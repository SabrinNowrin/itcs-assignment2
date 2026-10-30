import csv
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_with_warmup(optimizer, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def param_groups(module, wd):
    """No weight decay on biases and norm parameters."""
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") or "pos" in n or "cls" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


class CSVLogger:
    def __init__(self, path):
        self.path, self.fields = path, None

    def log(self, row):
        new = self.fields is None
        if new:
            self.fields = list(row.keys())
        with open(self.path, "a" if not new else "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_out_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device, num_classes=None):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    preds_all, ys_all = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        total += y.numel()
        preds_all.append(p.cpu())
        ys_all.append(y.cpu())
    if total == 0:
        return float("nan"), float("nan"), None
    preds, ys = torch.cat(preds_all), torch.cat(ys_all)
    conf = None
    if num_classes:
        conf = torch.zeros(num_classes, num_classes, dtype=torch.long)
        conf.index_put_((ys, preds), torch.ones_like(ys), accumulate=True)
    return 100.0 * correct / total, loss_sum / total, conf


@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, ys = [], []
    for x, y in loader:
        feats.append(encoder(x.to(device, non_blocking=True)).float().cpu())
        ys.append(y)
    if not feats:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(feats), torch.cat(ys)


def knn_accuracy(train_f, train_y, test_f, test_y, num_classes, k=20, t=0.07):
    """Weighted kNN on cosine similarity (as in InstDisc / DINO). No training involved,
    so it is a cheap, hyperparameter-light probe of representation quality."""
    if len(test_y) == 0:
        return float("nan")
    train_f, test_f = F.normalize(train_f, dim=1), F.normalize(test_f, dim=1)
    sim = test_f @ train_f.t()
    w, idx = sim.topk(min(k, train_f.shape[0]), dim=1)
    w = (w / t).exp()
    votes = torch.zeros(test_f.shape[0], num_classes)
    votes.scatter_add_(1, train_y[idx], w)
    return 100.0 * (votes.argmax(1) == test_y).float().mean().item()


def linear_probe(train_f, train_y, test_f, test_y, num_classes, epochs=300, lr=1e-2, wd=1e-4, device="cpu"):
    """Logistic regression on frozen (standardised) features."""
    if len(test_y) == 0:
        return float("nan")
    mu, sd = train_f.mean(0), train_f.std(0) + 1e-6
    xtr, xte = ((train_f - mu) / sd).to(device), ((test_f - mu) / sd).to(device)
    ytr = train_y.to(device)
    clf = nn.Linear(xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = torch.randperm(len(ytr), device=device)
        for i in range(0, len(ytr), 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(clf(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        return 100.0 * (clf(xte).argmax(1).cpu() == test_y).float().mean().item()


# --------------------------------------------------------------------------- mixing
def mix_batch(x, y, mode, alpha):
    """MixUp / CutMix. Returns mixed x, the two label sets and the mixing weight."""
    if mode == "none" or alpha <= 0:
        return x, y, y, 1.0
    if mode == "both":
        mode = random.choice(["mixup", "cutmix"])
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.shape[0], device=x.device)
    if mode == "mixup":
        x = lam * x + (1 - lam) * x[perm]
    else:
        H, W = x.shape[-2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, y, y[perm], lam
