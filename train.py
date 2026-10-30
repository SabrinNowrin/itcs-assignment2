"""Supervised training, joint supervised + SSL training, and fine-tuning of an SSL encoder.

Modes (all share one training loop so comparisons are controlled):
  supervised baseline : python train.py --data_dir data --out runs/sup
  joint CE + SimCLR   : python train.py --data_dir data --out runs/joint --ssl_weight 0.5
  SSL -> fine-tune    : python train.py --data_dir data --out runs/ft --init_from runs/simclr/encoder.pt
  SSL -> linear (aug) : python train.py ... --init_from runs/simclr/encoder.pt --freeze_backbone

In joint mode every image yields three views: one supervised-augmented view for the
cross-entropy loss (identical to the baseline) and two SimCLR views for the NT-Xent loss,
computed through a separate projection head.
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ssl_cv.data import (CachedDataset, MultiView, add_ssl_aug_args, build_data, eval_transform,
                         ssl_transform_from_args, supervised_transform)
from ssl_cv.losses import nt_xent
from ssl_cv.models import Classifier, ProjectionHead, build_encoder
from ssl_cv.utils import (CSVLogger, cosine_with_warmup, evaluate, get_device, make_out_dir, mix_batch,
                          param_groups, save_json, set_seed)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--stem", default="auto", choices=["auto", "cifar", "small", "imagenet"])
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--aug", choices=["weak", "standard", "strong"], default="standard")
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--mix", choices=["none", "mixup", "cutmix", "both"], default="none")
    p.add_argument("--mix_alpha", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    # SSL initialisation / fine-tuning
    p.add_argument("--init_from", default="", help="encoder.pt produced by pretrain.py")
    p.add_argument("--backbone_lr_mult", type=float, default=1.0, help="backbone lr = lr * mult")
    p.add_argument("--freeze_backbone", action="store_true", help="linear evaluation (with augmentation)")
    # joint training
    p.add_argument("--ssl_weight", type=float, default=0.0, help=">0 adds lambda * NT-Xent (joint training)")
    p.add_argument("--temperature", type=float, default=0.2)
    # data / bookkeeping
    p.add_argument("--val_frac", type=float, default=0.0, help="stratified hold-out from train for model selection")
    p.add_argument("--label_frac", type=float, default=1.0, help="use only this fraction of training labels")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--log_test", action="store_true",
                   help="also log test accuracy during training (for learning curves only, never for selection)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no_amp", action="store_true")
    add_ssl_aug_args(p)
    return p.parse_args()


def main():
    a = get_args()
    set_seed(a.seed)
    out = make_out_dir(a.out)
    dev = get_device()
    amp = dev.type == "cuda" and not a.no_amp

    init = None
    if a.init_from:
        init = torch.load(a.init_from, map_location="cpu", weights_only=False)
        a.arch, a.img_size, a.stem = init["arch"], init["img_size"], init["stem"]
        if a.stem.startswith("patch"):
            a.stem = "auto"

    d = build_data(a.data_dir, a.img_size, val_frac=a.val_frac, label_frac=a.label_frac, seed=a.seed)
    C = len(d["classes"])
    mean, std = (init["mean"], init["std"]) if init else (d["mean"], d["std"])
    ev_t = eval_transform(a.img_size, mean, std)
    sup_t = supervised_transform(a.img_size, mean, std, a.aug)
    joint = a.ssl_weight > 0
    if joint:
        v = ssl_transform_from_args(a, mean, std)
        train_t = MultiView([sup_t, v, v])
    else:
        train_t = sup_t

    bs = min(a.batch_size, len(d["train"][1]))
    pin = dev.type == "cuda"
    train_loader = DataLoader(CachedDataset(*d["train"], train_t), batch_size=bs, shuffle=True, drop_last=True,
                              num_workers=a.workers, pin_memory=pin, persistent_workers=a.workers > 0)
    mk = lambda split: DataLoader(CachedDataset(*d[split], ev_t), batch_size=256, num_workers=a.workers)
    train_eval, val_loader, test_loader = mk("train"), mk("val"), mk("test")

    encoder = build_encoder(a.arch, a.img_size, a.stem)
    if init:
        encoder.load_state_dict(init["encoder"])
        print(f"[init] loaded SSL encoder from {a.init_from}")
    model = Classifier(encoder, C, a.dropout).to(dev)
    proj = ProjectionHead(encoder.out_dim).to(dev) if joint else None

    if a.freeze_backbone:
        for p_ in model.encoder.parameters():
            p_.requires_grad = False
    groups = []
    if not a.freeze_backbone:
        groups += [dict(g, lr=a.lr * a.backbone_lr_mult) for g in param_groups(model.encoder, a.wd)]
    groups += [dict(g, lr=a.lr) for g in param_groups(model.head, a.wd)]
    if proj is not None:
        groups += [dict(g, lr=a.lr) for g in param_groups(proj, a.wd)]
    opt = (torch.optim.SGD(groups, lr=a.lr, momentum=0.9, nesterov=True) if a.optimizer == "sgd"
           else torch.optim.AdamW(groups, lr=a.lr))
    sched = cosine_with_warmup(opt, a.epochs * len(train_loader), a.warmup_epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log = CSVLogger(f"{out}/log.csv")
    mode = ("linear" if a.freeze_backbone else "finetune") if init else ("joint" if joint else "supervised")
    print(f"[train] mode={mode} | {a.arch} | {len(train_loader)} steps/epoch | device {dev}")

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        if a.freeze_backbone:
            model.encoder.eval()  # keep BN statistics fixed
        ce_sum = ssl_sum = 0.0
        nb = 0
        for x, y in train_loader:
            y = y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                xs = (x[0] if joint else x).to(dev, non_blocking=True)
                xs, ya, yb, lam = mix_batch(xs, y, a.mix, a.mix_alpha)
                logits = model(xs).float()
                ce = (lam * F.cross_entropy(logits, ya, label_smoothing=a.label_smoothing)
                      + (1 - lam) * F.cross_entropy(logits, yb, label_smoothing=a.label_smoothing))
                loss = ce
                if joint:
                    x1, x2 = x[1].to(dev, non_blocking=True), x[2].to(dev, non_blocking=True)
                    z = proj(model.encoder(torch.cat([x1, x2])))
                    ssl = nt_xent(*z.chunk(2), temperature=a.temperature)
                    loss = ce + a.ssl_weight * ssl
                    ssl_sum += ssl.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ce_sum += ce.item()
            nb += 1
        row = {"epoch": ep, "ce": round(ce_sum / nb, 4), "ssl": round(ssl_sum / nb, 4) if joint else "",
               "lr": round(sched.get_last_lr()[-1], 5), "time_min": round((time.time() - t0) / 60, 2),
               "train_acc": "", "val_acc": "", "test_acc": ""}
        if ep % a.eval_every == 0 or ep == a.epochs:
            row["train_acc"] = round(evaluate(model, train_eval, dev)[0], 2)  # clean (eval-transform) train acc
            if len(d["val"][1]):
                row["val_acc"] = round(evaluate(model, val_loader, dev)[0], 2)
            if a.log_test:
                row["test_acc"] = round(evaluate(model, test_loader, dev)[0], 2)
        log.log(row)
        print(" ".join(f"{k}={v}" for k, v in row.items() if v != ""), flush=True)

    test_acc, test_loss, conf = evaluate(model, test_loader, dev, C)
    train_acc = evaluate(model, train_eval, dev)[0]
    val_acc = evaluate(model, val_loader, dev)[0] if len(d["val"][1]) else None
    torch.save({"model": model.state_dict(), "arch": a.arch, "stem": encoder.stem, "img_size": a.img_size,
                "mean": mean, "std": std, "classes": d["classes"], "dropout": a.dropout, "args": vars(a)},
               f"{out}/model.pt")
    res = {"mode": mode, "arch": a.arch, "test_acc": round(test_acc, 2), "train_acc": round(train_acc, 2),
           "val_acc": None if val_acc is None else round(val_acc, 2),
           "generalization_gap": round(train_acc - test_acc, 2), "test_loss": round(test_loss, 4),
           "minutes": round((time.time() - t0) / 60, 1), "args": vars(a)}
    save_json(res, f"{out}/results.json")
    if conf is not None:
        torch.save(conf, f"{out}/confusion.pt")
    print(f"[done] test acc {test_acc:.2f}% | clean train acc {train_acc:.2f}% | gap {train_acc - test_acc:.2f}")


if __name__ == "__main__":
    main()
