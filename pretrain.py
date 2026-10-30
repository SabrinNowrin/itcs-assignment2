"""Self-supervised pretraining on the training images only (labels are NOT used for training).

Methods
  simclr   : contrastive learning (NT-Xent between two augmented views of each image)
  rotation : RotNet pretext task (predict 0/90/180/270 degree rotation) -- a simpler SSL baseline

Labels are only used for *monitoring* (weighted kNN accuracy on a held-out split) and the
final kNN / linear-probe evaluation.

Example:
  python pretrain.py --data_dir data --out runs/simclr_r18 --method simclr --epochs 500
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import v2 as T

from ssl_cv.data import (CachedDataset, MultiView, add_ssl_aug_args, build_data, eval_transform,
                         ssl_transform_from_args)
from ssl_cv.losses import nt_xent
from ssl_cv.models import ProjectionHead, build_encoder
from ssl_cv.utils import (CSVLogger, cosine_with_warmup, extract_features, get_device, knn_accuracy,
                          linear_probe, make_out_dir, param_groups, save_json, set_seed)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--method", choices=["simclr", "rotation"], default="simclr")
    p.add_argument("--arch", default="resnet18", help="resnet18 | resnet34 | resnet50 | vit_tiny")
    p.add_argument("--stem", default="auto", choices=["auto", "cifar", "small", "imagenet"])
    p.add_argument("--img_size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=0.12, help="SGD: ~0.06 per 128 images; AdamW: ~1e-3")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--proj_hidden", type=int, default=512)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--val_frac", type=float, default=0.0,
                   help="hold out part of train for kNN monitoring (otherwise test is monitored, report only final)")
    p.add_argument("--knn_every", type=int, default=25)
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

    d = build_data(a.data_dir, a.img_size, val_frac=a.val_frac, seed=a.seed)
    C = len(d["classes"])
    mean, std = d["mean"], d["std"]
    ev_t = eval_transform(a.img_size, mean, std)

    # SSL uses every training image (labeled + unlabeled); labels are ignored.
    if a.method == "simclr":
        view = ssl_transform_from_args(a, mean, std)
        train_t = MultiView([view, view])
    else:
        train_t = T.Compose([T.RandomResizedCrop(a.img_size, scale=(0.4, 1.0)), T.RandomHorizontalFlip(),
                             T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])
    ssl_set = ConcatDataset([CachedDataset(*d["train"], train_t), CachedDataset(*d["unlabeled"], train_t)])
    bs = min(a.batch_size, len(ssl_set))
    loader = DataLoader(ssl_set, batch_size=bs, shuffle=True, drop_last=True, num_workers=a.workers,
                        pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)

    mon_name = "val" if len(d["val"][1]) else "test"
    knn_train = DataLoader(CachedDataset(*d["train"], ev_t), batch_size=256, num_workers=a.workers)
    knn_mon = DataLoader(CachedDataset(*d[mon_name], ev_t), batch_size=256, num_workers=a.workers)

    encoder = build_encoder(a.arch, a.img_size, a.stem).to(dev)
    head = (ProjectionHead(encoder.out_dim, a.proj_hidden, a.proj_dim) if a.method == "simclr"
            else nn.Linear(encoder.out_dim, 4)).to(dev)
    params = param_groups(nn.ModuleList([encoder, head]), a.wd)
    opt = (torch.optim.SGD(params, lr=a.lr, momentum=0.9) if a.optimizer == "sgd"
           else torch.optim.AdamW(params, lr=a.lr))
    steps = a.epochs * len(loader)
    sched = cosine_with_warmup(opt, steps, a.warmup_epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log = CSVLogger(f"{out}/log.csv")
    print(f"[pretrain] {a.method} | {a.arch} ({encoder.stem} stem, {encoder.out_dim}-d) | "
          f"{len(loader)} steps/epoch | device {dev}")

    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        encoder.train()
        head.train()
        tot, nb = 0.0, 0
        for x, _ in loader:
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                if a.method == "simclr":
                    x1, x2 = x[0].to(dev, non_blocking=True), x[1].to(dev, non_blocking=True)
                    z = head(encoder(torch.cat([x1, x2])))
                    loss = nt_xent(*z.chunk(2), temperature=a.temperature)
                else:
                    x = x.to(dev, non_blocking=True)
                    xr = torch.cat([torch.rot90(x, k, dims=(2, 3)) for k in range(4)])
                    yr = torch.arange(4, device=dev).repeat_interleave(x.shape[0])
                    loss = F.cross_entropy(head(encoder(xr)).float(), yr)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        row = {"epoch": ep, "loss": round(tot / nb, 4), "lr": round(sched.get_last_lr()[0], 5),
               "time_min": round((time.time() - t0) / 60, 2), f"knn_{mon_name}": ""}
        if ep % a.knn_every == 0 or ep == a.epochs:
            ftr, ytr = extract_features(encoder, knn_train, dev)
            fm, ym = extract_features(encoder, knn_mon, dev)
            row[f"knn_{mon_name}"] = round(knn_accuracy(ftr, ytr, fm, ym, C), 2)
        log.log(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    ckpt = {"encoder": encoder.state_dict(), "arch": a.arch, "stem": encoder.stem, "img_size": a.img_size,
            "mean": mean, "std": std, "classes": d["classes"], "args": vars(a)}
    torch.save(ckpt, f"{out}/encoder.pt")

    # Final representation quality on the test set (frozen features, labels of train split only).
    test_loader = DataLoader(CachedDataset(*d["test"], ev_t), batch_size=256, num_workers=a.workers)
    ftr, ytr = extract_features(encoder, knn_train, dev)
    fte, yte = extract_features(encoder, test_loader, dev)
    res = {"method": a.method, "arch": a.arch, "epochs": a.epochs,
           "knn_test": round(knn_accuracy(ftr, ytr, fte, yte, C), 2),
           "linear_probe_test": round(linear_probe(ftr, ytr, fte, yte, C, device=dev), 2),
           "minutes": round((time.time() - t0) / 60, 1), "args": vars(a)}
    save_json(res, f"{out}/results.json")
    print(f"[done] kNN test {res['knn_test']:.2f}% | linear probe test {res['linear_probe_test']:.2f}%")


if __name__ == "__main__":
    main()
