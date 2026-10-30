"""Reproduce the reported test accuracy from a checkpoint.

  Classifier (model.pt from train.py):  test accuracy, per-class accuracy, confusion matrix
  Encoder (encoder.pt from pretrain.py): weighted kNN and linear-probe accuracy on frozen features

  python evaluate.py --ckpt checkpoints/best_model.pt --data_dir data
"""
import argparse

import torch
from torch.utils.data import DataLoader

from ssl_cv.data import CachedDataset, eval_transform, load_images, scan_folder
from ssl_cv.models import Classifier, build_encoder
from ssl_cv.utils import evaluate, extract_features, get_device, knn_accuracy, linear_probe


def loader_for(folder, ckpt, workers):
    classes, samples = scan_folder(folder)
    if classes != ckpt["classes"]:
        raise ValueError("class folders do not match the checkpoint's classes")
    imgs = load_images(samples, int(round(ckpt["img_size"] * 1.14)))
    t = eval_transform(ckpt["img_size"], ckpt["mean"], ckpt["std"])
    return DataLoader(CachedDataset(imgs, [y for _, y in samples], t), batch_size=256, num_workers=workers)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    dev = get_device()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    stem = "auto" if ck["stem"].startswith("patch") else ck["stem"]
    encoder = build_encoder(ck["arch"], ck["img_size"], stem)
    classes = ck["classes"]
    test = loader_for(f"{a.data_dir}/{a.split}", ck, a.workers)

    if "model" in ck:
        model = Classifier(encoder, len(classes), ck.get("dropout", 0.0))
        model.load_state_dict(ck["model"])
        model.to(dev)
        acc, loss, conf = evaluate(model, test, dev, len(classes))
        print(f"{a.split} accuracy: {acc:.2f}%  (loss {loss:.4f}, n={int(conf.sum())})")
        per = conf.diag().float() / conf.sum(1).clamp(min=1).float() * 100
        print("\nper-class accuracy:")
        for c, v in sorted(zip(classes, per.tolist()), key=lambda t: t[1]):
            print(f"  {v:6.2f}%  {c}")
        torch.save(conf, "confusion_matrix.pt")
    else:
        encoder.load_state_dict(ck["encoder"])
        encoder.to(dev)
        train = loader_for(f"{a.data_dir}/train", ck, a.workers)
        ftr, ytr = extract_features(encoder, train, dev)
        fte, yte = extract_features(encoder, test, dev)
        print(f"kNN (k=20) {a.split} accuracy:     {knn_accuracy(ftr, ytr, fte, yte, len(classes)):.2f}%")
        print(f"linear probe {a.split} accuracy:   {linear_probe(ftr, ytr, fte, yte, len(classes), device=dev):.2f}%")


if __name__ == "__main__":
    main()
