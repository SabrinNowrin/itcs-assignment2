"""Inference: predict the class of individual images or every image in a folder.

  python predict.py --ckpt checkpoints/best_model.pt --images some_dir_or_file.jpg [more ...]
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from ssl_cv.data import IMG_EXTS, eval_transform
from ssl_cv.models import Classifier, build_encoder
from ssl_cv.utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--topk", type=int, default=3)
    a = p.parse_args()
    dev = get_device()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    stem = "auto" if ck["stem"].startswith("patch") else ck["stem"]
    model = Classifier(build_encoder(ck["arch"], ck["img_size"], stem), len(ck["classes"]), ck.get("dropout", 0.0))
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    t = eval_transform(ck["img_size"], ck["mean"], ck["std"])

    paths = []
    for s in a.images:
        s = Path(s)
        paths += sorted(q for q in s.rglob("*") if q.suffix.lower() in IMG_EXTS) if s.is_dir() else [s]
    with torch.no_grad():
        for path in paths:
            x = t(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
            prob = model(x).softmax(1)[0].cpu()
            top = prob.topk(min(a.topk, len(prob)))
            preds = ", ".join(f"{ck['classes'][i]} ({v * 100:.1f}%)" for v, i in zip(top.values, top.indices))
            print(f"{path}: {preds}")


if __name__ == "__main__":
    main()
