"""Data loading and augmentation.

Expected layout (ImageFolder style):
    DATA_DIR/train/<class_name>/*.jpg
    DATA_DIR/test/<class_name>/*.jpg

All images are decoded once and cached in RAM (2,400 images is small), resized so the
shorter side is ~1.14x the training resolution. This makes every epoch CPU-light.
Normalization statistics are computed from the training images only.
"""
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- files
def scan_folder(root):
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class sub-folders found in {root}")
    samples = []
    for idx, c in enumerate(classes):
        for p in sorted((root / c).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), idx))
    return classes, samples


def stratified_split(samples, frac, seed):
    """Return (picked, rest): `frac` of each class goes to `picked` (at least 1 per class)."""
    by_cls = defaultdict(list)
    for s in samples:
        by_cls[s[1]].append(s)
    rng = random.Random(seed)
    picked, rest = [], []
    for c in sorted(by_cls):
        items = by_cls[c][:]
        rng.shuffle(items)
        k = max(1, round(frac * len(items)))
        picked += items[:k]
        rest += items[k:]
    return picked, rest


def load_images(samples, cache_size):
    imgs = []
    for path, _ in samples:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = cache_size / min(w, h)
            im = im.resize((max(cache_size, round(w * s)), max(cache_size, round(h * s))), Image.BICUBIC)
            imgs.append(im.copy())
    return imgs


def channel_stats(images):
    to_t = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    s = torch.zeros(3)
    sq = torch.zeros(3)
    n = 0
    for im in images:
        x = to_t(im).flatten(1)
        s += x.sum(1)
        sq += (x ** 2).sum(1)
        n += x.shape[1]
    mean = s / n
    std = (sq / n - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


class CachedDataset(Dataset):
    def __init__(self, images, targets, transform):
        self.images, self.targets, self.transform = images, targets, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]


def build_data(data_dir, img_size, val_frac=0.0, label_frac=1.0, seed=0,
               train_dirname="train", test_dirname="test"):
    """Load and cache all splits.

    val_frac:   fraction of the training set held out (stratified) for model selection.
    label_frac: fraction of the remaining training set whose labels are used (for
                low-label experiments). Unlabeled images are returned separately so
                SSL methods can still use them.
    """
    data_dir = Path(data_dir)
    classes, train_samples = scan_folder(data_dir / train_dirname)
    val_samples = []
    if val_frac > 0:
        val_samples, train_samples = stratified_split(train_samples, val_frac, seed)
    unlabeled_samples = []
    if label_frac < 1.0:
        train_samples, unlabeled_samples = stratified_split(train_samples, label_frac, seed + 1)

    test_samples = []
    test_dir = data_dir / test_dirname
    if test_dir.is_dir():
        test_classes, test_samples = scan_folder(test_dir)
        if test_classes != classes:
            raise ValueError(f"Class folders differ between train and test:\n{classes}\n{test_classes}")
    else:
        print(f"[warn] no test folder at {test_dir}; test accuracy will be skipped")

    cache_size = int(round(img_size * 1.14))
    out = {"classes": classes, "img_size": img_size}
    for name, samples in [("train", train_samples), ("unlabeled", unlabeled_samples),
                          ("val", val_samples), ("test", test_samples)]:
        out[name] = (load_images(samples, cache_size), [y for _, y in samples])
    out["mean"], out["std"] = channel_stats(out["train"][0] + out["unlabeled"][0])
    print(f"[data] {len(classes)} classes | train {len(train_samples)} | unlabeled {len(unlabeled_samples)}"
          f" | val {len(val_samples)} | test {len(test_samples)} | cache {cache_size}px")
    return out


# --------------------------------------------------------------------------- transforms
def _finish(mean, std):
    return [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)]


def eval_transform(img_size, mean, std):
    return T.Compose([T.Resize(img_size), T.CenterCrop(img_size)] + _finish(mean, std))


def supervised_transform(img_size, mean, std, strength="standard"):
    if strength == "weak":
        ops = [T.RandomResizedCrop(img_size, scale=(0.6, 1.0)), T.RandomHorizontalFlip()]
        return T.Compose(ops + _finish(mean, std))
    if strength == "standard":
        ops = [T.RandomResizedCrop(img_size, scale=(0.35, 1.0)), T.RandomHorizontalFlip(),
               T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.5)]
        return T.Compose(ops + _finish(mean, std))
    if strength == "strong":
        ops = [T.RandomResizedCrop(img_size, scale=(0.25, 1.0)), T.RandomHorizontalFlip(),
               T.TrivialAugmentWide()]
        return T.Compose(ops + _finish(mean, std) + [T.RandomErasing(p=0.25)])
    raise ValueError(strength)


def ssl_transform(img_size, mean, std, crop_min=0.2, color=True, color_strength=0.4,
                  gray=True, blur=True, solarize=False, flip=True):
    """SimCLR-style view generator. Each flag lets you ablate one augmentation."""
    ops = [T.RandomResizedCrop(img_size, scale=(crop_min, 1.0))]
    if flip:
        ops.append(T.RandomHorizontalFlip())
    if color:
        s = color_strength
        ops.append(T.RandomApply([T.ColorJitter(s, s, s, min(0.1, s / 4))], p=0.8))
    if gray:
        ops.append(T.RandomGrayscale(p=0.2))
    if blur:
        k = max(3, int(0.1 * img_size) // 2 * 2 + 1)
        ops.append(T.RandomApply([T.GaussianBlur(k, sigma=(0.1, 2.0))], p=0.5))
    if solarize:
        ops.append(T.RandomSolarize(threshold=128, p=0.2))
    return T.Compose(ops + _finish(mean, std))


class MultiView:
    """Apply a list of transforms to the same image, returning a list of views."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return [t(img) for t in self.transforms]


def add_ssl_aug_args(p):
    g = p.add_argument_group("SSL augmentations (for ablations)")
    g.add_argument("--crop_min", type=float, default=0.2, help="min scale of RandomResizedCrop")
    g.add_argument("--color_strength", type=float, default=0.4)
    g.add_argument("--no_color", action="store_true")
    g.add_argument("--no_gray", action="store_true")
    g.add_argument("--no_blur", action="store_true")
    g.add_argument("--no_flip", action="store_true")
    g.add_argument("--solarize", action="store_true")


def ssl_transform_from_args(a, mean, std):
    return ssl_transform(a.img_size, mean, std, crop_min=a.crop_min, color=not a.no_color,
                         color_strength=a.color_strength, gray=not a.no_gray, blur=not a.no_blur,
                         solarize=a.solarize, flip=not a.no_flip)
