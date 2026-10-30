# Self-Supervised Representation Learning with 2,400 Images

ITCS 6169/8169 Assignment 2. All models are **randomly initialised** (`weights=None`) and trained
**only** on the provided 2,400 training images (16 classes). No external data or pretrained weights.

**Best test accuracy: `81.75%`** (joint CE + SimCLR, run `joint_w0.5`) (`checkpoints/best_model.pt`, produced by the command under *Reproduce the best model*)

## What is in here

| File | Purpose |
|---|---|
| `pretrain.py` | SSL pretraining: **SimCLR** (NT-Xent contrastive) or **RotNet** (rotation prediction). Logs kNN accuracy during training; reports kNN + linear-probe accuracy at the end. |
| `train.py` | One training loop for: supervised baseline, **joint CE + SimCLR**, **fine-tuning** an SSL encoder, frozen-backbone linear eval. Supports low-label (`--label_frac`) runs. |
| `evaluate.py` | Reproduces test accuracy (+ per-class accuracy, confusion matrix) from a checkpoint. |
| `predict.py` | Inference on single images or folders. |
| `summarize.py` | Collects all `runs/*/results.json` into `runs/summary.md` and plots learning curves. |
| `scripts/run_experiments.sh` | The full experiment plan (tiers 1–3). |
| `ssl_cv/` | Data loading & augmentation, models (ResNet with small-image stem, small ViT), losses, utilities. |

## Setup

```bash
pip install -r requirements.txt
```

Data layout (ImageFolder):

```
data/
  train/<class_name>/*.jpg    # 2,400 images, 16 classes
  test/<class_name>/*.jpg
```

Images are cached in RAM once per run; normalisation statistics are computed from the training images.

### Google Colab

```python
!git clone https://github.com/<you>/<repo>.git && cd <repo>
from google.colab import drive; drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/<dataset_folder> data      # copy locally: much faster than reading from Drive
!pip install -r requirements.txt
!bash scripts/run_experiments.sh 1                         # DATA defaults to ./data
```

## Methods in one paragraph each

**Supervised baseline.** ResNet-18 (stem: 3×3 stride-1 conv for small images), cross-entropy,
SGD + Nesterov, cosine LR with warm-up, random-resized-crop + flip + mild colour jitter.

**SimCLR.** Two random views of each image are encoded and passed through an MLP projection head.
The NT-Xent loss treats the other view of the same image as the positive and the other 2N−2 views in
the batch as negatives, so the encoder must become invariant to the augmentations while still
distinguishing individual images. The projection head is discarded afterwards; the encoder is
fine-tuned with a lower backbone learning rate (`--backbone_lr_mult`).

**Joint training.** Each image produces one supervised view (for CE, identical to the baseline) and
two SimCLR views (for NT-Xent, separate projection head): `loss = CE + λ·NT-Xent`.

**RotNet.** Predict which of 4 rotations was applied — a much simpler pretext task, used to test
whether a more complex SSL objective is actually needed.

## Running experiments

```bash
DATA=data bash scripts/run_experiments.sh 1     # baseline, SimCLR->FT, joint     (required comparisons)
DATA=data bash scripts/run_experiments.sh 2     # RotNet, augmentation ablation, low-label, λ sweep
DATA=data bash scripts/run_experiments.sh 3     # best-recipe candidate + its fair supervised control
python summarize.py --runs runs
```

Use `IMG=64` for faster runs. Each run folder contains `log.csv` (per-epoch), `results.json`
(final numbers + all hyperparameters) and the checkpoint (`model.pt` or `encoder.pt`).

Useful flags: `--val_frac 0.1` holds out a stratified validation split for hyperparameter choices
(recommended, so the test set is used only for the final number). `--log_test` records test accuracy
during training for learning-curve plots only. SSL augmentation ablations: `--no_color`, `--no_gray`,
`--no_blur`, `--no_flip`, `--solarize`, `--crop_min`, `--color_strength`.

## Reproduce the best model

```bash
# best model: joint cross-entropy + SimCLR training (81.75% test accuracy)
python train.py --data_dir data --out runs/joint_w0.5 --arch resnet18 --img_size 96 \
    --epochs 200 --aug standard --ssl_weight 0.5
```

## Evaluate the released checkpoint

```bash
# download best_model.pt (link: <Google Drive / Hugging Face URL>) into checkpoints/
python evaluate.py --ckpt checkpoints/best_model.pt --data_dir data
python predict.py  --ckpt checkpoints/best_model.pt --images path/to/image_or_folder
```

## Results

| Method | Test Acc. |
|---|---|
| Supervised baseline | 81.25% |
| SimCLR frozen features (kNN / linear probe) | 74.50% / 76.50% |
| SimCLR → fine-tune | 77.75% |
| **Joint CE + SimCLR (λ = 0.5), best model** | **81.75%** |
