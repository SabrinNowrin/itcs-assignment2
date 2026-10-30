# Report skeleton (max 2 pages)

Replace every `XX` and every *[bracketed prompt]* with your own numbers and interpretation.
The prompts are questions to answer from YOUR results — don't keep claims your data doesn't support.

---

## Self-Supervised Representation Learning from 2,400 Images
*Name · ITCS 6169/8169 · GitHub: <link>*

**Best test accuracy: XX.X%** (SimCLR-pretrained ResNet-18, fine-tuned; checkpoint + reproduction commands in repo).

### 1. Supervised baseline
ResNet-18 trained from scratch at XX px (3×3 stride-1 stem so small images are not down-sampled too early),
cross-entropy, SGD (Nesterov, lr 0.05, wd 5e-4), cosine schedule, 200 epochs, random-resized-crop + flip
+ colour jitter. **Test accuracy: XX.X%**, clean training accuracy XX.X% (gap XX points).
*[One sentence: what does the gap tell you? Overfitting?]*

### 2. SSL strategy
*[Explain the idea in your own words, not "we used SimCLR".]* Key points to cover:
- what the positive pair is (two augmentations of one image) and where negatives come from (other images
  in the batch, 2N−2 per anchor, N = 256); NT-Xent = softmax over similarities, temperature τ = 0.2.
- why the projection head exists (loss on g(h); keep h, which retains augmentation-relevant info).
- **adaptations to this dataset**: small-image stem, all 2,400 images used without labels, batch size limited by
  dataset size (only ~9 steps/epoch → many epochs), augmentation choices from your ablation, temperature.
- pretraining → fine-tuning protocol (lower backbone LR ×0.2, 100 epochs) and the joint variant (CE on a
  standard view + λ·NT-Xent on two SimCLR views, λ = 0.5).

### 3. Experimental analysis

| Method | Test Acc. | Train–test gap |
|---|---|---|
| Supervised baseline | XX.X% | XX |
| RotNet → fine-tune | XX.X% | XX |
| SimCLR (frozen, linear probe / kNN) | XX.X% / XX.X% | – |
| SimCLR → fine-tune | XX.X% | XX |
| Joint CE + SimCLR (λ = 0.5 / 1.0) | XX.X% / XX.X% | XX |
| Strong-aug supervised (control) | XX.X% | XX |
| **Best: SimCLR-1000 → strong-aug FT** | **XX.X%** | XX |

Small second table (pick ONE that tells a story):
- *Augmentation ablation* (kNN acc, 200-ep SimCLR): full XX · no colour XX · crop_min 0.5 XX · no blur XX
- *Low-label*: 10% labels: sup XX vs SimCLR-FT XX · 25%: XX vs XX · 100%: XX vs XX

Interpretation prompts (answer the 3–4 your data speaks to most clearly):
- Did SSL help with only 2,400 images? By how much, and is that larger than run-to-run noise (if you can,
  re-run the key comparison with a 2nd seed)?
- Is the strong-aug supervised control close to the best SSL model? If so, how much of the gain is SSL vs.
  just better augmentation/regularisation? (This is the most important fairness check.)
- Pretrain→FT vs. joint: which won and why might that be (e.g., competing objectives, batch size for negatives)?
- Which augmentation mattered most for SimCLR here, and why does that make sense for these images?
- Does the SSL benefit grow as labels shrink? (Low-label table.)
- Did the gap shrink with SSL (overfitting)? Refer to curves.
- Was RotNet competitive? Is complexity paying for itself?

### 4. Limitations & conclusion
*[Single seed / small test set → confidence intervals; few negatives per batch; limited compute meant short
schedules; hyperparameters tuned on ... ; only CNN backbone (or ViT result if you ran it).]*
*[2–3 sentences: what you learned about representation learning when labels are scarce.]*
