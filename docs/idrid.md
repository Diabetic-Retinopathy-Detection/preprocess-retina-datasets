# IDRiD dataset preparation

The [IDRiD dataset](https://idrid.grand-challenge.org/) is a publicly available
retinal fundus dataset that includes a disease-grading task. This document
explains how to crop the raw images and stage them into a grade-labelled
ImageFolder (with a seeded stratified validation split) suitable for downstream
fine-tuning or evaluation.

## Dataset facts

The official source is located at:

```text
datasets/IDRiD/
```

Within it, the Disease Grading images and labels live under:

```text
B. Disease Grading/
  1. Original Images/
    a. Training Set/   # 413 images (IDRiD_001.jpg ... IDRiD_413.jpg)
    b. Testing Set/    # 103 images (IDRiD_414.jpg ... IDRiD_516.jpg)
  2. Groundtruths/
    a. IDRiD_Disease Grading_Training Labels.csv
    b. IDRiD_Disease Grading_Testing Labels.csv
```

IDRiD uses five disease-grading classes, `0` through `4`:

| Grade | Meaning |
|-------|---------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

There are **413 training** and **103 test** images. The disease-grading label CSV
files map image names (without extension, e.g. `IDRiD_001`) to a grade. IDRiD has
**no official validation split**, so the training images are carved into
`train`/`valid` below.

## Crop the raw images

As with DDR, crop the raw original images (this is the crop-first step the
normalization statistics below rely on). IDRiD ships single images per
directory (no per-split subfolders), so crop each set into a separate folder:

```bash
crop-images \
    --image-folder "datasets/IDRiD/B. Disease Grading/1. Original Images/a. Training Set" \
    --output-folder data/IDRiD-cropped/train \
    --crop-size 512 \
    --skip-existing \
    -n 8

crop-images \
    --image-folder "datasets/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set" \
    --output-folder data/IDRiD-cropped/test \
    --crop-size 512 \
    --skip-existing \
    -n 8
```

`--skip-existing` makes the crop step safe to rerun after an interruption.

## Prepare the ImageFolder

Stage the cropped images into a grade-labelled ImageFolder with `prepare-idrid`:

```bash
prepare-idrid \
    --train-images data/IDRiD-cropped/train \
    --test-images data/IDRiD-cropped/test \
    --labels-train "datasets/IDRiD/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv" \
    --labels-test "datasets/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv" \
    --output data/IDRiD-ImageFolder
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-images` | (required) | Directory of the (cropped) training images |
| `--test-images` | (required) | Directory of the (cropped) test images |
| `--labels-train` | (required) | IDRiD training label CSV |
| `--labels-test` | (required) | IDRiD test label CSV |
| `--output` | (required) | Output directory for the ImageFolder structure |
| `--valid-ratio` | `0.2` | Fraction of each training grade reserved for `valid` |
| `--seed` | `42` | Random seed for the stratified valid split |
| `--copy` | symlinks | Copy image files instead of creating symlinks |

By default images are **symlinked** into place, so the source images are not
duplicated. On this repository the symlinks resolve to **absolute** target paths,
which break when the dataset is transferred to a cluster or another machine —
use `--copy` for a cluster-safe transferable dataset.

The resulting layout is:

```text
IDRiD-ImageFolder/
  train/0/  train/1/  train/2/  train/3/  train/4/
  valid/0/  valid/1/  valid/2/  valid/3/  valid/4/
  test/0/   test/1/   test/2/   test/3/   test/4/
```

### Split

The official 413 training images are carved with an **80/20 grade-stratified**
split (seed `42`, `valid_ratio 0.2`) into `train`/`valid`; the official 103 test
images are used as-is. Real per-grade counts:

| Split | 0 | 1 | 2 | 3 | 4 | Total |
|-------|-----|-----|------|-----|-----|-------|
| train | 107 | 16 | 109 | 59 | 39 | 330 |
| valid | 27 | 4 | 27 | 15 | 10 | 83 |
| test | 34 | 5 | 32 | 19 | 13 | 103 |

Total: 516 images across the three splits.

## Normalization statistics

Compute the per-channel mean/std over the exact consumer transform used at
evaluation/fine-tuning time:

```bash
compute-dataset-stats \
    --image-folder data/IDRiD-cropped/train \
    --resize 384 \
    --output ../data/idrid-mean-std-384.json
```

`--resize` defaults to `384` and matches the model repository's
`finetune_input_size: 384`; the transform is `Resize((384, 384))` (PIL BILINEAR)
then `ToTensor()` (divides by 255, values in `[0, 1]`). The statistics are
computed on the **full 413 official cropped training images** (not the carved
330) as a pixel-weighted population mean/std over every resized pixel.

Real values, computed with `compute-dataset-stats` on the full 413-image
cropped train set at Resize 384:

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.543465 | 0.25007 |
| G | 0.264543 | 0.143724 |
| B | 0.087818 | 0.086933 |

```json
{"mean": [0.543465, 0.264543, 0.087818], "std": [0.25007, 0.143724, 0.086933]}
```

### Normalization statistics: design note

This implementation intentionally diverges from the original DRC stats ticket,
which was flawed for this repository:

- **Pre-crop vs post-crop.** The ticket computed statistics on raw (pre-crop)
  512px images. That is wrong here: downstream fine-tuning consumes the
  **cropped** output, so statistics must be measured on the exact consumer
  transform — `crop -> Resize 384 -> ToTensor [0, 1]`. Computing on the raw
  pre-crop images would double-count dark borders (the crop removes them) and
  mismatch the consumer's normalization.
- **numpy-only, not torch.** The ticket required a torch `GradingDataset`. This
  repository has no torch dependency (deps are pillow/numpy/opencv-headless/tqdm
  per `pyproject.toml`). The implemented `compute-dataset-stats` is numpy-only,
  using PIL for the image decode and the consumer-matching BILINEAR resize, then
  `/255` and numpy accumulation.
- **This repo's `crop.py`, not stale `utils/crop.py`.** The ticket referenced a
  stale `utils/crop.py`; cropping now lives in this repository's `crop.py`
  (`crop-images`), and the stats tool consumes its output directly.

The final values are pixel-weighted float64 over the 413 official cropped
training images at Resize 384 (BILINEAR, matching the consumer's
`transforms.Resize` default), stored at 6-decimal precision.

## After preparation

Point the evaluation CLI in the model repository at the output folder. As with
DDR, the evaluation uses `train/` as the reference set and `test/` as the query
set; `valid/` mirrors the carve used during fine-tuning. Use the `--copy` output
when transferring to a cluster.
