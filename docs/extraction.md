# Dataset extraction

The EyePACS Kaggle dataset is distributed as split zip archives. This document explains how to
extract them before running `crop-images`.

## Prerequisites

Install [p7zip](https://www.7-zip.org/):

```bash
brew install p7zip       # macOS
apt install p7zip         # Debian/Ubuntu
```

## Usage

Run the extraction script from the repo root:

```bash
bash scripts/extract_eyepacs.sh
```

This extracts:

| Archive | Parts | Output | Images |
|---------|-------|--------|--------|
| Train | `train.zip.001` – `.005` | `data/.../extracted/train/` | 35,126 |
| Test | `test.zip.001` – `.007` | `data/.../extracted/test/` | 53,576 |

Extracting only the first part of each split (`.001`) automatically references the remaining parts.

## After extraction

Point `crop-images` at the extracted folders:

```bash
crop-images \
    --image-folder data/EyePACS-Kaggle/diabetic-retinopathy-detection/extracted/train \
    --output-folder data/cropped \
    --crop-size 512 \
    --skip-existing \
    -n 8
```
