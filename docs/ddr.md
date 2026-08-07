# DDR dataset preparation

The [DDR dataset](https://deepblueai.github.io/project/ddr.html) is distributed as a multi-part zip archive.
This document explains how to extract it and stage the images into a grade-labelled ImageFolder suitable for
k-NN evaluation of pretrained encoders.

## Prerequisites

Install [p7zip](https://www.7-zip.org/):

```bash
brew install p7zip       # macOS
apt install p7zip         # Debian/Ubuntu
```

## Usage

Extract the archive:

```bash
bash scripts/extract_ddr.sh
```

| Archive | Parts | Output | Images |
|---------|-------|--------|--------|
| DDR | `DDR-dataset.zip.001` – `.010` | `data/DDR-dataset/DR_grading/` | 13,673 |

Extracting only the first part (`.001`) automatically references the remaining parts.

Then stage the images into an ImageFolder layout with `prepare-ddr`:

```bash
prepare-ddr \
    --input data/DDR-dataset/DR_grading \
    --output data/DDR-ImageFolder
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | DDR `DR_grading/` directory (contains `train.txt`, `valid.txt`, `test.txt`) |
| `--output` | (required) | Output directory for the ImageFolder structure |
| `--exclude-grade` | `5` | Grade to exclude (`5` = unreadable) |
| `--copy` | symlinks | Copy image files instead of creating symlinks |

By default images are **symlinked** into place, so the 15 GB of source images are not duplicated.
Use `--copy` on filesystems or for transfers (e.g. tar/scp) where symlinks are not preserved.

## Grades

| Grade | Meaning |
|-------|---------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |
| 5 | Unreadable (excluded, matching SSiT's 5-class evaluation) |

## Output

```text
DDR-ImageFolder/
  train/
    0/  1/  2/  3/  4/
  valid/
    0/  1/  2/  3/  4/
  test/
    0/  1/  2/  3/  4/
```

Images per split and grade (grade 5 excluded):

| Split | 0 | 1 | 2 | 3 | 4 | Total |
|-------|-----|-----|------|-----|-----|-------|
| train | 3,133 | 315 | 2,238 | 118 | 456 | 6,260 |
| valid | 1,253 | 126 | 895 | 47 | 182 | 2,503 |
| test | 1,880 | 189 | 1,344 | 71 | 275 | 3,759 |

## After preparation

Point the evaluation CLI in the model repository at the output folder:

```bash
uv run dr-evaluate \
    --method knn \
    --encoder checkpoints/epoch_60_encoder.pt \
    --data-path data/DDR-ImageFolder \
    --dataset ddr \
    --device auto \
    --output knn_results_epoch60.json
```

The evaluation uses `train/` as the reference set and `test/` as the query set.
