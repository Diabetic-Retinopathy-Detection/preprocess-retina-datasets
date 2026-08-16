# preprocess-retina-datasets

Preprocessing tools for retinal fundus image datasets. Converts raw datasets into a uniform structure for downstream training and evaluation.

- **GitHub**: <https://github.com/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/>
- **Documentation**: <https://Meier-Stefan.github.io/preprocess-retina-datasets/>

## How to use

### Set up DDR for finetuning

For fine-tuning, crop the raw DDR images before creating the labelled
ImageFolder. The crop tool removes dark borders, extracts the retinal region,
and resizes each image to a uniform square. The commands below keep the raw
dataset unchanged and create separate cropped output directories:

```bash
RAW=data/DDR-dataset/DR_grading
CROPPED=data/DDR-dataset-cropped/DR_grading

for split in train valid test; do
    crop-images \
        --image-folder "${RAW}/${split}" \
        --output-folder "${CROPPED}/${split}" \
        --crop-size 512 \
        --skip-existing \
        --num-workers 8
    cp "${RAW}/${split}.txt" "${CROPPED}/${split}.txt"
done
```

Then create the grade-labelled ImageFolder using real file copies. Using
`--copy` avoids broken absolute symlinks when the dataset is transferred to a
cluster or another machine:

```bash
prepare-ddr \
    --input data/DDR-dataset-cropped/DR_grading \
    --output data/DDR-ImageFolder-cropped \
    --copy
```

The resulting layout is:

```text
DDR-ImageFolder-cropped/
  train/0/  train/1/  train/2/  train/3/  train/4/
  valid/0/  valid/1/  valid/2/  valid/3/  valid/4/
  test/0/   test/1/   test/2/   test/3/   test/4/
```

`--skip-existing` makes the crop step safe to rerun after an interruption.
Saliency maps and a dataset index are not required for supervised DDR
fine-tuning; those are used by the self-supervised pretraining pipeline.

## Tools

### `crop-images` — Bounding-box crop and resize

Removes dark borders from retinal fundus images and resizes to a uniform square. The image is blurred, thresholded to create a foreground mask, and the bounding box of the retina region is extracted. Falls back to a centered square crop if the retina region cannot be detected.

```bash
crop-images \
    --image-folder data/EyePACS-Kaggle/diabetic-retinopathy-detection/extracted/train \
    --output-folder data/cropped \
    --crop-size 512 \
    --skip-existing \
    -n 8
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-folder` | (required) | Input folder of images (subfolders preserved) |
| `--output-folder` | (required) | Output folder for cropped images |
| `--crop-size` | `512` | Target size in pixels |
| `-n` / `--num-workers` | `8` | Number of parallel worker processes |
| `--skip-existing` | off | Skip images where the output file already exists (for resume) |

### `detect-saliency` — Generate saliency maps for training

Takes cropped retina fundus images (output of `crop-images`) and produces saliency maps as `.npy` files (float32). The output preserves the input directory structure and is designed for direct use in training pipelines that expect per-image `.npy` saliency maps.

The preprocessing pipeline applies unsharp masking, a circular fundus mask, and JPEG simulation before running OpenCV's StaticSaliencyFineGrained detector.

```bash
detect-saliency \
    --image-folder data/cropped/train \
    --output-folder data/saliency/train \
    --skip-existing \
    -n 8
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-folder` | (required) | Input folder of cropped images (subfolders preserved) |
| `--output-folder` | (required) | Output folder for `.npy` saliency maps |
| `-n` / `--num-workers` | `8` | Number of parallel worker processes |
| `--skip-existing` | off | Skip images where the output file already exists (for resume) |

### `build-dataset-index` — Build a dataset index pickle

Pairs cropped images with their saliency maps by relative path key and writes a pickle containing `list[tuple[Path, Path]]` of relative paths. Requires that every image has a matching saliency map and vice versa; raises an error on mismatch with a bounded sample of the missing/extra keys.

```bash
build-dataset-index \
    --image-folder data/cropped/train \
    --saliency-folder data/saliency/train \
    --output-file data/data_index/train.pkl
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-folder` | (required) | Folder of cropped images |
| `--saliency-folder` | (required) | Folder of `.npy` saliency maps |
| `--output-file` | (required) | Output pickle file path |

### `prepare-ddr` — Build a grade-labelled DDR ImageFolder

Reads the DDR `DR_grading` split files (`train.txt`, `valid.txt`, `test.txt`, one `<filename> <grade>` pair per line) and stages the images into a PyTorch ImageFolder layout `output/{split}/{grade}/`. Images are symlinked into place by default so the source files are not duplicated; pass `--copy` on filesystems or for transfers where symlinks are not preserved.

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

### `visualise_saliency.py` — Visually inspect saliency maps

Utility script that shows an image, its saliency map (hot colormap), and a
semi-transparent overlay side by side. Useful for qualitative checks after
running `detect-saliency`.

```bash
uv run python scripts/visualise_saliency.py \
    data/cropped/train/10_left.jpeg \
    data/saliency/train/10_left.npy
```

### `visualise_saliency_thresholds.py` — Compare saliency thresholds

Shows the image, the raw saliency map, and the binary saliency mask overlaid on
the image at each of two thresholds (default `0.25` and `0.5`), with the
percentage of pixels passing each threshold in the panel title. Useful to see
how the BCE segmentation target changes with `saliency_threshold` in pretraining.

```bash
uv run python scripts/visualise_saliency_thresholds.py \
    data/cropped/train/10_left.jpeg \
    data/saliency/train/10_left.npy \
    --output saliency_thresholds.png
```

| Argument | Description |
|----------|-------------|
| `<image>` | Cropped fundus image |
| `<saliency.npy>` | Corresponding saliency map |
| `--output` | Optional path to save the figure (defaults to an interactive window) |

## Dataset preparation

The EyePACS Kaggle dataset is distributed as split zip archives. Extract them first:

```bash
bash scripts/extract_eyepacs.sh
```

See [`docs/extraction.md`](docs/extraction.md) for details.

The DDR dataset is distributed as a multi-part zip archive. Extract and stage it into an ImageFolder:

```bash
bash scripts/extract_ddr.sh
prepare-ddr --input data/DDR-dataset/DR_grading --output data/DDR-ImageFolder
```

See [`docs/ddr.md`](docs/ddr.md) for details.

## Install

```bash
make install
```

## Development

```bash
make check     # lint, type-check, deptry
make test      # run tests with coverage
```
