# preprocess-retina-datasets

Preprocessing tools for retinal fundus image datasets. Converts raw datasets into a uniform structure for downstream training and evaluation.

- **GitHub**: <https://github.com/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/>

## How to use

### Set up EyePACS for pretraining

Place the multipart EyePACS archives in:

```text
data/EyePACS-Kaggle/diabetic-retinopathy-detection/
  train.zip.001 ... train.zip.005
  test.zip.001  ... test.zip.007
```

Install the project dependencies, then extract both splits. Python 3.10 or
newer is required. The extraction scripts additionally require `p7zip`, and
saliency detection uses `opencv-contrib-python-headless`.

```bash
make install
bash scripts/extract_eyepacs.sh
```

Crop the extracted training images before pretraining:

```bash
crop-images \
    --image-folder data/EyePACS-Kaggle/diabetic-retinopathy-detection/extracted/train \
    --output-folder data/cropped/train \
    --crop-size 512 \
    --skip-existing \
    --num-workers 8
```

Generate saliency maps for the cropped training images:

```bash
detect-saliency \
    --image-folder data/cropped/train \
    --output-folder data/saliency/train \
    --skip-existing \
    --num-workers 8
```

Build the image/saliency index used by the pretraining datamodule:

```bash
build-dataset-index \
    --image-folder data/cropped/train \
    --saliency-folder data/saliency/train \
    --output-file data/data_index/train.pkl
```

The pretraining-ready files are then:

```text
data/cropped/train/          # cropped EyePACS images
data/saliency/train/         # matching .npy saliency maps
data/data_index/train.pkl    # paired image/saliency index
```

The test images are extracted by `extract_eyepacs.sh` but are not needed for
self-supervised pretraining. Crop them separately if they are needed for a
later evaluation:

```bash
crop-images \
    --image-folder data/EyePACS-Kaggle/diabetic-retinopathy-detection/extracted/test \
    --output-folder data/cropped/test \
    --crop-size 512 \
    --skip-existing \
    --num-workers 8
```

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

The project does not download or redistribute datasets. Obtain each dataset
from its authoritative source and follow its licence, Kaggle terms, and
redistribution restrictions.

## Tools

### `crop-images` — Bounding-box crop and resize

Removes dark borders from retinal fundus images and resizes to a uniform RGB
square. The image is blurred, thresholded to create a foreground mask, and the
bounding box of the retina region is extracted. It falls back to a centered
square crop if the retina region cannot be detected. Output files retain the
input suffix; metadata is not preserved by the Pillow conversion. Supported
input extensions are `.jpeg`, `.jpg`, `.png`, `.tiff`, `.tif`, and `.bmp`.

```bash
crop-images \
    --image-folder data/EyePACS-Kaggle/diabetic-retinopathy-detection/extracted/train \
    --output-folder data/cropped/train \
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

Takes cropped retina fundus images (output of `crop-images`) and produces
saliency maps as `.npy` files (`float32`). The output preserves the input
directory structure and replaces the image suffix with `.npy`. Maps have the
processed image dimensions, values in `[0, 1]`, and zeros outside the circular
fundus mask.

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

Pairs cropped images with their saliency maps by relative path key and writes a
pickle containing `list[tuple[Path, Path]]` of relative paths. Each image path
is relative to `--image-folder` and each saliency path is relative to
`--saliency-folder`; they are not relative to one common root. Relative
subdirectories and filename stems must match, with the image suffix changed to
`.npy` for saliency maps. Requires that every image has a matching saliency map
and vice versa; raises an error on mismatch with a bounded sample of the
missing/extra keys. Files with the same stem and different image extensions can
collide during matching, so avoid such layouts.

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

Reads the DDR `DR_grading` split files (`train.txt`, `valid.txt`, `test.txt`,
one `<filename> <grade>` pair per line) and stages the images into a PyTorch
ImageFolder layout `output/{split}/{grade}/`. DDR emits `valid/`; the model
repository accepts both `valid/` and `val/` as the validation directory.
Images are symlinked into place by default so the source files are not
duplicated; pass `--copy` when symlinks will not be preserved. Grades outside
`0` through `4` are excluded; `--exclude-grade` defaults to `5`.

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

The DDR dataset is distributed as a multi-part zip archive. Place the archive
parts under `data/DDR-dataset/ZIP/`, extract them, and stage the result into an
ImageFolder:

```bash
bash scripts/extract_ddr.sh
prepare-ddr --input data/DDR-dataset/DR_grading --output data/DDR-ImageFolder
```

See [`docs/ddr.md`](docs/ddr.md) for details.

## Using the outputs with the model repository

For pretraining, configure `diabetic-retinopathy-model` with:

```yaml
data_dir: ../data
data_index_path: ../data/data_index/train.pkl
```

The shared data root must contain `cropped/train/` and `saliency/train/`.
For DDR fine-tuning, set `finetune_dataset_root` to the directory produced by
`prepare-ddr`; it contains `train/`, `valid/`, and `test/` ImageFolder splits.

## Install

```bash
make install
```

The extraction scripts require the `7z` command. On macOS install it with
`brew install p7zip`; on Debian or Ubuntu use `apt install p7zip`.

## Development

```bash
make check     # lint, type-check, deptry
make test      # run tests with coverage
```
