# preprocess-retina-datasets

Preprocessing tools for retinal fundus image datasets. Converts raw datasets into a uniform structure for downstream training and evaluation.

- **GitHub**: <https://github.com/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/>
- **Documentation**: <https://Meier-Stefan.github.io/preprocess-retina-datasets/>

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

### `detect-saliency` — Generate saliency maps

Generates saliency maps using OpenCV's StaticSaliencyFineGrained algorithm with a preprocessing pipeline (unsharp masking, circular fundus mask, JPEG simulation). Outputs `.npy` files (float32) at the same relative paths as input.

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

## Dataset preparation

The EyePACS Kaggle dataset is distributed as split zip archives. Extract them first:

```bash
bash scripts/extract_eyepacs.sh
```

See [`docs/extraction.md`](docs/extraction.md) for details.

## Install

```bash
make install
```

## Development

```bash
make check     # lint, type-check, deptry
make test      # run tests with coverage
```
