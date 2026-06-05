# preprocess-retina-datasets

[![Release](https://img.shields.io/github/v/release/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)](https://img.shields.io/github/v/release/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)
[![Build status](https://img.shields.io/github/actions/workflow/status/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/main.yml?branch=main)](https://github.com/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/branch/main/graph/badge.svg)](https://codecov.io/gh/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)
[![License](https://img.shields.io/github/license/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)](https://img.shields.io/github/license/Diabetic-Retinopathy-Detection/preprocess-retina-datasets)

Preprocessing tools for retinal fundus image datasets. Converts raw datasets into a uniform structure for downstream training and evaluation.

- **GitHub**: <https://github.com/Diabetic-Retinopathy-Detection/preprocess-retina-datasets/>
- **Documentation**: <https://Meier-Stefan.github.io/preprocess-retina-datasets/>

## Tools

### `crop-images` — Bounding-box crop and resize

Removes dark borders from retinal fundus images and resizes to a uniform square. The image is blurred, thresholded to create a foreground mask, and the bounding box of the retina region is extracted. Falls back to a centered square crop if the retina region cannot be detected.

```bash
crop-images \
    --image-folder /path/to/raw/images \
    --output-folder /path/to/cropped \
    --crop-size 512 \
    -n 8
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-folder` | (required) | Input folder of images (subfolders preserved) |
| `--output-folder` | (required) | Output folder for cropped images |
| `--crop-size` | `512` | Target size in pixels |
| `-n` / `--num-workers` | `8` | Number of parallel worker processes |

## Install

```bash
make install
```

## Development

```bash
make check     # lint, type-check, deptry
make test      # run tests with coverage
```
