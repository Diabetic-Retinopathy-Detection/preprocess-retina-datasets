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

### `package-dataset` — Build a dataset index pickle

Pairs cropped images with their saliency maps by relative path key and writes a pickle containing `list[tuple[Path, Path]]` of absolute paths. Requires that every image has a matching saliency map and vice versa; raises an error on mismatch with a bounded sample of the missing/extra keys.

```bash
package-dataset \
    --image-folder data/cropped/train \
    --saliency-folder data/saliency/train \
    --output-file data/data_index/train.pkl
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--image-folder` | (required) | Folder of cropped images |
| `--saliency-folder` | (required) | Folder of `.npy` saliency maps |
| `--output-file` | (required) | Output pickle file path |

### `visualise_saliency.py` — Visually inspect saliency maps

Utility script that shows an image, its saliency map (hot colormap), and a
semi-transparent overlay side by side. Useful for qualitative checks after
running `detect-saliency`.

```bash
uv run python scripts/visualise_saliency.py \
    data/cropped/train/10_left.jpeg \
    data/saliency/train/10_left.npy
```

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
