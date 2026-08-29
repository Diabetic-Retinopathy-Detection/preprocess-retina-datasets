"""Per-channel dataset statistics over the exact consumer transform.

Image pixels are resized to a square of ``resize`` pixels using PIL bilinear
resampling (torchvision ``Resize`` default) and scaled to ``[0, 1]`` by
dividing by ``255``. Mean and standard deviation are computed pixel-weighted
over the whole dataset (every pixel counts once, no per-image averaging).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".tiff", ".tif", ".bmp"})

_VARIANCE_EPSILON = 1e-12


def compute_mean_std(
    image_dir: Path,
    resize: int = 384,
    output: Path | None = None,
) -> dict[str, list[float]]:
    """Compute per-channel mean and std over all images in ``image_dir``.

    Each image is opened, converted to RGB, resized to
    ``(resize, resize)`` with PIL bilinear resampling (matching the
    consumer's ``transforms.Resize`` default), and scaled to ``[0, 1]``.
    Pixel-wise sums and squared sums are accumulated across all images and
    the population mean/std (ddof=0) are derived from those totals.

    Args:
        image_dir: Root directory; images are discovered recursively.
        resize: Target side length in pixels.
        output: If given, write the result as JSON (values rounded to 6
            decimals) to this path.

    Returns:
        ``{"mean": [m0, m1, m2], "std": [s0, s1, s2]}``.

    Raises:
        FileNotFoundError: If ``image_dir`` does not exist.
        ValueError: If ``image_dir`` contains no supported images.
    """
    if not image_dir.is_dir():
        msg = f"Image directory does not exist: {image_dir}"
        raise FileNotFoundError(msg)

    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        msg = f"No images found in {image_dir}"
        raise ValueError(msg)

    total_sum: np.ndarray = np.zeros(3, dtype=np.float64)
    total_sq: np.ndarray = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in images:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        resized = rgb.resize((resize, resize), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float64) / 255.0
        channel_sum = np.asarray(arr.sum(axis=(0, 1)), dtype=np.float64)
        channel_sq = np.asarray((arr**2).sum(axis=(0, 1)), dtype=np.float64)
        total_sum = total_sum + channel_sum
        total_sq = total_sq + channel_sq
        pixel_count += resize * resize

    denom = float(pixel_count)
    mean = total_sum / denom
    variance = total_sq / denom - mean * mean
    variance = np.where(variance < _VARIANCE_EPSILON, 0.0, variance)
    std = np.sqrt(variance)

    result = {"mean": mean.tolist(), "std": std.tolist()}

    if output is not None:
        rounded = {
            "mean": [round(float(v), 6) for v in result["mean"]],
            "std": [round(float(v), 6) for v in result["std"]],
        }
        output.write_text(json.dumps(rounded) + "\n")

    return result
