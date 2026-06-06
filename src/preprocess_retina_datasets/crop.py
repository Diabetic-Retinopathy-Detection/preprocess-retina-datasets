"""Crop retina fundus images by detecting and extracting the retinal region.

This module provides functionality to crop retinal fundus images by detecting
the bounding box of the retina region (removing dark borders) and resizing to
a uniform square. It supports both single-image and batch processing.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from preprocess_retina_datasets.errors import ImageProcessingError


def detect_retina_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Detect the bounding box of the retina region in a fundus image.

    The image is blurred with a Gaussian filter, thresholded to separate
    foreground (retina) from background (dark borders), and the bounding
    box of the foreground is computed.

    Args:
        image: Input PIL Image.

    Returns:
        Bounding box ``(left, upper, right, lower)`` in pixel coordinates,
        or ``None`` if the retina region cannot be detected.
    """
    blurred = image.filter(ImageFilter.BLUR)
    array = np.array(blurred)
    h, w, _ = array.shape

    if w <= 1.2 * h:
        return None

    left_edge = array[:, : w // 32, :].max(axis=(0, 1)).astype(int)
    right_edge = array[:, -w // 32 :, :].max(axis=(0, 1)).astype(int)
    background = np.maximum(left_edge, right_edge)

    mask = (array > background + 10).astype(np.uint8)
    mask_image = Image.fromarray(mask)
    bbox = mask_image.getbbox()

    if bbox is None:
        return None

    left, upper, right, lower = bbox
    if right - left < 0.8 * h or lower - upper < 0.8 * h:
        return None

    return bbox


def square_center_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """Return the largest centered square crop of the image.

    Args:
        image: Input PIL Image.

    Returns:
        Bounding box ``(left, upper, right, lower)`` for a centered square crop.
    """
    w, h = image.size
    if w > h:
        offset = (w - h) // 2
        return (offset, 0, w - offset, h)
    offset = (h - w) // 2
    return (0, offset, w, h - offset)


def crop_image(
    src_path: Path,
    dst_path: Path,
    crop_size: int = 512,
    skip_existing: bool = False,
) -> None:
    """Crop a single retina image to a uniform square.

    The image is opened, the retina bounding box is detected (with a
    centered-square fallback), the crop is applied, and the result is
    resized to ``(crop_size, crop_size)`` and saved.

    Args:
        src_path: Path to the source image.
        dst_path: Path to save the cropped image.
        crop_size: Target size in pixels (width and height).
        skip_existing: If True, skip processing if ``dst_path`` already exists.
    """
    if skip_existing and dst_path.exists():
        return

    try:
        image = Image.open(src_path).convert("RGB")

        bbox = detect_retina_bbox(image)
        if bbox is None:
            bbox = square_center_bbox(image)

        cropped = image.crop(bbox)
        resized = cropped.resize((crop_size, crop_size), Image.Resampling.LANCZOS)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst_path, quality=100, subsampling=0)
    except Exception as exc:
        msg = f"Failed to process {src_path}: {exc}"
        raise ImageProcessingError(msg) from exc


def crop_dataset(
    input_dir: Path,
    output_dir: Path,
    crop_size: int = 512,
    num_workers: int = 8,
    skip_existing: bool = False,
) -> int:
    """Crop all images in ``input_dir`` and write them to ``output_dir``.

    The directory structure under ``input_dir`` is preserved. Processing is
    parallelised across ``num_workers`` processes.

    Args:
        input_dir: Root directory of input images.
        output_dir: Root directory for cropped output images.
        crop_size: Target size in pixels (width and height).
        num_workers: Number of parallel worker processes.
        skip_existing: If True, skip images where the output already exists.

    Returns:
        Number of images that failed to process.
    """
    if not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise FileNotFoundError(msg)

    extensions = frozenset({".jpeg", ".jpg", ".png", ".tiff", ".tif", ".bmp"})
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]

    if not images:
        return 0

    jobs: list[tuple[Path, Path]] = []
    for src in images:
        dst = output_dir / src.relative_to(input_dir)
        if skip_existing and dst.exists():
            continue
        jobs.append((src, dst))

    if not jobs:
        return 0

    num_workers = min(num_workers, len(jobs))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for src, dst in jobs:
            futures[executor.submit(crop_image, src, dst, crop_size, skip_existing)] = src

        failed = 0
        with tqdm(total=len(jobs), desc="Cropping images", unit="img") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except ImageProcessingError:
                    failed += 1
                pbar.update(1)

    return failed
