"""Crop retina fundus images by detecting and extracting the retinal region.

This module provides functionality to crop retinal fundus images by detecting
the bounding box of the retina region (removing dark borders) and resizing to
a uniform square. It supports both single-image and batch processing.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


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
) -> None:
    """Crop a single retina image to a uniform square.

    The image is opened, the retina bounding box is detected (with a
    centered-square fallback), the crop is applied, and the result is
    resized to ``(crop_size, crop_size)`` and saved.

    Args:
        src_path: Path to the source image.
        dst_path: Path to save the cropped image.
        crop_size: Target size in pixels (width and height).
    """
    image = Image.open(src_path).convert("RGB")

    bbox = detect_retina_bbox(image)
    if bbox is None:
        bbox = square_center_bbox(image)

    cropped = image.crop(bbox)
    resized = cropped.resize((crop_size, crop_size), Image.Resampling.LANCZOS)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    resized.save(dst_path, quality=100, subsampling=0)


def _collect_jobs(
    input_dir: Path,
    output_dir: Path,
    crop_size: int,
) -> list[tuple[Path, Path, int]]:
    """Walk ``input_dir`` and return a list of ``(src, dst, crop_size)`` tuples.

    The directory structure under ``input_dir`` is preserved in ``output_dir``.
    """
    jobs: list[tuple[Path, Path, int]] = []
    for root_str, _dirs, files in os.walk(input_dir):
        root = Path(root_str)
        for file in files:
            if file.lower().endswith((".jpeg", ".jpg", ".png", ".tiff", ".tif", ".bmp")):
                src_path = root / file
                rel = root.relative_to(input_dir)
                dst_path = output_dir / rel / file
                jobs.append((src_path, dst_path, crop_size))
    return jobs


def _worker(args: tuple[Path, Path, int]) -> str | None:
    """Worker function for multiprocessing pool.

    Args:
        args: Tuple of ``(src_path, dst_path, crop_size)``.

    Returns:
        ``str`` representation of the source path on success, or ``None`` on
        failure.
    """
    src_path, dst_path, crop_size = args
    try:
        crop_image(src_path, dst_path, crop_size)
        return str(src_path)
    except Exception:
        return None


def crop_dataset(
    input_dir: Path,
    output_dir: Path,
    crop_size: int = 512,
    num_workers: int = 8,
) -> None:
    """Crop all images in ``input_dir`` and write them to ``output_dir``.

    The directory structure under ``input_dir`` is preserved. Processing is
    parallelised across ``num_workers`` processes.

    Args:
        input_dir: Root directory of input images.
        output_dir: Root directory for cropped output images.
        crop_size: Target size in pixels (width and height).
        num_workers: Number of parallel worker processes.
    """
    jobs = _collect_jobs(input_dir, output_dir, crop_size)
    if not jobs:
        return

    num_workers = min(num_workers, len(jobs))

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_worker, jobs),
                total=len(jobs),
                desc="Cropping images",
                unit="img",
            )
        )

    failed = [r for r in results if r is None]
    if failed:
        print(f"Warning: {len(failed)} / {len(jobs)} images failed to process.")


def main() -> None:
    """Entry point for the ``crop-images`` CLI.

    Parses arguments and runs ``crop_dataset``.
    """
    parser = argparse.ArgumentParser(
        description="Crop retina fundus images to uniform square size.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to the input folder containing images.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder for cropped images.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Target crop size in pixels (default: 512).",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    args = parser.parse_args()

    crop_dataset(
        input_dir=Path(args.image_folder),
        output_dir=Path(args.output_folder),
        crop_size=args.crop_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
