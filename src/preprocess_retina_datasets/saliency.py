"""Generate saliency maps for retinal fundus images.

Applies a preprocessing pipeline (unsharp masking, circular fundus mask,
JPEG simulation) and then runs OpenCV's :class:`cv2.saliency.StaticSaliencyFineGrained`
detector.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import NoReturn

import cv2
import numpy as np
from tqdm import tqdm

from preprocess_retina_datasets.errors import ImageProcessingError

_PREPROCESS_CIRCLE_FACTOR = 0.98
_FINAL_CIRCLE_FACTOR = 240 / 256


def _raise_error(msg: str) -> NoReturn:
    raise ImageProcessingError(msg)


def _preprocess(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(h, w)

    mask = np.zeros_like(image)
    center = (w // 2, h // 2)
    radius = int(scale / 2 * _PREPROCESS_CIRCLE_FACTOR)
    cv2.circle(mask, center, radius, (1, 1, 1), -1)

    blurred = cv2.GaussianBlur(image, (0, 0), scale / 30)
    weighted = cv2.addWeighted(image, 4, blurred, -4, 128)
    processed = weighted * mask + 128 * (1 - mask)
    processed = processed.astype(np.uint8)

    # JPEG encode/decode simulates lossy compression, making the pipeline
    # robust to JPEG artifacts commonly present in fundus datasets.
    _, jpeg = cv2.imencode(".jpeg", processed)
    decoded = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
    if decoded is None:
        _raise_error("JPEG decode failed during preprocessing")
    return decoded


def _build_final_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    radius = int(max(h, w) / 2 * _FINAL_CIRCLE_FACTOR)
    cv2.circle(mask, center, radius, 1.0, -1)
    return mask


def generate_saliency_map(
    src_path: Path,
    dst_path: Path,
    *,
    skip_existing: bool = False,
) -> None:
    """Generate a saliency map for a single fundus image.

    The image is preprocessed with unsharp masking and a circular mask,
    then passed through OpenCV's :class:`StaticSaliencyFineGrained`
    detector. The output is a float32 array in ``[0, 1]``, masked to the
    retinal disk and saved as a ``.npy`` file.

    Args:
        src_path: Path to the input image.
        dst_path: Path to save the ``.npy`` saliency map.
        skip_existing: If True, skip processing if ``dst_path`` already exists.
    """
    if skip_existing and dst_path.exists():
        return

    try:
        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            _raise_error(f"Failed to read image: {src_path}")

        processed = _preprocess(image)

        saliency = cv2.saliency.StaticSaliencyFineGrained_create()  # type: ignore[attr-defined]
        success, raw_map = saliency.computeSaliency(processed)
        if not success:
            _raise_error(f"Saliency detection returned failure for {src_path}")

        h, w = raw_map.shape
        final_mask = _build_final_mask(h, w)
        raw_map *= final_mask

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(dst_path), raw_map)
    except ImageProcessingError:
        raise
    except Exception as exc:
        msg = f"Failed to process {src_path}: {exc}"
        raise ImageProcessingError(msg) from exc


def _collect_jobs(
    input_dir: Path,
    output_dir: Path,
    *,
    skip_existing: bool = False,
) -> list[tuple[Path, Path]]:
    extensions = frozenset({".jpeg", ".jpg", ".png", ".tiff", ".tif", ".bmp"})
    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in extensions]

    jobs: list[tuple[Path, Path]] = []
    for src in images:
        dst = output_dir / src.relative_to(input_dir)
        dst = dst.with_suffix(".npy")
        if skip_existing and dst.exists():
            continue
        jobs.append((src, dst))
    return jobs


def _execute_jobs(
    jobs: list[tuple[Path, Path]],
    num_workers: int,
    *,
    skip_existing: bool = False,
    show_progress: bool = True,
) -> int:
    num_workers = min(num_workers, len(jobs))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for src, dst in jobs:
            futures[executor.submit(generate_saliency_map, src, dst, skip_existing=skip_existing)] = src

        failed = 0
        cm: object = (
            tqdm(total=len(jobs), desc="Generating saliency maps", unit="img") if show_progress else nullcontext()
        )
        with cm as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except ImageProcessingError:
                    failed += 1
                if show_progress:
                    pbar.update(1)

    return failed


def generate_saliency_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    num_workers: int = 8,
    skip_existing: bool = False,
    show_progress: bool = True,
) -> int:
    """Generate saliency maps for all images in ``input_dir``.

    The directory structure under ``input_dir`` is preserved. Each image
    gets a corresponding ``.npy`` file with the same relative path and
    stem. Processing is parallelised across ``num_workers`` processes.

    Args:
        input_dir: Root directory of input images (cropped).
        output_dir: Root directory for saliency map output.
        num_workers: Number of parallel worker processes.
        skip_existing: If True, skip images where the output already exists.
        show_progress: If True, display a progress bar via tqdm.

    Returns:
        Number of images that failed to process.
    """
    if not input_dir.is_dir():
        msg = f"Input directory does not exist: {input_dir}"
        raise FileNotFoundError(msg)

    if num_workers < 1:
        msg = f"num_workers must be >= 1, got {num_workers}"
        raise ValueError(msg)

    jobs = _collect_jobs(input_dir, output_dir, skip_existing=skip_existing)
    if not jobs:
        return 0

    return _execute_jobs(jobs, num_workers, skip_existing=skip_existing, show_progress=show_progress)
