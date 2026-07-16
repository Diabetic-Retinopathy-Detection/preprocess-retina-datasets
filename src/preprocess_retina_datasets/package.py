"""Build a dataset index by pairing images with their saliency maps.

This module collects images and saliency maps from two directories,
matches them by relative path key, and writes a pickle file containing
a list of ``(image_path, saliency_path)`` tuples with paths relative
to their respective input directories.
"""

from __future__ import annotations

import pickle
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".tiff", ".tif", ".bmp"})


def _collect_files_by_key(root: Path, allowed_suffixes: frozenset[str]) -> dict[str, Path]:
    if not root.is_dir():
        msg = f"Directory does not exist: {root}"
        raise FileNotFoundError(msg)

    files: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in allowed_suffixes:
            key = str(p.relative_to(root).with_suffix(""))
            files[key] = p.absolute()
    return files


def _compare_keys(images: dict[str, Path], saliency: dict[str, Path]) -> None:
    image_keys = set(images)
    saliency_keys = set(saliency)

    missing_saliency = sorted(image_keys - saliency_keys)
    extra_saliency = sorted(saliency_keys - image_keys)

    parts: list[str] = []
    if missing_saliency:
        n = len(missing_saliency)
        sample = ", ".join(missing_saliency[:5])
        part = f"{n} image(s) missing a saliency map:\n  {sample}"
        if n > 5:
            part += f"\n  (showing first 5 of {n} missing keys)"
        parts.append(part)
    if extra_saliency:
        n = len(extra_saliency)
        sample = ", ".join(extra_saliency[:5])
        part = f"{n} saliency map(s) without a matching image:\n  {sample}"
        if n > 5:
            part += f"\n  (showing first 5 of {n} extra keys)"
        parts.append(part)

    if parts:
        raise ValueError("\n".join(parts))


def _build_pairs(
    images: dict[str, Path],
    saliency: dict[str, Path],
) -> list[tuple[Path, Path]]:
    keys = sorted(images)
    return [(images[k], saliency[k]) for k in keys]


def build_dataset_index(
    image_dir: Path,
    saliency_dir: Path,
    output_file: Path,
) -> None:
    """Pair images with their saliency maps and write a dataset index pickle.

    The pickle contains ``list[tuple[Path, Path]]`` where each tuple is
    ``(image_path, saliency_path)`` stored **relative** to its respective
    input directory.  The consumer resolves paths by joining its own data
    directory with these relative paths.

    Args:
        image_dir: Directory containing cropped images.
        saliency_dir: Directory containing saliency ``.npy`` files.
        output_file: Path for the output pickle file.

    Raises:
        FileNotFoundError: If either input directory does not exist.
        ValueError: If the set of image keys does not match the set of
            saliency keys.
    """
    images = _collect_files_by_key(image_dir, IMAGE_SUFFIXES)
    saliency = _collect_files_by_key(saliency_dir, frozenset({".npy"}))

    _compare_keys(images, saliency)

    pairs = _build_pairs(images, saliency)
    pairs = [(p.relative_to(image_dir), s.relative_to(saliency_dir)) for p, s in pairs]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pickle.dump(pairs, f)
