"""Prepare the IDRiD dataset as a PyTorch ImageFolder for evaluation.

Reads the IDRiD disease grading label CSV files (one for training and one for
testing) and stages the images into an ImageFolder layout::

    IDRiD-ImageFolder/
      train/
        0/  1/  2/  3/  4/
      test/
        0/  1/  2/  3/  4/

The training split is stratified-split into ``train`` and ``valid`` using a
seeded per-grade carve; the test split is used as-is. Images are symlinked
into place by default so the source files are not duplicated; pass
``copy=True`` to copy instead (e.g. for filesystems or transfers where
symlinks are not preserved).
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")

GradeRow = tuple[str, int]


def parse_idrid_csv(csv_path: Path) -> list[GradeRow]:
    """Parse an IDRiD disease grading CSV into ``(image_name, grade)`` pairs.

    The CSV has a header row whose first cell is ``Image name``, which is
    skipped. Image names carry no extension (e.g. ``IDRiD_001``) and rows may
    have many trailing empty fields, which are ignored.

    Args:
        csv_path: Path to an IDRiD grade label CSV.

    Returns:
        List of ``(image_name, grade)`` pairs in file order. Blank rows are
        ignored.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If a non-blank row is missing a grade or the grade is not
            an integer.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        msg = f"File not found: {csv_path}"
        raise FileNotFoundError(msg)

    rows: list[GradeRow] = []
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or not row[0]:
                continue
            if row[0] == "Image name":
                continue
            if len(row) < 2 or not row[1]:
                msg = f"Malformed row in {csv_path}: {','.join(row)}"
                raise ValueError(msg)
            name = row[0].strip()
            try:
                grade = int(row[1])
            except ValueError as exc:
                msg = f"Invalid grade {row[1]!r} in {csv_path}: {','.join(row)}"
                raise ValueError(msg) from exc
            rows.append((name, grade))
    return rows


def carve_valid_split(
    rows: list[GradeRow],
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[GradeRow], list[GradeRow]]:
    """Stratified-split ``rows`` into ``(valid, train)`` by grade.

    A fixed fraction of each grade is reserved for the validation split using
    a seeded permutation, so the result is deterministic for a given
    ``valid_ratio`` and ``seed``.

    Args:
        rows: ``(image_name, grade)`` pairs.
        valid_ratio: Fraction of each grade to reserve for ``valid``.
        seed: Random seed governing the per-grade permutation.

    Returns:
        A ``(valid, train)`` pair of row lists.
    """
    by_grade: dict[int, list[GradeRow]] = {}
    for row in rows:
        by_grade.setdefault(row[1], []).append(row)

    valid: list[GradeRow] = []
    train: list[GradeRow] = []
    for grade in sorted(by_grade):
        group = by_grade[grade]
        k = round(len(group) * valid_ratio)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(group))
        valid.extend(group[i] for i in idx[:k])
        train.extend(group[i] for i in idx[k:])

    return valid, train


def prepare_idrid(
    train_image_dir: Path,
    test_image_dir: Path,
    train_labels_csv: Path,
    test_labels_csv: Path,
    output_dir: Path,
    valid_ratio: float = 0.2,
    seed: int = 42,
    copy: bool = False,
) -> dict[str, tuple[int, dict[int, int]]]:
    """Stage IDRiD images into a grade-labelled ImageFolder layout.

    Args:
        train_image_dir: Directory of the training images (``<name>.jpg``).
        test_image_dir: Directory of the test images (``<name>.jpg``).
        train_labels_csv: IDRiD training label CSV.
        test_labels_csv: IDRiD test label CSV.
        output_dir: Target directory for the ImageFolder structure.
        valid_ratio: Fraction of each training grade to reserve for ``valid``.
        seed: Random seed for the stratified validation carve.
        copy: If ``True``, copy image files instead of creating symlinks.

    Returns:
        Mapping of ``split`` to ``(staged, {grade: count})``.

    Raises:
        FileNotFoundError: If an input path does not exist.
        ValueError: If a labelled image has no matching file on disk, or an
            image on disk has no matching label.
    """
    _check_paths(train_image_dir, test_image_dir, train_labels_csv, test_labels_csv)

    train_rows = parse_idrid_csv(train_labels_csv)
    test_rows = parse_idrid_csv(test_labels_csv)

    valid_rows, train_rows_split = carve_valid_split(train_rows, valid_ratio=valid_ratio, seed=seed)

    _check_consistency(train_image_dir, "train", {name for name, _ in train_rows})
    _check_consistency(test_image_dir, "test", {name for name, _ in test_rows})

    summary: dict[str, tuple[int, dict[int, int]]] = {}
    summary["train"] = _stage(train_image_dir, train_rows_split, "train", output_dir, copy)
    summary["valid"] = _stage(train_image_dir, valid_rows, "valid", output_dir, copy)
    summary["test"] = _stage(test_image_dir, test_rows, "test", output_dir, copy)

    return summary


def _check_paths(
    train_image_dir: Path,
    test_image_dir: Path,
    train_labels_csv: Path,
    test_labels_csv: Path,
) -> None:
    for path, kind in (
        (train_image_dir, "Directory"),
        (test_image_dir, "Directory"),
        (train_labels_csv, "File"),
        (test_labels_csv, "File"),
    ):
        if not path.exists():
            msg = f"{kind} does not exist: {path}"
            raise FileNotFoundError(msg)


def _check_consistency(image_dir: Path, dir_label: str, labelled: set[str]) -> None:
    on_disk = {p.stem for p in image_dir.iterdir() if p.suffix.lower() == ".jpg"}
    missing = sorted(labelled - on_disk)
    if missing:
        n = len(missing)
        sample = ", ".join(missing[:5])
        part = f"{dir_label}: {n} labelled image(s) missing a file on disk:\n  {sample}"
        if n > 5:
            part += f"\n  (showing first 5 of {n} missing names)"
        raise ValueError(part)
    orphans = sorted(on_disk - labelled)
    if orphans:
        n = len(orphans)
        sample = ", ".join(orphans[:5])
        part = f"{dir_label}: {n} image(s) on disk without a label:\n  {sample}"
        if n > 5:
            part += f"\n  (showing first 5 of {n} orphan names)"
        raise ValueError(part)


def _stage(
    image_dir: Path,
    rows: list[GradeRow],
    split: str,
    output_dir: Path,
    copy: bool,
) -> tuple[int, dict[int, int]]:
    staged = 0
    per_grade: dict[int, int] = {}
    for name, grade in rows:
        src = image_dir / f"{name}.jpg"
        dst_dir = output_dir / split / str(grade)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{name}.jpg"
        if not dst.exists():
            if copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())
        staged += 1
        per_grade[grade] = per_grade.get(grade, 0) + 1
    return staged, per_grade
