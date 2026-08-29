"""Prepare the DDR dataset as a PyTorch ImageFolder for evaluation.

Reads the DDR ``DR_grading`` split files (``train.txt``, ``valid.txt``,
``test.txt``, one ``<filename> <grade>`` pair per line) and stages the
images into an ImageFolder layout::

    DDR-ImageFolder/
      train/
        0/  1/  2/  3/  4/
      valid/
        0/  1/  2/  3/  4/
      test/
        0/  1/  2/  3/  4/

Images are symlinked into place by default so the source files are not
duplicated; pass ``copy=True`` to copy instead (e.g. for filesystems or
transfers where symlinks are not preserved).
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

SPLITS = ("train", "valid", "test")

MIN_GRADE = 0
MAX_GRADE = 4
DEFAULT_EXCLUDE_GRADES = frozenset({5})


def parse_grading_txt(txt_path: Path) -> list[tuple[str, int]]:
    """Parse a DDR grading file into ``(filename, grade)`` pairs.

    Args:
        txt_path: Path to a ``<split>.txt`` file, one ``<filename> <grade>``
            pair per line.

    Returns:
        List of ``(filename, grade)`` pairs in file order. Blank lines are
        ignored.

    Raises:
        FileNotFoundError: If ``txt_path`` does not exist.
        ValueError: If a line is not a ``<filename> <grade>`` pair or the
            grade is not an integer.
    """
    lines = txt_path.read_text().splitlines()
    pairs: list[tuple[str, int]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            msg = f"Malformed line in {txt_path}: {line!r}"
            raise ValueError(msg)
        filename, grade_str = parts
        try:
            grade = int(grade_str)
        except ValueError as exc:
            msg = f"Invalid grade {grade_str!r} in {txt_path}: {line!r}"
            raise ValueError(msg) from exc
        pairs.append((filename, grade))
    return pairs


def prepare_ddr(
    input_dir: Path,
    output_dir: Path,
    exclude_grades: frozenset[int] = DEFAULT_EXCLUDE_GRADES,
    copy: bool = False,
) -> dict[str, tuple[int, int]]:
    """Stage DDR images into a grade-labelled ImageFolder layout.

    Args:
        input_dir: Directory containing ``{split}.txt`` grading files and the
            matching per-split image folders (the DDR ``DR_grading`` root).
        output_dir: Target directory for the ImageFolder structure.
        exclude_grades: Grades to skip. Unreadable images (grade 5) are
            excluded by default, matching SSiT's 5-class evaluation. Grades
            outside the valid range 0-4 are always excluded.
        copy: If ``True``, copy image files instead of creating symlinks.

    Returns:
        Mapping of ``split`` to ``(staged, excluded)`` counts.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.

    Warns:
        UserWarning: If a split grading file (``{split}.txt``) is missing.
    """
    if not input_dir.is_dir():
        msg = f"Directory does not exist: {input_dir}"
        raise FileNotFoundError(msg)

    summary: dict[str, tuple[int, int]] = {}
    for split in SPLITS:
        txt_path = input_dir / f"{split}.txt"
        if not txt_path.exists():
            warnings.warn(f"Split file not found, skipping: {txt_path}", UserWarning, stacklevel=2)
            continue

        image_dir = input_dir / split
        staged = 0
        excluded = 0
        for filename, grade in parse_grading_txt(txt_path):
            if grade in exclude_grades or not MIN_GRADE <= grade <= MAX_GRADE:
                excluded += 1
                continue

            src = image_dir / filename
            dst_dir = output_dir / split / str(grade)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / filename

            if not dst.exists():
                if copy:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.resolve())
            staged += 1

        summary[split] = (staged, excluded)

    return summary
