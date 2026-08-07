"""CLI for :mod:`preprocess_retina_datasets.ddr`.

Parses command-line arguments and delegates to the library functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from preprocess_retina_datasets.ddr import prepare_ddr


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``prepare-ddr`` console script.

    Args:
        argv: Argument list (defaults to :data:`sys.argv`).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(
        description="Stage DDR images into a grade-labelled ImageFolder for evaluation.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the DDR DR_grading directory (contains train.txt, valid.txt, test.txt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the ImageFolder structure.",
    )
    parser.add_argument(
        "--exclude-grade",
        type=int,
        default=5,
        help="Grade to exclude (default 5 = unreadable).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image files instead of creating symlinks.",
    )
    args = parser.parse_args(argv)

    try:
        summary = prepare_ddr(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            exclude_grades=frozenset({args.exclude_grade}),
            copy=args.copy,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for split, (staged, excluded) in summary.items():
        print(f"{split}: {staged} images staged, {excluded} excluded")

    print(f"Done. ImageFolder structure written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
