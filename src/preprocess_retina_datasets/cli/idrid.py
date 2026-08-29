"""CLI for :mod:`preprocess_retina_datasets.idrid`.

Parses command-line arguments and delegates to the library functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from preprocess_retina_datasets.idrid import prepare_idrid


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``prepare-idrid`` console script.

    Args:
        argv: Argument list (defaults to :data:`sys.argv`).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(
        description="Stage IDRiD images into a grade-labelled ImageFolder for evaluation.",
    )
    parser.add_argument(
        "--train-images",
        type=str,
        required=True,
        help="Path to the IDRiD training images directory.",
    )
    parser.add_argument(
        "--test-images",
        type=str,
        required=True,
        help="Path to the IDRiD test images directory.",
    )
    parser.add_argument(
        "--labels-train",
        type=str,
        required=True,
        help="Path to the IDRiD training label CSV.",
    )
    parser.add_argument(
        "--labels-test",
        type=str,
        required=True,
        help="Path to the IDRiD test label CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the ImageFolder structure.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Fraction of each training grade to reserve for the valid split (default 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the stratified valid split (default 42).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image files instead of creating symlinks.",
    )
    args = parser.parse_args(argv)

    try:
        summary = prepare_idrid(
            train_image_dir=Path(args.train_images),
            test_image_dir=Path(args.test_images),
            train_labels_csv=Path(args.labels_train),
            test_labels_csv=Path(args.labels_test),
            output_dir=Path(args.output),
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            copy=args.copy,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for split, (staged, per_grade) in summary.items():
        grade_counts = ", ".join(f"g{g}:{c}" for g, c in sorted(per_grade.items()))
        print(f"{split}: {staged} images staged  [{grade_counts}]")

    print(f"Done. ImageFolder structure written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
