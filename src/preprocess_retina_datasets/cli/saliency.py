"""CLI for :mod:`preprocess_retina_datasets.saliency`.

Parses command-line arguments and delegates to the library functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from preprocess_retina_datasets.errors import ImageProcessingError
from preprocess_retina_datasets.saliency import generate_saliency_dataset


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``detect-saliency`` console script.

    Args:
        argv: Argument list (defaults to :data:`sys.argv`).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(
        description="Generate saliency maps for cropped retina fundus images.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to the input folder containing cropped images.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder for saliency map .npy files.",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images where the output file already exists.",
    )
    args = parser.parse_args(argv)

    try:
        failed = generate_saliency_dataset(
            input_dir=Path(args.image_folder),
            output_dir=Path(args.output_folder),
            num_workers=args.num_workers,
            skip_existing=args.skip_existing,
        )
    except (OSError, ImageProcessingError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if failed:
        print(f"Warning: {failed} image(s) failed to process.", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
