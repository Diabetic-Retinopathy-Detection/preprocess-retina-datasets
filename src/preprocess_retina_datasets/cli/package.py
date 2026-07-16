"""CLI for :mod:`preprocess_retina_datasets.package`.

Parses command-line arguments and delegates to the library functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from preprocess_retina_datasets.errors import ImageProcessingError
from preprocess_retina_datasets.package import build_dataset_index


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``build-dataset-index`` console script.

    Args:
        argv: Argument list (defaults to :data:`sys.argv`).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(
        description="Build a dataset index pickle by pairing cropped images with their saliency maps.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to the folder containing cropped images.",
    )
    parser.add_argument(
        "--saliency-folder",
        type=str,
        required=True,
        help="Path to the folder containing saliency .npy files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path for the output pickle file.",
    )
    args = parser.parse_args(argv)

    try:
        build_dataset_index(
            image_dir=Path(args.image_folder),
            saliency_dir=Path(args.saliency_folder),
            output_file=Path(args.output_file),
        )
    except (OSError, ImageProcessingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
