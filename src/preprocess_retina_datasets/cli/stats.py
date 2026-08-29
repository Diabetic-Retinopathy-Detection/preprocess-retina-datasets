"""CLI for :mod:`preprocess_retina_datasets.stats`.

Parses command-line arguments and delegates to the library functions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from preprocess_retina_datasets.stats import compute_mean_std


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``compute-dataset-stats`` console script.

    Args:
        argv: Argument list (defaults to :data:`sys.argv`).

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std of a dataset over the finetune transform.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Path to the folder containing images (subfolders included).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=384,
        help="Resize images to this square size in pixels (default: 384).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the JSON result to this path instead of printing to stdout.",
    )
    args = parser.parse_args(argv)

    output_path = Path(args.output) if args.output else None

    try:
        result = compute_mean_std(
            image_dir=Path(args.image_folder),
            resize=args.resize,
            output=output_path,
        )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if output_path is not None:
        return 0

    payload = {
        "mean": [round(float(v), 6) for v in result["mean"]],
        "std": [round(float(v), 6) for v in result["std"]],
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
