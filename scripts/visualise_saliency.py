"""Visualise a cropped fundus image alongside its saliency map.

Usage:
    uv run python scripts/visualise_saliency.py <image_path> <saliency_path>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    if len(args) != 2:
        print(f"Usage: {sys.argv[0]} <image> <saliency.npy>", file=sys.stderr)
        return 1

    img_path, sal_path = args

    if not Path(img_path).exists():
        print(f"Error: image not found: {img_path}", file=sys.stderr)
        return 1
    if not Path(sal_path).exists():
        print(f"Error: saliency map not found: {sal_path}", file=sys.stderr)
        return 1

    image = np.array(Image.open(img_path))
    saliency = np.load(sal_path)

    axes = plt.subplots(1, 3, figsize=(14, 4))[1]
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    im = axes[1].imshow(saliency, cmap="hot")
    axes[1].set_title("Saliency (hot)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    axes[2].imshow(image)
    axes[2].imshow(saliency, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
