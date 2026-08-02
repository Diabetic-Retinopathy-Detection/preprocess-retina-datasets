"""Visualise a cropped fundus image with its saliency map binarised at two thresholds.

Shows the original image, the raw saliency map, and the saliency mask overlaid on
the image at each threshold (default 0.25 and 0.5), to illustrate how the BCE
segmentation target changes with ``saliency_threshold``.

Usage:
    uv run python scripts/visualise_saliency_thresholds.py <image_path> <saliency_path> [--output out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_THRESHOLDS = (0.25, 0.5)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    output = None
    if "--output" in args:
        i = args.index("--output")
        if i + 1 >= len(args):
            print("Error: --output requires a path", file=sys.stderr)
            return 1
        output = args[i + 1]
        del args[i : i + 2]

    pos = args

    if len(pos) != 2:
        print(
            f"Usage: {sys.argv[0]} <image> <saliency.npy> [--output out.png]",
            file=sys.stderr,
        )
        return 1

    img_path, sal_path = pos

    if not Path(img_path).exists():
        print(f"Error: image not found: {img_path}", file=sys.stderr)
        return 1
    if not Path(sal_path).exists():
        print(f"Error: saliency map not found: {sal_path}", file=sys.stderr)
        return 1

    image = np.array(Image.open(img_path))
    saliency = np.load(sal_path)
    saliency = saliency.astype(np.float32)

    if output is not None:
        matplotlib.use("Agg")

    panels = 2 + len(DEFAULT_THRESHOLDS)
    axes = plt.subplots(1, panels, figsize=(4 * panels, 4))[1]

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    im = axes[1].imshow(saliency, cmap="hot")
    axes[1].set_title("Saliency (raw)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    for ax, t in zip(axes[2:], DEFAULT_THRESHOLDS, strict=True):
        mask = (saliency > t).astype(float)
        ax.imshow(image)
        ax.imshow(mask, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        frac = mask.mean() * 100
        ax.set_title(f"Mask > {t}  ({frac:.1f}% pixels)")
        ax.axis("off")

    plt.tight_layout()

    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved illustration to {output}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
