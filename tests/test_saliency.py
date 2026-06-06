from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess_retina_datasets.errors import ImageProcessingError
from preprocess_retina_datasets.saliency import (
    generate_saliency_dataset,
    generate_saliency_map,
)


def _make_image(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


class TestGenerateSaliencyMap:
    def test_basic(self, tmp_path: Path) -> None:
        src = tmp_path / "img.jpeg"
        dst = tmp_path / "img.npy"
        _make_image(src)
        generate_saliency_map(src, dst)
        assert dst.exists()
        arr = np.load(str(dst))
        assert arr.dtype == np.float32
        assert arr.shape == (512, 512)
        assert 0.0 <= arr.min() <= arr.max() <= 1.0

    def test_skip_existing(self, tmp_path: Path) -> None:
        src = tmp_path / "img.jpeg"
        dst = tmp_path / "img.npy"
        _make_image(src)
        dst.write_text("stale")
        generate_saliency_map(src, dst, skip_existing=True)
        assert dst.read_text() == "stale"

    def test_missing_source(self, tmp_path: Path) -> None:
        dst = tmp_path / "img.npy"
        with pytest.raises(ImageProcessingError):
            generate_saliency_map(tmp_path / "nonexistent.jpeg", dst)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        src = tmp_path / "img.jpeg"
        dst = tmp_path / "sub" / "img.npy"
        _make_image(src)
        generate_saliency_map(src, dst)
        assert dst.exists()

    def test_non_512_size(self, tmp_path: Path) -> None:
        src = tmp_path / "img.jpeg"
        dst = tmp_path / "img.npy"
        _make_image(src, (128, 256))
        generate_saliency_map(src, dst)
        arr = np.load(str(dst))
        assert arr.shape == (128, 256)
        assert arr.dtype == np.float32

    def test_circular_mask_zeros_corners(self, tmp_path: Path) -> None:
        src = tmp_path / "white.jpeg"
        dst = tmp_path / "white.npy"
        img = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(src), img)
        generate_saliency_map(src, dst)
        arr = np.load(str(dst))
        assert arr[0, 0] == 0.0
        assert arr[0, -1] == 0.0
        assert arr[-1, 0] == 0.0
        assert arr[-1, -1] == 0.0
        center = arr[256, 256]
        assert center > 0.0


class TestGenerateSaliencyDataset:
    def test_processes_all_images(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        for i in range(3):
            _make_image(input_dir / f"img{i}.jpeg")

        failed = generate_saliency_dataset(input_dir, output_dir)
        assert failed == 0
        for i in range(3):
            assert (output_dir / f"img{i}.npy").exists()

    def test_preserves_subfolder_structure(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _make_image(input_dir / "sub" / "img.jpeg")

        generate_saliency_dataset(input_dir, output_dir)
        assert (output_dir / "sub" / "img.npy").exists()

    def test_empty_directory_returns_zero(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "empty"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        failed = generate_saliency_dataset(input_dir, output_dir)
        assert failed == 0

    def test_skip_existing(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _make_image(input_dir / "a.jpeg")
        _make_image(input_dir / "b.jpeg")
        output_dir.mkdir(parents=True)
        (output_dir / "a.npy").write_text("old")

        generate_saliency_dataset(input_dir, output_dir, skip_existing=True)
        assert (output_dir / "a.npy").read_text() == "old"
        assert (output_dir / "b.npy").exists()

    def test_nonexistent_input_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            generate_saliency_dataset(tmp_path / "nope", tmp_path / "out")

    @pytest.mark.parametrize("workers", [0, -1])
    def test_invalid_num_workers(self, tmp_path: Path, workers: int) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        with pytest.raises(ValueError, match="num_workers must be >= 1"):
            generate_saliency_dataset(input_dir, tmp_path / "out", num_workers=workers)

    def test_bad_file_does_not_block_others(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _make_image(input_dir / "good.jpeg")
        (input_dir / "bad.jpeg").write_text("not an image")

        failed = generate_saliency_dataset(input_dir, output_dir)
        assert failed == 1
        assert (output_dir / "good.npy").exists()
