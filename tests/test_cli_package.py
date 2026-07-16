from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from preprocess_retina_datasets.cli.package import main as cli_main


def _make_image(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_saliency(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(*size).astype(np.float32)
    np.save(str(path), arr)


def _load_pickle(pkl_path: Path) -> list:
    with pkl_path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


class TestCliDirect:
    def test_success(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"
        _make_image(image_dir / "a.jpeg")
        _make_saliency(saliency_dir / "a.npy")

        exit_code = cli_main([
            "--image-folder",
            str(image_dir),
            "--saliency-folder",
            str(saliency_dir),
            "--output-file",
            str(output_file),
        ])
        assert exit_code == 0
        assert output_file.exists()

        pairs = _load_pickle(output_file)
        assert len(pairs) == 1


class TestCliSubprocess:
    def test_success(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"
        _make_image(image_dir / "a.jpeg")
        _make_saliency(saliency_dir / "a.npy")

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.package",
                "--image-folder",
                str(image_dir),
                "--saliency-folder",
                str(saliency_dir),
                "--output-file",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert output_file.exists()

    def test_mismatch_returns_one(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"
        _make_image(image_dir / "a.jpeg")
        _make_image(image_dir / "b.jpeg")
        _make_saliency(saliency_dir / "a.npy")

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.package",
                "--image-folder",
                str(image_dir),
                "--saliency-folder",
                str(saliency_dir),
                "--output-file",
                str(output_file),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_missing_dir_returns_one(self, tmp_path: Path) -> None:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.package",
                "--image-folder",
                str(tmp_path / "nope"),
                "--saliency-folder",
                str(tmp_path / "saliency"),
                "--output-file",
                str(tmp_path / "index.pkl"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr
