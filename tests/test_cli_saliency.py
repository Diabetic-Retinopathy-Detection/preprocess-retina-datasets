from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from preprocess_retina_datasets.cli.saliency import main as cli_main


def _make_image(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


class TestCliDirect:
    def test_success(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _make_image(input_dir / "img.jpeg")

        exit_code = cli_main([
            "--image-folder",
            str(input_dir),
            "--output-folder",
            str(output_dir),
        ])
        assert exit_code == 0
        assert (output_dir / "img.npy").exists()

    def test_skip_existing(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _make_image(input_dir / "img.jpeg")
        output_dir.mkdir()
        (output_dir / "img.npy").write_text("old")

        exit_code = cli_main([
            "--image-folder",
            str(input_dir),
            "--output-folder",
            str(output_dir),
            "--skip-existing",
        ])
        assert exit_code == 0
        assert (output_dir / "img.npy").read_text() == "old"

    def test_missing_input_dir(self, tmp_path: Path) -> None:
        exit_code = cli_main([
            "--image-folder",
            str(tmp_path / "nope"),
            "--output-folder",
            str(tmp_path / "out"),
        ])
        assert exit_code == 1

    def test_partial_failure_returns_one(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _make_image(input_dir / "good.jpeg")
        (input_dir / "bad.jpeg").write_text("not an image")

        exit_code = cli_main([
            "--image-folder",
            str(input_dir),
            "--output-folder",
            str(output_dir),
        ])
        assert exit_code == 1


class TestCliSubprocess:
    def test_success(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _make_image(input_dir / "img.jpeg")

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.saliency",
                "--image-folder",
                str(input_dir),
                "--output-folder",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stderr == ""
        assert (output_dir / "img.npy").exists()

    def test_missing_dir_returns_one(self, tmp_path: Path) -> None:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.saliency",
                "--image-folder",
                str(tmp_path / "nope"),
                "--output-folder",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_warning_on_partial_failure(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _make_image(input_dir / "good.jpeg")
        (input_dir / "bad.jpeg").write_text("not an image")

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.saliency",
                "--image-folder",
                str(input_dir),
                "--output-folder",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Warning" in result.stderr
