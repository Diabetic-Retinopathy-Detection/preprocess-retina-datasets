from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

from preprocess_retina_datasets.cli.crop import main

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_DIR = FIXTURES / "eyepacs_sample"
CROP_SIZE = 512


class TestMain:
    def test_returns_zero_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main([
                "--image-folder",
                str(SAMPLE_DIR),
                "--output-folder",
                str(tmp),
                "--crop-size",
                str(CROP_SIZE),
                "-n",
                "2",
            ])
            assert exit_code == 0

    def test_returns_nonzero_on_bad_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main([
                "--image-folder",
                "/nonexistent",
                "--output-folder",
                str(Path(tmp) / "out"),
            ])
            assert exit_code == 1

    def test_prints_help(self) -> None:
        try:
            main(["--help"])
        except SystemExit as exc:
            assert exc.code == 0


class TestSubprocess:
    def test_cli_crops_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(  # noqa: S603
                [
                    sys.executable,
                    "-m",
                    "preprocess_retina_datasets.cli.crop",
                    "--image-folder",
                    str(SAMPLE_DIR),
                    "--output-folder",
                    str(tmp),
                    "--crop-size",
                    str(CROP_SIZE),
                    "-n",
                    "2",
                ],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            out_files = sorted(Path(tmp).iterdir())
            assert len(out_files) == len(list(SAMPLE_DIR.iterdir()))
            for f in out_files:
                assert Image.open(f).size == (CROP_SIZE, CROP_SIZE)
