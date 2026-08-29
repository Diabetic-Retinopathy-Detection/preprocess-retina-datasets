from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess_retina_datasets.cli.stats import main


def _make_fixture_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, value in (("a.png", 128), ("b.png", 200)):
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path / name)


class TestCliDirect:
    def test_success_prints_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        fixture = tmp_path / "imgs"
        _make_fixture_dir(fixture)

        exit_code = main(["--image-folder", str(fixture)])

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"mean", "std"}
        assert len(payload["mean"]) == 3
        assert len(payload["std"]) == 3

    def test_output_written(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        fixture = tmp_path / "imgs"
        _make_fixture_dir(fixture)
        output = tmp_path / "stats.json"

        exit_code = main(["--image-folder", str(fixture), "--output", str(output)])

        assert exit_code == 0
        assert capsys.readouterr().out == ""
        assert output.is_file()
        payload = json.loads(output.read_text())
        assert set(payload) == {"mean", "std"}

    def test_missing_dir_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--image-folder", str(tmp_path / "nope")])

        assert exit_code == 1
        assert "Error" in capsys.readouterr().err


class TestCliSubprocess:
    def test_success(self, tmp_path: Path) -> None:
        fixture = tmp_path / "imgs"
        _make_fixture_dir(fixture)

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.stats",
                "--image-folder",
                str(fixture),
                "--resize",
                "16",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(result.stdout)
        assert len(payload["mean"]) == 3
        assert len(payload["std"]) == 3

    def test_missing_dir_returns_one(self, tmp_path: Path) -> None:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.stats",
                "--image-folder",
                str(tmp_path / "nope"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error" in result.stderr
