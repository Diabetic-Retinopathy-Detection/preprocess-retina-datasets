from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from preprocess_retina_datasets.cli.ddr import main as cli_main


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _make_grading(tmp_path: Path) -> Path:
    input_dir = tmp_path / "DR_grading"
    for split in ("train", "valid", "test"):
        lines = []
        for grade in range(5):
            filename = f"{split}_g{grade}.jpg"
            _make_image(input_dir / split / filename)
            lines.append(f"{filename} {grade}")
        filename = f"{split}_unreadable.jpg"
        _make_image(input_dir / split / filename)
        lines.append(f"{filename} 5")
        (input_dir / f"{split}.txt").write_text("\n".join(lines) + "\n")
    return input_dir


class TestCliDirect:
    def test_success_symlinks(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "out"

        exit_code = cli_main(["--input", str(input_dir), "--output", str(output_dir)])

        assert exit_code == 0
        assert (output_dir / "train" / "2" / "train_g2.jpg").is_symlink()
        out = capsys.readouterr().out
        assert "train: 5 images staged, 1 excluded" in out
        assert "valid: 5 images staged, 1 excluded" in out
        assert "test: 5 images staged, 1 excluded" in out

    def test_success_copy(self, tmp_path: Path) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "out"

        exit_code = cli_main(["--input", str(input_dir), "--output", str(output_dir), "--copy"])

        assert exit_code == 0
        copied = output_dir / "train" / "2" / "train_g2.jpg"
        assert copied.is_file()
        assert not copied.is_symlink()

    def test_exclude_grade(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "out"

        exit_code = cli_main(["--input", str(input_dir), "--output", str(output_dir), "--exclude-grade", "3"])

        assert exit_code == 0
        assert not (output_dir / "train" / "3").exists()
        out = capsys.readouterr().out
        assert "train: 4 images staged, 2 excluded" in out

    def test_missing_input_dir_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = cli_main(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "out")])

        assert exit_code == 1
        assert "Error" in capsys.readouterr().err


class TestCliSubprocess:
    def test_success(self, tmp_path: Path) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "out"

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.ddr",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Done. ImageFolder structure written to" in result.stdout
        assert (output_dir / "test" / "0" / "test_g0.jpg").is_symlink()

    def test_missing_input_dir_returns_one(self, tmp_path: Path) -> None:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.ddr",
                "--input",
                str(tmp_path / "nope"),
                "--output",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr
