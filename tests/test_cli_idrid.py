from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from preprocess_retina_datasets.cli.idrid import main as cli_main


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train_dir = tmp_path / "train_img"
    test_dir = tmp_path / "test_img"
    train_labels = tmp_path / "train_labels.csv"
    test_labels = tmp_path / "test_labels.csv"

    train_lines = []
    for i in range(50):
        name = f"IDRiD_{i:03d}"
        _make_image(train_dir / f"{name}.jpg")
        train_lines.append(f"{name},{i % 5},2")
    train_labels.write_text("\n".join(train_lines) + "\n")

    test_lines = []
    for i in range(5):
        name = f"IDRiD_T{i:03d}"
        _make_image(test_dir / f"{name}.jpg")
        test_lines.append(f"{name},{i % 5},2")
    test_labels.write_text("\n".join(test_lines) + "\n")

    return train_dir, test_dir, train_labels, test_labels


def _args(train_dir: Path, test_dir: Path, train_labels: Path, test_labels: Path, output_dir: Path) -> list[str]:
    return [
        "--train-images",
        str(train_dir),
        "--test-images",
        str(test_dir),
        "--labels-train",
        str(train_labels),
        "--labels-test",
        str(test_labels),
        "--output",
        str(output_dir),
    ]


class TestCliDirect:
    def test_success_symlinks(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        train_dir, test_dir, train_labels, test_labels = _make_fixture(tmp_path)
        output_dir = tmp_path / "out"

        exit_code = cli_main(_args(train_dir, test_dir, train_labels, test_labels, output_dir))

        assert exit_code == 0
        assert (output_dir / "train" / "0" / "IDRiD_000.jpg").is_symlink()
        out = capsys.readouterr().out
        assert "train: 40 images staged" in out
        assert "test: 5 images staged" in out
        assert "Done. ImageFolder structure written to" in out

    def test_success_copy(self, tmp_path: Path) -> None:
        train_dir, test_dir, train_labels, test_labels = _make_fixture(tmp_path)
        output_dir = tmp_path / "out"

        exit_code = cli_main([*_args(train_dir, test_dir, train_labels, test_labels, output_dir), "--copy"])

        assert exit_code == 0
        copied = output_dir / "train" / "0" / "IDRiD_000.jpg"
        assert copied.is_file()
        assert not copied.is_symlink()

    def test_valid_ratio_and_seed(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        train_dir, test_dir, train_labels, test_labels = _make_fixture(tmp_path)
        output_dir = tmp_path / "out"

        exit_code = cli_main([
            *_args(train_dir, test_dir, train_labels, test_labels, output_dir),
            "--valid-ratio",
            "0.5",
            "--seed",
            "7",
        ])

        assert exit_code == 0
        assert (output_dir / "valid" / "0" / "IDRiD_000.jpg").exists()
        out = capsys.readouterr().out
        assert "valid:" in out

    def test_missing_train_images_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _train_dir, test_dir, train_labels, test_labels = _make_fixture(tmp_path)
        exit_code = cli_main(_args(tmp_path / "nope", test_dir, train_labels, test_labels, tmp_path / "out"))
        assert exit_code == 1
        assert "Error" in capsys.readouterr().err


class TestCliSubprocess:
    def test_success(self, tmp_path: Path) -> None:
        train_dir, test_dir, train_labels, test_labels = _make_fixture(tmp_path)
        output_dir = tmp_path / "out"

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.idrid",
                *_args(train_dir, test_dir, train_labels, test_labels, output_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Done. ImageFolder structure written to" in result.stdout
        assert (output_dir / "test" / "0" / "IDRiD_T000.jpg").is_symlink()

    def test_missing_train_images_returns_one(self, tmp_path: Path) -> None:
        _train_dir, test_dir, train_labels, test_labels = _make_fixture(tmp_path)
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "preprocess_retina_datasets.cli.idrid",
                *_args(tmp_path / "nope", test_dir, train_labels, test_labels, tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr
