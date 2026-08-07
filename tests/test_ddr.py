from __future__ import annotations

from pathlib import Path

import pytest

from preprocess_retina_datasets.ddr import parse_grading_txt, prepare_ddr


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _make_grading(tmp_path: Path) -> Path:
    input_dir = tmp_path / "DR_grading"
    for split, _n in (("train", 3), ("valid", 2), ("test", 1)):
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


def _class_dirs(output_dir: Path) -> list[Path]:
    return sorted(p for p in output_dir.rglob("*") if p.is_dir() and p.parent.name in ("train", "valid", "test"))


class TestParseGradingTxt:
    def test_parses_pairs(self, tmp_path: Path) -> None:
        txt = tmp_path / "train.txt"
        txt.write_text("a.jpg 0\nb.jpg 4\n")
        assert parse_grading_txt(txt) == [("a.jpg", 0), ("b.jpg", 4)]

    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        txt = tmp_path / "train.txt"
        txt.write_text("a.jpg 0\n\n  \nb.jpg 1\n")
        assert parse_grading_txt(txt) == [("a.jpg", 0), ("b.jpg", 1)]

    def test_splits_on_last_space(self, tmp_path: Path) -> None:
        txt = tmp_path / "train.txt"
        txt.write_text("my file name.jpg 3\n")
        assert parse_grading_txt(txt) == [("my file name.jpg", 3)]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_grading_txt(tmp_path / "nope.txt")

    def test_malformed_line(self, tmp_path: Path) -> None:
        txt = tmp_path / "train.txt"
        txt.write_text("a.jpg\n")
        with pytest.raises(ValueError, match="Malformed line"):
            parse_grading_txt(txt)

    def test_invalid_grade(self, tmp_path: Path) -> None:
        txt = tmp_path / "train.txt"
        txt.write_text("a.jpg abc\n")
        with pytest.raises(ValueError, match="Invalid grade"):
            parse_grading_txt(txt)


class TestPrepareDdr:
    def test_symlink_default(self, tmp_path: Path) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "DDR-ImageFolder"

        summary = prepare_ddr(input_dir, output_dir)

        assert summary["train"] == (5, 1)
        assert summary["valid"] == (5, 1)
        assert summary["test"] == (5, 1)

        link = output_dir / "train" / "2" / "train_g2.jpg"
        assert link.is_symlink()
        assert link.resolve() == (input_dir / "train" / "train_g2.jpg").resolve()

        unreadable = output_dir / "train" / "5"
        assert not unreadable.exists()

    def test_copy(self, tmp_path: Path) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "DDR-ImageFolder"

        summary = prepare_ddr(input_dir, output_dir, copy=True)

        assert summary["train"] == (5, 1)
        copied = output_dir / "train" / "2" / "train_g2.jpg"
        assert copied.is_file()
        assert not copied.is_symlink()
        assert copied.read_text() == "x"

    def test_custom_exclude_grades(self, tmp_path: Path) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "DDR-ImageFolder"

        summary = prepare_ddr(input_dir, output_dir, exclude_grades=frozenset({3}))

        assert summary["train"] == (4, 2)
        assert not (output_dir / "train" / "3").exists()

    def test_grades_outside_range_always_excluded(self, tmp_path: Path) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "DDR-ImageFolder"

        prepare_ddr(input_dir, output_dir, exclude_grades=frozenset())

        assert not (output_dir / "train" / "5").exists()

    def test_missing_input_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            prepare_ddr(tmp_path / "nope", tmp_path / "out")

    def test_missing_split_warns_and_skips(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "DR_grading"
        input_dir.mkdir()
        (input_dir / "train.txt").write_text("a.jpg 0\n")
        _make_image(input_dir / "train" / "a.jpg")
        output_dir = tmp_path / "out"

        with pytest.warns(UserWarning, match="Split file not found"):
            summary = prepare_ddr(input_dir, output_dir)

        assert "train" in summary
        assert "valid" not in summary
        assert "test" not in summary

    def test_idempotent(self, tmp_path: Path) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "DDR-ImageFolder"

        first = prepare_ddr(input_dir, output_dir)
        second = prepare_ddr(input_dir, output_dir)

        assert first == second
        n_links = len(list(output_dir.rglob("*.jpg")))
        assert n_links == 15
        assert len(_class_dirs(output_dir)) == 15


class TestPrepareDdrOutputLayout:
    def test_splits_and_grades(self, tmp_path: Path) -> None:
        input_dir = _make_grading(tmp_path)
        output_dir = tmp_path / "DDR-ImageFolder"

        prepare_ddr(input_dir, output_dir)

        dirs = _class_dirs(output_dir)
        assert len(dirs) == 15
        for split in ("train", "valid", "test"):
            for grade in range(5):
                assert output_dir / split / str(grade) in dirs
