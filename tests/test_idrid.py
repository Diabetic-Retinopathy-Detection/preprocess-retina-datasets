from __future__ import annotations

from pathlib import Path

import pytest

from preprocess_retina_datasets.idrid import carve_valid_split, parse_idrid_csv, prepare_idrid


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _class_dirs(output_dir: Path) -> list[Path]:
    return sorted(p for p in output_dir.rglob("*") if p.is_dir() and p.parent.name in ("train", "valid", "test"))


class TestParseIdridCsv:
    def test_parses_pairs(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("IDRiD_001,0,2\nIDRiD_002,4,1\n")
        assert parse_idrid_csv(csv_path) == [("IDRiD_001", 0), ("IDRiD_002", 4)]

    def test_skips_header_and_tolerates_trailing_commas(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("Image name,Retinopathy grade,\nIDRiD_001,0,2,,,,,,,,,\nIDRiD_002,3,1,,,,,,,,,\n")
        assert parse_idrid_csv(csv_path) == [("IDRiD_001", 0), ("IDRiD_002", 3)]

    def test_skips_blank_middle_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("IDRiD_001,0,\n\nIDRiD_002,2,\n")
        assert parse_idrid_csv(csv_path) == [("IDRiD_001", 0), ("IDRiD_002", 2)]

    def test_name_has_no_extension(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("IDRiD_007,1,\n")
        rows = parse_idrid_csv(csv_path)
        assert rows[0][0] == "IDRiD_007"

    def test_malformed_grade_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("IDRiD_001,abc,\n")
        with pytest.raises(ValueError, match="Invalid grade"):
            parse_idrid_csv(csv_path)

    def test_missing_grade_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("IDRiD_001,\n")
        with pytest.raises(ValueError, match="Malformed row"):
            parse_idrid_csv(csv_path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            parse_idrid_csv(tmp_path / "nope.csv")


class TestCarveValidSplit:
    def test_exact_counts_three_grades(self) -> None:
        rows = [("a", 0)] * 10 + [("b", 1)] * 5 + [("c", 2)] * 20
        valid, train = carve_valid_split(rows, valid_ratio=0.2, seed=42)
        assert _grade_counts(valid) == {0: 2, 1: 1, 2: 4}
        assert _grade_counts(train) == {0: 8, 1: 4, 2: 16}

    def test_determinism_same_seed(self) -> None:
        rows = [("x", 0)] * 134 + [("y", 1)] * 20 + [("z", 2)] * 136 + [("w", 3)] * 74 + [("v", 4)] * 49
        v1, t1 = carve_valid_split(rows, valid_ratio=0.2, seed=42)
        v2, t2 = carve_valid_split(rows, valid_ratio=0.2, seed=42)
        assert _grade_counts(v1) == _grade_counts(v2)
        assert _grade_counts(t1) == _grade_counts(t2)
        assert sorted(v1) == sorted(v2)
        assert sorted(t1) == sorted(t2)

    def test_reference_counts_five_grades(self) -> None:
        rows = [("x", 0)] * 134 + [("y", 1)] * 20 + [("z", 2)] * 136 + [("w", 3)] * 74 + [("v", 4)] * 49
        valid, train = carve_valid_split(rows, valid_ratio=0.2, seed=42)
        assert _grade_counts(valid) == {0: 27, 1: 4, 2: 27, 3: 15, 4: 10}
        assert _grade_counts(train) == {0: 107, 1: 16, 2: 109, 3: 59, 4: 39}

    def test_different_seed_differs(self) -> None:
        rows = [(f"x{i:02d}", 0) for i in range(20)]
        v1, _t1 = carve_valid_split(rows, valid_ratio=0.2, seed=42)
        v2, _t2 = carve_valid_split(rows, valid_ratio=0.2, seed=1)
        assert _grade_counts(v1) == {0: 4}
        assert _grade_counts(v2) == {0: 4}
        assert sorted(v1) != sorted(v2)


def _grade_counts(rows: list[tuple[str, int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for _name, grade in rows:
        counts[grade] = counts.get(grade, 0) + 1
    return counts


class TestPrepareIdrid:
    def _fixture(self, tmp_path: Path, *, extra_orphans: bool = False) -> tuple[Path, Path, Path, Path, Path]:
        train_dir = tmp_path / "train_img"
        test_dir = tmp_path / "test_img"
        train_labels = tmp_path / "train_labels.csv"
        test_labels = tmp_path / "test_labels.csv"

        train_rows = []
        train_csv_lines = []
        for i in range(50):
            name = f"IDRiD_{i:03d}"
            _make_image(train_dir / f"{name}.jpg")
            grade = i % 5
            train_rows.append((name, grade))
            train_csv_lines.append(f"{name},{grade},2")
        train_labels.write_text("\n".join(train_csv_lines) + "\n")

        test_rows = []
        test_csv_lines = []
        for i in range(5):
            name = f"IDRiD_T{i:03d}"
            _make_image(test_dir / f"{name}.jpg")
            grade = i % 5
            test_rows.append((name, grade))
            test_csv_lines.append(f"{name},{grade},2")
        test_labels.write_text("\n".join(test_csv_lines) + "\n")

        if extra_orphans:
            _make_image(train_dir / "orphan_extra.jpg")

        return train_dir, test_dir, train_labels, test_labels, train_rows

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            prepare_idrid(tmp_path / "nope", tmp_path / "t", tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "out")

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            prepare_idrid(tmp_path, tmp_path, tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "out")

    def test_symlink_default(self, tmp_path: Path) -> None:
        train_dir, test_dir, train_labels, test_labels, _train_rows = self._fixture(tmp_path)
        output_dir = tmp_path / "out"

        summary = prepare_idrid(train_dir, test_dir, train_labels, test_labels, output_dir)

        total = sum(c for _g, c in summary["train"][1].items())
        assert total == 40
        assert sum(summary["valid"][1].values()) == 10

        link = output_dir / "train" / "0" / "IDRiD_000.jpg"
        assert link.is_symlink()
        assert link.resolve() == (train_dir / "IDRiD_000.jpg").resolve()

    def test_copy(self, tmp_path: Path) -> None:
        train_dir, test_dir, train_labels, test_labels, _train_rows = self._fixture(tmp_path)
        output_dir = tmp_path / "out"

        summary = prepare_idrid(train_dir, test_dir, train_labels, test_labels, output_dir, copy=True)

        copied = output_dir / "train" / "0" / "IDRiD_000.jpg"
        assert copied.is_file()
        assert not copied.is_symlink()
        assert copied.read_text() == "x"
        assert sum(summary["test"][1].values()) == 5

    def test_idempotent(self, tmp_path: Path) -> None:
        train_dir, test_dir, train_labels, test_labels, _train_rows = self._fixture(tmp_path)
        output_dir = tmp_path / "out"

        first = prepare_idrid(train_dir, test_dir, train_labels, test_labels, output_dir)
        second = prepare_idrid(train_dir, test_dir, train_labels, test_labels, output_dir)

        assert first == second
        n_links = len(list(output_dir.rglob("*.jpg")))
        assert n_links == 55

    def test_orphan_raises(self, tmp_path: Path) -> None:
        train_dir, test_dir, train_labels, test_labels, _train_rows = self._fixture(tmp_path, extra_orphans=True)
        with pytest.raises(ValueError, match="orphan_extra"):
            prepare_idrid(train_dir, test_dir, train_labels, test_labels, tmp_path / "out")

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        train_dir, test_dir, train_labels, test_labels, _train_rows = self._fixture(tmp_path)
        (train_dir / "IDRiD_000.jpg").unlink()
        with pytest.raises(ValueError, match="missing a file"):
            prepare_idrid(train_dir, test_dir, train_labels, test_labels, tmp_path / "out")

    def test_output_layout(self, tmp_path: Path) -> None:
        train_dir, test_dir, train_labels, test_labels, _train_rows = self._fixture(tmp_path)
        output_dir = tmp_path / "out"

        prepare_idrid(train_dir, test_dir, train_labels, test_labels, output_dir)

        dirs = set(_class_dirs(output_dir))
        for split in ("train", "valid", "test"):
            for grade in range(5):
                assert output_dir / split / str(grade) in dirs
