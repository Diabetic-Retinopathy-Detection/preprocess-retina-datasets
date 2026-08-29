from __future__ import annotations

import pickle
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from preprocess_retina_datasets.package import (
    _collect_files_by_key,
    _compare_keys,
    build_dataset_index,
)


def _make_image(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_saliency(path: Path, size: tuple[int, int] = (512, 512)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.rand(*size).astype(np.float32)
    np.save(str(path), arr)


def _load_pickle(pkl_path: Path) -> list[tuple[Path, Path]]:
    with pkl_path.open("rb") as f:
        return cast(list[tuple[Path, Path]], pickle.load(f))  # noqa: S301


class TestCollectFilesByKey:
    def test_collects_files(self, tmp_path: Path) -> None:
        _make_image(tmp_path / "sub" / "a.jpeg")
        _make_image(tmp_path / "sub" / "b.jpeg")
        result = _collect_files_by_key(tmp_path, frozenset({".jpeg"}))
        assert result == {
            "sub/a": (tmp_path / "sub" / "a.jpeg").absolute(),
            "sub/b": (tmp_path / "sub" / "b.jpeg").absolute(),
        }

    def test_filters_by_suffix(self, tmp_path: Path) -> None:
        _make_image(tmp_path / "a.jpeg")
        _make_saliency(tmp_path / "a.npy")
        result = _collect_files_by_key(tmp_path, frozenset({".jpeg"}))
        assert result == {"a": (tmp_path / "a.jpeg").absolute()}

    def test_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            _collect_files_by_key(tmp_path / "nope", frozenset({".jpeg"}))


class TestCompareKeys:
    def test_matching_keys_does_not_raise(self) -> None:
        d = {"a": Path("/a"), "b": Path("/b")}
        _compare_keys(d, d)

    def test_missing_saliency_raises(self) -> None:
        images = {"a": Path("/a"), "b": Path("/b"), "c": Path("/c")}
        saliency = {"a": Path("/a")}
        with pytest.raises(ValueError, match="2 image\\(s\\) missing"):
            _compare_keys(images, saliency)

    def test_extra_saliency_raises(self) -> None:
        images = {"a": Path("/a")}
        saliency = {"a": Path("/a"), "b": Path("/b"), "c": Path("/c")}
        with pytest.raises(ValueError, match="2 saliency map\\(s\\) without"):
            _compare_keys(images, saliency)

    def test_both_mismatch_reports_both(self) -> None:
        images = {"a": Path("/a"), "b": Path("/b")}
        saliency = {"a": Path("/a"), "c": Path("/c")}
        with pytest.raises(ValueError) as exc:
            _compare_keys(images, saliency)
        msg = str(exc.value)
        assert "1 image(s) missing" in msg
        assert "1 saliency map(s) without" in msg


class TestBuildDatasetIndex:
    def test_basic(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"

        _make_image(image_dir / "a.jpeg")
        _make_image(image_dir / "b.jpeg")
        _make_saliency(saliency_dir / "a.npy")
        _make_saliency(saliency_dir / "b.npy")

        build_dataset_index(image_dir, saliency_dir, output_file)
        assert output_file.exists()

        pairs = _load_pickle(output_file)

        assert isinstance(pairs, list)
        assert len(pairs) == 2
        for img_path, sal_path in pairs:
            assert isinstance(img_path, Path)
            assert isinstance(sal_path, Path)
            assert not img_path.is_absolute()
            assert not sal_path.is_absolute()

    def test_paths_relative_to_respective_dirs(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "data" / "images"
        saliency_dir = tmp_path / "data" / "saliency"
        output_file = tmp_path / "index.pkl"

        _make_image(image_dir / "a.jpeg")
        _make_saliency(saliency_dir / "a.npy")

        build_dataset_index(image_dir, saliency_dir, output_file)

        pairs = _load_pickle(output_file)
        img_path, sal_path = pairs[0]
        assert str(img_path) == "a.jpeg"
        assert str(sal_path) == "a.npy"

    def test_preserves_subdirectory_structure(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"

        _make_image(image_dir / "train" / "a.jpeg")
        _make_image(image_dir / "val" / "a.jpeg")
        _make_saliency(saliency_dir / "train" / "a.npy")
        _make_saliency(saliency_dir / "val" / "a.npy")

        build_dataset_index(image_dir, saliency_dir, output_file)

        pairs = _load_pickle(output_file)
        assert len(pairs) == 2
        rel_paths = {str(img) for img, _ in pairs}
        assert rel_paths == {"train/a.jpeg", "val/a.jpeg"}

    def test_count_mismatch_image_extra(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"

        _make_image(image_dir / "a.jpeg")
        _make_image(image_dir / "b.jpeg")
        _make_saliency(saliency_dir / "a.npy")

        with pytest.raises(ValueError, match="1 image\\(s\\) missing"):
            build_dataset_index(image_dir, saliency_dir, output_file)

    def test_count_mismatch_saliency_extra(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"

        _make_image(image_dir / "a.jpeg")
        _make_saliency(saliency_dir / "a.npy")
        _make_saliency(saliency_dir / "b.npy")

        with pytest.raises(ValueError, match="1 saliency map\\(s\\) without"):
            build_dataset_index(image_dir, saliency_dir, output_file)

    def test_empty_image_dir(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"

        image_dir.mkdir()
        _make_saliency(saliency_dir / "a.npy")

        with pytest.raises(ValueError, match="1 saliency map\\(s\\) without"):
            build_dataset_index(image_dir, saliency_dir, output_file)

    def test_empty_both_dirs_produces_empty_list(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "index.pkl"

        image_dir.mkdir()
        saliency_dir.mkdir()

        build_dataset_index(image_dir, saliency_dir, output_file)

        pairs = _load_pickle(output_file)
        assert pairs == []

    def test_missing_input_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_dataset_index(
                tmp_path / "nope",
                tmp_path / "saliency",
                tmp_path / "index.pkl",
            )

    def test_output_dir_creation(self, tmp_path: Path) -> None:
        image_dir = tmp_path / "images"
        saliency_dir = tmp_path / "saliency"
        output_file = tmp_path / "nested" / "sub" / "index.pkl"

        _make_image(image_dir / "a.jpeg")
        _make_saliency(saliency_dir / "a.npy")

        build_dataset_index(image_dir, saliency_dir, output_file)
        assert output_file.exists()
