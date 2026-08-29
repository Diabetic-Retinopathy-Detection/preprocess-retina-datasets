from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from preprocess_retina_datasets.stats import compute_mean_std

RESIZE = 16


def _save_image(path: Path, values: int | np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((RESIZE, RESIZE, 3), values, dtype=np.uint8) if isinstance(values, int) else values
    Image.fromarray(arr).save(path)
    return path


def _reference_stats(images: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.concatenate([
        np.asarray(
            Image.open(p).convert("RGB").resize((RESIZE, RESIZE), Image.Resampling.BILINEAR),
            dtype=np.float64,
        )
        / 255.0
        for p in images
    ]).reshape(-1, 3)
    return pixels.mean(axis=0), pixels.std(axis=0)


class TestComputeMeanStd:
    def test_constant_gray_mean_and_std(self, tmp_path: Path) -> None:
        _save_image(tmp_path / "gray.png", 128)

        result = compute_mean_std(tmp_path, resize=RESIZE)

        assert result["mean"] == pytest.approx([128 / 255] * 3, abs=1e-9)
        assert result["std"] == [0.0, 0.0, 0.0]

    def test_checkerboard_matches_numpy_reference(self, tmp_path: Path) -> None:
        h = w = RESIZE * 2
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((xx + yy) % 2) == 1
        checker = np.where(np.broadcast_to(mask[..., np.newaxis], (h, w, 3)), 255, 0).astype(np.uint8)
        path = _save_image(tmp_path / "checker.png", checker)

        result = compute_mean_std(tmp_path, resize=RESIZE)

        ref_mean, ref_std = _reference_stats([path])
        assert all(0 < s <= 0.5 for s in result["std"])
        assert np.allclose(result["mean"], ref_mean, atol=1e-9)
        assert np.allclose(result["std"], ref_std, atol=1e-9)

    def test_resize_matches_pil_reference(self, tmp_path: Path) -> None:
        h, w = 16, 32
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, : w // 2, :] = 255
        arr[:, w // 2 :, :] = 64
        path = _save_image(tmp_path / "halves.png", arr)

        result = compute_mean_std(tmp_path, resize=RESIZE)

        ref_mean, ref_std = _reference_stats([path])
        assert np.allclose(result["mean"], ref_mean, atol=1e-9)
        assert np.allclose(result["std"], ref_std, atol=1e-9)

    def test_pixel_weighted_pooled_std(self, tmp_path: Path) -> None:
        a = _save_image(tmp_path / "a.png", 0)
        b = _save_image(tmp_path / "b.png", 255)

        result = compute_mean_std(tmp_path, resize=RESIZE)

        ref_mean, ref_std = _reference_stats([a, b])
        assert np.allclose(result["mean"], ref_mean, atol=1e-9)
        assert np.allclose(result["std"], ref_std, atol=1e-9)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            compute_mean_std(tmp_path)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_mean_std(tmp_path / "nope")

    def test_output_json_written(self, tmp_path: Path) -> None:
        _save_image(tmp_path / "gray.png", 128)
        output = tmp_path / "stats.json"

        result = compute_mean_std(tmp_path, resize=RESIZE, output=output)

        payload = json.loads(output.read_text())
        assert payload["mean"] == [round(v, 6) for v in result["mean"]]
        assert payload["std"] == [round(v, 6) for v in result["std"]]
