from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from preprocess_retina_datasets.crop import (
    crop_dataset,
    crop_image,
    detect_retina_bbox,
    square_center_bbox,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_DIR = FIXTURES / "eyepacs_sample"

SAMPLE_IMAGES = sorted(SAMPLE_DIR.iterdir())

FIRST = SAMPLE_IMAGES[0] if SAMPLE_IMAGES else Path("/nonexistent")
CROP_SIZE = 512


class TestDetectRetinaBbox:
    def test_all_sample_images_detect_bbox(self) -> None:
        for path in SAMPLE_IMAGES:
            image = Image.open(path)
            bbox = detect_retina_bbox(image)
            assert bbox is not None, f"{path.name}: expected retina bbox"
            left, upper, right, lower = bbox
            assert left < right, f"{path.name}: invalid bbox {bbox}"
            assert upper < lower, f"{path.name}: invalid bbox {bbox}"

    def test_square_image_returns_none(self) -> None:
        image = Image.new("RGB", (512, 512), color="black")
        bbox = detect_retina_bbox(image)
        assert bbox is None

    def test_tall_image_returns_none(self) -> None:
        image = Image.new("RGB", (100, 200), color="black")
        bbox = detect_retina_bbox(image)
        assert bbox is None


class TestSquareCenterBbox:
    def test_wider_than_tall(self) -> None:
        image = Image.new("RGB", (200, 100), color="black")
        bbox = square_center_bbox(image)
        assert bbox == (50, 0, 150, 100)

    def test_taller_than_wide(self) -> None:
        image = Image.new("RGB", (100, 200), color="black")
        bbox = square_center_bbox(image)
        assert bbox == (0, 50, 100, 150)

    def test_already_square(self) -> None:
        image = Image.new("RGB", (100, 100), color="black")
        bbox = square_center_bbox(image)
        assert bbox == (0, 0, 100, 100)


class TestCropImage:
    def test_output_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "out.jpeg"
            crop_image(FIRST, dst, crop_size=CROP_SIZE)
            out = Image.open(dst)
            assert out.size == (CROP_SIZE, CROP_SIZE)

    def test_output_has_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "out.jpeg"
            crop_image(FIRST, dst, crop_size=CROP_SIZE)
            arr = np.array(Image.open(dst))
            assert arr.min() < arr.max(), "output image should not be uniform"

    def test_fallback_square_crop(self) -> None:
        image = Image.new("RGB", (200, 100), color=(10, 20, 30))
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "input.jpeg"
            dst = Path(tmp) / "output.jpeg"
            image.save(src)
            crop_image(src, dst, crop_size=CROP_SIZE)
            out = Image.open(dst)
            assert out.size == (CROP_SIZE, CROP_SIZE)

    def test_crops_all_sample_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            for path in SAMPLE_IMAGES:
                dst = out_dir / path.name
                crop_image(path, dst, crop_size=CROP_SIZE)
                img = Image.open(dst)
                assert img.size == (CROP_SIZE, CROP_SIZE)

    def test_skip_existing_skips_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "out.jpeg"
            crop_image(FIRST, dst, crop_size=CROP_SIZE, skip_existing=True)
            assert dst.exists()
            mtime_before = dst.stat().st_mtime
            crop_image(FIRST, dst, crop_size=CROP_SIZE, skip_existing=True)
            mtime_after = dst.stat().st_mtime
            assert mtime_before == mtime_after, "file should not have been overwritten"

    def test_no_skip_existing_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "out.jpeg"
            crop_image(FIRST, dst, crop_size=CROP_SIZE)
            mtime_before = dst.stat().st_mtime
            crop_image(FIRST, dst, crop_size=CROP_SIZE)
            mtime_after = dst.stat().st_mtime
            assert mtime_before != mtime_after, "file should have been overwritten"


class TestCropDataset:
    def test_crops_all_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            failed = crop_dataset(
                input_dir=SAMPLE_DIR,
                output_dir=out_dir,
                crop_size=CROP_SIZE,
                num_workers=2,
            )
            assert failed == 0
            out_files = sorted(out_dir.iterdir())
            assert len(out_files) == len(SAMPLE_IMAGES)
            for f in out_files:
                img = Image.open(f)
                assert img.size == (CROP_SIZE, CROP_SIZE)

    def test_preserves_subfolder_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            sub_dir = Path(tmp) / "sub"
            sub_dir.mkdir()
            for f in list(SAMPLE_IMAGES)[:2]:
                (sub_dir / f.name).write_bytes(f.read_bytes())

            failed = crop_dataset(
                input_dir=Path(tmp),
                output_dir=out_dir,
                crop_size=CROP_SIZE,
                num_workers=2,
            )
            assert failed == 0
            assert len(list(out_dir.iterdir())) == 1
            assert len(list((out_dir / "sub").iterdir())) == 2

    def test_skip_existing_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            crop_dataset(
                input_dir=SAMPLE_DIR,
                output_dir=out_dir,
                crop_size=CROP_SIZE,
                num_workers=2,
            )
            count_first = len(list(out_dir.iterdir()))

            crop_dataset(
                input_dir=SAMPLE_DIR,
                output_dir=out_dir,
                crop_size=CROP_SIZE,
                num_workers=2,
                skip_existing=True,
            )
            count_second = len(list(out_dir.iterdir()))
            assert count_first == count_second, "no new files should be created"

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_dir = Path(tmp) / "empty"
            empty_dir.mkdir()
            out_dir = Path(tmp) / "out"
            failed = crop_dataset(
                input_dir=empty_dir,
                output_dir=out_dir,
                crop_size=CROP_SIZE,
                num_workers=2,
            )
            assert failed == 0
            assert not out_dir.exists() or not any(out_dir.iterdir())
