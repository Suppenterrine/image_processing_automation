from pathlib import Path

import cv2
import numpy as np
import pytest

from yatzar import io_utils


def _write_jpg(path: Path, img: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Test helper failed to write image: {path}")


def test_collect_input_files_discovers_images(tmp_path):
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    _write_jpg(img_dir / "a.jpg", np.zeros((4, 4, 3), dtype=np.uint8))
    _write_jpg(img_dir / "b.JPG", np.zeros((4, 4, 3), dtype=np.uint8))
    _write_jpg(img_dir / "c.png", np.zeros((4, 4, 3), dtype=np.uint8))
    (img_dir / "d.txt").write_text("x", encoding="utf-8")
    files = io_utils.collect_input_files(str(img_dir))
    assert [p.name for p in files] == ["a.jpg", "b.JPG", "c.png"]


def test_save_image_writes_expected_path(tmp_path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out = io_utils.save_image(tmp_path / "out", "my_look", tmp_path / "in.png", img, ext=".png")
    assert out.name == "my_look_in.png"
    assert out.exists()


def test_save_image_does_not_overwrite_without_flag(tmp_path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out1 = io_utils.save_image(tmp_path, "look", Path("img.png"), img, ext=".png")
    img2 = np.full((8, 8, 3), 255, dtype=np.uint8)
    out2 = io_utils.save_image(tmp_path, "look", Path("img.png"), img2, ext=".png", overwrite=False)
    assert out1.name == "look_img.png"
    assert out2.name == "look_img_1.png"
    loaded1 = cv2.imread(str(out1), cv2.IMREAD_UNCHANGED)
    loaded2 = cv2.imread(str(out2), cv2.IMREAD_UNCHANGED)
    assert float(loaded1.mean()) == 0.0
    assert float(loaded2.mean()) == 255.0


def test_save_image_overwrite_overwrites_with_flag(tmp_path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    out1 = io_utils.save_image(tmp_path, "look", Path("img.png"), img, ext=".png")
    img2 = np.full((8, 8, 3), 200, dtype=np.uint8)
    out2 = io_utils.save_image(tmp_path, "look", Path("img.png"), img2, ext=".png", overwrite=True)
    assert out1 == out2
    loaded = cv2.imread(str(out2), cv2.IMREAD_UNCHANGED)
    assert float(loaded.mean()) == 200.0


def test_save_image_auto_creates_output_dir(tmp_path):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = io_utils.save_image(tmp_path / "nested" / "out", "look", Path("x.png"), img)
    assert out.exists()
