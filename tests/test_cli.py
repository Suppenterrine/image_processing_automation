import cv2
from pathlib import Path

import numpy as np
import pytest

from yatzar.cli.apply_cmd import apply_command


def _make_test_image(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path / "a.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))


def test_apply_command_creates_outputs(tmp_path):
    img_dir = tmp_path / "img"
    out_dir = tmp_path / "out"
    _make_test_image(img_dir)
    apply_command(
        look="film_soft",
        input_dir=str(img_dir),
        output=str(out_dir),
        looks_dir=None,
        seed=None,
        make_gif=False,
        batch_size=25,
        workers=None,
        no_parallel=False,
        ext=".jpg",
        overwrite=False,
    )
    assert any(out_dir.iterdir())


def test_apply_command_reports_missing_input(tmp_path, capsys):
    apply_command(
        look="film_soft",
        input_dir=str(tmp_path / "missing"),
        output=str(tmp_path / "out"),
        looks_dir=None,
        seed=None,
        make_gif=False,
        batch_size=25,
        workers=None,
        no_parallel=False,
        ext=".jpg",
        overwrite=False,
    )
    captured = capsys.readouterr()
    assert "Keine Bilder" in captured.out
