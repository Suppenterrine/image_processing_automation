"""Datei-Discovery und Speichern von Bildern."""

from pathlib import Path

import cv2
import numpy as np

INPUT_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def collect_input_files(input_dir: str) -> list[Path]:
    files = []
    for pattern in INPUT_PATTERNS:
        files.extend(Path(input_dir).glob(pattern))
    return sorted(files)


def save_image(
    output_dir: Path,
    look_name: str,
    input_path: Path,
    img: np.ndarray,
    ext: str = ".jpg",
    overwrite: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{look_name}_{input_path.stem}{ext}"
    out_path = output_dir / out_name
    if out_path.exists() and not overwrite:
        # Append a numeric suffix instead of failing silently or clobbering.
        stem = f"{look_name}_{input_path.stem}"
        candidate = output_dir / f"{stem}_1{ext}"
        idx = 1
        while candidate.exists():
            idx += 1
            candidate = output_dir / f"{stem}_{idx}{ext}"
        out_path = candidate
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f"Konnte Bild nicht speichern: {out_path}")
    return out_path
