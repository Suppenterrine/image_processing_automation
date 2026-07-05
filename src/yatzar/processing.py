"""Verarbeitung eines einzelnen Batches von Bilddateien. Läuft entweder direkt
im Hauptprozess (Fast-Path) oder als Worker-Funktion in einem ProcessPoolExecutor
(muss deshalb picklebar sein: nur Strings/Dicts/Pfade als Argumente, keine
Funktionsobjekte oder Live-Handles)."""

import random
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from yatzar import io_utils, looks


@dataclass
class BatchResult:
    batch_id: int
    saved: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)


def process_batch(
    files: list[Path],
    look_name: str,
    look_cfg: dict,
    output_dir: Path,
    seed: int | None,
    batch_id: int,
    ext: str,
    progress_state,
) -> BatchResult:
    look_fn = looks.get(look_cfg["type"])
    rng = random.Random(seed + batch_id) if seed is not None else random.Random()
    result = BatchResult(batch_id=batch_id)
    total = len(files)
    progress_state[batch_id] = {"completed": 0, "total": total}

    for i, file_path in enumerate(files):
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                raise ValueError("Bild konnte nicht geladen werden")
            out = look_fn(img, look_cfg, rng)
            saved = io_utils.save_image(output_dir, look_name, file_path, out, ext=ext)
            result.saved.append(saved)
        except Exception as exc:
            result.errors.append((file_path, str(exc)))
        finally:
            progress_state[batch_id] = {"completed": i + 1, "total": total}

    return result
