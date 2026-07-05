"""Reine Arithmetik für automatisches, ausgeglichenes Batching. Kein I/O, keine Prozess-Logik."""

import math
import os


def compute_batches(n_items: int, batch_size: int) -> list[tuple[int, int]]:
    """Liefert (start, end)-Indexpaare, ausgeglichen auf die Batches verteilt.

    Beispiel: n_items=100, batch_size=25 -> 4 Batches à 25.
    n_items=110, batch_size=25 -> 5 Batches à 22 statt unausgeglichen 4x25+1x10.
    """
    if n_items <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size muss > 0 sein")

    num_batches = math.ceil(n_items / batch_size)
    base, remainder = divmod(n_items, num_batches)
    sizes = [base + 1] * remainder + [base] * (num_batches - remainder)

    bounds = []
    start = 0
    for size in sizes:
        bounds.append((start, start + size))
        start += size
    return bounds


def decide_workers(num_batches: int, requested: int | None) -> int:
    """Anzahl paralleler Worker: nie mehr als es Batches gibt."""
    if num_batches <= 0:
        return 0
    if requested is not None:
        return max(1, min(requested, num_batches))
    return max(1, min(os.cpu_count() or 1, num_batches))
