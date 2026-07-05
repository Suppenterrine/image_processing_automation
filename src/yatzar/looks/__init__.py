"""Registry für Look-Engines. Jedes Look-Modul registriert sich per @register(name) beim Import."""

import random
from collections.abc import Callable

import numpy as np

LookFn = Callable[[np.ndarray, dict, random.Random], np.ndarray]

_REGISTRY: dict[str, LookFn] = {}


def register(name: str) -> Callable[[LookFn], LookFn]:
    def decorator(fn: LookFn) -> LookFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get(name: str) -> LookFn:
    if name not in _REGISTRY:
        available_names = ", ".join(available())
        raise KeyError(f"Unbekannter Look-Typ '{name}'. Verfügbar: {available_names}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


# Import als Seiteneffekt: jedes Modul registriert seine Look-Funktion.
from yatzar.looks import engraving, gazette, gazette_halftone, standard, tri_x  # noqa: E402,F401
