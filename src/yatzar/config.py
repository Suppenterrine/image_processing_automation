"""Laden von Look-Presets aus einem Verzeichnis mit einer YAML-Datei pro Look."""

from importlib import resources
from pathlib import Path

import yaml


def default_looks_dir() -> Path:
    return resources.files("yatzar.data.looks")


def load_look_configs(dir_path: Path) -> dict[str, dict]:
    """Liest jede *.yaml-Datei im Verzeichnis; Dateiname (ohne Endung) wird zum Look-Namen."""
    configs = {}
    for path in sorted(Path(dir_path).glob("*.yaml")):
        configs[path.stem] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return configs
