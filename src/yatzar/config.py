"""Laden von Look-Presets aus einem Verzeichnis mit einer YAML-Datei pro Look."""

from importlib import resources
from importlib.abc import Traversable
from pathlib import Path

import yaml

from yatzar.looks import available as look_types


def default_looks_dir() -> Traversable:
    return resources.files("yatzar.data.looks")


def load_look_configs(dir_path: Path) -> dict[str, dict]:
    """Liest jede *.yaml-Datei im Verzeichnis; Dateiname (ohne Endung) wird zum Look-Namen."""
    configs = {}
    for path in sorted(Path(dir_path).glob("*.yaml")):
        configs[path.stem] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return configs


def validate_look_config(name: str, cfg: dict) -> list[str]:
    """Minimale, best-effort-Validierung eines geladenen Look-YAMLs.

    Gibt eine Liste von Fehlermeldungen zurück. Bei leerer Liste ist das Config
    syntaktisch/strukturell okay. Diese Validierung ist kein Schema-Ersatz, sondern
    ein schnelles Netz gegen offensichtliche Tippfehler und kaputte Strukturen.
    """
    errors: list[str] = []
    if not isinstance(cfg, dict):
        return [f"Look '{name}': Root ist kein Mapping."]

    look_type = cfg.get("type")
    if not look_type:
        errors.append(f"Look '{name}': Fehlendes Feld 'type'.")
    elif look_type not in look_types():
        errors.append(
            f"Look '{name}': Unbekannter Typ '{look_type}'. Verfügbar: {', '.join(sorted(look_types()))}"
        )

    active_effect_blocks = {
        "barrel_distortion",
        "gaussian_blur",
        "motion_blur",
        "grain",
        "color",
        "tone",
        "vignette",
        "paper_tone",
        "pre_blur",
        "soft_blur",
        "outline",
    }
    for key in cfg:
        if key == "type":
            continue
        block = cfg[key]
        if key in active_effect_blocks:
            if not isinstance(block, dict):
                errors.append(f"Look '{name}': '{key}' sollte ein Mapping sein.")
                continue
            if "enabled" not in block:
                errors.append(
                    f"Look '{name}': Effektblock '{key}' fehlt das Feld 'enabled'."
                )
            if not isinstance(block.get("enabled"), bool):
                errors.append(
                    f"Look '{name}': '{key}.enabled' muss ein boolescher Wert sein."
                )

    return errors
