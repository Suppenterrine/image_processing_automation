import pytest
import yaml

from pathlib import Path

from yatzar.config import validate_look_config


def _cfg(name, **overrides):
    base = {"type": name}
    base.update(overrides)
    return base


def test_valid_standard_config_passes():
    errors = validate_look_config("film_soft", _cfg("standard"))
    assert errors == []


def test_missing_type_is_reported():
    errors = validate_look_config("broken", {})
    assert any("Fehlendes Feld 'type'" in e for e in errors)


def test_unknown_type_is_reported():
    errors = validate_look_config("broken", _cfg("doesnotexist"))
    assert any("Unbekannter Typ" in e for e in errors)


def test_effect_block_without_enabled_is_reported():
    cfg = _cfg("standard", grain={"amount": 1})
    errors = validate_look_config("film_soft", cfg)
    assert any("'grain' fehlt das Feld 'enabled'" in e for e in errors)


def test_effect_block_enabled_must_be_bool():
    cfg = _cfg("standard", grain={"enabled": "yes", "amount": 1})
    errors = validate_look_config("film_soft", cfg)
    assert any("'grain.enabled' muss ein boolescher Wert sein" in e for e in errors)


def test_root_must_be_mapping():
    errors = validate_look_config("film_soft", "bad")
    assert errors != []


def test_tri_x_pushed_preset_has_enabled_flags():
    path = Path("src/yatzar/data/looks/tri_x_pushed.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = validate_look_config("tri_x_pushed", cfg)
    assert errors == []


def test_film_profile_presets_validate():
    preset_paths = [
        "src/yatzar/data/looks/kodak_vision3_2383_print.yaml",
        "src/yatzar/data/looks/kodak_vision3_5219.yaml",
        "src/yatzar/data/looks/kodak_5247.yaml",
        "src/yatzar/data/looks/kodak_ektachrome.yaml",
        "src/yatzar/data/looks/kodak_portra.yaml",
        "src/yatzar/data/looks/fujifilm_eterna.yaml",
        "src/yatzar/data/looks/fujifilm_velvia.yaml",
        "src/yatzar/data/looks/kodak_tri_x_400.yaml",
        "src/yatzar/data/looks/cinestill_800t.yaml",
    ]

    for path_str in preset_paths:
        path = Path(path_str)
        assert path.exists(), f"Preset fehlt: {path_str}"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        errors = validate_look_config(path.stem, cfg)
        assert errors == [], f"Preset {path_str} ist ungültig: {errors}"
