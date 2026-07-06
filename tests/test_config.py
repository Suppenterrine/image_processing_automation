import pytest

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
