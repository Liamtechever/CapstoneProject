import json

import pytest

from core.config import Settings, load_settings, save_settings
from core.errors import ConfigurationError


def test_missing_settings_return_defaults_without_writing(tmp_path):
    path = tmp_path / "missing.json"
    assert load_settings(path) == Settings()
    assert not path.exists()


@pytest.mark.parametrize(
    "data",
    [
        {"use_internet": "false"},
        {"user_vram": True},
        {"api_cost_tolerance": 11},
        {"user_vram": -1},
        {"api_cost_tolerance": 1.5},
        {"extra": 5},
        [],
        "invalid",
    ],
)
def test_invalid_settings_rejected(tmp_path, data):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_settings(path)


def test_settings_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    expected = Settings(8, True, 5)
    save_settings(expected, path)
    assert load_settings(path) == expected
