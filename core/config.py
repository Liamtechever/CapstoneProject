"""Explicit configuration; reading defaults never writes files or starts setup."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from core.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = PROJECT_ROOT / "settings.json"


@dataclass(frozen=True)
class Settings:
    user_vram: int = 4
    use_internet: bool = False
    api_cost_tolerance: int = 0

    def __post_init__(self):
        if type(self.use_internet) is not bool:
            raise ConfigurationError("use_internet must be a JSON boolean.")
        for name, upper in (("user_vram", 4096), ("api_cost_tolerance", 10)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= upper:
                raise ConfigurationError(f"{name} must be an integer from 0 to {upper}.")


def load_settings(path=None):
    path = Path(path or os.getenv("CAPSTONE_SETTINGS_FILE", SETTINGS_FILE))
    if not path.exists():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - set(Settings.__dataclass_fields__):
            raise ValueError("Unknown settings fields")
        return Settings(**data)
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigurationError("Invalid settings file. See settings.example.json.") from exc


def save_settings(settings, path=None):
    path = Path(path or os.getenv("CAPSTONE_SETTINGS_FILE", SETTINGS_FILE))
    path.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")


def load_environment():
    """Load optional local credentials only when a provider/configuration needs them."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / "secrets.env", override=False)  # Legacy local setup.
