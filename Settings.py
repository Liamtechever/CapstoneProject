"""Explicit configuration. Importing this module never starts voice setup."""

from core.config import Settings, load_settings, save_settings

__all__ = ["Settings", "load_settings", "save_settings"]


def ask_user_for_settings():
    from interfaces.voice_setup import configure

    return configure()


if __name__ == "__main__":
    ask_user_for_settings()
