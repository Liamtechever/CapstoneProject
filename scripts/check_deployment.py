"""Validate production defaults with an ephemeral key; never load deployment secrets."""

import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["DJANGO_SETTINGS_MODULE"] = "chatproj.settings"
os.environ["DJANGO_SECRET_KEY"] = secrets.token_urlsafe(64)
os.environ["DJANGO_DEBUG"] = "false"
os.environ["DJANGO_ALLOWED_HOSTS"] = "example.invalid"
os.environ["CAPSTONE_DB_PATH"] = ":memory:"

if __name__ == "__main__":
    import django
    from django.core.management import call_command

    django.setup()
    call_command("check", deploy=True, fail_level="WARNING")
