"""Isolated test configuration; never use a user's database."""

import os
import secrets

os.environ["DJANGO_SECRET_KEY"] = secrets.token_urlsafe(64)
os.environ["DJANGO_DEBUG"] = "true"
os.environ["DJANGO_ALLOWED_HOSTS"] = "testserver,localhost,127.0.0.1"
from chatproj.settings import *  # noqa: E402,F403

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
