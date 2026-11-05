"""Create local configuration without overwriting existing credentials or settings."""

import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def setup(root=ROOT):
    root = Path(root)
    (root / "var").mkdir(exist_ok=True)
    environment = root / ".env"
    if not environment.exists():
        template = (root / ".env.example").read_text(encoding="utf-8")
        environment.write_text(
            template.replace(
                "DJANGO_SECRET_KEY=\n", f"DJANGO_SECRET_KEY={secrets.token_urlsafe(64)}\n"
            ),
            encoding="utf-8",
        )
        print("Created private .env with a new local Django key.")
    else:
        print("Kept existing .env.")
    settings = root / "settings.json"
    if not settings.exists():
        settings.write_text(
            (root / "settings.example.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        print("Created settings.json with local-only defaults.")
    else:
        print("Kept existing settings.json.")


if __name__ == "__main__":
    setup()
