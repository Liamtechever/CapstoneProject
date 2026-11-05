"""Small committed-file hygiene check; not a comprehensive historical secret scan."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{30,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"""(?i)(?:api_key|secret_key)\s*=\s*["'][^"'\r\n]{35,}["']"""),
]
FORBIDDEN_SUFFIXES = {".sqlite3", ".db", ".mp3", ".wav", ".pyc"}


def main():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    failures = []
    for relative in set(result.stdout.decode().split("\0")) - {""}:
        path = ROOT / relative
        if not path.exists():
            continue
        if (
            path.suffix in FORBIDDEN_SUFFIXES
            or relative.startswith(".idea/")
            or relative in {".env", "secrets.env", "settings.json"}
        ):
            failures.append(f"{relative}: generated or private file")
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"{relative}: unexpected binary file")
            continue
        for line, text in enumerate(content.splitlines(), 1):
            if any(pattern.search(text) for pattern in PATTERNS):
                failures.append(f"{relative}:{line}: possible secret (value redacted)")
    print("\n".join(failures) if failures else "Tracked-file hygiene check passed.")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
