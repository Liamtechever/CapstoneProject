# Maintenance

## Automated checks

CI runs lint, formatting, tests, Django checks, and tracked-file hygiene on Windows/Linux with Python 3.11/3.12. A separate job checks locked runtime packages for published vulnerabilities. GitHub dependency update configuration covers pip and Actions monthly.

Tests use an in-memory Django database, temporary files, and mocked providers/devices. No real credentials, model weights, or original database are fixtures. If a Windows sandbox blocks its default temporary directory, use a new scratch path inside `var/`:
`python -m pytest --basetemp=var/test-scratch`.
Pytest may remove that scratch directory; do not point it at valuable files.

## Dependencies

Edit the small `.in` files, then regenerate the associated lock. Do not hand-edit hashes:

```sh
python -m piptools compile --generate-hashes --strip-extras -o requirements.txt requirements.in
uv pip compile requirements-dev.in --generate-hashes -o requirements-dev.txt
uv pip compile requirements-media.in --python-version 3.12 --python-platform windows --generate-hashes -o requirements-media.txt
python -m pip install --require-hashes -r requirements-dev.txt
python -m pip check
python -m pytest
python -m pip_audit -r requirements.txt --no-deps --disable-pip
```

Resolve the runtime lock before the development/media locks. Check both supported Python versions and OSes after changing versions; a successful install on one machine is not proof of cross-platform compatibility. Keep platform-specific media updates separate from the text app.

## Before a portfolio release

- Confirm the README commands from a fresh environment.
- Verify login, invalid login, logout, keyboard navigation, mobile layout, empty history, request errors, and repeated refresh.
- Run a real model request using accounts/models you are authorized to use; record model ID, settings, date, and observed result.
- Run media hardware checks if showcasing those paths.
- Verify every screenshot is an actual UI capture, with synthetic/non-sensitive inputs clearly identified if used.
- Complete the credential/history actions in [SECURITY.md](SECURITY.md).
- Choose a license only when ownership and intended permissions are established.
- Publish a release only after its scope and verification are accurate.

Suggested GitHub description: "Python capstone exploring local/hosted LLM routing, Django chat, and optional speech and vision integrations."
Suggested topics: `python`, `django`, `ollama`, `llm`, `speech-recognition`, `capstone-project`.

These metadata suggestions do not imply a live deployment, measured performance, or production support.
