# Cleanup validation — November 4–5, 2026

The cleanup was checked locally on Windows with Python 3.12.10. These results describe this revision, not a claim about future changes or model quality.

- 70 automated tests passed. They cover routing limits, malformed classification/settings, provider payloads and errors, authentication, CSRF, escaping, session isolation/history limits, import side effects, speech caching/silence logic, camera cleanup, wardrobe selection/CRUD/path boundaries, and non-overwriting local setup.
- Ruff lint and formatting checks passed.
- Runtime and optional media locks returned no known vulnerabilities from the advisory service at the time of checking.
- The text/development environment passed `pip check`.
- Standard Django checks and production-default deployment checks passed.
- Fresh migrations succeeded with zero bundled user accounts.
- The installed development environment matches every version in the final lock. An offline pip dry run found the locked requirements already satisfied; this was not a fresh reinstall. A metadata check found no missing declared dependencies for the Windows/Linux and Python 3.11/3.12 combinations.
- A real browser verified local login/logout, empty authenticated chat, and the mobile layout at a 390-pixel viewport. No horizontal overflow was observed. The disposable preview account was removed.
- Screenshots in `docs/images/` are actual UI captures with no fabricated model responses.
- Current-tree hygiene checks passed. Old sensitive commits remain in history.

The media lock was resolved and audited but the heavy media packages and model weights were not installed for these checks. Hardware, live Ollama inference, hosted model access, and output quality require manual verification. No paid model calls were made.

GitHub CI is configured for Windows and Linux on Python 3.11/3.12. Its remote result should be read from the pull request checks; local results alone do not establish that matrix.

See [security follow-up](SECURITY.md) for deployed credential resets and the separate history-cleanup decision. Repository About metadata suggestions are in [maintenance](MAINTENANCE.md); they require repository-settings access.
