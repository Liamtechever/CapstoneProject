# Security and migration notes

## Previously committed data requires owner follow-up

The original history contains a Django secret key, a SQLite database with privileged account records/password hashes and session data, and recordings. The cleanup removes these from the current tracked tree and generates a fresh key for new local setups. **It does not erase old commits, rotate any deployed environment, reset existing account passwords, or invalidate sessions on another installation.**

Before using an existing deployment:

1. Replace any reused Django key with a newly generated secret in that deployment's private environment. Do not retain the exposed key as a fallback.
2. Reset affected privileged account passwords; review any reuse elsewhere. Invalidate deployment sessions.
3. Rebuild the demo database through migrations and create new accounts. Never reuse the published database as a fixture.
4. Review historical recordings and decide whether to remove sensitive history. Coordinate a separate history rewrite with collaborators; it changes commit IDs and does not revoke credentials or remove other people's copies.
5. Review any deployed provider keys independently. The text audit did not identify a literal provider token, but it was not proof of an exposure-free environment.

Local copies of the original database/recording may remain ignored in a cleanup checkout. They are not used by the new app, whose database defaults to `var/db.sqlite3`.

## Configuration

`scripts/setup_local.py` generates a private local key and leaves existing files unchanged. `.env`, legacy `secrets.env`, settings, databases, images, recordings, caches, and environments are excluded from Git. Example configuration contains no real keys.

Production defaults disable debug, require a sufficiently long private key, use explicit allowed hosts, redirect to HTTPS, and enable secure cookies and HSTS. Local `.env.example` explicitly enables development mode for loopback use.

Run deployment checks against the real deployment environment:

```sh
python manage.py check --deploy
```

Do not use the Django development server for production. Configure a supported WSGI/ASGI server, HTTPS termination, static-file serving, database backups, and appropriate operational limits. HSTS is enabled for a year including subdomains with preload eligibility when debug is off; review this policy for your domain before enabling it. No proxy trust header is configured automatically.

## Application protections and limits

The web view requires login. Forms use CSRF tokens, Django templates escape messages, and logout is POST-only. Input length and display history are bounded. Provider failures do not expose raw provider messages, prompts, or credentials in the web UI/logs.

There is no distributed rate limiter or background request queue. Use only trusted demo accounts until deployment-specific rate and concurrency limits are configured. The relative cost setting is not a spending cap; use provider account controls for financial limits.

Offline mode excludes hosted inference, but installing dependencies or downloading Whisper/Coqui/Ollama weights initially requires network access. The first explicit speech call may download model weights. Hosted vision sends the captured image to the selected provider. CLI capture files are temporary; wardrobe images are retained locally until deleted.

The tracked-file hygiene script checks a narrow set of patterns and file types. It is not a replacement for GitHub secret scanning or a comprehensive full-history scan. Do not publish sensitive findings in a public issue.
