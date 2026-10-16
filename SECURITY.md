# Security policy

## Reporting a vulnerability

Please open a private security advisory on GitHub (Security → Report a vulnerability) rather than a
public issue. Include steps to reproduce and the affected version or commit.

## Design notes

- **Model artifacts.** joblib files can execute code when loaded. The app loads only artifacts listed
  in `models/registry.json` whose SHA-256 matches both the registry and the version manifest, and only
  from inside the models directory. It never loads uploaded or unknown model files.
- **Admin actions.** Training jobs and promotion require `ADMIN_TOKEN` (constant-time comparison). With
  no token configured, admin endpoints return 403. Training runs out of process with a timeout and
  one job at a time.
- **Public API.** Strict Pydantic schemas (unknown fields rejected), 16 KB body limit, per-client rate
  limits, sanitized error messages, and `nosniff` / `DENY` / CSP / `no-store` headers.
- **Privacy.** Only fictional scenarios are accepted. The monitoring database stores counts, latency,
  status and binned feature counts, never scenario values, IP addresses or prompts.
- **AI analyst.** API keys stay server-side. Tools are allow-listed and read-only. Tool output and user
  text are wrapped as untrusted data. Requests containing personal identifiers or asking for credit
  decisions are refused before any LLM call. Daily caps and rate limits bound cost.
- **Dependencies.** `pip-audit` runs in CI against `requirements.txt`.
