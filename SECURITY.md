# Security policy

## Reporting a vulnerability

Please **don't open a public issue** for security problems.

Report privately through GitHub instead:
[**Report a vulnerability**](https://github.com/davidpurkiss/jevex/security/advisories/new)
(repository **Security** tab → **Report a vulnerability**). Only the maintainer can see
the report.

Please include:
- what's affected (module, function, extra) and the jevex version
- how to reproduce it, ideally a minimal document and schema
- what an attacker could do with it

You should get an acknowledgement within 7 days. Fixes are released as soon as
practical and credited in the advisory unless you'd rather stay anonymous.

## Supported versions

jevex is pre-1.0. Only the latest release on PyPI gets security fixes.

| Version | Supported |
| --- | --- |
| latest `0.x` | ✅ |
| older | ❌ |

## Scope notes

jevex processes untrusted documents (HTML, PDF, images) and, when enabled, LLM output.
Reports about these are especially welcome:
- parsing untrusted input: resource exhaustion, parser crashes, path traversal
- learned generators: regexes must stay linear-time (RE2); nothing may execute LLM-written code
- secrets: Jev or LLM API keys leaking into logs, results, packs or the store

Legal compliance when scraping third-party sites is the caller's responsibility (see
the README). It's not a vulnerability in jevex.
