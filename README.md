# jevex

[![CI](https://github.com/davidpurkiss/jevex/actions/workflows/ci.yml/badge.svg)](https://github.com/davidpurkiss/jevex/actions/workflows/ci.yml)

Extract typed records from web pages and PDFs using [Jev](https://docs.typesafe.ai/introduction), TypeSafe AI's System One model.

jevex narrows each document step by step (document → component → statement → value), asking Jev small atomic questions at every level. Every LLM fallback is turned into a declarative generator, so each run needs fewer LLM calls than the last.

> **Status:** early planning. This release only reserves the package name; there is no usable API yet.

## Development

```sh
uv sync                     # create .venv with dev tools
uv run pre-commit install   # ruff on every commit
uv run ruff check && uv run ruff format --check
uv run pyright
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow, and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache 2.0
