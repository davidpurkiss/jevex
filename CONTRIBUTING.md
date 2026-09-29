# Contributing to jevex

Thanks for your interest in jevex. The project is at an early stage: the core pipeline is
being built issue by issue from the design spec, so the best way to help right now is to
report bugs, comment on design questions, or pick up an open issue.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Where things are

- **Design spec:** [`docs/design-spec.md`](docs/design-spec.md). Most issues cite one of
  its sections in a **Spec:** line; changes should follow that section.
- **Backlog:** [GitHub issues](https://github.com/davidpurkiss/jevex/issues), grouped into
  milestones `0`–`8`. An issue's "blocked by" links show what has to land first.
- **Conventions and architecture:** [`CLAUDE.md`](CLAUDE.md) has the module map, coding
  conventions and gotchas. It is written for AI agents working on the repo, but it is the
  canonical list for human contributors too.

Before starting on something non-trivial, comment on its issue (or open one) so the work
isn't duplicated and any design questions get settled first.

## Development setup

You need [uv](https://docs.astral.sh/uv/). It installs the right Python (3.12, from
`.python-version`) for you.

```sh
git clone https://github.com/davidpurkiss/jevex.git
cd jevex
uv sync                     # create .venv with the package and dev tools
uv run pre-commit install   # run ruff on every commit
```

## Checks

These are the checks CI runs (lint, then pyright and pytest on Python 3.12 and 3.13). Run
them before you push:

```sh
uv run ruff check && uv run ruff format --check
uv run pyright                            # strict mode, src + tests
uv run pytest                             # network blocked; live tests skipped
```

Tests never touch the network: `pytest-socket` blocks it. Use `jevex.testing.FakeJev` for
scripted Jev answers, or a `Cassette` to replay recorded ones. Two opt-in modes need API
keys in your environment (or a gitignored `.env`) and cost money:

```sh
uv run pytest --live -m live              # run tests marked `live` against real APIs
JEVEX_RECORD=1 uv run pytest tests/...    # re-record Jev cassettes
```

## Writing code

The full list is in [`CLAUDE.md`](CLAUDE.md#conventions). The short version:

- Everything is typed and must pass pyright in strict mode. Use frozen Pydantic v2 models
  for values that cross a boundary or get serialised, and dataclasses for internal state.
- Every module starts with `from __future__ import annotations`.
- Anything that does I/O or calls Jev or an LLM is `async`; CPU-only work is sync.
- Only `jevex/jev.py` imports `typesafe_sdk`. Question text comes from the schema layer,
  never hard-coded in a stage.
- Raise specific exceptions; never swallow them.
- Heavy dependencies are optional extras, imported lazily where they're needed.
- One test module per source module (`tests/test_<module>.py`). Cover edge cases and at
  least one failure path, and assert generated question text exactly.
- Public names get docstrings and are exported from `jevex/__init__.py`.

## Pull requests

- Keep one issue per PR, and start the description with `Closes #<issue>`.
- A change is done when the checks above pass, the new behaviour has tests, it follows the
  cited spec section, public API is documented and exported, and every item in the issue's
  **Done when** list is met. The [PR template](.github/pull_request_template.md) walks
  through this.
- If you depart from the spec, say so under **Departures from the spec** and explain why.
- If you find a bug or missing piece outside the issue, open a new issue rather than
  widening the PR.
- `main` is protected: all CI checks must pass, and PRs are squash-merged.

## Reporting bugs and requesting features

Use the [issue templates](https://github.com/davidpurkiss/jevex/issues/new/choose). For
bugs, a minimal document (or URL) and schema that reproduce the problem help most.

**Security problems** go through private reporting instead of issues. See
[SECURITY.md](SECURITY.md).

## License

jevex is licensed under [Apache 2.0](LICENSE). By contributing, you agree that your
contributions are licensed under the same terms.
