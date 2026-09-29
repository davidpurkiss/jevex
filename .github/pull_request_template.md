Closes #

## What changed and why

<!-- A few lines. Link the spec section the issue cites. -->

## Departures from the spec

<!-- Anything that differs from the cited section of docs/design-spec.md, and why. "None" if nothing. -->

## Checklist

- [ ] `uv run ruff check && uv run ruff format --check` pass
- [ ] `uv run pyright` passes
- [ ] `uv run pytest` passes (no network outside tests marked `live`)
- [ ] Tests cover the new behaviour, edge cases and at least one failure path
- [ ] Public API has docstrings and is exported from `jevex/__init__.py`
- [ ] README/docs updated if usage changed
- [ ] Every **Done when** item in the issue is met, or listed here with the reason it isn't
