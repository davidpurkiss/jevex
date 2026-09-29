# jevex

Python library that extracts typed records from HTML and PDFs using Jev (TypeSafe AI's
System One model). It narrows each document step by step (document → component →
statement → value), asking Jev small atomic questions at each level. LLM fallbacks are
turned into declarative generators, so each run needs fewer LLM calls than the last.

- Design spec: `docs/design-spec.md`. Issues cite its sections as **Spec:** lines.
- Backlog: GitHub issues in milestones `0`–`8`. Ordering uses GitHub's built-in
  "blocked by" dependencies.

## Commands

```sh
uv sync --all-extras                      # dev env with every extra (as CI does)
uv run ruff check && uv run ruff format --check
uv run pyright                            # strict mode, src + tests
uv run pytest                             # network blocked; live tests skipped
uv run pytest --live -m live              # real APIs (needs keys; costs money)
JEVEX_RECORD=1 uv run pytest tests/...    # re-record Jev cassettes
```

The package version lives only in `src/jevex/__init__.py` (`__version__`). Releases
come from publishing a GitHub release (`release.yml`); agents never do that.

CI (`.github/workflows/ci.yml`) runs lint, then pyright + pytest on 3.12 and 3.13.
`main` is protected: changes go through a PR and all three checks must pass.

## Architecture map

| Module | What lives there |
| --- | --- |
| `categorise.py` | Statement categorisation (stage 10): `JevStatementClassifier` (one request per statement with one Choice per schema, options limited to the fields the component gate passed; items are `ToClassify`), `CategoriseStage` (answers with full distributions on `SchemaRun.categories`). `select.field_statements` routes the top field plus any other at p ≥ `ALSO_CATEGORY_P` |
| `document.py` | `Document` (bytes + content type; base64 in JSON), content sniffing |
| `layout.py` | `Component` tree, `Location` union (`DomLocation`, `PageLocation`, `ImageLocation`), `BBox` |
| `statements.py` | `Statement`, `Span`, `Candidate`, `NormaliserStep` (compact YAML form) |
| `split.py` | Statement splitting (stage 9): `DefaultSplitter` (pysbd sentences, list items, `Label: value` pairs, headings, captions, alt text; tables are #25), `StatementStage` |
| `entities.py` | `EntityScope` |
| `jev.py` | The only code that talks to Jev: `Noul`/`Choice`/`Score` questions, answers, `JevClient` (batching, splitting, metering), `JevBackend` protocol, `TypeSafeBackend` |
| `schema.py` | `jevex.Field`, `Questions`, `SchemaConfig`, `SchemaSpec`/`FieldSpec` and every generated question |
| `interfaces.py` | Protocols for the 15 pluggable parts + shared types (`ParsedDocument`, `GateDecision`, `Selection`...). Its docstring maps each protocol to the issue that ships its default |
| `component_gate.py` | Component gate (stage 7): `gate_units` (a container's run of blocks, chunked; tables alone), `NoulComponentGate` (one Noul per unit × field group, all schemas in one request), `ComponentGateStage`. Results on `SchemaRun.component_ids`; `EntityStage` keeps only passing components; `SchemaRun.relevant_fields(cid)` for the classifier |
| `pipeline.py` | `Stage` protocol, `Context`/`SchemaRun` (per-document state), `Pipeline` composition, `for_each_scope`/`for_each_schema` |
| `extractor.py` | `Extractor`, `STAGE_ORDER`/`DEFAULT_STAGES`/`default_pipeline()`, `ExtractionResult`, `DocumentMeta` |
| `results.py` | `FieldMeta`, `Source`, `Extracted` records, `partial_model`, thresholds |
| `eval.py` | `jevex eval`: corpus (`truth.json`) loading, record matching, per-field tolerances and scores, `EvalReport` |
| `testing.py` | `FakeJev` / `FakeLLM` (scripted answers), `Cassette` / `LLMCassette` record/replay |
| `llm/` | `LLM` protocol, `LLMResponse`, price table, spend cap; adapters in `llm/anthropic.py`, `llm/openai.py`, `llm/litellm.py` (extras) |

How the parts fit together:
- A stage is anything with `name` and `async run(ctx)`. It reads earlier results from
  `ctx` and writes its own there.
- A pluggable part (for example a `Cleaner`) is a narrow protocol. A stage adapter wraps
  it, and the default adapter gets added to `DEFAULT_STAGES` in spec order.
- Within a stage, fan out with `for_each_scope`/`for_each_schema`, and put every question
  about one state into a single `ctx.jev.ask(...)` call.
- A default stage's `name` must be one of `extractor.STAGE_ORDER` (the spec's order). Add
  it to `DEFAULT_STAGES` in any position; `default_pipeline()` sorts it into place.
- Stages record what they find with `run.set_field(scope, field, FieldMeta(...))`
  (`jevex.results`). Records, thresholds and `result.one()` are built from that; a record
  is a partial model holding only the fields (`strict()` gives the real model).

Put new modules where the spec's structure suggests. For example: `jevex/clean.py`,
`jevex/layout_html.py`, `jevex/generators/`, `jevex/normalise.py`, `jevex/store/`,
`jevex/llm/`, `jevex/contrib/scrapy.py`, `jevex/cli.py`. Heavy dependencies are optional
extras (`pdf`, `ocr`, `anthropic`, `openai`, `litellm`, `postgres`, `server`, `scrapy`).
Import them lazily inside the code that needs them.

## Conventions

- **Types:** everything is typed and pyright strict must pass. Use Pydantic v2 models for
  data that crosses a boundary or gets serialised (make them frozen when they're values),
  and dataclasses for internal mutable state. Avoid `Any` in public signatures unless the
  value really is arbitrary (field values, normaliser args).
- **Imports:** annotations on Pydantic models are evaluated at runtime, so keep those
  imports at module level. Put other annotation-only imports under `if TYPE_CHECKING:`
  (ruff's TC rules enforce this). Every module starts with
  `from __future__ import annotations`.
- **Async:** anything that does I/O or calls Jev or an LLM is `async`. CPU-only work (parsing,
  splitting, generators, normalisers) is sync.
- **Jev:** only `jev.py` imports `typesafe_sdk`. Stages build jevex's own question models.
  Question text comes from `SchemaSpec`/`FieldSpec`, never hard-coded in a stage, so that
  `Questions(...)` overrides keep working.
- **Errors:** raise specific exceptions (subclass `JevError` or add your own). Never swallow
  exceptions. Record expected, recoverable conditions as `ctx.event(...)` instead of raising.
- **Style:** ruff with line length 100. Docstrings explain why or give the contract, not
  a restatement of the code. Match the surrounding code's comment density.
- **Tests:** one test module per source module (`tests/test_<module>.py`). Use `FakeJev`
  (or a `Cassette`) and never the network; `pytest-socket` blocks it. Async tests are
  plain `async def` (asyncio auto mode). When a stage generates questions, assert the exact
  question text, because it is user-visible behaviour.
- **Exports:** add public names to `jevex/__init__.py` and `__all__`.

## Definition of done

A change is done when all of these hold:
1. `ruff check`, `ruff format --check`, `pyright` and `pytest` all pass locally.
2. There are tests for the new behaviour, including edge cases and at least one failure path.
3. It follows the spec section the issue cites. Any departure is named in the PR
   description under **Departures from the spec**.
4. Public API has docstrings and is exported. README/docs are updated if usage changed.
5. Every checkbox in the issue's **Done when** list is met, or the PR says which ones
   aren't and why (with a follow-up comment on the relevant issue).

## Rules for agents working unattended

The loop's run procedure is `.claude/loop.md`. The run log is issue #84. It normally
runs locally via `scripts/agent-loop.sh`: fresh headless sessions on the owner's
subscription, in a dedicated clone under `~/.jevex-agent`. A cloud routine can use the
same prompt ("Follow .claude/loop.md...").

- **One issue per branch per PR.** Branch `agent/<issue>-<slug>`; the PR body starts
  with `Closes #<issue>`.
- **Build mode (for now):** once CI is green and the reviewer says `ready`, squash-merge
  your own PR and label it `needs-review`; the owner reviews merged work later. Draft
  PRs are never merged. #74 tracks switching to review mode.
- **No AI attribution** in commits, PRs, issues or comments (no `Co-Authored-By`,
  "Generated with" lines or session links). `.claude/settings.json` turns off Claude
  Code's automatic attribution.
- **Never push to `main`,** force-push, rewrite history, or change branch protection, CI
  workflows or repo settings unless the issue is about exactly that.
- **Never publish or contact anyone:** no PyPI releases, tags, GitHub releases, emails
  or posts outside this repo's issues and PRs.
- **No real API calls** (Jev, LLMs, websites) outside tests marked `live`, and only in
  runs where the loop's secrets and budget rules (#72, `.claude/loop.md` **Live calls**)
  allow them. Never commit keys; `.env` is gitignored. `JEVEX_JEV_MAX_COST_USD` hard-caps
  Jev spend per process (`JevBudgetExceededError`). Live tests take the
  `typesafe_api_key` fixture, which skips them when there's no key.
- **Stay in scope.** If you find a bug or missing piece outside the issue, open a new
  issue for it (with an `agent-ready` or `needs-human` label and "blocked by" links)
  instead of fixing it in the same PR.
- **Only work on `agent-ready` issues** whose blockers are all closed. Leave
  `needs-human` issues alone.
- **When stuck** (unclear spec, a decision needed, a missing credential, repeated CI
  failure): stop, comment on the issue with what you tried and what you need, and label
  it `agent-blocked`. Don't guess at product decisions.
- **Before opening a PR,** run the review described below and fix what it finds.

## Gotchas

- `typesafe-sdk` uses `httpx2`, not `httpx`. For tests of the real backend, use
  `httpx2.MockTransport` with `AsyncTypeSafeClient(transport=...)`.
- Jev limits (Jev 1.13): 32k tokens for state plus the longest question, 64k per
  request, Choice up to 255 options, Score 2–10 levels. `JevClient` estimates tokens and
  splits or raises (`StateTooLargeError`), so stages must chunk oversized components.
- Noul answers have only a probability (`NoulAnswer.p`) and no confidence.
- `extract_sync` reuses one private event loop. Don't create per-call loops around the
  shared `JevClient`.
- `jevex.Field` isn't recognised as a field specifier by type checkers, so a required
  field looks optional in direct constructor calls. This is a known limitation.

## Review before a PR

Before opening a PR, run the `reviewer` subagent (`.claude/agents/reviewer.md`) with
the issue number. Fix every MUST FIX, then run it again until the verdict is `ready`,
up to 3 rounds. Paste its final DONE-WHEN CHECK and any SHOULD FIX items you chose not
to fix (with a reason) into the PR description under **Review**. If it is still
`changes-needed` after 3 rounds, open the PR as a draft with the open findings listed.
