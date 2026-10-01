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
| `budgets.py` | `Budgets`/`DocBudget`/`RunBudget`; `RunLedger` (run budget + the store's spend ledger, owned by the `Extractor`, usable without a document); `DocumentBudget` on `ctx.budget`: every LLM call in a stage goes through `ctx.budget.call_llm(...)` (returns `None` when a budget says no). `Extractor(budgets=, store=, run_id=)` |
| `categorise.py` | Statement categorisation (stage 10): `JevStatementClassifier` (one request per statement with one Choice per schema, options limited to the fields the component gate passed; items are `ToClassify`), `CategoriseStage` (answers with full distributions on `SchemaRun.categories`). `select.field_statements` routes the top field plus any other at p ≥ `ALSO_CATEGORY_P` |
| `fallback.py` | LLM fallback (stage 14): `FallbackStage` (opt-in through `Extractor(extraction_llm=)` / `ctx.extraction_llm`; asks per statement and candidate field when there are no candidates, "none" despite category p ≥ `category_threshold`, or confidence < `fallback_threshold`; drops answers whose evidence isn't verbatim in the statement or whose value doesn't fit; one Jev request per statement verifies them with `verify_question` / `member_question`; p ≥ `verify_threshold` → `method="llm"`, `verified=True`, queued on `ctx.verified`; otherwise the Jev answer stays and the LLM value becomes an alternative), `LLMFieldExtractor` (default `LLMExtractor`, calls through `budget.call_llm`) |
| `learn.py` | Learner (stage 15): `GeneratorLearner` (opt-in through `Extractor(generator_llm=)`; `submit` stores and queues `ctx.verified` examples with p ≥ `learn_threshold`, a background worker skips values the generators in use already find, asks `generator_llm` for a `GeneratorDraft`, validates it as a `GeneratorSpec`, tests it end to end through Jev selection on the triggering statement and on stored examples it changes, then publishes it; every example ends in a `LearnOutcome`), `LearnedGenerators` (copy-on-write `GeneratorSnapshot`s loaded from and published to the store, or only in memory with `persist=False`; `refresh()` reloads at most every `refresh_after` seconds (`Extractor(refresh_generators=)`) so generators other processes publish or disable reach later documents; a document takes `ctx.generators` when it starts and `CandidateStage` runs them after its own registry), `LearnStage`. `LearnMode` (`Extractor(learn_mode=)`): `compile` documents only log examples (`ExampleLogger`), and `compile_pack` / `Extractor.compile_pack` / `jevex learn` learns from them in a batch into a `PackDiff` (`generators/<id>.yaml`, against `pack_generators(dir)`), publishing nothing |
| `housekeeping.py` | Generator housekeeping: `generator_use(ctx)` (generators the candidate stage ran, `ctx.generators_ran`; hits; wins = generators the normalise stage put in a value that stood, `SchemaRun.value_generators`), `Housekeeper` (on `ctx.housekeeper`, run by `LearnStage`: adds each document to the store's `GeneratorStats`, disables a learned generator with no wins after `prune_after` documents and withdraws it from the snapshot; `dedupe()` disables stored generators with the same field, scope and candidates on every stored example, keeping the most wins). `Extractor(prune_after=)`, `Extractor.dedupe_generators()` |
| `review.py` | Review sink: `ReviewItem` (an uncertain value: `field` `"Schema.field"`, entity, URL, `FieldMeta`, the statement's `context`; `example(value)` → a human `VerifiedExample` with the same id the fallback would give), `ReviewSink` (`async send(items)`, once per document), `ReviewQueue`, `review_items` (found values with confidence below `review_threshold`/`review_thresholds`, children too). `Extractor(review_sink=)` sends after building the result; `Extractor.feedback(item, value)` stores the example and hands it to the learner |
| `packs.py` | Packs (#41): `Pack`/`PackManifest` (a directory: `manifest.yaml`, `generators/<id>.yaml`, `key_mappings/<fingerprint>.yaml`, `examples/<field>.yaml`; `Pack.load`/`write`, `PackError`), `community_packs` (`jevex.packs` entry points), `load_pack` (directory or installed name), `layered_generators` (store → project → community, first id wins, the store's and each pack's `disables` turn off lower layers; `LearnedGenerators(packs=)`, `Extractor(packs=, community_packs=)`), `export_pack`/`import_pack` (store ↔ pack), `diff_packs` → `PackChanges`. CLI: `jevex pack export|import|diff` |
| `document.py` | `Document` (bytes + content type; base64 in JSON), content sniffing |
| `keypaths.py` | Structured-data stage (stage 4): `flatten` (key paths, collapsed shapes, entity candidates), fingerprints, `KeyPathMapper` (store lookup; one batched Choice per blob and schema on a miss; stores confident answers incl. "none", and a path after `UNSURE_LIMIT` unsure answers as an `unsure` "none"; enum/bool values Jev can't read directly are asked as the field's own question), `StructuredStage` (values on the default entity, `method="structured"`; its `mode` is `structured_only`, `fill_gaps` or `merge`: a schema needing no layout route is `SchemaRun.finish()`ed, and later routes record values with `run.offer_field`, which in merge mode settles disagreements into `meta.conflicts`) |
| `layout.py` | `Component` tree, `Location` union (`DomLocation`, `PageLocation`, `ImageLocation`), `BBox`; `section_text` (the capped heading trail every gate and statement state sends as `section`) |
| `statements.py` | `Statement`, `Span`, `Candidate`, `NormaliserStep` (compact YAML form) |
| `split.py` | Statement splitting (stage 8): `DefaultSplitter` (pysbd sentences, list items, `Label: value` pairs, headings, captions, alt text; tables are #25), `StatementStage` (cuts statements over `MAX_STATEMENT_CHARS` with `cut_statement`, repeating a cell's headers or a pair's label) |
| `images.py` | Image stage (stage 6): `ImageStage` (image components, scanned PDF pages found with pypdfium2, image documents), `DefaultImageLoader` (`data:` URIs, PDF renders, a fetcher for remote images), `OcrProcessor`/`RapidOcrEngine` (`ocr` extra), `text_components` (OCR lines → paragraphs and headed sections with `ImageLocation`s); vision processors' statements become `vision` statements |
| `entities.py` | `EntityScope` |
| `resolve.py` | Entity resolution (stage 9, after statements): `EntityStage` (gets only what passed the component gate), `SingleEntity`, `MultiEntity` (table columns, repeated siblings, headed sections, each label checked with a Noul; a Choice per unclaimed statement, "all of them" → `shared_statement_ids`). `ParentChild` (caller says where children live: table columns/rows or a component type; no questions; child scopes carry `parent` + `field`, and `EntityStage` gives each nested-model field its own `SchemaRun` named `"Parent.field"` with `parent` set, holding the parent scopes too; the extractor builds child records into `Extracted.children`, inheriting parent-scope values as `shared`). Downstream stages read a scope's statements with `ParsedDocument.scope_statements`; values from shared statements get `meta.shared` and lose to the entity's own |
| `jev.py` | The only code that talks to Jev: `Noul`/`Choice`/`Score` questions, answers, `JevClient` (batching, splitting, metering), `JevBackend` protocol, `TypeSafeBackend` |
| `schema.py` | `jevex.Field`, `Questions`, `SchemaConfig`, `SchemaSpec`/`FieldSpec` and every generated question |
| `interfaces.py` | Protocols for the 15 pluggable parts + shared types (`ParsedDocument`, `GateDecision`, `Selection`...). Its docstring maps each protocol to the issue that ships its default |
| `component_gate.py` | Component gate (stage 7): `gate_units` (a container's run of blocks, chunked; tables alone), `NoulComponentGate` (one Noul per unit × field group, all schemas in one request), `ComponentGateStage`. Results on `SchemaRun.component_ids`; nested models are gated per field in the same requests (`SchemaRun.child_component_ids`, the child run's `component_ids`); `EntityStage` keeps only passing components; `SchemaRun.relevant_fields(cid)` for the classifier. Under a `SingleEntity` entity stage (read from `ctx.pipeline`), or with `skip_found=True`, groups an earlier route fully found aren't asked about (`SchemaRun.ungated_groups`; their fields stay categorise options) |
| `pipeline.py` | `Stage` protocol, `Context`/`SchemaRun` (per-document state), `Pipeline` composition, `for_each_scope`/`for_each_schema` |
| `extractor.py` | `Extractor`, `STAGE_ORDER`/`DEFAULT_STAGES`/`default_pipeline()`, `ExtractionResult`, `DocumentMeta` |
| `results.py` | `FieldMeta`, `Source`, `Extracted` records, `partial_model`, thresholds |
| `eval.py` | `jevex eval`: corpus (`truth.json`) loading, record matching, per-field tolerances and scores, `EvalReport` |
| `store/` | Learned state: `Store` protocol and records (`base.py`), `SQLiteStore` (WAL, `BEGIN IMMEDIATE` writes, integer nano-dollar spend ledger, own worker thread), `PostgresStore` (`postgres.py`, `postgres` extra: psycopg pool, tables in their own Postgres schema, writes shielded from cancellation; tests use a bundled server from `pixeltable-pgserver`), `open_store(url)` |
| `tables.py` | `table_statements`: one `table_cell` statement per data cell, rendered `[band › ][row headers · ][column headers: ]value`, with headers structured on `Statement.table` (`TableCellRef`) for entity resolvers; `header_prefix`; `blank_rows` (a label whose data cells are empty: not a band or header row, for the gate too); `row_roles` (each row a header, band or body row, as the statements read it; the gate splits oversized tables by it); `infer_headers` (a header-less comparison table's first row and column, guarded by number-like values) |
| `testing.py` | `FakeJev` / `FakeLLM` (scripted answers), `Cassette` / `LLMCassette` record/replay |
| `llm/` | `LLM` protocol, `LLMResponse`, price table, spend cap; adapters in `llm/anthropic.py`, `llm/openai.py`, `llm/gemini.py`, `llm/litellm.py` (extras) |

How the parts fit together:
- A stage is anything with `name` and `async run(ctx)`. It reads earlier results from
  `ctx` and writes its own there.
- A pluggable part (for example a `Cleaner`) is a narrow protocol. A stage adapter wraps
  it, and the default adapter gets added to `DEFAULT_STAGES` in spec order.
- Within a stage, fan out with `for_each_scope`/`for_each_schema`, and put every question
  about one state into a single `ctx.jev.ask(...)` call. For other fan-outs use
  `jevex._tasks.gather`, not `asyncio.gather`: a failure cancels the siblings, so no
  branch is still running (and spending) after the document's result is built.
- A default stage's `name` must be one of `extractor.STAGE_ORDER` (the spec's order). Add
  it to `DEFAULT_STAGES` in any position; `default_pipeline()` sorts it into place.
- Stages record what they find with `run.set_field(scope, field, FieldMeta(...))`
  (`jevex.results`). Records, thresholds and `result.one()` are built from that; a record
  is a partial model holding only the fields (`strict()` gives the real model).

Put new modules where the spec's structure suggests. For example: `jevex/clean.py`,
`jevex/layout_html.py`, `jevex/generators/`, `jevex/normalise.py`, `jevex/store/`,
`jevex/llm/`, `jevex/contrib/scrapy.py`, `jevex/cli.py`. Heavy dependencies are optional
extras (`pdf`, `ocr`, `anthropic`, `openai`, `gemini`, `litellm`, `postgres`, `server`, `scrapy`).
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
  allow them. Never commit keys; `.env` is gitignored. `JEVEX_JEV_MAX_COST_USD` and
  `JEVEX_LLM_MAX_COST_USD` hard-cap spend per process (`JevBudgetExceededError`,
  `LLMBudgetExceededError`), or per run when the runner sets `JEVEX_SPEND_LEDGER`, a
  ledger every process adds to. Tests drop the ledger and both caps unless they're live
  or recording cassettes, so fake calls are never charged or refused. Live tests take the
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
