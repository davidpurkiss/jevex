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
| `budgets.py` | `Budgets`/`DocBudget`/`RunBudget`; `RunLedger` (run budget + the `SpendLedger` it's kept in, owned by the `Extractor`, usable without a document; what the ledger raises comes out as `LedgerError`); `DocumentBudget` on `ctx.budget`: every LLM call in a stage goes through `ctx.budget.call_llm(...)` (returns `None` when a budget says no, or the ledger fails: `on_ledger_error` reports it, kind `ledger`, and the call isn't made). `Extractor(budgets=, store=, ledger=, run_id=)` |
| `categorise.py` | Statement categorisation (stage 10): `JevStatementClassifier` (one request per statement with one Choice per schema, options limited to the fields the component gate passed; items are `ToClassify`), `CategoriseStage` (answers with full distributions on `SchemaRun.categories`). `select.field_statements` routes the top field plus any other at p ≥ `ALSO_CATEGORY_P` |
| `fallback.py` | LLM fallback (stage 14): `FallbackStage` (opt-in through `Extractor(extraction_llm=)` / `ctx.extraction_llm`; asks per statement and candidate field when there are no candidates, "none" despite category p ≥ `category_threshold`, or confidence < `fallback_threshold`; drops answers whose evidence isn't verbatim in the statement or whose value doesn't fit; one Jev request per statement verifies them with `verify_question` / `member_question`; p ≥ `verify_threshold` → `method="llm"`, `verified=True`, queued on `ctx.verified`; otherwise the Jev answer stays and the LLM value becomes an alternative; before that, with or without an LLM, verifies vision values (`SchemaRun.vision_values`, recorded by select/normalise: what only vision statements gave) the same way, `source="vision"` examples, rejected ones become alternatives; a list that loses some items is described by the best pick left, `SchemaRun.value_picks`, and keeps only those picks' `value_generators`), `LLMFieldExtractor` (default `LLMExtractor`, calls through `budget.call_llm`) |
| `learn.py` | Learner (stage 15): `GeneratorLearner` (opt-in through `Extractor(generator_llm=)`; `submit` stores and queues `ctx.verified` examples with p ≥ `learn_threshold`, a background worker skips values the generators in use already find, asks `generator_llm` for a `GeneratorDraft`, validates it as a `GeneratorSpec`, tests it end to end through Jev selection (asked and read as the select stage does, a bare number's unit included: `select.select_in_units`) on the triggering statement and on stored examples it changes, then publishes it; every example ends in a `LearnOutcome`; an example's field may be a nested model's, `"Parent.nested.field"`, resolved through `all_schemas`; generators run with the example's `document_source` and locale (`VerifiedExample.locale`, `context["locale"]`, the document's own: `example_context`), so scoped ones count; an example with a locale gives a generator scoped to that tag, its chain localised by `draft_spec`), `LearnedGenerators` (copy-on-write `GeneratorSnapshot`s loaded from and published to the store, or only in memory with `persist=False`; `refresh()` reloads at most every `refresh_after` seconds (`Extractor(refresh_generators=)`) so generators other processes publish or disable reach later documents; a document takes `ctx.generators` when it starts and `CandidateStage` runs them after its own registry), `LearnStage`. `LearnMode` (`Extractor(learn_mode=)`): `compile` documents only log examples (`ExampleLogger`), and `compile_pack` / `Extractor.compile_pack` / `jevex learn` learns from them in a batch into a `PackDiff` (`generators/<id>.yaml`, against `pack_generators(dir)`), publishing nothing |
| `housekeeping.py` | Generator housekeeping: `generator_use(ctx)` (generators the candidate stage ran, `ctx.generators_ran`; hits; wins = generators the normalise stage put in a value that stood, `SchemaRun.value_generators`), `Housekeeper` (on `ctx.housekeeper`, run by `LearnStage`: adds each document to the store's `GeneratorStats`, disables a learned generator with no wins after `prune_after` documents and withdraws it from the snapshot, and quarantines (disables) one with `QUARANTINE_AFTER` failures (`GeneratorStats.failures`); `dedupe()` disables stored generators with the same field, scope and candidates on every stored example, keeping the most wins). `Extractor(prune_after=)`, `Extractor.dedupe_generators()` |
| `errors.py` | Failure reporting (#230: the library reports, the caller decides): `PartError` (stage, kind, part, type, message, count, `fatal`), `PartErrors` on `ctx.errors` (merged per stage/kind/part/type; `ctx.part_failed(...)` for a part skipped on one input), `status_of` → `ok`/`partial`/`failed`, `DocumentError` (base of the "can't read the document" errors), `ExtractionError` (`ExtractionResult.raise_for_errors()`). `Extractor.extract` records a core failure (`ctx.stage` names the stage) as a fatal error instead of raising; only the process spend caps raise. Pluggable-part calls are wrapped where they're made (generators via `GeneratorRegistry.generate(on_error=)`, `NormaliserFailedError`, image loader/processors, structured extractor, fallback extractor, review sink); Jev and store errors are never swallowed by a part wrapper. Store and ledger failures are handled where the store or ledger is called (#238): a lookup counts as nothing found, a write is skipped, an LLM call the ledger can't clear isn't made, each reported (kinds `store`, `ledger`) |
| `logs.py` | Logging: `get_logger(__name__)` (a `ContextLogger` over `logging.getLogger("jevex.<module>")`) gives every record `run_id`, `document_id`, `url`, `stage`, `part` from a context var set by `log_context(...)` (the extractor per document, `Pipeline.run` per stage, the learner's worker, started in a fresh context, per run); the `jevex` logger has a `NullHandler`; no `print` outside the CLI (a test checks). `PartErrors.add` logs each failure (first `WARNING`, repeats `DEBUG`, fatal `ERROR`) |
| `tracing.py` | OpenTelemetry (`otel` extra, imported lazily): `resolve_tracer` (`Extractor(tracer=)`: a tracer, `False`, or `None` for the global one when installed), `trace_span`, `set_attributes`, `record_failure`. A `jevex.extract` span per document, `jevex.stage <name>` per stage (`Pipeline.run`, `ctx.tracer`/`ctx.span`); `ctx.part_failed` records exception events on the stage's span |
| `monitoring.py` | Signals across documents: `DriftWindow` (per-field "none" rate, fallback rate, mean confidence over the last documents, `FieldDrift`), `budget_headroom` (run budget and process caps, `Headroom`; separately `run_headroom`, `process_headroom`), `store_error` (a health read). `GeneratorLearner.alive`/`deaths`/`outcome_counts`, `Extractor.running_learner`. `jevex serve` puts them on `/metrics` and `/health`; `docs/monitoring.md` lists every signal and suggested alerts |
| `review.py` | Review sink: `ReviewItem` (an uncertain value: `field` `"Schema.field"`, entity, URL, `FieldMeta`, the statement's `context`; `example(value)` → a human `VerifiedExample` with the same id the fallback would give), `ReviewSink` (`async send(items)`, once per document), `ReviewQueue`, `review_items` (found values with confidence below `review_threshold`/`review_thresholds`, children too). `Extractor(review_sink=)` sends after building the result; `Extractor.feedback(item, value)` stores the example and hands it to the learner |
| `packs.py` | Packs (#41): `Pack`/`PackManifest` (a directory: `manifest.yaml`, `generators/<id>.yaml`, `key_mappings/<fingerprint>.yaml`, `examples/<field>.yaml`; `Pack.load`/`write`, `PackError`), `community_packs` (`jevex.packs` entry points), `load_pack` (directory or installed name), `layered_generators` (store → project → community, first id wins, the store's and each pack's `disables` turn off lower layers; `LearnedGenerators(packs=)`, `Extractor(packs=, community_packs=)`; key mappings are layered too, in `KeyPathMapper`, from `ctx.packs`), `export_pack`/`import_pack` (store ↔ pack), `diff_packs` → `PackChanges`. CLI: `jevex pack export|import|diff` |
| `locales.py` | `LocaleConventions` (decimal mark, numeric date order, gallon, currency after the amount, language, Swiss `apostrophe_groups`, the region's `dollar`/`yen` for "$"/"¥": `currency(symbol)`), `locale_conventions(tag)` (unknown → `EN_GB`; decimal-comma languages, except point-decimal regions such as `de-CH`/`es-MX`; US regions), the language's words on top of English ones: `MONTH_NAMES` (en, de, fr, es, it, nl), `MULTIPLIERS`, `RANGE_WORDS` (normalisers read every language's months and multipliers, so a word must mean one thing; its docstring lists what isn't read), `localise_steps` (adds `decimal`/`order`/`gallon` a chain doesn't set), `document_locale` (`Document.locale`, then `<html lang>`, a `Content-Language` pragma, `Document.content_language`; `html_language` reads the page's; `Context.locale`, that or `Context.default_locale` (`Extractor(locale=)`), which `CandidateStage` uses before its own `locale`), `canonical_locale` (`de_de` → `de-DE`: document, extractor, example and scope locales are kept canonical), `checked_locale` (the same after checking it's a tag; `Extractor(locale=)`, `jevex eval`/`serve --locale` and Scrapy's `JEVEX_LOCALE` use it). Built-in generators are `LocaleAwareGenerator`s (`generate_in(statement, field, locale)`, the document's locale); a `RegexGenerator` localises its chain by its own scope locale. Splitting, tables and PDF layout collapse only ASCII whitespace, so no-break spaces (thousands groups) reach the generators |
| `document.py` | `Document` (bytes + content type; base64 in JSON), content sniffing |
| `keypaths.py` | Structured-data stage (stage 4): `flatten` (key paths, collapsed shapes, entity candidates), fingerprints, `KeyPathMapper` (lookup in the store, then the packs (`ctx.packs`, or its own `packs=`), first layer per path wins, nothing from packs stored; one batched Choice per blob and schema on a miss; stores confident answers incl. "none", and a path after `UNSURE_LIMIT` unsure answers as an `unsure` "none"; enum/bool values Jev can't read directly are asked as the field's own question; a single-value field given distinct values is settled by Jev, `KeyPathMapper._settle`: the field's `select_question` over them, the leaves of the blobs giving them as the state, one request per set of blobs, the others alternatives, "none" leaving it unfound; the rest's values, `StructuredResult.rest`, are settled in the same requests, the same options asked once), `StructuredStage` (values on the default entity, `method="structured"`; its `mode` is `structured_only`, `fill_gaps` or `merge`: a schema needing no layout route is `SchemaRun.finish()`ed, and later routes record values with `run.offer_field`, which in merge mode settles disagreements into `meta.conflicts`; array objects (`ArrayItem`) and their values (`StructuredItem`) and the rest go on `SchemaRun.structured_items`/`structured_rest`) |
| `layout.py` | `Component` tree, `Location` union (`DomLocation`, `PageLocation`, `ImageLocation`), `BBox`; `section_text` (the capped heading trail every gate and statement state sends as `section`) |
| `statements.py` | `Statement`, `Span`, `Candidate`, `NormaliserStep` (compact YAML form) |
| `split.py` | Statement splitting (stage 8): `DefaultSplitter` (pysbd sentences, rejoined after mid-sentence abbreviations: `ABBREVIATIONS` plus the language's own `LANGUAGE_ABBREVIATIONS`; list items, `Label: value` pairs, headings, captions, alt text; tables through `table_statements`), `StatementStage` (cuts statements over `MAX_STATEMENT_CHARS` with `cut_statement`, repeating a cell's headers or a pair's label; a `LocaleAwareSplitter`, as `DefaultSplitter` is, gets `split_in(component, locale)` with `ctx.locale`, else the stage's `locale`, else `CandidateStage`'s) |
| `images.py` | Image stage (stage 6): `ImageStage` (image components, scanned PDF pages found with pypdfium2, image documents), `DefaultImageLoader` (`data:` URIs, PDF renders, a fetcher for remote images), `OcrProcessor`/`RapidOcrEngine` (`ocr` extra), `text_components` (OCR lines → paragraphs and headed sections with `ImageLocation`s); vision processors' statements become `vision` statements; `VisionProcessor` (opt-in, `Extractor(vision_llm=)` / `ctx.vision_llm`: one LLM call per image with the image, through `budget.call_llm`; a `BudgetedImageProcessor` gets `process_within(image, data, budget)`); a processor's `UnreadableImageError` keeps the others' readings |
| `entities.py` | `EntityScope` |
| `resolve.py` | Entity resolution (stage 9, after statements): `EntityStage` (gets only what passed the component gate), `SingleEntity`, `MultiEntity` (table columns, repeated siblings, headed sections, each label checked with a Noul; a Choice per unclaimed statement, "all of them" → `shared_statement_ids`; a `table_header` joins its column's or row's entity, an unclaimed one on the axis across a table's entities is shared without a Choice, and a label owning only headings or headers is dropped). `ParentChild` (caller says where children live: table columns/rows or a component type; no questions; child scopes carry `parent` + `field`, and `EntityStage` gives each nested-model field its own `SchemaRun` named `"Parent.field"` with `parent` set, holding the parent scopes too; the extractor builds child records into `Extracted.children`, inheriting parent-scope values as `shared`). Downstream stages read a scope's statements with `ParsedDocument.scope_statements`; values from shared statements get `meta.shared` and lose to the entity's own. `place_document_values`: values found for the whole document (embedded data) go to the entity Jev says an embedded array object describes (one `entity_question` Choice per object giving a value, its leaves as the state), else are shared by every entity (`offer_field` lets an own value replace a shared one; when the page's value is an entity's, or Jev's "none" while entities have their own, the shared one comes from the rest or an unplaced object) |
| `jev.py` | The only code that talks to Jev: `Noul`/`Choice`/`Score` questions, answers, `JevClient` (batching, splitting, metering, retrying `JevTransientError`s by its `RetryPolicy`, counted in `usage.retries`; `Extractor(jev_retry=)`), `JevBackend` protocol, `TypeSafeBackend` (SDK retries off, transient errors mapped) |
| `schema.py` | `jevex.Field`, `Questions`, `SchemaConfig`, `SchemaSpec`/`FieldSpec` and every generated question |
| `interfaces.py` | Protocols for the 15 pluggable parts + shared types (`ParsedDocument`, `GateDecision`, `Selection`...). Its docstring maps each protocol to the issue that ships its default |
| `component_gate.py` | Component gate (stage 7): `gate_units` (a container's run of blocks, chunked; tables alone), `NoulComponentGate` (one Noul per unit × field group, all schemas in one request, plus one per header-less table with a `header_shape` in the first unit holding it, worded by the first schema's `table_headers_question`/`table_labels_question`; returns a `ComponentGateResult`), `ComponentGateStage`. Results on `SchemaRun.component_ids`; nested models are gated per field in the same requests (`SchemaRun.child_component_ids`, the child run's `component_ids`); `EntityStage` keeps only passing components; `SchemaRun.relevant_fields(cid)` for the classifier. Under a `SingleEntity` entity stage (read from `ctx.pipeline`), groups an earlier route fully found aren't asked about (`SchemaRun.ungated_groups`; their fields stay categorise options) |
| `pipeline.py` | `Stage` protocol, `Context`/`SchemaRun` (per-document state), `Pipeline` composition, `for_each_scope`/`for_each_schema` |
| `extractor.py` | `Extractor`, `STAGE_ORDER`/`DEFAULT_STAGES`/`default_pipeline()`, `ExtractionResult`, `DocumentMeta` |
| `results.py` | `FieldMeta`, `Source`, `Extracted` records, `partial_model`, thresholds |
| `eval.py` | `jevex eval`: corpus (`truth.json`) loading, record matching, per-field tolerances and scores (precision, recall, accuracy), `run_document`, `EvalReport` |
| `replay.py` | `jevex eval --replay`: `replay` (an extractor over a corpus from an empty store, one document at a time in order, `wait_for_learning` after each), `ReplayReport` (per-batch `ReplayBatch`es: accuracy, cost, LLM calls, the learner's spend after each document (`GeneratorLearner.spend`, a `LearningSpend`; in the cost charts), resolution mix, generators; `to_csv` with `CSV_COLUMNS`, `to_html`: the stats page in report mode, `jevex.stats.render_page`, waves marked) |
| `baseline.py` | `jevex eval --gate` / `--write-baseline`: `Baseline` (JSON: overall and per-field accuracy, LLM calls per document, `corpus_digest`, mode `eval`/`replay`, `GateTolerances`), `check_baseline` → `GateResult` of `Check`s (absolute tolerances; improvements never fail; another corpus or mode is a `BaselineError`, `ensure_comparable` checks before a run). CI gates a test-site corpus replayed from recordings in `tests/fixtures/testsite_gate/` |
| `benchmarks.py` | Benchmark corpora and pins (#61, `docs/benchmarks.md`): `CorpusLock` (`benchmarks/corpora/<name>.lock`: hashes of `truth.json` and every document, plus `corpus_digest`), `lock_corpus`/`check_lock` → `LockCheck`/`verify_lock`; `books_corpus` (a seeded sample of books.toscrape.com fetched with `SimpleFetcher`, labelled from markup by `book_values`); `BenchmarkConfig` (`benchmarks/config.yaml`: seeds, `PinnedModel`s with prices, corpora as `CorpusSpec`s, budget); `bootstrap_interval`. CLI `jevex corpus lock\|check\|books` |
| `baselines.py` | Benchmark baselines (#62, `docs/benchmarks.md` › *How the baselines run*): `BaselineSystem` protocol (`extract(BaselineInput) → BaselineOutput`), `prepare_input` (a pipeline's clean/layout/image stages, no Jev; `render_text`), `write_inputs`/`read_inputs` (prepared inputs JSONL every system shares), `baseline_instructions` (`benchmarks/baselines/prompt-v1.md` + `schemas_text`), `records_model` (one optional-field list per schema), `LLMBaseline` (one structured call per document) and `pinned_llm` (adapter at a `PinnedModel`'s prices), `charge_usage` (a tool's reported usage against the cap and ledger), `lenient_records`, `run_baseline` → `ResultRow` JSONL, `score_results` → `EvalReport`. CLI `jevex baseline inputs\|run`, `jevex eval --results`; tool scripts `benchmarks/baselines/<tool>_baseline.py` (uv scripts, own env) pass their system to `cli.main(system=)` |
| `bench.py` | Benchmark runs (#63, `docs/benchmarks.md` › *Protocol*): `run_benchmarks` (config → results directory: every system in `SYSTEMS` over every corpus, `prepare_corpus` building the test site or finding a kept corpus and checking its lock; jevex (cold) is a replay from an empty store, jevex (warm) a second pass over what it learned, the LLM-only baselines `run_baseline`, the tools their uv scripts; every system's rows are `ResultRow`s scored by `score_system` into a `SystemScore` (bootstrap intervals, record completeness); `RunManifest` with each `StepRecord`; a live run needs both spend caps and the ledger, capped at the config's budget (`check_budget_env`), and stops after the step that reaches a cap). Script `benchmarks/run.py` |
| `bench_report.py` | The results page: `results_page`/`write_results_page` (a results directory → `docs/benchmarks-results.md` and its charts: cost per 1,000 documents against LLM-only fast, a table per corpus, `stats.charts.accuracy_cost_svg`, the replay's learning and mix SVGs, unfinished steps, the pinned setup). Script `benchmarks/report.py` |
| `server.py` | `jevex serve` (`server` extra, FastAPI; only the CLI imports it): `Service` (registered schemas, an `Extractor` per set of schemas a request names, sharing one Jev client, store (an in-memory one if none), spend ledger (`Service(ledger=)`, else the store, else one `MemoryLedger`) and run id; `Metrics` for `/metrics`, Prometheus text, with the learners, headroom and drift from `jevex.monitoring`; `Service.health()`), `create_app(service)` (`POST /extract` with an `ExtractRequest`, `/health` (503 when the store or a learner is down), `/metrics`, opt-in `/stats` routes over the store). `Dockerfile` at the root runs it |
| `contrib/scrapy.py` | Scrapy integration (`scrapy` extra; nothing in core imports it): `document_from_response` (body, URL, media type from the header, else sniffed, else the URL's extension), `JevexPipeline` (item pipeline on Scrapy's asyncio loop, `AsyncioRequiredError` otherwise: swaps an item's `document` for `records`, settings `JEVEX_SCHEMAS`/`JEVEX_STORE`/`JEVEX_THRESHOLD`/`JEVEX_META`..., counts in Scrapy stats under `jevex/`; subclasses override `make_extractor`/`fill_item`). Example spider: `examples/books_spider.py` |
| `stats/` | Stats UI (#51): `data.py` (`Stats` tables from a store, `from_store`, reading each document's `DocumentStat` (recorded by `Extractor(record_stats=)`, `document_stat`), or a replay, `from_replay`/`from_replay_csv`; queries `curve`, `summary`, `field_stats`, `to_json` per view), `charts.py` (learning/mix/cost as SVG, embedded or standalone, static or CSS-animated), `page.py` (`render_page`: one server-rendered page, live or report mode; `ReplayReport.to_html` uses it), `server.py` (`stats_server` on `http.server`: `/stats/`, `/stats/api/<view>`, `/stats/api/chart/<view>.svg`; `store_loader`/`replay_loader`). CLI `jevex stats [export]` |
| `store/` | Learned state: `Store` protocol and records (`base.py`; `DocumentStat` per extracted document), `SpendLedger` protocol (`ledger.py`, not part of `Store`; `MemoryLedger`, `LedgerError`; the built-in stores implement both), `SQLiteStore` (WAL, `BEGIN IMMEDIATE` writes, integer nano-dollar spend ledger, own worker thread), `PostgresStore` (`postgres.py`, `postgres` extra: psycopg pool, tables in their own Postgres schema, writes shielded from cancellation; tests use a bundled server from `pixeltable-pgserver`), `open_store(url)` |
| `tables.py` | `table_statements`: one `table_cell` statement per data cell, rendered `[band › ][row headers · ][column headers: ]value`, with headers structured on `Statement.table` (`TableCellRef`) for entity resolvers, and, in a table with headers on both axes, one `table_header` statement per header label over data (the bare label; `TableCellRef.corner` and `axis` (capped by `axis_text`, `MAX_AXIS_CHARS`) reach Jev as context through `statement_state`; never falls back to the LLM); `header_prefix`; `blank_rows` (a label whose data cells are empty: not a band or header row, for the gate too); `row_roles` (each row a header, band or body row, as the statements read it; the gate splits oversized tables by it); `header_shape` (where a header-less table could have headers: a comparison table's first row and column, or a two-column table's labels; the guard on asking Jev) and `infer_headers` (marks them, for a table the component gate found headers in, `Context.headed_tables`) |
| `testsite/` | Synthetic test site: `dataset.py` (seeded makes, models, trims, listings), `phrasing.py` (the phrasing bank: sentences, labels, value formats, listing facts), `render.py` (families `table`, `kv`, `prose`, `grid`, `listing`, `pdf` spec sheets, `scanned` image-only PDFs, `infographic` PNGs; `Page.content` bytes + `content_type`), `drawing.py` (`Drawing` → vector PDF, PNG or scanned PDF; rasterising needs the `testsite` extra, Pillow, and ASCII text: `plain` drawings), `waves.py` (wave schedules: `DEFAULT_WAVES`, `parse_waves`; `build` lists pages wave by wave and builds only scheduled families), `build`/`digest`/`server` in `__init__`; CLI `jevex testsite build|serve` |
| `testing.py` | `FakeJev` / `FakeLLM` (scripted answers), `Cassette` / `LLMCassette` record/replay |
| `llm/` | `LLM` protocol (`structured(prompt, schema, *, images=)`; `LLMImage`, PNG/JPEG/WebP; callers pass `images` only when there are some), `LLMResponse`, price table, spend cap; adapters in `llm/anthropic.py`, `llm/openai.py`, `llm/gemini.py`, `llm/litellm.py` (extras) |

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
extras (`pdf`, `ocr`, `anthropic`, `openai`, `gemini`, `litellm`, `postgres`, `server`, `scrapy`,
`testsite`, `otel`).
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
- **Ask Jev for judgements, compute structure:** when code has to decide what text
  *means* (which field, which entity, what a label refers to, whether something is
  relevant), ask Jev an atomic question, batched into the stage's existing `ask`, instead
  of hard-coding a heuristic or keyword list. Pass the context Jev needs to judge, rather
  than baking a conclusion into the text. Structure that code can read for certain
  (markup, header cells, spans, number syntax) stays in code. A heuristic that guesses
  meaning needs its reason in its docstring (for example, it runs before Jev can be asked).
- **Jev:** only `jev.py` imports `typesafe_sdk`. Stages build jevex's own question models.
  Question text comes from `SchemaSpec`/`FieldSpec`, never hard-coded in a stage, so that
  `Questions(...)` overrides keep working.
- **Errors:** raise specific exceptions (subclass `JevError` or add your own). Never swallow
  exceptions. Record expected, recoverable conditions as `ctx.event(...)` instead of raising.
- **Resources:** whoever creates a client, store or connection closes it; a component never
  closes one it was given.
- **Style:** ruff with line length 100. Docstrings explain why or give the contract, not
  a restatement of the code. Match the surrounding code's comment density.
- **Tests:** one test module per source module (`tests/test_<module>.py`). Use `FakeJev`
  (or a `Cassette`) and never the network; `pytest-socket` blocks it. Async tests are
  plain `async def` (asyncio auto mode). When a stage generates questions, assert the exact
  question text, because it is user-visible behaviour.
- **No slop code:** every line must earn its place in a design someone would choose on
  purpose, not just make the change look done or keep checks green. In particular:
  - **Change the tests, not the API.** When a deliberate behaviour change breaks
    existing tests, update them to the new behaviour (after checking the new outputs by
    hand). Never add an opt-in flag, a default that keeps the old behaviour, an alias or
    a compatibility shim just so old tests keep passing. jevex is pre-v1: there's no
    backwards compatibility to preserve.
  - **No speculative code.** No parameters, options, hooks or abstractions without a real
    caller. No dead code, commented-out code or "TODO: later" stubs.
  - **Don't silence the checks.** No `# type: ignore`, `cast`, `Any`, `noqa` or
    `getattr`/`hasattr` probing to get past pyright or ruff when the types can be made
    right. No `try`/`except` that hides a failure (see **Errors**). Never weaken, skip or
    loosen a test's assertion to make it pass.
  - **Reuse before you write.** Use the helper that already exists rather than writing a
    near-copy, and fix it at its source rather than working around it at a call site.
  - **Comments say why.** Don't narrate what the code does, restate a name, or leave
    notes about the change itself ("now also...", "fixed").
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
- Jev limits (Jev 1.13, measured in `docs/jev-limits.md`): 32k tokens for state plus the
  longest question, 64k per request (over either: `400 max_tokens_exceeded`), no cap on the
  number of questions (bounded only by the 64k request), Choice up to 255 options, Score
  2–10 levels. `JevClient` estimates tokens (`estimate_tokens`, content-aware: digits and
  punctuation are a token each, so tables and JSON cost ~3x prose per character) and splits
  or raises (`StateTooLargeError`), so stages must chunk oversized components. A
  `max_tokens_exceeded` rejection is handled the same way (`JevTokenLimitError` never leaves
  the client); `JevClient.fit_state` cuts a long text to fit.
- Noul answers have only a probability (`NoulAnswer.p`) and no confidence.
- `extract_sync` reuses one private event loop. Don't create per-call loops around the
  shared `JevClient`.
- A change to what jevex asks Jev or the fallback LLM makes the recorded replay tests
  (books smoke test, test-site eval gate) stale, and a stale recording **fails in CI**
  (`jevex.testing.stale_recording`; locally it xfails, so run with `CI=1` to see it).
  Re-record in the same PR when the run may make live calls. Otherwise label the PR
  `cassette-stale-ok` (CI then xfails them; re-run the failed jobs), say so in the PR
  body, and open a `live-api` issue to re-record. Until that's done `main`'s recording is
  stale and fails every PR: if an open re-record issue explains the failure, label your
  PR `cassette-stale-ok` too and cite that issue in the PR body.
- `jevex.Field` isn't recognised as a field specifier by type checkers, so a required
  field looks optional in direct constructor calls. This is a known limitation.

## Review before a PR

Before opening a PR, run the `reviewer` subagent (`.claude/agents/reviewer.md`) with
the issue number. Fix every MUST FIX, then run it again until the verdict is `ready`,
up to 3 rounds. Paste its final DONE-WHEN CHECK and any SHOULD FIX items you chose not
to fix (with a reason) into the PR description under **Review**. If it is still
`changes-needed` after 3 rounds, open the PR as a draft with the open findings listed.
