# Benchmarks: methodology

Status: **agreed** (#61; the owner's decisions of 2026-09-30 are folded in). Baselines:
#62. Published results: #63.

## The claims to prove

1. **Accuracy:** jevex's field accuracy is comparable to LLM-only extraction on the same
   documents and schemas.
2. **Cost:** jevex costs much less per document, and the gap widens as generators are learned.
3. **Learning:** on a stable corpus, jevex's LLM-call rate decays toward zero while accuracy holds.
4. **Speed:** latency per document is competitive, and it falls as generators replace LLM calls.

Every published number must be reproducible with one command from a tagged commit.

## What's pinned

`benchmarks/config.yaml` pins everything a run depends on. `jevex.benchmarks.BenchmarkConfig`
reads and validates it, and a test checks the committed file. Change it only together
with a new results run.

| Setting | Value |
| --- | --- |
| Seed | `42`. Used for the test site, the books sample and bootstrap resampling |
| Jev | `jev-1.13.0`, $0.042 per million input tokens (output is free) |
| jevex's fallback LLM (`extraction_llm`) | `claude-haiku-4-5-20251001`, the same model as the fast baseline, so the two differ in method rather than model |
| jevex's learner (`generator_llm`) | `claude-opus-5-5`, the library default |
| LLM-only, fast | Claude Haiku 4.5, `claude-haiku-4-5-20251001` |
| LLM-only, strong | Claude Opus 5.5, `claude-opus-5-5` |
| LLM-only, Gemini | Gemini 3.8 Flash, `gemini-3.8-flash` (`models.baseline_gemini`), Gemini's current Flash model, the one most Gemini users extract with. Its pinned price is Google's offer that runs until 2026-12-31, so a run in 2027 needs new prices |
| Open-source tools | ScrapeGraphAI `2.3.0` and Crawl4AI `0.9.4`, each pinned in its script under `benchmarks/baselines/` with a `uv` lock. Both use the fast baseline's model, so they differ from it in method only |
| Concurrency | 8 documents in flight while latency is measured |
| Budget | $25 for one full run, hard-capped (Jev and LLMs together) |
| Bootstrap | 1000 resamples, 95% intervals |

Every model is pinned to an exact version. The config rejects `-latest` aliases. Each
model also records the prices the run is costed at and the date they were read
(`price_date`). A results file copies these prices, so old results keep their original
costs after list prices change. The jevex commit and the `uv.lock` used are recorded with
the results too. Together they pin every library version, including Pillow, which draws
the test site's raster pages.

## Corpora

| Corpus | Why | Ground truth | Size | Published |
| --- | --- | --- | --- | --- |
| **`testsite`**: the synthetic test site, seed 42, default waves | Exact truth. It covers every feature, and its waves show the learning curve | Generated with the site | 203 pages: 155 HTML, 32 PDFs, 16 PNGs | In full |
| **`books`**: [books.toscrape.com](https://books.toscrape.com) (`Book` schema) | A real site built for scraping practice | Read from each page's markup, then spot-checked by hand | 200 product pages, sampled with seed 42 | In full |
| **`spec-sheets`**: real UK price guides, spec PDFs and spec pages from 11 makes (`VehicleSpec` schema) | Real documents with trim columns, fetched from the manufacturers by anyone rerunning | Hand-labelled, one record per priced variant (`spec-sheets/LABELLING.md`), every value checked against the document's text by script | 25 documents (13 PDFs, 12 HTML pages), 318 records | In full |

Every corpus is a directory in `jevex eval`'s format: a `truth.json` that lists each
document and its expected records.

**Freezing.** A corpus is frozen by a lock, `benchmarks/corpora/<name>.lock`. The lock is
JSON holding a SHA-256 of `truth.json`, a SHA-256 of every document, and the corpus digest
a `jevex eval` baseline records. A run checks each corpus against its lock first and
refuses to go on if anything differs. The check names the documents that changed, went
missing or aren't in the lock.

```sh
jevex corpus lock DIR --name NAME --out benchmarks/corpora/NAME.lock [--publish aggregate]
jevex corpus check DIR benchmarks/corpora/NAME.lock     # exit 1, listing what differs
```

- **`testsite`** isn't stored. The run rebuilds it from its seed
  (`jevex testsite build --seed 42`) and checks the build against the committed
  `testsite.lock`, which CI tests as well. HTML and PDFs are byte-identical everywhere.
  The raster pages (scans and infographics) depend on the Pillow version, which
  `uv.lock` pins. The PNGs also need a Python whose `zlib` is classic zlib rather than
  zlib-ng (`zlib.ZLIB_RUNTIME_VERSION`), as in uv's 3.12 and 3.13 builds and CI.
- **`books`** is fetched once with `jevex corpus books --out DIR --lock
  benchmarks/corpora/books.lock`, which makes real requests to the site. The fetch walks
  the whole catalogue (50 pages, 1000 books) through `SimpleFetcher`, which honours
  robots.txt and waits a second between requests. `random.Random(42)` then picks 200 books
  in catalogue order, and each page is saved byte for byte. Labels come from the page's own
  markup: the `<h1>` title, the product table's "Price (incl. tax)" and "Availability"
  rows, and the `star-rating` class. A human spot-checks them before the lock is committed.
  The pages are kept with the results rather than in the repo. The committed
  `books.lock` is the fetch of 2026-10-01. Its labels were checked against the pages'
  markup by script, and a 20-label spot-check was posted on #211 for the owner to confirm. If
  the site starts serving different bytes, the lock check fails and a new fetch needs a
  new lock.
- **`spec-sheets`** is fetched, like `books`, and never committed: the documents are the
  manufacturers'. The repo holds what's needed to fetch and check them, in
  `benchmarks/corpora/spec-sheets/`: `manifest.json` gives each document's file name, URL,
  SHA-256, kind and the date it was captured, and `truth.json` holds the labels. Results
  can be published for as long as the links live. Fetch the documents with the
  `fetch.py` uv script, into the directory `$JEVEX_BENCH_SPEC_SHEETS` names (the config's
  `env`), then check them against the lock:

  ```sh
  uv run --script benchmarks/corpora/fetch.py benchmarks/corpora/spec-sheets/manifest.json \
      --out "$JEVEX_BENCH_SPEC_SHEETS"
  jevex corpus check "$JEVEX_BENCH_SPEC_SHEETS" benchmarks/corpora/spec-sheets.lock
  ```

  `fetch.py` downloads through `SimpleFetcher` (robots.txt honoured, a second between
  requests to one host), checks every document's SHA-256 and copies `truth.json` alongside.
  It reports each document. A dead link, a robots.txt refusal or a different hash fails the
  run, and nothing already in the directory is overwritten. A document already there with
  the right hash isn't fetched again, so a run can be resumed. If a manufacturer replaces
  a document, the corpus needs a new capture: a new manifest entry, labels and lock.

## Systems compared

| System | Configuration |
| --- | --- |
| **jevex (cold)** | Empty store; LLM fallback on (`extraction_llm` pinned above); learning `inline`, one document at a time in the corpus's order, so each learns from the ones before it (`jevex eval --replay`) |
| **jevex (warm)** | The same, after the cold pass (generators learned): a second pass over the corpus with the store the cold pass filled, at the pinned concurrency, with the fallback on and learning off |
| **jevex (no LLM)** | Jev plus built-in generators only; the floor for accuracy and cost |
| **LLM-only, fast** | The same schema as structured output, the whole cleaned document per call (Claude Haiku 4.5) |
| **LLM-only, strong** | The same with Claude Opus 5.5; the accuracy ceiling |
| **LLM-only, Gemini** | The same through jevex's native Gemini adapter (Gemini 3.8 Flash) |
| **ScrapeGraphAI** | `SmartScraperGraph` with its recommended settings, on Claude Haiku 4.5 |
| **Crawl4AI** | `LLMExtractionStrategy` (schema extraction) with its recommended settings, on Claude Haiku 4.5 |

LLM-only baselines get the same cleaned text that jevex's layout stage sees, and the PDF
text from the same parser. That way the comparison is about extraction rather than input
quality. Prompts are fixed in `benchmarks/baselines/` and versioned.

### How the baselines run

`jevex.baselines` holds the baselines (#62). Each one runs over a corpus into a results
file, and `jevex eval --results` scores it with the same record matching, tolerances and
metrics as jevex.

- **Same input.** `jevex baseline inputs` prepares every document once. It runs the clean,
  layout and image stages of the pipeline jevex uses on that corpus, including a site's
  cleaner such as the books corpus's star ratings. It writes the cleaned document and the
  text those stages read (`render_text`) to one JSONL file, and every system reads that
  file. A baseline's `seconds` cover only its extraction, because every system shares
  the prepared input. jevex's own latency includes cleaning and layout, so #63 compares
  latency with that in mind.
- **Same instructions.** `benchmarks/baselines/prompt-v1.md` lists the schemas field by
  field: description, unit and type, in the words jevex's fallback prompt uses. Every
  baseline gets it. A changed prompt gets a new file (`prompt-v2.md`), never an edit.
- **Same output.** Every baseline returns `records_model`: one list of records per schema,
  every field optional. A document can hold several records, as a spec sheet with several
  trims does, and any of the schemas, so a baseline has to pick the schema just as jevex
  does. A tool's raw output is checked value by value (`lenient_records`), so an "NA" it
  writes for a missing value counts as missing, not as a lost record.
- **LLM-only** (`jevex baseline run --model fast|strong|gemini`): one structured-output
  call per document through jevex's own adapters: the instructions, then the document's
  text in `<document>` tags. Anthropic's server-side refusal fallback is off, so every
  call is served by the pinned model.
- **Open-source tools** (`uv run --script benchmarks/baselines/<tool>_baseline.py`, with
  the same arguments): each script runs its tool in its own environment, because
  Crawl4AI's LiteLLM fork can't share one with `jevex[litellm]`. Each page goes to the
  tool as HTML, the cleaned document. A PDF or image goes as the text jevex read from it,
  wrapped in `<pre>`, since neither tool reads PDFs or images from bytes in its
  recommended setup.
  - ScrapeGraphAI: `SmartScraperGraph` with the page as `source` and the records model
    as `schema`. Its model is a `ChatAnthropic` with Claude's context window as
    `model_tokens`, because ScrapeGraphAI's table doesn't know Haiku 4.5 and would cut
    pages into 8k-token chunks. Telemetry is off.
  - Crawl4AI: `LLMExtractionStrategy(extraction_type="schema")` with its default
    markdown input, 2048-token chunking and output limit, through the HTTP crawler
    strategy. It reads `raw:` HTML without starting a browser. Crawl4AI reports a failed
    chunk as an error block rather than raising. A document fails only when every chunk
    failed; otherwise the records the other chunks found count.
- **Cost from real usage.** Every row records the tokens the API reported and their cost
  at the pinned prices. jevex's adapters report them for LLM-only. For the tools, a
  callback on ScrapeGraphAI's model reads each response's `usage_metadata`, and Crawl4AI
  gives LiteLLM's per-call usage (`strategy.usages`). Every charge goes through the spend
  cap and the shared ledger, as jevex's own calls do. A call that fails is in the ledger
  but not in its document's row.
- **Choosing the tools.** The owner asked for ScrapeGraphAI, Crawl4AI and any other
  widely used schema-driven extractor. Others considered:
  - Instructor-style structured output is what the LLM-only baselines already do.
  - Firecrawl's `/extract` is a hosted service rather than a library, so it isn't run.
  - Google's LangExtract is driven by few-shot examples rather than a schema.

Results files hold one JSON object per document (`ResultRow`): its path, the records
found, seconds, calls, tokens, cost, the model that served it, and an `error` if the system
failed on it. A document whose system failed scores as all missing.

## Metrics

`jevex eval` measures each of these for every document (`EvalReport`). `jevex eval --replay`
gives them over time (`ReplayReport`).

| Metric | How | Where |
| --- | --- | --- |
| **Field accuracy** | Exact match for strings and enums (after normalising case and whitespace). Numbers match within the field's tolerance: exact for counts and money, ±0.5% for measured quantities, plus the rounding a page's unit adds. Lists get precision and recall item by item. Accuracy is correct ÷ (correct + wrong + missing + spurious) | `EvalReport.field_scores()`, `overall()` |
| **Record completeness** | Strict-complete rate: expected records paired (as scoring pairs them) with a found record that has every field right, nothing wrong, missing or spurious | `jevex.bench.score_system` (`SystemScore.complete_records`) |
| **Cost per document** | Jev: input tokens × the pinned Jev price. LLMs: billed token usage × the pinned prices. Learner spend counts too | `summary()["cost_per_document"]`; replay `learning_*` columns |
| **Calls** | Jev requests and questions, and LLM calls, per document | `summary()` |
| **Latency** | Wall time per document at the pinned concurrency, p50 and p95. jevex (cold) runs one document at a time, as its learning needs | `summary()["latency_p50"]`, `["latency_p95"]` |
| **Resolution mix** | How many values came from structured data, Jev, generators, the LLM and vision | `summary()["resolution_mix"]` |
| **LLM-call rate over time** | The replay curves. The x-axis is documents processed; the y-axes are LLM calls per document, cost per document and accuracy, with the test site's waves marked | `ReplayReport.to_csv()`, `to_html()` |

Means are reported with 95% bootstrap confidence intervals over documents:
`jevex.benchmarks.bootstrap_interval`, with the pinned number of resamples and seed, so the
same results always give the same intervals. Accuracy and record completeness are ratios
(correct over scored values, complete over expected records), so documents are resampled
and each weighted by what it holds. LLM baselines take one sample per document: three, as
first planned, would triple the LLM-only spend, which already takes most of the budget.

## Protocol

1. Load `benchmarks/config.yaml`. Build or locate each corpus and check it against its
   lock. Each corpus names the schemas it's labelled in, the pipeline jevex runs on it
   (`pipeline`, `module:name`) and whether its documents hold several records
   (`entities: multi`, resolved with `MultiEntity`). A corpus that doesn't match its lock
   fails its steps, and the run goes on to the next.
2. For each system and corpus, run and save the raw per-document results to
   `benchmarks/results/<date>/<system>/<corpus>.jsonl`: one `ResultRow` per document, the
   same format for jevex as for the baselines (`jevex.baselines.result_row`; jevex's rows
   add Jev usage, the methods its values came from and what the learner spent). The
   baselines read inputs prepared once per corpus with that corpus's pipeline. Record the
   pinned models and prices, the jevex commit and the `uv.lock` hash in `manifest.json`,
   with every step's outcome and the run's spend from the ledger. For an `aggregate`
   corpus, the rows stay in the work directory (`<results>.work/`, never committed) and
   only its scores are kept.
3. Each results file is scored as `jevex eval --results` scores it (which rescores any of
   them, jevex's included) into `<system>/<corpus>.score.json` (`SystemScore`). jevex
   (cold)'s replay gives the learning curve, `jevex-cold/<corpus>.replay.csv`.
4. `uv run benchmarks/report.py benchmarks/results/<date>` builds
   `docs/benchmarks-results.md` and its charts in `docs/benchmark-results/`: the stats
   UI's animated learning curve and resolution mix (#59), and accuracy against cost per
   document on a log scale. It reads only the results directory, never the corpora.

One command: `uv run benchmarks/run.py --all` (#63; `--system` and `--corpus` choose
fewer). It stops after the step that reaches a spend cap. A step that fails for any other
reason is recorded in the manifest and on the page, and the run goes on.

## Budget, guardrails and cadence

- Hard caps via `JEVEX_JEV_MAX_COST_USD` and `JEVEX_LLM_MAX_COST_USD` (#32 and #72).
  Point `JEVEX_SPEND_LEDGER` at a file to make them cover every process in the run. The
  first full run is capped at **$25**; LLM-only strong takes most of it. `run.py` refuses
  a live run unless both caps and the ledger are set, with the caps adding up to no more
  than the config's `budget_usd`.
- The runner passes each `PinnedModel`'s prices to its adapter (`prices=`), keyed by the
  model the API reports as serving the call, so the cap counts every call at the pinned
  price. A dated ID (`<name>-YYYYMMDD`) missing from a price table is costed at its base
  name's price; a model with no price at all would be costed at nothing.
- Live runs happen only under #72's rules. Results files never contain keys.
- A dry run (`run.py --dry-run`: `FakeJev`, fake LLMs and the tools' `--fake`) checks
  every step for free. It drops the caps and the ledger, so its metered fake answers never
  count against a real budget, and its page says it's a dry run.
- **Cadence:** on demand, not on every release. Rerun when a change is expected to move
  the numbers, and before quoting them anywhere new.
