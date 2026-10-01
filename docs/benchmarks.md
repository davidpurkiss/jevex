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
| LLM-only, Gemini | Wanted. #62 picks the model and pins it as `models.baseline_gemini` |
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
| **`spec-sheets-local`**: real spec PDFs we may not redistribute | Realistic PDFs with trim columns | Hand-labelled | 20–40 PDFs | Aggregate numbers only |
| **`spec-sheets-press`**: press-kit spec PDFs whose terms allow committing them | Real PDFs anyone can rerun | Hand-labelled | As many as the terms allow | In full |

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
  The raster pages depend on the Pillow version, which `uv.lock` pins.
- **`books`** is fetched once with `jevex corpus books --out DIR --lock
  benchmarks/corpora/books.lock`, which makes real requests to the site. The fetch walks
  the whole catalogue (50 pages, 1000 books) through `SimpleFetcher`, which honours
  robots.txt and waits a second between requests. `random.Random(42)` then picks 200 books
  in catalogue order, and each page is saved byte for byte. Labels come from the page's own
  markup: the `<h1>` title, the product table's "Price (incl. tax)" and "Availability"
  rows, and the `star-rating` class. A human spot-checks them before the lock is committed.
  The pages are kept with the results rather than in the repo. If the site starts serving
  different bytes, the lock check fails and a new fetch needs a new lock.
- **`spec-sheets-local`** lives outside the repo, in the directory `$JEVEX_BENCH_SPEC_SHEETS`
  names. Only its lock is committed. That lock holds file names and hashes, never content.
  Its `publish: aggregate` means results show only the corpus's summary numbers, with no
  per-document rows, values or file names.
- **`spec-sheets-press`** is committed under `benchmarks/corpora/spec-sheets-press/`. Each
  source's terms are confirmed and recorded next to its files before they're added.

## Systems compared

| System | Configuration |
| --- | --- |
| **jevex (cold)** | Empty store; LLM fallback on (`extraction_llm` pinned above); learning `inline` |
| **jevex (warm)** | The same, after one full pass (generators learned); measured on a second pass or a held-out split |
| **jevex (no LLM)** | Jev plus built-in generators only; the floor for accuracy and cost |
| **LLM-only, fast** | The same schema as structured output, the whole cleaned document per call (Claude Haiku 4.5) |
| **LLM-only, strong** | The same with Claude Opus 5.5; the accuracy ceiling |
| **LLM-only, Gemini** | The same through jevex's native Gemini adapter; #62 picks the model |
| **Open-source tool** | One existing schema-driven extractor run with its recommended settings (#62 picks it) |

LLM-only baselines get the same cleaned text that jevex's layout stage sees, and the PDF
text from the same parser. That way the comparison is about extraction rather than input
quality. Prompts are fixed in `benchmarks/baselines/` and versioned.

## Metrics

`jevex eval` measures each of these for every document (`EvalReport`). `jevex eval --replay`
gives them over time (`ReplayReport`).

| Metric | How | Where |
| --- | --- | --- |
| **Field accuracy** | Exact match for strings and enums (after normalising case and whitespace). Numbers match within the field's tolerance: exact for counts and money, ±0.5% for measured quantities, plus the rounding a page's unit adds. Lists get precision and recall item by item. Accuracy is correct ÷ (correct + wrong + missing + spurious) | `EvalReport.field_scores()`, `overall()` |
| **Record completeness** | Strict-complete rate: records with every expected field correct | Computed from per-document field scores by #63's report |
| **Cost per document** | Jev: input tokens × the pinned Jev price. LLMs: billed token usage × the pinned prices. Learner spend counts too | `summary()["cost_per_document"]`; replay `learning_*` columns |
| **Calls** | Jev requests and questions, and LLM calls, per document | `summary()` |
| **Latency** | Wall time per document at the pinned concurrency, p50 and p95 | `summary()["latency_p50"]`, `["latency_p95"]` |
| **Resolution mix** | How many values came from structured data, Jev, generators, the LLM and vision | `summary()["resolution_mix"]` |
| **LLM-call rate over time** | The replay curves. The x-axis is documents processed; the y-axes are LLM calls per document, cost per document and accuracy, with the test site's waves marked | `ReplayReport.to_csv()`, `to_html()` |

Means are reported with 95% bootstrap confidence intervals over documents:
`jevex.benchmarks.bootstrap_interval`, with the pinned number of resamples and seed, so the
same results always give the same intervals. LLM baselines run 3 samples where the API
allows, to show variance.

## Protocol

1. Load `benchmarks/config.yaml`. Build or locate each corpus and check it against its
   lock. Stop on any mismatch.
2. For each system and corpus, run and save the raw per-document results to
   `benchmarks/results/<date>/<system>/<corpus>.jsonl`. Record the pinned models and
   prices, the jevex commit and the `uv.lock` hash in `manifest.json`. For an `aggregate`
   corpus, keep only its summary.
3. `jevex eval` scores every results file against the ground truth. `jevex eval --replay`
   produces the learning curves.
4. `benchmarks/report.py` builds `docs/benchmarks-results.md` and the charts, using the
   same animated SVGs the stats UI exports (#59).

One command: `uv run benchmarks/run.py --all` (#63). It stops at the budget cap.

## Budget, guardrails and cadence

- Hard caps via `JEVEX_JEV_MAX_COST_USD` and `JEVEX_LLM_MAX_COST_USD` (#32 and #72).
  Point `JEVEX_SPEND_LEDGER` at a file to make them cover every process in the run. The
  first full run is capped at **$25**; LLM-only strong takes most of it.
- Live runs happen only under #72's rules. Results files never contain keys.
- A dry run with `FakeJev` and a fake LLM checks the whole pipeline for free.
- **Cadence:** on demand, not on every release. Rerun when a change is expected to move
  the numbers, and before quoting them anywhere new.
