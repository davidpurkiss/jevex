# Benchmarks

The methodology is in [`docs/benchmarks.md`](../docs/benchmarks.md).

| File | What it holds |
| --- | --- |
| `config.yaml` | The pinned setup: seeds, model versions and prices, concurrency, the budget, and the corpora (`jevex.benchmarks.BenchmarkConfig`) |
| `corpora/<name>.lock` | A corpus's lock: hashes of its `truth.json` and of every document (`jevex corpus lock`, `jevex corpus check`) |
| `corpora/spec-sheets/` | The real spec-sheet corpus as the repo holds it: `manifest.json` (each document's URL and sha256), the labels (`truth.json`, labelled by `LABELLING.md`'s rules) and the labelling's checking scripts |
| `corpora/fetch.py` | Fetches a corpus from its `manifest.json`: a uv script (`fetch.py.lock`) |
| `baselines/prompt-v1.md` | The instructions every baseline gets, with `{schemas}` where the schemas are written out (`jevex.baselines.baseline_instructions`) |
| `baselines/<tool>_baseline.py` | An open-source tool as a baseline: a uv script with its pinned tool version, run in its own environment (`<tool>_baseline.py.lock`) |
| `gate_wording/` | The live comparisons of document-gate wordings behind #242's default (`compare.py`, results per date) and of the test site's `Listing` docstring behind #261's (`listing_docstring.py`, `listing-docstring-<date>.json`) |
| `thresholds/` | The threshold sweep behind #49's and #297's defaults (`sweep.py`; results per run, written up in `docs/thresholds.md`) |

`corpora/testsite.lock` is the seed-42 test site. Rebuild the site and check it:

```sh
jevex testsite build --seed 42 --out /tmp/testsite
jevex corpus check /tmp/testsite benchmarks/corpora/testsite.lock
```

`corpora/books.lock` is the books.toscrape.com sample (seed 42, 200 books), fetched on
2026-10-01. Its labels were checked against the pages' markup by script, all 200 of
them, and a 20-label spot-check was posted on #211 for the owner to confirm. Its pages aren't in the repo. Refetching them makes
real requests to the site, then the check confirms the site still serves the same bytes:

```sh
jevex corpus books --out /tmp/books
jevex corpus check /tmp/books benchmarks/corpora/books.lock
```

`corpora/spec-sheets.lock` is the real spec-sheet corpus: 25 UK manufacturers' price
guides and spec pages, labelled for #212. The documents are the manufacturers' and are
never committed. Fetch them from their URLs (robots.txt honoured, a second between requests
to one host), which checks each document's sha256 and copies the labels alongside, then
check the lot:

```sh
uv run --script benchmarks/corpora/fetch.py benchmarks/corpora/spec-sheets/manifest.json \
    --out /tmp/spec-sheets
jevex corpus check /tmp/spec-sheets benchmarks/corpora/spec-sheets.lock
```

A dead link, a robots.txt refusal or a changed document fails the fetch, naming the
document, and nothing already in the directory is overwritten.

## Baselines

Every baseline reads the same prepared inputs: each document after jevex's clean stage,
and the text jevex's layout and image stages read from it. Prepare them once per corpus,
with the pipeline jevex runs on it (the books corpus has a star-rating cleaner):

```sh
jevex baseline inputs /tmp/books --out /tmp/books-inputs.jsonl \
    --pipeline jevex.examples.books:books_pipeline --lock benchmarks/corpora/books.lock
```

Then run each system into a results file, and score it as jevex is scored:

```sh
B="/tmp/books --schema jevex.examples.books:Book --inputs /tmp/books-inputs.jsonl"
jevex baseline run $B --model fast --out results/llm-fast.jsonl     # or strong, gemini
uv run --script benchmarks/baselines/scrapegraphai_baseline.py $B --model fast --out results/scrapegraphai.jsonl
uv run --script benchmarks/baselines/crawl4ai_baseline.py $B --model fast --out results/crawl4ai.jsonl
jevex eval /tmp/books --schema jevex.examples.books:Book --results results/crawl4ai.jsonl
```

They make real LLM calls with the provider's key from the environment (`ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`). Every call is charged at the config's pinned prices against
`JEVEX_LLM_MAX_COST_USD` and `JEVEX_SPEND_LEDGER`, and a run stopped by the cap keeps the
rows it paid for. Add `--fake` to a tool script to check its plumbing with a scripted model,
for free.
