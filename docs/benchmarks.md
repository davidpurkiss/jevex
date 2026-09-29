# Benchmarks: methodology (draft)

Status: **draft for owner review** (#61). Baselines: #62. Published results: #63.

## The claims to prove

1. **Accuracy:** jevex's field accuracy is comparable to LLM-only extraction on the same
   documents and schemas.
2. **Cost:** jevex costs much less per document, and the gap widens as generators are learned.
3. **Learning:** on a stable corpus, jevex's LLM-call rate decays toward zero while accuracy holds.
4. **Speed:** latency per document is competitive, and it falls as generators replace LLM calls.

Every published number must be reproducible with one command from a tagged commit.

## Corpora

| Corpus | Why | Ground truth | Size (v1) |
| --- | --- | --- | --- |
| **Synthetic test site** (#43–#45), `--seed 42`, waves 1–3 | Exact truth, covers every feature, and waves show the learning curve | Generated alongside the site | ~600 pages + 60 PDFs |
| **books.toscrape.com** (`Book` schema) | A real site built for scraping practice | Hand-checked labels for a fixed sample | 200 pages |
| **Real spec sheets** (optional) | Realistic PDFs with trim columns | Hand-labelled | 20–40 PDFs |

The real spec sheets need care. Only include documents we're allowed to redistribute
(e.g. manufacturer press kits whose terms allow it), or keep them local-only and publish
just the aggregate numbers. **Owner decision.**

All corpora are frozen: documents are stored (or regenerated from a seed) and hashed in
`benchmarks/corpora/*.lock`, so reruns see identical inputs.

## Systems compared

| System | Configuration |
| --- | --- |
| **jevex (cold)** | Empty store; LLM fallback on (`extraction_llm` = a fast model); learning `inline` |
| **jevex (warm)** | The same, after one full pass (generators learned); measured on a second pass or a held-out split |
| **jevex (no LLM)** | Jev plus built-in generators only; the floor for accuracy and cost |
| **LLM-only, fast** | The same schema as structured output, the whole cleaned document per call (e.g. Claude Haiku) |
| **LLM-only, strong** | The same with a frontier model (e.g. Claude Sonnet or GPT); the accuracy ceiling |
| **Open-source tool** | One existing schema-driven extractor run with its recommended settings (#62 picks it) |

LLM-only baselines get the same cleaned text jevex's layout stage sees (and the PDF text
from the same parser), so the comparison is about extraction rather than input quality.
Prompts are fixed in `benchmarks/baselines/` and versioned.

## Metrics

- **Per field:** exact match (strings, enums), numeric match within tolerance (default ±0.5% or
  the field's unit precision), date match at the stated precision, and precision/recall for lists.
- **Per record:** strict-complete rate (every required field correct).
- **Cost:** real billed usage. Jev uses input tokens × price from the response usage; LLMs use
  token usage × the published price on the run date, written into the results file.
- **Calls:** Jev requests and questions, and LLM calls per document.
- **Latency:** wall time per document at a fixed concurrency (default 8), p50 and p95.
- **Learning:** the replay curves, with documents processed on the x-axis and LLM calls per
  document, cost per document and accuracy on the y-axis.

Report means with 95% bootstrap confidence intervals over documents. LLM baselines run
3 seeds or samples where the API allows, to show variance.

## Protocol

1. `jevex testsite build --seed 42`, then check the corpus lock hashes.
2. For each system and corpus: run, save raw per-document results to
   `benchmarks/results/<date>/<system>/<corpus>.jsonl`, and record the model versions, prices
   and jevex commit in `manifest.json`.
3. `jevex eval` scores every results file against the ground truth. `jevex eval --replay`
   produces the learning curves.
4. `benchmarks/report.py` builds `docs/benchmarks-results.md` and the charts, the same
   animated SVGs the stats UI exports (#59).

One command: `uv run benchmarks/run.py --all`. It stops at the budget cap (below).

## Budget and guardrails

- Hard caps via `JEVEX_JEV_MAX_COST_USD` and the LLM adapters' caps (#32 and #72).
  Proposed v1 budget: **$25 total** (LLM-only strong is most of it).
- Live runs happen only under #72's rules. Results files never contain keys.
- A dry run with `FakeJev` and a fake LLM validates the whole pipeline for free.

## Open questions for the owner

- Include real spec sheets (and on what licensing basis), or only synthetic data plus
  books.toscrape for v1?
- Which baseline models (a fast one and a strong one), and which open-source tool?
- Is the $25 budget for the first full run acceptable?
- Should benchmarks re-run on every release (CI, budget-capped) or on demand?
