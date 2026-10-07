# Jev limits, measured

Measured live against `jev-1.13.0` on 2026-10-07 with `benchmarks/jev_limits/probe.py`. The
raw numbers are in `benchmarks/jev_limits/results-2026-10-07.json`. Total spend was about
$0.035, at $0.042 per million input tokens. Re-run the probe after a Jev release:

```sh
(set -a; . .env; set +a; JEVEX_JEV_MAX_COST_USD=0.30 uv run python benchmarks/jev_limits/probe.py)
```

Each pass stops at the lower of $0.30 and `JEVEX_JEV_MAX_COST_USD`. The script applies the
cap itself, because its raw SDK calls bypass `JevClient`. `--edges` (the limit search) and
`--pipeline` (the per-page pricing) re-run one section and add it to the day's file.

## Limits

| | Measured | Documented |
| --- | --- | --- |
| State (+ longest question) | 32,193 real tokens accepted; ~34k rejected | 32k |
| Whole request | 65,160 accepted; ~70k rejected | 64k |
| Questions per request | **no cap found**: 4,400 tiny Nouls in one request worked | none stated |
| Over a limit | `400 {"detail": {"error_type": "max_tokens_exceeded"}}` (`TypeSafeBadRequestError`) | |
| Rate | 300 requests at concurrency 50 in 1.7 s (~10,500/min burst), **no 429s** | 1,200 rpm; 250k tokens/s (not tested) |

The rate test was a single burst of 300 requests: under 1,200 in the minute, but faster than
1,200 per minute while it lasted. It doesn't show a sustained limit, and the token-rate
limit wasn't tested, so `JevClient` keeps its concurrency of 16.

## Tokens: jevex's estimate vs the API's count

`estimate_tokens` assumes about 4 characters per token. Real counts for one request with
one Noul:

| State | Chars | Estimate | Real | Real ÷ estimate |
| --- | --- | --- | --- | --- |
| Prose | 4,000 | 1,001 | 1,159 | 1.16 |
| Table text (`a \| b \| 123`) | 2,997 | 750 | 2,522 | **3.4** |
| JSON | 2,602 | 651 | 1,859 | **2.9** |

Numbers, separators and punctuation tokenise far more densely than prose. So `JevClient`
under-plans table and JSON states by about 3x. A table component it thinks fits can be
rejected with `max_tokens_exceeded`, and cost estimates for table-heavy pages are low.
Short questions are estimated about right (19 against ~18). One long, repetitive question
(about 1.2k characters) was over-estimated by about 2.3x, but that is a single artificial
sample.

## Billing and latency

- **The state is billed once per request.** Each extra Noul on the same state adds about
  18 tokens (1 question: 1,159 tokens; 10: 1,321; 50: 2,081).
- **Latency is flat:** median 0.2–0.4 s, whether a request has 1 or 100 questions, and for
  states from 1k to 60k characters.
- **Batching doesn't change answers.** Ten labelled Nouls gave the same answers alone and
  among 100 or 400 distractor questions (10/10 correct each time, p within ±0.03), and
  repeated requests gave identical probabilities and choices.

## A spec page, end to end

The default pipeline with `VehicleSpec` on test-site pages (seed 42, `--pipeline`), metered by
`JevClient`. **One setting differs from the default:** the document-gate wording is
`SchemaConfig(document_question="Does this document give technical specifications for one or
more vehicle variants?")`, because the default wording rules out multi-variant pages (point 5).

| Page | Requests | Questions | Tokens | Cost | Wall time |
| --- | --- | --- | --- | --- | --- |
| Spec-sheet PDF, 4 trims | 76 | 98 | 35k | $0.0015 | 9 s (mostly Docling layout; 32 s with a cold model) |
| HTML spec table, 4 trims | 74 | 96 | 34k | $0.0014 | 1.9 s |
| Key/value page, 4 trims | 79 | 123 | 36k | $0.0015 | 2.1 s |
| Prose page, 1 car | 20 | 31 | 9k | $0.0004 | 1.2 s |

So jevex costs about **$0.0015 per spec page**, or **$1.50 per thousand**, before any
LLM fallback. Most requests are one statement each (categorise, then select). That's
cheap at Jev's price, but it sets the latency and request volume.

## Implications

1. **Batch every question about one state into one request** (as stages already do).
   Extra questions are almost free in tokens and time.
2. **Fix the token estimate for tables and JSON** (#241), and treat
   `max_tokens_exceeded` as "split and retry smaller" rather than a failed document.
3. **Fan out across states with concurrency.** Latency per request is flat, so wall time
   is the number of sequential rounds, not the number of questions.
4. **Per-statement requests dominate the request count.** Statements can't share a state
   without changing what each question sees, so the levers are to ask fewer of them: the
   component gate and categorisation filters, the structured-data route, and learned
   generators.
5. **Found live, not by the scripted tests:** the document gate's question comes from the
   schema docstring. `VehicleSpec` says "for one vehicle variant", so Jev correctly answers
   *no* for a page with four variants (p = 0.15–0.18 across runs) and the page is skipped.
   The page-level wording above gives p = 0.99 (#242). Both values are in the results file.
