# Default thresholds

The defaults below were set from live eval runs on the synthetic test site (#49), on
2026-10-10, with Jev 1.13 and `claude-opus-5-5` as the fallback LLM. The script is
`benchmarks/thresholds/sweep.py` and the raw results are in
`benchmarks/thresholds/results-2026-10-10.json`.

| Setting | Was | Now | Why |
| --- | --- | --- | --- |
| `ALSO_CATEGORY_P` (`jevex.select`) | 0.3 | **0.1** | The biggest gain: about 3 points of accuracy, precision unchanged |
| `fallback_threshold` | 0.5 | **0.3** | Same accuracy as 0.5 and 0.7, with 60% fewer LLM calls |
| `verify_threshold` | 0.8 | **0.7** | Every verified LLM answer outside one ambiguity was right from p 0.5 up |
| `learn_threshold` | 0.9 | **0.95** | A learned generator repeats its example's mistake; 0.95 had no wrong answers |
| `category_threshold` | 0.5 | 0.5 | No effect anywhere from 0.3 to 0.7 |
| `prune_after` | 50 | 50 | Not measured (see below) |

## How it was measured

The tuning set was 20 pages: the first four of each HTML family (table, kv, prose, grid,
listing) of test-site seed 7. Settings were scored with `jevex.eval`, under `MultiEntity`.

**One live pass, then the grid offline.** The most permissive setting asks a superset of
every other setting's questions: category 0.3, fallback 0.9, verify 0 and
`ALSO_CATEGORY_P` 0.1. Lower thresholds send more statements to the LLM and to more
fields. So one live pass at that setting filled a cache of Jev answers (per question) and
LLM answers (per prompt). The grid then replayed offline from the cache, end to end
through the default pipeline:
- category × fallback × verify, 120 points;
- `ALSO_CATEGORY_P` from 0.1 to 0.5;
- a handful of combinations.

A Jev answer doesn't depend on the other questions in its request (#6,
`docs/jev-limits.md`), which is what makes the per-question cache valid. Offline replays
are deterministic. The permissive setting replayed scores within 1.5 points of its own
live run. The gap is Jev's run-to-run variance on questions asked twice (the cache keeps
one answer), so the grid compares every setting on one consistent set of answers.

**Then validation on unseen pages.** Seed 11, 20 different pages. The old defaults and
the chosen ones were each run live.

| Seed | Setting | Accuracy | Precision | Recall | LLM calls/doc |
| --- | --- | --- | --- | --- | --- |
| 7 (tuning) | old defaults | 0.640 | 0.850 | 0.702 | 1.25 |
| 7 (tuning) | **chosen** | **0.678** | 0.857 | 0.744 | **0.50** |
| 7 (tuning) | chosen, fallback 0.9 | 0.685 | 0.866 | 0.752 | 5.80 |
| 11 (validation) | old defaults | 0.682 | 0.925 | 0.697 | 0.65 |
| 11 (validation) | **chosen** | **0.709** | 0.926 | 0.726 | **0.40** |

On unseen pages the chosen setting keeps its gain: +2.7 points of accuracy and 38% fewer
LLM calls, with precision unchanged. A fallback threshold of 0.9 buys under a point more
for about 11 times the LLM calls, so it isn't the default. A caller who wants the last point
can set `FallbackStage(fallback_threshold=0.9)`.

The first live run of the chosen setting on seed 11 had 9 documents fail with transient
Jev errors. Their cause wasn't recorded; the script records errors now. A rerun was clean,
and its numbers are the ones above.

## What the fallback's answers look like

Over the permissive pass, the LLM's verified answers were judged against the truth. Each
answer counts once, even when it's copied into several records (shared values, extra
entities), and only in records paired with an expected one:

- **Outside `model`, all 63 were right, at every verification p from 0.5 up.** Jev's
  verification lets through what the LLM gets right. It doesn't separate right from wrong
  here, because there was almost nothing wrong to separate.
- **All 5 wrong ones are `model` with the make in it** ("Esquel Wrenna" where the truth is
  "Wrenna"), verified at p 0.6–0.91. That's a reading Jev rightly calls plausible: the
  statement does say it. It's the make/model routing question (#274), not something a
  threshold can fix.

So `verify_threshold` 0.7 accepts the right answers between 0.7 and 0.8 that 0.8 dropped.
`learn_threshold` 0.95 keeps 32 of 33 high-p answers and drops the one wrong one, which
matters more for learning than for one record: a generator learned from it would repeat
it on every later page.

## Caveats

- **The test site is synthetic and English.** The thresholds are a starting point. The
  same sweep should be rerun on the real spec-sheet corpora once they're chosen (#212).
- **`prune_after` isn't measured.** The sweep runs no learning (no `generator_llm`), and
  the test site's pages are too alike for a replay to tell a dead generator from a rare
  one. It keeps 50 until the real corpora can show how often a field's fallback fires per
  scoped document.
- **Values in another unit sit on the categorise boundary** ("173 bhp" for a kW field, p
  around 0.5; see #49's comments). No threshold here is the fix; the field description
  or a unit-aware question would be.
