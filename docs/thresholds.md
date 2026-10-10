# Default thresholds

The defaults below were set from live eval runs on the synthetic test site (#49), on
2026-10-10, with Jev 1.13 and `claude-opus-5-5` as the fallback LLM. The script is
`benchmarks/thresholds/sweep.py` and the raw results are in
`benchmarks/thresholds/results-2026-10-10.json`.

| Setting | Was | Now | Why |
| --- | --- | --- | --- |
| `ALSO_CATEGORY_P` (`jevex.select`) | 0.3 | **0.1** | The biggest gain: about 3 points of accuracy, precision unchanged |
| `fallback_threshold` | 0.5 | **0.3** | About two-thirds fewer LLM calls (68%) for at most 0.3 points of accuracy |
| `verify_threshold` | 0.8 | **0.7** | Every verified LLM answer except a few `model` ones was right from p 0.5 up |
| `learn_threshold` | 0.9 | **0.95** | A learned generator repeats its example's mistake; 0.95 had no wrong answers |
| `category_threshold` | 0.5 | 0.5 | No effect anywhere from 0.3 to 0.7 |
| `prune_after` | 50 | 50 | Not measured (see below) |

## How it was measured

The tuning set was 20 pages: the first four of each HTML family (table, kv, prose, grid,
listing) of test-site seed 7. Settings were scored with `jevex.eval`, under `MultiEntity`.

The sweep ran on main at `d48a134` (#290) with this change applied, before #292 (numbers
alone as candidates) and #294 were merged. Those change the candidates the thresholds act
on. The gate and books recordings were redone on the merged code, and they agree in
direction: against main's gate baseline, accuracy 0.739 → 0.749 and LLM calls per
document 1.17 → 0.67. A rerun of the sweep would measure the current code.

**One live pass, then the grid offline.** One live pass at a permissive setting (category
0.3, fallback 0.9, verify 0, `ALSO_CATEGORY_P` 0.1) filled a cache of Jev answers (per
question) and LLM answers (per prompt). For a given `ALSO_CATEGORY_P`, a lower category
threshold and a higher fallback threshold only ask more. Across `ALSO_CATEGORY_P` that
isn't guaranteed, because a field another route already found confidently isn't asked
about. So the claim was checked rather than assumed: every replayed row counts its cache
misses, and **all 246 rows had none**. They were:
- category × fallback × verify (120 points) at `ALSO_CATEGORY_P` 0.3 and again at 0.1;
- `ALSO_CATEGORY_P` from 0.1 to 0.5 at the old and the chosen other thresholds.

The script runs one live pass per `ALSO_CATEGORY_P` value, so a future sweep doesn't rely
on that.

A Jev answer doesn't depend on the other questions in its request (#6,
`docs/jev-limits.md`), which is what makes the per-question cache valid. Offline replays
are deterministic. The permissive setting scored 0.684 live and 0.687 replayed. The gap is
Jev's run-to-run variance on questions asked twice (the cache keeps one answer), so the
grid compares every setting on one consistent set of answers.

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
LLM calls, with precision unchanged.

**`fallback_threshold` against cost**, at the chosen `ALSO_CATEGORY_P` 0.1 and verify 0.7:

| Fallback | 0.3 | 0.5 | 0.7 | 0.9 |
| --- | --- | --- | --- | --- |
| Accuracy | 0.678 | 0.678 | 0.681 | 0.685 |
| LLM calls/doc | 0.50 | 1.55 | 2.60 | 5.80 |

At the old `ALSO_CATEGORY_P` 0.3, accuracy was 0.640 at 0.3, 0.5 and 0.7 alike. Going to 0.9
buys under a point for about 11 times the calls, so it isn't the default. A caller who
wants that last point can set `FallbackStage(fallback_threshold=0.9)`.

`ALSO_CATEGORY_P` isn't a setting callers can pass, unlike the three `FallbackStage`
thresholds. It's a module constant (`jevex.select.ALSO_CATEGORY_P`), and the sweep varies
it by patching the module.

The first live run of the chosen setting on seed 11 failed 9 documents. That was a bug in
the sweep, not in jevex or Jev. The run came second in a two-setting live run, and the
sweep's caches passed each extractor's close on to the live Jev client. So every question
the first setting hadn't asked went to a closed client. The caches no longer close their
clients. A clean single-setting rerun gave the numbers above.

## What the fallback's answers look like

Over the permissive pass, the LLM's verified answers that ended up in records were judged
against the truth. Each answer counts once, even when it's copied into several records
(shared values, extra entities), and only in records paired with an expected one. A
shared answer counts as right if it's right in any record it was paired into.

These are the answers that won their field. Others the fallback accepted but that lost to a
better one became alternatives, and aren't judged here. Learning sees every accepted
answer (`ctx.verified`), so this is a lower bound on what learning is offered.

- **Outside `model`, all 63 were right, at every verification p from 0.5 up.** Jev's
  verification lets through what the LLM gets right. It doesn't separate right from wrong
  here, because there was almost nothing wrong to separate.
- **All 4 wrong ones are `model` with the make or the trim in it,** verified at p
  0.62–0.91. Two have the make: "Halden Quorra" (p 0.82) and "Esquel Wrenna" (p 0.91),
  where the truth is "Quorra" and "Wrenna". Two have the trim: "Tamsin S" (0.62) and
  "Tamsin GT" (0.68), where the truth is model "Tamsin" with trim "S" or "GT". Each is a
  reading Jev rightly calls plausible, because the statement says it. It's the
  make/model/trim routing question (#274), not something a threshold can fix.

So `verify_threshold` 0.7 accepts the right answers between 0.7 and 0.8 that 0.8 dropped.
`learn_threshold` 0.95 keeps 32 of 33 high-p answers and drops the one wrong one, which
matters more for learning than for one record: a generator learned from it would repeat
it on every later page.

The learner's own acceptance test uses `fallback_threshold` too (a candidate generator
passes when Jev picks its value at that confidence), so it's 0.3 there now. The sweep ran
no learning, so that effect is unmeasured.

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
