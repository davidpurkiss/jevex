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
| `component_gate.DEFAULT_THRESHOLD` | 0.3 | **0.1** | About 4 points of accuracy, precision up too, for about a quarter more LLM calls (0.55 → 0.70 per document; #297, [below](#the-component-gates-threshold-297)) |

## How it was measured

The tuning set was 20 pages: the first four of each HTML family (table, kv, prose, grid,
listing) of test-site seed 7. Settings were scored with `jevex.eval`, under `MultiEntity`.

The sweep ran on main at `d48a134` (#290) with this change applied, before #292 (numbers
alone as candidates) and #294 were merged. Those change the candidates the thresholds act
on. The gate and books recordings were redone on the merged code, and they agree in
direction: against main's gate baseline, accuracy 0.739 → 0.749 and LLM calls per
document 1.17 → 0.67. One field went down there: `VehicleSpec.price_gbp` 1.0 → 0.6. That's
not these thresholds. On the gate's spec table, the component gate's price question came
back at p 0.29 in this recording and 0.30 in main's, against its own 0.3 threshold, so
price wasn't offered for those cells. #297 found why and swept that threshold too
([below](#the-component-gates-threshold-297)), on the code after #292 and #294. Since then
`sweep.py` runs this part of the grid at the gate threshold #297 chose (0.1), so a rerun
won't reproduce `results-2026-10-10.json`, which was measured at 0.3.

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

## The component gate's threshold (#297)

Measured the same way, later the same day, on main at `741d217` (#314) with this change.
Raw results are in `benchmarks/thresholds/results-2026-10-10-gate.json`.

**Why the price row scored 0.3.** On `delmaro-kestrova-table.html` the whole spec table is
one gate unit, "Price from | 19,000 GBP | £21,495 | ..." row and section heading included,
so the label wasn't missing. The question was: the test site described
`VehicleSpec.price_gbp` as "On-the-road price", and Jev reads that literally. Asked live
about the same unit with only the row's label changed:

| Row label | "...the on-the-road price (GBP)?" | "...the price (GBP)?" |
| --- | --- | --- |
| Price from | 0.25–0.33 | 0.99 |
| Price | 0.42–0.45 | 0.99 |
| OTR price, On-the-road price | 0.99 | 0.99 |
| (no price row) | 0.01 | 0.01 |

The kv page's "Price from: 19,000 GBP" gave 0.36–0.42, and the price row alone (no other
rows, no heading) 0.45–0.48, so no context the gate could add moves it. A page that says
"Price from" doesn't say the price is on the road, and Jev is right not to be sure. The
test site's truth counts every price label its phrasing bank uses ("OTR price", "Price
from", "Price", "On-the-road price"), so the schema now describes the field as "Price".
Of six descriptions tried ("on-the-road price", "price", "price new", "list price",
"price, on the road", "retail price"), "price" was the only one at 0.98 or above on every
label and prose phrasing the test site uses, and it stays at 0.01 on the table without a
price row.

**The threshold, swept.** A lower gate threshold doesn't only ask more. The categorise
Choice offers a statement only the fields its component passed for, so its options, and
with them the question, change with the threshold. The sweep runs one live permissive pass
per gate threshold (`sweep.py`'s `PERMISSIVE`), and every replayed row had no cache misses.
At #49's other thresholds (category 0.5, verify 0.7, `ALSO_CATEGORY_P` 0.1), seed 7:

| Gate threshold | 0 | 0.05 | **0.1** | 0.2 | 0.3 | 0.4 | 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 0.730 | 0.730 | **0.734** | 0.717 | 0.692 | 0.681 | 0.675 |
| Precision | 0.894 | 0.894 | **0.895** | 0.882 | 0.863 | 0.859 | 0.856 |
| Recall | 0.782 | 0.782 | **0.787** | 0.777 | 0.760 | 0.747 | 0.740 |
| LLM calls/doc | 1.05 | 1.05 | **0.70** | 0.55 | 0.55 | 0.60 | 0.60 |

The fallback threshold doesn't change the picture: at 0.5, 0.7 and 0.9 the accuracy for
each gate threshold from 0.1 to 0.5 is within 0.004 of the row above, with 0.1 best each
time (0 and 0.05 were replayed at fallback 0.3 only). Jev's
questions per document hardly move (97.4 at 0.3, 98.3 at 0.1): the test site's pages are
nearly all relevant, so the gate rarely saves a categorise request here.

The gain is in the identity fields, whose sections Jev rated between 0.1 and 0.3:
`Listing.make` 0.385 → 0.615, `VehicleSpec.trim` 0.395 → 0.447, `VehicleSpec.model`
0.087 → 0.167 (and 16 fewer spurious models), `VehicleSpec.power_kw` 0.174 → 0.217.
Below 0.1 nothing more passes that helps, and the fallback is asked more. Gate 0 is no
gate at all, and scores the same as 0.05.

**Validation on seed 11** (20 unseen pages, live):

| Gate threshold | Accuracy | Precision | Recall | LLM calls/doc |
| --- | --- | --- | --- | --- |
| 0.3 | 0.713 | 0.923 | 0.732 | 0.35 |
| **0.1** | **0.724** | **0.942** | **0.743** | 0.40 |

So `DEFAULT_THRESHOLD` is 0.1. It costs some LLM calls: 0.55 → 0.70 per document on seed
7, 0.35 → 0.40 on seed 11, as sections that now pass offer fields the fallback is then
asked about. On the CI gate's corpus (seed 42), re-recorded with both changes: accuracy
0.763 → 0.773, LLM calls per document 1.17 → 0.83, `VehicleSpec.price_gbp` stays at 1.0
with the price question at p 0.99 rather than 0.30. Two fields went down there:
`Listing.model` 0.462 → 0.385 and `VehicleSpec.model` 0.2 → 0.1. On six pages one
record moves a field by 0.08–0.1, and model is the make/model/trim routing question
(#274), which went up on seed 7.

The caveat above applies more here: on real pages, with navigation, footers and
unrelated sections, a section the gate passes wrongly costs categorise requests, and this
corpus can't show that cost. Rerun the gate sweep on the real corpora (#212).
