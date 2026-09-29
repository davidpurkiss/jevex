# jevex stats UI: design spec (draft)

Status: **draft for owner review** (#50). It will become a section of
`docs/design-spec.md` once agreed. Build issue: #51. README graphics that come from it: #59.

## Why

jevex's headline claim is that each run needs fewer LLM calls than the last, while
accuracy holds. The stats UI shows that claim happening: live on a developer's machine,
behind `jevex serve`, and in eval reports. The README's graphics are drawn from the same
data by the same code, so the marketing *is* the measurement.

## Audience and use cases

| Who | Where | What they want to know |
| --- | --- | --- |
| Developer tuning schemas | `jevex stats` locally, pointed at a store | Is extraction working? Which fields fall back to the LLM? Which generators earn their keep? |
| Operator of `jevex serve` (the car finder) | `/stats` route on the service | Spend today vs budget, LLM-call rate trend, budget events, errors |
| Anyone evaluating jevex | `jevex eval --replay` HTML report, README | Does the learning curve bend? What does it cost against LLM-only extraction? |

Out of scope: editing anything (review sinks are separate), multi-user auth (`jevex serve`
sits behind the caller's own auth), and long-term metrics storage (export to Prometheus via
`/metrics` instead).

## Data sources

The UI reads data; it never computes extraction. Two sources, one schema:

1. **Store** (`Store` interface, #35): generator stats (documents, hit and win counts),
   generator disables and the spend ledger (one row per charge, by kind and run). It's
   live, and the ledger is append-only. Per-document results meta and budget events aren't
   stored yet; the stats UI (#51) needs them added to the store or kept in a run log.
2. **Replay output** (`jevex eval --replay`, #47): per-batch CSV rows, plus the
   ground-truth accuracy the store doesn't have.

Both are normalised into one event table that the UI queries:

```
DocEvent:  ts, doc_id, url, schema, n_records, jev_requests, jev_questions, jev_tokens,
           jev_cost, llm_calls, llm_cost, seconds, method_counts{structured,jev,generator,
           llm,vision}, budget_events[], snapshot_version, accuracy? (replay only)
GeneratorStat: generator_id, field, scope, hits, wins, disabled, created, learned_from
FieldStat: schema, field, n, mean_confidence, method_counts, fallback_rate
```

Stats are served as JSON from `GET /stats/api/*` (by `jevex stats` or `jevex serve`), so the
UI, the tests and the README exporter all use the same queries.

## Views

1. **Learning curve (home).** The x-axis is documents processed (or time, as a toggle). Three
   lines on aligned y-axes: *LLM calls per doc* (hero, falling), *cost per doc*, and
   *accuracy* (replay only, flat is good). Markers show test-site waves or new template
   families, where spikes are expected, plus generator-learned events (small ticks).
2. **Resolution mix.** A stacked area over the same x-axis, showing the share of field values
   resolved by structured / jev / generator / llm / vision. The "llm" band should shrink
   as "generator" grows. This is the most intuitive single picture of learning.
3. **Cost.** Cumulative spend (Jev vs LLM) against the budget line; today vs the period cap;
   an optional LLM-only baseline cost line from #62.
4. **Generators.** A sortable table: id, field, scope, hits, wins, win rate, age, and status
   (active, disabled or pruned). Clicking a row shows its spec YAML and the example it was learned from.
5. **Fields.** One row per schema field: mean confidence, fallback rate, method mix as a
   sparkbar, and the lowest-confidence recent examples. "Which fields need attention" at a glance.
6. **Budget and errors.** A timeline of budget events (LLM stopped, caps hit) and Jev/LLM errors.

## Wireframe: home

```
┌──────────────────────────────────────────────────────────────────────────┐
│ jevex · stats     store: sqlite:///jevex.db     [docs|time]  [7d ▾]      │
├──────────────┬──────────────┬──────────────┬─────────────────────────────┤
│ LLM calls/doc│ cost/doc     │ accuracy     │ generators learned          │
│ 0.4  ▼ 92%   │ $0.0008 ▼71% │ 97.8%  ≈     │ 143  (+12 today)            │
├──────────────┴──────────────┴──────────────┴─────────────────────────────┤
│ LLM calls per doc                                                        │
│ 3 ┤█▇                  wave 2 ↓                                          │
│ 2 ┤  ▆▅▃               ▇▅▃                                               │
│ 1 ┤     ▂▂▁▁▁▁           ▂▁▁▁▁▁                                          │
│ 0 ┼──────────────────────────────────────────────── documents ──▶        │
├──────────────────────────────────────────────────────────────────────────┤
│ Resolution mix   ■ structured ■ jev ■ generator ■ llm ■ vision           │
│ ████████████████████████████████████████████████████                     │
│ (stacked area; the llm band shrinks, the generator band grows)           │
├───────────────────────────────────┬──────────────────────────────────────┤
│ Fields needing attention          │ Recent budget events                 │
│ trim        conf .61  llm 34%     │ 14:02 LLM cap hit (doc 1203)         │
│ fuel_type   conf .72  llm 12%     │ 13:40 run budget 80%                 │
└───────────────────────────────────┴──────────────────────────────────────┘
```

## Technology

- **No front-end build step.** A single static HTML page with vanilla JS and one small
  charting library (uPlot, ~50 KB), shipped inside the wheel. It's served by `jevex stats`
  (stdlib `http.server`, so no extra dependency) and mounted by `jevex serve` at `/stats`.
- **Report mode.** `jevex eval --replay --html` writes the same page with its JSON inlined,
  as a single file that works offline and can be attached to CI runs.
- **README export.** `jevex stats export --svg <view>` renders views to **animated SVG**
  (CSS keyframes inside the SVG), which GitHub renders in READMEs. The line draws itself and
  the resolution mix fills over time. Static SVG and PNG fallbacks are included. The export
  uses the same JSON queries, so README graphics are always real data (#59).
- **Styling.** Uses the brand palette (#57) with light and dark themes. The value mint is
  reserved for "resolved without an LLM".

## Build order (for #51)

1. Stats JSON queries over the store and replay CSV (pure Python, tested).
2. `jevex stats export --svg` for views 1–2, which unblocks the README hero (#59).
3. The static page with views 1–3, then `jevex stats` serving it.
4. Views 4–6, the `/stats` mount in `jevex serve`, and report mode.

## Open questions for the owner

- Is a static page plus uPlot acceptable, or do you want a richer framework later? The
  site (#65) could reuse it either way.
- Should `/stats` on `jevex serve` be on by default or opt-in? The proposal is opt-in,
  because it exposes URLs and spend.
- Should the x-axis default be documents processed (the clearest learning story) or time
  (better for operators)? The proposal is documents for replay reports and time for live stores.
