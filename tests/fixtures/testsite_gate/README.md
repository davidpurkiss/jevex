# Test-site eval gate

CI's regression gate (`test_the_test_site_passes_the_eval_gate` in
`tests/test_baseline.py`) runs a small test-site corpus through the default pipeline:
the first page of each HTML family of seed 42 (`table`, `kv`, `prose`, `grid`,
`listing`), plus its first page with JSON-LD if none of those has any. Entities are
resolved with `MultiEntity`, since most of these pages hold several records. Jev and the
fallback LLM are replayed from recordings, and the run is checked against a baseline
with `jevex.baseline.check_baseline`.

| File | What it holds |
| --- | --- |
| `jev-cassette.json` | Jev's answers |
| `llm-cassette.json` | The fallback LLM's answers (Anthropic, the default model) |
| `baseline.json` | The run's accuracy, per-field accuracy and LLM calls per document, the corpus digest and the gate's tolerances |

Until they're recorded, the gate test skips. To record all three (real API calls):

    JEVEX_RECORD=1 TYPESAFE_API_KEY=... ANTHROPIC_API_KEY=... \
        JEVEX_JEV_MAX_COST_USD=0.50 JEVEX_LLM_MAX_COST_USD=2 \
        uv run pytest tests/test_baseline.py -k eval_gate

When a change moves the numbers on purpose without changing what jevex asks, rewrite only
the baseline from the recordings, offline:

    JEVEX_UPDATE_BASELINE=1 uv run pytest tests/test_baseline.py -k eval_gate

A change to the questions, the pipeline's states or the test site's pages makes the
recordings stale. The gate then xfails locally and **fails in CI**, like the books smoke
test (`jevex.testing.stale_recording`): re-record them in the same PR. If nobody with
keys can, label the PR `cassette-stale-ok` and re-run the failed jobs; CI then xfails
instead. `main` stays red on a stale recording until it's re-recorded.
