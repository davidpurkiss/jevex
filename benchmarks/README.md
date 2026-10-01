# Benchmarks

The methodology is in [`docs/benchmarks.md`](../docs/benchmarks.md).

| File | What it holds |
| --- | --- |
| `config.yaml` | The pinned setup: seeds, model versions and prices, concurrency, the budget, and the corpora (`jevex.benchmarks.BenchmarkConfig`) |
| `corpora/<name>.lock` | A corpus's lock: hashes of its `truth.json` and of every document (`jevex corpus lock`, `jevex corpus check`) |

`corpora/testsite.lock` is the seed-42 test site. Rebuild the site and check it:

```sh
jevex testsite build --seed 42 --out /tmp/testsite
jevex corpus check /tmp/testsite benchmarks/corpora/testsite.lock
```

The other locks are committed when their corpora are first built. `books.lock` comes from
`jevex corpus books`, which makes real requests to books.toscrape.com. The spec-sheet
locks follow once their sources are chosen.
