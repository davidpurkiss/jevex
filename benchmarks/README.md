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

`corpora/books.lock` is the books.toscrape.com sample (seed 42, 200 books), fetched and
spot-checked on 2026-10-01 (#211). Its pages aren't in the repo. Refetching them makes
real requests to the site, then the check confirms the site still serves the same bytes:

```sh
jevex corpus books --out /tmp/books
jevex corpus check /tmp/books benchmarks/corpora/books.lock
```

The spec-sheet locks follow once their sources are chosen.
