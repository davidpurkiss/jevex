# books.toscrape.com fixtures

Product pages from [books.toscrape.com](https://books.toscrape.com), Zyte's public
sandbox for practising web scraping. Its prices and ratings are random, and its
descriptions are publisher blurbs.

| File | Source |
| --- | --- |
| `a-light-in-the-attic_1000.html` | https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html |
| `tipping-the-velvet_999.html` | https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html |
| `sapiens-a-brief-history-of-humankind_996.html` | https://books.toscrape.com/catalogue/sapiens-a-brief-history-of-humankind_996/index.html |

Saved on 2026-09-30 with `curl`, unmodified. The "Products you recently viewed" blocks
reflect the session that fetched them: the pages were fetched in the order above, so
later pages list the earlier books.

`jev-cassette.json` holds Jev's answers for these pages (see
`tests/test_smoke_books.py`). A change to what jevex asks makes it stale, which fails CI
unless the PR is labelled `cassette-stale-ok` (see `../testsite_gate/README.md`).
