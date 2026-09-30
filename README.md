# jevex

[![CI](https://github.com/davidpurkiss/jevex/actions/workflows/ci.yml/badge.svg)](https://github.com/davidpurkiss/jevex/actions/workflows/ci.yml)

Extract typed records from web pages and PDFs using [Jev](https://docs.typesafe.ai/introduction), TypeSafe AI's System One model.

jevex narrows each document step by step (document → component → statement → value), asking Jev small atomic questions at every level. Every LLM fallback is turned into a declarative generator, so each run needs fewer LLM calls than the last.

> **Status:** in active development. The core types, Jev client, schema layer and pipeline
> skeleton are in place, and the stages are landing now. The PyPI release (0.0.1) only
> reserves the name; there is no usable extraction yet.

## PDFs

PDF layout uses [Docling](https://github.com/docling-project/docling), an optional extra:
`pip install "jevex[pdf]"`. It's heavy (about 1 GB with torch, and Docling downloads
about 500 MB of layout and table models from Hugging Face the first time it runs).
Without it, PDFs skip layout, with a `layout_skipped` event saying so.

## Images

Text in pictures (infographics, scanned PDF pages, image documents) is read with OCR
when the `ocr` extra is installed: `pip install "jevex[ocr]"`. It uses
[RapidOCR](https://github.com/RapidAI/RapidOCR), whose models ship with it, so nothing is
downloaded. jevex makes no requests of its own, so images on a web page are read only
when they're inline (`data:` URIs) unless you give the loader a fetcher:

```python
from jevex import DefaultImageLoader, ImageStage, SimpleFetcher
from jevex.extractor import default_pipeline

pipeline = default_pipeline().replace(
    "images", ImageStage(loader=DefaultImageLoader(fetcher=SimpleFetcher()))
)
```

A vision model plugs in as another `ImageProcessor` passed to `ImageStage(processors=[...])`;
the statements it returns are tagged `vision`, and so are the values taken from them.

## Development

```sh
uv sync                     # create .venv with dev tools
uv run pre-commit install   # ruff on every commit
uv run ruff check && uv run ruff format --check
uv run pyright
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow, and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Scraping responsibly

jevex takes documents, not URLs. Crawling is up to you, and so is compliance with each
site's terms of service, robots.txt and the law. Some sites forbid scraping outright.
The built-in `SimpleFetcher` honours robots.txt and rate-limits per host, but it doesn't
make any particular use lawful.

## License

Apache 2.0. jevex is an independent open-source project. It isn't affiliated with or
endorsed by TypeSafe AI; "Jev" is TypeSafe AI's model.
