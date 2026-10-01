# jevex

[![CI](https://github.com/davidpurkiss/jevex/actions/workflows/ci.yml/badge.svg)](https://github.com/davidpurkiss/jevex/actions/workflows/ci.yml)

Extract typed records from web pages and PDFs using [Jev](https://docs.typesafe.ai/introduction), TypeSafe AI's System One model.

jevex narrows each document step by step (document → component → statement → value), asking Jev small atomic questions at every level. Every LLM fallback is turned into a declarative generator, so each run needs fewer LLM calls than the last.

> **Status:** in active development. The core types, Jev client, schema layer and pipeline
> skeleton are in place, and the stages are landing now. The PyPI release (0.0.1) only
> reserves the name; there is no usable extraction yet.

## Embedded data

When a page embeds machine-readable data (JSON-LD, microdata, RDFa, app state such as
`__NEXT_DATA__`, `data-*` attributes), jevex maps its key paths to your fields first. The
structured stage's `mode` decides what happens next:

| Mode | Behaviour |
| --- | --- |
| `structured_only` (default) | A schema the embedded data gave any value skips the layout route; the rest go on to it. |
| `fill_gaps` | The layout route runs while some field is still empty, and only those fields get values from it. With the default `SingleEntity` resolver, the component gate skips field groups whose fields were all found; under other resolvers it asks about every group, since a field found for the page can still be empty for each entity. The categoriser still offers every field. |
| `merge` | The layout route looks for every field. The more confident value wins (a direct read of embedded data counts as certain), and the other is kept in `meta.conflicts`. |

```python
from jevex import StructuredStage
from jevex.extractor import default_pipeline

pipeline = default_pipeline().replace("structured", StructuredStage(mode="fill_gaps"))
```

## PDFs

PDF layout uses [Docling](https://github.com/docling-project/docling), an optional extra:
`pip install "jevex[pdf]"`. It's heavy (about 1 GB with torch, and Docling downloads
about 500 MB of layout and table models from Hugging Face the first time it runs).
Without it, PDFs skip layout, with a `layout_skipped` event saying so.

The document gate reads a PDF's text layer page by page, without Docling's models. A
schema with `SchemaConfig(gate_unit="page")` is asked about each page. When every active
schema is gated by page, layout converts only the pages some schema passed (plus any page
with no text layer, such as a scan), so a 40-page brochure only lays out its spec pages:

```python
class VehicleSpec(BaseModel):
    """A car's technical specification."""

    __jevex__ = SchemaConfig(gate_unit="page")
```

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

## LLM fallback

An LLM is optional. With one set as `extraction_llm`, jevex asks it about a field only
where Jev's selection failed: no candidate values were found, Jev chose "none" for a
statement it had categorised as that field, or Jev's pick had low confidence. The LLM
must return the value and the exact words of the statement that state it. Jev then checks
the answer ("The statement states that the 0-62 mph time (s) is 9.1."). A verified
answer is used with `method="llm"` and `verified=True`. A rejected one is dropped, the
field keeps Jev's best answer, and the rejected value is listed in `meta.alternatives`.

```python
from jevex import Budgets, DocBudget, Extractor, FallbackStage
from jevex.extractor import default_pipeline
from jevex.llm.anthropic import AnthropicLLM

extractor = Extractor(
    schemas=[VehicleSpec],
    extraction_llm=AnthropicLLM(model="claude-sonnet-5-5"),
    budgets=Budgets(per_document=DocBudget(max_llm_calls=5)),
    # Optional: tune when the LLM is asked and what Jev must confirm.
    pipeline=default_pipeline().replace(
        "fallback", FallbackStage(fallback_threshold=0.6, verify_threshold=0.85)
    ),
)
```

Adapters ship as extras: `jevex[anthropic]` (`jevex.llm.anthropic.AnthropicLLM`),
`jevex[openai]` (`jevex.llm.openai.OpenAILLM`), `jevex[gemini]`
(`jevex.llm.gemini.GeminiLLM`, e.g. `GeminiLLM("gemini-3.5-flash")`, with the key in
`GEMINI_API_KEY`) and `jevex[litellm]` (`jevex.llm.litellm.LiteLLM`, any provider LiteLLM
supports, including local Ollama). Any object with an async `structured(prompt, schema)`
method works too.

## Learning generators

With a `generator_llm` as well, each answer Jev verified with probability at least
`learn_threshold` (0.9 by default) is turned into a generator in the background, so the
same pattern never needs the LLM again. The LLM writes an RE2 pattern and a chain of
built-in normalisers. The generator is kept only if it finds the value in the statement it
came from, Jev picks that value there, and it doesn't make Jev wrong on the field's stored
examples. Documents that start after it is accepted use it; documents already running keep
the generators they started with (`meta.generator_snapshot` says which).

```python
extractor = Extractor(
    schemas=[VehicleSpec],
    extraction_llm=AnthropicLLM(model="claude-sonnet-5-5"),
    generator_llm=AnthropicLLM(model="claude-opus-5-5"),
    store="sqlite:///jevex.db",  # keeps learned generators and examples between runs
)
result = await extractor.extract(document)
await extractor.wait_for_learning()  # optional: let queued examples finish
learner = await extractor.learner()
print([(o.status, o.spec.id if o.spec else None) for o in learner.outcomes])
```

Learning stops when the extractor is closed; examples still queued then stay in the store.
A store's learned generators are used whether or not a `generator_llm` is set.

That is the default `learn_mode="inline"`. To review generators before any document uses
them, use `learn_mode="compile"`: documents only log their verified examples to the store,
and `jevex learn` synthesises and tests generators from them in a batch, then writes the
ones it accepts as a pack diff (`OUT/generators/<id>.yaml`) without publishing anything:

```sh
jevex learn --schema cars:VehicleSpec --store sqlite:///jevex.db --out review/ \
    --pack packs/cars --llm anthropic --max-spend 2
```

`--pack` is the pack the diff is against: its generators count as learned, so examples
they already find cost nothing. `learn_mode="hybrid"` learns inline as usual (like compile
mode, it needs a `store=`); a periodic `jevex learn` then also puts the generators learned
inline that the pack lacks into the diff. In Python, `await extractor.compile_pack(pack)`
returns the same `PackDiff`.

With a store, every document adds to each generator's stats: documents it ran on, hits
(it gave a candidate) and wins (its candidate became a value that stood). A learned
generator with no wins after `prune_after` documents (50 by default; `None` turns this off)
is disabled, not deleted. `await extractor.dedupe_generators()` disables stored generators
that give the same candidates as an older one on every stored example of their field.

```python
stats = await store.generator_stats("gen-0f3a9c")
print(stats.hit_rate, stats.win_rate)
```

## Learned state

jevex keeps what it learns (key mappings, generators, verified examples, stats) and the
run's spend ledger in a store. The default is SQLite, safe for several processes on one
host. For several hosts or many workers, use Postgres (`pip install "jevex[postgres]"`):

```python
extractor = Extractor(schemas=[VehicleSpec], store="postgresql://jevex@db.internal/jevex")
```

Its tables go in a `jevex` schema. To choose another, pass a store instead:
`store=PostgresStore(url, db_schema="jevex_staging")` (from `jevex.store.postgres`).

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
