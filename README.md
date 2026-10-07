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

On a page with several entities (a comparison table under `MultiEntity`), embedded values
don't make a record of their own. A value from an object that names an entity (a JSON-LD
`offers[]` item called "Kestrova SE L" when a table column is "SE L") goes to that entity.
The rest (the make, the model) are copied into every entity with `meta.shared` set, and a
value the layout route finds for the entity itself replaces a shared one.

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
downloaded. When the OCR model first loads, jevex turns ONNX Runtime's telemetry off for
the whole process (`onnxruntime.disable_telemetry_events()`). That stops ONNX Runtime's
usage telemetry, and its telemetry thread can no longer abort a macOS process at exit.
jevex makes no requests of its own, so images on a web page are read only
when they're inline (`data:` URIs) unless you give the loader a fetcher:

```python
from jevex import DefaultImageLoader, ImageStage, SimpleFetcher
from jevex.extractor import default_pipeline

pipeline = default_pipeline().replace(
    "images", ImageStage(loader=DefaultImageLoader(fetcher=SimpleFetcher()))
)
```

A vision model can read images too, alongside OCR. It's off by default; pass any LLM
adapter that reads images (Claude, OpenAI, Gemini) as `vision_llm`:

```python
from jevex import Extractor
from jevex.llm.anthropic import AnthropicLLM

extractor = Extractor([VehicleSpec], vision_llm=AnthropicLLM("claude-sonnet-5-5"))
```

It asks the model for the facts each image shows (one call per image, counted against your
`budgets` like any LLM call) and adds them as `vision` statements. Values Jev picks from them
get `method="vision"`, and Jev then checks each one against its statement, as it does the LLM
fallback's answers: a value that passes is marked `verified`, and one that fails is moved to
the field's `alternatives`. Your own vision model plugs in as another `ImageProcessor` in
`ImageStage(processors=[...])`.

## Locales

Generators read numbers, amounts and dates the way the page's locale writes them. Each
document's locale comes from, in order: the caller (`Document.from_bytes(..., locale="de-DE")`),
the page's `<html lang>` (or a `<meta http-equiv="Content-Language">`), and the HTTP
`Content-Language` header (`Document(content_language=...)`; `SimpleFetcher` and the Scrapy
integration fill it in). When none of them says, as for most PDFs and images, the
extractor's `locale` is the document's, as if the document had said it:

```python
extractor = Extractor([VehicleSpec], locale="en-GB")
```

The other entry points take it too: `jevex eval --locale en-GB` (plain and `--replay`),
`jevex serve --locale en-GB`, and the Scrapy pipeline's `JEVEX_LOCALE` setting. Each
checks the tag the way `Extractor` does (`jevex.checked_locale`), so a bad one is a usage
error or a settings error before anything runs.

Generators scoped to it run on those documents, and what the learner learns from them is
scoped to it. It comes before the candidate and statement stages' own `locale`. Without
one, such documents have no locale of their own: generators are scoped and numbers and dates
read by the candidate stage's `locale` if it has one (else only unscoped generators run),
and what the learner learns from them stays unscoped:

```python
from jevex import CandidateStage
from jevex.extractor import default_pipeline

pipeline = default_pipeline().replace("candidates", CandidateStage(locale="de-DE"))
```

Tags are kept canonical (`jevex.canonical_locale`): `de_DE`, `de-de` and `DE-de` are all
`de-DE`, so they're one generator scope. `jevex.document_locale(document)` shows which
locale a document says it has (without the extractor's default). The statement
stage splits sentences by the same locale's language (German pages keep "z. B." and
"3. Mai" mid-sentence), falling back to `StatementStage(locale=...)`, then the candidate
stage's locale, then English; a language the sentence splitter doesn't know is split as
English.

| Locale | Reads |
| --- | --- |
| unset, `en-GB`, other decimal-point languages, and regions such as `de-CH` and `es-MX` | "1,234.5", "£18,495", "03/12/2024" as 3 December, mpg in UK gallons; Swiss regions also "1’250.50" and "CHF 1’250.–" |
| decimal-comma languages (`de`, `fr`, `es`, `it`, `nl`, `pt`, `pl`, `sv`, ...) | "1.234,5 kg", "18 495 €" grouped with a dot or a no-break space (not a plain one), "18.495,- €", "01.12.2023" day first |
| German, French, Spanish, Italian, Dutch (`de`, `fr`, `es`, `it`, `nl`) | their month names ("12. März 2024", "1er août 2024", "12 de marzo de 2024"), multipliers ("1,5 Mio. €", "2 Mds €") and range words ("1,4 bis 2,0 l", "zwischen 4 und 5"), on top of English ones |
| a US region (`en-US`, `es-US`) | "03/12/2024" as 12 March, mpg in US gallons |

Normaliser steps take the matching arguments (`{parse_number: {decimal: ","}}`,
`{parse_date: {order: mdy}}`, `{unit: {from: mpg, gallon: us}}`), and a learned or pack
generator scoped to a locale (`scope: {locale: de-DE}`) gets them added to its chain, so
it reads its locale's numbers without spelling that out. A locale-scoped generator runs
only when the document's locale matches it (`de` matches `de-AT`), never when it's unknown. `locale_conventions("de-DE")` shows what jevex
assumes for a tag. The learner scopes a generator it learns from a page with a locale to
that page's tag, with the chain written out for it, so one learned on a `de-DE` page
doesn't misread "1.234" on an `en-GB` one; pages with no locale give unscoped generators.

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
the generators they started with (`meta.generator_snapshot` says which). A nested model's
fields are learned too: a generator for `ModelPage.variants.price` runs only on the
variants, not on the page's own fields.

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
A store's learned generators are used whether or not a `generator_llm` is set. Workers
sharing a store (Scrapy, `jevex serve`) pick up the generators the others learn or disable:
each extractor checks the store at most every `refresh_generators` seconds (30 by
default; `None` turns this off), and documents that start after a check use what it found.

With `extract_sync`, the learner runs only while a blocking call drives the extractor's
private event loop. Call `extractor.wait_for_learning_sync()` after your last
`extract_sync` to finish the queued examples.

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
store = await extractor.store()
stats = await store.generator_stats("gen-0f3a9c")
print(stats.hit_rate, stats.win_rate)
```

## Review

A `review_sink` gets every value whose confidence is below `review_threshold` (0.8 by
default; `review_thresholds` per field, keyed like `thresholds`) as a `ReviewItem`: the
field, the entity, the document URL and the value's full `FieldMeta`, in one `send(items)`
call per document. Any object with an async `send` works; `ReviewQueue` keeps them in
memory. Send a person's answer back with `feedback`, and it becomes a verified example:
stored, and learned from like an LLM answer Jev verified.

```python
queue = ReviewQueue()
extractor = Extractor(schemas=[VehicleSpec], store="sqlite:///jevex.db", review_sink=queue)
await extractor.extract(document)
for item in queue.items:
    print(item.field, item.meta.value, item.meta.source.statement)
await extractor.feedback(queue.items[0], 7.4)  # the right value in that statement
```

## When something fails

`extract()` reports what went wrong on the result instead of raising, and you decide what
to do about it. `result.status` is `ok`, `partial` or `failed`, and `result.errors` lists
each failure (`PartError`: stage, kind, part, exception type, message, and how many times
it happened).

- **A part fails** (a generator, a normaliser, an image loader or processor, the
  structured extractor, the LLM fallback, the review sink): it's skipped for that input,
  the rest of the pipeline carries on, and the result is `partial`.
- **The core fails** (Jev after retries, a document that can't be read, a bug in a stage):
  the result is `failed`, with what was found before the failure.
- Only the process spend caps (`JEVEX_*_MAX_COST_USD`) and setup problems (a store or pack
  that won't open) raise.

```python
result = await extractor.extract(document)
if result.status != "ok":
    for error in result.errors:
        log.warning(error.describe())  # "candidates generator gen-3f2a: IndexError: ..."
result.raise_for_errors()  # or fail loudly: ExtractionError (partial=False: only failed)
```

Transient Jev errors (timeouts, 429, 5xx) are retried with backoff:
`Extractor(jev_retry=RetryPolicy(max_retries=4))` (default two retries). LLM adapters take
`max_retries` for their SDKs' retries. Retries are counted in `meta.jev.retries` and
`meta.llm.retries` (Anthropic and OpenAI report theirs; Gemini and LiteLLM don't).
Generators and normalisers are never retried. With a store, a learned generator that has
failed 3 times is disabled (kept for review, like a pruned one) with a
`generator_quarantined` event; other generators and the LLM fallback still cover its field.

`jevex extract` prints a partial result with a `jevex: warning:` line per error and exits
1 for a failed one; `jevex eval` scores failed documents as all missing and lists each
partial one's errors.

## Learned state

jevex keeps what it learns (key mappings, generators, verified examples, stats) and the
run's spend ledger in a store. The default is SQLite, safe for several processes on one
host. For several hosts or many workers, use Postgres (`pip install "jevex[postgres]"`):

```python
extractor = Extractor(schemas=[VehicleSpec], store="postgresql://jevex@db.internal/jevex")
```

Its tables go in a `jevex` schema. To choose another, pass a store instead:
`store=PostgresStore(url, db_schema="jevex_staging")` (from `jevex.store.postgres`).

If the store fails while a document runs, the document carries on and the failure is in
`result.errors` (kind `store`): a lookup that fails counts as nothing found (a key path is
asked about again, which costs an extra Jev call), and a write that fails is skipped.

### Spend ledger

A run budget (`RunBudget`) is kept in a spend ledger that every worker on the same run
shares. By default that's the store (SQLite and Postgres are ledgers too; with no store, the
extractor opens an in-memory SQLite one). A custom store that isn't a ledger gets a
`MemoryLedger` for the extractor. To keep spend somewhere else, such as a Redis
counter, a billing system or a per-process cap, implement `SpendLedger` and pass it in:

```python
from jevex import Budgets, Extractor, RunBudget, SpendEntry


class BillingLedger:  # four async methods; see jevex.store.ledger
    async def record_spend(self, entry: SpendEntry) -> None: ...
    async def spend(self, *, since=None, kind=None, run_id=None) -> float: ...
    async def try_spend(
        self, entry, *, cap_usd=None, max_count=None, since=None, kind=None
    ) -> bool: ...
    async def spend_entries(self, *, since=None, kind=None) -> list[SpendEntry]: ...


extractor = Extractor(
    schemas=[VehicleSpec],
    budgets=Budgets(run=RunBudget(max_spend=5.00, llm_rpm=60)),
    store="sqlite:///jevex.db",  # learned state stays here
    ledger=BillingLedger(),  # spend goes here
)
```

`try_spend` must check its limits and record the entry atomically, so workers sharing a
cap can't overshoot it together. Sums should be exact to a nano-dollar, because Jev
charges a few nano-dollars per token. When the ledger raises, jevex can't confirm the
spend, so the LLM call that needed it isn't made. The failure is reported in
`result.errors` (kind `ledger`), and Jev and generators carry on. Retrying, buffering or
failing open during an outage is up to the ledger, for example a wrapper around it.

### Packs

A pack is learned state as reviewable YAML: a directory with a `manifest.yaml` (name,
version, schemas, locales, and `disables`: generator ids from lower layers it turns off),
`generators/<id>.yaml`, `key_mappings/<fingerprint>.yaml` and optional
`examples/<Schema.field>.yaml`.

```sh
jevex pack export --store sqlite:///jevex.db --out packs/cars --name cars --version 1.0.0
jevex pack diff packs/cars sqlite:///jevex.db   # what the store learned since
jevex pack import packs/cars --store sqlite:///other.db
```

An extractor uses packs' generators under its store's: the store first, then the project
packs you pass, then the community packs installed. The first layer with an id wins, and
any layer can disable a lower layer's generator without editing it. Key mappings are
looked up through the same layers: for each key path the first layer with a mapping wins,
so the store's own answers (including "none") beat a pack's, and nothing from a pack is
copied into the store. Examples are used once a pack is imported into a store.

```python
extractor = Extractor(schemas=[VehicleSpec], store="sqlite:///jevex.db", packs=["packs/cars"])
```

A generator can be limited to some sources with `scope: {sources: [example.com]}`. A
document's source is its URL's host, lower-cased and without `www.`
(`https://www.Example.com/cars/1` is `example.com`; `shop.example.com` is a different
source). When the URL doesn't say, pass one: `Document.from_path(path, site="example.com")`.
Source-scoped generators don't run on a document with no source. Verified examples keep
their document's source (`document_source`), so the learner tests generators on each one
as its own source's documents would run them.

A community pack is a PyPI package that registers its directory under the `jevex.packs`
entry point; every installed one is used unless you pass `community_packs=False` (or the
names to use):

```toml
[project.entry-points."jevex.packs"]
automotive-uk = "jevex_pack_automotive_uk"  # the package directory holding manifest.yaml
```

## Microservice

`jevex serve` (the `server` extra: `pip install "jevex[server]"`) puts extraction behind
HTTP for callers that aren't Python. Schemas are registered by module path, as for
`jevex extract`, and requests name them by class name:

```sh
jevex serve --schema carfinder.schemas:VehicleSpec --schema carfinder.schemas:Listing \
    --store sqlite:///jevex.db --llm anthropic --max-spend 5    # http://127.0.0.1:8080/
```

```sh
curl -s localhost:8080/extract -H 'content-type: application/json' -d '{
  "document": {"content": "<the page, base64>", "content_type": "text/html",
               "url": "https://example.com/cars/1"},
  "schema": "VehicleSpec"
}'
# {"status": "ok", "errors": [], "records": [{"schema": "VehicleSpec", "entity": "document", "record": {...}}]}
```

`schema` is one name or a list (one extractor per set, so a document is only asked about
the schemas it's for). `"meta": true` adds per-field and document metadata, as
`jevex extract --meta` does. `content_type` is sniffed from the bytes when left out.
A [partial result](#when-something-fails) is a 200 whose `status` and `errors` say what was
skipped. Unknown schemas and unreadable documents answer 422, a Jev failure 502, any other
failed extraction 500, and a process spend cap (`JEVEX_*_MAX_COST_USD`) 503.

`GET /health` names the schemas; `GET /metrics` is Prometheus text (documents by outcome:
`ok`, `partial`, `stopped` or `error`; errors by stage and kind; records, values by
resolution method, Jev and LLM calls, retries and spend, budget hits, extraction time).
`--stats` (with `--store`) also serves the [stats UI](#stats) at `/stats/`. It's off by
default because it shows URLs and spend. The service has no auth of its own: run it behind
yours.

`--max-spend` and `--max-jev-spend` cap LLM and Jev spend per `--period` (default `day`)
across every request. `--locale TAG` is the [locale](#locales) of documents that don't say
their own; a request's `"locale"` in `document` overrides it for that document. In Python, `jevex.server.create_app(Service([...], store=...))`
gives the FastAPI app to mount or run yourself. The run budget is kept in the store
(the built-in ones are [spend ledgers](#spend-ledger)); `Service(ledger=...)` keeps it in another
`SpendLedger` instead, shared by every schema set the service serves, as is the in-memory
ledger it makes for a store that isn't one.

The `Dockerfile` builds an image that runs `jevex serve` on port 8080. Your schemas'
package must be importable inside it (build an image `FROM` it that installs the
package, or mount it):

```sh
docker build -t jevex .            # --build-arg EXTRAS=server,postgres,anthropic for more
docker run -p 8080:8080 -e TYPESAFE_API_KEY -v jevex-data:/data jevex \
    --schema carfinder.schemas:VehicleSpec --store sqlite:////data/jevex.db
```

## Scrapy

With the `scrapy` extra (`pip install "jevex[scrapy]"`), Scrapy does the crawling and
`jevex.contrib.scrapy.JevexPipeline`, an item pipeline, does the extraction on Scrapy's
asyncio reactor (its default). The spider yields each page as a document, made with
`document_from_response`:

```python
from jevex.contrib.scrapy import document_from_response


class BookSpider(scrapy.Spider):
    name = "books"
    custom_settings = {
        "ITEM_PIPELINES": {"jevex.contrib.scrapy.JevexPipeline": 300},
        "JEVEX_SCHEMAS": ["myproject.schemas:Book"],
        "JEVEX_STORE": "sqlite:///jevex.db",
    }

    def parse_book(self, response):
        yield {"url": response.url, "document": document_from_response(response)}
```

The pipeline swaps the item's `document` for `records` (each record's schema, entity and
values, as `jevex extract` prints them), so `scrapy crawl books -O books.jsonl` writes
them out. Items without a document pass through. `JEVEX_THRESHOLD`, `JEVEX_LOCALE` (the
locale of documents that don't say their own) and `JEVEX_META` (per-field and document
meta) work as their `Extractor` and `jevex extract` namesakes,
and counts go to Scrapy's stats under `jevex/` (with `partial`, `failed` and
`errors/<kind>`). A [failed](#when-something-fails) document fails its item with an
`ExtractionError`, which Scrapy logs and drops. For anything else (LLMs, budgets, a
custom pipeline), subclass `JevexPipeline` and override `make_extractor`; `fill_item`
decides what the item gets. When the spider closes, the pipeline lets queued learning
finish (`JEVEX_WAIT_FOR_LEARNING`), then closes the extractor. Several Scrapy workers
can share learned generators and a run budget through one store (Postgres across hosts).

`src/jevex/examples/books_spider.py` is a complete spider for
[books.toscrape.com](https://books.toscrape.com), a practice site:

```sh
scrapy runspider src/jevex/examples/books_spider.py -O books.jsonl -s CLOSESPIDER_ITEMCOUNT=5
```

## Stats

With a store, each document's numbers are recorded in it (`Extractor(record_stats=False)`
turns that off): Jev and LLM calls and cost, how each value was resolved, budget hits and
errors. `jevex stats` serves a page over them, or over a replay's CSV: the learning curve
(LLM calls and cost per document, and accuracy for a replay), the resolution mix, cumulative
spend against an optional budget, the generators with their hits and wins, the fields that
fall back to the LLM most, and budget events. The page reads the store again on every load
and reloads itself, and the same numbers are JSON at `/stats/api/<view>`
(`summary`, `learning`, `mix`, `cost`, `generators`, `fields`, `events` or `all`).

```sh
jevex stats --store sqlite:///jevex.db --budget 5     # http://127.0.0.1:8765/stats/
jevex stats --replay curve.csv
jevex stats export --replay curve.csv --svg learning --out learning.svg   # animated
jevex stats export --store sqlite:///jevex.db --svg mix --out mix.svg --static
```

`export` writes one chart (`learning`, `mix` or `cost`) as an SVG file with its own light
and dark styles. Unless `--static`, its lines draw themselves and its areas fill in (CSS
animation, which GitHub shows in READMEs). The x-axis is time for a store and documents
processed for a replay; `--x` picks the other.

## Development

```sh
uv sync                     # create .venv with dev tools
uv run pre-commit install   # ruff on every commit
uv run ruff check && uv run ruff format --check
uv run pyright
uv run pytest
```

`jevex.testsite.build(seed, out_dir)` writes a synthetic car site with exact ground truth
(`truth.json`) for `jevex eval`: HTML pages in several template families, spec-sheet PDFs,
scanned (image-only) PDFs and infographic PNGs, with each fact worded several ways. The
scans and infographics need Pillow: `uv sync --extra testsite` (or `--all-extras`).

```sh
jevex testsite build --seed 42                  # deterministic: same seed, same files
jevex testsite build --waves "table;kv,grid"    # only these families, in two waves
jevex testsite serve                            # http://127.0.0.1:8000/
```

The site is `en-GB`. HTML pages say so in `<html lang>`; PDFs and images can't, so their
`truth.json` entries carry `"locale": "en-GB"`, which `jevex eval` gives the document
(any corpus can do the same), and `jevex testsite serve` sends `Content-Language: en-GB`.
`jevex eval --locale TAG` gives every page without a `locale` of its own (in `truth.json`
or the page itself) that one instead.

`truth.json` lists the pages wave by wave, so a replay meets each wave's template
families together and the LLM-call rate spikes, then falls as generators are learned. The
default schedule is `table,listing;kv,grid;prose;pdf;scanned,infographic`. Families left
out of `--waves` aren't built.

`jevex eval` scores extraction against a build (or any directory with a `truth.json`).
`--replay` draws the learning curve: it starts from an empty in-memory store, runs the
pages one at a time in order (letting learning finish after each), and reports accuracy,
cost per document and LLM calls per document per batch, as CSV and as the stats page in
report mode (a single HTML file that works offline) with the waves marked. `--llm` turns on the LLM fallback and learning; without
it the replay runs on Jev alone. On a plain run, `--llm` turns on only the fallback. The
curve counts what the learner spends too: the CSV's `learning_*` columns hold its
generator LLM calls and cost and the Jev cost of testing drafts, per document, and the
cost charts include them.

`--gate BASELINE` turns any run into a regression check: it exits 1 if overall accuracy,
or any field's accuracy, fell, or LLM calls per document rose, by more than the
baseline's tolerances. `--write-baseline PATH` records a run as the baseline (a JSON file
with the numbers, a digest of the corpus, the run's `--locale`, and the tolerances). The
default tolerances are 0.02 for accuracy, 0.05 per field and 0.1 LLM calls per document;
`--max-accuracy-drop`, `--max-field-drop` (`none` turns it off) and `--max-llm-rise`
change them. A baseline only gates runs of the same corpus, mode (plain or `--replay`)
and `--locale` (a baseline file without a locale means no `--locale`). jevex's own CI
gates a small test-site corpus replayed from recorded Jev and LLM answers
(`tests/test_baseline.py`).

```sh
jevex eval testsite/build --schema jevex.testsite:VehicleSpec --schema jevex.testsite:Listing
jevex eval testsite/build --schema jevex.testsite:VehicleSpec --schema jevex.testsite:Listing \
    --replay --batch-size 20 --llm anthropic --csv curve.csv --html curve.html
jevex eval testsite/build --schema jevex.testsite:VehicleSpec --schema jevex.testsite:Listing \
    --llm anthropic --write-baseline baseline.json   # then, later: --gate baseline.json
```

Benchmarks follow [`docs/benchmarks.md`](docs/benchmarks.md). `benchmarks/config.yaml`
pins the seeds, the model versions and prices, and the corpora. Each corpus is frozen by a
lock of hashes in `benchmarks/corpora/`:

```sh
jevex corpus lock DIR --name NAME --out NAME.lock   # hash truth.json and every document
jevex corpus check DIR NAME.lock                    # exit 1, listing what differs
jevex corpus books --out books --lock books.lock    # fetch the books.toscrape.com sample
```

Baselines run over the same corpora into results files that `jevex eval` scores like jevex:
LLM-only extraction with a pinned model, and open-source tools (ScrapeGraphAI, Crawl4AI)
from scripts in `benchmarks/baselines/`, each in its own environment. They make real LLM
calls, capped by `JEVEX_LLM_MAX_COST_USD`:

```sh
jevex baseline inputs books --out inputs.jsonl --pipeline jevex.examples.books:books_pipeline
jevex baseline run books --schema jevex.examples.books:Book --model fast \
    --inputs inputs.jsonl --out fast.jsonl                  # fast, strong or gemini
uv run --script benchmarks/baselines/crawl4ai_baseline.py books \
    --schema jevex.examples.books:Book --model fast --inputs inputs.jsonl --out crawl4ai.jsonl
jevex eval books --schema jevex.examples.books:Book --results fast.jsonl
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
