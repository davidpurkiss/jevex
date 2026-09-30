# jevex — Design Spec

Sep 29, 2026 · @David Purkiss

## Overview

jevex is an open-source Python library (Apache 2.0) that extracts typed records from web pages and PDFs using Jev, TypeSafe AI's System One model. It narrows each document step by step (document → component → statement → value), asking Jev small atomic questions at every level. Every LLM fallback is turned into a declarative generator, so each run needs fewer LLM calls than the last.

**Goals**

- **Generic:** one layout model for any HTML or PDF, with no per-site configuration.
- **Cheaper and more deterministic over time:** the LLM-call rate is the headline metric, and it should decay on a stable corpus.
- **Simple for consumers:** pass a Pydantic model and a document, get a plain record back. Metadata is one step away.
- **Extensible:** every stage is an interface with a sensible default.
- **Composable:** runs inside Scrapy or any crawler, and ships a basic fetcher for standalone use.

**Non-goals**

- Crawling, JavaScript rendering, anti-bot handling and rate limiting. That's the caller's job; jevex only ships a simple fetcher for demos.
- Cross-document entity matching and merging.
- Executing LLM-generated code in core.
- Review UIs. Results carry full metadata; a review sink is an optional plugin.

**First consumer:** the car finder, which builds a vehicle catalogue from manufacturer sites and spec PDFs, and listings from used-car sites. It calls jevex as a microservice.

## Background: what Jev can and can't do

Jev answers typed questions about a *state* and never generates text. That one constraint shapes every stage. ([Jev docs](https://docs.typesafe.ai/introduction))

| Primitive | Asks | Returns | jevex uses it for |
| --- | --- | --- | --- |
| Choice | Pick one option from a list | choice, probabilities, confidence | Statement category, candidate selection, enum fields, entity assignment, structured-data key mapping |
| Score | Rate against ordered levels | score, probabilities, confidence | Ranking components by relevance, breaking candidate ties |
| Noul | Is this statement true? | probability (0–1) | Document gate, component gate, boolean fields, verifying LLM answers |

**What follows from that**

- Jev can't lay out a page, so a parser does it.
- Jev can't write a value. Code generates verbatim candidate spans, Jev picks one, and code normalises it. This is the pattern in TypeSafe's [pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook) cookbook.
- All questions in a request are evaluated in parallel and in isolation. jevex batches every question at a level into one call, as in their [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out) pattern.
- Atomic questions work best, so jevex asks one question per field and combines the answers in code.
- Choice and Score return confidence, and Noul returns a probability. Gating uses both.
- Official SDKs exist for Python and JavaScript. jevex uses the async Python client.

## Pipeline architecture

A document flows through a fixed sequence of stages, each defined by an interface with a default implementation. Each level narrows the work: only relevant components are split into statements, and only categorised statements get candidates. Cost therefore scales with the amount of relevant content, not the size of the document.

&#91;embedded content: jevex pipeline · narrowing stages, structured-data branch, learning loop\]

Most documents only take the left-hand path. The right-hand column runs when selection fails, and each accepted generator removes the need for the LLM on that pattern next time.

| # | Stage | Interface | Default | Jev question |
| --- | --- | --- | --- | --- |
| 1 | Fetch (optional) | `Fetcher` | `SimpleFetcher` (httpx, honours robots.txt) | none |
| 2 | Clean | `Cleaner` | Boilerplate stripper (nav, footer, cookie banners, scripts) | none |
| 3 | Document gate | `DocumentGate` | One Noul per registered schema, per document or per PDF page | "Does this document describe {schema description}?" |
| 4 | Structured data | `StructuredExtractor` | JSON-LD, microdata, embedded app JSON | Choice: which field does this key path hold? |
| 5 | Layout | `LayoutParser` | HTML DOM segmenter; Docling for PDF | none |
| 6 | Images | `ImageProcessor` | OCR; vision models as plugins | none |
| 7 | Component gate | `ComponentGate` | One Noul per component × field group | "Does this section contain the {field description}?" |
| 8 | Entities | `EntityResolver` | Single, multi or parent/child | Choice: which entity does this statement apply to? |
| 9 | Statements | `StatementSplitter` | Sentences, list items, key/value pairs, table cells | none |
| 10 | Categorise | `StatementClassifier` | One Choice per statement over the schema's fields + "none" | "Which detail does this statement state?" |
| 11 | Candidates | `CandidateGenerator` registry | Built-in plus learned generators | none |
| 12 | Select | `CandidateSelector` | One Choice over candidates + "none" | "Which of these is the {field description}?" |
| 13 | Normalise | Normaliser chain | Declarative, built-in normalisers | none |
| 14 | Fallback | `LLMExtractor` + verifier | Opt-in LLM, then a Jev Noul check | "The statement says {field} is {value}." |
| 15 | Learn | `Learner` (async worker) | Synthesise, test, hot-swap | none |

Stages can be replaced, removed or added through `Pipeline([...])`. The default pipeline is what `Extractor(...)` builds when no stages are given.

## Schema definition

Consumers declare what they want as a Pydantic model. jevex generates every stage's questions from it, and any question can be overridden per field or per schema.

```python
from typing import Literal
from pydantic import BaseModel
from jevex import Field, Questions, SchemaConfig

class VehicleSpec(BaseModel):
    """A manufacturer's technical specification for one vehicle variant."""

    __jevex__ = SchemaConfig(
        document_question="Does this document contain a car's technical specification?",
    )

    model: str = Field(description="Model name, e.g. Golf")
    trim: str = Field(description="Trim or grade name, e.g. SE L")
    fuel_type: Literal["petrol", "diesel", "hybrid", "phev", "ev"] = Field(
        description="Fuel or powertrain type",
    )
    engine_size_cc: int | None = Field(description="Engine displacement", unit="cc")
    zero_to_62_s: float = Field(
        description="0-62 mph acceleration time",
        unit="s",
        questions=Questions(select="Which value is the 0-62 mph time in seconds?"),
    )
```

`jevex.Field` is a thin wrapper over `pydantic.Field` that stores its extras in `json_schema_extra`, so the model stays plain Pydantic.

**How the field type drives extraction**

| Field type | Resolved by | Candidates needed |
| --- | --- | --- |
| `Literal[...]` / `Enum` | Jev Choice over the options + "not stated" | No |
| `bool` | Jev Noul | No |
| `int`, `float`, `Decimal`, with `unit` | Number-with-unit candidates, Jev Choice, then unit conversion | Yes |
| `date`, `datetime` | Date candidates, Jev Choice, then parsing | Yes |
| `str` | Span candidates (noun phrases, key/value values), Jev Choice | Yes |
| `list[T]` | As `T`, keeping every accepted candidate instead of one | Yes, per item |
| Nested `BaseModel` | Treated as a child entity (see Entity models) | Per field |

**Generated questions**

- Document gate: the schema docstring, or `document_question` if given. It runs once per document by default, or per page for long PDFs (gate\_unit="page"), so a 40-page brochure only processes its spec pages.
- Component gate: one Noul per field group (fields sharing a `group`, or one group per field by default).
- Categorise: one Choice per statement, options = field descriptions + "none of these".
- Select: "Which of these is the {description}?", options = candidates + "none".

For full control, `Pipeline([...])` accepts hand-written stages, and `Questions(...)` overrides any single question.

## Entity models

One document can yield one record, many records, or a parent with children. The caller picks a resolver, or supplies their own through the `EntityResolver` interface.

| Resolver | Use when | How entities are found | Cost |
| --- | --- | --- | --- |
| `SingleEntity` | You know the page holds one record (a listing detail page) | The whole document is one entity | Lowest: no entity questions |
| `MultiEntity` (car-finder default) | Structure is unknown: spec tables with trim columns, listing grids, a section per trim | Boundary detection, then Jev assignment | One Choice per ambiguous statement |
| `ParentChild` | You know the shape: a model page with variant children | Caller declares where children live (e.g. "table columns", a component type) | Low: statements go to parent or child |

**MultiEntity boundary detection**, in priority order:

1. Table headers: each column (or row) keyed by a variant label becomes an entity.
2. Repeated sibling structures: components with the same shape and type sequence (listing cards).
3. Headed sections: a heading per trim or engine, with its subtree.
4. Anything still ambiguous: Jev Choice "Which vehicle does this statement apply to?", with options = entity labels + "all of them".

Statements assigned to "all of them" become shared values. With `ParentChild`, they land on the parent and are inherited by every child. With `MultiEntity`, they are copied into each record, with `meta` marking them as shared.

```python
class EntityResolver(Protocol):
    async def resolve(
        self, doc: ParsedDocument, schema: type[BaseModel], jev: JevClient,
    ) -> list[EntityScope]: ...
```

An `EntityScope` is a label plus the components (or table cells) belonging to it. Downstream stages run once per scope, with all of their questions batched into shared Jev calls.

## Structured-data stage

When a page embeds machine-readable data, jevex uses it first. By default it stops there: the layout route runs only when the page has no embedded data, or when the caller opts in.

**Sources read:** JSON-LD (schema.org `Vehicle`, `Car`, `Product`, `Offer`), microdata and RDFa, embedded app state (`__NEXT_DATA__`, Nuxt payloads, `window.__INITIAL_STATE__`), and `data-*` attributes.

**How it works**

1. Flatten each blob into key-path statements, e.g. `props.vehicle.engine.displacement: 1498`. Arrays of objects become entity candidates.
2. Fingerprint the blob's shape (a hash of its sorted key-path set, with array indices collapsed).
3. Look up learned key mappings for that fingerprint. A hit is a pure lookup, with no Jev call.
4. On a miss, ask one batched Jev Choice per key path: which field (or none) does this hold? Values then go through the normal candidate and normaliser route.
5. Store accepted mappings (fingerprint + path → field + normaliser chain) as learned state, like generators.

| Mode | Behaviour |
| --- | --- |
| `structured_only` (default) | Use embedded data only. The layout route runs only when the page has none. |
| `fill_gaps` | Structured data first, then the layout route for fields it left empty. |
| `merge` | Both routes run for every field. Confidence settles disagreements, which are recorded in `meta.conflicts`. |

## Layout, statements, tables and images

Every parser produces the same generic component tree, so all later stages are format-agnostic.

```python
class Component(BaseModel):
    id: str
    type: Literal["section", "column", "heading", "paragraph", "list",
                  "list_item", "table", "image", "breakout", "caption"]
    text: str
    children: list["Component"]
    heading_trail: list[str]      # e.g. ["Specifications", "Performance"]
    location: Location            # dom_path for HTML; page + bbox for PDF
```

**Defaults**

- HTML: a DOM segmenter that uses block-level elements, the heading hierarchy and semantic tags (`aside`, `figure`, `table`).
- PDF: [Docling](https://github.com/docling-project/docling) for layout, reading order, table structure and OCR. It's an optional extra.
- A `heading_trail` travels with every component. "7.9 s" under "Performance" means something that "7.9 s" alone doesn't.

**Statement splitting by component type**

| Component | Statements |
| --- | --- |
| Paragraph | One per sentence (pysbd) |
| List item | One each |
| Key/value line, definition list | One per pair |
| Table | One per cell, rendered with its headers: `Performance › 0-62 mph (s) · 1.5 TSI SE: 9.1` |
| Image | Alt text and caption as statements; OCR text parsed into components recursively |

Table headers do double duty: they give each cell its meaning, and they mark entity boundaries for `MultiEntity`.

**Image stage**

- `ImageProcessor` is an interface. The default runs OCR on images and on scanned PDF pages (pages with no text layer).
- Vision-model plugins can return statements directly. These are tagged `method="vision"`, then verified and learned from like LLM output.
- Image provenance keeps the image URL or page, plus the bbox of the OCR line.

## Value extraction

Values come from verbatim candidate spans that Jev chooses between. An LLM is used only when that fails, and only if the caller enables it. Generators are built for recall; Jev provides the precision.

**Order of resolution, per statement and field**

1. Enum or bool field: Jev answers directly (Choice or Noul). Done.
2. Run the generators in scope for the field (by type, field, schema, locale and source) to get candidate spans with character offsets.
3. Jev Choice: "Which of these is the {description}?", with options = candidates + "none".
4. Normalise the chosen span through its generator's normaliser chain, then validate it against the field type.
5. Fall back to the LLM if any of these hold:
   - no candidates were generated
   - Jev chose "none", but the statement was categorised as this field with probability ≥ `category_threshold`
   - selection confidence is below `fallback_threshold`
6. LLM fallback: `extraction_llm` receives the statement, its heading trail and the field schema. It must return a value **and** the verbatim evidence span.
7. Verify: Jev Noul "The statement states that {description} is {value}."
   - If p ≥ `verify_threshold`, accept with `method="llm"` and queue it for learning.
   - Otherwise, discard it and return the best Jev answer, marked low-confidence.

**Built-in generators**

| Generator | Finds | Example span → value |
| --- | --- | --- |
| `number_with_unit` | Numbers with units from a unit lexicon (s, mph, km/h, bhp, PS, kW, Nm, lb ft, mpg, l/100km, g/km, kg, mm, cc, litres) | `150PS` → 150 PS → 110 kW |
| `money` | Amounts with currency symbols or codes | `£18,495` → 18495 GBP |
| `date`, `year` | Absolute dates, month-year, model years | `March 2024` → 2024-03 |
| `range` | Numeric ranges | `5–7 seats` → \[5, 7\] |
| `key_value` | The value side of `label: value` statements | `Colour: Moonstone Grey` |
| `noun_phrase` | Noun-phrase chunks, for strings | `Moonstone Grey metallic` |
| `regex` | Any declarative regex spec (what the learner produces) | as defined |

```python
class CandidateGenerator(Protocol):
    id: str
    scope: Scope
    def generate(self, statement: Statement) -> list[Candidate]: ...

class Candidate(BaseModel):
    span: Span            # start, end offsets into statement.text
    raw: str
    normalise: list[NormaliserStep]
    generator_id: str
```

Custom generators are ordinary plugins and can do anything internally, including calling an LLM. Only the core learner is restricted to declarative output.

## Learning loop

Every verified LLM extraction becomes a declarative generator, so the same pattern never needs the LLM again. Learning runs asynchronously, and new generators are used as soon as they pass their tests.

**From fallback to generator**

1. **Verify first.** Only answers that pass Jev verification with p ≥ `learn_threshold` become verified examples. This stops hallucinations being baked in.
2. **Queue.** The example (statement, field, value, evidence span, context) goes to the learning worker, and the document carries on.
3. **Synthesise.** `generator_llm` writes a generator spec as structured output matching the spec's JSON schema. It is asked for recall: a pattern that finds the value, not one that is the only match.
4. **Validate.** The spec must parse; its normalisers must come from the built-in set; its regex must compile under a linear-time engine (google-re2), with a length cap.
5. **Test.**
   - It must produce the verified value on the triggering statement, end to end through Jev selection.
   - It must not lower accuracy on the stored examples for that field: a sample, or all of them if there are fewer than N.
6. **Hot-swap.** Accepted generators are published to the registry as a new copy-on-write snapshot. Documents in flight keep their snapshot; new documents get the new one.

**A learned generator**

```yaml
id: gen-0f3a9c
field: VehicleSpec.zero_to_62_s
scope: {locale: en-GB}
match:
  regex: '0\s*[-–]\s*62(?:\s*mph)?\D{0,20}?(\d+(?:\.\d+)?)\s*(?:s|secs?|seconds)\b'
  group: 1
normalise:
  - parse_number
  - unit: {from: s, to: s}
provenance:
  learned_from: [ex-91c2]
  synthesised_by: generator_llm
  created: 2026-09-29
```

Structured-data key mappings (fingerprint + path → field) are learned the same way.

| Mode | Behaviour |
| --- | --- |
| `inline` (default) | Learn asynchronously during the run; use generators as soon as they are accepted. |
| `compile` | Only log verified examples. `jevex learn` synthesises and tests in a batch, then writes a pack diff for review. |
| `hybrid` | Inline into the local layer, with periodic compiles into a reviewable pack. |

**Housekeeping**

- Hit rate (produced a candidate) and win rate (candidate chosen and correct) are tracked per generator.
- Generators with no wins after `prune_after` scoped documents are disabled, not deleted.
- Specs whose candidate sets match on every stored example are deduplicated.
- Human answers from an optional review sink enter as verified examples, and are learned from the same way.

## Learned state

A `Store` interface holds everything learned. A database handles runtime writes and stats, and file packs handle review, versioning and distribution.

| Held in the store | Purpose |
| --- | --- |
| Generator specs | Candidate generation |
| Key mappings | Structured-data lookup by fingerprint |
| Verified examples | Regression tests and eval corpus |
| Generator stats | Hit and win rates, pruning |
| Spend ledger | Run-level budgets shared across workers |

**Backends**

- SQLite (default, WAL mode): safe for several processes on one host.
- Postgres (`jevex[postgres]`): for multiple hosts and many Scrapy workers.
- Custom backends implement `Store`.

**Packs**

- A pack is a directory of YAML files (generators, key mappings, optional examples) plus a `manifest.yaml` (name, version, compatible schemas, locales).
- Packs are distributable as git directories or PyPI packages registered under the `jevex.packs` entry point, e.g. `jevex-pack-automotive-uk`.
- CLI: `jevex pack export`, `jevex pack import`, `jevex pack diff`.

**Layering**

The store resolves generators through ordered layers, and the first match on an id wins:

1. Local learned layer (database)
2. Project packs
3. Community packs

Any layer can disable a generator from a lower layer without editing that layer.

## Results API

Each extracted record is a plain Pydantic instance, with full metadata alongside it. By default nothing is filtered: `record` holds the best answer for every field, and a threshold is opt-in.

```python
from jevex import Extractor, Document

extractor = Extractor(schemas=[VehicleSpec], store="sqlite:///jevex.db")

result = await extractor.extract(Document.from_bytes(pdf, url=url))

for item in result.records:          # list[Extracted[VehicleSpec]]
    spec = item.record               # plain VehicleSpec values
    m = item.meta.zero_to_62_s       # FieldMeta
    m.confidence, m.method, m.source.span, m.alternatives

spec = result.one().record           # convenience for single-entity documents
```

**Thresholds**

- `threshold=0` (default): `record` holds the raw best answers.
- `Extractor(threshold=0.8)` or `thresholds={"price": 0.95}`: values below the threshold become `None` in `record`. They are still available in full in `meta`.

**Required fields.** Records are built from a partial variant of the model (all fields optional), so a missing value never raises. `item.complete` reports whether the strict model validates, and `item.strict()` returns the validated instance or raises.

**FieldMeta**

| Attribute | Contents |
| --- | --- |
| `value` | The normalised best answer, even if filtered out of `record` |
| `confidence` | Jev confidence for the deciding question (probability for Noul) |
| `method` | `structured`, `jev`, `generator`, `llm` or `vision` |
| `generator_id` | The generator or key mapping that produced it |
| `source` | Document URL, component id, statement text, span offsets, and page + bbox or DOM path |
| `alternatives` | Other candidates with their probabilities |
| `verified` | Whether an LLM or vision value passed Jev verification |
| `shared` | The value came from an "all entities" statement |
| `conflicts` | Disagreeing values from other routes (merge mode) |

**Document-level meta:** document-gate results, Jev and LLM call counts, cost, timings, budget events and the generator snapshot version used.

**Review sink (optional).** A `ReviewSink` plugin receives fields below a threshold. Answers sent back through `extractor.feedback(...)` become verified examples.

## LLM adapters and budgets

The LLM is optional and has two separately configured roles. Every LLM and Jev call is metered against budgets you set.

```python
from jevex.llm.anthropic import AnthropicLLM

extractor = Extractor(
    schemas=[VehicleSpec],
    extraction_llm=AnthropicLLM(model="<fast model>"),     # fallback extraction
    generator_llm=AnthropicLLM(model="<strong model>"),    # generator synthesis
    budgets=Budgets(
        per_document=DocBudget(max_llm_calls=5, max_spend=0.05, timeout_s=30),
        run=RunBudget(max_spend=5.00, period="day", llm_rpm=60),
    ),
)
```

**Adapters**

| Extra | Adapter |
| --- | --- |
| `jevex[anthropic]` | `AnthropicLLM` (native) |
| `jevex[openai]` | `OpenAILLM` (native) |
| `jevex[litellm]` | `LiteLLM`, which covers any provider LiteLLM supports, including local Ollama |

Each adapter implements one small protocol, so anything else is a few lines:

```python
class LLM(Protocol):
    async def structured(self, prompt: str, schema: type[T]) -> LLMResponse[T]: ...
```

`LLMResponse` carries the parsed output plus token usage, which is used for cost accounting.

**Budgets**

- **Per document:** max LLM calls, max spend and a timeout.
- **Per run:** a spend cap per period and rate limits, shared across concurrent workers through the store's spend ledger.
- **When a budget is hit:** LLM use stops, and Jev and generators carry on. Affected fields get the best Jev answer or `None`, and the event is recorded in `meta.budget_events`.
- **Jev calls:** counted and costed too, and can have their own cap.

## Integration

jevex takes documents, not URLs, so it drops into any crawler. The built-in fetcher and cleaner exist for demos, tests and small standalone jobs.

**Input**

```python
class Document(BaseModel):
    content: bytes
    content_type: str          # text/html, application/pdf, image/*
    url: str | None
    fetched_at: datetime | None
```

**Ways in**

| Surface | What it is |
| --- | --- |
| Library | Async-first `await extractor.extract(doc)`, with `extract_sync` as a wrapper. Python 3.12+. |
| Scrapy | `jevex.contrib.scrapy.JevexPipeline`, an item pipeline that runs on Scrapy's asyncio reactor, plus a helper that turns a `Response` into a `Document` |
| Built-in fetcher | `jevex.fetch.SimpleFetcher`: httpx, honours robots.txt, polite per-host delay, no JavaScript rendering |
| Microservice | `jevex serve` (`jevex[server]`, FastAPI): `POST /extract` takes a document plus a schema name; schemas are registered by module path. Also `/health` and `/metrics`. |
| CLI | `jevex extract`, `learn`, `pack`, `eval`, `testsite`, `serve` |

**Car finder:** the Go app handles crawling and calls `jevex serve` over HTTP. `VehicleSpec` and `Listing` schemas live in a small Python package the service loads.

**Scraping responsibly:** some target sites forbid scraping in their terms, Autotrader included. The built-in fetcher honours robots.txt by default. Legal compliance stays with the caller, and the README should say so.

## Evaluation and test site

The eval harness has to prove the headline claim: accuracy holds while cost and the LLM-call rate fall as generators are learned. A synthetic test site with exact ground truth drives it.

**`jevex eval`**

- Input: a corpus directory of documents plus expected records (YAML or JSON).
- Per-field metrics: exact match, numeric match within a tolerance, precision and recall for lists.
- Run metrics: cost per document, latency, Jev and LLM calls, and the resolution mix (structured, jev, generator, llm, vision).
- `--replay`: starts from an empty store, processes the corpus in order, and reports metrics per batch. The output is CSV plus an HTML chart of accuracy, cost per document and LLM-call rate over documents processed.
- `--gate`: fails CI if accuracy drops or the LLM-call rate rises beyond set tolerances against a baseline.

**Synthetic test site (`testsite/` in the repo)**

- A seeded generator creates a dataset of fictional makes, models and variants, then renders a static site from it, along with the ground truth.
- It covers every feature:
  - several HTML template families
  - JSON-LD on some pages only
  - multi-entity listing grids
  - spec-sheet PDFs with trim columns
  - rasterised "scanned" PDFs
  - infographic images
  - a phrasing bank of varied wordings for the same fact
- It's deterministic for a given seed. `jevex testsite build --seed 42` builds it, and `jevex testsite serve` serves it locally.
- **Learning demo:** template families are released in waves. Each wave spikes LLM calls, which then fall away as generators are learned. This chart goes in the README.

**Real-world smoke test:** [books.toscrape.com](https://books.toscrape.com), which exists for scraping practice, with a small `Book` schema.

## Packaging and project

The project is `jevex`: Apache 2.0, Python 3.12+, Pydantic v2, fully typed (`py.typed`). The core install stays light, and heavy dependencies are extras.

| Extra | Pulls in |
| --- | --- |
| `pdf` | Docling (layout, tables, reading order) |
| `ocr` | OCR engine for images and scanned pages |
| `anthropic` / `openai` / `litellm` | LLM adapters |
| `postgres` | Postgres store |
| `server` | FastAPI microservice |
| `scrapy` | Scrapy pipeline integration |
| `all` | Everything |

## Decisions log

| Decision | Choice |
| --- | --- |
| Layout definition | Generic component model for all sites; no per-site config |
| Language | Python |
| Schema input | Pydantic model generates the questions; any stage can be overridden |
| Records per document | Single, multi and parent/child, behind a pluggable `EntityResolver` |
| Non-enum values | Candidate generators + Jev choice, with LLM fallback; all pluggable |
| Learning | Every verified LLM answer becomes a generator |
| Generator form | Declarative only in core; custom plugins can do anything |
| Library boundary | Documents in; a pipeline with optional fetcher and cleaner stages |
| When learning runs | Inline and async by default; compile-and-review and hybrid as options |
| Learned state | `Store` interface: database at runtime, file packs for review and distribution, layered |
| Uncertain fields | Return everything with metadata; review sink optional |
| Result shape | Plain record + side-channel `meta`; opt-in thresholds (default: no filtering) |
| Embedded data | Structured-only by default; `fill_gaps` and `merge` modes |
| Images | OCR on by default; pluggable image stage for vision models |
| Cross-document merging | Out of scope |
| LLM | Native adapters + optional LiteLLM; separate extraction and generator models |
| Evaluation | Harness with learning curves and a CI gate; synthetic test site |
| Budgets | Per document and per run |
| Name and license | jevex, Apache 2.0 |
| Required fields (decided in #20) | Records are instances of a generated partial model: the same fields, all optional, keeping each field's constraints, `Annotated` validators and alias, but not model-level validators, computed fields or serializers, so a missing value never raises. It's a separate class with only the fields: `isinstance` against the user's model is false and the model's methods and properties aren't on it. `strict()` returns the real model, validating every found value, including any the type rejected. |

## Open questions

- [ ] Register `jevex` on PyPI and GitHub now. PyPI returned 404 for the name on 2026-09-29.
- [ ] Email TypeSafe about using "Jev" in the name (trademark), and about a listing in their docs or cookbooks.
- [ ] Check Jev's limits: max state size (long tables and big components), questions per request, rate limits and pricing at batch scale.
- [ ] Locale handling beyond the UK: decimal commas, unit systems, and date order in generators and normalisers.
- [x] Confirm Docling as the default PDF parser, given its install weight. Decided in #24: Docling stays the `pdf` extra. It measured 1.1 GB and 103 packages on macOS (torch, transformers, opencv), plus about 500 MB of models downloaded on first use; on Linux, PyPI's torch also pulls in 15 CUDA packages (several GB) unless the CPU build is chosen. The lighter options give text with positions but no layout model: pypdfium2 finds no headings or tables, pdfplumber only ruled tables, and PyMuPDF is AGPL. The owner may revisit.
- [x] Confirm the partial-model approach for required fields. Decided in #20 (see the Decisions log); the owner may revisit.
- [ ] Set default values for `fallback_threshold`, `verify_threshold`, `learn_threshold` and `prune_after` from eval runs on the test site.

## Sources

- [TypeSafe AI](https://typesafe.ai/)
- [Jev docs: introduction](https://docs.typesafe.ai/introduction)
- [Jev docs index](https://docs.typesafe.ai/llms.txt)
- [winnow on PyPI](https://pypi.org/project/winnow/) (name taken, the reason for choosing jevex)
