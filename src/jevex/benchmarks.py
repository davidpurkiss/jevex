"""Benchmark corpora and the pinned benchmark setup (``docs/benchmarks.md``, #61).

Three things make a benchmark run reproducible:

- **Frozen corpora.** A :class:`CorpusLock` records the hash of a corpus's ``truth.json``
  and of every document it lists (:func:`lock_corpus`). :func:`check_lock` says which
  documents changed, went missing or appeared, so a rerun can prove it saw the same inputs
  (:func:`verify_lock` raises :class:`CorpusLockError` instead). The test site is rebuilt
  from its seed and checked against its lock; fetched or hand-made corpora are kept and
  checked the same way.
- **The real-world corpus.** :func:`books_corpus` fetches a seeded sample of product pages
  from books.toscrape.com (Zyte's scraping sandbox) through :class:`~jevex.fetch.SimpleFetcher`,
  so robots.txt and a per-host delay apply, and labels each page from its own markup
  (:func:`book_values`): the product table's price and availability, the ``<h1>`` title
  and the ``star-rating`` class. The labels are mechanical; a human spot-checks them.
- **Pinned settings.** :class:`BenchmarkConfig` (``benchmarks/config.yaml``) pins the seeds,
  the corpora and their locks, the Jev and LLM model versions with the prices a run is
  costed at, the concurrency latency is measured at, and the budget cap.

:func:`bootstrap_interval` gives the seeded 95% bootstrap confidence intervals the
results report around per-document means.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast
from urllib.parse import urljoin, urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jevex.baseline import corpus_digest
from jevex.clean import html_text_of
from jevex.eval import TRUTH_FILE, load_corpus
from jevex.fetch import SimpleFetcher

if TYPE_CHECKING:
    from collections.abc import Sequence

LOCK_VERSION = 1
CONFIG_VERSION = 1

Publish = Literal["full", "aggregate"]
"""``full``: per-document results may be published. ``aggregate``: only the corpus's
summary numbers may (documents kept local, e.g. spec sheets we may not redistribute)."""


# --- corpus locks ----------------------------------------------------------------------


class CorpusLockError(ValueError):
    """A corpus doesn't match its lock, or a lock file can't be read."""


class CorpusLock(BaseModel):
    """What a frozen corpus holds: hashes of its ``truth.json`` and of every document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = LOCK_VERSION
    name: str
    publish: Publish = "full"
    settings: dict[str, Any] = Field(default_factory=dict[str, Any])
    """``truth.json``'s top-level settings (everything but ``pages`` and the test site's
    own ``digest``): the seed and waves a test site was built with, a sample's seed and
    size. For reading; the hashes are what's checked."""
    truth: str
    """SHA-256 of ``truth.json``'s bytes: the labels."""
    documents: dict[str, str]
    """Each document's path (relative, ``/``-separated) → SHA-256 of its bytes."""
    digest: str
    """:func:`~jevex.baseline.corpus_digest`, the hash a :class:`~jevex.baseline.Baseline`
    records, so a baseline can be tied to a locked corpus."""

    @classmethod
    def load(cls, path: str | Path) -> CorpusLock:
        """Read a lock file. Raises :class:`CorpusLockError` naming the file."""
        try:
            return cls.model_validate_json(Path(path).read_bytes())
        except OSError as exc:
            raise CorpusLockError(f"can't read lock {path}: {exc.strerror or exc}") from exc
        except ValidationError as exc:
            raise CorpusLockError(f"{path} isn't a corpus lock: {exc}") from exc

    def write(self, path: str | Path) -> None:
        """Write the lock as indented JSON with sorted keys, so diffs stay readable."""
        data = self.model_dump(mode="json")
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_corpus(directory: str | Path, name: str, *, publish: Publish = "full") -> CorpusLock:
    """Lock the corpus in ``directory`` as it is now.

    Raises ``ValueError`` (from :func:`~jevex.eval.load_corpus`) for a malformed corpus or
    one whose ``truth.json`` lists a document that doesn't exist.
    """
    root = Path(directory)
    items = load_corpus(root)
    manifest = cast("dict[str, Any]", json.loads((root / TRUTH_FILE).read_text()))
    return CorpusLock(
        name=name,
        publish=publish,
        settings={k: v for k, v in manifest.items() if k not in ("pages", "digest")},
        truth=_sha256(root / TRUTH_FILE),
        documents={item.path.relative_to(root).as_posix(): _sha256(item.path) for item in items},
        digest=corpus_digest(root),
    )


@dataclass(frozen=True)
class LockCheck:
    """How a corpus differs from its lock. Paths are relative to the corpus."""

    truth_changed: bool
    changed: tuple[str, ...]
    """Documents whose bytes differ from the lock's."""
    missing: tuple[str, ...]
    """Documents the lock lists that aren't there."""
    extra: tuple[str, ...]
    """Documents ``truth.json`` lists that the lock doesn't."""

    @property
    def ok(self) -> bool:
        return not (self.truth_changed or self.changed or self.missing or self.extra)

    def describe(self) -> str:
        """One line per difference (at most five documents each), or ``matches``."""
        if self.ok:
            return "matches"
        lines = ["truth.json changed"] if self.truth_changed else []
        for label, paths in (
            ("changed", self.changed),
            ("missing", self.missing),
            ("not in the lock", self.extra),
        ):
            if paths:
                shown = ", ".join(paths[:5]) + (f" and {len(paths) - 5} more" if paths[5:] else "")
                lines.append(f"{len(paths)} {label}: {shown}")
        return "\n".join(lines)


def check_lock(directory: str | Path, lock: CorpusLock) -> LockCheck:
    """Compare the corpus in ``directory`` with ``lock``.

    Unlike :func:`lock_corpus` this doesn't need the corpus to load, so a deleted document
    shows up as ``missing`` rather than an error. Raises ``ValueError`` only if there's no
    readable ``truth.json`` listing pages.
    """
    root = Path(directory)
    truth = root / TRUTH_FILE
    try:
        pages = cast("list[Any]", json.loads(truth.read_text())["pages"])
        listed = {str(cast("dict[str, Any]", p)["path"]) for p in pages}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{truth} isn't a corpus manifest: {exc}") from exc
    changed: list[str] = []
    missing: list[str] = []
    for path, sha in sorted(lock.documents.items()):
        file = root / path
        if not file.is_file():
            missing.append(path)
        elif _sha256(file) != sha:
            changed.append(path)
    extra = sorted(p for p in listed if Path(p).as_posix() not in lock.documents)
    return LockCheck(
        truth_changed=_sha256(truth) != lock.truth,
        changed=tuple(changed),
        missing=tuple(missing),
        extra=tuple(extra),
    )


def verify_lock(directory: str | Path, lock: CorpusLock) -> None:
    """Raise :class:`CorpusLockError` saying what differs unless the corpus matches."""
    check = check_lock(directory, lock)
    if not check.ok:
        raise CorpusLockError(
            f"{directory} doesn't match the {lock.name!r} lock:\n{check.describe()}"
        )


# --- books.toscrape.com ----------------------------------------------------------------

BOOKS_URL = "https://books.toscrape.com/"
BOOKS_SAMPLE = 200
BOOKS_SCHEMA = "Book"
"""The schema the corpus is labelled in: :class:`jevex.examples.books.Book`."""
MAX_CATALOGUE_PAGES = 200
"""A stop for a catalogue whose "next" links never end (the real one has 50 pages)."""

_RATINGS = ("One", "Two", "Three", "Four", "Five")
_PRICE = re.compile(r"\d+(?:\.\d+)?")
_IN_STOCK = re.compile(r"^in stock\s*\((\d+) available\)$", re.I)


class BookPageError(ValueError):
    """A page doesn't have what a books.toscrape.com product page has."""


class _BookPage(HTMLParser):
    """Reads a product page: the ``<h1>``, the first ``star-rating`` after it (later ones
    belong to the "recently viewed" books) and the product table's ``<th>``/``<td>`` rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self.rating: str | None = None
        self.rows: dict[str, str] = {}
        self._capture: list[str] | None = None
        self._capturing = ""
        self._header: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if (tag == "h1" and self.title is None) or tag in ("th", "td"):
            self._capture, self._capturing = [], tag
        classes = (dict(attrs).get("class") or "").split()
        if self.title is not None and self.rating is None and "star-rating" in classes:
            self.rating = next((c for c in classes if c in _RATINGS), None)

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None or tag != self._capturing:
            return
        text = " ".join("".join(self._capture).split())
        self._capture = None
        if tag == "h1":
            self.title = text
        elif tag == "th":
            self._header = text
        elif self._header is not None:
            self.rows.setdefault(self._header, text)
            self._header = None

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._capture.append(data)


def book_values(html: str) -> dict[str, Any]:
    """A product page's :class:`~jevex.examples.books.Book` values, as ``truth.json`` holds
    them (the price as a decimal string). Raises :class:`BookPageError` naming what's
    missing."""
    page = _BookPage()
    page.feed(html)
    page.close()
    if not page.title:
        raise BookPageError("no <h1> title")
    if page.rating is None:
        raise BookPageError("no star-rating after the title")
    price = _PRICE.search(page.rows.get("Price (incl. tax)", ""))
    if price is None:
        raise BookPageError("no 'Price (incl. tax)' row")
    availability = page.rows.get("Availability")
    if availability is None:
        raise BookPageError("no 'Availability' row")
    if (stock := _IN_STOCK.match(availability)) is not None:
        in_stock, count = True, int(stock.group(1))
    elif availability.casefold() == "out of stock":
        in_stock, count = False, 0
    else:
        raise BookPageError(f"can't read availability {availability!r}")
    return {
        "title": page.title,
        "price": price.group(),
        "in_stock": in_stock,
        "stock_count": count,
        "rating": page.rating,
    }


class _Catalogue(HTMLParser):
    """A catalogue page's product links (each ``<h3><a href>``) and its "next" link."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.products: list[str] = []
        self.next: str | None = None
        self._in_h3 = False
        self._in_next = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "h3":
            self._in_h3 = True
        elif tag == "li" and "next" in (attributes.get("class") or "").split():
            self._in_next = True
        elif tag == "a" and (href := attributes.get("href")):
            if self._in_h3:
                self.products.append(href)
            elif self._in_next and self.next is None:
                self.next = href

    def handle_endtag(self, tag: str) -> None:
        if tag == "h3":
            self._in_h3 = False
        elif tag == "li":
            self._in_next = False


def catalogue_page(html: str, url: str) -> tuple[list[str], str | None]:
    """The absolute product URLs on a catalogue page, in page order, and the next page's
    URL (``None`` on the last page)."""
    page = _Catalogue()
    page.feed(html)
    page.close()
    products = [urljoin(url, href) for href in page.products]
    return products, urljoin(url, page.next) if page.next else None


def _slug(url: str) -> str:
    """``.../catalogue/a-light-in-the-attic_1000/index.html`` → ``a-light-in-the-attic_1000``."""
    parts = [p for p in urlsplit(url).path.split("/") if p and p != "index.html"]
    if not parts:
        raise BookPageError(f"can't name a file for {url}")
    return re.sub(r"[^A-Za-z0-9_.-]", "-", parts[-1].removesuffix(".html"))


async def books_corpus(
    out: str | Path,
    *,
    sample: int = BOOKS_SAMPLE,
    seed: int = 42,
    fetcher: SimpleFetcher | None = None,
    base_url: str = BOOKS_URL,
) -> CorpusLock:
    """Fetch a seeded sample of ``sample`` books.toscrape.com product pages into ``out``,
    write their ``truth.json``, and return the corpus's lock (named ``books``).

    The whole catalogue is walked (its "next" links from ``catalogue/page-1.html``), then
    ``random.Random(seed).sample`` picks the books, which are fetched and listed in
    catalogue order. Pages are saved byte for byte as ``pages/<slug>.html``. The same seed
    picks the same books; the lock tells whether the site still serves the same bytes.

    ``fetcher`` defaults to a :class:`~jevex.fetch.SimpleFetcher` (robots.txt, one request
    a second), closed afterwards; a fetcher passed in is left open. ``out`` must be empty
    or not exist. Raises ``ValueError`` if the catalogue has fewer than ``sample`` books,
    if its "next" links loop or run past :data:`MAX_CATALOGUE_PAGES`, :class:`BookPageError`
    naming a page that can't be labelled, and
    :class:`~jevex.fetch.FetchError` (or its ``RobotsDisallowedError``) from fetching.
    Nothing is written until every page is fetched and labelled.
    """
    root = Path(out)
    if sample < 1:
        raise ValueError(f"sample must be at least 1, not {sample}")
    await asyncio.to_thread(_check_empty, root)
    own = fetcher is None
    client = fetcher or SimpleFetcher()
    try:
        catalogue: list[str] = []
        url: str | None = urljoin(base_url, "catalogue/page-1.html")
        seen: set[str] = set()
        while url is not None and url not in seen and len(seen) < MAX_CATALOGUE_PAGES:
            seen.add(url)
            document = await client.fetch(url)
            products, url = catalogue_page(html_text_of(document.content), document.url or url)
            catalogue.extend(products)
        if url is not None:  # a loop or a runaway catalogue: sampling it would mislead
            raise ValueError(f"stopped walking the catalogue at {url} after {len(seen)} pages")
        catalogue = list(dict.fromkeys(catalogue))  # a book listed twice counts once
        if len(catalogue) < sample:
            raise ValueError(f"the catalogue has {len(catalogue)} books, fewer than {sample}")
        picked = set(random.Random(seed).sample(catalogue, sample))
        chosen = {f"pages/{_slug(p)}.html": p for p in catalogue if p in picked}
        if len(chosen) != sample:
            raise BookPageError("two sampled books map to the same file name")
        pages: list[_SavedPage] = []
        for path, product in chosen.items():
            document = await client.fetch(product)
            try:
                values = book_values(html_text_of(document.content))
            except BookPageError as exc:
                raise BookPageError(f"{product}: {exc}") from exc
            pages.append(_SavedPage(path, product, document.content, values))
    finally:
        if own:
            await client.aclose()
    manifest = {"source": base_url, "seed": seed, "sample": sample, "labels": "markup"}
    return await asyncio.to_thread(_write_books, root, manifest, pages)


@dataclass(frozen=True)
class _SavedPage:
    path: str
    url: str
    content: bytes
    values: dict[str, Any]


def _check_empty(root: Path) -> None:
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"{root} isn't an empty directory; refusing to write a corpus there")


def _write_books(root: Path, settings: dict[str, Any], pages: list[_SavedPage]) -> CorpusLock:
    for page in pages:
        target = root / page.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(page.content)
    entries = [
        {
            "path": page.path,
            "url": page.url,
            "schema": BOOKS_SCHEMA,
            "content_type": "text/html",
            "records": [{"entity": "document", "values": page.values}],
        }
        for page in pages
    ]
    manifest = settings | {"pages": entries}
    (root / TRUTH_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return lock_corpus(root, "books")


# --- the pinned configuration ----------------------------------------------------------


class BenchmarkConfigError(ValueError):
    """``benchmarks/config.yaml`` can't be read or isn't a valid benchmark config."""


class PinnedModel(BaseModel):
    """One model at an exact version, with the prices a run is costed at."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["jev", "anthropic", "openai", "gemini", "litellm"]
    model: str = Field(min_length=1)
    """The exact version: a dated snapshot or a versioned name, never a ``-latest`` alias."""
    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)
    price_date: date
    """When the prices were read from the provider's price page."""

    @model_validator(mode="after")
    def _pinned(self) -> Self:
        if self.model.endswith("latest"):
            raise ValueError(f"{self.model!r} is an alias; pin an exact version")
        return self

    @property
    def spec(self) -> str:
        """``provider:model``, the form ``jevex --llm`` takes."""
        return f"{self.provider}:{self.model}"

    def cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        """USD for this much usage at the pinned prices."""
        return (
            input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000


class PinnedModels(BaseModel):
    """The LLMs each system uses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extraction: PinnedModel
    """jevex's fallback (``Extractor(extraction_llm=)``)."""
    generator: PinnedModel
    """jevex's learner (``Extractor(generator_llm=)``)."""
    baseline_fast: PinnedModel
    baseline_strong: PinnedModel
    baseline_gemini: PinnedModel | None = None
    """Wanted by the owner; #62 picks the model."""

    @model_validator(mode="after")
    def _llms(self) -> Self:
        for role, pinned in self:
            if isinstance(pinned, PinnedModel) and pinned.provider == "jev":
                raise ValueError(f"models.{role} must be an LLM, not Jev")
        return self


class CorpusSpec(BaseModel):
    """One benchmark corpus: how to get it, its lock, and what may be published."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    kind: Literal["testsite", "books", "directory"]
    """``testsite``: rebuilt from ``seed`` and ``waves``. ``books``: fetched once with
    :func:`books_corpus` (``seed``, ``sample``) and kept. ``directory``: a corpus kept as
    files, at ``path`` (relative to the config) or in the directory ``env`` names."""
    lock: str
    """The lock file, relative to the config file."""
    publish: Publish = "full"
    seed: int | None = None
    waves: str | None = None
    """``testsite`` only: the wave schedule (``jevex testsite build --waves`` syntax);
    the default schedule if unset."""
    sample: int | None = Field(default=None, ge=1)
    path: str | None = None
    env: str | None = None

    @model_validator(mode="after")
    def _fits_kind(self) -> Self:
        needs: dict[str, tuple[str, ...]] = {
            "testsite": ("seed",),
            "books": ("seed", "sample"),
            "directory": (),
        }
        missing = [f for f in needs[self.kind] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"a {self.kind} corpus needs {', '.join(missing)}")
        if self.kind == "directory" and (self.path is None) == (self.env is None):
            raise ValueError("a directory corpus needs exactly one of path and env")
        if self.kind != "directory" and (self.path is not None or self.env is not None):
            raise ValueError("path and env are only for directory corpora")
        if self.waves is not None and self.kind != "testsite":
            raise ValueError("waves is only for testsite corpora")
        return self


class BootstrapSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int = Field(default=1000, ge=100)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)


class BenchmarkConfig(BaseModel):
    """The pinned benchmark setup. Change it only alongside a new results run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = CONFIG_VERSION
    seed: int
    """Seeds everything not given its own: bootstrap resampling, LLM sampling."""
    concurrency: int = Field(ge=1)
    """Documents in flight while latency is measured."""
    budget_usd: float = Field(gt=0)
    """The hard cap for one full run, Jev and LLMs together."""
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    jev: PinnedModel
    models: PinnedModels
    corpora: tuple[CorpusSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.jev.provider != "jev":
            raise ValueError("jev must be a Jev model (provider: jev)")
        names = [c.name for c in self.corpora]
        if len(set(names)) != len(names):
            raise ValueError(f"corpus names must be unique: {names}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        """Read a YAML config. Raises :class:`BenchmarkConfigError` naming the file."""
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except OSError as exc:
            raise BenchmarkConfigError(f"can't read {path}: {exc.strerror or exc}") from exc
        except (yaml.YAMLError, ValidationError) as exc:
            raise BenchmarkConfigError(f"{path} isn't a benchmark config: {exc}") from exc

    def corpus(self, name: str) -> CorpusSpec:
        """The corpus called ``name``. Raises ``KeyError`` if there's none."""
        for spec in self.corpora:
            if spec.name == name:
                return spec
        raise KeyError(name)


# --- statistics ------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A mean and its bootstrap confidence interval."""

    mean: float
    low: float
    high: float


def bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Interval:
    """The mean of ``values`` (e.g. one accuracy or cost per document) and its percentile
    bootstrap interval: ``samples`` resamples with replacement, drawn from
    ``random.Random(seed)``, so the same inputs always give the same interval.

    Raises ``ValueError`` for no values, fewer than one resample, or a confidence outside
    (0, 1).
    """
    if not values:
        raise ValueError("no values to bootstrap")
    if samples < 1:
        raise ValueError(f"samples must be at least 1, not {samples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be between 0 and 1, not {confidence}")
    data = [float(v) for v in values]
    n = len(data)
    rng = random.Random(seed)
    means = sorted(math.fsum(rng.choices(data, k=n)) / n for _ in range(samples))
    tail = (1.0 - confidence) / 2
    return Interval(
        mean=math.fsum(data) / n,
        low=_quantile(means, tail),
        high=_quantile(means, 1.0 - tail),
    )


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks."""
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


__all__ = [
    "BOOKS_SAMPLE",
    "BOOKS_URL",
    "BenchmarkConfig",
    "BenchmarkConfigError",
    "BookPageError",
    "BootstrapSettings",
    "CorpusLock",
    "CorpusLockError",
    "CorpusSpec",
    "Interval",
    "LockCheck",
    "PinnedModel",
    "PinnedModels",
    "book_values",
    "books_corpus",
    "bootstrap_interval",
    "catalogue_page",
    "check_lock",
    "lock_corpus",
    "verify_lock",
]
