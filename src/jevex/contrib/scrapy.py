"""Scrapy integration (spec: *Integration*, ``scrapy`` extra).

Scrapy does the crawling; jevex extracts. A spider turns each page it wants records from
into a :class:`~jevex.document.Document` with :func:`document_from_response` and yields it
in an item, under ``document``::

    def parse_book(self, response):
        yield {"url": response.url, "document": document_from_response(response)}

:class:`JevexPipeline`, an item pipeline, extracts the registered schemas from it on
Scrapy's asyncio loop and puts the records in the item instead (``records``: each one's
schema, entity and values as JSON types, as ``jevex extract`` prints them), so feed exports
write them as they are. Items without a document (or with ``None``) pass through untouched.

Settings:

- ``JEVEX_SCHEMAS``: the schemas, as model classes or ``module:Class`` paths (as
  ``jevex extract --schema`` takes them; ``-s JEVEX_SCHEMAS=a:A,b:B`` works too).
- ``JEVEX_STORE``: a store URL such as ``sqlite:///jevex.db`` (default: none). Workers
  sharing a store share learned generators and the run budget's ledger.
- ``JEVEX_THRESHOLD``: the confidence below which a value is left out of the record
  (default 0: keep everything).
- ``JEVEX_META``: add each field's meta to the records, and the document's meta to the
  item under ``document_meta`` (default ``False``).
- ``JEVEX_DOCUMENT_FIELD`` / ``JEVEX_RECORDS_FIELD`` / ``JEVEX_META_FIELD``: the item
  fields (``document``, ``records`` and ``document_meta``). A ``scrapy.Item`` must declare
  the ones it gets.
- ``JEVEX_KEEP_DOCUMENT``: leave the document in the item (default ``False``: its bytes
  are dropped, so they don't end up in the feed).
- ``JEVEX_WAIT_FOR_LEARNING``: when the spider closes, let examples still queued for the
  learner finish before closing the extractor (default ``True``).

Anything else (a custom pipeline, LLMs, budgets, packs, a review sink) is a subclass that
overrides :meth:`JevexPipeline.make_extractor`; :meth:`JevexPipeline.fill_item` decides
what the item gets.

The pipeline needs the asyncio reactor (``TWISTED_REACTOR`` set to
``twisted.internet.asyncioreactor.AsyncioSelectorReactor``, Scrapy's default) or Scrapy
running without a reactor: jevex is asyncio code.
"""

from __future__ import annotations

import mimetypes
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Self

from itemadapter import ItemAdapter
from pydantic import BaseModel
from scrapy.utils.asyncio import is_asyncio_available

from jevex.document import OCTET_STREAM, Document, sniff_content_type
from jevex.extractor import Extractor

if TYPE_CHECKING:
    from scrapy.crawler import Crawler
    from scrapy.http import Response
    from scrapy.settings import Settings

    from jevex.extractor import ExtractionResult

__all__ = ["AsyncioRequiredError", "JevexPipeline", "document_from_response"]


class AsyncioRequiredError(RuntimeError):
    """Scrapy is running on a reactor other than the asyncio one."""


def document_from_response(response: Response, *, site: str | None = None) -> Document:
    """The response as a :class:`~jevex.document.Document`: its body, URL and media type.

    The media type comes from the ``Content-Type`` header. Without one, or when it's
    ``application/octet-stream`` (as servers often send PDFs), it's sniffed from the bytes,
    then guessed from the URL's extension. ``fetched_at`` is when this runs: Scrapy doesn't
    record when a response arrived. ``site`` is passed on (see ``Document.site``).
    """
    header = response.headers.get(b"Content-Type")
    content_type = header.decode("latin-1").split(";", 1)[0].strip().lower() if header else ""
    if not content_type or content_type == OCTET_STREAM:
        content_type = (
            sniff_content_type(response.body)
            or mimetypes.guess_type(response.url)[0]
            or content_type
        )
    return Document.from_bytes(
        response.body,
        url=response.url,
        content_type=content_type or None,
        fetched_at=datetime.now(UTC),
        site=site,
    )


class JevexPipeline:
    """An item pipeline that extracts records from the documents in items.

    Enable it in ``ITEM_PIPELINES`` and set ``JEVEX_SCHEMAS`` (see the module docs for
    every setting). The pipeline owns its extractor: it's made when the spider opens
    (:meth:`make_extractor`) and closed when it closes. Items are extracted concurrently,
    up to Scrapy's ``CONCURRENT_ITEMS``. An extraction that raises fails its item, which
    Scrapy logs and drops.

    Counts go to Scrapy's stats under ``jevex/``: ``documents``, ``records``, ``stopped``
    (documents a budget or gate stopped early), ``jev_requests``, ``llm_calls``,
    ``jev_cost_usd`` and ``llm_cost_usd``.
    """

    def __init__(self, crawler: Crawler) -> None:
        self.crawler = crawler
        self.settings: Settings = crawler.settings
        self.document_field: str = self.settings.get("JEVEX_DOCUMENT_FIELD", "document")
        self.records_field: str = self.settings.get("JEVEX_RECORDS_FIELD", "records")
        self.meta_field: str = self.settings.get("JEVEX_META_FIELD", "document_meta")
        self.meta = self.settings.getbool("JEVEX_META", False)
        self.keep_document = self.settings.getbool("JEVEX_KEEP_DOCUMENT", False)
        self.wait_for_learning = self.settings.getbool("JEVEX_WAIT_FOR_LEARNING", True)
        self._extractor: Extractor | None = None

    @classmethod
    def from_crawler(cls, crawler: Crawler) -> Self:
        return cls(crawler)

    @property
    def extractor(self) -> Extractor:
        """The extractor, once the spider has opened."""
        if self._extractor is None:
            raise RuntimeError("the spider hasn't opened yet, so there's no extractor")
        return self._extractor

    def make_extractor(self) -> Extractor:
        """The extractor for this crawl, from ``JEVEX_SCHEMAS``, ``JEVEX_STORE`` and
        ``JEVEX_THRESHOLD``. Override it to configure anything else.

        It runs on Scrapy's event loop when the spider opens, so the extractor's Jev client
        and store belong to the loop the items are processed on.
        """
        specs: list[object] = self.settings.getlist("JEVEX_SCHEMAS")
        if not specs:
            raise ValueError(
                "set JEVEX_SCHEMAS to the schemas to extract (model classes or module:Class "
                "paths), or override JevexPipeline.make_extractor"
            )
        return Extractor(
            [_schema(s) for s in specs],
            store=self.settings.get("JEVEX_STORE") or None,
            threshold=self.settings.getfloat("JEVEX_THRESHOLD", 0.0),
        )

    async def open_spider(self) -> None:
        if not is_asyncio_available():
            raise AsyncioRequiredError(
                "JevexPipeline needs Scrapy's asyncio reactor: set TWISTED_REACTOR to "
                "'twisted.internet.asyncioreactor.AsyncioSelectorReactor'"
            )
        self._extractor = self.make_extractor()

    async def process_item(self, item: Any) -> Any:
        adapter = ItemAdapter(item)
        if self.document_field not in adapter:
            return item
        document: object = adapter[self.document_field]
        if document is None:
            return item
        if not isinstance(document, Document):
            raise TypeError(
                f"item field {self.document_field!r} holds a {type(document).__name__}, not a "
                "jevex Document (make one with jevex.contrib.scrapy.document_from_response)"
            )
        result = await self.extractor.extract(document)
        self._count(result)
        return self.fill_item(item, result)

    def fill_item(self, item: Any, result: ExtractionResult) -> Any:
        """Put the result in the item, and drop the document unless ``JEVEX_KEEP_DOCUMENT``.

        Override it to build typed items from ``result`` (``result.for_schema(Model)``).
        """
        adapter = ItemAdapter(item)
        if not self.keep_document:
            del adapter[self.document_field]
        if self.meta:
            adapter[self.records_field] = [r.to_dict() for r in result.records]
            adapter[self.meta_field] = result.meta.model_dump(mode="json")
        else:
            adapter[self.records_field] = result.to_plain_dict()["records"]
        return item

    async def close_spider(self) -> None:
        extractor, self._extractor = self._extractor, None
        if extractor is None:
            return
        try:
            if self.wait_for_learning:
                await extractor.wait_for_learning()
        finally:
            await extractor.aclose()

    def _count(self, result: ExtractionResult) -> None:
        stats = self.crawler.stats
        meta = result.meta
        stats.inc_value("jevex/documents")
        stats.inc_value("jevex/records", len(result.records))
        stats.inc_value("jevex/jev_requests", meta.jev.requests)
        stats.inc_value("jevex/llm_calls", meta.llm.calls)
        if meta.stopped:
            stats.inc_value("jevex/stopped")
        for key, cost in (
            ("jevex/jev_cost_usd", meta.jev.cost),
            ("jevex/llm_cost_usd", meta.llm.cost),
        ):
            stats.set_value(key, stats.get_value(key, 0.0) + cost)


def _schema(spec: object) -> type[BaseModel]:
    if isinstance(spec, type) and issubclass(spec, BaseModel):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"JEVEX_SCHEMAS takes model classes or module:Class paths, not {spec!r}")
    from jevex.cli import CliError, load_schema

    try:
        return load_schema(spec.strip())
    except CliError as exc:
        raise ValueError(f"JEVEX_SCHEMAS: {exc}") from exc
