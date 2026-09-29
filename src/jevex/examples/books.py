"""A ``Book`` schema for bookshop product pages, e.g. https://books.toscrape.com.

``StarRatingCleaner`` shows how a site-specific cleaner fills a gap: books.toscrape.com
gives a book's rating only as a CSS class (``<p class="star-rating Three">``), which has
no text for Jev to read, so the cleaner writes it out ("Rating: Three out of five stars")
before the default boilerplate cleaner runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from jevex.clean import BoilerplateCleaner, CleanStage, decode_html
from jevex.document import Document
from jevex.extractor import default_pipeline
from jevex.pipeline import Pipeline
from jevex.schema import Field


class Book(BaseModel):
    """A book for sale in an online bookshop."""

    title: str = Field(description="Title of the book")
    price: Decimal = Field(description="Price", unit="GBP")
    in_stock: bool = Field(description="The book is in stock")
    stock_count: int = Field(description="Number of copies available")
    rating: Literal["One", "Two", "Three", "Four", "Five"] = Field(
        description="Star rating out of five"
    )


_STAR_RATING = re.compile(
    r"""(<[a-z]+\b[^>]*\bclass=["'][^"']*\bstar-rating\s+(One|Two|Three|Four|Five)\b[^>]*>)""",
    re.I,
)


@dataclass(frozen=True)
class StarRatingCleaner:
    """Writes ``star-rating N`` classes out as text, then runs ``inner``."""

    inner: BoilerplateCleaner = field(default_factory=BoilerplateCleaner)

    def clean(self, document: Document) -> Document:
        if not document.is_html:
            return self.inner.clean(document)
        html, _ = decode_html(document.content)
        rewritten = _STAR_RATING.sub(r"\1Rating: \2 out of five stars", html)
        if rewritten != html:
            document = document.model_copy(update={"content": rewritten.encode("utf-8")})
        return self.inner.clean(document)


def books_pipeline() -> Pipeline:
    """The default pipeline with :class:`StarRatingCleaner` as its cleaner."""
    return default_pipeline().replace("clean", CleanStage(StarRatingCleaner()))
