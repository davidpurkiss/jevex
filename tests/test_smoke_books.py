"""Smoke test: real product pages from books.toscrape.com through the default pipeline.

The pages are saved in ``tests/fixtures/books/``. Jev's answers are replayed from
``tests/fixtures/books/jev-cassette.json``, recorded once against the real API:

    JEVEX_RECORD=1 TYPESAFE_API_KEY=... JEVEX_JEV_MAX_COST_USD=0.50 \\
        uv run pytest tests/test_smoke_books.py

Until a recording exists, the replay tests skip. Any pipeline change that alters what
jevex asks Jev makes the recording stale; those tests then xfail with a note to
re-record, rather than failing builds that can't reach the API.
"""

import os
import re
from decimal import Decimal
from pathlib import Path

import pytest

from jevex import Document, Extractor
from jevex.examples.books import Book, StarRatingCleaner, books_pipeline
from jevex.jev import Choice, JevClient
from jevex.layout_html import HtmlLayoutParser
from jevex.testing import RECORD_ENV, CassetteMissError, FakeJev, cassette

FIXTURES = Path(__file__).parent / "fixtures" / "books"
CASSETTE = FIXTURES / "jev-cassette.json"

PAGES = {
    "a-light-in-the-attic_1000": Book(
        title="A Light in the Attic",
        price=Decimal("51.77"),
        in_stock=True,
        stock_count=22,
        rating="Three",
    ),
    "tipping-the-velvet_999": Book(
        title="Tipping the Velvet",
        price=Decimal("53.74"),
        in_stock=True,
        stock_count=20,
        rating="One",
    ),
    "sapiens-a-brief-history-of-humankind_996": Book(
        title="Sapiens: A Brief History of Humankind",
        price=Decimal("54.23"),
        in_stock=True,
        stock_count=20,
        rating="Five",
    ),
}


def page(name: str) -> Document:
    return Document.from_bytes(
        (FIXTURES / f"{name}.html").read_bytes(),
        url=f"https://books.toscrape.com/catalogue/{name}/index.html",
    )


# --- with recorded Jev answers -------------------------------------------------------


@pytest.fixture
def recorded_jev() -> JevClient:
    if not CASSETTE.exists() and os.environ.get(RECORD_ENV) != "1":
        pytest.skip(
            "no Jev recording yet; record with JEVEX_RECORD=1 TYPESAFE_API_KEY=... "
            "uv run pytest tests/test_smoke_books.py"
        )
    return JevClient(cassette(CASSETTE))


@pytest.mark.parametrize("name", list(PAGES))
async def test_books_page_with_recorded_jev(name: str, recorded_jev: JevClient) -> None:
    async with Extractor([Book], jev=recorded_jev, pipeline=books_pipeline()) as ex:
        try:
            result = await ex.extract(page(name))
        except CassetteMissError as exc:
            pytest.xfail(f"the Jev recording is stale; re-record it ({exc})")
    found = result.one(Book).record.model_dump(exclude_unset=True)
    assert found == PAGES[name].model_dump()


# --- wiring, with scripted answers (always runs) -------------------------------------


async def test_star_ratings_are_written_out_as_text() -> None:
    doc = StarRatingCleaner().clean(page("a-light-in-the-attic_1000"))
    root = await HtmlLayoutParser().parse(doc)
    texts = [c.text for c in root.walk()]
    assert "Rating: Three out of five stars" in [t.strip() for t in texts]
    assert "A Light in the Attic" in texts


def scripted_jev(expected: Book) -> FakeJev:
    """Answers a careful reader would give for this page."""

    def pick(value: str):
        def choose(q: Choice) -> str:
            return value if value in q.options else "none"

        return choose

    return (
        FakeJev(strict=True)
        .noul("Does this document describe", p=0.97)
        .noul("Does this section", p=0.05)
        .noul("Does this section", p=0.9, state=f"Rating: {expected.rating} out of")
        .choice("Which detail", "none")
        .choice("Which detail", "title", state=f'"statement": "{expected.title}"')
        .choice("Which detail", "price", state=f'"statement": "£{expected.price}"')
        # "In stock (22 available)" states two fields: the count, and that it's in stock.
        .choice(
            "Which detail",
            "stock_count",
            confidence=0.55,
            probabilities={"in_stock": 0.4},
            state="available)",
        )
        .choice("Which detail", "rating", state=f"Rating: {expected.rating}")
        .choice("Which of these", "none")
        .choice(re.compile("(?i)which of these is the title"), pick(expected.title))
        .choice(re.compile("(?i)which of these is the price"), pick(f"£{expected.price}"))
        .choice(re.compile("(?i)which of these is the number"), pick(str(expected.stock_count)))
        .choice(re.compile("(?i)what is the star rating"), expected.rating)
        .noul(re.compile("(?i)does the statement say the book is in stock"), p=0.95)
    )


@pytest.mark.parametrize("name", list(PAGES))
async def test_books_page_with_scripted_jev(name: str) -> None:
    expected = PAGES[name]
    fake = scripted_jev(expected)
    async with Extractor([Book], jev=fake.client(), pipeline=books_pipeline()) as ex:
        result = await ex.extract(page(name))
    assert result.one(Book).strict() == expected
