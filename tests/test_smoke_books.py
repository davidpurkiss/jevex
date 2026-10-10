"""Smoke test: real product pages from books.toscrape.com through the default pipeline.

The pages are saved in ``tests/fixtures/books/``. Jev's answers are replayed from
``tests/fixtures/books/jev-cassette.json``, recorded once against the real API:

    JEVEX_RECORD=1 TYPESAFE_API_KEY=... JEVEX_JEV_MAX_COST_USD=0.50 \\
        uv run pytest tests/test_smoke_books.py

Until a recording exists, the replay tests skip. Any pipeline change that alters what
jevex asks Jev makes the recording stale; those tests then xfail locally and fail in CI
(see ``jevex.testing.stale_recording``), unless the PR is labelled ``cassette-stale-ok``.

The other tests always run: the rating cleaner, and the whole pipeline over the saved
pages with Jev answering by pattern (any price is "the price") rather than from the
expected record. See ``fixtures/books/README.md`` for where the pages came from.
"""

import json
import os
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import pytest

from jevex import Document, ExtractionResult, Extractor
from jevex.examples.books import Book, StarRatingCleaner, books_pipeline
from jevex.jev import Choice, JevClient, JevResponse, JSONContent, Question
from jevex.layout_html import HtmlLayoutParser
from jevex.testing import (
    RECORD_ENV,
    STALE_OK_ENV,
    Cassette,
    CassetteMissError,
    FakeJev,
    cassette,
    request_key,
    stale_recording,
)

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
            "JEVEX_JEV_MAX_COST_USD=0.50 uv run pytest tests/test_smoke_books.py"
        )
    return JevClient(cassette(CASSETTE))


async def extract_recorded(name: str, jev: JevClient) -> ExtractionResult:
    """The page through the books pipeline, ended by ``stale_recording`` when it asks Jev
    something the recording doesn't have.

    ``extract`` doesn't raise the cassette's miss: it fails the document and reports the
    miss among the result's errors, so that's where to look.
    """
    async with Extractor([Book], jev=jev, pipeline=books_pipeline()) as ex:
        result = await ex.extract(page(name))
    missed = [e for e in result.errors if e.type == CassetteMissError.__name__]
    if missed:
        stale_recording(missed[0].message)
    return result


@pytest.mark.parametrize("name", list(PAGES))
async def test_books_page_with_recorded_jev(name: str, recorded_jev: JevClient) -> None:
    result = await extract_recorded(name, recorded_jev)
    found = result.one(Book).record.model_dump(exclude_unset=True)
    assert found == PAGES[name].model_dump()


class KeyLog:
    """Replays a cassette and notes each request's key."""

    def __init__(self, inner: Cassette) -> None:
        self.inner = inner
        self.keys: list[str] = []

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        self.keys.append(request_key(state, questions))
        return await self.inner.system_one(state, questions)


@pytest.mark.parametrize(
    ("env", "outcome"),
    [
        ({}, pytest.xfail.Exception),
        ({"CI": "true"}, pytest.fail.Exception),
        ({"CI": "true", STALE_OK_ENV: "1"}, pytest.xfail.Exception),
    ],
    ids=["local", "ci", "ci-stale-ok"],
)
async def test_a_request_missing_from_the_recording_is_a_stale_recording(
    env: dict[str, str],
    outcome: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not CASSETTE.exists():
        pytest.skip("no Jev recording yet")
    name = next(iter(PAGES))
    log = KeyLog(Cassette(CASSETTE))
    await extract_recorded(name, JevClient(log))
    # A copy without the page's last request, as if the pipeline now asked it differently.
    original = CASSETTE.read_bytes()
    entries = json.loads(original)
    del entries[log.keys[-1]]
    stale = tmp_path / "jev-cassette.json"
    stale.write_text(json.dumps(entries))
    for var in ("CI", STALE_OK_ENV):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    with pytest.raises(
        outcome, match=f"the recording is stale: no recording for request {log.keys[-1]}"
    ):
        await extract_recorded(name, JevClient(Cassette(stale)))
    assert CASSETTE.read_bytes() == original


# --- the cleaner, and the pipeline with pattern-based answers (always run) ---------


async def test_star_ratings_are_written_out_as_text() -> None:
    doc = StarRatingCleaner().clean(page("a-light-in-the-attic_1000"))
    root = await HtmlLayoutParser().parse(doc)
    texts = [c.text for c in root.walk()]
    assert "Rating: Three out of five stars" in [t.strip() for t in texts]
    assert "A Light in the Attic" in texts


async def test_the_cleaner_keeps_a_legacy_encoding() -> None:
    html = '<meta charset="windows-1252"><p class="star-rating Two"></p><p>Price: £51.77 café</p>'
    doc = Document.from_bytes(html.encode("cp1252"), content_type="text/html")
    cleaned = StarRatingCleaner().clean(doc)
    root = await HtmlLayoutParser().parse(cleaned)
    texts = [c.text.strip() for c in root.walk() if c.text.strip()]
    assert texts == ["Rating: Two out of five stars", "Price: £51.77 café"]


def test_the_cleaner_keeps_undecodable_bytes() -> None:
    raw = b'<p class="star-rating Four"></p><p>odd \xff byte</p>'
    cleaned = StarRatingCleaner().clean(Document.from_bytes(raw, content_type="text/html"))
    assert b"Rating: Four out of five stars" in cleaned.content
    assert b"odd \xff byte" in cleaned.content


def test_the_cleaner_leaves_other_documents_alone() -> None:
    plain = Document.from_bytes(b"star-rating Three", content_type="text/plain")
    assert StarRatingCleaner().clean(plain) == plain
    html = b"<p class='no-star-rating Four'>x</p><p>No rating here</p>"
    doc = Document.from_bytes(html, content_type="text/html")
    assert b"Rating:" not in StarRatingCleaner().clean(doc).content


def test_the_cleaner_is_linear_on_unclosed_tags() -> None:
    import time

    doc = Document.from_bytes(b'<a class="q ' * 20_000, content_type="text/html")
    start = time.perf_counter()
    StarRatingCleaner().clean(doc)
    assert time.perf_counter() - start < 2


RATINGS = ("One", "Two", "Three", "Four", "Five")


def page_titles(name: str) -> list[str]:
    """Every book title on the page (product and "recently viewed"), from its markup."""
    html = (FIXTURES / f"{name}.html").read_text(encoding="utf-8")
    titles = re.findall(r"<h1>([^<]+)</h1>", html) + re.findall(r'alt="([^"]+)"', html)
    return list(dict.fromkeys(titles))


def pattern_jev(name: str) -> FakeJev:
    """A reader that answers by pattern, not from the expected record.

    Every section with a price, a rating or a stock line passes the gate, including the
    "Products you recently viewed" block, which lists other books' titles, prices,
    ratings and stock. Any price is categorised as the price, any title as the title, and
    so on, so the pipeline itself has to keep the product's values ahead of the others.
    """
    titles = page_titles(name)

    def longest(q: Choice) -> str:
        return max((o for o in q.options if o != "none"), key=len, default="none")

    fake = (
        FakeJev(strict=True)
        .noul("Does this document describe", p=0.97)
        .noul("Does this section", p=0.05)
        .noul("Does this section", p=0.9, state=re.compile(r"£|Rating:|In stock"))
        .choice("Which detail", "none")
        .choice("Which detail", "price", state=re.compile(r'"statement": "£'))
        .choice("Which detail", "in_stock", state=re.compile(r'"statement": "In stock"'))
        # "In stock (22 available)" states two fields: the count, and that it's in stock.
        .choice(
            "Which detail",
            "stock_count",
            confidence=0.55,
            probabilities={"in_stock": 0.4},
            state="available)",
        )
        .choice(
            "Which detail",
            "title",
            state=re.compile(f'"statement": "(?:{"|".join(map(re.escape, titles))})"'),
        )
        .choice(re.compile("(?i)which of these is the (title|price|number)"), longest)
        .noul(re.compile("(?i)does the statement say the book is in stock"), p=0.95)
    )
    for rating in RATINGS:
        fake.choice("Which detail", "rating", state=f"Rating: {rating}")
        fake.choice(re.compile("(?i)what is the star rating"), rating, state=f"Rating: {rating}")
    return fake


@pytest.mark.parametrize("name", list(PAGES))
async def test_books_page_with_pattern_answers(name: str) -> None:
    expected = PAGES[name]
    fake = pattern_jev(name)
    async with Extractor([Book], jev=fake.client(), pipeline=books_pipeline()) as ex:
        result = await ex.extract(page(name))
    assert result.one(Book).strict() == expected
