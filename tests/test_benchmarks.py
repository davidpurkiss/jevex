import hashlib
import json
import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx2
import pytest

import jevex
from jevex.baseline import corpus_digest
from jevex.benchmarks import (
    BOOKS_URL,
    BenchmarkConfig,
    BenchmarkConfigError,
    BookPageError,
    CorpusLock,
    CorpusLockError,
    CorpusSpec,
    LockCheck,
    PinnedModel,
    book_values,
    books_corpus,
    bootstrap_interval,
    catalogue_page,
    check_lock,
    lock_corpus,
    verify_lock,
)
from jevex.eval import load_corpus
from jevex.examples.books import Book
from jevex.fetch import RobotsDisallowedError, SimpleFetcher
from jevex.llm import gemini_flash_3x_price
from jevex.testsite import build

ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "benchmarks" / "config.yaml"
BOOK_FIXTURES = Path(__file__).parent / "fixtures" / "books"

# The saved real pages (tests/fixtures/books) and what they say.
REAL_BOOKS = {
    "a-light-in-the-attic_1000": {
        "title": "A Light in the Attic",
        "price": "51.77",
        "in_stock": True,
        "stock_count": 22,
        "rating": "Three",
    },
    "tipping-the-velvet_999": {
        "title": "Tipping the Velvet",
        "price": "53.74",
        "in_stock": True,
        "stock_count": 20,
        "rating": "One",
    },
    # Its "recently viewed" block lists the other two books, with their own ratings.
    "sapiens-a-brief-history-of-humankind_996": {
        "title": "Sapiens: A Brief History of Humankind",
        "price": "54.23",
        "in_stock": True,
        "stock_count": 20,
        "rating": "Five",
    },
}


def product_page(title: str, price: str, availability: str, rating: str) -> str:
    """A product page shaped like books.toscrape.com's, for books the fixtures don't have."""
    return f"""<html><body><article class="product_page">
<div class="col-sm-6 product_main"><h1>{title}</h1>
<p class="price_color">&pound;{price}</p>
<p class="instock availability"><i class="icon-ok"></i> {availability}</p>
<p class="star-rating {rating}"><i class="icon-star"></i></p></div>
<table class="table table-striped">
<tr><th>UPC</th><td>0000</td></tr>
<tr><th>Price (incl. tax)</th><td>&pound;{price}</td></tr>
<tr><th>Availability</th>
<td>{availability}</td></tr>
</table></article></body></html>"""


def catalogue(products: list[str], next_page: str | None) -> str:
    pods = "\n".join(
        f'<article class="product_pod"><a href="{p}/index.html"><img></a>'
        f'<p class="star-rating Two"></p><h3><a href="{p}/index.html" title="{p}">{p}</a></h3>'
        "</article>"
        for p in products
    )
    pager = f'<li class="next"><a href="{next_page}">next</a></li>' if next_page else ""
    return f'<html><body><ol class="row">{pods}</ol><ul class="pager">{pager}</ul></body></html>'


EXTRA_BOOKS = {
    "out-of-print_7": ("Out of Print", "10.00", "Out of stock", "Two"),
    "soumission_998": ("Soumission", "50.10", "In stock (20 available)", "One"),
}
CATALOGUE = {
    "/catalogue/page-1.html": catalogue(
        ["a-light-in-the-attic_1000", "tipping-the-velvet_999"], "page-2.html"
    ),
    "/catalogue/page-2.html": catalogue(["soumission_998", "out-of-print_7"], "page-3.html"),
    # A book listed again counts once.
    "/catalogue/page-3.html": catalogue(
        ["sapiens-a-brief-history-of-humankind_996", "soumission_998"], None
    ),
}
ORDER = [
    "a-light-in-the-attic_1000",
    "tipping-the-velvet_999",
    "soumission_998",
    "out-of-print_7",
    "sapiens-a-brief-history-of-humankind_996",
]


def book_site(
    *,
    robots: str = "",
    seen: list[str] | None = None,
    broken: str | None = None,
    pages: dict[str, str] = CATALOGUE,
) -> SimpleFetcher:
    def handler(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if seen is not None:
            seen.append(path)
        if path == "/robots.txt":
            return httpx2.Response(200, text=robots) if robots else httpx2.Response(404)
        if path in pages:
            return httpx2.Response(200, text=pages[path])
        slug = path.removeprefix("/catalogue/").removesuffix("/index.html")
        if slug == broken:
            return httpx2.Response(200, text="<html><body><h1>Gone</h1></body></html>")
        if slug in REAL_BOOKS:
            content = (BOOK_FIXTURES / f"{slug}.html").read_bytes()
            return httpx2.Response(200, content=content, headers={"content-type": "text/html"})
        if slug in EXTRA_BOOKS:
            return httpx2.Response(200, text=product_page(*EXTRA_BOOKS[slug]))
        return httpx2.Response(404)

    async def no_wait(_: float) -> None:
        return None

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return SimpleFetcher(client=client, sleep=no_wait)


# --- labelling product pages -----------------------------------------------------------


@pytest.mark.parametrize("slug", sorted(REAL_BOOKS))
def test_book_values_read_the_real_pages(slug: str) -> None:
    html = (BOOK_FIXTURES / f"{slug}.html").read_text(encoding="utf-8")
    values = book_values(html)
    assert values == REAL_BOOKS[slug]
    # The labels fit the schema the corpus is scored in.
    book = Book.model_validate(values)
    assert book.price == Decimal(REAL_BOOKS[slug]["price"])


def test_an_out_of_stock_book_has_none_available() -> None:
    values = book_values(product_page("Out of Print", "10.00", "Out of stock", "Two"))
    assert values == {
        "title": "Out of Print",
        "price": "10.00",
        "in_stock": False,
        "stock_count": 0,
        "rating": "Two",
    }


@pytest.mark.parametrize(
    ("html", "message"),
    [
        ("<p class='star-rating One'></p>", "no <h1> title"),
        (product_page("T", "1.00", "Out of stock", "Six"), "no star-rating after the title"),
        ("<h1>T</h1><p class='star-rating One'></p>", "no 'Price (incl. tax)' row"),
        (
            "<h1>T</h1><p class='star-rating One'></p>"
            "<table><tr><th>Price (incl. tax)</th><td>£1</td></tr></table>",
            "no 'Availability' row",
        ),
        (product_page("T", "1.00", "Back soon", "One"), "can't read availability 'Back soon'"),
    ],
)
def test_book_values_name_what_a_page_lacks(html: str, message: str) -> None:
    with pytest.raises(BookPageError, match=f"^{re.escape(message)}$"):
        book_values(html)


def test_a_rating_before_the_title_is_not_the_books() -> None:
    html = "<p class='star-rating Four'></p>" + product_page("T", "1.00", "Out of stock", "One")
    assert book_values(html)["rating"] == "One"


def test_catalogue_page_finds_products_and_the_next_page() -> None:
    url = "https://books.toscrape.com/catalogue/page-1.html"
    products, next_url = catalogue_page(CATALOGUE["/catalogue/page-1.html"], url)
    assert products == [
        "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
        "https://books.toscrape.com/catalogue/tipping-the-velvet_999/index.html",
    ]
    assert next_url == "https://books.toscrape.com/catalogue/page-2.html"
    assert catalogue_page(CATALOGUE["/catalogue/page-3.html"], url)[1] is None


# --- fetching the books corpus ---------------------------------------------------------


async def test_books_corpus_writes_a_labelled_locked_sample(tmp_path: Path) -> None:
    out = tmp_path / "books"
    seen: list[str] = []
    lock = await books_corpus(out, sample=3, seed=42, fetcher=book_site(seen=seen))

    truth = json.loads((out / "truth.json").read_text())
    assert {k: v for k, v in truth.items() if k != "pages"} == {
        "source": BOOKS_URL,
        "seed": 42,
        "sample": 3,
        "labels": "markup",
    }
    slugs = [Path(p["path"]).stem for p in truth["pages"]]
    # Seed 42's three, in catalogue order.
    assert slugs == [
        "a-light-in-the-attic_1000",
        "soumission_998",
        "sapiens-a-brief-history-of-humankind_996",
    ]
    first = truth["pages"][0]
    assert first["url"] == f"{BOOKS_URL}catalogue/a-light-in-the-attic_1000/index.html"
    assert first["schema"] == "Book"
    assert first["content_type"] == "text/html"
    soumission = EXTRA_BOOKS["soumission_998"]
    expected = {
        "a-light-in-the-attic_1000": REAL_BOOKS["a-light-in-the-attic_1000"],
        "soumission_998": book_values(product_page(*soumission)),
        "sapiens-a-brief-history-of-humankind_996": REAL_BOOKS[
            "sapiens-a-brief-history-of-humankind_996"
        ],
    }
    for page, slug in zip(truth["pages"], slugs, strict=True):
        assert page["records"] == [{"entity": "document", "values": expected[slug]}]
    # Pages are saved byte for byte, and only the sampled ones were fetched.
    for page, slug in zip(truth["pages"], slugs, strict=True):
        served = (
            (BOOK_FIXTURES / f"{slug}.html").read_bytes()
            if slug in REAL_BOOKS
            else product_page(*EXTRA_BOOKS[slug]).encode()
        )
        assert (out / page["path"]).read_bytes() == served
    fetched = [p.removeprefix("/catalogue/").removesuffix("/index.html") for p in seen]
    assert sorted(s for s in fetched if s in ORDER) == sorted(slugs)
    assert seen[0] == "/robots.txt"

    assert [i.schema for i in load_corpus(out)] == ["Book"] * 3
    assert lock.name == "books"
    assert lock.settings == {"source": BOOKS_URL, "seed": 42, "sample": 3, "labels": "markup"}
    assert sorted(lock.documents) == sorted(p["path"] for p in truth["pages"])
    assert lock.digest == corpus_digest(out)
    assert check_lock(out, lock).ok


async def test_the_same_seed_picks_the_same_books(tmp_path: Path) -> None:
    def picked(lock: CorpusLock) -> list[str]:
        return sorted(lock.documents)

    first = await books_corpus(tmp_path / "a", sample=2, seed=7, fetcher=book_site())
    again = await books_corpus(tmp_path / "b", sample=2, seed=7, fetcher=book_site())
    assert picked(first) == picked(again)
    assert first.digest == again.digest
    everything = await books_corpus(tmp_path / "c", sample=5, seed=1, fetcher=book_site())
    assert picked(everything) == sorted(f"pages/{s}.html" for s in ORDER)


async def test_books_corpus_honours_robots_txt(tmp_path: Path) -> None:
    fetcher = book_site(robots="User-agent: *\nDisallow: /catalogue/\n")
    with pytest.raises(RobotsDisallowedError):
        await books_corpus(tmp_path / "books", sample=1, fetcher=fetcher)
    assert not (tmp_path / "books").exists()


async def test_a_catalogue_that_loops_is_refused(tmp_path: Path) -> None:
    last = catalogue(["sapiens-a-brief-history-of-humankind_996"], "page-1.html")
    looping = CATALOGUE | {"/catalogue/page-3.html": last}
    stopped = r"stopped walking the catalogue at \S+/page-1\.html after 3 pages"
    with pytest.raises(ValueError, match=stopped):
        await books_corpus(tmp_path / "books", sample=1, fetcher=book_site(pages=looping))
    assert not (tmp_path / "books").exists()


async def test_books_corpus_needs_enough_books(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="the catalogue has 5 books, fewer than 6"):
        await books_corpus(tmp_path / "books", sample=6, fetcher=book_site())
    with pytest.raises(ValueError, match="sample must be at least 1, not 0"):
        await books_corpus(tmp_path / "books", sample=0, fetcher=book_site())
    assert not (tmp_path / "books").exists()


async def test_a_page_that_cannot_be_labelled_stops_the_corpus(tmp_path: Path) -> None:
    out = tmp_path / "books"
    with pytest.raises(BookPageError, match=r"soumission_998/index\.html: no star-rating"):
        await books_corpus(out, sample=5, fetcher=book_site(broken="soumission_998"))
    assert not out.exists()  # nothing is written until every page is labelled


async def test_books_corpus_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("mine")
    seen: list[str] = []
    with pytest.raises(ValueError, match="isn't an empty directory"):
        await books_corpus(tmp_path, sample=1, fetcher=book_site(seen=seen))
    assert seen == []
    assert (tmp_path / "keep.txt").read_text() == "mine"


async def test_a_fetcher_passed_in_stays_open(tmp_path: Path) -> None:
    fetcher = book_site()
    await books_corpus(tmp_path / "a", sample=1, fetcher=fetcher)
    await books_corpus(tmp_path / "b", sample=1, fetcher=fetcher)  # still usable
    await fetcher.aclose()


# --- locks -----------------------------------------------------------------------------


def write_corpus(root: Path, docs: dict[str, str], **settings: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    for name, text in docs.items():
        (root / f"{name}.html").write_text(f"<h1>{text}</h1>")
        pages.append(
            {"path": f"{name}.html", "schema": "Book", "records": [{"values": {"title": text}}]}
        )
    (root / "truth.json").write_text(json.dumps(settings | {"pages": pages}))
    return root


def test_a_lock_records_the_truth_and_every_document(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "c", {"a": "Dune", "b": "Emma"}, seed=3, digest="site")
    lock = lock_corpus(root, "mine", publish="aggregate")
    assert lock.name == "mine"
    assert lock.publish == "aggregate"
    assert lock.settings == {"seed": 3}  # not the pages, nor a test site's own digest
    assert sorted(lock.documents) == ["a.html", "b.html"]
    assert lock.digest == corpus_digest(root)
    assert check_lock(root, lock) == LockCheck(False, (), (), ())
    assert check_lock(root, lock).describe() == "matches"
    verify_lock(root, lock)

    path = tmp_path / "mine.lock"
    lock.write(path)
    assert CorpusLock.load(path) == lock
    assert path.read_text().endswith("}\n")


def test_check_lock_says_what_differs(tmp_path: Path) -> None:
    root = write_corpus(tmp_path / "c", {"a": "Dune", "b": "Emma", "c": "Ulysses"})
    lock = lock_corpus(root, "mine")
    (root / "a.html").write_text("<h1>Dune Messiah</h1>")
    (root / "b.html").unlink()
    truth = json.loads((root / "truth.json").read_text())
    truth["pages"].append({"path": "d.html", "schema": "Book", "records": []})
    (root / "truth.json").write_text(json.dumps(truth))

    check = check_lock(root, lock)
    assert check == LockCheck(True, ("a.html",), ("b.html",), ("d.html",))
    assert not check.ok
    assert check.describe() == (
        "truth.json changed\n1 changed: a.html\n1 missing: b.html\n1 not in the lock: d.html"
    )
    with pytest.raises(CorpusLockError, match=r"doesn't match the 'mine' lock:\ntruth\.json"):
        verify_lock(root, lock)


def test_describe_shortens_long_lists() -> None:
    check = LockCheck(False, tuple(f"{i}.html" for i in range(7)), (), ())
    assert check.describe() == "7 changed: 0.html, 1.html, 2.html, 3.html, 4.html and 2 more"


def test_locks_that_cannot_be_read(tmp_path: Path) -> None:
    with pytest.raises(CorpusLockError, match="can't read lock"):
        CorpusLock.load(tmp_path / "nope.lock")
    (tmp_path / "bad.lock").write_text('{"name": "x"}')
    with pytest.raises(CorpusLockError, match="isn't a corpus lock"):
        CorpusLock.load(tmp_path / "bad.lock")
    lock = lock_corpus(write_corpus(tmp_path / "c", {"a": "Dune"}), "mine")
    with pytest.raises(ValueError, match="isn't a corpus manifest"):
        check_lock(tmp_path / "empty", lock)
    (tmp_path / "c" / "a.html").unlink()
    with pytest.raises(ValueError, match=r"lists a\.html, which doesn't exist"):
        lock_corpus(tmp_path / "c", "mine")


def test_the_committed_test_site_lock_matches_a_fresh_build(tmp_path: Path) -> None:
    """The benchmark's synthetic corpus is rebuilt from its seed: the build must stay
    byte for byte what the lock says. Rebuild the lock only alongside a new results run:
    ``jevex testsite build --out X && jevex corpus lock X --name testsite --out
    benchmarks/corpora/testsite.lock``."""
    pytest.importorskip("PIL")  # the scanned and infographic pages
    spec = BenchmarkConfig.load(CONFIG).corpus("testsite")
    assert spec.seed is not None
    lock = CorpusLock.load(CONFIG.parent / spec.lock)
    build(spec.seed, tmp_path)
    check = check_lock(tmp_path, lock)
    assert check.ok, check.describe()
    assert lock.settings["seed"] == spec.seed


def test_the_committed_books_lock_is_the_configured_sample() -> None:
    """The books pages live outside the repo, so only the lock can be checked here: it must
    be the sample the config pins. Refetch it with ``jevex corpus books --out X --lock
    benchmarks/corpora/books.lock``."""
    spec = BenchmarkConfig.load(CONFIG).corpus("books")
    lock = CorpusLock.load(CONFIG.parent / spec.lock)
    assert lock.name == "books"
    assert lock.publish == "full"
    assert lock.settings == {
        "labels": "markup",
        "sample": spec.sample,
        "seed": spec.seed,
        "source": BOOKS_URL,
    }
    assert len(lock.documents) == spec.sample
    assert all(re.fullmatch(r"pages/[\w.-]+\.html", path) for path in lock.documents)


def test_the_committed_spec_sheets_lock_is_its_manifest_and_labels() -> None:
    """The spec sheets aren't in the repo either: what fetch.py downloads (the manifest's
    hashes) and the labels it copies must be what the lock checks."""
    spec = BenchmarkConfig.load(CONFIG).corpus("spec-sheets")
    lock = CorpusLock.load(CONFIG.parent / spec.lock)
    manifest = fetch_script().Manifest.model_validate_json(
        (SPEC_SHEETS / "manifest.json").read_bytes()
    )
    assert lock.name == "spec-sheets"
    assert lock.publish == spec.publish == "full"
    assert lock.documents == {d.file: d.sha256 for d in manifest.documents}
    truth = (SPEC_SHEETS / "truth.json").read_bytes()
    assert lock.truth == hashlib.sha256(truth).hexdigest()
    assert [p["path"] for p in json.loads(truth)["pages"]] == [d.file for d in manifest.documents]


# --- fetching a corpus from its manifest (benchmarks/corpora/fetch.py) -----------------

SPEC_SHEETS = ROOT / "benchmarks" / "corpora" / "spec-sheets"
PDF = b"%PDF-1.4\nprice list\n"
HTML = b"<html><body><h1>Spec</h1></body></html>"


def fetch_script() -> Any:
    """The fetch script as a module (it isn't part of the package)."""
    import importlib.util

    path = ROOT / "benchmarks" / "corpora" / "fetch.py"
    spec = importlib.util.spec_from_file_location("_bench_fetch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass looks itself up there
    spec.loader.exec_module(module)
    return module


def manifest_entry(file: str, path: str, content: bytes, kind: str = "pdf") -> dict[str, str]:
    return {
        "file": file,
        "url": f"https://maker.example{path}",
        "sha256": hashlib.sha256(content).hexdigest(),
        "captured": "2026-10-10",
        "kind": kind,
    }


def write_manifest(root: Path, *documents: dict[str, str], truth: str | None = "{}") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"purpose": "test", "documents": list(documents)}))
    if truth is not None:
        (root / "truth.json").write_text(truth)
    return manifest


def maker_site(
    seen: list[str] | None = None, robots: str = "User-agent: *\nDisallow: /private/\n"
) -> SimpleFetcher:
    files = {"/a.pdf": PDF, "/b.html": HTML, "/private/c.pdf": PDF}

    def handler(request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if seen is not None:
            seen.append(path)
        if path == "/robots.txt":
            return httpx2.Response(200, text=robots)
        if path in files:
            return httpx2.Response(200, content=files[path])
        return httpx2.Response(404)

    async def no_wait(_: float) -> None:
        return None

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return SimpleFetcher(client=client, sleep=no_wait)


def statuses(outcomes: list[Any]) -> dict[str, str]:
    return {o.file: o.status for o in outcomes}


async def test_fetch_downloads_each_document_and_copies_the_labels(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "src",
        manifest_entry("a.pdf", "/a.pdf", PDF),
        manifest_entry("b.html", "/b.html", HTML, kind="html"),
        truth='{"pages": []}',
    )
    out = tmp_path / "corpus"
    outcomes = await fetch_script().fetch_corpus(manifest, out, maker_site())
    assert statuses(outcomes) == {"a.pdf": "fetched", "b.html": "fetched", "truth.json": "copied"}
    assert (out / "a.pdf").read_bytes() == PDF
    assert (out / "b.html").read_bytes() == HTML
    assert (out / "truth.json").read_text() == '{"pages": []}'
    assert sorted(p.name for p in out.iterdir()) == ["a.pdf", "b.html", "truth.json"]


async def test_fetch_reports_each_failure_and_writes_nothing_for_it(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "src",
        manifest_entry("a.pdf", "/a.pdf", PDF),
        manifest_entry("changed.pdf", "/a.pdf", b"%PDF-1.4\nlast year's prices\n"),
        manifest_entry("dead.pdf", "/gone.pdf", PDF),
        manifest_entry("private.pdf", "/private/c.pdf", PDF),
    )
    out = tmp_path / "corpus"
    outcomes = await fetch_script().fetch_corpus(manifest, out, maker_site())
    assert statuses(outcomes) == {
        "a.pdf": "fetched",
        "changed.pdf": "failed",
        "dead.pdf": "failed",
        "private.pdf": "failed",
        "truth.json": "copied",
    }
    details = {o.file: o.describe() for o in outcomes}
    found = hashlib.sha256(PDF).hexdigest()
    assert details["changed.pdf"].startswith(
        f"failed   changed.pdf: https://maker.example/a.pdf has sha256 {found}, not "
    )
    assert details["changed.pdf"].endswith("; not written")
    assert details["dead.pdf"] == (
        "failed   dead.pdf: GET https://maker.example/gone.pdf returned HTTP 404"
    )
    assert details["private.pdf"] == (
        "failed   private.pdf: robots.txt disallows https://maker.example/private/c.pdf for "
        "'jevex (+https://github.com/davidpurkiss/jevex)'"
    )
    assert sorted(p.name for p in out.iterdir()) == ["a.pdf", "truth.json"]


async def test_fetch_keeps_what_is_there_and_never_overwrites(tmp_path: Path) -> None:
    manifest = write_manifest(
        tmp_path / "src",
        manifest_entry("a.pdf", "/a.pdf", PDF),
        manifest_entry("b.html", "/b.html", HTML, kind="html"),
        truth='{"pages": [1]}',
    )
    out = tmp_path / "corpus"
    out.mkdir()
    (out / "a.pdf").write_bytes(PDF)
    (out / "b.html").write_bytes(b"<html>edited</html>")
    (out / "truth.json").write_text('{"pages": []}')
    seen: list[str] = []
    outcomes = await fetch_script().fetch_corpus(manifest, out, maker_site(seen))
    assert statuses(outcomes) == {"a.pdf": "present", "b.html": "failed", "truth.json": "failed"}
    assert seen == []  # neither document was fetched again
    details = [o.describe() for o in outcomes]
    edited = hashlib.sha256(b"<html>edited</html>").hexdigest()
    assert details[1] == f"failed   b.html: already in {out} with sha256 {edited}; not overwritten"
    assert details[2] == (
        f"failed   truth.json: {out / 'truth.json'} differs from "
        f"{tmp_path / 'src' / 'truth.json'}; not overwritten"
    )
    assert (out / "b.html").read_bytes() == b"<html>edited</html>"
    assert (out / "truth.json").read_text() == '{"pages": []}'

    (out / "truth.json").write_text('{"pages": [1]}')
    rerun = await fetch_script().fetch_corpus(manifest, out, maker_site())
    assert statuses(rerun)["truth.json"] == "present"


async def test_fetch_works_before_the_labels_exist(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path / "src", manifest_entry("a.pdf", "/a.pdf", PDF), truth=None)
    out = tmp_path / "corpus"
    outcomes = await fetch_script().fetch_corpus(manifest, out, maker_site())
    assert statuses(outcomes) == {"a.pdf": "fetched", "truth.json": "missing"}
    truth = tmp_path / "src" / "truth.json"
    assert outcomes[1].describe() == (
        f"missing  truth.json: no labels next to the manifest yet ({truth})"
    )


def test_fetch_main_runs_an_empty_manifest_without_a_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path / "src")
    out = tmp_path / "corpus"
    assert fetch_script().main([str(manifest), "--out", str(out)]) == 0
    assert capsys.readouterr().out == "copied   truth.json\n1 copied\n"
    assert (out / "truth.json").read_text() == "{}"


def test_fetch_main_fails_when_a_document_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(tmp_path / "src", manifest_entry("a.pdf", "/a.pdf", PDF))
    out = tmp_path / "corpus"
    out.mkdir()
    (out / "a.pdf").write_bytes(b"something else")
    assert fetch_script().main([str(manifest), "--out", str(out)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "1 copied, 1 failed"


@pytest.mark.parametrize(
    ("documents", "message"),
    [
        ([manifest_entry("../a.pdf", "/a.pdf", PDF)], "String should match pattern"),
        ([manifest_entry("truth.json", "/a.pdf", PDF)], "used twice or for the labels: truth"),
        (
            [manifest_entry("a.pdf", "/a.pdf", PDF), manifest_entry("a.pdf", "/b.pdf", PDF)],
            r"used twice or for the labels: a\.pdf",
        ),
        ([manifest_entry("a.pdf", "/a.pdf", PDF, kind="docx")], "Input should be 'pdf' or"),
    ],
)
def test_fetch_main_refuses_a_bad_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    documents: list[dict[str, str]],
    message: str,
) -> None:
    manifest = write_manifest(tmp_path / "src", *documents)
    out = tmp_path / "corpus"
    assert fetch_script().main([str(manifest), "--out", str(out)]) == 1
    error = capsys.readouterr().err
    assert error.startswith(f"can't fetch {manifest}: ")
    assert re.search(message, error)
    assert not out.exists()


# --- the pinned config -----------------------------------------------------------------


def test_the_committed_config_pins_everything() -> None:
    config = BenchmarkConfig.load(CONFIG)
    assert config.seed == 42
    assert config.budget_usd == 25.0
    assert config.jev.model == "jev-1.13.0"
    assert config.models.baseline_fast.model == "claude-haiku-4-5-20251001"
    assert config.models.baseline_strong.model == "claude-opus-5-5"
    assert config.models.extraction == config.models.baseline_fast
    gemini = config.models.baseline_gemini
    assert gemini is not None
    assert gemini.spec == "gemini:gemini-3.8-flash"
    # Pinned at the list price on its price date, so it can't drift with jevex's table.
    assert gemini.prices() == {"gemini-3.8-flash": gemini_flash_3x_price(gemini.price_date)}
    assert [c.name for c in config.corpora] == [
        "testsite",
        "books",
        "spec-sheets",
    ]
    with pytest.raises(KeyError):
        config.corpus("nope")


def pinned(**overrides: Any) -> dict[str, Any]:
    return {
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "input_usd_per_mtok": 4.0,
        "output_usd_per_mtok": 20.0,
        "price_date": "2026-10-01",
    } | overrides


def config(**overrides: Any) -> dict[str, Any]:
    llm = pinned()
    return {
        "seed": 1,
        "concurrency": 2,
        "budget_usd": 5.0,
        "jev": pinned(provider="jev", model="jev-1.13.0", input_usd_per_mtok=0.042),
        "models": {
            "extraction": llm,
            "generator": llm,
            "baseline_fast": llm,
            "baseline_strong": llm,
        },
        "corpora": [{"name": "testsite", "kind": "testsite", "seed": 42, "lock": "t.lock"}],
    } | overrides


def test_a_pinned_model_costs_usage_at_its_prices() -> None:
    model = PinnedModel.model_validate(pinned())
    assert model.spec == "anthropic:claude-opus-5-5"
    assert math.isclose(model.cost(1_000_000, 500_000), 14.0)
    assert math.isclose(model.cost(250_000), 1.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"jev": pinned(model="jev-1.13.0")}, "jev must be a Jev model"),
        (
            {"jev": pinned(provider="jev", model="jev-latest")},
            "'jev-latest' is an alias; pin an exact version",
        ),
        (
            {"models": config()["models"] | {"generator": pinned(provider="jev")}},
            "models.generator must be an LLM",
        ),
        (
            {"corpora": [config()["corpora"][0], config()["corpora"][0]]},
            "corpus names must be unique",
        ),
        ({"corpora": []}, "at least 1 item"),
        ({"concurrency": 0}, "greater than or equal to 1"),
        ({"budget_usd": 0}, "greater than 0"),
        ({"extra": 1}, "Extra inputs are not permitted"),
    ],
)
def test_bad_configs_are_refused(tmp_path: Path, overrides: dict[str, Any], message: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config(**overrides)))  # JSON is YAML
    with pytest.raises(BenchmarkConfigError, match=message):
        BenchmarkConfig.load(path)


@pytest.mark.parametrize(
    ("corpus", "message"),
    [
        ({"kind": "testsite"}, "a testsite corpus needs seed"),
        ({"kind": "books", "seed": 1}, "a books corpus needs sample"),
        ({"kind": "directory"}, "exactly one of path and env"),
        ({"kind": "directory", "path": "x", "env": "X"}, "exactly one of path and env"),
        ({"kind": "testsite", "seed": 1, "path": "x"}, "only for directory corpora"),
        ({"kind": "books", "seed": 1, "sample": 2, "waves": "table"}, "only for testsite"),
        ({"kind": "books", "seed": 1, "sample": 0}, "greater than or equal to 1"),
        ({"kind": "directory", "path": "x", "name": "Bad Name"}, "should match pattern"),
    ],
)
def test_corpus_specs_must_fit_their_kind(corpus: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CorpusSpec.model_validate({"name": "c", "lock": "c.lock"} | corpus)


def test_configs_that_cannot_be_read(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkConfigError, match="can't read"):
        BenchmarkConfig.load(tmp_path / "nope.yaml")
    (tmp_path / "bad.yaml").write_text("seed: [unclosed")
    with pytest.raises(BenchmarkConfigError, match="isn't a benchmark config"):
        BenchmarkConfig.load(tmp_path / "bad.yaml")


# --- bootstrap intervals ---------------------------------------------------------------


def test_bootstrap_intervals_are_seeded() -> None:
    values = [0.0, 1.0, 1.0, 0.5, 0.75, 1.0, 0.25, 1.0]
    interval = bootstrap_interval(values)
    assert interval == bootstrap_interval(values)  # same seed, same interval
    assert interval.mean == pytest.approx(0.6875)
    assert 0.0 <= interval.low < interval.mean < interval.high <= 1.0
    wider = bootstrap_interval(values, confidence=0.99)
    assert wider.low <= interval.low
    assert wider.high >= interval.high
    spread = [float(i * i % 17) for i in range(30)]
    assert bootstrap_interval(spread, seed=1) != bootstrap_interval(spread, seed=2)


def test_a_constant_has_no_spread() -> None:
    interval = bootstrap_interval([2.0, 2.0, 2.0], samples=10)
    assert (interval.mean, interval.low, interval.high) == (2.0, 2.0, 2.0)


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        ([], {}, "no values to bootstrap"),
        ([1.0], {"samples": 0}, "samples must be at least 1"),
        ([1.0], {"confidence": 1.0}, "confidence must be between 0 and 1"),
    ],
)
def test_bootstrap_refuses_bad_input(
    values: list[float], kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_interval(values, **kwargs)


def test_the_public_names_are_exported() -> None:
    for name in ("BenchmarkConfig", "CorpusLock", "books_corpus", "bootstrap_interval"):
        assert name in jevex.__all__
        assert getattr(jevex, name) is not None
