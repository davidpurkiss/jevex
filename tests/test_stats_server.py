import asyncio
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jevex.stats import Stats, from_replay_csv, replay_loader, stats_server, store_loader
from jevex.stats.server import redact
from jevex.store import DocumentStat, SQLiteStore, StoreError, ValueStat

T0 = datetime(2026, 9, 30, 8, 0, tzinfo=UTC)

CSV = """\
batch,documents,size,waves,accuracy,llm_calls_per_document,jev_cost_per_document,\
llm_cost_per_document,errors,generators,values_llm
1,2,2,1,1.0,2.0,0.001,0.02,0,1,3
2,4,2,2,0.5,0.0,0.001,0.0,0,3,0
"""


def sqlite_store(path: Path, documents: int) -> str:
    async def fill() -> None:
        store = SQLiteStore(path)
        for i in range(documents):
            value = ValueStat(field="Car.model", method="llm", confidence=0.5, value="Golf")
            await store.record_document(
                DocumentStat(id=f"d{i}", at=T0 + timedelta(minutes=i), llm_calls=1, values=[value])
            )
        await store.aclose()

    asyncio.run(fill())
    return f"sqlite:///{path}"


def test_a_store_loader_reads_the_store_again_each_time(tmp_path: Path) -> None:
    url = sqlite_store(tmp_path / "s.db", 2)
    load = store_loader(url, budget_usd=3.0)
    first = load()
    assert (first.source, first.documents, first.budget_usd) == (url, 2, 3.0)
    sqlite_store(tmp_path / "s.db", 3)  # another process adds to it
    assert load().documents == 3


def test_a_store_loader_fails_for_an_unreadable_store(tmp_path: Path) -> None:
    with pytest.raises(StoreError):
        store_loader(tmp_path / "missing" / "s.db")()


def test_a_replay_loader_reads_its_csv(tmp_path: Path) -> None:
    path = tmp_path / "curve.csv"
    path.write_text(CSV)
    stats = replay_loader(path, budget_usd=1.0)()
    assert (stats.kind, stats.source, stats.documents, stats.budget_usd) == (
        "replay",
        str(path),
        4,
        1.0,
    )
    with pytest.raises(FileNotFoundError):
        replay_loader(tmp_path / "nope.csv")()


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("postgresql://jx:secret@db.example:5432/jevex", "postgresql://jx@db.example:5432/jevex"),
        ("postgresql://db/jevex", "postgresql://db/jevex"),
        ("sqlite:///jevex.db", "sqlite:///jevex.db"),
    ],
)
def test_passwords_never_reach_the_page(url: str, shown: str) -> None:
    assert redact(url) == shown


class Source:
    """A loader whose stats (or failure) a test controls."""

    def __init__(self) -> None:
        self.stats: Stats = from_replay_csv(CSV, source="curve.csv")
        self.error: Exception | None = None
        self.loads = 0

    def __call__(self) -> Stats:
        self.loads += 1
        if self.error is not None:
            raise self.error
        return self.stats


@pytest.fixture
def source() -> Source:
    return Source()


@pytest.fixture
def served(source: Source) -> Iterator[str]:
    httpd = stats_server(source, port=0)
    thread = threading.Thread(target=httpd.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host!s}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def get(url: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url) as response:
            return response.status, response.headers["Content-Type"], response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers["Content-Type"], exc.read().decode()


pytestmark = [pytest.mark.enable_socket, pytest.mark.allow_hosts(["127.0.0.1"])]


def test_the_page_is_served_live(served: str, source: Source) -> None:
    status, kind, body = get(f"{served}/")  # redirected to /stats/
    assert (status, kind) == (200, "text/html; charset=utf-8")
    assert "replay: curve.csv" in body
    assert 'data-refresh="30"' in body
    get(f"{served}/stats/")
    assert source.loads == 2  # read again for each request


@pytest.mark.parametrize("view", ["summary", "learning", "mix", "cost", "generators", "all"])
def test_the_api_serves_each_view_as_json(served: str, view: str) -> None:
    status, kind, body = get(f"{served}/stats/api/{view}")
    assert (status, kind) == (200, "application/json; charset=utf-8")
    data = json.loads(body)
    if view == "summary":
        assert data["documents"] == 4
    if view == "all":
        assert data["learning"]["waves"] == [{"documents": 2, "wave": 2}]


def test_charts_are_served_as_standalone_svg(served: str) -> None:
    status, kind, body = get(f"{served}/stats/api/chart/learning.svg?animate=1")
    assert (status, kind) == (200, "image/svg+xml; charset=utf-8")
    assert body.startswith('<svg xmlns="http://www.w3.org/2000/svg" class="chart viz-root animate"')
    _, _, still = get(f"{served}/stats/api/chart/mix.svg?x=docs")
    assert 'class="chart viz-root"' in still


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/elsewhere", 404),
        ("/stats/api/nope", 404),
        ("/stats/api/chart/pie.svg", 404),
        ("/stats/api/chart/learning", 404),
        ("/stats/api/chart/learning.svg?x=sideways", 400),
        ("/stats/api/chart/learning.svg?x=time", 400),  # a replay has no times
    ],
)
def test_bad_requests(served: str, path: str, status: int) -> None:
    assert get(f"{served}{path}")[0] == status


def test_a_source_that_cant_be_read_is_a_500_with_the_reason(served: str, source: Source) -> None:
    source.error = StoreError("database is locked")
    status, _, body = get(f"{served}/stats/api/summary")
    assert status == 500
    assert "read stats: database is locked" in body
