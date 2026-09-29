from collections.abc import Callable

import httpx2
import pytest

from jevex.fetch import FetchError, RobotsDisallowedError, SimpleFetcher

type Handler = Callable[[httpx2.Request], httpx2.Response]

ROBOTS = "User-agent: *\nDisallow: /private/\nCrawl-delay: 5\n"


class FakeTime:
    """A controllable clock; sleep() advances it and records how long it slept."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 3))
        self.now += seconds


def site(robots: httpx2.Response | None = None, seen: list[str] | None = None) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(f"{request.url.host}{request.url.path}")
        if request.url.path == "/robots.txt":
            return robots or httpx2.Response(200, text=ROBOTS)
        if request.url.path == "/moved":
            return httpx2.Response(301, headers={"location": "/page"})
        if request.url.path == "/missing":
            return httpx2.Response(404, text="nope")
        if request.url.path == "/big":
            return httpx2.Response(200, content=b"x" * 2048)
        return httpx2.Response(
            200,
            content=b"<html><body>Golf</body></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    return handler


def fetcher(handler: Handler, fake: FakeTime | None = None, **kwargs: object) -> SimpleFetcher:
    fake = fake or FakeTime()
    client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        follow_redirects=True,
        headers={"User-Agent": "jevex-test"},
    )
    return SimpleFetcher(
        user_agent="jevex-test",
        client=client,
        clock=fake.clock,
        sleep=fake.sleep,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )


async def test_fetch_returns_a_document() -> None:
    doc = await fetcher(site()).fetch("https://example.com/page")
    assert doc.content == b"<html><body>Golf</body></html>"
    assert doc.content_type == "text/html"
    assert doc.url == "https://example.com/page"
    assert doc.fetched_at is not None


async def test_redirects_are_followed_and_final_url_kept() -> None:
    doc = await fetcher(site()).fetch("https://example.com/moved")
    assert doc.url == "https://example.com/page"


async def test_robots_disallow_is_honoured() -> None:
    with pytest.raises(RobotsDisallowedError, match="disallows"):
        await fetcher(site()).fetch("https://example.com/private/secret")


async def test_robots_can_be_ignored_explicitly() -> None:
    seen: list[str] = []
    f = fetcher(site(seen=seen), respect_robots=False)
    await f.fetch("https://example.com/private/secret")
    assert seen == ["example.com/private/secret"]


@pytest.mark.parametrize(
    ("robots", "allowed"),
    [
        (httpx2.Response(404), True),  # RFC 9309: 4xx means no restrictions
        (httpx2.Response(403), True),
        (httpx2.Response(503), False),  # 5xx: assume complete disallow
    ],
)
async def test_robots_status_codes(robots: httpx2.Response, allowed: bool) -> None:
    f = fetcher(site(robots=robots))
    if allowed:
        await f.fetch("https://example.com/private/x")
    else:
        with pytest.raises(RobotsDisallowedError):
            await f.fetch("https://example.com/page")


async def test_unreachable_robots_means_disallowed() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/robots.txt":
            raise httpx2.ConnectError("down", request=request)
        return httpx2.Response(200, text="ok")

    with pytest.raises(RobotsDisallowedError):
        await fetcher(handler).fetch("https://example.com/page")


async def test_robots_fetched_once_per_host() -> None:
    seen: list[str] = []
    f = fetcher(site(seen=seen))
    await f.fetch("https://example.com/a")
    await f.fetch("https://example.com/b")
    await f.fetch("https://other.example/c")
    assert seen.count("example.com/robots.txt") == 1
    assert seen.count("other.example/robots.txt") == 1


async def test_crawl_delay_spaces_requests_to_the_same_host() -> None:
    fake = FakeTime()
    f = fetcher(site(), fake, delay=1.0)
    await f.fetch("https://example.com/a")
    await f.fetch("https://example.com/b")
    # robots.txt (no wait) is read first, so its Crawl-delay of 5 s already applies to /a
    # and again to /b, overriding the 1 s default
    assert fake.slept == [5.0, 5.0]


async def test_default_delay_when_no_crawl_delay() -> None:
    fake = FakeTime()
    f = fetcher(site(robots=httpx2.Response(200, text="User-agent: *\nAllow: /\n")), fake, delay=2)
    await f.fetch("https://example.com/a")
    fake.now += 0.5  # half a second of other work
    await f.fetch("https://example.com/b")
    assert fake.slept == [2.0, 1.5]


async def test_non_2xx_raises() -> None:
    with pytest.raises(FetchError, match="HTTP 404"):
        await fetcher(site()).fetch("https://example.com/missing")


async def test_size_limit() -> None:
    with pytest.raises(FetchError, match="limit"):
        await fetcher(site(), max_bytes=1024).fetch("https://example.com/big")


async def test_network_errors_become_fetch_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(FetchError, match="failed"):
        await fetcher(handler, respect_robots=False).fetch("https://example.com/page")


@pytest.mark.parametrize("url", ["ftp://example.com/x", "not a url", "/relative"])
async def test_rejects_non_http_urls(url: str) -> None:
    with pytest.raises(FetchError, match="not an http"):
        await fetcher(site()).fetch(url)


async def test_user_agent_is_sent_and_context_manager_closes() -> None:
    agents: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        agents.append(request.headers["user-agent"])
        return httpx2.Response(200, text="ok")

    async with SimpleFetcher(
        user_agent="jevex-test",
        client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler), headers={"User-Agent": "jevex-test"}
        ),
        respect_robots=False,
        delay=0,
    ) as f:
        await f.fetch("https://example.com/")
    assert agents == ["jevex-test"]


def test_satisfies_fetcher_protocol() -> None:
    from jevex.interfaces import Fetcher

    assert isinstance(fetcher(site()), Fetcher)
