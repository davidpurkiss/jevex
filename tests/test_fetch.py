import asyncio
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from jevex import FetchError, RobotsDisallowedError, SimpleFetcher
from jevex.interfaces import Fetcher

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
        await asyncio.sleep(0)


def site(
    robots: str | httpx2.Response = ROBOTS,
    seen: list[str] | None = None,
    extra: dict[str, httpx2.Response] | None = None,
) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        key = f"{request.url.host}{request.url.path}"
        if seen is not None:
            seen.append(key)
        if extra and key in extra:
            return extra[key]
        if request.url.path == "/robots.txt":
            if isinstance(robots, httpx2.Response):
                return robots
            return httpx2.Response(200, text=robots)
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


def fetcher(handler: Handler, fake: FakeTime | None = None, **kwargs: Any) -> SimpleFetcher:
    fake = fake or FakeTime()
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return SimpleFetcher(
        user_agent="jevex-test", client=client, clock=fake.clock, sleep=fake.sleep, **kwargs
    )


# --- basics ----------------------------------------------------------------------------


async def test_fetch_returns_a_document() -> None:
    doc = await fetcher(site()).fetch("https://example.com/page")
    assert doc.content == b"<html><body>Golf</body></html>"
    assert doc.content_type == "text/html"
    assert doc.url == "https://example.com/page"
    assert doc.fetched_at is not None


async def test_non_2xx_raises() -> None:
    with pytest.raises(FetchError, match="HTTP 404"):
        await fetcher(site()).fetch("https://example.com/missing")


async def test_network_errors_become_fetch_errors() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(FetchError, match="failed"):
        await fetcher(handler, respect_robots=False).fetch("https://example.com/page")


@pytest.mark.parametrize(
    "url", ["ftp://example.com/x", "not a url", "/relative", "http://example.com:abc/"]
)
async def test_rejects_bad_urls(url: str) -> None:
    with pytest.raises(FetchError, match="not an http"):
        await fetcher(site()).fetch(url)


async def test_user_agent_is_sent_even_with_an_injected_client() -> None:
    agents: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        agents.append(request.headers["user-agent"])
        return httpx2.Response(200, text="User-agent: *\nAllow: /\n")

    async with fetcher(handler, delay=0) as f:
        await f.fetch("https://example.com/")
    assert agents == ["jevex-test", "jevex-test"]  # robots.txt, then the page


def test_satisfies_fetcher_protocol() -> None:
    assert isinstance(fetcher(site()), Fetcher)


# --- robots.txt (RFC 9309 via protego) -------------------------------------------------


async def test_robots_disallow_is_honoured() -> None:
    with pytest.raises(RobotsDisallowedError, match="disallows"):
        await fetcher(site()).fetch("https://example.com/private/secret")


async def test_longest_match_wins_over_file_order() -> None:
    robots = "User-agent: *\nAllow: /\nDisallow: /private/\n"
    with pytest.raises(RobotsDisallowedError):
        await fetcher(site(robots)).fetch("https://example.com/private/x")


async def test_wildcards_and_end_anchor() -> None:
    f = fetcher(site("User-agent: *\nDisallow: /*.pdf$\n"))
    with pytest.raises(RobotsDisallowedError):
        await f.fetch("https://example.com/brochure.pdf")
    await f.fetch("https://example.com/brochure.pdfx")


async def test_robots_can_be_ignored_explicitly() -> None:
    seen: list[str] = []
    await fetcher(site(seen=seen), respect_robots=False).fetch("https://example.com/private/x")
    assert seen == ["example.com/private/x"]


@pytest.mark.parametrize(
    ("robots", "allowed"),
    [
        (httpx2.Response(404), True),  # RFC 9309: 4xx means no restrictions
        (httpx2.Response(403), True),
        (httpx2.Response(503), False),  # 5xx: assume complete disallow
    ],
)
async def test_robots_status_codes(robots: httpx2.Response, allowed: bool) -> None:
    f = fetcher(site(robots))
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


async def test_robots_fetched_once_per_origin_ignoring_case_and_default_port() -> None:
    seen: list[str] = []
    f = fetcher(site(seen=seen))
    await asyncio.gather(
        f.fetch("https://example.com/a"),
        f.fetch("https://EXAMPLE.com/b"),
        f.fetch("https://example.com:443/c"),
    )
    await f.fetch("https://other.example/d")
    assert seen.count("example.com/robots.txt") == 1
    assert seen.count("other.example/robots.txt") == 1


# --- redirects -------------------------------------------------------------------------


async def test_redirects_are_followed_and_final_url_kept() -> None:
    doc = await fetcher(site()).fetch("https://example.com/moved")
    assert doc.url == "https://example.com/page"


async def test_redirect_into_a_disallowed_path_is_blocked() -> None:
    go = httpx2.Response(302, headers={"location": "/private/secret"})
    seen: list[str] = []
    f = fetcher(site(seen=seen, extra={"example.com/go": go}))
    with pytest.raises(RobotsDisallowedError):
        await f.fetch("https://example.com/go")
    assert "example.com/private/secret" not in seen


async def test_cross_host_redirect_checks_the_other_hosts_robots() -> None:
    hop = httpx2.Response(302, headers={"location": "https://other.example/y"})
    blocked = httpx2.Response(200, text="User-agent: *\nDisallow: /\n")
    seen: list[str] = []
    f = fetcher(site(seen=seen, extra={"example.com/x": hop, "other.example/robots.txt": blocked}))
    with pytest.raises(RobotsDisallowedError, match=r"other\.example"):
        await f.fetch("https://example.com/x")
    assert "other.example/robots.txt" in seen
    assert "other.example/y" not in seen


async def test_too_many_redirects() -> None:
    loop = httpx2.Response(302, headers={"location": "/loop"})
    f = fetcher(site(extra={"example.com/loop": loop}), delay=0)
    with pytest.raises(FetchError, match="redirected more than"):
        await f.fetch("https://example.com/loop")


# --- politeness ------------------------------------------------------------------------


async def test_crawl_delay_spaces_requests_to_the_same_host() -> None:
    fake = FakeTime()
    f = fetcher(site(), fake, delay=1.0)
    await f.fetch("https://example.com/a")
    await f.fetch("https://example.com/b")
    # robots.txt (no wait) is read first, so its Crawl-delay of 5 s already applies to /a
    # and again to /b, overriding the 1 s default
    assert fake.slept == [5.0, 5.0]


async def test_fractional_crawl_delay() -> None:
    fake = FakeTime()
    f = fetcher(site("User-agent: *\nCrawl-delay: 1.5\n"), fake, delay=0.1)
    await f.fetch("https://example.com/a")
    assert fake.slept == [1.5]


async def test_default_delay_when_no_crawl_delay() -> None:
    fake = FakeTime()
    f = fetcher(site("User-agent: *\nAllow: /\n"), fake, delay=2)
    await f.fetch("https://example.com/a")
    fake.now += 0.5  # half a second of other work
    await f.fetch("https://example.com/b")
    assert fake.slept == [2.0, 1.5]


async def test_concurrent_fetches_to_one_host_are_serialised() -> None:
    fake = FakeTime()
    order: list[str] = []
    f = fetcher(site("User-agent: *\nAllow: /\n", seen=order), fake, delay=2)
    await asyncio.gather(*(f.fetch(f"https://example.com/{i}") for i in range(3)))
    assert fake.slept == [2.0, 2.0, 2.0]
    assert order[0] == "example.com/robots.txt"
    assert sorted(order[1:]) == ["example.com/0", "example.com/1", "example.com/2"]


# --- size limit ------------------------------------------------------------------------


async def test_size_limit_from_the_body() -> None:
    with pytest.raises(FetchError, match="limit"):
        await fetcher(site(), max_bytes=1024).fetch("https://example.com/big")


async def test_size_limit_from_content_length_rejects_before_reading() -> None:
    declared = httpx2.Response(200, headers={"content-length": "999999999"}, content=b"x")
    f = fetcher(site(extra={"example.com/huge": declared}), max_bytes=1024)
    with pytest.raises(FetchError, match="999999999 bytes"):
        await f.fetch("https://example.com/huge")
