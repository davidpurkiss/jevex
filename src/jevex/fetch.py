"""A small, polite fetcher for demos, tests and standalone jobs.

It's not a crawler: no JavaScript, no retries or proxies, no anti-bot handling. Real
crawls should use a crawler (Scrapy, your own) and hand jevex :class:`~jevex.Document`s.

Robots.txt follows RFC 9309: a 4xx response means "no rules, crawl freely", while a 5xx
response or an unreachable server means "assume everything is disallowed".
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx2

from jevex.document import Document

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType

DEFAULT_USER_AGENT = "jevex (+https://github.com/davidpurkiss/jevex)"
DEFAULT_MAX_BYTES = 20 * 1024 * 1024
_DISALLOW_ALL = ["User-agent: *", "Disallow: /"]


class FetchError(Exception):
    """The document could not be fetched (network error, non-2xx status, too large)."""


class RobotsDisallowedError(FetchError):
    """robots.txt forbids fetching this URL for our user agent."""


class SimpleFetcher:
    """Fetches one URL at a time per host, honouring robots.txt and a politeness delay.

    ``delay`` is the minimum number of seconds between requests to the same host. A
    robots.txt ``Crawl-delay`` longer than that wins. Requests to different hosts run
    concurrently.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_USER_AGENT,
        delay: float = 1.0,
        timeout: float = 20.0,
        respect_robots: bool = True,
        max_bytes: int = DEFAULT_MAX_BYTES,
        client: httpx2.AsyncClient | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.user_agent = user_agent
        self.delay = delay
        self.respect_robots = respect_robots
        self.max_bytes = max_bytes
        self._client = client or httpx2.AsyncClient(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": user_agent}
        )
        self._owns_client = client is None
        self._clock = clock
        self._sleep = sleep
        self._robots: dict[str, RobotFileParser] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    async def fetch(self, url: str) -> Document:
        """Fetch ``url`` and return it as a :class:`~jevex.Document`."""
        origin = _origin(url)
        if self.respect_robots:
            robots = await self._robots_for(origin)
            if not robots.can_fetch(self.user_agent, url):
                raise RobotsDisallowedError(f"robots.txt disallows {url} for {self.user_agent!r}")
        response = await self._polite_get(origin, url)
        if not response.is_success:
            raise FetchError(f"GET {url} returned HTTP {response.status_code}")
        content = response.content
        if len(content) > self.max_bytes:
            raise FetchError(f"{url} is {len(content)} bytes; the limit is {self.max_bytes}")
        return Document.from_bytes(
            content,
            url=str(response.url),
            content_type=response.headers.get("content-type") or None,
            fetched_at=datetime.now(UTC),
        )

    async def _polite_get(self, origin: str, url: str) -> httpx2.Response:
        lock = self._host_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            wait = self._delay_for(origin) - (self._clock() - self._last_request.get(origin, -1e9))
            if wait > 0:
                await self._sleep(wait)
            try:
                return await self._client.get(url)
            except httpx2.HTTPError as exc:
                raise FetchError(f"GET {url} failed: {exc}") from exc
            finally:
                self._last_request[origin] = self._clock()

    def _delay_for(self, origin: str) -> float:
        robots = self._robots.get(origin)
        crawl_delay = robots.crawl_delay(self.user_agent) if robots else None
        return max(self.delay, float(crawl_delay or 0))

    async def _robots_for(self, origin: str) -> RobotFileParser:
        lock = self._robots_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self._robots:
                self._robots[origin] = await self._load_robots(origin)
            return self._robots[origin]

    async def _load_robots(self, origin: str) -> RobotFileParser:
        parser = RobotFileParser(f"{origin}/robots.txt")
        try:
            response = await self._polite_get(origin, f"{origin}/robots.txt")
        except FetchError:
            parser.parse(_DISALLOW_ALL)  # unreachable: assume disallowed (RFC 9309 §2.3.1.4)
            return parser
        if response.status_code >= 500:
            parser.parse(_DISALLOW_ALL)
        elif response.status_code >= 400:
            parser.parse([])  # "unavailable": no restrictions (RFC 9309 §2.3.1.3)
        else:
            parser.parse(response.text.splitlines())
        return parser

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def _origin(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise FetchError(f"not an http(s) URL: {url!r}")
    return f"{parts.scheme}://{parts.netloc}"
