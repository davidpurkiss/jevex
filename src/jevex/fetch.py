"""A small, polite fetcher for demos, tests and standalone jobs.

It's not a crawler: no JavaScript, no retries or proxies, no anti-bot handling. Real
crawls should use a crawler (Scrapy, your own) and hand jevex :class:`~jevex.Document`s.

robots.txt is honoured per RFC 9309, using ``protego`` (the matcher Scrapy uses): the
longest matching rule wins, ``Allow`` wins ties, and ``*``/``$`` wildcards work. A 4xx
robots.txt means "no rules"; a 5xx or an unreachable server means "assume everything is
disallowed". Redirects are followed by hand so every hop is checked against *its* host's
robots.txt and politeness delay.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self

import httpx2
from protego import Protego

from jevex.document import Document

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType

DEFAULT_USER_AGENT = "jevex (+https://github.com/davidpurkiss/jevex)"
DEFAULT_MAX_BYTES = 20 * 1024 * 1024
MAX_REDIRECTS = 10
_REDIRECTS = frozenset({301, 302, 303, 307, 308})
_ALLOW_ALL = Protego.parse("")
_DISALLOW_ALL = Protego.parse("User-agent: *\nDisallow: /\n")


class FetchError(Exception):
    """The document could not be fetched (bad URL, network error, non-2xx, too large)."""


class RobotsDisallowedError(FetchError):
    """robots.txt forbids fetching this URL for our user agent."""


class _Response:
    __slots__ = ("content", "headers", "status", "url")

    def __init__(self, status: int, url: str, headers: httpx2.Headers, content: bytes) -> None:
        self.status = status
        self.url = url
        self.headers = headers
        self.content = content


class SimpleFetcher:
    """Fetches URLs politely: robots.txt, a per-host delay, redirects checked hop by hop.

    - ``user_agent``: sent with every request and used for robots.txt matching.
    - ``delay``: the minimum number of seconds between requests to one host. A longer
      robots.txt ``Crawl-delay`` wins. Different hosts are fetched concurrently.
    - ``timeout``: applies only to the client the fetcher creates itself.
    - ``respect_robots``: turn robots.txt off only for sites you own or may test against.
    - ``max_bytes``: bodies are streamed and abandoned once larger than this.
    - ``client``: bring your own ``httpx2.AsyncClient`` (e.g. with a mock transport). The
      fetcher doesn't close it.
    - ``clock``/``sleep``: injectable for tests.
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
        self._client = client or httpx2.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._clock = clock
        self._sleep = sleep
        self._robots: dict[str, Protego] = {}
        self._robots_locks: dict[str, asyncio.Lock] = {}
        self._host_locks: dict[str, asyncio.Lock] = {}
        self._last_request: dict[str, float] = {}

    async def fetch(self, url: str) -> Document:
        """Fetch ``url`` (following redirects) and return it as a :class:`~jevex.Document`."""
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            origin = _origin(current)
            if self.respect_robots:
                robots = await self._robots_for(origin)
                if not robots.can_fetch(current, self.user_agent):
                    raise RobotsDisallowedError(
                        f"robots.txt disallows {current} for {self.user_agent!r}"
                    )
            response = await self._polite_get(origin, current, read_body=True)
            location = response.headers.get("location")
            if response.status in _REDIRECTS and location:
                current = _join(current, location)
                continue
            if not 200 <= response.status < 300:
                raise FetchError(f"GET {current} returned HTTP {response.status}")
            return Document.from_bytes(
                response.content,
                url=response.url,
                content_type=response.headers.get("content-type") or None,
                fetched_at=datetime.now(UTC),
                content_language=response.headers.get("content-language") or None,
            )
        raise FetchError(f"{url} redirected more than {MAX_REDIRECTS} times")

    async def _polite_get(
        self, origin: str, url: str, *, read_body: bool, follow_redirects: bool = False
    ) -> _Response:
        lock = self._host_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            wait = self._delay_for(origin) - (self._clock() - self._last_request.get(origin, -1e9))
            if wait > 0:
                await self._sleep(wait)
            try:
                return await self._get(url, read_body=read_body, follow_redirects=follow_redirects)
            finally:
                self._last_request[origin] = self._clock()

    async def _get(self, url: str, *, read_body: bool, follow_redirects: bool) -> _Response:
        try:
            async with self._client.stream(
                "GET",
                url,
                headers={"User-Agent": self.user_agent},
                follow_redirects=follow_redirects,
            ) as response:
                content = b""
                if read_body and not (response.status_code in _REDIRECTS and not follow_redirects):
                    content = await self._read_limited(response, url)
                return _Response(response.status_code, str(response.url), response.headers, content)
        except (httpx2.HTTPError, httpx2.InvalidURL) as exc:
            raise FetchError(f"GET {url} failed: {exc}") from exc

    async def _read_limited(self, response: httpx2.Response, url: str) -> bytes:
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            raise FetchError(f"{url} is {declared} bytes; the limit is {self.max_bytes}")
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > self.max_bytes:
                raise FetchError(f"{url} is over the {self.max_bytes}-byte limit")
            chunks.append(chunk)
        return b"".join(chunks)

    def _delay_for(self, origin: str) -> float:
        robots = self._robots.get(origin)
        crawl_delay = robots.crawl_delay(self.user_agent) if robots else None
        return max(self.delay, float(crawl_delay or 0))

    async def _robots_for(self, origin: str) -> Protego:
        lock = self._robots_locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self._robots:
                self._robots[origin] = await self._load_robots(origin)
            return self._robots[origin]

    async def _load_robots(self, origin: str) -> Protego:
        try:
            # RFC 9309 §2.3.1.2: follow redirects for robots.txt itself.
            response = await self._polite_get(
                origin, f"{origin}/robots.txt", read_body=True, follow_redirects=True
            )
        except FetchError:
            return _DISALLOW_ALL  # unreachable: assume disallowed (RFC 9309 §2.3.1.4)
        if response.status >= 500:
            return _DISALLOW_ALL
        if response.status >= 400:
            return _ALLOW_ALL  # "unavailable": no restrictions (RFC 9309 §2.3.1.3)
        return Protego.parse(response.content.decode("utf-8", errors="replace"))

    async def aclose(self) -> None:
        """Close the HTTP client, if the fetcher created it."""
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
    """``scheme://host[:port]``, lower-cased, without userinfo or a default port."""
    try:
        parsed = httpx2.URL(url)
    except (httpx2.InvalidURL, TypeError, ValueError) as exc:
        raise FetchError(f"not an http(s) URL: {url!r} ({exc})") from exc
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise FetchError(f"not an http(s) URL: {url!r}")
    port = f":{parsed.port}" if parsed.port is not None else ""
    host = parsed.host.lower()
    if ":" in host:  # IPv6 literal: keep the brackets so the origin is still a valid URL
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}{port}"


def _join(base: str, location: str) -> str:
    try:
        return str(httpx2.URL(base).join(location))
    except (httpx2.InvalidURL, TypeError, ValueError) as exc:
        raise FetchError(f"bad redirect from {base} to {location!r}") from exc
