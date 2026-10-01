"""``jevex stats``: the stats page and its JSON API over HTTP, with the standard library
only (spec: *Stats UI › Technology*).

Routes: ``/stats/`` (the page, reloading itself), ``/stats/api/<view>`` (JSON, one of
:data:`~jevex.stats.data.VIEWS`) and ``/stats/api/chart/<view>.svg?x=docs|time&animate=1``
(a standalone chart, as ``jevex stats export`` writes it). Every request reads the source
again, so the page follows a store other processes are writing to.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit, urlunsplit

from jevex.stats.charts import CHART_VIEWS, chart_svg
from jevex.stats.data import DOCUMENT_LIMIT, VIEWS, from_replay_csv, from_store, to_json
from jevex.stats.page import render_page
from jevex.store import StoreError, open_store

if TYPE_CHECKING:
    from datetime import datetime

    from jevex.stats.data import Stats

type Loader = Callable[[], Stats]
"""Reads a source's stats afresh (called once per request)."""

DEFAULT_PORT = 8765


def redact(url: str) -> str:
    """``url`` without a password, to show as the page's source."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.hostname or ""
    user = f"{parts.username}@" if parts.username else ""
    netloc = f"{user}{host}" + (f":{parts.port}" if parts.port else "")
    return urlunsplit(parts._replace(netloc=netloc))


def store_loader(
    url: str | Path,
    *,
    since: datetime | None = None,
    limit: int | None = DOCUMENT_LIMIT,
    budget_usd: float | None = None,
) -> Loader:
    """Reads a store's stats (:func:`~jevex.stats.data.from_store`), opening and closing
    the store each time. Raises :class:`~jevex.store.StoreError` if it can't be read."""
    source = redact(str(url))

    async def read() -> Stats:
        store = await asyncio.to_thread(open_store, url)
        try:
            return await from_store(
                store, source=source, since=since, limit=limit, budget_usd=budget_usd
            )
        finally:
            await store.aclose()

    return lambda: asyncio.run(read())


def replay_loader(path: str | Path, *, budget_usd: float | None = None) -> Loader:
    """Reads a replay's CSV (:func:`~jevex.stats.data.from_replay_csv`). Raises ``OSError``
    if it can't be read and ``ValueError`` if it isn't a replay's CSV."""

    def read() -> Stats:
        stats = from_replay_csv(Path(path).read_text(encoding="utf-8"), source=str(path))
        stats.budget_usd = budget_usd
        return stats

    return read


def stats_server(
    load: Loader,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    *,
    title: str = "jevex · stats",
) -> ThreadingHTTPServer:
    """An HTTP server for the stats of ``load``, bound but not yet serving: call
    ``serve_forever()`` (and ``server_close()`` when done). ``port=0`` picks a free port.
    A source that can't be read answers 500 with the reason. Raises ``OSError`` if the
    address can't be bound."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlsplit(self.path)
            path = url.path
            if path in ("/", "/stats"):
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/stats/")
                self.end_headers()
                return
            if path != "/stats/" and not path.startswith("/stats/api/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            name = path.removeprefix("/stats/api/")
            if path.startswith("/stats/api/chart/"):
                view = name.removeprefix("chart/").removesuffix(".svg")
                if view not in CHART_VIEWS or not name.endswith(".svg"):
                    self.send_error(HTTPStatus.NOT_FOUND, f"no chart {view!r}")
                    return
            elif path != "/stats/" and name not in VIEWS:
                self.send_error(HTTPStatus.NOT_FOUND, f"no view {name!r}")
                return
            try:
                stats = load()
            except (StoreError, OSError, ValueError) as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"can't read stats: {exc}")
                return
            if path == "/stats/":
                self._send(render_page(stats, title=title, live=True), "text/html")
            elif path.startswith("/stats/api/chart/"):
                query = parse_qs(url.query)
                axis = query.get("x", [stats.default_axis()])[0]
                if axis not in ("docs", "time"):
                    self.send_error(HTTPStatus.BAD_REQUEST, "x must be docs or time")
                    return
                try:
                    svg = chart_svg(
                        stats,
                        name.removeprefix("chart/").removesuffix(".svg"),
                        "time" if axis == "time" else "docs",
                        standalone=True,
                        animate=query.get("animate", ["0"])[0] == "1",
                    )
                except ValueError as exc:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._send(svg, "image/svg+xml")
            else:
                self._send(json.dumps(to_json(stats, name), ensure_ascii=False), "application/json")

        def _send(self, body: str, content_type: str) -> None:
            data = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)
