"""Input documents."""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

HTML = "text/html"
PDF = "application/pdf"
OCTET_STREAM = "application/octet-stream"

LOCALE_TAG = r"^[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,8})*$"
"""What jevex accepts as a locale: a BCP 47 language tag (``de``, ``de-DE``, ``en_GB``)."""

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", PDF),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class Document(BaseModel):
    """A fetched document. jevex takes documents, not URLs, so any crawler can supply them.

    ``site`` names where the document came from when its URL doesn't say (a local file, or
    several hosts that are one site); it overrides the URL's host as :attr:`source`.

    ``content_language`` is the HTTP ``Content-Language`` header as the server sent it.
    ``locale`` is the caller's word on the page's locale; it overrides what the page and
    the header say (see :func:`jevex.locales.document_locale`).
    """

    # Bytes travel as base64 in JSON (e.g. the `jevex serve` API).
    model_config = ConfigDict(frozen=True, ser_json_bytes="base64", val_json_bytes="base64")

    content: bytes
    content_type: str
    url: str | None = None
    fetched_at: datetime | None = None
    site: str | None = None
    content_language: str | None = None
    locale: str | None = Field(default=None, pattern=LOCALE_TAG)

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        *,
        url: str | None = None,
        content_type: str | None = None,
        fetched_at: datetime | None = None,
        site: str | None = None,
        content_language: str | None = None,
        locale: str | None = None,
    ) -> Document:
        """Build a document, sniffing the content type from the bytes if not given."""
        return cls(
            content=content,
            content_type=(
                _media_type(content_type)
                if content_type
                else sniff_content_type(content) or OCTET_STREAM
            ),
            url=url,
            fetched_at=fetched_at,
            site=site,
            content_language=content_language,
            locale=locale,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        url: str | None = None,
        site: str | None = None,
        locale: str | None = None,
    ) -> Document:
        """Read a local file, using its extension when the bytes don't say what it is."""
        path = Path(path)
        content = path.read_bytes()
        content_type = sniff_content_type(content) or mimetypes.guess_type(path)[0]
        return cls.from_bytes(content, url=url, content_type=content_type, site=site, locale=locale)

    @property
    def source(self) -> str | None:
        """Where the document came from, as generator scopes name it (``Scope.sources``).

        ``site`` if given, else the URL's host; both pass through :func:`normalise_source`,
        so ``https://WWW.Example.com:8080/a`` is ``example.com``. Subdomains are distinct
        sources (``shop.example.com`` isn't ``example.com``). ``None`` when neither is
        known (an unparseable URL has no host), and then no source-scoped generator runs.
        """
        if self.site and self.site.strip():
            return normalise_source(self.site)
        if not self.url:
            return None
        try:
            host = urlsplit(self.url).hostname
        except ValueError:  # e.g. unbalanced IPv6 brackets
            return None
        return normalise_source(host) if host else None

    @property
    def is_html(self) -> bool:
        return self.content_type in (HTML, "application/xhtml+xml")

    @property
    def is_pdf(self) -> bool:
        return self.content_type == PDF

    @property
    def is_image(self) -> bool:
        return self.content_type.startswith("image/")


def normalise_source(source: str) -> str:
    """A source as scopes compare it: trimmed, lower-cased, without a leading ``www.``
    or trailing dot."""
    return source.strip().lower().rstrip(".").removeprefix("www.")


def sniff_content_type(content: bytes) -> str | None:
    """Guess a media type from leading bytes, or ``None`` when unsure."""
    for magic, media_type in _MAGIC:
        if content.startswith(magic):
            return media_type
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "image/webp"
    head = content[:1024].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")) or b"<body" in head or b"<head" in head:
        return HTML
    return None


def _media_type(content_type: str) -> str:
    """Strip parameters such as ``; charset=utf-8`` and normalise case."""
    return content_type.split(";", 1)[0].strip().lower()
