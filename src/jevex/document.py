"""Input documents."""

from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

HTML = "text/html"
PDF = "application/pdf"
OCTET_STREAM = "application/octet-stream"

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", PDF),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


class Document(BaseModel):
    """A fetched document. jevex takes documents, not URLs, so any crawler can supply them."""

    # Bytes travel as base64 in JSON (e.g. the `jevex serve` API).
    model_config = ConfigDict(frozen=True, ser_json_bytes="base64", val_json_bytes="base64")

    content: bytes
    content_type: str
    url: str | None = None
    fetched_at: datetime | None = None

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        *,
        url: str | None = None,
        content_type: str | None = None,
        fetched_at: datetime | None = None,
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
        )

    @classmethod
    def from_path(cls, path: str | Path, *, url: str | None = None) -> Document:
        """Read a local file, using its extension when the bytes don't say what it is."""
        path = Path(path)
        content = path.read_bytes()
        content_type = sniff_content_type(content) or mimetypes.guess_type(path)[0]
        return cls.from_bytes(content, url=url, content_type=content_type)

    @property
    def is_html(self) -> bool:
        return self.content_type in (HTML, "application/xhtml+xml")

    @property
    def is_pdf(self) -> bool:
        return self.content_type == PDF

    @property
    def is_image(self) -> bool:
        return self.content_type.startswith("image/")


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
