from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from jevex import Document
from jevex.document import sniff_content_type


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"%PDF-1.7\n...", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n....", "image/png"),
        (b"\xff\xd8\xff\xe0....", "image/jpeg"),
        (b"GIF89a....", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"  <!DOCTYPE html><html>", "text/html"),
        (b"<html><body>hi</body></html>", "text/html"),
        (b"<div><head></head></div>", "text/html"),
        (b"just some bytes", None),
    ],
)
def test_sniff_content_type(content: bytes, expected: str | None) -> None:
    assert sniff_content_type(content) == expected


def test_from_bytes_sniffs_when_type_missing() -> None:
    doc = Document.from_bytes(b"%PDF-1.4", url="https://example.com/spec.pdf")
    assert doc.content_type == "application/pdf"
    assert doc.is_pdf
    assert not doc.is_html
    assert doc.url == "https://example.com/spec.pdf"


def test_from_bytes_unknown_is_octet_stream() -> None:
    assert Document.from_bytes(b"\x00\x01").content_type == "application/octet-stream"


def test_from_bytes_normalises_given_type() -> None:
    doc = Document.from_bytes(b"<p>x</p>", content_type="Text/HTML; charset=utf-8")
    assert doc.content_type == "text/html"
    assert doc.is_html


def test_from_path_falls_back_to_extension(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_bytes(b"<p>no doctype</p>")
    assert Document.from_path(path).content_type == "text/html"


def test_from_path_prefers_magic_bytes(tmp_path: Path) -> None:
    path = tmp_path / "mislabelled.html"
    path.write_bytes(b"%PDF-1.4")
    assert Document.from_path(path).is_pdf


def test_document_is_immutable_and_round_trips() -> None:
    fetched = datetime(2026, 9, 29, tzinfo=UTC)
    doc = Document.from_bytes(b"\x89PNG\r\n\x1a\n", fetched_at=fetched)
    assert doc.is_image
    assert Document.model_validate_json(doc.model_dump_json()) == doc
    assert '"content":"iVBORw0KGgo="' in doc.model_dump_json()
    with pytest.raises(ValidationError):
        doc.url = "https://example.com"  # pyright: ignore[reportAttributeAccessIssue]
