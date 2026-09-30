"""pypdfium2 (``pdf`` extra), typed by what jevex uses of it.

The document gate reads page text with it, the PDF layout parser cuts pages out with it and
the image stage renders pictures and scanned pages with it. pdfium isn't thread-safe, even
across documents, so every use holds :data:`LOCK`, which is Docling's own pdfium lock when
Docling is installed.
"""

from __future__ import annotations

import importlib.util
import io
import threading
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL.Image import Image as PILImage


def _lock() -> threading.Lock:
    if importlib.util.find_spec("docling") is None:
        return threading.Lock()
    # Docling converts in worker threads too; sharing its lock keeps us out of its way.
    from docling.utils.locks import pypdfium2_lock

    return pypdfium2_lock


LOCK = _lock()
"""Held for every call into pdfium. Not reentrant."""


class TextPage(Protocol):
    def get_text_range(self) -> str: ...


class Bitmap(Protocol):
    def to_pil(self) -> PILImage: ...


class Page(Protocol):
    def get_size(self) -> tuple[float, float]: ...

    def get_textpage(self) -> TextPage: ...

    def render(self, *, scale: float, crop: tuple[float, float, float, float]) -> Bitmap: ...


class Pdf(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Page: ...

    def import_pages(self, pdf: Pdf, pages: Sequence[int] | None = None) -> None: ...

    def save(self, dest: io.BytesIO) -> None: ...

    def close(self) -> None: ...


class UnreadablePdfError(Exception):
    """pdfium couldn't open the PDF (damaged, encrypted, or not a PDF)."""


def installed() -> bool:
    return importlib.util.find_spec("pypdfium2") is not None


def open_pdf(content: bytes) -> Pdf:
    """A ``PdfDocument`` for ``content``. Call it holding :data:`LOCK`."""
    import pypdfium2  # pyright: ignore[reportMissingTypeStubs]

    try:
        return cast("Pdf", pypdfium2.PdfDocument(content))
    except pypdfium2.PdfiumError as exc:
        raise UnreadablePdfError(f"pdfium couldn't open the PDF: {exc}") from exc


def page_count(content: bytes) -> int:
    with LOCK:
        pdf = open_pdf(content)
        try:
            return len(pdf)
        finally:
            pdf.close()


def page_texts(content: bytes) -> list[str]:
    """Each page's text layer, in page order (``""`` for a page without one)."""
    with LOCK:
        pdf = open_pdf(content)
        try:
            return [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
        finally:
            pdf.close()


def keep_pages(content: bytes, pages: Sequence[int]) -> bytes:
    """A new PDF holding only ``pages`` (1-based) of ``content``, in that order.

    Page content is copied as it is; document-level parts such as bookmarks are not.
    """
    import pypdfium2  # pyright: ignore[reportMissingTypeStubs]

    with LOCK:
        source = open_pdf(content)
        out = cast("Pdf", pypdfium2.PdfDocument.new())
        try:
            out.import_pages(source, [page - 1 for page in pages])
            buffer = io.BytesIO()
            out.save(buffer)
            return buffer.getvalue()
        finally:
            out.close()
            source.close()
