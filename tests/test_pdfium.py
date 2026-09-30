"""pypdfium2 helpers. Skipped when the ``pdf`` extra isn't installed."""

from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("pypdfium2")

from jevex import UnreadablePdfError, _pdfium

SPEC_PDF = (Path(__file__).parent / "fixtures" / "pdf" / "spec.pdf").read_bytes()


def test_page_count_and_texts() -> None:
    assert _pdfium.page_count(SPEC_PDF) == 2
    texts = _pdfium.page_texts(SPEC_PDF)
    assert [text.split("\r\n")[0] for text in texts] == ["Skoda Octavia Estate", "Dimensions"]


def test_keep_pages_copies_the_pages_asked_for_in_order() -> None:
    both = _pdfium.page_texts(SPEC_PDF)
    assert _pdfium.page_texts(_pdfium.keep_pages(SPEC_PDF, [2])) == both[1:]
    assert _pdfium.page_texts(_pdfium.keep_pages(SPEC_PDF, [2, 1])) == both[::-1]


def keep_first(content: bytes) -> bytes:
    return _pdfium.keep_pages(content, [1])


@pytest.mark.parametrize("call", [_pdfium.page_count, _pdfium.page_texts, keep_first])
def test_unreadable_pdfs_raise(call: Callable[[bytes], object]) -> None:
    with pytest.raises(UnreadablePdfError, match="pdfium couldn't open the PDF"):
        call(b"%PDF-1.7 broken")


def test_the_lock_is_released_after_a_failure() -> None:
    with pytest.raises(UnreadablePdfError):
        _pdfium.page_texts(b"not a pdf")
    assert not _pdfium.LOCK.locked()


def test_the_lock_is_doclings_when_docling_is_installed() -> None:
    locks = pytest.importorskip("docling.utils.locks")
    assert _pdfium.LOCK is locks.pypdfium2_lock
