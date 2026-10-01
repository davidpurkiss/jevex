import dataclasses
import hashlib
import io
import random
import re
import sys

import pytest

from jevex.testsite import generate
from jevex.testsite.drawing import (
    Drawing,
    _png,  # pyright: ignore[reportPrivateUsage]
    _scanned_pdf,  # pyright: ignore[reportPrivateUsage]
    plain_text,
    png_bytes,
    rasterise,
    to_pdf,
    to_png,
    to_scanned_pdf,
)
from jevex.testsite.render import CELL_SIZE, spec_sheet


def sample(text: str = "Price from £17,000 (0–62 mph) a\\b") -> Drawing:
    d = Drawing(300, 200, title="A (title)")
    d.box(10, 10, 100, 40, (0.2, 0.3, 0.4))
    d.rule(10, 60, 290, 60, 1.0)
    d.text(20, 40, "Header", 14, bold=True, fill=(1.0, 1.0, 1.0))
    d.text(20, 90, text, 10)
    return d


def test_pdfs_are_well_formed_and_identical_every_time() -> None:
    pdf = to_pdf(sample())
    assert pdf == to_pdf(sample())
    assert pdf.startswith(b"%PDF-1.4\n")
    assert pdf.endswith(b"%%EOF\n")
    assert b"CreationDate" not in pdf
    # Every xref entry points at its object, and startxref at the xref table.
    xref = int(re.findall(rb"startxref\n(\d+)", pdf)[0])
    assert pdf[xref:].startswith(b"xref\n0 8\n")
    offsets = re.findall(rb"(\d{10}) 00000 n", pdf[xref:])
    for number, offset in enumerate(offsets, start=1):
        assert pdf[int(offset) :].startswith(f"{number} 0 obj".encode())


def test_pdf_text_round_trips_through_a_reader() -> None:
    pdfium = pytest.importorskip("pypdfium2")
    page = pdfium.PdfDocument(to_pdf(sample()))[0]
    assert page.get_size() == (300, 200)
    text = page.get_textpage().get_text_range()
    assert "Header" in text
    assert "Price from £17,000 (0–62 mph) a\\b" in text


def test_pdf_text_must_be_winansi() -> None:
    with pytest.raises(UnicodeEncodeError):
        to_pdf(sample("Price 17 000 ₹"))


def test_plain_drawings_write_ascii() -> None:
    assert plain_text("£17,000 · 0–62") == "GBP 17,000 · 0-62"
    d = Drawing(100, 100, plain=True)
    d.text(0, 10, "£17,000 0–62mph", 10)
    assert [t.text for t in d.texts] == ["GBP 17,000 0-62mph"]
    assert d.shown("£5") == "GBP 5"
    assert Drawing(100, 100).shown("£5") == "£5"


def test_rasterising_refuses_text_the_font_cant_draw() -> None:
    pytest.importorskip("PIL")
    with pytest.raises(ValueError, match="can't rasterise 'Price from £17,000"):
        rasterise(sample())
    with pytest.raises(ValueError, match="not ASCII"):
        to_png(sample())


def test_png_bytes_depend_only_on_the_pixels() -> None:
    """The same pixels give the same bytes on every machine. Pillow's encoder (zlib-ng)
    writes different bytes on x86 and arm64, which broke the benchmark's test-site lock."""
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (40, 30))
    pixels = [((x * 7) % 256, (y * 11) % 256, (x * y) % 256) for y in range(30) for x in range(40)]
    image.putdata(pixels)
    png = png_bytes(image)
    assert hashlib.sha256(png).hexdigest() == (
        "f740c453bbdd28cce64400126d633fa3cb10ca9b1b3c1b687fd0b4dbb5fab642"
    )
    with image_module.open(io.BytesIO(png)) as decoded:
        assert decoded.mode == "RGB"
        assert decoded.tobytes() == image.tobytes()
    assert png_bytes(image.convert("RGBA")) == png  # written as RGB whatever the mode


def test_pngs_and_scans() -> None:
    image_module = pytest.importorskip("PIL.Image")
    pdfium = pytest.importorskip("pypdfium2")
    drawing = sample("Plain text")
    png = to_png(drawing)
    _png.cache_clear()
    assert to_png(drawing) == png
    with image_module.open(io.BytesIO(png)) as image:
        assert image.size == (600, 400)  # 144 dpi
        assert image.getpixel((100, 50)) == (51, 76, 102)  # inside the box
    scan = to_scanned_pdf(drawing, "seed-1")
    _scanned_pdf.cache_clear()
    assert scan == to_scanned_pdf(drawing, "seed-1")  # drawn again, not from the cache
    assert scan != to_scanned_pdf(drawing, "seed-2")
    page = pdfium.PdfDocument(scan)[0]
    assert page.get_size() == (300, 200)
    assert page.get_textpage().get_text_range().strip() == ""
    assert b"/Title (A \\(title\\))" in scan


def test_rasterising_needs_pillow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "PIL", None)
    drawing = sample("Not drawn before, so not cached")
    with pytest.raises(ImportError, match=r"install jevex\[testsite\]"):
        to_png(drawing)
    with pytest.raises(ImportError, match=r"install jevex\[testsite\]"):
        to_scanned_pdf(drawing, "seed")
    assert to_pdf(drawing).startswith(b"%PDF")  # vector PDFs don't need it


def test_spec_sheets_refuse_text_that_overflows_a_column() -> None:
    model = generate(42).models[0]
    sheet = spec_sheet(random.Random(1), model)
    assert {t.size for t in sheet.texts if t.text in [v.trim for v in model.variants]} == {
        CELL_SIZE
    }
    long = model.variants[0].model_copy(update={"trim": "An extraordinarily long trim name"})
    wide = dataclasses.replace(model, variants=(long, *model.variants[1:]))
    with pytest.raises(ValueError, match="'An extraordinarily long trim name' doesn't fit"):
        spec_sheet(random.Random(1), wide)
