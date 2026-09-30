"""The image stage: loading, OCR, text to components, vision statements.

Most tests use a scripted OCR engine and a tiny PNG, so they need no extra. Tests that
render PDFs (``pdf`` extra) or run RapidOCR for real (``ocr`` extra) skip without them;
CI installs both. RapidOCR's models ship in its wheel, so the real OCR test is offline.
"""

import base64
import io
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from jevex import (
    BBox,
    Component,
    Context,
    DefaultImageLoader,
    Document,
    DomLocation,
    Extractor,
    Field,
    ImageData,
    ImageLoader,
    ImageLocation,
    ImageReading,
    ImageStage,
    ImageText,
    OcrEngine,
    OcrProcessor,
    PageLocation,
    RapidOcrEngine,
    StatementStage,
    UnreadableImageError,
    text_components,
)
from jevex.extractor import STAGE_ORDER, default_pipeline
from jevex.fetch import FetchError
from jevex.images import MAX_IMAGES, decode_data_uri, pages_without_text, raster, render_pdf
from jevex.interfaces import ImageProcessor, ParsedDocument
from jevex.jev import Choice
from jevex.layout import LayoutStage
from jevex.testing import FakeJev

PDF_FIXTURE = Path(__file__).parent / "fixtures" / "pdf" / "spec.pdf"

# A 1x1 white PNG.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
PNG_URI = "data:image/png;base64," + base64.b64encode(PNG).decode()


def line(text: str, x0: float, y0: float, x1: float, y1: float) -> ImageText:
    return ImageText(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), confidence=0.9)


@dataclass
class FakeOcr:
    """An OCR engine that reads the same lines from every image."""

    lines: list[ImageText]
    seen: list[bytes] = field(default_factory=list[bytes])

    def read(self, image: bytes) -> list[ImageText]:
        self.seen.append(image)
        return self.lines


@dataclass
class FakeVision:
    """A vision plugin: statements about the image, no text."""

    said: list[str]

    async def process(self, image: Component, data: ImageData) -> ImageReading:
        return ImageReading(statements=[ImageText(text=s) for s in self.said])


@dataclass
class FakeFetcher:
    content: bytes = PNG
    content_type: str = "image/png"
    fail: bool = False
    urls: list[str] = field(default_factory=list[str])

    async def fetch(self, url: str) -> Document:
        self.urls.append(url)
        if self.fail:
            raise FetchError(f"GET {url} returned HTTP 404")
        return Document.from_bytes(self.content, url=url, content_type=self.content_type)


def stage(*lines: ImageText, max_images: int = MAX_IMAGES) -> tuple[ImageStage, FakeOcr]:
    ocr = FakeOcr(list(lines))
    return ImageStage(processors=[OcrProcessor(engine=ocr)], max_images=max_images), ocr


async def html_context(body: str, url: str | None = "https://cars.test/golf") -> Context:
    document = Document.from_bytes(body.encode(), url=url, content_type="text/html")
    ctx = Context.create(document, [], FakeJev().client())
    await LayoutStage().run(ctx)
    return ctx


def image_data(**kwargs: object) -> ImageData:
    return ImageData.model_validate({"content": PNG, "content_type": "image/png", **kwargs})


def image_component(cid: str = "c1", trail: list[str] | None = None) -> Component:
    return Component(
        id=cid,
        type="image",
        location=DomLocation(dom_path="/html/body/img"),
        heading_trail=trail or [],
    )


def outline(components: list[Component]) -> list[tuple[int, str, str, list[str]]]:
    out: list[tuple[int, str, str, list[str]]] = []

    def visit(c: Component, depth: int) -> None:
        out.append((depth, c.type, c.text, c.heading_trail))
        for child in c.children:
            visit(child, depth + 1)

    for c in components:
        visit(c, 0)
    return out


def test_defaults_satisfy_the_protocols() -> None:
    assert isinstance(OcrProcessor(engine=FakeOcr([])), ImageProcessor)
    assert isinstance(FakeVision([]), ImageProcessor)
    assert isinstance(DefaultImageLoader(), ImageLoader)
    assert isinstance(RapidOcrEngine(), OcrEngine)


def test_image_stage_is_a_default_stage_in_spec_order() -> None:
    names = default_pipeline().names
    assert "images" in STAGE_ORDER
    assert names.index("layout") < names.index("images") < names.index("component_gate")


# --- text to components ------------------------------------------------------------------


def test_lines_become_paragraphs_in_reading_order_with_rows_joined() -> None:
    texts = [
        line("Top speed", 10, 60, 90, 70),
        line("130 mph", 110, 61, 160, 71),  # same row, close: joined
        line("0-62 mph", 10, 40, 80, 50),
        line("9.1 s", 95, 40, 130, 50),
        line("Far column", 400, 41, 480, 51),  # same row, far: a line of its own
    ]
    components = text_components(texts, image_component(trail=["Specs"]), image_data())
    assert [(c.id, c.type, c.text, c.heading_trail) for c in components] == [
        ("c1-t0", "paragraph", "0-62 mph 9.1 s", ["Specs"]),
        ("c1-t1", "paragraph", "Far column", ["Specs"]),
        ("c1-t2", "paragraph", "Top speed 130 mph", ["Specs"]),
    ]
    assert components[0].location == ImageLocation(bbox=BBox(x0=10, y0=40, x1=130, y1=50))


def test_wrapped_lines_join_but_pairs_stand_alone() -> None:
    texts = [
        line("The engine is quiet and it", 10, 10, 200, 20),
        line("pulls from low revs.", 10, 23, 150, 33),
        line("Power: 150 PS", 10, 36, 110, 46),
        line("Torque: 250 Nm", 10, 49, 110, 59),
        line("Indented, so not a wrap", 60, 62, 200, 72),
        line("A gap, so a new paragraph", 10, 100, 200, 110),
    ]
    assert [c.text for c in text_components(texts, image_component(), image_data())] == [
        "The engine is quiet and it pulls from low revs.",
        "Power: 150 PS",
        "Torque: 250 Nm",
        "Indented, so not a wrap",
        "A gap, so a new paragraph",
    ]


def test_tall_short_lines_are_headings_opening_sections() -> None:
    texts = [
        line("Intro line", 10, 0, 100, 10),
        line("Performance", 10, 20, 200, 40),
        line("0-62 mph: 9.1 s", 10, 50, 120, 60),
        line("Top speed: 130 mph", 10, 70, 140, 80),
        line("Economy", 10, 100, 150, 120),
        line("Combined: 52 mpg", 10, 130, 140, 140),
    ]
    components = text_components(texts, image_component(trail=["Golf"]), image_data())
    assert outline(components) == [
        (0, "paragraph", "Intro line", ["Golf"]),
        (0, "section", "", ["Golf"]),
        (1, "heading", "Performance", ["Golf"]),
        (1, "paragraph", "0-62 mph: 9.1 s", ["Golf", "Performance"]),
        (1, "paragraph", "Top speed: 130 mph", ["Golf", "Performance"]),
        (0, "section", "", ["Golf"]),
        (1, "heading", "Economy", ["Golf"]),
        (1, "paragraph", "Combined: 52 mpg", ["Golf", "Economy"]),
    ]
    ids = [c.id for top in components for c in top.walk()]
    assert ids == [f"c1-t{n}" for n in range(8)]
    assert components[1].location == ImageLocation(bbox=BBox(x0=10, y0=20, x1=200, y1=80))


def test_a_long_or_lone_tall_line_is_not_a_heading() -> None:
    long = "A very long line set in large type that runs on and on well past any title's length"
    texts = [line(long, 0, 0, 900, 40), line("small print", 0, 50, 90, 60)]
    assert [c.type for c in text_components(texts, image_component(), image_data())] == [
        "paragraph",
        "paragraph",
    ]
    lone = text_components([line("Big", 0, 0, 90, 60)], image_component(), image_data())
    assert [c.type for c in lone] == ["paragraph"]


def test_pieces_without_boxes_stay_in_order_one_paragraph_each() -> None:
    texts = [ImageText(text="second"), line("first", 0, 0, 10, 10)]
    components = text_components(texts, image_component(), image_data(), start=5)
    assert [(c.id, c.text, c.location) for c in components] == [
        ("c1-t5", "second", ImageLocation()),
        ("c1-t6", "first", ImageLocation(bbox=BBox(x0=0, y0=0, x1=10, y1=10))),
    ]


def test_boxes_on_a_pdf_page_map_to_page_points() -> None:
    data = image_data(page=2, region=BBox(x0=100, y0=200, x1=300, y1=400), scale=2.0)
    assert data.location(BBox(x0=10, y0=20, x1=30, y1=40)) == ImageLocation(
        page=2, bbox=BBox(x0=105, y0=210, x1=115, y1=220)
    )
    assert data.location() == ImageLocation(page=2)


def test_no_text_gives_no_components() -> None:
    assert text_components([], image_component(), image_data()) == []


# --- loading -----------------------------------------------------------------------------


def test_data_uris_decode() -> None:
    assert decode_data_uri(PNG_URI) == (PNG, "image/png")
    assert decode_data_uri("data:image/svg+xml,%3Csvg%3E") == (b"<svg>", "image/svg+xml")
    assert decode_data_uri("data:;base64,aGk=") == (b"hi", "text/plain")
    with pytest.raises(UnreadableImageError, match="no ','"):
        decode_data_uri("data:image/png;base64")
    with pytest.raises(UnreadableImageError, match="base64"):
        decode_data_uri("data:image/png;base64,not*base64")


def test_raster_trusts_the_bytes_over_the_declared_type() -> None:
    assert raster(PNG, "application/octet-stream").content_type == "image/png"
    assert raster(b"BM...", "image/bmp; x=1").content_type == "image/bmp"
    with pytest.raises(UnreadableImageError, match=r"image/svg\+xml isn't a raster image"):
        raster(b"<svg/>", "image/svg+xml")
    with pytest.raises(UnreadableImageError, match="text/html"):
        raster(b"<html><body>404</body></html>", "image/png")


async def test_loader_reads_data_uris_and_image_documents() -> None:
    loader = DefaultImageLoader()
    html = Document.from_bytes(b"<p/>", url="https://cars.test/")
    image = image_component().model_copy(update={"src": PNG_URI})
    assert await loader.load(image, html) == ImageData(content=PNG, content_type="image/png")
    assert await loader.load(image_component(), html) is None  # no src

    png = Document.from_bytes(PNG, url="https://cars.test/a.png")
    loaded = await loader.load(image_component(), png)
    assert loaded == ImageData(content=PNG, content_type="image/png", src="https://cars.test/a.png")


async def test_loader_fetches_remote_images_only_with_a_fetcher() -> None:
    html = Document.from_bytes(b"<p/>", url="https://cars.test/")
    image = image_component().model_copy(update={"src": "https://cars.test/a.png"})
    assert await DefaultImageLoader().load(image, html) is None

    fetcher = FakeFetcher()
    loaded = await DefaultImageLoader(fetcher=fetcher).load(image, html)
    assert loaded is not None
    assert (loaded.content, loaded.src) == (PNG, "https://cars.test/a.png")
    assert fetcher.urls == ["https://cars.test/a.png"]

    relative = image_component().model_copy(update={"src": "a.png"})
    assert await DefaultImageLoader(fetcher=fetcher).load(relative, html) is None


async def test_loader_reports_failed_fetches_and_non_images() -> None:
    html = Document.from_bytes(b"<p/>", url="https://cars.test/")
    image = image_component().model_copy(update={"src": "https://cars.test/a.png"})
    with pytest.raises(UnreadableImageError, match="404"):
        await DefaultImageLoader(fetcher=FakeFetcher(fail=True)).load(image, html)
    page = FakeFetcher(content=b"<html><body>Not found</body></html>", content_type="text/html")
    with pytest.raises(UnreadableImageError, match="isn't a raster image"):
        await DefaultImageLoader(fetcher=page).load(image, html)


def test_loader_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        DefaultImageLoader(scale=0)


# --- PDFs (pdf extra) --------------------------------------------------------------------


def scanned_page_pdf() -> bytes:
    """``spec.pdf`` (two pages with text) plus a third page that's only a picture."""
    pdfium = pytest.importorskip("pypdfium2")
    image_module = pytest.importorskip("PIL.Image")
    picture = image_module.new("RGB", (400, 300), "white")
    scan = io.BytesIO()
    picture.save(scan, format="PDF", resolution=72)
    pdf = pdfium.PdfDocument(PDF_FIXTURE.read_bytes())
    pdf.import_pages(pdfium.PdfDocument(scan.getvalue()))
    out = io.BytesIO()
    pdf.save(out)
    return out.getvalue()


def test_pages_without_a_text_layer_are_found() -> None:
    document = Document.from_bytes(scanned_page_pdf())
    assert pages_without_text(document) == [3]
    assert pages_without_text(Document.from_path(PDF_FIXTURE)) == []


def test_pdf_pictures_render_cropped_at_scale() -> None:
    pytest.importorskip("pypdfium2")
    image_module = pytest.importorskip("PIL.Image")
    document = Document.from_path(PDF_FIXTURE)
    box = BBox(x0=72, y0=100, x1=272, y1=150)
    data = render_pdf(document, 1, box, scale=2.0)
    assert (data.content_type, data.page, data.region, data.scale) == ("image/png", 1, box, 2.0)
    assert image_module.open(io.BytesIO(data.content)).size == (400, 100)

    whole = render_pdf(document, 2, scale=1.0)
    assert whole.region == BBox(x0=0, y0=0, x1=612, y1=792)
    # A box running off the page is clipped to it.
    clipped = render_pdf(document, 1, BBox(x0=500, y0=700, x1=900, y1=900), scale=1.0)
    assert clipped.region == BBox(x0=500, y0=700, x1=612, y1=792)


def test_pdf_render_refuses_missing_pages_and_empty_boxes() -> None:
    pytest.importorskip("pypdfium2")
    document = Document.from_path(PDF_FIXTURE)
    with pytest.raises(UnreadableImageError, match="no page 9"):
        render_pdf(document, 9)
    with pytest.raises(UnreadableImageError, match="empty"):
        render_pdf(document, 1, BBox(x0=700, y0=0, x1=800, y1=10))


def pdf_tree() -> Component:
    def at(cid: str, type_: str, page: int, text: str = "") -> Component:
        return Component.model_validate(
            {
                "id": cid,
                "type": type_,
                "text": text,
                "location": PageLocation(page=page, bbox=BBox(x0=72, y0=72, x1=272, y1=172)),
            }
        )

    return Component(
        id="c0",
        type="section",
        location=PageLocation(page=1),
        children=[
            at("c1", "paragraph", 1, "Golf"),
            at("c2", "image", 2),
            at("c3", "image", 3),  # a picture Docling found on the scanned page
            at("c4", "paragraph", 4, "Back cover"),
        ],
    )


async def test_scanned_pages_are_read_whole_instead_of_their_pictures() -> None:
    pdfium = pytest.importorskip("pypdfium2")
    pdf = pdfium.PdfDocument(scanned_page_pdf())
    pdf.import_pages(pdfium.PdfDocument(PDF_FIXTURE.read_bytes()), [0])  # page 4 has text
    out = io.BytesIO()
    pdf.save(out)
    document = Document.from_bytes(out.getvalue())
    ctx = Context.create(document, [], FakeJev().client())
    ctx.parsed = ParsedDocument(document=document, root=pdf_tree())
    images, ocr = stage(line("Boot: 380 litres", 30, 60, 330, 90))
    images.loader = DefaultImageLoader(scale=1.5)
    await images.run(ctx)

    root = ctx.parsed.root
    assert [(c.id, c.type) for c in root.children] == [
        ("c1", "paragraph"),
        ("c2", "image"),
        ("c3", "image"),
        ("page3", "image"),
        ("c4", "paragraph"),
    ]
    assert len(ocr.seen) == 2  # c2 and the page, not the picture on it
    page = root.children[3]
    assert page.location == PageLocation(page=3)
    [text] = page.children
    assert (text.id, text.text) == ("page3-t0", "Boot: 380 litres")
    # Pixels at 1.5 per point, on a whole-page render.
    assert text.location == ImageLocation(page=3, bbox=BBox(x0=20, y0=40, x1=220, y1=60))
    [picture] = root.children[1].children
    assert picture.location == ImageLocation(page=2, bbox=BBox(x0=92, y0=112, x1=292, y1=132))
    assert root.children[2].children == []
    assert ctx.events == []


async def test_pdf_images_need_pypdfium2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jevex.images._pdfium_installed", lambda: False)
    document = Document.from_path(PDF_FIXTURE)
    ctx = Context.create(document, [], FakeJev().client())
    ctx.parsed = ParsedDocument(document=document, root=pdf_tree())
    images, ocr = stage(line("x", 0, 0, 1, 1))
    await images.run(ctx)
    assert [e.kind for e in ctx.events] == ["images_skipped"]
    assert "jevex[pdf]" in ctx.events[0].message
    assert ocr.seen == []


# --- the stage ---------------------------------------------------------------------------


async def test_ocr_text_becomes_components_and_ocr_statements() -> None:
    ctx = await html_context(
        f"<h1>Golf</h1><p>Intro.</p><img src='{PNG_URI}' alt='Spec card'><p>After.</p>"
    )
    images, ocr = stage(
        line("Quiet and quick.", 0, 0, 100, 10), line("Power: 150 PS", 0, 30, 100, 40)
    )
    await images.run(ctx)
    await StatementStage().run(ctx)
    assert ctx.parsed is not None
    assert ocr.seen == [PNG]
    [image] = [c for c in ctx.parsed.root.walk() if c.type == "image"]
    assert [(c.type, c.text, c.heading_trail) for c in image.children] == [
        ("paragraph", "Quiet and quick.", ["Golf"]),
        ("paragraph", "Power: 150 PS", ["Golf"]),
    ]
    assert [(s.text, s.kind) for s in ctx.parsed.statements.values()] == [
        ("Golf", "sentence"),
        ("Intro.", "sentence"),
        ("Spec card", "alt_text"),
        ("Quiet and quick.", "ocr"),
        ("Power: 150 PS", "key_value"),
        ("After.", "sentence"),
    ]
    pair = ctx.parsed.statements[f"{image.id}-t1.0"]
    # A data: image has no URL worth repeating on every statement.
    assert pair.location == ImageLocation(bbox=BBox(x0=0, y0=30, x1=100, y1=40))
    assert ctx.events == []


async def test_image_documents_become_a_tree_holding_the_image() -> None:
    document = Document.from_bytes(PNG, url="https://cars.test/card.png")
    ctx = Context.create(document, [], FakeJev().client())
    await LayoutStage().run(ctx)
    assert [e.kind for e in ctx.events] == ["layout_skipped"]
    images, _ = stage(line("Price: £24,995", 5, 5, 120, 20))
    await images.run(ctx)
    parsed = ctx.parsed
    assert parsed is not None
    [image] = parsed.root.children
    assert (image.type, image.src) == ("image", "https://cars.test/card.png")
    [text] = image.children
    assert text.location == ImageLocation(
        src="https://cars.test/card.png", bbox=BBox(x0=5, y0=5, x1=120, y1=20)
    )


async def test_vision_statements_are_kept_as_they_are() -> None:
    ctx = await html_context(f"<h1>Golf</h1><img src='{PNG_URI}'><p>After.</p>")
    ocr = FakeOcr([line("Golf GTI", 0, 0, 50, 10)])
    vision = FakeVision(["A red five-door hatchback. It has alloy wheels.", "The badge says GTI"])
    await ImageStage(processors=[OcrProcessor(engine=ocr), vision]).run(ctx)
    await StatementStage().run(ctx)
    assert ctx.parsed is not None
    [image] = [c for c in ctx.parsed.root.walk() if c.type == "image"]
    assert [(c.id, c.text) for c in image.children] == [
        (f"{image.id}-t0", "Golf GTI"),
        (f"{image.id}-v0", "A red five-door hatchback. It has alloy wheels."),
        (f"{image.id}-v1", "The badge says GTI"),
    ]
    kinds = [(s.text, s.kind) for s in ctx.parsed.statements.values()]
    assert kinds == [
        ("Golf", "sentence"),
        ("Golf GTI", "ocr"),
        ("A red five-door hatchback. It has alloy wheels.", "vision"),  # not split
        ("The badge says GTI", "vision"),
        ("After.", "sentence"),
    ]
    said = ctx.parsed.statements[f"{image.id}-v1.0"]
    assert (said.component_id, said.heading_trail) == (f"{image.id}-v1", ["Golf"])


async def test_long_vision_statements_are_cut() -> None:
    ctx = await html_context(f"<img src='{PNG_URI}'>")
    await ImageStage(processors=[FakeVision(["word " * 2000])]).run(ctx)
    assert ctx.parsed is not None
    assert len(ctx.parsed.statements) > 1
    assert all(s.kind == "vision" for s in ctx.parsed.statements.values())
    assert all(":" in sid for sid in ctx.parsed.statements)


async def test_unreadable_and_unloadable_images_are_reported() -> None:
    ctx = await html_context(
        "<img src='data:image/png;base64,***' alt=A>"
        "<img src='data:image/svg+xml,%3Csvg%3E' alt=B>"
        "<img src='/remote.png' alt=C>"
        f"<img src='{PNG_URI}' alt=D>"
    )
    images, ocr = stage(line("read", 0, 0, 9, 9))
    await images.run(ctx)
    assert ctx.parsed is not None
    ids = [c.id for c in ctx.parsed.root.walk() if c.type == "image"]
    events = {e.kind: e for e in ctx.events}
    assert set(events) == {"images_unread", "images_not_loaded"}
    assert list(events["images_unread"].data["images"]) == ids[:2]
    assert "base64" in events["images_unread"].data["images"][ids[0]]
    assert events["images_not_loaded"].data["images"] == [ids[2]]
    assert "fetcher" in events["images_not_loaded"].message
    assert len(ocr.seen) == 1


async def test_an_engine_that_cant_decode_an_image_is_reported() -> None:
    class Broken:
        def read(self, image: bytes) -> list[ImageText]:
            raise UnreadableImageError("OCR can't decode the image")

    ctx = await html_context(f"<img src='{PNG_URI}'>")
    await ImageStage(processors=[OcrProcessor(engine=Broken())]).run(ctx)
    [event] = ctx.events
    assert event.kind == "images_unread"
    assert list(event.data["images"].values()) == ["OCR can't decode the image"]


async def test_other_errors_propagate() -> None:
    class Crashes:
        def read(self, image: bytes) -> list[ImageText]:
            raise RuntimeError("model file missing")

    ctx = await html_context(f"<img src='{PNG_URI}'>")
    with pytest.raises(RuntimeError, match="model file missing"):
        await ImageStage(processors=[OcrProcessor(engine=Crashes())]).run(ctx)


async def test_images_beyond_the_cap_are_counted_not_read() -> None:
    ctx = await html_context("".join(f"<img src='{PNG_URI}' alt={n}>" for n in range(4)))
    images, ocr = stage(line("x", 0, 0, 9, 9), max_images=3)
    await images.run(ctx)
    assert len(ocr.seen) == 3
    [event] = ctx.events
    assert (event.kind, event.data) == ("images_capped", {"unread": 1})
    with pytest.raises(ValueError, match="max_images"):
        ImageStage(max_images=-1)
    assert ImageStage().max_images == MAX_IMAGES


async def test_without_a_processor_images_are_skipped_with_an_event() -> None:
    ctx = await html_context(f"<p>x</p><img src='{PNG_URI}'><img src=a.png>")
    await ImageStage(processors=[]).run(ctx)
    [event] = ctx.events
    assert (event.kind, event.data) == ("images_skipped", {"images": 2})
    assert "jevex[ocr]" in event.message

    document = Document.from_bytes(PNG)
    ctx = Context.create(document, [], FakeJev().client())
    await ImageStage(processors=[]).run(ctx)
    assert ctx.parsed is None
    assert [e.data for e in ctx.events] == [{"images": 1}]


async def test_nothing_happens_without_a_tree_or_images() -> None:
    images, ocr = stage(line("x", 0, 0, 9, 9))
    ctx = Context.create(Document.from_bytes(b"%PDF-1.7"), [], FakeJev().client())
    await images.run(ctx)
    assert ctx.parsed is None
    ctx = await html_context("<p>No pictures here.</p>")
    await images.run(ctx)
    assert (ocr.seen, ctx.events) == ([], [])
    ctx = await html_context("<p>x</p>")
    await ImageStage(processors=[]).run(ctx)
    assert ctx.events == []


def test_default_processors_follow_the_ocr_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jevex.images._ocr_installed", lambda: False)
    assert ImageStage().processors == []
    monkeypatch.setattr("jevex.images._ocr_installed", lambda: True)
    [processor] = ImageStage().processors
    assert isinstance(processor, OcrProcessor)
    assert isinstance(processor.engine, RapidOcrEngine)


# --- end to end --------------------------------------------------------------------------


class Car(BaseModel):
    price: Decimal = Field(description="Price", unit="GBP")


async def test_a_price_on_an_image_document_is_extracted() -> None:
    fake = (
        FakeJev(default_p=1.0)
        .choice("Which detail", "price", state="24,995")
        .choice("Which of these", lambda q: next(o for o in q.options if o != "none"))
    )
    ocr = FakeOcr([line("Kestrova SE", 0, 0, 200, 40), line("Price: £24,995", 0, 60, 200, 80)])
    pipeline = default_pipeline().replace(
        "images", ImageStage(processors=[OcrProcessor(engine=ocr)])
    )
    async with Extractor([Car], jev=fake.client(), pipeline=pipeline) as ex:
        result = await ex.extract(Document.from_bytes(PNG, url="https://cars.test/card.png"))
    item = result.one(Car)
    assert item.strict() == Car(price=Decimal("24995"))
    source = item.meta.price.source
    assert source is not None
    assert source.location == ImageLocation(
        src="https://cars.test/card.png", bbox=BBox(x0=0, y0=60, x1=200, y1=80)
    )
    assert item.meta.price.method == "generator"
    questions = [q for q in fake.questions if isinstance(q, Choice)]
    assert questions


# --- RapidOCR (ocr extra) ----------------------------------------------------------------


def test_rapidocr_reads_rendered_text() -> None:
    pytest.importorskip("rapidocr")
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    font_module = pytest.importorskip("PIL.ImageFont")
    picture = image_module.new("RGB", (700, 200), "white")
    draw = draw_module.Draw(picture)
    font = font_module.load_default(size=32)
    draw.text((20, 20), "0-62 mph: 9.1 seconds", fill="black", font=font)
    draw.text((20, 110), "Top speed 130 mph", fill="black", font=font)
    out = io.BytesIO()
    picture.save(out, format="PNG")

    lines = RapidOcrEngine().read(out.getvalue())
    assert [t.text for t in lines] == ["0-62 mph: 9.1 seconds", "Top speed 130 mph"]
    first, second = lines
    assert first.bbox is not None
    assert second.bbox is not None
    assert first.bbox.y1 < 100 < second.bbox.y0
    assert first.confidence is not None
    assert first.confidence > 0.8

    assert RapidOcrEngine().read(PNG) == []  # blank
    with pytest.raises(UnreadableImageError, match="decode"):
        RapidOcrEngine().read(b"not an image")


def test_rapidocr_drops_lines_below_min_confidence() -> None:
    pytest.importorskip("rapidocr")
    with pytest.raises(ValueError, match="min_confidence"):
        RapidOcrEngine(min_confidence=1.5)
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    font_module = pytest.importorskip("PIL.ImageFont")
    picture = image_module.new("RGB", (500, 100), "white")
    draw_module.Draw(picture).text(
        (20, 20), "Boot 380 litres", fill="black", font=font_module.load_default(size=32)
    )
    out = io.BytesIO()
    picture.save(out, format="PNG")
    assert RapidOcrEngine(min_confidence=1.0).read(out.getvalue()) == []
