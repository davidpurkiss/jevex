"""The image stage: text in pictures, as components and statements (spec: *Image stage*,
stage 6).

Pictures carry facts too: an infographic's "0-62 mph 7.9 s", a scanned spec sheet, a
brochure page saved as a JPEG. :class:`ImageStage` reads them so later stages see that text
like any other.

- **Which images.** Every ``image`` component in the layout tree (HTML ``img``, PDF
  pictures). In a PDF, each page with no text layer (a scan) is read whole, as a new
  ``image`` component for the page, instead of the pictures on it. An image document
  (``image/png``...), which no layout parser reads, becomes a tree holding one image.
- **Loading.** An :class:`ImageLoader` gets each image's bytes. The default,
  :class:`DefaultImageLoader`, decodes ``data:`` URIs, renders PDF pictures and pages with
  pypdfium2 (``pdf`` extra) and takes an image document's own bytes. It fetches remote
  images only through a :class:`~jevex.interfaces.Fetcher` given to it: jevex takes
  documents, and makes no requests of its own unless asked to.
- **Reading.** Every :class:`~jevex.interfaces.ImageProcessor` reads every loaded image.
  The default, :class:`OcrProcessor`, runs OCR (:class:`RapidOcrEngine`, ``ocr`` extra).
  A vision-model plugin is a processor that returns statements.
- **Text becomes components.** Text read from an image is parsed into child components
  of the image (:func:`text_components`): lines in reading order, wrapped lines joined into
  paragraphs, and a short line clearly taller than the rest as a heading that opens a
  section, so what follows carries it in its heading trail. The statement stage splits
  them like any paragraph; their sentences are ``ocr`` statements.
- **Vision statements** are taken as they are: each becomes a ``vision`` statement on a
  paragraph child of its own (so the component gate sees its text), and isn't split
  again. Values selected from one are recorded with ``method="vision"``.

Everything found in an image has an :class:`~jevex.layout.ImageLocation`: the image's URL
or PDF page, plus the bbox of its line (on a PDF page, in page points; otherwise in the
image's pixels). A ``data:`` image has no URL worth repeating, so its ``src`` is unset;
the component id still points at the image.

An image that can't be read (a broken ``data:`` URI, a failed fetch, bytes that aren't a
raster image) is recorded in an ``images_unread`` event and skipped. With no processor
(the ``ocr`` extra isn't installed and none was given) the stage only records
``images_skipped``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib.util
import io
import statistics
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable
from urllib.parse import unquote_to_bytes

from pydantic import BaseModel, ConfigDict, Field

from jevex._tasks import gather
from jevex.document import sniff_content_type
from jevex.fetch import FetchError
from jevex.interfaces import ParsedDocument
from jevex.layout import BBox, Component, ImageLocation, PageLocation
from jevex.split import cut_statement, is_key_value
from jevex.statements import Statement

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from PIL.Image import Image as PILImage
    from rapidocr import RapidOCR

    from jevex.document import Document
    from jevex.interfaces import Fetcher, ImageProcessor
    from jevex.layout import ComponentType
    from jevex.pipeline import Context

RASTER_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp", "image/tiff"}
)
"""Image types the processors are given. Anything else (SVG, a fetched HTML error page)
is unreadable."""

PDF_RENDER_SCALE = 3.0
"""Pixels per PDF point when rendering a picture or page for OCR: 216 dpi, enough for
small print."""

MAX_IMAGES = 50
"""The most images the stage reads per document; OCR takes a second or so each."""


class UnreadableImageError(Exception):
    """An image couldn't be loaded or read: a broken ``data:`` URI, a failed fetch, bytes
    that aren't a raster image. The image stage records it and moves on."""


class ImageData(BaseModel):
    """An image's bytes, loaded for the processors, and where they came from.

    For a PDF, ``page`` is the page, ``region`` the part of it the image shows (page
    points, top-left origin) and ``scale`` the image's pixels per point, so a box found in
    the image maps back onto the page (:meth:`location`). Elsewhere boxes stay in pixels.
    """

    model_config = ConfigDict(frozen=True, ser_json_bytes="base64", val_json_bytes="base64")

    content: bytes
    content_type: str
    src: str | None = None
    page: int | None = Field(default=None, ge=1)
    region: BBox | None = None
    scale: float = Field(default=1.0, gt=0)

    def location(self, bbox: BBox | None = None) -> ImageLocation:
        """Where a box found in the image (in its pixels) sits: on the page, for a PDF."""
        if bbox is not None and self.region is not None:
            x, y, s = self.region.x0, self.region.y0, self.scale
            bbox = BBox(
                x0=x + bbox.x0 / s, y0=y + bbox.y0 / s, x1=x + bbox.x1 / s, y1=y + bbox.y1 / s
            )
        return ImageLocation(src=self.src, page=self.page, bbox=bbox)


class ImageText(BaseModel):
    """A piece of text from an image: an OCR line, or a vision model's statement.

    ``bbox`` is in the image's pixels (top-left origin), when known.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    bbox: BBox | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class ImageReading(BaseModel):
    """What an :class:`~jevex.interfaces.ImageProcessor` found in one image.

    ``text`` is text seen in the image (OCR lines), parsed into components by the stage.
    ``statements`` are statements about the image (a vision model's), used as they are.
    """

    model_config = ConfigDict(frozen=True)

    text: list[ImageText] = Field(default_factory=list[ImageText])
    statements: list[ImageText] = Field(default_factory=list[ImageText])


@runtime_checkable
class ImageLoader(Protocol):
    async def load(self, image: Component, document: Document) -> ImageData | None:
        """The image's bytes, or ``None`` when it has nothing to load (no URL, or a remote
        one this loader can't fetch). Raise :class:`UnreadableImageError` when loading
        fails."""
        ...


@runtime_checkable
class OcrEngine(Protocol):
    def read(self, image: bytes) -> list[ImageText]:
        """The text lines in an encoded image (PNG, JPEG...), boxes in pixels.

        CPU-bound and called in a worker thread. Raise :class:`UnreadableImageError` for
        bytes that aren't an image the engine can decode.
        """
        ...


# --- Loading -----------------------------------------------------------------------------


@dataclass
class DefaultImageLoader:
    """The default :class:`ImageLoader`.

    - A picture or page of a PDF (a :class:`~jevex.layout.PageLocation`) is rendered with
      pypdfium2 at ``scale`` pixels per point, cropped to its bbox (no bbox: the whole
      page).
    - An image document is its own bytes.
    - A ``data:`` URI is decoded.
    - An ``http(s)`` URL is fetched with ``fetcher``; with none (the default) it isn't
      loaded. :class:`~jevex.fetch.SimpleFetcher` honours robots.txt.
    """

    fetcher: Fetcher | None = None
    scale: float = PDF_RENDER_SCALE

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

    async def load(self, image: Component, document: Document) -> ImageData | None:
        location = image.location
        if document.is_pdf and isinstance(location, PageLocation):
            return await asyncio.to_thread(
                render_pdf, document, location.page, location.bbox, scale=self.scale
            )
        if document.is_image:
            return raster(document.content, document.content_type, src=document.url)
        src = image.src
        if src is None:
            return None
        if src.startswith("data:"):
            content, declared = decode_data_uri(src)
            return raster(content, declared)
        if self.fetcher is None or not src.startswith(("http://", "https://")):
            return None
        try:
            fetched = await self.fetcher.fetch(src)
        except FetchError as exc:
            raise UnreadableImageError(f"fetching {src} failed: {exc}") from exc
        return raster(fetched.content, fetched.content_type, src=src)


def raster(content: bytes, content_type: str, *, src: str | None = None) -> ImageData:
    """``content`` as :class:`ImageData`, typed by its bytes where they say, else by
    ``content_type``. Raises :class:`UnreadableImageError` unless it's a raster image."""
    kind = sniff_content_type(content) or content_type.split(";", 1)[0].strip().lower()
    if kind not in RASTER_TYPES:
        raise UnreadableImageError(f"{kind or 'unknown content'} isn't a raster image")
    return ImageData(content=content, content_type=kind, src=src)


def decode_data_uri(uri: str) -> tuple[bytes, str]:
    """The bytes and media type of a ``data:`` URI. Raises :class:`UnreadableImageError`
    when it's malformed."""
    header, comma, data = uri.removeprefix("data:").partition(",")
    if not comma:
        raise UnreadableImageError("data: URI has no ',' before its data")
    params = [p.strip() for p in header.split(";")]
    media_type = params[0].lower() or "text/plain"
    if "base64" in (p.lower() for p in params[1:]):
        try:
            return base64.b64decode("".join(data.split()), validate=True), media_type
        except (binascii.Error, ValueError) as exc:
            raise UnreadableImageError(f"data: URI isn't valid base64: {exc}") from exc
    return unquote_to_bytes(data), media_type


_PDFIUM_LOCK = threading.Lock()
"""pdfium isn't thread-safe, so every call into it is serialised."""


def _pdfium_installed() -> bool:
    return importlib.util.find_spec("pypdfium2") is not None


class _TextPage(Protocol):
    def get_text_range(self) -> str: ...


class _Bitmap(Protocol):
    def to_pil(self) -> PILImage: ...


class _Page(Protocol):
    def get_size(self) -> tuple[float, float]: ...

    def get_textpage(self) -> _TextPage: ...

    def render(self, *, scale: float, crop: tuple[float, float, float, float]) -> _Bitmap: ...


class _Pdf(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> _Page: ...

    def close(self) -> None: ...


def _open_pdf(content: bytes) -> _Pdf:
    """A pypdfium2 ``PdfDocument`` (``pdf`` extra), typed by what's used of it."""
    import pypdfium2  # pyright: ignore[reportMissingTypeStubs]

    return cast("_Pdf", pypdfium2.PdfDocument(content))


def pages_without_text(document: Document) -> list[int]:
    """The 1-based numbers of a PDF's pages with no text layer (scans, or pages that are
    one big picture), found with pypdfium2 (``pdf`` extra)."""
    with _PDFIUM_LOCK:
        pdf = _open_pdf(document.content)
        try:
            empty: list[int] = []
            for index in range(len(pdf)):
                textpage = pdf[index].get_textpage()
                if not textpage.get_text_range().strip():
                    empty.append(index + 1)
            return empty
        finally:
            pdf.close()


def render_pdf(
    document: Document, page: int, bbox: BBox | None = None, *, scale: float = PDF_RENDER_SCALE
) -> ImageData:
    """Page ``page`` of a PDF as a PNG, cropped to ``bbox`` (page points, top-left
    origin; ``None`` for the whole page), at ``scale`` pixels per point."""
    with _PDFIUM_LOCK:
        pdf = _open_pdf(document.content)
        try:
            if not 1 <= page <= len(pdf):
                raise UnreadableImageError(f"the PDF has no page {page}")
            pdf_page = pdf[page - 1]
            width, height = pdf_page.get_size()
            region = BBox(x0=0, y0=0, x1=width, y1=height)
            if bbox is not None:
                region = BBox(
                    x0=min(max(bbox.x0, 0), width),
                    y0=min(max(bbox.y0, 0), height),
                    x1=min(max(bbox.x1, 0), width),
                    y1=min(max(bbox.y1, 0), height),
                )
            if region.x1 - region.x0 < 1 or region.y1 - region.y0 < 1:
                raise UnreadableImageError(f"the picture's box {bbox} is empty on page {page}")
            # pdfium crops by how much to take off each side (left, bottom, right, top).
            crop = (region.x0, height - region.y1, width - region.x1, region.y0)
            bitmap = pdf_page.render(scale=scale, crop=crop)
            picture = bitmap.to_pil()
        finally:
            pdf.close()
    out = io.BytesIO()
    picture.save(out, format="PNG")
    return ImageData(
        content=out.getvalue(), content_type="image/png", page=page, region=region, scale=scale
    )


# --- Reading -----------------------------------------------------------------------------


class RapidOcrEngine:
    """OCR with `RapidOCR <https://github.com/RapidAI/RapidOCR>`_ (``ocr`` extra): PaddleOCR's
    PP-OCR models on ONNX Runtime. The models ship in the wheel, so nothing is downloaded.
    They read Latin script and Chinese.

    Lines Jev would be misled by, those below ``min_confidence``, are dropped. ``params``
    go to ``RapidOCR(params=...)`` (keys as in its ``config.yaml``, e.g.
    ``{"Det.box_thresh": 0.6}``). The model loads on first use, and calls are serialised,
    since one engine isn't safe to share between threads.
    """

    def __init__(
        self, *, min_confidence: float = 0.5, params: Mapping[str, Any] | None = None
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence}")
        self.min_confidence = min_confidence
        self._params: dict[str, Any] = {"Global.log_level": "error", **(params or {})}
        self._ocr: RapidOCR | None = None
        self._lock = threading.Lock()

    def read(self, image: bytes) -> list[ImageText]:
        from PIL import UnidentifiedImageError
        from rapidocr import LoadImageError, RapidOCR
        from rapidocr.utils.output import RapidOCROutput

        with self._lock:
            if self._ocr is None:
                self._ocr = RapidOCR(params=dict(self._params))
            try:
                result = self._ocr(image)
            except (UnidentifiedImageError, LoadImageError) as exc:
                raise UnreadableImageError(f"OCR can't decode the image: {exc}") from exc
        if not isinstance(result, RapidOCROutput) or result.boxes is None:
            return []
        txts = result.txts or ()
        scores = result.scores or ()
        lines: list[ImageText] = []
        for box, text, score in zip(result.boxes.tolist(), txts, scores, strict=True):
            text = " ".join(text.split())
            if not text or score < self.min_confidence:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            bbox = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
            lines.append(ImageText(text=text, bbox=bbox, confidence=min(max(score, 0.0), 1.0)))
        return lines


@dataclass
class OcrProcessor:
    """The default :class:`~jevex.interfaces.ImageProcessor`: OCR with ``engine``, in a
    worker thread."""

    engine: OcrEngine = field(default_factory=RapidOcrEngine)

    async def process(self, image: Component, data: ImageData) -> ImageReading:
        return ImageReading(text=await asyncio.to_thread(self.engine.read, data.content))


# --- Text to components ------------------------------------------------------------------

HEADING_SCALE = 1.4
"""A line at least this many times the median line height, and short, reads as a
heading."""
MAX_HEADING_CHARS = 80
ROW_GAP = 2.0
"""Pieces on one row join into one line when the gap between them is at most this many
line heights (a label and its value); wider gaps are separate columns."""
LINE_GAP = 0.6
"""A line continues the paragraph above when the gap between them is at most this many
line heights, their left edges align and their heights match."""


@dataclass
class _Line:
    text: str
    bbox: BBox | None

    @property
    def height(self) -> float:
        return self.bbox.y1 - self.bbox.y0 if self.bbox else 0.0


@dataclass
class _Block:
    type: ComponentType
    lines: list[_Line]
    children: list[_Block] = field(default_factory=list["_Block"])


def _union(boxes: Sequence[BBox | None]) -> BBox | None:
    if not boxes or any(b is None for b in boxes):
        return None
    placed = [b for b in boxes if b is not None]
    return BBox(
        x0=min(b.x0 for b in placed),
        y0=min(b.y0 for b in placed),
        x1=max(b.x1 for b in placed),
        y1=max(b.y1 for b in placed),
    )


def _rows(texts: Sequence[ImageText]) -> list[_Line]:
    """The pieces in reading order, top to bottom and left to right, with pieces close
    together on one row joined into a line."""
    placed = sorted(
        (t for t in texts if t.bbox is not None),
        key=lambda t: (t.bbox.y0, t.bbox.x0) if t.bbox else (0.0, 0.0),
    )
    rows: list[list[ImageText]] = []
    for piece in placed:
        assert piece.bbox is not None
        middle = (piece.bbox.y0 + piece.bbox.y1) / 2
        row = rows[-1] if rows else None
        if row is not None and (last := row[-1].bbox) is not None and last.y0 <= middle <= last.y1:
            row.append(piece)
        else:
            rows.append([piece])
    lines: list[_Line] = []
    for row in rows:
        row.sort(key=lambda t: t.bbox.x0 if t.bbox else 0.0)
        current = _Line(row[0].text, row[0].bbox)
        for piece in row[1:]:
            assert current.bbox is not None
            assert piece.bbox is not None
            height = max(current.height, piece.bbox.y1 - piece.bbox.y0)
            if piece.bbox.x0 - current.bbox.x1 <= ROW_GAP * height:
                current = _Line(f"{current.text} {piece.text}", _union([current.bbox, piece.bbox]))
            else:
                lines.append(current)
                current = _Line(piece.text, piece.bbox)
        lines.append(current)
    return lines


def _continues(block: _Block, line: _Line) -> bool:
    """Whether ``line`` wraps on from the paragraph ``block``."""
    last = block.lines[-1]
    if block.type != "paragraph" or last.bbox is None or line.bbox is None:
        return False
    if is_key_value(last.text) or is_key_value(line.text):
        return False  # each pair is a line of its own
    height = max(last.height, line.height)
    first = block.lines[0].bbox
    assert first is not None
    return (
        0 <= line.bbox.y0 - last.bbox.y1 <= LINE_GAP * height
        and abs(line.bbox.x0 - first.x0) <= height
        and min(last.height, line.height) >= 0.8 * height
    )


def _blocks(texts: Sequence[ImageText]) -> list[_Block]:
    if any(t.bbox is None for t in texts):
        # Without boxes there's no layout to read: each piece is a paragraph, in order.
        return [_Block("paragraph", [_Line(t.text, t.bbox)]) for t in texts]
    lines = _rows(texts)
    median = statistics.median(line.height for line in lines) if lines else 0.0
    out: list[_Block] = []
    section: _Block | None = None
    for line in lines:
        heading = (
            len(lines) > 1
            and line.height >= HEADING_SCALE * median
            and len(line.text) <= MAX_HEADING_CHARS
            and not is_key_value(line.text)
        )
        if heading:
            section = _Block("section", [], children=[_Block("heading", [line])])
            out.append(section)
            continue
        siblings = section.children if section is not None else out
        if siblings and _continues(siblings[-1], line):
            siblings[-1].lines.append(line)
        else:
            siblings.append(_Block("paragraph", [line]))
    return out


def text_components(
    texts: Sequence[ImageText], image: Component, data: ImageData, *, start: int = 0
) -> list[Component]:
    """Text read from ``image`` as its child components.

    Pieces are put in reading order, top to bottom and left to right, and pieces close
    together on one row become one line ("0-62 mph" and "7.9 s"). Lines that wrap (just
    below the one before, left-aligned, the same height) join into a paragraph; a
    ``Label: value`` line stays a paragraph of its own. A short line at least
    :data:`HEADING_SCALE` times the median line height is a heading, opening a section
    that holds the lines after it, up to the next heading. With a piece lacking a box,
    there's no layout to read, and each piece is a paragraph, in the order given.

    Ids are ``<image id>-t<n>`` from ``n = start``, in reading order.
    """
    counter = start

    def convert(block: _Block, trail: list[str]) -> tuple[Component, BBox | None]:
        nonlocal counter
        component_id = f"{image.id}-t{counter}"
        counter += 1
        children: list[Component] = []
        boxes = [line.bbox for line in block.lines]
        inner = trail
        for child in block.children:
            component, box = convert(child, inner)
            children.append(component)
            boxes.append(box)
            if child.type == "heading":
                inner = [*trail, component.text]
        bbox = _union(boxes)
        component = Component(
            id=component_id,
            type=block.type,
            text=" ".join(line.text for line in block.lines),
            children=children,
            heading_trail=trail,
            location=data.location(bbox),
        )
        return component, bbox

    return [convert(block, list(image.heading_trail))[0] for block in _blocks(texts)]


def vision_components(
    statements: Sequence[ImageText], image: Component, data: ImageData, *, start: int = 0
) -> tuple[list[Component], list[Statement]]:
    """A vision model's statements about ``image``: one paragraph child each, so the
    component gate sees the text, and the ``vision`` statement on it (cut like any other
    when too long). Ids are ``<image id>-v<n>`` from ``n = start``."""
    components: list[Component] = []
    out: list[Statement] = []
    for n, said in enumerate(statements, start):
        location = data.location(said.bbox)
        component = Component(
            id=f"{image.id}-v{n}",
            type="paragraph",
            text=said.text,
            heading_trail=list(image.heading_trail),
            location=location,
        )
        components.append(component)
        statement = Statement(
            id=f"{component.id}.0",
            text=said.text,
            kind="vision",
            component_id=component.id,
            heading_trail=list(image.heading_trail),
            location=location,
        )
        out.extend(cut_statement(statement))
    return components, out


# --- The stage ---------------------------------------------------------------------------


def _ocr_installed() -> bool:
    """Whether RapidOCR and ONNX Runtime (the ``ocr`` extra) are importable."""
    return all(importlib.util.find_spec(m) is not None for m in ("rapidocr", "onnxruntime"))


def _default_processors() -> list[ImageProcessor]:
    return [OcrProcessor()] if _ocr_installed() else []


def _image_document_root(document: Document) -> Component:
    """The tree for an image document: a section holding the one image."""
    location = ImageLocation(src=document.url)
    image = Component(id="c1", type="image", location=location, src=document.url)
    return Component(id="c0", type="section", children=[image], location=location)


def _first_page(component: Component) -> int:
    location = component.location
    return location.page if isinstance(location, PageLocation) else 0


@dataclass
class _Read:
    image: Component
    data: ImageData | None = None
    readings: list[ImageReading] = field(default_factory=list[ImageReading])
    error: str | None = None


@dataclass
class ImageStage:
    """Reads the text in a document's images (stage 6; see the module docstring).

    ``processors`` default to OCR when the ``ocr`` extra is installed, and to none
    otherwise. Add a vision model by passing it alongside:
    ``ImageStage(processors=[OcrProcessor(), MyVisionModel()])``. At most ``max_images``
    images are read per document, in reading order; an ``images_capped`` event counts the
    rest.
    """

    processors: list[ImageProcessor] = field(default_factory=_default_processors)
    loader: ImageLoader = field(default_factory=DefaultImageLoader)
    max_images: int = MAX_IMAGES
    name: str = "images"

    def __post_init__(self) -> None:
        if self.max_images < 0:
            raise ValueError(f"max_images must not be negative, got {self.max_images}")

    async def run(self, ctx: Context) -> None:
        document = ctx.document
        if ctx.parsed is None and not document.is_image:
            return
        if not self.processors:
            root = ctx.parsed.root if ctx.parsed else _image_document_root(document)
            count = sum(c.type == "image" for c in root.walk())
            if count:
                ctx.event(
                    self.name,
                    "images_skipped",
                    f"{count} image(s) not read: no image processor (install jevex[ocr] for OCR)",
                    images=count,
                )
            return
        if document.is_pdf and not _pdfium_installed():
            ctx.event(
                self.name,
                "images_skipped",
                "PDF images not read: rendering them needs pypdfium2 (install jevex[pdf])",
            )
            return
        if ctx.parsed is None:
            ctx.parsed = ParsedDocument(document=document, root=_image_document_root(document))
        parsed = ctx.parsed
        images = self._images(parsed.root, document)
        if len(images) > self.max_images:
            ctx.event(
                self.name,
                "images_capped",
                f"read the first {self.max_images} of {len(images)} images",
                unread=len(images) - self.max_images,
            )
            images = images[: self.max_images]
        reads = await gather(self._read(image, document) for image in images)

        unread = {r.image.id: r.error for r in reads if r.error is not None}
        if unread:
            ctx.event(
                self.name,
                "images_unread",
                f"{len(unread)} image(s) couldn't be read",
                images=unread,
            )
        not_loaded = [r.image.id for r in reads if r.data is None and r.error is None]
        if not_loaded:
            ctx.event(
                self.name,
                "images_not_loaded",
                f"{len(not_loaded)} image(s) have no source to load (remote images need "
                "DefaultImageLoader(fetcher=...))",
                images=not_loaded,
            )
        for read in reads:
            if read.data is not None:
                self._attach(parsed, read.image, read.data, read.readings)

    def _images(self, root: Component, document: Document) -> list[Component]:
        """The images to read, in reading order. For a PDF, each page without a text
        layer is added to the tree as an image, and the pictures on it are left out."""
        scanned = pages_without_text(document) if document.is_pdf else []
        pages: set[str] = set()
        for page in scanned:
            index = next(
                (i for i, c in enumerate(root.children) if _first_page(c) > page),
                len(root.children),
            )
            whole = Component(id=f"page{page}", type="image", location=PageLocation(page=page))
            root.children.insert(index, whole)
            pages.add(whole.id)
        return [
            c
            for c in root.walk()
            if c.type == "image" and (c.id in pages or _first_page(c) not in scanned)
        ]

    async def _read(self, image: Component, document: Document) -> _Read:
        read = _Read(image)
        try:
            read.data = await self.loader.load(image, document)
            if read.data is not None:
                data = read.data
                read.readings = await gather(p.process(image, data) for p in self.processors)
        except UnreadableImageError as exc:
            read.error = str(exc)
        return read

    def _attach(
        self,
        parsed: ParsedDocument,
        image: Component,
        data: ImageData,
        readings: list[ImageReading],
    ) -> None:
        texts = 0
        seen = 0
        for reading in readings:
            children = text_components(reading.text, image, data, start=texts)
            texts += sum(1 for c in children for _ in c.walk())
            components, statements = vision_components(reading.statements, image, data, start=seen)
            seen += len(components)
            image.children.extend([*children, *components])
            for statement in statements:
                parsed.statements[statement.id] = statement
