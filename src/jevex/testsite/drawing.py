"""A tiny page-drawing model the test site writes as a PDF, a PNG or a "scanned" PDF.

A :class:`Drawing` is text, rules and filled boxes in points, with the origin at the top
left and text placed by its baseline. :func:`to_pdf` writes it as a one-page PDF with a
text layer (Helvetica, WinAnsi), byte for byte the same every time: no dates, no IDs.
:func:`to_png` and :func:`to_scanned_pdf` rasterise it with Pillow (the ``testsite``
extra), the second as an image-only PDF with a scanner's tilt, paper tone, speckle and
blur. Rasterised text is set in Pillow's bundled font, so it's a little narrower or wider
than in the PDF; layouts leave room for that. That font has no ``£`` or en dash, so a
drawing meant for rasterising is ``plain``: its text is ASCII (:func:`plain_text`).
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from functools import cache, lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image, ImageFont

RGB = tuple[float, float, float]
"""A colour, each channel from 0 to 1."""

BLACK: RGB = (0.0, 0.0, 0.0)
PRODUCER = "jevex test site"
PLAIN = {"£": "GBP ", "–": "-"}


def plain_text(text: str) -> str:
    """``text`` with the characters Pillow's bundled font lacks spelled in ASCII."""
    for char, ascii_text in PLAIN.items():
        text = text.replace(char, ascii_text)
    return text


@dataclass(frozen=True)
class Text:
    """A run of text with its baseline starting at (``x``, ``y``)."""

    x: float
    y: float
    text: str
    size: float
    bold: bool = False
    fill: RGB = BLACK


@dataclass(frozen=True)
class Rule:
    """A straight line."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float = 0.5
    stroke: RGB = BLACK


@dataclass(frozen=True)
class Box:
    """A filled rectangle with its top-left corner at (``x``, ``y``)."""

    x: float
    y: float
    w: float
    h: float
    fill: RGB


Op = Text | Rule | Box


@dataclass
class Drawing:
    """One page: its size in points and what's on it, painted in order.

    A ``plain`` drawing writes its text through :func:`plain_text`.
    """

    width: float
    height: float
    title: str = ""
    ops: list[Op] = field(default_factory=list[Op])
    plain: bool = False

    def shown(self, text: str) -> str:
        """``text`` as :meth:`text` would draw it."""
        return plain_text(text) if self.plain else text

    def text(
        self, x: float, y: float, text: str, size: float, *, bold: bool = False, fill: RGB = BLACK
    ) -> None:
        self.ops.append(Text(x, y, self.shown(text), size, bold, fill))

    def rule(self, x0: float, y0: float, x1: float, y1: float, width: float = 0.5) -> None:
        self.ops.append(Rule(x0, y0, x1, y1, width))

    def box(self, x: float, y: float, w: float, h: float, fill: RGB) -> None:
        self.ops.append(Box(x, y, w, h, fill))

    @property
    def texts(self) -> list[Text]:
        return [op for op in self.ops if isinstance(op, Text)]


def text_width(text: str, size: float, *, bold: bool = False) -> float:
    """A generous estimate of ``text``'s width in points, for laying out columns."""
    return len(text) * size * (0.62 if bold else 0.58)


# --- PDF -------------------------------------------------------------------------------


def to_pdf(drawing: Drawing) -> bytes:
    """``drawing`` as a one-page PDF with real (selectable) text."""
    content = "\n".join(_pdf_op(op, drawing.height) for op in drawing.ops).encode("latin-1")
    fonts = "/Font << /F1 4 0 R /F2 5 0 R >>"
    return _pdf(drawing, content, fonts, extra=[])


def _pdf_op(op: Op, height: float) -> str:
    match op:
        case Text():
            font = "/F2" if op.bold else "/F1"
            return (
                f"BT {font} {_n(op.size)} Tf {_rgb(op.fill)} rg "
                f"{_n(op.x)} {_n(height - op.y)} Td ({_pdf_string(op.text)}) Tj ET"
            )
        case Rule():
            return (
                f"{_rgb(op.stroke)} RG {_n(op.width)} w {_n(op.x0)} {_n(height - op.y0)} m "
                f"{_n(op.x1)} {_n(height - op.y1)} l S"
            )
        case Box():
            return (
                f"{_rgb(op.fill)} rg {_n(op.x)} {_n(height - op.y - op.h)} "
                f"{_n(op.w)} {_n(op.h)} re f"
            )


def _pdf(drawing: Drawing, content: bytes, resources: str, extra: list[bytes]) -> bytes:
    """Assemble the objects: catalog, pages, page, two fonts, content, info, then ``extra``
    (numbered from 8)."""
    font = "<< /Type /Font /Subtype /Type1 /BaseFont /{} /Encoding /WinAnsiEncoding >>"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_n(drawing.width)} "
            f"{_n(drawing.height)}] /Resources << {resources} >> /Contents 6 0 R >>"
        ).encode(),
        font.format("Helvetica").encode(),
        font.format("Helvetica-Bold").encode(),
        _stream(b"", content),
        (f"<< /Title ({_pdf_string(drawing.title)}) /Producer ({PRODUCER}) >>".encode("latin-1")),
        *extra,
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    out.write("".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 7 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def _stream(dictionary: bytes, data: bytes) -> bytes:
    head = b"<< " + dictionary + f"/Length {len(data)} >>\nstream\n".encode()
    return head + data + b"\nendstream"


def _pdf_string(text: str) -> str:
    """``text`` in WinAnsi, escaped for a PDF literal string (non-ASCII as octal)."""
    out: list[str] = []
    for byte in text.encode("cp1252"):
        char = chr(byte)
        if char in "\\()":
            out.append("\\" + char)
        elif 32 <= byte < 127:
            out.append(char)
        else:
            out.append(f"\\{byte:03o}")
    return "".join(out)


def _n(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _rgb(colour: RGB) -> str:
    return " ".join(_n(c) for c in colour)


# --- raster ----------------------------------------------------------------------------


def rasterise(drawing: Drawing, dpi: float = 150) -> Image.Image:
    """``drawing`` as an RGB image at ``dpi``. Raises ``ValueError`` for text that isn't
    ASCII, which the font may not have (make the drawing ``plain``)."""
    for op in drawing.texts:
        if not op.text.isascii():
            raise ValueError(f"can't rasterise {op.text!r}: not ASCII")
    _need_pillow()
    from PIL import Image, ImageDraw

    scale = dpi / 72
    size = (round(drawing.width * scale), round(drawing.height * scale))
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for op in drawing.ops:
        match op:
            case Box():
                draw.rectangle(
                    (op.x * scale, op.y * scale, (op.x + op.w) * scale, (op.y + op.h) * scale),
                    fill=_rgb255(op.fill),
                )
            case Rule():
                draw.line(
                    (op.x0 * scale, op.y0 * scale, op.x1 * scale, op.y1 * scale),
                    fill=_rgb255(op.stroke),
                    width=max(1, round(op.width * scale)),
                )
            case Text():
                px = op.size * scale
                draw.text(
                    (op.x * scale, op.y * scale),
                    op.text,
                    font=_font(round(px)),
                    fill=_rgb255(op.fill),
                    anchor="ls",
                    # Pillow's default font has no bold face; thicken large bold text.
                    stroke_width=1 if op.bold and px >= 24 else 0,
                    stroke_fill=_rgb255(op.fill),
                )
    return image


def to_png(drawing: Drawing, dpi: float = 144) -> bytes:
    """``drawing`` as a PNG."""
    return _png(drawing.width, drawing.height, tuple(drawing.ops), dpi)


def to_scanned_pdf(drawing: Drawing, seed: str, dpi: float = 150) -> bytes:
    """``drawing`` printed and scanned: a JPEG of the page in grey, slightly tilted, on
    off-white paper with speckle and a little blur, in a PDF with no text layer. ``seed``
    picks the tilt, paper and speckle."""
    width, height, title = drawing.width, drawing.height, drawing.title
    return _scanned_pdf(width, height, title, tuple(drawing.ops), seed, dpi)


# Rasterising is slow next to the rest of a build and a site renders the same drawings
# again and again (every build, every test), so the bytes are cached by what's drawn.


@lru_cache(maxsize=128)
def _png(width: float, height: float, ops: tuple[Op, ...], dpi: float) -> bytes:
    out = io.BytesIO()
    rasterise(Drawing(width, height, ops=list(ops)), dpi).save(out, "PNG")
    return out.getvalue()


@lru_cache(maxsize=128)
def _scanned_pdf(
    width: float, height: float, title: str, ops: tuple[Op, ...], seed: str, dpi: float
) -> bytes:
    _need_pillow()
    from PIL import Image, ImageDraw, ImageFilter

    drawing = Drawing(width, height, title, list(ops))
    rng = random.Random(seed)
    page = rasterise(drawing, dpi).convert("L")
    paper = rng.randint(236, 248)
    page = page.point([round(paper * v / 255) for v in range(256)])
    page = page.rotate(rng.uniform(-1.2, 1.2), resample=Image.Resampling.BILINEAR, fillcolor=paper)
    draw = ImageDraw.Draw(page)
    pixels_w, pixels_h = page.size
    for _ in range(pixels_w * pixels_h // 4000):
        x, y = rng.randrange(pixels_w), rng.randrange(pixels_h)
        r = rng.choice((0, 0, 1))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=rng.randint(60, 160))
    page = page.filter(ImageFilter.GaussianBlur(0.6))
    jpeg = io.BytesIO()
    page.save(jpeg, "JPEG", quality=80)
    image = _stream(
        (
            f"/Type /XObject /Subtype /Image /Width {pixels_w} /Height {pixels_h} "
            "/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode "
        ).encode(),
        jpeg.getvalue(),
    )
    content = f"q {_n(width)} 0 0 {_n(height)} 0 0 cm /Im1 Do Q".encode()
    resources = "/Font << /F1 4 0 R /F2 5 0 R >> /XObject << /Im1 8 0 R >>"
    return _pdf(drawing, content, resources, extra=[image])


def _need_pillow() -> None:
    try:
        import PIL  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as exc:
        raise ImportError(
            "Rendering the test site's images and scanned PDFs needs Pillow: "
            "install jevex[testsite]"
        ) from exc


@cache
def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    from PIL import ImageFont

    return ImageFont.load_default(size=px)


def _rgb255(colour: RGB) -> tuple[int, int, int]:
    r, g, b = (round(c * 255) for c in colour)
    return (r, g, b)
