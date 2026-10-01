# /// script
# requires-python = ">=3.12"
# dependencies = ["resvg-py>=0.3", "pillow>=11"]
# ///
"""Render the raster brand assets in docs/brand/ from their SVG sources.

Run `uv run scripts/render_brand.py` after changing an SVG there. The SVGs are the
sources; the PNGs and the ICO are committed renders of them. The social preview's text
uses installed fonts: Nunito if present, else Avenir Next (macOS), else system-ui. Its
committed PNG was rendered with Avenir Next, so a render elsewhere differs slightly.
"""

from __future__ import annotations

import io
from pathlib import Path

import resvg_py
from PIL import Image

BRAND = Path(__file__).resolve().parent.parent / "docs" / "brand"
FAVICON_SIZES = (16, 32, 48)


def render(name: str, size: int) -> bytes:
    return bytes(resvg_py.svg_to_bytes(svg_path=str(BRAND / name), width=size, height=size))


def main() -> None:
    # Each size is rendered from the SVG rather than downscaled, so 16 px stays crisp.
    icons = [Image.open(io.BytesIO(render("favicon.svg", size))) for size in FAVICON_SIZES]
    icons[-1].save(
        BRAND / "favicon.ico",
        sizes=[(size, size) for size in FAVICON_SIZES],
        append_images=icons[:-1],
    )
    (BRAND / "jevex-icon-512.png").write_bytes(render("jevex-icon.svg", 512))
    social = resvg_py.svg_to_bytes(svg_path=str(BRAND / "social-preview.svg"))
    (BRAND / "social-preview.png").write_bytes(bytes(social))
    for name in ("favicon.ico", "jevex-icon-512.png", "social-preview.png"):
        print(f"docs/brand/{name}")


if __name__ == "__main__":
    main()
