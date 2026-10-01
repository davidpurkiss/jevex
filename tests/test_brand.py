"""The brand assets in docs/brand/ are hand-edited SVGs plus renders of them, so a broken
file or a stale link would otherwise only show up on GitHub."""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "docs" / "brand"
SVG = "{http://www.w3.org/2000/svg}"
SVGS = sorted(p.name for p in BRAND.glob("*.svg"))


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def ico_sizes(data: bytes) -> set[int]:
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    if (reserved, kind) != (0, 1):
        raise ValueError("not an ICO")
    # A width byte of 0 means 256.
    return {data[6 + 16 * i] or 256 for i in range(count)}


def odd_edges(svg: ET.Element) -> list[str]:
    """Shapes whose edges fall between pixels when a 32-unit drawing is shown at 16 px."""
    problems: list[str] = []
    for rect in svg.iter(f"{SVG}rect"):
        x, y = float(rect.get("x", 0)), float(rect.get("y", 0))
        w, h = float(rect.get("width", 0)), float(rect.get("height", 0))
        if any(edge % 2 for edge in (x, y, x + w, y + h)):
            problems.append(f"rect {x} {y} {w} {h}")
    for circle in svg.iter(f"{SVG}circle"):
        cx, cy, r = (float(circle.get(a, 0)) for a in ("cx", "cy", "r"))
        if any(edge % 2 for edge in (cx - r, cx + r, cy - r, cy + r)):
            problems.append(f"circle {cx} {cy} {r}")
    return problems


def missing_links(markdown: str, base: Path) -> list[str]:
    targets = re.findall(r"\]\(([^)#\s]+)\)|srcset=\"([^\"]+)\"|src=\"([^\"]+)\"", markdown)
    links = [t for groups in targets for t in groups if t and "://" not in t]
    # Snippets for the repo README use repo-root paths.
    return [
        link
        for link in links
        if not (ROOT / link if link.startswith("docs/") else base / link).exists()
    ]


def test_brand_assets_exist() -> None:
    expected = {
        "jevex-logo.svg",
        "jevex-logo-dark.svg",
        "jevex-mark.svg",
        "jevex-mark-dark.svg",
        "jevex-icon.svg",
        "favicon.svg",
        "mascot.svg",
        "social-preview.svg",
        "palette.svg",
    }
    assert expected <= set(SVGS)
    assert (BRAND / "favicon.ico").is_file()
    assert (BRAND / "jevex-icon-512.png").is_file()


@pytest.mark.parametrize("name", SVGS)
def test_svgs_parse_and_are_labelled(name: str) -> None:
    svg = ET.parse(BRAND / name).getroot()
    assert svg.tag == f"{SVG}svg"
    assert svg.get("viewBox")
    assert svg.get("role") == "img"
    assert svg.get("aria-label")
    assert svg.find(f"{SVG}title") is not None


def test_social_preview_is_github_size() -> None:
    assert png_size((BRAND / "social-preview.png").read_bytes()) == (1280, 640)
    svg = ET.parse(BRAND / "social-preview.svg").getroot()
    assert svg.get("viewBox") == "0 0 1280 640"


def test_icon_render_is_512() -> None:
    assert png_size((BRAND / "jevex-icon-512.png").read_bytes()) == (512, 512)


def test_favicon_ico_holds_each_size() -> None:
    assert ico_sizes((BRAND / "favicon.ico").read_bytes()) == {16, 32, 48}


def test_favicon_is_crisp_at_16px() -> None:
    svg = ET.parse(BRAND / "favicon.svg").getroot()
    assert svg.get("viewBox") == "0 0 32 32"
    assert odd_edges(svg) == []


def test_odd_edges_flags_half_pixels() -> None:
    svg = ET.fromstring(
        f'<svg xmlns="{SVG[1:-1]}"><rect x="5" y="4" width="22" height="4"/>'
        '<circle cx="16" cy="26" r="3"/></svg>'
    )
    assert odd_edges(svg) == ["rect 5.0 4.0 22.0 4.0", "circle 16.0 26.0 3.0"]


def test_binary_readers_reject_other_files() -> None:
    with pytest.raises(ValueError, match="not a PNG"):
        png_size((BRAND / "favicon.ico").read_bytes())
    with pytest.raises(ValueError, match="not an ICO"):
        ico_sizes((BRAND / "social-preview.png").read_bytes())


@pytest.mark.parametrize("readme", ["README.md", "concepts/README.md"])
def test_readme_links_resolve(readme: str) -> None:
    path = BRAND / readme
    assert missing_links(path.read_text(), path.parent) == []


def test_missing_links_reports_broken_targets() -> None:
    markdown = (
        "![a](palette.svg) [b](nope.svg) [c](https://example.com/x.svg) "
        '<img src="docs/brand/jevex-logo.svg"> <img src="docs/brand/nope.svg">'
    )
    assert missing_links(markdown, BRAND) == ["nope.svg", "docs/brand/nope.svg"]
