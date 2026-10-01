"""The brand assets in docs/brand/ are hand-edited SVGs plus renders of them, so a broken
file or a stale link would otherwise only show up on GitHub."""

from __future__ import annotations

import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "docs" / "brand"
SVG = "{http://www.w3.org/2000/svg}"
SVGS = sorted(p.name for p in BRAND.glob("*.svg"))
ILLUSTRATIONS = BRAND / "illustrations"
DRAW = ROOT / "scripts" / "draw_illustrations.py"
# The brand sheet's colours, plus the white and black the mascot's eyes and shadow use.
PALETTE = {
    "#1E1B4B", "#0F0D24", "#F5F3FF", "#7C3AED", "#5B21B6", "#8B5CF6", "#A78BFA",
    "#C4B5FD", "#DDD6FE", "#EDE9FE", "#34D399", "#064E3B", "#F59E0B", "#4F46E5",
    "#FFFFFF", "#000000",
}  # fmt: skip


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


def draw(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DRAW), *args], capture_output=True, text=True, check=False
    )


def test_illustrations_match_their_script() -> None:
    result = draw("--check")
    assert result.returncode == 0, result.stdout


def test_draw_check_reports_stale_and_stray_files(tmp_path: Path) -> None:
    missing = draw("--check", "--out", str(tmp_path))
    assert missing.returncode == 1
    assert "stale:" in missing.stdout
    assert list(tmp_path.iterdir()) == []

    assert draw("--out", str(tmp_path)).returncode == 0
    assert draw("--check", "--out", str(tmp_path)).returncode == 0

    (tmp_path / "pipeline.svg").write_text("<svg/>")
    (tmp_path / "old-drawing.svg").write_text("<svg/>")
    stale = draw("--check", "--out", str(tmp_path))
    assert stale.returncode == 1
    assert f"stale: {tmp_path / 'pipeline.svg'}" in stale.stdout
    assert f"not drawn by this script: {tmp_path / 'old-drawing.svg'}" in stale.stdout


def test_each_illustration_has_every_variant() -> None:
    names = {p.name for p in ILLUSTRATIONS.glob("*.svg")}
    assert names == {
        f"{drawing}{kind}{theme}.svg"
        for drawing in ("pipeline", "learning-loop")
        for kind in ("", "-animated")
        for theme in ("", "-dark")
    }


@pytest.mark.parametrize("path", sorted(ILLUSTRATIONS.glob("*.svg")), ids=lambda p: p.name)
def test_illustrations_are_labelled_and_on_palette(path: Path) -> None:
    svg = ET.parse(path).getroot()
    assert svg.get("role") == "img"
    assert svg.get("aria-label")
    assert svg.find(f"{SVG}title") is not None
    colours = {c.upper() for c in re.findall(r"#[0-9A-Fa-f]{6}\b", path.read_text())}
    assert colours <= PALETTE


@pytest.mark.parametrize("path", sorted(ILLUSTRATIONS.glob("*.svg")), ids=lambda p: p.name)
def test_only_animated_illustrations_move(path: Path) -> None:
    styles = ET.parse(path).getroot().findall(f"{SVG}style")
    if "-animated" not in path.name:
        assert styles == []
        return
    (style,) = styles
    css = style.text or ""
    assert "@keyframes" in css
    # GitHub shows README images through <img>, where scripts and SMIL don't run.
    assert "<script" not in path.read_text()
    assert "<animate" not in path.read_text()
    assert "@media (prefers-reduced-motion: reduce)" in css


@pytest.mark.parametrize("theme", ["", "-dark"])
@pytest.mark.parametrize("drawing", ["pipeline", "learning-loop"])
def test_animated_illustrations_rest_on_the_static_picture(drawing: str, theme: str) -> None:
    """With animations off (reduced motion), an animated file must show what the static
    one shows: the same elements, apart from ones hidden at rest."""

    def picture(svg: str) -> str:
        svg = re.sub(r"\n\s*<style>.*?</style>", "", svg, flags=re.S)
        svg = re.sub(r' class="[^"]*"', "", svg)
        return re.sub(r'\n\s*<rect [^>]*opacity="0"[^>]*/>', "", svg)

    still = (ILLUSTRATIONS / f"{drawing}{theme}.svg").read_text()
    moving = (ILLUSTRATIONS / f"{drawing}-animated{theme}.svg").read_text()
    assert picture(moving) == picture(still)
