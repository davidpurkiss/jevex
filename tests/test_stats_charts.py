import itertools
import math
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from jevex.stats import CHART_VIEWS, Point, SpendPoint, Stats, chart_svg, cost_svg
from jevex.stats.charts import (
    CHART_CSS,
    DARK,
    LIGHT,
    learning_svg,
    mix_svg,
    nice_ticks,
    theme_css,
    usd,
)
from jevex.stats.data import METHODS

T0 = datetime(2026, 9, 30, 8, 0, tzinfo=UTC)


def point(documents: int, **kw: Any) -> Point:
    return Point(
        documents=documents,
        size=kw.pop("size", 1),
        llm_calls_per_document=kw.pop("llm", 0.0),
        jev_cost_per_document=kw.pop("jev_cost", 0.001),
        llm_cost_per_document=kw.pop("llm_cost", 0.0),
        **kw,
    )


def replay_stats() -> Stats:
    return Stats(
        source="curve.csv",
        kind="replay",
        points=[
            point(10, size=10, llm=3.0, accuracy=1.0, methods={"llm": 6, "jev": 4}),
            point(20, size=10, llm=1.0, accuracy=0.9, methods={"llm": 2, "generator": 8}),
            point(30, size=10, llm=0.0, accuracy=None, methods={"generator": 9, "jev": 1}),
        ],
        spend=[SpendPoint(10, 0.01, 0.06), SpendPoint(20, 0.02, 0.08), SpendPoint(30, 0.03, 0.08)],
        waves=[(10, 2)],
        learned=[10, 10],
    )


def store_stats() -> Stats:
    return Stats(
        source="sqlite:///x.db",
        kind="store",
        points=[point(i + 1, llm=float(i), at=T0 + timedelta(hours=i)) for i in range(3)],
        spend=[SpendPoint(i + 1, 0.001 * i, 0.0, at=T0 + timedelta(hours=i)) for i in range(3)],
        budget_usd=0.5,
    )


def parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)  # well-formed XML, or this raises


@pytest.mark.parametrize("view", CHART_VIEWS)
@pytest.mark.parametrize("animate", [False, True])
def test_standalone_charts_are_svg_files_with_their_own_styles(view: str, animate: bool) -> None:
    svg = chart_svg(replay_stats(), view, standalone=True, animate=animate)
    root = parse(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "--series-1" in svg  # light and dark colour roles travel with the file
    assert "prefers-color-scheme: dark" in svg
    assert ("animate" in root.attrib["class"].split()) is animate
    assert "prefers-reduced-motion" in svg


def test_an_embedded_chart_leaves_styles_to_the_page() -> None:
    svg = learning_svg(replay_stats())
    assert "<style>" not in svg
    assert "xmlns=" not in svg


def test_the_learning_curve_has_a_panel_per_metric_with_waves_and_ticks() -> None:
    svg = learning_svg(replay_stats())
    titles = re.findall(r'<text class="title"[^>]*>([^<]*)</text>', svg)
    assert titles == ["LLM calls per document", "Cost per document (USD)", "Accuracy"]
    assert svg.count(">wave 2</text>") == 3
    assert svg.count('class="learned"') == 6  # two generators, three panels
    assert "documents 11–20\nLLM calls per document: 1.00" in svg
    assert ">30 documents</text>" in svg


def test_cost_titles_say_when_they_include_the_learners_spend() -> None:
    stats = replay_stats()
    stats.points[0] = point(10, size=10, llm=3.0, learning_jev_cost_per_document=0.002)
    titles = re.findall(r'<text class="title"[^>]*>([^<]*)</text>', learning_svg(stats))
    assert titles[1] == "Cost per document, learning included (USD)"
    assert (
        "Cost per document, learning included (USD)"
        in parse(learning_svg(stats)).attrib["aria-label"]
    )
    cost = cost_svg(stats)
    assert ">Cumulative spend, learning included (USD)</text>" in cost
    assert "learning included" in parse(cost).attrib["aria-label"]
    assert "learning included" not in cost_svg(replay_stats())


def test_a_store_has_no_accuracy_panel_and_a_time_axis() -> None:
    svg = learning_svg(store_stats(), "time")
    assert "Accuracy" not in svg
    assert ">30 Sep 08:00</text>" in svg
    assert ">30 Sep 10:00</text>" in svg
    assert "document 2, 30 Sep 09:00" in svg


def test_learned_ticks_on_the_time_axis_stay_in_the_plot() -> None:
    stats = store_stats()
    stats.learned_at = [T0 + timedelta(minutes=30), T0 + timedelta(days=1)]
    svg = learning_svg(stats, "time")
    assert svg.count('class="learned"') == 2  # one per panel; the late one is off the plot


def test_a_missing_value_breaks_the_line() -> None:
    stats = replay_stats()
    stats.points.append(point(40, size=10, accuracy=0.8))
    accuracy = learning_svg(stats).split('class="title"')[3]
    assert accuracy.count("<polyline") == 1  # 1.0 → 0.9, then a gap, then 0.8 alone
    assert accuracy.count('class="dot"') == 3


def test_the_mix_stacks_methods_in_order_with_labels() -> None:
    svg = mix_svg(replay_stats())
    bands = re.findall(r'class="band m-(\w+)"', svg)
    assert bands == ["jev", "generator", "llm"]  # the methods present, bottom up
    assert "documents 1–10\njev 40% · llm 60%" in svg
    for name in ("structured", "jev", "generator", "llm", "vision"):
        assert f">{name}</text>" in svg  # the legend lists every method


def test_the_animated_mix_reveals_from_the_left() -> None:
    svg = mix_svg(replay_stats(), animate=True)
    assert 'clip-path="url(#jx-mix-clip)"' in svg
    assert 'class="reveal"' in svg


def test_cost_stacks_jev_and_llm_against_the_budget() -> None:
    svg = cost_svg(store_stats(), "time")
    assert re.findall(r'class="band m-(\w+)"', svg) == ["jev"]  # no LLM spend
    assert f">budget {usd(0.5)}</text>" in svg
    assert "Jev $0.002 · LLM $0 · total $0.002" in svg


def band_xs(svg: str) -> list[float]:
    return [
        float(xy.split(",")[0])
        for points in re.findall(r'class="band[^"]*" points="([^"]*)"', svg)
        for xy in points.split()
    ]


def test_spend_while_the_first_document_ran_is_on_the_plot() -> None:
    stats = store_stats()  # the first document finished at T0
    stats.started = T0 - timedelta(seconds=10)
    stats.spend = [
        SpendPoint(1, 0.1, 0.0, at=T0 - timedelta(seconds=8)),
        SpendPoint(1, 0.2, 0.0, at=T0 - timedelta(seconds=1)),
        SpendPoint(2, 0.3, 0.0, at=T0 + timedelta(hours=1)),
    ]
    stats.learned_at = [T0 - timedelta(seconds=5)]
    svg = cost_svg(stats, "time")
    xs = band_xs(svg)
    assert all(72.0 <= x <= 720 - 96 for x in xs)  # inside the plot
    assert ">30 Sep 07:59</text>" in svg
    # The learning curve starts there too, with the generator learned during it.
    assert learning_svg(stats, "time").count('class="learned"') == 2


def test_a_long_ledger_is_thinned_to_the_last_entry() -> None:
    stats = store_stats()
    stats.spend = [SpendPoint(1, 0.001 * i, 0.0) for i in range(1000)]
    svg = cost_svg(stats)
    assert svg.count('<rect class="hit"') == 121  # the start, then 120 points
    assert "total $0.999" in svg


def test_empty_stats_still_draw() -> None:
    empty = Stats(source="s", kind="store", points=[])
    for view in CHART_VIEWS:
        parse(chart_svg(empty, view, standalone=True))
    with pytest.raises(ValueError, match="no times"):
        chart_svg(empty, "learning", "time")


def test_unknown_views_and_axes() -> None:
    with pytest.raises(KeyError):
        chart_svg(replay_stats(), "pie")
    with pytest.raises(ValueError, match="use the documents axis"):
        chart_svg(replay_stats(), "mix", "time")


@pytest.mark.parametrize(
    ("top", "ticks"),
    [
        (0.0, [0.0, 1.0]),
        (3.0, [0.0, 1.0, 2.0, 3.0]),
        (0.0042, [0.0, 0.002, 0.004, 0.006]),
        (7.0, [0.0, 2.0, 4.0, 6.0, 8.0]),
        (10.0, [0.0, 2.5, 5.0, 7.5, 10.0]),
    ],
)
def test_nice_ticks(top: float, ticks: list[float]) -> None:
    assert nice_ticks(top) == ticks


# The dataviz palette checks the stats UI's colours were chosen with: OKLCH lightness
# inside the theme's band, a chroma floor, and OKLab ΔE×100 between neighbouring bands of
# at least 8 under protanopia and deuteranopia (Machado et al. 2009, severity 1) and 15
# under normal vision. WCAG contrast against the surface may dip below 3:1 for a series
# (charts carry direct labels and the page has tables), never for text.
BANDS = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET = 8.0
NORMAL_FLOOR = 15.0
MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
}
THEMES = {"light": LIGHT, "dark": DARK}


def linear(hex_colour: str) -> tuple[float, float, float]:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(int(hex_colour[i : i + 2], 16) / 255) for i in (1, 3, 5))
    return r, g, b


def oklab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    l_ = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * l_ + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l_ - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l_ + 0.7827717662 * m - 0.8086757660 * s,
    )


def simulate(hex_colour: str, kind: str) -> tuple[float, float, float]:
    rgb = linear(hex_colour)
    r, g, b = (
        min(1.0, max(0.0, sum(k * c for k, c in zip(row, rgb, strict=True))))
        for row in MACHADO[kind]
    )
    return r, g, b


def delta_e(a: str, b: str, kind: str | None = None) -> float:
    x = oklab(simulate(a, kind) if kind else linear(a))
    y = oklab(simulate(b, kind) if kind else linear(b))
    return 100 * math.dist(x, y)


def lightness_chroma_hue(hex_colour: str) -> tuple[float, float, float]:
    lightness, a, b = oklab(linear(hex_colour))
    return lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def contrast(a: str, b: str) -> float:
    def luminance(c: str) -> float:
        r, g, b = linear(c)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def palette_failures(colours: list[str], theme: str) -> list[str]:
    """What the palette checks reject in ``colours``, a categorical palette in stacking
    order (only neighbours are compared, as in a stacked chart)."""
    low, high = BANDS[theme]
    failures: list[str] = []
    for c in colours:
        lightness, chroma, _ = lightness_chroma_hue(c)
        if not low <= lightness <= high:
            failures.append(f"{c} lightness {lightness:.3f}")
        if chroma < CHROMA_FLOOR:
            failures.append(f"{c} chroma {chroma:.3f}")
    for a, b in itertools.pairwise(colours):
        cvd = min(delta_e(a, b, kind) for kind in MACHADO)
        if cvd < CVD_TARGET:
            failures.append(f"{a}/{b} colour-blind ΔE {cvd:.1f}")
        if delta_e(a, b) < NORMAL_FLOOR:
            failures.append(f"{a}/{b} ΔE {delta_e(a, b):.1f}")
    return failures


def series(theme: str, *methods: str) -> list[str]:
    return [THEMES[theme][f"series-{METHODS.index(m) + 1}"] for m in methods]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_series_colours_pass_the_palette_checks(theme: str) -> None:
    assert palette_failures(series(theme, *METHODS), theme) == []  # the mix's bands
    assert palette_failures(series(theme, "jev", "llm"), theme) == []  # the cost chart's


def test_the_brand_mint_itself_is_too_light_for_a_chart_band() -> None:
    """Why the generator slot is a darker step of the mint."""
    assert palette_failures(["#34d399"], "light") == ["#34d399 lightness 0.773"]
    assert palette_failures(["#34d399"], "dark") == ["#34d399 lightness 0.773"]
    assert palette_failures(["#7c3aed", "#6366f1"], "light") == [
        "#7c3aed/#6366f1 colour-blind ΔE 3.5",
        "#7c3aed/#6366f1 ΔE 8.7",
    ]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_text_and_accents_are_readable_on_the_surface(theme: str) -> None:
    roles = THEMES[theme]
    surface = roles["surface-1"]
    for role in ("text-primary", "text-secondary", "text-muted"):
        assert contrast(roles[role], surface) >= 4.5, role
    assert contrast(roles["accent"], surface) >= 3.0


def test_the_brand_palette_sets_surfaces_text_and_accents() -> None:
    assert (LIGHT["surface-1"], LIGHT["text-primary"], LIGHT["accent"]) == (
        "#ffffff",
        "#1e1b4b",  # ink
        "#7c3aed",  # violet
    )
    assert (DARK["surface-1"], DARK["text-primary"], DARK["accent"]) == (
        "#0f0d24",  # night
        "#f5f3ff",  # lavender
        "#a78bfa",
    )
    assert LIGHT["series-4"] == "#f59e0b"  # the brand's amber marks LLM spend


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_mint_is_only_for_values_resolved_without_an_llm(theme: str) -> None:
    def minty(c: str) -> bool:
        _, chroma, hue = lightness_chroma_hue(c)
        return chroma >= CHROMA_FLOOR and 140 <= hue <= 180

    mint = {role for role, c in THEMES[theme].items() if minty(c)}
    assert mint == {f"series-{METHODS.index('generator') + 1}"}
    slot = f"var(--series-{METHODS.index('generator') + 1})"
    rules = [rule for rule in CHART_CSS.split("\n") if slot in rule]
    assert rules == ["svg.chart .m-generator { fill: var(--series-3); }"]


def test_theme_css_puts_dark_behind_the_media_query_and_data_theme() -> None:
    css = theme_css({"x": "#fff"}, {"x": "#000"})
    assert css.startswith(".viz-root {\n  color-scheme: light;\n  --x: #fff;\n}\n")
    media = css.index("@media (prefers-color-scheme: dark)")
    assert css.index("--x: #000;") > media
    assert css.count("--x: #000;") == 2
    assert ':root[data-theme="dark"] .viz-root' in css
    assert ':not([data-theme="light"])' in css


def test_the_learning_curve_is_drawn_in_the_accent() -> None:
    svg = learning_svg(replay_stats(), standalone=True)
    assert "svg.chart .line { fill: none; stroke: var(--accent);" in svg
    assert "svg.chart .dot { fill: var(--accent);" in svg
    assert "--accent: #7c3aed;" in svg
