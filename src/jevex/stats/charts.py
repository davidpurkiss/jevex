"""SVG charts for the stats UI's views (spec: *Stats UI › Views*, *Technology*).

The page, the JSON API's ``chart`` route and ``jevex stats export --svg`` all draw with
these functions, so README graphics are the same charts over the same data. Each takes a
:class:`~jevex.stats.data.Stats` and an x-axis and returns one ``<svg>`` element. With
``standalone=True`` it carries its own styles (light and dark themes) for use as a file;
with ``animate=True`` lines draw themselves and areas fill in from the left (CSS
keyframes, which GitHub renders in READMEs), except for readers who prefer reduced
motion.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from jevex.stats.data import CURVE_POINTS, METHODS, curve, shares

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from jevex.stats.data import Point, Stats, XAxis

CHART_VIEWS: tuple[str, ...] = ("learning", "mix", "cost")
"""The views drawn as charts (``jevex stats export --svg <view>``)."""

WIDTH = 720
_LEFT, _RIGHT, _TOP, _BOTTOM = 72, 96, 34, 30
_PANEL = 170

LIGHT: Mapping[str, str] = {
    "surface-1": "#ffffff",
    "text-primary": "#1e1b4b",  # ink
    "text-secondary": "#4c4878",
    "text-muted": "#6b6893",
    "grid": "#ede9fe",
    "axis": "#c4b5fd",
    "accent": "#7c3aed",  # violet
    "series-1": "#818cf8",
    "series-2": "#7c3aed",
    "series-3": "#10b981",
    "series-4": "#f59e0b",  # amber
    "series-5": "#4338ca",
}
"""The light theme's colour roles, from the brand palette (``docs/brand/README.md``): ink
text, violet for accents, and the categorical slots in method order (structured, jev,
generator, llm, vision). Only the generator slot is mint: values resolved without an LLM.
The LLM slot is amber so the two contrast."""

DARK: Mapping[str, str] = {
    "surface-1": "#0f0d24",  # night
    "text-primary": "#f5f3ff",  # lavender
    "text-secondary": "#c4b5fd",
    "text-muted": "#9a95c2",
    "grid": "#1e1b4b",
    "axis": "#3b3775",
    "accent": "#a78bfa",
    "series-1": "#7c86fe",
    "series-2": "#7c3aed",
    "series-3": "#14ac7a",
    "series-4": "#c7800e",
    "series-5": "#5149e8",
}
"""The dark theme's colour roles: night surface, lavender text, and its own steps of the
same hues. Categorical slots in both themes are steps of the brand's hues chosen to pass
the dataviz palette checks against their surface (lightness band, chroma floor,
colour-blind separation of neighbouring bands), so the mint and amber are darker steps
than the brand's ``#34D399`` and ``#F59E0B``, which are too light for a dark chart."""


def theme_css(light: Mapping[str, str], dark: Mapping[str, str]) -> str:
    """CSS custom properties on ``.viz-root``: ``light`` by default, ``dark`` when the
    reader prefers it, and either when the document root says ``data-theme``."""

    def block(selector: str, scheme: str, roles: Mapping[str, str], indent: str) -> str:
        lines = [f"{indent}color-scheme: {scheme};"]
        lines += [f"{indent}--{name}: {value};" for name, value in roles.items()]
        return f"{selector} {{\n" + "\n".join(lines) + f"\n{indent[:-2]}}}\n"

    dark_auto = (
        ':root:where(:not([data-theme="light"])) .viz-root,\n'
        '  :root.viz-root:where(:not([data-theme="light"]))'
    )
    return (
        block(".viz-root", "light", light, "  ")
        + "@media (prefers-color-scheme: dark) {\n  "
        + block(dark_auto, "dark", dark, "    ")
        + "}\n"
        + block(
            ':root[data-theme="dark"] .viz-root, :root[data-theme="dark"].viz-root',
            "dark",
            dark,
            "  ",
        )
    )


PALETTE_CSS = theme_css(LIGHT, DARK)
"""The colour roles, light and dark (:data:`LIGHT`, :data:`DARK`)."""

CHART_CSS = """\
svg.chart { display: block; max-width: 100%; height: auto; overflow: visible;
  font: 12px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif; }
svg.chart .bg { fill: var(--surface-1); }
svg.chart text { fill: var(--text-secondary); font-size: 12px; }
svg.chart text.title { fill: var(--text-primary); font-size: 13px; font-weight: 600; }
svg.chart text.muted { fill: var(--text-muted); }
svg.chart .grid { stroke: var(--grid); stroke-width: 1; }
svg.chart .axis { stroke: var(--axis); stroke-width: 1; }
svg.chart .wave { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 3 3; }
svg.chart .learned { stroke: var(--text-primary); stroke-width: 2; }
svg.chart .budget { stroke: var(--text-secondary); stroke-width: 1.5; stroke-dasharray: 6 4; }
svg.chart .line { fill: none; stroke: var(--accent); stroke-width: 2;
  stroke-linejoin: round; stroke-linecap: round; }
svg.chart .dot { fill: var(--accent); stroke: var(--surface-1); stroke-width: 2; }
svg.chart .hit { fill: transparent; }
svg.chart .point:hover .dot { stroke: var(--text-primary); }
svg.chart .band { stroke: var(--surface-1); stroke-width: 2; stroke-linejoin: round; }
svg.chart .m-structured { fill: var(--series-1); }
svg.chart .m-jev { fill: var(--series-2); }
svg.chart .m-generator { fill: var(--series-3); }
svg.chart .m-llm { fill: var(--series-4); }
svg.chart .m-vision { fill: var(--series-5); }
@keyframes jx-draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }
@keyframes jx-reveal { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes jx-fade { from { opacity: 0; } to { opacity: 1; } }
svg.animate .line { stroke-dasharray: 1; stroke-dashoffset: 1;
  animation: jx-draw 2.4s ease-out forwards; }
svg.animate .reveal { transform-box: fill-box; transform-origin: left; transform: scaleX(0);
  animation: jx-reveal 2.4s ease-out forwards; }
svg.animate .late { opacity: 0; animation: jx-fade 0.4s ease-out 2.4s forwards; }
@media (prefers-reduced-motion: reduce) {
  svg.animate .line, svg.animate .reveal, svg.animate .late {
    animation: none; stroke-dashoffset: 0; transform: none; opacity: 1; }
}
"""


def nice_ticks(top: float) -> list[float]:
    """Ticks from 0 to the smallest 1, 2, 2.5 or 5 × 10ⁿ step multiple at or above ``top``,
    three to five of them (``[0, 1]`` for a ``top`` of 0 or less)."""
    if top <= 0:
        return [0.0, 1.0]
    raw = top / 4
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    count = math.ceil(top / step - 1e-9)
    return [round(i * step, 12) for i in range(count + 1)]


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def usd(x: float) -> str:
    text = f"{x:.5f}".rstrip("0").rstrip(".") if abs(x) < 0.01 else f"{x:.3f}"
    return f"${text}"


def calls(x: float) -> str:
    return f"{x:.2f}"


@dataclass(frozen=True)
class _Metric:
    key: str
    title: str
    fmt: Callable[[float], str]
    top: float | None = None
    """A fixed top for the y-axis (accuracy's 100%); otherwise from the data."""


COST = _Metric("cost_per_document", "Cost per document (USD)", usd)
LEARNING_METRICS = (
    _Metric("llm_calls_per_document", "LLM calls per document", calls),
    COST,
    _Metric("accuracy", "Accuracy", pct, top=1.0),
)
LEARNING_COST_TITLE = "Cost per document, learning included (USD)"
"""The cost panel's title when the points count the learner's spend (a replay's)."""


def _learning_counted(stats: Stats) -> bool:
    return any(p.learning_counted for p in stats.points)


class _Frame:
    """One plot area: maps data to SVG coordinates and draws its axes."""

    def __init__(
        self, stats: Stats, axis: XAxis, top: float, y_top: float, extra: Sequence[float] = ()
    ) -> None:
        self.axis = axis
        self.top = top
        self.y_top = y_top or 1.0
        self.height = _PANEL - _TOP - _BOTTOM
        self.width = WIDTH - _LEFT - _RIGHT
        points = stats.points
        if axis == "docs":
            self.x0, self.x1 = 0.0, float(max(stats.documents, 1))
        elif not stats.has_time:
            raise ValueError(f"{stats.source} has no times: use the documents axis")
        else:
            self.x0, self.x1 = points[0].x("time"), points[-1].x("time")
            if stats.started is not None:
                self.x0 = min(self.x0, stats.started.timestamp())
        if extra:  # a series reaching past the documents (the learner's later spend)
            self.x0, self.x1 = min(self.x0, *extra), max(self.x1, *extra)
        if self.x1 <= self.x0:
            self.x1 = self.x0 + 1.0

    def x(self, value: float) -> float:
        return _LEFT + self.width * (value - self.x0) / (self.x1 - self.x0)

    def y(self, value: float) -> float:
        return self.top + _TOP + self.height * (1 - value / self.y_top)

    @property
    def bottom(self) -> float:
        return self.top + _TOP + self.height

    def axes(self, ticks: Sequence[float], fmt: Callable[[float], str]) -> list[str]:
        parts = [
            f'<line class="grid" x1="{_LEFT}" x2="{WIDTH - _RIGHT}" y1="{self.y(t):.1f}" '
            f'y2="{self.y(t):.1f}"/><text x="{_LEFT - 8}" y="{self.y(t) + 4:.1f}" '
            f'text-anchor="end">{_esc(fmt(t))}</text>'
            for t in ticks
        ]
        parts.append(
            f'<line class="axis" x1="{_LEFT}" x2="{WIDTH - _RIGHT}" y1="{self.bottom:.1f}" '
            f'y2="{self.bottom:.1f}"/>'
        )
        if self.axis == "docs":
            start, end = "0", f"{int(self.x1)} documents"
        else:
            start, end = _when(self.x0), _when(self.x1)
        parts.append(
            f'<text x="{_LEFT}" y="{self.bottom + 18:.1f}">{_esc(start)}</text>'
            f'<text x="{WIDTH - _RIGHT}" y="{self.bottom + 18:.1f}" text-anchor="end">'
            f"{_esc(end)}</text>"
        )
        return parts

    def markers(self, stats: Stats) -> list[str]:
        """Wave starts (the documents axis only) and generator-learned ticks."""
        parts: list[str] = []
        if self.axis == "docs":
            for documents, wave in stats.waves:
                x = self.x(documents)
                parts.append(
                    f'<line class="wave" x1="{x:.1f}" x2="{x:.1f}" y1="{self.top + _TOP - 6:.1f}" '
                    f'y2="{self.bottom:.1f}"/><text class="muted" x="{x + 4:.1f}" '
                    f'y="{self.top + _TOP - 8:.1f}">wave {wave}</text>'
                )
        ticks = (
            [float(d) for d in stats.learned]
            if self.axis == "docs"
            else [when.timestamp() for when in stats.learned_at]
        )
        for tick in (t for t in ticks if self.x0 <= t <= self.x1):
            x = self.x(tick)
            parts.append(
                f'<line class="learned" x1="{x:.1f}" x2="{x:.1f}" y1="{self.bottom:.1f}" '
                f'y2="{self.bottom - 6:.1f}"><title>generator learned</title></line>'
            )
        return parts


def _when(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).strftime("%d %b %H:%M")


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _open(
    name: str, title: str, height: float, *, standalone: bool, animate: bool, label: str
) -> list[str]:
    classes = "chart viz-root animate" if animate else "chart viz-root"
    attrs = 'xmlns="http://www.w3.org/2000/svg" ' if standalone else ""
    parts = [
        f'<svg {attrs}class="{classes}" data-view="{name}" viewBox="0 0 {WIDTH} {height:.0f}" '
        f'width="{WIDTH}" height="{height:.0f}" role="img" aria-label="{_esc(label)}">'
        f"<title>{_esc(title)}</title>"
    ]
    if standalone:
        parts.append(f"<style>{PALETTE_CSS}{CHART_CSS}</style>")
        parts.append(f'<rect class="bg" width="{WIDTH}" height="{height:.0f}"/>')
    return parts


def learning_svg(
    stats: Stats, axis: XAxis = "docs", *, standalone: bool = False, animate: bool = False
) -> str:
    """View 1, the learning curve: LLM calls per document (the hero), cost per document
    and, for a replay, accuracy, one panel each over the same x-axis, with test-site
    waves marked and a tick where each generator was learned. A replay's cost includes
    what the learner spent, and its title says so."""
    points = curve(stats)
    metrics = [
        m
        for m in LEARNING_METRICS
        if m.key != "accuracy" or any(p.accuracy is not None for p in points)
    ]
    if _learning_counted(stats):
        metrics = [replace(m, title=LEARNING_COST_TITLE) if m is COST else m for m in metrics]
    x_name = "documents processed" if axis == "docs" else "time"
    parts = _open(
        "learning",
        "Learning curve",
        _PANEL * len(metrics),
        standalone=standalone,
        animate=animate,
        label=f"{', '.join(m.title for m in metrics)} over {x_name}",
    )
    for i, metric in enumerate(metrics):
        parts += _line_panel(stats, points, axis, metric, top=_PANEL * i)
    parts.append("</svg>")
    return "".join(parts)


def _line_panel(
    stats: Stats, points: Sequence[Point], axis: XAxis, metric: _Metric, *, top: float
) -> list[str]:
    values: list[float | None] = [getattr(p, metric.key) for p in points]
    present = [v for v in values if v is not None]
    ticks = nice_ticks(metric.top if metric.top is not None else max(present, default=0.0))
    frame = _Frame(stats, axis, top, ticks[-1])
    parts = [f'<text class="title" x="{_LEFT}" y="{top + 14:.1f}">{_esc(metric.title)}</text>']
    parts += frame.axes(ticks, metric.fmt)
    parts += frame.markers(stats)
    segment: list[str] = []
    segments: list[list[str]] = [segment]
    for p, v in zip(points, values, strict=True):
        if v is None:
            segment = []
            segments.append(segment)
        else:
            segment.append(f"{frame.x(p.x(axis)):.1f},{frame.y(v):.1f}")
    for seg in segments:
        if len(seg) > 1:
            parts.append(f'<polyline class="line" pathLength="1" points="{" ".join(seg)}"/>')
    for p, v in zip(points, values, strict=True):
        if v is None:
            continue
        cx, cy = frame.x(p.x(axis)), frame.y(v)
        tip = f"{_span(p, axis)}\n{metric.title}: {metric.fmt(v)}"
        parts.append(
            f'<g class="point late"><title>{_esc(tip)}</title>'
            f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>'
            f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4"/></g>'
        )
    last = next(
        ((p, v) for p, v in zip(reversed(points), reversed(values), strict=True) if v is not None),
        None,
    )
    if last is not None:
        p, v = last
        parts.append(
            f'<text class="late" x="{frame.x(p.x(axis)) + 10:.1f}" y="{frame.y(v) + 4:.1f}">'
            f"{_esc(metric.fmt(v))}</text>"
        )
    return parts


def _span(p: Point, axis: XAxis) -> str:
    first = p.documents - p.size + 1
    docs = f"document {p.documents}" if p.size == 1 else f"documents {first}–{p.documents}"
    if axis == "time" and p.at is not None:
        return f"{docs}, {p.at.strftime('%d %b %H:%M')}"
    return docs


def mix_svg(
    stats: Stats, axis: XAxis = "docs", *, standalone: bool = False, animate: bool = False
) -> str:
    """View 2, the resolution mix: each method's share of the values found, stacked to
    100%. The llm band should shrink as the generator band grows."""
    points = [p for p in curve(stats) if sum(p.methods.values())]
    parts = _open(
        "mix",
        "Resolution mix",
        _PANEL + 24,
        standalone=standalone,
        animate=animate,
        label="Share of values resolved by structured, jev, generator, llm and vision",
    )
    frame = _Frame(stats, axis, 24, 1.0)
    parts.append(f'<text class="title" x="{_LEFT}" y="14">Resolution mix</text>')
    parts += _legend(METHODS, x=_LEFT + 120, y=14)
    parts += frame.axes(nice_ticks(1.0), lambda t: f"{t * 100:.0f}%")
    parts += _bands(
        "mix",
        frame,
        [(p.x(axis), shares(p)) for p in points],
        METHODS,
        tips=[f"{_span(p, axis)}\n{_mix_tip(p)}" for p in points],
        animate=animate,
    )
    parts += frame.markers(stats)
    parts.append("</svg>")
    return "".join(parts)


def _mix_tip(p: Point) -> str:
    share = shares(p)
    return " · ".join(f"{m} {share[m] * 100:.0f}%" for m in METHODS if p.methods.get(m))


def _legend(names: Sequence[str], *, x: float, y: float, labels: Sequence[str] = ()) -> list[str]:
    parts: list[str] = []
    for i, name in enumerate(names):
        text = labels[i] if labels else name
        parts.append(
            f'<rect class="m-{name}" x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2"/>'
            f'<text x="{x + 14:.1f}" y="{y:.1f}">{_esc(text)}</text>'
        )
        x += 24 + 7 * len(text)
    return parts


def _bands(
    name: str,
    frame: _Frame,
    columns: Sequence[tuple[float, dict[str, float]]],
    order: Sequence[str],
    *,
    tips: Sequence[str],
    animate: bool,
    labels: Sequence[str] = (),
) -> list[str]:
    """Stacked areas, ``order`` from the bottom, one column of values per x."""
    if not columns:
        return []
    if len(columns) == 1:  # one column: draw it across the plot
        columns = [(frame.x0, columns[0][1]), (frame.x1, columns[0][1])]
    xs = [frame.x(x) for x, _ in columns]
    below = [0.0] * len(columns)
    clip = f"jx-{name}-clip"
    parts = [
        f'<clipPath id="{clip}"><rect class="reveal" x="{_LEFT}" y="{frame.top:.1f}" '
        f'width="{frame.width}" height="{frame.bottom - frame.top:.1f}"/></clipPath>'
        f'<g clip-path="url(#{clip})">'
        if animate
        else "<g>"
    ]
    ends: list[tuple[str, float, float]] = []
    for i, key in enumerate(order):
        above = [b + col.get(key, 0.0) for b, (_, col) in zip(below, columns, strict=True)]
        if any(a > b for a, b in zip(above, below, strict=True)):
            upper = [f"{x:.1f},{frame.y(a):.1f}" for x, a in zip(xs, above, strict=True)]
            lower = [f"{x:.1f},{frame.y(b):.1f}" for x, b in zip(xs, below, strict=True)]
            parts.append(
                f'<polygon class="band m-{key}" points="{" ".join(upper + lower[::-1])}">'
                f"<title>{_esc(labels[i] if labels else key)}</title></polygon>"
            )
            ends.append((labels[i] if labels else key, below[-1], above[-1]))
        below = above
    parts.append("</g>")
    for x, tip in zip(xs, tips, strict=False):
        parts.append(
            f'<rect class="hit" x="{x - 6:.1f}" y="{frame.top + _TOP:.1f}" width="12" '
            f'height="{frame.height:.1f}"><title>{_esc(tip)}</title></rect>'
        )
    for label, lo, hi in ends:
        if frame.y(lo) - frame.y(hi) >= 12:  # room for a direct label
            parts.append(
                f'<text class="late" x="{WIDTH - _RIGHT + 8}" '
                f'y="{(frame.y(lo) + frame.y(hi)) / 2 + 4:.1f}">{_esc(label)}</text>'
            )
    return parts


def cost_svg(
    stats: Stats, axis: XAxis = "docs", *, standalone: bool = False, animate: bool = False
) -> str:
    """View 3, cost: cumulative Jev and LLM spend, stacked so the top edge is the total,
    against the budget line when there is one."""
    spend = _thin([s for s in stats.spend if axis == "docs" or s.at is not None])
    total = max((s.total for s in spend), default=0.0)
    ticks = nice_ticks(max(total, stats.budget_usd or 0.0))
    learning = ", learning included" if _learning_counted(stats) else ""
    parts = _open(
        "cost",
        "Cumulative spend",
        _PANEL + 24,
        standalone=standalone,
        animate=animate,
        label=f"Cumulative Jev and LLM spend in USD{learning}"
        + (" against the budget" if stats.budget_usd is not None else ""),
    )
    xs = [s.at.timestamp() if axis == "time" and s.at else float(s.documents) for s in spend]
    frame = _Frame(stats, axis, 24, ticks[-1], extra=xs)
    title = f"Cumulative spend{learning} (USD)"
    parts.append(f'<text class="title" x="{_LEFT}" y="14">{_esc(title)}</text>')
    parts += _legend(("jev", "llm"), x=_LEFT + 8 * len(title) + 4, y=14, labels=("Jev", "LLM"))
    parts += frame.axes(ticks, usd)
    columns = [(frame.x0, {"jev": 0.0, "llm": 0.0})]
    columns += [(x, {"jev": s.jev, "llm": s.llm}) for x, s in zip(xs, spend, strict=True)]
    tips = ["start"] + [
        f"Jev {usd(s.jev)} · LLM {usd(s.llm)} · total {usd(s.total)}" for s in spend
    ]
    parts += _bands(
        "cost", frame, columns, ("jev", "llm"), tips=tips, animate=animate, labels=("Jev", "LLM")
    )
    if stats.budget_usd is not None:
        y = frame.y(stats.budget_usd)
        parts.append(
            f'<line class="budget" x1="{_LEFT}" x2="{WIDTH - _RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<text x="{WIDTH - _RIGHT + 8}" y="{y + 4:.1f}">budget {usd(stats.budget_usd)}</text>'
        )
    parts += frame.markers(stats)
    parts.append("</svg>")
    return "".join(parts)


def _thin[T](items: Sequence[T], most: int = CURVE_POINTS * 2) -> list[T]:
    """At most ``most`` of ``items``, evenly spaced, always with the last (cumulative
    series lose nothing that matters)."""
    if len(items) <= most:
        return list(items)
    step = len(items) / most
    return [items[min(len(items) - 1, math.ceil((i + 1) * step) - 1)] for i in range(most)]


def chart_svg(
    stats: Stats,
    view: str,
    axis: XAxis = "docs",
    *,
    standalone: bool = False,
    animate: bool = False,
) -> str:
    """One of :data:`CHART_VIEWS` as SVG. Raises ``KeyError`` for another view and
    ``ValueError`` for the time axis on a source without times (a replay)."""
    draw = {"learning": learning_svg, "mix": mix_svg, "cost": cost_svg}[view]
    return draw(stats, axis, standalone=standalone, animate=animate)
