# /// script
# requires-python = ">=3.12"
# ///
"""Draw the explainer illustrations in docs/brand/illustrations/.

Run `uv run scripts/draw_illustrations.py` after changing a drawing here; this script is
their source. `--check` writes nothing and exits 1 if a committed file differs from what
it would draw (the tests run it).

Each illustration comes in light and dark, static and animated. The animated file is the
static drawing plus a `<style>` of CSS keyframes, which GitHub renders in an `<img>`
(SMIL and scripts are either unreliable or stripped). Every element's resting state is the
finished picture, so `prefers-reduced-motion` just turns the animations off.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "brand" / "illustrations"

HEADING = "'Nunito','Avenir Next',ui-rounded,system-ui,sans-serif"
BODY = "system-ui,sans-serif"
MONO = "ui-monospace,Menlo,monospace"

MINT = "#34D399"
DEEP_MINT = "#064E3B"
AMBER = "#F59E0B"
VIOLET = "#7C3AED"
INK = "#1E1B4B"

# Every keyframed element finishes appearing by HOLD (as a percentage of the loop), stays
# until FADE, and is gone again by the loop's end, so the next loop starts from the page.
HOLD = 90.0
FADE = 96.0


@dataclass(frozen=True)
class Theme:
    suffix: str
    text: str
    muted: str
    card: str
    card_edge: str
    line: str
    band: str
    ramp: tuple[str, str, str, str]  # gains contrast toward the value, as in the mark
    on_ramp: str  # text and marks on ramp[2] and ramp[3]
    shadow: str
    shadow_opacity: float


LIGHT = Theme(
    suffix="",
    text=INK,
    muted="#5B21B6",
    card="#FFFFFF",
    card_edge="#DDD6FE",
    line="#DDD6FE",
    band="#F5F3FF",
    ramp=("#C4B5FD", "#A78BFA", "#7C3AED", "#5B21B6"),
    on_ramp="#FFFFFF",
    shadow=INK,
    shadow_opacity=0.08,
)
DARK = Theme(
    suffix="-dark",
    text="#F5F3FF",
    muted="#C4B5FD",
    card="#0F0D24",
    card_edge="#5B21B6",
    line="#5B21B6",
    band="#1E1B4B",
    ramp=("#5B21B6", "#7C3AED", "#A78BFA", "#DDD6FE"),
    on_ramp=INK,
    shadow="#000000",
    shadow_opacity=0.25,
)


# ---- drawing helpers -----------------------------------------------------------------


def rect(x: float, y: float, w: float, h: float, rx: float, **attrs: str | float) -> str:
    return f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx:g}"{_attrs(attrs)}/>'


def text(x: float, y: float, body: str, size: float, fill: str, **attrs: str | float) -> str:
    attrs.setdefault("font_family", BODY)
    return (
        f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" fill="{fill}"{_attrs(attrs)}>{body}</text>'
    )


def group(children: list[str], **attrs: str | float) -> str:
    inner = "\n".join(f"  {line}" for child in children for line in child.split("\n"))
    return f"<g{_attrs(attrs)}>\n{inner}\n</g>"


def _attrs(attrs: dict[str, str | float]) -> str:
    out = ""
    for key, value in attrs.items():
        name = "class" if key == "cls" else key.replace("_", "-")
        out += f' {name}="{value:g}"' if isinstance(value, float) else f' {name}="{value}"'
    return out


def rule(x1: float, y1: float, x2: float, y2: float, colour: str, width: float = 1.5) -> str:
    return f'<path d="M{x1:g} {y1:g} L{x2:g} {y2:g}" stroke="{colour}" stroke-width="{width:g}"/>'


def lines(x: float, y: float, widths: list[float], fill: str, step: float = 11) -> list[str]:
    """Placeholder text: rounded bars, so the drawings stay free of example data."""
    return [rect(x, y + i * step, w, 5, 2.5, fill=fill) for i, w in enumerate(widths)]


def arrow(x1: float, y1: float, x2: float, y2: float, colour: str, **attrs: str | float) -> str:
    head = f"M{x2 - 7:g} {y2 - 6:g} L{x2:g} {y2:g} L{x2 - 7:g} {y2 + 6:g}"
    return group(
        [f'<path d="M{x1:g} {y1:g} L{x2:g} {y2:g} {head}"/>'],
        fill="none",
        stroke=colour,
        stroke_width=3,
        stroke_linecap="round",
        stroke_linejoin="round",
        **attrs,
    )


def check(x: float, y: float, size: float, colour: str, width: float = 4) -> str:
    """A check mark centred on (x, y), drawn like the mascot's badge."""
    s = size / 20
    d = f"M{x - 10 * s:g} {y:g} L{x - 3 * s:g} {y + 7 * s:g} L{x + 10 * s:g} {y - 7 * s:g}"
    return (
        f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width:g}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def value_pill(x: float, y: float, w: float, h: float) -> str:
    """The value: mint, as everywhere in the brand, with the badge's check."""
    return group(
        [rect(x, y, w, h, h * 0.32, fill=MINT), check(x + w / 2, y + h / 2, h * 0.6, DEEP_MINT)]
    )


def mascot(theme: Theme, x: float, y: float, mirror: bool, badge_cls: str = "") -> str:
    """Jev-ex from docs/brand/mascot.svg, its top-left at (x, y). Mirrored, it faces left
    and holds the badge out on that side, toward whatever came before it; the badge itself
    is not mirrored, so its check still reads as one."""
    flip = " translate(172 0) scale(-1 1)" if mirror else ""
    badge = group(
        [
            rect(108, 18, 44, 28, 9, fill=MINT),
            '<path d="M120 32 L 127 39 L 140 25" fill="none" stroke="#064E3B" stroke-width="5" '
            'stroke-linecap="round" stroke-linejoin="round"/>',
        ],
        **({"cls": badge_cls} if badge_cls else {}),
    )
    if mirror:
        badge = group([badge], transform="translate(260 0) scale(-1 1)")
    body = [
        f'<ellipse cx="62" cy="140" rx="46" ry="6" fill="{theme.shadow}" '
        f'opacity="{theme.shadow_opacity:g}"/>',
        '<path d="M18 128 C 6 70, 34 22, 62 22 C 92 22, 118 70, 106 128 Z" fill="#8B5CF6"/>',
        '<path d="M30 124 C 24 84, 40 44, 62 44 C 86 44, 100 84, 94 124 Z" fill="#A78BFA" '
        'opacity="0.45"/>',
        '<circle cx="48" cy="70" r="11" fill="#FFFFFF"/><circle cx="51" cy="72" r="5" '
        f'fill="{INK}"/>',
        '<circle cx="78" cy="70" r="11" fill="#FFFFFF"/><circle cx="81" cy="72" r="5" '
        f'fill="{INK}"/>',
        f'<path d="M52 92 Q 63 102 74 92" fill="none" stroke="{INK}" stroke-width="4" '
        'stroke-linecap="round"/>',
        '<path d="M104 86 L 122 50" stroke="#8B5CF6" stroke-width="9" stroke-linecap="round"/>',
        badge,
    ]
    return group(
        [group(body, transform="translate(6 2)")], transform=f"translate({x:g} {y:g}){flip}"
    )


def caption(x: float, y: float, title: str, detail: str, theme: Theme) -> list[str]:
    return [
        text(
            x, y, title, 17, theme.text, font_family=HEADING, font_weight=700, text_anchor="middle"
        ),
        text(x, y + 21, detail, 12.5, theme.muted, text_anchor="middle"),
    ]


# ---- animation -------------------------------------------------------------------------


@dataclass
class Motion:
    """CSS keyframes for one illustration. Each method animates one class, timed in
    percentages of a loop of `seconds`."""

    seconds: float
    rules: list[str]

    def _add(self, cls: str, frames: list[tuple[str, str]], origin: bool = False) -> None:
        steps = " ".join(f"{at} {{ {style} }}" for at, style in frames)
        self.rules.append(f"@keyframes {cls} {{ {steps} }}")
        box = " transform-box: fill-box; transform-origin: center;" if origin else ""
        self.rules.append(
            f".{cls} {{ animation: {cls} {self.seconds:g}s ease-in-out infinite;{box} }}"
        )

    def appear(self, cls: str, start: float, end: float, rest: float = 1) -> None:
        self._add(cls, _span(start, end, "opacity: 0", f"opacity: {rest:g}"))

    def dim(self, cls: str, start: float, end: float, fade_from: float, rest: float) -> None:
        """Appear, then fade to `rest`: what a stage dropped."""
        self._add(
            cls,
            [
                (f"0%, {start:g}%", "opacity: 0"),
                (f"{end:g}%, {fade_from:g}%", "opacity: 1"),
                (f"{fade_from + 6:g}%, {HOLD:g}%", f"opacity: {rest:g}"),
                (f"{FADE:g}%, 100%", "opacity: 0"),
            ],
        )

    def pop(self, cls: str, start: float, end: float) -> None:
        mid = start + (end - start) * 0.6
        self._add(
            cls,
            [
                (f"0%, {start:g}%", "opacity: 0; transform: scale(0.4)"),
                (f"{mid:g}%", "opacity: 1; transform: scale(1.15)"),
                (f"{end:g}%, {HOLD:g}%", "opacity: 1; transform: scale(1)"),
                (f"{FADE:g}%, 100%", "opacity: 0; transform: scale(1)"),
            ],
            origin=True,
        )

    def draw(self, cls: str, start: float, end: float) -> None:
        """Draw a path along its length; the path needs pathLength="1"."""
        self._add(
            cls,
            [
                (f"0%, {start:g}%", "stroke-dashoffset: 1; opacity: 1"),
                (f"{end:g}%, {HOLD:g}%", "stroke-dashoffset: 0; opacity: 1"),
                (f"{FADE:g}%, 100%", "stroke-dashoffset: 0; opacity: 0"),
            ],
        )
        self.rules.append(f".{cls} {{ stroke-dasharray: 1 1; }}")

    def hop(self, cls: str, start: float, end: float) -> None:
        mid = (start + end) / 2
        self._add(
            cls,
            [
                (f"0%, {start:g}%, {end:g}%, 100%", "transform: translateY(0)"),
                (f"{mid:g}%", "transform: translateY(-14px)"),
            ],
        )

    def pulse(self, cls: str, start: float, end: float) -> None:
        mid = (start + end) / 2
        self._add(
            cls,
            [
                (f"0%, {start:g}%, {end:g}%, 100%", "transform: scale(1)"),
                (f"{mid:g}%", "transform: scale(1.3)"),
            ],
            origin=True,
        )

    def sweep(self, cls: str, start: float, end: float, distance: float) -> None:
        """Move down by `distance` while visible; at rest it is hidden (opacity 0)."""
        self._add(
            cls,
            [
                (f"0%, {start:g}%", "opacity: 0; transform: translateY(0)"),
                (f"{start + 2:g}%", "opacity: 0.55; transform: translateY(0)"),
                (f"{end - 2:g}%", f"opacity: 0.55; transform: translateY({distance:g}px)"),
                (f"{end:g}%, 100%", f"opacity: 0; transform: translateY({distance:g}px)"),
            ],
        )

    def style(self) -> str:
        reduced = "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"
        body = "\n".join(f"  {rule}" for rule in [*self.rules, reduced])
        return f"<style>\n{body}\n</style>"


def _span(start: float, end: float, before: str, after: str) -> list[tuple[str, str]]:
    return [
        (f"0%, {start:g}%", before),
        (f"{end:g}%, {HOLD:g}%", after),
        (f"{FADE:g}%, 100%", before),
    ]


def svg(width: int, height: int, label: str, comment: str, body: list[str], style: str) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{label}">',
        f"  <title>{label}</title>",
        f"  <!-- {comment}\n       Drawn by scripts/draw_illustrations.py; edit it there. -->",
    ]
    if style:
        parts.extend(f"  {line}" for line in style.split("\n"))
    parts.extend(f"  {line}" for child in body for line in child.split("\n"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ---- pipeline narrowing ----------------------------------------------------------------

PIPELINE_LABEL = (
    "How jevex narrows a document: the page is split into components, Jev keeps the "
    "relevant ones, splits them into statements, keeps the one that states the field, "
    "and picks the value from it"
)


def pipeline(theme: Theme, animated: bool) -> str:
    t = theme
    r0, r1, r2, r3 = t.ramp
    motion = Motion(seconds=10, rules=[])

    # The funnel: wide at the document, narrowing into the value the mascot holds.
    band = (
        '<path d="M30 30 Q30 14 46 14 L210 14 C380 14 560 110 770 116 '
        f'L770 144 C560 150 380 246 210 246 L46 246 Q30 246 30 230 Z" fill="{t.band}"/>'
    )

    # 1 · The document: a laid-out page. The table is what the field lives in.
    table = [
        rect(68, 96, 120, 52, 3, fill="none", stroke=t.line, stroke_width=2),
        *[rule(68, 96 + 13 * i, 188, 96 + 13 * i, t.line) for i in (1, 2, 3)],
        rule(112, 96, 112, 148, t.line),
    ]
    document = group(
        [
            rect(54, 30, 148, 200, 10, fill=t.card, stroke=r0, stroke_width=2),
            rect(68, 44, 78, 10, 5, fill=r1),
            *lines(68, 64, [120, 96], t.line),
            *table,
            rect(64, 92, 128, 60, 6, fill="none", stroke=r1, stroke_width=2.5),
            rect(68, 162, 50, 40, 4, fill=t.line),
            f'<path d="M74 196 L88 178 L98 188 L104 182 L114 196 Z" fill="{r1}"/>',
            *lines(126, 166, [62, 48, 56], t.line),
            *lines(68, 210, [112], t.line),
        ]
    )
    scan = rect(58, 34, 140, 14, 6, fill=r1, opacity=0, cls="scan")
    motion.sweep("scan", 0, 16, 178)

    # 2 · Components: the gate keeps the table and drops the rest.
    zoom1 = group(
        [
            f'<path d="M192 92 L270 98 M192 152 L270 164" stroke="{r1}" stroke-width="1.5" '
            'stroke-dasharray="4 4"/>'
        ],
        cls="zoom1",
    )
    motion.appear("zoom1", 16, 22)
    kept_component = group(
        [
            rect(270, 98, 140, 66, 8, fill=t.card, stroke=r1, stroke_width=2.5),
            *[rule(280, 105 + 13 * i, 400, 105 + 13 * i, t.line) for i in (1, 2, 3)],
            rule(322, 106, 322, 156, t.line),
            *lines(284, 108, [28, 28, 28, 28], t.line, step=13),
            *lines(330, 108, [50, 40, 58, 34], t.line, step=13),
        ],
        cls="comp",
    )
    motion.appear("comp", 18, 26)
    dropped_components = group(
        [
            rect(
                270, 40, 140, 46, 8, fill="none", stroke=r0, stroke_width=2, stroke_dasharray="6 5"
            ),
            *lines(282, 52, [104, 92, 70], r0),
            rect(
                270, 176, 140, 46, 8, fill="none", stroke=r0, stroke_width=2, stroke_dasharray="6 5"
            ),
            rect(282, 186, 34, 26, 3, fill=r0),
            *lines(324, 188, [74, 56], r0),
        ],
        opacity=0.45,
        cls="comp-out",
    )
    motion.dim("comp-out", 18, 26, 30, 0.45)

    # 3 · Statements: one per table cell; the classifier keeps the one stating the field.
    zoom2 = group(
        [
            f'<path d="M414 98 L474 77 M414 164 L474 183" stroke="{r2}" stroke-width="1.5" '
            'stroke-dasharray="4 4"/>'
        ],
        cls="zoom2",
    )
    motion.appear("zoom2", 36, 42)
    kept_statement = group(
        [
            rect(474, 105, 152, 22, 11, fill=r2),
            *lines(488, 113.5, [40], t.on_ramp),
            *lines(536, 113.5, [74], t.on_ramp),
        ],
        cls="stmt",
    )
    motion.appear("stmt", 38, 46)
    dropped_statements = group(
        [
            piece
            for y in (77, 133, 161)
            for piece in (
                rect(
                    474,
                    y,
                    152,
                    22,
                    11,
                    fill="none",
                    stroke=r1,
                    stroke_width=2,
                    stroke_dasharray="6 5",
                ),
                *lines(488, y + 8.5, [40], r1),
                *lines(536, y + 8.5, [60 + (y % 3) * 8], r1),
            )
        ],
        opacity=0.45,
        cls="stmt-out",
    )
    motion.dim("stmt-out", 38, 46, 50, 0.45)

    # 4 · The value: selected from the statement's candidates, held up by the mascot.
    to_value = arrow(632, 116, 760, 128, r3, cls="tovalue")
    motion.appear("tovalue", 56, 60)
    holder = group([mascot(t, 752, 96, mirror=True, badge_cls="badge")], cls="hop")
    motion.pop("badge", 58, 66)
    motion.hop("hop", 64, 72)

    captions = [
        *caption(128, 286, "document", "cleaned and laid out", t),
        *caption(340, 286, "component", "Noul: does it contain the field?", t),
        *caption(550, 286, "statement", "Choice: which detail does it state?", t),
        *caption(838, 286, "value", "Choice: which candidate is it?", t),
    ]

    body = [
        band,
        document,
        *([scan] if animated else []),
        zoom1,
        dropped_components,
        kept_component,
        zoom2,
        dropped_statements,
        kept_statement,
        to_value,
        holder,
        *captions,
    ]
    comment = "Pipeline narrowing: document → component → statement → value."
    return svg(960, 330, PIPELINE_LABEL, comment, body, motion.style() if animated else "")


# ---- learning loop ---------------------------------------------------------------------

LEARNING_LABEL = (
    "How jevex learns: the first time, no candidate fits, so an LLM reads the statement, "
    "Jev verifies its answer and a generator is learned from it; the next time, the "
    "generator finds the value and no LLM is called"
)


def statement_card(theme: Theme, x: float, y: float, widths: list[float]) -> str:
    return group(
        [
            rect(x, y, 170, 56, 10, fill=theme.card, stroke=theme.card_edge, stroke_width=2),
            *lines(x + 16, y + 15, widths, theme.ramp[1], step=12),
        ]
    )


def jev_node(theme: Theme, x: float, y: float, cls: str, tick_cls: str) -> str:
    return group(
        [
            rect(x, y, 110, 56, 14, fill=VIOLET),
            text(
                x + 55,
                y + 35,
                "Jev",
                20,
                "#FFFFFF",
                font_family=HEADING,
                font_weight=800,
                text_anchor="middle",
            ),
            group(
                [
                    f'<circle cx="{x + 108:g}" cy="{y + 2:g}" r="13" fill="{theme.card}" '
                    f'stroke="{VIOLET}" stroke-width="2.5"/>',
                    check(x + 108, y + 2, 12, VIOLET, 3.5),
                ],
                cls=tick_cls,
            ),
        ],
        cls=cls,
    )


def lane(theme: Theme, y: float, title: str, llm_calls: str, used_llm: bool) -> list[str]:
    dot = (
        f'<circle cx="0" cy="0" r="7" fill="{AMBER}"/>'
        if used_llm
        else f'<circle cx="0" cy="0" r="6" fill="none" stroke="{AMBER}" stroke-width="2" '
        'stroke-dasharray="3 3"/>'
    )
    return [
        rect(16, y, 768, 186, 18, fill=theme.band),
        text(40, y + 32, title, 18, theme.text, font_family=HEADING, font_weight=800),
        group([dot], transform=f"translate(170 {y + 26:g})"),
        text(184, y + 31, llm_calls, 14, theme.text),
    ]


def learning_loop(theme: Theme, animated: bool) -> str:
    t = theme
    r3 = t.ramp[3]
    motion = Motion(seconds=12, rules=[])
    y1, y2 = 70, 306  # the two lanes' node rows
    gap = 234  # where the learning arrow crosses between the lanes

    first = [
        *lane(t, 14, "First time", "one LLM call", used_llm=True),
        statement_card(t, 40, y1, [96, 130, 70]),
        group(
            [
                f'<circle cx="210" cy="{y1:g}" r="14" fill="{r3}"/>',
                text(
                    210,
                    y1 + 6,
                    "?",
                    18,
                    t.on_ramp,
                    font_family=HEADING,
                    font_weight=800,
                    text_anchor="middle",
                ),
            ],
            cls="unsure",
        ),
        arrow(226, y1 + 28, 262, y1 + 28, r3, cls="a1"),
        group(
            [
                rect(270, y1, 110, 56, 14, fill=AMBER),
                text(
                    325,
                    y1 + 35,
                    "LLM",
                    20,
                    INK,
                    font_family=HEADING,
                    font_weight=800,
                    text_anchor="middle",
                ),
            ],
            cls="llm",
        ),
        arrow(388, y1 + 28, 432, y1 + 28, r3, cls="a2"),
        jev_node(t, 440, y1, "jev1", "tick1"),
        arrow(564, y1 + 28, 602, y1 + 28, r3, cls="a3"),
        group([value_pill(610, y1 + 12, 80, 32)], cls="value1"),
        *caption(125, y1 + 82, "statement", "no candidate fits", t),
        *caption(325, y1 + 82, "LLM fallback", "reads the statement", t),
        *caption(495, y1 + 82, "verified", "Jev checks the answer", t),
        *caption(650, y1 + 82, "value", "a verified example", t),
    ]
    motion.pulse("unsure", 2, 9)
    motion.appear("a1", 9, 11)
    motion.pop("llm", 10, 15)
    motion.appear("a2", 17, 19)
    motion.pop("jev1", 19, 23)
    motion.pop("tick1", 23, 27)
    motion.appear("a3", 27, 29)
    motion.pop("value1", 29, 33)

    learn = group(
        [
            f'<path d="M694 {y1 + 28:g} C748 {y1 + 28:g}, 748 {gap:g}, 694 {gap:g} L360 {gap:g} '
            f'C325 {gap:g}, 325 {gap + 14:g}, 325 {y2 - 10:g}" '
            f'fill="none" stroke="{r3}" stroke-width="3" stroke-linecap="round" '
            'pathLength="1" class="learn"/>',
        ]
    )
    learn_head = group(
        [
            f'<path d="M318 {y2 - 18:g} L325 {y2 - 9:g} L332 {y2 - 18:g}" fill="none" '
            f'stroke="{r3}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
            text(
                520,
                gap - 10,
                "learn a generator: write · test · publish",
                13,
                t.text,
                font_weight=600,
                text_anchor="middle",
            ),
        ],
        cls="learn-end",
    )
    motion.draw("learn", 35, 45)
    motion.appear("learn-end", 44, 47)

    generator = group(
        [
            rect(270, y2, 110, 56, 10, fill=t.card, stroke=t.ramp[2], stroke_width=2.5),
            text(282, y2 + 23, "match: …", 12, t.text, font_family=MONO),
            text(282, y2 + 41, "normalise: …", 12, t.text, font_family=MONO),
        ],
        cls="gen",
    )
    motion.pop("gen", 46, 51)

    second = [
        *lane(t, 250, "Next time", "no LLM call", used_llm=False),
        statement_card(t, 40, y2, [80, 120, 104]),
        arrow(218, y2 + 28, 262, y2 + 28, r3, cls="b1"),
        generator,
        arrow(388, y2 + 28, 432, y2 + 28, r3, cls="b2"),
        jev_node(t, 440, y2, "jev2", "tick2"),
        arrow(564, y2 + 28, 610, y2 + 34, r3, cls="b3"),
        group([mascot(t, 600, y2 - 24, mirror=True, badge_cls="value2")], cls="cheer"),
        *caption(125, y2 + 82, "statement", "same pattern, new page", t),
        *caption(325, y2 + 82, "generator", "in place of the LLM", t),
        *caption(495, y2 + 82, "selected", "Jev picks the candidate", t),
    ]
    motion.appear("b1", 51, 53)
    motion.appear("b2", 53, 55)
    motion.pop("jev2", 55, 59)
    motion.pop("tick2", 58, 62)
    motion.appear("b3", 62, 64)
    motion.pop("value2", 64, 68)
    motion.hop("cheer", 68, 75)

    body = [*first, *second, learn, learn_head]
    comment = "Learning loop: LLM fallback → verified → generator → no LLM next time."
    return svg(800, 450, LEARNING_LABEL, comment, body, motion.style() if animated else "")


# ---- files -----------------------------------------------------------------------------


def drawings() -> dict[str, str]:
    files: dict[str, str] = {}
    for theme in (LIGHT, DARK):
        for animated in (False, True):
            kind = "-animated" if animated else ""
            files[f"pipeline{kind}{theme.suffix}.svg"] = pipeline(theme, animated)
            files[f"learning-loop{kind}{theme.suffix}.svg"] = learning_loop(theme, animated)
    return files


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Draw the illustrations in docs/brand/.")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if stale")
    parser.add_argument("--out", type=Path, default=OUT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    out: Path = args.out
    files = drawings()
    changed = [
        name
        for name, content in files.items()
        if not (out / name).is_file() or (out / name).read_text() != content
    ]
    # A drawing renamed or dropped here would otherwise leave its old file behind.
    extra = sorted(p.name for p in out.glob("*.svg") if p.name not in files)
    if not args.check:
        out.mkdir(parents=True, exist_ok=True)
        for name in changed:
            (out / name).write_text(files[name])
    for name in changed:
        print(f"{'stale' if args.check else 'wrote'}: {out / name}")
    for name in extra:
        print(f"not drawn by this script: {out / name}")
    return 1 if extra or (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
