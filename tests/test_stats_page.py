import json
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from jevex.stats import (
    Event,
    FieldStat,
    FieldWindow,
    GeneratorStat,
    Point,
    SpendPoint,
    Stats,
    render_page,
    to_json,
)
from jevex.stats.charts import PALETTE_CSS
from jevex.stats.page import MARK_CSS, MARK_SVG

T0 = datetime(2026, 9, 30, 8, 0, tzinfo=UTC)


def store_stats(**kw: object) -> Stats:
    stats = Stats(
        source="sqlite:///jevex.db",
        kind="store",
        points=[
            Point(
                documents=i + 1,
                size=1,
                llm_calls_per_document=2.0 if i < 5 else 0.5,
                jev_cost_per_document=0.001,
                llm_cost_per_document=0.01,
                methods={"llm": 1, "generator": 1},
                at=T0 + timedelta(minutes=i),
            )
            for i in range(20)
        ],
        generators=[
            GeneratorStat(
                generator_id="power-<ps>",
                field="Car.power_ps",
                scope={"locale": "en-GB"},
                documents=10,
                hits=4,
                wins=3,
                disabled=False,
                created=T0,
                learned_from=("ex1",),
                spec={"id": "power-ps", "match": {"regex": r"(\d+) PS"}},
                example="Power: 150 PS",
            ),
            GeneratorStat(
                generator_id="old",
                field="Car.model",
                scope={},
                documents=50,
                hits=0,
                wins=0,
                disabled=True,
                created=T0 - timedelta(days=3),
                learned_from=(),
                spec={"id": "old"},
            ),
        ],
        fields=[
            FieldStat("Car.model", 20, 0.95, {"jev": 20}, ()),
            FieldStat("Car.trim", 10, 0.6, {"llm": 4, "generator": 6}, (("GTI", 0.41),)),
        ],
        events=[
            Event("budget", "document max_llm_calls: 1 LLM calls", 3, T0, "https://cars.test/3"),
            Event("error", "JevBackendError: </script><script>alert(1)</script>", 7, T0),
            Event("error", "<!--<script>", 8, T0),
        ],
        spend=[SpendPoint(20, 0.02, 0.2, at=T0 + timedelta(minutes=19))],
        budget_usd=1.0,
    )
    for key, value in kw.items():
        setattr(stats, key, value)
    return stats


def inlined(page: str) -> Any:
    return json.loads(page.split('id="stats-data">')[1].split("</script>")[0])


def test_a_live_page_has_every_view_and_reloads_itself() -> None:
    stats = store_stats()
    page = render_page(stats, live=True)
    assert page.startswith("<!doctype html>")
    assert 'data-refresh="30"' in page
    assert "store: sqlite:///jevex.db" in page
    # Tiles: the latest tenth against the first.
    assert ">0.50</div>" in page
    assert "▼ 75% since the start" in page
    assert "replays only" in page
    assert "of a $1.000 budget" in page
    # Both axes, time first for a store; the documents view hidden until picked.
    assert page.count('<svg class="chart') == 6
    assert page.index('data-axis="time"') < page.index('data-axis="docs"')
    assert '<div data-axis="docs" hidden>' in page
    assert '<button type="button" data-pick="time" aria-pressed="true">time</button>' in page
    for heading in ("Learning curve", "Generators", "Fields", "Budget and errors"):
        assert f"<h2>{heading}</h2>" in page
    assert inlined(page) == json.loads(json.dumps(to_json(stats, "all")))


def test_generators_are_listed_with_their_spec_and_example() -> None:
    page = render_page(store_stats())
    assert "<td>power-&lt;ps&gt;</td>" in page
    assert "<td>locale=en-GB</td>" in page
    assert '<td class="num" data-value="0.7500">75.0%</td>' in page
    assert "<td>active</td>" in page
    assert "<td>disabled</td>" in page
    assert "regex: (\\d+) PS" in page  # the spec as YAML
    assert "Learned from: <q>Power: 150 PS</q>" in page
    assert "isn't in the store" in page  # the generator without an example


def test_fields_needing_attention_come_first() -> None:
    page = render_page(store_stats()).split("<h2>Fields</h2>")[1]
    assert page.index("<td>Car.trim</td>") < page.index("<td>Car.model</td>")
    assert "GTI (0.41)" in page
    assert 'aria-label="generator 60%, llm 40%"' in page


def test_fields_show_none_rates_and_mark_recent_drift() -> None:
    drifting = FieldStat(
        "Car.trim",
        10,
        0.9,
        {"jev": 9, "llm": 1},
        (),
        none_rate=0.1,
        recent=FieldWindow(documents=5, found=3, values=3, llm=1, mean_confidence=0.85),
    )
    steady = FieldStat(
        "Car.model",
        20,
        0.95,
        {"jev": 20},
        (),
        none_rate=0.0,
        recent=FieldWindow(documents=5, found=5, values=5, llm=0, mean_confidence=0.95),
    )
    unknown = FieldStat("Car.price", 2, None, {"structured": 2}, ())
    page = render_page(store_stats(fields=[drifting, steady, unknown])).split("<h2>Fields</h2>")[1]
    assert "recent: llm · confidence · none" in page
    # Fallback 10% → 33% and none 10% → 40% are marked; confidence 0.90 → 0.85 isn't.
    assert '<td><b class="drift">33.3%</b> · 0.85 · <b class="drift">40.0%</b></td>' in page
    assert "<td>0.0% · 0.95 · 0.0%</td>" in page
    row = page.split("<td>Car.price</td>")[1].split("</tr>")[0]
    assert '<td class="num" data-value="0.0000">–</td>' in row  # no none rate known
    assert "<td>–</td>" in row  # nor a recent window
    [trim] = to_json(store_stats(fields=[drifting]), "fields")
    assert trim["none_rate"] == 0.1
    assert trim["recent"]["none_rate"] == 0.4


def test_events_are_newest_first_and_escaped() -> None:
    page = render_page(store_stats())
    errors = page.index("JevBackendError")
    assert errors < page.index("max_llm_calls")
    assert "</script><script>alert(1)" not in page
    assert "<!--<script>" not in page
    data = inlined(page)
    assert data["events"][1]["message"].endswith("alert(1)</script>")
    assert data["events"][2]["message"] == "<!--<script>"
    assert page.endswith("</script></body></html>\n")  # the page's own script survives


def test_a_report_has_one_axis_and_no_reload() -> None:
    stats = store_stats()
    for p in range(len(stats.points)):
        stats.points[p] = Point(
            documents=p + 1,
            size=1,
            llm_calls_per_document=0,
            jev_cost_per_document=0,
            llm_cost_per_document=0,
        )
    page = render_page(stats, title="Replay", generated="10 documents")
    assert "data-refresh" not in page
    assert '<div class="toggle"' not in page
    assert page.count('<svg class="chart') == 3
    assert "<title>Replay</title>" in page
    assert "10 documents" in page


def test_empty_views_say_why() -> None:
    page = render_page(store_stats(points=[], generators=[], fields=[], events=[], spend=[]))
    assert "No generators yet" in page
    assert "No field values yet" in page
    assert "No budget hits, stopped documents or errors." in page
    assert page.count('<svg class="chart') == 3


BRAND = Path(__file__).parent.parent / "docs" / "brand"
SVG_NS = "{http://www.w3.org/2000/svg}"


def shapes(svg: ET.Element) -> list[dict[str, str]]:
    """Each rect and circle's geometry, in order, without its colours."""
    keep = ("x", "y", "width", "height", "rx", "cx", "cy", "r")
    return [
        {k: v for k, v in el.attrib.items() if k in keep}
        for el in svg.iter()
        if el.tag.removeprefix(SVG_NS) in ("rect", "circle")
    ]


def fills(svg: ET.Element) -> list[str]:
    """Each rect and circle's colour: its fill, or its stroke for an outline."""
    return [
        (el.get("stroke" if el.get("fill") == "none" else "fill") or "").lower()
        for el in svg.iter()
        if el.tag.removeprefix(SVG_NS) in ("rect", "circle")
    ]


def test_the_header_carries_the_jevex_mark_in_both_themes() -> None:
    page = render_page(store_stats())
    assert f"<header><h1>{MARK_SVG}jevex · stats</h1>" in page
    mark = ET.fromstring(MARK_SVG)
    assert shapes(mark) == shapes(ET.parse(BRAND / "jevex-mark.svg").getroot())
    light = fills(ET.parse(BRAND / "jevex-mark.svg").getroot())
    dark = fills(ET.parse(BRAND / "jevex-mark-dark.svg").getroot())
    css = MARK_CSS.split("@media")
    for i in range(4):  # the bars, as the brand's light and dark marks colour them
        assert f"--bar-{i + 1}: {light[i]};" in css[0]
        assert f"--bar-{i + 1}: {dark[i]};" in css[1]
    assert light[4:] == dark[4:] == ["#34d399", "#34d399"]
    for theme, file in ((css[0], "jevex-mark.svg"), (css[1], "jevex-mark-dark.svg")):
        ring = list(ET.parse(BRAND / file).getroot().iter(f"{SVG_NS}circle"))[-1]
        assert f"--ring: {ring.get('opacity')};" in theme
    assert ".mark .value { fill: #34d399; }" in MARK_CSS
    assert f"{PALETTE_CSS}{MARK_CSS}" in page


def test_the_favicon_is_inline_and_the_page_makes_no_requests() -> None:
    page = render_page(store_stats(), live=True)
    icon = re.search(
        r'<link rel="icon" type="image/svg\+xml" href="data:image/svg\+xml,([^"]+)">', page
    )
    assert icon is not None
    favicon = ET.fromstring(unquote(icon.group(1)))
    brand = ET.parse(BRAND / "favicon.svg").getroot()
    assert shapes(favicon) == shapes(brand)
    assert fills(favicon) == fills(brand)
    assert favicon.get("viewBox") == brand.get("viewBox")
    for request in ("src=", 'href="http', "url(http", "@import", "@font-face"):
        assert request not in page
