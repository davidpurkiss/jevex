import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from jevex.stats import CHART_VIEWS, Point, SpendPoint, Stats, chart_svg, cost_svg
from jevex.stats.charts import learning_svg, mix_svg, nice_ticks, usd

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


def test_a_store_has_no_accuracy_panel_and_a_time_axis() -> None:
    svg = learning_svg(store_stats(), "time")
    assert "Accuracy" not in svg
    assert ">30 Sep 08:00</text>" in svg
    assert ">30 Sep 10:00</text>" in svg
    assert "document 2, 30 Sep 09:00" in svg


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
