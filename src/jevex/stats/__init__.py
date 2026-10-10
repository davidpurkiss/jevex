"""The stats UI (spec: *Stats UI*): each run needing fewer LLM calls than the last, while
accuracy holds, shown from a store (live) or a replay (``jevex eval --replay``).

- :mod:`jevex.stats.data`: the normalised tables (:class:`Stats`) from a store
  (:func:`from_store`, reading the :class:`~jevex.store.DocumentStat` each document
  records) or a replay (:func:`from_replay`, :func:`from_replay_csv`), and the queries.
- :mod:`jevex.stats.charts`: the learning curve, resolution mix and cost as SVG, static or
  animated (``jevex stats export --svg <view>``, for README graphics).
- :mod:`jevex.stats.page`: the page, served live or written as a report.
- :mod:`jevex.stats.server`: ``jevex stats``, the page and ``/stats/api/*`` over HTTP.
"""

from __future__ import annotations

from jevex.stats.charts import (
    CHART_VIEWS,
    CostPoint,
    accuracy_cost_svg,
    chart_svg,
    cost_svg,
    learning_svg,
    mix_svg,
)
from jevex.stats.data import (
    DRIFT_DOCUMENTS,
    VIEWS,
    Event,
    FieldStat,
    FieldWindow,
    GeneratorStat,
    Point,
    SpendPoint,
    Stats,
    curve,
    field_stats,
    field_window,
    from_replay,
    from_replay_csv,
    from_store,
    summary,
    to_json,
)
from jevex.stats.page import render_page
from jevex.stats.server import replay_loader, stats_server, store_loader

__all__ = [
    "CHART_VIEWS",
    "DRIFT_DOCUMENTS",
    "VIEWS",
    "CostPoint",
    "Event",
    "FieldStat",
    "FieldWindow",
    "GeneratorStat",
    "Point",
    "SpendPoint",
    "Stats",
    "accuracy_cost_svg",
    "chart_svg",
    "cost_svg",
    "curve",
    "field_stats",
    "field_window",
    "from_replay",
    "from_replay_csv",
    "from_store",
    "learning_svg",
    "mix_svg",
    "render_page",
    "replay_loader",
    "stats_server",
    "store_loader",
    "summary",
    "to_json",
]
