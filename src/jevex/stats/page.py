"""The stats page (spec: *Stats UI › Views*): one HTML page, the same for ``jevex stats``
(live, reloading itself) and for report mode (``jevex eval --replay --html``: a single file
that works offline).

The page is rendered here, charts included, so it reads without a script; a small inline
script adds the docs/time toggle, sortable tables, generator details and the live
reload. Its data is inlined as JSON (``<script type="application/json" id="stats-data">``),
the same payload as ``GET /stats/api/all``.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import TYPE_CHECKING

import yaml

from jevex.stats.charts import CHART_CSS, PALETTE_CSS, calls, chart_svg, pct, usd
from jevex.stats.data import METHODS, method_shares, summary, to_json

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jevex.stats.data import FieldStat, GeneratorStat, Stats, XAxis

LIVE_REFRESH_SECONDS = 30
EVENT_LIMIT = 50

_PAGE_CSS = """\
body.viz-root { background: var(--surface-1); color: var(--text-primary); margin: 0;
  padding: 24px; font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline; }
h1 { font-size: 20px; margin: 0; }
h2 { font-size: 15px; margin: 28px 0 8px; }
section.chart-only { margin-top: 28px; }
.source { color: var(--text-secondary); }
.toggle { margin-left: auto; display: flex; gap: 4px; }
.toggle button { font: inherit; color: var(--text-secondary); background: none;
  border: 1px solid var(--grid); border-radius: 6px; padding: 2px 10px; cursor: pointer; }
.toggle button[aria-pressed="true"] { color: var(--text-primary); border-color: var(--axis);
  font-weight: 600; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin: 16px 0; }
.tile { border: 1px solid var(--grid); border-radius: 8px; padding: 10px 12px; }
.tile .label { color: var(--text-secondary); font-size: 12px; }
.tile .value { font-size: 22px; font-weight: 600; }
.tile .note { color: var(--text-muted); font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; text-align: left; border-bottom: 1px solid var(--grid);
  vertical-align: top; }
th { color: var(--text-secondary); font-weight: 600; white-space: nowrap; }
th[data-sort] { cursor: pointer; }
td.num, th.num { text-align: right; }
tr.row { cursor: pointer; }
tr.row:hover { background: color-mix(in srgb, var(--grid) 40%, transparent); }
pre { margin: 4px 0; font-size: 12px; white-space: pre-wrap; }
.mixbar { display: flex; width: 120px; height: 10px; border-radius: 3px; overflow: hidden;
  gap: 1px; background: var(--surface-1); }
.mixbar span { display: block; }
.mixbar .m-structured { background: var(--series-1); }
.mixbar .m-jev { background: var(--series-2); }
.mixbar .m-generator { background: var(--series-3); }
.mixbar .m-llm { background: var(--series-4); }
.mixbar .m-vision { background: var(--series-5); }
.empty { color: var(--text-muted); }
ul.events { list-style: none; padding: 0; margin: 0; }
ul.events li { padding: 3px 0; border-bottom: 1px solid var(--grid); }
.kind { display: inline-block; min-width: 64px; color: var(--text-secondary); }
"""

_SCRIPT = """\
(() => {
  const root = document.body;
  const pick = (axis) => {
    for (const el of document.querySelectorAll("[data-axis]")) {
      el.hidden = el.dataset.axis !== axis;
    }
    for (const b of document.querySelectorAll(".toggle button")) {
      b.setAttribute("aria-pressed", String(b.dataset.pick === axis));
    }
    try { localStorage.setItem("jevex-stats-axis", axis); } catch (e) {}
  };
  for (const b of document.querySelectorAll(".toggle button")) {
    b.addEventListener("click", () => pick(b.dataset.pick));
  }
  let saved = null;
  try { saved = localStorage.getItem("jevex-stats-axis"); } catch (e) {}
  if (saved && document.querySelector(`.toggle button[data-pick="${saved}"]`)) pick(saved);
  for (const table of document.querySelectorAll("table[data-sortable]")) {
    const body = table.tBodies[0];
    table.querySelectorAll("th[data-sort]").forEach((th, col) => {
      th.addEventListener("click", () => {
        const down = th.dataset.dir !== "desc";
        th.dataset.dir = down ? "desc" : "asc";
        const rows = [...body.querySelectorAll("tr.row")];
        const key = (tr) => tr.cells[col].dataset.value ?? tr.cells[col].textContent;
        const num = th.dataset.sort === "num";
        rows.sort((a, b) => {
          const x = key(a), y = key(b);
          const c = num ? (parseFloat(x) || 0) - (parseFloat(y) || 0) : x.localeCompare(y);
          return down ? -c : c;
        });
        for (const tr of rows) {
          const detail = tr.nextElementSibling;
          body.appendChild(tr);
          if (detail && detail.classList.contains("detail")) body.appendChild(detail);
        }
      });
    });
  }
  for (const tr of document.querySelectorAll("tr.row")) {
    tr.addEventListener("click", () => {
      const detail = tr.nextElementSibling;
      if (detail && detail.classList.contains("detail")) detail.hidden = !detail.hidden;
    });
  }
  const refresh = Number(root.dataset.refresh || 0);
  if (refresh > 0) setTimeout(() => location.reload(), refresh * 1000);
})();
"""


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def render_page(
    stats: Stats, *, title: str = "jevex · stats", live: bool = False, generated: str = ""
) -> str:
    """The stats page for ``stats``: headline tiles, the learning curve, resolution mix and
    cost charts, then the generators, fields and budget-and-errors views.

    ``live`` reloads it every :data:`LIVE_REFRESH_SECONDS` (``jevex stats``); otherwise it
    is a report. ``generated`` is shown under the title (say, when it was made).
    """
    axes: list[XAxis] = ["time", "docs"] if stats.has_time else ["docs"]
    if stats.default_axis() == "docs":
        axes.reverse()
    toggle = (
        '<div class="toggle" role="group" aria-label="x-axis">'
        + "".join(
            f'<button type="button" data-pick="{a}" aria-pressed="{str(a == axes[0]).lower()}">'
            f"{'documents' if a == 'docs' else 'time'}</button>"
            for a in axes
        )
        + "</div>"
        if len(axes) > 1
        else ""
    )
    charts = "".join(
        (f"<section><h2>{heading}</h2>" if heading else '<section class="chart-only">')
        + "".join(
            f'<div data-axis="{a}"{"" if a == axes[0] else " hidden"}>'
            f"{chart_svg(stats, v, a)}</div>"
            for a in axes
        )
        + "</section>"
        # The mix and cost charts carry their own titles.
        for v, heading in (("learning", "Learning curve"), ("mix", ""), ("cost", ""))
    )
    data = json.dumps(to_json(stats, "all"), ensure_ascii=False).replace("</", "<\\/")
    refresh = f' data-refresh="{LIVE_REFRESH_SECONDS}"' if live else ""
    note = f" · {_esc(generated)}" if generated else ""
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{PALETTE_CSS}{CHART_CSS}{_PAGE_CSS}</style></head>"
        f'<body class="viz-root"{refresh}><header><h1>{_esc(title)}</h1>'
        f'<span class="source">{stats.kind}: {_esc(stats.source)}{note}</span>{toggle}</header>'
        f"{_tiles(stats)}{charts}"
        f"<section><h2>Generators</h2>{_generators(stats.generators)}</section>"
        f"<section><h2>Fields</h2>{_fields(stats.fields)}</section>"
        f"<section><h2>Budget and errors</h2>{_events(stats)}</section>"
        f'<script type="application/json" id="stats-data">{data}</script>'
        f"<script>{_SCRIPT}</script></body></html>\n"
    )


def _change(value: float | None) -> str:
    if value is None:
        return ""
    arrow = "▼" if value < 0 else "▲"
    return f"{arrow} {abs(value) * 100:.0f}% since the start"


def _tile(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="tile"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div><div class="note">{_esc(note)}</div></div>'
    )


def _tiles(stats: Stats) -> str:
    s = summary(stats)
    llm = s["llm_calls_per_document"]
    cost = s["cost_per_document"]
    accuracy = s["accuracy"]
    learned = s["generators"]
    recent = s["generators_last_day"]
    spent = s["spent_usd"]
    budget = s["budget_usd"]
    tiles = [
        _tile(
            "LLM calls / doc", "–" if llm is None else calls(llm), _change(s["llm_calls_change"])
        ),
        _tile("Cost / doc", "–" if cost is None else usd(cost), _change(s["cost_change"])),
        _tile(
            "Accuracy",
            "–" if accuracy is None else pct(accuracy),
            "replays only" if stats.kind == "store" else "over every scored value",
        ),
        _tile(
            "Generators learned",
            "–" if learned is None else str(learned),
            "" if recent is None else f"+{recent} in the last day",
        ),
        _tile(
            "Spent",
            usd(spent),
            f"of a {usd(budget)} budget" if budget is not None else f"{s['documents']} documents",
        ),
    ]
    return f'<div class="tiles">{"".join(tiles)}</div>'


def _num(value: float | None, fmt: str = "{:.2f}") -> str:
    return "–" if value is None else fmt.format(value)


def _generators(generators: Sequence[GeneratorStat]) -> str:
    if not generators:
        return '<p class="empty">No generators yet: they come from a store\'s learner.</p>'
    head = (
        '<tr><th data-sort="text">id</th><th data-sort="text">field</th>'
        '<th data-sort="text">scope</th><th class="num" data-sort="num">hits</th>'
        '<th class="num" data-sort="num">wins</th><th class="num" data-sort="num">win rate</th>'
        '<th data-sort="text">created</th><th data-sort="text">status</th></tr>'
    )
    rows: list[str] = []
    for g in generators:
        scope = ", ".join(f"{k}={v}" for k, v in sorted(g.scope.items())) or "any"
        spec = yaml.safe_dump(dict(g.spec), sort_keys=False, allow_unicode=True)
        example = (
            f"<p>Learned from: <q>{_esc(g.example)}</q></p>"
            if g.example
            else '<p class="empty">The example it was learned from isn\'t in the store.</p>'
        )
        rows.append(
            f'<tr class="row"><td>{_esc(g.generator_id)}</td><td>{_esc(g.field)}</td>'
            f'<td>{_esc(scope)}</td><td class="num">{g.hits}</td>'
            f'<td class="num">{g.wins}</td>'
            f'<td class="num" data-value="{_num(g.win_rate, "{:.4f}")}">'
            f"{'–' if g.win_rate is None else pct(g.win_rate)}</td>"
            f'<td data-value="{g.created.isoformat()}">{g.created:%Y-%m-%d}</td>'
            f"<td>{g.status}</td></tr>"
            f'<tr class="detail" hidden><td colspan="8"><pre>{_esc(spec)}</pre>{example}</td></tr>'
        )
    return f"<table data-sortable><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _mixbar(counts: Mapping[str, int]) -> str:
    share = method_shares(counts)
    spans = "".join(
        f'<span class="m-{m}" style="width:{share[m] * 100:.1f}%" '
        f'title="{m} {share[m] * 100:.0f}%"></span>'
        for m in METHODS
        if share[m]
    )
    label = ", ".join(f"{m} {share[m] * 100:.0f}%" for m in METHODS if share[m])
    return f'<div class="mixbar" role="img" aria-label="{_esc(label)}">{spans}</div>'


def _fields(fields: Sequence[FieldStat]) -> str:
    if not fields:
        return '<p class="empty">No field values yet: they come from a store\'s documents.</p>'
    ordered = sorted(fields, key=lambda f: (-f.fallback_rate, f.mean_confidence or 1.0, f.field))
    head = (
        '<tr><th data-sort="text">field</th><th class="num" data-sort="num">values</th>'
        '<th class="num" data-sort="num">mean confidence</th>'
        '<th class="num" data-sort="num">llm</th><th>method mix</th>'
        "<th>lowest confidence</th></tr>"
    )
    rows = [
        f'<tr class="row"><td>{_esc(f.field)}</td><td class="num">{f.n}</td>'
        f'<td class="num">{_num(f.mean_confidence)}</td>'
        f'<td class="num" data-value="{f.fallback_rate:.4f}">{pct(f.fallback_rate)}</td>'
        f"<td>{_mixbar(f.methods)}</td>"
        f"<td>{'; '.join(f'{_esc(v)} ({c:.2f})' for v, c in f.lowest) or '–'}</td></tr>"
        for f in ordered
    ]
    return f"<table data-sortable><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _events(stats: Stats) -> str:
    if not stats.events:
        return '<p class="empty">No budget hits, stopped documents or errors.</p>'
    items: list[str] = []
    for e in reversed(stats.events[-EVENT_LIMIT:]):
        when = e.at.strftime("%Y-%m-%d %H:%M") if isinstance(e.at, datetime) else ""
        where = f" ({_esc(e.url)})" if e.url else ""
        items.append(
            f'<li><span class="kind">{_esc(e.kind)}</span> {_esc(when)} '
            f"document {e.documents}{where}: {_esc(e.message)}</li>"
        )
    more = len(stats.events) - EVENT_LIMIT
    tail = f'<p class="empty">and {more} earlier</p>' if more > 0 else ""
    return f'<ul class="events">{"".join(items)}</ul>{tail}'
