"""Learning curves: ``jevex eval --replay`` (spec: *Evaluation and test site › jevex eval*).

:func:`replay` starts from an empty store and runs an extractor over a corpus one document
at a time, in the corpus's order. After each document it waits for the learner to finish
what the document queued, so every document sees what the ones before it taught (and a
replay with recorded answers gives the same curve every time). It reports per batch of
documents (:class:`ReplayBatch`): accuracy, cost per document, LLM calls per document, the
resolution mix and the learned generators in use. That is the headline claim's evidence:
accuracy holds while cost and the LLM-call rate fall as generators are learned.

:meth:`ReplayReport.to_csv` gives one row per batch (:data:`CSV_COLUMNS`), and
:meth:`ReplayReport.to_html` a self-contained page charting accuracy, cost per document
and LLM calls per document over documents processed, with the test site's waves marked.
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

from jevex.eval import EvalReport, check_schemas, resolve_tolerances, run_document
from jevex.results import Method

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jevex.eval import CorpusItem, DocumentRun, Tolerance
    from jevex.extractor import Extractor
    from jevex.store import Store

REPLAY_BATCH_SIZE = 10


class LearningStoppedError(Exception):
    """The learner failed during a replay. The documents after it would learn nothing, so
    the curve would stop measuring learning: the replay ends instead."""


METHODS: tuple[str, ...] = get_args(Method)

CSV_COLUMNS: tuple[str, ...] = (
    "batch",
    "documents",
    "size",
    "waves",
    "accuracy",
    "precision",
    "recall",
    "cost_per_document",
    "jev_cost_per_document",
    "llm_cost_per_document",
    "llm_calls_per_document",
    "jev_requests_per_document",
    "jev_questions_per_document",
    "seconds_per_document",
    "errors",
    "generators",
    *(f"values_{m}" for m in METHODS),
)
"""The CSV's columns. ``documents`` is how many documents were processed by the batch's
end (the x-axis), ``waves`` the test-site waves its documents came from, ``generators``
the learned generators enabled in the store after it, and ``values_<method>`` how many
values each method resolved in it."""


@dataclass(frozen=True)
class ReplayBatch:
    """The metrics of one batch of documents in a replay (per document unless counted)."""

    number: int
    """1-based."""
    documents: int
    """Documents processed by the end of this batch."""
    size: int
    waves: tuple[int, ...]
    accuracy: float | None
    precision: float | None
    recall: float | None
    cost_per_document: float
    jev_cost_per_document: float
    llm_cost_per_document: float
    llm_calls_per_document: float
    jev_requests_per_document: float
    jev_questions_per_document: float
    seconds_per_document: float | None
    errors: int
    generators: int
    methods: dict[str, int]

    def row(self) -> dict[str, Any]:
        """The batch as a flat dict keyed by :data:`CSV_COLUMNS` (JSON types)."""
        return {
            "batch": self.number,
            "documents": self.documents,
            "size": self.size,
            "waves": " ".join(str(w) for w in self.waves),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "cost_per_document": self.cost_per_document,
            "jev_cost_per_document": self.jev_cost_per_document,
            "llm_cost_per_document": self.llm_cost_per_document,
            "llm_calls_per_document": self.llm_calls_per_document,
            "jev_requests_per_document": self.jev_requests_per_document,
            "jev_questions_per_document": self.jev_questions_per_document,
            "seconds_per_document": self.seconds_per_document,
            "errors": self.errors,
            "generators": self.generators,
            **{f"values_{m}": self.methods.get(m, 0) for m in METHODS},
        }


@dataclass
class ReplayReport:
    """The result of :func:`replay`: every document's run, in the order it ran."""

    report: EvalReport
    batch_size: int
    waves: list[int | None]
    """Each document's test-site wave (``None`` if the corpus doesn't say)."""
    generators: list[int]
    """Learned generators enabled in the store after each document."""

    def batches(self) -> list[ReplayBatch]:
        """The documents in batches of ``batch_size`` (the last one may be smaller)."""
        docs = self.report.documents
        out: list[ReplayBatch] = []
        for start in range(0, len(docs), self.batch_size):
            end = min(start + self.batch_size, len(docs))
            out.append(self._batch(len(out) + 1, start, end))
        return out

    def _batch(self, number: int, start: int, end: int) -> ReplayBatch:
        s = EvalReport(documents=self.report.documents[start:end]).summary()
        return ReplayBatch(
            number=number,
            documents=end,
            size=end - start,
            waves=tuple(sorted({w for w in self.waves[start:end] if w is not None})),
            accuracy=s["accuracy"],
            precision=s["precision"],
            recall=s["recall"],
            cost_per_document=s["cost_per_document"],
            jev_cost_per_document=s["jev_cost_per_document"],
            llm_cost_per_document=s["llm_cost_per_document"],
            llm_calls_per_document=s["llm_calls_per_document"],
            jev_requests_per_document=s["jev_requests_per_document"],
            jev_questions_per_document=s["jev_questions_per_document"],
            seconds_per_document=s["seconds_per_document"],
            errors=s["errors"],
            generators=self.generators[end - 1],
            methods=s["resolution_mix"],
        )

    def wave_starts(self) -> list[tuple[int, int]]:
        """``(documents processed before it, wave)`` for each wave after the first."""
        starts: list[tuple[int, int]] = []
        previous: int | None = None
        for i, wave in enumerate(self.waves):
            if wave is not None and previous is not None and wave != previous:
                starts.append((i, wave))
            previous = wave if wave is not None else previous
        return starts

    @property
    def failed(self) -> list[DocumentRun]:
        """Documents whose extraction raised."""
        return self.report.failed

    def to_dict(self) -> dict[str, Any]:
        """The run's summary and its batches as JSON types."""
        return {
            "batch_size": self.batch_size,
            "summary": self.report.summary(),
            "batches": [b.row() for b in self.batches()],
        }

    def to_csv(self) -> str:
        """One row per batch, with a header row of :data:`CSV_COLUMNS`. Empty cells are
        metrics with nothing to measure (no values scored, or every document failed)."""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for batch in self.batches():
            writer.writerow({k: "" if v is None else v for k, v in batch.row().items()})
        return buffer.getvalue()

    def to_html(self, title: str = "jevex replay") -> str:
        """A self-contained page (no scripts, works offline): accuracy, cost per document
        and LLM calls per document over documents processed, one chart each on a shared
        x-axis, with every wave's start marked, then the batches as a table. The data is
        inlined as JSON (``<script type="application/json" id="replay-data">``)."""
        return _render_html(self, title)


async def replay(
    extractor: Extractor,
    corpus: list[CorpusItem],
    *,
    batch_size: int = REPLAY_BATCH_SIZE,
    tolerances: Mapping[str, Tolerance] | None = None,
) -> ReplayReport:
    """Run ``extractor`` over ``corpus`` in order, one document at a time, and report
    the learning curve in batches of ``batch_size`` documents.

    The extractor's store must start empty: no generators, key mappings or examples (the
    curve would otherwise start part way down). An extractor with a ``generator_llm``
    and no store has an in-memory one; one with neither learns nothing, so its curve
    shows only what the run budget and the structured stage's cache change. Packs still
    apply (pass ``community_packs=False`` for a curve from nothing). ``tolerances`` is as
    for :func:`~jevex.eval.evaluate`.

    Raises ``ValueError`` for a ``batch_size`` below 1, a schema the extractor lacks or a
    store that isn't empty; the :data:`~jevex.eval.RUN_ERRORS` and an error that stopped
    the learner end the replay (:class:`LearningStoppedError`). Other per-document errors
    are scored as all missing.

    Cost and LLM calls are the documents' own (their ``meta``). What the learner spends
    between documents (generator LLM calls, Jev requests testing generators) isn't
    counted.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    resolved = resolve_tolerances(extractor, tolerances)
    check_schemas(corpus, resolved)
    store = await extractor.store()
    if store is not None and await _holds_learned_state(store):
        raise ValueError(
            "replay starts from an empty store, but this one already holds generators, "
            "key mappings or examples"
        )
    runs: list[DocumentRun] = []
    generators: list[int] = []
    for item in corpus:
        runs.append(await run_document(extractor, item, resolved))
        try:
            await extractor.wait_for_learning()
        except RuntimeError as exc:  # the learner's worker failed (GeneratorLearner.drain)
            cause = exc.__cause__ or exc
            raise LearningStoppedError(f"learning stopped after {item.path}: {cause}") from exc
        generators.append(len(await store.generators()) if store is not None else 0)
    return ReplayReport(
        report=EvalReport(documents=runs),
        batch_size=batch_size,
        waves=[item.wave for item in corpus],
        generators=generators,
    )


async def _holds_learned_state(store: Store) -> bool:
    return bool(
        await store.generators(include_disabled=True)
        or await store.disabled_generator_ids()
        or await store.key_mappings()
        or await store.examples(limit=1)
    )


# --- HTML ------------------------------------------------------------------------------

_WIDTH, _HEIGHT = 720, 180
_LEFT, _RIGHT, _TOP, _BOTTOM = 64, 80, 22, 30

_STYLE = """\
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #6b6a65;
  --grid: #e4e3df;
  --series-1: #2a78d6;
  background: var(--surface-1);
  color: var(--text-primary);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  margin: 0;
  padding: 24px;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #a3a29a;
    --grid: #33332f;
    --series-1: #3987e5;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #a3a29a;
  --grid: #33332f;
  --series-1: #3987e5;
}
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 24px 0 4px; }
p.lede { color: var(--text-secondary); margin: 0 0 8px; }
svg { display: block; max-width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-secondary); font-size: 12px; }
svg .muted { fill: var(--text-muted); }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .wave { stroke: var(--text-muted); stroke-width: 1; opacity: 0.6; }
svg .line { fill: none; stroke: var(--series-1); stroke-width: 2;
  stroke-linejoin: round; stroke-linecap: round; }
svg .dot { fill: var(--series-1); stroke: var(--surface-1); stroke-width: 2; }
svg .hit { fill: transparent; }
svg .hit:hover + .dot, svg .point:hover .dot { stroke: var(--text-primary); }
table { border-collapse: collapse; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; text-align: right; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 600; }
"""


@dataclass(frozen=True)
class _Metric:
    key: str
    title: str
    fmt: Callable[[float], str]
    top: float | None = None
    """A fixed top for the y-axis (accuracy's 100%); otherwise from the data."""


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _usd(x: float) -> str:
    text = f"{x:.5f}".rstrip("0").rstrip(".") if abs(x) < 0.01 else f"{x:.3f}"
    return f"${text}"


def _calls(x: float) -> str:
    return f"{x:.2f}"


_METRICS = (
    _Metric("accuracy", "Accuracy", _pct, top=1.0),
    _Metric("cost_per_document", "Cost per document (USD)", _usd),
    _Metric("llm_calls_per_document", "LLM calls per document", _calls),
)


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


def _chart(report: ReplayReport, batches: list[ReplayBatch], metric: _Metric) -> str:
    total = max(len(report.report.documents), 1)
    values: list[float | None] = [getattr(b, metric.key) for b in batches]
    present = [v for v in values if v is not None]
    ticks = (
        [0.0, 0.25, 0.5, 0.75, 1.0] if metric.top == 1.0 else nice_ticks(max(present, default=0.0))
    )
    y_top = ticks[-1]
    plot_w = _WIDTH - _LEFT - _RIGHT
    plot_h = _HEIGHT - _TOP - _BOTTOM

    def x(documents: float) -> float:
        return _LEFT + plot_w * documents / total

    def y(value: float) -> float:
        return _TOP + plot_h * (1 - value / y_top)

    parts = [
        f'<svg viewBox="0 0 {_WIDTH} {_HEIGHT}" role="img" '
        f'aria-label="{html.escape(metric.title)} over documents processed">'
    ]
    for t in ticks:
        parts.append(
            f'<line class="grid" x1="{_LEFT}" x2="{_WIDTH - _RIGHT}" '
            f'y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
            f'<text x="{_LEFT - 8}" y="{y(t) + 4:.1f}" text-anchor="end">'
            f"{html.escape(metric.fmt(t))}</text>"
        )
    for documents, wave in report.wave_starts():
        parts.append(
            f'<line class="wave" x1="{x(documents):.1f}" x2="{x(documents):.1f}" '
            f'y1="{_TOP - 6}" y2="{_TOP + plot_h}"/>'
            f'<text class="muted" x="{x(documents) + 4:.1f}" y="{_TOP - 8}">wave {wave}</text>'
        )
    parts.append(
        f'<text x="{_LEFT}" y="{_HEIGHT - 6}" text-anchor="start">0</text>'
        f'<text x="{_WIDTH - _RIGHT}" y="{_HEIGHT - 6}" text-anchor="end">'
        f"{total} documents</text>"
    )
    for segment in _segments(batches, values):
        points = " ".join(f"{x(b.documents):.1f},{y(v):.1f}" for b, v in segment)
        if len(segment) > 1:
            parts.append(f'<polyline class="line" points="{points}"/>')
    for batch, value in zip(batches, values, strict=True):
        if value is None:
            continue
        label = (
            f"batch {batch.number}: documents {batch.documents - batch.size + 1}–"
            f"{batch.documents}\n{metric.title}: {metric.fmt(value)}"
        )
        cx, cy = x(batch.documents), y(value)
        parts.append(
            f'<g class="point"><title>{html.escape(label)}</title>'
            f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="12"/>'
            f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="4"/></g>'
        )
    found = [(b, v) for b, v in zip(batches, values, strict=True) if v is not None]
    if found:
        last = found[-1]
        parts.append(
            f'<text x="{x(last[0].documents) + 10:.1f}" y="{y(last[1]) + 4:.1f}">'
            f"{html.escape(metric.fmt(last[1]))}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _segments(
    batches: list[ReplayBatch], values: list[float | None]
) -> list[list[tuple[ReplayBatch, float]]]:
    """Runs of batches with a value: the line breaks where a batch has none."""
    segments: list[list[tuple[ReplayBatch, float]]] = [[]]
    for batch, value in zip(batches, values, strict=True):
        if value is None:
            segments.append([])
        else:
            segments[-1].append((batch, value))
    return [s for s in segments if s]


def _requests(x: float) -> str:
    return f"{x:.1f}"


_TABLE: tuple[tuple[str, str, Callable[[Any], str]], ...] = (
    ("batch", "Batch", str),
    ("documents", "Documents", str),
    ("waves", "Waves", str),
    ("accuracy", "Accuracy", _pct),
    ("cost_per_document", "Cost/doc", _usd),
    ("llm_calls_per_document", "LLM calls/doc", _calls),
    ("jev_requests_per_document", "Jev requests/doc", _requests),
    ("generators", "Generators", str),
    ("errors", "Errors", str),
)


def _table_row(row: Mapping[str, Any]) -> str:
    cells = (
        "–" if row[key] in (None, "") else html.escape(fmt(row[key])) for key, _, fmt in _TABLE
    )
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _render_html(report: ReplayReport, title: str) -> str:
    batches = report.batches()
    summary = report.report.summary()
    lede = (
        f"{summary['documents']} documents from an empty store in batches of "
        f"{report.batch_size}; accuracy "
        f"{'–' if summary['accuracy'] is None else _pct(summary['accuracy'])} overall, "
        f"{summary['llm_calls_per_document']:.2f} LLM calls and "
        f"{_usd(summary['cost_per_document'])} per document."
    )
    charts = "".join(
        f"<h2>{html.escape(m.title)}</h2>{_chart(report, batches, m)}" for m in _METRICS
    )
    rows = "".join(_table_row(b.row()) for b in batches)
    head = "".join(f"<th>{html.escape(name)}</th>" for _, name, _ in _TABLE)
    data = json.dumps(report.to_dict(), ensure_ascii=False).replace("</", "<\\/")
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f'<body class="viz-root"><h1>{html.escape(title)}</h1>'
        f'<p class="lede">{html.escape(lede)}</p>{charts}'
        f"<h2>Batches</h2><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
        f'<script type="application/json" id="replay-data">{data}</script>'
        "</body></html>\n"
    )
