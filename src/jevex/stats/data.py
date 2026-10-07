"""The stats UI's data: one set of tables from a store or a replay (spec: *Stats UI ›
Data sources*), and the queries every view, the JSON API and the SVG export share.

A :class:`Stats` holds :class:`Point` rows (one per document from a store, one per batch
from a replay), generators, fields, events and cumulative spend. :func:`curve` buckets the
points for a chart; :func:`summary` gives the headline tiles; :func:`to_json` the API's
payloads. The UI only reads: nothing here runs extraction or writes to a store.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from jevex.results import Method
from jevex.store import SpendLedger

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from jevex.replay import ReplayReport
    from jevex.store import DocumentStat, GeneratorRecord, SpendEntry, Store, VerifiedExample

METHODS: tuple[str, ...] = get_args(Method)

XAxis = Literal["docs", "time"]
"""Documents processed (replays, eval reports) or wall-clock time (live stores)."""

SourceKind = Literal["store", "replay"]

DOCUMENT_LIMIT = 10_000
"""Most recent documents a store's stats read."""

CURVE_POINTS = 60
"""Most points a chart draws: documents are bucketed down to this many."""

LOWEST_VALUES = 3
"""Lowest-confidence values the fields view lists per field."""


@dataclass(frozen=True)
class Point:
    """``size`` documents ending ``documents`` documents in (at ``at``, if known), with
    their per-document averages. ``accuracy`` is a replay's (the store has no ground
    truth); ``generators`` is how many learned generators were enabled after it, if
    known. The ``learning_*`` costs are what the learner spent on their examples, also a
    replay's (``None`` where not measured); :attr:`cost_per_document` includes them."""

    documents: int
    size: int
    llm_calls_per_document: float
    jev_cost_per_document: float
    llm_cost_per_document: float
    methods: Mapping[str, int] = field(default_factory=dict[str, int])
    at: datetime | None = None
    accuracy: float | None = None
    generators: int | None = None
    waves: tuple[int, ...] = ()
    learning_jev_cost_per_document: float | None = None
    learning_llm_cost_per_document: float | None = None

    @property
    def learning_counted(self) -> bool:
        return (
            self.learning_jev_cost_per_document is not None
            or self.learning_llm_cost_per_document is not None
        )

    @property
    def cost_per_document(self) -> float:
        return (
            self.jev_cost_per_document
            + self.llm_cost_per_document
            + (self.learning_jev_cost_per_document or 0.0)
            + (self.learning_llm_cost_per_document or 0.0)
        )

    def x(self, axis: XAxis) -> float:
        """The point's position on ``axis``: documents, or POSIX seconds."""
        if axis == "docs":
            return float(self.documents)
        if self.at is None:
            raise ValueError("this point has no time: use the documents axis")
        return self.at.timestamp()


@dataclass(frozen=True)
class GeneratorStat:
    """A stored generator with its counts (spec: *Stats UI › Generators*)."""

    generator_id: str
    field: str
    scope: Mapping[str, str]
    documents: int
    hits: int
    wins: int
    disabled: bool
    created: datetime
    learned_from: tuple[str, ...]
    spec: Mapping[str, Any]
    example: str | None = None
    """The statement of the first example it was learned from, if the store has it."""

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.hits if self.hits else None

    @property
    def status(self) -> str:
        return "disabled" if self.disabled else "active"


@dataclass(frozen=True)
class FieldStat:
    """One ``"Schema.field"``: how its found values were resolved and how sure Jev was."""

    field: str
    n: int
    mean_confidence: float | None
    methods: Mapping[str, int]
    lowest: tuple[tuple[str, float], ...]
    """Its lowest-confidence recent values, ``(value, confidence)``, lowest first."""

    @property
    def fallback_rate(self) -> float:
        """The share of its values the LLM fallback resolved."""
        return self.methods.get("llm", 0) / self.n if self.n else 0.0


@dataclass(frozen=True)
class Event:
    """A budget hit, a stopped document or an error, ``documents`` documents in."""

    kind: str
    message: str
    documents: int
    at: datetime | None = None
    url: str | None = None


@dataclass(frozen=True)
class SpendPoint:
    """Cumulative spend (USD) by ``documents`` documents in (at ``at``, if known)."""

    documents: int
    jev: float
    llm: float
    at: datetime | None = None

    @property
    def total(self) -> float:
        return self.jev + self.llm


@dataclass
class Stats:
    """Everything the stats UI shows about one source.

    ``source`` names it (a store URL or a CSV path). ``waves`` is ``(documents before it,
    wave)`` for each test-site wave after the first, ``learned`` how many documents had
    been processed when each generator was learned and ``learned_at`` when (the curve's
    ticks on each axis; a replay has no times), and ``budget_usd`` an optional budget
    line for the cost view.
    """

    source: str
    kind: SourceKind
    points: list[Point]
    generators: list[GeneratorStat] = field(default_factory=list[GeneratorStat])
    fields: list[FieldStat] = field(default_factory=list[FieldStat])
    events: list[Event] = field(default_factory=list[Event])
    spend: list[SpendPoint] = field(default_factory=list[SpendPoint])
    waves: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    learned: list[int] = field(default_factory=list[int])
    learned_at: list[datetime] = field(default_factory=list[datetime])
    budget_usd: float | None = None
    started: datetime | None = None
    """When the first document started: the time axis begins here, since its charges and
    anything learned while it ran come before it finished."""

    @property
    def documents(self) -> int:
        return self.points[-1].documents if self.points else 0

    @property
    def has_time(self) -> bool:
        """Whether the time axis is available: every point has a time."""
        return bool(self.points) and all(p.at is not None for p in self.points)

    def default_axis(self) -> XAxis:
        """Time for a live store, documents for a replay (spec: *Stats UI › Views*)."""
        return "time" if self.kind == "store" and self.has_time else "docs"


# --- queries -------------------------------------------------------------------------


def curve(stats: Stats, max_points: int = CURVE_POINTS) -> list[Point]:
    """The points merged into at most ``max_points`` consecutive buckets of equal size
    (the last may be smaller), each weighted by its documents."""
    if max_points < 1:
        raise ValueError(f"max_points must be at least 1, got {max_points}")
    points = stats.points
    if len(points) <= max_points:
        return list(points)
    size = math.ceil(len(points) / max_points)
    return [merge(points[i : i + size]) for i in range(0, len(points), size)]


def merge(points: Sequence[Point]) -> Point:
    """Consecutive points as one: averages weighted by size, method counts added."""
    if not points:
        raise ValueError("nothing to merge")
    n = sum(p.size for p in points)

    def mean(key: str) -> float:
        return sum(getattr(p, key) * p.size for p in points) / n if n else 0.0

    def measured_mean(key: str) -> float | None:
        values = [(cast("float | None", getattr(p, key)), p.size) for p in points]
        if all(v is None for v, _ in values):
            return None
        return sum((v or 0.0) * size for v, size in values) / n if n else 0.0

    scored = [p for p in points if p.accuracy is not None]
    scored_n = sum(p.size for p in scored)
    methods: dict[str, int] = {}
    for p in points:
        for m, count in p.methods.items():
            methods[m] = methods.get(m, 0) + count
    last = points[-1]
    return Point(
        documents=last.documents,
        size=n,
        llm_calls_per_document=mean("llm_calls_per_document"),
        jev_cost_per_document=mean("jev_cost_per_document"),
        llm_cost_per_document=mean("llm_cost_per_document"),
        methods=methods,
        at=last.at,
        accuracy=(
            sum(p.accuracy * p.size for p in scored if p.accuracy is not None) / scored_n
            if scored_n
            else None
        ),
        generators=last.generators,
        waves=tuple(sorted({w for p in points for w in p.waves})),
        learning_jev_cost_per_document=measured_mean("learning_jev_cost_per_document"),
        learning_llm_cost_per_document=measured_mean("learning_llm_cost_per_document"),
    )


def method_shares(counts: Mapping[str, int]) -> dict[str, float]:
    """Each method's share of ``counts`` (all zero when there are none)."""
    total = sum(counts.get(m, 0) for m in METHODS)
    return {m: counts.get(m, 0) / total if total else 0.0 for m in METHODS}


def shares(point: Point) -> dict[str, float]:
    """Each method's share of the point's values."""
    return method_shares(point.methods)


def summary(stats: Stats) -> dict[str, Any]:
    """The headline tiles: LLM calls and cost per document over the latest tenth of the
    documents, their change from the first tenth, accuracy over all of them (replays),
    and the learned generators enabled."""
    points = stats.points
    tenth = max(1, len(points) // 10)
    first = merge(points[:tenth]) if points else None
    last = merge(points[-tenth:]) if points else None
    everything = merge(points) if points else None

    def change(key: str) -> float | None:
        if first is None or last is None:
            return None
        before = getattr(first, key)
        return (getattr(last, key) - before) / before if before else None

    active = [g for g in stats.generators if not g.disabled]
    newest = max((p.at for p in points if p.at is not None), default=None)
    recent = (
        sum(1 for g in active if g.created >= newest - timedelta(days=1))
        if newest is not None
        else None
    )
    return {
        "source": stats.source,
        "kind": stats.kind,
        "documents": stats.documents,
        "llm_calls_per_document": last.llm_calls_per_document if last else None,
        "llm_calls_change": change("llm_calls_per_document"),
        "cost_per_document": last.cost_per_document if last else None,
        "cost_change": change("cost_per_document"),
        "accuracy": everything.accuracy if everything else None,
        "generators": len(active) if stats.kind == "store" else (last.generators if last else 0),
        "generators_last_day": recent if stats.kind == "store" else None,
        "budget_usd": stats.budget_usd,
        "spent_usd": stats.spend[-1].total if stats.spend else 0.0,
    }


def to_json(stats: Stats, view: str, *, max_points: int = CURVE_POINTS) -> Any:
    """A view's data as JSON types: ``summary``, ``learning``, ``mix``, ``cost``,
    ``generators``, ``fields``, ``events`` or ``all`` (every one, by name). Raises
    ``KeyError`` for another view."""
    views = {
        "summary": lambda: summary(stats),
        "learning": lambda: {
            "points": [_point_json(p) for p in curve(stats, max_points)],
            "waves": [{"documents": d, "wave": w} for d, w in stats.waves],
            "learned": stats.learned,
        },
        "mix": lambda: [
            {"documents": p.documents, "at": _iso(p.at), **shares(p)}
            for p in curve(stats, max_points)
        ],
        "cost": lambda: {
            "budget_usd": stats.budget_usd,
            "points": [
                {"documents": s.documents, "at": _iso(s.at), "jev": s.jev, "llm": s.llm}
                for s in stats.spend
            ],
        },
        "generators": lambda: [_generator_json(g) for g in stats.generators],
        "fields": lambda: [_field_json(f) for f in stats.fields],
        "events": lambda: [
            {
                "kind": e.kind,
                "message": e.message,
                "documents": e.documents,
                "at": _iso(e.at),
                "url": e.url,
            }
            for e in stats.events
        ],
    }
    if view == "all":
        return {name: build() for name, build in views.items()}
    return views[view]()


VIEWS: tuple[str, ...] = (
    "summary",
    "learning",
    "mix",
    "cost",
    "generators",
    "fields",
    "events",
    "all",
)
"""The views :func:`to_json` answers (``GET /stats/api/<view>``)."""


def _iso(at: datetime | None) -> str | None:
    return at.isoformat() if at is not None else None


def _point_json(p: Point) -> dict[str, Any]:
    return {
        "documents": p.documents,
        "size": p.size,
        "at": _iso(p.at),
        "llm_calls_per_document": p.llm_calls_per_document,
        "cost_per_document": p.cost_per_document,
        "jev_cost_per_document": p.jev_cost_per_document,
        "llm_cost_per_document": p.llm_cost_per_document,
        "learning_jev_cost_per_document": p.learning_jev_cost_per_document,
        "learning_llm_cost_per_document": p.learning_llm_cost_per_document,
        "accuracy": p.accuracy,
        "generators": p.generators,
        "waves": list(p.waves),
        "methods": dict(p.methods),
    }


def _generator_json(g: GeneratorStat) -> dict[str, Any]:
    return {
        "id": g.generator_id,
        "field": g.field,
        "scope": dict(g.scope),
        "documents": g.documents,
        "hits": g.hits,
        "wins": g.wins,
        "win_rate": g.win_rate,
        "status": g.status,
        "created": g.created.isoformat(),
        "learned_from": list(g.learned_from),
        "example": g.example,
        "spec": dict(g.spec),
    }


def _field_json(f: FieldStat) -> dict[str, Any]:
    return {
        "field": f.field,
        "n": f.n,
        "mean_confidence": f.mean_confidence,
        "fallback_rate": f.fallback_rate,
        "methods": dict(f.methods),
        "lowest": [{"value": v, "confidence": c} for v, c in f.lowest],
    }


# --- from a store --------------------------------------------------------------------


async def from_store(
    store: Store,
    *,
    source: str,
    since: datetime | None = None,
    limit: int | None = DOCUMENT_LIMIT,
    budget_usd: float | None = None,
    ledger: SpendLedger | None = None,
) -> Stats:
    """A store's stats: its recorded documents (the newest ``limit``, at or after
    ``since``), its generators with their counts, and its spend.

    Spend comes from the spend ledger (``ledger``, else the store when it is one), which
    also has what the learner spent between documents, from the first document's time
    on. Without ledger entries (no run budget was set, or no ledger) it comes from the
    documents instead.
    """
    docs = await store.documents(since=since, limit=limit)
    records = await store.generators(include_disabled=True)
    counts = [await store.generator_stats(g.id) for g in records]
    examples = await _learned_from(store, records)
    # A document's charges are written while it runs, before its stat (``at`` is when it
    # finished), so spend is counted from when the first one started, and a charge
    # against the documents started by then.
    starts = sorted(d.at - timedelta(seconds=d.seconds) for d in docs)
    start = starts[0] if starts else since
    if ledger is None and isinstance(store, SpendLedger):
        ledger = store
    spent = await ledger.spend_entries(since=start) if ledger is not None else []
    entries = [e for e in spent if e.kind in ("jev", "llm")]
    times = [d.at for d in docs]
    learned_at = sorted(g.created_at for g in records if start is None or g.created_at >= start)
    generators = [
        GeneratorStat(
            generator_id=g.id,
            field=g.field,
            scope=g.scope,
            documents=c.documents,
            hits=c.hits,
            wins=c.wins,
            disabled=not g.enabled,
            created=g.created_at,
            learned_from=tuple(_provenance(g.spec)),
            spec=g.spec,
            example=next(
                (examples[i].statement for i in _provenance(g.spec) if i in examples), None
            ),
        )
        for g, c in zip(records, counts, strict=True)
    ]
    return Stats(
        source=source,
        kind="store",
        points=_document_points(docs),
        generators=generators,
        fields=field_stats(docs),
        events=[
            Event(kind=e.kind, message=e.message, documents=i + 1, at=d.at, url=d.url)
            for i, d in enumerate(docs)
            for e in d.events
        ],
        spend=_ledger_spend(entries, starts) if entries else _point_spend(_document_points(docs)),
        learned=[_processed_by(times, when) for when in learned_at] if docs else [],
        learned_at=learned_at if docs else [],
        budget_usd=budget_usd,
        started=starts[0] if starts else None,
    )


def _provenance(spec: Mapping[str, Any]) -> list[str]:
    """The ids of the examples a stored spec says it was learned from."""
    provenance: object = spec.get("provenance")
    if not isinstance(provenance, dict):
        return []
    learned: object = cast("dict[str, object]", provenance).get("learned_from")
    return [str(i) for i in cast("list[object]", learned)] if isinstance(learned, list) else []


async def _learned_from(
    store: Store, records: Sequence[GeneratorRecord]
) -> dict[str, VerifiedExample]:
    wanted = {i for g in records for i in _provenance(g.spec)}
    if not wanted:
        return {}
    found: dict[str, VerifiedExample] = {}
    for name in sorted({g.field for g in records}):
        found.update({e.id: e for e in await store.examples(name) if e.id in wanted})
    return found


def _document_points(docs: Sequence[DocumentStat]) -> list[Point]:
    return [
        Point(
            documents=i + 1,
            size=1,
            llm_calls_per_document=d.llm_calls,
            jev_cost_per_document=d.jev_cost,
            llm_cost_per_document=d.llm_cost,
            methods=_count(v.method for v in d.values if v.method),
            at=d.at,
        )
        for i, d in enumerate(docs)
    ]


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def _processed_by(times: Sequence[datetime], when: datetime) -> int:
    """How many of ``times`` (sorted) are at or before ``when``."""
    lo, hi = 0, len(times)
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] <= when:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _ledger_spend(entries: Sequence[SpendEntry], starts: Sequence[datetime]) -> list[SpendPoint]:
    out: list[SpendPoint] = []
    jev = llm = 0.0
    for e in entries:
        if e.kind == "jev":
            jev += e.amount_usd
        else:
            llm += e.amount_usd
        out.append(SpendPoint(documents=_processed_by(starts, e.at), jev=jev, llm=llm, at=e.at))
    return out


def _point_spend(points: Sequence[Point]) -> list[SpendPoint]:
    out: list[SpendPoint] = []
    jev = llm = 0.0
    for p in points:
        jev += (p.jev_cost_per_document + (p.learning_jev_cost_per_document or 0.0)) * p.size
        llm += (p.llm_cost_per_document + (p.learning_llm_cost_per_document or 0.0)) * p.size
        out.append(SpendPoint(documents=p.documents, jev=jev, llm=llm, at=p.at))
    return out


def field_stats(docs: Sequence[DocumentStat], lowest: int = LOWEST_VALUES) -> list[FieldStat]:
    """Per ``"Schema.field"``, sorted by field: the documents' found values counted by
    method, their mean confidence (over values that have one) and the ``lowest``
    lowest-confidence ones, the most recent first among equals."""
    by_field: dict[str, list[tuple[int, str | None, float | None, str]]] = {}
    for i, d in enumerate(docs):
        for v in d.values:
            by_field.setdefault(v.field, []).append((i, v.method, v.confidence, v.value))
    out: list[FieldStat] = []
    for name in sorted(by_field):
        values = by_field[name]
        confident = [c for _, _, c, _ in values if c is not None]
        low = sorted(
            ((i, c, text) for i, _, c, text in values if c is not None),
            key=lambda t: (t[1], -t[0]),
        )[:lowest]
        out.append(
            FieldStat(
                field=name,
                n=len(values),
                mean_confidence=sum(confident) / len(confident) if confident else None,
                methods=_count(m for _, m, _, _ in values if m),
                lowest=tuple((text, c) for _, c, text in low),
            )
        )
    return out


# --- from a replay -------------------------------------------------------------------


def from_replay(report: ReplayReport, *, source: str = "replay") -> Stats:
    """A replay's stats: one point per batch, with its waves and accuracy."""
    batches = [_replay_batch(b.row()) for b in report.batches()]
    return replace(_replay_stats(batches, source), waves=report.wave_starts())


def from_replay_csv(text: str, *, source: str = "replay") -> Stats:
    """The stats of a replay's CSV (:meth:`~jevex.replay.ReplayReport.to_csv`).

    Waves are marked at the start of the batch each one first appears in. Raises
    ``ValueError`` for a CSV without the replay's columns or with a bad number.
    """
    reader = csv.DictReader(io.StringIO(text))
    needed = {"documents", "size", "llm_calls_per_document", "jev_cost_per_document"}
    missing = needed - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"not a jevex replay CSV: no {', '.join(sorted(missing))} column")
    batches: list[tuple[Point, int]] = []
    for line, raw in enumerate(reader, start=2):
        try:
            batches.append(_replay_batch({k: _cell(v) for k, v in raw.items() if k is not None}))
        except ValueError as exc:
            raise ValueError(f"replay CSV line {line}: {exc}") from None
    return _replay_stats(batches, source)


def _cell(text: str | None) -> Any:
    if text is None or text == "":
        return None
    for parse in (int, float):
        try:
            return parse(text)
        except ValueError:
            pass
    return text  # the waves column: "1 2"


def _waves(value: Any) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(w) for w in str(value).split())


def _number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"{key} is {value!r}, not a number")


def _measured(row: Mapping[str, Any], key: str) -> float | None:
    """A column older CSVs lack: ``None`` there, a number otherwise."""
    return None if row.get(key) is None else _number(row, key)


def _replay_batch(r: Mapping[str, Any]) -> tuple[Point, int]:
    """A CSV row as a point and its count of failed documents."""
    point = Point(
        documents=int(_number(r, "documents")),
        size=int(_number(r, "size")),
        llm_calls_per_document=_number(r, "llm_calls_per_document"),
        jev_cost_per_document=_number(r, "jev_cost_per_document"),
        llm_cost_per_document=_number(r, "llm_cost_per_document"),
        methods={m: int(_number(r, f"values_{m}")) for m in METHODS},
        accuracy=None if r.get("accuracy") is None else _number(r, "accuracy"),
        generators=None if r.get("generators") is None else int(_number(r, "generators")),
        waves=_waves(r.get("waves")),
        learning_jev_cost_per_document=_measured(r, "learning_jev_cost_per_document"),
        learning_llm_cost_per_document=_measured(r, "learning_llm_cost_per_document"),
    )
    return point, int(_number(r, "errors"))


def _replay_stats(batches: Sequence[tuple[Point, int]], source: str) -> Stats:
    points = [p for p, _ in batches]
    waves: list[tuple[int, int]] = []
    seen: set[int] = set()
    for p in points:
        for w in p.waves:
            if w not in seen and seen:
                waves.append((p.documents - p.size, w))
            seen.add(w)
    learned: list[int] = []
    before = 0
    for p in points:
        if p.generators is not None and p.generators > before:
            learned += [p.documents] * (p.generators - before)
            before = p.generators
    events = [
        Event(kind="error", message=f"{errors} documents failed", documents=p.documents)
        for p, errors in batches
        if errors
    ]
    return Stats(
        source=source,
        kind="replay",
        points=points,
        events=events,
        spend=_point_spend(points),
        waves=waves,
        learned=learned,
    )
