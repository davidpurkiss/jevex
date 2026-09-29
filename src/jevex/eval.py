"""Evaluate extraction against a labelled corpus (spec: *Evaluation and test site › jevex eval*).

A corpus is a directory with a ``truth.json`` in the test site's format (so
``jevex.testsite.build`` output works as-is)::

    {
        "pages": [
            {
                "path": "specs/x.html",
                "schema": "VehicleSpec",
                "records": [{"entity": "SE", "values": {...}}],
            }
        ]
    }

:func:`evaluate` runs an :class:`~jevex.Extractor` over every document and scores the
records it returns against the expected ones:

- per field: correct, wrong, missing (expected but not found) and spurious (found but not
  expected), giving precision and recall. Numbers match within a tolerance, strings
  after normalising case and whitespace, lists by item precision/recall;
- per run: cost per document, latency, Jev requests and questions, LLM calls, and the
  resolution mix (how many values each method produced).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from jevex.document import Document

if TYPE_CHECKING:
    from jevex.extractor import ExtractionResult, Extractor

DEFAULT_REL_TOL = 0.005
DEFAULT_ABS_TOL = 0.5
TRUTH_FILE = "truth.json"


# --- corpus ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Expected:
    entity: str
    values: dict[str, Any]


@dataclass(frozen=True)
class CorpusItem:
    path: Path
    schema: str
    records: tuple[Expected, ...]


def load_corpus(directory: str | Path) -> list[CorpusItem]:
    """Read ``truth.json`` from ``directory``; every listed document must exist."""
    root = Path(directory)
    truth = root / TRUTH_FILE
    try:
        manifest = json.loads(truth.read_text())
        pages = cast("list[dict[str, Any]]", manifest["pages"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{truth} isn't a corpus manifest: {exc}") from exc
    items: list[CorpusItem] = []
    for page in pages:
        path = root / str(page["path"])
        if not path.is_file():
            raise ValueError(f"{truth} lists {page['path']}, which doesn't exist")
        records = tuple(
            Expected(str(r["entity"]), dict(r["values"]))
            for r in cast("list[dict[str, Any]]", page["records"])
        )
        items.append(CorpusItem(path=path, schema=str(page["schema"]), records=records))
    return items


# --- comparing values ------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """Numbers match if within ``rel`` (fraction) or ``abs`` of each other."""

    rel: float = DEFAULT_REL_TOL
    abs: float = DEFAULT_ABS_TOL


DEFAULT_TOLERANCE = Tolerance()


def _canonical(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


def values_match(expected: Any, actual: Any, tolerance: Tolerance = DEFAULT_TOLERANCE) -> bool:
    """Whether a found value counts as the expected one (scalars; see :func:`list_scores`)."""
    e, a = _canonical(expected), _canonical(actual)
    if isinstance(e, bool) or isinstance(a, bool):  # True == 1 in Python; not here
        return isinstance(e, bool) and isinstance(a, bool) and e == a
    if isinstance(e, int | float) and isinstance(a, int | float):
        return math.isclose(float(e), float(a), rel_tol=tolerance.rel, abs_tol=tolerance.abs)
    if isinstance(e, str) and isinstance(a, int | float):
        try:
            return values_match(float(e), a, tolerance)
        except ValueError:
            return False
    return e == a


def list_scores(
    expected: list[Any], actual: list[Any], tolerance: Tolerance = DEFAULT_TOLERANCE
) -> tuple[int, int, int]:
    """(matched, expected count, found count) for list fields, each item matched once."""
    remaining = list(actual)
    matched = 0
    for item in expected:
        for i, candidate in enumerate(remaining):
            if values_match(item, candidate, tolerance):
                matched += 1
                del remaining[i]
                break
    return matched, len(expected), len(actual)


# --- scores ----------------------------------------------------------------------------


@dataclass
class FieldScore:
    """Counts for one schema field across the corpus."""

    correct: int = 0
    wrong: int = 0
    missing: int = 0
    spurious: int = 0
    empty: int = 0
    """Expected ``None`` and nothing found: correct, but not counted as a hit."""

    @property
    def precision(self) -> float | None:
        found = self.correct + self.wrong + self.spurious
        return self.correct / found if found else None

    @property
    def recall(self) -> float | None:
        expected = self.correct + self.wrong + self.missing
        return self.correct / expected if expected else None

    def add(self, other: FieldScore) -> None:
        self.correct += other.correct
        self.wrong += other.wrong
        self.missing += other.missing
        self.spurious += other.spurious
        self.empty += other.empty

    def to_dict(self) -> dict[str, Any]:
        return {
            "correct": self.correct,
            "wrong": self.wrong,
            "missing": self.missing,
            "spurious": self.spurious,
            "empty": self.empty,
            "precision": self.precision,
            "recall": self.recall,
        }


def score_value(expected: Any, actual: Any, tolerance: Tolerance = DEFAULT_TOLERANCE) -> FieldScore:
    """Score one expected value against one found value (``None`` = not found)."""
    s = FieldScore()
    exp_empty = expected is None or expected == []
    act_empty = actual is None or actual == []
    if exp_empty and act_empty:
        s.empty = 1
    elif exp_empty:
        s.spurious = 1
    elif act_empty:
        s.missing = 1
    elif isinstance(expected, list) or isinstance(actual, list):
        exp_list = cast("list[Any]", expected if isinstance(expected, list) else [expected])
        act_list = cast("list[Any]", actual if isinstance(actual, list) else [actual])
        matched, n_exp, n_act = list_scores(exp_list, act_list, tolerance)
        s.correct = matched
        s.wrong = min(n_exp, n_act) - matched
        s.missing = max(0, n_exp - n_act)
        s.spurious = max(0, n_act - n_exp)
    elif values_match(expected, actual, tolerance):
        s.correct = 1
    else:
        s.wrong = 1
    return s


def match_records(
    expected: tuple[Expected, ...],
    found: list[dict[str, Any]],
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> list[tuple[Expected | None, dict[str, Any] | None]]:
    """Pair expected and found records: same entity label first, then by field agreement.

    Unpaired expected records pair with ``None`` (every field missing) and unpaired found
    records with ``None`` (every field spurious).
    """
    pairs: list[tuple[Expected | None, dict[str, Any] | None]] = []
    left = list(expected)
    right = list(found)
    for exp in list(left):
        same = next((f for f in right if f["entity"] == exp.entity), None)
        if same is not None:
            pairs.append((exp, same))
            left.remove(exp)
            right.remove(same)

    def agreement(exp: Expected, rec: dict[str, Any]) -> int:
        values = cast("dict[str, Any]", rec["values"])
        return sum(
            1
            for k, v in exp.values.items()
            if v is not None and values_match(v, values.get(k), tolerance)
        )

    while left and right:
        exp, rec = max(
            ((e, r) for e in left for r in right), key=lambda er: agreement(er[0], er[1])
        )
        pairs.append((exp, rec))
        left.remove(exp)
        right.remove(rec)
    pairs.extend((exp, None) for exp in left)
    pairs.extend((None, rec) for rec in right)
    return pairs


# --- running ---------------------------------------------------------------------------


@dataclass
class DocumentRun:
    path: str
    schema: str
    seconds: float
    jev_requests: int
    jev_questions: int
    jev_cost: float
    llm_calls: int
    """LLM calls for this document. Always 0 until the LLM fallback (#33) reports them."""
    llm_cost: float
    methods: Counter[str]
    fields: dict[str, FieldScore]
    error: str | None = None

    @property
    def cost(self) -> float:
        return self.jev_cost + self.llm_cost


@dataclass
class EvalReport:
    documents: list[DocumentRun] = field(default_factory=list[DocumentRun])

    def field_scores(self) -> dict[str, FieldScore]:
        """Per ``Schema.field`` totals across documents."""
        out: dict[str, FieldScore] = {}
        for doc in self.documents:
            for name, s in doc.fields.items():
                out.setdefault(f"{doc.schema}.{name}", FieldScore()).add(s)
        return dict(sorted(out.items()))

    def overall(self) -> FieldScore:
        total = FieldScore()
        for s in self.field_scores().values():
            total.add(s)
        return total

    def summary(self) -> dict[str, Any]:
        n = len(self.documents) or 1
        overall = self.overall()
        methods: Counter[str] = Counter()
        for doc in self.documents:
            methods.update(doc.methods)
        return {
            "documents": len(self.documents),
            "errors": sum(1 for d in self.documents if d.error),
            "precision": overall.precision,
            "recall": overall.recall,
            "cost_per_document": sum(d.cost for d in self.documents) / n,
            "seconds_per_document": sum(d.seconds for d in self.documents) / n,
            "jev_requests_per_document": sum(d.jev_requests for d in self.documents) / n,
            "llm_calls_per_document": sum(d.llm_calls for d in self.documents) / n,
            "resolution_mix": dict(sorted(methods.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "fields": {k: v.to_dict() for k, v in self.field_scores().items()},
            "documents": [
                {
                    "path": d.path,
                    "schema": d.schema,
                    "seconds": d.seconds,
                    "cost": d.cost,
                    "jev_requests": d.jev_requests,
                    "llm_calls": d.llm_calls,
                    "error": d.error,
                }
                for d in self.documents
            ],
        }


def _found_records(result: ExtractionResult, schema: str) -> list[dict[str, Any]]:
    return [
        {"entity": r.entity, "values": {n: getattr(r.record, n) for n in r.record.model_fields_set}}
        for r in result.records
        if r.schema_name == schema
    ]


def score_document(
    item: CorpusItem,
    result: ExtractionResult,
    fields: list[str],
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> dict[str, FieldScore]:
    """Per-field scores for one document."""
    scores = {name: FieldScore() for name in fields}
    for exp, rec in match_records(item.records, _found_records(result, item.schema), tolerance):
        exp_values = exp.values if exp else {}
        rec_values = cast("dict[str, Any]", rec["values"]) if rec else {}
        for name in fields:
            scores[name].add(score_value(exp_values.get(name), rec_values.get(name), tolerance))
    return scores


async def evaluate(
    extractor: Extractor,
    corpus: list[CorpusItem],
    *,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
    concurrency: int = 4,
) -> EvalReport:
    """Run ``extractor`` over the corpus and score it. Documents run ``concurrency`` at a time."""
    fields = {s.name: [f.name for f in s.fields] for s in extractor.schemas}
    unknown = sorted({i.schema for i in corpus} - set(fields))
    if unknown:
        raise ValueError(f"the corpus uses schemas the extractor doesn't have: {unknown}")
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: CorpusItem) -> DocumentRun:
        async with semaphore:
            document = Document.from_path(item.path, url=item.path.as_posix())
            start = time.perf_counter()
            try:
                result = await extractor.extract(document)
            except Exception as exc:  # scored as all-missing; the run carries on
                return DocumentRun(
                    path=item.path.as_posix(),
                    schema=item.schema,
                    seconds=time.perf_counter() - start,
                    jev_requests=0,
                    jev_questions=0,
                    jev_cost=0.0,
                    llm_calls=0,
                    llm_cost=0.0,
                    methods=Counter(),
                    fields=_all_missing(item, fields[item.schema], tolerance),
                    error=f"{type(exc).__name__}: {exc}",
                )
            seconds = time.perf_counter() - start
        methods: Counter[str] = Counter(
            m.method
            for r in result.records
            for m in r.meta.values()
            if m.found and not m.filtered and m.method
        )
        return DocumentRun(
            path=item.path.as_posix(),
            schema=item.schema,
            seconds=seconds,
            jev_requests=result.meta.jev.requests,
            jev_questions=result.meta.jev.questions,
            jev_cost=result.meta.jev.cost,
            llm_calls=0,
            llm_cost=0.0,
            methods=methods,
            fields=score_document(item, result, fields[item.schema], tolerance),
        )

    return EvalReport(documents=list(await asyncio.gather(*(one(i) for i in corpus))))


def _all_missing(
    item: CorpusItem, fields: list[str], tolerance: Tolerance
) -> dict[str, FieldScore]:
    scores = {n: FieldScore() for n in fields}
    for exp in item.records:
        for n in fields:
            scores[n].add(score_value(exp.values.get(n), None, tolerance))
    return scores
