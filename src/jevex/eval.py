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

A page may also give its ``locale`` (a BCP 47 tag): the document gets it as
:attr:`~jevex.Document.locale`, for documents that can't say it themselves (the test
site's PDFs and images).

:func:`evaluate` runs an :class:`~jevex.Extractor` over every document and scores the
records it returns against the expected ones:

- per field: correct, wrong, missing (expected but not found) and spurious (found but not
  expected), giving precision and recall. Numbers match within the field's tolerance
  (:func:`field_tolerance`), strings after normalising case and whitespace, lists item
  by item;
- per run: cost per document, latency (mean, p50, p95), Jev requests and questions, LLM
  calls, and the resolution mix (how many values each method produced).

Errors that make the whole run meaningless (Jev's spend cap, a bad key or an unreachable
API: :data:`RUN_ERRORS`) stop :func:`evaluate`. Any other failed document (its result's
``status`` is ``failed``, :mod:`jevex.errors`) is recorded on its :class:`DocumentRun`
and scored as all missing; a ``partial`` one is scored as it is, with its errors listed.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
import types
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast, get_args, get_origin

from jevex.document import LOCALE_TAG, Document
from jevex.errors import ExtractionError
from jevex.generators.units import UNITS
from jevex.jev import JevBackendError, JevBudgetExceededError
from jevex.normalise import NormaliseError, canonical_unit, convert

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from jevex.extractor import ExtractionResult, Extractor
    from jevex.schema import FieldSpec, SchemaSpec

TRUTH_FILE = "truth.json"
MEASURED_REL_TOL = 0.005
"""±0.5% for measured quantities (floats), per docs/benchmarks.md."""

RUN_ERRORS: tuple[type[Exception], ...] = (JevBudgetExceededError, ExtractionError)
"""Errors that stop :func:`evaluate` instead of being scored against one document: the
spend cap, and a document failed by the Jev API (bad key, network, 5xx after retries),
raised as its result's :class:`~jevex.errors.ExtractionError` (the
:class:`~jevex.jev.JevBackendError` is its ``__cause__``)."""


# --- corpus ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Expected:
    """One expected record: its entity label and field values (JSON types)."""

    entity: str
    values: dict[str, Any]


@dataclass(frozen=True)
class CorpusItem:
    """One labelled document: where it is, the schema it's labelled in, what it holds."""

    path: Path
    schema: str
    records: tuple[Expected, ...]
    wave: int | None = None
    """The test-site wave the page was released in (its ``wave``), if the manifest says."""
    locale: str | None = None
    """The page's locale (its ``locale``), if the manifest says: the document's
    :attr:`~jevex.Document.locale`."""


def load_corpus(directory: str | Path) -> list[CorpusItem]:
    """Read ``truth.json`` from ``directory``; every listed document must exist.

    Raises ``ValueError`` naming the page for a malformed manifest.
    """
    root = Path(directory)
    truth = root / TRUTH_FILE
    try:
        manifest = json.loads(truth.read_text())
        pages = manifest["pages"]
        if not isinstance(pages, list):
            raise TypeError("'pages' must be a list")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{truth} isn't a corpus manifest: {exc}") from exc
    items: list[CorpusItem] = []
    for i, page in enumerate(cast("list[Any]", pages)):
        try:
            item = _corpus_item(root, page)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{truth} page {i}: {_describe(exc)}") from exc
        if not item.path.is_file():
            raise ValueError(f"{truth} lists {page['path']}, which doesn't exist")
        items.append(item)
    return items


def _corpus_item(root: Path, page: Any) -> CorpusItem:
    if not isinstance(page, dict):
        raise TypeError(f"a page must be an object, not {type(page).__name__}")
    entry = cast("dict[str, Any]", page)
    records = entry["records"]
    if not isinstance(records, list):
        raise TypeError("'records' must be a list")
    expected: list[Expected] = []
    for j, record in enumerate(cast("list[Any]", records)):
        r = cast("dict[str, Any]", record) if isinstance(record, dict) else None
        if r is None or not isinstance(r.get("values"), dict):
            raise TypeError(f"record {j} must be an object with a 'values' object")
        expected.append(Expected(str(r.get("entity", "")), dict(r["values"])))
    wave = entry.get("wave")
    if wave is not None and (not isinstance(wave, int) or isinstance(wave, bool)):
        raise TypeError(f"'wave' must be a number, not {wave!r}")
    locale = entry.get("locale")
    if locale is not None and not (isinstance(locale, str) and re.fullmatch(LOCALE_TAG, locale)):
        raise TypeError(f"'locale' must be a language tag such as 'en-GB', not {locale!r}")
    return CorpusItem(
        path=root / str(entry["path"]),
        schema=str(entry["schema"]),
        records=tuple(expected),
        wave=wave,
        locale=locale,
    )


def _describe(exc: Exception) -> str:
    return f"missing {exc}" if isinstance(exc, KeyError) else str(exc)


# --- comparing values ------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    """Numbers match if within ``rel`` (a fraction) or ``abs`` of each other. The default
    is an exact match."""

    rel: float = 0.0
    abs: float = 0.0


EXACT = Tolerance()


def field_tolerance(spec: FieldSpec) -> Tolerance:
    """The default tolerance for a field, from its type and unit.

    Counts (``int`` with no unit, like a year or seats) and money (``Decimal``, or a
    currency unit) are stated exactly, so they match exactly. Measured quantities (``float``)
    get ±0.5%. A field with a unit also allows the rounding a page introduces when it shows
    the value in a comparable unit: power as whole PS or bhp is up to 0.37 kW out, a speed in
    whole km/h up to 0.31 mph (:func:`rounding_slack`).
    """
    if spec.kind != "number":
        return EXACT
    base = _base_type(spec.annotation)
    if base is Decimal or (spec.unit and _is_currency(spec.unit)):
        return EXACT
    rel = MEASURED_REL_TOL if base is float else 0.0
    slack = rounding_slack(spec.unit) if spec.unit else 0.0
    return Tolerance(rel=rel, abs=slack)


def rounding_slack(unit: str) -> float:
    """How far a value in ``unit`` can be off after a page rounds it to a whole number in a
    comparable unit (one within a factor of two, so PS for kW but not W). 0 if none."""
    try:
        target = canonical_unit(unit)
    except NormaliseError:
        return 0.0
    slack = 0.0
    for other in {u.canonical for u in UNITS} - {target}:
        try:
            one = convert(1.0, other, target)
        except NormaliseError:
            continue
        if 0.5 <= one <= 2.0:
            slack = max(slack, one / 2)
    return slack


def _is_currency(unit: str) -> bool:
    return len(unit) == 3 and unit.isalpha() and unit.isupper()


def _base_type(annotation: Any) -> Any:
    """``int | None`` → ``int``; ``list[float]`` → ``float``."""
    while True:
        origin = get_origin(annotation)
        if origin in (Union, types.UnionType):
            rest = [a for a in get_args(annotation) if a is not type(None)]
            if len(rest) != 1:
                return annotation
            annotation = rest[0]
        elif origin is list:
            (annotation,) = get_args(annotation) or (Any,)
        else:
            return annotation


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


def values_match(expected: Any, actual: Any, tolerance: Tolerance = EXACT) -> bool:
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
    expected: list[Any], actual: list[Any], tolerance: Tolerance = EXACT
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

    @property
    def accuracy(self) -> float | None:
        """Correct values over every value that was expected, found, or both: one number
        that a wrong, a missing and a spurious value each lower."""
        scored = self.correct + self.wrong + self.missing + self.spurious
        return self.correct / scored if scored else None

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
            "accuracy": self.accuracy,
        }


def score_value(expected: Any, actual: Any, tolerance: Tolerance = EXACT) -> FieldScore:
    """Score one expected value against one found value (``None`` = not found).

    Lists count per item, including against an empty side: expecting three items and
    finding none is three missing, not one.
    """
    s = FieldScore()
    exp_empty = expected is None or expected == []
    act_empty = actual is None or actual == []
    if exp_empty and act_empty:
        s.empty = 1
    elif exp_empty:
        s.spurious = len(cast("list[Any]", actual)) if isinstance(actual, list) else 1
    elif act_empty:
        s.missing = len(cast("list[Any]", expected)) if isinstance(expected, list) else 1
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
    tolerances: Mapping[str, Tolerance] | None = None,
) -> list[tuple[Expected | None, dict[str, Any] | None]]:
    """Pair expected and found records by how many field values agree.

    The entity label only breaks ties: on a listing grid one skipped card shifts every
    positional label, so trusting labels first would pair every record wrongly. Pairing is
    greedy on the agreement matrix (best pair first), which is not always the optimal
    assignment but is close for records that mostly agree or mostly don't.

    Unpaired expected records pair with ``None`` (every field missing) and unpaired found
    records with ``None`` (every field spurious).
    """
    tol = tolerances or {}

    def agreement(exp: Expected, rec: dict[str, Any]) -> int:
        values = cast("dict[str, Any]", rec["values"])
        return sum(
            score_value(v, values.get(k), tol.get(k, EXACT)).correct
            for k, v in exp.values.items()
            if v is not None
        )

    ranked = sorted(
        (
            (agreement(e, r), e.entity == r["entity"], -i, -j)
            for i, e in enumerate(expected)
            for j, r in enumerate(found)
        ),
        reverse=True,
    )
    pairs: list[tuple[Expected | None, dict[str, Any] | None]] = []
    used_exp: set[int] = set()
    used_found: set[int] = set()
    for _, _, neg_i, neg_j in ranked:
        i, j = -neg_i, -neg_j
        if i not in used_exp and j not in used_found:
            pairs.append((expected[i], found[j]))
            used_exp.add(i)
            used_found.add(j)
    pairs.extend((e, None) for i, e in enumerate(expected) if i not in used_exp)
    pairs.extend((None, r) for j, r in enumerate(found) if j not in used_found)
    return pairs


# --- running ---------------------------------------------------------------------------


@dataclass
class DocumentRun:
    """What happened on one document: its metrics and per-field scores."""

    path: str
    schema: str
    seconds: float
    jev_requests: int
    jev_questions: int
    jev_cost: float
    llm_calls: int
    """LLM calls for this document (``meta.llm.calls``: the fallback's)."""
    llm_cost: float
    methods: Counter[str]
    fields: dict[str, FieldScore]
    """Keyed ``Schema.field``. Usually the document's own schema, plus spurious counts for
    any other schema the extractor wrongly found records of."""
    error: str | None = None
    """Set when extraction failed (``status="failed"``): what failed. The document then
    scores as all missing."""
    status: str = "ok"
    """The result's ``status``: ``ok``, ``partial`` or ``failed``."""
    warnings: list[str] = field(default_factory=list[str])
    """A ``partial`` document's errors: the parts that failed and were skipped."""

    @property
    def cost(self) -> float:
        return self.jev_cost + self.llm_cost


@dataclass
class EvalReport:
    """The result of :func:`evaluate`: one :class:`DocumentRun` per corpus document."""

    documents: list[DocumentRun] = field(default_factory=list[DocumentRun])

    @property
    def failed(self) -> list[DocumentRun]:
        """Documents whose extraction failed."""
        return [d for d in self.documents if d.error]

    @property
    def partial(self) -> list[DocumentRun]:
        """Documents where a part failed and was skipped (scored as found)."""
        return [d for d in self.documents if d.status == "partial"]

    def field_scores(self) -> dict[str, FieldScore]:
        """Per ``Schema.field`` totals across documents."""
        out: dict[str, FieldScore] = {}
        for doc in self.documents:
            for name, s in doc.fields.items():
                out.setdefault(name, FieldScore()).add(s)
        return dict(sorted(out.items()))

    def overall(self) -> FieldScore:
        """Every field's counts added together (micro-averaged precision and recall)."""
        total = FieldScore()
        for s in self.field_scores().values():
            total.add(s)
        return total

    def summary(self) -> dict[str, Any]:
        """Run-level metrics. Latency covers only documents that didn't fail."""
        n = len(self.documents) or 1
        overall = self.overall()
        methods: Counter[str] = Counter()
        for doc in self.documents:
            methods.update(doc.methods)
        seconds = sorted(d.seconds for d in self.documents if not d.error)
        return {
            "documents": len(self.documents),
            "errors": len(self.failed),
            "partial": len(self.partial),
            "precision": overall.precision,
            "recall": overall.recall,
            "accuracy": overall.accuracy,
            "cost_per_document": sum(d.cost for d in self.documents) / n,
            "jev_cost_per_document": sum(d.jev_cost for d in self.documents) / n,
            "llm_cost_per_document": sum(d.llm_cost for d in self.documents) / n,
            "seconds_per_document": sum(seconds) / len(seconds) if seconds else None,
            "latency_p50": _percentile(seconds, 0.5),
            "latency_p95": _percentile(seconds, 0.95),
            "jev_requests_per_document": sum(d.jev_requests for d in self.documents) / n,
            "jev_questions_per_document": sum(d.jev_questions for d in self.documents) / n,
            "llm_calls_per_document": sum(d.llm_calls for d in self.documents) / n,
            "resolution_mix": dict(sorted(methods.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        """The full report as JSON types: summary, per-field scores, per-document runs."""
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
                    "jev_questions": d.jev_questions,
                    "llm_calls": d.llm_calls,
                    "methods": dict(sorted(d.methods.items())),
                    "error": d.error,
                    "status": d.status,
                    "warnings": d.warnings,
                }
                for d in self.documents
            ],
        }


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """Linear interpolation between closest ranks; ``None`` for no values."""
    if not sorted_values:
        return None
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _found_records(result: ExtractionResult) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in result.records:
        values = {n: getattr(r.record, n) for n in r.record.model_fields_set}
        out.setdefault(r.schema_name, []).append({"entity": r.entity, "values": values})
    return out


def score_document(
    item: CorpusItem,
    result: ExtractionResult,
    tolerances: Mapping[str, Mapping[str, Tolerance]],
) -> dict[str, FieldScore]:
    """Per ``Schema.field`` scores for one document.

    ``tolerances`` maps each schema the extractor has to its per-field tolerances. The
    document's own schema is scored against its expected records; any record found for
    another schema is wrong by definition, so its values count as spurious there.
    """
    return score_records(item, _found_records(result), tolerances)


def score_records(
    item: CorpusItem,
    found: Mapping[str, list[dict[str, Any]]],
    tolerances: Mapping[str, Mapping[str, Tolerance]],
) -> dict[str, FieldScore]:
    """:func:`score_document` for records found some other way (a baseline's): ``found``
    maps each schema name to its records, each ``{"entity": ..., "values": {...}}`` holding
    only the values found."""
    own = tolerances[item.schema]
    scores = {f"{item.schema}.{name}": FieldScore() for name in own}
    for exp, rec in match_records(item.records, found.get(item.schema, []), own):
        exp_values = exp.values if exp else {}
        rec_values = cast("dict[str, Any]", rec["values"]) if rec else {}
        for name, tol in own.items():
            scores[f"{item.schema}.{name}"].add(
                score_value(exp_values.get(name), rec_values.get(name), tol)
            )
    for schema, records in found.items():
        if schema == item.schema:
            continue
        for rec in records:
            for name, value in cast("dict[str, Any]", rec["values"]).items():
                if value is not None:
                    key = f"{schema}.{name}"
                    scores.setdefault(key, FieldScore()).add(score_value(None, value))
    return scores


def resolve_tolerances(
    extractor: Extractor, overrides: Mapping[str, Tolerance] | None = None
) -> dict[str, dict[str, Tolerance]]:
    """Each schema's per-field tolerance: an override keyed ``Schema.field`` or ``field``,
    else :func:`field_tolerance`."""
    return schema_tolerances(extractor.schemas, overrides)


def schema_tolerances(
    schemas: Iterable[SchemaSpec], overrides: Mapping[str, Tolerance] | None = None
) -> dict[str, dict[str, Tolerance]]:
    """:func:`resolve_tolerances` for schemas without an extractor."""
    given = overrides or {}
    return {
        spec.name: {
            f.name: given.get(f"{spec.name}.{f.name}", given.get(f.name, field_tolerance(f)))
            for f in spec.fields
        }
        for spec in schemas
    }


def check_schemas(corpus: list[CorpusItem], tolerances: Mapping[str, object]) -> None:
    """Raise ``ValueError`` if the corpus uses a schema the extractor doesn't have
    (``tolerances`` is :func:`resolve_tolerances`' result, keyed by schema)."""
    unknown = sorted({i.schema for i in corpus} - set(tolerances))
    if unknown:
        raise ValueError(f"the corpus uses schemas the extractor doesn't have: {unknown}")


async def run_document(
    extractor: Extractor,
    item: CorpusItem,
    tolerances: Mapping[str, Mapping[str, Tolerance]],
) -> DocumentRun:
    """Extract one corpus document and score it (``tolerances`` as for
    :func:`score_document`).

    Raises the :data:`RUN_ERRORS`; any other failed document is recorded on the run,
    which then scores as all missing.
    """
    document = await asyncio.to_thread(
        Document.from_path, item.path, url=item.path.as_posix(), locale=item.locale
    )
    start = time.perf_counter()
    result = await extractor.extract(document)
    seconds = time.perf_counter() - start
    if isinstance(result.cause, JevBackendError):
        result.raise_for_errors()
    meta = result.meta
    failed = result.status == "failed"
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
        jev_requests=meta.jev.requests,
        jev_questions=meta.jev.questions,
        jev_cost=meta.jev.cost,
        llm_calls=meta.llm.calls,
        llm_cost=meta.llm.cost,
        methods=Counter() if failed else methods,
        fields=(
            all_missing(item, tolerances[item.schema])
            if failed
            else score_document(item, result, tolerances)
        ),
        error="; ".join(e.describe() for e in result.errors if e.fatal) if failed else None,
        status=result.status,
        warnings=[] if failed else [e.describe() for e in result.errors],
    )


async def evaluate(
    extractor: Extractor,
    corpus: list[CorpusItem],
    *,
    tolerances: Mapping[str, Tolerance] | None = None,
    concurrency: int = 4,
) -> EvalReport:
    """Run ``extractor`` over the corpus and score it. Documents run ``concurrency`` at a time.

    ``tolerances`` overrides :func:`field_tolerance` per field (keys ``Schema.field`` or
    ``field``). Raises the :data:`RUN_ERRORS` (spend cap, Jev API failure) rather than
    scoring them; other per-document errors are recorded and scored as all missing.
    """
    resolved = resolve_tolerances(extractor, tolerances)
    check_schemas(corpus, resolved)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: CorpusItem) -> DocumentRun:
        async with semaphore:
            return await run_document(extractor, item, resolved)

    tasks = [asyncio.create_task(one(i)) for i in corpus]
    try:
        runs = await asyncio.gather(*tasks)
    except BaseException:
        # gather doesn't cancel the rest on the first error: stop them, so a hit spend cap
        # or a dead API doesn't keep spending on documents whose results are discarded.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return EvalReport(documents=list(runs))


def all_missing(item: CorpusItem, tolerances: Mapping[str, Tolerance]) -> dict[str, FieldScore]:
    """Scores for a document nothing was found in (its extraction failed): every expected
    value missing. ``tolerances`` are its own schema's."""
    scores = {f"{item.schema}.{n}": FieldScore() for n in tolerances}
    for exp in item.records:
        for n, tol in tolerances.items():
            scores[f"{item.schema}.{n}"].add(score_value(exp.values.get(n), None, tol))
    return scores
