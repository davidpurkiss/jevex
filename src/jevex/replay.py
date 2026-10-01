"""Learning curves: ``jevex eval --replay`` (spec: *Evaluation and test site › jevex eval*).

:func:`replay` starts from an empty store and runs an extractor over a corpus one document
at a time, in the corpus's order. After each document it waits for the learner to finish
what the document queued, so every document sees what the ones before it taught (and a
replay with recorded answers gives the same curve every time). It reports per batch of
documents (:class:`ReplayBatch`): accuracy, cost per document, LLM calls per document, what
the learner spent learning from the documents' examples, the resolution mix and the learned
generators in use. That is the headline claim's evidence: accuracy holds while cost
(learning included) and the LLM-call rate fall as generators are learned.

:meth:`ReplayReport.to_csv` gives one row per batch (:data:`CSV_COLUMNS`), and
:meth:`ReplayReport.to_html` the stats page (:mod:`jevex.stats`) as a self-contained report
charting them over documents processed, with the test site's waves marked.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jevex.eval import EvalReport, check_schemas, resolve_tolerances, run_document
from jevex.learn import LearningSpend
from jevex.stats.charts import pct, usd
from jevex.stats.data import METHODS, from_replay
from jevex.stats.page import render_page

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jevex.eval import CorpusItem, DocumentRun, Tolerance
    from jevex.extractor import Extractor
    from jevex.learn import GeneratorLearner
    from jevex.store import Store

REPLAY_BATCH_SIZE = 10


class LearningStoppedError(Exception):
    """The learner failed during a replay. The documents after it would learn nothing, so
    the curve would stop measuring learning: the replay ends instead."""


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
    "learning_jev_cost_per_document",
    "learning_llm_cost_per_document",
    "learning_llm_calls_per_document",
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
values each method resolved in it. The cost and LLM-call columns are the documents' own
(their ``meta``); the ``learning_*`` columns are what the learner spent learning from
their examples (``generator_llm`` calls and the Jev requests testing drafts), so a
batch's whole bill is ``cost_per_document`` plus the two ``learning_*_cost`` columns."""


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
    learning_jev_cost_per_document: float
    learning_llm_cost_per_document: float
    learning_llm_calls_per_document: float
    jev_requests_per_document: float
    jev_questions_per_document: float
    seconds_per_document: float | None
    errors: int
    generators: int
    methods: dict[str, int]

    @property
    def learning_cost_per_document(self) -> float:
        return self.learning_jev_cost_per_document + self.learning_llm_cost_per_document

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
            "learning_jev_cost_per_document": self.learning_jev_cost_per_document,
            "learning_llm_cost_per_document": self.learning_llm_cost_per_document,
            "learning_llm_calls_per_document": self.learning_llm_calls_per_document,
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
    learning: list[LearningSpend] = field(default_factory=list[LearningSpend])
    """What the learner spent after each document, learning from its examples (a
    document missing here spent nothing)."""

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
        learning = self._learning(start, end)
        size = end - start
        return ReplayBatch(
            number=number,
            documents=end,
            size=size,
            waves=tuple(sorted({w for w in self.waves[start:end] if w is not None})),
            accuracy=s["accuracy"],
            precision=s["precision"],
            recall=s["recall"],
            cost_per_document=s["cost_per_document"],
            jev_cost_per_document=s["jev_cost_per_document"],
            llm_cost_per_document=s["llm_cost_per_document"],
            llm_calls_per_document=s["llm_calls_per_document"],
            learning_jev_cost_per_document=learning.jev_cost / size,
            learning_llm_cost_per_document=learning.llm_cost / size,
            learning_llm_calls_per_document=learning.llm_calls / size,
            jev_requests_per_document=s["jev_requests_per_document"],
            jev_questions_per_document=s["jev_questions_per_document"],
            seconds_per_document=s["seconds_per_document"],
            errors=s["errors"],
            generators=self.generators[end - 1],
            methods=s["resolution_mix"],
        )

    def _learning(self, start: int, end: int) -> LearningSpend:
        total = LearningSpend()
        for spend in self.learning[start:end]:
            total = LearningSpend(
                jev_cost=total.jev_cost + spend.jev_cost,
                llm_calls=total.llm_calls + spend.llm_calls,
                llm_cost=total.llm_cost + spend.llm_cost,
            )
        return total

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
        """The run's summary (the documents' own), what the learner spent in all, and the
        batches as JSON types."""
        learning = self._learning(0, len(self.report.documents))
        return {
            "batch_size": self.batch_size,
            "summary": self.report.summary(),
            "learning": {
                "jev_cost": learning.jev_cost,
                "llm_calls": learning.llm_calls,
                "llm_cost": learning.llm_cost,
            },
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
        """The stats page in report mode (:func:`~jevex.stats.render_page`): a single file
        that works offline, charting LLM calls per document, cost per document (learning
        included) and accuracy, the resolution mix and cumulative cost over documents
        processed, with every wave's start marked. Its data is inlined as JSON
        (``<script type="application/json" id="stats-data">``)."""
        return render_page(from_replay(self), title=title, generated=self._lede())

    def _lede(self) -> str:
        s = self.report.summary()
        accuracy = "–" if s["accuracy"] is None else pct(s["accuracy"])
        learning = self._learning(0, len(self.report.documents))
        return (
            f"{s['documents']} documents from an empty store in batches of {self.batch_size}; "
            f"accuracy {accuracy} overall; learning cost {usd(learning.cost)} "
            f"({learning.llm_calls} generator LLM calls)"
        )


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

    Each document's cost and LLM calls are its own (its ``meta``). What the learner
    spends after it, learning from its examples (``generator_llm`` calls and the Jev
    requests testing drafts, :attr:`~jevex.learn.GeneratorLearner.spend`), is counted
    separately in :attr:`ReplayReport.learning`.
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
    learning: list[LearningSpend] = []
    learner: GeneratorLearner | None = None
    before = LearningSpend()
    for item in corpus:
        runs.append(await run_document(extractor, item, resolved))
        try:
            await extractor.wait_for_learning()
        except RuntimeError as exc:  # the learner's worker failed (GeneratorLearner.drain)
            cause = exc.__cause__ or exc
            raise LearningStoppedError(f"learning stopped after {item.path}: {cause}") from exc
        generators.append(len(await store.generators()) if store is not None else 0)
        current = await extractor.learner()
        if current is not learner:  # a new learner starts its totals from nothing
            learner, before = current, LearningSpend()
        spent = learner.spend if learner is not None else LearningSpend()
        learning.append(spent - before)
        before = spent
    return ReplayReport(
        report=EvalReport(documents=runs),
        batch_size=batch_size,
        waves=[item.wave for item in corpus],
        generators=generators,
        learning=learning,
    )


async def _holds_learned_state(store: Store) -> bool:
    return bool(
        await store.generators(include_disabled=True)
        or await store.disabled_generator_ids()
        or await store.key_mappings()
        or await store.examples(limit=1)
    )
