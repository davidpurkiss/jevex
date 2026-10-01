"""The regression gate for ``jevex eval --gate`` (spec: *Evaluation and test site › jevex
eval*).

A :class:`Baseline` is a JSON file holding one eval run's headline numbers: overall and
per-field accuracy and the LLM-call rate, plus the :class:`GateTolerances` the gate
allows. ``jevex eval --write-baseline PATH`` writes one; ``jevex eval --gate PATH``
compares a new run with it and fails (exit 1) when :func:`check_baseline` finds a regression:

- overall accuracy fell by more than ``accuracy_drop``;
- a field's accuracy fell by more than ``field_accuracy_drop`` (``None`` skips this);
- LLM calls per document rose by more than ``llm_rate_rise``.

Improvements never fail the gate. Every tolerance is absolute: accuracies are fractions
(0.02 is two percentage points), the LLM-call rate is calls per document.

A baseline only means something for the corpus and mode it was recorded on, so it
records both: :func:`corpus_digest` (a hash of ``truth.json`` and every document it
lists) and whether it came from a plain run or a ``--replay``. :func:`check_baseline` refuses a
run of anything else with :class:`BaselineError` rather than comparing unlike numbers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jevex.eval import TRUTH_FILE, load_corpus

if TYPE_CHECKING:
    from jevex.eval import EvalReport

BASELINE_VERSION = 1

EvalMode = Literal["eval", "replay"]

_MODES: dict[EvalMode, str] = {"eval": "a plain eval", "replay": "a --replay run"}

_SLACK = 1e-9
"""Float noise allowed on top of a tolerance, so a drop of exactly the tolerance passes."""


class BaselineError(ValueError):
    """A baseline that can't be read, or doesn't fit the run it's compared with."""


class GateTolerances(BaseModel):
    """How far a run may fall behind its baseline before the gate fails."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accuracy_drop: float = Field(default=0.02, ge=0.0, le=1.0)
    """Overall accuracy may fall by this much (a fraction: 0.02 is two points)."""
    field_accuracy_drop: float | None = Field(default=0.05, ge=0.0, le=1.0)
    """Each field's accuracy may fall by this much; ``None`` checks only the overall."""
    llm_rate_rise: float = Field(default=0.1, ge=0.0)
    """LLM calls per document may rise by this much."""


class Baseline(BaseModel):
    """One eval run's numbers to gate later runs against. Serialised as JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = BASELINE_VERSION
    corpus: str
    """:func:`corpus_digest` of the corpus the run scored."""
    mode: EvalMode = "eval"
    """``"replay"`` for a ``jevex eval --replay`` run, whose LLM-call rate includes learning
    from an empty store; ``"eval"`` otherwise."""
    documents: int = Field(ge=0)
    accuracy: float | None
    """Overall accuracy (``None`` when nothing was scored)."""
    llm_calls_per_document: float = Field(ge=0.0)
    fields: dict[str, float | None] = Field(default_factory=dict[str, float | None])
    """Accuracy per ``Schema.field``."""
    tolerances: GateTolerances = GateTolerances()

    @classmethod
    def from_report(
        cls,
        report: EvalReport,
        *,
        corpus: str,
        mode: EvalMode = "eval",
        tolerances: GateTolerances | None = None,
    ) -> Baseline:
        """The baseline a run of ``report`` sets, keeping ``tolerances`` (default ones if
        not given)."""
        summary = report.summary()
        return cls(
            corpus=corpus,
            mode=mode,
            documents=summary["documents"],
            accuracy=summary["accuracy"],
            llm_calls_per_document=summary["llm_calls_per_document"],
            fields={name: s.accuracy for name, s in report.field_scores().items()},
            tolerances=tolerances or GateTolerances(),
        )

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        """Read a baseline file. Raises :class:`BaselineError` naming the file if it's
        missing or not a baseline."""
        try:
            return cls.model_validate_json(Path(path).read_bytes())
        except OSError as exc:
            raise BaselineError(f"can't read baseline {path}: {exc.strerror or exc}") from exc
        except ValidationError as exc:
            raise BaselineError(f"{path} isn't a jevex baseline: {exc}") from exc

    def write(self, path: str | Path) -> None:
        """Write the baseline as indented JSON with sorted keys, so diffs stay readable."""
        data = self.model_dump(mode="json")
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def corpus_digest(directory: str | Path) -> str:
    """A hash of a corpus: its ``truth.json`` and the bytes of every document it lists.

    Raises ``ValueError`` (from :func:`~jevex.eval.load_corpus`) for a malformed corpus.
    """
    root = Path(directory)
    items = load_corpus(root)
    h = hashlib.sha256((root / TRUTH_FILE).read_bytes())
    for item in items:
        h.update(item.path.relative_to(root).as_posix().encode())
        h.update(item.path.read_bytes())
    return h.hexdigest()


class Check(BaseModel):
    """One number the gate compared: the baseline's, the run's, and the worst it allows."""

    model_config = ConfigDict(frozen=True)

    metric: str
    """``accuracy``, ``llm_calls_per_document`` or ``accuracy:<Schema.field>``."""
    baseline: float | None
    value: float | None
    limit: float
    higher_is_better: bool

    @property
    def passed(self) -> bool:
        """Whether the run's value is within ``limit`` (a run scoring nothing passes only
        when the baseline scored nothing too)."""
        if self.value is None:
            # Nothing scored now: only a regression if something was scored before.
            return self.baseline is None
        if self.higher_is_better:
            return self.value >= self.limit - _SLACK
        return self.value <= self.limit + _SLACK

    def describe(self) -> str:
        """``accuracy 88.0% (baseline 92.0%, at least 90.0%)``, or the rate's equivalent."""

        def show(x: float | None) -> str:
            if x is None:
                return "none"
            return f"{x * 100:.1f}%" if self.higher_is_better else f"{x:.2f}"

        bound = "at least" if self.higher_is_better else "at most"
        return (
            f"{self.metric} {show(self.value)} "
            f"(baseline {show(self.baseline)}, {bound} {show(self.limit)})"
        )


class GateResult(BaseModel):
    """Every check :func:`check_baseline` made; the gate passes when all of them do."""

    model_config = ConfigDict(frozen=True)

    checks: tuple[Check, ...]

    @property
    def regressions(self) -> list[Check]:
        """The checks that failed, in order."""
        return [c for c in self.checks if not c.passed]

    @property
    def passed(self) -> bool:
        """True when no check failed."""
        return not self.regressions


def ensure_comparable(baseline: Baseline, *, corpus: str, mode: EvalMode) -> None:
    """Raise :class:`BaselineError` unless a run of ``mode`` on ``corpus`` can be gated
    against ``baseline``. Cheap, so callers can check before spending on the run."""
    if corpus != baseline.corpus:
        raise BaselineError(
            "the baseline was recorded on a different corpus (its documents or truth.json "
            "changed); record a new one with --write-baseline"
        )
    if mode != baseline.mode:
        raise BaselineError(
            f"the baseline is from {_MODES[baseline.mode]}, not {_MODES[mode]}: their "
            "LLM-call rates don't compare"
        )


def check_baseline(
    report: EvalReport,
    baseline: Baseline,
    *,
    corpus: str,
    mode: EvalMode = "eval",
    tolerances: GateTolerances | None = None,
) -> GateResult:
    """Compare a run with its baseline, using ``tolerances`` or else the baseline's own.

    ``corpus`` (:func:`corpus_digest`) and ``mode`` must match the baseline's
    (:func:`ensure_comparable`). A field in the baseline that the run scored nothing for
    counts as a regression; a field new since the baseline isn't checked.
    """
    ensure_comparable(baseline, corpus=corpus, mode=mode)
    tol = tolerances or baseline.tolerances
    summary = report.summary()
    checks: list[Check] = []
    if baseline.accuracy is not None:
        checks.append(
            Check(
                metric="accuracy",
                baseline=baseline.accuracy,
                value=summary["accuracy"],
                limit=max(0.0, baseline.accuracy - tol.accuracy_drop),
                higher_is_better=True,
            )
        )
    if tol.field_accuracy_drop is not None:
        scores = report.field_scores()
        for name, before in sorted(baseline.fields.items()):
            if before is None:
                continue
            score = scores.get(name)
            checks.append(
                Check(
                    metric=f"accuracy:{name}",
                    baseline=before,
                    value=score.accuracy if score else None,
                    limit=max(0.0, before - tol.field_accuracy_drop),
                    higher_is_better=True,
                )
            )
    checks.append(
        Check(
            metric="llm_calls_per_document",
            baseline=baseline.llm_calls_per_document,
            value=summary["llm_calls_per_document"],
            limit=baseline.llm_calls_per_document + tol.llm_rate_rise,
            higher_is_better=False,
        )
    )
    return GateResult(checks=tuple(checks))
