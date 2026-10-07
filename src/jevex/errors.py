"""What went wrong while extracting a document, reported on the result (#230's policy).

jevex reports and the caller decides. :meth:`Extractor.extract <jevex.Extractor.extract>`
returns a result for everything that happened while extracting a document, and raises
only for caller mistakes (bad arguments, an unknown schema) and the process spend caps.

- A **part** failing (a generator, a normaliser, an image loader or processor, the
  structured extractor, the LLM fallback, the review sink) is skipped for that input, the
  failure is recorded as a :class:`PartError`, and the pipeline carries on: the result's
  ``status`` is ``"partial"``.
- A **core** failure (Jev after retries, a document that can't be read, a bug in a stage)
  ends the document: the result's ``status`` is ``"failed"``, with the error.
- :meth:`ExtractionResult.raise_for_errors <jevex.ExtractionResult.raise_for_errors>` is
  the opt-in way to fail loudly: it raises :class:`ExtractionError`.

Each failure is recorded once per stage, part and exception type, with a count, so a
generator that fails on every statement is one error, not hundreds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type Status = Literal["ok", "partial", "failed"]
"""``ok``: nothing failed. ``partial``: a part failed and was skipped. ``failed``: a core
failure ended the document."""

PartKind = Literal[
    "generator",
    "normaliser",
    "image_loader",
    "image_processor",
    "structured_extractor",
    "llm_extractor",
    "review_sink",
    "jev",
    "store",
    "document",
    "stage",
]
"""What failed. Parts that are skipped: ``generator``, ``normaliser``, ``image_loader``,
``image_processor``, ``structured_extractor``, ``llm_extractor`` (the fallback's) and
``review_sink``. Core failures, which fail the document: ``jev`` (Jev after retries),
``store`` (while the document runs), ``document`` (it can't be read:
:class:`DocumentError`) and ``stage`` (any other error a stage raised). A ``store`` error
writing the spend ledger or stats after the document ran isn't fatal: its result stands,
``partial``."""

MAX_MESSAGE_CHARS = 500
"""Longer messages are cut: one is kept per error, and results are serialised."""


class PartError(BaseModel):
    """One kind of failure in one document: where it happened, what raised, how often.

    ``part`` names the part (a generator id, a normaliser's name, a processor's class)
    when there is one. ``message`` is the first occurrence's. ``fatal`` errors are core
    failures: they failed the document.
    """

    model_config = ConfigDict(frozen=True)

    stage: str
    kind: PartKind
    part: str | None = None
    type: str
    """The exception's class name."""
    message: str
    count: int = Field(default=1, ge=1)
    fatal: bool = False

    def describe(self) -> str:
        """One line for logs and CLI output."""
        where = f"{self.stage} {self.kind}" + (f" {self.part}" if self.part else "")
        times = f" (x{self.count})" if self.count > 1 else ""
        return f"{where}: {self.type}: {self.message}{times}"


class DocumentError(Exception):
    """The document can't be read (a broken PDF or image, an unsupported type): a core
    failure, reported with kind ``document``."""


class ExtractionError(Exception):
    """Raised by :meth:`~jevex.ExtractionResult.raise_for_errors` for a result that isn't
    ``ok``. ``errors`` are the result's; a failed result's original exception is the
    ``__cause__``."""

    def __init__(self, status: Status, errors: list[PartError], url: str | None = None) -> None:
        self.status: Status = status
        self.errors = errors
        self.url = url
        where = f" for {url}" if url else ""
        lines = "; ".join(e.describe() for e in errors)
        super().__init__(f"extraction {status}{where}: {lines}")


@dataclass
class PartErrors:
    """A document's errors as they're recorded, merged per (stage, kind, part, type)."""

    _errors: dict[tuple[str, str, str | None, str], PartError] = field(
        default_factory=dict[tuple[str, str, str | None, str], PartError]
    )
    cause: BaseException | None = None
    """The first fatal error's exception, for :class:`ExtractionError`'s ``__cause__``."""

    def add(
        self,
        stage: str,
        kind: PartKind,
        part: str | None,
        exc: BaseException,
        *,
        fatal: bool = False,
    ) -> None:
        key = (stage, kind, part, type(exc).__name__)
        found = self._errors.get(key)
        if found is not None:
            self._errors[key] = found.model_copy(update={"count": found.count + 1})
            return
        self._errors[key] = PartError(
            stage=stage,
            kind=kind,
            part=part,
            type=type(exc).__name__,
            message=_message(exc),
            fatal=fatal,
        )
        if fatal and self.cause is None:
            self.cause = exc

    @property
    def errors(self) -> list[PartError]:
        """Every error, in the order each was first recorded."""
        return list(self._errors.values())

    @property
    def status(self) -> Status:
        return status_of(self.errors)

    def __len__(self) -> int:
        return len(self._errors)


def status_of(errors: list[PartError]) -> Status:
    """``failed`` with a fatal error, ``partial`` with any other, else ``ok``."""
    if any(e.fatal for e in errors):
        return "failed"
    return "partial" if errors else "ok"


def _message(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    return text if len(text) <= MAX_MESSAGE_CHARS else f"{text[: MAX_MESSAGE_CHARS - 1]}…"
