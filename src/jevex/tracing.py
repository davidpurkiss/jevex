"""OpenTelemetry tracing (the ``otel`` extra: ``pip install 'jevex[otel]'``).

Each document is a ``jevex.extract`` span, with a ``jevex.stage <name>`` span per stage
under it. The document's span is a root span, or a child of the span current when
``extract`` was called (a crawler's span for the page, say), so a document's stages are
one trace either way. Attributes:

- on the document span: ``jevex.url``, ``jevex.run_id``, ``jevex.document_id``,
  ``jevex.content_type``, ``jevex.schemas``, ``jevex.status``, ``jevex.records``,
  ``jevex.entities``, ``jevex.errors``, and the document's Jev and LLM use
  (``jevex.jev.requests``, ``jevex.jev.cost_usd``, ``jevex.jev.retries``,
  ``jevex.llm.calls``, ``jevex.llm.cost_usd``);
- on a stage span: ``jevex.stage``, the schemas still active and their entity count when
  it ends (``jevex.schemas``, ``jevex.entities``), and the stage's own Jev and LLM use.

A core failure is recorded as an exception on its stage's span and the document's, with
an error status; a part that fails and is skipped is an exception event on its stage's
span, with ``jevex.kind`` and ``jevex.part`` attributes.

Tracing is off unless configured. ``Extractor(tracer=)`` takes a tracer; by default
(``None``) jevex uses OpenTelemetry's global tracer when ``opentelemetry-api`` is
installed, which records nothing until the application (or ``opentelemetry-instrument``
reading the standard ``OTEL_*`` environment) sets a tracer provider. ``tracer=False``
turns tracing off. Without the extra nothing is imported.
"""

from __future__ import annotations

import contextlib
import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from opentelemetry.trace import Span, Tracer

    from jevex.errors import PartKind

TRACER_NAME = "jevex"

type Attribute = str | bool | int | float | Sequence[str]


def resolve_tracer(tracer: Tracer | bool | None) -> Tracer | None:
    """The tracer an extractor uses (see the module docstring). ``True`` insists on
    OpenTelemetry's global tracer: ``ImportError`` without the ``otel`` extra."""
    if tracer is False:
        return None
    if tracer is None or tracer is True:
        if tracer is None and importlib.util.find_spec("opentelemetry") is None:
            return None
        from jevex import __version__  # not at import: jevex/__init__ imports this module

        try:
            from opentelemetry import trace
        except ImportError as exc:
            # Another opentelemetry-* package without the API: tracing just stays off.
            if tracer is None:
                return None
            raise ImportError("tracing needs the otel extra: pip install 'jevex[otel]'") from exc
        return trace.get_tracer(TRACER_NAME, __version__)
    return tracer


@contextlib.contextmanager
def trace_span(
    tracer: Tracer | None, name: str, attributes: Mapping[str, Attribute] | None = None
) -> Generator[Span | None]:
    """A span (current inside the block) when tracing, else ``None``. An exception
    leaving the block is recorded on it with an error status."""
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name, attributes=attributes) as current:
        yield current


def set_attributes(current: Span | None, attributes: Mapping[str, Attribute | None]) -> None:
    """Set the attributes that have a value, when the span records."""
    if current is None or not current.is_recording():
        return
    current.set_attributes({k: v for k, v in attributes.items() if v is not None})


def record_failure(
    current: Span | None,
    exc: BaseException,
    *,
    stage: str,
    kind: PartKind,
    part: str | None,
    fatal: bool,
) -> None:
    """Record a failure on ``current``: an exception event, and for a fatal one an error
    status."""
    if current is None or not current.is_recording():
        return
    from opentelemetry.trace import Status, StatusCode

    attributes: dict[str, Attribute] = {"jevex.stage": stage, "jevex.kind": kind}
    if part is not None:
        attributes["jevex.part"] = part
    current.record_exception(exc, attributes=attributes)
    if fatal:
        current.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
