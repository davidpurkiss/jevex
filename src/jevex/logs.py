"""Logging: what jevex does and what went wrong, through the standard :mod:`logging`.

Every module logs to ``logging.getLogger("jevex.<module>")`` (``jevex.extractor``,
``jevex.learn``...), wrapped by :func:`get_logger` so that each record carries where it
happened as attributes a formatter or a JSON handler can use:

- ``run_id``: the extractor's run (``Extractor(run_id=)``);
- ``document_id`` and ``url``: the document being extracted;
- ``stage``: the pipeline stage running;
- ``part``: the pluggable part involved (a generator id, a processor's class), if any.

Each is ``None`` when it doesn't apply (the learner works outside any document, so its
records have a ``run_id`` and no ``url``). The ``jevex`` logger has a
:class:`~logging.NullHandler`, so jevex is silent until the application configures
logging, for example::

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s [%(stage)s %(part)s] %(url)s: %(message)s",
    )

Levels: ``DEBUG`` for each stage, Jev request and event; ``INFO`` for each finished
document, learner outcome and generator published or pruned; ``WARNING`` for a part
skipped (the first of each kind in a document), a Jev retry or a budget hit; ``ERROR``
for a failed document or a learner that died. Failures are also on the result
(:mod:`jevex.errors`): logs are for operators, results for callers.
"""

from __future__ import annotations

import contextlib
import logging
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator, MutableMapping

ROOT = "jevex"

FIELDS = ("run_id", "document_id", "url", "stage", "part")
"""The attributes every jevex log record has."""

logging.getLogger(ROOT).addHandler(logging.NullHandler())


@dataclass(frozen=True)
class LogContext:
    """Where the code logging now is working (see the module docstring)."""

    run_id: str | None = None
    document_id: str | None = None
    url: str | None = None
    stage: str | None = None
    part: str | None = None


_NOWHERE = LogContext()
_CONTEXT: ContextVar[LogContext] = ContextVar("jevex_log_context", default=_NOWHERE)


def current() -> LogContext:
    """The context records logged now get."""
    return _CONTEXT.get()


@contextlib.contextmanager
def log_context(**changes: str | None) -> Generator[LogContext]:
    """Set some of :class:`LogContext`'s fields for the code inside the block (and the
    tasks it starts, which copy the context)."""
    token = _CONTEXT.set(replace(_CONTEXT.get(), **changes))
    try:
        yield _CONTEXT.get()
    finally:
        _CONTEXT.reset(token)


class ContextLogger(logging.LoggerAdapter[logging.Logger]):
    """A logger whose records carry the :class:`LogContext` (an ``extra`` given to one
    call overrides it)."""

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, Any]:
        ctx = _CONTEXT.get()
        extra: dict[str, Any] = {name: getattr(ctx, name) for name in FIELDS}
        extra.update(kwargs.get("extra") or {})
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> ContextLogger:
    """The logger for a jevex module (pass ``__name__``)."""
    return ContextLogger(logging.getLogger(name))
