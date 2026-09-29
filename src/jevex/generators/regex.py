"""Declarative regex generators: the only kind the learner may create.

Patterns compile under RE2 (linear time, no backreferences), so a learned pattern can't
cause catastrophic backtracking on hostile input. The pattern length is capped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

from jevex.interfaces import Scope
from jevex.statements import Candidate, NormaliserStep, Span, Statement

if TYPE_CHECKING:
    from collections.abc import Iterator

MAX_PATTERN_LENGTH = 500


class InvalidGeneratorError(ValueError):
    """A declarative generator spec failed validation."""


class _Match(Protocol):
    def span(self, group: int = 0) -> tuple[int, int]: ...


class _Pattern(Protocol):
    @property
    def groups(self) -> int: ...

    def finditer(self, text: str) -> Iterator[_Match]: ...


def _has_byte_escape(pattern: str) -> bool:
    """Whether ``pattern`` uses RE2's ``\\C`` (any single byte), outside an escaped backslash."""
    i = 0
    while i < len(pattern):
        if pattern[i] == "\\":
            if pattern[i + 1 : i + 2] == "C":
                return True
            i += 2
        else:
            i += 1
    return False


def compile_re2(pattern: str) -> _Pattern:
    """Compile under RE2, raising :class:`InvalidGeneratorError` on anything RE2 rejects.

    ``\\C`` is refused too: it matches one byte of UTF-8, so on non-ASCII text its match
    offsets fall inside a character and the wrapper's byte-to-character mapping breaks.
    """
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise InvalidGeneratorError(
            f"pattern is {len(pattern)} characters; the limit is {MAX_PATTERN_LENGTH}"
        )
    if _has_byte_escape(pattern):
        raise InvalidGeneratorError(
            r"pattern uses \C (match one byte), which breaks on non-ASCII text"
        )
    # google-re2 ships without type stubs; the Protocols above describe what we use.
    import re2  # pyright: ignore[reportMissingTypeStubs]

    options = re2.Options()  # pyright: ignore[reportUnknownMemberType]
    options.log_errors = False  # otherwise every rejected pattern logs to stderr
    re2_error = cast("type[Exception]", re2.error)  # pyright: ignore[reportUnknownMemberType]
    try:
        compiled = re2.compile(pattern, options)  # pyright: ignore[reportUnknownMemberType]
    except re2_error as exc:
        message: object = exc.args[0] if exc.args else exc
        if isinstance(message, bytes):
            message = message.decode(errors="replace")
        raise InvalidGeneratorError(f"pattern does not compile under RE2: {message}") from None
    return cast("_Pattern", compiled)


@dataclass(frozen=True)
class RegexGenerator:
    """Proposes the text matched by ``group`` of an RE2 ``pattern`` as a candidate."""

    id: str
    pattern: str
    group: int = 0
    normalise: tuple[NormaliserStep, ...] = ()
    scope: Scope = field(default_factory=Scope)
    _compiled: _Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        compiled = compile_re2(self.pattern)
        if not 0 <= self.group <= compiled.groups:
            raise InvalidGeneratorError(
                f"group {self.group} doesn't exist; the pattern has {compiled.groups} group(s)"
            )
        object.__setattr__(self, "_compiled", compiled)  # frozen dataclass

    def generate(self, statement: Statement) -> list[Candidate]:
        out: list[Candidate] = []
        for match in self._compiled.finditer(statement.text):
            start, end = match.span(self.group)
            if start < 0 or start == end:  # optional group didn't take part, or empty
                continue
            out.append(
                Candidate.from_statement(
                    statement,
                    Span(start=start, end=end),
                    generator_id=self.id,
                    normalise=list(self.normalise),
                )
            )
        return out
