"""Declarative regex generators: the only kind the learner may create.

Patterns compile under RE2 (linear time, no backreferences), so a learned pattern can't
cause catastrophic backtracking on hostile input. The pattern length is capped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from jevex.interfaces import Scope
from jevex.statements import Candidate, NormaliserStep, Span, Statement

MAX_PATTERN_LENGTH = 500


class InvalidGeneratorError(ValueError):
    """A declarative generator spec failed validation."""


class _Match(Protocol):
    def span(self, group: int = 0) -> tuple[int, int]: ...


class _Pattern(Protocol):
    groups: int

    def finditer(self, text: str) -> Any: ...


def compile_re2(pattern: str) -> _Pattern:
    """Compile under RE2, raising :class:`InvalidGeneratorError` on anything RE2 rejects."""
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise InvalidGeneratorError(
            f"pattern is {len(pattern)} characters; the limit is {MAX_PATTERN_LENGTH}"
        )
    import re2  # pyright: ignore[reportMissingTypeStubs]

    try:
        return cast("_Pattern", re2.compile(pattern))  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:  # re2 raises its own error type for unsupported syntax
        raise InvalidGeneratorError(f"pattern does not compile under RE2: {exc}") from exc


@dataclass(frozen=True)
class RegexGenerator:
    """Proposes the text matched by ``group`` of an RE2 ``pattern`` as a candidate."""

    id: str
    pattern: str
    group: int = 0
    normalise: tuple[NormaliserStep, ...] = ()
    scope: Scope = field(default_factory=Scope)

    def __post_init__(self) -> None:
        compiled = compile_re2(self.pattern)
        if not 0 <= self.group <= compiled.groups:
            raise InvalidGeneratorError(
                f"group {self.group} doesn't exist; the pattern has {compiled.groups} group(s)"
            )
        object.__setattr__(self, "_compiled", compiled)

    def generate(self, statement: Statement) -> list[Candidate]:
        compiled = cast("_Pattern", self.__dict__["_compiled"])
        out: list[Candidate] = []
        for match in cast("list[_Match]", list(compiled.finditer(statement.text))):
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
