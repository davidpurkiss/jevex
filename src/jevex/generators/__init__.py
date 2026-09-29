"""Candidate generators: verbatim spans Jev chooses between (spec: *Value extraction*)."""

from __future__ import annotations

from jevex.generators.builtin import (
    BUILTIN_GENERATORS,
    DateGenerator,
    KeyValue,
    Money,
    NounPhrase,
    NumberWithUnit,
    Range,
    Year,
)
from jevex.generators.regex import InvalidGeneratorError, RegexGenerator, compile_re2
from jevex.generators.registry import GeneratorRegistry, scope_matches


def default_registry() -> GeneratorRegistry:
    """The built-in generators, in the order they win span ties."""
    return GeneratorRegistry(BUILTIN_GENERATORS)


__all__ = [
    "BUILTIN_GENERATORS",
    "DateGenerator",
    "GeneratorRegistry",
    "InvalidGeneratorError",
    "KeyValue",
    "Money",
    "NounPhrase",
    "NumberWithUnit",
    "Range",
    "RegexGenerator",
    "Year",
    "compile_re2",
    "default_registry",
    "scope_matches",
]
