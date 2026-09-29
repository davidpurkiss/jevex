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
    WholeStatement,
    Year,
)
from jevex.generators.regex import InvalidGeneratorError, RegexGenerator, compile_re2
from jevex.generators.registry import GeneratorRegistry, scope_matches
from jevex.generators.spec import (
    BUILTIN_NORMALISER_ARGS,
    GeneratorSpec,
    MatchSpec,
    Provenance,
    SpecScope,
    generator_spec_json_schema,
)


def default_registry() -> GeneratorRegistry:
    """The built-in generators, in the order they win span ties."""
    return GeneratorRegistry(BUILTIN_GENERATORS)


__all__ = [
    "BUILTIN_GENERATORS",
    "BUILTIN_NORMALISER_ARGS",
    "DateGenerator",
    "GeneratorRegistry",
    "GeneratorSpec",
    "InvalidGeneratorError",
    "KeyValue",
    "MatchSpec",
    "Money",
    "NounPhrase",
    "NumberWithUnit",
    "Provenance",
    "Range",
    "RegexGenerator",
    "SpecScope",
    "WholeStatement",
    "Year",
    "compile_re2",
    "default_registry",
    "generator_spec_json_schema",
    "scope_matches",
]
