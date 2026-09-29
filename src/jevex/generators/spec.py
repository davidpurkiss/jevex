"""Declarative generator specs: the only thing the learner may produce (spec: *Learning loop*).

A spec is data, never code. It names a field, a scope, an RE2 pattern with the group
holding the value, and a chain of built-in normalisers. Validation guarantees:

- the pattern compiles under RE2 (linear time; no backreferences or lookaround) and is at
  most :data:`~jevex.generators.regex.MAX_PATTERN_LENGTH` characters;
- the group exists in the pattern;
- every normaliser is one of :data:`BUILTIN_NORMALISER_ARGS`, with only its known
  arguments, and unit names come from the unit lexicon;
- no unknown keys anywhere, so a typo fails loudly instead of being ignored.

On disk a spec is YAML (see :meth:`GeneratorSpec.to_yaml`), e.g.::

    id: gen-0f3a9c
    field: VehicleSpec.zero_to_62_s
    scope: {locale: en-GB}
    match:
      regex: '0\\s*[-–]\\s*62(?:\\s*mph)?\\D{0,20}?(\\d+(?:\\.\\d+)?)\\s*(?:s|secs?|seconds)\\b'
      group: 1
    normalise:
      - parse_number
      - unit: {from: s, to: s}
    provenance:
      learned_from: [ex-91c2]
      synthesised_by: generator_llm
      created: 2026-09-29

:func:`generator_spec_json_schema` is the JSON schema, which is also what the learner asks
``generator_llm`` to produce (#38). Anything that fails validation raises
:class:`~jevex.generators.regex.InvalidGeneratorError`.
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from jevex.generators.regex import (
    MAX_PATTERN_LENGTH,
    InvalidGeneratorError,
    RegexGenerator,
    compile_re2,
)
from jevex.interfaces import Scope
from jevex.statements import NormaliserStep

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_NORMALISERS = 8
MAX_SPEC_CHARS = 20_000

_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_FIELD = rf"^{_NAME}\.{_NAME}$"


def _unit_name(value: object) -> None:
    from jevex.normalise import canonical_unit  # jevex.normalise imports this package

    if not isinstance(value, str):
        raise ValueError(f"a unit must be a string, not {value!r}")
    canonical_unit(value)  # NormaliseError (a ValueError) for unknown units


def _one_of(*allowed: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        if value not in allowed:
            raise ValueError(f"{value!r} isn't one of {', '.join(allowed)}")

    return check


def _currency(value: object) -> None:
    if not (isinstance(value, str) and len(value) == 3 and value.isalpha() and value.isupper()):
        raise ValueError(f"a currency must be an ISO 4217 code like 'GBP', not {value!r}")


BUILTIN_NORMALISER_ARGS: dict[str, dict[str, Callable[[object], None]]] = {
    "strip": {},
    "parse_number": {},
    "parse_range": {},
    "unit": {"from": _unit_name, "to": _unit_name, "gallon": _one_of("uk", "us")},
    "parse_money": {"currency": _currency},
    "parse_date": {
        "order": _one_of("ymd", "dmy", "mdy"),
        "precision": _one_of("day", "month", "year"),
    },
}
"""The normalisers a spec may use, and a check for each argument they accept."""


def _check_step(step: NormaliserStep) -> None:
    rules = BUILTIN_NORMALISER_ARGS.get(step.name)
    if rules is None:
        raise ValueError(
            f"normaliser {step.name!r} isn't built in; use one of "
            f"{', '.join(sorted(BUILTIN_NORMALISER_ARGS))}"
        )
    for arg, value in step.args.items():
        check = rules.get(arg)
        if check is None:
            accepted = ", ".join(sorted(rules)) or "no arguments"
            raise ValueError(f"{step.name} doesn't take {arg!r} (it takes {accepted})")
        try:
            check(value)
        except ValueError as exc:
            raise ValueError(f"{step.name}.{arg}: {exc}") from None


class SpecScope(BaseModel):
    """Where a learned generator runs, beyond its own field. Empty means everywhere."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    locale: str | None = Field(default=None, pattern=r"^[A-Za-z]{2,3}([-_][A-Za-z0-9]{2,8})*$")
    sources: list[str] = Field(
        default_factory=list[str], description="Document sources (e.g. site hosts)."
    )


class MatchSpec(BaseModel):
    """An RE2 pattern; ``group`` is the capture group holding the value (0: the match)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regex: str = Field(min_length=1, max_length=MAX_PATTERN_LENGTH)
    group: int = Field(default=0, ge=0)


class Provenance(BaseModel):
    """Where a spec came from: the verified examples and what wrote it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    learned_from: list[str] = Field(
        default_factory=list[str], description="Ids of the verified examples it was learned from."
    )
    synthesised_by: str | None = Field(
        default=None, description="What wrote it: 'generator_llm', 'human', a pack name..."
    )
    created: date | None = None
    note: str | None = Field(default=None, max_length=1000)


class GeneratorSpec(BaseModel):
    """A declarative candidate generator for one field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=_ID)
    field: str = Field(pattern=_FIELD, description="Schema.field, e.g. VehicleSpec.price")
    scope: SpecScope = Field(default_factory=SpecScope)
    match: MatchSpec
    normalise: list[NormaliserStep] = Field(
        default_factory=list[NormaliserStep], max_length=MAX_NORMALISERS
    )
    provenance: Provenance = Field(default_factory=Provenance)

    @field_validator("match")
    @classmethod
    def _compiles(cls, match: MatchSpec) -> MatchSpec:
        try:
            compiled = compile_re2(match.regex)
        except InvalidGeneratorError as exc:
            raise ValueError(str(exc)) from None
        if match.group > compiled.groups:
            raise ValueError(
                f"group {match.group} doesn't exist; the pattern has {compiled.groups} group(s)"
            )
        return match

    @field_validator("normalise")
    @classmethod
    def _built_in_normalisers(cls, steps: list[NormaliserStep]) -> list[NormaliserStep]:
        for step in steps:
            _check_step(step)
        return steps

    @property
    def schema_name(self) -> str:
        return self.field.split(".", 1)[0]

    @property
    def field_name(self) -> str:
        return self.field.split(".", 1)[1]

    @classmethod
    def parse(cls, data: object) -> GeneratorSpec:
        """Validate a spec from parsed YAML/JSON, raising :class:`InvalidGeneratorError`."""
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise InvalidGeneratorError(_describe(exc)) from None

    @classmethod
    def from_yaml(cls, text: str) -> GeneratorSpec:
        if len(text) > MAX_SPEC_CHARS:
            raise InvalidGeneratorError(
                f"spec is {len(text)} characters; the limit is {MAX_SPEC_CHARS}"
            )
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise InvalidGeneratorError(f"spec isn't valid YAML: {exc}") from None
        return cls.parse(data)

    def to_data(self) -> dict[str, Any]:
        """The spec as plain JSON-compatible data, omitting empty optional parts."""
        data = cast("dict[str, Any]", json.loads(self.model_dump_json(exclude_defaults=True)))
        data["match"] = self.match.model_dump()  # always show the group
        return data

    def to_yaml(self) -> str:
        data = self.to_data()
        if self.provenance.created:  # a YAML date, not a quoted string
            data["provenance"]["created"] = self.provenance.created
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)

    def to_generator(self) -> RegexGenerator:
        """The runnable generator, scoped to this spec's field (and locale and sources)."""
        return RegexGenerator(
            id=self.id,
            pattern=self.match.regex,
            group=self.match.group,
            normalise=tuple(self.normalise),
            scope=Scope(
                fields=frozenset({self.field}),
                schemas=frozenset({self.schema_name}),
                locale=self.scope.locale,
                sources=frozenset(self.scope.sources),
            ),
        )


def _describe(exc: ValidationError) -> str:
    problems: list[str] = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "spec"
        problems.append(f"{where}: {err['msg'].removeprefix('Value error, ')}")
    return "invalid generator spec: " + "; ".join(problems)


def _normaliser_item_schema() -> dict[str, Any]:
    """The compact form specs are written in: a name, or ``{name: {arg: value}}``."""
    options: list[dict[str, Any]] = []
    for name, rules in sorted(BUILTIN_NORMALISER_ARGS.items()):
        options.append({"const": name})
        if rules:
            options.append(
                {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": "object",
                            "properties": {arg: {"type": "string"} for arg in rules},
                            "additionalProperties": False,
                        }
                    },
                    "required": [name],
                    "additionalProperties": False,
                }
            )
    return {"anyOf": options}


def generator_spec_json_schema() -> dict[str, Any]:
    """The JSON schema for :class:`GeneratorSpec` (checked in as
    ``docs/generator-spec.schema.json``).

    Normaliser steps are described in their compact form. Unit names, the RE2 check and the
    group check can't be expressed in JSON schema; :meth:`GeneratorSpec.parse` enforces them.
    """
    schema = GeneratorSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    normalise = schema["properties"]["normalise"]
    normalise["items"] = _normaliser_item_schema()
    normalise["description"] = "Built-in normalisers, applied in order."
    schema.get("$defs", {}).pop("NormaliserStep", None)
    return schema
