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
import reprlib
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Annotated, Any, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    WithJsonSchema,
    field_validator,
)
from yaml.composer import ComposerError
from yaml.constructor import ConstructorError

from jevex.generators.regex import (
    MAX_PATTERN_LENGTH,
    InvalidGeneratorError,
    RegexGenerator,
    compile_re2,
)
from jevex.generators.units import spellings
from jevex.interfaces import Scope
from jevex.statements import NormaliserStep

if TYPE_CHECKING:
    from collections.abc import Callable

    from yaml.error import Mark

MAX_NORMALISERS = 8
MAX_SPEC_CHARS = 20_000

_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_FIELD = rf"^{_NAME}\.{_NAME}$"


# Error messages quote untrusted input; keep the quotes short whatever its shape.
_REPR = reprlib.Repr(maxlevel=2, maxstring=40, maxother=40, maxlist=4, maxdict=4)


def _short(value: object) -> str:
    return _REPR.repr(value)


def _unit_name(value: object) -> None:
    from jevex.normalise import canonical_unit  # jevex.normalise imports this package

    if not isinstance(value, str):
        raise ValueError(f"a unit must be a string, not {_short(value)}")
    canonical_unit(value)  # NormaliseError (a ValueError) for unknown units


def _one_of(*allowed: str) -> Callable[[object], None]:
    def check(value: object) -> None:
        if value not in allowed:
            raise ValueError(f"{_short(value)} isn't one of {', '.join(allowed)}")

    return check


def _currency(value: object) -> None:
    if not (isinstance(value, str) and len(value) == 3 and value.isalpha() and value.isupper()):
        raise ValueError(f"a currency must be an ISO 4217 code like 'GBP', not {_short(value)}")


@dataclass(frozen=True)
class ArgRule:
    """How one normaliser argument is checked, and how the JSON schema describes it."""

    check: Callable[[object], None]
    schema: dict[str, Any]


def _enum(*values: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


_UNIT_NAMES = sorted(
    {name for spelling, canonical, _ in spellings() for name in (spelling, canonical)}
)
_UNIT = ArgRule(_unit_name, _enum(*_UNIT_NAMES))

BUILTIN_NORMALISER_ARGS: dict[str, dict[str, ArgRule]] = {
    "strip": {},
    "parse_number": {},
    "parse_range": {},
    "unit": {
        "from": _UNIT,
        "to": _UNIT,
        "gallon": ArgRule(_one_of("uk", "us"), _enum("uk", "us")),
    },
    "parse_money": {
        "currency": ArgRule(_currency, {"type": "string", "pattern": "^[A-Z]{3}$"}),
    },
    "parse_date": {
        "order": ArgRule(_one_of("ymd", "dmy", "mdy"), _enum("ymd", "dmy", "mdy")),
        "precision": ArgRule(_one_of("day", "month", "year"), _enum("day", "month", "year")),
    },
}
"""The normalisers a spec may use, and a rule for each argument they accept."""


def _check_step(step: NormaliserStep) -> None:
    rules = BUILTIN_NORMALISER_ARGS.get(step.name)
    if rules is None:
        raise ValueError(
            f"normaliser {_short(step.name)} isn't built in; use one of "
            f"{', '.join(sorted(BUILTIN_NORMALISER_ARGS))}"
        )
    for arg, value in step.args.items():
        rule = rules.get(arg)
        if rule is None:
            accepted = ", ".join(sorted(rules)) or "no arguments"
            raise ValueError(f"{step.name} doesn't take {_short(arg)} (it takes {accepted})")
        try:
            rule.check(value)
        except ValueError as exc:
            raise ValueError(f"{step.name}.{arg}: {exc}") from None
    if step.name == "unit" and "from" in step.args and "to" in step.args:
        from jevex.normalise import convert

        try:  # e.g. mph → kW would fail on every value at run time
            convert(1.0, step.args["from"], step.args["to"], gallon=step.args.get("gallon", "uk"))
        except ValueError as exc:
            raise ValueError(f"unit: {exc}") from None


def _normaliser_item_schema() -> dict[str, Any]:
    """The compact form specs are written in: a name, or ``{name: {arg: value}}``.

    Every option has a ``type``, and arguments with a fixed set of values are enums, so
    the schema works for LLM structured output (#38) and output is valid by construction.
    """
    options: list[dict[str, Any]] = [_enum(*sorted(BUILTIN_NORMALISER_ARGS))]
    for name, rules in sorted(BUILTIN_NORMALISER_ARGS.items()):
        if rules:
            options.append(
                {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": "object",
                            "properties": {arg: dict(rule.schema) for arg, rule in rules.items()},
                            "additionalProperties": False,
                        }
                    },
                    "required": [name],
                    "additionalProperties": False,
                }
            )
    return {"anyOf": options}


_NORMALISE_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": _normaliser_item_schema(),
    "maxItems": MAX_NORMALISERS,
    "description": "Built-in normalisers, applied in order.",
}


def _check_step_keys(steps: object) -> None:
    """Long-form steps (``{name: ..., args: ...}``) may hold nothing else.

    ``NormaliserStep`` ignores extra keys, so ``{name: unit, arg: {...}}`` (a typo for
    ``args``) would otherwise become a bare ``unit`` without its arguments.
    """
    if isinstance(steps, list):
        for i, step in enumerate(cast("list[object]", steps)):
            if isinstance(step, dict) and "name" in step:
                extra = set(cast("dict[str, object]", step)) - {"name", "args"}
                if extra:
                    raise ValueError(
                        f"step {i} has unknown keys {', '.join(sorted(map(str, extra)))} "
                        "(a long-form step takes only name and args)"
                    )


class _SpecLoader(yaml.SafeLoader):
    """``SafeLoader`` without aliases or duplicate keys.

    Aliases let a few hundred characters expand into gigabytes (a "billion laughs" spec)
    and specs never need them. A duplicate key would silently win, hiding a typo.
    """

    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.check_event(yaml.AliasEvent):
            event = cast("object", self.peek_event())  # pyright: ignore[reportUnknownMemberType]
            mark = cast("Mark | None", getattr(event, "start_mark", None))
            raise ComposerError(None, None, "aliases aren't allowed in specs", mark)
        return super().compose_node(parent, index)  # pyright: ignore[reportUnknownMemberType]

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[object] = set()
        for key_node, _ in node.value:
            key: object = self.construct_object(key_node, deep=True)  # pyright: ignore[reportUnknownMemberType]
            try:
                duplicate = key in seen
            except TypeError:  # unhashable: the base class reports it
                continue
            if duplicate:
                raise ConstructorError(
                    None, None, f"duplicate key {_short(key)}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


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
    group: int = Field(default=0, ge=0, strict=True)


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
    normalise: Annotated[list[NormaliserStep], WithJsonSchema(_NORMALISE_SCHEMA)] = Field(
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

    @field_validator("normalise", mode="before")
    @classmethod
    def _strict_steps(cls, steps: object) -> object:
        _check_step_keys(steps)
        return steps

    @field_validator("normalise")
    @classmethod
    def _built_in_normalisers(cls, steps: list[NormaliserStep]) -> list[NormaliserStep]:
        for step in steps:
            _check_step(step)
        return steps

    @property
    def schema_name(self) -> str:
        """The schema part of ``field``: ``VehicleSpec`` for ``VehicleSpec.price``."""
        return self.field.split(".", 1)[0]

    @property
    def field_name(self) -> str:
        """The field part of ``field``: ``price`` for ``VehicleSpec.price``."""
        return self.field.split(".", 1)[1]

    @classmethod
    def parse(cls, data: object) -> GeneratorSpec:
        """Validate a spec from parsed YAML/JSON, raising :class:`InvalidGeneratorError`."""
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise InvalidGeneratorError(_describe(exc)) from None
        except RecursionError:
            raise InvalidGeneratorError("spec is nested too deeply") from None

    @classmethod
    def from_yaml(cls, text: str) -> GeneratorSpec:
        """Load and validate one spec from YAML.

        The text is at most :data:`MAX_SPEC_CHARS` characters. It is loaded safely (no
        Python tags), without aliases or duplicate keys. Every failure, including bad
        YAML and over-deep nesting, raises :class:`InvalidGeneratorError`.
        """
        if len(text) > MAX_SPEC_CHARS:
            raise InvalidGeneratorError(
                f"spec is {len(text)} characters; the limit is {MAX_SPEC_CHARS}"
            )
        try:
            data = _load(text)
        except yaml.YAMLError as exc:
            raise InvalidGeneratorError(f"spec isn't valid YAML: {exc}") from None
        except RecursionError:
            raise InvalidGeneratorError("spec is nested too deeply") from None
        return cls.parse(data)

    def to_data(self) -> dict[str, Any]:
        """The spec as plain JSON-compatible data, omitting empty optional parts."""
        data = cast("dict[str, Any]", json.loads(self.model_dump_json(exclude_defaults=True)))
        data["match"] = self.match.model_dump()  # always show the group
        return data

    def to_yaml(self) -> str:
        """The spec as YAML in the spec's own format; :meth:`from_yaml` reads it back.

        Text is written as Unicode where that round-trips. PyYAML folds a few characters
        (U+0085 NEL) when writing them raw, so then everything is escaped instead.
        """
        data = self.to_data()
        if self.provenance.created:  # a YAML date, not a quoted string
            data["provenance"]["created"] = self.provenance.created
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
        if _load(text) != data:
            text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False, width=100)
        return text

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


def _load(text: str) -> object:
    return yaml.load(text, Loader=_SpecLoader)


def _describe(exc: ValidationError) -> str:
    problems: list[str] = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"]) or "spec"
        problems.append(f"{where}: {err['msg'].removeprefix('Value error, ')}")
    return "invalid generator spec: " + "; ".join(problems)


def generator_spec_json_schema() -> dict[str, Any]:
    """The JSON schema for :class:`GeneratorSpec` (checked in as
    ``docs/generator-spec.schema.json``).

    It is ``GeneratorSpec.model_json_schema()``, so structured output built from the model
    uses the same compact normaliser form. The RE2, group and unit-conversion checks
    can't be expressed in JSON schema; :meth:`GeneratorSpec.parse` enforces them.
    """
    schema = GeneratorSpec.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema
