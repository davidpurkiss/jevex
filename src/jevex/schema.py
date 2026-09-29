"""Turn a Pydantic model into the questions every stage asks.

Consumers declare what they want as a plain Pydantic model, using :func:`Field` for the
extras jevex needs (description, unit, group, question overrides) and an optional
``__jevex__ = SchemaConfig(...)`` class attribute. :meth:`SchemaSpec.from_model`
reads the model once and generates each stage's questions from it.
"""

from __future__ import annotations

import enum
import inspect
import types
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Union, cast, get_args, get_origin

import pydantic
from pydantic import BaseModel, ConfigDict
from pydantic_core import PydanticUndefined

from jevex.jev import Choice, JSONContent, Noul

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

EXTRA_KEY = "jevex"
NONE_OPTION = "none"
NOT_STATED_OPTION = "not stated"

FieldKind = Literal["enum", "bool", "number", "date", "str", "model"]


class Questions(BaseModel):
    """Per-field overrides for generated question text.

    - ``component_gate``: Noul asked of each component, e.g. "Does this section contain...?"
    - ``categorise``: this field's option description in the categorise Choice.
    - ``select``: instructions for choosing among candidate values.
    - ``verify``: Noul checking an LLM answer; ``{value}`` is replaced by the value.
    """

    model_config = ConfigDict(frozen=True)

    component_gate: str | None = None
    categorise: str | None = None
    select: str | None = None
    verify: str | None = None


class SchemaConfig(BaseModel):
    """Schema-level settings, set as ``__jevex__ = SchemaConfig(...)`` on the model."""

    model_config = ConfigDict(frozen=True)

    document_question: str | None = None
    categorise_question: str | None = None
    gate_unit: Literal["document", "page"] = "document"


def Field(
    default: Any = PydanticUndefined,
    *,
    description: str,
    unit: str | None = None,
    group: str | None = None,
    questions: Questions | None = None,
    **kwargs: Any,
) -> Any:
    """``pydantic.Field`` plus jevex extras, stored under ``json_schema_extra["jevex"]``.

    The model stays plain Pydantic: the extras only add to its JSON schema.
    """
    extra: dict[str, Any] = {}
    if unit is not None:
        extra["unit"] = unit
    if group is not None:
        extra["group"] = group
    if questions is not None:
        extra["questions"] = questions.model_dump(exclude_none=True)
    json_schema_extra = cast("dict[str, Any]", kwargs.pop("json_schema_extra", None) or {})
    if extra:
        json_schema_extra = {**json_schema_extra, EXTRA_KEY: extra}
    return pydantic.Field(
        default,
        description=description,
        json_schema_extra=json_schema_extra or None,
        **kwargs,
    )


class UnsupportedFieldError(TypeError):
    """A field's type has no extraction strategy."""


@dataclass(frozen=True)
class FieldSpec:
    """Everything jevex needs to know about one field."""

    name: str
    description: str
    kind: FieldKind
    annotation: Any
    required: bool
    nullable: bool
    many: bool
    options: tuple[str, ...] = ()
    unit: str | None = None
    group: str | None = None
    questions: Questions = field(default_factory=Questions)
    model: type[BaseModel] | None = None

    @property
    def needs_candidates(self) -> bool:
        """Enums and bools are answered directly; everything else needs candidate spans."""
        return self.kind not in ("enum", "bool", "model")

    @property
    def label(self) -> str:
        """Description with its unit, as shown to Jev in option lists."""
        return f"{self.description} ({self.unit})" if self.unit else self.description

    def select_instructions(self) -> str:
        return self.questions.select or f"Which of these is the {self.label}?"

    def select_question(self, candidates: list[str]) -> Choice:
        """Choice over candidate spans plus "none". Duplicate spans are merged."""
        if NONE_OPTION in candidates:
            raise ValueError(f"a candidate span may not be the reserved option {NONE_OPTION!r}")
        options: dict[str, JSONContent | None] = dict.fromkeys(candidates)
        options[NONE_OPTION] = f"None of these is the {self.description}"
        return Choice(instructions=self.select_instructions(), options=dict(options))

    def enum_question(self) -> Choice:
        """For ``Literal``/``Enum`` fields: pick an option or "not stated"."""
        if self.kind != "enum":
            raise ValueError(f"{self.name} is not an enum field")
        options: dict[str, JSONContent | None] = dict.fromkeys(self.options)
        options[NOT_STATED_OPTION] = f"The statement does not state the {self.description}"
        return Choice(
            instructions=self.questions.select or f"What is the {self.label}?", options=options
        )

    def bool_question(self) -> Noul:
        if self.kind != "bool":
            raise ValueError(f"{self.name} is not a bool field")
        return Noul(instructions=self.questions.select or f"Does the statement say {self.label}?")

    def verify_question(self, value: object) -> Noul:
        """Checks an LLM or vision answer against the statement."""
        template = (
            self.questions.verify or "The statement states that the {description} is {value}."
        )
        return Noul(instructions=template.format(description=self.label, value=value))


@dataclass(frozen=True)
class SchemaSpec:
    """A model analysed once, with question builders for every stage."""

    model: type[BaseModel]
    name: str
    description: str
    config: SchemaConfig
    fields: tuple[FieldSpec, ...]

    @classmethod
    def from_model(cls, model: type[BaseModel]) -> SchemaSpec:
        config = getattr(model, "__jevex__", None) or SchemaConfig()
        if not isinstance(config, SchemaConfig):
            raise TypeError(f"{model.__name__}.__jevex__ must be a SchemaConfig")
        docstring = model.__dict__.get("__doc__")
        description = inspect.cleandoc(docstring) if docstring else _humanise(model.__name__)
        fields = tuple(_field_spec(name, info) for name, info in model.model_fields.items())
        return cls(model, model.__name__, description, config, fields)

    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def groups(self) -> dict[str, tuple[FieldSpec, ...]]:
        """Fields grouped by ``group``; ungrouped fields form a group of their own."""
        out: dict[str, list[FieldSpec]] = {}
        for f in self.fields:
            out.setdefault(f.group or f.name, []).append(f)
        return {k: tuple(v) for k, v in out.items()}

    def document_gate_question(self) -> Noul:
        return Noul(
            instructions=self.config.document_question
            or f"Does this document describe {_lower_first(_first_sentence(self.description))}?"
        )

    def component_gate_questions(self) -> dict[str, Noul]:
        """One Noul per field group, keyed by group name."""
        questions: dict[str, Noul] = {}
        for group, members in self.groups.items():
            override = next(
                (f.questions.component_gate for f in members if f.questions.component_gate), None
            )
            listed = _join_or([f.label for f in members])
            questions[group] = Noul(
                instructions=override or f"Does this section contain the {listed}?"
            )
        return questions

    def categorise_question(self) -> Choice:
        """One Choice per statement: which field does it state, or none of them."""
        options: dict[str, JSONContent | None] = {
            f.name: f.questions.categorise or f.label for f in self.fields if f.kind != "model"
        }
        options[NONE_OPTION] = "None of these details"
        return Choice(
            instructions=self.config.categorise_question
            or "Which detail does this statement state?",
            options=options,
        )


def _field_spec(name: str, info: FieldInfo) -> FieldSpec:
    raw_extra: object = info.json_schema_extra  # a dict, a callable or None
    extra = cast("dict[str, Any]", raw_extra) if isinstance(raw_extra, dict) else {}
    jevex_extra = cast("dict[str, Any]", extra.get(EXTRA_KEY) or {})
    annotation, nullable = _strip_optional(info.annotation)
    many = get_origin(annotation) is list
    if many:
        (annotation,) = get_args(annotation) or (Any,)
        annotation, _ = _strip_optional(annotation)
    kind, options = _classify(name, annotation)
    return FieldSpec(
        name=name,
        description=info.description or _humanise(name),
        kind=kind,
        annotation=info.annotation,
        required=info.is_required(),
        nullable=nullable,
        many=many,
        options=options,
        unit=jevex_extra.get("unit"),
        group=jevex_extra.get("group"),
        questions=Questions.model_validate(jevex_extra.get("questions") or {}),
        model=annotation if kind == "model" else None,
    )


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    if get_origin(annotation) in (Union, types.UnionType):
        args = get_args(annotation)
        rest = tuple(a for a in args if a is not type(None))
        if len(rest) == 1:
            return rest[0], len(rest) < len(args)
        return annotation, len(rest) < len(args)
    return annotation, False


def _classify(name: str, annotation: Any) -> tuple[FieldKind, tuple[str, ...]]:
    if get_origin(annotation) is Literal:
        return "enum", tuple(str(v) for v in get_args(annotation))
    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            return "enum", tuple(str(m.value) for m in annotation)
        if issubclass(annotation, bool):
            return "bool", ()
        if issubclass(annotation, int | float | Decimal):
            return "number", ()
        if issubclass(annotation, date | datetime):
            return "date", ()
        if issubclass(annotation, str):
            return "str", ()
        if issubclass(annotation, BaseModel):
            return "model", ()
    raise UnsupportedFieldError(f"field {name!r} has unsupported type {annotation!r}")


def _humanise(name: str) -> str:
    words: list[str] = []
    for part in name.replace("_", " ").split():
        spaced = "".join(f" {c}" if c.isupper() and i else c for i, c in enumerate(part))
        words.extend(spaced.split())
    return " ".join(w.lower() for w in words)


def _first_sentence(text: str) -> str:
    return text.split("\n\n", 1)[0].strip().rstrip(".")


def _lower_first(text: str) -> str:
    """Lowercase the first letter, unless the first word is an acronym ("EV", "SUV")."""
    if len(text) > 1 and text[1].isupper():
        return text
    return text[:1].lower() + text[1:]


def _join_or(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"
