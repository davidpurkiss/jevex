"""Turn a Pydantic model into the questions every stage asks.

Consumers declare what they want as a plain Pydantic model, using :func:`Field` for the
extras jevex needs (description, unit, group, question overrides) and an optional
``__jevex__ = SchemaConfig(...)`` class attribute. :meth:`SchemaSpec.from_model`
reads the model once and generates each stage's questions from it.
"""

from __future__ import annotations

import enum
import inspect
import re
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
    from collections.abc import Sequence

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
    - ``member``: for ``list[...]`` fields, the Noul asking whether one value (``{value}``)
      is stated as one of them.
    """

    model_config = ConfigDict(frozen=True)

    component_gate: str | None = None
    categorise: str | None = None
    select: str | None = None
    verify: str | None = None
    member: str | None = None


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

    ``description`` is read mid-sentence in jevex's questions ("Which of these is the
    {description}?"), so write it as a noun phrase: "Price", "Engine displacement". For a
    ``bool``, write the claim that makes it True, as a clause: "The book is in stock", or
    "has an automatic gearbox" (read as "it has..."). A bare noun ("Sunroof") works too
    it is asked as "Does the statement say it has sunroof?".
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


class ReservedFieldNameError(ValueError):
    """A schema field uses a name jevex reserves for its own question options."""


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
    constraints: tuple[Any, ...] = ()
    """The field's own validation metadata (``ge``, ``max_length``, ``Annotated`` validators)."""

    @property
    def needs_candidates(self) -> bool:
        """Enums and bools are answered directly; everything else needs candidate spans."""
        return self.kind not in ("enum", "bool", "model")

    @property
    def label(self) -> str:
        """Description with its unit, as shown to Jev in option lists."""
        return f"{self.description} ({self.unit})" if self.unit else self.description

    @property
    def phrase(self) -> str:
        """The label as it reads mid-sentence: "Price (GBP)" → "price (GBP)".

        Acronyms keep their case ("VIN", "EV range").
        """
        return _lower_first(self.label)

    def select_instructions(self) -> str:
        return self.questions.select or f"Which of these is the {self.phrase}?"

    def select_question(self, candidates: list[str]) -> Choice:
        """Choice over candidate spans plus "none". Duplicate spans are merged."""
        if NONE_OPTION in candidates:
            raise ValueError(f"a candidate span may not be the reserved option {NONE_OPTION!r}")
        options: dict[str, JSONContent | None] = dict.fromkeys(candidates)
        options[NONE_OPTION] = f"None of these is the {_lower_first(self.description)}"
        return Choice(instructions=self.select_instructions(), options=dict(options))

    def enum_question(self) -> Choice:
        """For ``Literal``/``Enum`` fields: pick an option or "not stated"."""
        if self.kind != "enum":
            raise ValueError(f"{self.name} is not an enum field")
        options: dict[str, JSONContent | None] = dict.fromkeys(self.options)
        options[NOT_STATED_OPTION] = (
            f"The statement does not state the {_lower_first(self.description)}"
        )
        return Choice(
            instructions=self.questions.select or f"What is the {self.phrase}?", options=options
        )

    def bool_question(self) -> Noul:
        if self.kind != "bool":
            raise ValueError(f"{self.name} is not a bool field")
        if self.questions.select:
            return Noul(instructions=self.questions.select)
        shape, claim = _claim(self)
        # A noun is asked about as present, not merely mentioned: "No sunroof" mentions one.
        if shape == "noun":
            claim = f"it has {claim}"
        return Noul(instructions=f"Does the statement say {claim}?")

    def member_question(self, value: str) -> Noul:
        """For ``list[...]`` fields: does the statement give ``value`` as one of them?"""
        # A custom template gets the description as written; the default reads it mid-sentence.
        template = self.questions.member
        if template is None:
            template, description = (
                'Does the statement give "{value}" as one of the {description}?',
                self.phrase,
            )
        else:
            description = self.label
        return Noul(instructions=template.format(description=description, value=value))

    def verify_question(self, value: object) -> Noul:
        """Checks an LLM or vision answer against the statement."""
        template = self.questions.verify
        if template is None and self.kind == "bool":
            # "the book is in stock is True" doesn't read; say the claim itself.
            shape, claim = _claim(self)
            if shape == "clause":
                text = claim if value else f"it is not the case that {claim}"
                return Noul(instructions=f"The statement says {text}.")
            text = f"it has {claim}" if value else f"there is no {claim}"
            return Noul(instructions=f"The statement says {text}.")
        if template is None:
            template, description = (
                "The statement states that the {description} is {value}.",
                self.phrase,
            )
        else:
            description = self.label
        return Noul(instructions=template.format(description=description, value=value))


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
        if NONE_OPTION in model.model_fields:
            raise ReservedFieldNameError(
                f"{model.__name__}.{NONE_OPTION}: {NONE_OPTION!r} is reserved (it is the "
                '"none of these" option in jevex\'s questions); rename the field and set '
                f"alias={NONE_OPTION!r} if the data needs that name"
            )
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
            questions[group] = Noul(instructions=override or _gate_instructions(members))
        return questions

    def categorise_question(self, fields: Sequence[str] | None = None) -> Choice:
        """One Choice per statement: which field does it state, or none of them.

        ``fields`` limits the options to those field names (in schema order).
        """
        allowed = None if fields is None else set(fields)
        options: dict[str, JSONContent | None] = {
            f.name: f.questions.categorise or f.label
            for f in self.fields
            if f.kind != "model" and (allowed is None or f.name in allowed)
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
        constraints=tuple(info.metadata),
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


_FIRST_TOKEN = re.compile(r"[^\W_]+")


def _lower_first(text: str) -> str:
    """Lowercase the first letter, unless the first word is an acronym or a letter name.

    Kept: any capital after the first character of the first word ("EV range", "iPhone",
    "VIN"), and a lone capital that isn't the article "A" ("X", "A-pillar colour").
    Proper nouns ("Google rating") can't be told apart and are lowercased; override the
    question with ``Questions(...)`` if that matters.
    """
    m = _FIRST_TOKEN.match(text)
    if m is None:
        return text
    token = m.group()
    rest = text[m.end() :]
    if any(ch.isupper() for ch in token[1:]):
        return text
    if len(token) == 1 and token.isupper() and not (token == "A" and rest[:1] == " "):
        return text
    return text[:1].lower() + text[1:]


# Words that make a bool description a clause ("the book is in stock") rather than a noun
# ("sunroof").
_VERBS = frozenset(
    [
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "can",
        "could",
        "will",
        "would",
        "does",
        "do",
        "did",
        "comes",
        "come",
        "includes",
        "include",
        "offers",
        "offer",
        "supports",
        "support",
        "needs",
        "need",
        "uses",
        "use",
        "runs",
        "run",
    ]
)


# Only these, first, mean the subject is missing ("has an automatic gearbox"): base forms
# such as "use" or "can" start nouns too ("Use of pool", "Can opener included").
_LEADING_VERBS = frozenset(
    ["is", "has", "was", "does", "comes", "includes", "offers", "supports", "needs", "uses", "runs"]
)


def _claim(field: FieldSpec) -> tuple[Literal["clause", "noun"], str]:
    """How a bool's description reads: as a claim ("the book is in stock", "it has an
    automatic gearbox") or as a thing ("sunroof")."""
    phrase = field.phrase
    words = [w.lower() for w in phrase.split()]
    if words and words[0] in _LEADING_VERBS:
        return "clause", f"it {phrase}"
    if any(w in _VERBS for w in words[1:]):
        return "clause", phrase
    return "noun", phrase


def _gate_instructions(members: Sequence[FieldSpec]) -> str:
    """ "Does this section contain the price (GBP)?"; bools ask for their claim: "Does this
    section say whether the book is in stock?", or "...mention a sunroof?" for nouns."""
    things = [f.phrase for f in members if f.kind != "bool"]
    claims = [_claim(f) for f in members if f.kind == "bool"]
    clauses = [f"whether {text}" for shape, text in claims if shape == "clause"]
    nouns = [text for shape, text in claims if shape == "noun"]
    parts: list[str] = []
    if things:
        parts.append(f"contain the {_join_or(things)}")
    if clauses:
        parts.append(f"say {_join_or(clauses)}")
    if nouns:
        parts.append(f"mention {_join_or(nouns)}")
    return f"Does this section {' or '.join(parts)}?"


def _join_or(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"
