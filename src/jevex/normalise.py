"""Normalisers: turn a chosen candidate's raw text into a typed value (spec: *Value extraction*).

A candidate carries a declarative chain, e.g. ``["parse_number", {"unit": {"from": "PS"}}]``.
:func:`normalise` runs it step by step, then validates the result against the field's type.
Normalisers are small named functions in a :class:`NormaliserRegistry`; the built-ins below
are the only ones the learner may use in generated specs.

Unit conversion is by dimension: each unit has a factor to its dimension's base unit
(power → kW, speed → km/h ...). Fuel economy is the exception: l/100km is *inverse* to mpg.
mpg means UK (imperial) gallons unless a step says ``{"gallon": "us"}``.

Numbers are read with a decimal point unless a step says ``{"decimal": ","}``; then "." and
no-break spaces group thousands ("1.234,5" → 1234.5). With a decimal point, commas and
Swiss apostrophes group them ("1’250.50"). :mod:`jevex.locales` adds these arguments for
a locale.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import TYPE_CHECKING, Annotated, Any, Union, cast, get_args, get_origin

from pydantic import TypeAdapter, ValidationError

from jevex.generators.units import spellings
from jevex.locales import ALL_MONTH_NAMES, ALL_MULTIPLIERS, THOUSANDS_AFTER_DECIMAL_COMMA
from jevex.pipeline import ValuePick, vision_values
from jevex.results import Alternative, FieldMeta, Source

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from jevex.interfaces import Selection
    from jevex.pipeline import Context, SchemaRun, VisionValue
    from jevex.results import Method
    from jevex.schema import FieldSpec
    from jevex.statements import Candidate, NormaliserStep, Span, Statement


class NormaliseError(ValueError):
    """A value couldn't be normalised or doesn't fit the field."""


class NormaliserFailedError(Exception):
    """A normaliser raised something other than :class:`NormaliseError` (a bug in it, not
    a value that doesn't fit). ``normaliser`` is its name and ``error`` what it raised
    (also the ``__cause__``). The normalise stage skips that candidate and records the
    failure (:mod:`jevex.errors`)."""

    def __init__(self, normaliser: str, error: Exception) -> None:
        super().__init__(f"normaliser {normaliser!r} raised {type(error).__name__}: {error}")
        self.normaliser = normaliser
        self.error = error


@dataclass(frozen=True)
class FunctionNormaliser:
    """A named normaliser backed by a function ``(value, field, **args) -> value``."""

    name: str
    fn: Callable[..., Any]

    def apply(self, value: Any, **args: Any) -> Any:
        """Run the normaliser. ``args`` include the step's arguments and ``field``."""
        return self.fn(value, **args)


# --- numbers ---------------------------------------------------------------------------

_POINT_GROUP = "[,'’]"
"""Thousands separators with a decimal point: a comma, or a Swiss apostrophe."""
_NUMBER = re.compile(rf"[-+]?(?:\d{{1,3}}(?:{_POINT_GROUP}\d{{3}})+|\d+)(?:\.\d+)?|[-+]?\.\d+")
_GROUP = f"[{THOUSANDS_AFTER_DECIMAL_COMMA}]"
_NUMBER_DECIMAL_COMMA = re.compile(
    rf"[-+]?(?:\d{{1,3}}(?:{_GROUP}\d{{3}})+|\d+)(?:,\d+)?|[-+]?,\d+"
)


def _number_pattern(decimal: str) -> re.Pattern[str]:
    if decimal == ".":
        return _NUMBER
    if decimal == ",":
        return _NUMBER_DECIMAL_COMMA
    raise NormaliseError(f"decimal must be '.' or ',', not {decimal!r}")


def _plain_number(text: str, decimal: str) -> str:
    """A matched number with its grouping removed and a decimal point: "1.234,5" → "1234.5"."""
    if decimal == ".":
        return re.sub(_POINT_GROUP, "", text)
    return re.sub(_GROUP, "", text).replace(",", ".")


def parse_number(value: Any, *, decimal: str = ".") -> int | float:
    """The first number in ``value``: "18,495" → 18495, "9.1 s" → 9.1, "150PS" → 150,
    "1’250.50" → 1250.5.

    With ``decimal=","``: "18.495" → 18495, "9,1 s" → 9.1.
    """
    if isinstance(value, bool):
        raise NormaliseError(f"not a number: {value!r}")
    if isinstance(value, int | float):
        return value
    if isinstance(value, Decimal):
        return float(value)
    m = _number_pattern(decimal).search(str(value))
    if not m:
        raise NormaliseError(f"no number in {value!r}")
    text = _plain_number(m.group(), decimal)
    return float(text) if "." in text else int(text)


_SPACE_RUN = re.compile(r"\s+")


def strip(value: Any) -> Any:
    """Trim whitespace and trailing punctuation from strings; other values pass through.

    Runs of whitespace, no-break spaces included (statements keep them for numbers'
    thousands), become one plain space: "A\u00a0Light" → "A Light".
    Leading punctuation is part of the text (".NET", "...And Justice for All").
    """
    if isinstance(value, str):
        return _SPACE_RUN.sub(" ", value).strip().rstrip(" .;,")
    return value


# --- units -----------------------------------------------------------------------------

# (dimension, factor to the dimension's base unit)
_UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "s": ("time", 1.0),
    "mph": ("speed", 1.609344),  # base km/h
    "km/h": ("speed", 1.0),
    "kW": ("power", 1.0),  # base kW
    "bhp": ("power", 0.745699872),
    "hp": ("power", 0.745699872),
    "PS": ("power", 0.73549875),
    "Nm": ("torque", 1.0),
    "lb ft": ("torque", 1.3558179483),
    "kWh": ("energy", 1.0),
    "g/km": ("emissions", 1.0),
    "kg": ("mass", 1.0),
    "mm": ("length", 0.001),  # base m
    "cm": ("length", 0.01),
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "miles": ("length", 1609.344),
    "cc": ("volume", 0.001),  # base litres
    "l": ("volume", 1.0),
}
_ECONOMY = {"mpg", "l/100km"}
_LITRES_PER_100KM_TIMES_MPG = {"uk": 282.480936, "us": 235.214583}
_PRECISION = 6  # decimal places kept after a conversion, dropping float noise
_UNIT_LOOKUP = {s.lower(): canonical for s, canonical, _ in spellings()} | {
    canonical.lower(): canonical for _, canonical, _ in spellings()
}


def canonical_unit(unit: str) -> str:
    """The lexicon's canonical spelling: "seconds" → "s", "litres" → "l", "PS" → "PS"."""
    key = re.sub(r"\s+", " ", unit.strip())
    if key == "PS":  # case matters for the metric horsepower
        return "PS"
    try:
        return _UNIT_LOOKUP[key.lower()]
    except KeyError:
        raise NormaliseError(f"unknown unit {unit!r}") from None


def dimension(unit: str) -> str | None:
    """What ``unit`` measures ("power"; "economy" for mpg and l/100km): units of one
    dimension convert into each other. ``None`` for a unit jevex can't convert (a currency,
    one outside the lexicon)."""
    try:
        found = canonical_unit(unit)
    except NormaliseError:
        return None
    if found in _ECONOMY:
        return "economy"
    known = _UNIT_FACTORS.get(found)
    return known[0] if known else None


def convert(value: float, from_unit: str, to_unit: str, *, gallon: str = "uk") -> float:
    """Convert between units of the same dimension. Raises for incompatible units."""
    src, dst = canonical_unit(from_unit), canonical_unit(to_unit)
    if src == dst:
        return value
    if {src, dst} == _ECONOMY:
        if gallon not in _LITRES_PER_100KM_TIMES_MPG:
            raise NormaliseError(f"gallon must be 'uk' or 'us', not {gallon!r}")
        if value == 0:
            raise NormaliseError("can't convert zero fuel economy")
        return round(_LITRES_PER_100KM_TIMES_MPG[gallon] / value, _PRECISION)
    if src not in _UNIT_FACTORS or dst not in _UNIT_FACTORS:
        raise NormaliseError(f"can't convert {from_unit} to {to_unit}")
    (src_dim, src_factor), (dst_dim, dst_factor) = _UNIT_FACTORS[src], _UNIT_FACTORS[dst]
    if src_dim != dst_dim:
        raise NormaliseError(f"can't convert {from_unit} ({src_dim}) to {to_unit} ({dst_dim})")
    return round(value * src_factor / dst_factor, _PRECISION)


def unit(value: Any, *, field: FieldSpec | None = None, gallon: str = "uk", **args: Any) -> Any:
    """Convert a number (or range) from ``from`` to ``to``, or else to the field's unit.

    Without a target unit, the number is returned as it is.
    """
    source = args.get("from")
    target = args.get("to") or (field.unit if field else None)
    if source is None or target is None:
        return value
    if isinstance(value, list):
        # Sorted: inverse conversions (mpg ↔ l/100km) would otherwise flip a range.
        return sorted(convert(float(v), source, target, gallon=gallon) for v in value)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return convert(float(parse_number(value)), source, target, gallon=gallon)


# --- money -----------------------------------------------------------------------------

_MULTIPLIERS = {
    "k": 10**3,
    "thousand": 10**3,
    "m": 10**6,
    "mn": 10**6,
    "million": 10**6,
    "bn": 10**9,
    "billion": 10**9,
}
_OTHER_MULTIPLIERS = "|".join(
    re.escape(word).replace(r"\ ", r"\s+")
    for word in sorted(ALL_MULTIPLIERS, key=lambda w: (-len(w), w))
)
# Other languages' words first, so "Millionen" isn't read as "million".
_MONEY_MULTIPLIER = (
    rf"(?P<mult>(?i:{_OTHER_MULTIPLIERS})(?![^\W\d_])"
    r"|(?i:bn|billion|million|thousand|mn)|[kKmM](?![a-zA-Z]))?"
)
_MONEY = re.compile(
    rf"(?P<num>\d{{1,3}}(?:{_POINT_GROUP}\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?" + _MONEY_MULTIPLIER
)
_MONEY_DECIMAL_COMMA = re.compile(
    rf"(?P<num>\d{{1,3}}(?:{_GROUP}\d{{3}})+(?:,\d+)?|\d+(?:,\d+)?)\s?" + _MONEY_MULTIPLIER
)
_CURRENCY_UNITS = {
    "GBP",
    "USD",
    "EUR",
    "JPY",
    "CHF",
    "AUD",
    "CAD",
    "NZD",
    "HKD",
    "SGD",
    "MXN",
    "CNY",
}


def parse_money(
    value: Any,
    *,
    field: FieldSpec | None = None,
    currency: str | None = None,
    decimal: str = ".",
) -> Decimal:
    """An amount as ``Decimal``: "£18,495" → 18495, "£1.5m" / "£1.5 million" → 1500000;
    with ``decimal=","``, "18.495,50 €" → 18495.50 and "1,5 Mio. €" → 1500000.

    Multipliers are English ones plus every language's in :data:`~jevex.locales.MULTIPLIERS`,
    whatever the page's language.

    If the field's unit is a currency code, the amount must be in that currency; jevex
    doesn't convert between currencies.
    """
    if isinstance(value, int | float | Decimal) and not isinstance(value, bool):
        amount = Decimal(str(value))
    else:
        _number_pattern(decimal)  # rejects a bad decimal mark
        pattern = _MONEY if decimal == "." else _MONEY_DECIMAL_COMMA
        m = pattern.search(str(value))
        if not m:
            raise NormaliseError(f"no amount in {value!r}")
        try:
            amount = Decimal(_plain_number(m.group("num"), decimal))
        except InvalidOperation as exc:
            raise NormaliseError(f"bad amount in {value!r}") from exc
        if m.group("mult"):
            word = " ".join(m.group("mult").lower().split())
            amount *= _MULTIPLIERS.get(word) or ALL_MULTIPLIERS[word]
    wanted = field.unit.upper() if field and field.unit else None
    if wanted in _CURRENCY_UNITS and currency and currency.upper() != wanted:
        raise NormaliseError(f"amount is in {currency}, the field wants {wanted}")
    # Whole amounts as plain integers (Decimal("1500000"), never "1.5E+6").
    return Decimal(int(amount)) if amount == amount.to_integral() else amount


# --- dates -----------------------------------------------------------------------------

_WORDS = re.compile(r"[^\W\d_]+|\d+")
_NUMERIC_DATE = re.compile(r"(?<!\d)\d{1,4}([./-])\d{1,2}\1\d{1,4}(?!\d)")


def parse_date(
    value: Any,
    *,
    field: FieldSpec | None = None,
    order: str | None = None,
    precision: str = "day",
) -> date | int:
    """Dates in the forms the date/year generators find.

    ``order`` is ``ymd``/``dmy``/``mdy`` for all-numeric dates (default ``dmy``, en-GB); a
    four-digit first number is always the year. Month names may be in any language in
    :data:`~jevex.locales.MONTH_NAMES` ("12. März 2024", "12 de marzo de 2024"); with
    several, the one nearest the year counts ("2 years ago, in March 2024": March, not
    Spanish "ago"), and an all-numeric date beats them all. ``precision="month"`` gives the
    1st of the month; ``"year"`` gives an ``int`` year for number fields, or 1 January for
    date fields.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parts = _WORDS.findall(str(value))
    numbers = [int(p) for p in parts if p.isdigit()]
    month_names = [p.lower() for p in parts if not p.isdigit() and p.lower() in ALL_MONTH_NAMES]
    if _NUMERIC_DATE.search(str(value)):
        month_names = []  # "01/05/2022 (3 years ago)": not August
    elif len(month_names) > 1:
        month_names = [_month_nearest_year(parts)]
    try:
        if precision == "year":
            year = next(n for n in numbers if 1000 <= n <= 9999)
            return year if field is not None and field.kind == "number" else date(year, 1, 1)
        if month_names:
            month = ALL_MONTH_NAMES[month_names[0]]
            year = next(n for n in numbers if n >= 1000)
            day_numbers = [n for n in numbers if n < 1000]
            day = 1 if precision == "month" or not day_numbers else day_numbers[0]
            return date(year, month, day)
        if len(numbers) >= 3:
            a, b, c = numbers[:3]
            resolved = "ymd" if a >= 1000 else order or "dmy"
            if resolved == "ymd" and a < 100:
                a = _four_digit_year(a)
            elif resolved in ("dmy", "mdy") and c < 100:
                c = _four_digit_year(c)
            if resolved == "ymd":
                return date(a, b, c)
            if resolved == "mdy":
                return date(c, a, b)
            return date(c, b, a)
        if len(numbers) == 2 and precision == "month":
            month, year = (
                (numbers[0], numbers[1]) if numbers[1] >= 1000 else (numbers[1], numbers[0])
            )
            return date(year, month, 1)
    except (StopIteration, ValueError) as exc:
        raise NormaliseError(f"not a date: {value!r}") from exc
    raise NormaliseError(f"not a date: {value!r}")


def _month_nearest_year(parts: list[str]) -> str:
    """The month name nearest the first four-digit number (the first name if there's none):
    other languages' short names are English words ("ago", "set", "mag")."""
    names = [i for i, p in enumerate(parts) if not p.isdigit() and p.lower() in ALL_MONTH_NAMES]
    year = next((i for i, p in enumerate(parts) if p.isdigit() and int(p) >= 1000), names[0])
    return parts[min(names, key=lambda i: (abs(i - year), i))].lower()


def _four_digit_year(yy: int) -> int:
    """Two-digit years pivot at 70: 00–69 → 2000s, 70–99 → 1900s.

    Jev picks the span but never sees how it's read, and "03/04/71" doesn't say its
    century, so a fixed pivot decides."""
    return 2000 + yy if yy < 70 else 1900 + yy


_SIGN_BEFORE = re.compile(r"(?:^|[\s(:-]|\bto)$", re.IGNORECASE)
"""What may come right before a range's signed number (searched up to the number)."""


def parse_range(value: Any, *, decimal: str = ".") -> list[int | float]:
    """ "5–7" / "380 to 1,237 litres" / "between 4 and 5" → [lo, hi]; with
    ``decimal=","``, "1,4–2,0 l" → [1.4, 2.0].

    A sign that starts the text or follows a space, "(", ":", a range's dash or "to" is the
    number's own: "-5 to -2" → [-5, -2], "-5--2" → [-5, -2] (a minus sign, U+2212, counts
    as "-"). One right after a number or unit is the range's dash: "5-7" → [5, 7].
    """
    if isinstance(value, list | tuple):
        return [parse_number(v, decimal=decimal) for v in value]  # pyright: ignore[reportUnknownVariableType]
    text = str(value).replace("\u2212", "-").replace("–", " ").replace("—", " ")
    numbers = [
        m.group() if _SIGN_BEFORE.search(text, 0, m.start()) else m.group().lstrip("+-")
        for m in _number_pattern(decimal).finditer(text)
    ]
    if len(numbers) < 2:
        raise NormaliseError(f"not a range: {value!r}")
    return [parse_number(numbers[0], decimal=decimal), parse_number(numbers[1], decimal=decimal)]


# --- registry and chains ---------------------------------------------------------------


class NormaliserRegistry:
    """Named normalisers. Immutable; ``with_normaliser`` returns a new registry."""

    def __init__(self, normalisers: Iterable[FunctionNormaliser] = ()) -> None:
        self._by_name = {n.name: n for n in normalisers}

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    @property
    def names(self) -> list[str]:
        """Registered normaliser names."""
        return list(self._by_name)

    def get(self, name: str) -> FunctionNormaliser:
        """The normaliser called ``name``; ``NormaliseError`` if there's none."""
        try:
            return self._by_name[name]
        except KeyError:
            raise NormaliseError(f"unknown normaliser {name!r}") from None

    def with_normaliser(self, normaliser: FunctionNormaliser) -> NormaliserRegistry:
        """A new registry with ``normaliser`` added (replacing one of the same name)."""
        return NormaliserRegistry([*self._by_name.values(), normaliser])


def _field_aware(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a plain ``fn(value)`` to the ``(value, field=..., **args)`` calling convention."""

    def call(value: Any, *, field: FieldSpec | None = None, **args: Any) -> Any:
        return fn(value, **args)

    return call


BUILTIN_NORMALISERS = NormaliserRegistry(
    [
        FunctionNormaliser("strip", _field_aware(strip)),
        FunctionNormaliser("parse_number", _field_aware(parse_number)),
        FunctionNormaliser("unit", unit),
        FunctionNormaliser("parse_money", parse_money),
        FunctionNormaliser("parse_date", parse_date),
        FunctionNormaliser("parse_range", _field_aware(parse_range)),
    ]
)


def run_chain(
    raw: Any,
    steps: Iterable[NormaliserStep],
    field: FieldSpec | None = None,
    *,
    registry: NormaliserRegistry = BUILTIN_NORMALISERS,
) -> Any:
    """Apply each step in order. Unknown step names or bad arguments raise NormaliseError;
    anything else a normaliser raises, :class:`NormaliserFailedError`."""
    value = raw
    for step in steps:
        normaliser = registry.get(step.name)
        try:
            value = normaliser.apply(value, field=field, **step.args)
        except NormaliseError:
            raise
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise NormaliseError(f"{step.name} failed on {value!r}: {exc}") from exc
        except Exception as exc:
            raise NormaliserFailedError(step.name, exc) from exc
    return value


def normalise(
    raw: Any,
    steps: Iterable[NormaliserStep],
    field: FieldSpec,
    *,
    registry: NormaliserRegistry = BUILTIN_NORMALISERS,
) -> Any:
    """Run the chain, then validate against the field, including its own constraints.

    For ``list[T]`` fields one value is a ``T``; a chain that yields a list (a range) is
    validated as ``list[T]``. Numbers bound for an ``int`` field are rounded to the
    nearest integer when a unit conversion made them fractional (1.4 l → 1400 cc).
    If the field has a unit but the chain never says what unit the text was in (a bare
    number), the number is taken as already being in the field's unit; the select stage
    adds a ``unit`` step when the statement names another unit it could be in.
    """
    chain = list(steps)
    value = run_chain(raw, chain, field, registry=registry)
    target = _target_type(field, value)
    converted = any(_converts(step, field) for step in chain)
    value = _round_for_int(value, target, converted=converted)
    try:
        return TypeAdapter(target).validate_python(value)
    except ValidationError as exc:
        message = exc.errors()[0]["msg"]
        raise NormaliseError(f"{value!r} doesn't fit {field.name}: {message}") from exc


def _target_type(field: FieldSpec, value: Any) -> Any:
    """The type ``value`` must have: the field (with its constraints) or one list item."""
    if not field.many:
        return (
            Annotated[field.annotation, *field.constraints]
            if field.constraints
            else field.annotation
        )
    annotation = field.annotation
    if get_origin(annotation) in (Union, types.UnionType):
        annotation = next(a for a in get_args(annotation) if a is not type(None))
    (item,) = get_args(annotation) or (Any,)
    return list[item] if isinstance(value, list) else item


def _converts(step: NormaliserStep, field: FieldSpec) -> bool:
    """Whether a step is a unit conversion between two different units."""
    if step.name != "unit":
        return False
    source, target = step.args.get("from"), step.args.get("to") or field.unit
    if not source or not target:
        return False
    try:
        return canonical_unit(source) != canonical_unit(target)
    except NormaliseError:
        return False


def _is_int_type(target: Any) -> bool:
    while get_origin(target) is Annotated:
        target = get_args(target)[0]
    if get_origin(target) in (Union, types.UnionType):
        return any(_is_int_type(a) for a in get_args(target) if a is not type(None))
    if get_origin(target) is list:
        return _is_int_type(get_args(target)[0])
    return target is int


def _round_for_int(value: Any, target: Any, *, converted: bool) -> Any:
    if not _is_int_type(target):
        return value
    if isinstance(value, list):
        return [_round_for_int(v, target, converted=converted) for v in value]  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, float) and (converted or value.is_integer()):
        return int(Decimal(str(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return value


# --- stage -----------------------------------------------------------------------------


@dataclass
class NormaliseStage:
    """Turns the selector's picks (``SchemaRun.selections``) into field values.

    Per scope and field, the most confident candidate that normalises and validates wins,
    and the rest become alternatives. List fields keep every accepted value, deduplicated
    in document order. An entity's own statements win over those it shares with every
    entity (``MultiEntity``'s "all of them"); a value from a shared one is marked
    ``shared``. If nothing normalises, the field's meta carries the error. A normaliser
    that raises (:class:`NormaliserFailedError`) skips that candidate and is recorded with
    :meth:`Context.part_failed <jevex.pipeline.Context.part_failed>`. A field
    another route already filled (e.g. structured data) isn't overwritten; in ``merge``
    mode the two are weighed (:meth:`~jevex.pipeline.SchemaRun.offer_field`). Values are
    recorded with ``method="generator"``, or ``"vision"`` when the statement came from a
    vision model.
    """

    registry: NormaliserRegistry = BUILTIN_NORMALISERS
    name: str = "normalise"

    async def run(self, ctx: Context) -> None:
        for run in ctx.active:
            grouped: dict[tuple[str, str], list[tuple[str, Selection]]] = {}
            for (scope, field_name, statement_id), selection in run.selections.items():
                if selection.candidate is not None:
                    grouped.setdefault((scope, field_name), []).append((statement_id, selection))
            for (scope, field_name), picks in grouped.items():
                if not run.needs(scope, field_name):
                    continue
                meta, generator_ids, checks, taken = self._field_meta(
                    ctx, run, field_name, picks, shared=run.shared_statements(scope)
                )
                run.offer_field(scope, field_name, meta)
                run.value_generators[(scope, field_name)] = generator_ids
                if checks:
                    run.vision_values[(scope, field_name)] = checks
                    if run.spec.field(field_name).many:
                        run.value_picks[(scope, field_name)] = taken

    def _field_meta(
        self,
        ctx: Context,
        run: SchemaRun,
        field_name: str,
        picks: list[tuple[str, Selection]],
        *,
        shared: set[str],
    ) -> tuple[FieldMeta, set[str], list[VisionValue], list[ValuePick]]:
        """The field's meta, the ids of the generators whose candidates are in its value,
        the values in it only vision statements gave, and the picks it took, best first."""
        field = run.spec.field(field_name)
        # The entity's own statements outrank those it shares with every entity.
        ranked = sorted(picks, key=lambda p: (p[0] in shared, -p[1].confidence))
        accepted: list[tuple[str, Selection, Candidate, Any]] = []
        errors: list[str] = []
        for statement_id, selection in ranked:
            assert selection.candidate is not None
            # List fields take every candidate the statement states; others take the pick.
            wanted = (
                (selection.accepted or [selection.candidate])
                if field.many
                else [selection.candidate]
            )
            for candidate in wanted:
                try:
                    value = normalise(
                        candidate.raw, candidate.normalise, field, registry=self.registry
                    )
                except NormaliseError as exc:
                    errors.append(str(exc))
                    continue
                except NormaliserFailedError as exc:
                    ctx.part_failed(self.name, "normaliser", exc.normaliser, exc.error)
                    errors.append(str(exc))
                    continue
                accepted.append((statement_id, selection, candidate, value))

        if not accepted:
            statement_id, selection = ranked[0]
            return (
                FieldMeta(
                    confidence=selection.confidence,
                    method=_method(ctx, statement_id),
                    generator_id=selection.candidate.generator_id if selection.candidate else None,
                    source=_source(ctx, statement_id, selection),
                    alternatives=_alternatives(ranked, []),
                    shared=statement_id in shared,
                    error="; ".join(errors),
                ),
                set(),
                [],
                [],
            )

        best_id, best, _, best_value = accepted[0]
        from_shared = best_id in shared
        if field.many:
            from_shared = all(a[0] in shared for a in accepted)
            in_order = sorted(accepted, key=lambda a: _position(ctx, a[0], a[2]))
            items: list[Any] = []
            for _, _, _, v in in_order:
                for item in v if isinstance(v, list) else [v]:  # pyright: ignore[reportUnknownVariableType]
                    if item not in items:
                        items.append(item)
            best_value = items
        used = accepted if field.many else accepted[:1]
        given: list[tuple[Statement, Any, Span | None]] = []
        for statement_id, _, candidate, v in used:
            statement = ctx.parsed.statements.get(statement_id) if ctx.parsed else None
            if statement is not None:
                items = cast("list[Any]", v) if field.many and isinstance(v, list) else [v]
                given.extend((statement, item, candidate.span) for item in items)
        return (
            FieldMeta(
                value=best_value,
                confidence=best.confidence,
                method=_method(ctx, best_id),
                generator_id=best.candidate.generator_id if best.candidate else None,
                source=_source(ctx, best_id, best),
                alternatives=_alternatives(
                    ranked,
                    [] if field.many else [best_id],
                    accepted_raws={c.raw for _, _, c, _ in accepted} if field.many else set(),
                ),
                shared=from_shared,
            ),
            {c.generator_id for _, _, c, _ in used},
            vision_values(given),
            _picks(ctx, used, shared),
        )


def _picks(
    ctx: Context, used: list[tuple[str, Selection, Candidate, Any]], shared: set[str]
) -> list[ValuePick]:
    """One pick per candidate a value took, each with its own span, best first: within a
    statement, the selection's own candidate first, which is the one the meta describes."""
    rank: dict[str, int] = {}
    for i, (statement_id, *_) in enumerate(used):
        rank.setdefault(statement_id, i)
    ordered = sorted(used, key=lambda u: (rank[u[0]], u[2] != u[1].candidate))
    return [
        ValuePick(
            items=tuple(cast("list[Any]", v) if isinstance(v, list) else [v]),
            method=_method(ctx, statement_id),
            source=_source(ctx, statement_id, selection).model_copy(
                update={"span": candidate.span}
            ),
            confidence=selection.confidence,
            generator_id=candidate.generator_id,
            shared=statement_id in shared,
        )
        for statement_id, selection, candidate, v in ordered
    ]


def _method(ctx: Context, statement_id: str) -> Method:
    """``vision`` for a value a vision model stated, else ``generator``."""
    statement = ctx.parsed.statements.get(statement_id) if ctx.parsed else None
    return "vision" if statement is not None and statement.kind == "vision" else "generator"


def _source(ctx: Context, statement_id: str, selection: Selection) -> Source:
    statement = ctx.parsed.statements.get(statement_id) if ctx.parsed else None
    return Source(
        url=ctx.document.url,
        component_id=statement.component_id if statement else None,
        statement_id=statement_id,
        statement=statement.text if statement else None,
        span=selection.candidate.span if selection.candidate else None,
        location=statement.location if statement else None,
    )


def _alternatives(
    ranked: list[tuple[str, Selection]],
    exclude: list[str],
    *,
    accepted_raws: set[str] | None = None,
) -> list[Alternative]:
    """Other picks and the options each question weighed, one per raw span, most likely
    first. Chosen spans (and, for list fields, every accepted span) are never listed."""
    chosen = {s.candidate.raw for sid, s in ranked if sid in exclude and s.candidate is not None}
    chosen |= accepted_raws or set()
    best: dict[str, float] = {}
    for statement_id, selection in ranked:
        if selection.candidate and statement_id not in exclude:
            raw = selection.candidate.raw
            best[raw] = max(best.get(raw, 0.0), selection.confidence)
        picked = selection.candidate.raw if selection.candidate else None
        for raw, p in selection.alternatives.items():
            if raw != picked:
                best[raw] = max(best.get(raw, 0.0), p)
    return [
        Alternative(value=raw, raw=raw, p=p)
        for raw, p in sorted(best.items(), key=lambda kv: -kv[1])
        if raw not in chosen
    ]


def _position(ctx: Context, statement_id: str, candidate: Candidate) -> tuple[int, int]:
    """Document order: the statement's position, then the span's offset within it."""
    order = list(ctx.parsed.statements) if ctx.parsed else []
    index = order.index(statement_id) if statement_id in order else len(order)
    return index, candidate.span.start
