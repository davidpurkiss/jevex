from datetime import date
from decimal import Decimal
from typing import Literal

import pytest
from pydantic import BaseModel

from jevex import DomLocation, Field, NormaliserStep, SchemaSpec, Statement
from jevex.generators import (
    DateGenerator,
    GeneratorRegistry,
    InvalidGeneratorError,
    KeyValue,
    Money,
    NounPhrase,
    NumberWithUnit,
    Range,
    RegexGenerator,
    WholeStatement,
    Year,
    default_registry,
)
from jevex.generators.regex import MAX_PATTERN_LENGTH
from jevex.interfaces import CandidateGenerator, FieldAwareGenerator, Scope
from jevex.normalise import NormaliseError, normalise


def st(text: str) -> Statement:
    return Statement(
        id="s1", text=text, kind="sentence", component_id="c1", location=DomLocation(dom_path="/")
    )


def raws(generator: CandidateGenerator, text: str) -> list[str]:
    return [c.raw for c in generator.generate(st(text))]


def chain(generator: CandidateGenerator, text: str, raw: str) -> list[object]:
    for c in generator.generate(st(text)):
        if c.raw == raw:
            return [step.model_dump() for step in c.normalise]
    raise AssertionError(f"{raw!r} not generated from {text!r}")


class VehicleSpec(BaseModel):
    model: str = Field(description="Model name")
    fuel_type: Literal["petrol", "diesel"] = Field(description="Fuel type")
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")
    first_registered: str = Field(description="Registration")


SPEC = SchemaSpec.from_model(VehicleSpec)


# --- number_with_unit ------------------------------------------------------------------


def test_number_with_unit_finds_units_and_bare_numbers() -> None:
    assert raws(NumberWithUnit(), "0-62 mph in 9.1 seconds") == ["0", "62 mph", "9.1 seconds"]
    assert raws(NumberWithUnit(), "150PS (110 kW) and 250 Nm") == ["150PS", "110 kW", "250 Nm"]


@pytest.mark.parametrize(
    ("text", "raw", "unit"),
    [
        ("9.1s", "9.1s", "s"),
        ("52.3 mpg", "52.3 mpg", "mpg"),
        ("5.4 l/100km", "5.4 l/100km", "l/100km"),
        ("123 g/km", "123 g/km", "g/km"),
        ("1,300kg", "1,300kg", "kg"),
        ("1498cc", "1498cc", "cc"),
        ("1.5 L", "1.5 L", "l"),
        ("380 litres", "380 litres", "l"),
        ("320 lb ft", "320 lb ft", "lb ft"),
        ("155 km/h", "155 km/h", "km/h"),
        ("110 KW", "110 KW", "kW"),
    ],
)
def test_number_with_unit_canonicalises_units(text: str, raw: str, unit: str) -> None:
    assert chain(NumberWithUnit(), text, raw) == ["parse_number", {"unit": {"from": unit}}]


def test_unit_must_end_at_a_word_boundary() -> None:
    # "5 seats" must not read as 5 seconds, "150 psi" isn't PS, and "2 Lanes" isn't litres.
    assert chain(NumberWithUnit(), "5 seats", "5") == ["parse_number"]
    assert raws(NumberWithUnit(), "150 psi") == ["150"]
    assert raws(NumberWithUnit(), "2 Lanes") == ["2"]


def test_numbers_inside_words_are_ignored() -> None:
    assert raws(NumberWithUnit(), "CO2 and A4 and v2.0") == []


# --- money -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "raw", "currency"),
    [
        ("Price £18,495 on the road", "£18,495", "GBP"),
        ("$32,000", "$32,000", "USD"),
        ("€ 25,990.50", "€ 25,990.50", "EUR"),
        ("25k GBP on finance", "25k GBP", "GBP"),
        ("EUR 30,000 abroad", "EUR 30,000", "EUR"),
        ("from £199k", "£199k", "GBP"),
    ],
)
def test_money(text: str, raw: str, currency: str) -> None:
    assert chain(Money(), text, raw) == [{"parse_money": {"currency": currency}}]


def test_money_ignores_plain_numbers() -> None:
    assert raws(Money(), "18,495 miles") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("£1.5m budget", ["£1.5m"]),
        ("EUR 2.5bn deal", ["EUR 2.5bn"]),
        ("£18,4950", []),  # malformed: never truncate to a wrong "£18,495"
        ("£5.99p", []),
        ("£1.5x", []),
        ("a £1.5 million refit", ["£1.5 million"]),
        ("£2 billion", ["£2 billion"]),
        ("EUR 3 million", ["EUR 3 million"]),
        ("£2 m", ["£2 m"]),
        ("£299pm or £1,200pcm", ["£299", "£1,200"]),
        ("£299 pm", ["£299"]),
    ],
)
def test_money_never_truncates_amounts(text: str, expected: list[str]) -> None:
    assert raws(Money(), text) == expected


# --- dates and years -------------------------------------------------------------------


def test_dates() -> None:
    text = "Registered 12 March 2024, facelift Mar 2025, MOT 2024-09-01, sold 03/04/2026"
    assert raws(DateGenerator(), text) == [
        "12 March 2024",
        "March 2024",
        "Mar 2025",
        "2024-09-01",
        "03/04/2026",
    ]
    assert chain(DateGenerator(), text, "03/04/2026") == [{"parse_date": {"order": "dmy"}}]
    assert chain(DateGenerator(), text, "Mar 2025") == [{"parse_date": {"precision": "month"}}]
    assert chain(DateGenerator(), text, "2024-09-01") == [{"parse_date": {"order": "ymd"}}]


def test_us_style_month_first_date() -> None:
    assert chain(DateGenerator(), "on March 12, 2024", "March 12, 2024") == [
        {"parse_date": {"order": "mdy"}}
    ]


def test_years_are_not_found_inside_identifiers() -> None:
    assert raws(Year(), "VIN WVW2024ZZZ, part A2024, 2024x") == []


def test_years() -> None:
    assert raws(Year(), "the 2024 model year, built 1999, ref 12024, 2024.5") == ["2024", "1999"]
    assert raws(Year(), "MY2024 and MY24") == ["2024"]
    assert chain(Year(), "2024 model year", "2024") == [{"parse_date": {"precision": "year"}}]


# --- ranges ----------------------------------------------------------------------------


def test_ranges() -> None:
    text = "Seats 5–7 adults, boot 380 to 1,237 litres, between 4 and 5 stars"
    assert raws(Range(), text) == ["5–7", "380 to 1,237 litres", "between 4 and 5"]
    assert chain(Range(), text, "380 to 1,237 litres") == [
        "parse_range",
        {"unit": {"from": "l"}},
    ]


def test_dates_and_references_are_not_ranges() -> None:
    assert raws(Range(), "MOT 2024-09-01") == []
    assert raws(Range(), "ref 12-34/56") == []


# --- key/value -------------------------------------------------------------------------


def test_key_value_takes_the_value_side() -> None:
    assert raws(KeyValue(), "Colour: Moonstone Grey metallic.") == ["Moonstone Grey metallic"]
    assert chain(KeyValue(), "Seats: 5", "5") == ["strip"]


def test_key_value_reads_rendered_table_cells() -> None:
    assert raws(KeyValue(), "Performance › 0-62 mph (s) · 1.5 TSI SE: 9.1") == ["9.1"]


def test_key_value_strips_trailing_newlines() -> None:
    assert raws(KeyValue(), "Colour: Grey\n") == ["Grey"]


def test_key_value_needs_a_separator_and_a_value() -> None:
    assert raws(KeyValue(), "No separator here") == []
    assert raws(KeyValue(), "Colour: ") == []
    assert raws(KeyValue(), "Time 10:30") == []


class Listing(BaseModel):
    title: str = Field(description="Title")
    power_kw: float = Field(description="Power", unit="kW")
    price: Decimal = Field(description="Price", unit="GBP")
    seats: int = Field(description="Seats")
    boot_litres: list[float] = Field(default_factory=list, description="Boot", unit="l")
    registered: date = Field(description="Registered")


LISTING = SchemaSpec.from_model(Listing)


def kv_chain(text: str, field: str) -> list[object] | None:
    """The key_value chain for ``field``, or None when there's no candidate."""
    cands = KeyValue().generate_for(st(text), LISTING.field(field))
    assert len(cands) <= 1
    return [step.model_dump() for step in cands[0].normalise] if cands else None


@pytest.mark.parametrize(
    ("text", "field", "steps"),
    [
        ("Power: 150PS (110kW)", "power_kw", ["parse_number", {"unit": {"from": "PS"}}]),
        ("0-62 mph: 9.1 seconds", "power_kw", ["parse_number", {"unit": {"from": "s"}}]),
        ("Seats: 5", "seats", ["parse_number"]),
        ("Price: £18,495 on the road", "price", [{"parse_money": {"currency": "GBP"}}]),
        ("Price: 25k GBP", "price", [{"parse_money": {"currency": "GBP"}}]),
        ("Price: from EUR 30,000", "price", [{"parse_money": {"currency": "EUR"}}]),
        ("Boot: 380 to 1,237 litres", "boot_litres", ["parse_range", {"unit": {"from": "l"}}]),
        ("Registered: 12 March 2024", "registered", [{"parse_date": {"order": "dmy"}}]),
        ("Registered: 2024-03-12", "registered", [{"parse_date": {"order": "ymd"}}]),
        ("Registered: 12/03/2024.", "registered", [{"parse_date": {"order": "dmy"}}]),
        ("Registered: March 2024", "registered", [{"parse_date": {"precision": "month"}}]),
        ("Registered: 2019 (UK)", "registered", [{"parse_date": {"precision": "year"}}]),
        ("Title: 12 March 2024", "title", ["strip"]),
        ("Title: 150PS", "title", ["strip"]),
    ],
)
def test_key_value_chain_fits_the_field_kind(text: str, field: str, steps: list[object]) -> None:
    assert kv_chain(text, field) == steps


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("Colour: Moonstone Grey", "power_kw"),
        ("Registered: soon", "registered"),
        # A bare number with more numbers after it is ambiguous.
        ("Engine: 1.5 TSI 150PS", "power_kw"),
        ("Seats: 5 (7 optional)", "seats"),
        # A digit before the quantity would be what the parser reads.
        ("Warranty: 3 years, then £500", "price"),
        ("Registered: Q1 2024", "registered"),
    ],
)
def test_key_value_skips_values_that_wont_parse_for_the_field(text: str, field: str) -> None:
    assert kv_chain(text, field) is None


def test_key_value_without_a_field_keeps_strip_and_needs_a_separator() -> None:
    assert chain(KeyValue(), "Registered: 12 March 2024", "12 March 2024") == ["strip"]
    assert (
        KeyValue().generate_for(st("Registered 12 March 2024"), LISTING.field("registered")) == []
    )
    assert isinstance(KeyValue(), FieldAwareGenerator)
    assert not isinstance(NumberWithUnit(), FieldAwareGenerator)


@pytest.mark.parametrize(
    ("text", "field", "raw", "value"),
    [
        ("Power: 150PS (110kW)", "power_kw", "150PS (110kW)", 110.324812),
        ("Price: £18,495 on the road", "price", "£18,495 on the road", Decimal(18495)),
        (
            "Registered: 12 March 2024 (first owner)",
            "registered",
            "12 March 2024 (first owner)",
            date(2024, 3, 12),
        ),
        (
            "Boot: 380 to 1,237 litres (seats down)",
            "boot_litres",
            "380 to 1,237 litres (seats down)",
            [380.0, 1237.0],
        ),
    ],
)
def test_key_value_spans_normalise_end_to_end(
    text: str, field: str, raw: str, value: object
) -> None:
    spec = LISTING.field(field)
    [cand] = [
        c
        for c in default_registry().generate(st(text), spec, schema="Listing")
        if c.generator_id == "key_value"
    ]
    assert cand.raw == raw
    assert normalise(cand.raw, cand.normalise, spec) == value


def test_key_value_money_in_another_currency_fails_validation() -> None:
    spec = LISTING.field("price")
    [cand] = KeyValue().generate_for(st("Price: from EUR 30,000"), spec)
    with pytest.raises(NormaliseError, match="the field wants GBP"):
        normalise(cand.raw, cand.normalise, spec)


# --- noun phrases ----------------------------------------------------------------------


def test_noun_phrases_split_on_stopwords_and_punctuation() -> None:
    assert raws(NounPhrase(), "Available in Moonstone Grey metallic and Pure White") == [
        "Available",
        "Moonstone Grey metallic",
        "Pure White",
    ]


def test_noun_phrases_keep_numbers_with_thousands_separators_whole() -> None:
    assert raws(NounPhrase(), "Price £18,495, or 25k GBP on finance") == [
        "Price £18,495",
        "25k GBP",
        "finance",
    ]


def test_noun_phrases_skip_pure_numbers_and_chunk_long_runs() -> None:
    assert raws(NounPhrase(), "42, 7.5") == []
    long = " ".join(f"Word{i}" for i in range(12))
    assert raws(NounPhrase(), long) == [
        " ".join(f"Word{i}" for i in range(8)),
        " ".join(f"Word{i}" for i in range(8, 12)),
    ]


# --- declarative regex generators ------------------------------------------------------


def test_regex_generator_uses_the_group_span() -> None:
    gen = RegexGenerator(
        id="gen-0f3a9c",
        pattern=r"0\s*[-–]\s*62(?:\s*mph)?\D{0,20}?(\d+(?:\.\d+)?)\s*(?:s|secs?|seconds)\b",
        group=1,
        normalise=(NormaliserStep(name="parse_number"),),
    )
    [cand] = gen.generate(st("0-62 mph in 9.1 seconds"))
    assert (cand.raw, cand.span.start, cand.span.end) == ("9.1", 12, 15)
    assert cand.generator_id == "gen-0f3a9c"
    assert [s.name for s in cand.normalise] == ["parse_number"]


def test_regex_generator_skips_missing_optional_groups() -> None:
    gen = RegexGenerator(id="g", pattern=r"x(\d)?", group=1)
    assert [c.raw for c in gen.generate(st("x1 x x2"))] == ["1", "2"]


@pytest.mark.parametrize(
    ("pattern", "group", "message"),
    [
        (r"(a)\1", 0, "RE2"),  # backreferences aren't linear-time
        (r"(?=a)a", 0, "RE2"),  # lookaround either
        (r"(a", 0, "RE2"),
        (r"(a)", 2, "group 2"),
        ("a" * (MAX_PATTERN_LENGTH + 1), 0, "limit"),
    ],
)
def test_regex_generator_rejects_unsafe_or_invalid_specs(
    pattern: str, group: int, message: str
) -> None:
    with pytest.raises(InvalidGeneratorError, match=message):
        RegexGenerator(id="bad", pattern=pattern, group=group)


# --- registry --------------------------------------------------------------------------


def test_default_registry_picks_generators_by_field_kind() -> None:
    reg = default_registry()
    number = [g.id for g in reg.for_field(SPEC.field("zero_to_62_s"), schema="VehicleSpec")]
    string = [g.id for g in reg.for_field(SPEC.field("model"), schema="VehicleSpec")]
    enum = [g.id for g in reg.for_field(SPEC.field("fuel_type"), schema="VehicleSpec")]
    assert number == ["number_with_unit", "money", "year", "range", "key_value"]
    assert string == ["key_value", "noun_phrase", "whole_statement"]
    assert enum == []


def test_whole_statement_proposes_short_statements_whole() -> None:
    gen = WholeStatement()
    assert raws(gen, "A Light in the Attic") == ["A Light in the Attic"]
    assert raws(gen, "  Tipping the Velvet. ") == ["Tipping the Velvet"]
    assert raws(gen, "Sapiens: A Brief History of Humankind") == [
        "Sapiens: A Brief History of Humankind"
    ]
    assert raws(gen, " ".join(["word"] * 17)) == []
    assert raws(gen, "...") == []
    pair = Statement(
        id="s1",
        text="Colour: Red",
        kind="key_value",
        component_id="c1",
        location=DomLocation(dom_path="/"),
    )
    assert gen.generate(pair) == []


def test_registry_generate_dedupes_spans_and_sorts() -> None:
    cands = default_registry().generate(
        st("Performance › 0-62 mph (s) · 1.5 TSI SE: 9.1"),
        SPEC.field("zero_to_62_s"),
        schema="VehicleSpec",
    )
    assert [c.raw for c in cands] == ["0", "0-62 mph", "62 mph", "1.5", "9.1"]
    # number_with_unit is earlier in the registry, so it owns the "9.1" span
    assert cands[-1].generator_id == "number_with_unit"


@pytest.mark.parametrize(
    ("scope", "matches"),
    [
        (Scope(), True),
        (Scope(fields=frozenset({"zero_to_62_s"})), True),
        (Scope(fields=frozenset({"VehicleSpec.zero_to_62_s"})), True),
        (Scope(fields=frozenset({"Listing.zero_to_62_s"})), False),
        (Scope(schemas=frozenset({"Listing"})), False),
        (Scope(kinds=frozenset({"date"})), False),
        (Scope(locale="en"), True),
        (Scope(locale="en-GB"), True),
        (Scope(locale="de-DE"), False),
        (Scope(locale="EN_gb"), True),
        (Scope(sources=frozenset({"structured"})), False),
    ],
)
def test_scope_matching(scope: Scope, matches: bool) -> None:
    gen = RegexGenerator(id="g", pattern=r"\d+", scope=scope)
    reg = GeneratorRegistry([gen])
    found = reg.for_field(SPEC.field("zero_to_62_s"), schema="VehicleSpec", locale="en-GB")
    assert (found == [gen]) is matches


def test_locale_scoped_generators_skip_documents_of_unknown_locale() -> None:
    gen = RegexGenerator(id="g", pattern=r"\d+,\d+", scope=Scope(locale="de"))
    reg = GeneratorRegistry([gen])
    assert reg.for_field(SPEC.field("zero_to_62_s"), schema="VehicleSpec", locale=None) == []
    assert reg.for_field(SPEC.field("zero_to_62_s"), schema="VehicleSpec", locale="de-AT") == [gen]


def test_re2_errors_are_readable(capfd: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(InvalidGeneratorError) as info:
        RegexGenerator(id="bad", pattern=r"(a")
    assert "missing )" in str(info.value)
    assert "b'" not in str(info.value)
    assert capfd.readouterr().err == ""


def test_registry_is_immutable_and_replaces_by_id() -> None:
    base = default_registry()
    learned = RegexGenerator(id="gen-1", pattern=r"\d+")
    grown = base.with_generator(learned)
    assert "gen-1" in grown
    assert "gen-1" not in base
    replaced = grown.with_generator(RegexGenerator(id="gen-1", pattern=r"\d{2}"))
    assert len(replaced) == len(grown)
    assert isinstance(replaced.get("gen-1"), RegexGenerator)
    assert replaced.ids.index("gen-1") == grown.ids.index("gen-1")
    assert "gen-1" not in grown.without("gen-1")
    with pytest.raises(KeyError):
        base.without("nope")
    with pytest.raises(ValueError, match="duplicate"):
        GeneratorRegistry([learned, learned])


def test_builtins_satisfy_the_protocol() -> None:
    for gen in default_registry():
        assert isinstance(gen, CandidateGenerator)
