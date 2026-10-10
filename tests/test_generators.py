from datetime import date
from decimal import Decimal
from typing import Literal

import pytest
from pydantic import BaseModel

from jevex import DomLocation, Field, NormaliserStep, SchemaSpec, Statement
from jevex.generators import (
    DateGenerator,
    GeneratorRegistry,
    GeneratorSpec,
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
from jevex.generators.builtin import MAX_PHRASE_WORDS
from jevex.generators.regex import MAX_PATTERN_LENGTH
from jevex.generators.units import mentioned
from jevex.interfaces import (
    CandidateGenerator,
    FieldAwareGenerator,
    LocaleAwareGenerator,
    Scope,
)
from jevex.normalise import NormaliseError, normalise, parse_money


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


@pytest.mark.parametrize(
    ("text", "units"),
    [
        ("Power (kW) · SE: 110", ["kW"]),
        ("150PS / 110 kW, 0-62 mph in 9.1 secs", ["PS", "kW", "mph", "s"]),
        ("Torque: 250 Nm (184 lb-ft), 250 nm", ["Nm", "lb ft"]),
        ("5 seats, 150 psi, 2 Lanes, CO2, Ps", []),
    ],
)
def test_mentioned_finds_units_with_or_without_a_number(text: str, units: list[str]) -> None:
    assert mentioned(text) == units


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
        "Moonstone",
        "Moonstone Grey",
        "Moonstone Grey metallic",
        "Grey",
        "Grey metallic",
        "metallic",
        "Pure",
        "Pure White",
        "White",
    ]


def test_noun_phrases_propose_every_part_of_a_name_run() -> None:
    """Nothing marks where the make ends and the model starts, so Jev gets each part."""
    assert raws(NounPhrase(), "Delmaro Kestrova SE") == [
        "Delmaro",
        "Delmaro Kestrova",
        "Delmaro Kestrova SE",
        "Kestrova",
        "Kestrova SE",
        "SE",
    ]


def test_noun_phrases_keep_numbers_with_thousands_separators_whole() -> None:
    assert raws(NounPhrase(), "Price £18,495, or 25k GBP on finance") == [
        "Price",
        "Price £18,495",
        "18,495",
        "25k",
        "25k GBP",
        "GBP",
        "finance",
    ]


def test_noun_phrases_propose_numbers_alone() -> None:
    """A model name can be a number; select's Choice tells it from a quantity."""
    assert raws(NounPhrase(), "The Peugeot 308 GT") == [
        "Peugeot",
        "Peugeot 308",
        "Peugeot 308 GT",
        "308",
        "308 GT",
        "GT",
    ]
    assert raws(NounPhrase(), "42, 7.5, 18,495") == ["42", "7.5", "18,495"]
    assert raws(NounPhrase(), "Kestrova 2.0 SE") == [
        "Kestrova",
        "Kestrova 2.0",
        "Kestrova 2.0 SE",
        "2.0",
        "2.0 SE",
        "SE",
    ]


def test_noun_phrases_chunk_long_runs() -> None:
    long = " ".join(f"Word{i}" for i in range(12))
    assert raws(NounPhrase(), long) == [
        " ".join(f"Word{i}" for i in range(8)),
        " ".join(f"Word{i}" for i in range(8, 12)),
    ]


def test_a_run_of_max_phrase_words_still_gives_every_sub_run() -> None:
    n = MAX_PHRASE_WORDS
    words = [f"Word{i}" for i in range(n)]
    found = raws(NounPhrase(), " ".join(words))
    assert found == [" ".join(words[i:j]) for i in range(n) for j in range(i + 1, n + 1)]
    assert len(found) == n * (n + 1) // 2


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
    assert raws(gen, " ".join(["word"] * 16)) == [" ".join(["word"] * 16)]
    assert raws(gen, " ".join(["word"] * 17)) == []
    assert raws(WholeStatement(max_words=3), "one two three four") == []
    assert raws(gen, "...") == []
    assert raws(gen, "Who Moved My Cheese?") == ["Who Moved My Cheese?"]
    assert raws(gen, "Stop! ") == ["Stop!"]
    assert raws(gen, "Title\r") == ["Title"]
    pair = Statement(
        id="s1",
        text="Colour: Red",
        kind="key_value",
        component_id="c1",
        location=DomLocation(dom_path="/"),
    )
    assert gen.generate(pair) == []


def test_key_value_strips_only_its_own_trailing_characters() -> None:
    assert raws(KeyValue(), "Colour: Red\r") == ["Red"]
    assert raws(KeyValue(), "Time: 10:30:") == ["10:30:"]


def test_whole_statement_skips_list_fields() -> None:
    class Car(BaseModel):
        trims: list[str] = Field(default_factory=list, description="Trim names")
        model: str = Field(description="Model name")

    spec = SchemaSpec.from_model(Car)
    statement = st("Available in SE, SE L and R-Line trims")
    assert WholeStatement().generate_for(statement, spec.field("trims")) == []
    [cand] = WholeStatement().generate_for(statement, spec.field("model"))
    assert cand.raw == "Available in SE, SE L and R-Line trims"


def test_whole_statement_is_linear_on_long_statements() -> None:
    import time

    start = time.perf_counter()
    assert raws(WholeStatement(), ". " * 40_000) == []
    assert time.perf_counter() - start < 0.5


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


@pytest.mark.parametrize(
    ("sources", "source", "matches"),
    [
        ({"example.com"}, "example.com", True),
        ({"WWW.Example.com"}, "example.com", True),
        ({"example.com"}, "Example.COM", True),
        ({"other.com", "example.com"}, "example.com", True),
        ({"example.com"}, "shop.example.com", False),
        ({"example.com"}, "other.com", False),
        ({"example.com"}, None, False),
        (set[str](), None, True),
    ],
)
def test_source_scoped_generators_run_only_on_their_sources(
    sources: set[str], source: str | None, matches: bool
) -> None:
    gen = RegexGenerator(id="g", pattern=r"\d+", scope=Scope(sources=frozenset(sources)))
    reg = GeneratorRegistry([gen])
    field = SPEC.field("zero_to_62_s")
    assert (reg.for_field(field, schema="VehicleSpec", source=source) == [gen]) is matches
    statement = st("9 s")
    found = reg.generate(statement, field, schema="VehicleSpec", source=source)
    assert [c.raw for c in found] == (["9"] if matches else [])


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


def test_extended_appends_new_ids_and_keeps_its_own() -> None:
    base = GeneratorRegistry([RegexGenerator(id="a", pattern=r"\d+")])
    clash = RegexGenerator(id="a", pattern=r"x")
    extra = RegexGenerator(id="b", pattern=r"y")
    grown = base.extended([clash, extra])
    assert grown.ids == ["a", "b"]
    assert grown.get("a") is base.get("a")
    assert base.ids == ["a"]
    assert base.extended([clash]) is base


def test_builtins_satisfy_the_protocol() -> None:
    for gen in default_registry():
        assert isinstance(gen, CandidateGenerator)


# --- locales (#56) ---------------------------------------------------------------------


class Angebot(BaseModel):
    gewicht_kg: float = Field(description="Kerb weight", unit="kg")
    hubraum_l: float = Field(description="Engine size", unit="l")
    preis: Decimal = Field(description="Price", unit="EUR")
    verbrauch: list[float] = Field(default_factory=list, description="Economy", unit="l/100km")
    zugelassen: date = Field(description="First registered")
    sitze: int = Field(description="Seats")


ANGEBOT = SchemaSpec.from_model(Angebot)


def values_in(locale: str | None, text: str, field: str) -> dict[str, object]:
    """Every candidate the default registry proposes, normalised: raw → value (or the error)."""
    spec = ANGEBOT.field(field)
    out: dict[str, object] = {}
    for cand in default_registry().generate(st(text), spec, schema="Angebot", locale=locale):
        try:
            out[cand.raw] = normalise(cand.raw, cand.normalise, spec)
        except NormaliseError:
            out[cand.raw] = "error"
    return out


def test_german_numbers_with_units_read_a_decimal_comma() -> None:
    found = values_in("de-DE", "Leergewicht 1.234,5 kg, Hubraum 1,4 l", "gewicht_kg")
    assert found["1.234,5 kg"] == 1234.5
    assert values_in("de-DE", "Hubraum 1,4 Liter", "hubraum_l") == {"1,4 Liter": 1.4}
    # The same text read as en-GB has no 1234.5 in it.
    assert 1234.5 not in values_in(None, "Leergewicht 1.234,5 kg", "gewicht_kg").values()


def test_number_chains_carry_the_decimal_mark() -> None:
    [cand] = NumberWithUnit().generate_in(st("9,1 s"), SPEC.field("zero_to_62_s"), "de-DE")
    assert [s.model_dump() for s in cand.normalise] == [
        {"parse_number": {"decimal": ","}},
        {"unit": {"from": "s"}},
    ]


def test_german_amounts_put_the_symbol_after_the_number() -> None:
    found = values_in("de-DE", "Preis: 18.495 € inkl. MwSt.", "preis")
    assert found["18.495 €"] == Decimal(18495)
    assert values_in("de-DE", "nur 1.299,99 €", "preis")["1.299,99 €"] == Decimal("1299.99")
    # A symbol after the number isn't an amount in en-GB text.
    assert "18,495 €" not in values_in(None, "only 18,495 €", "preis")


def test_german_dates_and_month_names() -> None:
    found = values_in("de-DE", "Erstzulassung am 12. März 2024", "zugelassen")
    assert found["12. März 2024"] == date(2024, 3, 12)
    assert found["März 2024"] == date(2024, 3, 1)  # Jev chooses between the spans
    found = values_in("de-DE", "Erstzulassung: 01.12.2023", "zugelassen")
    assert found["01.12.2023"] == date(2023, 12, 1)
    assert values_in("de-DE", "Lieferbar ab Mai 2025", "zugelassen")["Mai 2025"] == date(2025, 5, 1)
    # English month names still match on a German page; German ones don't on an English one.
    assert values_in("de", "seit March 2024", "zugelassen")["March 2024"] == date(2024, 3, 1)
    assert "März 2024" not in values_in(None, "seit März 2024", "zugelassen")


def test_german_ranges() -> None:
    assert values_in("de-DE", "Verbrauch 4,8–5,6 l/100km", "verbrauch")["4,8–5,6 l/100km"] == [
        4.8,
        5.6,
    ]


def test_german_key_value_chains() -> None:
    def kv(text: str, field: str) -> list[object]:
        [cand] = KeyValue().generate_in(st(text), ANGEBOT.field(field), "de-DE")
        return [s.model_dump() for s in cand.normalise]

    assert kv("Leergewicht: 1.234,5 kg", "gewicht_kg") == [
        {"parse_number": {"decimal": ","}},
        {"unit": {"from": "kg"}},
    ]
    assert kv("Preis: 18.495 €", "preis") == [{"parse_money": {"currency": "EUR", "decimal": ","}}]
    assert kv("Verbrauch: 4,8–5,6 l/100km", "verbrauch") == [
        {"parse_range": {"decimal": ","}},
        {"unit": {"from": "l/100km"}},
    ]
    assert kv("Erstzulassung: 12. März 2024", "zugelassen") == [{"parse_date": {"order": "dmy"}}]
    assert values_in("de-DE", "Sitze: 5", "sitze") == {"5": 5}


# --- more decimal-comma forms (#219) --------------------------------------------------


@pytest.mark.parametrize(
    ("locale", "text", "raw", "value"),
    [
        ("de-DE", "Umsatz 1,5 Mio. € im Jahr", "1,5 Mio. €", Decimal(1500000)),
        ("de-DE", "Budget: 2 Mrd. EUR", "2 Mrd. EUR", Decimal(2000000000)),
        ("de-AT", "rund 1,5 Millionen € netto", "1,5 Millionen €", Decimal(1500000)),
        ("de-DE", "ab € 1,5 Mio.", "€ 1,5 Mio.", Decimal(1500000)),
        ("fr-FR", "un budget de 2 Mds €", "2 Mds €", Decimal(2000000000)),
        ("es-ES", "unos 2 mil millones €", "2 mil millones €", Decimal(2000000000)),
        ("it-IT", "circa 3 mln €", "3 mln €", Decimal(3000000)),
        ("nl-NL", "ruim 1,2 miljoen €", "1,2 miljoen €", Decimal(1200000)),
    ],
)
def test_amounts_with_the_page_languages_multipliers(
    locale: str, text: str, raw: str, value: Decimal
) -> None:
    assert values_in(locale, text, "preis")[raw] == value


def test_other_languages_multipliers_are_not_read_on_english_pages() -> None:
    found = values_in(None, "about 1,5 Mio. € a year", "preis")
    assert "1,5 Mio. €" not in found
    assert "€2bn" in values_in("de-DE", "nur €2bn", "preis")  # English ones still are


@pytest.mark.parametrize(
    ("locale", "text"),
    [("de-DE", "€ 1 Billion"), ("de-DE", "ab € 1,2 Bio."), ("fr-FR", "1 billion €")],
)
def test_long_scale_billions_give_no_amount(locale: str, text: str) -> None:
    # A German or French "billion" is 10^12: no amount, rather than a wrong or truncated one.
    assert Money().generate_in(st(text), ANGEBOT.field("preis"), locale) == []


@pytest.mark.parametrize(
    ("locale", "text", "raw"),
    [
        ("de-DE", "Preis: 18.495,- € inkl. MwSt.", "18.495,- €"),
        ("de-DE", "nur € 18.495,–", "€ 18.495,–"),
        ("de-DE", "Preis 18.495,-- EUR", "18.495,-- EUR"),
    ],
)
def test_round_amounts_with_a_dash(locale: str, text: str, raw: str) -> None:
    assert values_in(locale, text, "preis")[raw] == Decimal(18495)


def test_swiss_round_amounts_and_apostrophe_grouping() -> None:
    [cand] = Money().generate_in(st("CHF 1’250.– inkl."), ANGEBOT.field("preis"), "de-CH")
    assert cand.raw == "CHF 1’250.–"
    assert [s.model_dump() for s in cand.normalise] == [{"parse_money": {"currency": "CHF"}}]
    assert parse_money(cand.raw, currency="CHF") == Decimal(1250)
    assert values_in("de-CH", "Leergewicht 1’250.50 kg", "gewicht_kg") == {"1’250.50 kg": 1250.5}
    assert values_in("de-CH", "Leergewicht 1'250 kg", "gewicht_kg") == {"1'250 kg": 1250}
    assert values_in("it-CH", "peso 1’250.50", "gewicht_kg") == {"1’250.50": 1250.5}
    # Elsewhere an apostrophe doesn't group: en-GB reads as before.
    assert "1’250.50 kg" not in values_in("en-GB", "Leergewicht 1’250.50 kg", "gewicht_kg")


@pytest.mark.parametrize(
    ("locale", "text", "raw", "value"),
    [
        ("de-DE", "Verbrauch 4,8 bis 5,6 l/100km", "4,8 bis 5,6 l/100km", [4.8, 5.6]),
        ("de-DE", "zwischen 4 und 5 Sitze", "zwischen 4 und 5", [4, 5]),
        ("fr-FR", "de 4,5 à 5,2 l/100km", "4,5 à 5,2 l/100km", [4.5, 5.2]),
        ("fr-FR", "entre 4 et 5", "entre 4 et 5", [4, 5]),
        ("es-ES", "de 4 a 5 l/100km", "4 a 5 l/100km", [4, 5]),
        ("it-IT", "tra 4 e 5", "tra 4 e 5", [4, 5]),
        ("nl-NL", "van 4 tot 5", "4 tot 5", [4, 5]),
        ("nl-BE", "tussen 4 en 5", "tussen 4 en 5", [4, 5]),
        # English words still match on other pages.
        ("de-DE", "4,8 to 5,6 l/100km", "4,8 to 5,6 l/100km", [4.8, 5.6]),
    ],
)
def test_ranges_in_the_page_languages_words(
    locale: str, text: str, raw: str, value: list[float]
) -> None:
    [cand] = Range().generate_in(st(text), ANGEBOT.field("verbrauch"), locale)
    assert cand.raw == raw
    assert values_in(locale, text, "verbrauch")[raw] == value


def test_range_words_need_a_space_and_the_page_language() -> None:
    field = ANGEBOT.field("verbrauch")
    assert Range().generate_in(st("4 bis 5"), field, "en-GB") == []
    assert Range().generate_in(st("4 und 5"), field, "de-DE") == []
    assert Range().generate_in(st("4bis5"), field, "de-DE") == []


@pytest.mark.parametrize(
    ("locale", "text", "raw", "value"),
    [
        ("fr-FR", "Livraison le 12 mars 2024", "12 mars 2024", date(2024, 3, 12)),
        ("fr-BE", "le 1er août 2024", "1er août 2024", date(2024, 8, 1)),
        ("fr-FR", "depuis févr. 2025", "févr. 2025", date(2025, 2, 1)),
        ("es-ES", "el 12 de marzo de 2024", "12 de marzo de 2024", date(2024, 3, 12)),
        ("es-AR", "desde septiembre de 2024", "septiembre de 2024", date(2024, 9, 1)),
        ("it-IT", "dal 3 giugno 2025", "3 giugno 2025", date(2025, 6, 3)),
        ("nl-NL", "per 12 maart 2024", "12 maart 2024", date(2024, 3, 12)),
        ("nl-BE", "sinds mei 2024", "mei 2024", date(2024, 5, 1)),
    ],
)
def test_dates_with_the_page_languages_month_names(
    locale: str, text: str, raw: str, value: date
) -> None:
    assert values_in(locale, text, "zugelassen")[raw] == value


def test_other_languages_month_names_are_not_matched_on_english_pages() -> None:
    assert set(values_in(None, "le 12 mars 2024", "zugelassen")) == {"2024"}
    assert set(values_in("de-DE", "le 12 mars 2024", "zugelassen")) == {"2024"}


@pytest.mark.parametrize(
    "text", ["First registered: 01/05/2022 (3 years ago)", "Registered: 01/05/2022, set"]
)
def test_english_words_that_are_month_names_elsewhere_dont_change_en_gb_dates(text: str) -> None:
    [cand] = KeyValue().generate_in(st(text), ANGEBOT.field("zugelassen"), None)
    assert normalise(cand.raw, cand.normalise, ANGEBOT.field("zugelassen")) == date(2022, 5, 1)


def test_us_dates_are_month_first_and_mpg_is_us_gallons() -> None:
    us = values_in("en-US", "Registered 03/12/2024", "zugelassen")
    uk = values_in("en-GB", "Registered 03/12/2024", "zugelassen")
    assert us["03/12/2024"] == date(2024, 3, 12)
    assert uk["03/12/2024"] == date(2024, 12, 3)
    us = values_in("en-US", "Economy: 35 mpg", "verbrauch")
    uk = values_in("en-GB", "Economy: 35 mpg", "verbrauch")
    assert us["35 mpg"] == pytest.approx(6.720417)
    assert uk["35 mpg"] == pytest.approx(8.070884)


@pytest.mark.parametrize(
    "text",
    [
        "0-62 mph in 9.1 seconds, £18,495, 12 March 2024, 380 to 1,237 litres",
        "Price: £18,495 on the road",
        "Registered: 12/03/2024.",
    ],
)
@pytest.mark.parametrize("field", ["zero_to_62_s", "first_registered"])
def test_an_unknown_locale_is_read_as_en_gb(text: str, field: str) -> None:
    spec = SPEC.field(field)
    for gen in (NumberWithUnit(), Money(), DateGenerator(), Range(), KeyValue()):
        default = (
            gen.generate_for(st(text), spec)
            if isinstance(gen, KeyValue)
            else gen.generate(st(text))
        )
        assert gen.generate_in(st(text), spec, None) == default
        assert gen.generate_in(st(text), spec, "en-GB") == default


def test_locale_scoped_learned_generators_read_their_locales_numbers() -> None:
    spec = GeneratorSpec.parse(
        {
            "id": "gen-de",
            "field": "Angebot.gewicht_kg",
            "scope": {"locale": "de-DE"},
            "match": {"regex": r"Leergewicht\s+([\d.,]+)\s*kg", "group": 1},
            "normalise": ["parse_number"],
        }
    )
    field = ANGEBOT.field("gewicht_kg")
    text = st("Leergewicht 1.234,5 kg")
    [cand] = GeneratorRegistry([spec.to_generator()]).generate(
        text, field, schema="Angebot", locale="de-DE"
    )
    assert [s.model_dump() for s in cand.normalise] == [{"parse_number": {"decimal": ","}}]
    assert normalise(cand.raw, cand.normalise, field) == 1234.5
    # Its spec keeps the chain as written.
    assert [s.model_dump() for s in spec.normalise] == ["parse_number"]


def test_unscoped_and_explicit_learned_chains_are_used_as_written() -> None:
    field = ANGEBOT.field("gewicht_kg")
    text = st("Leergewicht 1.234 kg")
    unscoped = RegexGenerator(
        id="u", pattern=r"([\d.,]+) kg", group=1, normalise=(NormaliserStep(name="parse_number"),)
    )
    explicit = RegexGenerator(
        id="e",
        pattern=r"([\d.,]+) kg",
        group=1,
        normalise=(NormaliserStep(name="parse_number", args={"decimal": "."}),),
        scope=Scope(locale="de-DE"),
    )
    [u] = unscoped.generate(text)
    [e] = explicit.generate(text)
    assert normalise(u.raw, u.normalise, field) == 1.234
    assert normalise(e.raw, e.normalise, field) == 1.234


def test_builtins_that_depend_on_the_locale_are_locale_aware() -> None:
    aware = {g.id for g in default_registry() if isinstance(g, LocaleAwareGenerator)}
    assert aware == {"number_with_unit", "money", "date", "range", "key_value"}
