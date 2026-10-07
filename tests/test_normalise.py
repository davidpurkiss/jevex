from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel

from jevex import (
    Candidate,
    Context,
    Document,
    DomLocation,
    Field,
    NormaliserStep,
    SchemaSpec,
    Span,
    Statement,
)
from jevex.entities import EntityScope
from jevex.interfaces import ParsedDocument, Selection
from jevex.layout import Component
from jevex.normalise import (
    BUILTIN_NORMALISERS,
    FunctionNormaliser,
    NormaliseError,
    NormaliseStage,
    canonical_unit,
    convert,
    normalise,
    parse_date,
    parse_money,
    parse_number,
    parse_range,
    run_chain,
    strip,
)
from jevex.results import FieldMeta
from jevex.testing import FakeJev


def steps(*items: object) -> list[NormaliserStep]:
    return [NormaliserStep.model_validate(i) for i in items]


class Car(BaseModel):
    power_kw: float = Field(description="Power", unit="kW")
    zero_to_62_s: float = Field(description="0-62 time", unit="s")
    economy: float = Field(description="Economy", unit="l/100km")
    price: Decimal = Field(description="Price", unit="GBP")
    seats: int = Field(description="Seats")
    registered: date = Field(description="Registered")
    model_year: int = Field(description="Model year")
    colours: list[str] = Field(default_factory=list, description="Colours")
    boot_litres: list[float] = Field(default_factory=list, description="Boot", unit="l")


SPEC = SchemaSpec.from_model(Car)


def f(name: str):
    return SPEC.field(name)


# --- parsing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "value"),
    [("18,495", 18495), ("9.1 s", 9.1), ("150PS", 150), ("-3", -3), (".5", 0.5), (7, 7)],
)
def test_parse_number(raw: object, value: float) -> None:
    assert parse_number(raw) == value


@pytest.mark.parametrize("raw", ["no digits", True])
def test_parse_number_rejects(raw: object) -> None:
    with pytest.raises(NormaliseError):
        parse_number(raw)


def test_strip() -> None:
    assert strip("  Moonstone Grey. \n") == "Moonstone Grey"
    assert strip("A\u00a0Light in\u202f the\u2009Attic\u00a0") == "A Light in the Attic"
    assert strip(5) == 5


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("£18,495", Decimal(18495)),
        ("£1.5m", Decimal(1_500_000)),
        ("£1.5 million", Decimal(1_500_000)),
        ("25k GBP", Decimal(25_000)),
        ("€2bn", Decimal(2_000_000_000)),
        ("£19.99", Decimal("19.99")),
    ],
)
def test_parse_money(raw: str, value: Decimal) -> None:
    assert parse_money(raw) == value
    assert "E" not in str(parse_money(raw))  # never exponent notation


def test_parse_money_refuses_the_wrong_currency_for_the_field() -> None:
    with pytest.raises(NormaliseError, match="EUR"):
        parse_money("€30,000", field=f("price"), currency="EUR")
    assert parse_money("£30,000", field=f("price"), currency="GBP") == Decimal(30000)


@pytest.mark.parametrize(
    ("raw", "args", "value"),
    [
        ("12 March 2024", {}, date(2024, 3, 12)),
        ("12th March, 2024", {"order": "dmy"}, date(2024, 3, 12)),
        ("March 12, 2024", {"order": "mdy"}, date(2024, 3, 12)),
        ("03/04/2026", {"order": "dmy"}, date(2026, 4, 3)),
        ("03/04/2026", {"order": "mdy"}, date(2026, 3, 4)),
        ("2024-09-01", {"order": "ymd"}, date(2024, 9, 1)),
        ("2024-09-01", {}, date(2024, 9, 1)),  # year first: ymd without being told
        ("Mar 2025", {"precision": "month"}, date(2025, 3, 1)),
        ("2024", {"precision": "year"}, date(2024, 1, 1)),
    ],
)
def test_parse_date(raw: str, args: dict[str, Any], value: date) -> None:
    assert parse_date(raw, **args) == value


def test_year_precision_gives_an_int_for_number_fields() -> None:
    assert parse_date("MY2024", field=f("model_year"), precision="year") == 2024


@pytest.mark.parametrize("raw", ["31/02/2024", "no date here", "March"])
def test_parse_date_rejects(raw: str) -> None:
    with pytest.raises(NormaliseError):
        parse_date(raw)


@pytest.mark.parametrize(
    ("raw", "value"),
    [("5–7", [5, 7]), ("380 to 1,237 litres", [380, 1237]), ("between 4 and 5", [4, 5])],
)
def test_parse_range(raw: str, value: list[int]) -> None:
    assert parse_range(raw) == value


# --- decimal commas and other locales (#56) -------------------------------------------


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("1.234,5", 1234.5),
        ("18.495", 18495),
        ("9,1 s", 9.1),
        ("1\u00a0234,5 kg", 1234.5),
        ("1\u202f234", 1234),
        ("1\u2009234,5", 1234.5),
        ("-0,5", -0.5),
        (",5", 0.5),
        ("1.234.567", 1234567),
    ],
)
def test_parse_number_with_a_decimal_comma(raw: str, value: float) -> None:
    assert parse_number(raw, decimal=",") == value


def test_parse_number_reads_a_decimal_point_by_default() -> None:
    assert parse_number("1.234,5") == 1.234  # en-GB: the first number is "1.234"


@pytest.mark.parametrize("decimal", [";", "", "·"])
def test_a_decimal_mark_other_than_point_or_comma_is_refused(decimal: str) -> None:
    with pytest.raises(NormaliseError, match="decimal must be"):
        parse_number("1,5", decimal=decimal)
    with pytest.raises(NormaliseError, match="decimal must be"):
        parse_money("1,5 €", decimal=decimal)
    with pytest.raises(NormaliseError, match="decimal must be"):
        parse_range("1,5-2,5", decimal=decimal)


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("18.495 €", Decimal(18495)),
        ("18.495,50 €", Decimal("18495.50")),
        ("€ 1.299,99", Decimal("1299.99")),
        ("EUR 30.000", Decimal(30000)),
        ("2,5k EUR", Decimal(2500)),
    ],
)
def test_parse_money_with_a_decimal_comma(raw: str, value: Decimal) -> None:
    assert parse_money(raw, decimal=",") == value


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("1,4–2,0", [1.4, 2.0]),
        ("1.200 bis 1.500 kg", [1200, 1500]),
        ("5-7", [5, 7]),
        ("zwischen 4 und 5", [4, 5]),
        ("de 4 à 5", [4, 5]),
    ],
)
def test_parse_range_with_a_decimal_comma(raw: str, value: list[float]) -> None:
    assert parse_range(raw, decimal=",") == value


def test_parse_range_reads_list_items_with_the_decimal_mark() -> None:
    assert parse_range(["1,4", "2,0"], decimal=",") == [1.4, 2.0]
    assert parse_range(["1,400", "2,000"]) == [1400, 2000]


@pytest.mark.parametrize(
    ("raw", "args", "value"),
    [
        ("12. März 2024", {"order": "dmy"}, date(2024, 3, 12)),
        ("1. Dezember 2023", {}, date(2023, 12, 1)),
        ("3. Jänner 2025", {}, date(2025, 1, 3)),
        ("Mai 2024", {"precision": "month"}, date(2024, 5, 1)),
        ("Okt. 2024", {"precision": "month"}, date(2024, 10, 1)),
        ("12.03.2024", {"order": "dmy"}, date(2024, 3, 12)),
        # A four-digit first number is the year, whatever the locale's order says.
        ("2024-03-12", {"order": "mdy"}, date(2024, 3, 12)),
        ("2024-03-12", {"order": "dmy"}, date(2024, 3, 12)),
    ],
)
def test_parse_date_in_other_locales(raw: str, args: dict[str, Any], value: date) -> None:
    assert parse_date(raw, **args) == value


@pytest.mark.parametrize(
    ("raw", "args", "value"),
    [
        ("12 mars 2024", {}, date(2024, 3, 12)),
        ("1er août 2024", {}, date(2024, 8, 1)),
        ("févr. 2025", {"precision": "month"}, date(2025, 2, 1)),
        ("12 de marzo de 2024", {}, date(2024, 3, 12)),
        ("septiembre de 2024", {"precision": "month"}, date(2024, 9, 1)),
        ("3 giugno 2025", {}, date(2025, 6, 3)),
        ("12 maart 2024", {}, date(2024, 3, 12)),
        ("1 mei 2024", {}, date(2024, 5, 1)),
    ],
)
def test_parse_date_reads_french_spanish_italian_and_dutch_months(
    raw: str, args: dict[str, Any], value: date
) -> None:
    assert parse_date(raw, **args) == value


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        # Other languages' short month names are English words: the name by the year counts.
        ("2 years ago, in March 2024", date(2024, 3, 1)),
        ("set for May 2025", date(2025, 5, 1)),
        ("März 2024, nicht Mai", date(2024, 3, 1)),
    ],
)
def test_parse_date_takes_the_month_name_nearest_the_year(raw: str, value: date) -> None:
    assert parse_date(raw, precision="month") == value


def test_parse_date_with_month_names_but_no_year_is_refused() -> None:
    with pytest.raises(NormaliseError, match="not a date"):
        parse_date("mars ou avril")


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("1,5 Mio. €", Decimal(1500000)),
        ("1,5 Millionen €", Decimal(1500000)),
        ("2 Mrd. EUR", Decimal(2000000000)),
        ("500 Tsd. €", Decimal(500000)),
        ("2 Mds €", Decimal(2000000000)),
        ("1,2 milliard €", Decimal(1200000000)),
        ("2 mil millones €", Decimal(2000000000)),
        ("3,5 millones €", Decimal(3500000)),
        ("3 mln €", Decimal(3000000)),
        ("1,2 miljoen €", Decimal(1200000)),
        ("18.495,- €", Decimal(18495)),
        ("€ 18.495,–", Decimal(18495)),
    ],
)
def test_parse_money_reads_other_languages_multipliers(raw: str, value: Decimal) -> None:
    assert parse_money(raw, decimal=",") == value


def test_parse_money_multiplier_words_need_a_word_end() -> None:
    assert parse_money("2 Mioxx €", decimal=",") == Decimal(2)
    # Spanish "mil" on its own isn't read: in English it's a million, or a thousandth of an inch.
    assert parse_money("£5 mil") == Decimal(5)
    assert parse_money("£1.5 billion") == Decimal(1500000000)  # English is short scale


@pytest.mark.parametrize(
    ("raw", "value"),
    [("1’250.50", 1250.5), ("1'250", 1250), ("1’250’000 kg", 1250000), ("12’5", 12)],
)
def test_parse_number_reads_swiss_apostrophe_grouping(raw: str, value: float) -> None:
    assert parse_number(raw) == value


def test_parse_money_and_range_read_swiss_apostrophe_grouping() -> None:
    assert parse_money("CHF 1’250.50") == Decimal("1250.50")
    assert parse_money("CHF 1’250.–") == Decimal(1250)
    assert parse_range("1’200–1’500 kg") == [1200, 1500]


def test_decimal_comma_chains_validate_against_the_field() -> None:
    chain = steps({"parse_number": {"decimal": ","}}, {"unit": {"from": "l", "to": "l"}})
    assert normalise("Kofferraum: 1.234,5 l", chain, f("boot_litres")) == 1234.5
    assert normalise("1,4 Liter", chain, f("boot_litres")) == 1.4
    price = steps({"parse_money": {"currency": "EUR", "decimal": ","}})
    with pytest.raises(NormaliseError, match="EUR"):  # still no currency conversion
        normalise("18.495 €", price, f("price"))


def test_us_gallons_convert_mpg_differently() -> None:
    uk = normalise("35 mpg", steps("parse_number", {"unit": {"from": "mpg"}}), f("economy"))
    us = normalise(
        "35 mpg", steps("parse_number", {"unit": {"from": "mpg", "gallon": "us"}}), f("economy")
    )
    assert uk == pytest.approx(8.070884)
    assert us == pytest.approx(6.720417)


# --- units -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (150, "PS", "kW", 110.325),
        (110, "kW", "PS", 149.558),
        (148, "bhp", "kW", 110.364),
        (62, "mph", "km/h", 99.779),
        (100, "km/h", "mph", 62.137),
        (250, "Nm", "lb ft", 184.391),
        (1498, "cc", "l", 1.498),
        (1.5, "litres", "cc", 1500),
        (4284, "mm", "m", 4.284),
        (52.3, "mpg", "l/100km", 5.401),  # UK gallons
        (5.4, "l/100km", "mpg", 52.311),
        (9.1, "seconds", "s", 9.1),
    ],
)
def test_convert(value: float, src: str, dst: str, expected: float) -> None:
    assert convert(value, src, dst) == pytest.approx(expected, rel=1e-4)


def test_us_gallons() -> None:
    assert convert(40, "mpg", "l/100km", gallon="us") == pytest.approx(5.880, rel=1e-3)


@pytest.mark.parametrize(("src", "dst"), [("kW", "mph"), ("kg", "l"), ("s", "km")])
def test_convert_rejects_mismatched_dimensions(src: str, dst: str) -> None:
    with pytest.raises(NormaliseError, match="can't convert"):
        convert(1, src, dst)


def test_canonical_unit() -> None:
    assert canonical_unit("Litres") == "l"
    assert canonical_unit("PS") == "PS"
    assert canonical_unit("kw") == "kW"
    with pytest.raises(NormaliseError, match="unknown unit"):
        canonical_unit("furlongs")


# --- chains and validation -------------------------------------------------------------


def test_chain_converts_to_the_fields_unit() -> None:
    chain = steps("parse_number", {"unit": {"from": "PS"}})
    assert normalise("150PS", chain, f("power_kw")) == pytest.approx(110.325, rel=1e-4)


def test_explicit_target_unit_wins() -> None:
    chain = steps("parse_number", {"unit": {"from": "PS", "to": "bhp"}})
    assert run_chain("150PS", chain) == pytest.approx(147.95, rel=1e-3)


def test_bare_number_is_taken_as_the_fields_unit() -> None:
    assert normalise("9.1", steps("parse_number"), f("zero_to_62_s")) == 9.1


def test_range_with_unit_converts_each_end() -> None:
    chain = steps("parse_range", {"unit": {"from": "cc"}})
    assert run_chain("5000 to 6000 cc", chain, f("boot_litres")) == [5.0, 6.0]


def test_result_is_validated_against_the_field_type() -> None:
    assert normalise("5", steps("strip"), f("seats")) == 5  # coerced
    with pytest.raises(NormaliseError, match="doesn't fit seats"):
        normalise("five", steps("strip"), f("seats"))


def test_list_fields_validate_one_item() -> None:
    assert normalise(" Pure White ", steps("strip"), f("colours")) == "Pure White"


def test_unknown_step_and_bad_arguments() -> None:
    with pytest.raises(NormaliseError, match="unknown normaliser 'shout'"):
        run_chain("x", steps("shout"))
    with pytest.raises(NormaliseError, match="parse_number failed"):
        run_chain("5", steps({"parse_number": {"base": 16}}))


def test_custom_normalisers_can_be_registered() -> None:
    def upper(value: object, *, field: object = None) -> str:
        return str(value).upper()

    registry = BUILTIN_NORMALISERS.with_normaliser(FunctionNormaliser("upper", upper))
    assert run_chain("golf", steps("upper"), registry=registry) == "GOLF"
    assert "upper" not in BUILTIN_NORMALISERS
    assert set(BUILTIN_NORMALISERS.names) == {
        "strip",
        "parse_number",
        "unit",
        "parse_money",
        "parse_date",
        "parse_range",
    }


# --- stage -----------------------------------------------------------------------------

LOC = DomLocation(dom_path="/p")


def statement(sid: str, text: str) -> Statement:
    return Statement(id=sid, text=text, kind="sentence", component_id=f"c-{sid}", location=LOC)


def pick(st: Statement, raw: str, confidence: float, *chain: object, **alts: float) -> Selection:
    start = st.text.index(raw)
    cand = Candidate.from_statement(
        st, Span(start=start, end=start + len(raw)), generator_id="gen", normalise=steps(*chain)
    )
    return Selection(candidate=cand, confidence=confidence, alternatives=alts)


def context(*statements: Statement) -> Context:
    ctx = Context.create(
        Document.from_bytes(b"<p/>", url="https://example.com"), [SPEC], FakeJev().client()
    )
    root = Component(id="root", type="section", location=LOC)
    ctx.parsed = ParsedDocument(
        document=ctx.document, root=root, statements={s.id: s for s in statements}
    )
    return ctx


async def test_stage_picks_the_most_confident_valid_candidate() -> None:
    a, b = statement("s1", "0-62 mph in 9.1 s"), statement("s2", "0-62: 9.4 seconds")
    ctx = context(a, b)
    run = ctx.schemas["Car"]
    run.selections[("doc", "zero_to_62_s", "s1")] = pick(
        a, "9.1 s", 0.9, "parse_number", **{"62 mph": 0.05}
    )
    run.selections[("doc", "zero_to_62_s", "s2")] = pick(b, "9.4 seconds", 0.6, "parse_number")
    await NormaliseStage().run(ctx)
    meta = run.fields["doc"]["zero_to_62_s"]
    assert meta.value == 9.1
    assert meta.confidence == 0.9
    assert meta.method == "generator"
    assert meta.generator_id == "gen"
    assert meta.source is not None
    assert meta.source.statement == "0-62 mph in 9.1 s"
    assert meta.source.url == "https://example.com"
    assert meta.source.span == Span(start=12, end=17)
    assert [(alt.raw, alt.p) for alt in meta.alternatives] == [
        ("9.4 seconds", 0.6),
        ("62 mph", 0.05),
    ]


async def test_stage_falls_back_when_the_best_candidate_doesnt_normalise() -> None:
    a, b = statement("s1", "Seats: five"), statement("s2", "Seats 5")
    ctx = context(a, b)
    run = ctx.schemas["Car"]
    run.selections[("doc", "seats", "s1")] = pick(a, "five", 0.9, "strip")
    run.selections[("doc", "seats", "s2")] = pick(b, "5", 0.7, "parse_number")
    await NormaliseStage().run(ctx)
    assert run.fields["doc"]["seats"].value == 5
    assert run.fields["doc"]["seats"].confidence == 0.7


async def test_stage_records_the_error_when_nothing_normalises() -> None:
    a = statement("s1", "Seats: five")
    ctx = context(a)
    run = ctx.schemas["Car"]
    run.selections[("doc", "seats", "s1")] = pick(a, "five", 0.9, "strip")
    await NormaliseStage().run(ctx)
    meta = run.fields["doc"]["seats"]
    assert not meta.found
    assert meta.error is not None
    assert "doesn't fit seats" in meta.error


async def test_stage_keeps_every_value_for_list_fields_in_document_order() -> None:
    a, b = statement("s1", "In Pure White"), statement("s2", "Also Moonstone Grey or Pure White")
    ctx = context(a, b)
    run = ctx.schemas["Car"]
    run.selections[("doc", "colours", "s2")] = pick(b, "Moonstone Grey", 0.95, "strip")
    run.selections[("doc", "colours", "s1")] = pick(a, "Pure White", 0.8, "strip")
    await NormaliseStage().run(ctx)
    assert run.fields["doc"]["colours"].value == ["Pure White", "Moonstone Grey"]


async def test_stage_prefers_the_entitys_own_statements_over_shared_ones() -> None:
    own, everyone = statement("s1", "Seats 4"), statement("s2", "Every trim seats 5")
    ctx = context(own, everyone)
    run = ctx.schemas["Car"]
    run.scopes = [EntityScope(label="doc", statement_ids=["s1"], shared_statement_ids=["s2"])]
    run.selections[("doc", "seats", "s1")] = pick(own, "4", 0.6, "parse_number")
    run.selections[("doc", "seats", "s2")] = pick(everyone, "5", 0.95, "parse_number")
    run.selections[("doc", "price", "s2")] = pick(everyone, "5", 0.9, "parse_number")
    await NormaliseStage().run(ctx)
    seats, price = run.fields["doc"]["seats"], run.fields["doc"]["price"]
    assert (seats.value, seats.shared) == (4, False)
    assert [a.raw for a in seats.alternatives] == ["5"]
    assert (price.value, price.shared) == (5, True)  # only a shared statement gave it


async def test_stage_marks_a_list_shared_only_when_every_value_is() -> None:
    own, everyone = statement("s1", "In Pure White"), statement("s2", "All come in Moonstone Grey")
    ctx = context(own, everyone)
    run = ctx.schemas["Car"]
    run.scopes = [
        EntityScope(label="a", statement_ids=["s1"], shared_statement_ids=["s2"]),
        EntityScope(label="b", shared_statement_ids=["s2"]),
    ]
    for scope in ("a", "b"):
        run.selections[(scope, "colours", "s2")] = pick(everyone, "Moonstone Grey", 0.9, "strip")
    run.selections[("a", "colours", "s1")] = pick(own, "Pure White", 0.8, "strip")
    await NormaliseStage().run(ctx)
    a, b = run.fields["a"]["colours"], run.fields["b"]["colours"]
    assert (a.value, a.shared) == (["Pure White", "Moonstone Grey"], False)
    assert (b.value, b.shared) == (["Moonstone Grey"], True)


async def test_stage_skips_none_picks_and_never_overwrites_found_values() -> None:
    a = statement("s1", "Price £18,495")
    ctx = context(a)
    run = ctx.schemas["Car"]
    run.selections[("doc", "seats", "s1")] = Selection(candidate=None, confidence=0.9)
    run.selections[("doc", "price", "s1")] = pick(
        a, "£18,495", 0.9, {"parse_money": {"currency": "GBP"}}
    )
    run.set_field("doc", "price", FieldMeta(value=Decimal(17000), method="structured"))
    await NormaliseStage().run(ctx)
    assert "seats" not in run.fields["doc"]
    assert run.fields["doc"]["price"].value == Decimal(17000)
    assert run.fields["doc"]["price"].method == "structured"


async def test_in_merge_mode_a_disagreeing_value_from_embedded_data_is_a_conflict() -> None:
    a = statement("s1", "Price £18,495")
    ctx = context(a)
    run = ctx.schemas["Car"]
    run.merge = True
    run.selections[("doc", "price", "s1")] = pick(
        a, "£18,495", 0.9, {"parse_money": {"currency": "GBP"}}
    )
    run.set_field("doc", "price", FieldMeta(value=Decimal(17000), method="structured"))
    await NormaliseStage().run(ctx)
    meta = run.fields["doc"]["price"]
    assert (meta.value, meta.method) == (Decimal(17000), "structured")  # no confidence: certain
    [conflict] = meta.conflicts
    assert (conflict.value, conflict.method, conflict.confidence) == (
        Decimal(18495),
        "generator",
        0.9,
    )
    assert conflict.source is not None
    assert conflict.source.statement == "Price £18,495"


async def test_stage_is_in_the_default_pipeline() -> None:
    from jevex.extractor import default_pipeline

    assert "normalise" in default_pipeline()


# --- review fixes ----------------------------------------------------------------------


class Extra(BaseModel):
    engine_cc: int = Field(description="Engine", unit="cc")
    power_ps: int = Field(description="Power", unit="PS")
    seats: int = Field(ge=1, le=9, description="Seats")
    seat_options: list[int] = Field(default_factory=list, description="Seat options")
    boot: list[float] = Field(default_factory=list, description="Boot", unit="l")


EXTRA = SchemaSpec.from_model(Extra)


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("£1.5 Million", Decimal(1_500_000)),
        ("£2Bn", Decimal(2_000_000_000)),
        ("£3 Thousand", Decimal(3000)),
        ("€2 MN", Decimal(2_000_000)),
    ],
)
def test_money_multipliers_ignore_case(raw: str, value: Decimal) -> None:
    assert parse_money(raw) == value


def test_int_fields_with_units_round_after_conversion() -> None:
    assert (
        normalise(
            "1.4 litre", steps("parse_number", {"unit": {"from": "l"}}), EXTRA.field("engine_cc")
        )
        == 1400
    )
    assert (
        normalise(
            "110 kW", steps("parse_number", {"unit": {"from": "kW"}}), EXTRA.field("power_ps")
        )
        == 150
    )
    assert convert(1.4, "l", "cc") == 1400.0  # no float noise


def test_field_constraints_are_enforced() -> None:
    with pytest.raises(NormaliseError, match="less than or equal to 9"):
        normalise("12", steps("parse_number"), EXTRA.field("seats"))


def test_ranges_validate_on_list_fields() -> None:
    assert normalise("5–7", steps("parse_range"), EXTRA.field("seat_options")) == [5, 7]
    assert normalise(
        "380 to 1,237 litres", steps("parse_range", {"unit": {"from": "l"}}), EXTRA.field("boot")
    ) == [380.0, 1237.0]
    with pytest.raises(NormaliseError):
        normalise("5–7", steps("parse_range"), EXTRA.field("seats"))  # a range isn't one int


def test_inverse_range_conversions_stay_ordered() -> None:
    assert run_chain(
        "40-50 mpg", steps("parse_range", {"unit": {"from": "mpg", "to": "l/100km"}})
    ) == [
        pytest.approx(5.6496, rel=1e-3),
        pytest.approx(7.0620, rel=1e-3),
    ]


@pytest.mark.parametrize(
    ("raw", "order", "value"),
    [
        ("12/03/24", "dmy", date(2024, 3, 12)),
        ("12/03/98", "dmy", date(1998, 3, 12)),
        ("24-03-12", "ymd", date(2024, 3, 12)),
    ],
)
def test_two_digit_years_pivot(raw: str, order: str, value: date) -> None:
    assert parse_date(raw, order=order) == value


def test_unknown_gallon_is_a_normalise_error() -> None:
    with pytest.raises(NormaliseError, match="gallon"):
        run_chain(
            "40 mpg",
            steps("parse_number", {"unit": {"from": "mpg", "to": "l/100km", "gallon": "imperial"}}),
        )


async def test_stage_extends_list_fields_with_ranges() -> None:
    a = statement("s1", "Seats 5–7")
    ctx = Context.create(Document.from_bytes(b"<p/>"), [EXTRA], FakeJev().client())
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={"s1": a},
    )
    run = ctx.schemas["Extra"]
    run.selections[("doc", "seat_options", "s1")] = pick(a, "5–7", 0.9, "parse_range")
    await NormaliseStage().run(ctx)
    assert run.fields["doc"]["seat_options"].value == [5, 7]


async def test_alternatives_never_repeat_the_chosen_span() -> None:
    a, b = statement("s1", "0-62 in 9.1 s"), statement("s2", "0-62: 9.1 s")
    ctx = context(a, b)
    run = ctx.schemas["Car"]
    run.selections[("doc", "zero_to_62_s", "s1")] = pick(
        a, "9.1 s", 0.9, "parse_number", **{"62": 0.1}
    )
    run.selections[("doc", "zero_to_62_s", "s2")] = pick(
        b, "9.1 s", 0.8, "parse_number", **{"62": 0.2}
    )
    await NormaliseStage().run(ctx)
    alts = run.fields["doc"]["zero_to_62_s"].alternatives
    assert [(x.raw, x.p) for x in alts] == [("62", 0.2)]


def test_int_rounding_only_after_a_real_conversion_and_half_up() -> None:
    same_unit = steps("parse_number", {"unit": {"from": "PS"}})
    with pytest.raises(NormaliseError):
        normalise(
            "9.5 PS", same_unit, EXTRA.field("power_ps")
        )  # nothing converted: 9.5 isn't an int
    to_ps = steps("parse_number", {"unit": {"from": "kW"}})
    # 1.838746875 kW is exactly 2.5 PS: half rounds up, not to even
    assert normalise("1.838746875 kW", to_ps, EXTRA.field("power_ps")) == 3


# --- list fields: several accepted candidates per statement (Selection.accepted) -------


class Wheels(BaseModel):
    wheel_sizes: list[float] = Field(default_factory=list, description="Wheel sizes")
    front_wheel: float = Field(default=0, description="Front wheel size")


WHEELS = SchemaSpec.from_model(Wheels)


def many_pick(st: Statement, raws: list[str], confidence: float, *chain: object) -> Selection:
    cands: list[Candidate] = []
    for raw in raws:
        start = st.text.index(raw)
        cands.append(
            Candidate.from_statement(
                st,
                Span(start=start, end=start + len(raw)),
                generator_id="gen",
                normalise=steps(*chain),
            )
        )
    return Selection(candidate=cands[0], confidence=confidence, accepted=cands)


def wheels_context(*statements: Statement) -> Context:
    ctx = Context.create(Document.from_bytes(b"<p/>"), [WHEELS], FakeJev().client())
    ctx.parsed = ParsedDocument(
        document=ctx.document,
        root=Component(id="root", type="section", location=LOC),
        statements={s.id: s for s in statements},
    )
    return ctx


async def test_list_fields_take_every_accepted_candidate_in_text_order() -> None:
    a = statement("s1", "Wheel sizes: 19, 17 or 18 inch")
    ctx = wheels_context(a)
    run = ctx.schemas["Wheels"]
    run.selections[("doc", "wheel_sizes", "s1")] = many_pick(a, ["19", "17"], 0.9, "parse_number")
    await NormaliseStage().run(ctx)
    meta = run.fields["doc"]["wheel_sizes"]
    assert meta.value == [19.0, 17.0]
    assert all(alt.raw not in ("19", "17") for alt in meta.alternatives)


async def test_one_bad_accepted_candidate_keeps_the_others() -> None:
    a = statement("s1", "Sizes: 17, eighteen")
    ctx = wheels_context(a)
    run = ctx.schemas["Wheels"]
    run.selections[("doc", "wheel_sizes", "s1")] = many_pick(a, ["17", "eighteen"], 0.9, "strip")
    await NormaliseStage().run(ctx)
    assert run.fields["doc"]["wheel_sizes"].value == [17.0]


async def test_scalar_fields_ignore_accepted() -> None:
    a = statement("s1", "Front 18, rear 19")
    ctx = wheels_context(a)
    run = ctx.schemas["Wheels"]
    sel = many_pick(a, ["18", "19"], 0.9, "parse_number")
    run.selections[("doc", "front_wheel", "s1")] = sel
    await NormaliseStage().run(ctx)
    assert run.fields["doc"]["front_wheel"].value == 18.0


async def test_values_from_a_vision_statement_are_tagged_vision() -> None:
    said = statement("s1", "The dial tops out at 9.1 s").model_copy(update={"kind": "vision"})
    ctx = context(said)
    run = ctx.schemas["Car"]
    run.selections[("doc", "zero_to_62_s", "s1")] = pick(said, "9.1 s", 0.8, "parse_number")
    await NormaliseStage().run(ctx)
    meta = run.fields["doc"]["zero_to_62_s"]
    assert (meta.value, meta.method) == (9.1, "vision")
