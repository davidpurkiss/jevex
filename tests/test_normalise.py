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


async def test_stage_is_in_the_default_pipeline() -> None:
    from jevex.extractor import default_pipeline

    assert "normalise" in default_pipeline()
