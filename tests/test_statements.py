import pytest
from pydantic import TypeAdapter, ValidationError

from jevex import Candidate, DomLocation, EntityScope, NormaliserStep, Span, Statement


def statement(text: str = "0-62 mph in 9.1 seconds") -> Statement:
    return Statement(
        id="s1",
        text=text,
        kind="sentence",
        component_id="c1",
        heading_trail=["Performance"],
        location=DomLocation(dom_path="/p[1]"),
    )


def test_span_of() -> None:
    assert Span(start=12, end=15).of("0-62 mph in 9.1 seconds") == "9.1"


def test_span_rejects_reversed_or_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Span(start=5, end=2)
    with pytest.raises(ValueError, match="outside text"):
        Span(start=0, end=99).of("short")


@pytest.mark.parametrize(
    ("compact", "name", "args"),
    [
        ("parse_number", "parse_number", {}),
        ({"unit": {"from": "s", "to": "s"}}, "unit", {"from": "s", "to": "s"}),
        ({"trim": None}, "trim", {}),
        ({"name": "parse_date", "args": {"order": "dmy"}}, "parse_date", {"order": "dmy"}),
    ],
)
def test_normaliser_step_accepts_compact_forms(
    compact: object, name: str, args: dict[str, object]
) -> None:
    step = NormaliserStep.model_validate(compact)
    assert (step.name, step.args) == (name, args)


def test_normaliser_chain_serialises_to_compact_yaml_form() -> None:
    chain = TypeAdapter(list[NormaliserStep]).validate_python(
        ["parse_number", {"unit": {"from": "s", "to": "s"}}]
    )
    assert TypeAdapter(list[NormaliserStep]).dump_python(chain) == [
        "parse_number",
        {"unit": {"from": "s", "to": "s"}},
    ]


def test_normaliser_step_rejects_non_mapping_args() -> None:
    with pytest.raises(ValidationError):
        NormaliserStep.model_validate({"unit": "s"})


def test_candidate_from_statement_takes_raw_from_text() -> None:
    cand = Candidate.from_statement(
        statement(),
        Span(start=12, end=15),
        generator_id="number_with_unit",
        normalise=[NormaliserStep(name="parse_number")],
    )
    assert cand.raw == "9.1"
    assert Candidate.model_validate_json(cand.model_dump_json()) == cand


def test_entity_scope_defaults() -> None:
    scope = EntityScope(label="1.5 TSI SE", component_ids=["c2"])
    assert scope.statement_ids == []
    assert scope.parent is None
