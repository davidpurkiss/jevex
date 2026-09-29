import json
import os
from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel

from jevex import DomLocation, Field, GeneratorRegistry, SchemaSpec, Statement
from jevex.generators import (
    BUILTIN_NORMALISER_ARGS,
    GeneratorSpec,
    InvalidGeneratorError,
    generator_spec_json_schema,
)
from jevex.generators.regex import MAX_PATTERN_LENGTH
from jevex.generators.spec import MAX_SPEC_CHARS
from jevex.normalise import BUILTIN_NORMALISERS, normalise

SCHEMA_FILE = Path(__file__).parent.parent / "docs" / "generator-spec.schema.json"

# The example from the spec (*Learning loop*), verbatim.
SPEC_YAML = r"""
id: gen-0f3a9c
field: VehicleSpec.zero_to_62_s
scope: {locale: en-GB}
match:
  regex: '0\s*[-–]\s*62(?:\s*mph)?\D{0,20}?(\d+(?:\.\d+)?)\s*(?:s|secs?|seconds)\b'
  group: 1
normalise:
  - parse_number
  - unit: {from: s, to: s}
provenance:
  learned_from: [ex-91c2]
  synthesised_by: generator_llm
  created: 2026-09-29
"""


class VehicleSpec(BaseModel):
    zero_to_62_s: float = Field(description="0-62 mph time", unit="s")


def st(text: str) -> Statement:
    return Statement(
        id="s1", text=text, kind="sentence", component_id="c1", location=DomLocation(dom_path="/")
    )


def minimal(**overrides: object) -> dict[str, object]:
    return {"id": "gen-1", "field": "Book.price", "match": {"regex": r"(\d+)", "group": 1}} | (
        overrides
    )


def test_the_spec_example_parses() -> None:
    spec = GeneratorSpec.from_yaml(SPEC_YAML)
    assert spec.id == "gen-0f3a9c"
    assert (spec.schema_name, spec.field_name) == ("VehicleSpec", "zero_to_62_s")
    assert spec.scope.locale == "en-GB"
    assert spec.match.group == 1
    assert [s.name for s in spec.normalise] == ["parse_number", "unit"]
    assert spec.normalise[1].args == {"from": "s", "to": "s"}
    assert spec.provenance.learned_from == ["ex-91c2"]
    assert spec.provenance.created == date(2026, 9, 29)


def test_yaml_round_trip_is_lossless_and_compact() -> None:
    spec = GeneratorSpec.from_yaml(SPEC_YAML)
    text = spec.to_yaml()
    assert GeneratorSpec.from_yaml(text) == spec
    assert "- parse_number\n" in text  # compact normaliser form
    assert "created: 2026-09-29\n" in text  # a YAML date, unquoted
    minimal_text = GeneratorSpec.parse(minimal()).to_yaml()
    assert "scope" not in minimal_text
    assert "provenance" not in minimal_text


def test_to_generator_runs_end_to_end_through_the_registry() -> None:
    spec = GeneratorSpec.from_yaml(SPEC_YAML)
    registry = GeneratorRegistry([spec.to_generator()])
    field = SchemaSpec.from_model(VehicleSpec).field("zero_to_62_s")

    assert registry.for_field(field, schema="VehicleSpec", locale="en-GB")
    assert not registry.for_field(field, schema="VehicleSpec", locale="de-DE")
    assert not registry.for_field(field, schema="OtherSchema", locale="en-GB")

    [cand] = registry.generate(
        st("It does 0-62 mph in 9.1 seconds."), field, schema="VehicleSpec", locale="en-GB"
    )
    assert cand.raw == "9.1"
    assert cand.generator_id == "gen-0f3a9c"
    assert normalise(cand.raw, cand.normalise, field) == 9.1


def test_rejects_patterns_re2_rejects() -> None:
    with pytest.raises(InvalidGeneratorError, match="RE2"):
        GeneratorSpec.parse(minimal(match={"regex": r"(a)\1", "group": 1}))
    with pytest.raises(InvalidGeneratorError, match="RE2"):
        GeneratorSpec.parse(minimal(match={"regex": r"a(?=b)"}))


def test_rejects_long_patterns() -> None:
    with pytest.raises(InvalidGeneratorError, match=r"match\.regex"):
        GeneratorSpec.parse(minimal(match={"regex": "a" * (MAX_PATTERN_LENGTH + 1)}))
    GeneratorSpec.parse(minimal(match={"regex": "a" * MAX_PATTERN_LENGTH}))


def test_rejects_a_missing_group() -> None:
    with pytest.raises(InvalidGeneratorError, match="group 2 doesn't exist"):
        GeneratorSpec.parse(minimal(match={"regex": r"(\d+)", "group": 2}))


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (["eval"], "isn't built in"),
        (["custom_normaliser"], "isn't built in"),
        ([{"unit": {"from": "parsecs"}}], "unknown unit 'parsecs'"),
        ([{"unit": {"from": "s", "via": "x"}}], "doesn't take 'via'"),
        ([{"parse_number": {"locale": "de"}}], "takes no arguments"),
        ([{"parse_date": {"order": "ydm"}}], "order"),
        ([{"parse_money": {"currency": "pounds"}}], "ISO 4217"),
        ([{"unit": {"gallon": "imperial"}}], "gallon"),
        (["strip"] * 9, "at most 8"),
    ],
)
def test_rejects_normalisers_outside_the_built_in_set(steps: list[object], message: str) -> None:
    with pytest.raises(InvalidGeneratorError, match=message):
        GeneratorSpec.parse(minimal(normalise=steps))


def test_every_built_in_normaliser_is_allowed_and_real() -> None:
    assert set(BUILTIN_NORMALISER_ARGS) == set(BUILTIN_NORMALISERS.names)
    spec = GeneratorSpec.parse(
        minimal(
            normalise=[
                "strip",
                "parse_range",
                {"unit": {"from": "km/h", "to": "mph", "gallon": "us"}},
                {"parse_money": {"currency": "GBP"}},
                {"parse_date": {"order": "dmy", "precision": "month"}},
            ]
        )
    )
    assert len(spec.normalise) == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"surprise": 1}, "surprise"),
        ({"scope": {"locale": "en-GB", "kinds": ["number"]}}, "scope.kinds"),
        ({"field": "price"}, "field"),
        ({"field": "Book.price.amount"}, "field"),
        ({"id": "has space"}, "id"),
        ({"id": ""}, "id"),
        ({"scope": {"locale": "not a locale"}}, "scope.locale"),
        ({"provenance": {"created": "yesterday"}}, "provenance.created"),
    ],
)
def test_rejects_malformed_specs(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(InvalidGeneratorError, match=message):
        GeneratorSpec.parse(minimal(**overrides))


def test_rejects_bad_yaml_and_non_mappings() -> None:
    with pytest.raises(InvalidGeneratorError, match="YAML"):
        GeneratorSpec.from_yaml("id: [unclosed")
    with pytest.raises(InvalidGeneratorError):
        GeneratorSpec.from_yaml("- just\n- a list\n")
    with pytest.raises(InvalidGeneratorError, match="limit"):
        GeneratorSpec.from_yaml("#" * (MAX_SPEC_CHARS + 1))


def test_yaml_is_loaded_safely() -> None:
    with pytest.raises(InvalidGeneratorError):
        GeneratorSpec.from_yaml("!!python/object/apply:os.system ['echo hi']")


def test_checked_in_json_schema_is_current() -> None:
    schema = generator_spec_json_schema()
    if os.environ.get("JEVEX_UPDATE_SCHEMA") == "1":
        SCHEMA_FILE.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    assert json.loads(SCHEMA_FILE.read_text()) == schema, (
        "docs/generator-spec.schema.json is stale; regenerate it with "
        "JEVEX_UPDATE_SCHEMA=1 uv run pytest tests/test_generator_spec.py"
    )


def test_json_schema_describes_the_compact_normaliser_form() -> None:
    schema = generator_spec_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"id", "field", "match"}
    options = schema["properties"]["normalise"]["items"]["anyOf"]
    assert {"const": "parse_number"} in options
    unit = next(o for o in options if o.get("required") == ["unit"])
    assert set(unit["properties"]["unit"]["properties"]) == {"from", "to", "gallon"}
    assert "NormaliserStep" not in schema.get("$defs", {})
