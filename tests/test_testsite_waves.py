import pytest

from jevex.testsite import FAMILIES
from jevex.testsite.waves import (
    DEFAULT_WAVES,
    check_waves,
    format_waves,
    parse_waves,
    wave_numbers,
)


def test_the_default_schedule_releases_every_family_once() -> None:
    families = [f for wave in DEFAULT_WAVES for f in wave]
    assert sorted(families) == sorted(FAMILIES)
    assert check_waves(DEFAULT_WAVES) == DEFAULT_WAVES
    assert DEFAULT_WAVES[0] == ("table", "listing")
    assert DEFAULT_WAVES[-1] == ("scanned", "infographic")  # OCR last


def test_parse_and_format_round_trip() -> None:
    waves = parse_waves(" table , listing ;kv; pdf,scanned ")
    assert waves == (("table", "listing"), ("kv",), ("pdf", "scanned"))
    assert format_waves(waves) == "table,listing;kv;pdf,scanned"
    assert parse_waves(format_waves(DEFAULT_WAVES)) == DEFAULT_WAVES


def test_wave_numbers_count_from_one() -> None:
    assert wave_numbers((("table", "listing"), ("kv",))) == {"table": 1, "listing": 1, "kv": 2}


def test_check_waves_accepts_lists() -> None:
    assert check_waves([["prose"], ["grid", "kv"]]) == (("prose",), ("grid", "kv"))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "wave 1 must be a non-empty list"),
        ("table;;kv", "wave 2 must be a non-empty list"),
        ("table;", "wave 2 must be a non-empty list"),
        ("table;tables", "wave 2: unknown family 'tables'; they're table, kv, prose"),
        ("table,kv;kv", "wave 2: 'kv' is already in wave 1"),
        ("table,table", "wave 1 lists 'table' twice"),
    ],
)
def test_bad_schedules_are_refused(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_waves(text)


def test_check_waves_refuses_empty_schedules_and_strings() -> None:
    with pytest.raises(ValueError, match="at least one wave"):
        check_waves([])
    with pytest.raises(ValueError, match="wave 2 must be a non-empty list"):
        check_waves([["table"], "kv"])  # a bare string would read as letters
    with pytest.raises(TypeError, match="use parse_waves"):
        check_waves("table;kv")
