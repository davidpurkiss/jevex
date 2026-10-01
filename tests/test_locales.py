import pytest

from jevex import LocaleConventions, NormaliserStep, locale_conventions, localise_steps
from jevex.locales import EN_GB


def steps(*items: object) -> list[NormaliserStep]:
    return [NormaliserStep.model_validate(i) for i in items]


@pytest.mark.parametrize("locale", [None, "", "en", "en-GB", "en_gb", "ja-JP", "xx"])
def test_en_gb_conventions_for_unknown_and_point_decimal_locales(locale: str | None) -> None:
    conventions = locale_conventions(locale)
    assert conventions.decimal == "."
    assert conventions.date_order == "dmy"
    assert conventions.gallon == "uk"
    assert not conventions.currency_after


def test_unknown_locales_are_en_gb() -> None:
    assert locale_conventions(None) is EN_GB
    assert locale_conventions("en-GB") == LocaleConventions(language="en")


@pytest.mark.parametrize("locale", ["de-DE", "de_AT", "DE", "fr-FR", "es-ES", "nl", "pt-BR"])
def test_decimal_comma_languages(locale: str) -> None:
    conventions = locale_conventions(locale)
    assert conventions.decimal == ","
    assert conventions.thousands == ".\u00a0\u202f\u2009"
    assert conventions.date_order == "dmy"
    assert conventions.currency_after
    assert conventions.language == locale[:2].lower()


@pytest.mark.parametrize("locale", ["de-CH", "de-LI", "it-CH", "es-MX", "es_pe"])
def test_regions_that_write_a_decimal_point_in_decimal_comma_languages(locale: str) -> None:
    conventions = locale_conventions(locale)
    assert conventions.decimal == "."
    assert conventions.date_order == "dmy"
    assert not conventions.currency_after
    assert locale_conventions("fr-CH").decimal == ","  # French Switzerland keeps the comma


@pytest.mark.parametrize("locale", ["en-US", "en_us", "es-US", "en-Latn-US"])
def test_us_regions_write_month_first_dates_and_use_us_gallons(locale: str) -> None:
    conventions = locale_conventions(locale)
    assert conventions.decimal == "."
    assert conventions.thousands == ","
    assert conventions.date_order == "mdy"
    assert conventions.gallon == "us"


def test_en_gb_chains_are_left_as_they_are() -> None:
    chain = steps("parse_number", {"unit": {"from": "mpg"}}, {"parse_date": {"precision": "day"}})
    assert localise_steps(chain, EN_GB) == chain


def test_decimal_comma_locales_add_decimal_to_number_range_and_money_steps() -> None:
    chain = steps(
        "parse_number",
        "parse_range",
        {"parse_money": {"currency": "EUR"}},
        {"unit": {"from": "kg"}},
        "parse_date",
        "strip",
    )
    out = localise_steps(chain, locale_conventions("de-DE"))
    assert [s.model_dump() for s in out] == [
        {"parse_number": {"decimal": ","}},
        {"parse_range": {"decimal": ","}},
        {"parse_money": {"currency": "EUR", "decimal": ","}},
        {"unit": {"from": "kg"}},
        "parse_date",
        "strip",
    ]


def test_us_locales_add_date_order_and_gallon() -> None:
    chain = steps("parse_number", {"unit": {"from": "mpg"}}, "parse_date")
    out = localise_steps(chain, locale_conventions("en-US"))
    assert [s.model_dump() for s in out] == [
        "parse_number",
        {"unit": {"from": "mpg", "gallon": "us"}},
        {"parse_date": {"order": "mdy"}},
    ]


def test_arguments_a_step_sets_are_never_overridden() -> None:
    chain = steps(
        {"parse_number": {"decimal": "."}},
        {"parse_date": {"order": "ymd"}},
        {"unit": {"from": "mpg", "gallon": "uk"}},
    )
    assert localise_steps(chain, locale_conventions("de-DE")) == chain
    assert localise_steps(chain, locale_conventions("en-US")) == chain
