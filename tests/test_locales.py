import pytest

from jevex import (
    Document,
    LocaleConventions,
    NormaliserStep,
    canonical_locale,
    checked_locale,
    document_locale,
    html_language,
    locale_conventions,
    localise_steps,
)
from jevex.locales import EN_GB, MONTH_NAMES, MULTIPLIERS, RANGE_WORDS


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


@pytest.mark.parametrize("locale", ["de-CH", "de-LI", "it-CH", "en-CH"])
def test_swiss_decimal_points_come_with_apostrophe_grouping(locale: str) -> None:
    conventions = locale_conventions(locale)
    assert conventions.apostrophe_groups
    assert conventions.thousands == ",'’"


@pytest.mark.parametrize("locale", [None, "en-GB", "de-DE", "fr-CH", "es-MX", "en-US"])
def test_other_locales_group_without_apostrophes(locale: str | None) -> None:
    assert not locale_conventions(locale).apostrophe_groups
    assert "'" not in locale_conventions(locale).thousands


@pytest.mark.parametrize("table", [MONTH_NAMES, MULTIPLIERS])
def test_a_word_means_the_same_in_every_language_that_has_it(
    table: dict[str, dict[str, int]],
) -> None:
    # The normalisers read every language's words, whatever the page's language.
    meanings: dict[str, set[int]] = {}
    for words in table.values():
        for word, value in words.items():
            assert word == word.lower()
            meanings.setdefault(word, set()).add(value)
    assert {word: values for word, values in meanings.items() if len(values) > 1} == {}


def test_languages_with_their_own_words() -> None:
    assert set(MONTH_NAMES) == {"en", "de", "fr", "es", "it", "nl"}
    assert set(MULTIPLIERS) == set(RANGE_WORDS) == {"de", "fr", "es", "it", "nl"}
    assert "mil" not in MULTIPLIERS["es"]  # a million, or a thousandth of an inch, in English


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


# --- document_locale -------------------------------------------------------------------


def html(
    markup: str, *, content_language: str | None = None, locale: str | None = None
) -> Document:
    return Document.from_bytes(
        markup.encode(), content_type="text/html", content_language=content_language, locale=locale
    )


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (html('<!doctype html><html lang="de-DE"><body><p>9,1 s</p>'), "de-DE"),
        (html("<HTML LANG='fr'><HEAD></HEAD><BODY>"), "fr"),
        (html('<html xml:lang="nl-BE"><body>'), "nl-BE"),
        (html('<html lang="de-AT" xml:lang="en"><body>'), "de-AT"),
        (
            html('<html><head><meta http-equiv="Content-Language" content="es-ES, en">'),
            "es-ES",
        ),
        (html('<html lang="de"><body>', content_language="fr-FR"), "de"),
        (html("<html><body><p>Hallo</p>", content_language="de-CH"), "de-CH"),
        (html("<html><body>", content_language="de-DE, en-GB"), "de-DE"),
        (html('<html lang="de-DE"><body>', locale="en_US"), "en-US"),
        (html('<html lang="en-gb"><body>'), "en-GB"),
        (html("<html><body>", content_language="DE_de"), "de-DE"),
        (html("<p>Hallo</p>"), None),
        (html('<html lang=""><body>'), None),
        (html('<html lang="English"><body>', content_language="en-GB"), "en-GB"),
        (html("<html><body>", content_language="*"), None),
        (Document.from_bytes(b"%PDF-1.7", content_language="de"), "de"),
        (Document.from_bytes(b"%PDF-1.7", locale="pl-PL"), "pl-PL"),
        (Document.from_bytes(b"%PDF-1.7"), None),
    ],
)
def test_document_locale_takes_the_caller_then_the_page_then_the_header(
    document: Document, expected: str | None
) -> None:
    assert document_locale(document) == expected


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("de-DE", "de-DE"),
        ("de_DE", "de-DE"),
        ("de-de", "de-DE"),
        ("DE_de", "de-DE"),
        ("FR", "fr"),
        ("zh_hant_tw", "zh-Hant-TW"),
        ("ES-419", "es-419"),
        ("sr-LATN-rs", "sr-Latn-RS"),
        ("de-CH-1996", "de-CH-1996"),
        ("de-DE-u-CO-PhoneBk", "de-DE-u-co-phonebk"),  # extensions are lower case
        ("en-x-GB", "en-x-gb"),
    ],
)
def test_canonical_locales_write_one_locale_one_way(tag: str, expected: str) -> None:
    assert canonical_locale(tag) == expected
    assert canonical_locale(expected) == expected


def test_checked_locales_come_back_canonical() -> None:
    assert checked_locale("de_de") == "de-DE"
    assert checked_locale("zh-hant-tw") == "zh-Hant-TW"


@pytest.mark.parametrize("tag", ["English", "", "en GB", "en-GB\n", "e", "en-"])
def test_checked_locales_must_be_language_tags(tag: str) -> None:
    with pytest.raises(ValueError, match="locale must be a BCP 47 language tag"):
        checked_locale(tag)


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        (b'\xef\xbb\xbf<html lang="de-DE"><body>', "de-DE"),
        (
            b'<html><head><script>document.write("<html lang=fr>")</script>'
            b'<meta http-equiv="content-language" content="it"></head>',
            "it",
        ),
        (
            b'<html><head><noscript><img src="/pixel.gif"></noscript>'
            b'<meta http-equiv="Content-Language" content="sv-SE"></head>',
            "sv-SE",
        ),
        (b'<html><body><div lang="de">Hallo</div><meta http-equiv="content-language"', None),
        (b'<div lang="de"><html lang="de">', None),
        (b'<html lang="de-DE"', None),
        (b"", None),
    ],
)
def test_html_language_reads_only_what_comes_before_the_body(
    markup: bytes, expected: str | None
) -> None:
    assert html_language(markup) == expected
