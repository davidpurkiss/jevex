import re
from pathlib import Path
from typing import Any

import pytest

from jevex import (
    BoilerplateCleaner,
    Document,
    DomLocation,
    EmbeddedData,
    EmbeddedDataReader,
    StructuredBlob,
)
from jevex.structured import schema_type, unflatten_devalue

FIXTURES = Path(__file__).parent / "fixtures" / "structured"


def page(body: str, head: str = "", url: str | None = None) -> Document:
    return Document.from_bytes(
        f"<!DOCTYPE html><html><head>{head}</head><body>{body}</body></html>".encode(),
        content_type="text/html",
        url=url,
    )


def read(body: str, head: str = "", url: str | None = None, **kwargs: Any) -> EmbeddedData:
    return EmbeddedDataReader(**kwargs).read(page(body, head, url))


def fixture(name: str, url: str | None = None) -> Document:
    return Document.from_path(FIXTURES / name, url=url)


def only(data: EmbeddedData) -> StructuredBlob:
    assert len(data.blobs) == 1, data.blobs
    return data.blobs[0]


def ld(value: str) -> str:
    return f'<script type="application/ld+json">{value}</script>'


# --- JSON-LD ---------------------------------------------------------------------------


def test_dealer_page_json_ld_car_with_offer_inlined() -> None:
    data = EmbeddedDataReader(sources=frozenset({"json_ld"})).read(fixture("dealer_car.html"))

    assert [b.types for b in data.blobs] == [
        ["Organization"],
        ["BreadcrumbList"],
        ["Car", "Product"],
        ["Offer"],
        ["AutoDealer"],
    ]
    car = data.of_type("Car")[0]
    assert car.source == "json_ld"
    assert car.location == DomLocation(dom_path="/html/head/script")
    assert "@context" not in car.data
    assert car.data["mileageFromOdometer"] == {
        "@type": "QuantitativeValue",
        "value": 23450,
        "unitCode": "SMI",
    }
    # A raw newline inside a string is invalid JSON, but common; it is read as-is.
    assert car.data["description"] == "One owner.\nFull service history."
    # The Offer is a separate graph node, referenced by @id. Inlining goes one level, so
    # the Offer's own reference to its seller stays a reference.
    assert car.data["offers"]["price"] == 18995
    assert car.data["offers"]["seller"] == {"@id": "https://example-motors.test/#org"}
    # Referenced nodes are still blobs in their own right.
    assert data.of_type("Offer")[0].data["priceCurrency"] == "GBP"
    # The footer script sits in a different element.
    assert data.of_type("AutoDealer")[0].location.dom_path == "/html/body/footer/script"
    assert data.skipped == []


@pytest.mark.parametrize(
    ("value", "types"),
    [
        ('{"@type": "Vehicle", "name": "Van"}', ["Vehicle"]),
        ('{"@type": "http://schema.org/Car", "name": "Golf"}', ["Car"]),
        ('{"@type": "schema:Product", "name": "Shoe"}', ["Product"]),
        ('{"@type": ["https://schema.org/Offer", ""], "price": "9"}', ["Offer"]),
        ('{"name": "untyped"}', []),
    ],
)
def test_json_ld_types_are_short_schema_org_names(value: str, types: list[str]) -> None:
    assert only(read("", ld(value))).types == types


def test_json_ld_top_level_array_gives_one_blob_per_node() -> None:
    data = read("", ld('[{"@type": "Product", "name": "A"}, 3, {"@type": "Offer", "price": 1}]'))
    assert [(b.types, b.data) for b in data.blobs] == [
        (["Product"], {"@type": "Product", "name": "A"}),
        (["Offer"], {"@type": "Offer", "price": 1}),
    ]


def test_json_ld_references_inline_one_level_deep() -> None:
    data = read(
        "",
        ld(
            '{"@graph": ['
            '{"@type": "WebPage", "@id": "#page", "mainEntity": {"@id": "#car"}},'
            '{"@type": "Car", "@id": "#car", "mainEntityOfPage": {"@id": "#page"},'
            ' "offers": {"@id": "#elsewhere"}, "self": {"@id": "#car"}}]}'
        ),
    )
    web_page, car = data.blobs
    assert web_page.data["mainEntity"]["@type"] == "Car"
    assert web_page.data["mainEntity"]["mainEntityOfPage"] == {"@id": "#page"}
    assert car.data["mainEntityOfPage"]["mainEntity"] == {"@id": "#car"}
    # Unknown ids and references to the node itself stay as they are.
    assert car.data["offers"] == {"@id": "#elsewhere"}
    assert car.data["self"] == {"@id": "#car"}


def test_json_ld_references_resolve_across_scripts() -> None:
    data = read(
        "",
        ld('{"@type": "Product", "offers": {"@id": "#o"}}')
        + ld('{"@type": "Offer", "@id": "#o", "price": "5"}'),
    )
    assert data.blobs[0].data["offers"] == {"@type": "Offer", "@id": "#o", "price": "5"}


@pytest.mark.parametrize(
    "wrapped",
    [
        '<!-- {"@type": "Product", "name": "A"} -->',
        '//<![CDATA[\n{"@type": "Product", "name": "A"}\n//]]>',
        '/*<![CDATA[*/ {"@type": "Product", "name": "A"} /*]]>*/',
        '{"@type": "Product", "name": "A",};',
        "{'@type': 'Product', name: 'A'} // trailing comment",
    ],
)
def test_json_ld_tolerates_wrappers_and_sloppy_json(wrapped: str) -> None:
    assert only(read("", ld(wrapped))).data == {"@type": "Product", "name": "A"}


def test_invalid_json_ld_is_skipped_with_the_reason() -> None:
    data = read("", ld('{"@type": "Product", "name": ') + ld('{"@type": "Offer"}'))

    assert [b.types for b in data.blobs] == [["Offer"]]
    (skipped,) = data.skipped
    assert skipped.source == "json_ld"
    assert skipped.reason == "invalid JSON-LD: expected a value at offset 28"
    assert skipped.location.dom_path == "/html/head/script[1]"


def test_json_ld_with_trailing_junk_is_skipped() -> None:
    data = read("", ld('{"@type": "Product"} {"@type": "Offer"}'))
    assert data.blobs == []
    assert data.skipped[0].reason == "invalid JSON-LD: unexpected text after the value at offset 21"


def test_json_ld_nested_too_deeply_is_skipped() -> None:
    deep = "[" * 5000 + "]" * 5000
    data = read("", ld(f'{{"@type": "Product", "a": {deep}}}') + ld('{"@type": "Offer"}'))
    assert [b.types for b in data.blobs] == [["Offer"]]
    assert data.skipped[0].reason == "invalid JSON-LD: nested too deeply"


def test_empty_json_ld_script_is_skipped() -> None:
    assert read("", ld("  ")).skipped[0].reason.startswith("invalid JSON-LD: expected a value")


# --- Microdata -------------------------------------------------------------------------


def test_shop_microdata_product_with_nested_items() -> None:
    data = EmbeddedDataReader(sources=frozenset({"microdata"})).read(
        fixture("shop_microdata.html", url="https://ignored.test/")
    )

    product = only(data)
    assert product.source == "microdata"
    assert product.types == ["Product"]
    assert product.location.dom_path == "/html/body/div[1]"
    assert product.data == {
        "@type": "Product",
        # URLs resolve against <base href>, which itself resolves against the page URL.
        "@id": "https://shop.test/products/#trail-runner-2",
        "sku": "TR2-BLK-42",
        "image": "https://shop.test/products/trail-runner-2.jpg",
        "name": "Trail Runner 2",
        "description": "A light trail shoe.",
        "brand": {"@type": "Brand", "name": "Fleet"},
        "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.6", "reviewCount": "128"},
        # Unclosed <li>s are separate offers, as in a browser; repeats become a list.
        "offers": [
            {
                "@type": "Offer",
                "price": "89.00",
                "priceCurrency": "GBP",
                "availability": "https://schema.org/InStock",
            },
            {
                "@type": "Offer",
                "price": "94.00",
                "priceCurrency": "GBP",
                "priceValidUntil": "2026-12-31",
            },
        ],
    }


@pytest.mark.parametrize(
    ("element", "value"),
    [
        ('<meta itemprop="p" content="c">', "c"),
        ('<span itemprop="p" content="c">shown</span>', "c"),
        ('<img itemprop="p" src="/a.jpg">', "https://ex.test/a.jpg"),
        ('<a itemprop="p" href="b">link</a>', "https://ex.test/b"),
        ('<object itemprop="p" data="c.pdf"></object>', "https://ex.test/c.pdf"),
        ('<data itemprop="p" value="42">forty-two</data>', "42"),
        ('<meter itemprop="p" value="0.5">half</meter>', "0.5"),
        ('<time itemprop="p" datetime="2026-09-29">today</time>', "2026-09-29"),
        ("<time itemprop='p'>today</time>", "today"),
        ("<div itemprop='p'> a <b>b</b>\n c <script>x()</script></div>", "a b c"),
        ('<a itemprop="p">no href</a>', "no href"),
    ],
)
def test_microdata_property_values(element: str, value: str) -> None:
    body = f"<div itemscope>{element}</div>"
    assert only(read(body, url="https://ex.test/")).data == {"p": value}


def test_microdata_relative_urls_stay_relative_without_a_page_url() -> None:
    assert only(read('<div itemscope><a itemprop="u" href="/x">x</a></div>')).data == {"u": "/x"}


def test_microdata_multiple_names_types_and_itemref() -> None:
    body = (
        '<div itemscope itemtype="https://schema.org/Car https://schema.org/Product"'
        ' itemref="price missing">'
        '<span itemprop="name model">Golf</span></div>'
        '<p id="price" itemprop="price">18995</p>'
        '<span itemprop="orphan">not in any item</span>'
    )
    blob = only(read(body))
    assert blob.types == ["Car", "Product"]
    assert blob.data == {
        "@type": ["Car", "Product"],
        "name": "Golf",
        "model": "Golf",
        "price": "18995",
    }


def test_microdata_itemref_cycles_stop() -> None:
    body = (
        '<div id="a" itemscope itemref="b"><span itemprop="x">1</span></div>'
        '<div id="b" itemprop="child" itemscope itemref="a"><span itemprop="y">2</span></div>'
    )
    assert only(read(body)).data == {"x": "1", "child": {"y": "2"}}


def test_microdata_item_nested_without_itemprop_is_its_own_item() -> None:
    body = (
        '<div itemscope itemtype="https://schema.org/ItemList"><span itemprop="name">Cars</span>'
        '<div itemscope itemtype="https://schema.org/Car"><span itemprop="name">Golf</span></div>'
        "</div>"
    )
    data = read(body)
    assert [b.data for b in data.blobs] == [
        {"@type": "ItemList", "name": "Cars"},
        {"@type": "Car", "name": "Golf"},
    ]


def test_microdata_nested_too_deeply_is_skipped() -> None:
    nested = '<div itemprop="p" itemscope>' * 3000 + "</div>" * 3000
    data = read(f"<div itemscope>{nested}</div><div itemscope><b itemprop='a'>1</b></div>")
    assert [b.data for b in data.blobs] == [{"a": "1"}]
    assert data.skipped[0].reason == "items nested too deeply"
    assert data.skipped[0].location.dom_path == "/html/body/div[1]"


# --- RDFa ------------------------------------------------------------------------------


def test_shop_rdfa_review_and_page_level_properties() -> None:
    data = EmbeddedDataReader(sources=frozenset({"rdfa"})).read(fixture("shop_microdata.html"))
    assert [(b.types, b.data) for b in data.blobs] == [
        (
            ["Review"],
            {
                "@type": "Review",
                "reviewBody": "Great grip on wet rock.",
                "author": {"@type": "Person", "name": "Sam"},
                "reviewRating": {"@type": "Rating", "ratingValue": "5"},
            },
        )
    ]


def test_open_graph_properties_form_one_untyped_blob() -> None:
    data = EmbeddedDataReader(sources=frozenset({"rdfa"})).read(fixture("dealer_car.html"))
    blob = only(data)
    assert blob.types == []
    assert blob.location.dom_path == "/"
    assert blob.data == {
        "og:type": "product",
        "og:title": "2021 Volkswagen Golf 1.5 TSI Life",
        "product:price:amount": "18995.00",
        "product:price:currency": "GBP",
    }


def test_rdfa_values_prefixes_and_resources() -> None:
    body = (
        '<div vocab="https://schema.org/" typeof="Car" resource="#golf">'
        '<span property="schema:name">Golf</span>'
        '<a property="url" href="/golf">see</a>'
        '<span property="sameAs" resource="https://wiki.test/golf">wiki</span>'
        '<time property="dateVehicleFirstRegistered" datetime="2021-03-01">March</time>'
        '<span property="color">Red</span><span property="color">Blue</span>'
        "</div>"
    )
    assert only(read(body, url="https://cars.test/list")).data == {
        "@type": "Car",
        "@id": "https://cars.test/list#golf",
        "name": "Golf",
        "url": "https://cars.test/golf",
        "sameAs": "https://wiki.test/golf",
        "dateVehicleFirstRegistered": "2021-03-01",
        "color": ["Red", "Blue"],
    }


# --- App state -------------------------------------------------------------------------


def test_next_data_is_read_as_json() -> None:
    blob = only(EmbeddedDataReader().read(fixture("next_listing.html")))
    assert blob.source == "app_state"
    assert blob.name == "__NEXT_DATA__"
    assert blob.location.dom_path == "/html/body/script"
    assert blob.data["props"]["pageProps"]["listings"][1] == {
        "id": "b2",
        "make": "Skoda",
        "model": "Octavia",
        "price": {"amount": 16900, "currency": "GBP"},
    }


def test_nuxt_3_payload_is_decoded_from_devalue() -> None:
    data = EmbeddedDataReader().read(fixture("nuxt_listing.html"))
    nuxt_data, nuxt = data.blobs
    assert nuxt_data.name == "__NUXT_DATA__"
    assert nuxt_data.data == {
        "data": {
            "van-4": {
                "make": "Ford",
                "model": "Transit Custom",
                "firstRegistered": "2022-03-01T00:00:00.000Z",
                "features": ["Air conditioning"],
            }
        },
        "state": {},
        "once": [],
    }
    # A later ``window.__NUXT__.config = ...`` is a property write, not a new blob.
    assert nuxt.name == "window.__NUXT__"
    assert nuxt.data == {}


def test_nuxt_2_function_payload_is_read_without_running_it() -> None:
    script = (
        "<script>window.__NUXT__=(function(a,b,c,d){b.shared=a;"
        'return {layout:"default",data:[{car:{make:a,model:"Mondeo",year:c,sold:!1,'
        'extras:[d,"Tow bar"],"price":1.2e4,old:void 0,rego:new Date("2019-05-01"),'
        "count:0x1F,note:'it\\'s \\u00a3\\x41',emoji:\"\\u{1F697}\"}}],"
        "fetch:{},error:null,serverRendered:true}}"
        '("Ford",{},2019,"Heated seats"));</script>'
    )
    blob = only(read(script))
    assert blob.name == "window.__NUXT__"
    assert blob.data == {
        "layout": "default",
        "data": [
            {
                "car": {
                    "make": "Ford",
                    "model": "Mondeo",
                    "year": 2019,
                    "sold": False,
                    "extras": ["Heated seats", "Tow bar"],
                    "price": 12000.0,
                    "old": None,
                    "rego": "2019-05-01",
                    "count": 31,
                    "note": "it's £A",
                    "emoji": "\U0001f697",
                }
            }
        ],
        "fetch": {},
        "error": None,
        "serverRendered": True,
    }


@pytest.mark.parametrize(
    ("script", "name", "value"),
    [
        (
            'window.__INITIAL_STATE__ = {"car": {"make": "Audi"}};',
            "window.__INITIAL_STATE__",
            {"car": {"make": "Audi"}},
        ),
        (
            "window['__PRELOADED_STATE__'] = {cars: [{make: 'BMW', doors: 5,}, ,],};",
            "window.__PRELOADED_STATE__",
            {"cars": [{"make": "BMW", "doors": 5}, None]},
        ),
        (
            'window.__INITIAL_STATE__=JSON.parse("{\\"price\\":-3.5,\\"tags\\":[]}")',
            "window.__INITIAL_STATE__",
            {"price": -3.5, "tags": []},
        ),
        (
            "__APOLLO_STATE__ = {/* cache */ 'Car:1': {id: 1, make: `Seat`, neg: -Infinity}}",
            "window.__APOLLO_STATE__",
            {"Car:1": {"id": 1, "make": "Seat", "neg": None}},
        ),
        (
            "self.__NEXT_DATA__ = {page: '/'}; var x = 1;",
            "window.__NEXT_DATA__",
            {"page": "/"},
        ),
        (
            "window.__INITIAL_STATE__ = (function(a){var b={};b.self=b;return {v:a}})(7)",
            "window.__INITIAL_STATE__",
            {"v": 7},
        ),
        (
            "window.__INITIAL_STATE__ = {now: Date.now(), cfg: window.config, 1: 'one'}",
            "window.__INITIAL_STATE__",
            {"now": None, "cfg": None, "1": "one"},
        ),
    ],
)
def test_state_assignments(script: str, name: str, value: Any) -> None:
    blob = only(read(f"<script>{script}</script>"))
    assert (blob.name, blob.data) == (name, value)


def test_several_assignments_in_one_script() -> None:
    data = read(
        "<script>window.__INITIAL_STATE__={a:1};window.__APOLLO_STATE__={b:2};"
        "if (window.__INITIAL_STATE__ == null) {}</script>"
    )
    assert [(b.name, b.data) for b in data.blobs] == [
        ("window.__INITIAL_STATE__", {"a": 1}),
        ("window.__APOLLO_STATE__", {"b": 2}),
    ]


@pytest.mark.parametrize(
    ("script", "reason"),
    [
        (
            "window.__INITIAL_STATE__ = {a: [1, 2};",
            "unreadable window.__INITIAL_STATE__: expected ']' at offset 36",
        ),
        (
            "window.__INITIAL_STATE__ = {[key]: 1}",
            "unreadable window.__INITIAL_STATE__: computed keys are not supported at offset 28",
        ),
        (
            "window.__INITIAL_STATE__ = {a: `x${y}`}",
            "unreadable window.__INITIAL_STATE__: template substitutions are not supported"
            " at offset 33",
        ),
        (
            "window.__INITIAL_STATE__ = 'unterminated",
            "unreadable window.__INITIAL_STATE__: unterminated string at offset 40",
        ),
        (
            "window.__INITIAL_STATE__ = JSON.parse('{nope}')",
            "unreadable window.__INITIAL_STATE__: JSON.parse argument is not JSON (Expecting"
            " property name enclosed in double quotes: line 1 column 2 (char 1)) at offset 47",
        ),
        (
            "window.__INITIAL_STATE__ = (function(a){return {a:a}})",
            "unreadable window.__INITIAL_STATE__: expected a function call at offset 54",
        ),
        (
            "window.__INITIAL_STATE__ = {a: '\\u12'}",
            'unreadable window.__INITIAL_STATE__: bad escape "12\'}" at offset 38',
        ),
    ],
)
def test_unreadable_state_is_skipped_with_the_reason(script: str, reason: str) -> None:
    data = read(f"<script>{script}</script>")
    assert data.blobs == []
    (skipped,) = data.skipped
    assert (skipped.source, skipped.name, skipped.reason) == (
        "app_state",
        "window.__INITIAL_STATE__",
        reason,
    )


def test_deeply_nested_state_is_skipped_not_raised() -> None:
    data = read(f"<script>window.__INITIAL_STATE__ = {'[' * 5000}{']' * 5000}</script>")
    assert data.skipped[0].reason == "unreadable window.__INITIAL_STATE__: nested too deeply"


def test_other_json_scripts_are_app_state_named_by_id() -> None:
    data = read(
        '<script type="application/json" id="config">{"currency": "GBP"}</script>'
        '<script type="application/vnd.shop+json">[1, 2]</script>'
        '<script type="application/json"> </script>'
        '<script type="application/json" id="bad">{</script>'
        "<script>var plain = {not: 'state'};</script>"
        '<script type="text/x-template">{"a": 1}</script>'
    )
    assert [(b.name, b.data) for b in data.blobs] == [
        ("config", {"currency": "GBP"}),
        (None, [1, 2]),
    ]
    assert [(s.name, s.reason) for s in data.skipped] == [
        ("bad", "invalid JSON: expected a property name at offset 1")
    ]


def test_bad_devalue_payload_is_skipped() -> None:
    data = read('<script type="application/json" id="__NUXT_DATA__">{"a": 1}</script>')
    assert data.skipped[0].reason == "invalid JSON: devalue payload must be a non-empty array"


# --- devalue ---------------------------------------------------------------------------


def test_unflatten_devalue_special_values_and_cycles() -> None:
    flat = [
        {"u": -1, "nan": -3, "inf": -4, "ninf": -5, "nz": -6, "holes": 1, "self": 0, "m": 2},
        [3, -2, 3],
        ["Map", 3, 4, 5, 3],
        "x",
        "y",
        7,
    ]
    assert unflatten_devalue(flat) == {
        "u": None,
        "nan": None,
        "inf": None,
        "ninf": None,
        "nz": -0.0,
        "holes": ["x", None, "x"],
        "self": None,
        # A map with a non-string key stays a list of pairs.
        "m": [["x", "y"], [7, "x"]],
    }


@pytest.mark.parametrize(
    ("flat", "value"),
    [
        (-1, None),
        ([["BigInt", "12"]], 12),
        ([["null", "a", 1], 5], {"a": 5}),
        ([["RegExp", "a+", "g"]], "a+"),
        ([["Ref"]], None),
    ],
)
def test_unflatten_devalue_values(flat: Any, value: Any) -> None:
    assert unflatten_devalue(flat) == value


@pytest.mark.parametrize(
    ("flat", "message"),
    [
        ([], "devalue payload must be a non-empty array"),
        ({"a": 1}, "devalue payload must be a non-empty array"),
        ([{"a": 5}], "devalue reference 5 is out of range"),
        ([{"a": "1"}], "devalue reference '1' is not an index"),
        ([{"a": -9}], "devalue reference -9 is not a special value"),
        (3, "devalue reference 3 is not a special value"),
    ],
)
def test_unflatten_devalue_rejects_malformed_payloads(flat: Any, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        unflatten_devalue(flat)


# --- data-* attributes -----------------------------------------------------------------


def test_dealer_page_data_attributes_leave_out_plumbing() -> None:
    data = EmbeddedDataReader(sources=frozenset({"data_attributes"})).read(
        fixture("dealer_car.html")
    )
    blob = only(data)
    assert blob.location.dom_path == "/html/body/main/div"
    assert blob.data == {"vehicle-id": "4411", "price": "18995", "mileage": "23450"}


def test_data_attributes_parse_json_and_keep_everything_without_a_noise_filter() -> None:
    body = (
        "<div data-car='{\"make\": \"Kia\"}' data-list='[1,' data-empty='' data-testid='t'"
        " data->x</div><script data-price='1'></script>"
    )
    kept = only(read(body, data_attribute_noise=None)).data
    assert kept == {"car": {"make": "Kia"}, "list": "[1,", "testid": "t"}
    filtered = only(read(body)).data
    assert filtered == {"car": {"make": "Kia"}, "list": "[1,"}


# --- The reader as a whole -------------------------------------------------------------


def test_every_source_on_one_page_in_source_order() -> None:
    data = EmbeddedDataReader().read(fixture("dealer_car.html"))
    assert [b.source for b in data.blobs] == [
        "json_ld",
        "json_ld",
        "json_ld",
        "json_ld",
        "json_ld",
        "rdfa",
        "data_attributes",
    ]
    assert data.from_source("rdfa")[0].data["og:type"] == "product"


def test_reads_the_cleaned_page() -> None:
    """Data scripts survive cleaning even inside a removed footer."""
    cleaned = BoilerplateCleaner().clean(fixture("dealer_car.html"))
    assert b"site-footer" not in cleaned.content
    data = EmbeddedDataReader().read(cleaned)
    assert [b.types[0] for b in data.from_source("json_ld")] == [
        "Organization",
        "BreadcrumbList",
        "Car",
        "Offer",
        "AutoDealer",
    ]


def test_sources_can_be_narrowed_or_emptied() -> None:
    body = '<div itemscope><b itemprop="a">1</b></div><div data-b="2"></div>'
    assert [b.source for b in read(body, sources=frozenset({"data_attributes"})).blobs] == [
        "data_attributes"
    ]
    assert read(body, sources=frozenset()) == EmbeddedData()


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"unknown structured sources \['jsonld'\]"):
        EmbeddedDataReader(sources=frozenset({"jsonld"}))  # pyright: ignore[reportArgumentType]


def test_non_html_documents_have_no_embedded_data() -> None:
    pdf = Document.from_bytes(b"%PDF-1.7 <script type='application/ld+json'>{}</script>")
    assert EmbeddedDataReader().read(pdf) == EmbeddedData()


def test_page_without_embedded_data() -> None:
    assert read("<p>Just <b>text</b></p>") == EmbeddedData()


def test_dom_paths_follow_browser_recovery() -> None:
    body = (
        "<table><tr><td data-a='1'>x<td data-a='2'>y</table>"
        "<p>para<div data-a='3'>block</div>"
        "<dl><dt data-a='4'>k<dd data-a='5'>v</dl>"
        "<select><option data-a='6'>o<option data-a='7'>p</select></i></span>"
        "<br/><div data-a='8'/>"
    )
    paths = [b.location.dom_path for b in read(body).blobs]
    assert paths == [
        "/html/body/table/tr/td[1]",
        "/html/body/table/tr/td[2]",
        "/html/body/div[1]",
        "/html/body/dl/dt",
        "/html/body/dl/dd",
        "/html/body/select/option[1]",
        "/html/body/select/option[2]",
        "/html/body/div[2]",
    ]


def test_blobs_serialise_to_json() -> None:
    blob = only(read("", ld('{"@type": "Car", "name": "Golf"}')))
    assert StructuredBlob.model_validate_json(blob.model_dump_json()) == blob


@pytest.mark.parametrize(
    ("name", "short"),
    [
        ("https://schema.org/Car", "Car"),
        ("http://www.schema.org/Car", "Car"),
        (" schema:Offer ", "Offer"),
        ("og:title", "og:title"),
        ("https://example.test/Car", "https://example.test/Car"),
    ],
)
def test_schema_type(name: str, short: str) -> None:
    assert schema_type(name) == short
