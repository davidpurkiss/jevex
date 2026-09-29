import codecs
import re
from pathlib import Path

import pytest
from pydantic import BaseModel

from jevex import BoilerplateCleaner, CleanStage, Context, Document, Extractor, Field, SchemaSpec
from jevex.extractor import default_pipeline
from jevex.interfaces import Cleaner
from jevex.jev import JevClient
from jevex.testing import FakeJev

FIXTURES = Path(__file__).parent / "fixtures" / "clean"


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")


def html(body: str, head: str = "") -> Document:
    return Document.from_bytes(
        f"<!DOCTYPE html><html><head>{head}</head><body>{body}</body></html>".encode(),
        content_type="text/html",
    )


def clean(body: str, cleaner: BoilerplateCleaner | None = None) -> str:
    """Clean a page and return just its body markup."""
    out = (cleaner or BoilerplateCleaner()).clean(html(body)).content.decode()
    match = re.search(r"<body>(.*)</body>", out, re.DOTALL)
    assert match, out
    return match.group(1)


# Pages modelled on common templates: WordPress with Cookiebot, a Shopify product page
# with OneTrust, a windows-1252 news page with Sourcepoint and Quantcast, and
# books.toscrape.com. Each lists text that must survive and text that must go.
PAGES = {
    "wordpress_post.html": (
        [
            "Our week with the Golf SE L",
            "2 March 2026",
            "1,498&nbsp;cc",
            "0-62 mph in 8.5 s",
            "47.9 mpg",
            "Our test car",
            "Posted in Reviews",
            "Newsletter",
            '<script type="application/ld+json">',
            "<title>Our week with the Golf SE L",
        ],
        [
            "Primary menu",
            "Small cars, big opinions",
            "Previous: Polo long-termer",
            "Proudly powered by WordPress",
            "This website uses cookies",
            "Allow all",
            "wp-block-image img{",
            "ads.js",
            "html5shiv",
        ],
    ),
    "shop_product.html": (
        [
            "Anker PowerCore 10000",
            "&pound;21.99",
            "10,000 mAh",
            "5 V &#x2F; 2.4 A",
            '"@type":"Product"',
            "Free UK delivery over &pound;30",
            "Add to cart",
        ],
        [
            "Catalogue",
            "Home</a> / ",
            "Refund policy",
            "&copy; 2026, Gadget Store",
            "Accept All Cookies",
            "improve your experience",
            "Shopify.shop",
            "youtube-nocookie",
            "--font-body-family",
        ],
    ),
    "news_article.html": (
        [
            "Council approves new cycle lanes",
            "By Sam Reporter",
            "8.4 km",
            "&pound;4.2m",
            "The café on the corner",
            "Bus fares frozen",
            'window.__INITIAL_STATE__={"article":{"id":991',
        ],
        [
            "SP Consent Message",
            "We value your privacy",
            "Weather",
            "masthead-logo",
            "The Daily Example Ltd",
            "googletag",
            ".ad-slot{",
        ],
    ),
    "books_catalogue.html": (
        [
            "A Light in the Attic",
            "&#163;51.77",
            "In stock (22 available)",
            "a897fe39b1053632",
            "It's hard to imagine",
            "Poetry",
        ],
        [
            "We love being scraped!",
            "oscar.init",
            "bootstrap.min.js",
            "Start of product page",
            "lt IE 7",
        ],
    ),
}


@pytest.mark.parametrize("name", sorted(PAGES))
def test_realistic_pages_keep_content_and_lose_boilerplate(name: str) -> None:
    kept, dropped = PAGES[name]
    document = Document.from_path(FIXTURES / name)
    cleaned = BoilerplateCleaner().clean(document)
    encoding = "cp1252" if name == "news_article.html" else "utf-8"
    text = cleaned.content.decode(encoding)
    assert len(cleaned.content) < len(document.content)
    for phrase in kept:
        assert phrase in text, phrase
    for phrase in dropped:
        assert phrase not in text, phrase
    for tag in ("<style", "<nav", "<noscript", "<iframe", "<!--"):
        assert tag not in text.lower()
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", text, re.DOTALL)
    assert len(scripts) == (0 if name == "books_catalogue.html" else 1)
    assert all("@type" in s or "__INITIAL_STATE__" in s for s in scripts)
    assert cleaned.url == document.url
    assert cleaned.content_type == "text/html"


def test_drops_scripts_styles_and_non_content_embeds() -> None:
    body = (
        "<p>kept</p><script>var a = '<p>no</p>';</script><style>p{color:red}</style>"
        "<noscript>enable js</noscript><template><p>tpl</p></template>"
        '<iframe src="/ad"></iframe><object data="x.swf">flash</object><embed src="x.swf">'
        "<p>after</p>"
    )
    assert clean(body) == "<p>kept</p><p>after</p>"


@pytest.mark.parametrize(
    "script",
    [
        '<script type="application/ld+json">{"@type":"Car","name":"Golf"}</script>',
        '<script type="application/json" id="__NEXT_DATA__">{"props":{}}</script>',
        '<script type="application/vnd.example+json; charset=utf-8">{}</script>',
        "<script>window.__NUXT__=(function(a){return {data:[a]}}(1))</script>",
        '<script type="module">window.__INITIAL_STATE__ = {"car": "Golf"};</script>',
        "<script>window.__APOLLO_STATE__ = {};</script>",
    ],
)
def test_keeps_data_scripts_even_inside_boilerplate(script: str) -> None:
    assert clean(f"<p>a</p>{script}") == f"<p>a</p>{script}"
    assert clean(f"<footer><p>(c)</p>{script}</footer><p>b</p>") == f"{script}<p>b</p>"


@pytest.mark.parametrize(
    "script",
    [
        "<script>ga('send', 'pageview');</script>",
        '<script src="/app.js"></script>',
        '<script type="text/x-template"><div>__INITIAL_STATE__</div></script>',
        '<script type="text/javascript">var cfg = {"json": true};</script>',
    ],
)
def test_drops_code_scripts(script: str) -> None:
    assert clean(f"<p>a</p>{script}<p>b</p>") == "<p>a</p><p>b</p>"


def test_data_scripts_can_be_dropped_too() -> None:
    body = '<script type="application/ld+json">{}</script><p>kept</p>'
    assert clean(body, BoilerplateCleaner(keep_data_scripts=False)) == "<p>kept</p>"


def test_scripts_are_copied_when_not_dropped() -> None:
    body = "<script>ga('send');</script><p>kept</p><nav>x</nav>"
    cleaner = BoilerplateCleaner(drop_tags=frozenset({"nav"}))
    assert clean(body, cleaner) == "<script>ga('send');</script><p>kept</p>"


def test_drops_nav_anywhere() -> None:
    body = "<main><nav><a>Home</a></nav><article><nav>Contents</nav><p>kept</p></article></main>"
    assert clean(body) == "<main><article><p>kept</p></article></main>"


def test_drops_page_level_header_and_footer_but_keeps_sectioned_ones() -> None:
    body = (
        "<header>Site logo</header>"
        "<div><header>Also page level</header></div>"
        "<article><header><h1>Title</h1></header><footer>By Jo</footer></article>"
        "<section><footer>Section note</footer></section>"
        "<aside><header>Aside heading</header></aside>"
        "<footer>(c) Site</footer>"
    )
    assert clean(body) == (
        "<div></div>"
        "<article><header><h1>Title</h1></header><footer>By Jo</footer></article>"
        "<section><footer>Section note</footer></section>"
        "<aside><header>Aside heading</header></aside>"
    )


@pytest.mark.parametrize("role", ["navigation", "banner", "contentinfo", " Navigation "])
def test_drops_landmark_roles(role: str) -> None:
    assert clean(f'<div role="{role}"><a>x</a></div><p>kept</p>') == "<p>kept</p>"


@pytest.mark.parametrize(
    "attrs",
    [
        'id="onetrust-consent-sdk"',
        'id="CybotCookiebotDialog"',
        'class="cookie-banner is-visible"',
        'class="cookie_notice"',
        'class="cc-window cc-banner"',
        'id="qc-cmp2-container"',
        'id="sp_message_container_1"',
        'id="didomi-host"',
        'id="usercentrics-root"',
        'class="gdpr-popup"',
        'class="osano-cm-window"',
        'id="truste-consent-track"',
        'aria-label="Cookie consent"',
        'id="cookie-consent-settings"',
        'id="BorlabsCookieBox"',
        'id="moove_gdpr_cookie_info_bar"',
        'class="klaro"',
        'class="cky-consent-container"',
    ],
)
def test_drops_cookie_and_consent_banners(attrs: str) -> None:
    assert clean(f"<div {attrs}><p>We use cookies</p><button>OK</button></div><p>kept</p>") == (
        "<p>kept</p>"
    )


@pytest.mark.parametrize(
    "body",
    [
        '<div class="cookie-recipe"><p>Choc chip cookies</p></div>',
        '<div class="gdpr-guide"><p>What GDPR means</p></div>',
        '<section class="informed-consent"><p>Patient consent</p></section>',
        '<div class="page cookie-consent-active"><p>Whole page</p></div>',
        '<div class="cookieconsent-given"><p>Whole page</p></div>',
        '<a class="cookie-policy-link" href="/cookies">Cookie policy</a>',
        '<article id="cookie-policy"><p>Our policy</p></article>',
        '<div class="trusted-seller">Trusted</div>',
        '<div class="cmp-text"><p>AEM content component</p></div>',
    ],
)
def test_keeps_content_that_only_looks_like_boilerplate(body: str) -> None:
    assert clean(body) == body


def test_keeps_declarative_shadow_dom_templates() -> None:
    body = (
        '<product-card><template shadowrootmode="open"><h2>Golf</h2></template></product-card>'
        '<x-a><template shadowroot="open"><p>legacy</p></template></x-a>'
        "<template><p>inert</p></template>"
    )
    assert clean(body) == body.replace("<template><p>inert</p></template>", "")


def test_body_with_consent_flag_class_is_kept() -> None:
    document = Document.from_bytes(
        b'<html><body class="gdpr-active cookies-not-set"><p>kept</p><nav>x</nav></body></html>'
    )
    out = BoilerplateCleaner().clean(document).content
    assert out == b'<html><body class="gdpr-active cookies-not-set"><p>kept</p></body></html>'


def test_copies_kept_markup_verbatim() -> None:
    body = (
        "<p class=x data-a='1'>A &amp; B &pound;5 &#163;5 &#x2F; &nbsp;</p>"
        "<br><br/><img src=a.png><svg><path d='M0'/></svg><?php echo 1 ?>"
        "<!-- tracking pixel -->"
    )
    assert clean(body) == body.replace("<!-- tracking pixel -->", "")


def test_skipped_subtree_ends_when_an_ancestor_closes() -> None:
    # The <nav> and its <ul> are never closed; </div> closes them, as in a browser.
    body = "<div><nav><ul><li>Home<li>Shop</div><p>kept</p>"
    assert clean(body) == "<div></div><p>kept</p>"


def test_nested_same_tag_inside_a_skipped_subtree() -> None:
    body = '<div class="cookie-banner"><div><div>inner</div></div>still banner</div><p>kept</p>'
    assert clean(body) == "<p>kept</p>"


def test_script_text_that_looks_like_markup_does_not_end_the_skip() -> None:
    body = "<script>document.write('</div><p>x</p>')</script><p>kept</p>"
    assert clean(body) == "<p>kept</p>"


def test_stray_end_tags_are_kept() -> None:
    assert clean("<p>a</span></p>") == "<p>a</span></p>"


def test_unclosed_boilerplate_runs_to_the_end_of_its_parent() -> None:
    document = Document.from_bytes(b"<html><body><p>kept</p><footer>(c) 2026")
    assert BoilerplateCleaner().clean(document).content == b"<html><body><p>kept</p>"


def test_non_html_documents_are_returned_unchanged() -> None:
    pdf = Document.from_bytes(b"%PDF-1.7 <nav>not html</nav>")
    assert BoilerplateCleaner().clean(pdf) is pdf


def test_html_with_nothing_to_remove_is_returned_unchanged() -> None:
    document = html("<p>just content</p>")
    assert BoilerplateCleaner().clean(document) is document


@pytest.mark.parametrize(
    ("encoding", "declaration"),
    [
        ("iso-8859-1", '<meta charset="iso-8859-1">'),
        ("cp1252", '<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">'),
        ("shift_jis", "<meta charset=Shift_JIS>"),
    ],
)
def test_keeps_the_declared_charset(encoding: str, declaration: str) -> None:
    source = f"<html><head>{declaration}</head><body><nav>x</nav>café ¢ ×</body></html>"
    if encoding == "shift_jis":
        source = source.replace("café ¢ ×", "日本語")
    document = Document.from_bytes(source.encode(encoding))
    cleaned = BoilerplateCleaner().clean(document)
    assert cleaned.content == source.replace("<nav>x</nav>", "").encode(encoding)


@pytest.mark.parametrize("encoding", ["utf-16-le", "utf-16-be"])
def test_keeps_a_utf16_document_in_its_byte_order(encoding: str) -> None:
    source = "\ufeff<html><body><nav>x</nav>café</body></html>"
    document = Document.from_bytes(source.encode(encoding), content_type="text/html")
    cleaned = BoilerplateCleaner().clean(document)
    assert cleaned.content == "\ufeff<html><body>café</body></html>".encode(encoding)


def test_keeps_a_utf8_bom() -> None:
    document = Document.from_bytes(
        b"\xef\xbb\xbf<html><body><nav>x</nav>caf\xc3\xa9</body></html>", content_type="text/html"
    )
    cleaned = BoilerplateCleaner().clean(document)
    assert cleaned.content == b"\xef\xbb\xbf<html><body>caf\xc3\xa9</body></html>"


def test_undecodable_bytes_survive_untouched() -> None:
    document = Document.from_bytes(b"<html><body>\xff\xfe caf\xe9<nav>x</nav></body></html>")
    cleaned = BoilerplateCleaner().clean(document)
    assert cleaned.content == b"<html><body>\xff\xfe caf\xe9</body></html>"


@pytest.mark.parametrize("label", ["x-made-up", "utf-16", "utf-32", "utf-7", "hz", "cp037"])
def test_unusable_meta_charsets_fall_back_to_utf8(label: str) -> None:
    head = f'<head><meta charset="{label}"></head>'
    source = f"<html>{head}<body><nav>x</nav>c+b-afé~{{</body></html>"
    cleaned = BoilerplateCleaner().clean(Document.from_bytes(source.encode()))
    assert cleaned.content == source.replace("<nav>x</nav>", "").encode()


def test_bytes_invalid_in_the_declared_charset_fall_back_to_utf8() -> None:
    # A truncated ISO-2022-JP escape sequence can't be decoded, so the page is read as
    # UTF-8 with the stray bytes kept as they are.
    content = b'<html><head><meta charset="iso-2022-jp"></head><body>\x1b$B<nav>x</nav></body>'
    cleaned = BoilerplateCleaner().clean(Document.from_bytes(content))
    assert cleaned.content == content.replace(b"<nav>x</nav>", b"")


@pytest.mark.parametrize(
    "content",
    [
        "<html><body><nav>x</nav></body></html>".encode("utf-16") + b"\x00",
        codecs.BOM_UTF16_LE + b"\x00\xd8<\x00p\x00>\x00",
    ],
    ids=["odd-trailing-byte", "lone-surrogate"],
)
def test_malformed_utf16_passes_through_without_raising(content: bytes) -> None:
    document = Document.from_bytes(content, content_type="text/html")
    assert BoilerplateCleaner().clean(document).content == content


def test_configuration_widens_or_narrows_what_is_dropped() -> None:
    body = '<nav>menu</nav><form>search</form><div class="cookie-banner">c</div><p>kept</p>'
    wider = BoilerplateCleaner(drop_tags=frozenset({"nav", "form"}))
    assert clean(body, wider) == "<p>kept</p>"
    narrower = BoilerplateCleaner(drop_tags=frozenset(), pattern=None)
    assert clean(body, narrower) == body


def test_default_cleaner_satisfies_the_protocol() -> None:
    assert isinstance(BoilerplateCleaner(), Cleaner)


def context(document: Document) -> Context:
    return Context.create(document, [SchemaSpec.from_model(Car)], JevClient(FakeJev()))


async def test_stage_replaces_the_context_document() -> None:
    ctx = context(html("<nav>menu</nav><p>Golf</p>"))
    await CleanStage().run(ctx)
    assert b"menu" not in ctx.document.content
    assert b"<p>Golf</p>" in ctx.document.content


async def test_stage_runs_a_custom_cleaner() -> None:
    class Upper:
        def clean(self, document: Document) -> Document:
            return document.model_copy(update={"content": document.content.upper()})

    ctx = context(html("<p>golf</p>"))
    await CleanStage(Upper()).run(ctx)
    assert b"<P>GOLF</P>" in ctx.document.content


async def test_stage_does_not_swallow_cleaner_errors() -> None:
    class Broken:
        def clean(self, document: Document) -> Document:
            raise ValueError("bad markup")

    with pytest.raises(ValueError, match="bad markup"):
        await CleanStage(Broken()).run(context(html("<p>golf</p>")))


def test_default_pipeline_starts_with_clean() -> None:
    assert default_pipeline().names[0] == "clean"


async def test_extractor_cleans_by_default() -> None:
    fake = FakeJev()
    async with Extractor([Car], jev=JevClient(fake)) as ex:
        result = await ex.extract(html("<nav>menu</nav><p>Golf</p>"))
    assert "clean" in result.meta.timings
    # The document gate is the first stage to read the page, and sees it cleaned.
    assert [call.state for call in fake.calls] == ["Golf"]
