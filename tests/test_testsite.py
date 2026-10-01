import io
import json
import threading
import urllib.request
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from jevex import testsite
from jevex.document import sniff_content_type
from jevex.testsite import (
    DEFAULT_WAVES,
    FAMILIES,
    Dataset,
    Page,
    build,
    digest,
    drawing,
    generate,
    render,
    server,
)
from jevex.testsite.schemas import Listing, VehicleSpec

SCHEMAS = {"VehicleSpec": VehicleSpec, "Listing": Listing}


def test_same_seed_same_site_different_seed_different_site() -> None:
    first = digest(render(generate(42)))
    drawing._png.cache_clear()  # pyright: ignore[reportPrivateUsage]
    drawing._scanned_pdf.cache_clear()  # pyright: ignore[reportPrivateUsage]
    assert digest(render(generate(42))) == first  # rasterised again, not from the cache
    assert digest(render(generate(42))) != digest(render(generate(7)))


def test_pages_dont_change_when_the_dataset_grows() -> None:
    small = {p.path: p.content for p in render(generate(42, n_models=4, n_listings=12))}
    large = {p.path: p.content for p in render(generate(42, n_models=4, n_listings=24))}
    model_pages = [p for p in small if not p.startswith("used/")]
    assert {p.split("/")[0] for p in model_pages} == {"specs", "brochures", "scans", "infographics"}
    for path in model_pages:
        assert large[path] == small[path]


def test_every_family_is_present() -> None:
    pages = render(generate(42))
    families = Counter(p.family for p in pages)
    assert set(families) == {
        *("table", "kv", "prose", "grid", "listing"),
        *("pdf", "scanned", "infographic"),
    }
    assert families["grid"] >= 2
    assert families["pdf"] == families["scanned"] == families["infographic"] == 16
    types = {p.family: p.content_type for p in pages}
    assert types["pdf"] == types["scanned"] == "application/pdf"
    assert types["infographic"] == "image/png"
    assert {types[f] for f in ("table", "kv", "prose", "grid", "listing")} == {"text/html"}
    for page in pages:
        assert sniff_content_type(page.content) == page.content_type
        assert page.truth()["content_type"] == page.content_type


def test_html_is_only_for_html_pages() -> None:
    pages = render(generate(42, n_models=1, n_listings=0))
    assert next(p for p in pages if p.family == "table").html.startswith("<!doctype html>")
    pdf = next(p for p in pages if p.family == "pdf")
    with pytest.raises(ValueError, match="is application/pdf, not HTML"):
        _ = pdf.html


def test_spec_sheet_pdfs_have_a_text_layer_with_every_trim() -> None:
    pdfium = pytest.importorskip("pypdfium2")
    data = generate(42)
    pages = [p for p in render(data) if p.family == "pdf"]
    for model, page in zip(data.models, pages, strict=True):
        assert page.path == f"brochures/{model.slug}-spec-sheet.pdf"
        assert [r["entity"] for r in page.records] == [v.trim for v in model.variants]
        pdf = pdfium.PdfDocument(page.content)
        assert len(pdf) == 1
        assert pdf[0].get_size() == (595, 842)
        text = pdf[0].get_textpage().get_text_range()
        assert f"{model.make} {model.name}" in text
        for v in model.variants:
            assert v.trim in text
            assert f"{int(v.price_gbp):,}" in text


def test_scanned_pdfs_are_one_image_and_no_text() -> None:
    pdfium = pytest.importorskip("pypdfium2")
    scans = [p for p in render(generate(42)) if p.family == "scanned"]
    assert len({p.content for p in scans}) == len(scans)
    for page in scans:
        pdf = pdfium.PdfDocument(page.content)
        assert len(pdf) == 1
        assert pdf[0].get_textpage().get_text_range().strip() == ""
        objects = list(pdf[0].get_objects())
        assert [o.type for o in objects] == [pdfium.raw.FPDF_PAGEOBJ_IMAGE]
        assert b"/Filter /DCTDecode" in page.content


def test_infographics_are_pngs_of_one_trim() -> None:
    image_module = pytest.importorskip("PIL.Image")
    data = generate(42)
    pages = [p for p in render(data) if p.family == "infographic"]
    trims: set[str] = set()
    for model, page in zip(data.models, pages, strict=True):
        [record] = page.records
        assert record["entity"] == "document"
        trim = record["values"]["trim"]
        assert trim in {v.trim for v in model.variants}
        assert page.path == f"infographics/{model.slug}-{trim.lower().replace(' ', '-')}.png"
        trims.add(trim)
        with image_module.open(io.BytesIO(page.content)) as image:
            assert image.size == (1200, 888)
    assert len(trims) > 1  # not always the same trim


def test_json_ld_on_some_prose_pages_only() -> None:
    prose = [p for p in render(generate(42)) if p.family == "prose"]
    with_ld = [p for p in prose if p.json_ld]
    assert 0 < len(with_ld) < len(prose)
    for page in prose:
        assert ("application/ld+json" in page.html) is page.json_ld
    blob = with_ld[0].html.split('<script type="application/ld+json">')[1].split("</script>")[0]
    data = json.loads(blob)
    assert data["@type"] == "Car"
    assert data["model"] == with_ld[0].records[0]["values"]["model"]


def test_ground_truth_validates_against_the_schemas() -> None:
    for page in render(generate(42)):
        assert page.records
        for record in page.records:
            SCHEMAS[page.schema].model_validate(record["values"])


def test_multi_entity_pages_have_one_record_per_entity() -> None:
    pages = render(generate(42))
    table = next(p for p in pages if p.family == "table")
    trims = [r["entity"] for r in table.records]
    assert len(trims) >= 2
    for trim in trims:
        assert f"<th>{trim}</th>" in table.html
    grid = next(p for p in pages if p.family == "grid")
    assert len(grid.records) == 12
    assert grid.html.count('class="listing"') == 12


def test_truth_values_appear_on_the_page() -> None:
    for page in render(generate(42)):
        if page.content_type != "text/html":
            continue  # test_testsite_truth reads PDFs and drawings
        for record in page.records:
            values = record["values"]
            assert values["model"] in page.html
            if page.family == "prose":
                assert str(values["zero_to_62_s"]) in page.html
            if page.schema == "Listing":
                assert f"{values['mileage_miles']:,}" in page.html


REAL_NAMES = {
    # A sanity denylist of real makes and models that invented names must avoid.
    "vento",
    "strada",
    "sable",
    "nimbus",
    "golf",
    "polo",
    "focus",
    "fiesta",
    "civic",
    "corolla",
    "astra",
    "corsa",
    "clio",
    "fenwick",
    "ford",
    "fiat",
    "kia",
    "mini",
}


def test_names_are_fictional() -> None:
    data = generate(42)
    names = {m.make.lower() for m in data.models} | {m.name.lower() for m in data.models}
    assert not names & REAL_NAMES


def test_values_are_plausible_and_trims_step_up() -> None:
    data = generate(42)
    for model in data.models:
        prices = [v.price_gbp for v in model.variants]
        powers = [v.power_kw for v in model.variants]
        assert prices == sorted(prices)
        assert powers == sorted(powers)
        assert len({v.seats for v in model.variants}) == 1
    for v in data.variants:
        assert 3.0 <= v.zero_to_62_s <= 14.0
        assert v.top_speed_mph <= 155
        assert (v.engine_size_cc is None) == (v.fuel_type == "ev")
        assert (v.co2_g_km == 0) == (v.fuel_type == "ev")
        if v.fuel_type in ("hybrid", "phev", "ev"):
            assert v.automatic


@pytest.mark.parametrize(("n_models", "n_listings"), [(0, 5), (17, 5), (4, -1)])
def test_generate_rejects_bad_sizes(n_models: int, n_listings: int) -> None:
    with pytest.raises(ValueError, match=r"n_models|n_listings"):
        generate(42, n_models=n_models, n_listings=n_listings)


def test_rebuild_replaces_an_earlier_build(tmp_path: Path) -> None:
    build(42, tmp_path)
    build(7, tmp_path)
    on_disk = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.html")}
    listed = {p["path"] for p in json.loads((tmp_path / "truth.json").read_text())["pages"]}
    assert on_disk == {p for p in listed if p.endswith(".html")} | {"index.html"}
    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert files == listed | {"index.html", "truth.json"}


def test_build_refuses_a_foreign_truth_json_and_keeps_its_files(tmp_path: Path) -> None:
    (tmp_path / "truth.json").write_text('{"records": []}')
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="refusing"):
        build(42, tmp_path)
    assert (tmp_path / "docs" / "report.pdf").exists()
    assert (tmp_path / "truth.json").read_text() == '{"records": []}'


def test_a_failed_render_keeps_the_earlier_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build(42, tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def no_pillow(dataset: Dataset, **kwargs: object) -> list[Page]:
        raise ImportError("install jevex[testsite]")

    monkeypatch.setattr(testsite, "render", no_pillow)
    with pytest.raises(ImportError):
        build(7, tmp_path)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_rebuild_keeps_files_it_did_not_write(tmp_path: Path) -> None:
    build(42, tmp_path)
    mine = tmp_path / "specs" / "my-notes.md"
    mine.write_text("keep me")
    build(7, tmp_path)
    assert mine.read_text() == "keep me"


def test_build_refuses_an_unrelated_directory(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="refusing"):
        build(42, tmp_path)
    assert (tmp_path / "notes.txt").exists()


def test_build_writes_the_site_and_truth(tmp_path: Path) -> None:
    manifest = build(42, tmp_path)
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "truth.json").exists()
    on_disk = json.loads((tmp_path / "truth.json").read_text())
    assert on_disk == json.loads(json.dumps(manifest))
    pages = {p.path: p for p in render(generate(42))}
    assert {p["path"] for p in manifest["pages"]} == set(pages)
    for page in manifest["pages"]:
        assert (tmp_path / page["path"]).read_bytes() == pages[page["path"]].content
    assert on_disk["digest"] == digest([pages[p["path"]] for p in manifest["pages"]])


@pytest.mark.parametrize("escape", ["absolute", "dotdot"])
def test_rebuild_never_deletes_outside_the_directory(tmp_path: Path, escape: str) -> None:
    site, victim = tmp_path / "site", tmp_path / "victim.txt"
    site.mkdir()
    victim.write_text("keep")
    path = str(victim) if escape == "absolute" else "../victim.txt"
    manifest = {"seed": 1, "digest": "x", "pages": [{"path": path}]}
    (site / "truth.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="refusing"):
        build(42, site)
    assert victim.read_text() == "keep"


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_build_is_deterministic(tmp_path: Path) -> None:
    build(42, tmp_path / "a")
    drawing._png.cache_clear()  # pyright: ignore[reportPrivateUsage]
    drawing._scanned_pdf.cache_clear()  # pyright: ignore[reportPrivateUsage]
    build(42, tmp_path / "b")
    first, second = _files(tmp_path / "a"), _files(tmp_path / "b")
    assert first == second  # every page, index.html and truth.json, byte for byte


def test_build_lists_pages_wave_by_wave(tmp_path: Path) -> None:
    manifest = build(42, tmp_path)
    assert manifest["waves"] == [list(wave) for wave in DEFAULT_WAVES]
    pages = manifest["pages"]
    assert {p["family"] for p in pages} == set(FAMILIES)
    numbers = {f: n for n, wave in enumerate(DEFAULT_WAVES, start=1) for f in wave}
    assert [p["wave"] for p in pages] == sorted(p["wave"] for p in pages)
    for page in pages:
        assert page["wave"] == numbers[page["family"]]
    # Within a wave, render's order stands.
    rendered = [p.path for p in render(generate(42)) if p.family in DEFAULT_WAVES[0]]
    assert [p["path"] for p in pages if p["wave"] == 1] == rendered


def test_a_custom_schedule_builds_only_its_families(tmp_path: Path) -> None:
    manifest = build(42, tmp_path, waves=[["prose"], ["grid", "table"]])
    pages = manifest["pages"]
    assert manifest["waves"] == [["prose"], ["grid", "table"]]
    variants = len(generate(42).variants)
    assert [p["family"] for p in pages if p["wave"] == 1] == ["prose"] * variants
    assert {p["family"] for p in pages if p["wave"] == 2} == {"grid", "table"}
    assert {p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*.*")} == {
        *("specs", "used", "index.html", "truth.json")
    }
    index = (tmp_path / "index.html").read_text()
    first, second = index.index("<h2>Wave 1: prose</h2>"), index.index("<h2>Wave 2: grid, table")
    assert first < index.index('href="specs/') < second  # prose links under wave 1
    assert second < index.index("-table.html")
    assert second < index.index("used/page-1")


def test_a_bad_schedule_leaves_the_earlier_build(tmp_path: Path) -> None:
    build(42, tmp_path, waves=[["table"]])
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="unknown family 'tables'"):
        build(7, tmp_path, waves=[["tables"]])
    assert _files(tmp_path) == before


def test_render_only_the_families_asked_for() -> None:
    data = generate(42, n_models=2, n_listings=3)
    assert render(data, families=[]) == []
    assert {p.family for p in render(data, families={"kv", "listing"})} == {"kv", "listing"}
    with pytest.raises(ValueError, match=r"unknown template families \['tabel'\]"):
        render(data, families=["table", "tabel"])


@pytest.fixture
def site(tmp_path: Path) -> Iterator[str]:
    build(42, tmp_path, waves=[["pdf"], ["listing"]])
    httpd = server(tmp_path, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host!s}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_server_serves_the_build(site: str, tmp_path: Path) -> None:
    with urllib.request.urlopen(f"{site}/") as response:
        assert response.headers["Content-Type"].startswith("text/html")
        assert b"<h2>Wave 1: pdf</h2>" in response.read()
    truth = json.loads((tmp_path / "truth.json").read_text())
    pdf = next(p["path"] for p in truth["pages"] if p["family"] == "pdf")
    with urllib.request.urlopen(f"{site}/{pdf}") as response:
        assert response.headers["Content-Type"] == "application/pdf"
        assert response.read() == (tmp_path / pdf).read_bytes()


def test_server_needs_a_build(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html></html>")
    with pytest.raises(ValueError, match="has no test site; build one first"):
        server(tmp_path, port=0)
