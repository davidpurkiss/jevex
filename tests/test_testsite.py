import json
from collections import Counter
from pathlib import Path

import pytest

from jevex.testsite import build, digest, generate, render
from jevex.testsite.schemas import Listing, VehicleSpec

SCHEMAS = {"VehicleSpec": VehicleSpec, "Listing": Listing}


def test_same_seed_same_site_different_seed_different_site() -> None:
    assert digest(render(generate(42))) == digest(render(generate(42)))
    assert digest(render(generate(42))) != digest(render(generate(7)))


def test_pages_dont_change_when_the_dataset_grows() -> None:
    small = {p.path: p.html for p in render(generate(42, n_models=4, n_listings=12))}
    large = {p.path: p.html for p in render(generate(42, n_models=4, n_listings=24))}
    for path, html in small.items():
        if path.startswith("specs/"):
            assert large[path] == html


def test_every_family_is_present() -> None:
    families = Counter(p.family for p in render(generate(42)))
    assert set(families) == {"table", "kv", "prose", "grid", "listing"}
    assert families["grid"] >= 2


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
    assert on_disk == listed | {"index.html"}


def test_build_refuses_a_foreign_truth_json_and_keeps_its_files(tmp_path: Path) -> None:
    (tmp_path / "truth.json").write_text('{"records": []}')
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "report.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="refusing"):
        build(42, tmp_path)
    assert (tmp_path / "docs" / "report.pdf").exists()
    assert (tmp_path / "truth.json").read_text() == '{"records": []}'


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
    for page in manifest["pages"]:
        assert (tmp_path / page["path"]).is_file()
    assert on_disk["digest"] == digest(render(generate(42)))


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
