import json
from collections import Counter
from pathlib import Path

from jevex.testsite import build, digest, generate, render
from jevex.testsite.dataset import MAKES
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


def test_makes_are_fictional_and_values_plausible() -> None:
    data = generate(42)
    assert {m.make for m in data.models} <= set(MAKES)
    for v in data.variants:
        assert 3.0 <= v.zero_to_62_s <= 14.0
        assert (v.engine_size_cc is None) == (v.fuel_type == "ev")


def test_build_writes_the_site_and_truth(tmp_path: Path) -> None:
    manifest = build(42, tmp_path)
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "truth.json").exists()
    on_disk = json.loads((tmp_path / "truth.json").read_text())
    assert on_disk == json.loads(json.dumps(manifest))
    for page in manifest["pages"]:
        assert (tmp_path / page["path"]).is_file()
    assert on_disk["digest"] == digest(render(generate(42)))
