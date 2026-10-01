import importlib
import sys
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from jevex import (
    GeneratorSpec,
    KeyMapping,
    Pack,
    PackError,
    PackManifest,
    StoreError,
    VerifiedExample,
    community_packs,
    diff_packs,
    export_pack,
    import_pack,
    layered_generators,
    load_pack,
)
from jevex import packs as packs_module
from jevex.packs import ENTRY_POINT_GROUP, generator_record
from jevex.store import GeneratorRecord, open_store

FIELD = "Car.zero_to_62_s"


def spec(
    gid: str = "gen-a", field: str = FIELD, regex: str = r"(\d+) secs", **kw: Any
) -> GeneratorSpec:
    return GeneratorSpec.parse(
        {"id": gid, "field": field, "match": {"regex": regex, "group": 1}, **kw}
    )


def mapping(path: str = "$.price", field: str | None = "price", **kw: Any) -> KeyMapping:
    data = {"fingerprint": "fp1", "schema": "Car", "path": path, "field": field, **kw}
    return KeyMapping.model_validate(data)


def ex(eid: str = "ex-1", field: str = FIELD, **kw: Any) -> VerifiedExample:
    data: dict[str, Any] = {
        "id": eid,
        "field": field,
        "statement": "0–62 mph takes 9.1 seconds",
        "value": 9.1,
        "evidence": (17, 20),
        "context": {"heading_trail": ["Performance"]},
        "probability": 0.95,
    }
    return VerifiedExample.model_validate({**data, **kw})


def manifest(**kw: Any) -> PackManifest:
    return PackManifest.model_validate({"name": "cars", "version": "1.0.0", **kw})


def pack(*generators: GeneratorSpec, disables: list[str] | None = None, **kw: Any) -> Pack:
    return Pack(
        manifest=manifest(disables=disables or [], name=kw.pop("name", "cars")),
        generators=list(generators),
        **kw,
    )


def full_pack() -> Pack:
    return Pack(
        manifest=manifest(
            description="Cars sold in the UK",
            schemas=["Car"],
            locales=["en-GB"],
            disables=["gen-old"],
        ),
        generators=[spec("gen-b"), spec("gen-a", scope={"locale": "en-GB"})],
        key_mappings=[
            mapping(),
            mapping("$.sku", None, unsure=True),
            mapping("$.mpg", "mpg", normalisers=["parse_number", {"unit": {"from": "mpg"}}]),
            KeyMapping(fingerprint="fp/../2", schema="Car", path="$.name", field="model"),
        ],
        examples=[ex("ex-2", value="Café", evidence=None), ex()],
    )


def write(p: Pack, directory: Path) -> Path:
    p.write(directory)
    return directory


def contents(p: Pack) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Key mappings and examples without created_at (files don't keep it), sorted."""
    mappings = sorted(
        (m.model_dump(exclude={"created_at"}) for m in p.key_mappings),
        key=lambda m: (m["fingerprint"], m["path"]),
    )
    examples = sorted(
        (e.model_dump(exclude={"created_at"}) for e in p.examples), key=lambda e: e["id"]
    )
    return mappings, examples


# --- files -------------------------------------------------------------------------------


def test_a_pack_round_trips_through_its_directory(tmp_path: Path) -> None:
    original = full_pack()
    written = original.write(tmp_path / "cars")
    names = sorted(str(p.relative_to(tmp_path / "cars")) for p in written)
    hashed = [n for n in names if n.startswith("key_mappings/") and n != "key_mappings/fp1.yaml"]
    assert names == sorted(
        [
            "manifest.yaml",
            "generators/gen-a.yaml",
            "generators/gen-b.yaml",
            "key_mappings/fp1.yaml",
            *hashed,  # "fp/../2" isn't a safe file name: it's hashed
            "examples/Car.zero_to_62_s.yaml",
        ]
    )
    assert len(hashed) == 1
    assert ".." not in hashed[0]

    loaded = Pack.load(tmp_path / "cars")
    assert loaded.path == tmp_path / "cars"
    assert loaded.manifest == original.manifest
    assert [g.id for g in loaded.generators] == ["gen-a", "gen-b"]  # by file name
    assert {g.id: g for g in loaded.generators} == {g.id: g for g in original.generators}
    assert contents(loaded) == contents(original)


def test_the_files_are_readable_yaml(tmp_path: Path) -> None:
    full_pack().write(tmp_path)
    assert (tmp_path / "manifest.yaml").read_text() == (
        "name: cars\n"
        "version: 1.0.0\n"
        "description: Cars sold in the UK\n"
        "schemas:\n- Car\n"
        "locales:\n- en-GB\n"
        "disables:\n- gen-old\n"
    )
    assert (tmp_path / "key_mappings/fp1.yaml").read_text() == (
        "fingerprint: fp1\n"
        "mappings:\n"
        "- schema: Car\n  path: $.mpg\n  field: mpg\n  normalisers:\n  - parse_number\n"
        "  - unit:\n      from: mpg\n"
        "- schema: Car\n  path: $.price\n  field: price\n"
        "- schema: Car\n  path: $.sku\n  field: null\n  unsure: true\n"
    )
    examples = (tmp_path / "examples/Car.zero_to_62_s.yaml").read_text()
    assert examples.startswith("field: Car.zero_to_62_s\nexamples:\n- id: ex-1\n")
    assert "value: Café" in examples  # Unicode stays readable


def test_writing_the_same_state_twice_writes_the_same_bytes(tmp_path: Path) -> None:
    a = full_pack()
    b = a.model_copy(
        update={
            "generators": list(reversed(a.generators)),
            "key_mappings": list(reversed(a.key_mappings)),
            "examples": list(reversed(a.examples)),
        }
    )
    a.write(tmp_path / "a")
    b.write(tmp_path / "b")
    for path in (tmp_path / "a").rglob("*.yaml"):
        assert path.read_bytes() == (tmp_path / "b" / path.relative_to(tmp_path / "a")).read_bytes()


def test_writing_refuses_a_used_directory(tmp_path: Path) -> None:
    (tmp_path / "old.yaml").write_text("x: 1")
    with pytest.raises(FileExistsError, match="isn't an empty directory"):
        full_pack().write(tmp_path)
    (tmp_path / "file").write_text("")
    with pytest.raises(FileExistsError):
        full_pack().write(tmp_path / "file")
    full_pack().write(tmp_path / "new")  # an empty or missing directory is fine


def test_a_minimal_pack_is_only_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "manifest.yaml").write_text("name: tiny\nversion: '0.1'\n")
    loaded = Pack.load(tmp_path)
    assert loaded.name == "tiny"
    assert (loaded.generators, loaded.key_mappings, loaded.examples) == ([], [], [])


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({}, "has no manifest.yaml"),
        ({"manifest.yaml": "name: [unclosed"}, "manifest.yaml isn't valid YAML"),
        (
            {"manifest.yaml": "name: cars\nversion: 1.10\n"},
            "version: .*quote it, as version: '1.1'",
        ),
        (
            {"manifest.yaml": "name: cars\nversion: [1]\n"},
            "version: Input should be a valid string",
        ),
        ({"manifest.yaml": "name: cars\nversion: '1'\nowner: me\n"}, "owner: Extra inputs"),
        ({"manifest.yaml": "name: ../cars\nversion: '1'\n"}, "name: String should match"),
        ({"manifest.yaml": "name: cars\nname: vans\nversion: '1'\n"}, "duplicate key"),
        ({"manifest.yaml": "a: &x [1]\nb: *x\n"}, "isn't valid YAML"),
        (
            {"generators/gen-a.yaml": "id: gen-a\nfield: Car.x\nmatch: {regex: '(a'}\n"},
            "generators/gen-a.yaml",
        ),
        (
            {"key_mappings/fp.yaml": "fingerprint: fp\nmappings:\n- {schema: Car, path: p}\n"},
            r"key_mappings/fp.yaml: mappings.0.field: Field required",
        ),
        (
            {
                "examples/x.yaml": "field: Car.x\nexamples:\n"
                "- {id: e, statement: s, value: 1, at: 2}"
            },
            "examples/x.yaml: examples.0.at: Extra inputs",
        ),
        (
            {
                "examples/a.yaml": "field: Car.a\nexamples:\n- {id: e, statement: s, value: 1}",
                "examples/b.yaml": "field: Car.b\nexamples:\n- {id: e, statement: s, value: 2}",
            },
            r"duplicate example id: \['e'\]",
        ),
    ],
)
def test_bad_pack_files_are_pack_errors_naming_the_file(
    tmp_path: Path, files: dict[str, str], message: str
) -> None:
    if "manifest.yaml" not in files and files:
        files = {"manifest.yaml": "name: cars\nversion: '1'\n", **files}
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    with pytest.raises(PackError, match=message):
        Pack.load(tmp_path)


def test_a_missing_directory_is_a_pack_error(tmp_path: Path) -> None:
    with pytest.raises(PackError, match="no such pack directory"):
        Pack.load(tmp_path / "missing")


def test_a_pack_cant_disable_its_own_generators() -> None:
    with pytest.raises(ValidationError, match=r"disables the pack's own generators: \['gen-a'\]"):
        pack(spec("gen-a"), disables=["gen-a"])


def test_a_pack_holds_only_the_schemas_its_manifest_lists() -> None:
    with pytest.raises(
        ValidationError, match=r"schemas the manifest doesn't list: \['Bus', 'Van'\]"
    ):
        Pack(
            manifest=manifest(schemas=["Car"]),
            generators=[spec(field="Van.price")],
            key_mappings=[mapping()],
            examples=[ex(field="Bus.seats")],
        )


def test_a_nested_models_fields_belong_to_its_top_level_schema() -> None:
    nested = "Car.trims.price"
    p = Pack(
        manifest=manifest(schemas=["Car"]),
        generators=[spec(field=nested)],
        examples=[ex(field=nested)],
    )
    assert ids(p.generators) == ["gen-a"]
    with pytest.raises(ValidationError, match=r"schemas the manifest doesn't list: \['Van'\]"):
        Pack(manifest=manifest(schemas=["Car"]), generators=[spec(field="Van.trims.price")])


def test_duplicates_within_a_pack_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate generator id"):
        pack(spec("gen-a"), spec("gen-a"))
    with pytest.raises(ValidationError, match="duplicate key mapping"):
        Pack(manifest=manifest(), key_mappings=[mapping(), mapping(field="cost")])


# --- layering ----------------------------------------------------------------------------


def ids(specs: list[GeneratorSpec]) -> list[str]:
    return [s.id for s in specs]


def test_the_first_layer_with_an_id_wins() -> None:
    local = [spec("gen-a", regex=r"(\d+) local")]
    project = pack(spec("gen-a", regex=r"(\d+) project"), spec("gen-b"))
    community = pack(spec("gen-b", regex=r"(\d+) community"), spec("gen-c"))
    layered = layered_generators(local, set(), [project, community])
    assert ids(layered) == ["gen-a", "gen-b", "gen-c"]
    assert layered[0].match.regex == r"(\d+) local"
    assert layered[1] == project.generators[1]


def test_the_store_disables_generators_in_every_pack() -> None:
    project = pack(spec("gen-a"), spec("gen-b"))
    community = pack(spec("gen-c"))
    layered = layered_generators([], {"gen-a", "gen-c"}, [project, community])
    assert ids(layered) == ["gen-b"]


def test_a_pack_disables_generators_only_in_the_layers_below_it() -> None:
    project = pack(spec("gen-a"), disables=["gen-b", "gen-local"])
    community = pack(spec("gen-b"), spec("gen-c"), disables=["gen-a"])
    local = [spec("gen-local")]
    layered = layered_generators(local, set(), [project, community])
    # gen-local is above the project pack, and gen-a above the community pack.
    assert ids(layered) == ["gen-local", "gen-a", "gen-c"]


def test_a_shadowed_id_stays_shadowed_even_when_disabled_above() -> None:
    # The project pack disables gen-a; the store's own gen-a still wins, and the
    # community copy stays out.
    layered = layered_generators(
        [spec("gen-a")], set(), [pack(disables=["gen-a"]), pack(spec("gen-a"))]
    )
    assert ids(layered) == ["gen-a"]


# --- installed packs ---------------------------------------------------------------------


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, EntryPoint]]:
    """Installed packs: a package holding its files, a path, and a function."""
    site = tmp_path / "site"
    package = site / "jevex_pack_test_cars"
    full_pack().write(package)
    (package / "__init__.py").write_text("")
    module = site / "jevex_pack_test_helpers.py"
    vans = tmp_path / "vans"
    pack(spec("gen-v", field="Van.price"), name="vans").write(vans)
    module.write_text(
        f"from pathlib import Path\nPATH = {str(vans)!r}\n"
        f"def where():\n    return Path(PATH)\n"
        "NUMBER = 3\n"
        "def broken():\n    raise RuntimeError('no files')\n"
    )
    monkeypatch.setattr(sys, "path", [str(site), *sys.path])
    importlib.invalidate_caches()
    eps = {
        "cars": EntryPoint("cars", "jevex_pack_test_cars", ENTRY_POINT_GROUP),
        "vans": EntryPoint("vans", "jevex_pack_test_helpers:where", ENTRY_POINT_GROUP),
        "vans-path": EntryPoint("vans-path", "jevex_pack_test_helpers:PATH", ENTRY_POINT_GROUP),
    }

    def fake_entry_points(*, group: str) -> list[EntryPoint]:
        assert group == ENTRY_POINT_GROUP
        return list(eps.values())

    monkeypatch.setattr(packs_module, "entry_points", fake_entry_points)
    yield eps
    for name in ("jevex_pack_test_cars", "jevex_pack_test_helpers"):
        sys.modules.pop(name, None)


def test_community_packs_are_the_installed_entry_points(installed: dict[str, EntryPoint]) -> None:
    found = community_packs()
    assert [p.name for p in found] == ["cars", "vans", "vans"]  # by entry point name
    assert [p.name for p in community_packs(["vans", "cars"])] == ["vans", "cars"]
    assert ids(found[0].generators) == ["gen-a", "gen-b"]


def test_an_unknown_installed_pack_is_a_pack_error(installed: dict[str, EntryPoint]) -> None:
    with pytest.raises(PackError, match=r"no installed pack named 'trucks' \(installed: cars"):
        community_packs(["trucks"])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("jevex_pack_test_helpers", "isn't a package directory"),
        ("jevex_pack_test_helpers:NUMBER", "is a int, not a package, a path or a function"),
        ("jevex_pack_test_helpers:broken", "failed to load: no files"),
        ("jevex_pack_no_such_module", "failed to load: No module named"),
    ],
)
def test_a_bad_installed_pack_is_a_pack_error(
    installed: dict[str, EntryPoint], value: str, message: str
) -> None:
    installed["bad"] = EntryPoint("bad", value, ENTRY_POINT_GROUP)
    with pytest.raises(PackError, match=f"installed pack 'bad'.*{message}"):
        community_packs(["bad"])


def test_load_pack_takes_a_pack_a_directory_or_an_installed_name(
    installed: dict[str, EntryPoint], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = full_pack()
    assert load_pack(p) is p
    write(p, tmp_path / "local")
    assert load_pack(tmp_path / "local").path == tmp_path / "local"
    assert load_pack(str(tmp_path / "local")).name == "cars"
    assert load_pack("vans").name == "vans"
    monkeypatch.chdir(tmp_path)
    assert load_pack("local").path == Path("local")  # a directory wins over a name
    with pytest.raises(PackError, match="no such pack directory"):
        load_pack("./trucks")


# --- stores ------------------------------------------------------------------------------


async def test_export_writes_the_stores_state_as_a_pack() -> None:
    store = open_store(":memory:")
    await store.put_generator(generator_record(spec("gen-a", scope={"locale": "en-GB"})))
    await store.put_generator(generator_record(spec("gen-v", field="Van.price")))
    await store.put_generator(generator_record(spec("gen-off")))
    await store.set_generator_enabled("gen-off", False)
    await store.set_generator_enabled("gen-community", False)
    await store.put_key_mapping(mapping())
    await store.put_key_mapping(mapping(schema="Van"))
    await store.add_example(ex())

    everything = await export_pack(store, "all", "2.0", examples=True)
    assert everything.manifest == PackManifest(
        name="all",
        version="2.0",
        schemas=["Car", "Van"],
        locales=["en-GB"],
        disables=["gen-community", "gen-off"],
    )
    assert ids(everything.generators) == ["gen-a", "gen-v"]
    assert len(everything.key_mappings) == 2
    assert [e.id for e in everything.examples] == ["ex-1"]

    cars = await export_pack(store, "cars", "1", schemas=["Car"], locales=["en"], description="d")
    assert (cars.manifest.schemas, cars.manifest.locales) == (["Car"], ["en"])
    assert ids(cars.generators) == ["gen-a"]
    assert [m.schema_name for m in cars.key_mappings] == ["Car"]
    assert cars.examples == []  # only with examples=True
    await store.aclose()


async def test_export_keeps_a_nested_models_state_with_its_top_level_schema() -> None:
    store = open_store(":memory:")
    await store.put_generator(generator_record(spec("gen-t", field="Car.trims.price")))
    await store.put_generator(generator_record(spec("gen-v", field="Van.trims.price")))
    await store.add_example(ex(field="Car.trims.price"))
    everything = await export_pack(store, "all", "1", examples=True)
    assert everything.manifest.schemas == ["Car", "Van"]
    cars = await export_pack(store, "cars", "1", schemas=["Car"], examples=True)
    assert ids(cars.generators) == ["gen-t"]
    assert [e.id for e in cars.examples] == ["ex-1"]
    await store.aclose()


async def test_export_refuses_an_invalid_stored_generator() -> None:
    store = open_store(":memory:")
    await store.put_generator(GeneratorRecord(id="gen-x", field=FIELD, spec={"id": "gen-x"}))
    with pytest.raises(StoreError, match="stored generator 'gen-x' is invalid"):
        await export_pack(store, "x", "1")


async def test_import_puts_a_pack_in_the_local_layer() -> None:
    store = open_store(":memory:")
    await store.put_generator(generator_record(spec("gen-a", regex=r"(\d+) old")))
    await store.set_generator_enabled("gen-a", False)  # pruned before the review
    await store.put_key_mapping(mapping(field="cost"))

    await import_pack(full_pack(), store)
    by_id = {r.id: r for r in await store.generators()}
    assert sorted(by_id) == ["gen-a", "gen-b"]
    assert by_id["gen-a"].spec["match"]["regex"] == r"(\d+) secs"  # replaced and enabled
    assert by_id["gen-a"].scope == {"locale": "en-GB"}
    assert await store.disabled_generator_ids() == {"gen-old"}
    assert {(m.path, m.field) for m in await store.key_mappings("fp1")} == {
        ("$.price", "price"),
        ("$.sku", None),
        ("$.mpg", "mpg"),
    }
    assert {e.id for e in await store.examples()} == {"ex-1", "ex-2"}

    again = open_store(":memory:")
    await import_pack(full_pack(), again, examples=False)
    assert await again.examples() == []


async def test_export_then_import_carries_a_store_over() -> None:
    source = open_store(":memory:")
    await import_pack(full_pack(), source)
    copy = open_store(":memory:")
    await import_pack(await export_pack(source, "copy", "1", examples=True), copy)
    exported = await export_pack(copy, "copy", "1", examples=True)
    assert diff_packs(await export_pack(source, "copy", "1", examples=True), exported).empty


# --- diffs -------------------------------------------------------------------------------


def test_diff_lists_added_removed_and_changed_entries() -> None:
    old = full_pack()
    new = Pack(
        manifest=old.manifest.model_copy(update={"version": "1.1.0", "disables": ["gen-new"]}),
        generators=[spec("gen-a", regex=r"(\d+) seconds"), spec("gen-c")],
        key_mappings=[mapping(), mapping("$.sku", "sku"), mapping("$.colour", "colour")],
        examples=[ex(value=9.2)],
    )
    changes = diff_packs(old, new)
    assert not changes.empty
    assert changes.manifest == {"version": ("1.0.0", "1.1.0")}
    assert ids(changes.generators.added) == ["gen-c"]
    assert ids(changes.generators.removed) == ["gen-b"]
    assert ids(changes.generators.changed) == ["gen-a"]
    assert changes.generators.changed[0].match.regex == r"(\d+) seconds"
    assert [m.path for m in changes.key_mappings.added] == ["$.colour"]
    assert sorted(m.path for m in changes.key_mappings.removed) == ["$.mpg", "$.name"]
    assert [m.path for m in changes.key_mappings.changed] == ["$.sku"]
    assert [e.id for e in changes.examples.removed] == ["ex-2"]
    assert [e.id for e in changes.examples.changed] == ["ex-1"]
    assert (changes.disables.added, changes.disables.removed) == (["gen-new"], ["gen-old"])

    quiet = diff_packs(old, new, manifest=False, examples=False)
    assert quiet.manifest == {}
    assert quiet.examples.empty


def test_creation_times_dont_count_as_changes() -> None:
    old = full_pack()
    new = old.model_copy(
        update={
            "key_mappings": [
                m.model_copy(update={"created_at": m.created_at.replace(year=2000)})
                for m in old.key_mappings
            ],
            "examples": [ex("ex-2", value="Café", evidence=None), ex()],
        }
    )
    assert diff_packs(old, new).empty
