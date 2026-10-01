import asyncio
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from jevex import Context, GeneratorRecord, GeneratorSpec, VerifiedExample, cli
from jevex.cli import CliError, load_llm, load_schema, main
from jevex.generators import GeneratorRegistry
from jevex.jev import Choice
from jevex.llm import ANTHROPIC_MODEL
from jevex.llm.anthropic import AnthropicLLM
from jevex.normalise import NormaliseStage
from jevex.packs import Pack, PackError, generator_record
from jevex.results import FieldMeta
from jevex.select import CandidateStage, SelectStage
from jevex.store import KeyMapping, open_store
from jevex.testing import FakeJev, FakeLLM

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = f"{FIXTURES / 'cli_schemas.py'}:Book"


@dataclass
class FindTitle:
    """Stand-in for the real stages: records a title so the CLI has output to print."""

    name: str = "select"

    async def run(self, ctx: Context) -> None:
        for run in ctx.active:
            run.set_field(
                "document", "title", FieldMeta(value="Dune", confidence=0.9, method="jev")
            )


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), jev=FakeJev().client(), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def page(tmp_path: Path) -> Path:
    path = tmp_path / "book.html"
    path.write_text("<html><body><h1>Dune</h1><p>Title: Dune</p></body></html>")
    return path


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import jevex.extractor as extractor

    monkeypatch.setattr(extractor, "DEFAULT_STAGES", (FindTitle(),))


@pytest.mark.usefixtures("pipeline")
def test_extract_prints_records_as_json(page: Path) -> None:
    code, out, err = run_cli("extract", str(page), "--schema", SCHEMA)
    assert (code, err) == (0, "")
    assert json.loads(out) == {
        "records": [{"schema": "Book", "entity": "document", "record": {"title": "Dune"}}]
    }


@pytest.mark.usefixtures("pipeline")
def test_extract_with_meta(page: Path) -> None:
    code, out, _ = run_cli("extract", str(page), "--schema", SCHEMA, "--meta", "--indent", "0")
    assert code == 0
    assert "\n" not in out.strip()
    data = json.loads(out)
    assert data["records"][0]["meta"]["title"]["confidence"] == 0.9
    assert data["meta"]["content_type"] == "text/html"


@pytest.mark.usefixtures("pipeline")
def test_extract_threshold(page: Path) -> None:
    code, out, _ = run_cli("extract", str(page), "--schema", SCHEMA, "--threshold", "0.95")
    assert code == 0
    assert json.loads(out)["records"][0]["record"] == {"title": None}


def test_missing_file_is_a_clean_error() -> None:
    code, out, err = run_cli("extract", "nope.html", "--schema", SCHEMA)
    assert code == 1
    assert out == ""
    assert "no such file: nope.html" in err


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("no-colon", "must look like"),
        ("jevex.nothing_here:X", "couldn't import"),
        ("jevex:__version__", "not a Pydantic model"),
        (f"{FIXTURES / 'cli_schemas.py'}:NotAModel", "not a Pydantic model"),
        ("missing/file.py:Book", "no such schema file"),
    ],
)
def test_bad_schemas(spec: str, message: str) -> None:
    with pytest.raises(CliError, match=message):
        load_schema(spec)


def test_schema_by_module_path_and_by_file() -> None:
    from pydantic import BaseModel

    assert load_schema("pydantic:BaseModel") is BaseModel
    assert load_schema(SCHEMA).__name__ == "Book"


def test_planned_commands_say_so() -> None:
    code, _, err = run_cli("serve")
    assert code == 2
    assert "not implemented yet" in err
    assert "#" in err


def test_no_command_prints_help_to_stderr() -> None:
    code, out, err = run_cli()
    assert code == 2
    assert out == ""
    assert "extract" in err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    import jevex

    assert f"jevex {jevex.__version__}" in capsys.readouterr().out


def test_console_script_is_declared() -> None:
    import tomllib

    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    assert scripts == {"jevex": "jevex.cli:entrypoint"}


# --- review round 1 --------------------------------------------------------------------


def test_missing_api_key_is_a_clean_error(page: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    out, err = io.StringIO(), io.StringIO()
    code = main(["extract", str(page), "--schema", SCHEMA], out=out, err=err)  # no jev injected
    assert code == 1
    assert "TYPESAFE_API_KEY is not set" in err.getvalue()
    assert "Traceback" not in err.getvalue()


def test_unsupported_schema_is_a_clean_error(page: Path, tmp_path: Path) -> None:
    schema = tmp_path / "bad_schema.py"
    schema.write_text("from pydantic import BaseModel\nclass Bad(BaseModel):\n    tags: set[int]\n")
    code, _, err = run_cli("extract", str(page), "--schema", f"{schema}:Bad")
    assert code == 1
    assert "unsupported type" in err


@pytest.mark.usefixtures("pipeline")
def test_jev_errors_are_clean_errors(page: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import dataclass

    import jevex.extractor as extractor
    from jevex.jev import StateTooLargeError

    @dataclass
    class Boom:
        name: str = "select"

        async def run(self, ctx: Context) -> None:
            raise StateTooLargeError("state is too big")

    monkeypatch.setattr(extractor, "DEFAULT_STAGES", (Boom(),))
    code, _, err = run_cli("extract", str(page), "--schema", SCHEMA)
    assert code == 1
    assert "jevex: error: Jev: state is too big" in err


def test_schema_files_never_shadow_real_modules(tmp_path: Path) -> None:
    import json as real_json
    import sys

    fake = tmp_path / "json.py"
    fake.write_text("from pydantic import BaseModel\nclass Book(BaseModel):\n    title: str\n")
    assert load_schema(f"{fake}:Book").__name__ == "Book"
    assert sys.modules["json"] is real_json


def test_modules_in_the_current_directory_are_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "here_schemas.py").write_text(
        "from pydantic import BaseModel\nclass Here(BaseModel):\n    x: int = 0\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.path", [p for p in __import__("sys").path if p != str(tmp_path)])
    assert load_schema("here_schemas:Here").__name__ == "Here"


def test_missing_and_dotted_attributes() -> None:
    with pytest.raises(CliError, match="has no attribute 'Nope'"):
        load_schema("jevex.results:Nope")
    assert load_schema(f"{FIXTURES / 'cli_schemas.py'}:Outer.Inner").__name__ == "Inner"


def test_a_failing_schema_file_leaves_nothing_in_sys_modules(tmp_path: Path) -> None:
    import sys

    broken = tmp_path / "broken_schema.py"
    broken.write_text("raise RuntimeError('boom')\n")
    with pytest.raises(CliError, match="boom"):
        load_schema(f"{broken}:X")
    assert not [m for m in sys.modules if m.startswith("_jevex_schema_broken_schema_")]


def test_whitespace_api_key_counts_as_missing(page: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    code = main(["extract", str(page), "--schema", SCHEMA], out=io.StringIO(), err=io.StringIO())
    assert code == 1


@pytest.mark.parametrize("value", ["5", "-0.1"])
def test_threshold_must_be_a_probability(page: Path, value: str) -> None:
    with pytest.raises(SystemExit) as info:
        main(["extract", str(page), "--schema", SCHEMA, "--threshold", value], err=io.StringIO())
    assert info.value.code == 2


def test_url_fetch_errors_are_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    import jevex.cli as cli
    from jevex.fetch import FetchError

    class NoFetch:
        async def __aenter__(self) -> "NoFetch":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        async def fetch(self, url: str) -> object:
            raise FetchError(f"GET {url} returned HTTP 503")

    monkeypatch.setattr(cli, "SimpleFetcher", NoFetch)
    code, _, err = run_cli("extract", "https://example.com/book", "--schema", SCHEMA)
    assert code == 1
    assert "HTTP 503" in err


@pytest.mark.usefixtures("pipeline")
def test_non_ascii_is_printed_as_is(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import jevex.extractor as extractor

    @dataclass
    class Accents:
        name: str = "select"

        async def run(self, ctx: Context) -> None:
            for run in ctx.active:
                run.set_field("document", "title", FieldMeta(value="Café — 東京", confidence=1.0))

    monkeypatch.setattr(extractor, "DEFAULT_STAGES", (Accents(),))
    page = tmp_path / "p.html"
    page.write_text("<p>x</p>")
    code, out, _ = run_cli("extract", str(page), "--schema", SCHEMA)
    assert code == 0
    assert "Café — 東京" in out


@pytest.mark.usefixtures("pipeline")
def test_the_jev_client_is_closed(page: Path) -> None:
    from jevex.jev import JevClient

    closed: list[bool] = []

    class Closing(FakeJev):
        async def aclose(self) -> None:
            closed.append(True)

    code = main(
        ["extract", str(page), "--schema", SCHEMA],
        jev=JevClient(Closing()),
        out=io.StringIO(),
        err=io.StringIO(),
    )
    assert code == 0
    assert closed == [True]


# --- jevex learn ---------------------------------------------------------------------------

CAR = f"{FIXTURES / 'cli_schemas.py'}:Car"
CAR_TEXT = "62 mph takes 9.1 seconds"
CAR_DRAFT = {"regex": r"(\d+(?:\.\d+)?) seconds", "group": 1, "normalise": ["parse_number"]}


def pick_91(q: Choice) -> str:
    return "9.1" if "9.1" in q.options else "none"


@pytest.fixture
def logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A SQLite store holding one logged example for Car.zero_to_62_s. The pipeline has no
    built-in generators, which would already find the value."""
    import jevex.extractor as extractor

    stages = (CandidateStage(registry=GeneratorRegistry()), SelectStage(), NormaliseStage())
    monkeypatch.setattr(extractor, "DEFAULT_STAGES", stages)
    url = f"sqlite:///{tmp_path / 'jevex.db'}"

    async def fill() -> None:
        store = open_store(url)
        await store.add_example(
            VerifiedExample(
                id="ex-1",
                field="Car.zero_to_62_s",
                statement=CAR_TEXT,
                value=9.1,
                evidence=(13, 16),
                context={"heading_trail": [], "kind": "sentence"},
                probability=0.95,
            )
        )
        await store.aclose()

    asyncio.run(fill())
    return url


def run_learn(*argv: str, llm: FakeLLM, fake: FakeJev | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    jev = (fake or FakeJev().choice(None, pick_91)).client()
    code = main(["learn", *argv], jev=jev, llm=llm, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_learn_writes_a_pack_diff_and_publishes_nothing(logged: str, tmp_path: Path) -> None:
    out = tmp_path / "diff"
    llm = FakeLLM([CAR_DRAFT])
    code, stdout, err = run_learn("--schema", CAR, "--store", logged, "--out", str(out), llm=llm)
    assert (code, err) == (0, "")
    [path] = sorted((out / "generators").iterdir())
    spec = GeneratorSpec.from_yaml(path.read_text())
    assert path.name == f"{spec.id}.yaml"
    assert spec.field == "Car.zero_to_62_s"
    assert spec.provenance.learned_from == ["ex-1"]
    assert stdout == (
        "examples: 1 (accepted 1)\n"
        f"wrote 1 generator(s) to {out / 'generators'}\n"
        f"  {spec.id}  Car.zero_to_62_s  {CAR_DRAFT['regex']}\n"
    )

    async def stored() -> list[GeneratorRecord]:
        store = open_store(logged)
        try:
            return await store.generators()
        finally:
            await store.aclose()

    assert asyncio.run(stored()) == []


def test_learn_diffs_against_a_pack(logged: str, tmp_path: Path) -> None:
    first = tmp_path / "first"
    assert (
        run_learn(
            "--schema", CAR, "--store", logged, "--out", str(first), llm=FakeLLM([CAR_DRAFT])
        )[0]
        == 0
    )
    llm = FakeLLM([])
    code, stdout, _ = run_learn(
        "--schema", CAR, "--store", logged, "--out", str(tmp_path / "second"),
        "--pack", str(first), "--json", llm=llm,
    )  # fmt: skip
    assert code == 0
    payload = json.loads(stdout)
    assert payload["generators"] == []
    assert [o["status"] for o in payload["outcomes"]] == ["covered"]
    assert llm.calls == []


def test_learn_refuses_a_used_out_directory_before_spending(logged: str, tmp_path: Path) -> None:
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "old.yaml").write_text("")
    llm = FakeLLM([])
    code, _, err = run_learn(
        "--schema", CAR, "--store", logged, "--out", str(tmp_path / "used"), llm=llm
    )
    assert code == 1
    assert "isn't an empty directory" in err
    assert llm.calls == []


def test_learn_reports_a_bad_pack_cleanly(logged: str, tmp_path: Path) -> None:
    out = str(tmp_path / "out")
    code, _, err = run_learn(
        "--schema", CAR, "--store", logged, "--out", out, "--pack", str(tmp_path / "nope"),
        llm=FakeLLM([]),
    )  # fmt: skip
    assert code == 1
    assert "no such pack directory" in err
    (tmp_path / "pack" / "generators").mkdir(parents=True)
    (tmp_path / "pack" / "generators" / "x.yaml").write_text("id: [\n")
    code, _, err = run_learn(
        "--schema", CAR, "--store", logged, "--out", out, "--pack", str(tmp_path / "pack"),
        llm=FakeLLM([]),
    )  # fmt: skip
    assert code == 1
    assert "x.yaml: spec isn't valid YAML" in err


def test_learn_needs_a_jev_key(
    logged: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    out, err = io.StringIO(), io.StringIO()
    argv = ["learn", "--schema", CAR, "--store", logged, "--out", str(tmp_path / "o")]
    code = main(argv, llm=FakeLLM([]), out=out, err=err)
    assert code == 1
    assert "TYPESAFE_API_KEY is not set" in err.getvalue()


def test_learn_with_spend_caps_records_a_budget_outcome(logged: str, tmp_path: Path) -> None:
    llm = FakeLLM([CAR_DRAFT])
    code, stdout, _ = run_learn(
        "--schema", CAR, "--store", logged, "--out", str(tmp_path / "o"),
        "--max-spend", "0", "--json", llm=llm,
    )  # fmt: skip
    assert code == 0
    assert [o["status"] for o in json.loads(stdout)["outcomes"]] == ["budget"]
    assert llm.calls == []


def test_learn_spend_caps_cant_be_negative(logged: str) -> None:
    with pytest.raises(SystemExit):
        main(["learn", "--schema", CAR, "--store", logged, "--out", "o", "--max-spend", "-1"],
             err=io.StringIO())  # fmt: skip


def test_load_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(CliError, match="must start with one of"):
        load_llm("mistral:large")
    with pytest.raises(CliError, match="needs a model, as openai:<model>"):
        load_llm("openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    llm = load_llm("anthropic")
    assert isinstance(llm, AnthropicLLM)
    assert llm.model == ANTHROPIC_MODEL
    monkeypatch.setitem(sys.modules, "jevex.llm.gemini", None)  # as if the extra is missing
    with pytest.raises(CliError, match="needs the gemini extra"):
        load_llm("gemini:gemini-3-flash")


def test_load_llm_reports_a_client_that_wont_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_key(*_: object, **__: object) -> None:
        raise RuntimeError("no API key")

    monkeypatch.setattr(AnthropicLLM, "__init__", no_key)
    with pytest.raises(CliError, match="--llm anthropic:claude-opus-5-5: no API key"):
        load_llm("anthropic:claude-opus-5-5")


def test_learn_closes_the_llm_it_builds(
    logged: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ClosingLLM(FakeLLM):
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    built = ClosingLLM([{"regex": "(", "group": 0, "normalise": []}])

    def build(_spec: str) -> FakeLLM:
        return built

    monkeypatch.setattr(cli, "load_llm", build)
    argv = ["learn", "--schema", CAR, "--store", logged, "--out", str(tmp_path / "o")]
    code = main(argv, jev=FakeJev().client(), out=io.StringIO(), err=io.StringIO())
    assert code == 0
    assert built.closed
    injected = ClosingLLM([{"regex": "(", "group": 0, "normalise": []}])
    main([*argv[:-1], str(tmp_path / "o2")], jev=FakeJev().client(), llm=injected,
         out=io.StringIO(), err=io.StringIO())  # fmt: skip
    assert not injected.closed  # the caller's to close


# --- jevex pack ---------------------------------------------------------------------------


def car_spec(gid: str, regex: str = r"(\d+) secs") -> GeneratorSpec:
    return GeneratorSpec.parse(
        {"id": gid, "field": "Car.zero_to_62_s", "match": {"regex": regex, "group": 1}}
    )


@pytest.fixture
def filled(tmp_path: Path) -> str:
    """A SQLite store with two generators (one disabled), a key mapping and an example."""
    url = f"sqlite:///{tmp_path / 'learned.db'}"

    async def fill() -> None:
        store = open_store(url)
        await store.put_generator(generator_record(car_spec("gen-a")))
        await store.put_generator(generator_record(car_spec("gen-b")))
        await store.set_generator_enabled("gen-b", False)
        await store.put_key_mapping(
            KeyMapping(fingerprint="fp1", schema="Car", path="$.price", field="price")
        )
        await store.add_example(
            VerifiedExample(id="ex-1", field="Car.zero_to_62_s", statement=CAR_TEXT, value=9.1)
        )
        await store.aclose()

    asyncio.run(fill())
    return url


def test_pack_export_import_and_diff(filled: str, tmp_path: Path) -> None:
    out = tmp_path / "pack"
    code, stdout, err = run_cli(
        "pack", "export", "--store", filled, "--out", str(out), "--name", "cars", "--examples"
    )
    assert (code, err) == (0, "")
    assert stdout == (
        f"exported pack cars 0.1.0 (1 generator, 1 key mapping, 1 example, 1 disable) to {out}\n"
    )
    pack = Pack.load(out)
    assert [g.id for g in pack.generators] == ["gen-a"]
    assert pack.manifest.disables == ["gen-b"]

    target = f"sqlite:///{tmp_path / 'fresh.db'}"
    code, stdout, _ = run_cli("pack", "import", str(out), "--store", target)
    assert code == 0
    assert stdout == (
        f"imported pack cars 0.1.0 (1 generator, 1 key mapping, 1 example, 1 disable) "
        f"into {target}\n"
    )

    async def check() -> None:
        store = open_store(target)
        assert [r.id for r in await store.generators()] == ["gen-a"]
        assert await store.disabled_generator_ids() == {"gen-b"}
        assert [m.field for m in await store.key_mappings("fp1")] == ["price"]
        assert [e.id for e in await store.examples()] == ["ex-1"]
        await store.aclose()

    asyncio.run(check())
    code, stdout, _ = run_cli("pack", "diff", filled, target, "--examples")
    assert (code, stdout) == (0, "no changes\n")


def test_pack_diff_shows_what_changes(filled: str, tmp_path: Path) -> None:
    run_cli("pack", "export", "--store", filled, "--out", str(tmp_path / "v1"), "--name", "cars")
    v1 = Pack.load(tmp_path / "v1")
    v2 = v1.model_copy(
        update={
            "manifest": v1.manifest.model_copy(update={"version": "0.2.0", "disables": []}),
            "generators": [car_spec("gen-a", r"(\d+) seconds"), car_spec("gen-c")],
            "key_mappings": [],
        }
    )
    v2.write(tmp_path / "v2")
    code, stdout, _ = run_cli("pack", "diff", str(tmp_path / "v1"), str(tmp_path / "v2"))
    assert code == 0
    assert stdout == (
        "manifest: version '0.1.0' -> '0.2.0'\n"
        "generators:\n"
        "  + gen-c  Car.zero_to_62_s  (\\d+) secs\n"
        "  ~ gen-a  Car.zero_to_62_s  (\\d+) seconds\n"
        "key mappings:\n"
        "  - fp1  Car  $.price -> price\n"
        "disables:\n"
        "  - gen-b\n"
    )
    code, stdout, _ = run_cli("pack", "diff", str(tmp_path / "v1"), str(tmp_path / "v2"), "--json")
    data = json.loads(stdout)
    assert data["manifest"] == {"version": ["0.1.0", "0.2.0"]}
    assert [g["id"] for g in data["generators"]["added"]] == ["gen-c"]
    # Against a store, the manifest isn't compared: the store has none.
    code, stdout, _ = run_cli("pack", "diff", filled, str(tmp_path / "v2"))
    assert not stdout.startswith("manifest:")
    assert "  + gen-c" in stdout


def test_pack_export_refuses_bad_arguments_before_writing(filled: str, tmp_path: Path) -> None:
    out = tmp_path / "pack"
    code, _, err = run_cli("pack", "export", "--store", filled, "--out", str(out), "--name", "../x")
    assert code == 1
    assert "jevex: error: --name: String should match pattern" in err
    missing = f"sqlite:///{tmp_path / 'missing.db'}"
    code, _, err = run_cli("pack", "export", "--store", missing, "--out", str(out), "--name", "x")
    assert (code, err) == (1, f"jevex: error: no such store: {missing}\n")
    assert not (tmp_path / "missing.db").exists()  # reading never creates a database
    out.mkdir()
    (out / "old.yaml").write_text("")
    code, _, err = run_cli("pack", "export", "--store", filled, "--out", str(out), "--name", "x")
    assert "isn't an empty directory" in err


def test_pack_import_and_diff_report_bad_packs_cleanly(filled: str, tmp_path: Path) -> None:
    (tmp_path / "bad").mkdir()
    code, _, err = run_cli("pack", "import", str(tmp_path / "bad"), "--store", filled)
    assert (code, err) == (1, f"jevex: error: {tmp_path / 'bad'} has no manifest.yaml\n")
    code, _, err = run_cli("pack", "diff", "no-such-pack", filled)
    assert code == 1
    assert "no installed pack named 'no-such-pack'" in err


def test_pack_needs_a_subcommand() -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli("pack")
    assert exc.value.code == 2


@pytest.mark.usefixtures("pipeline")
def test_a_broken_community_pack_is_a_clean_error(
    page: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jevex.extractor as extractor

    def broken(names: list[str] | None = None) -> list[Pack]:
        raise PackError("installed pack 'cars' (cars_pack) failed to load: boom")

    monkeypatch.setattr(extractor, "community_packs", broken)
    code, out, err = run_cli("extract", str(page), "--schema", SCHEMA)
    assert (code, out) == (1, "")
    assert err == "jevex: error: pack: installed pack 'cars' (cars_pack) failed to load: boom\n"


def test_testsite_build_writes_the_site_and_says_what_it_built(tmp_path: Path) -> None:
    out = tmp_path / "site"
    code, stdout, err = run_cli(
        "testsite", "build", "--seed", "7", "--out", str(out), "--waves", "table,listing;kv"
    )
    assert (code, err) == (0, "")
    manifest = json.loads((out / "truth.json").read_text())
    assert manifest["seed"] == 7
    assert manifest["waves"] == [["table", "listing"], ["kv"]]
    lines = stdout.splitlines()
    assert lines[0] == f"built {len(manifest['pages'])} pages for seed 7 in {out}"
    assert lines[1] == "  wave 1: table, listing (88 pages)"
    assert lines[2] == "  wave 2: kv (16 pages)"
    assert lines[3] == f"digest {manifest['digest']}"
    again = tmp_path / "again"
    run_cli("testsite", "build", "--seed", "7", "--out", str(again), "--waves", "table,listing;kv")
    assert (again / "truth.json").read_bytes() == (out / "truth.json").read_bytes()


def test_testsite_build_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def fake_build(seed: int, out_dir: str, *, waves: object) -> dict[str, object]:
        calls.append((seed, out_dir, waves))
        return {"waves": [], "pages": [], "digest": "d"}

    monkeypatch.setattr(cli, "build", fake_build)
    code, stdout, _ = run_cli("testsite", "build")
    assert code == 0
    assert calls == [(42, "testsite/build", cli.DEFAULT_WAVES)]
    assert stdout == "built 0 pages for seed 42 in testsite/build\ndigest d\n"


def test_testsite_build_refuses_a_bad_schedule(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli("testsite", "build", "--waves", "table;tables")
    assert exc.value.code == 2
    assert "--waves: wave 2: unknown family 'tables'" in capsys.readouterr().err


def test_testsite_build_errors_are_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "notes.txt").write_text("mine")
    code, _, err = run_cli("testsite", "build", "--out", str(tmp_path))
    assert code == 1
    assert err.startswith("jevex: error: ")
    assert "refusing" in err
    assert (tmp_path / "notes.txt").read_text() == "mine"

    def no_pillow(*args: object, **kwargs: object) -> None:
        raise ImportError("Rendering the test site's images needs Pillow: install jevex[testsite]")

    monkeypatch.setattr(cli, "build", no_pillow)
    code, _, err = run_cli("testsite", "build", "--out", str(tmp_path / "new"))
    assert code == 1
    assert err.endswith("jevex[testsite], or leave scanned and infographic out of --waves\n")


def test_testsite_serve_serves_until_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    served: list[tuple[str, str, int]] = []

    class FakeServer:
        server_address = ("127.0.0.1", 8123)
        closed = False

        def __enter__(self) -> "FakeServer":
            return self

        def __exit__(self, *exc: object) -> None:
            self.closed = True

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

    fake = FakeServer()

    def fake_server(directory: str, host: str, port: int) -> FakeServer:
        served.append((directory, host, port))
        return fake

    monkeypatch.setattr(cli, "server", fake_server)
    code, out, err = run_cli("testsite", "serve", "--dir", str(tmp_path), "--port", "8123")
    assert (code, err) == (0, "")
    assert served == [(str(tmp_path), "127.0.0.1", 8123)]
    assert out == f"serving {tmp_path} at http://127.0.0.1:8123/ (Ctrl-C to stop)\n"
    assert fake.closed


def test_testsite_serve_needs_a_build(tmp_path: Path) -> None:
    code, out, err = run_cli("testsite", "serve", "--dir", str(tmp_path))
    assert (code, out) == (1, "")
    assert err == (
        f"jevex: error: {tmp_path} has no test site; build one first (jevex testsite build)\n"
    )


def test_testsite_needs_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli("testsite")
    assert exc.value.code == 2
    assert "testsite command" in capsys.readouterr().err
