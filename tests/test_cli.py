import asyncio
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI

from jevex import Context, CorpusLock, GeneratorRecord, GeneratorSpec, VerifiedExample, cli
from jevex.cli import CliError, load_llm, load_schema, main
from jevex.fetch import RobotsDisallowedError
from jevex.generators import GeneratorRegistry
from jevex.jev import Choice
from jevex.llm import ANTHROPIC_MODEL
from jevex.llm.anthropic import AnthropicLLM
from jevex.normalise import NormaliseStage
from jevex.packs import Pack, PackError, generator_record
from jevex.results import FieldMeta
from jevex.select import CandidateStage, SelectStage
from jevex.store import DocumentStat, KeyMapping, open_store
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
        "status": "ok",
        "errors": [],
        "records": [{"schema": "Book", "entity": "document", "record": {"title": "Dune"}}],
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


@dataclass
class TitleIsLocale:
    """Records the document's locale (or ``"none"``) as the title, so a test can see it."""

    name: str = "select"

    async def run(self, ctx: Context) -> None:
        meta = FieldMeta(value=ctx.locale or "none", confidence=0.9, method="jev")
        for run in ctx.active:
            run.set_field("document", "title", meta)


@pytest.mark.parametrize(
    ("html", "locale"),
    [
        ("<html><body><p>Title: Dune</p></body></html>", "de-AT"),
        ('<html lang="fr-FR"><body><p>Titre : Dune</p></body></html>', "fr-FR"),
    ],
)
def test_extract_locale_is_for_documents_without_their_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, html: str, locale: str
) -> None:
    import jevex.extractor as extractor

    monkeypatch.setattr(extractor, "DEFAULT_STAGES", (TitleIsLocale(),))
    path = tmp_path / "book.html"
    path.write_text(html)
    code, out, err = run_cli("extract", str(path), "--schema", SCHEMA, "--locale", "de_at")
    assert (code, err) == (0, "")
    assert json.loads(out)["records"][0]["record"] == {"title": locale}


def test_extract_without_a_locale_leaves_it_unknown(
    page: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jevex.extractor as extractor

    monkeypatch.setattr(extractor, "DEFAULT_STAGES", (TitleIsLocale(),))
    code, out, _ = run_cli("extract", str(page), "--schema", SCHEMA)
    assert code == 0
    assert json.loads(out)["records"][0]["record"] == {"title": "none"}


def test_extract_rejects_a_bad_locale(page: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli("extract", str(page), "--schema", SCHEMA, "--locale", "en GB")
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "argument --locale: locale must be a BCP 47 language tag" in captured.err
    assert captured.out == ""


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


type Served = list[tuple[FastAPI, dict[str, object]]]


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Served:
    """The apps ``jevex serve`` runs, and where; serving starts and stops at once."""
    import uvicorn

    calls: Served = []

    async def serve(self: uvicorn.Server, sockets: object = None) -> None:
        config = self.config
        assert isinstance(config.app, FastAPI)
        calls.append((config.app, {"host": config.host, "port": config.port}))
        self.started = True

    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    return calls


@pytest.mark.usefixtures("pipeline")
def test_serve_runs_the_app_until_interrupted(served: Served, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from jevex.server import Service

    url = f"sqlite:///{tmp_path / 'jevex.db'}"
    code, out, err = run_cli(
        "serve", "--schema", SCHEMA, "--store", url, "--stats", "--port", "9001",
        "--max-spend", "2", "--period", "week",
    )  # fmt: skip
    assert (code, err) == (0, "")
    assert out == (
        "serving Book at http://127.0.0.1:9001/extract; stats at "
        "http://127.0.0.1:9001/stats/ (Ctrl-C to stop)\n"
    )
    [(app, options)] = served
    assert options == {"host": "127.0.0.1", "port": 9001}
    service: object = app.state.service
    assert isinstance(service, Service)
    run = service.budgets.run if service.budgets else None
    assert run is not None
    assert (run.max_spend, run.period) == (2, "week")
    with TestClient(app) as client:
        response = client.post(
            "/extract", json={"document": {"content": "PGgxPkR1bmU8L2gxPg=="}, "schema": "Book"}
        )
        assert response.json()["records"][0]["record"] == {"title": "Dune"}
        assert client.get("/stats/api/summary").json()["documents"] == 1


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--stats"], "--stats reads the store: give --store too"),
        (["--stats-budget", "5"], "--stats-budget only applies with --stats"),
        (["--period", "week"], "--period only applies with --max-spend or --max-jev-spend"),
    ],
)
def test_serve_usage_errors(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli("serve", "--schema", SCHEMA, *argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--schema", "nope:Nope"], "couldn't import 'nope'"),
        (["--schema", SCHEMA, "--store", "mysql://x"], "store: unsupported store URL"),
    ],
)
def test_serve_errors_before_serving(argv: list[str], message: str, served: Served) -> None:
    code, out, err = run_cli("serve", *argv)
    assert (code, out) == (1, "")
    assert message in err
    assert served == []


@pytest.mark.usefixtures("pipeline")
def test_serve_passes_its_locale_to_the_extractors(served: Served) -> None:
    from fastapi.testclient import TestClient

    from jevex.server import Service

    code, _, err = run_cli("serve", "--schema", SCHEMA, "--locale", "de_at")
    assert (code, err) == (0, "")
    [(app, _)] = served
    service: object = app.state.service
    assert isinstance(service, Service)
    assert service.locale == "de-AT"
    with TestClient(app):
        assert service.extractor(["Book"]).locale == "de-AT"


def test_serve_rejects_a_bad_locale(served: Served, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli("serve", "--schema", SCHEMA, "--locale", "en GB")
    assert exc.value.code == 2
    assert "argument --locale: locale must be a BCP 47 language tag" in capsys.readouterr().err
    assert served == []


def test_serve_needs_an_api_key(monkeypatch: pytest.MonkeyPatch, served: Served) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    err = io.StringIO()
    assert main(["serve", "--schema", SCHEMA], err=err) == 1
    assert "TYPESAFE_API_KEY is not set" in err.getvalue()
    assert served == []


class ClosingLLM(FakeLLM):
    closed = False

    async def aclose(self) -> None:
        self.closed = True


def test_serve_closes_the_llm_it_builds_once_serving_stops(
    monkeypatch: pytest.MonkeyPatch, served: Served
) -> None:
    built = ClosingLLM([])

    def build(_spec: str) -> FakeLLM:
        return built

    monkeypatch.setattr(cli, "load_llm", build)
    code, _, err = run_cli("serve", "--schema", SCHEMA, "--llm", "anthropic:claude-sonnet-5-5")
    assert (code, err) == (0, "")
    assert len(served) == 1
    assert built.closed
    injected = ClosingLLM([])
    argv = ["serve", "--schema", SCHEMA, "--llm", "anthropic:claude-sonnet-5-5"]
    code = main(argv, jev=FakeJev().client(), llm=injected, out=io.StringIO(), err=io.StringIO())
    assert code == 0
    assert not injected.closed  # the caller's to close


def test_serve_fails_when_the_server_doesnt_start(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    async def refused(self: uvicorn.Server, sockets: object = None) -> None:
        sys.exit(3)  # as uvicorn does when it can't bind the port, after logging why

    monkeypatch.setattr(uvicorn.Server, "serve", refused)
    built = ClosingLLM([])

    def build(_spec: str) -> FakeLLM:
        return built

    monkeypatch.setattr(cli, "load_llm", build)
    code, _, err = run_cli(
        "serve", "--schema", SCHEMA, "--port", "9002", "--llm", "anthropic:claude-sonnet-5-5"
    )
    assert code == 1
    assert "couldn't serve at http://127.0.0.1:9002 (see the log above)" in err
    assert built.closed


def test_serve_needs_the_server_extra(monkeypatch: pytest.MonkeyPatch, served: Served) -> None:
    monkeypatch.setitem(sys.modules, "jevex.server", None)
    code, out, err = run_cli("serve", "--schema", SCHEMA)
    assert (code, out) == (1, "")
    assert "jevex serve needs the server extra: pip install 'jevex[server]'" in err


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
    code, out, err = run_cli("extract", str(page), "--schema", SCHEMA)
    assert (code, out) == (1, "")
    assert err == "jevex: error: select jev: StateTooLargeError: state is too big\n"


def test_extract_prints_a_partial_result_and_warns(
    page: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jevex.extractor as extractor

    @dataclass
    class SkipsAPart:
        name: str = "select"

        async def run(self, ctx: Context) -> None:
            await FindTitle().run(ctx)
            ctx.part_failed("candidates", "generator", "gen-1", RuntimeError("bad regex"))

    monkeypatch.setattr(extractor, "DEFAULT_STAGES", (SkipsAPart(),))
    code, out, err = run_cli("extract", str(page), "--schema", SCHEMA)
    assert code == 0
    data = json.loads(out)
    assert data["status"] == "partial"
    assert data["records"][0]["record"] == {"title": "Dune"}
    assert err == "jevex: warning: candidates generator gen-1: RuntimeError: bad regex\n"


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


class Closing(FakeJev):
    """A Jev backend that notes when it's closed."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


@pytest.mark.usefixtures("pipeline")
def test_the_jev_client_made_for_the_command_is_closed(
    page: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = Closing()
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    def api_backend(model: str | None = None) -> Closing:
        return backend

    monkeypatch.setattr("jevex.jev.TypeSafeBackend", api_backend)
    err = io.StringIO()
    code = main(["extract", str(page), "--schema", SCHEMA], out=io.StringIO(), err=err)
    assert (code, err.getvalue()) == (0, "")
    assert backend.closed == 1


@pytest.mark.usefixtures("pipeline")
def test_a_jev_client_passed_in_is_left_open(page: Path) -> None:
    from jevex.jev import JevClient

    backend = Closing()
    code = main(
        ["extract", str(page), "--schema", SCHEMA],
        jev=JevClient(backend),
        out=io.StringIO(),
        err=io.StringIO(),
    )
    assert code == 0
    assert backend.closed == 0  # its maker closes it


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


# --- stats -----------------------------------------------------------------------------

REPLAY_CSV = """\
batch,documents,size,waves,accuracy,llm_calls_per_document,jev_cost_per_document,\
llm_cost_per_document,errors,generators,values_llm
1,2,2,1,1.0,2.0,0.001,0.02,0,1,3
2,4,2,2,0.5,0.0,0.001,0.0,0,3,0
"""


def stats_store(path: Path) -> str:
    async def fill() -> None:
        store = open_store(path)
        await store.record_document(DocumentStat(id="d1", llm_calls=2))
        await store.aclose()

    asyncio.run(fill())
    return f"sqlite:///{path}"


def test_stats_export_writes_an_animated_svg(tmp_path: Path) -> None:
    (tmp_path / "curve.csv").write_text(REPLAY_CSV)
    out = tmp_path / "learning.svg"
    code, stdout, err = run_cli(
        "stats",
        "export",
        "--replay",
        str(tmp_path / "curve.csv"),
        "--svg",
        "learning",
        "--out",
        str(out),
    )
    assert (code, stdout, err) == (0, f"wrote {out}\n", "")
    svg = out.read_text()
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" class="chart viz-root animate"')
    assert ">wave 2</text>" in svg  # a replay's default axis is documents


def test_stats_export_static_from_a_store_with_a_budget(tmp_path: Path) -> None:
    url = stats_store(tmp_path / "s.db")
    out = tmp_path / "cost.svg"
    code, _, err = run_cli(
        "stats",
        "export",
        "--store",
        url,
        "--svg",
        "cost",
        "--out",
        str(out),
        "--static",
        "--budget",
        "2",
        "--x",
        "docs",
    )
    assert (code, err) == (0, "")
    svg = out.read_text()
    assert 'class="chart viz-root"' in svg
    assert ">budget $2.000</text>" in svg
    assert ">1 documents</text>" in svg


def test_stats_export_errors_are_clean(tmp_path: Path) -> None:
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    code, out, err = run_cli(
        "stats",
        "export",
        "--replay",
        str(tmp_path / "bad.csv"),
        "--svg",
        "mix",
        "--out",
        str(tmp_path / "mix.svg"),
    )
    assert (code, out) == (1, "")
    assert err == (
        "jevex: error: not a jevex replay CSV: no documents, jev_cost_per_document, "
        "llm_calls_per_document, size column\n"
    )
    (tmp_path / "ok.csv").write_text(REPLAY_CSV)
    code, _, err = run_cli(
        "stats",
        "export",
        "--replay",
        str(tmp_path / "ok.csv"),
        "--svg",
        "mix",
        "--x",
        "time",
        "--out",
        str(tmp_path / "mix.svg"),
    )
    assert code == 1
    assert "use the documents axis" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["stats"], "one of the arguments --store --replay is required"),
        (["stats", "--store", "a", "--replay", "b"], "not allowed with argument"),
        (["stats", "export", "--replay", "c.csv"], "jevex stats export needs --svg and --out"),
        (["stats", "--replay", "c.csv", "--svg", "mix"], "--svg only apply to jevex stats export"),
        (["stats", "--replay", "c.csv", "--static"], "--static only apply to jevex stats export"),
        (["stats", "export", "--replay", "c.csv", "--svg", "pie"], "invalid choice: 'pie'"),
    ],
)
def test_stats_usage_errors(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli(*argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_stats_serves_until_interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = stats_store(tmp_path / "s.db")
    served: list[tuple[str, int]] = []

    class FakeServer:
        server_address = ("127.0.0.1", 8765)
        closed = False

        def __enter__(self) -> "FakeServer":
            return self

        def __exit__(self, *exc: object) -> None:
            self.closed = True

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

    fake = FakeServer()

    def fake_server(load: object, host: str, port: int) -> FakeServer:
        assert callable(load)
        served.append((host, port))
        return fake

    monkeypatch.setattr(cli, "stats_server", fake_server)
    code, out, err = run_cli("stats", "--store", url)
    assert (code, err) == (0, "")
    assert served == [("127.0.0.1", 8765)]
    assert out == "serving stats at http://127.0.0.1:8765/stats/ (Ctrl-C to stop)\n"
    assert fake.closed


def test_stats_serve_checks_the_source_first(tmp_path: Path) -> None:
    code, out, err = run_cli("stats", "--replay", str(tmp_path / "nope.csv"))
    assert (code, out) == (1, "")
    assert err.startswith("jevex: error: [Errno 2] No such file or directory")


# --- jevex corpus ----------------------------------------------------------------------


def test_corpus_lock_and_check(tmp_path: Path) -> None:
    corpus = tmp_path / "site"
    run_cli("testsite", "build", "--seed", "7", "--out", str(corpus), "--waves", "kv")
    lock = tmp_path / "site.lock"
    code, stdout, err = run_cli(
        "corpus", "lock", str(corpus), "--name", "kv", "--out", str(lock), "--publish", "aggregate"
    )
    assert (code, err) == (0, "")
    assert stdout == f"locked 16 documents of {corpus} as 'kv' in {lock}\n"
    written = json.loads(lock.read_text())
    assert (written["name"], written["publish"]) == ("kv", "aggregate")
    assert written["settings"] == {"seed": 7, "waves": [["kv"]]}

    code, stdout, err = run_cli("corpus", "check", str(corpus), str(lock))
    assert (code, err) == (0, "")
    assert stdout == f"{corpus} matches the 'kv' lock (16 documents)\n"

    page = sorted(written["documents"])[0]
    (corpus / page).write_text("changed")
    code, stdout, err = run_cli("corpus", "check", str(corpus), str(lock))
    assert (code, stdout) == (1, "")
    assert err == f"jevex: error: {corpus} doesn't match the 'kv' lock:\n1 changed: {page}\n"


def test_corpus_lock_and_check_report_bad_input(tmp_path: Path) -> None:
    code, _, err = run_cli("corpus", "lock", str(tmp_path), "--name", "x", "--out", "x.lock")
    assert code == 1
    assert err.startswith(f"jevex: error: {tmp_path / 'truth.json'} isn't a corpus manifest")
    code, _, err = run_cli("corpus", "check", str(tmp_path), str(tmp_path / "x.lock"))
    assert code == 1
    assert err.startswith("jevex: error: can't read lock")


def test_corpus_books_fetches_a_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    async def fake_books(out: str, *, sample: int, seed: int) -> CorpusLock:
        calls.append((out, sample, seed))
        return CorpusLock(name="books", truth="t", documents={"pages/a.html": "h"}, digest="d")

    monkeypatch.setattr(cli, "books_corpus", fake_books)
    lock = tmp_path / "books.lock"
    code, stdout, err = run_cli(
        "corpus", "books", "--out", "b", "--sample", "1", "--seed", "3", "--lock", str(lock)
    )
    assert (code, err) == (0, "")
    assert calls == [("b", 1, 3)]
    assert stdout == f"fetched 1 book (seed 3) into b\nwrote {lock}\n"
    assert CorpusLock.load(lock).digest == "d"

    code, stdout, _ = run_cli("corpus", "books", "--out", "b")
    assert calls[-1] == ("b", 200, 42)
    assert stdout == "fetched 1 book (seed 42) into b\n"


def test_corpus_books_reports_fetch_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def blocked(out: str, *, sample: int, seed: int) -> CorpusLock:
        raise RobotsDisallowedError("robots.txt disallows it")

    monkeypatch.setattr(cli, "books_corpus", blocked)
    code, stdout, err = run_cli("corpus", "books", "--out", str(tmp_path))
    assert (code, stdout) == (1, "")
    assert err == "jevex: error: robots.txt disallows it\n"


def test_corpus_needs_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli("corpus")
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run_cli("corpus", "books", "--out", "b", "--sample", "0")
    assert exc.value.code == 2
    assert "must be at least 1, not 0" in capsys.readouterr().err


# --- jevex baseline / eval --results ---------------------------------------------------

BOOK_SCHEMA = "jevex.examples.books:Book"
CONFIG = Path(__file__).parent.parent / "benchmarks" / "config.yaml"
BOOK_PAGES = ("a-light-in-the-attic_1000", "sapiens-a-brief-history-of-humankind_996")


@pytest.fixture
def books(tmp_path: Path) -> Path:
    from jevex.benchmarks import book_values
    from jevex.clean import html_text_of

    root = tmp_path / "books"
    (root / "pages").mkdir(parents=True)
    pages: list[dict[str, object]] = []
    for name in BOOK_PAGES:
        content = (FIXTURES / "books" / f"{name}.html").read_bytes()
        (root / "pages" / f"{name}.html").write_bytes(content)
        values = book_values(html_text_of(content))
        pages.append(
            {"path": f"pages/{name}.html", "schema": "Book", "records": [{"values": values}]}
        )
    (root / "truth.json").write_text(json.dumps({"pages": pages}))
    return root


def _book_answer(prompt: str, schema: type) -> dict[str, object]:
    from jevex.benchmarks import book_values
    from jevex.clean import html_text_of

    for name in BOOK_PAGES:
        values = book_values(html_text_of((FIXTURES / "books" / f"{name}.html").read_bytes()))
        if f"# {values['title']}\n" in prompt:
            return {"Book": [values]}
    return {"Book": []}


def run_baseline_cli(*argv: str, **kwargs: object) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err, **kwargs)  # type: ignore[arg-type]
    return code, out.getvalue(), err.getvalue()


def test_baseline_inputs_then_run_then_eval_results(books: Path, tmp_path: Path) -> None:
    inputs, results = tmp_path / "inputs.jsonl", tmp_path / "results.jsonl"
    code, out, err = run_baseline_cli(
        "baseline",
        "inputs",
        str(books),
        "--out",
        str(inputs),
        "--pipeline",
        "jevex.examples.books:books_pipeline",
    )
    assert (code, err) == (0, "")
    assert out == f"wrote the inputs of 2 documents to {inputs}\n"
    assert "out of five stars" in inputs.read_text()

    llm = FakeLLM(_book_answer, model="claude-haiku-4-5-20251001", price=(1.0, 5.0))
    code, out, err = run_baseline_cli(
        "baseline",
        "run",
        str(books),
        "--schema",
        BOOK_SCHEMA,
        "--model",
        "fast",
        "--out",
        str(results),
        "--inputs",
        str(inputs),
        "--config",
        str(CONFIG),
        llm=llm,
    )
    assert (code, err) == (0, "")
    assert out.startswith(f"wrote {results}: 2 documents, 0 failed, $0.00")
    assert len(llm.calls) == 2
    assert llm.calls[0].prompt.startswith("Extract structured records from the document.")
    assert "- price: Price, in GBP. A number." in llm.calls[0].prompt

    code, out, err = run_baseline_cli(
        "eval", str(books), "--schema", BOOK_SCHEMA, "--results", str(results), "--json"
    )
    assert (code, err) == (0, "")
    summary = json.loads(out)["summary"]
    assert summary["accuracy"] == 1.0
    assert summary["llm_calls_per_document"] == 1
    assert summary["jev_requests_per_document"] == 0
    assert summary["cost_per_document"] > 0


def test_baseline_run_builds_a_given_system_from_the_pinned_setup(
    books: Path, tmp_path: Path
) -> None:
    from jevex.baselines import BaselineInput, BaselineOutput, BaselineSetup

    seen: list[BaselineSetup] = []

    @dataclass
    class Tool:
        name: str = "tool"
        closed: bool = False

        async def extract(self, source: BaselineInput) -> BaselineOutput:
            if "# Sapiens" in source.text:
                raise RuntimeError("the tool crashed")
            return BaselineOutput({"Book": []}, calls=2, input_tokens=10, output_tokens=5, cost=0.5)

        async def aclose(self) -> None:
            self.closed = True

    tool = Tool()

    def make(setup: BaselineSetup) -> Tool:
        seen.append(setup)
        return tool

    code, out, err = run_baseline_cli(
        "baseline",
        "run",
        str(books),
        "--schema",
        BOOK_SCHEMA,
        "--model",
        "strong",
        "--out",
        str(tmp_path / "results.jsonl"),
        "--config",
        str(CONFIG),
        "--concurrency",
        "1",
        system=make,
    )
    assert code == 1
    assert out == f"wrote {tmp_path / 'results.jsonl'}: 2 documents, 1 failed, $0.5000\n"
    assert err == (
        "jevex: error: pages/sapiens-a-brief-history-of-humankind_996.html: "
        "RuntimeError: the tool crashed\n"
    )
    assert seen[0].model.model == "claude-opus-5-5"
    assert seen[0].instructions.count("## Book") == 1
    assert tool.closed


def test_baseline_run_reports_setup_errors(books: Path, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    text = CONFIG.read_text()
    config.write_text(text[: text.index("  baseline_gemini:")] + text[text.index("# Locks are") :])
    base = ["baseline", "run", str(books), "--schema", BOOK_SCHEMA, "--out", str(tmp_path / "r")]
    code, _, err = run_baseline_cli(*base, "--model", "gemini", "--config", str(config))
    assert code == 1
    assert err == f"jevex: error: {config} pins no models.baseline_gemini\n"

    code, _, err = run_baseline_cli(
        *base, "--model", "fast", "--config", str(CONFIG), "--prompt", str(tmp_path / "none")
    )
    assert code == 1
    assert err.startswith("jevex: error: can't read prompt")

    lock = tmp_path / "books.lock"
    assert run_cli("corpus", "lock", str(books), "--name", "books", "--out", str(lock))[0] == 0
    (books / "pages" / f"{BOOK_PAGES[0]}.html").write_text("<p>changed</p>")
    code, _, err = run_baseline_cli(
        *base, "--model", "fast", "--config", str(CONFIG), "--lock", str(lock), llm=FakeLLM([])
    )
    assert code == 1
    assert "doesn't match the 'books' lock" in err
    assert not (tmp_path / "r").exists()

    code, _, err = run_baseline_cli(
        *base, "--model", "fast", "--config", str(CONFIG), "--pipeline", "jevex.examples.books:Book"
    )
    assert code == 1
    assert "is not a Pipeline or a function returning one" in err


def test_eval_results_takes_no_run_options(
    books: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["eval", str(books), "--schema", BOOK_SCHEMA, "--results", "r", "--replay"])
    assert exit_info.value.code == 2
    assert "--results scores a results file: leave out --replay" in capsys.readouterr().err
    code, _, err = run_baseline_cli(
        "eval", str(books), "--schema", BOOK_SCHEMA, "--results", str(tmp_path / "none")
    )
    assert code == 1
    assert err.startswith("jevex: error: can't read results")
