import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from jevex import Context
from jevex.cli import CliError, load_schema, main
from jevex.results import FieldMeta
from jevex.testing import FakeJev

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


@pytest.mark.parametrize("command", ["learn", "pack", "eval", "testsite", "serve"])
def test_planned_commands_say_so(command: str) -> None:
    code, _, err = run_cli(command)
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
