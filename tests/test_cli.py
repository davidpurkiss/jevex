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


def test_no_command_prints_help() -> None:
    code, out, _ = run_cli()
    assert code == 2
    assert "extract" in out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert "jevex 0.0.1" in capsys.readouterr().out


def test_console_script_is_declared() -> None:
    import tomllib

    scripts = tomllib.loads(Path("pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {"jevex": "jevex.cli:entrypoint"}
