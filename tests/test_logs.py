import ast
import logging
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from jevex import Document, Extractor, Field, Pipeline
from jevex.generators import GeneratorRegistry
from jevex.interfaces import Scope
from jevex.jev import JevBackendError, Noul
from jevex.layout import DomLocation
from jevex.logs import FIELDS, LogContext, current, get_logger, log_context
from jevex.pipeline import Context
from jevex.schema import SchemaSpec
from jevex.statements import Candidate, Statement
from jevex.testing import FakeJev

SRC = Path(__file__).parent.parent / "src" / "jevex"


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")


class Crashes:
    """A generator with a bug."""

    id = "crashes"
    scope = Scope(fields=frozenset({"model"}))

    def generate(self, statement: Statement) -> list[Candidate]:
        raise IndexError("group 2 out of range")


def statement(sid: str) -> Statement:
    return Statement(
        id=sid, text="Golf", kind="sentence", component_id="c1", location=DomLocation(dom_path="/p")
    )


@dataclass
class Generates:
    """Runs a failing generator on two statements, as the candidate stage would."""

    name: str = "candidates"

    async def run(self, ctx: Context) -> None:
        registry = GeneratorRegistry([Crashes()])
        field = SchemaSpec.from_model(Car).field("model")
        for sid in ("s1", "s2"):
            registry.generate(
                statement(sid),
                field,
                schema="Car",
                on_error=lambda g, e: ctx.part_failed(self.name, "generator", g.id, e),
            )


@dataclass
class Fails:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("s", {"q": Noul(instructions="a?")})
        raise JevBackendError("backend down")


def doc() -> Document:
    return Document.from_bytes(b"<p>Golf</p>", url="https://cars.test/golf")


def jevex_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name.startswith("jevex")]


async def test_a_failing_generator_is_logged_with_its_document_stage_and_part(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="jevex")
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Generates()]), run_id="r1")
    result = await ex.extract(doc())
    assert result.status == "partial"
    failed = [r for r in jevex_records(caplog) if "crashes" in r.getMessage()]
    first, again = failed
    assert first.levelno == logging.WARNING
    assert first.name == "jevex.errors"
    assert first.getMessage() == (
        "candidates generator crashes failed and was skipped: group 2 out of range"
    )
    assert first.exc_info is not None
    assert first.exc_info[0] is IndexError
    assert again.levelno == logging.DEBUG  # repeats don't flood the log
    for record in failed:
        assert record.__dict__["run_id"] == "r1"
        assert record.__dict__["url"] == "https://cars.test/golf"
        assert record.__dict__["stage"] == "candidates"
        assert record.__dict__["part"] == "crashes"
        assert len(record.__dict__["document_id"]) == 32


async def test_every_record_has_the_context_fields(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="jevex")
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Generates()]), run_id="r1")
    await ex.extract(doc())
    records = jevex_records(caplog)
    assert {r.name for r in records} >= {"jevex.extractor", "jevex.pipeline", "jevex.errors"}
    for record in records:
        assert all(hasattr(record, name) for name in FIELDS)
    [done] = [r for r in records if r.getMessage().startswith("extracted ")]
    assert done.levelno == logging.INFO
    assert done.getMessage().startswith("extracted https://cars.test/golf: partial, 0 records")
    assert done.__dict__["stage"] is None  # after the stages
    stages = [r.getMessage() for r in records if r.name == "jevex.pipeline"]
    assert stages[0] == "stage candidates started"
    assert stages[1].startswith("stage candidates finished in ")


async def test_a_failed_document_is_logged_as_an_error_with_its_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="jevex")
    jev = FakeJev().client()
    ex = Extractor([Car], jev=jev, pipeline=Pipeline([Fails()]), run_id="r1")
    result = await ex.extract(doc())
    assert result.status == "failed"
    [error] = [r for r in jevex_records(caplog) if r.levelno == logging.ERROR]
    assert error.getMessage() == "select jev failed; the document failed: backend down"
    assert error.exc_info is not None
    assert error.exc_info[0] is JevBackendError
    assert error.__dict__["stage"] == "select"


def test_log_context_nests_and_resets() -> None:
    assert current() == LogContext()
    with log_context(run_id="r1", url="u"):
        with log_context(stage="select") as inner:
            assert inner == LogContext(run_id="r1", url="u", stage="select")
        assert current() == LogContext(run_id="r1", url="u")
    assert current() == LogContext()


def test_an_extra_overrides_the_context(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="jevex")
    log = get_logger("jevex.test")
    with log_context(stage="select", part="a"):
        log.info("hello", extra={"part": "b"})
    [record] = jevex_records(caplog)
    assert (record.__dict__["stage"], record.__dict__["part"]) == ("select", "b")


def test_jevex_is_silent_until_the_application_configures_logging() -> None:
    root = logging.getLogger("jevex")
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)


def test_no_module_prints() -> None:
    """Library code logs; only the CLI writes to the terminal."""
    printing: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name == "cli.py" or "testsite" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                printing.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert printing == []
