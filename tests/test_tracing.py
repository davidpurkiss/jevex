from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from jevex import Document, Extractor, Field, Pipeline
from jevex.jev import JevBackendError, Noul
from jevex.pipeline import Context
from jevex.results import FieldMeta
from jevex.testing import FakeJev
from jevex.tracing import resolve_tracer


class Car(BaseModel):
    """A car."""

    model: str = Field(description="Model name")


@dataclass
class Finds:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        await ctx.jev.ask("s", {"q": Noul(instructions="a?")})
        ctx.schemas["Car"].set_field("document", "model", FieldMeta(value="Golf", method="jev"))


@dataclass
class Skips:
    """A part fails and is skipped."""

    name: str = "candidates"

    async def run(self, ctx: Context) -> None:
        ctx.part_failed(self.name, "generator", "crashes", IndexError("group 2 out of range"))


@dataclass
class Fails:
    name: str = "select"

    async def run(self, ctx: Context) -> None:
        raise JevBackendError("backend down")


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    yield exporter
    exporter.clear()


def tracer(exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def doc() -> Document:
    return Document.from_bytes(
        b"<p>Golf</p>", url="https://cars.test/golf", content_type="text/html"
    )


def by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {s.name: s for s in exporter.get_finished_spans()}


async def test_a_document_is_one_trace_with_a_span_per_stage(
    exporter: InMemorySpanExporter,
) -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([Skips(), Finds()]),
        run_id="r1",
        tracer=tracer(exporter).get_tracer("test"),
    )
    result = await ex.extract(doc())
    spans = by_name(exporter)
    assert set(spans) == {"jevex.extract", "jevex.stage candidates", "jevex.stage select"}
    root, select = spans["jevex.extract"], spans["jevex.stage select"]
    assert root.parent is None
    assert root.context is not None
    assert select.context is not None
    assert select.parent is not None
    assert select.parent.span_id == root.context.span_id
    assert {s.context.trace_id for s in spans.values() if s.context} == {root.context.trace_id}
    assert root.attributes is not None
    assert dict(root.attributes) == {
        "jevex.run_id": "r1",
        "jevex.document_id": root.attributes["jevex.document_id"],
        "jevex.url": "https://cars.test/golf",
        "jevex.content_type": "text/html",
        "jevex.schemas": ("Car",),
        "jevex.status": "partial",
        "jevex.records": 1,
        "jevex.entities": 1,
        "jevex.errors": 1,
        "jevex.jev.requests": 1,
        "jevex.jev.cost_usd": result.meta.jev.cost,
        "jevex.jev.retries": 0,
        "jevex.llm.calls": 0,
        "jevex.llm.cost_usd": 0.0,
    }
    assert select.attributes is not None
    assert select.attributes["jevex.stage"] == "select"
    assert select.attributes["jevex.jev.requests"] == 1
    assert select.attributes["jevex.schemas"] == ("Car",)
    assert select.status.status_code == StatusCode.UNSET


async def test_a_skipped_part_is_an_exception_event_on_its_stage(
    exporter: InMemorySpanExporter,
) -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([Skips()]),
        tracer=tracer(exporter).get_tracer("test"),
    )
    await ex.extract(doc())
    stage = by_name(exporter)["jevex.stage candidates"]
    [event] = stage.events
    assert event.name == "exception"
    assert event.attributes is not None
    assert event.attributes["exception.type"] == "IndexError"
    assert event.attributes["jevex.part"] == "crashes"
    assert event.attributes["jevex.kind"] == "generator"
    assert stage.status.status_code == StatusCode.UNSET  # skipped, not failed


async def test_a_core_failure_marks_its_stage_and_the_document(
    exporter: InMemorySpanExporter,
) -> None:
    ex = Extractor(
        [Car],
        jev=FakeJev().client(),
        pipeline=Pipeline([Fails()]),
        tracer=tracer(exporter).get_tracer("test"),
    )
    result = await ex.extract(doc())
    assert result.status == "failed"
    spans = by_name(exporter)
    for name in ("jevex.stage select", "jevex.extract"):
        assert spans[name].status.status_code == StatusCode.ERROR
        assert [e.name for e in spans[name].events] == ["exception"]
    assert spans["jevex.extract"].attributes is not None
    assert spans["jevex.extract"].attributes["jevex.status"] == "failed"


async def test_no_tracer_no_spans(exporter: InMemorySpanExporter) -> None:
    ex = Extractor([Car], jev=FakeJev().client(), pipeline=Pipeline([Skips()]), tracer=False)
    assert ex.tracer is None
    result = await ex.extract(doc())
    assert result.status == "partial"
    assert exporter.get_finished_spans() == ()


def test_resolve_tracer() -> None:
    given = TracerProvider().get_tracer("mine")
    assert resolve_tracer(given) is given
    assert resolve_tracer(False) is None
    assert resolve_tracer(None) is not None  # the global (no-op until configured) tracer
    assert resolve_tracer(True) is not None


def test_without_opentelemetry_tracing_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    def find_spec(name: str) -> None:
        return None

    monkeypatch.setattr("importlib.util.find_spec", find_spec)
    assert resolve_tracer(None) is None


def test_an_opentelemetry_package_without_the_api_leaves_tracing_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)  # import fails
    monkeypatch.delattr("opentelemetry.trace", raising=False)
    assert resolve_tracer(None) is None
    with pytest.raises(ImportError, match="the otel extra"):
        resolve_tracer(True)
