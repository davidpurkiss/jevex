"""The consumer entry point: ``Extractor(schemas=[...]).extract(document)``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field

from jevex.clean import CleanStage
from jevex.gate import DocumentGateStage
from jevex.interfaces import GateDecision
from jevex.jev import JevClient
from jevex.layout import LayoutStage
from jevex.pipeline import Context, Pipeline
from jevex.schema import SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from jevex.document import Document
    from jevex.pipeline import Stage

# Default stages in spec order; each lands with its issue (see jevex.interfaces).
DEFAULT_STAGES: tuple[Stage, ...] = (CleanStage(), DocumentGateStage(), LayoutStage())


def default_pipeline() -> Pipeline:
    """The pipeline ``Extractor`` builds when none is given."""
    return Pipeline(DEFAULT_STAGES)


class JevUsageSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: int
    questions: int
    input_tokens: int
    cost: float
    seconds: float
    models: list[str]


class EventInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    kind: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict[str, Any])


class DocumentMeta(BaseModel):
    """Document-level metadata: gates, Jev usage, timings and events."""

    model_config = ConfigDict(frozen=True)

    url: str | None
    content_type: str
    gates: dict[str, GateDecision]
    active_schemas: list[str]
    jev: JevUsageSummary
    timings: dict[str, float]
    events: list[EventInfo]
    stopped: bool


class ExtractionResult(BaseModel):
    """What ``extract`` returns.

    ``values`` holds the normalised values by schema, entity scope and field. Typed
    records with per-field metadata arrive with the results API (#20).
    """

    model_config = ConfigDict(frozen=True)

    values: dict[str, dict[str, dict[str, Any]]]
    meta: DocumentMeta

    @classmethod
    def from_context(cls, ctx: Context) -> ExtractionResult:
        usage = ctx.jev.usage
        return cls(
            values={name: run.values for name, run in ctx.schemas.items() if run.values},
            meta=DocumentMeta(
                url=ctx.document.url,
                content_type=ctx.document.content_type,
                gates={name: run.gate for name, run in ctx.schemas.items() if run.gate},
                active_schemas=[run.name for run in ctx.active],
                jev=JevUsageSummary(
                    requests=usage.requests,
                    questions=usage.questions,
                    input_tokens=usage.input_tokens,
                    cost=usage.cost,
                    seconds=usage.seconds,
                    models=sorted(usage.models),
                ),
                timings=dict(ctx.timings),
                events=[
                    EventInfo(stage=e.stage, kind=e.kind, message=e.message, data=e.data)
                    for e in ctx.events
                ],
                stopped=ctx.stopped,
            ),
        )


class Extractor:
    """Extracts records for the registered schemas from documents.

    The Jev client is created from ``TYPESAFE_*`` environment variables on first use
    unless one is passed in. Use as an async context manager (or call ``aclose``) to
    release its connections.
    """

    def __init__(
        self,
        schemas: Sequence[type[BaseModel]],
        *,
        jev: JevClient | None = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        if not schemas:
            raise ValueError("register at least one schema")
        self.schemas = [SchemaSpec.from_model(m) for m in schemas]
        names = [s.name for s in self.schemas]
        if len(set(names)) != len(names):
            raise ValueError(f"schema names must be unique: {names}")
        self.pipeline = pipeline if pipeline is not None else default_pipeline()
        self._jev = jev
        self._sync_loop: asyncio.AbstractEventLoop | None = None

    @property
    def jev(self) -> JevClient:
        if self._jev is None:
            self._jev = JevClient.from_env()
        return self._jev

    async def extract(self, document: Document) -> ExtractionResult:
        """Run the pipeline over one document."""
        ctx = Context.create(document, self.schemas, self.jev.metered())
        await self.pipeline.run(ctx)
        return ExtractionResult.from_context(ctx)

    def extract_sync(self, document: Document) -> ExtractionResult:
        """Blocking wrapper for scripts and notebooks without a running event loop.

        Reuses one private event loop so HTTP connections stay valid between calls.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("extract_sync() called inside an event loop; use await extract()")
        if self._sync_loop is None or self._sync_loop.is_closed():
            self._sync_loop = asyncio.new_event_loop()
        return self._sync_loop.run_until_complete(self.extract(document))

    async def aclose(self) -> None:
        close = getattr(self._jev.backend, "aclose", None) if self._jev else None
        if close is not None:
            await close()

    def close(self) -> None:
        """Close the Jev client and the private loop used by ``extract_sync``."""
        loop = self._sync_loop
        if loop is not None and not loop.is_closed():
            loop.run_until_complete(self.aclose())
            loop.close()
        self._sync_loop = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
