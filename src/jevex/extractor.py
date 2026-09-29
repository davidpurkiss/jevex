"""The consumer entry point: ``Extractor(schemas=[...]).extract(document)``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, overload

from pydantic import BaseModel, ConfigDict, Field

from jevex.categorise import CategoriseStage
from jevex.clean import CleanStage
from jevex.component_gate import ComponentGateStage
from jevex.gate import DocumentGateStage
from jevex.interfaces import GateDecision
from jevex.jev import JevClient
from jevex.keypaths import StructuredStage
from jevex.layout import LayoutStage
from jevex.normalise import NormaliseStage
from jevex.pipeline import Context, Pipeline
from jevex.resolve import EntityStage
from jevex.results import Extracted, FieldMeta, build_extracted, select_records
from jevex.schema import SchemaSpec
from jevex.select import CandidateStage, SelectStage
from jevex.split import StatementStage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from jevex.document import Document
    from jevex.pipeline import SchemaRun, Stage

# The spec's stage order (see jevex.interfaces). Default stages must use these names; they
# are sorted into this order, so issues can add their stage without coordinating position.
STAGE_ORDER: tuple[str, ...] = (
    "fetch",
    "clean",
    "document_gate",
    "structured",
    "layout",
    "images",
    "component_gate",
    "entities",
    "statements",
    "categorise",
    "candidates",
    "select",
    "normalise",
    "fallback",
    "learn",
)

# Default stages, one per line to keep merges clean. Order here doesn't matter: see
# STAGE_ORDER.
DEFAULT_STAGES: tuple[Stage, ...] = (
    CleanStage(),
    DocumentGateStage(),
    StructuredStage(),
    LayoutStage(),
    ComponentGateStage(),
    StatementStage(),
    EntityStage(),
    CategoriseStage(),
    CandidateStage(),
    SelectStage(),
    NormaliseStage(),
)


def default_pipeline() -> Pipeline:
    """The pipeline ``Extractor`` builds when none is given, in :data:`STAGE_ORDER`."""
    unknown = [s.name for s in DEFAULT_STAGES if s.name not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"default stages {unknown} are not in STAGE_ORDER {STAGE_ORDER}")
    return Pipeline(sorted(DEFAULT_STAGES, key=lambda s: STAGE_ORDER.index(s.name)))


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


@dataclass(frozen=True)
class ExtractionResult:
    """What ``extract`` returns: typed records plus document-level metadata.

    ``records`` holds one :class:`~jevex.results.Extracted` per entity per schema, in schema
    registration order. Use ``for_schema(Model)`` for typed access, or ``one()`` for
    single-entity documents.
    """

    records: list[Extracted[BaseModel]]
    meta: DocumentMeta

    @overload
    def for_schema[T: BaseModel](self, model: type[T]) -> list[Extracted[T]]: ...
    @overload
    def for_schema(self, model: None = None) -> list[Extracted[BaseModel]]: ...
    def for_schema(self, model: type[BaseModel] | None = None) -> list[Extracted[Any]]:
        return select_records(self.records, model)

    @overload
    def one[T: BaseModel](self, model: type[T]) -> Extracted[T]: ...
    @overload
    def one(self, model: None = None) -> Extracted[BaseModel]: ...
    def one(self, model: type[BaseModel] | None = None) -> Extracted[Any]:
        """The only record (of ``model``, if given). Raises ``LookupError`` otherwise."""
        found = select_records(self.records, model)
        if len(found) != 1:
            which = f" for {model.__name__}" if model else ""
            raise LookupError(f"expected exactly one record{which}, found {len(found)}")
        return found[0]

    @property
    def values(self) -> dict[str, dict[str, dict[str, Any]]]:
        """The records' values (as coerced into them), by schema, entity and field.

        Only fields that made it into a record: not filtered, not rejected by the type.
        """
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for r in self.records:
            found = {name: getattr(r.record, name) for name in r.record.model_fields_set}
            if found:
                out.setdefault(r.schema_name, {})[r.entity] = found
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [r.to_dict() for r in self.records],
            "meta": self.meta.model_dump(mode="json"),
        }

    @classmethod
    def from_context(
        cls,
        ctx: Context,
        *,
        threshold: float = 0.0,
        thresholds: Mapping[str, float] | None = None,
    ) -> ExtractionResult:
        usage = ctx.jev.usage
        records: list[Extracted[BaseModel]] = []
        for run in ctx.schemas.values():
            for entity, metas in _entity_metas(run).items():
                records.append(
                    build_extracted(
                        run.spec, entity, metas, threshold=threshold, thresholds=thresholds
                    )
                )
        return cls(
            records=records,
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


def _entity_metas(run: SchemaRun) -> dict[str, dict[str, FieldMeta]]:
    """Field metadata per entity, in scope order; bare ``values`` fill any gaps."""
    order = [s.label for s in run.scopes]
    labels = [*order, *(k for k in [*run.fields, *run.values] if k not in order)]
    out: dict[str, dict[str, FieldMeta]] = {}
    for label in dict.fromkeys(labels):
        metas = dict(run.fields.get(label, {}))
        for name, value in run.values.get(label, {}).items():
            metas.setdefault(name, FieldMeta(value=value))
        if any(m.found or m.alternatives for m in metas.values()):
            out[label] = metas
    return out


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
        threshold: float = 0.0,
        thresholds: Mapping[str, float] | None = None,
    ) -> None:
        """``threshold`` (default 0: keep everything) and per-field ``thresholds`` (keys
        ``"field"`` or ``"Schema.field"``) set the confidence below which a value becomes
        ``None`` in the record. It stays in ``meta`` with ``filtered=True``."""
        if not schemas:
            raise ValueError("register at least one schema")
        self.schemas = [SchemaSpec.from_model(m) for m in schemas]
        names = [s.name for s in self.schemas]
        if len(set(names)) != len(names):
            raise ValueError(f"schema names must be unique: {names}")
        self.pipeline = pipeline if pipeline is not None else default_pipeline()
        self.threshold = threshold
        self.thresholds = dict(thresholds or {})
        known = {f.name for s in self.schemas for f in s.fields} | {
            f"{s.name}.{f.name}" for s in self.schemas for f in s.fields
        }
        unknown = sorted(set(self.thresholds) - known)
        if unknown:
            raise ValueError(f"thresholds for unknown fields: {unknown}")
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
        return ExtractionResult.from_context(
            ctx, threshold=self.threshold, thresholds=self.thresholds
        )

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
