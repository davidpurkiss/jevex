"""Baselines jevex is benchmarked against (``docs/benchmarks.md`` › *Systems compared*, #62).

A baseline is any :class:`BaselineSystem`: given a document, it returns the records it
found and the LLM usage that cost. Two kinds are compared with jevex:

- **LLM-only** (:class:`LLMBaseline`): one structured-output call per document, through
  jevex's own adapters (:func:`pinned_llm`), so token usage, prices and the spend cap work
  as they do for jevex. The model sees the whole document as text and returns every
  record of every schema (:func:`records_model`).
- **Open-source tools** (ScrapeGraphAI, Crawl4AI): scripts in ``benchmarks/baselines/``
  that each run in their own environment and implement the same protocol, charging their
  token usage through :func:`charge_usage`.

Every system sees the same input (:func:`prepare_input`): the document after the clean
stage jevex runs with (a site's cleaner too), and the text jevex's layout and image
stages read from it (:func:`render_text`), so a comparison is about extraction rather than
input quality. Every system gets the same instructions too: the prompt in
``benchmarks/baselines/`` with the schemas written out (:func:`baseline_instructions`).

:func:`run_baseline` runs a system over a corpus and writes one :class:`ResultRow` per
document to a JSONL results file as each finishes, so a run stopped by the spend cap keeps
what it paid for. :func:`score_results` scores a results file exactly as ``jevex eval``
scores jevex, into an :class:`~jevex.eval.EvalReport`. Costs come from the token usage
each API reported, at the pinned prices (:class:`~jevex.benchmarks.PinnedModel`).
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from jevex._tasks import gather
from jevex.document import Document
from jevex.eval import (
    DocumentRun,
    EvalReport,
    all_missing,
    check_schemas,
    load_corpus,
    schema_tolerances,
    score_records,
)
from jevex.fallback import field_type_text
from jevex.jev import JevClient
from jevex.llm import LLMBudgetExceededError, LLMUsage, check_budget, record
from jevex.pipeline import Context
from jevex.results import partial_model
from jevex.schema import SchemaSpec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

    from jevex.benchmarks import PinnedModel
    from jevex.eval import CorpusItem, Tolerance
    from jevex.jev import JevResponse, JSONContent, Question
    from jevex.layout import Component
    from jevex.llm import LLM
    from jevex.pipeline import Pipeline
    from jevex.schema import FieldSpec

INPUT_STAGES = ("clean", "layout", "images")
"""The stages whose output every baseline reads: the cleaned document and its text."""

SCHEMAS_PLACEHOLDER = "{schemas}"


class BaselineRunError(Exception):
    """A baseline can't be run or scored as configured (no input text, an unpriced model,
    a malformed results file). Stops the run instead of being recorded per document."""


RUN_ERRORS: tuple[type[Exception], ...] = (LLMBudgetExceededError, BaselineRunError)
"""Errors that stop :func:`run_baseline` instead of being recorded on one document, raised
by the system or by preparing an input (a document no parser reads)."""


# --- input -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineInput:
    """What a baseline gets for one document."""

    document: Document
    """The document after jevex's clean stage (boilerplate removed, a site's cleaner run)."""
    text: str
    """The text jevex's layout and image stages read from it (:func:`render_text`)."""


class _NoJev:
    """The input stages never ask Jev; a stage that does is a bug here, not a cost."""

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        raise BaselineRunError("a baseline input stage asked Jev a question")


async def prepare_input(document: Document, pipeline: Pipeline | None = None) -> BaselineInput:
    """Run ``pipeline``'s clean, layout and image stages (:data:`INPUT_STAGES`; default:
    jevex's own) over ``document`` and render what they read.

    Raises :class:`BaselineRunError` when no layout parser reads the document (a PDF without
    the ``pdf`` extra), since a baseline given no text would be scored on nothing.
    """
    if pipeline is None:
        from jevex.extractor import default_pipeline  # the extractor imports eval

        pipeline = default_pipeline()
    ctx = Context.create(document, [], JevClient(_NoJev()))
    for stage in pipeline:
        if stage.name in INPUT_STAGES:
            await stage.run(ctx)
    if ctx.parsed is None:
        reasons = [e.message for e in ctx.events if e.kind == "layout_skipped"]
        raise BaselineRunError(
            f"no text for {document.url or 'the document'}: {'; '.join(reasons) or 'not parsed'}"
        )
    return BaselineInput(document=ctx.document, text=render_text(ctx.parsed.root))


_PREFIX = {"heading": "# ", "list_item": "- "}


def render_text(root: Component) -> str:
    """A layout tree as plain text, one block per component in reading order, blank lines
    between: headings start ``# ``, list items ``- ``, tables are their rows (cells joined
    ``" | "``), an image is ``[Image: <alt text>]`` followed by the text read from it."""
    blocks: list[str] = []
    for component in root.walk():
        text = component.text.strip()
        if not text:
            continue
        if component.type == "image":
            text = f"[Image: {text}]"
        blocks.append(_PREFIX.get(component.type, "") + text)
    return "\n\n".join(blocks)


# --- instructions and output -----------------------------------------------------------


def load_prompt(path: str | Path) -> str:
    """Read a prompt template; it must contain ``{schemas}`` once.

    Raises :class:`BaselineRunError` naming the file.
    """
    try:
        template = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise BaselineRunError(f"can't read prompt {path}: {exc.strerror or exc}") from exc
    if template.count(SCHEMAS_PLACEHOLDER) != 1:
        raise BaselineRunError(f"{path} must contain {SCHEMAS_PLACEHOLDER} exactly once")
    return template


def baseline_instructions(template: str, schemas: Sequence[SchemaSpec]) -> str:
    """``template`` with ``{schemas}`` replaced by :func:`schemas_text`. The same
    instructions go to every baseline; the LLM-only one adds the document after them."""
    return template.replace(SCHEMAS_PLACEHOLDER, schemas_text(schemas)).strip()


def schemas_text(schemas: Sequence[SchemaSpec]) -> str:
    """Each schema as a heading, its description and one line per field: name,
    description, unit and type, in the words jevex's fallback prompt uses."""
    parts: list[str] = []
    for spec in schemas:
        lines = [f"## {spec.name}", spec.description]
        for f in spec.fields:
            unit = f", in {f.unit}" if f.unit else ""
            kind = "an object" if f.kind == "model" else field_type_text(f)
            lines.append(f"- {f.name}: {f.description}{unit}. {kind[0].upper()}{kind[1:]}.")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


_RECORDS: dict[tuple[type[BaseModel], ...], type[BaseModel]] = {}
_RECORD: dict[type[BaseModel], type[BaseModel]] = {}


def record_model(spec: SchemaSpec) -> type[BaseModel]:
    """One record of ``spec`` in :func:`records_model`'s output. Cached per schema."""
    if spec.model not in _RECORD:
        values: dict[str, Any] = {
            f.name: (_json_type(f) | None, Field(default=None, description=_describe(f)))
            for f in spec.fields
        }
        _RECORD[spec.model] = create_model(
            f"{spec.name}Record",
            __config__=ConfigDict(protected_namespaces=()),
            __doc__=spec.description,
            **values,
        )
    return _RECORD[spec.model]


def records_model(schemas: Sequence[SchemaSpec]) -> type[BaseModel]:
    """The structured output a baseline returns: one list of records per schema, under
    the schema's name. Every field is optional (``null`` when the document doesn't state
    it) and has a JSON-friendly type: numbers as numbers, dates as ``YYYY-MM-DD`` strings,
    enums as their options, a nested model as its partial. Cached per set of schemas."""
    key = tuple(s.model for s in schemas)
    if key not in _RECORDS:
        lists: dict[str, Any] = {
            spec.name: (
                list[record_model(spec)],
                Field(default_factory=list, description=f"Every {spec.name} in the document"),
            )
            for spec in schemas
        }
        _RECORDS[key] = create_model("Records", **lists)
    return _RECORDS[key]


def _json_type(spec: FieldSpec) -> Any:
    item: Any
    if spec.kind == "enum":
        item = Literal[spec.options]  # pyright: ignore[reportInvalidTypeForm]
    elif spec.kind == "model" and spec.model is not None:
        item = partial_model(spec.model)
    else:
        item = {"number": float, "date": str, "str": str, "bool": bool}.get(spec.kind, str)
    return list[item] if spec.many else item


def _describe(spec: FieldSpec) -> str:
    return f"{spec.description}, in {spec.unit}" if spec.unit else spec.description


def found_records(output: BaseModel) -> dict[str, list[dict[str, Any]]]:
    """A :func:`records_model` output as :func:`~jevex.eval.score_records` takes it: per
    schema, ``{"entity": "<n>", "values": {...}}`` with only the values found (not
    ``null``), numbered in the order returned. A record with no values is dropped."""
    out: dict[str, list[dict[str, Any]]] = {}
    for schema, records in output.model_dump(mode="json").items():
        kept: list[dict[str, Any]] = []
        for record_ in cast("list[dict[str, Any]]", records):
            values = {k: v for k, v in record_.items() if v is not None}
            if values:
                kept.append({"entity": str(len(kept) + 1), "values": values})
        out[schema] = kept
    return out


def lenient_records(
    data: Mapping[str, Any], schemas: Sequence[SchemaSpec]
) -> dict[str, list[dict[str, Any]]]:
    """:func:`found_records` for a tool's raw output: a dict shaped like
    :func:`records_model`'s, that the tool didn't validate. Each value is checked against
    its field alone, so one that doesn't fit ("NA", a word for a number) is dropped
    (scored missing) without losing the rest of its record. Keys that aren't a schema or
    a field, and records that aren't objects, are ignored."""
    out: dict[str, list[dict[str, Any]]] = {}
    for spec in schemas:
        model = record_model(spec)
        raw = data.get(spec.name)
        kept: list[dict[str, Any]] = []
        for item in cast("list[Any]", raw) if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            values: dict[str, Any] = {}
            for name, value in cast("dict[str, Any]", item).items():
                if name not in model.model_fields or value is None:
                    continue
                try:
                    parsed = model.model_validate({name: value}).model_dump(mode="json")[name]
                except ValidationError:
                    continue
                if parsed is not None:
                    values[name] = parsed
            if values:
                kept.append({"entity": str(len(kept) + 1), "values": values})
        out[spec.name] = kept
    return out


# --- systems ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineOutput:
    """What a baseline found in one document and what finding it cost."""

    records: dict[str, list[dict[str, Any]]]
    """Per schema name, ``{"entity": ..., "values": {...}}`` (:func:`found_records`)."""
    calls: int
    input_tokens: int
    output_tokens: int
    cost: float
    """USD at the pinned prices, from the token usage the API reported."""
    model: str | None = None
    """The model that served the calls, as the API named it."""


class BaselineSystem(Protocol):
    """One system to compare: it extracts every schema's records from a document."""

    @property
    def name(self) -> str: ...

    async def extract(self, source: BaselineInput) -> BaselineOutput: ...


@dataclass(frozen=True)
class BaselineSetup:
    """What a :class:`BaselineSystem` is built from (see ``jevex baseline``)."""

    schemas: tuple[SchemaSpec, ...]
    instructions: str
    """:func:`baseline_instructions` for these schemas."""
    model: PinnedModel
    """The LLM the system uses, at its pinned version and prices."""

    @property
    def records_model(self) -> type[BaseModel]:
        return records_model(self.schemas)


@dataclass
class LLMBaseline:
    """LLM-only extraction: the instructions, then the document's text in ``<document>``
    tags, in one structured-output call to ``llm`` returning :func:`records_model`.

    The call's cost is what the adapter computed from the API's token usage; an adapter
    that can't price its model (``usage.cost is None``) stops the run with a
    :class:`BaselineRunError`, because its costs would read as free.
    """

    llm: LLM
    schemas: Sequence[SchemaSpec]
    instructions: str
    name: str = "llm-only"

    async def extract(self, source: BaselineInput) -> BaselineOutput:
        prompt = f"{self.instructions}\n\n<document>\n{source.text}\n</document>"
        response = await self.llm.structured(prompt, records_model(self.schemas))
        usage = response.usage
        if usage.cost is None:
            raise BaselineRunError(f"no price for {response.model}: pass the adapter prices=")
        return BaselineOutput(
            records=found_records(response.output),
            calls=1,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost=usage.cost,
            model=response.model,
        )


def pinned_llm(pinned: PinnedModel) -> LLM:
    """jevex's adapter for a pinned model, costing every call at its pinned prices.

    Anthropic's server-side refusal fallback is off, so every call is served by the pinned
    model. Needs the provider's extra and API key. Raises ``ValueError`` for Jev.
    """
    prices = pinned.prices()
    if pinned.provider == "anthropic":
        from jevex.llm.anthropic import AnthropicLLM

        return AnthropicLLM(pinned.model, prices=prices, fallbacks=False)
    if pinned.provider == "gemini":
        from jevex.llm.gemini import GeminiLLM

        return GeminiLLM(pinned.model, prices=prices)
    if pinned.provider == "openai":
        from jevex.llm.openai import OpenAILLM

        return OpenAILLM(pinned.model, prices=prices)
    if pinned.provider == "litellm":
        from jevex.llm.litellm import LiteLLM

        return LiteLLM(pinned.model, prices=prices)
    raise ValueError(f"{pinned.spec} is Jev, not an LLM")


def charge_usage(pinned: PinnedModel, input_tokens: int, output_tokens: int) -> float:
    """Record usage an outside tool reported, at ``pinned``'s prices, against
    ``JEVEX_LLM_MAX_COST_USD`` (and the shared ledger), as jevex's adapters do; returns
    the USD. Call :func:`~jevex.llm.check_budget` before each document."""
    usd = pinned.cost(input_tokens, output_tokens)
    record(LLMUsage(input_tokens, output_tokens, usd))
    return usd


# --- running ---------------------------------------------------------------------------


class ResultRow(BaseModel):
    """One document's result in a results file (one JSON object per line)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    """The document's path as ``truth.json`` lists it."""
    records: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict[str, list[dict[str, Any]]]
    )
    """Per schema name, the records found (:attr:`BaselineOutput.records`)."""
    seconds: float
    """Wall time extracting. Preparing the input is the same for every system and isn't
    timed."""
    calls: int = 0
    """LLM calls."""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    """The LLM calls' USD at the pinned prices. A failed call's cost is in the spend
    ledger, not here."""
    model: str | None = None
    error: str | None = None
    jev_requests: int = 0
    """jevex's own rows (:func:`result_row`) carry its Jev usage, the methods its values came
    from and its learner's spend; a baseline's leave them empty."""
    jev_questions: int = 0
    jev_cost: float = 0.0
    """Jev's USD (its input tokens at Jev's price)."""
    learning_cost: float = 0.0
    """What the learner spent learning from this document's examples (Jev and LLM), when
    it ran one document at a time (:func:`~jevex.replay.replay`)."""
    methods: dict[str, int] | None = None
    """How many values each method resolved. ``None`` (a baseline): every value found
    counts as ``llm``."""


def result_row(run: DocumentRun, corpus: str | Path, *, learning_cost: float = 0.0) -> ResultRow:
    """A jevex run of one corpus document (:func:`~jevex.eval.evaluate`'s or
    :func:`~jevex.replay.replay`'s :class:`~jevex.eval.DocumentRun`) as a results-file row,
    so jevex is saved and scored (:func:`score_results`) as the baselines are."""
    return ResultRow(
        path=Path(run.path).relative_to(corpus).as_posix(),
        records=run.records,
        seconds=run.seconds,
        calls=run.llm_calls,
        cost=run.llm_cost,
        error=run.error,
        jev_requests=run.jev_requests,
        jev_questions=run.jev_questions,
        jev_cost=run.jev_cost,
        learning_cost=learning_cost,
        methods=dict(run.methods),
    )


def read_results(path: str | Path) -> list[ResultRow]:
    """Read a results file. Raises :class:`BaselineRunError` naming the file and line."""
    rows: list[ResultRow] = []
    for number, line in _lines(path, "results"):
        try:
            rows.append(ResultRow.model_validate_json(line))
        except ValidationError as exc:
            raise BaselineRunError(f"{path} line {number} isn't a result: {exc}") from exc
    return rows


class InputRow(BaseModel):
    """One document's prepared input in an inputs file (one JSON object per line)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    """The document's path as ``truth.json`` lists it."""
    document: Document
    """:attr:`BaselineInput.document`."""
    text: str
    """:attr:`BaselineInput.text`."""


async def write_inputs(
    corpus: str | Path, out: str | Path, *, pipeline: Pipeline | None = None
) -> list[InputRow]:
    """Prepare every corpus document's input once (:func:`prepare_input` with ``pipeline``)
    and write them to ``out`` as JSONL, for :func:`run_baseline`'s ``inputs``.

    Every system then reads exactly the same input, and a tool's environment needs none of
    the parsers (``pdf``, ``ocr``) that made it. ``out`` must not exist; nothing is written
    unless every document was prepared. Raises ``ValueError`` for a malformed corpus or an
    existing ``out``, :class:`BaselineRunError` for a document no parser reads.
    """
    root = Path(corpus)
    items = await asyncio.to_thread(load_corpus, root)
    target = Path(out)
    if await asyncio.to_thread(target.exists):
        raise ValueError(f"{target} exists; refusing to overwrite an inputs file")
    rows: list[InputRow] = []
    for item in items:
        document = await _load(item)
        source = await prepare_input(document, pipeline)
        rows.append(
            InputRow(path=_relative(item, root), document=source.document, text=source.text)
        )
    text = "".join(row.model_dump_json() + "\n" for row in rows)
    await asyncio.to_thread(target.write_text, text, encoding="utf-8")
    return rows


def read_inputs(path: str | Path) -> dict[str, BaselineInput]:
    """Read an inputs file: each document's path → its input. Raises
    :class:`BaselineRunError` naming the file and line."""
    inputs: dict[str, BaselineInput] = {}
    for number, line in _lines(path, "inputs"):
        try:
            row = InputRow.model_validate_json(line)
        except ValidationError as exc:
            raise BaselineRunError(f"{path} line {number} isn't an input: {exc}") from exc
        inputs[row.path] = BaselineInput(document=row.document, text=row.text)
    return inputs


async def run_baseline(
    system: BaselineSystem,
    corpus: str | Path,
    out: str | Path,
    *,
    inputs: Mapping[str, BaselineInput] | None = None,
    pipeline: Pipeline | None = None,
    concurrency: int = 8,
) -> list[ResultRow]:
    """Run ``system`` over the corpus in ``corpus``, ``concurrency`` documents at a time,
    appending each document's :class:`ResultRow` to ``out`` as it finishes.

    ``inputs`` (:func:`read_inputs`) gives every document's prepared input; without it,
    each is prepared here with ``pipeline``'s input stages (:func:`prepare_input`).
    ``out`` must not exist. A document whose extraction raises gets a row with its
    ``error``; the :data:`RUN_ERRORS` (the spend cap, an unpriced model, inputs missing a
    document) stop the run, leaving the rows written so far. Raises ``ValueError`` for a
    malformed corpus or an existing ``out``.
    """
    root = Path(corpus)
    items = await asyncio.to_thread(load_corpus, root)
    if inputs is not None:
        missing = [p for p in (_relative(i, root) for i in items) if p not in inputs]
        if missing:
            raise BaselineRunError(f"the inputs have no {', '.join(missing[:5])}")
    target = Path(out)
    if await asyncio.to_thread(target.exists):
        raise ValueError(f"{target} exists; refusing to overwrite a results file")
    await asyncio.to_thread(target.touch)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(item: CorpusItem) -> ResultRow:
        async with semaphore:
            path = _relative(item, root)

            async def source() -> BaselineInput:
                if inputs is not None:
                    return inputs[path]
                return await prepare_input(await _load(item), pipeline)

            row = await _run_one(system, path, source)
            await asyncio.to_thread(_append, target, row)
            return row

    return await gather(one(item) for item in items)


def _relative(item: CorpusItem, root: Path) -> str:
    return item.path.relative_to(root).as_posix()


async def _load(item: CorpusItem) -> Document:
    return await asyncio.to_thread(
        Document.from_path, item.path, url=item.path.as_posix(), locale=item.locale
    )


async def _run_one(
    system: BaselineSystem, path: str, source: Callable[[], Awaitable[BaselineInput]]
) -> ResultRow:
    start = time.perf_counter()
    try:
        prepared = await source()
        start = time.perf_counter()
        check_budget()
        output = await system.extract(prepared)
    except RUN_ERRORS:
        raise
    except Exception as exc:  # recorded and scored as all missing; the run carries on
        return ResultRow(
            path=path, seconds=time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}"
        )
    return ResultRow(
        path=path,
        records=output.records,
        seconds=time.perf_counter() - start,
        calls=output.calls,
        input_tokens=output.input_tokens,
        output_tokens=output.output_tokens,
        cost=output.cost,
        model=output.model,
    )


def _lines(path: str | Path, what: str) -> list[tuple[int, str]]:
    """A JSONL file's non-blank lines with their numbers."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BaselineRunError(f"can't read {what} {path}: {exc.strerror or exc}") from exc
    return [(n, line) for n, line in enumerate(lines, start=1) if line.strip()]


def _append(path: Path, row: ResultRow) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(row.model_dump_json() + "\n")


# --- scoring ---------------------------------------------------------------------------


def score_results(
    corpus: str | Path,
    rows: Iterable[ResultRow],
    schemas: Sequence[SchemaSpec],
    *,
    tolerances: Mapping[str, Tolerance] | None = None,
) -> EvalReport:
    """Score a baseline's results against the corpus in ``corpus``, as ``jevex eval``
    scores jevex: the same record matching, tolerances (``tolerances`` overrides them per
    field) and per-document metrics.

    A document without a row, or whose row has an ``error``, scores as all missing (its
    ``error`` says which). A baseline's values count as method ``llm`` in the resolution
    mix; jevex's rows give their own. Raises :class:`BaselineRunError` for a row naming a
    document the corpus doesn't list, or two rows for one document, and ``ValueError`` for
    a corpus labelled in a schema not in ``schemas``.
    """
    root = Path(corpus)
    items = load_corpus(root)
    resolved = schema_tolerances(schemas, tolerances)
    check_schemas(items, resolved)
    by_path: dict[str, ResultRow] = {}
    for row in rows:
        if row.path in by_path:
            raise BaselineRunError(f"two results for {row.path}")
        by_path[row.path] = row
    listed = {item.path.relative_to(root).as_posix() for item in items}
    unknown = sorted(set(by_path) - listed)
    if unknown:
        raise BaselineRunError(f"results for documents the corpus doesn't list: {unknown[:5]}")
    return EvalReport(documents=[_scored(item, root, by_path, resolved) for item in items])


def _scored(
    item: CorpusItem,
    root: Path,
    rows: Mapping[str, ResultRow],
    tolerances: Mapping[str, Mapping[str, Tolerance]],
) -> DocumentRun:
    row = rows.get(item.path.relative_to(root).as_posix())
    error = "no result" if row is None else row.error
    if row is None or error is not None:
        fields = all_missing(item, tolerances[item.schema])
        methods: Counter[str] = Counter()
    else:
        fields = score_records(item, row.records, tolerances)
        if row.methods is not None:
            methods = Counter(row.methods)
        else:
            found = sum(len(r["values"]) for records in row.records.values() for r in records)
            methods = Counter({"llm": found} if found else {})
    return DocumentRun(
        path=item.path.as_posix(),
        schema=item.schema,
        seconds=row.seconds if row else 0.0,
        jev_requests=row.jev_requests if row else 0,
        jev_questions=row.jev_questions if row else 0,
        jev_cost=row.jev_cost if row else 0.0,
        llm_calls=row.calls if row else 0,
        llm_cost=row.cost if row else 0.0,
        methods=methods,
        fields=fields,
        error=error,
        records=row.records if row else {},
    )


def schema_specs(models: Iterable[type[BaseModel]]) -> tuple[SchemaSpec, ...]:
    """:meth:`SchemaSpec.from_model` for each model."""
    return tuple(SchemaSpec.from_model(m) for m in models)


def summarise_results(rows: Sequence[ResultRow]) -> str:
    """One line for a finished run: documents, failures and cost."""
    failed = sum(1 for r in rows if r.error)
    return f"{len(rows)} documents, {failed} failed, ${sum(r.cost for r in rows):.4f}\n"


__all__ = [
    "INPUT_STAGES",
    "RUN_ERRORS",
    "BaselineInput",
    "BaselineOutput",
    "BaselineRunError",
    "BaselineSetup",
    "BaselineSystem",
    "InputRow",
    "LLMBaseline",
    "ResultRow",
    "baseline_instructions",
    "charge_usage",
    "found_records",
    "lenient_records",
    "load_prompt",
    "pinned_llm",
    "prepare_input",
    "read_inputs",
    "read_results",
    "record_model",
    "records_model",
    "render_text",
    "result_row",
    "run_baseline",
    "schema_specs",
    "schemas_text",
    "score_results",
    "summarise_results",
    "write_inputs",
]
