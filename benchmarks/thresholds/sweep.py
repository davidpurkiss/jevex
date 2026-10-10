"""Sweep the extraction thresholds over the synthetic test site (#49, #297) or a corpus
directory (#296).

Live and billed: run only with keys, under spend caps. Live passes ask Jev and the
fallback LLM everything the grid's most permissive setting for each ``ALSO_CATEGORY_P`` and
each component gate threshold would ask (:data:`PERMISSIVE`), and cache every answer per
question (Jev) and per prompt (LLM). For a given ``ALSO_CATEGORY_P`` and gate threshold, a
lower category threshold and a higher fallback threshold only ask more, so every grid
setting should ask a subset of what's cached. Across ``ALSO_CATEGORY_P`` that doesn't hold
(a field another route already found confidently isn't asked about), nor across gate
thresholds (the categorise Choice offers only the fields the gate passed, so its options
change), hence one live pass per value. The grid then replays offline from the cache, end
to end through the default pipeline with ``MultiEntity``, scored with
:func:`jevex.score_result`. Every row reports its cache misses: a Jev miss fails a stage,
and an LLM miss is caught by the fallback and scored as no answer, so a row is only valid
with none::

    # live: fill the cache, one pass per ALSO_CATEGORY_P and gate threshold (the first, over
    # 20 pages, cost about $0.02 Jev and $0.57 LLM; later passes reuse cached answers and
    # cost far less, about $0.03 LLM each)
    (set -a; . .env; set +a; ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_FOR_TESTS" \\
        JEVEX_JEV_MAX_COST_USD=0.50 JEVEX_LLM_MAX_COST_USD=2 \\
        uv run python benchmarks/thresholds/sweep.py --live --cache /tmp/sweep-7 --seed 7)
    # offline: the grid, from the cache
    uv run python benchmarks/thresholds/sweep.py --cache /tmp/sweep-7 --seed 7 --out grid.json
    # live, a few settings only (a validation seed)
    ... sweep.py --live --cache /tmp/sweep-11 --seed 11 --only 0.5,0.5,0.8,0.3,0.3 --only ...
    # a corpus directory (truth.json and its documents) instead of the test site, with the
    # benchmark's fallback model (benchmarks/config.yaml)
    ... sweep.py --live --cache /tmp/sweep-real --corpus DIR --model claude-haiku-4-5-20251001

Swept: the fallback's ``category_threshold``, ``fallback_threshold`` and
``verify_threshold``, ``jevex.select.ALSO_CATEGORY_P`` (a statement's second fields), and
the component gate's ``threshold``.
``learn_threshold`` is read off the same runs: how often the fallback's verified values
are right, by verification probability. A Jev answer is assumed not to depend on the other
questions sent with it, which #6 measured (``docs/jev-limits.md``).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, TypeAdapter

import jevex.select
from jevex import (
    ComponentGateStage,
    EntityStage,
    EvalReport,
    Extractor,
    FallbackStage,
    MultiEntity,
    NoulComponentGate,
    load_corpus,
    score_result,
)
from jevex.document import Document
from jevex.eval import CorpusItem, match_records, resolve_tolerances, score_value
from jevex.extractor import default_pipeline
from jevex.jev import (
    Answer,
    JevBackend,
    JevClient,
    JevResponse,
    JSONContent,
    Question,
    TypeSafeBackend,
)
from jevex.llm import ANTHROPIC_MODEL, LLM, LLMImage, LLMResponse, LLMUsage
from jevex.testsite import build
from jevex.testsite.schemas import Listing, VehicleSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jevex import ExtractionResult
    from jevex.eval import Tolerance
    from jevex.llm.anthropic import AnthropicLLM

SCHEMAS: dict[str, type[BaseModel]] = {s.__name__: s for s in (VehicleSpec, Listing)}
"""The schemas a corpus may be labelled in: the test site's (and the spec-sheets corpus's)."""
FAMILIES = ("table", "kv", "prose", "grid", "listing")
"""HTML only: their bytes are the same on every machine, so the cache keys are too."""
CATEGORY = (0.3, 0.4, 0.5, 0.6, 0.7)
FALLBACK = (0.3, 0.5, 0.7, 0.9)
VERIFY = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
ALSO = (0.1, 0.2, 0.3, 0.4, 0.5)
GATE = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

type Setting = tuple[float, float, float, float, float]
"""(category, fallback, verify, also, gate)."""

OLD_DEFAULTS: Setting = (0.5, 0.5, 0.8, 0.3, 0.3)
"""The defaults before #49."""
CHOSEN: Setting = (0.5, 0.3, 0.7, 0.1, 0.1)
"""The defaults #49 and #297 chose (``docs/thresholds.md``)."""
PERMISSIVE: list[Setting] = list(
    dict.fromkeys(
        [(min(CATEGORY), max(FALLBACK), 0.0, also, CHOSEN[4]) for also in ALSO]
        + [(min(CATEGORY), max(FALLBACK), 0.0, CHOSEN[3], gate) for gate in GATE]
    )
)
"""For each ``ALSO_CATEGORY_P`` at the chosen gate threshold, and each gate threshold at the
chosen ``ALSO_CATEGORY_P``, the setting that asks a superset of the others' questions. A
verify threshold of 0 keeps every verified LLM value, so the calibration sees them all."""

_ANSWER: TypeAdapter[Answer] = TypeAdapter(Answer)
_QUESTION: TypeAdapter[Question] = TypeAdapter(Question)


class CacheMissError(LookupError):
    """An offline run asked something the live pass didn't."""


def _key(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class QuestionCache:
    """A Jev backend answering each question from a per-question cache. Misses go to
    ``inner`` (only they are charged), or raise :class:`CacheMissError` without one."""

    def __init__(self, path: Path, inner: JevBackend | None) -> None:
        self.path = path
        self.inner = inner
        self.entries: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        self.misses = 0
        """Questions an offline run asked that weren't cached."""

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        keys = {
            n: _key([state, _QUESTION.dump_python(q, mode="json")]) for n, q in questions.items()
        }
        missing = {n: q for n, q in questions.items() if keys[n] not in self.entries}
        tokens = 0
        if missing:
            if self.inner is None:
                self.misses += len(missing)
                raise CacheMissError(f"{len(missing)} question(s) not in {self.path}")
            response = await self.inner.system_one(state, missing)
            tokens = response.input_tokens or 0
            for name, answer in response.answers.items():
                self.entries[keys[name]] = _ANSWER.dump_python(answer, mode="json")
        answers = {n: _ANSWER.validate_python(self.entries[keys[n]]) for n in questions}
        return JevResponse(answers=answers, input_tokens=tokens)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, sort_keys=True) + "\n")


class PromptCache:
    """An LLM answered from a per-prompt cache; misses go to ``inner`` (or raise). A cached
    answer reports its original usage, so offline runs still price each setting.

    Every setting's extractor shares the live clients, so :func:`main`, which made them,
    closes them once at the end."""

    def __init__(self, path: Path, inner: LLM | None) -> None:
        self.path = path
        self.inner = inner
        self.entries: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        self.misses = 0
        """Prompts an offline run sent that weren't cached."""

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        key = _key([prompt, schema.model_json_schema(), [i.content.hex() for i in images]])
        if key not in self.entries:
            if self.inner is None:
                self.misses += 1
                raise CacheMissError(f"an LLM prompt not in {self.path}")
            response = await self.inner.structured(prompt, schema, images=images)
            u = response.usage
            self.entries[key] = {
                "output": response.output.model_dump(mode="json"),
                "usage": [u.input_tokens, u.output_tokens, u.cost],
                "model": response.model,
            }
        entry = self.entries[key]
        i, o, c = entry["usage"]
        return LLMResponse(
            output=schema.model_validate(entry["output"]),
            usage=LLMUsage(i, o, c),
            model=entry["model"],
        )

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, sort_keys=True) + "\n")


def testsite_corpus(root: Path, seed: int, per_family: int) -> list[CorpusItem]:
    """The first ``per_family`` pages of each HTML family of ``seed``, interleaved."""
    manifest = build(seed, root / "site", waves=[list(FAMILIES)])
    by_family = {f: [p for p in manifest["pages"] if p["family"] == f] for f in FAMILIES}
    pages = [
        p
        for group in itertools.zip_longest(*(by_family[f][:per_family] for f in FAMILIES))
        for p in group
        if p is not None
    ]
    (root / "site" / "truth.json").write_text(json.dumps(manifest | {"pages": pages}))
    return load_corpus(root / "site")


def document_url(item: CorpusItem, root: Path) -> str:
    """The document's path under the corpus root: cache keys hold it (through the
    statements' state), so they match across machines and checkouts."""
    return item.path.relative_to(root).as_posix()


def llm_values(
    item: CorpusItem,
    url: str,
    result: ExtractionResult,
    tolerances: Mapping[str, Mapping[str, Tolerance]],
) -> dict[tuple[str, str, str, str], tuple[float, bool]]:
    """(verification p, right?) per value the fallback answered, keyed by document,
    statement, field and value: an answer copied into several records (shared, or onto
    extra entities) counts once, and is judged only in records paired with an expected
    one. Extra records are entity resolution's error, not the LLM's."""
    own = tolerances[item.schema]
    found = [
        {
            "entity": r.entity,
            "values": {n: getattr(r.record, n) for n in r.record.model_fields_set},
            "meta": r.meta,
        }
        for r in result.records
        if r.schema_name == item.schema
    ]
    out: dict[tuple[str, str, str, str], tuple[float, bool]] = {}
    for exp, rec in match_records(item.records, found, own):
        if exp is None or rec is None:
            continue
        for name, meta in rec["meta"].items():
            if meta.method != "llm" or not meta.found or meta.confidence is None:
                continue
            sid = meta.source.statement_id if meta.source else ""
            right = score_value(exp.values.get(name), meta.value, own[name]).correct > 0
            key = (url, sid, name, repr(meta.value))
            # Right in any record it was paired into counts as right.
            out[key] = (meta.confidence, right or out.get(key, (0.0, False))[1])
    return out


async def run_setting(
    items: list[CorpusItem],
    root: Path,
    jev: QuestionCache,
    llm: PromptCache,
    setting: Setting,
) -> dict[str, Any]:
    category, fallback, verify, also, gate = setting
    jev.misses = llm.misses = 0
    pipeline = (
        default_pipeline()
        .replace("component_gate", ComponentGateStage(NoulComponentGate(threshold=gate)))
        .replace("entities", EntityStage(MultiEntity()))
        .replace(
            "fallback",
            FallbackStage(
                category_threshold=category,
                fallback_threshold=fallback,
                verify_threshold=verify,
            ),
        )
    )
    report = EvalReport()
    calibration: dict[tuple[str, str, str, str], tuple[float, bool]] = {}
    # A fresh in-memory store, one document at a time in order: the structured stage's
    # learned key mappings then reach later documents the same way in every setting.
    library_also = jevex.select.ALSO_CATEGORY_P
    jevex.select.ALSO_CATEGORY_P = also  # only select.field_statements reads it, per call
    try:
        async with Extractor(
            corpus_schemas(items),
            jev=JevClient(jev),
            pipeline=pipeline,
            extraction_llm=llm,
            store=":memory:",
            community_packs=False,
        ) as extractor:
            tolerances = resolve_tolerances(extractor)
            for item in items:
                url = document_url(item, root)
                document = Document.from_path(item.path, url=url, locale=item.locale)
                start = time.perf_counter()
                result = await extractor.extract(document)
                report.documents.append(
                    score_result(item, result, time.perf_counter() - start, tolerances)
                )
                calibration |= llm_values(item, url, result, tolerances)
    finally:
        jevex.select.ALSO_CATEGORY_P = library_also
    summary = report.summary()
    return {
        "category_threshold": category,
        "fallback_threshold": fallback,
        "verify_threshold": verify,
        "also_category_p": also,
        "component_gate_threshold": gate,
        "summary": summary,
        "fields": {k: v.to_dict() for k, v in report.field_scores().items()},
        "calibration": [
            {"page": page, "statement_id": sid, "field": f, "value": v, "p": p, "right": ok}
            for (page, sid, f, v), (p, ok) in calibration.items()
        ],
        "cache_misses": {"jev": jev.misses, "llm": llm.misses},
        "failed": {d.path: d.error for d in report.documents if d.status == "failed"},
        "partial": {d.path: d.warnings for d in report.documents if d.status == "partial"},
        "methods": dict(sum((d.methods for d in report.documents), Counter[str]())),
    }


def corpus_schemas(items: Sequence[CorpusItem]) -> list[type[BaseModel]]:
    """The schemas ``items`` are labelled in, so a corpus labelled in one isn't asked about
    the other."""
    return [SCHEMAS[name] for name in dict.fromkeys(item.schema for item in items)]


def grid() -> list[Setting]:
    """The fallback's three thresholds against each other at the old and the chosen
    ``ALSO_CATEGORY_P``, then ``ALSO_CATEGORY_P`` alone at the old and the chosen other
    thresholds, all at the chosen gate threshold; then the gate's threshold against the
    fallback threshold at the chosen others: a field the gate drops can still reach the
    fallback."""
    gate = CHOSEN[4]
    out: list[Setting] = [
        (c, f, v, a, gate)
        for a in (OLD_DEFAULTS[3], CHOSEN[3])
        for c, f, v in itertools.product(CATEGORY, FALLBACK, VERIFY)
    ]
    out += [(*base[:3], a, gate) for base in (OLD_DEFAULTS, CHOSEN) for a in ALSO]
    out += [(CHOSEN[0], f, CHOSEN[2], CHOSEN[3], g) for f in FALLBACK for g in GATE]
    return list(dict.fromkeys(out))


def _setting(text: str) -> Setting:
    c, f, v, a, g = (float(x) for x in text.split(","))
    return c, f, v, a, g


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Sweep jevex's extraction thresholds.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--live", action="store_true", help="Fill the cache from real APIs")
    parser.add_argument(
        "--corpus", type=Path, help="A corpus directory (truth.json) instead of the test site"
    )
    parser.add_argument("--seed", type=int, default=7, help="The test site's seed")
    parser.add_argument("--per-family", type=int, default=4, help="Test-site pages per family")
    parser.add_argument("--model", default=ANTHROPIC_MODEL, help="The fallback LLM (Anthropic)")
    parser.add_argument(
        "--only",
        type=_setting,
        action="append",
        help="category,fallback,verify,also,gate: run just these settings (repeatable)",
    )
    args = parser.parse_args(argv)

    args.cache.mkdir(parents=True, exist_ok=True)
    out_path = args.out.resolve() if args.out else None
    # The test site is built in the cache directory, and its URLs keep the "site/" prefix
    # earlier caches were keyed by.
    root = args.corpus.resolve() if args.corpus else Path(".")
    os.chdir(args.cache)
    items = (
        load_corpus(root) if args.corpus else testsite_corpus(Path("."), args.seed, args.per_family)
    )
    live: tuple[TypeSafeBackend, AnthropicLLM] | None = None
    if args.live:
        from jevex.llm.anthropic import AnthropicLLM

        live = (TypeSafeBackend(), AnthropicLLM(args.model))
    jev = QuestionCache(Path("jev.json"), live[0] if live else None)
    llm = PromptCache(Path("llm.json"), live[1] if live else None)
    settings = args.only or (PERMISSIVE if args.live else grid())
    rows: list[dict[str, Any]] = []
    try:
        for setting in settings:
            rows.append(await run_setting(items, root, jev, llm, setting))
            print(setting, rows[-1]["summary"]["accuracy"], file=sys.stderr, flush=True)
    finally:
        jev.save()
        llm.save()
        if live is not None:
            await live[0].aclose()
            await live[1].aclose()
    out = {
        "corpus": args.corpus.name if args.corpus else f"testsite seed {args.seed}",
        "model": args.model,
        "pages": [document_url(i, root) for i in items],
        "rows": rows,
    }
    text = json.dumps(out, indent=1, default=str)
    if out_path:
        out_path.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
