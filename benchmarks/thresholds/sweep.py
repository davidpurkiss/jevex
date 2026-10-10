"""Sweep the extraction thresholds over the synthetic test site (#49).

Live and billed: run only with keys, under spend caps. One live pass asks Jev and the
fallback LLM everything the grid's most permissive setting would ask, and caches every
answer per question (Jev) and per prompt (LLM). Every other setting asks a subset of
those, so the whole grid then replays offline from the cache, end to end through the
default pipeline with ``MultiEntity``, and is scored with :func:`jevex.score_result`::

    # live: fill the cache (Jev and the fallback LLM; about $1 for 15 pages)
    (set -a; . .env; set +a; ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_FOR_TESTS" \\
        JEVEX_JEV_MAX_COST_USD=0.50 JEVEX_LLM_MAX_COST_USD=2 \\
        uv run python benchmarks/thresholds/sweep.py --live --cache /tmp/sweep-7 --seed 7)
    # offline: the grid, from the cache
    uv run python benchmarks/thresholds/sweep.py --cache /tmp/sweep-7 --seed 7 --out grid.json
    # live, a few settings only (a validation seed)
    ... sweep.py --live --cache /tmp/sweep-11 --seed 11 --only 0.5,0.5,0.8,0.3 --only ...

Swept: the fallback's ``category_threshold``, ``fallback_threshold`` and
``verify_threshold``, and ``jevex.select.ALSO_CATEGORY_P`` (a statement's second fields).
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
    EntityStage,
    EvalReport,
    Extractor,
    FallbackStage,
    MultiEntity,
    load_corpus,
    score_result,
)
from jevex.document import Document
from jevex.eval import CorpusItem, match_records, resolve_tolerances, score_value
from jevex.extractor import default_pipeline
from jevex.jev import Answer, JevBackend, JevClient, JevResponse, JSONContent, Question
from jevex.llm import LLM, LLMImage, LLMResponse, LLMUsage
from jevex.testsite import build
from jevex.testsite.schemas import Listing, VehicleSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from jevex import ExtractionResult
    from jevex.eval import Tolerance

FAMILIES = ("table", "kv", "prose", "grid", "listing")
"""HTML only: their bytes are the same on every machine, so the cache keys are too."""
CATEGORY = (0.3, 0.4, 0.5, 0.6, 0.7)
FALLBACK = (0.3, 0.5, 0.7, 0.9)
VERIFY = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
ALSO = (0.1, 0.2, 0.3, 0.4, 0.5)
PERMISSIVE = (min(CATEGORY), max(FALLBACK), 0.0, min(ALSO))
"""(category, fallback, verify, also): asks a superset of every other setting's questions.
A verify threshold of 0 keeps every verified LLM value, so the calibration sees them all."""
DEFAULTS = (0.5, 0.5, 0.8, 0.3)

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
                raise CacheMissError(f"{len(missing)} question(s) not in {self.path}")
            response = await self.inner.system_one(state, missing)
            tokens = response.input_tokens or 0
            for name, answer in response.answers.items():
                self.entries[keys[name]] = _ANSWER.dump_python(answer, mode="json")
        answers = {n: _ANSWER.validate_python(self.entries[keys[n]]) for n in questions}
        return JevResponse(answers=answers, input_tokens=tokens)

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None)
        if close is not None:
            await close()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, sort_keys=True) + "\n")


class PromptCache:
    """An LLM answered from a per-prompt cache; misses go to ``inner`` (or raise). A cached
    answer reports its original usage, so offline runs still price each setting."""

    def __init__(self, path: Path, inner: LLM | None) -> None:
        self.path = path
        self.inner = inner
        self.entries: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    async def structured[T: BaseModel](
        self, prompt: str, schema: type[T], *, images: Sequence[LLMImage] = ()
    ) -> LLMResponse[T]:
        key = _key([prompt, schema.model_json_schema(), [i.content.hex() for i in images]])
        if key not in self.entries:
            if self.inner is None:
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

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None)
        if close is not None:
            await close()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, sort_keys=True) + "\n")


def corpus(root: Path, seed: int, per_family: int) -> list[CorpusItem]:
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


def llm_values(
    item: CorpusItem, result: ExtractionResult, tolerances: Mapping[str, Mapping[str, Tolerance]]
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
            key = (item.path.as_posix(), sid, name, repr(meta.value))
            # Right in any record it was paired into counts as right.
            out[key] = (meta.confidence, right or out.get(key, (0.0, False))[1])
    return out


async def run_setting(
    items: list[CorpusItem],
    jev: QuestionCache,
    llm: PromptCache,
    setting: tuple[float, float, float, float],
) -> dict[str, Any]:
    category, fallback, verify, also = setting
    jevex.select.ALSO_CATEGORY_P = also
    pipeline = (
        default_pipeline()
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
    async with Extractor(
        [VehicleSpec, Listing],
        jev=JevClient(jev),
        pipeline=pipeline,
        extraction_llm=llm,
        store=":memory:",
        community_packs=False,
    ) as extractor:
        tolerances = resolve_tolerances(extractor)
        for item in items:
            document = Document.from_path(item.path, url=item.path.as_posix())
            start = time.perf_counter()
            result = await extractor.extract(document)
            report.documents.append(
                score_result(item, result, time.perf_counter() - start, tolerances)
            )
            calibration |= llm_values(item, result, tolerances)
    jevex.select.ALSO_CATEGORY_P = DEFAULTS[3]
    summary = report.summary()
    return {
        "category_threshold": category,
        "fallback_threshold": fallback,
        "verify_threshold": verify,
        "also_category_p": also,
        "summary": summary,
        "fields": {k: v.to_dict() for k, v in report.field_scores().items()},
        "calibration": [
            {"page": page, "statement_id": sid, "field": f, "value": v, "p": p, "right": ok}
            for (page, sid, f, v), (p, ok) in calibration.items()
        ],
        "failed": {d.path: d.error for d in report.documents if d.status == "failed"},
        "methods": dict(sum((d.methods for d in report.documents), Counter[str]())),
    }


def grid() -> list[tuple[float, float, float, float]]:
    """The fallback's three thresholds against each other at the default
    ``ALSO_CATEGORY_P``, then ``ALSO_CATEGORY_P`` alone at the other defaults."""
    out = [(c, f, v, DEFAULTS[3]) for c, f, v in itertools.product(CATEGORY, FALLBACK, VERIFY)]
    out += [(*DEFAULTS[:3], a) for a in ALSO if a != DEFAULTS[3]]
    return out


def _setting(text: str) -> tuple[float, float, float, float]:
    c, f, v, a = (float(x) for x in text.split(","))
    return c, f, v, a


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Sweep jevex's extraction thresholds.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--live", action="store_true", help="Fill the cache from real APIs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument(
        "--only",
        type=_setting,
        action="append",
        help="category,fallback,verify,also: run just these settings (repeatable)",
    )
    args = parser.parse_args(argv)

    args.cache.mkdir(parents=True, exist_ok=True)
    out_path = args.out.resolve() if args.out else None
    os.chdir(args.cache)  # relative document URLs, so cached states match across machines
    items = corpus(Path("."), args.seed, args.per_family)
    inner_jev: JevBackend | None = None
    inner_llm: LLM | None = None
    if args.live:
        from jevex.jev import TypeSafeBackend
        from jevex.llm.anthropic import AnthropicLLM

        inner_jev, inner_llm = TypeSafeBackend(), AnthropicLLM()
    jev = QuestionCache(Path("jev.json"), inner_jev)
    llm = PromptCache(Path("llm.json"), inner_llm)
    settings = args.only or ([PERMISSIVE] if args.live else grid())
    rows: list[dict[str, Any]] = []
    try:
        for setting in settings:
            rows.append(await run_setting(items, jev, llm, setting))
            print(setting, rows[-1]["summary"]["accuracy"], file=sys.stderr, flush=True)
    finally:
        jev.save()
        llm.save()
        await jev.aclose()
        await llm.aclose()
    out = {
        "seed": args.seed,
        "pages": [i.path.as_posix() for i in items],
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
