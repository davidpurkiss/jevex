"""Sweep the LLM fallback's thresholds over the synthetic test site (#49).

One live pass asks Jev and the fallback LLM everything the most permissive setting in the
grid would ask, and caches every answer **per question** (Jev) and per prompt (LLM). Every
other setting asks a subset of those, so the whole grid then replays offline from the
cache, end to end through the default pipeline, and is scored with ``jevex.eval``.

Assumes a Jev answer doesn't depend on the other questions sent with it (the cache splits
requests into questions), which is how Noul, Choice and Score questions are defined.

    # live: fills the cache (needs TYPESAFE_API_KEY and ANTHROPIC_API_KEY)
    uv run python scripts/threshold_sweep.py --live --cache .sweep
    # offline: replays the grid from the cache
    uv run python scripts/threshold_sweep.py --cache .sweep --out sweep.json

Learning (``learn_threshold``, ``prune_after``) is read off the same runs: the verified
LLM answers' precision by verification probability, and how often a field's fallback
fires per scoped document.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, TypeAdapter

from jevex import Extractor
from jevex.document import Document
from jevex.eval import (
    CorpusItem,
    DocumentRun,
    EvalReport,
    load_corpus,
    match_records,
    resolve_tolerances,
    score_document,
    score_value,
)
from jevex.extractor import default_pipeline
from jevex.fallback import FallbackStage
from jevex.jev import Answer, JevBackend, JevClient, JevResponse, JSONContent, Question
from jevex.llm import LLM, LLMResponse, LLMUsage
from jevex.testsite import build
from jevex.testsite.schemas import Listing, VehicleSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

FAMILIES = ("table", "kv", "prose", "grid", "listing")
CATEGORY = (0.3, 0.5, 0.7)
FALLBACK = (0.3, 0.5, 0.7, 0.9)
VERIFY = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
P_BINS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)

_ANSWER: TypeAdapter[Answer] = TypeAdapter(Answer)
_QUESTION: TypeAdapter[Question] = TypeAdapter(Question)


class CacheMissError(LookupError):
    """An offline run asked something the live pass didn't."""


def _json_key(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


class QuestionCache:
    """A Jev backend that answers each question from a per-question cache and sends only
    the misses on (or raises :class:`CacheMissError` without an inner backend)."""

    def __init__(self, path: Path, inner: JevBackend | None) -> None:
        self.path = path
        self.inner = inner
        self.entries: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        self.sent = 0

    async def system_one(
        self, state: JSONContent, questions: Mapping[str, Question]
    ) -> JevResponse:
        keys = {
            name: _json_key([state, _QUESTION.dump_python(q, mode="json")])
            for name, q in questions.items()
        }
        missing = {name: q for name, q in questions.items() if keys[name] not in self.entries}
        tokens = 0
        if missing:
            if self.inner is None:
                raise CacheMissError(f"{len(missing)} question(s) not in {self.path}")
            response = await self.inner.system_one(state, missing)
            self.sent += 1
            tokens = response.input_tokens or 0
            for name, answer in response.answers.items():
                self.entries[keys[name]] = _ANSWER.dump_python(answer, mode="json")
        answers = {name: _ANSWER.validate_python(self.entries[keys[name]]) for name in questions}
        # Cached answers cost nothing; only what was sent is charged.
        return JevResponse(answers=answers, input_tokens=tokens)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, sort_keys=True) + "\n")


class PromptCache:
    """An LLM answered from a per-prompt cache; misses go to ``inner`` (or raise)."""

    def __init__(self, path: Path, inner: LLM | None) -> None:
        self.path = path
        self.inner = inner
        self.entries: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    async def structured[T: BaseModel](self, prompt: str, schema: type[T]) -> LLMResponse[T]:
        key = _json_key([prompt, schema.model_json_schema()])
        if key not in self.entries:
            if self.inner is None:
                raise CacheMissError(f"an LLM prompt not in {self.path}")
            response = await self.inner.structured(prompt, schema)
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


def corpus(root: Path, per_family: int, seed: int) -> list[CorpusItem]:
    """The first ``per_family`` pages of each HTML family, interleaved by family."""
    manifest = build(seed, root / "site", waves=[list(FAMILIES)])
    by_family = {f: [p for p in manifest["pages"] if p["family"] == f] for f in FAMILIES}
    pages = [
        p
        for group in itertools.zip_longest(*(by_family[f][:per_family] for f in FAMILIES))
        for p in group
        if p is not None
    ]
    # Still a build manifest (seed, digest), so the next run may rebuild over it.
    (root / "site" / "truth.json").write_text(json.dumps(manifest | {"pages": pages}))
    return load_corpus(root / "site")


def llm_values(item: CorpusItem, result: Any, tolerances: Any) -> list[tuple[float, bool]]:
    """(verification p, correct?) for every value the fallback put in a record."""
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
    out: list[tuple[float, bool]] = []
    for exp, rec in match_records(item.records, found, own):
        if rec is None:
            continue
        for name, meta in rec["meta"].items():
            if meta.method != "llm" or meta.confidence is None or not meta.found:
                continue
            expected = exp.values.get(name) if exp else None
            ok = score_value(expected, meta.value, own[name]).correct > 0
            out.append((meta.confidence, ok))
    return out


async def run_config(
    items: list[CorpusItem],
    jev: JevBackend,
    llm: LLM,
    category: float,
    fallback: float,
    verify: float,
) -> dict[str, Any]:
    stage = FallbackStage(
        category_threshold=category, fallback_threshold=fallback, verify_threshold=verify
    )
    pipeline = default_pipeline().replace("fallback", stage)
    report = EvalReport()
    calibration: list[tuple[float, bool]] = []
    triggers: Counter[str] = Counter()
    async with Extractor(
        [VehicleSpec, Listing],
        jev=JevClient(jev),
        pipeline=pipeline,
        extraction_llm=llm,
        store=":memory:",
        community_packs=False,
    ) as extractor:
        tolerances = resolve_tolerances(extractor)
        # One document at a time: the structured stage's learned mappings then reach later
        # documents in the same order in every configuration, so the questions repeat.
        for item in items:
            document = Document.from_path(item.path, url=item.path.as_posix())
            result = await extractor.extract(document)
            report.documents.append(
                DocumentRun(
                    path=item.path.as_posix(),
                    schema=item.schema,
                    seconds=0.0,
                    jev_requests=result.meta.jev.requests,
                    jev_questions=result.meta.jev.questions,
                    jev_cost=result.meta.jev.cost,
                    llm_calls=result.meta.llm.calls,
                    llm_cost=result.meta.llm.cost,
                    methods=Counter(
                        m.method
                        for r in result.records
                        for m in r.meta.values()
                        if m.found and m.method
                    ),
                    fields=score_document(item, result, tolerances),
                )
            )
            calibration += llm_values(item, result, tolerances)
            for event in result.meta.events:
                if event.stage == "fallback":
                    triggers[event.kind] += 1
    s = report.summary()
    return {
        "category_threshold": category,
        "fallback_threshold": fallback,
        "verify_threshold": verify,
        "accuracy": s["accuracy"],
        "precision": s["precision"],
        "recall": s["recall"],
        "llm_calls_per_document": s["llm_calls_per_document"],
        "llm_cost_per_document": s["llm_cost_per_document"],
        "jev_questions_per_document": s["jev_questions_per_document"],
        "llm_values": len(calibration),
        "llm_values_correct": sum(ok for _, ok in calibration),
        "resolution_mix": s["resolution_mix"],
        "fields": {k: v.to_dict() for k, v in report.field_scores().items()},
        "calibration": calibration,
        "fallback_events": dict(triggers),
    }


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Sweep the LLM fallback's thresholds.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--live", action="store_true", help="Fill the cache from real APIs")
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-model", default=None, help="Default: jevex's Anthropic default")
    args = parser.parse_args(argv)

    args.cache.mkdir(parents=True, exist_ok=True)
    out_path = args.out.resolve() if args.out else None
    os.chdir(args.cache)  # relative document URLs, so cached states match across machines
    items = corpus(Path("."), args.per_family, args.seed)
    inner_jev: JevBackend | None = None
    inner_llm: LLM | None = None
    if args.live:
        from jevex.jev import TypeSafeBackend
        from jevex.llm.anthropic import AnthropicLLM

        inner_jev = TypeSafeBackend()
        inner_llm = AnthropicLLM(args.llm_model) if args.llm_model else AnthropicLLM()
    jev = QuestionCache(Path("jev.json"), inner_jev)
    llm = PromptCache(Path("llm.json"), inner_llm)
    try:
        if args.live:
            # The grid's most permissive setting asks a superset of every other's questions.
            probe = await run_config(items, jev, llm, min(CATEGORY), max(FALLBACK), 0.0)
            print(
                json.dumps({k: v for k, v in probe.items() if k not in ("fields", "calibration")})
            )
            return 0
        rows = [
            await run_config(items, jev, llm, c, f, v)
            for c, f, v in itertools.product(CATEGORY, FALLBACK, (0.0, *VERIFY))
        ]
    finally:
        jev.save()
        llm.save()
        aclose = getattr(inner_llm, "aclose", None)
        if aclose is not None:
            await aclose()
    out = {"documents": len(items), "pages": [i.path.as_posix() for i in items], "rows": rows}
    text = json.dumps(out, indent=1, default=str)
    if out_path:
        out_path.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
