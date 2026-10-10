"""Measure how long a learned generator waits for its first win, to set ``prune_after`` (#296).

A learned generator with no wins after ``prune_after`` scoped documents is pruned
(:mod:`jevex.housekeeping`); one win keeps it for good. So ``prune_after`` must outlast the
documents a generator that will win runs on before its first win. This replays a corpus
directory from an empty store, one document at a time in its order, at the default
thresholds, with a fallback LLM and a ``generator_llm``, and pruning off. After each
document it reads every learned generator's stats from the store, and records which fields
the fallback answered.

Live and billed: run only with keys, under spend caps. Jev and the fallback LLM go through
``sweep.py``'s caches, so a sweep with the same ``--cache`` reuses (and fills) the same
answers; the generator LLM has its own cache (its prompts are other prompts, to another
model)::

    (set -a; . .env; set +a; ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_FOR_TESTS" \\
        uv run python benchmarks/thresholds/prune.py --live --cache /tmp/sweep-real \\
        --corpus DIR --model claude-haiku-4-5-20251001 --out prune.json)

An offline rerun (no ``--live``) replays from the caches and reports its misses: a draft
the learner tests differently asks other questions, so it's only valid with none.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sweep import (
    PromptCache,
    QuestionCache,
    corpus_schemas,
    document_url,
)

from jevex import EntityStage, EvalReport, Extractor, MultiEntity, load_corpus, score_result
from jevex.document import Document
from jevex.eval import resolve_tolerances
from jevex.extractor import default_pipeline
from jevex.jev import JevClient, TypeSafeBackend
from jevex.llm import ANTHROPIC_MODEL

if TYPE_CHECKING:
    from jevex import ExtractionResult
    from jevex.llm.anthropic import AnthropicLLM
    from jevex.store import GeneratorStats


def fallback_fields(result: ExtractionResult) -> Counter[str]:
    """``"Schema.field"`` → values the fallback LLM gave that stood, in this document."""
    return Counter(
        f"{r.schema_name}.{name}"
        for r in result.records
        for name, meta in r.meta.items()
        if meta.method == "llm" and meta.found
    )


def first_win_waits(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per learned generator: its field, the document it was learned after, the documents
    it ran on (and won) in all, and how many it ran on before its first win (``None``: it
    never won)."""
    out: dict[str, dict[str, Any]] = {}
    for doc in documents:
        for gid, use in doc["generators"].items():
            entry = out.setdefault(
                gid,
                {
                    "field": use["field"],
                    "learned_after": doc["index"],
                    "documents": 0,
                    "wins": 0,
                    "documents_before_first_win": None,
                },
            )
            if use["wins"] and entry["documents_before_first_win"] is None:
                entry["documents_before_first_win"] = entry["documents"]
            entry["documents"] += use["documents"]
            entry["wins"] += use["wins"]
    return out


def fallback_gaps(documents: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Per field, the documents between one the fallback answered it in and the next."""
    seen: dict[str, list[int]] = {}
    for doc in documents:
        for name in doc["fallback"]:
            seen.setdefault(name, []).append(doc["index"])
    return {name: [b - a for a, b in itertools.pairwise(at)] for name, at in seen.items()}


async def replay(
    corpus: Path,
    jev: QuestionCache,
    llm: PromptCache,
    generator_llm: PromptCache,
) -> dict[str, Any]:
    items = load_corpus(corpus)
    pipeline = default_pipeline().replace("entities", EntityStage(MultiEntity()))
    report = EvalReport()
    documents: list[dict[str, Any]] = []
    stats: dict[str, GeneratorStats] = {}
    async with Extractor(
        corpus_schemas(items),
        jev=JevClient(jev),
        pipeline=pipeline,
        extraction_llm=llm,
        generator_llm=generator_llm,
        store=":memory:",
        community_packs=False,
        prune_after=None,
    ) as extractor:
        tolerances = resolve_tolerances(extractor)
        store = await extractor.store()
        assert store is not None  # a generator_llm gives the extractor a store
        for index, item in enumerate(items):
            url = document_url(item, corpus)
            start = time.perf_counter()
            result = await extractor.extract(
                Document.from_path(item.path, url=url, locale=item.locale)
            )
            run = score_result(item, result, time.perf_counter() - start, tolerances)
            report.documents.append(run)
            await extractor.wait_for_learning()
            generators: dict[str, dict[str, Any]] = {}
            for spec in await store.generators(include_disabled=True):
                now = await store.generator_stats(spec.id)
                was = stats.get(spec.id)
                stats[spec.id] = now
                generators[spec.id] = {
                    "field": spec.field,
                    "documents": now.documents - (was.documents if was else 0),
                    "hits": now.hits - (was.hits if was else 0),
                    "wins": now.wins - (was.wins if was else 0),
                }
            documents.append(
                {
                    "index": index,
                    "page": url,
                    "summary": EvalReport(documents=[run]).summary(),
                    "fallback": dict(fallback_fields(result)),
                    "generators": generators,
                }
            )
            print(index, url, f"generators {len(generators)}", file=sys.stderr, flush=True)
        learner = await extractor.learner()
        outcomes = learner.outcome_counts() if learner is not None else {}
    return {
        "summary": report.summary(),
        "learn_outcomes": outcomes,
        "first_win_waits": first_win_waits(documents),
        "fallback_gaps": fallback_gaps(documents),
        "documents": documents,
    }


async def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Replay a corpus to measure prune_after.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--live", action="store_true", help="Fill the caches from real APIs")
    parser.add_argument("--model", default=ANTHROPIC_MODEL, help="The fallback LLM (Anthropic)")
    parser.add_argument(
        "--generator-model", default=ANTHROPIC_MODEL, help="The generator LLM (Anthropic)"
    )
    args = parser.parse_args(argv)

    args.cache.mkdir(parents=True, exist_ok=True)
    out_path = args.out.resolve() if args.out else None
    corpus = args.corpus.resolve()
    os.chdir(args.cache)
    live: tuple[TypeSafeBackend, AnthropicLLM, AnthropicLLM] | None = None
    if args.live:
        from jevex.llm.anthropic import AnthropicLLM

        live = (TypeSafeBackend(), AnthropicLLM(args.model), AnthropicLLM(args.generator_model))
    jev = QuestionCache(Path("jev.json"), live[0] if live else None)
    llm = PromptCache(Path("llm.json"), live[1] if live else None)
    generator_llm = PromptCache(Path("generator.json"), live[2] if live else None)
    try:
        out = await replay(corpus, jev, llm, generator_llm)
    finally:
        jev.save()
        llm.save()
        generator_llm.save()
        if live is not None:
            for client in live:
                await client.aclose()
    out = {
        "corpus": corpus.name,
        "model": args.model,
        "generator_model": args.generator_model,
        "cache_misses": {"jev": jev.misses, "llm": llm.misses, "generator": generator_llm.misses},
    } | out
    text = json.dumps(out, indent=1, default=str)
    if out_path:
        out_path.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
