# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "crawl4ai==0.9.4",
#     "jevex",
# ]
#
# [tool.uv.sources]
# jevex = { path = "../..", editable = true }
# ///
"""Crawl4AI's LLM extraction as a jevex benchmark baseline (docs/benchmarks.md, #62).

Run from the repository root, with an inputs file from ``jevex baseline inputs``::

    uv run --script benchmarks/baselines/crawl4ai_baseline.py CORPUS --schema MODULE:CLASS \\
        --model fast --inputs INPUTS --out RESULTS [--fake]

The arguments are ``jevex baseline run``'s. Crawl4AI runs with its recommended settings
for schema extraction: an ``LLMExtractionStrategy`` with ``extraction_type="schema"``, its
default markdown input and chunking, jevex's records model as the schema and the
baselines' shared prompt as the instruction. Each page goes in as ``raw:`` HTML (the
cleaned document; a PDF or image as the text jevex read from it), through the HTTP crawler
strategy, since no browser is needed for HTML already in hand. Token usage is what LiteLLM
reports for each call (``strategy.usages``), charged at the pinned prices against
``JEVEX_LLM_MAX_COST_USD``. ``--fake`` answers every call with LiteLLM's mock response
instead, so the plumbing can be checked without a key or spend.

Crawl4AI has its own LiteLLM fork, which can't share an environment with ``jevex[litellm]``,
so it runs here in its own.
"""

from __future__ import annotations

import json
import os
import sys
from html import escape
from typing import TYPE_CHECKING, Any, cast

from jevex.baselines import BaselineOutput, BaselineSetup, charge_usage, lenient_records
from jevex.cli import main

if TYPE_CHECKING:
    from jevex.baselines import BaselineInput

KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}


class ToolError(Exception):
    """Crawl4AI reported a failed extraction (it returns error blocks rather than raising)."""


def page_html(source: BaselineInput) -> str:
    """The cleaned page, or for a PDF or image the text jevex read from it."""
    if source.document.is_html:
        return source.document.content.decode("utf-8", "replace")
    return f"<html><body><pre>{escape(source.text)}</pre></body></html>"


class Crawl4AIBaseline:
    name = "crawl4ai"

    def __init__(self, setup: BaselineSetup, *, fake: bool = False) -> None:
        if setup.model.provider not in KEY_ENV:
            raise ValueError(f"this script doesn't run {setup.model.provider} models")
        self.setup = setup
        self.fake = fake

    async def extract(self, source: BaselineInput) -> BaselineOutput:
        from crawl4ai import (
            AsyncWebCrawler,
            CacheMode,
            CrawlerRunConfig,
            LLMConfig,
            LLMExtractionStrategy,
        )
        from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

        pinned = self.setup.model
        extra: dict[str, Any] = {}
        if self.fake:
            empty = {spec.name: [] for spec in self.setup.schemas}
            extra["mock_response"] = f"<blocks>{json.dumps([empty])}</blocks>"
        # A new strategy per document: its usage totals add up across runs.
        strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider=f"{pinned.provider}/{pinned.model}",
                api_token="fake" if self.fake else f"env:{KEY_ENV[pinned.provider]}",
            ),
            schema=self.setup.records_model.model_json_schema(),
            extraction_type="schema",
            instruction=self.setup.instructions,
            extra_args=extra,
        )
        config = CrawlerRunConfig(extraction_strategy=strategy, cache_mode=CacheMode.BYPASS)
        try:
            async with AsyncWebCrawler(crawler_strategy=AsyncHTTPCrawlerStrategy()) as crawler:
                result: Any = await crawler.arun(url="raw:" + page_html(source), config=config)
        finally:  # what was paid for counts even if the crawl then failed
            usages: list[Any] = list(strategy.usages)
            input_tokens = sum(u.prompt_tokens or 0 for u in usages)
            output_tokens = sum(u.completion_tokens or 0 for u in usages)
            cost = 0.0 if self.fake else charge_usage(pinned, input_tokens, output_tokens)
        if not result.success:
            raise ToolError(result.error_message or "the crawl failed")
        blocks = [b for b in json.loads(result.extracted_content or "[]") if isinstance(b, dict)]
        good = [b for b in blocks if b.get("error") is not True]
        if blocks and not good:
            # Every chunk failed. One failed chunk among good ones (or the unparsed
            # leftovers Crawl4AI reports as an error block) keeps what the rest found.
            raise ToolError("; ".join(str(b.get("content", "")) for b in blocks))
        merged: dict[str, list[Any]] = {}
        for block in good:
            for key, value in block.items():
                if isinstance(value, list):
                    merged.setdefault(key, []).extend(cast("list[Any]", value))
        return BaselineOutput(
            records=lenient_records(merged, self.setup.schemas),
            calls=len(usages),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model=pinned.model,
        )


if __name__ == "__main__":
    # LiteLLM would fetch its price table from GitHub; jevex prices calls itself.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    args = sys.argv[1:]
    fake = "--fake" in args
    args = [a for a in args if a != "--fake"]
    sys.exit(
        main(["baseline", "run", *args], system=lambda setup: Crawl4AIBaseline(setup, fake=fake))
    )
