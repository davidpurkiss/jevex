# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "jevex",
#     "langchain-anthropic==1.7.5",
#     "scrapegraphai==2.3.0",
# ]
#
# [tool.uv.sources]
# jevex = { path = "../..", editable = true }
# ///
"""ScrapeGraphAI's ``SmartScraperGraph`` as a jevex benchmark baseline (docs/benchmarks.md, #62).

Run from the repository root, with an inputs file from ``jevex baseline inputs``::

    uv run --script benchmarks/baselines/scrapegraphai_baseline.py CORPUS --schema MODULE:CLASS \\
        --model fast --inputs INPUTS --out RESULTS [--fake]

The arguments are ``jevex baseline run``'s. ``SmartScraperGraph`` runs with its recommended
settings: the page's HTML as the source (the cleaned document; a PDF or image as the text
jevex read from it), the baselines' shared prompt, and jevex's records model as the
schema. The model is a LangChain ``ChatAnthropic`` with the pinned model's context window
as ``model_tokens`` (ScrapeGraphAI's table doesn't know Claude Haiku 4.5 and would cut pages
into 8k-token chunks). Token usage is counted from each response's ``usage_metadata`` by a
callback on the model (ScrapeGraphAI's own counter drops usage when graphs run
concurrently) and charged at the pinned prices against ``JEVEX_LLM_MAX_COST_USD``.
Telemetry is turned off. ``--fake`` swaps in a scripted chat model, so the plumbing can be
checked without a key or spend.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from html import escape
from typing import TYPE_CHECKING, Any

from jevex.baselines import BaselineOutput, BaselineSetup, charge_usage, lenient_records
from jevex.cli import main

if TYPE_CHECKING:
    from jevex.baselines import BaselineInput

CONTEXT_TOKENS = {"anthropic": 200_000}
"""``model_tokens`` per provider: Claude's context window."""
CHARGE_LOCK = threading.Lock()
"""Graphs call the model from worker threads; charges change process-wide totals."""
MAX_OUTPUT_TOKENS = 16_000
"""As jevex's ``AnthropicLLM`` (LangChain's default would cut long answers short)."""


class ToolError(Exception):
    """ScrapeGraphAI returned an error instead of an answer."""


def page_html(source: BaselineInput) -> str:
    """The cleaned page, or for a PDF or image the text jevex read from it. It must not
    start with "http", or ScrapeGraphAI would fetch it as a URL."""
    if source.document.is_html:
        return source.document.content.decode("utf-8", "replace").lstrip()
    return f"<html><body><pre>{escape(source.text)}</pre></body></html>"


class ScrapeGraphAIBaseline:
    name = "scrapegraphai"

    def __init__(self, setup: BaselineSetup, *, fake: bool = False) -> None:
        if setup.model.provider not in CONTEXT_TOKENS:
            raise ValueError(f"this script doesn't run {setup.model.provider} models")
        self.setup = setup
        self.fake = fake

    def _model(self, callbacks: list[Any]) -> Any:
        if self.fake:
            from itertools import repeat

            from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
            from langchain_core.messages import AIMessage

            empty = {spec.name: [] for spec in self.setup.schemas}
            usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
            message = AIMessage(content=json.dumps(empty), usage_metadata=usage)
            return GenericFakeChatModel(messages=repeat(message), callbacks=callbacks)
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.setup.model.model, max_tokens=MAX_OUTPUT_TOKENS, callbacks=callbacks
        )

    async def extract(self, source: BaselineInput) -> BaselineOutput:
        from langchain_core.callbacks import BaseCallbackHandler
        from scrapegraphai.graphs import SmartScraperGraph

        pinned, fake = self.setup.model, self.fake

        class Usage(BaseCallbackHandler):
            """Charges each response as it arrives, so a call made after this document's
            task was cancelled (the graph runs on in its thread) still counts."""

            def __init__(self) -> None:
                self.calls = self.input_tokens = self.output_tokens = 0
                self.cost = 0.0

            def on_llm_end(self, response: Any, **kwargs: Any) -> None:
                tokens_in = tokens_out = 0
                for generations in response.generations:
                    for generation in generations:
                        message = getattr(generation, "message", None)
                        usage = getattr(message, "usage_metadata", None) or {}
                        tokens_in += usage.get("input_tokens", 0)
                        tokens_out += usage.get("output_tokens", 0)
                with CHARGE_LOCK:
                    self.calls += 1
                    self.input_tokens += tokens_in
                    self.output_tokens += tokens_out
                    if not fake:
                        self.cost += charge_usage(pinned, tokens_in, tokens_out)

        usage = Usage()
        graph = SmartScraperGraph(
            prompt=self.setup.instructions,
            source=page_html(source),
            config={
                "llm": {
                    "model_instance": self._model([usage]),
                    "model_tokens": CONTEXT_TOKENS[pinned.provider],
                },
                "verbose": False,
            },
            schema=self.setup.records_model,
        )
        answer: Any = await asyncio.to_thread(graph.run)
        if isinstance(answer, dict) and "error" in answer:
            raise ToolError(str(answer["error"]))
        if not isinstance(answer, dict):
            raise ToolError(f"expected an object, got {type(answer).__name__}")
        return BaselineOutput(
            records=lenient_records(answer, self.setup.schemas),
            calls=usage.calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost=usage.cost,
            model=pinned.model,
        )


if __name__ == "__main__":
    # Read when scrapegraphai is imported: no page content or prompts leave the machine.
    os.environ["SCRAPEGRAPHAI_TELEMETRY_ENABLED"] = "false"
    args = sys.argv[1:]
    fake = "--fake" in args
    args = [a for a in args if a != "--fake"]
    sys.exit(
        main(
            ["baseline", "run", *args],
            system=lambda setup: ScrapeGraphAIBaseline(setup, fake=fake),
        )
    )
