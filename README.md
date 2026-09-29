# jevex

Extract typed records from web pages and PDFs using [Jev](https://docs.typesafe.ai/introduction), TypeSafe AI's System One model.

jevex narrows each document step by step (document → component → statement → value), asking Jev small atomic questions at every level. Every LLM fallback is turned into a declarative generator, so each run needs fewer LLM calls than the last.

> **Status:** early planning. This release only reserves the package name; there is no usable API yet.

## License

Apache 2.0
