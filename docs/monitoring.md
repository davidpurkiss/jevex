# Monitoring jevex

A result says what happened to one document (`result.status`, `result.errors`; see the
README's *When something fails*). This page covers watching a running crawler or
`jevex serve`: logs, traces, metrics, health checks, drift, the alerts we suggest, and
what to do about failures jevex can't see itself.

| Signal | Where | Module |
| --- | --- | --- |
| Logs | `logging.getLogger("jevex.<module>")` | `jevex.logs` |
| Traces | OpenTelemetry, `Extractor(tracer=)` (`otel` extra) | `jevex.tracing` |
| Metrics | `jevex serve`'s `/metrics` (Prometheus), Scrapy stats under `jevex/` | `jevex.server`, `jevex.contrib.scrapy` |
| Health | `jevex serve`'s `/health` | `jevex.server`, `jevex.monitoring` |
| Drift | `/metrics` (a window of recent documents), the stats UI's fields view | `jevex.monitoring`, `jevex.stats` |
| Per document | `ExtractionResult.status`/`errors`, the store's `DocumentStat` (stats UI) | `jevex.errors`, `jevex.store` |

## Logs

jevex logs through the standard `logging` module and is silent until your application
configures logging (the `jevex` logger has a `NullHandler`). It never prints.

Every record carries where it happened as attributes: `run_id`, `document_id`, `url`,
`stage` and `part` (a generator id or a processor's class), each `None` when it doesn't
apply. Use them in a format string or a JSON formatter:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(stage)s %(part)s] %(url)s: %(message)s",
)
logging.getLogger("jevex.pipeline").setLevel(logging.WARNING)  # quieter stage timings
```

| Level | What |
| --- | --- |
| `DEBUG` | each stage starting and finishing, each Jev request, each pipeline event, a part failing again |
| `INFO` | each finished document (status, records, time, Jev spend, LLM calls), a stopped document, each learner outcome, a generator published, pruned or deduplicated |
| `WARNING` | the first failure of each part in a document (with its traceback), a Jev retry, the first hit of each budget limit, a quarantined generator, a failed health check |
| `ERROR` | a failed document (with its traceback), a process spend cap, a learner worker that died |

The learner works outside any document, so its records have a `run_id` and no `url`.

## Traces

With the `otel` extra (`pip install "jevex[otel]"`), each document is a `jevex.extract`
span with a `jevex.stage <name>` span per stage. Spans carry the URL, run and document
ids, schemas, entity count, status, records and errors, and each stage's own Jev requests
and spend and LLM calls and spend. A part that fails and is skipped is an `exception`
event on its stage's span (with `jevex.kind` and `jevex.part`); a core failure sets an
error status on the stage and the document.

By default jevex uses OpenTelemetry's global tracer, which records nothing until a tracer
provider is configured: by your application, or by `opentelemetry-instrument` from the
standard `OTEL_*` environment variables. Pass your own with `Extractor(tracer=...)`, or
`tracer=False` to turn tracing off.

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)  # jevex's documents are traced from here on
```

A document's span is a child of the span current when `extract` is called (a crawler's
span for the page, say), so a page's fetch and extraction can be one trace.

## Metrics

`jevex serve`'s `GET /metrics` is Prometheus text, counted since the process started:

| Series | Type | What |
| --- | --- | --- |
| `jevex_documents_total{outcome}` | counter | `ok`, `partial` (a part was skipped), `stopped` (a stage or budget stopped it), `error` (it failed) |
| `jevex_documents_in_progress` | gauge | documents being extracted now |
| `jevex_errors_total{stage,kind,part}` | counter | documents with each failure (`part` is `""` when there's none) |
| `jevex_extract_seconds` | summary | time to extract a document |
| `jevex_stage_seconds{stage}` | summary | time in each stage |
| `jevex_records_total{schema}`, `jevex_values_total{schema,method}` | counter | records, and values by how they were resolved |
| `jevex_jev_requests_total`, `jevex_jev_questions_total`, `jevex_jev_input_tokens_total`, `jevex_jev_cost_usd_total` | counter | Jev use |
| `jevex_llm_calls_total`, `jevex_llm_cost_usd_total` | counter | LLM use |
| `jevex_jev_retries_total`, `jevex_llm_retries_total` | counter | retried requests (LLM: as the SDKs report them) |
| `jevex_jev_rate_limited_total` | counter | Jev requests answered 429, retried or not |
| `jevex_llm_rate_limited_total` | counter | LLM calls that still failed on a 429 after the SDK's retries |
| `jevex_budget_events_total{scope,limit}` | counter | budget limits hit |
| `jevex_budget_remaining_usd{scope,kind,period}`, `jevex_budget_limit_usd{...}` | gauge | each spend cap and what's left: `scope="run"` is the run budget this period (`--max-spend`, `--max-jev-spend`), `scope="process"` the `JEVEX_*_MAX_COST_USD` caps |
| `jevex_store_errors_total` | counter | store failures: in documents (`jevex_errors_total` kind `store`), health checks and stats reads |
| `jevex_ledger_errors_total` | counter | spend-ledger failures: in documents (kind `ledger`: an LLM call the ledger couldn't clear isn't made) and the service's headroom reads |
| `jevex_learner_outcomes_total{status}` | counter | examples the learner finished, by outcome (`accepted`, `covered`, `llm_error`...) |
| `jevex_learner_alive` | gauge | 1 while every learner's worker can take examples, 0 once one has died |
| `jevex_learner_worker_deaths_total` | counter | times a learner's worker died |
| `jevex_drift_documents`, `jevex_field_records{field}`, `jevex_field_none_rate{field}`, `jevex_field_fallback_rate{field}`, `jevex_field_confidence_mean{field}` | gauge | drift over the last documents (below) |

Learner series appear once a learner exists (a `generator_llm`, after the first
document); headroom series only when a cap is set. The run budget's headroom is left
out while its ledger can't be read (counted in `jevex_ledger_errors_total`). A process
cap's is left out when it's misconfigured (`JEVEX_*_MAX_COST_USD` isn't a number, or the
`JEVEX_SPEND_LEDGER` file can't be read); that is logged as a warning on `jevex.server`,
not counted as a ledger failure.

The spend ledger is the store unless `jevex serve` was given another
(`Service(ledger=...)`), so a store outage usually shows in both error counters.

The Scrapy pipeline puts the counts that apply in Scrapy's stats under `jevex/`:
documents, records, `partial`, `failed`, `stopped`, `errors/<kind>`,
`errors/<stage>/<kind>/<part>`, Jev and LLM calls, retries, rate limits and spend,
`stage_seconds/<stage>`, and, when the spider closes, `learner/<status>`,
`learner_deaths` and `learner_alive`.

## Health

`GET /health` answers 200 with `"status": "ok"`, or 503 with `"status": "unhealthy"`
when the store doesn't answer a small read within 5 seconds or a learner's worker has
died. `checks` says which:

```json
{"status": "unhealthy", "version": "0.0.1", "schemas": ["VehicleSpec"],
 "checks": {"store": "ok", "learner": "1 worker(s) died"}}
```

Point your orchestrator's liveness probe at it. A dead learner's 503 is short-lived: the
next document's learn stage raises the worker's error (that document fails, with
`stage="learn"`) and the next example starts a new worker. So alert on
`jevex_learner_worker_deaths_total`, which keeps counting. A dead learner loses only
learning (the examples stay in the store). There is no separate `/ready`:
the service only answers once its store and Jev client are open.

## Drift

A site that changes its layout rarely makes extraction fail; values go missing or move to
the LLM fallback. jevex watches each `"Schema.field"` over a window of recent documents
(`Service(drift_window=200)`, `jevex.monitoring.DriftWindow`):

- the **"none" rate**: records with no value for the field;
- the **fallback rate**: values the LLM fallback gave;
- the **mean confidence** of its values.

On `/metrics` the "none" rate is per record: a document with three entities counts three
times, and a document that found nothing counts once with every field missing. The stats
UI's fields view counts per document instead (one with the field's schema that gave the
field no value at all), from the store's document stats, and shows each field's "none"
rate and the same three numbers over the last 100 documents, marked when they're more
than 10 points worse than over all of them.

## Suggested alerts

PromQL for Prometheus alerting rules; tune the thresholds to your traffic.

```yaml
groups:
  - name: jevex
    rules:
      - alert: JevexErrorRate          # documents failing
        expr: |
          sum(rate(jevex_documents_total{outcome="error"}[15m]))
            / sum(rate(jevex_documents_total[15m])) > 0.05
        for: 15m
      - alert: JevexPartFailing        # one generator or processor failing a lot
        expr: sum by (stage, kind, part) (rate(jevex_errors_total{part!=""}[1h])) > 0.1
        for: 30m
      - alert: JevexStopRate           # budgets or gates stopping documents
        expr: |
          sum(rate(jevex_documents_total{outcome="stopped"}[1h]))
            / sum(rate(jevex_documents_total[1h])) > 0.2
        for: 1h
      - alert: JevexSpendNearCap
        expr: jevex_budget_remaining_usd / jevex_budget_limit_usd < 0.1
      - alert: JevexLearnerDied        # alive goes back to 1 once a new worker starts
        expr: increase(jevex_learner_worker_deaths_total[15m]) > 0
      - alert: JevexLLMRateRising      # learning should make LLM calls rarer, not commoner
        expr: |
          (rate(jevex_llm_calls_total[1d]) / rate(jevex_documents_total[1d]))
            > 1.5 * (rate(jevex_llm_calls_total[1d] offset 7d)
                     / rate(jevex_documents_total[1d] offset 7d))
      - alert: JevexFieldDrift         # a field going missing
        expr: jevex_field_none_rate > 0.5 and jevex_field_records > 50
        for: 1h
      - alert: JevexRateLimited
        expr: rate(jevex_jev_rate_limited_total[15m]) > 0 or rate(jevex_llm_rate_limited_total[15m]) > 0
        for: 15m
      - alert: JevexStoreErrors
        expr: rate(jevex_store_errors_total[15m]) > 0
      - alert: JevexLedgerErrors       # LLM calls the run budget can't clear aren't made
        expr: rate(jevex_ledger_errors_total[15m]) > 0
```

Metrics can't see accuracy: a value can be confidently wrong. Schedule an accuracy check
against a labelled corpus, which exits 1 on a regression:

```sh
# nightly, e.g. from cron or CI
jevex eval corpus/ --schema carfinder.schemas:VehicleSpec --gate baselines/vehicles.json
```

## Failures jevex can't see

Some failures happen where jevex has no chance to log, count or report them:

- **Native aborts.** A native library can kill the process at exit or mid-run, with no
  Python traceback: for example onnxruntime's telemetry thread at interpreter exit on
  macOS (#181, exit code 134, `libc++abi: terminating`; jevex turns that telemetry off
  when its OCR engine loads). Look at the exit code: 134 is `SIGABRT`, 137 `SIGKILL`
  (often the out-of-memory killer: check `dmesg` or the container's `OOMKilled`), 139
  `SIGSEGV`. Keep core dumps or macOS crash reports (`~/Library/Logs/DiagnosticReports`)
  and run with `PYTHONFAULTHANDLER=1` so Python dumps every thread's stack on a crash.
- **Restarts.** A supervisor or orchestrator restarting the service resets
  `/metrics`' counters (Prometheus's `rate()` copes) and drops examples queued for
  learning (they stay in the store; `jevex learn` can learn from them later). Alert on
  restarts themselves: Kubernetes' `kube_pod_container_status_restarts_total`, or
  systemd's `NRestarts`. In Prometheus a restart shows as `jevex_documents_total`
  starting again from zero (`resets(jevex_documents_total{outcome="ok"}[1h]) > 2`).
- **A hung process.** A process that stops making progress can still answer
  `/health`: alert when `jevex_documents_total` stops increasing while you expect traffic
  (`rate(jevex_documents_total[15m]) == 0`), and when `jevex_documents_in_progress`
  stays high.
- **Spend outside jevex.** The spend ledger counts what jevex sent. Your providers'
  dashboards are the source of truth for what you're billed: compare them weekly.
