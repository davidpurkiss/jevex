#!/usr/bin/env bash
# Run the jevex agent loop locally on your Claude subscription.
#
# Each run is a fresh headless Claude Code session that follows .claude/loop.md in a
# dedicated clone (never your working copy). Runs never overlap.
#
#   scripts/agent-loop.sh --dry-run            # one run that only reports its pick
#   scripts/agent-loop.sh                      # one real run
#   caffeinate -i scripts/agent-loop.sh --continuous  # back-to-back runs, keep the Mac awake
#   caffeinate -i scripts/agent-loop.sh --every 2h    # fixed interval instead
#   scripts/agent-loop.sh --budget             # this week's live spend and the next caps
#
# --continuous starts the next run 1 minute after one that did work, waits 30 minutes
# after "nothing to do", and 15 minutes after a failed run (e.g. a usage limit).
#
# Options: --continuous | --every <N>[s|m|h]  --model <name> (default: opus)  --dry-run
# Env:     JEVEX_AGENT_DIR (default: ~/.jevex-agent)
#          JEVEX_JEV_RUN_USD / JEVEX_JEV_WEEK_USD (default: 0.50 / 2), Jev caps
#          JEVEX_LLM_RUN_USD / JEVEX_LLM_WEEK_USD (default: 2 / 10), LLM caps
#
# API keys for live-api issues go in $JEVEX_AGENT_DIR/.env (never in a clone). Runs get
# only its *path* as JEVEX_SECRETS_FILE; .claude/loop.md says when they may load it.
# Each run gets its own spend ledger ($JEVEX_AGENT_DIR/spend/<ISO week>/run-*.ledger)
# as JEVEX_SPEND_LEDGER. jevex counts every process's spend there, and the Jev and LLM
# caps are set to the smaller of the per-run cap and what's left of the week (Mon-Sun UTC).
set -euo pipefail

REPO_URL="git@github.com:davidpurkiss/jevex.git"
AGENT_DIR="${JEVEX_AGENT_DIR:-$HOME/.jevex-agent}"
CLONE="$AGENT_DIR/jevex"
LOGS="$AGENT_DIR/logs"
LOCK="$AGENT_DIR/lock"
SPEND="$AGENT_DIR/spend"
JEV_RUN_USD="${JEVEX_JEV_RUN_USD:-0.50}"
JEV_WEEK_USD="${JEVEX_JEV_WEEK_USD:-2}"
LLM_RUN_USD="${JEVEX_LLM_RUN_USD:-2}"
LLM_WEEK_USD="${JEVEX_LLM_WEEK_USD:-10}"
MODEL="opus"
EVERY=""
CONTINUOUS=""
DRY=""
BUDGET=""
LAST=""  # outcome of the last run: worked | idle | failed

usage() { sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

seconds() {
  case "$1" in
    *h) echo $(( ${1%h} * 3600 )) ;;
    *m) echo $(( ${1%m} * 60 )) ;;
    *s) echo "${1%s}" ;;
    *[!0-9]*|"") echo "bad duration: $1" >&2; exit 2 ;;
    *) echo "$1" ;;
  esac
}

while [ $# -gt 0 ]; do
  case "$1" in
    --every) EVERY="$(seconds "${2:?--every needs a duration}")"; shift 2 ;;
    --continuous) CONTINUOUS=1; shift ;;
    --model) MODEL="${2:?--model needs a name}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --budget) BUDGET=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

# Ledgers for the current ISO week (Monday 00:00 UTC onwards).
week_dir() { echo "$SPEND/$(date -u +%G-W%V)"; }

# "<jev> <llm>" USD in the ledger lines on stdin (none: 0 0).
spent() {
  awk '$1 == "jev" { j += $2 } $1 == "llm" { l += $2 } END { printf "%.4f %.4f\n", j, l }'
}

# The cap for one run: the per-run cap, or what's left of the weekly cap if that's less.
run_cap() {
  awk -v run="$1" -v week="$2" -v spent="$3" \
    'BEGIN { left = week - spent; if (left < 0) left = 0; printf "%.4f\n", (run < left ? run : left) }'
}

week_spent() {
  local dir; dir="$(week_dir)"
  if [ -d "$dir" ]; then
    find "$dir" -name '*.ledger' -type f -exec cat {} + | spent
  else
    spent </dev/null
  fi
}

set_caps() {
  local jev llm
  read -r jev llm < <(week_spent)
  WEEK_JEV="$jev"
  WEEK_LLM="$llm"
  JEV_CAP="$(run_cap "$JEV_RUN_USD" "$JEV_WEEK_USD" "$jev")"
  LLM_CAP="$(run_cap "$LLM_RUN_USD" "$LLM_WEEK_USD" "$llm")"
}

if [ -n "$BUDGET" ]; then
  set_caps
  echo "week $(basename "$(week_dir)"): Jev \$$WEEK_JEV of \$$JEV_WEEK_USD, LLM \$$WEEK_LLM of \$$LLM_WEEK_USD"
  echo "next run caps: Jev \$$JEV_CAP, LLM \$$LLM_CAP"
  exit 0
fi

command -v claude >/dev/null || { echo "claude CLI not found" >&2; exit 1; }
command -v gh >/dev/null || { echo "gh CLI not found" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv not found" >&2; exit 1; }
mkdir -p "$AGENT_DIR" "$LOGS"

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another loop run holds $LOCK (remove it if no run is active)" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

final_message() {
  python3 - "$1" <<'PY'
import json, sys
text = ""
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        event = json.loads(line)
    except ValueError:
        continue
    if event.get("type") == "result":
        text = event.get("result") or ""
print(text)
PY
}

run_once() {
  local stamp log prompt ledger jev llm
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$LOGS/run-$stamp.jsonl"
  set_caps
  mkdir -p "$(week_dir)"
  ledger="$(week_dir)/run-$stamp.ledger"
  : >"$ledger"

  if [ ! -d "$CLONE/.git" ]; then
    git clone -q "$REPO_URL" "$CLONE"
  fi
  git -C "$CLONE" fetch -q --prune origin
  git -C "$CLONE" checkout -q main
  git -C "$CLONE" reset -q --hard origin/main
  git -C "$CLONE" clean -fdq

  prompt="Follow .claude/loop.md for one run of the jevex agent loop."
  [ -n "$DRY" ] && prompt="$prompt DRY RUN"

  echo "[$stamp] run started (model: $MODEL${DRY:+, dry run}); log: $log"
  echo "caps: Jev \$$JEV_CAP, LLM \$$LLM_CAP (week so far: Jev \$$WEEK_JEV, LLM \$$WEEK_LLM)"
  # Unset ANTHROPIC_API_KEY so the session uses your logged-in subscription, not API credits.
  # Secrets stay in a file outside the clone; the session only learns where it is.
  local secrets=""
  [ -f "$AGENT_DIR/.env" ] && secrets="$AGENT_DIR/.env"
  if (cd "$CLONE" && env -u ANTHROPIC_API_KEY \
        JEVEX_SECRETS_FILE="$secrets" \
        JEVEX_SPEND_LEDGER="$ledger" \
        JEVEX_JEV_MAX_COST_USD="$JEV_CAP" \
        JEVEX_LLM_MAX_COST_USD="$LLM_CAP" \
        claude -p "$prompt" \
        --model "$MODEL" \
        --permission-mode auto \
        --output-format stream-json --verbose) >"$log" 2>&1; then
    status=ok
    # Only the final message counts: the log also holds loop.md, which mentions the word.
    if final_message "$log" | grep -q 'nothing-to-do'; then LAST=idle; else LAST=worked; fi
  else
    status="failed ($?)"
    LAST=failed
  fi
  echo "[$(date -u +%Y%m%dT%H%M%SZ)] run $status ($LAST)"
  read -r jev llm < <(spent <"$ledger")
  echo "spend this run: Jev \$$jev, LLM \$$llm"
  python3 - "$log" <<'PY' || true
import json, sys
result = None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    try:
        event = json.loads(line)
    except ValueError:
        continue
    if event.get("type") == "result":
        result = event
if result:
    print(result.get("result") or "(no final message)")
    cost = result.get("total_cost_usd")
    turns = result.get("num_turns")
    print(f"-- turns: {turns}" + (f", equivalent API cost: ${cost:.2f}" if cost else ""))
else:
    print("(no result event; see the log)")
PY
}

if [ -n "$CONTINUOUS" ]; then
  while true; do
    run_once || LAST=failed
    case "$LAST" in
      worked) pause=60 ;;
      idle) pause=1800 ;;
      *) pause=900 ;;
    esac
    echo "next run in $pause s (Ctrl-C to stop)"
    sleep "$pause"
  done
elif [ -z "$EVERY" ]; then
  run_once
else
  while true; do
    run_once || true
    echo "next run in $EVERY s (Ctrl-C to stop)"
    sleep "$EVERY"
  done
fi
