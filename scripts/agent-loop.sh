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
#
# --continuous starts the next run 1 minute after one that did work, waits 30 minutes
# after "nothing to do", and 15 minutes after a failed run (e.g. a usage limit).
#
# Options: --continuous | --every <N>[s|m|h]  --model <name> (default: opus)  --dry-run
# Env:     JEVEX_AGENT_DIR (default: ~/.jevex-agent)
set -euo pipefail

REPO_URL="git@github.com:davidpurkiss/jevex.git"
AGENT_DIR="${JEVEX_AGENT_DIR:-$HOME/.jevex-agent}"
CLONE="$AGENT_DIR/jevex"
LOGS="$AGENT_DIR/logs"
LOCK="$AGENT_DIR/lock"
MODEL="opus"
EVERY=""
CONTINUOUS=""
DRY=""
LAST=""  # outcome of the last run: worked | idle | failed

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

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
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

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
  local stamp log prompt
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$LOGS/run-$stamp.jsonl"

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
  # Unset ANTHROPIC_API_KEY so the session uses your logged-in subscription, not API credits.
  if (cd "$CLONE" && env -u ANTHROPIC_API_KEY claude -p "$prompt" \
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
