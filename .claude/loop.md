# jevex agent loop: one run

You are one run of jevex's unattended development loop. You start with no memory.
GitHub is the only state: issues, labels, "blocked by" dependencies, PRs and the run log
(issue #84). Do **one issue** per run, end with a merged PR, log the run, and stop.

Read `CLAUDE.md` first and follow it, especially **Rules for agents working unattended**
and **Definition of done**. If anything here conflicts with CLAUDE.md, CLAUDE.md wins.

`REPO=davidpurkiss/jevex`. `NOW` is `date -u +%Y-%m-%dT%H:%MZ`.
If the prompt that started you contains `DRY RUN`, go through steps 0–4 **read-only**.
Report what steps 1–2 would post or release, which issue step 4 would pick and why, then
stop. In a dry run, don't comment, label, branch, push or edit files.

## 0. Preflight
- Run `gh auth status`, `git status`, `uv --version`, then `uv sync`. If `gh` can't reach
  the repo, or `uv sync` fails, stop and report it in your final message (you can't log
  to #84 without `gh`).
- You should be on `main` at `origin/main` with a clean tree (the runner makes sure of
  this). If `git status --porcelain` shows changes, **stop and report**; never discard
  changes you didn't make. Otherwise `git fetch origin && git checkout main && git pull --ff-only`.

## 1. Weekly summary
If no comment on #84 starting with `**Week of**` was posted since the most recent Monday
00:00 UTC, post one using the format in #84's description. Build it from PRs and issues
changed in the last 7 days (`gh pr list --state all --search "head:agent/ updated:>=<date>"`,
`gh issue list --label agent-blocked`, ...).

## 2. Release stale locks
For each open issue labelled `agent-in-progress`: if the label was added more than 6
hours ago (check the issue's timeline events) and the issue has no open PR, remove the
label and comment `Stale lock released by loop run NOW.`

## 3. Capacity
Count open PRs from `agent/` branches (`gh pr list --state open --search "head:agent/"`).
These are drafts the loop could not merge. If there are **3 or more**, log
`nothing-to-do` (reason: drafts waiting on the owner) and stop.

## 3b. Finish a draft first
Before picking new work, look for an open agent draft PR whose issue's last `**Loop attempt**`
comment says a fix was made but not re-reviewed, and which isn't `agent-blocked`. If there's
one, work on it instead of a new issue: claim its issue (step 5 counts this as an attempt),
check out its branch, merge `origin/main` into it and fix any conflicts, run the checks, then
run **one** fresh `reviewer` round. If the verdict is `ready` and CI passes, mark it ready
(`gh pr ready <pr>`) and merge it (step 8). Otherwise fix what it found. If that's done within
the run, re-review once more; if it's still not `ready`, leave the draft and comment on the
issue with what's left. This keeps drafts from stalling everything blocked behind them.

## 4. Pick an issue
Candidates are open issues with label `agent-ready` and without `agent-in-progress`,
`agent-blocked` or `needs-human`.
- Drop any issue whose blockers aren't all closed:
  `gh api repos/$REPO/issues/<n>/dependencies/blocked_by --jq '[.[] | select(.state=="open")] | length'` must be `0`.
- Drop any issue that already has an open PR (`gh pr list --search "<n> in:body is:open"`,
  then check the PR body for `Closes #<n>`).
- Drop any issue labelled `live-api` unless live calls are allowed (see **Live calls**
  below). **Until #72 is closed, no run may make live calls.**
- Order by milestone (the leading number of its title; no milestone sorts last), then
  by issue number. Take the first.

If nothing is left, log `nothing-to-do` (reason: no unblocked agent-ready issues) and stop.

## 5. Claim it
- Count this issue's earlier attempts: comments starting with `**Loop attempt**`. If
  there are already 3, label it `agent-blocked`, comment that the attempt limit was
  reached with a summary of the earlier attempts, log `blocked`, and stop.
- `gh issue edit <n> --add-label agent-in-progress`
- Comment `**Loop attempt** <k>/3 started NOW.`

From here on, **any** failure must still go through step 9 (release the lock) and step 10 (log).

## 6. Implement
- Read the issue, the spec sections it cites in `docs/design-spec.md`, and the code it touches.
- Branch: `agent/<n>-<short-slug>` from `origin/main`. If that branch already exists on
  origin from an earlier attempt (and has no open PR), check it out and continue from it.
- Implement to CLAUDE.md's definition of done. Commit in logical steps. **No AI
  attribution** anywhere: no `Co-Authored-By` trailers, no "Generated with" lines, no
  session links in commits, PRs or comments.
- If the issue is ambiguous, needs a product decision, or needs a credential, **don't
  guess**. Comment what you need, label it `agent-blocked` (keep `agent-ready`), then go
  to step 9.
- If you find work outside the issue, open a new issue for it (label it `agent-ready` or
  `needs-human`, set its milestone, and add "blocked by" links with
  `gh api -X POST repos/$REPO/issues/<new>/dependencies/blocked_by -F issue_id=<id of blocker>`).
  Don't do that work in this PR.

## 7. Verify and review
- `uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest -q`
  must all pass.
- Run the `reviewer` subagent with the issue number. Fix every MUST FIX and re-run it,
  up to 3 rounds (see CLAUDE.md, **Review before a PR**).

## 8. Open the PR
- `git push -u origin <branch>`, then
  `gh pr create --title "<imperative summary>" --body-file <file>`. The body contains:
  - `Closes #<n>` on the first line
  - what changed and why, briefly
  - **Departures from the spec** (if any)
  - **Review**: the reviewer's final DONE-WHEN CHECK, plus SHOULD FIX items not done and why
- Label the PR `needs-review` (`gh pr edit <pr> --add-label needs-review`); the owner
  reviews merged work later and swaps it for `reviewed`.
- Open it as a **draft** if the reviewer still says `changes-needed` or checks fail.
- Wait for CI (`gh pr checks <pr> --watch`). If CI fails on something you can fix, fix
  it and push, up to 2 times. Otherwise convert the PR to a draft and explain in a comment.
- **Merge when ready** (build mode, until #74 switches the loop to review mode): if the
  PR is not a draft, the reviewer verdict is `ready` and every CI check passed, run
  `gh pr merge <pr> --squash --delete-branch`. If `main` moved and the branch is behind,
  `gh pr update-branch <pr>`, wait for CI again, then merge. Draft PRs are never merged;
  they wait for the owner and count towards step 3's limit.

## 9. Release the lock
`gh issue edit <n> --remove-label agent-in-progress`. Comment on the issue:
`**Loop attempt** <k>/3 finished NOW: <outcome>, PR #<pr> (merged | draft) / <what's needed>.`

## Live calls (only once #72 is closed)
A `live-api` issue may call real APIs only when all of these hold:
- #72 is closed, and `JEVEX_SECRETS_FILE` is set and non-empty. If not, label the issue
  `agent-blocked` with "needs keys" and stop.
- This week's live spend is still under the weekly caps in #72. Add up the `Spend:` lines
  of this week's #84 comments.
- Load keys **only in the command that runs the live step**, never exported for the
  whole session and never printed:
  `(set -a; . "$JEVEX_SECRETS_FILE"; set +a; uv run pytest --live -m live tests/...)`.
- `JEVEX_JEV_MAX_COST_USD` is already set by the runner and hard-stops Jev spend. Don't
  raise it.
- Websites: only the practice or test sites the issue names, honouring robots.txt.
- Commit recordings (cassettes, fixtures), never keys. Before committing, check that no
  secret value appears in `git diff --cached`.
- Report the real spend (`jevex.jev.process_cost()` or the test output) in the log.

## 10. Log the run
Comment on #84 in the run format from its description: outcome, issue, PR, reviewer
verdict, attempt, duration, spend (Jev and LLM; `none` until #72) and one or two lines of
notes. End your session with a message that repeats the log comment.
