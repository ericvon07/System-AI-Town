#!/usr/bin/env bash
# Run one arm of the decision-seam comparison on a fresh world.
#
#   scripts/run-arm.sh dice 25     upstream's dice and rules, recorded
#   scripts/run-arm.sh jev 25      Jev behind the seam
#   scripts/run-arm.sh gemma 25    Gemma (OpenRouter) behind the seam
#
# Needs `npx convex dev` running (it hosts the local backend); TYPESAFE_API_KEY
# in the Convex env for the jev arm, OPENROUTER_API_KEY for the gemma arm and
# for CHAT_BACKEND=openrouter. The decisions table survives the wipe; the rest
# of the world (messages, memories, who talked to whom) is exported to
# $RUNS_LOCAL/<start>-<arm>/ before the script returns, since the next arm's
# wipe deletes it. RUNS_LOCAL defaults to the lab's runs-local/, the folder
# above this repository, where tools/register_run.py picks the export up.
# The heartbeat keeps the world running with no browser tab.
#
# SPEND_CAP_USD (default 1.00): when the OpenRouter key has spent this much
# since the arm started, the world is stopped and the arm ends early.
set -euo pipefail
arm=${1:?usage: run-arm.sh dice|jev|gemma [minutes]}
minutes=${2:-25}
cap=${SPEND_CAP_USD:-1.00}
runs_local=${RUNS_LOCAL:-$(cd "$(dirname "$0")/../.." && pwd)/runs-local}
case "$arm" in dice | jev | gemma) ;; *) echo "arm must be dice, jev or gemma" >&2; exit 2 ;; esac
cd "$(dirname "$0")/.."
export NODE_NO_WARNINGS=1 CONVEX_AGENT_MODE=anonymous
cx() { npx convex "$@"; }
# What the OpenRouter key has spent, or "" when there is no key to ask.
spent() { cx run spend:openRouter 2>/dev/null | python3 -c 'import sys, json; print(json.load(sys.stdin)["usage"])' 2>/dev/null || true; }

cx run testing:stop >/dev/null 2>&1 || true
cx run testing:wipeAllTables >/dev/null
# The wipe deletes in scheduled pages; wait for the worlds table to empty.
# (The CLI says "no documents" on stderr; read it into a variable first, since
# grep -q closing the pipe early fails it under pipefail.)
until [[ "$(cx data worlds 2>&1)" == *"no documents"* ]]; do sleep 2; done
if [ "$arm" = dice ]; then
  cx env remove DECISION_BACKEND >/dev/null 2>&1 || true
else
  cx env set DECISION_BACKEND "$arm" >/dev/null
fi
chat=$(cx env get CHAT_BACKEND 2>/dev/null || true)
# The model that writes the messages, summaries and reflections, as llm.ts and
# openrouter.ts default it.
if [ "$chat" = openrouter ]; then
  chat_model=$(cx env get GEMMA_MODEL 2>/dev/null || true)
  chat_model=${chat_model:-google/gemma-4-31b-it}
else
  chat_model=$(cx env get OLLAMA_MODEL 2>/dev/null || true)
  chat_model=${chat_model:-llama3}
fi
message_length=$(cx env get MESSAGE_LENGTH 2>/dev/null || true)
cx run init >/dev/null
world=$(cx run world:defaultWorldStatus | python3 -c 'import sys, json; print(json.load(sys.stdin)["worldId"])')
start=$(date '+%F %T')
# Dirty means anything git would not reproduce: modified files and untracked
# ones (a new module that ran but was never committed). Read at the start.
code=$(git rev-parse --short HEAD)$([ -z "$(git status --porcelain)" ] || echo -dirty)
spent0=$(spent)
echo "arm=$arm chat=${chat:-ollama} world=$world start=$start openrouter_usage=${spent0:-n/a}"

end=$(($(date +%s) + minutes * 60))
while [ "$(date +%s)" -lt "$end" ]; do
  cx run world:heartbeatWorld "{\"worldId\":\"$world\"}" >/dev/null 2>&1 || true
  sleep 30
  if [ -n "$spent0" ]; then
    now=$(spent)
    if [ -n "$now" ] && python3 -c "import sys; sys.exit(0 if $now - $spent0 >= $cap else 1)"; then
      echo "spend cap reached: \$$(python3 -c "print(round($now - $spent0, 4))") >= \$$cap, stopping early"
      break
    fi
  fi
done
cx run testing:stop >/dev/null
spent1=$(spent)
echo "arm=$arm world=$world end=$(date '+%F %T')"
if [ -n "$spent0" ] && [ -n "$spent1" ]; then
  echo "openrouter spent this arm: \$$(python3 -c "print(round($spent1 - $spent0, 4))")"
fi

out="$runs_local/$(date -j -f '%F %T' "$start" '+%Y-%m-%dT%H%M')-$arm"
mkdir -p "$out"
for table in messages memories archivedConversations participatedTogether playerDescriptions; do
  cx data "$table" --limit 10000 --format jsonLines >"$out/$table.jsonl" 2>/dev/null || true
done
cx data decisions --limit 10000 --format jsonLines 2>/dev/null |
  python3 -c "import sys, json; [print(l, end='') for l in sys.stdin if json.loads(l)['worldId'] == '$world']" \
    >"$out/decisions.jsonl"
cat >"$out/run.json" <<JSON
{"arm": "$arm", "chat_backend": "${chat:-ollama}", "chat_model": "$chat_model",
 "message_length": "${message_length:-upstream}", "world": "$world", "start": "$start",
 "end": "$(date '+%F %T')", "minutes": $minutes, "code": "$code",
 "openrouter_usage_start": ${spent0:-null}, "openrouter_usage_end": ${spent1:-null}}
JSON
echo "exported to $out"
cx run decisions:summary "{\"worldId\":\"$world\"}"
