#!/usr/bin/env bash
# Run one arm of the decision-seam comparison on a fresh world.
#
#   scripts/run-arm.sh dice 25    upstream's dice, recorded
#   scripts/run-arm.sh jev 25     Jev behind the seam
#
# Needs `npx convex dev` running (it hosts the local backend) and, for the jev
# arm, TYPESAFE_API_KEY in the Convex env. The decisions table survives the
# wipe, so both arms are read together at the end with decisions:summary.
# The heartbeat keeps the world running with no browser tab open.
set -euo pipefail
arm=${1:?usage: run-arm.sh dice|jev [minutes]}
minutes=${2:-25}
case "$arm" in dice | jev) ;; *) echo "arm must be dice or jev" >&2; exit 2 ;; esac
cd "$(dirname "$0")/.."
export NODE_NO_WARNINGS=1 CONVEX_AGENT_MODE=anonymous
cx() { npx convex "$@"; }

cx run testing:stop >/dev/null 2>&1 || true
cx run testing:wipeAllTables >/dev/null
# The wipe deletes in scheduled pages; wait for the worlds table to empty.
# (The CLI says "no documents" on stderr; read it into a variable first, since
# grep -q closing the pipe early fails it under pipefail.)
until [[ "$(cx data worlds 2>&1)" == *"no documents"* ]]; do sleep 2; done
if [ "$arm" = jev ]; then
  cx env set DECISION_BACKEND jev >/dev/null
else
  cx env remove DECISION_BACKEND >/dev/null 2>&1 || true
fi
cx run init >/dev/null
world=$(cx run world:defaultWorldStatus | python3 -c 'import sys, json; print(json.load(sys.stdin)["worldId"])')
echo "arm=$arm world=$world start=$(date '+%F %T')"

end=$(($(date +%s) + minutes * 60))
while [ "$(date +%s)" -lt "$end" ]; do
  cx run world:heartbeatWorld "{\"worldId\":\"$world\"}" >/dev/null 2>&1 || true
  sleep 30
done
cx run testing:stop >/dev/null
echo "arm=$arm world=$world end=$(date '+%F %T')"
cx run decisions:summary "{\"worldId\":\"$world\"}"
