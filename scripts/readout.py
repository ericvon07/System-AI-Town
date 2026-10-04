#!/usr/bin/env python3
"""Read the decision trace of both arms side by side.

    python3 scripts/readout.py            all worlds in the decisions table
    python3 scripts/readout.py --jsonl F  read a saved export instead

For each decision and persona: how many decisions each backend made, the share
of each act, and (in brackets) the mean probability behind it. The die's
probabilities are known by construction; Jev's come back with every answer.
Rates only: the two arms stop visiting the same states at the first decision
that differs, so no row pairs a dice decision with a Jev one.
"""
import argparse
import json
import os
import subprocess
from collections import defaultdict

NO_ONE = "no one"


def load(path):
    if path:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    env = dict(os.environ, NODE_NO_WARNINGS="1", CONVEX_AGENT_MODE="anonymous")
    out = subprocess.run(
        ["npx", "convex", "data", "decisions", "--limit", "8000", "--format", "jsonl"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env=env, capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def fold(decision, option):
    # Whom to invite is a name that differs per state; read it as someone / no one.
    if decision == "invite_target" and option != NO_ONE:
        return "someone"
    return option


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl")
    args = parser.parse_args()
    rows = load(args.jsonl)

    groups = defaultdict(lambda: {"n": 0, "fallbacks": 0, "taken": defaultdict(int),
                                  "p": defaultdict(float), "latency": []})
    options_of = defaultdict(set)
    for r in rows:
        g = groups[(r["decision"], r["name"], r["backend"])]
        g["n"] += 1
        g["fallbacks"] += 1 if r.get("fallback") else 0
        g["taken"][fold(r["decision"], r["chosen"])] += 1
        for p in r["probabilities"]:
            g["p"][fold(r["decision"], p["option"])] += p["p"]
            options_of[r["decision"]].add(fold(r["decision"], p["option"]))
        if r.get("latencyMs") is not None:
            g["latency"].append(r["latencyMs"])

    for decision in ("activity", "accept_invite", "invite_target"):
        options = sorted(options_of[decision])
        if not options:
            continue
        print("\n## %s\n" % decision)
        header = "| persona | arm | n | " + " | ".join(options) + " |"
        print(header)
        print("|" + "---|" * (3 + len(options)))
        names = sorted({name for (d, name, _) in groups if d == decision})
        for name in names:
            for backend in ("dice", "jev"):
                g = groups.get((decision, name, backend))
                if not g:
                    continue
                cells = []
                for o in options:
                    share = g["taken"][o] / g["n"]
                    cells.append("%3.0f%% [%.2f]" % (100 * share, g["p"][o] / g["n"]))
                n = str(g["n"]) + (" (%d fallback)" % g["fallbacks"] if g["fallbacks"] else "")
                print("| %s | %s | %s | %s |" % (name, backend, n, " | ".join(cells)))

    jev = [r for r in rows if r["backend"] == "jev"]
    if jev:
        lat = sorted(r["latencyMs"] for r in jev if r.get("latencyMs") is not None)
        falls = sum(1 for r in jev if r.get("fallback"))
        print("\njev: %d decisions, %d fallbacks, latency median %d ms, p90 %d ms" % (
            len(jev), falls, lat[len(lat) // 2], lat[int(len(lat) * 0.9)]))


if __name__ == "__main__":
    main()
