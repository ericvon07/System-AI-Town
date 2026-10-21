#!/usr/bin/env python3
"""Read the decisions of the town's runs, and the probes, side by side.

    python3 scripts/readout.py                  all worlds in the local decisions table
    python3 scripts/readout.py --jsonl F        read a saved export instead
    python3 scripts/readout.py --probe F        print a saved probe:probeJev result
    python3 scripts/readout.py --runs D..       the conversations of exported arms
                                                (folders written by scripts/run-arm.sh)
    python3 scripts/readout.py --arms N..       the arms under runs/<name>/: pinned
                                                probabilities, invitations in two halves,
                                                P(leave) by length, how talks ended, pairs
    python3 scripts/readout.py --leave F        a probeLeave result, per persona and length
    python3 scripts/readout.py --variants F     the designed characters of a probeVariants
                                                result, per family
    python3 scripts/readout.py --interests F..  the interest town's probe by level, one
                                                probeInterests file per judge
    python3 scripts/readout.py --town D.. [F..] the interest town's arms by level (each an
                                                export or runs/<name>/; folders are arms,
                                                files probes), per world, then the criterion
                                                per world and pooled; beside the probe's
                                                P(leave) when F is given

With no file it asks the town's local Convex backend for the decisions table.

For each decision and persona: how many decisions each backend made, the share
of each act, and (in brackets) the mean probability behind it. The die's
probabilities are known by construction; Jev's come back with every answer.
Rates only: the two arms stop visiting the same states at the first decision
that differs, so no row pairs a dice decision with a Jev one.
"""
import argparse
import glob
import gzip
import json
import os
import re
import subprocess
from collections import defaultdict

FORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NO_ONE = "no one"


def load(path):
    if path:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    env = dict(os.environ, NODE_NO_WARNINGS="1", CONVEX_AGENT_MODE="anonymous")
    out = subprocess.run(
        ["npx", "convex", "data", "decisions", "--limit", "8000", "--format", "jsonl"],
        cwd=FORK,
        env=env, capture_output=True, text=True, check=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


BACKENDS = ("dice", "jev", "gemma")


def normalize(text):
    # decisions.ts: normalize, so an exact answer and a lenient match differ.
    return re.sub(r"\s+", " ", re.sub(r"[`*\"'.!:]", "", text.lower())).strip()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0


def fold(decision, option):
    # Whom to invite is a name that differs per state; read it as someone / no one.
    if decision == "invite_target" and option != NO_ONE:
        return "someone"
    return option


def print_probe(path):
    """A judge's probabilities on the grid's constructed states (probe.ts: probeJev, probeGemma)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    print("probe: %s, every candidate at %s tiles, everyone a stranger\n" % (d["model"], d["distance"]))
    grid = [{"probabilities": a["probabilities"]} for a in d["activity"]] + \
        [{"probabilities": {"accept": x["accept"], "decline": 1 - x["accept"]}} for x in d["accept"]] + \
        [{"probabilities": i["probabilities"]} for i in d["invite"]]
    print("pinned (likeliest option >= 0.98): %d of %d questions\n" % pinned(grid))
    names = [a["name"] for a in d["activity"]]
    options = sorted(d["activity"][0]["probabilities"])
    print("## activity: P(option)\n")
    print("| persona | " + " | ".join(options) + " |")
    print("|" + "---|" * (1 + len(options)))
    for a in d["activity"]:
        print("| %s | %s |" % (a["name"], " | ".join("%.2f" % a["probabilities"].get(o, 0) for o in options)))
    print("\n## accept_invite: P(accept), row = invited, column = inviter\n")
    print("| invited | " + " | ".join(names) + " | mean |")
    print("|" + "---|" * (2 + len(names)))
    for name in names:
        cells, ps = [], []
        for inviter in names:
            hit = [x["accept"] for x in d["accept"] if x["name"] == name and x["inviter"] == inviter]
            cells.append("%.2f" % hit[0] if hit else "-")
            ps += hit
        print("| %s | %s | %.2f |" % (name, " | ".join(cells), sum(ps) / len(ps)))
    print("\n## invite_target: P(target), row = the one choosing\n")
    cols = names + [NO_ONE]
    print("| chooser | " + " | ".join(cols) + " |")
    print("|" + "---|" * (1 + len(cols)))
    for i in d["invite"]:
        print("| %s | %s |" % (i["name"], " | ".join(
            "-" if c == i["name"] else "%.2f" % i["probabilities"].get(c, 0) for c in cols)))


def print_leave(path):
    """probe:probeLeave: mean P(leave) per persona at each length, and how many states are pinned."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    rows = d["rows"]
    states = [{"probabilities": {"leave": r["leave"], "continue": 1 - r["leave"]}} for r in rows]
    print("probe: probeLeave, %s (%s), small talk going in circles; pinned (>= 0.98 either way): %d of %d states\n"
          % (d.get("backend", "jev"), d["model"], *pinned(states)))
    print("| persona | " + " | ".join("%d msgs" % n for n in d["lengths"]) + " |")
    print("|" + "---|" * (1 + len(d["lengths"])))
    for name in sorted({r["name"] for r in rows}):
        print("| %s | %s |" % (name, " | ".join(
            "%.2f" % mean([r["leave"] for r in rows if r["name"] == name and r["messages"] == n])
            for n in d["lengths"])))


def print_variants(path):
    """probe:probeVariants: per designed character, the mean P(accept) over the four
    inviters, P(no one) when inviting, and the mean P(leave) at each length."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    for fam in d["families"].values():
        extra = ""
        if fam["backend"] == "gemma":
            extra = ", %d samples, %d calls, billed $%.4f, %d off the list" % (
                fam["samples"], fam["calls"], fam["costUsd"], len(fam["offList"]))
        print("\n## %s, %s (%s%s)\n" % (fam["family"], fam["backend"], fam["model"], extra))
        lengths = fam["lengths"]
        hit, n = pinned(fam["rows"])
        print("pinned (likeliest option >= 0.98): %d of %d questions" % (hit, n))
        print("| character | P(accept) | P(no one) | P(leave) at %s messages |" % " / ".join(map(str, lengths)))
        print("|---|---|---|---|")
        for v in fam["variants"]:
            rows = [r for r in fam["rows"] if r["variant"] == v["variant"]]
            accept = [r["probabilities"].get("accept", 0) for r in rows if r["decision"] == "accept_invite"]
            no_one = [r["probabilities"].get(NO_ONE, 0) for r in rows if r["decision"] == "invite_target"]
            leave = defaultdict(list)
            for r in rows:
                if r["decision"] == "leave_conversation":
                    leave[r["messages"]].append(r["probabilities"].get("leave", 0))
            print("| %s | %.2f | %.2f | %s |" % (
                v["variant"], sum(accept) / len(accept), no_one[0],
                " / ".join("%.2f" % (sum(leave[n]) / len(leave[n])) for n in lengths)))


def load_json(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


LEVELS = [3, 2, 1, 0]
# The criterion's lengths: P(leave) is read at and after the norm of 20; the talks at 20 and 30.
VERDICT_LENGTHS = (20, 30, 45)
TALK_VERDICT_LENGTHS = (20, 30)
TOO_FEW = 10


def it_k(values, rising, extra_spread=0.0, counts=None):
    """The criterion, as written before the answers were seen. `values` maps each level
    to one number per ordered pair (or per pair and world); the unit is their mean, and
    the spread the largest difference between two of them at one level (or, through
    `extra_spread`, between two worlds). The judge READS THE LEVEL when the four means
    are in order -- rising from 0 to 3, or falling for P(leave) -- and the gap between
    levels 3 and 0 beats the spread; IN PART when the gap beats the spread but the order
    breaks in the middle; otherwise it DOES NOT READ IT, whatever the sign. In an arm, a
    level with fewer than 10 decisions (`counts`) is TOO FEW TO READ."""
    if not any(values.get(lv) for lv in LEVELS):
        return "no rows", None, None
    if counts is not None and any(counts.get(lv, 0) < TOO_FEW for lv in LEVELS):
        return "too few to read", None, None
    if any(not values.get(lv) for lv in LEVELS):
        return "too few to read", None, None
    means = {lv: mean(values[lv]) for lv in LEVELS}
    spread = max([max(values[lv]) - min(values[lv]) for lv in LEVELS] + [extra_spread])
    seq = [means[lv] if rising else -means[lv] for lv in (0, 1, 2, 3)]
    gap = seq[3] - seq[0]
    if gap <= spread:
        return "does not read it", means[3] - means[0], spread
    ordered = all(seq[i] <= seq[i + 1] for i in range(3))
    return ("reads the level" if ordered else "in part"), means[3] - means[0], spread


def verdict_line(label, result):
    v, gap, spread = result
    if gap is None:
        return "- %s: **%s**" % (label, v)
    return "- %s: **%s** (level 3 minus level 0 %+.2f, spread %.2f)" % (label, v, gap, spread)


def merge_probes(paths):
    """probeInterests files joined by judge and dealing: Gemma's half comes in pieces,
    one person and one part (grid or talks) per call. Each row gains its dealing and,
    for a leave row written before the talks existed, the talk 'small'."""
    groups = {}
    for path in paths:
        pr = load_json(path)
        dealing = (pr.get("town") or {}).get("dealing", 1)
        key = (pr["backend"], dealing)
        g = groups.setdefault(key, {"backend": pr["backend"], "dealing": dealing, "model": pr.get("model"),
                                    "norm": pr["norm"], "wordings": pr["wordings"], "samples": pr.get("samples"),
                                    "calls": 0, "costUsd": 0.0, "offList": [], "rows": [], "seen": set()})
        g["calls"] += pr.get("calls") or 0
        g["costUsd"] += pr.get("costUsd") or 0
        g["offList"] += pr.get("offList") or []
        for r in pr["rows"]:
            r = dict(r, dealing=dealing)
            if r["decision"] == "leave_conversation":
                r.setdefault("talk", "small")
            ident = (r["decision"], r["name"], r.get("other"), r.get("wording"), r.get("talk"),
                     r.get("messages"), r.get("order"))
            if ident in g["seen"]:
                raise SystemExit("%s: a %s row asked twice for judge %s, dealing %d" % (
                    path, r["decision"], pr["backend"], dealing))
            g["seen"].add(ident)
            g["rows"].append(r)
    return groups


def whom_by_pair(rows):
    """Each ordered pair (dealing, chooser, partner) of the invite question: the mean,
    over the orders asked, of P(partner | someone) = P(partner) / (1 - P(no one))."""
    out = defaultdict(list)
    for r in rows:
        if r["decision"] != "invite_target":
            continue
        ps = probs(r)
        q = ps.get(NO_ONE, 0.0)
        if q >= 1:
            continue
        for name, lv in r["levels"].items():
            out[(r["dealing"], r["name"], name, lv)].append(ps.get(name, 0.0) / (1 - q))
    return {k: mean(v) for k, v in out.items()}


def by_level(pair_values, dealing=None):
    vals = defaultdict(list)
    for (d, _, _, lv), x in pair_values.items():
        if dealing is None or d == dealing:
            vals[lv].append(x)
    return vals


def leave_by_pair(rows, wording, talk, lengths):
    out = defaultdict(list)
    for r in rows:
        if (r["decision"] == "leave_conversation" and r["wording"] == wording
                and r["talk"] == talk and r["messages"] in lengths):
            out[(r["dealing"], r["name"], r["other"], r["level"])].append(probs(r).get("leave", 0.0))
    return {k: mean(v) for k, v in out.items()}


def print_interests(paths):
    """probe:probeInterests, read by level, the level being how many pastimes the
    two have in common: one or more files per judge and dealing, joined, judges side by
    side; each dealing alone and both pooled. Beside each table, the criterion's verdict; the
    invite read in two halves, the overlap rule as its ceiling, the leave
    question over the small talk and over the two talks about a pastime, and the
    pinned probabilities. Each pair is also listed alone."""
    groups = merge_probes(paths)
    judges = sorted({j for j, _ in groups}, key=lambda j: ("jev", "gemma").index(j) if j in ("jev", "gemma") else 9)
    dealings = sorted({d for _, d in groups})
    rows = {j: [r for (jj, _), g in groups.items() if jj == j for r in g["rows"]] for j in judges}
    for (j, d), g in sorted(groups.items()):
        extra = ""
        if j == "gemma":
            extra = ", %s samples, %d calls, billed $%.4f, %d off the list" % (
                g["samples"], g["calls"], g["costUsd"], len(g["offList"]))
        print("%s, dealing %d: %s%s, norm %d, %d rows" % (j, d, g["model"], extra, g["norm"], len(g["rows"])))
    scopes = [("dealing %d" % d, d) for d in dealings] + ([("pooled", None)] if len(dealings) > 1 else [])
    head = " | ".join(judges)

    def table(title, note, values_of, rising, fmt="%.2f"):
        print("\n## %s\n" % title)
        if note:
            print(note + "\n")
        print("| level | scope | " + head + " |")
        print("|---|---|" + "---|" * len(judges))
        per = {j: values_of(rows[j]) for j in judges}
        for lv in LEVELS:
            for label, d in scopes:
                cells = []
                for j in judges:
                    xs = by_level(per[j], d).get(lv, [])
                    cells.append((fmt % mean(xs)) + " (%d)" % len(xs) if xs else "-")
                print("| %d | %s | %s |" % (lv, label, " | ".join(cells)))
        print()
        for j in judges:
            for label, d in scopes:
                print(verdict_line("criterion, %s, %s" % (j, label), it_k(by_level(per[j], d), rising)))

    accept = lambda rs: {(r["dealing"], r["name"], r["other"], r["level"]): probs(r).get("accept", 0.0)
                         for r in rs if r["decision"] == "accept_invite"}
    table("accept_invite: P(accept) by level",
          "Mean over the ordered pairs (their number in brackets).", accept, True)

    table("invite_target, whom: P(a given partner | someone is invited), by the partner's level",
          "Mean over the ordered pairs, each the mean of the two orders. Five people are free, so\n"
          "a uniform pick among them gives 0.20; the overlap rule gives 1.00 at level 3\n"
          "and 0.00 below -- the ceiling, since here the level is a count of identical words.",
          whom_by_pair, True)
    print("\n## invite_target, whether: P(no one) by chooser\n")
    print("| chooser | scope | " + head + " |")
    print("|---|---|" + "---|" * len(judges))
    for name in sorted({r["name"] for j in judges for r in rows[j] if r["decision"] == "invite_target"}):
        for label, d in scopes:
            cells = []
            for j in judges:
                xs = [probs(r).get(NO_ONE, 0.0) for r in rows[j] if r["decision"] == "invite_target"
                      and r["name"] == name and (d is None or r["dealing"] == d)]
                cells.append("%.2f" % mean(xs) if xs else "-")
            print("| %s | %s | %s |" % (name, label, " | ".join(cells)))
    print("\nBy chooser and order (P of the partner at level 3 / 2 / 1 / each at 0 / no one):\n")
    print("| dealing | chooser | order | " + head + " |")
    print("|---|---|---|" + "---|" * len(judges))
    keys = sorted({(r["dealing"], r["name"], r["order"]) for j in judges for r in rows[j]
                   if r["decision"] == "invite_target"})
    for d, name, order in keys:
        cells = []
        for j in judges:
            mine = [r for r in rows[j] if r["decision"] == "invite_target" and r["dealing"] == d
                    and r["name"] == name and r["order"] == order]
            if not mine:
                cells.append("-")
                continue
            ps, r = [], mine[0]
            for lv in LEVELS:
                at = [n for n, l in r["levels"].items() if l == lv]
                ps.append(sum(probs(r).get(n, 0) for n in at) / len(at))
            ps.append(probs(r).get(NO_ONE, 0))
            cells.append(" / ".join("%.2f" % x for x in ps))
        print("| %d | %s | %s | %s |" % (d, name, order, " | ".join(cells)))

    wordings = groups[next(iter(groups))]["wordings"]
    lengths = sorted({r["messages"] for j in judges for r in rows[j]
                      if r["decision"] == "leave_conversation" and r["talk"] == "small"})
    for wording in wordings:
        print("\n## leave_conversation over the small talk, %s wording: P(leave) by level and messages so far\n"
              % wording)
        print("| level | judge | scope | " + " | ".join(map(str, lengths)) + " |")
        print("|---|---|---|" + "---|" * len(lengths))
        for lv in LEVELS:
            for j in judges:
                for label, d in scopes:
                    cells = []
                    for n in lengths:
                        xs = by_level(leave_by_pair(rows[j], wording, "small", (n,)), d).get(lv, [])
                        cells.append("%.2f" % mean(xs) if xs else "-")
                    print("| %d | %s | %s | %s |" % (lv, j, label, " | ".join(cells)))
        print()
        for j in judges:
            for label, d in scopes:
                print(verdict_line("criterion, %s, %s, P(leave) at %s messages, each pair's mean" % (
                    j, label, "/".join(map(str, VERDICT_LENGTHS))),
                    it_k(by_level(leave_by_pair(rows[j], wording, "small", VERDICT_LENGTHS), d), False)))

    print_talks(judges, rows, wordings, scopes)
    print_pinned_probe(judges, rows)

    print("\n## each ordered pair alone: P(accept); P(leave) over the small talk at %s messages, neutral / clause\n"
          % " / ".join(map(str, lengths)))
    print("| dealing | person | other | level | judge | accept | leave, neutral | leave, clause |")
    print("|---|---|---|---|---|---|---|---|")
    pairs = sorted({(r["dealing"], r["level"], r["name"], r["other"]) for j in judges for r in rows[j]
                    if r["decision"] == "accept_invite"}, key=lambda t: (t[0], -t[1], t[2], t[3]))
    for d, lv, name, other in pairs:
        for j in judges:
            mine = [r for r in rows[j] if r["dealing"] == d and r["name"] == name and r.get("other") == other]
            if not mine:
                continue
            acc = [probs(r).get("accept", 0) for r in mine if r["decision"] == "accept_invite"]
            leave = {}
            for w in wordings:
                leave[w] = " ".join("%.2f" % mean([probs(r).get("leave", 0) for r in mine
                                                   if r["decision"] == "leave_conversation"
                                                   and r["talk"] == "small" and r["wording"] == w
                                                   and r["messages"] == n]) for n in lengths)
            print("| %d | %s | %s | %d | %s | %.2f | %s | %s |" % (
                d, name, other, lv, j, mean(acc), leave.get("neutral", "-"), leave.get("clause", "-")))


def print_talks(judges, rows, wordings, scopes):
    """The leave question over a talk about a pastime both list against one only
    the other lists, on the ordered pairs that have both (levels 1-3). The criterion: the judge
    READS THE TALK when P(leave) over the pastime both list is below P(leave) over the
    other's own, at 20 and at 30 messages, by more than the spread within a talk; THE
    CLAUSE SHARPENS IT when that gap under the clause beats the gap under the neutral
    wording by more than the spread."""
    talk_rows = {j: [r for r in rows[j] if r["decision"] == "leave_conversation" and r["talk"] != "small"]
                 for j in judges}
    if not any(talk_rows.values()):
        return
    lengths = sorted({r["messages"] for j in judges for r in talk_rows[j]})
    print("\n## leave_conversation over a talk about a pastime: P(leave), both list / only the other lists\n")
    print("On the ordered pairs at levels 1-3, which have a pastime both list; the pairs at level 0,")
    print("asked over the other's own only, are in the last rows.\n")
    print("| wording | judge | scope | " + " | ".join(map(str, lengths)) + " |")
    print("|---|---|---|" + "---|" * len(lengths))

    def gap_at(rs, wording, n, d):
        """(mean both, mean other only, gap, spread) over the pairs that have both."""
        both = {k: v for k, v in leave_by_pair(rs, wording, "both", (n,)).items() if d is None or k[0] == d}
        own = {k: v for k, v in leave_by_pair(rs, wording, "other only", (n,)).items()
               if k in both}
        if not both:
            return None
        spread = max(max(both.values()) - min(both.values()), max(own.values()) - min(own.values()))
        return mean(list(both.values())), mean(list(own.values())), mean(list(own.values())) - mean(list(both.values())), spread

    for wording in wordings:
        for j in judges:
            for label, d in scopes:
                cells = []
                for n in lengths:
                    g = gap_at(talk_rows[j], wording, n, d)
                    cells.append("%.2f / %.2f" % g[:2] if g else "-")
                print("| %s | %s | %s | %s |" % (wording, j, label, " | ".join(cells)))
    for wording in wordings:
        for j in judges:
            for label, d in scopes:
                cells = []
                for n in lengths:
                    xs = [v for k, v in leave_by_pair(talk_rows[j], wording, "other only", (n,)).items()
                          if k[3] == 0 and (d is None or k[0] == d)]
                    cells.append("%.2f" % mean(xs) if xs else "-")
                print("| %s | %s | %s, level 0 | %s |" % (wording, j, label, " | ".join(cells)))
    print()
    for j in judges:
        for label, d in scopes:
            reads, sharpens, notes = True, True, []
            for n in TALK_VERDICT_LENGTHS:
                neu = gap_at(talk_rows[j], "neutral", n, d)
                cla = gap_at(talk_rows[j], "clause", n, d)
                if not neu or not cla:
                    reads = sharpens = None
                    break
                reads = reads and neu[2] > neu[3]
                spread = max(neu[3], cla[3])
                sharpens = sharpens and (cla[2] - neu[2]) > spread
                notes.append("at %d: gap %+.2f neutral, %+.2f clause, spread %.2f" % (n, neu[2], cla[2], spread))
            if reads is None:
                print("- criterion, %s, %s: **too few to read**" % (j, label))
                continue
            print("- criterion, %s, %s: **%s**; the clause **%s** (%s)" % (
                j, label, "reads the talk" if reads else "does not read the talk",
                "sharpens it" if sharpens else "does not sharpen it", "; ".join(notes)))


def print_pinned_probe(judges, rows):
    """Each person, decision and level whose mean probability is pinned (<= 0.02 or
    >= 0.98): P(accept); P(invite a partner at that level); P(no one); P(leave) at 20,
    30 and 45 messages over the small talk, by wording. And, beside it, the share of
    single rows pinned, read by the shared `pinned`."""
    print("\n## pinned probabilities: person, decision and level whose mean is <= %.2f or >= %.2f\n"
          % (PINNED_LOW, PINNED_HIGH))
    for j in judges:
        groups = defaultdict(list)
        for r in rows[j]:
            ps = probs(r)
            if r["decision"] == "accept_invite":
                groups[(r["dealing"], r["name"], "accept", "level %d" % r["level"])].append(ps.get("accept", 0))
            elif r["decision"] == "invite_target":
                groups[(r["dealing"], r["name"], "no one", "")].append(ps.get(NO_ONE, 0))
                for name, lv in r["levels"].items():
                    groups[(r["dealing"], r["name"], "invite", "level %d" % lv)].append(ps.get(name, 0))
            elif r["talk"] == "small" and r["messages"] in VERDICT_LENGTHS:
                groups[(r["dealing"], r["name"], "leave, %s" % r["wording"], "level %d" % r["level"])].append(
                    ps.get("leave", 0))
        stuck = sorted((k, mean(v)) for k, v in groups.items()
                       if mean(v) <= PINNED_LOW or mean(v) >= PINNED_HIGH)
        by = defaultdict(int)
        for (d, name, what, lv), _ in stuck:
            by[(what, lv)] += 1
        print("%s: %d of %d groups pinned%s" % (j, len(stuck), len(groups), ": " if by else "."))
        for (what, lv), n in sorted(by.items()):
            print("    %s %s: %d (%s)" % (what, lv, n, ", ".join(
                "%s d%d %.2f" % (name, d, m) for (d, name, w, l), m in stuck if (w, l) == (what, lv))))
        single = []
        for decision, option in (("accept_invite", "accept"), ("invite_target", NO_ONE), ("leave_conversation", "leave")):
            hit, n = pinned([r for r in rows[j] if r["decision"] == decision], option)
            single.append("%s %d of %d" % (decision, hit, n))
        print("    single rows pinned: %s" % "; ".join(single))


# Words that say a pastime came up in the talk (the arm readout of the interest town):
# a stem or a near word for each of the eighteen, matched case-blind on word starts.
PASTIME_WORDS = {
    "chess": r"chess",
    "birdwatching": r"bird",
    "sourdough baking": r"sourdough|bak(e|es|ed|ing|er)\b|bread|loaf|loaves|starter\b",
    "jazz records": r"jazz|vinyl",
    "pottery": r"potter|clay\b|kiln|ceramic|glaze",
    "stargazing": r"stargaz|stars?\b|telescope|constellation|night sky",
    "long-distance cycling": r"cycl|bik(e|es|ing)\b|bicycl",
    "learning Japanese": r"japan",
    "fly fishing": r"fish|angl(er|ing)|trout",
    "silent films": r"film|movie|cinema|chaplin|keaton",
    "poetry": r"poe(m|t)",
    "local history": r"histor",
    "salsa dancing": r"salsa|danc",
    "repairing old radios": r"radio",
    "yoga": r"yoga",
    "card magic": r"card|magic",
    "bonsai": r"bonsai",
    "stamp collecting": r"stamp|philatel",
}
LEAVE_BINS = [(1, 5, 2), (6, 14, 10), (15, 24, 20), (25, 37, 30), (38, 52, 45), (53, 70, 60), (71, 10 ** 6, 80)]


def read_table(d, name):
    for path, opener in ((os.path.join(d, name + ".jsonl"), open),
                         (os.path.join(d, name + ".jsonl.gz"), gzip.open)):
        if os.path.exists(path):
            with opener(path, "rt") as f:
                return [json.loads(line) for line in f if line.strip()]
    return []


def pastimes_of(identity):
    m = re.search(r"In free time, \S+ enjoys (.+)\.$", identity)
    return re.split(r", | and ", m.group(1)) if m else []


def print_town(d, probe_paths):
    """An arm of the interest town, read by level: whom people invite
    against who was free, P(accept), P(leave) by messages so far beside the probe's,
    conversation lengths and the share that reached the cap, conversations per pair, and
    how often a pair's pastimes in common come up in its talk against the others' own.
    Each pair is also listed alone."""
    people = {p["name"]: pastimes_of(p["description"]) for p in read_table(d, "playerDescriptions")}
    names_by_id = {p["playerId"]: p["name"] for p in read_table(d, "playerDescriptions")}
    if not all(people.values()):
        raise SystemExit("%s is not the interest town: no pastime list in its identities" % d)
    level = lambda a, b: len(set(people[a]) & set(people[b]))
    pair_of = lambda a, b: tuple(sorted((a, b)))
    rows = [r for r in read_table(d, "decisions") if r["backend"] != "dice" and not r.get("fallback")]
    falls = len([r for r in read_table(d, "decisions") if r.get("fallback")])
    meta = {}
    if os.path.exists(os.path.join(d, "run.json")):
        meta = json.load(open(os.path.join(d, "run.json")))
    elif os.path.exists(os.path.join(d, "manifest.json")):
        meta = json.load(open(os.path.join(d, "manifest.json")))
    town = {k: meta.get(k) or meta.get("condition", {}).get(k) for k in ("cast", "dealing", "norm", "cap")}
    dealing = town["dealing"] or 1
    print("%s: %d judged decisions (%d fallbacks left out); %s" % (
        os.path.basename(d.rstrip("/")), len(rows), falls, town))
    levels = [3, 2, 1, 0]
    P = lambda r, o: {p["option"]: p["p"] for p in r["probabilities"]}.get(o, 0)

    print("\n## invite_target: each free person as a candidate, by level\n")
    print("Chance is 1 / (free people + 1), the share a uniform pick over them and no one would give.\n")
    print("| level | times free | mean P(invited) | invited | share | chance |")
    print("|---|---|---|---|---|---|")
    invites = [r for r in rows if r["decision"] == "invite_target"]
    offers = defaultdict(list)
    by_pos = defaultdict(list)
    for r in invites:
        free = [f["name"] for f in r["state"]["free_people"]]
        for i, name in enumerate(free):
            offers[level(r["name"], name)].append((P(r, name), r["chosen"] == name, 1 / (len(free) + 1)))
            by_pos["last" if i == len(free) - 1 and len(free) > 1 else "first" if i == 0 else "middle"].append(P(r, name))
    for lv in levels:
        xs = offers[lv]
        if xs:
            print("| %d | %d | %.2f | %d | %.2f | %.2f |" % (
                lv, len(xs), mean([x[0] for x in xs]), sum(x[1] for x in xs),
                mean([x[1] for x in xs]), mean([x[2] for x in xs])))
    print("| no one | %d | %.2f | %d | %.2f | |" % (
        len(invites), mean([P(r, NO_ONE) for r in invites]),
        sum(r["chosen"] == NO_ONE for r in invites), mean([r["chosen"] == NO_ONE for r in invites])))
    print("\nBy place in the (shuffled) list: %s" % ", ".join(
        "%s %.2f (n=%d)" % (k, mean(v), len(v)) for k, v in sorted(by_pos.items())))

    print("\n## invite_target in two halves, and the overlap rule\n")
    print("Whether: P(no one) and how often no one was chosen, per person. Whom: given someone is")
    print("invited, P(a free partner | someone) = P(partner) / (1 - P(no one)), by the partner's level.\n")
    print("| person | invites | mean P(no one) | chose no one | chose someone |")
    print("|---|---|---|---|---|")
    for name in sorted(people):
        h = invite_halves([r for r in invites if r["name"] == name])
        if h["n"]:
            print("| %s | %d | %.2f | %d | %d |" % (name, h["n"], h["p_no_one"], h["chose_no_one"], h["someone"]))
    whom = defaultdict(list)
    rule_p, rule_chance, rule_hit = [], [], []
    for r in invites:
        free = [f["name"] for f in r["state"]["free_people"]]
        q = P(r, NO_ONE)
        if not free or q >= 1:
            continue
        for f in free:
            whom[level(r["name"], f)].append(P(r, f) / (1 - q))
        top = max(level(r["name"], f) for f in free)
        picks = [f for f in free if level(r["name"], f) == top]
        rule_p.append(sum(P(r, f) for f in picks) / (1 - q))
        rule_chance.append(len(picks) / len(free))
        if r["chosen"] != NO_ONE:
            rule_hit.append(r["chosen"] in picks)
    print("\n| level | offers | mean P(partner \\| someone) |")
    print("|---|---|---|")
    for lv in levels:
        if whom[lv]:
            print("| %d | %d | %.2f |" % (lv, len(whom[lv]), mean(whom[lv])))
    print("\nWhat else the invite state carries, beside the level: P(partner | someone) by how far the")
    print("partner stood and whether the two had talked before, and how far the free stood by level.\n")
    near = defaultdict(list)
    for r in invites:
        free = r["state"]["free_people"]
        q = P(r, NO_ONE)
        for f in free:
            near[("distance", level(r["name"], f["name"]))].append(f["distance_in_tiles"])
            if free and q < 1:
                tiles = f["distance_in_tiles"]
                band = "near (5 tiles or less)" if tiles <= 5 else "mid (6-15)" if tiles <= 15 else "far (over 15)"
                near[("band", band)].append(P(r, f["name"]) / (1 - q))
                past = "never talked" if f.get("last_talked_minutes_ago") is None else "talked before"
                near[("past", past)].append(P(r, f["name"]) / (1 - q))
    print("| | n | mean P(partner \\| someone) |")
    print("|---|---|---|")
    for kind in ("band", "past"):
        for k in sorted({k for (kk, k) in near if kk == kind}):
            print("| %s | %d | %.2f |" % (k, len(near[(kind, k)]), mean(near[(kind, k)])))
    print("\nMean distance of a free person, in tiles, by level: %s" % ", ".join(
        "%d: %.1f" % (lv, mean(near[("distance", lv)])) for lv in levels if near[("distance", lv)]))
    if rule_p:
        print("\nThe overlap rule picks the free person with the most pastimes in common (ties split): the")
        print("judge gave its pick P %.2f given someone, against %.2f for a uniform pick and 1.00 for the" % (
            mean(rule_p), mean(rule_chance)))
        print("rule; of %d invitations to someone, %d went to the rule's pick." % (len(rule_hit), sum(rule_hit)))

    print("\n## accept_invite by level\n")
    print("| level | n | mean P(accept) | accepted |")
    print("|---|---|---|---|")
    accepts = [r for r in rows if r["decision"] == "accept_invite"]
    for lv in levels:
        xs = [r for r in accepts if level(r["name"], r["otherName"]) == lv]
        if xs:
            print("| %d | %d | %.2f | %.2f |" % (lv, len(xs), mean([P(r, "accept") for r in xs]),
                                               mean([r["chosen"] == "accept" for r in xs])))

    probe = {}
    for path in probe_paths:
        pr = load_json(path)
        # The probe's small talk, neutral wording, in this arm's dealing.
        if (pr.get("town") or {}).get("dealing", 1) != dealing:
            continue
        for r in pr["rows"]:
            if (r["decision"] == "leave_conversation" and r["wording"] == "neutral"
                    and r.get("talk", "small") == "small"):
                probe.setdefault((pr["backend"], r["level"], r["messages"]), []).append(
                    r["probabilities"].get("leave", 0))
    print("\n## leave_conversation: mean P(leave) by level and messages so far (n)%s\n" % (
        ", the probe's at the bin's length in brackets" if probe else ""))
    print("| level | " + " | ".join("%d–%s" % (lo, hi if hi < 10 ** 6 else "") for lo, hi, _ in LEAVE_BINS) + " |")
    print("|---|" + "---|" * len(LEAVE_BINS))
    leaves = [r for r in rows if r["decision"] == "leave_conversation"]
    for lv in levels:
        cells = []
        for lo, hi, at in LEAVE_BINS:
            xs = [P(r, "leave") for r in leaves if level(r["name"], r["otherName"]) == lv
                  and lo <= r["messagesSoFar"] <= hi]
            cell = "%.2f (%d)" % (mean(xs), len(xs)) if xs else "-"
            if probe.get(("jev", lv, at)):
                cell += " [%.2f]" % mean(probe[("jev", lv, at)])
            cells.append(cell)
        print("| %d | %s |" % (lv, " | ".join(cells)))

    convs = defaultdict(list)
    for m in sorted(read_table(d, "messages"), key=lambda m: m["_creationTime"]):
        convs[m["conversationId"]].append(m)
    by_pair = defaultdict(list)
    for cid, ms in convs.items():
        who = sorted({names_by_id.get(m["author"], m["author"]) for m in ms})
        if len(who) != 2:
            continue
        a, b = who
        text = " ".join(m["text"] for m in ms).lower()
        said = {p for p in PASTIME_WORDS if re.search(r"\b(" + PASTIME_WORDS[p] + ")", text)}
        both = set(people[a]) & set(people[b])
        own = (set(people[a]) | set(people[b])) - both
        by_pair[(a, b)].append({"n": len(ms), "both": both, "own": own, "said": said})
    cap = meta.get("cap") or meta.get("condition", {}).get("cap") or 100
    print("\n## conversations by level\n")
    print("A pastime counts as come up when a word of it is in any message of the conversation.\n")
    print("| level | pairs | conversations | per pair | median length | mean length | reached %d | pastimes in common come up | each one's own come up |" % cap)
    print("|---|---|---|---|---|---|---|---|---|")
    for lv in levels:
        pairs = [p for p in by_pair if level(*p) == lv]
        n_pairs = len([1 for a in people for b in people if a < b and level(a, b) == lv])
        cs = [c for p in pairs for c in by_pair[p]]
        if not cs:
            print("| %d | %d | 0 | | | | | | |" % (lv, n_pairs))
            continue
        lengths = [c["n"] for c in cs]
        both = [len(c["said"] & c["both"]) / len(c["both"]) for c in cs if c["both"]]
        own = [len(c["said"] & c["own"]) / len(c["own"]) for c in cs if c["own"]]
        print("| %d | %d | %d | %.1f | %d | %.1f | %d | %s | %.2f |" % (
            lv, n_pairs, len(cs), len(cs) / n_pairs, pct(lengths, 0.5), mean(lengths),
            sum(1 for x in lengths if x > cap), "%.2f" % mean(both) if both else "-", mean(own)))

    print("\n## pinned\n")
    print("A person whose ten or more decisions of a kind were all one act, and the share of single")
    print("rows pinned (P <= %.2f or >= %.2f, the shared reading).\n" % (PINNED_LOW, PINNED_HIGH))
    acts = defaultdict(list)
    for r in rows:
        if r["decision"] in ("accept_invite", "invite_target", "leave_conversation", "activity"):
            acts[(r["name"], r["decision"])].append(fold(r["decision"], r["chosen"]))
    one_act = sorted((k, v[0], len(v)) for k, v in acts.items() if len(v) >= TOO_FEW and len(set(v)) == 1)
    for (name, decision), act, n in one_act:
        print("- %s, %s: %s in all %d" % (name, decision, act, n))
    if not one_act:
        print("- no person made ten or more decisions of a kind all one way")
    print("- single rows pinned: %s" % "; ".join(
        "%s %d of %d" % ((decision,) + pinned([r for r in rows if r["decision"] == decision], option))
        for decision, option in (("accept_invite", "accept"), ("invite_target", NO_ONE),
                                 ("leave_conversation", "leave"))))

    print("\n## each pair alone\n")
    print("| pair | level | conversations | lengths | P(invited) when free, each way | P(accept) | mean P(leave) |")
    print("|---|---|---|---|---|---|---|")
    for a, b in sorted({pair_of(x, y) for x in people for y in people if x != y},
                       key=lambda p: (-level(*p), p)):
        cs = by_pair.get((a, b), [])
        each = []
        for x, y in ((a, b), (b, a)):
            xs = [P(r, y) for r in invites if r["name"] == x
                  and y in [f["name"] for f in r["state"]["free_people"]]]
            each.append("%.2f (%d)" % (mean(xs), len(xs)) if xs else "-")
        acc = [P(r, "accept") for r in accepts if pair_of(r["name"], r["otherName"]) == (a, b)]
        lv_ = [P(r, "leave") for r in leaves if pair_of(r["name"], r["otherName"]) == (a, b)]
        print("| %s · %s | %d | %d | %s | %s | %s | %s |" % (
            a, b, level(a, b), len(cs), ", ".join(str(c["n"]) for c in cs) or "-",
            " / ".join(each), "%.2f (%d)" % (mean(acc), len(acc)) if acc else "-",
            "%.2f (%d)" % (mean(lv_), len(lv_)) if lv_ else "-"))


def town_values(d):
    """One interest-town arm's units for the criterion, by decision: {(world, person, other, level): mean},
    and the number of decisions behind each level. Whom: P(partner | someone) over the
    invitations where the partner was free; accept: P(accept) by inviter; leave: P(leave)
    at 15-52 messages (the bins of 20, 30 and 45)."""
    people = {p["name"]: pastimes_of(p["description"]) for p in read_table(d, "playerDescriptions")}
    level = lambda a, b: len(set(people[a]) & set(people[b]))
    world = os.path.basename(d.rstrip("/"))
    rows = [r for r in read_table(d, "decisions") if r["backend"] != "dice" and not r.get("fallback")]
    vals = {k: defaultdict(list) for k in ("whom", "accept", "leave")}
    counts = {k: defaultdict(int) for k in vals}
    for r in rows:
        ps = probs(r)
        if r["decision"] == "invite_target":
            q = ps.get(NO_ONE, 0.0)
            if q >= 1:
                continue
            for f in [f["name"] for f in r["state"]["free_people"]]:
                lv = level(r["name"], f)
                vals["whom"][(world, r["name"], f, lv)].append(ps.get(f, 0.0) / (1 - q))
                counts["whom"][lv] += 1
        elif r["decision"] == "accept_invite":
            lv = level(r["name"], r["otherName"])
            vals["accept"][(world, r["name"], r["otherName"], lv)].append(ps.get("accept", 0.0))
            counts["accept"][lv] += 1
        elif r["decision"] == "leave_conversation" and 15 <= (r.get("messagesSoFar") or 0) <= 52:
            lv = level(r["name"], r["otherName"])
            vals["leave"][(world, r["name"], r["otherName"], lv)].append(ps.get("leave", 0.0))
            counts["leave"][lv] += 1
    return ({k: {kk: mean(vv) for kk, vv in v.items()} for k, v in vals.items()},
            {k: dict(v) for k, v in counts.items()})


def print_town_verdicts(dirs):
    """The criterion over the interest town's arms: per world, and pooled with the spread taken over
    pairs and between the two worlds; a level with fewer than ten decisions of a kind is
    too few to read, and a verdict that holds in one world only is said so."""
    per = [(os.path.basename(d.rstrip("/")),) + town_values(d) for d in dirs]
    print("\n# The criterion over %s\n" % (" and ".join(w for w, _, _ in per) if len(per) > 1 else per[0][0]))
    for kind, rising, title in (("whom", True, "whom is invited, P(partner | someone)"),
                                ("accept", True, "P(accept)"),
                                ("leave", False, "P(leave) at 15-52 messages")):
        print("**%s**\n" % title)
        results = []
        for world, vals, counts in per:
            res = it_k(by_level(vals[kind]), rising, counts=counts[kind])
            results.append(res[0])
            print(verdict_line(world, res) + "; decisions by level %s" % ", ".join(
                "%d: %d" % (lv, counts[kind].get(lv, 0)) for lv in LEVELS))
        if len(per) > 1:
            pooled = {}
            for _, vals, _ in per:
                pooled.update(vals[kind])
            means = [{lv: mean(xs) for lv, xs in by_level(vals[kind]).items()} for _, vals, _ in per]
            between = max([abs(means[0].get(lv, 0) - means[1].get(lv, 0)) for lv in LEVELS
                           if lv in means[0] and lv in means[1]] or [0])
            total = defaultdict(int)
            for _, _, counts in per:
                for lv, n in counts[kind].items():
                    total[lv] += n
            res = it_k(by_level(pooled), rising, extra_spread=between, counts=total)
            note = ""
            if res[0] == "reads the level" and results.count("reads the level") == 1:
                note = " -- in one world only"
            print(verdict_line("pooled (between the worlds %.2f)" % between, res) + note)
        print()


# --- shared readings, used by the arms, the probes and the interest town ---

PINNED_LOW, PINNED_HIGH = 0.02, 0.98


def probs(row):
    """A row's probabilities as {option: p}, from an arm's list or a probe's dict."""
    ps = row["probabilities"]
    return {p["option"]: p["p"] for p in ps} if isinstance(ps, list) else dict(ps)


def pinned(rows, option=None):
    """(pinned, n): rows whose P(option) is <= 0.02 or >= 0.98; with no option, rows
    whose likeliest option is >= 0.98. A Jev row carries Jev's probabilities and a die's
    are known by construction; a row of a Gemma ARM is one sample, one-hot by
    construction, so it is always pinned and says nothing — Gemma's pinned reading comes
    from a probe's samples."""
    n = hit = 0
    for r in rows:
        ps = probs(r)
        if option is None:
            stuck = max(ps.values(), default=0) >= PINNED_HIGH
        else:
            p = ps.get(option, 0.0)
            stuck = p <= PINNED_LOW or p >= PINNED_HIGH
        n += 1
        hit += 1 if stuck else 0
    return hit, n


def invite_halves(rows):
    """The invite in two halves. Whether: the mean P(no one) and how many chose no one —
    for the judged backends only, the die cannot choose no one (D1). Whom: given someone
    was invited, how often each partner was chosen, and the mean P(partner | someone) =
    P(partner) / (1 - P(no one)) over the rows where that partner was on offer."""
    p_no_one, chose_no_one, chosen, cond = [], 0, defaultdict(int), defaultdict(list)
    for r in rows:
        ps = probs(r)
        q = ps.get(NO_ONE, 0.0)
        p_no_one.append(q)
        if r.get("chosen") == NO_ONE:
            chose_no_one += 1
        elif r.get("chosen") is not None:
            chosen[r["chosen"]] += 1
        if q < 1:
            for o, p in ps.items():
                if o != NO_ONE:
                    cond[o].append(p / (1 - q))
    return {"n": len(rows), "p_no_one": mean(p_no_one), "chose_no_one": chose_no_one,
            "someone": sum(chosen.values()), "whom_chosen": dict(chosen),
            "whom_p": {o: mean(ps) for o, ps in cond.items()}}


LENGTH_BINS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 40)]


def read_arm(name):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", name)
    def table(t):
        path = os.path.join(d, t + ".jsonl.gz")
        if not os.path.exists(path):
            return []
        with gzip.open(path, "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(d, "manifest.json")) as f:
        manifest = json.load(f)
    return manifest, {t: table(t) for t in ("decisions", "messages", "memories",
                                            "archivedConversations", "playerDescriptions")}


def print_arms(names):
    """The arms under runs/, read by rules fixed before they were read. One world per judge: every rate
    below is one trajectory's, never a pair across arms."""
    arms = [(n,) + read_arm(n) for n in names]
    print("One world per arm: every number is a rate of one trajectory; nothing pairs across arms.")
    for name, m, t in arms:
        c, ds = m["condition"], t["decisions"]
        lat = [r["latencyMs"] for r in ds if r.get("latencyMs") is not None and r["decision"] != "importance"]
        print("\n## %s — %s, chat %s, %s min, fork %s, $%.3f by the key's counter" % (
            name, c["backend"], c.get("chat_model") or c["chat"], m["window"]["minutes"],
            m["code"]["sha"], m["spend"]["openrouter_usd"] or 0))
        print("%d decisions, %d fallbacks, %d off the list; %d messages, %d memories (%s)" % (
            len(ds), m["counts"]["fallbacks"], m["counts"]["off_list"], len(t["messages"]),
            len(t["memories"]), ", ".join("%s %d" % kv for kv in sorted(
                defaultdict(int, {k: sum(1 for x in t["memories"] if x["data"]["type"] == k)
                                  for k in {x["data"]["type"] for x in t["memories"]}}).items()))))
        if c["backend"] != "dice" and lat:
            print("decision latency (not importance): median %d ms, p90 %d ms" % (pct(lat, 0.5), pct(lat, 0.9)))
        billed = sum(r.get("costUsd") or 0 for r in ds)
        if billed:
            print("decisions billed by the router: $%.4f" % billed)

    print("\n# Pinned probabilities (<= 0.02 or >= 0.98), per persona and decision")
    print("A Gemma arm's rows are one sample each, so always pinned: read Gemma's on p4-gemma / pl4-gemma / pv2-gemma.")
    print("A die's are known by construction: accept 0.8, activity 1/3, invite the first free player at 1 (pinned by rule).\n")
    print("| arm | persona | decision | n | pinned | mean P | act share |")
    print("|---|---|---|---|---|---|---|")
    key = {"accept_invite": "accept", "invite_target": NO_ONE, "leave_conversation": "leave", "activity": None}
    for name, m, t in arms:
        if m["condition"]["backend"] == "gemma":
            continue
        for decision in ("accept_invite", "invite_target", "leave_conversation", "activity"):
            for persona in sorted({r["name"] for r in t["decisions"] if r["decision"] == decision}):
                rows = [r for r in t["decisions"] if r["decision"] == decision and r["name"] == persona]
                hit, n = pinned(rows, key[decision])
                if key[decision]:
                    mp = mean([probs(r).get(key[decision], 0.0) for r in rows])
                    share = sum(1 for r in rows if r["chosen"] == key[decision]) / n
                    label = "P(%s)" % key[decision]
                else:
                    mp = mean([max(probs(r).values()) for r in rows])
                    share = None
                    label = "max P"
                print("| %s | %s | %s | %d | %d of %d | %s %.2f | %s |" % (
                    name, persona, decision, n, hit, n, label, mp,
                    "%.0f%%" % (100 * share) if share is not None else "-"))

    print("\n# The invite in two halves")
    print("Whether (P(no one)) for the judged backends only: the die cannot choose no one (D1).")
    print("Whom, given someone was invited, for all three. How many conversations happen is not the judge's alone.\n")
    print("| arm | persona | n | whether: mean P(no one) | chose no one | whom: chosen (given someone) |")
    print("|---|---|---|---|---|---|")
    for name, m, t in arms:
        for persona in sorted({r["name"] for r in t["decisions"] if r["decision"] == "invite_target"}):
            rows = [r for r in t["decisions"] if r["decision"] == "invite_target" and r["name"] == persona]
            h = invite_halves(rows)
            whether = "-" if m["condition"]["backend"] == "dice" else "%.2f" % h["p_no_one"]
            print("| %s | %s | %d | %s | %d | %s |" % (
                name, persona, h["n"], whether, h["chose_no_one"],
                ", ".join("%s %d" % kv for kv in sorted(h["whom_chosen"].items(), key=lambda kv: -kv[1])) or "-"))

    print("\n# P(leave) by message count (judged arms; norm 25, cap 40)\n")
    print("| arm | persona | " + " | ".join("%d-%d" % b for b in LENGTH_BINS) + " |")
    print("|---|---|" + "---|" * len(LENGTH_BINS))
    for name, m, t in arms:
        if m["condition"]["backend"] == "dice":
            continue
        rows = [r for r in t["decisions"] if r["decision"] == "leave_conversation"]
        for persona in sorted({r["name"] for r in rows}) + ["all"]:
            sel = rows if persona == "all" else [r for r in rows if r["name"] == persona]
            cells = []
            for lo, hi in LENGTH_BINS:
                b = [probs(r).get("leave", 0.0) for r in sel if lo <= (r.get("messagesSoFar") or 0) <= hi]
                cells.append("%.2f (%d)" % (mean(b), len(b)) if b else "-")
            print("| %s | %s | %s |" % (name, persona, " | ".join(cells)))

    print("\n# How conversations ended, and their lengths")
    print("Length is D4's before it is the judge's: the die ends every talk at 10 messages by rule;")
    print("the judged backends are told a norm of 25 under a cap of 40.\n")
    print("| arm | conversations | median length | max | chose leave | cap (40) | rule (dice) | running at the end | no message |")
    print("|---|---|---|---|---|---|---|---|---|")
    pair_rows = []
    for name, m, t in arms:
        names = {p["playerId"]: p["name"] for p in t["playerDescriptions"]}
        by = defaultdict(list)
        for msg in t["messages"]:
            by[msg["conversationId"]].append(msg)
        left = {r["conversationId"] for r in t["decisions"]
                if r["decision"] == "leave_conversation" and r["chosen"] == "leave"}
        archived = {a["id"]: a for a in t["archivedConversations"]}
        ended = defaultdict(int)
        lengths = []
        for cid, msgs in by.items():
            n = len(msgs)
            lengths.append(n)
            if cid not in archived:
                how = "running"
            elif m["condition"]["backend"] == "dice":
                how = "rule"
            elif cid in left:
                how = "leave"
            elif n >= 40:
                how = "cap"
            else:
                how = "other"
            ended[how] += 1
            who = " & ".join(sorted({names.get(x["author"], x["author"]) for x in msgs}))
            pair_rows.append((name, who, n, how))
        silent = sum(1 for a in t["archivedConversations"] if a.get("numMessages", 0) == 0)
        lengths.sort()
        print("| %s | %d | %d | %d | %d | %d | %d | %d | %d |" % (
            name, len(lengths), pct(lengths, 0.5) if lengths else 0, lengths[-1] if lengths else 0,
            ended["leave"], ended["cap"], ended["rule"], ended["running"], silent)
            + (" other %d" % ended["other"] if ended["other"] else ""))
    print("\n# Who talked with whom (conversations with at least one message: count, lengths, how they ended)\n")
    print("| arm | pair | conversations | lengths | ended |")
    print("|---|---|---|---|---|")
    grouped = defaultdict(list)
    for name, who, n, how in pair_rows:
        grouped[(name, who)].append((n, how))
    for (name, who), xs in sorted(grouped.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
        hows = defaultdict(int)
        for _, how in xs:
            hows[how] += 1
        print("| %s | %s | %d | %s | %s |" % (name, who, len(xs), ", ".join(str(n) for n, _ in xs),
                                           ", ".join("%s %d" % kv for kv in sorted(hows.items()))))

    print("\n# Importance, by each arm's own judge (reflection needs an agent's sum past 500)\n")
    print("| arm | persona | memories | mean score | sum |")
    print("|---|---|---|---|---|")
    for name, m, t in arms:
        rows = [r for r in t["decisions"] if r["decision"] == "importance"]
        for persona in sorted({r["name"] for r in rows}) + ["all"]:
            sel = rows if persona == "all" else [r for r in rows if r["name"] == persona]
            xs = [r["score"] for r in sel]
            print("| %s | %s | %d | %.2f | %.0f |" % (name, persona, len(xs), mean(xs), sum(xs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl")
    parser.add_argument("--probe")
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--variants")
    parser.add_argument("--interests", nargs="+")
    parser.add_argument("--town", nargs="+")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--leave")
    args = parser.parse_args()
    if args.arms:
        print_arms(args.arms)
        return
    if args.leave:
        print_leave(args.leave)
        return
    if args.town:
        arms = [p for p in args.town if os.path.isdir(p)]
        probe_files = [p for p in args.town if not os.path.isdir(p)]
        for d in arms:
            print_town(d, probe_files)
            print()
        print_town_verdicts(arms)
        return
    if args.interests:
        print_interests(args.interests)
        return
    if args.variants:
        print_variants(args.variants)
        return
    if args.runs:
        print_runs(args.runs)
        return
    if args.probe:
        print_probe(args.probe)
        return
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

    for decision in ("activity", "accept_invite", "invite_target", "leave_conversation"):
        options = sorted(options_of[decision])
        if not options:
            continue
        print("\n## %s\n" % decision)
        header = "| persona | arm | n | " + " | ".join(options) + " |"
        print(header)
        print("|" + "---|" * (3 + len(options)))
        names = sorted({name for (d, name, _) in groups if d == decision})
        for name in names:
            for backend in BACKENDS:
                g = groups.get((decision, name, backend))
                if not g:
                    continue
                cells = []
                for o in options:
                    share = g["taken"][o] / g["n"]
                    cells.append("%3.0f%% [%.2f]" % (100 * share, g["p"][o] / g["n"]))
                n = str(g["n"]) + (" (%d fallback)" % g["fallbacks"] if g["fallbacks"] else "")
                print("| %s | %s | %s | %s |" % (name, backend, n, " | ".join(cells)))

    importance = [r for r in rows if r["decision"] == "importance"]
    if importance:
        print("\n## importance: mean score (0-9), by persona and arm\n")
        print("| persona | " + " | ".join(BACKENDS) + " |")
        print("|" + "---|" * (1 + len(BACKENDS)))
        by = defaultdict(list)
        for r in importance:
            by[(r["name"], r["backend"])].append(r["score"])
        for name in sorted({r["name"] for r in importance}):
            cells = []
            for b in BACKENDS:
                xs = by.get((name, b))
                cells.append("%.2f (n=%d)" % (sum(xs) / len(xs), len(xs)) if xs else "-")
            print("| %s | %s |" % (name, " | ".join(cells)))
        cells = []
        for b in BACKENDS:
            xs = [r["score"] for r in importance if r["backend"] == b]
            cells.append("%.2f (n=%d)" % (sum(xs) / len(xs), len(xs)) if xs else "-")
        print("| **all** | %s |" % " | ".join(cells))

    print()
    for b in ("jev", "gemma"):
        judged = [r for r in rows if r["backend"] == b]
        if not judged:
            continue
        lat = [r["latencyMs"] for r in judged if r.get("latencyMs") is not None]
        falls = [r for r in judged if r.get("fallback")]
        offlist = [r for r in falls if (r.get("error") or "").startswith("off the list")]
        line = "%s: %d decisions, %d fallbacks (%d off the list), latency median %d ms, p90 %d ms" % (
            b, len(judged), len(falls), len(offlist), pct(lat, 0.5), pct(lat, 0.9))
        if b == "gemma":
            labelled = [r for r in judged if r.get("raw") is not None and r["decision"] != "importance"
                        and not r.get("fallback")]
            lenient = [r for r in labelled if normalize(r["raw"]) != normalize(r["chosen"])]
            cost = sum(r.get("costUsd") or 0 for r in judged)
            line += ", %d lenient matches, billed $%.4f" % (len(lenient), cost)
        print(line)
        for r in offlist[:10]:
            print("    off the list (%s): %r" % (r["decision"], r.get("raw")))


def print_runs(dirs):
    """The conversations of exported runs: how long, between whom."""
    for d in dirs:
        meta = json.load(open(os.path.join(d, "run.json")))
        names = {}
        for line in open(os.path.join(d, "playerDescriptions.jsonl")):
            if line.strip():
                p = json.loads(line)
                names[p["playerId"]] = p["name"]
        convs = defaultdict(list)
        for line in open(os.path.join(d, "messages.jsonl")):
            if line.strip():
                m = json.loads(line)
                convs[m["conversationId"]].append(m)
        lengths = sorted(len(ms) for ms in convs.values())
        pairs = defaultdict(list)
        for ms in convs.values():
            who = tuple(sorted({names.get(m["author"], m["author"]) for m in ms}))
            pairs[who].append(len(ms))
        spent = None
        if meta.get("openrouter_usage_end") is not None and meta.get("openrouter_usage_start") is not None:
            spent = meta["openrouter_usage_end"] - meta["openrouter_usage_start"]
        print("\n## %s: arm %s, chat %s, %s min%s\n" % (
            os.path.basename(d.rstrip("/")), meta["arm"], meta["chat_backend"], meta["minutes"],
            ", OpenRouter $%.4f" % spent if spent is not None else ""))
        if not lengths:
            print("no conversations")
            continue
        print("%d conversations, %d messages; messages per conversation: median %d, max %d" % (
            len(lengths), sum(lengths), pct(lengths, 0.5), lengths[-1]))
        print("\n| pair | conversations | messages each |")
        print("|---|---|---|")
        for who, ls in sorted(pairs.items(), key=lambda kv: -len(kv[1])):
            print("| %s | %d | %s |" % (" & ".join(who), len(ls), ", ".join(map(str, ls))))


if __name__ == "__main__":
    main()
