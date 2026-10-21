# Runs

The runs behind the AI Town results in version 20 of the System One essay (October 2026), so
that every number can be recomputed from the data. `results/` holds the first pair of runs,
from October 3 and 4, 2026.

There are two kinds of run:

- **Arms** ran the whole town on a fresh world for a fixed time with one backend deciding. Each
  folder holds the tables the town wrote, one JSON object per line: `decisions` (every
  decision with its options, probabilities and the act taken), `messages`,
  `archivedConversations`, `participatedTogether`, `memories` and `playerDescriptions`.
- **Probes** asked the same questions over constructed states, outside a running town
  (`convex/probe.ts`). Each folder holds the probe's output, `probe.json.gz`.

Every folder has a `manifest.json`: the backend and models, the code, the window, the counts,
what the OpenRouter key spent, and the sha1 of each data file. `code.sha` is the commit the
run was made on in the private repository where this fork was developed, and the one the data
files record; `code.public_sha` is the same code in this repository.

## What each part of the essay reads

All commands run from the repository root with Python 3 and no dependencies.

**The town's original rules against a judgment, and Bob.** Three hour-long arms on the five
original characters, one per backend: `b2-dice` (upstream's dice and rules), `b1-jev` and
`b3-gemma`. Bob's 19 of 23 accepted invitations under the dice and his 0 of 47 under Jev are
in the `accept_invite` rows; the decision latency and what each hour cost are at the top of
each arm's block.

```sh
python3 scripts/readout.py --arms b1-jev b2-dice b3-gemma
```

**Lorenzo, whose sociability changes one sentence at a time.** The designed characters of
`data/variants.ts`, asked by both judges on October 7, 2026. The code calls him Sam; the essay
calls him Lorenzo. The sociability family averages P(accept) over the four inviters.

```sh
python3 scripts/readout.py --variants runs/pv1-jev/probe.json.gz
python3 scripts/readout.py --variants runs/pv2-gemma/probe.json.gz
```

**A town of shared interests.** The isolated test of whom to invite, by how many pastimes two
people share, asked in two dealings of the pastimes on October 9, 2026 (`pt2-jev`, `pt3-jev`,
`pt4-gemma`, `pt5-gemma`), and the two hour-long worlds where Jev decided (`t1-jev`,
`t2-jev`).

```sh
python3 scripts/readout.py --interests runs/pt2-jev/probe.json.gz runs/pt3-jev/probe.json.gz \
    runs/pt4-gemma/probe.json.gz runs/pt5-gemma/probe.json.gz
python3 scripts/readout.py --town runs/t1-jev runs/t2-jev \
    runs/pt2-jev/probe.json.gz runs/pt3-jev/probe.json.gz
```

**Supporting probes.** Every character's answers over the whole grid of states (`p3-jev`,
`p4-gemma`) and the leave question over small talk at growing lengths (`pl3-jev`,
`pl4-gemma`). `s2-gemma` is a 10-minute arm with Gemma deciding in the afternoon, when its
latency was much higher than in `b3-gemma`.

```sh
python3 scripts/readout.py --probe runs/p3-jev/probe.json.gz
python3 scripts/readout.py --leave runs/pl3-jev/probe.json.gz
```

## Reading the numbers

Two arms stop visiting the same states at the first decision that differs, so the readout
gives rates per arm, decision and character and never pairs a decision of one arm with one of
another. Each arm and each world ran once, so another run could change its numbers.
