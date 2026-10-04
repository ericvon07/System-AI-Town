# Results

Each file here is a trace or a probe a number in a write-up can point to.

## 2026-10-03: the first pair of arms

`2026-10-03-arms.jsonl` (149 rows) and its readout `2026-10-03-arms-readout.md`
(`python3 scripts/readout.py --jsonl results/2026-10-03-arms.jsonl`).

| | dice | jev |
|---|---|---|
| code | `3c6884b` | `3c6884b` |
| world | `m17eh2e8be4nfrk08n2vxpf02s8fm1fs` | `m172anzxq0wnmpagk0zqh2jaxh8fnd2t` |
| window | 2026-10-03 23:33:22 to 23:58:50 CDT | 23:59:25 to 2026-10-04 00:24:37 CDT |
| decisions | 69 | 80 (0 fallbacks) |

Both: five agents from `data/characters.ts`, `scripts/run-arm.sh <arm> 25`,
conversations on `llama3.1:8b` through a local Ollama, embeddings on
`mxbai-embed-large`, Jev `jev-1.13.0`, `JEV_POLICY=sample`. Jev latency:
median 174 ms, p90 291 ms.

Two dice rows of the raw export were left out: they belong to the world that
was running before the seam was deployed (`m175fq2pz8dvwnpsvt7hrjt0x98fngf3`),
which recorded two decisions after the push and before the wipe.

Rates, not pairs: the two worlds stop visiting the same states at the first
decision that differs. With 25 minutes and five agents, most persona rows hold
one to sixteen decisions, so the act shares are anecdotes; the mean probability
beside each share is the steadier reading for the jev arm.

## 2026-10-04: the probe

`2026-10-04-probe-d5.json` and `2026-10-04-probe-d5.md`
(`npx convex run probe:probeJev`, code `db14918`, whose questions are
`3c6884b`'s, checked by diffing every string literal).
Jev `jev-1.13.0`. Every persona's activity, every invitee x inviter pair, and
every persona choosing among the other four, everyone a stranger, every
candidate 5 tiles away: 30 questions, one request each.
