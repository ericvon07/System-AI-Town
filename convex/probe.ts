// Jev's answers for every persona and every pair, on constructed states,
// outside any world.
//
// A 25-minute window gives each persona a handful of invitations, too few to
// read a rate from. The probe asks the very same questions the seam asks
// (decisions.ts) for the whole grid instead: each persona's activity, each
// invitee x inviter pair, and each persona choosing among the other four.
// The states are built from data/characters.ts exactly as decisionContext
// builds them in a world, with everyone a stranger (never talked) and every
// candidate at the same distance, so what moves an answer is the persona.
//
//   npx convex run probe:probeJev                 distance 5 tiles
//   npx convex run probe:probeJev '{"distance": 20}'

import { v } from 'convex/values';
import { internalAction, internalQuery } from './_generated/server';
import { internal } from './_generated/api';
import { Descriptions } from '../data/characters';
import { ACTIVITIES } from './constants';
import { askJev, choice, score, ScoreAnswer } from './util/jev';
import { askOpenRouter, gemmaModel } from './util/openrouter';
import { getLLMConfig } from './util/llm';
import {
  Me,
  acceptQuestion,
  choicePrompt,
  matchOption,
  activityQuestion,
  importanceQuestion,
  inviteQuestion,
  leaveQuestion,
  parseImportance,
  upstreamImportancePrompt,
} from './decisions';

type Question = { state: unknown; instructions: string; criteria: Record<string, string> };

// The grid, asked through `ask`: each persona's activity, each invitee x
// inviter pair, each persona choosing among the other four.
async function probeGrid(
  distance: number,
  ask: (q: Question) => Promise<Record<string, number>>,
) {
  const people: Me[] = Descriptions.map((d) => ({
    name: d.name,
    identity: d.identity,
    goal: d.plan,
  }));
  const activity: Array<{ name: string; probabilities: Record<string, number> }> = [];
  const accept: Array<{ name: string; inviter: string; accept: number }> = [];
  const invite: Array<{ name: string; probabilities: Record<string, number> }> = [];
  for (const me of people) {
    activity.push({
      name: me.name,
      probabilities: await ask(
        activityQuestion(
          me,
          ACTIVITIES.map((a) => a.description),
        ),
      ),
    });
    const others = people.filter((p) => p.name !== me.name);
    for (const inviter of others) {
      const probabilities = await ask(
        acceptQuestion(me, { name: inviter.name, identity: inviter.identity, lastEnded: null }, 0),
      );
      accept.push({ name: me.name, inviter: inviter.name, accept: probabilities.accept ?? 0 });
    }
    invite.push({
      name: me.name,
      probabilities: await ask(
        inviteQuestion(
          me,
          others.map((p) => ({ label: p.name, identity: p.identity, distance, lastEnded: null })),
          0,
        ),
      ),
    });
  }
  return { activity, accept, invite };
}

export const probeJev = internalAction({
  args: { distance: v.optional(v.number()) },
  handler: async (_ctx, args) => {
    const distance = args.distance ?? 5;
    let model: string | undefined;
    const grid = await probeGrid(distance, async (q) => {
      const response = await askJev(q.state, { q: choice(q.instructions, q.criteria) });
      model = response.model;
      return response.answers.q.probabilities;
    });
    return { model, distance, ...grid };
  },
});

// The same grid asked of Gemma through OpenRouter, the gemma backend's prompt
// (decisions.ts: choicePrompt), `samples` times per question at temperature 1.
// A probability is the share of samples that named the option; the share
// that named none is reported as OFF_LIST.
//
//   npx convex run probe:probeGemma                    10 samples, 5 tiles
//   npx convex run probe:probeGemma '{"samples": 20}'
export const OFF_LIST = '(off the list)';

export const probeGemma = internalAction({
  args: { distance: v.optional(v.number()), samples: v.optional(v.number()) },
  handler: async (_ctx, args) => {
    const distance = args.distance ?? 5;
    const samples = args.samples ?? 10;
    let costUsd = 0;
    let calls = 0;
    const offList: Array<{ question: string; raw: string }> = [];
    const providers = new Set<string>();
    const grid = await probeGrid(distance, async (q) => {
      const options = Object.keys(q.criteria);
      const counts: Record<string, number> = Object.fromEntries(
        [...options, OFF_LIST].map((o) => [o, 0]),
      );
      const answers = await Promise.all(
        Array.from({ length: samples }, () =>
          askOpenRouter(choicePrompt(q), { maxTokens: 24, temperature: 1 }),
        ),
      );
      for (const answer of answers) {
        calls += 1;
        costUsd += answer.costUsd ?? 0;
        if (answer.provider) {
          providers.add(answer.provider);
        }
        const chosen = matchOption(answer.content, options);
        if (chosen) {
          counts[chosen] += 1;
        } else {
          counts[OFF_LIST] += 1;
          offList.push({ question: q.instructions.slice(0, 60), raw: answer.content });
        }
      }
      return Object.fromEntries(Object.entries(counts).map(([o, n]) => [o, n / samples]));
    });
    return {
      model: gemmaModel(),
      providers: [...providers],
      distance,
      samples,
      calls,
      costUsd,
      offList,
      ...grid,
    };
  },
});

// The importance of the same memories by each of the three judges, outside
// any world: upstream's prompt on the local Ollama model, the same prompt on
// Gemma through OpenRouter, and Jev's typed Score. Reads the memories of the
// world that is in the database (run it after an arm, before the next wipe),
// or the descriptions it is given.
//
//   npx convex run probe:probeImportance
//   npx convex run probe:probeImportance '{"judges": ["jev"]}'
export const probeImportance = internalAction({
  args: {
    descriptions: v.optional(v.array(v.string())),
    judges: v.optional(v.array(v.string())),
  },
  handler: async (ctx, args) => {
    const judges = args.judges ?? ['ollama', 'gemma', 'jev'];
    const memories: Array<{ name: string; kind: string; description: string; arm: number }> =
      args.descriptions
        ? args.descriptions.map((description) => ({
            name: '',
            kind: 'given',
            description,
            arm: NaN,
          }))
        : await ctx.runQuery(internal.probe.worldMemories, {});
    const ollama = getLLMConfig();
    const ask: Record<string, (description: string) => Promise<{ score: number; raw?: string }>> =
      {
        ollama: async (description) => {
          const response = await fetch(ollama.url + '/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: ollama.chatModel,
              messages: [{ role: 'user', content: upstreamImportancePrompt(description) }],
              temperature: 0,
              max_tokens: 1,
            }),
          });
          const raw = (await response.json()).choices[0].message.content as string;
          return { ...parseImportance(raw), raw };
        },
        gemma: async (description) => {
          const answer = await askOpenRouter(
            [{ role: 'user', content: upstreamImportancePrompt(description) }],
            { maxTokens: 1, temperature: 0 },
          );
          return { ...parseImportance(answer.content), raw: answer.content };
        },
        jev: async (description) => {
          const q = importanceQuestion(description);
          const { answers } = await askJev<ScoreAnswer>(q.state, {
            importance: score(q.instructions, q.criteria),
          });
          return { score: answers.importance.score };
        },
      };
    const rows = [];
    for (const memory of memories) {
      const scores: Record<string, number | null> = {};
      const raws: Record<string, string | undefined> = {};
      await Promise.all(
        judges.map(async (judge) => {
          try {
            const { score, raw } = await ask[judge](memory.description);
            scores[judge] = score;
            raws[judge] = raw;
          } catch (e) {
            scores[judge] = null;
            raws[judge] = `error: ${e instanceof Error ? e.message : String(e)}`;
          }
        }),
      );
      rows.push({ ...memory, scores, raws });
    }
    return { model: { ollama: ollama.chatModel, gemma: gemmaModel() }, judges, rows };
  },
});

export const worldMemories = internalQuery({
  args: {},
  handler: async (ctx) => {
    const memories = await ctx.db.query('memories').collect();
    const names = new Map<string, string>();
    for (const d of await ctx.db.query('playerDescriptions').collect()) {
      names.set(d.playerId, d.name);
    }
    return memories.map((m) => ({
      name: names.get(m.playerId) ?? m.playerId,
      kind: m.data.type,
      description: m.description,
      arm: m.importance,
    }));
  },
});

// leave_conversation on constructed conversations: each persona talking with
// each other one, the transcript a run of small talk that repeats itself, cut
// at each of `lengths` messages. What should move P(leave) is who `me` is and
// how long the talk has gone round in circles.
//
//   npx convex run probe:probeLeave                                  Jev
//   npx convex run probe:probeLeave '{"backend": "gemma", "samples": 10}'
//
// Under gemma, P(leave) is the share of `samples` answers naming `leave` (the
// gemma backend's prompt, choicePrompt, at temperature 1); answers naming
// neither option are counted in `offList`, not in either share.
const SMALL_TALK = [
  'Nice weather today, isn’t it?',
  'Yes, very nice. Sunny.',
  'Have you been well?',
  'Well enough, thanks. And you?',
];

function smallTalkQuestion(me: Me, other: Me, n: number) {
  const transcript = Array.from({ length: n }, (_, i) => ({
    speaker: i % 2 === 0 ? other.name : me.name,
    text: SMALL_TALK[i % SMALL_TALK.length],
  }));
  return leaveQuestion(me, other, Math.round(n * 0.35 * 10) / 10, transcript);
}

export const probeLeave = internalAction({
  args: {
    lengths: v.optional(v.array(v.number())),
    backend: v.optional(v.union(v.literal('jev'), v.literal('gemma'))),
    samples: v.optional(v.number()),
  },
  handler: async (_ctx, args) => {
    const lengths = args.lengths ?? [2, 8, 20, 25, 30, 38];
    const backend = args.backend ?? 'jev';
    const samples = args.samples ?? 10;
    const people: Me[] = Descriptions.map((d) => ({
      name: d.name,
      identity: d.identity,
      goal: d.plan,
    }));
    let model: string | undefined;
    let costUsd = 0;
    let calls = 0;
    const providers = new Set<string>();
    const offList: Array<{ name: string; other: string; messages: number; raw: string }> = [];
    const rows: Array<{ name: string; other: string; messages: number; leave: number }> = [];
    for (const me of people) {
      for (const other of people.filter((p) => p.name !== me.name)) {
        await Promise.all(
          lengths.map(async (n) => {
            const q = smallTalkQuestion(me, other, n);
            let leave: number;
            if (backend === 'jev') {
              const response = await askJev(q.state, { q: choice(q.instructions, q.criteria) });
              model = response.model;
              leave = response.answers.q.probabilities.leave ?? 0;
            } else {
              const options = Object.keys(q.criteria);
              const answers = await Promise.all(
                Array.from({ length: samples }, () =>
                  askOpenRouter(choicePrompt(q), { maxTokens: 24, temperature: 1 }),
                ),
              );
              let leaves = 0;
              for (const answer of answers) {
                calls += 1;
                costUsd += answer.costUsd ?? 0;
                model = answer.model;
                if (answer.provider) {
                  providers.add(answer.provider);
                }
                const chosen = matchOption(answer.content, options);
                if (chosen === 'leave') {
                  leaves += 1;
                } else if (!chosen) {
                  offList.push({ name: me.name, other: other.name, messages: n, raw: answer.content });
                }
              }
              leave = leaves / samples;
            }
            rows.push({ name: me.name, other: other.name, messages: n, leave });
          }),
        );
      }
    }
    return backend === 'jev'
      ? { backend, model, lengths, rows }
      : { backend, model, providers: [...providers], samples, calls, costUsd, offList, lengths, rows };
  },
});

// What each judge is handed for one persona, side by side, with no call made:
// for every decision, the question Jev receives (state, instructions and the
// options with their descriptions) and the prompt Gemma reads, built from that
// same object (choicePrompt; for importance, upstream's prompt, FT-E). The
// states are the probes' — everyone a stranger at `distance` tiles, the leave
// question on eight messages of small talk.
//
//   npx convex run probe:showQuestions '{"name": "Bob"}'
export const showQuestions = internalQuery({
  args: { name: v.optional(v.string()), distance: v.optional(v.number()) },
  handler: async (_ctx, args) => {
    const distance = args.distance ?? 5;
    const people: Me[] = Descriptions.map((d) => ({
      name: d.name,
      identity: d.identity,
      goal: d.plan,
    }));
    const me = people.find((p) => p.name === (args.name ?? 'Bob'));
    if (!me) {
      throw new Error(`no persona named ${args.name}`);
    }
    const others = people.filter((p) => p.name !== me.name);
    const questions: Record<string, Question> = {
      activity: activityQuestion(
        me,
        ACTIVITIES.map((a) => a.description),
      ),
      accept_invite: acceptQuestion(
        me,
        { name: others[0].name, identity: others[0].identity, lastEnded: null },
        0,
      ),
      invite_target: inviteQuestion(
        me,
        others.map((p) => ({ label: p.name, identity: p.identity, distance, lastEnded: null })),
        0,
      ),
      leave_conversation: smallTalkQuestion(me, others[0], 8),
    };
    const memory = `${me.name} talked with ${others[0].name} about the weather.`;
    const importance = importanceQuestion(memory);
    return {
      ...Object.fromEntries(
        Object.entries(questions).map(([decision, q]) => [
          decision,
          { jev: q, gemma: choicePrompt(q)[0].content },
        ]),
      ),
      importance: { jev: importance, gemma: upstreamImportancePrompt(memory) },
    };
  },
});
