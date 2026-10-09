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
import { COUNTERPARTS, VARIANT_FAMILIES } from '../data/variants';
import {
  Dealing,
  NAMES,
  PASTIMES,
  PASTIMES_OF,
  SEEDS,
  TownCast,
  interestDealing,
  interestDescriptions,
  level,
  townCast,
} from '../data/interests';
import { ACTIVITIES } from './constants';
import { askJev, choice, score, ScoreAnswer } from './util/jev';
import { askOpenRouter, gemmaModel } from './util/openrouter';
import { getLLMConfig } from './util/llm';
import {
  LeaveWording,
  Me,
  acceptQuestion,
  choicePrompt,
  conversationCap,
  matchOption,
  activityQuestion,
  importanceQuestion,
  inviteQuestion,
  leaveNorm,
  leaveQuestion,
  parseImportance,
  upstreamImportancePrompt,
} from './decisions';
import {
  continueConversationPrompt,
  historyMessages,
  leaveConversationPrompt,
  startConversationPrompt,
} from './agent/conversation';

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

function smallTalkQuestion(me: Me, other: Me, n: number, norm?: number, wording?: LeaveWording) {
  const transcript = Array.from({ length: n }, (_, i) => ({
    speaker: i % 2 === 0 ? other.name : me.name,
    text: SMALL_TALK[i % SMALL_TALK.length],
  }));
  return leaveQuestion(me, other, Math.round(n * 0.35 * 10) / 10, transcript, norm, wording);
}

// The people of a town, as the questions read them; the interest town in the
// dealing given, or the deployment's (INTEREST_DEALING).
function peopleOf(cast: TownCast, dealing: Dealing = interestDealing()): Me[] {
  return (cast === 'interests' ? interestDescriptions(dealing) : Descriptions).map((d) => ({
    name: d.name,
    identity: d.identity,
    goal: d.plan,
  }));
}

// The interest town's talks about a pastime (003-IP, IT-J): `other` speaks of
// one of their pastimes and `me` answers with the same neutral lines, four
// lines repeating as the small talk does. The two talks of a pair differ in the
// pastime's name alone: one both list ('both'), or one only `other` lists
// ('other only'), the first of `other`'s list, in its fixed order, that fits.
// What differs is whether `me`'s own identity lists it. Like every text the lab
// writes for this town, no line holds interest, interested, interesting,
// share, shared, common or similar.
export type Talk = 'small' | 'both' | 'other only';

function talkLines(pastime: string) {
  return [
    `I spent a few hours on ${pastime} again today.`,
    'Oh? How did it go?',
    `Well. I always lose track of time with ${pastime}.`,
    'I can see why.',
  ];
}

function talkPastime(me: Me, other: Me, talk: 'both' | 'other only', dealing: Dealing) {
  const mine = PASTIMES_OF[dealing][me.name];
  return PASTIMES_OF[dealing][other.name].find((p) =>
    talk === 'both' ? mine.includes(p) : !mine.includes(p),
  );
}

function pastimeTalkQuestion(
  me: Me,
  other: Me,
  pastime: string,
  n: number,
  norm?: number,
  wording?: LeaveWording,
) {
  const lines = talkLines(pastime);
  const transcript = Array.from({ length: n }, (_, i) => ({
    speaker: i % 2 === 0 ? other.name : me.name,
    text: lines[i % lines.length],
  }));
  return leaveQuestion(me, other, Math.round(n * 0.35 * 10) / 10, transcript, norm, wording);
}

// The three prompts of a conversation between `me` and `other` as the chat
// model receives them (agent/conversation.ts), on constructed inputs: the
// opening at a first meeting and at a later one, a reply and a goodbye after
// three messages of the small talk above. Where the town puts what an agent
// remembers, a marked slot stands instead; the lines about memories and about
// the last conversation appear only when the town has them.
function conversationPrompts(me: Me, other: Me, now: number) {
  const player = { id: 'me', name: me.name };
  const otherPlayer = { id: 'other', name: other.name };
  const agent = { identity: me.identity, plan: me.goal ?? '' };
  const otherAgent = { identity: other.identity, plan: other.goal ?? '' };
  const memories = [
    { description: `[slot: up to three of ${me.name}'s memories, the most relevant to this talk]` },
  ];
  const history = historyMessages(
    player,
    otherPlayer,
    SMALL_TALK.slice(0, 3).map((text, i) => ({
      author: i % 2 === 0 ? otherPlayer.id : player.id,
      text,
    })),
  );
  return {
    start_first_meeting: startConversationPrompt(
      player,
      otherPlayer,
      agent,
      otherAgent,
      null,
      memories,
      false,
    ),
    start_again: startConversationPrompt(
      player,
      otherPlayer,
      agent,
      otherAgent,
      { created: now - 60 * 60_000 },
      memories,
      true,
    ),
    continue: continueConversationPrompt(
      player,
      otherPlayer,
      agent,
      otherAgent,
      memories,
      new Date(now - 3 * 60_000),
      now,
      history,
    ),
    leave: leaveConversationPrompt(player, otherPlayer, agent, otherAgent, history),
  };
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
// question on eight messages of small talk — and `other` is the inviter and
// the partner in the leave question. `conversation` is the three prompts of a
// conversation between the two as the chat model receives them.
//
// The town is the deployment's (TOWN_CAST) unless `cast` names one. In the
// interest town (003-IP) the dealing is the deployment's (INTEREST_DEALING)
// unless `dealing` names one, and the default pair is Avery and their partner
// at level 3; `town` adds the six, the pastimes dealt and how many each pair
// shares; the leave question comes in both wordings, the town's neutral one
// and, for the probe only, the one with the five-person town's clause; and,
// for the probe only, over the two talks about a pastime (IT-J) -- one both
// list (when the pair has one) and one only `other` lists.
//
//   npx convex run probe:showQuestions '{"name": "Bob"}'
//   npx convex run probe:showQuestions '{"cast": "interests", "dealing": 2, "name": "Avery"}'
export const showQuestions = internalQuery({
  args: {
    name: v.optional(v.string()),
    other: v.optional(v.string()),
    distance: v.optional(v.number()),
    cast: v.optional(v.union(v.literal('upstream'), v.literal('interests'))),
    dealing: v.optional(v.union(v.literal(1), v.literal(2))),
  },
  handler: async (_ctx, args) => {
    const distance = args.distance ?? 5;
    const cast = args.cast ?? townCast();
    const interests = cast === 'interests';
    const dealing: Dealing = args.dealing ?? interestDealing();
    const people = peopleOf(cast, dealing);
    const me = people.find((p) => p.name === (args.name ?? (interests ? 'Avery' : 'Bob')));
    if (!me) {
      throw new Error(`no persona named ${args.name}`);
    }
    const others = people.filter((p) => p.name !== me.name);
    const other = args.other
      ? others.find((p) => p.name === args.other)
      : interests
        ? others.find((p) => level(me.name, p.name, dealing) === 3)
        : others[0];
    if (!other) {
      throw new Error(`no other persona named ${args.other}`);
    }
    const { norm, wording } = leaveNorm(cast);
    const questions: Record<string, Question> = {
      activity: activityQuestion(
        me,
        ACTIVITIES.map((a) => a.description),
      ),
      accept_invite: acceptQuestion(
        me,
        { name: other.name, identity: other.identity, lastEnded: null },
        0,
      ),
      invite_target: inviteQuestion(
        me,
        others.map((p) => ({ label: p.name, identity: p.identity, distance, lastEnded: null })),
        0,
      ),
      leave_conversation: smallTalkQuestion(me, other, 8, norm, wording),
      ...(interests
        ? { leave_conversation_clause: smallTalkQuestion(me, other, 8, norm, 'clause') }
        : {}),
    };
    const talks: Record<string, string | undefined> = interests
      ? {
          both: talkPastime(me, other, 'both', dealing),
          'other only': talkPastime(me, other, 'other only', dealing),
        }
      : {};
    for (const [talk, pastime] of Object.entries(talks)) {
      if (pastime) {
        questions[`leave_conversation_talk_${talk.replace(' ', '_')}`] = pastimeTalkQuestion(
          me,
          other,
          pastime,
          8,
          norm,
          wording,
        );
      }
    }
    const memory = `${me.name} talked with ${other.name} about the weather.`;
    const importance = importanceQuestion(memory);
    return {
      condition: {
        cast,
        norm,
        wording,
        cap: conversationCap(cast),
        message_length: process.env.MESSAGE_LENGTH ?? 'upstream',
      },
      ...(interests
        ? {
            town: {
              dealing,
              seed: SEEDS[dealing],
              pastimes: PASTIMES,
              people: interestDescriptions(dealing).map((d) => ({
                name: d.name,
                identity: d.identity,
                goal: d.plan,
                pastimes: PASTIMES_OF[dealing][d.name],
              })),
              overlap: Object.fromEntries(
                NAMES.map((a) => [
                  a,
                  Object.fromEntries(
                    NAMES.filter((b) => b !== a).map((b) => [b, level(a, b, dealing)]),
                  ),
                ]),
              ),
              pair: { me: me.name, other: other.name, level: level(me.name, other.name, dealing) },
              talks,
              // The pastime each talk of IT-J is about, for every ordered pair: what
              // probeInterests asks, so the page shows it rather than rebuilding it.
              talkPastimes: Object.fromEntries(
                people.map((a) => [
                  a.name,
                  Object.fromEntries(
                    people
                      .filter((b) => b.name !== a.name)
                      .map((b) => [
                        b.name,
                        {
                          both: talkPastime(a, b, 'both', dealing) ?? null,
                          'other only': talkPastime(a, b, 'other only', dealing) ?? null,
                        },
                      ]),
                  ),
                ]),
              ),
            },
          }
        : {}),
      ...Object.fromEntries(
        Object.entries(questions).map(([decision, q]) => [
          decision,
          { jev: q, gemma: choicePrompt(q)[0].content },
        ]),
      ),
      importance: { jev: importance, gemma: upstreamImportancePrompt(memory) },
      conversation: conversationPrompts(me, other, Date.now()),
    };
  },
});

// The designed characters of data/variants.ts, one family per call (a family
// of Gemma's answers takes minutes, and an action has ten): for each
// character, the accept question from each counterpart, the invite question
// over the four and no one, and the leave question with each counterpart on
// small talk cut at `lengths` messages. Under gemma a probability is the share
// of `samples` answers, the off-list ones counted apart.
//
//   npx convex run probe:probeVariants '{"family": "sociability"}'
//   npx convex run probe:probeVariants '{"family": "sociability", "backend": "gemma", "samples": 10}'
//
// A call has about five minutes; a family of Gemma's answers can pass that, so
// `only` asks a subset of the family's characters, by name, and the halves
// are joined afterwards.
export const probeVariants = internalAction({
  args: {
    family: v.string(),
    only: v.optional(v.array(v.string())),
    backend: v.optional(v.union(v.literal('jev'), v.literal('gemma'))),
    samples: v.optional(v.number()),
    lengths: v.optional(v.array(v.number())),
  },
  handler: async (_ctx, args) => {
    const family = VARIANT_FAMILIES[args.family];
    if (!family) {
      throw new Error(`no family ${args.family}: ${Object.keys(VARIANT_FAMILIES).join(', ')}`);
    }
    const variants = args.only ? family.filter((d) => args.only!.includes(d.variant)) : family;
    const backend = args.backend ?? 'jev';
    const samples = args.samples ?? 10;
    const lengths = args.lengths ?? [2, 8, 20, 25, 30, 38];
    let model: string | undefined;
    let costUsd = 0;
    let calls = 0;
    const providers = new Set<string>();
    const offList: Array<{ variant: string; decision: string; raw: string }> = [];
    const ask = async (variant: string, decision: string, q: Question) => {
      if (backend === 'jev') {
        const response = await askJev(q.state, { q: choice(q.instructions, q.criteria) });
        model = response.model;
        return response.answers.q.probabilities as Record<string, number>;
      }
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
        model = answer.model;
        if (answer.provider) {
          providers.add(answer.provider);
        }
        const chosen = matchOption(answer.content, options);
        if (chosen) {
          counts[chosen] += 1;
        } else {
          counts[OFF_LIST] += 1;
          offList.push({ variant, decision, raw: answer.content });
        }
      }
      return Object.fromEntries(Object.entries(counts).map(([o, n]) => [o, n / samples]));
    };
    const rows: Array<{
      family: string;
      variant: string;
      decision: string;
      other?: string;
      messages?: number;
      probabilities: Record<string, number>;
    }> = [];
    for (const d of variants) {
      const me: Me = { name: d.name, identity: d.identity, goal: d.plan };
      for (const other of COUNTERPARTS) {
        const asked = [
          ask(
            d.variant,
            'accept_invite',
            acceptQuestion(me, { name: other.name, identity: other.identity, lastEnded: null }, 0),
          ).then((probabilities) =>
            rows.push({
              family: args.family,
              variant: d.variant,
              decision: 'accept_invite',
              other: other.name,
              probabilities,
            }),
          ),
          ...lengths.map((n) =>
            ask(d.variant, 'leave_conversation', smallTalkQuestion(me, other, n)).then(
              (probabilities) =>
                rows.push({
                  family: args.family,
                  variant: d.variant,
                  decision: 'leave_conversation',
                  other: other.name,
                  messages: n,
                  probabilities,
                }),
            ),
          ),
        ];
        await Promise.all(asked);
      }
      const probabilities = await ask(
        d.variant,
        'invite_target',
        inviteQuestion(
          me,
          COUNTERPARTS.map((p) => ({
            label: p.name,
            identity: p.identity,
            distance: 5,
            lastEnded: null,
          })),
          0,
        ),
      );
      rows.push({
        family: args.family,
        variant: d.variant,
        decision: 'invite_target',
        probabilities,
      });
    }
    return {
      family: args.family,
      backend,
      model,
      variants,
      ...(backend === 'gemma'
        ? { providers: [...providers], samples, calls, costUsd, offList }
        : {}),
      lengths,
      rows,
    };
  },
});

// One judge's answer to a question as probabilities over its options: Jev's,
// or the share of `samples` Gemma answers naming each option (the gemma
// backend's prompt, choicePrompt, at temperature 1), the off-list ones counted
// apart. `tally` is what the calls cost and where they went.
function judgeOf(backend: 'jev' | 'gemma', samples: number) {
  let model: string | undefined;
  let costUsd = 0;
  let calls = 0;
  const providers = new Set<string>();
  const offList: Array<{ name: string; decision: string; raw: string }> = [];
  const ask = async (name: string, decision: string, q: Question) => {
    if (backend === 'jev') {
      const response = await askJev(q.state, { q: choice(q.instructions, q.criteria) });
      model = response.model;
      return response.answers.q.probabilities as Record<string, number>;
    }
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
      model = answer.model;
      if (answer.provider) {
        providers.add(answer.provider);
      }
      const chosen = matchOption(answer.content, options);
      if (chosen) {
        counts[chosen] += 1;
      } else {
        counts[OFF_LIST] += 1;
        offList.push({ name, decision, raw: answer.content });
      }
    }
    return Object.fromEntries(Object.entries(counts).map(([o, n]) => [o, n / samples]));
  };
  const tally = () =>
    backend === 'jev'
      ? { model }
      : { model, providers: [...providers], samples, calls, costUsd, offList };
  return { ask, tally };
}

// The interest town's paired probe (003-IP, IT-D and IT-J), on the six of
// data/interests.ts in `dealing` (1 or 2, IT-I), everyone a stranger. The
// grid: for each person, the accept question from each of the five others;
// the invite question over the five at `distance` tiles and no one, asked
// with the five in the town's order and reversed, so that no level is always
// listed first or last; and the leave question with each of the five over the
// small talk above cut at `lengths` messages. The talks: the leave question
// with each of the five over the two talks about a pastime -- one both list,
// when the pair has one, and one only the other lists -- cut at `talkLengths`
// messages. Every leave question under both wordings with the norm of 20: the
// neutral one the town asks, and the one with the five-person town's clause.
// With the talk held fixed, a difference by level is the texts alone; the
// difference between the wordings is what the clause adds; the difference
// between the two talks is the judge connecting what is said to who `me` is.
// Every row carries the pair's level and its talk.
//
//   npx convex run probe:probeInterests '{"dealing": 2}'                     Jev, all six
//   npx convex run probe:probeInterests '{"backend": "gemma", "only": ["Avery"], "part": "talks"}'
//
// A call has about five minutes: under gemma, one person and one `part`
// (grid or talks) per call, and the pieces are joined afterwards.
export const probeInterests = internalAction({
  args: {
    only: v.optional(v.array(v.string())),
    backend: v.optional(v.union(v.literal('jev'), v.literal('gemma'))),
    dealing: v.optional(v.union(v.literal(1), v.literal(2))),
    part: v.optional(v.union(v.literal('grid'), v.literal('talks'), v.literal('all'))),
    samples: v.optional(v.number()),
    lengths: v.optional(v.array(v.number())),
    talkLengths: v.optional(v.array(v.number())),
    distance: v.optional(v.number()),
  },
  handler: async (_ctx, args) => {
    const backend = args.backend ?? 'jev';
    const dealing: Dealing = args.dealing ?? 1;
    const part = args.part ?? 'all';
    const grid = part !== 'talks';
    const talks = part !== 'grid';
    const samples = args.samples ?? 10;
    const lengths = args.lengths ?? [2, 10, 20, 30, 45, 60, 80];
    const talkLengths = args.talkLengths ?? [10, 20, 30];
    const distance = args.distance ?? 5;
    const { norm } = leaveNorm('interests');
    const wordings: LeaveWording[] = ['neutral', 'clause'];
    const people = peopleOf('interests', dealing);
    const judge = judgeOf(backend, samples);
    const rows: Array<{
      name: string;
      decision: string;
      other?: string;
      level?: number;
      wording?: LeaveWording;
      talk?: Talk;
      pastime?: string;
      messages?: number;
      order?: string;
      levels?: Record<string, number>;
      probabilities: Record<string, number>;
    }> = [];
    for (const me of people.filter((p) => !args.only || args.only.includes(p.name))) {
      const others = people.filter((p) => p.name !== me.name);
      for (const other of others) {
        const pair = {
          name: me.name,
          other: other.name,
          level: level(me.name, other.name, dealing),
        };
        const pastimes = (['both', 'other only'] as const)
          .map((talk) => ({ talk, pastime: talkPastime(me, other, talk, dealing) }))
          .filter((t): t is { talk: 'both' | 'other only'; pastime: string } => !!t.pastime);
        for (const wording of wordings) {
          await Promise.all([
            ...(grid && wording === wordings[0]
              ? [
                  judge
                    .ask(
                      me.name,
                      'accept_invite',
                      acceptQuestion(
                        me,
                        { name: other.name, identity: other.identity, lastEnded: null },
                        0,
                      ),
                    )
                    .then((probabilities) =>
                      rows.push({ ...pair, decision: 'accept_invite', probabilities }),
                    ),
                ]
              : []),
            ...(grid ? lengths : []).map((n) =>
              judge
                .ask(me.name, 'leave_conversation', smallTalkQuestion(me, other, n, norm, wording))
                .then((probabilities) =>
                  rows.push({
                    ...pair,
                    decision: 'leave_conversation',
                    wording,
                    talk: 'small',
                    messages: n,
                    probabilities,
                  }),
                ),
            ),
            ...(talks ? pastimes : []).flatMap(({ talk, pastime }) =>
              talkLengths.map((n) =>
                judge
                  .ask(
                    me.name,
                    'leave_conversation',
                    pastimeTalkQuestion(me, other, pastime, n, norm, wording),
                  )
                  .then((probabilities) =>
                    rows.push({
                      ...pair,
                      decision: 'leave_conversation',
                      wording,
                      talk,
                      pastime,
                      messages: n,
                      probabilities,
                    }),
                  ),
              ),
            ),
          ]);
        }
      }
      if (grid) {
        const levels = Object.fromEntries(
          others.map((p) => [p.name, level(me.name, p.name, dealing)]),
        );
        for (const order of ['as listed', 'reversed']) {
          const listed = order === 'reversed' ? [...others].reverse() : others;
          const probabilities = await judge.ask(
            me.name,
            'invite_target',
            inviteQuestion(
              me,
              listed.map((p) => ({ label: p.name, identity: p.identity, distance, lastEnded: null })),
              0,
            ),
          );
          rows.push({ name: me.name, decision: 'invite_target', order, levels, probabilities });
        }
      }
    }
    return {
      backend,
      ...judge.tally(),
      town: { cast: 'interests', dealing, seed: SEEDS[dealing], pastimes: PASTIMES_OF[dealing] },
      only: args.only ?? null,
      part,
      distance,
      norm,
      wordings,
      lengths: grid ? lengths : [],
      talkLengths: talks ? talkLengths : [],
      rows,
    };
  },
});
