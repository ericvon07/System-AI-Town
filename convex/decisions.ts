// The decision seam: the three choices upstream AI Town makes with a die or a
// fixed rule, behind one flag.
//
//   DECISION_BACKEND=dice  (default) upstream behaviour, unchanged: the same
//                          random draws in the same order. The seam records,
//                          it does not correct.
//   DECISION_BACKEND=jev   each decision is a typed Choice asked of Jev over a
//                          named-object state built from the persona texts the
//                          conversation prompts already use.
//   DECISION_BACKEND=gemma the same question, the same state, asked of Gemma
//                          through OpenRouter (util/openrouter.ts) as a prompt;
//                          the label it writes back is matched to the options,
//                          and an answer off the list falls back to the die.
//                          Sampled at temperature 1, the like-for-like of
//                          JEV_POLICY=sample.
//
//   JEV_POLICY=sample      (default) draw the act from Jev's probabilities, the
//                          like-for-like replacement of a die.
//   JEV_POLICY=argmax      take the label Jev returns.
//
// The three decisions:
//   activity       which of ACTIVITIES to do when idle   (dice: uniform 1/3)
//   accept_invite  accept another agent's invitation      (dice: 0.8, everyone)
//   invite_target  whom to invite while wandering         (dice: the first free
//                  player not in cooldown -- upstream sorts candidates by a
//                  distance that is always 0, see findConversationCandidate)
//
// One deliberate change of set, the judged backends only: invite_target
// offers 'no one'. Upstream invites whenever a candidate exists, so a persona
// that wants to avoid people has no way to say so in that decision.
//
// Two more decisions, which upstream makes with a rule or with the LLM:
//   leave_conversation  before each reply, say more or leave (dice: a rule,
//                       leave after 8 messages or 10 minutes; the judged
//                       backends ask, told that conversations here usually
//                       wrap up after about 25 messages, under a cap of 40,
//                       agent.ts; in the interest town, TOWN_CAST=interests,
//                       about 20 in the neutral wording, under a cap of 100)
//   importance          the 0-9 poignancy of a new memory (dice: upstream's
//                       prompt on the chat LLM; gemma: the same prompt on
//                       Gemma; jev: a typed Score over the same 0-9 scale)
//
// Every decision of both backends lands in the `decisions` table with its
// options, the act taken and the probabilities behind it (by construction for
// the die), so either policy can be read off the same trace.

import { v } from 'convex/values';
import { ActionCtx, internalMutation, internalQuery, query } from './_generated/server';
import { internal } from './_generated/api';
import { Id } from './_generated/dataModel';
import {
  ACTIVITIES,
  INTEREST_MAX_CONVERSATION_MESSAGES,
  INTEREST_USUAL_CONVERSATION_MESSAGES,
  INVITE_ACCEPT_PROBABILITY,
  JUDGED_MAX_CONVERSATION_MESSAGES,
  PLAYER_CONVERSATION_COOLDOWN,
  USUAL_CONVERSATION_MESSAGES,
} from './constants';
import { askJev, choice, score, ScoreAnswer } from './util/jev';
import { askOpenRouter, gemmaModel } from './util/openrouter';
import { chatCompletion, getLLMConfig } from './util/llm';
import { distance } from './util/geometry';
import { SerializedPlayer } from './aiTown/player';
import { decisionFields } from './decisionFields';
import { TownCast, shuffled, townCast } from '../data/interests';

export type DecisionBackend = 'dice' | 'jev' | 'gemma';

export function decisionBackend(): DecisionBackend {
  const backend = process.env.DECISION_BACKEND;
  return backend === 'jev' || backend === 'gemma' ? backend : 'dice';
}

function jevPolicy(): 'sample' | 'argmax' {
  return process.env.JEV_POLICY === 'argmax' ? 'argmax' : 'sample';
}

// The leave question's norm and wording, by town: 25 with the clause that
// people stay longer only when the talk is unusually interesting to them (D4);
// in the interest town, 20 and the norm alone (D8).
export type LeaveWording = 'clause' | 'neutral';

export function leaveNorm(cast: TownCast = townCast()): { norm: number; wording: LeaveWording } {
  return cast === 'interests'
    ? { norm: INTEREST_USUAL_CONVERSATION_MESSAGES, wording: 'neutral' }
    : { norm: USUAL_CONVERSATION_MESSAGES, wording: 'clause' };
}

// The messages after which the code ends a conversation under the judged
// backends (agent.ts), by town.
export function conversationCap(cast: TownCast = townCast()): number {
  return cast === 'interests'
    ? INTEREST_MAX_CONVERSATION_MESSAGES
    : JUDGED_MAX_CONVERSATION_MESSAGES;
}

// The town an arm runs in, which run-arm.sh writes into run.json: the cast,
// and the norm and the cap the judged backends use (a dice arm leaves by
// upstream's rule whatever they say).
export const town = internalQuery({
  args: {},
  handler: async () => ({ cast: townCast(), norm: leaveNorm().norm, cap: conversationCap() }),
});

type DecisionName = 'activity' | 'accept_invite' | 'invite_target' | 'leave_conversation';
type Probability = { option: string; p: number };

export const record = internalMutation({
  args: decisionFields,
  handler: async (ctx, args) => {
    await ctx.db.insert('decisions', args);
  },
});

// --- what the typed backend reads ---------------------------------------

export const decisionContext = internalQuery({
  args: {
    worldId: v.id('worlds'),
    playerId: v.string(),
    otherIds: v.array(v.string()),
    now: v.number(),
  },
  handler: async (ctx, args) => {
    const world = await ctx.db.get(args.worldId);
    if (!world) {
      throw new Error(`World ${args.worldId} not found`);
    }
    const describe = async (playerId: string) =>
      await ctx.db
        .query('playerDescriptions')
        .withIndex('worldId', (q) => q.eq('worldId', args.worldId).eq('playerId', playerId))
        .first();
    const me = await describe(args.playerId);
    if (!me) {
      throw new Error(`Player description for ${args.playerId} not found`);
    }
    const agent = world.agents.find((a) => a.playerId === args.playerId);
    const agentDescription =
      agent &&
      (await ctx.db
        .query('agentDescriptions')
        .withIndex('worldId', (q) => q.eq('worldId', args.worldId).eq('agentId', agent.id))
        .first());
    const others = [];
    for (const otherId of args.otherIds) {
      const description = await describe(otherId);
      const lastTogether = await ctx.db
        .query('participatedTogether')
        .withIndex('edge', (q) =>
          q.eq('worldId', args.worldId).eq('player1', args.playerId).eq('player2', otherId),
        )
        .order('desc')
        .first();
      others.push({
        playerId: otherId,
        name: description?.name ?? otherId,
        identity: description?.description ?? '',
        human: !!world.players.find((p) => p.id === otherId)?.human,
        lastEnded: lastTogether?.ended ?? null,
        // The same veto findConversationCandidate applies.
        inCooldown: !!lastTogether && args.now < lastTogether.ended + PLAYER_CONVERSATION_COOLDOWN,
      });
    }
    return {
      me: {
        name: me.name,
        identity: agentDescription?.identity ?? me.description,
        goal: agentDescription?.plan ?? null,
      },
      others,
    };
  },
});

// The conversation so far, for leave_conversation: when it started and who
// said what, in order.
export const conversationSoFar = internalQuery({
  args: { worldId: v.id('worlds'), conversationId: v.string() },
  handler: async (ctx, args) => {
    const world = await ctx.db.get(args.worldId);
    if (!world) {
      throw new Error(`World ${args.worldId} not found`);
    }
    const conversation = world.conversations.find((c) => c.id === args.conversationId);
    const messages = await ctx.db
      .query('messages')
      .withIndex('conversationId', (q) =>
        q.eq('worldId', args.worldId).eq('conversationId', args.conversationId as any),
      )
      .collect();
    const names = new Map<string, string>();
    for (const message of messages) {
      if (!names.has(message.author)) {
        const description = await ctx.db
          .query('playerDescriptions')
          .withIndex('worldId', (q) =>
            q.eq('worldId', args.worldId).eq('playerId', message.author),
          )
          .first();
        names.set(message.author, description?.name ?? message.author);
      }
    }
    return {
      created: conversation?.created ?? messages[0]?._creationTime ?? Date.now(),
      transcript: messages.map((m) => ({ speaker: names.get(m.author)!, text: m.text })),
    };
  },
});

function minutesAgo(now: number, ts: number | null) {
  return ts === null ? null : Math.round((now - ts) / 60_000);
}

// --- the shared path -----------------------------------------------------

function sample(probabilities: Probability[]) {
  const total = probabilities.reduce((acc, { p }) => acc + p, 0);
  let r = Math.random() * total;
  for (const { option, p } of probabilities) {
    r -= p;
    if (r < 0) {
      return option;
    }
  }
  return probabilities[probabilities.length - 1].option;
}

type Base = {
  worldId: Id<'worlds'>;
  decision: DecisionName;
  playerId: string;
  name: string;
  otherName?: string;
  options: string[];
  conversationId?: string;
  messagesSoFar?: number;
};

type Question = { state: unknown; instructions: string; criteria: Record<string, string> };

// The seam's question as a prompt, for the gemma backend: the same state, the
// same instructions, the same options with the same descriptions. Only the
// shape of the answer changes, from a typed Choice to a label written back.
export function choicePrompt(q: Question) {
  const options = Object.entries(q.criteria)
    .map(([label, description]) => `- ${label}: ${description}`)
    .join('\n');
  return [
    {
      role: 'user' as const,
      content:
        `Here is the current state, as JSON:\n\n${JSON.stringify(q.state, null, 2)}\n\n` +
        `${q.instructions}\n\nOptions:\n${options}\n\n` +
        'Answer with exactly one option name from the list above, written exactly as it ' +
        'appears there, and nothing else.',
    },
  ];
}

function normalize(text: string) {
  return text
    .toLowerCase()
    .replace(/[`*"'.!:]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

// An answer is on the list when, stripped of quotes, emphasis and a full
// stop, it is one of the options; or when it names exactly one option and no
// other (a label wrapped in a sentence). The readout tells the two apart by
// comparing `raw` with `chosen`. Anything else is off the list.
export function matchOption(raw: string, options: string[]): string | null {
  const answer = normalize(raw);
  const exact = options.find((o) => normalize(o) === answer);
  if (exact) {
    return exact;
  }
  const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const named = options.filter((o) => new RegExp(`\\b${escape(normalize(o))}\\b`).test(answer));
  return named.length === 1 ? named[0] : null;
}

async function decide(
  ctx: ActionCtx,
  base: Base,
  dice: () => { chosen: string; probabilities: Probability[] },
  question: () => Promise<Question>,
): Promise<string> {
  const backend = decisionBackend();
  if (backend === 'dice') {
    const { chosen, probabilities } = dice();
    await ctx.runMutation(internal.decisions.record, {
      ...base,
      backend: 'dice',
      chosen,
      probabilities,
    });
    return chosen;
  }
  const policy = backend === 'gemma' ? 'sample' : jevPolicy();
  const q = await question();
  const started = Date.now();
  const fallBack = async (
    e: unknown,
    extra: { raw?: string; model?: string; provider?: string; costUsd?: number } = {},
  ) => {
    const { chosen, probabilities } = dice();
    console.error(`${backend} ${base.decision} for ${base.name} failed, the die decided:`, e);
    await ctx.runMutation(internal.decisions.record, {
      ...base,
      backend,
      policy,
      chosen,
      probabilities,
      state: q.state,
      latencyMs: Date.now() - started,
      fallback: true,
      error: e instanceof Error ? e.message : String(e),
      ...extra,
    });
    return chosen;
  };
  if (backend === 'gemma') {
    let answer;
    try {
      answer = await askOpenRouter(choicePrompt(q), { maxTokens: 24, temperature: 1 });
    } catch (e) {
      return await fallBack(e, { model: gemmaModel() });
    }
    const meta = {
      raw: answer.content,
      model: answer.model,
      provider: answer.provider ?? undefined,
      costUsd: answer.costUsd ?? undefined,
    };
    const chosen = matchOption(answer.content, base.options);
    if (!chosen) {
      return await fallBack(new Error(`off the list: ${JSON.stringify(answer.content)}`), meta);
    }
    await ctx.runMutation(internal.decisions.record, {
      ...base,
      backend: 'gemma',
      policy,
      chosen,
      // One sample, so the only probability there is: the act itself.
      probabilities: base.options.map((option) => ({ option, p: option === chosen ? 1 : 0 })),
      state: q.state,
      latencyMs: answer.latencyMs,
      ...meta,
    });
    return chosen;
  }
  try {
    const { answers, model } = await askJev(q.state, {
      [base.decision]: choice(q.instructions, q.criteria),
    });
    const answer = answers[base.decision];
    if (!answer || !base.options.includes(answer.choice)) {
      throw new Error(`no usable answer: ${JSON.stringify(answer)}`);
    }
    const probabilities = base.options.map((option) => ({
      option,
      p: answer.probabilities?.[option] ?? 0,
    }));
    const chosen = policy === 'argmax' ? answer.choice : sample(probabilities);
    await ctx.runMutation(internal.decisions.record, {
      ...base,
      backend: 'jev',
      policy,
      chosen,
      probabilities,
      jevChoice: answer.choice,
      state: q.state,
      latencyMs: Date.now() - started,
      model,
    });
    return chosen;
  } catch (e) {
    return await fallBack(e);
  }
}

// --- the questions --------------------------------------------------------
// What Jev is asked, kept apart from where it is asked so the probe
// (probe.ts) puts the very same questions to constructed states.

export type Me = { name: string; identity: string; goal: string | null };

export const NO_ONE = 'no one';

export function activityQuestion(me: Me, options: string[]) {
  return {
    state: { me },
    instructions:
      '`me` is free and alone for the next while, not talking to anyone. Which of these ' +
      'does `me.name` choose to spend that time on, given who they are (`me.identity`) ' +
      'and what they want (`me.goal`)?',
    criteria: Object.fromEntries(options.map((o) => [o, `spend the time ${o}`])),
  };
}

export function acceptQuestion(
  me: Me,
  inviter: { name: string; identity: string; lastEnded: number | null },
  now: number,
) {
  return {
    state: {
      me,
      inviter: {
        name: inviter.name,
        identity: inviter.identity,
        last_talked_minutes_ago: minutesAgo(now, inviter.lastEnded),
      },
    },
    instructions:
      '`inviter.name` has just invited `me.name` to stop and talk. Does `me.name` accept, ' +
      'given who they are (`me.identity`), what they want (`me.goal`) and who ' +
      '`inviter` is (`inviter.identity`)?',
    criteria: {
      accept: 'agree, and walk over to talk',
      decline: 'turn the invitation down and carry on alone',
    },
  };
}

export function inviteQuestion(
  me: Me,
  people: Array<{ label: string; identity: string; distance: number; lastEnded: number | null }>,
  now: number,
) {
  const options = people.map((p) => p.label).concat([NO_ONE]);
  return {
    state: {
      me,
      free_people: people.map((p) => ({
        name: p.label,
        identity: p.identity,
        distance_in_tiles: p.distance,
        last_talked_minutes_ago: minutesAgo(now, p.lastEnded),
      })),
    },
    instructions:
      '`me` is walking around town and could start a conversation now. Whom does ' +
      '`me.name` walk up to and start talking with: one of `free_people`, or no one? ' +
      'Weigh who `me` is (`me.identity`), what they want (`me.goal`), who each person ' +
      'is, how far away they are and how recently they last talked.',
    criteria: Object.fromEntries(
      options.map((o) => [
        o,
        o === NO_ONE ? 'start no conversation and keep walking' : `start a conversation with ${o}`,
      ]),
    ),
  };
}

export function leaveQuestion(
  me: Me,
  other: { name: string; identity: string },
  minutesSoFar: number,
  transcript: Array<{ speaker: string; text: string }>,
  usualMessages: number = USUAL_CONVERSATION_MESSAGES,
  wording: LeaveWording = 'clause',
) {
  return {
    state: {
      me,
      other: { name: other.name, identity: other.identity },
      conversation: {
        messages_so_far: transcript.length,
        minutes_so_far: minutesSoFar,
        usual_length_messages: usualMessages,
        transcript,
      },
    },
    instructions:
      "`me.name` is talking with `other.name`, and it is `me.name`'s turn to speak. Does " +
      '`me.name` keep the conversation going, or say goodbye and leave it now? In this ' +
      'town a conversation usually wraps up after about `conversation.usual_length_messages` ' +
      (wording === 'clause'
        ? 'messages; people stay longer only when the talk is unusually interesting to them. '
        : 'messages. ') +
      'Weigh who `me` is (`me.identity`), what they want (`me.goal`), who `other` is ' +
      '(`other.identity`), what has been said so far (`conversation.transcript`) and how ' +
      'long they have been talking (`conversation.messages_so_far` messages over ' +
      '`conversation.minutes_so_far` minutes).',
    criteria: {
      continue: 'keep talking: say something more',
      leave: 'say goodbye and leave the conversation',
    },
  };
}

// Upstream's importance prompt, verbatim (agent/memory.ts before the seam):
// the dice and gemma backends send it to an LLM and parse a digit back.
export function upstreamImportancePrompt(description: string) {
  return `On the scale of 0 to 9, where 0 is purely mundane (e.g., brushing teeth, making bed) and 9 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.
      Memory: ${description}
      Answer on a scale of 0 to 9. Respond with number only, e.g. "5"`;
}

// The same scale as a typed Score: ten levels, 0 to 9. The two ends quote
// upstream's anchors; the eight between are the System One Lab's (its
// jev_questions.py, deviation D2, the paper's event poignancy), whose ends
// are the same two anchors.
export const IMPORTANCE_LEVELS = [
  'Purely mundane: brushing teeth, making the bed.',
  'Routine upkeep of the day, noticed and immediately forgotten.',
  'Ordinary, with some small personal texture to it.',
  'Mildly notable: a small departure from how the day usually goes.',
  'Noticeable: the sort of thing worth mentioning to someone later.',
  'Personally meaningful: it touches something this person cares about.',
  'Clearly significant: likely to still be on their mind in a few days.',
  'Emotionally charged, or consequential for their plans.',
  'Deeply affecting: it changes how they feel about someone or something.',
  'Extremely poignant: a break up, a college acceptance.',
];

export function importanceQuestion(description: string) {
  return {
    state: { memory: description },
    instructions: 'Rate the likely poignancy of `memory`, a piece of memory.',
    criteria: IMPORTANCE_LEVELS,
  };
}

// Upstream's parse: a number, else the first digit, else 5.
export function parseImportance(raw: string): { score: number; parsed: boolean } {
  let score = parseFloat(raw);
  if (isNaN(score)) {
    score = +(raw.match(/\d+/)?.[0] ?? NaN);
  }
  return isNaN(score) ? { score: 5, parsed: false } : { score, parsed: true };
}

// --- the three decisions -------------------------------------------------

export async function decideActivity(
  ctx: ActionCtx,
  worldId: Id<'worlds'>,
  playerId: string,
  now: number,
) {
  const { me } = await ctx.runQuery(internal.decisions.decisionContext, {
    worldId,
    playerId,
    otherIds: [],
    now,
  });
  const options = ACTIVITIES.map((a) => a.description);
  const chosen = await decide(
    ctx,
    { worldId, decision: 'activity', playerId, name: me.name, options },
    () => ({
      // Upstream's own draw, unchanged.
      chosen: ACTIVITIES[Math.floor(Math.random() * ACTIVITIES.length)].description,
      probabilities: options.map((option) => ({ option, p: 1 / options.length })),
    }),
    async () => activityQuestion(me, options),
  );
  return ACTIVITIES.find((a) => a.description === chosen)!;
}

export async function decideAcceptInvite(
  ctx: ActionCtx,
  worldId: Id<'worlds'>,
  playerId: string,
  inviterId: string,
  now: number,
) {
  const { me, others } = await ctx.runQuery(internal.decisions.decisionContext, {
    worldId,
    playerId,
    otherIds: [inviterId],
    now,
  });
  const inviter = others[0];
  const chosen = await decide(
    ctx,
    {
      worldId,
      decision: 'accept_invite',
      playerId,
      name: me.name,
      otherName: inviter.name,
      options: ['accept', 'decline'],
    },
    () => ({
      chosen: Math.random() < INVITE_ACCEPT_PROBABILITY ? 'accept' : 'decline',
      probabilities: [
        { option: 'accept', p: INVITE_ACCEPT_PROBABILITY },
        { option: 'decline', p: 1 - INVITE_ACCEPT_PROBABILITY },
      ],
    }),
    async () => acceptQuestion(me, inviter, now),
  );
  return chosen === 'accept';
}

// Returns the player to invite, or `declined: true` when the typed backend
// chose no one -- which counts as an invite attempt, so the agent does not
// ask again on the very next tick.
export async function decideInviteTarget(
  ctx: ActionCtx,
  worldId: Id<'worlds'>,
  now: number,
  player: SerializedPlayer,
  otherFreePlayers: SerializedPlayer[],
): Promise<{ invitee?: string; declined?: boolean }> {
  const { me, others } = await ctx.runQuery(internal.decisions.decisionContext, {
    worldId,
    playerId: player.id,
    otherIds: otherFreePlayers.map((p) => p.id),
    now,
  });
  const candidates = others.filter((o) => !o.inCooldown);
  if (candidates.length === 0) {
    return {};
  }
  // Labels are names; a repeated name carries its id so no two labels collide.
  const seen = new Set<string>();
  const labelled = candidates.map((c) => {
    const label = seen.has(c.name) ? `${c.name} (${c.playerId})` : c.name;
    seen.add(label);
    return { ...c, label };
  });
  const byLabel = new Map(labelled.map((c) => [c.label, c]));
  const backend = decisionBackend();
  // In the interest town (D8) the judged backends are asked about the free
  // people in a random order: Jev leans toward the last name listed (the lab's
  // probeInterests), and the town's own order, by creation, would mix that
  // pull with how much a pair has in common. The state records the order.
  const asked =
    backend !== 'dice' && townCast() === 'interests' ? shuffled(labelled, Math.random) : labelled;
  const options = asked.map((c) => c.label).concat(backend === 'dice' ? [] : [NO_ONE]);

  let diceInvitee: string | undefined;
  if (backend === 'dice') {
    // Upstream's own rule, unchanged.
    diceInvitee = await ctx.runQuery(internal.aiTown.agent.findConversationCandidate, {
      now,
      worldId,
      player,
      otherFreePlayers,
    });
  }
  const chosen = await decide(
    ctx,
    { worldId, decision: 'invite_target', playerId: player.id, name: me.name, options },
    () => {
      // Under the judged backends this is the fallback: upstream's pick is the
      // first candidate, so take it without a second query.
      const pick = diceInvitee ?? labelled[0].playerId;
      const label = labelled.find((c) => c.playerId === pick)?.label ?? labelled[0].label;
      return {
        chosen: label,
        probabilities: options.map((option) => ({ option, p: option === label ? 1 : 0 })),
      };
    },
    async () =>
      inviteQuestion(
        me,
        asked.map((c) => ({
          label: c.label,
          identity: c.identity,
          distance: Math.round(
            distance(player.position, otherFreePlayers.find((p) => p.id === c.playerId)!.position),
          ),
          lastEnded: c.lastEnded,
        })),
        now,
      ),
  );
  if (chosen === NO_ONE) {
    return { declined: true };
  }
  return { invitee: byLabel.get(chosen)!.playerId };
}

// Under the judged backends, before each reply: say more, or leave. Never
// asked under dice, where the rule in agent.ts decides; the die's stand-in
// for a failed call is that rule's answer below its cap, 'continue'.
export async function decideLeave(
  ctx: ActionCtx,
  worldId: Id<'worlds'>,
  playerId: string,
  otherPlayerId: string,
  conversationId: string,
  now: number,
) {
  const { me, others } = await ctx.runQuery(internal.decisions.decisionContext, {
    worldId,
    playerId,
    otherIds: [otherPlayerId],
    now,
  });
  const other = others[0];
  const { created, transcript } = await ctx.runQuery(internal.decisions.conversationSoFar, {
    worldId,
    conversationId,
  });
  const chosen = await decide(
    ctx,
    {
      worldId,
      decision: 'leave_conversation',
      playerId,
      name: me.name,
      otherName: other.name,
      options: ['continue', 'leave'],
      conversationId,
      messagesSoFar: transcript.length,
    },
    () => ({
      chosen: 'continue',
      probabilities: [
        { option: 'continue', p: 1 },
        { option: 'leave', p: 0 },
      ],
    }),
    async () => {
      const { norm, wording } = leaveNorm();
      return leaveQuestion(
        me,
        other,
        Math.round((now - created) / 6_000) / 10,
        transcript,
        norm,
        wording,
      );
    },
  );
  return chosen === 'leave';
}

// The importance of a new memory, by the arm's judge: upstream's prompt on
// the chat LLM (dice), the same prompt on Gemma (gemma), a typed Score (jev).
// A failed Jev call falls back to upstream's path; an unparseable answer gets
// upstream's 5. Every score is a row.
export async function scoreImportance(
  ctx: ActionCtx,
  worldId: Id<'worlds'>,
  playerId: string,
  name: string,
  description: string,
  memoryKind: 'conversation' | 'reflection',
): Promise<number> {
  const backend = decisionBackend();
  const base = {
    worldId,
    decision: 'importance' as const,
    playerId,
    name,
    options: [],
    probabilities: [],
    state: { memory: description },
    memoryKind,
    backend,
  };
  const started = Date.now();
  const llmPath = async (error?: string) => {
    const { content } = await chatCompletion({
      messages: [{ role: 'user', content: upstreamImportancePrompt(description) }],
      temperature: 0.0,
      max_tokens: 1,
    });
    const { score, parsed } = parseImportance(content);
    if (!parsed) {
      console.debug('Could not parse memory importance from: ', content);
    }
    await ctx.runMutation(internal.decisions.record, {
      ...base,
      chosen: String(score),
      score,
      raw: content,
      model: process.env.CHAT_BACKEND === 'openrouter' ? gemmaModel() : getLLMConfig().chatModel,
      latencyMs: Date.now() - started,
      ...(parsed && !error ? {} : { fallback: true }),
      ...(error ? { error } : parsed ? {} : { error: 'unparseable, scored 5' }),
    });
    return score;
  };
  if (backend === 'dice') {
    return await llmPath();
  }
  if (backend === 'gemma') {
    try {
      const answer = await askOpenRouter(
        [{ role: 'user', content: upstreamImportancePrompt(description) }],
        { maxTokens: 1, temperature: 0 },
      );
      const { score, parsed } = parseImportance(answer.content);
      await ctx.runMutation(internal.decisions.record, {
        ...base,
        chosen: String(score),
        score,
        raw: answer.content,
        model: answer.model,
        provider: answer.provider ?? undefined,
        costUsd: answer.costUsd ?? undefined,
        latencyMs: answer.latencyMs,
        ...(parsed ? {} : { fallback: true, error: 'unparseable, scored 5' }),
      });
      return score;
    } catch (e) {
      return await llmPath(`gemma failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  try {
    const q = importanceQuestion(description);
    const { answers, model } = await askJev<ScoreAnswer>(q.state, {
      importance: score(q.instructions, q.criteria),
    });
    const value = answers.importance?.score;
    if (typeof value !== 'number' || isNaN(value)) {
      throw new Error(`no usable answer: ${JSON.stringify(answers.importance)}`);
    }
    await ctx.runMutation(internal.decisions.record, {
      ...base,
      chosen: String(Math.round(value * 100) / 100),
      score: value,
      model,
      latencyMs: Date.now() - started,
    });
    return value;
  } catch (e) {
    return await llmPath(`jev failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

// --- the readout ---------------------------------------------------------

// Per backend, decision and persona: how many decisions, how often each
// option was taken, and the mean probability behind each option. Rates only:
// two backends stop visiting the same states at the first decision that
// differs, so nothing here pairs one decision with another.
export const summary = query({
  args: { worldId: v.optional(v.id('worlds')) },
  handler: async (ctx, args) => {
    const rows = await ctx.db.query('decisions').collect();
    const groups = new Map<
      string,
      {
        backend: string;
        decision: string;
        name: string;
        n: number;
        fallbacks: number;
        taken: Map<string, number>;
        pSum: Map<string, number>;
        scoreSum: number;
      }
    >();
    for (const row of rows) {
      if (args.worldId && row.worldId !== args.worldId) {
        continue;
      }
      const key = `${row.backend}|${row.decision}|${row.name}`;
      let g = groups.get(key);
      if (!g) {
        g = {
          backend: row.backend,
          decision: row.decision,
          name: row.name,
          n: 0,
          fallbacks: 0,
          taken: new Map(),
          pSum: new Map(),
          scoreSum: 0,
        };
        groups.set(key, g);
      }
      g.n += 1;
      if (row.fallback) {
        g.fallbacks += 1;
      }
      // importance has no options: its reading is the mean score.
      if (row.decision === 'importance') {
        g.scoreSum += row.score ?? 0;
        continue;
      }
      // invite_target labels are names, which differ per state; group them
      // as 'someone' vs 'no one' so the rate reads across states.
      const fold = (o: string) =>
        row.decision === 'invite_target' && o !== NO_ONE ? 'someone' : o;
      g.taken.set(fold(row.chosen), (g.taken.get(fold(row.chosen)) ?? 0) + 1);
      const folded = new Map<string, number>();
      for (const { option, p } of row.probabilities) {
        folded.set(fold(option), (folded.get(fold(option)) ?? 0) + p);
      }
      for (const [option, p] of folded) {
        g.pSum.set(option, (g.pSum.get(option) ?? 0) + p);
      }
    }
    return [...groups.values()]
      .sort((a, b) =>
        `${a.decision}|${a.backend}|${a.name}`.localeCompare(`${b.decision}|${b.backend}|${b.name}`),
      )
      .map((g) => ({
        backend: g.backend,
        decision: g.decision,
        name: g.name,
        n: g.n,
        fallbacks: g.fallbacks,
        ...(g.decision === 'importance'
          ? { meanScore: Math.round((g.scoreSum / g.n) * 100) / 100 }
          : {}),
        options: [...new Set([...g.taken.keys(), ...g.pSum.keys()])].sort().map((option) => ({
          option,
          taken: g.taken.get(option) ?? 0,
          meanP: Math.round(((g.pSum.get(option) ?? 0) / g.n) * 1000) / 1000,
        })),
      }));
  },
});
