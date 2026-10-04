// The decision seam: the three choices upstream AI Town makes with a die or a
// fixed rule, behind one flag.
//
//   DECISION_BACKEND=dice  (default) upstream behaviour, unchanged: the same
//                          random draws in the same order. The seam records,
//                          it does not correct.
//   DECISION_BACKEND=jev   each decision is a typed Choice asked of Jev over a
//                          named-object state built from the persona texts the
//                          conversation prompts already use.
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
// One deliberate change of set, the jev backend only: invite_target offers
// 'no one'. Upstream invites whenever a candidate exists, so a persona that
// wants to avoid people has no way to say so in that decision.
//
// Every decision of both backends lands in the `decisions` table with its
// options, the act taken and the probabilities behind it (by construction for
// the die), so either policy can be read off the same trace.

import { v } from 'convex/values';
import { ActionCtx, internalMutation, internalQuery, query } from './_generated/server';
import { internal } from './_generated/api';
import { Id } from './_generated/dataModel';
import { ACTIVITIES, INVITE_ACCEPT_PROBABILITY, PLAYER_CONVERSATION_COOLDOWN } from './constants';
import { askJev, choice } from './util/jev';
import { distance } from './util/geometry';
import { SerializedPlayer } from './aiTown/player';
import { decisionFields } from './schema';

export type DecisionBackend = 'dice' | 'jev';

export function decisionBackend(): DecisionBackend {
  return process.env.DECISION_BACKEND === 'jev' ? 'jev' : 'dice';
}

function jevPolicy(): 'sample' | 'argmax' {
  return process.env.JEV_POLICY === 'argmax' ? 'argmax' : 'sample';
}

type DecisionName = 'activity' | 'accept_invite' | 'invite_target';
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
};

async function decide(
  ctx: ActionCtx,
  base: Base,
  dice: () => { chosen: string; probabilities: Probability[] },
  jev: () => Promise<{ state: unknown; instructions: string; criteria: Record<string, string> }>,
): Promise<string> {
  if (decisionBackend() === 'dice') {
    const { chosen, probabilities } = dice();
    await ctx.runMutation(internal.decisions.record, {
      ...base,
      backend: 'dice',
      chosen,
      probabilities,
    });
    return chosen;
  }
  const policy = jevPolicy();
  const { state, instructions, criteria } = await jev();
  const started = Date.now();
  try {
    const { answers } = await askJev(state, { [base.decision]: choice(instructions, criteria) });
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
      state,
      latencyMs: Date.now() - started,
    });
    return chosen;
  } catch (e) {
    const { chosen, probabilities } = dice();
    console.error(`jev ${base.decision} for ${base.name} failed, the die decided:`, e);
    await ctx.runMutation(internal.decisions.record, {
      ...base,
      backend: 'jev',
      policy,
      chosen,
      probabilities,
      state,
      latencyMs: Date.now() - started,
      fallback: true,
      error: e instanceof Error ? e.message : String(e),
    });
    return chosen;
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
  const options = labelled.map((c) => c.label).concat(backend === 'jev' ? [NO_ONE] : []);

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
      // Under the jev backend this is the fallback: upstream's pick is the
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
        labelled.map((c) => ({
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
        };
        groups.set(key, g);
      }
      g.n += 1;
      if (row.fallback) {
        g.fallbacks += 1;
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
        options: [...new Set([...g.taken.keys(), ...g.pSum.keys()])].sort().map((option) => ({
          option,
          taken: g.taken.get(option) ?? 0,
          meanP: Math.round(((g.pSum.get(option) ?? 0) / g.n) * 1000) / 1000,
        })),
      }));
  },
});
