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
import { internalAction } from './_generated/server';
import { Descriptions } from '../data/characters';
import { ACTIVITIES } from './constants';
import { askJev, choice } from './util/jev';
import { Me, acceptQuestion, activityQuestion, inviteQuestion } from './decisions';

type Question = { state: unknown; instructions: string; criteria: Record<string, string> };

export const probeJev = internalAction({
  args: { distance: v.optional(v.number()) },
  handler: async (_ctx, args) => {
    const distance = args.distance ?? 5;
    const people: Me[] = Descriptions.map((d) => ({
      name: d.name,
      identity: d.identity,
      goal: d.plan,
    }));
    let model: string | undefined;
    const ask = async (q: Question) => {
      const response = await askJev(q.state, { q: choice(q.instructions, q.criteria) });
      model = response.model;
      return response.answers.q.probabilities;
    };
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
    return { model, distance, activity, accept, invite };
  },
});
