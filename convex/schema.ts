import { defineSchema, defineTable } from 'convex/server';
import { v } from 'convex/values';
import { agentTables } from './agent/schema';
import { aiTownTables } from './aiTown/schema';
import { conversationId, playerId } from './aiTown/ids';
import { engineTables } from './engine/schema';

// One row per decision of the seam in decisions.ts, under either backend.
export const decisionFields = {
  worldId: v.id('worlds'),
  backend: v.union(v.literal('dice'), v.literal('jev')),
  policy: v.optional(v.string()),
  decision: v.union(v.literal('activity'), v.literal('accept_invite'), v.literal('invite_target')),
  playerId: v.string(),
  name: v.string(),
  // The other party, for the two social decisions.
  otherName: v.optional(v.string()),
  options: v.array(v.string()),
  chosen: v.string(),
  probabilities: v.array(v.object({ option: v.string(), p: v.number() })),
  // The label Jev returned, before the policy turned it into an act.
  jevChoice: v.optional(v.string()),
  state: v.optional(v.any()),
  latencyMs: v.optional(v.number()),
  // Jev failed after its retries and the die decided instead.
  fallback: v.optional(v.boolean()),
  error: v.optional(v.string()),
};

export default defineSchema({
  music: defineTable({
    storageId: v.string(),
    type: v.union(v.literal('background'), v.literal('player')),
  }),

  messages: defineTable({
    conversationId,
    messageUuid: v.string(),
    author: playerId,
    text: v.string(),
    worldId: v.optional(v.id('worlds')),
  })
    .index('conversationId', ['worldId', 'conversationId'])
    .index('messageUuid', ['conversationId', 'messageUuid']),

  decisions: defineTable(decisionFields).index('worldId', ['worldId']),

  ...agentTables,
  ...aiTownTables,
  ...engineTables,
});
