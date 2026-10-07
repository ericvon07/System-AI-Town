// The decisions table's fields, apart from schema.ts so decisions.ts can
// import them without importing the whole schema: agent/memory.ts imports
// decisions.ts, and the schema imports the world, which imports the memory.

import { v } from 'convex/values';

// One row per decision of the seam in decisions.ts, under either backend.
export const decisionFields = {
  worldId: v.id('worlds'),
  backend: v.union(v.literal('dice'), v.literal('jev'), v.literal('gemma')),
  policy: v.optional(v.string()),
  decision: v.union(
    v.literal('activity'),
    v.literal('accept_invite'),
    v.literal('invite_target'),
    v.literal('leave_conversation'),
    v.literal('importance'),
  ),
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
  // Jev or Gemma failed after its retries, or Gemma answered off the list,
  // and the die decided instead.
  fallback: v.optional(v.boolean()),
  error: v.optional(v.string()),
  // Gemma's answer as it came, before it was matched to an option.
  raw: v.optional(v.string()),
  // The model and, for a routed call, the provider that answered, and what
  // the router billed for the call.
  model: v.optional(v.string()),
  provider: v.optional(v.string()),
  costUsd: v.optional(v.number()),
  // importance: the score (0-9) the memory was given, and what was scored.
  score: v.optional(v.number()),
  memoryKind: v.optional(v.string()),
  // leave_conversation: the conversation, and how far it had gone.
  conversationId: v.optional(v.string()),
  messagesSoFar: v.optional(v.number()),
};
