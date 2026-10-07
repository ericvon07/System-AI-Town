// What the OpenRouter key has spent, as the router counts it. The run script
// reads it before and after an arm, and between heartbeats as the spend guard.
//
//   npx convex run spend:openRouter

import { internalAction } from './_generated/server';
import { openRouterUsage } from './util/openrouter';

export const openRouter = internalAction({
  args: {},
  handler: async () => await openRouterUsage(),
});
