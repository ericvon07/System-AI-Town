// Designed characters for the probes only (probe.ts: probeVariants); none of
// them lives in the town. Each family changes one thing at a time, so a
// difference in what a judge answers is caused by that one thing:
//
//   sociability  Sam, one sentence in five steps from avoiding people to
//                seeking them out; should move accept, invite and leave.
//   stamina      Sam, one sentence in five steps from tiring of talk quickly
//                to talking for hours; should move leave as the talk goes on.
//   ablation     Bob with and without his two avoidance texts (the sentence
//                in his identity, the goal), two by two: which words carry
//                his behaviour.
//   names        Bob's text, word for word, under five names: the judge
//                should read the trait, not the name.
//
// Sam's texts name Sam and use no pronoun, so nothing else changes between
// steps; the goal is the same neutral line in every family that replaces one.
// The counterparts in every question are the town's Lucky, Stella, Alice and
// Pete (data/characters.ts), never Bob.

import { Descriptions } from './characters';

export type Variant = { variant: string; name: string; identity: string; plan: string };

const SAM =
  'Sam has lived in town for years, works mornings at the bakery on the square, ' +
  'and spends the afternoons walking around town.';
export const NEUTRAL_GOAL = 'You want to have a good day in town.';

const BOB = Descriptions.find((d) => d.name === 'Bob')!;
const BOB_AVOIDANCE =
  " When spoken to he'll respond but try and get out of the conversation as quickly as possible.";

function sam(variant: string, sentence: string): Variant {
  return { variant, name: 'Sam', identity: `${SAM} ${sentence}`, plan: NEUTRAL_GOAL };
}

export const VARIANT_FAMILIES: Record<string, Variant[]> = {
  sociability: [
    sam('s0 avoids people', 'Sam avoids other people as much as possible.'),
    sam('s1 prefers to be alone', 'Sam mostly prefers to be alone, but is polite when spoken to.'),
    sam('s2 neither', 'Sam neither seeks out company nor avoids it.'),
    sam('s3 enjoys company', 'Sam enjoys company and is usually glad when someone stops to talk.'),
    sam(
      's4 seeks people out',
      'Sam loves being around people and seeks out company at every chance.',
    ),
  ],
  stamina: [
    sam('t0 tires quickly', 'Sam tires of talking quickly; a few exchanges are plenty.'),
    sam('t1 short chats', 'Sam is happy with a short chat and starts to drift after a while.'),
    sam('t2 a reasonable while', 'Sam can keep a conversation going for a reasonable while.'),
    sam(
      't3 long conversations',
      'Sam enjoys long conversations and rarely feels the need to wrap them up.',
    ),
    sam(
      't4 talks for hours',
      'Sam could talk for hours and never seems to run out of things to say.',
    ),
  ],
  ablation: [
    { variant: 'bob as written', name: 'Bob', identity: BOB.identity, plan: BOB.plan },
    {
      variant: 'without the avoidance sentence',
      name: 'Bob',
      identity: BOB.identity.replace(BOB_AVOIDANCE, ''),
      plan: BOB.plan,
    },
    {
      variant: 'without the avoidance goal',
      name: 'Bob',
      identity: BOB.identity,
      plan: NEUTRAL_GOAL,
    },
    {
      variant: 'without both',
      name: 'Bob',
      identity: BOB.identity.replace(BOB_AVOIDANCE, ''),
      plan: NEUTRAL_GOAL,
    },
  ],
  names: ['Bob', 'Tom', 'Ahmed', 'Kenji', 'Mateo'].map((name) => ({
    variant: name,
    name,
    identity: BOB.identity.replace(/\bBob\b/g, name),
    plan: BOB.plan,
  })),
};

// The four the designed characters meet, invite and are invited by.
export const COUNTERPARTS = Descriptions.filter((d) => d.name !== 'Bob').map((d) => ({
  name: d.name,
  identity: d.identity,
  goal: d.plan,
}));
