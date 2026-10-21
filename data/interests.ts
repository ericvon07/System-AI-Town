// The interest town (deviation D8), under TOWN_CAST=interests:
// six people who are the same text but for their names and six pastimes.
// Unset, the town is the five of data/characters.ts, as published.
//
// Every pastime belongs to exactly two people, so nothing one pair shares
// reaches a third. The pairs that share 3, 2 and 1 pastimes are three disjoint
// matchings of the six, so everyone has one partner at each of 3, 2 and 1 and
// two at 0: each person is their own control.
//
// Nothing says who shares what. The text is Sam's base of
// data/variants.ts without the bakery (six coworkers would share a workplace
// across every pair), with the middle step of sociability, the neutral goal,
// and no pronoun. No text here contains interest, interested, interesting,
// share, shared, common or similar. The eighteen pastimes come from eighteen
// different domains, so two lists that share nothing are not near each other
// either; they are dealt to the pairs by a fixed seed, and each list is put in
// its own shuffled order, so the shared ones are not always first.
//
// Two dealings, picked by INTEREST_DEALING (1, the default, or
// 2): the same six names, texts and eighteen pastimes, but in the second every
// name takes another row of the first's table -- one of the 24 permutations
// under which no pair keeps its level -- and the pastimes are dealt again by
// another seed, none falling to the pair it had. So a level rests on six pairs
// across the two worlds, tied to no one set of pastimes and no one pair of names.

import { Descriptions } from './characters';
import { NEUTRAL_GOAL } from './variants';

export type TownCast = 'upstream' | 'interests';

export function townCast(): TownCast {
  return process.env.TOWN_CAST === 'interests' ? 'interests' : 'upstream';
}

export type Dealing = 1 | 2;

export function interestDealing(): Dealing {
  return process.env.INTEREST_DEALING === '2' ? 2 : 1;
}

export type TownPerson = { name: string; character: string; identity: string; plan: string };

export const NAMES = ['Avery', 'Blair', 'Laura', 'Drew', 'Emery', 'Frankie'];
const SPRITES = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6'];

export const PASTIMES = [
  'chess',
  'birdwatching',
  'sourdough baking',
  'jazz records',
  'pottery',
  'stargazing',
  'long-distance cycling',
  'learning Japanese',
  'fly fishing',
  'silent films',
  'poetry',
  'local history',
  'salsa dancing',
  'repairing old radios',
  'yoga',
  'card magic',
  'bonsai',
  'stamp collecting',
];

// The pairs at each level, by dealing: how many pastimes the two have in their
// lists. In the second, Avery, Blair, Laura, Drew, Emery and Frankie take the
// first's rows of Avery, Emery, Frankie, Laura, Drew and Blair.
type Matchings = Array<{ level: number; pairs: Array<[string, string]> }>;

export const MATCHINGS: Record<Dealing, Matchings> = {
  1: [
    {
      level: 3,
      pairs: [
        ['Avery', 'Blair'],
        ['Laura', 'Drew'],
        ['Emery', 'Frankie'],
      ],
    },
    {
      level: 2,
      pairs: [
        ['Avery', 'Laura'],
        ['Blair', 'Emery'],
        ['Drew', 'Frankie'],
      ],
    },
    {
      level: 1,
      pairs: [
        ['Avery', 'Drew'],
        ['Blair', 'Frankie'],
        ['Laura', 'Emery'],
      ],
    },
  ],
  2: [
    {
      level: 3,
      pairs: [
        ['Avery', 'Frankie'],
        ['Blair', 'Laura'],
        ['Drew', 'Emery'],
      ],
    },
    {
      level: 2,
      pairs: [
        ['Avery', 'Drew'],
        ['Blair', 'Frankie'],
        ['Laura', 'Emery'],
      ],
    },
    {
      level: 1,
      pairs: [
        ['Avery', 'Emery'],
        ['Blair', 'Drew'],
        ['Laura', 'Frankie'],
      ],
    },
  ],
};

export const SEEDS: Record<Dealing, number> = { 1: 20261008, 2: 20261009 };

// mulberry32: a small seeded generator, so the deal is the same on every run.
function seeded(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function shuffled<T>(items: T[], random: () => number): T[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// The deal: the pool shuffled once, then handed out in order, three to each
// pair at level 3, two at level 2, one at level 1; then each list shuffled.
function deal(dealing: Dealing): Record<string, string[]> {
  const random = seeded(SEEDS[dealing]);
  const pool = shuffled(PASTIMES, random);
  const lists: Record<string, string[]> = Object.fromEntries(NAMES.map((n) => [n, []]));
  let next = 0;
  for (const { level, pairs } of MATCHINGS[dealing]) {
    for (const [a, b] of pairs) {
      const dealt = pool.slice(next, next + level);
      next += level;
      lists[a].push(...dealt);
      lists[b].push(...dealt);
    }
  }
  for (const name of NAMES) {
    lists[name] = shuffled(lists[name], random);
  }
  return lists;
}

export const PASTIMES_OF: Record<Dealing, Record<string, string[]>> = { 1: deal(1), 2: deal(2) };

// How many pastimes two people of the town have in both their lists, 0 to 3.
export function level(a: string, b: string, dealing: Dealing = interestDealing()): number {
  return PASTIMES_OF[dealing][a].filter((p) => PASTIMES_OF[dealing][b].includes(p)).length;
}

function listed(items: string[]) {
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

export function interestDescriptions(dealing: Dealing): TownPerson[] {
  return NAMES.map((name, i) => ({
    name,
    character: SPRITES[i],
    identity:
      `${name} has lived in town for years and spends the afternoons walking around town. ` +
      `${name} neither seeks out company nor avoids it. ` +
      `In free time, ${name} enjoys ${listed(PASTIMES_OF[dealing][name])}.`,
    plan: NEUTRAL_GOAL,
  }));
}

// The people `init` creates: the interest town under TOWN_CAST=interests, in
// the dealing INTEREST_DEALING names, otherwise upstream's five.
export function townDescriptions(): TownPerson[] {
  return townCast() === 'interests' ? interestDescriptions(interestDealing()) : Descriptions;
}
