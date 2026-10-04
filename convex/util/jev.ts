// Transport for TypeSafe's System One model (Jev).
//
// One HTTP call carries one named-object `state` and a map of typed
// `questions`; every question is evaluated in parallel against that state and
// comes back with its own typed answer. Only `choice` is used here:
//
//   choice -> { choice: <label>, probabilities: { label: p }, confidence }
//
// The wire format mirrors the System One Lab's Python client
// (generative_agents/reverie/backend_server/persona/prompt_template/jev.py),
// which talks HTTP directly for the same reason this file does: the format is
// small and stable, and it keeps the dependency list unchanged.
//
// Configuration, all through `npx convex env set`:
//   TYPESAFE_API_KEY   the key (required for the jev backend)
//   JEV_MODEL          default 'jev-latest'
//   JEV_BASE_URL       default 'https://api.typesafe.ai/v1'

export type ChoiceQuestion = {
  type: 'choice';
  instructions: string;
  criteria: Record<string, string | null>;
};

export type ChoiceAnswer = {
  choice: string;
  probabilities: Record<string, number>;
  confidence?: number;
};

export class JevError extends Error {}

const RETRYABLE_STATUS = new Set([408, 409, 425, 429, 500, 502, 503, 504, 529]);
// Five attempts, waits of 2, 4, 8 and 16 seconds: half a minute of patience,
// well inside the 120 s an agent operation has before the engine times it out.
const MAX_ATTEMPTS = 5;
const BASE_DELAY_MS = 2000;
const TIMEOUT_MS = 30_000;

export function choice(
  instructions: string,
  criteria: Record<string, string | null>,
): ChoiceQuestion {
  return { type: 'choice', instructions, criteria };
}

export function jevConfigured() {
  return !!process.env.TYPESAFE_API_KEY;
}

export async function askJev(
  state: unknown,
  questions: Record<string, ChoiceQuestion>,
): Promise<{ answers: Record<string, ChoiceAnswer>; model?: string; latencyMs: number }> {
  const apiKey = process.env.TYPESAFE_API_KEY;
  if (!apiKey) {
    throw new JevError('no API key: run npx convex env set TYPESAFE_API_KEY <key>');
  }
  const baseUrl = (process.env.JEV_BASE_URL ?? 'https://api.typesafe.ai/v1').replace(/\/$/, '');
  const body = JSON.stringify({
    state,
    model: process.env.JEV_MODEL ?? 'jev-latest',
    questions,
  });

  let lastError: unknown;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const started = Date.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    let retryable = true;
    try {
      const response = await fetch(baseUrl + '/systemone', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          'User-Agent': 'ai-town-jev/0.1 (research)',
        },
        body,
        signal: controller.signal,
      });
      if (!response.ok) {
        retryable = RETRYABLE_STATUS.has(response.status);
        throw new JevError(`HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
      }
      const json = await response.json();
      if (!json || typeof json.answers !== 'object') {
        retryable = false;
        throw new JevError('response carried no answers map');
      }
      return { answers: json.answers, model: json.model, latencyMs: Date.now() - started };
    } catch (e) {
      lastError = e;
      if (!retryable || attempt === MAX_ATTEMPTS) {
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, BASE_DELAY_MS * 2 ** (attempt - 1)));
    } finally {
      clearTimeout(timer);
    }
  }
  throw new JevError(lastError instanceof Error ? lastError.message : String(lastError));
}
