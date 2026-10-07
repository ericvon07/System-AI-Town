// Transport for Gemma through OpenRouter, pinned to one provider.
//
// Two callers: the gemma backend of the decision seam (decisions.ts), which
// asks the seam's questions as a prompt and reads a label back, and
// chatCompletion (llm.ts) under CHAT_BACKEND=openrouter, which sends the
// conversations, summaries and reflections here instead of to Ollama.
//
// The pin is what makes a run one set of weights: `order` names one provider
// and `allow_fallbacks` is off, so the router never moves a call elsewhere.
// The provider that answered comes back in the response and is recorded.
//
// Configuration, all through `npx convex env set`:
//   OPENROUTER_API_KEY    the key (required)
//   GEMMA_MODEL           default 'google/gemma-4-31b-it'
//   OPENROUTER_PROVIDER   default 'friendli'
//   OPENROUTER_BASE_URL   default 'https://openrouter.ai/api/v1'

export class OpenRouterError extends Error {}

// A request the router refused for good: malformed, key, balance, permission.
// Everything else (429, 5xx, 404 from a provider that blinks, a timeout, a
// 200 with no choices) is waited out.
const FATAL_STATUS = new Set([400, 401, 402, 403]);
// Five attempts, waits of 2, 4, 8 and 16 seconds, as jev.ts: inside the 120 s
// an agent operation has before the engine times it out.
const MAX_ATTEMPTS = 5;
const BASE_DELAY_MS = 2000;
const TIMEOUT_MS = 20_000;

export function gemmaModel() {
  return process.env.GEMMA_MODEL ?? 'google/gemma-4-31b-it';
}

export function openRouterProvider() {
  return process.env.OPENROUTER_PROVIDER ?? 'friendli';
}

export function openRouterBaseUrl() {
  return (process.env.OPENROUTER_BASE_URL ?? 'https://openrouter.ai/api/v1').replace(/\/$/, '');
}

export function openRouterKey() {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) {
    throw new OpenRouterError('no API key: run npx convex env set OPENROUTER_API_KEY <key>');
  }
  return key;
}

// The fields every request to the router carries besides the chat body.
export function openRouterExtras() {
  return {
    model: gemmaModel(),
    provider: { order: [openRouterProvider()], allow_fallbacks: false },
    usage: { include: true },
  };
}

export type OpenRouterAnswer = {
  content: string;
  model: string;
  provider: string | null;
  costUsd: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  latencyMs: number;
};

export async function askOpenRouter(
  messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
  options: { maxTokens: number; temperature: number },
): Promise<OpenRouterAnswer> {
  const body = JSON.stringify({
    ...openRouterExtras(),
    messages,
    max_tokens: options.maxTokens,
    temperature: options.temperature,
  });
  const key = openRouterKey();
  let lastError: unknown;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const started = Date.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    let retryable = true;
    try {
      const response = await fetch(openRouterBaseUrl() + '/chat/completions', {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
        body,
        signal: controller.signal,
      });
      if (!response.ok) {
        retryable = !FATAL_STATUS.has(response.status);
        throw new OpenRouterError(
          `HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`,
        );
      }
      const json = await response.json();
      const content = json?.choices?.[0]?.message?.content;
      if (typeof content !== 'string') {
        throw new OpenRouterError(`no content: ${JSON.stringify(json).slice(0, 200)}`);
      }
      return {
        content,
        model: json.model ?? gemmaModel(),
        provider: json.provider ?? null,
        costUsd: typeof json.usage?.cost === 'number' ? json.usage.cost : null,
        promptTokens: json.usage?.prompt_tokens ?? null,
        completionTokens: json.usage?.completion_tokens ?? null,
        latencyMs: Date.now() - started,
      };
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
  throw new OpenRouterError(lastError instanceof Error ? lastError.message : String(lastError));
}

// What the key has spent so far, in dollars, as the router counts it. The
// run script reads it before and after an arm, and between heartbeats as the
// spend guard.
export async function openRouterUsage(): Promise<{ usage: number; limitRemaining: number | null }> {
  const response = await fetch(openRouterBaseUrl() + '/key', {
    headers: { Authorization: `Bearer ${openRouterKey()}` },
  });
  if (!response.ok) {
    throw new OpenRouterError(`HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
  }
  const { data } = await response.json();
  return { usage: data?.usage ?? 0, limitRemaining: data?.limit_remaining ?? null };
}
