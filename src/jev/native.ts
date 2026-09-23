/** The native TypeSafe API serves explicit advice and the enabled request-boundary policy. */
export const JEV_MODEL = 'jev-1.13.0';
const ENDPOINT = 'https://api.typesafe.ai/v1/systemone';
export type Question = { type: 'noul'; instructions: string } |
  { type: 'choice'; instructions: string; criteria: Record<string, string> } |
  { type: 'score'; instructions: string; criteria: string[] };
export type Answer = { type: 'noul'; noul: number } |
  { type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number } |
  { type: 'score'; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number };
export interface Evaluation { model: string; answers: Record<string, Answer>; usage: { input_tokens: number; output_tokens: number } }
export class DecisionError extends Error {
  constructor(public readonly code: string) { super(code); }
}
function object(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}
function unit(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1; }
function sameKeys(value: Record<string, unknown>, keys: string[]): boolean {
  return Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));
}
function distribution(value: unknown, keys: string[]): value is Record<string, number> {
  return object(value) && sameKeys(value, keys) && Object.values(value).every(unit) &&
    Math.abs(Object.values(value).reduce<number>((sum, item) => sum + (item as number), 0) - 1) <= 0.001;
}
export function validateEvaluation(value: unknown, questions: Record<string, Question>): Evaluation {
  const invalid = () => { throw new DecisionError('invalid_response'); };
  if (!object(value) || value.model !== JEV_MODEL || !object(value.answers) || !sameKeys(value.answers, Object.keys(questions))) return invalid();
  if (!object(value.usage) || !['input_tokens', 'output_tokens'].every(key => Number.isSafeInteger(value.usage && (value.usage as Record<string, unknown>)[key]) && ((value.usage as Record<string, number>)[key] >= 0))) return invalid();
  for (const [id, question] of Object.entries(questions)) {
    const answer = value.answers[id];
    if (!object(answer) || answer.type !== question.type) return invalid();
    if (question.type === 'noul') { if (!unit(answer.noul)) return invalid(); continue; }
    if (!unit(answer.confidence)) return invalid();
    const keys = question.type === 'choice' ? Object.keys(question.criteria) : question.criteria.map((_, index) => String(index));
    if (!distribution(answer.probabilities, keys)) return invalid();
    if (question.type === 'choice') {
      if (typeof answer.choice !== 'string' || !keys.includes(answer.choice) || answer.probabilities[answer.choice] + 0.001 < Math.max(...Object.values(answer.probabilities))) return invalid();
    } else {
      if (typeof answer.score !== 'number' || !Number.isFinite(answer.score) || answer.score < 0 || answer.score > keys.length - 1 || !object(answer.legend) || !sameKeys(answer.legend, keys)) return invalid();
      if (!keys.every(key => answer.legend && (answer.legend as Record<string, unknown>)[key] === question.criteria[Number(key)])) return invalid();
      const expected = keys.reduce((sum, key) => sum + Number(key) * (answer.probabilities as Record<string, number>)[key], 0);
      if (Math.abs(expected - answer.score) > 0.02) return invalid();
    }
  }
  return value as unknown as Evaluation;
}
export class TypeSafeClient {
  constructor(private readonly apiKey: string, private readonly timeoutMs = 5000, private readonly transport: typeof fetch = fetch) {
    if (!apiKey.trim()) throw new DecisionError('missing_credentials');
    if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 30000) throw new DecisionError('invalid_timeout');
  }
  async evaluate(state: unknown, questions: Record<string, Question>, signal?: AbortSignal): Promise<Evaluation> {
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (signal?.aborted) throw new DecisionError('cancelled');
    signal?.addEventListener('abort', abort, { once: true });
    const timer = setTimeout(abort, this.timeoutMs);
    try {
      const response = await this.transport(ENDPOINT, { method: 'POST', redirect: 'error', signal: controller.signal,
        headers: { Authorization: `Bearer ${this.apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: JEV_MODEL, state, questions }) });
      if (!response.ok) throw new DecisionError(`http_${response.status}`);
      if (!response.body) throw new DecisionError('invalid_response');
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let size = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.length;
        if (size > 65536) { await reader.cancel(); throw new DecisionError('response_too_large'); }
        chunks.push(value);
      }
      return validateEvaluation(JSON.parse(Buffer.concat(chunks).toString('utf8')), questions);
    } catch (error) {
      if (signal?.aborted) throw new DecisionError('cancelled');
      if (controller.signal.aborted) throw new DecisionError('timeout');
      if (error instanceof DecisionError) throw error;
      throw new DecisionError('request_failed');
    } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort); }
  }
}
