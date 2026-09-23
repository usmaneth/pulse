import assert from 'node:assert/strict';
import { adviseContinuity, CONTINUITY_QUESTIONS, continuityFallback, validateContinuity, ContinuityInput } from './continuity.js';
import { JEV_MODEL, TypeSafeClient } from './native.js';
const input: ContinuityInput = { objective: 'Repair retry logic', latest_user: 'Continue the retry fix', revision: 3,
  context_pressure: 0.5, completed_count: 0, next_steps_count: 2, blockers_count: 0, evidence_count: 1 };
assert.equal(continuityFallback(input).actions.checkpoint, 'defer');
assert.equal(continuityFallback({ ...input, context_pressure: 0.75 }).actions.checkpoint, 'now');
assert.equal(continuityFallback({ ...input, evidence_count: 0 }).actions.retrieve, 'now');
for (const invalid of [{ ...input, transcript: 'not allowed' }, { ...input, revision: -1 }, { ...input, context_pressure: NaN },
  { ...input, objective: 'x'.repeat(3001) }, { ...input, latest_user: 'x'.repeat(4097) }, { ...input, evidence_count: 0.5 }]) {
  assert.throws(() => validateContinuity(invalid));
}
function response(confidence = 0.95) {
  return { model: JEV_MODEL, usage: { input_tokens: 1, output_tokens: 1 }, answers: Object.fromEntries(
    Object.entries(CONTINUITY_QUESTIONS).map(([key, question]) => {
      assert.equal(question.type, 'choice'); if (question.type !== 'choice') throw Error();
      const choice = key === 'topic' ? 'same' : 'defer';
      return [key, { type: 'choice', choice, confidence, probabilities: Object.fromEntries(Object.keys(question.criteria).map(k => [k, k === choice ? 1 : 0])) }];
    })) };
}
let calls = 0; let posted = '';
const transport = (async (_url, options) => { calls++; posted = String(options?.body); return new Response(JSON.stringify(response())); }) as typeof fetch;
const advised = await adviseContinuity({ ...input, objective: 'API_KEY=private123 repair', latest_user: 'Bearer secret456 continue' }, async () => new TypeSafeClient('test', 750, transport));
assert.equal(calls, 1); assert.equal(advised.source, 'typesafe_native'); assert.equal(advised.actions.topic, 'same');
assert(!posted.includes('private123')); assert(!posted.includes('secret456')); assert(posted.includes('[credential removed]'));
const low = await adviseContinuity(input, async () => new TypeSafeClient('test', 750, (async () => new Response(JSON.stringify(response(0.2)))) as typeof fetch));
assert.equal(low.status, 'low_confidence'); assert.equal(low.source, 'local_fallback');
const malformed = await adviseContinuity(input, async () => new TypeSafeClient('test', 750, (async () => new Response('{}')) as typeof fetch));
assert.equal(malformed.status, 'invalid_response');
const timeout = await adviseContinuity(input, async () => new TypeSafeClient('test', 10, (async (_u, options) => new Promise((_resolve, reject) => {
  options?.signal?.addEventListener('abort', () => reject(Error('aborted')), { once: true });
})) as typeof fetch));
assert.equal(timeout.status, 'timeout'); assert.deepEqual(timeout.actions, continuityFallback(input).actions);
const missing = await adviseContinuity(input, async () => new TypeSafeClient(''));
assert.equal(missing.status, 'missing_credentials');
console.log('continuity advisory tests passed');
