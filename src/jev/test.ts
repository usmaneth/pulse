import test from 'node:test';
import assert from 'node:assert/strict';
import { TypeSafeClient, JEV_MODEL, validateEvaluation, Question } from './native.js';
import { decide, QUESTIONS, validateRequest } from './decide.js';
import { JevDecisionClient } from './client.js';
const request = { task: 'Repair a test failure.', evidence: [], available_inspections: ['Read the failing test.'], unresolved: ['Failure cause.'] };
function response(action = 'inspect', probability = 0.98, missing = 0.01) {
  return { model: JEV_MODEL, usage: { input_tokens: 12, output_tokens: 20 }, answers: {
    next_step: { type: 'choice', choice: action, probabilities: Object.fromEntries(['inspect', 'ready', 'clarify'].map(key => [key, key === action ? probability : (1 - probability) / 2])), confidence: probability },
    blocking_missing: { type: 'noul', noul: missing },
    task_category: { type: 'choice', choice: 'code', probabilities: { code: 1, research: 0, operations: 0, other: 0 }, confidence: 1 },
    evidence_sufficiency: { type: 'score', score: 1, legend: Object.fromEntries((QUESTIONS.evidence_sufficiency as Extract<Question, {type: 'score'}>).criteria.map((value, i) => [i, value])), probabilities: { '0': 0, '1': 1, '2': 0 }, confidence: 1 },
  } };
}
function client(value: unknown) { return new TypeSafeClient('test-only', 1000, async () => new Response(JSON.stringify(value))); }
test('native request pins endpoint and model and sends only supplied state', async () => {
  let calls = 0;
  const api = new TypeSafeClient('test-only', 1000, async (url, init) => {
    calls++;
    assert.equal(url, 'https://api.typesafe.ai/v1/systemone');
    assert.equal(init?.redirect, 'error');
    const body = JSON.parse(String(init?.body));
    assert.equal(body.model, JEV_MODEL); assert.deepEqual(body.state, request);
    return new Response(JSON.stringify(response()));
  });
  const result = await decide(api, request);
  assert.equal(calls, 1); assert.equal(result.action, 'inspect'); assert.equal(result.advisory_only, true);
});
test('all Choice options and finite normalized probabilities are required', () => {
  for (const mutate of [
    (r: any) => { delete r.answers.next_step.probabilities.ready; },
    (r: any) => { r.answers.next_step.probabilities.extra = 0; },
    (r: any) => { r.answers.next_step.probabilities.inspect = NaN; },
    (r: any) => { r.answers.next_step.probabilities.inspect = -1; },
    (r: any) => { r.answers.next_step.probabilities.inspect = 0.4; },
    (r: any) => { r.answers.next_step.choice = 'ready'; },
    (r: any) => { delete r.answers.next_step.confidence; },
    (r: any) => { r.answers.next_step.confidence = 2; },
  ]) { const r = response(); mutate(r); assert.throws(() => validateEvaluation(r, QUESTIONS)); }
});
test('Score, Noul, model and complete answer schemas are validated', () => {
  for (const mutate of [
    (r: any) => { r.answers.blocking_missing.noul = Infinity; },
    (r: any) => { r.answers.evidence_sufficiency.score = 2; },
    (r: any) => { delete r.answers.evidence_sufficiency.legend['1']; },
    (r: any) => { r.answers.evidence_sufficiency.legend['1'] = 'other'; },
    (r: any) => { r.model = 'jev-latest'; },
    (r: any) => { delete r.answers.next_step; },
    (r: any) => { r.usage.input_tokens = -1; },
  ]) { const r = response(); mutate(r); assert.throws(() => validateEvaluation(r, QUESTIONS)); }
});
test('policy abstains on uncertainty and inconsistent clarification answers', async () => {
  assert.equal((await decide(client(response('inspect', 0.5)), request)).action, 'abstain');
  assert.equal((await decide(client(response('clarify', 0.98, 0.01)), request)).action, 'abstain');
  assert.equal((await decide(client(response('clarify', 0.98, 0.99)), request)).action, 'clarify');
  assert.equal((await decide(client(response()), { ...request, available_inspections: [] })).action, 'abstain');
  assert.equal((await decide(client({}), request)).reason, 'invalid_response');
});
test('timeout and caller cancellation return bounded abstention', async () => {
  const transport: typeof fetch = async (_url, init) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(new Error('private failure text')), { once: true });
  });
  assert.equal((await decide(new TypeSafeClient('test-only', 10, transport), request)).reason, 'timeout');
  const controller = new AbortController(); controller.abort();
  assert.equal((await decide(new TypeSafeClient('test-only', 1000, transport), request, controller.signal)).reason, 'cancelled');
});
test('HTTP failures and oversized bodies do not expose response content', async () => {
  const api = new TypeSafeClient('test-only', 1000, async () => new Response('private upstream text', { status: 429 }));
  const result = await decide(api, request); assert.equal(result.reason, 'http_429'); assert.equal(result.answers, null);
  assert.equal((await decide(client('x'.repeat(70000)), request)).reason, 'response_too_large');
});
test('request schema rejects unknown fields and malformed evidence', () => {
  assert.throws(() => validateRequest({ ...request, auto_read_vault: true }));
  assert.throws(() => validateRequest({ ...request, evidence: 'read everything' }));
});
test('legacy methods use local policy with no invented probabilities', async () => {
  const local = new JevDecisionClient();
  assert.equal((await local.decideSpeculationK({ promptSnippet: 'JSON schema', taskDomain: 'code' })).confidence, null);
  assert.equal((await local.decideMemoryAdmission({ activeSessions: 0, usedKVMemoryGB: 101, totalAddressableGB: 121, incomingTokens: 10 })).action, 'queue_wait');
  assert.equal((await local.decideMemoryAdmission({ activeSessions: 0, usedKVMemoryGB: NaN, totalAddressableGB: 121, incomingTokens: 10 })).action, 'queue_wait');
  assert.equal((await local.decideToolRouting('read a file')).probability, null);
});

test('request boundary changes only budget and respects the user override', async () => {
  const { routeReasoning, boundaryState } = await import('./boundary.js');
  const payload = { messages: [{ role: 'user', content: 'Read the README. OPENAI_API_KEY=sk-test-secret-secret apikey_synthetic_key_12345 AI_GATEWAY_API_KEY=syntheticgate DATABASE_PASSWORD=syntheticpass ACCESS_TOKEN=synthetictoken' }, { role: 'tool', content: 'Process exited with code 0\nPRIVATE SOURCE CONTENT' }], tools: [{ type: 'function', function: { name: 'read_file' } }] };
  const before = structuredClone(payload);
  const state = boundaryState(payload);
  assert.ok(!JSON.stringify(state).includes('PRIVATE SOURCE'));
  assert.ok(!JSON.stringify(state).includes('sk-test'));
  assert.ok(!JSON.stringify(state).includes('synthetic'));
  let calls = 0;
  const factory = async () => new TypeSafeClient('test-only', 1000, async () => {
    calls++;
    return new Response(JSON.stringify({ model: JEV_MODEL, usage: { input_tokens: 1, output_tokens: 1 }, answers: {
      workload: { type: 'choice', choice: 'simple', confidence: 0.98, probabilities: { simple: 0.99, normal: 0.01 } }, clarification: { type: 'noul', noul: 0.01 },
    } }));
  });
  const result = await routeReasoning(payload, false, undefined, factory);
  assert.equal(result.selected_reasoning_budget, 512);
  assert.deepEqual(payload.messages, before.messages); assert.deepEqual(payload.tools, before.tools);
  const explicit = { ...before, reasoning_budget_tokens: -1 };
  assert.equal((await routeReasoning(explicit, true, undefined, factory)).status, 'user_override');
  assert.equal(explicit.reasoning_budget_tokens, -1); assert.equal(calls, 1);
});
test('boundary errors preserve the normal budget and prompt', async () => {
  const { routeReasoning } = await import('./boundary.js');
  const payload = { messages: [{ role: 'user', content: 'Explain a failure.' }], tools: [] };
  const before = structuredClone(payload);
  const result = await routeReasoning(payload, false, undefined, async () => client({}));
  assert.equal(result.status, 'invalid_response'); assert.deepEqual(payload, before);
  const slow = async () => new TypeSafeClient('test-only', 10, async (_url, init) => new Promise((_resolve, reject) => init?.signal?.addEventListener('abort', () => reject(new Error('timeout')))));
  assert.equal((await routeReasoning(payload, false, undefined, slow)).status, 'timeout');
  assert.deepEqual(payload, before);
});
test('candidate policy changes only the clarification gate and keeps the same budget', async () => {
  const { selectedBudget } = await import('./boundary.js');
  const simple = { type: 'choice' as const, choice: 'simple', confidence: 0.9, probabilities: { simple: 0.95, normal: 0.05 } };
  const uncertain = { type: 'noul' as const, noul: 0.6 };
  assert.equal(selectedBudget(simple, uncertain, 'conservative-v1'), null);
  assert.equal(selectedBudget(simple, uncertain, 'choice-only-v2'), 512);
  assert.equal(selectedBudget(simple, { type: 'noul', noul: 0.1 }, 'conservative-v1'), 512);
  for (const policy of ['conservative-v1', 'choice-only-v2'] as const) {
    assert.equal(selectedBudget({ ...simple, confidence: 0.79 }, uncertain, policy), null);
    assert.equal(selectedBudget({ ...simple, choice: 'normal', probabilities: { normal: 0.95, simple: 0.05 } }, uncertain, policy), null);
    assert.throws(() => selectedBudget(simple, uncertain, policy, 0.79));
  }
});
test('unknown environment policy preserves the payload without a network call', async () => {
  const { routeReasoning } = await import('./boundary.js');
  const previous = process.env.PULSE_JEV_POLICY;
  process.env.PULSE_JEV_POLICY = 'unknown';
  try {
    const payload = { messages: [{ role: 'user', content: 'Read the README.' }] };
    const before = structuredClone(payload);
    let calls = 0;
    const result = await routeReasoning(payload, false, undefined, async () => { calls++; return client({}); });
    assert.equal(result.status, 'invalid_reasoning_policy'); assert.equal(calls, 0); assert.deepEqual(payload, before);
  } finally { if (previous === undefined) delete process.env.PULSE_JEV_POLICY; else process.env.PULSE_JEV_POLICY = previous; }
});
