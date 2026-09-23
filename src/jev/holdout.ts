/** Run the frozen synthetic API fixture only after explicit --live selection. */
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { credentials } from './cli.js';
import { TypeSafeClient } from './native.js';
import { BOUNDARY_QUESTIONS, boundaryState, selectedBudget } from './boundary.js';
const args = process.argv.slice(2);
if (args.length !== 3 || args[0] !== '--live') {
  process.stderr.write('Usage: node dist/jev/holdout.js --live FIXTURE_JSON NEW_RESULT_JSON\n');
  process.exitCode = 2;
} else {
  const text = await readFile(resolve(args[1]), 'utf8');
  const fixture = JSON.parse(text);
  const receipt = { fixture_sha256: createHash('sha256').update(text).digest('hex'),
    questions_sha256: createHash('sha256').update(JSON.stringify(BOUNDARY_QUESTIONS)).digest('hex'),
    started_at: new Date().toISOString(), fixture_version: fixture.version,
    note: 'API classification only. This does not measure downstream token savings or task correctness.', cases: [] as unknown[] };
  // Reserve a new result file before any request. Do not overwrite earlier evidence.
  await writeFile(resolve(args[2]), JSON.stringify(receipt, null, 2), { flag: 'wx', mode: 0o600 });
  const client = new TypeSafeClient(await credentials(), 750);
  for (const item of fixture.cases) {
    const payload = { messages: [{ role: 'user', content: item.task }, ...(item.tool_output ? [{ role: 'tool', content: item.tool_output }] : [])],
      tools: item.tools.map((name: string) => ({ function: { name } })) };
    const started = performance.now();
    try {
      const evaluation = await client.evaluate(boundaryState(payload), BOUNDARY_QUESTIONS);
      const budgets = Object.fromEntries((['conservative-v1', 'choice-only-v2'] as const).map(policy => [policy,
        selectedBudget(evaluation.answers.workload, evaluation.answers.clarification, policy, fixture.confidence_threshold)]));
      receipt.cases.push({ id: item.id, expected: item.expected, latency_ms: performance.now() - started,
        answers: evaluation.answers, budgets,
        unsafe_shortening: Object.fromEntries(Object.entries(budgets).map(([policy, budget]) => [policy, item.expected === 'normal_required' && budget !== null])) });
    } catch {
      receipt.cases.push({ id: item.id, expected: item.expected, latency_ms: performance.now() - started, status: 'api_failure_default_preserved' });
    }
    await writeFile(resolve(args[2]), JSON.stringify(receipt, null, 2), { mode: 0o600 });
  }
  process.stdout.write(`Recorded ${receipt.cases.length} synthetic cases.\n`);
}
