import { constants } from 'node:fs';
import { open } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { decide, validateRequest } from './decide.js';
import { DecisionError, JEV_MODEL, TypeSafeClient } from './native.js';
export async function credentials(): Promise<string> {
  if (process.env.TYPESAFE_API_KEY?.trim()) return process.env.TYPESAFE_API_KEY.trim();
  const path = process.env.TYPESAFE_API_KEY_FILE || join(homedir(), '.config/pulse/typesafe.env');
  let file;
  try {
    file = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
    const stat = await file.stat();
    if (!stat.isFile() || (stat.mode & 0o077) !== 0 || stat.uid !== process.getuid?.() || stat.size > 16384) throw new DecisionError('unsafe_credential_file');
    const text = await file.readFile('utf8');
    const match = text.match(/^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*?)\s*$/m);
    if (!match) throw new DecisionError('missing_credentials');
    let key = match[1];
    if ((key.startsWith('"') && key.endsWith('"')) || (key.startsWith("'") && key.endsWith("'"))) key = key.slice(1, -1);
    if (!key || /\s/.test(key)) throw new DecisionError('invalid_credentials');
    return key;
  } catch (error) { if (error instanceof DecisionError) throw error; throw new DecisionError('missing_credentials'); }
  finally { await file?.close(); }
}
export async function main(): Promise<void> {
  if (process.argv.slice(2).length) throw new DecisionError('usage_stdin_json_only');
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    size += chunk.length;
    if (size > 65536) throw new DecisionError('request_too_large');
    chunks.push(Buffer.from(chunk));
  }
  let input;
  try { input = validateRequest(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
  catch (error) { throw error instanceof DecisionError ? error : new DecisionError('invalid_request'); }
  const controller = new AbortController();
  const cancel = () => controller.abort();
  process.once('SIGINT', cancel); process.once('SIGTERM', cancel);
  try {
    const result = await decide(new TypeSafeClient(await credentials()), input, controller.signal);
    process.stdout.write(JSON.stringify(result) + '\n');
    if (result.action === 'abstain') process.exitCode = 2;
  } finally { process.removeListener('SIGINT', cancel); process.removeListener('SIGTERM', cancel); }
}
if (process.argv[1]?.endsWith('/jev/cli.js')) {
  main().catch(error => {
    process.stdout.write(JSON.stringify({ source: 'typesafe_native', model: JEV_MODEL, advisory_only: true, action: 'abstain',
      reason: error instanceof DecisionError ? error.code : 'request_failed', answers: null }) + '\n');
    process.exitCode = 2;
  });
}
