// Tests for `pulse model`: profile rendering, plans, parsers, the gateway
// fragment, and a full `up` and `down` against a fake recipe with stub docker
// and curl commands. No ssh, no GPU and no model server is needed.
// Run: npm run test:runtime

import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Writable } from 'node:stream';
import { fileURLToPath } from 'node:url';
import { EnvFileError, parseEnv, formatAssignment } from './envfile.js';
import { DockerArgsError, mergeDockerArgs, pairs, parseDockerArgs, tokenize } from './dockerargs.js';
import {
  ProfileError, diffMaps, loadNodes, loadProfile, listProfiles, readOverlay, renderEnv,
} from './profiles.js';
import type { NodeConfig, NodesFile, Rendered } from './profiles.js';
import { planDown, planUp, refusals, shq } from './plan.js';
import { ExecRunner, RecordingRunner, SSH_OPTIONS, commandFor, wrapScript } from './runner.js';
import { parseContainerState, parseEnvHeader, parseKvLine, parseModels, parseSmoke, parseStatus } from './status.js';
import { buildFragment, writeFragment, writeNodeState } from './fragment.js';
import type { NodeState } from './fragment.js';
import { main } from './cli.js';
import { loadConfig } from '../gateway/config.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const NODES_FILE = path.join(ROOT, 'runtime', 'nodes.json');
const PROFILES = path.join(ROOT, 'runtime', 'qwen38', 'profiles');
const NOW = new Date('2026-09-23T10:15:00Z');

function tmpdir(prefix = 'pulse-runtime-'): string {
  return mkdtempSync(path.join(os.tmpdir(), prefix));
}

function nodes(): NodesFile {
  return loadNodes(NODES_FILE);
}

function render(profile: string, node: string): Rendered {
  const n = nodes();
  return renderEnv(loadProfile(PROFILES, profile), n.nodes[node], { now: NOW });
}

/** Source a rendered file with bash in an empty environment and return the key map. */
function bashMap(text: string): Record<string, string> {
  const dir = tmpdir();
  const file = path.join(dir, '.env');
  writeFileSync(file, text);
  const out = execFileSync('env', ['-i', 'bash', '-c', 'set -a; source "$1"; env', '_', file], { encoding: 'utf8' });
  const map: Record<string, string> = {};
  for (const line of out.trim().split('\n')) {
    const eq = line.indexOf('=');
    const k = line.slice(0, eq);
    if (['PWD', 'OLDPWD', 'SHLVL', '_'].includes(k)) continue;
    map[k] = line.slice(eq + 1);
  }
  return map;
}

/** EXTRA_DOCKER_ARGS as a sorted list of pairs: docker does not depend on their order. */
function withSortedDocker(map: Record<string, string>): Record<string, string> {
  const out = { ...map };
  if (out.EXTRA_DOCKER_ARGS !== undefined) out.EXTRA_DOCKER_ARGS = pairs(out.EXTRA_DOCKER_ARGS.split(/\s+/).filter(Boolean)).sort().join(' | ');
  return out;
}

const SNAP = '/root/.cache/huggingface/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/925d7be6c14c6c9442ef83e8f05b5a3c39304f69';
const PKG = '/usr/local/lib/python3.12/dist-packages/vllm';

// The effective key map of /models/usman/qwen38-flash/.env on spark1
// (= profiles/spark1-best.env), from `env -i bash -c 'set -a; source .env; env'`.
const SPARK1_BEST: Record<string, string> = {
  ABLIT: '0',
  BIND: '127.0.0.1',
  CHAT_TEMPLATE: 'files/chat-template/chat_template.jinja',
  COMPILATION_MODE: '0',
  CUDAGRAPH_CAPTURE_SIZES: 'auto',
  EXTRA_DOCKER_ARGS: [
    '-e VLLM_USE_V2_MODEL_RUNNER=1',
    '-e VLLM_MTP_DRAFT_HEAD_FP8=1',
    '-e TRITON_CACHE_DIR=/triton-cache',
    '-v /models/usman/triton-cache:/triton-cache',
    '-v /models/usman/vllm-ple-cache:/models/usman/vllm-ple-cache',
    `-v /models/usman/qwen38-flash/files/ours/speculative.py:${PKG}/config/speculative.py:ro`,
    `-v /models/usman/qwen38-flash/files/ours/kv_cache_utils.py:${PKG}/v1/core/kv_cache_utils.py:ro`,
    `-v /models/usman/qwen38-flash/files/ours/scheduler.py:${PKG}/v1/core/sched/scheduler.py:ro`,
    `-v /models/usman/distill/shard34_r2.safetensors:${SNAP}/model-00034-of-00034.safetensors:ro`,
  ].join(' '),
  EXTRA_VLLM_ARGS: '--kv-cache-memory-bytes 10737418240',
  HF_HOME: '/models/usman/hf',
  HOST_RESERVE_GIB: '32',
  HOST_SLACK_GIB: '4',
  IMAGE: 'vllm/vllm-openai:qwen38-flash-next',
  KV_CACHE_DTYPE: 'fp8',
  KV_TARGET_GIB: '20',
  MAMBA_SSM_CACHE_DTYPE: 'bfloat16',
  MAX_MODEL_LEN: '262144',
  MAX_NUM_BATCHED_TOKENS: '8192',
  MAX_NUM_SEQS: '4',
  MTP_DISABLE_BLOCK_DROP: '1',
  MTP_DRAFT_VOCAB: 'files/draft_vocab_en_code_47k.txt',
  MTP_INDEX_SHARE: '1',
  MTP_K_SCHEDULE: '',
  MTP_NUM_SPECULATIVE_TOKENS: '3',
  PORT: '8888',
  SERVED_MODEL_NAME: 'qwen3.8-flash-next',
  YARN: '1',
  YARN_MAX_MODEL_LEN: '524288',
};

// The effective key map of /models/usman/qwen38-flash/.env.datagen on spark2.
const SPARK2_DATAGEN: Record<string, string> = {
  ...Object.fromEntries(Object.entries(SPARK1_BEST).filter(([k]) => !['MTP_DISABLE_BLOCK_DROP', 'MTP_INDEX_SHARE'].includes(k))),
  EXTRA_DOCKER_ARGS: '-e VLLM_USE_V2_MODEL_RUNNER=1 -v /models/usman/vllm-ple-cache:/models/usman/vllm-ple-cache',
  HF_HOME: '/home/usman/.cache/huggingface',
  MAX_NUM_BATCHED_TOKENS: '2048',
  MAX_NUM_SEQS: '8',
  REQUIRE_IDLE_GPU: 'false',
  YARN: '0',
};

// The effective key map of /models/usman/qwen38-flash/.env.capture on spark1.
const SPARK1_CAPTURE: Record<string, string> = {
  ...Object.fromEntries(Object.entries(SPARK1_BEST).filter(([k]) => k !== 'MTP_DISABLE_BLOCK_DROP')),
  EXTRA_DOCKER_ARGS: [
    '-e VLLM_USE_V2_MODEL_RUNNER=1',
    '-e VLLM_MTP_CAPTURE_DIR=/cap',
    '-v /models/usman/distill/cap:/cap',
    '-v /models/usman/vllm-ple-cache:/models/usman/vllm-ple-cache',
    `-v /models/usman/qwen38-flash/files/ours/ar_speculator_capture.py:${PKG}/v1/worker/gpu/spec_decode/autoregressive/speculator.py:ro`,
  ].join(' '),
  MAX_NUM_BATCHED_TOKENS: '2048',
};

// ---------------------------------------------------------------- env files

test('envfile: parses quoted, unquoted, empty and multi-line values and directives', () => {
  const p = parseEnv([
    '# pulse-recipe: qwen38-flash',
    '# a comment',
    'A=1   # trailing comment',
    'B="two words"',
    'C=',
    'D=""',
    'EXTRA_DOCKER_ARGS="',
    '  -e X=1',
    '  -v /a:/b:ro',
    '"   # comment after the value',
    'A=2',
  ].join('\n'));
  assert.deepEqual(p.directives.map((d) => [d.name, d.value]), [['recipe', 'qwen38-flash']]);
  assert.deepEqual(p.assignments.map((a) => [a.key, a.value]), [
    ['A', '1'], ['B', 'two words'], ['C', ''], ['D', ''], ['EXTRA_DOCKER_ARGS', '-e X=1 -v /a:/b:ro'], ['A', '2'],
  ]);
});

test('envfile: refuses values that are not safe in bash double quotes', () => {
  for (const bad of ['A="$(id)"', 'A="`id`"', 'A="a\\b"', 'A=$HOME', 'A=a;b', "A='x'", 'A=one two', 'export A=1', 'B="no end']) {
    assert.throws(() => parseEnv(bad), EnvFileError, bad);
  }
  assert.throws(() => formatAssignment('A', 'x"y'), EnvFileError);
  assert.throws(() => formatAssignment('A', 'x\ny'), EnvFileError);
  assert.equal(formatAssignment('A', 'x y'), 'A="x y"');
});

// ---------------------------------------------------------------- docker args

test('dockerargs: parses -e and -v pairs and refuses anything else', () => {
  const args = parseDockerArgs(tokenize('-e A=1 -v /x:/y:ro -v /p:/q'), 't');
  assert.deepEqual(args.map((a) => a.kind === 'env' ? `${a.name}=${a.value}` : `${a.src}>${a.dst}:${a.mode ?? ''}`), ['A=1', '/x>/y:ro', '/p>/q:']);
  assert.throws(() => tokenize('-e A=$(id)'), DockerArgsError);
  assert.throws(() => tokenize('-v /a:/b;reboot'), DockerArgsError);
  assert.throws(() => parseDockerArgs(['--gpus', 'all'], 't'), /unsupported docker argument/);
  assert.throws(() => parseDockerArgs(['-e', 'A'], 't'), /NAME=VALUE/);
  assert.throws(() => parseDockerArgs(['-v', 'rel:/b'], 't'), /absolute/);
  assert.throws(() => parseDockerArgs(['-v', '/a:/b:z'], 't'), /ro or rw/);
  assert.throws(() => parseDockerArgs(['-v'], 't'), /no value/);
});

test('dockerargs: a duplicate mount destination names both sources', () => {
  const a = parseDockerArgs(['-v', '/one:/dst'], 'profile:best');
  const b = parseDockerArgs(['-v', '/two:/dst'], 'overlay:/tmp/o');
  assert.throws(() => mergeDockerArgs([a, b]), (e: Error) => /\/one \(profile:best\)/.test(e.message) && /\/two \(overlay:\/tmp\/o\)/.test(e.message) && /--overlay-wins/.test(e.message));
  const m = mergeDockerArgs([a, b], { overlayWins: true });
  assert.equal(m.args.length, 1);
  assert.equal((m.args[0] as { src: string }).src, '/two');
  assert.match(m.replaced[0], /\/one \(profile:best\) replaced by \/two/);
  // Two profile entries on one destination stay an error even with --overlay-wins.
  assert.throws(() => mergeDockerArgs([a, parseDockerArgs(['-v', '/three:/dst'], 'node:x')], { overlayWins: true }), /duplicate mount/);
  assert.throws(() => mergeDockerArgs([parseDockerArgs(['-e', 'A=1', '-e', 'A=2'], 'p')]), /duplicate env A/);
});

test('dockerargs: destinations and variables that start.sh sets itself are refused', () => {
  const mtp = parseDockerArgs(['-v', `/x:${PKG}/models/qwen3_8_flash_next/nvidia/mtp.py:ro`], 'overlay:o');
  assert.throws(() => mergeDockerArgs([mtp], { recipe: 'qwen38-flash' }), /mounts this path itself/);
  assert.throws(() => mergeDockerArgs([parseDockerArgs(['-e', 'HF_TOKEN=x'], 'p')], { recipe: 'qwen38-flash' }), /sets this variable itself/);
});

// ---------------------------------------------------------------- profiles

test('profiles: the shipped profiles load, and tp2 is experimental and render-only', () => {
  assert.deepEqual(listProfiles(PROFILES), ['best', 'capture', 'datagen', 'tp2']);
  const tp2 = loadProfile(PROFILES, 'tp2');
  assert.equal(tp2.status, 'experimental');
  assert.equal(tp2.runnable, false);
  assert.equal(tp2.recipe, 'qwen38-dual');
  assert.equal(loadProfile(PROFILES, 'best').proofs.length, 3);
});

test('render: best on spark1 equals the effective spark1 .env (spark1-best.env)', () => {
  const r = render('best', 'spark1');
  assert.deepEqual(withSortedDocker(r.map), withSortedDocker(SPARK1_BEST));
  assert.equal(r.expectedMaxModelLen, 524288);
  assert.deepEqual(r.mkdirs, ['/models/usman/triton-cache']);
});

test('render: datagen on spark2 equals the effective spark2 .env.datagen', () => {
  const r = render('datagen', 'spark2');
  assert.deepEqual(withSortedDocker(r.map), withSortedDocker(SPARK2_DATAGEN));
  assert.equal(r.expectedMaxModelLen, 262144);
});

test('render: capture on spark1 equals the effective spark1 .env.capture', () => {
  const r = render('capture', 'spark1');
  assert.deepEqual(withSortedDocker(r.map), withSortedDocker(SPARK1_CAPTURE));
  assert.deepEqual(r.mkdirs, ['/models/usman/distill/cap']);
});

test('render: bash reads every rendered profile back to the same key map', () => {
  for (const [profile, node] of [['best', 'spark1'], ['best', 'spark2'], ['datagen', 'spark2'], ['capture', 'spark1'], ['tp2', 'spark1']]) {
    const r = render(profile, node);
    assert.deepEqual(bashMap(r.text), r.map, `${profile} on ${node}`);
  }
});

test('render: the header carries the profile, node and the body sha256; the body has no comments', () => {
  const r = render('best', 'spark2');
  const header = parseEnvHeader(r.text.split('\n'));
  assert.equal(header.profile, 'best');
  assert.equal(header.node, 'spark2');
  assert.equal(header.sha256, r.sha256);
  assert.equal(header.rendered, '2026-09-23T10:15:00Z');
  assert.ok(r.bodyLines.every((l) => /^[A-Z_][A-Z0-9_]*=".*"$/.test(l)));
  // The same sha256 as the node computes with grep and sha256sum.
  const dir = tmpdir();
  writeFileSync(path.join(dir, '.env'), r.text + '\n# pulse-carried: x\nHF_TOKEN=secret\n');
  const sha = execFileSync('bash', ['-c', "grep -vE '^(#|$)' .env | grep -vE '^(HF_TOKEN|API_KEY)=' | sha256sum | cut -d' ' -f1"], { cwd: dir, encoding: 'utf8' }).trim();
  assert.equal(sha, r.sha256);
  // Rendering again at another time gives the same body and sha256.
  const again = renderEnv(loadProfile(PROFILES, 'best'), nodes().nodes.spark2, { now: new Date() });
  assert.equal(again.sha256, r.sha256);
});

function tmpProfile(body: string, name = 'x'): string {
  const dir = tmpdir();
  writeFileSync(path.join(dir, `${name}.env`), body);
  return dir;
}

test('render: unknown variable, node key, refused key, unknown directive and duplicate key are errors', () => {
  const n = nodes().nodes.spark2;
  const head = '# pulse-recipe: qwen38-flash\nIMAGE="i"\n';
  assert.throws(() => renderEnv(loadProfile(tmpProfile(head + 'A="{{nope}}"\n'), 'x'), n), /unknown variable \{\{nope\}\}/);
  assert.throws(() => loadProfile(tmpProfile(head + 'HF_HOME=/x\n'), 'x'), /node key/);
  assert.throws(() => loadProfile(tmpProfile(head + 'PORT=1\n'), 'x'), /node key/);
  assert.throws(() => loadProfile(tmpProfile(head + 'HF_TOKEN=abc\n'), 'x'), /not allowed/);
  assert.throws(() => loadProfile(tmpProfile(head + 'TP1_CONTAINER_NAME=c\n'), 'x'), /not allowed/);
  assert.throws(() => loadProfile(tmpProfile(head + '# pulse-mkdri: /x\n'), 'x'), /unknown directive/);
  assert.throws(() => loadProfile(tmpProfile(head + 'IMAGE="j"\n'), 'x'), /set twice/);
  assert.throws(() => loadProfile(tmpProfile('IMAGE="i"\n'), 'x'), /pulse-recipe/);
  assert.throws(() => renderEnv(loadProfile(tmpProfile('# pulse-recipe: qwen38-dual\nA=1\n'), 'x'), n), /no recipe qwen38-dual/);
  assert.throws(() => loadProfile(PROFILES, '../best'), ProfileError);
});

test('overlay: docker-args.txt adds pairs, a clash needs --overlay-wins, a missing file is an error', () => {
  const dir = tmpdir();
  writeFileSync(path.join(dir, 'docker-args.txt'), [
    '# built by overlays/qwen38/build.sh',
    '-v /ov/speculative.py:/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py:ro',
    '-e VLLM_OVERLAY=1',
    '',
  ].join('\n'));
  writeFileSync(path.join(dir, 'manifest.json'), JSON.stringify({ overlays: ['block-drop'] }));
  const ov = readOverlay(dir);
  assert.deepEqual(ov.manifest, { overlays: ['block-drop'] });
  assert.equal(ov.words.length, 4);
  const best = loadProfile(PROFILES, 'best');
  const spark2 = nodes().nodes.spark2;
  assert.throws(() => renderEnv(best, spark2, { overlay: ov }), /duplicate mount destination .*speculative\.py.*--overlay-wins/);
  const r = renderEnv(best, spark2, { overlay: ov, overlayWins: true });
  assert.match(r.map.EXTRA_DOCKER_ARGS, /-v \/ov\/speculative\.py:/);
  assert.match(r.map.EXTRA_DOCKER_ARGS, /-e VLLM_OVERLAY=1/);
  assert.equal(r.replaced.length, 1);
  assert.match(r.text, /# pulse-overlay: .* manifest-sha256=/);
  const d = renderEnv(loadProfile(PROFILES, 'datagen'), spark2, { overlay: ov });
  assert.match(d.map.EXTRA_DOCKER_ARGS, /speculative\.py/);
  assert.throws(() => readOverlay(tmpdir()), /docker-args\.txt is missing/);
});

test('diffMaps: EXTRA_DOCKER_ARGS compares as pairs without order', () => {
  assert.deepEqual(diffMaps({ EXTRA_DOCKER_ARGS: '-e A=1 -v /a:/b' }, { EXTRA_DOCKER_ARGS: '-v /a:/b -e A=1' }), []);
  assert.deepEqual(diffMaps({ A: '1', B: '2' }, { A: '1', C: '3' }), ['- B=2', '+ C=3']);
  assert.deepEqual(diffMaps({ YARN: '0' }, { YARN: '1' }), ['~ YARN: 0 -> 1']);
});

// ---------------------------------------------------------------- plans

const planOpts = { ts: '20260923T101500Z', stateDir: '/tmp/state' };

test('plan: up has the steps in order, and only preflight, wait-mem, wait-ready and verify read only', () => {
  const plan = planUp(render('best', 'spark2'), nodes(), planOpts);
  assert.deepEqual(plan.steps.map((s) => s.id), ['preflight', 'mkdir', 'write-env', 'stop', 'wait-mem', 'start', 'wait-ready', 'verify', 'state']);
  assert.deepEqual(plan.steps.filter((s) => !s.mutates).map((s) => s.id), ['preflight', 'wait-mem', 'wait-ready', 'verify']);
  assert.deepEqual(plan.refusals, []);
  assert.equal(plan.backupPath, '/models/usman/qwen38-flash/.env.pulse-bak-20260923T101500Z');
  const dg = planUp(render('datagen', 'spark2'), nodes(), planOpts);
  assert.ok(!dg.steps.some((s) => s.id === 'mkdir'), 'datagen has no directories to create');
});

test('plan: spark1 is refused without --yes, tp2 is refused always, experimental needs --experimental', () => {
  const n = nodes();
  const p1 = planUp(render('best', 'spark1'), n, planOpts);
  assert.equal(p1.refusals.length, 1);
  assert.match(p1.refusals[0], /spark1 is protected/);
  assert.deepEqual(planUp(render('best', 'spark1'), n, { ...planOpts, yes: true }).refusals, []);
  const tp2 = planUp(render('tp2', 'spark1'), n, { ...planOpts, yes: true, experimental: true });
  assert.equal(tp2.steps.length, 0);
  assert.match(tp2.refusals.join('\n'), /render-only/);
  assert.match(refusals(n.nodes.spark2, loadProfile(PROFILES, 'tp2'), {}).join('\n'), /experimental/);
  assert.match(planDown(n.nodes.spark1, 'qwen38-flash', planOpts).refusals.join('\n'), /protected/);
});

test('plan: preflight reads only; no step talks to the server before the start; no --no-launch', () => {
  const plan = planUp(render('best', 'spark2'), nodes(), planOpts);
  const pre = plan.steps[0].script!;
  for (const bad of ['curl', ':8888/', 'docker run', 'docker rm', 'docker stop', 'docker pull', 'bash -c \'./', 'mkdir -p', 'cp ', 'mv ', 'rm ', 'sysctl', 'nvidia-smi', 'setsid']) {
    assert.ok(!pre.includes(bad), `preflight contains ${bad}`);
  }
  // The start and stop scripts appear only in `[ -x ... ]` tests.
  for (const line of pre.split('\n').filter((l) => /\.\/(start|stop)\.sh/.test(l))) assert.match(line, /^if \[ -x '\.\/(start|stop)\.sh' \]/);
  const all = plan.steps.map((s) => s.script ?? '').join('\n');
  assert.ok(!all.includes('--no-launch'));
  assert.ok(!all.includes('docker run'));
  assert.ok(!all.includes('pkill'));
  const beforeStart = plan.steps.slice(0, plan.steps.findIndex((s) => s.id === 'start')).map((s) => s.script ?? '').join('\n');
  assert.ok(!beforeStart.includes('curl'), 'no HTTP request before the start step');
  const start = plan.steps.find((s) => s.id === 'start')!.script!;
  assert.match(start, /setsid env -i HOME="\$HOME"/);
  assert.match(start, /<\/dev\/null >\/dev\/null 2>&1 &/);
  const write = plan.steps.find((s) => s.id === 'write-env')!.script!;
  assert.match(write, /cp -p \.env '\/models\/usman\/qwen38-flash\/\.env\.pulse-bak-20260923T101500Z'/);
  assert.ok(write.indexOf('cp -p .env') < write.indexOf('mv -f "$tmp" .env'));
});

test('plan: every node script is valid bash', () => {
  const n = nodes();
  for (const [profile, node] of [['best', 'spark2'], ['capture', 'spark1'], ['datagen', 'spark2']]) {
    const plan = planUp(render(profile, node), n, { ...planOpts, yes: true });
    for (const s of plan.steps) {
      if (!s.script) continue;
      execFileSync('bash', ['-n'], { input: wrapScript(s.script) });
    }
  }
  execFileSync('bash', ['-n'], { input: wrapScript(planDown(n.nodes.spark2, 'qwen38-flash', planOpts).steps[0].script!) });
});

test('runner: ssh runs `bash -s` with batch mode, and the script reaches bash on stdin', async () => {
  const [cmd, args] = commandFor(nodes().nodes.spark2);
  assert.equal(cmd, 'ssh');
  assert.deepEqual(args, [...SSH_OPTIONS, 'spark2', 'bash -s']);
  assert.ok(SSH_OPTIONS.includes('BatchMode=yes'));
  assert.deepEqual(commandFor(nodes().nodes.spark1), ['bash', ['-s']]);
  // A command in the script that reads stdin must not eat the rest of the script.
  const r = await new ExecRunner(nodes().nodes.spark1).run('cat\necho "@pulse after cat"\necho "@pulse quote $(printf %s \'a b\')"\n');
  assert.equal(r.code, 0);
  assert.deepEqual(r.records, ['after cat', 'quote a b']);
  assert.equal(shq("it's"), `'it'\\''s'`);
});

// ---------------------------------------------------------------- parsers

test('parsers: KV line, container state, /v1/models, env header, smoke reply', () => {
  const kv = parseKvLine('(EngineCore pid=185) INFO 09-23 15:20:11 [kv_cache_utils.py:2258] GPU KV cache size: 667,808 tokens, Maximum concurrency for 262,144 tokens per request: 2.55x');
  assert.deepEqual({ ...kv, line: undefined }, { tokens: 667808, perRequestTokens: 262144, maxConcurrency: 2.55, line: undefined });
  assert.equal(parseKvLine('nothing here'), null);

  const st = parseContainerState('{"Status":"running","Running":true,"Paused":false,"Restarting":false,"OOMKilled":false,"Dead":false,"Pid":1,"ExitCode":0,"Error":"","StartedAt":"2026-09-23T10:48:04.643805093Z","FinishedAt":"0001-01-01T00:00:00Z"}');
  assert.equal(st.running, true);
  assert.equal(st.startedAt, '2026-09-23T10:48:04.643805093Z');
  const ex = parseContainerState('{"Status":"exited","Running":false,"OOMKilled":false,"ExitCode":137,"StartedAt":"2026-09-23T10:48:04Z"}');
  assert.equal(ex.exitCode, 137);
  assert.equal(parseContainerState('absent').exists, false);

  const models = parseModels('{"object":"list","data":[{"id":"qwen3.8-flash-next","object":"model","created":1,"owned_by":"vllm","root":"Mia-AiLab/Qwen3.8-Flash-Next-NVFP4","parent":null,"max_model_len":524288,"permission":[]}]}');
  assert.deepEqual(models, { id: 'qwen3.8-flash-next', maxModelLen: 524288 });
  assert.equal(parseModels(''), null);
  assert.equal(parseModels('{"data":[]}'), null);

  const s = parseSmoke(['smoke-ms 812', 'smoke-body {"choices":[{"message":{"content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2}}']);
  assert.equal(s.ok, true);
  assert.equal(s.content, 'ready');
  assert.equal(s.latencyMs, 812);
  assert.equal(parseSmoke(['smoke-ms 5', 'smoke-body ']).ok, false);
  assert.equal(parseSmoke(['smoke-body {"choices":[{"message":{"content":""},"finish_reason":"length"}]}']).ok, false);
});

test('parsers: status reports drift, unmanaged files and the saved KV line', () => {
  const recs = [
    'state {"Status":"running","Running":true,"StartedAt":"2026-09-23T10:00:00Z"}',
    'env-header # pulse-profile: best',
    'env-header # pulse-sha256: aaa',
    'env-body-sha bbb',
    `env-mtime ${Date.parse('2026-09-23T11:00:00Z') / 1000}`,
    'kv ',
    'mem-available-gib 18.0',
  ];
  const s = parseStatus('spark2', recs, 'GPU KV cache size: 709,422 tokens, Maximum concurrency for 524,288 tokens per request: 1.35x', new Date('2026-09-23T12:30:00Z'));
  assert.equal(s.uptime, '2h 30m');
  assert.equal(s.env.managed, true);
  assert.deepEqual(s.env.drift, ['the .env was edited after pulse rendered it', 'the .env changed after the container started']);
  assert.equal(s.kv?.tokens, 709422);
  assert.equal(s.kvSource, 'state');
  const u = parseStatus('spark2', ['state absent', 'env-body-sha x'], undefined);
  assert.deepEqual(u.env.drift, ['unmanaged: the .env has no pulse header']);
});

// ---------------------------------------------------------------- gateway fragment

function state(node: string, s: Partial<NodeState>): NodeState {
  return { node, state: 'up', updatedAt: NOW.toISOString(), servedModel: 'qwen3.8-flash-next', bind: '127.0.0.1', ...s };
}

test('fragment: loads through the gateway loadConfig; loopback spark2 stays disabled', () => {
  const dir = tmpdir();
  const n = nodes();
  writeNodeState(dir, state('spark2', { profile: 'best' }));
  const { file, endpoints, warnings } = writeFragment(dir, n);
  const config = loadConfig({ PULSE_GATEWAY_CONFIG: file }, []);
  assert.equal(config.models.length, 1);
  assert.equal(config.models[0].id, 'qwen3.8-flash-next');
  assert.equal(config.models[0].profile, 'qwen38');
  assert.deepEqual(config.models[0].endpoints.map((e) => [e.name, e.baseUrl, e.enabled]), [
    ['spark1', 'http://127.0.0.1:8888/v1', true],
    ['spark2', 'http://10.99.0.2:8888/v1', false],
  ]);
  assert.deepEqual(warnings, []);
  assert.match(endpoints[1].reason, /loopback/);
  // The --config form of the gateway reads the same file.
  assert.equal(loadConfig({}, ['--config', file]).models[0].endpoints.length, 2);
});

test('fragment: a node that listens on its link address is enabled; an all-down set keeps the first node on', () => {
  const n = nodes();
  const reach = buildFragment(n, { spark1: state('spark1', {}), spark2: state('spark2', { bind: '10.99.0.2' }) });
  assert.deepEqual(reach.fragment.models![0].endpoints.map((e) => e.enabled), [true, true]);
  const down = buildFragment(n, { spark1: state('spark1', { state: 'down' }), spark2: state('spark2', { state: 'failed' }) });
  assert.deepEqual(down.fragment.models![0].endpoints.map((e) => e.enabled), [true, false]);
  assert.match(down.warnings.join('\n'), /stays enabled/);
});

// ---------------------------------------------------------------- CLI

function capture(): { out: string[]; err: string[]; deps: { out: (s: string) => void; err: (s: string) => void } } {
  const out: string[] = [];
  const err: string[] = [];
  return { out, err, deps: { out: (s) => out.push(s), err: (s) => err.push(s) } };
}

test('cli: --dry-run creates no runner and changes no state', async () => {
  const stateDir = tmpdir();
  let made = 0;
  const rec = new RecordingRunner();
  const c = capture();
  const code = await main(['up', 'best', '--node', 'spark2', '--dry-run'], {
    ...c.deps, env: { PULSE_STATE_DIR: stateDir }, now: () => NOW, runnerFor: () => { made++; return rec; },
  });
  assert.equal(code, 0);
  assert.equal(made, 0);
  assert.equal(rec.calls.length, 0);
  assert.deepEqual(readdirSync(stateDir), []);
  const text = c.out.join('\n');
  assert.match(text, /rendered \.env for spark2:\/models\/usman\/qwen38-flash\/\.env/);
  assert.match(text, /1\. \[preflight\]/);
  assert.match(text, /9\. \[state\]/);
  assert.match(text, /dry run: nothing ran and nothing changed/);
  // The down dry run does not run anything either.
  assert.equal(await main(['down', '--node', 'spark2', '--dry-run'], { ...c.deps, env: { PULSE_STATE_DIR: stateDir }, runnerFor: () => { made++; return rec; } }), 0);
  assert.equal(made, 0);
});

test('cli: --dry-run --preflight runs exactly the read-only preflight script', async () => {
  const rec = new RecordingRunner(() => ({
    code: 0,
    stdout: [
      '@pulse ok recipe-dir /models/usman/qwen38-flash',
      '@pulse container running 2026-09-23T10:48:04Z',
      ...Object.entries(SPARK2_DATAGEN).map(([k, v]) => `@pulse env-var ${k}=${v}`),
      '@pulse env-body-sha 0000',
      '@pulse clients 8',
      '@pulse mem-available-gib 18.0',
      '@pulse preflight-fails 0',
    ].join('\n'),
  }));
  const c = capture();
  const code = await main(['up', 'datagen', '--node', 'spark2', '--dry-run', '--preflight'], { ...c.deps, env: { PULSE_STATE_DIR: tmpdir() }, now: () => NOW, runnerFor: () => rec });
  assert.equal(code, 0);
  assert.equal(rec.calls.length, 1);
  assert.match(rec.calls[0].script, /@pulse preflight-fails/);
  const text = c.out.join('\n');
  assert.match(text, /change none: the effective key map of the current \.env equals the render \(25 keys\)/);
  assert.match(text, /8 open client connection/);

  const failing = new RecordingRunner(() => ({ code: 1, stdout: '@pulse fail mount-source /models/usman/distill/shard34_r2.safetensors is missing (profile:best)\n@pulse preflight-fails 1\n' }));
  const c2 = capture();
  assert.equal(await main(['up', 'best', '--node', 'spark2', '--dry-run', '--preflight'], { ...c2.deps, env: { PULSE_STATE_DIR: tmpdir() }, runnerFor: () => failing }), 1);
  assert.match(c2.out.join('\n'), /FAIL   mount-source \/models\/usman\/distill\/shard34_r2\.safetensors/);
});

test('cli: a real up on spark1 without --yes and on tp2 runs nothing', async () => {
  let made = 0;
  const c = capture();
  const deps = { ...c.deps, env: { PULSE_STATE_DIR: tmpdir() }, runnerFor: () => { made++; return new RecordingRunner(); } };
  assert.equal(await main(['up', 'best', '--node', 'spark1'], deps), 2);
  assert.equal(await main(['down', '--node', 'spark1'], deps), 2);
  assert.equal(await main(['up', 'tp2', '--node', 'spark1', '--yes', '--experimental'], deps), 2);
  assert.equal(made, 0);
  assert.match(c.err.join('\n'), /protected/);
  assert.match(c.err.join('\n'), /render-only/);
  assert.equal(await main(['up', 'best', '--node', 'spark1', '--preflight'], deps), 2);
  assert.equal(await main(['up', 'nope', '--node', 'spark2', '--dry-run'], deps), 2);
  assert.equal(await main(['up', 'best', '--node', 'spark9', '--dry-run'], deps), 2);
  assert.equal(made, 0);
});

// ---------------------------------------------------------------- end to end with a fake recipe

/** A fake node: a recipe dir with start.sh and stop.sh, and stub docker, curl and ss commands. */
function fakeNode(): { root: string; nodesFile: string; env: NodeJS.ProcessEnv } {
  const root = tmpdir('pulse-fake-');
  const bin = path.join(root, 'bin');
  const recipe = path.join(root, 'recipe');
  const st = path.join(root, 'state');
  for (const d of [bin, recipe, st, path.join(recipe, 'files', 'chat-template'), path.join(root, 'hf', 'hub'), path.join(root, 'ple')]) mkdirSync(d, { recursive: true });
  writeFileSync(path.join(recipe, 'files', 'chat-template', 'chat_template.jinja'), 'x');
  writeFileSync(path.join(recipe, 'files', 'draft_vocab_en_code_47k.txt'), '1\n');
  const script = (file: string, body: string) => { writeFileSync(file, `#!/usr/bin/env bash\n${body}\n`); chmodSync(file, 0o755); };
  const S = st;
  // start.sh and stop.sh run under env -i, so they find the state dir from their own path.
  script(path.join(recipe, 'start.sh'), `S="$(dirname "$0")/../state"; echo "fake start, HF_TOKEN=\${HF_TOKEN:-unset}"; date -u +%Y-%m-%dT%H:%M:%S.%NZ > "$S/started"; touch "$S/running"; exit 0`);
  script(path.join(recipe, 'stop.sh'), `S="$(dirname "$0")/../state"; rm -f "$S/running"; echo "fake stop"`);
  script(path.join(bin, 'docker'), `S=${shq(S)}
case "$1 $2" in
  "image inspect") echo '["sha256:aaa","sha256:bbb"]' ;;
  "ps --format") [ -f "$S/running" ] && echo "vllm-fn-tp1 vllm/vllm-openai:qwen38-flash-next"; echo "open-webui ghcr.io/open-webui/open-webui:main" ;;
  "ps -aq") [ -f "$S/running" ] && echo 0123abcd ;;
  "logs "*) printf '%s\\n' "INFO GPU KV cache size: 667,808 tokens, Maximum concurrency for 262,144 tokens per request: 2.55x" ;;
  "inspect -f")
    [ -f "$S/running" ] || exit 1
    case "$3" in
      *json*) echo "{\\"Status\\":\\"running\\",\\"Running\\":true,\\"StartedAt\\":\\"$(cat "$S/started")\\"}" ;;
      *StartedAt*) echo "running $(cat "$S/started")" ;;
      *Config.Image*) echo vllm/vllm-openai:qwen38-flash-next ;;
      '{{.Image}}') echo sha256:img ;;
      *) echo running ;;
    esac ;;
  "inspect "*) [ -f "$S/running" ] ;;
  *) exit 1 ;;
esac`);
  script(path.join(bin, 'curl'), `S=${shq(S)}
[ -f "$S/running" ] || { case "$*" in *http_code*) printf 000 ;; esac; exit 7; }
case "$*" in
  *http_code*) printf 200 ;;
  *chat/completions*) echo '{"choices":[{"message":{"content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}' ;;
  *) echo '{"object":"list","data":[{"id":"qwen3.8-flash-next","max_model_len":262144}]}' ;;
esac`);
  script(path.join(bin, 'ss'), 'exit 0');
  script(path.join(bin, 'nvidia-smi'), 'exit 0');
  writeFileSync(path.join(recipe, '.env'), 'IMAGE="old"\nMAX_NUM_SEQS=4\nHF_TOKEN=hf_secret_value\n');
  const nodesFile = path.join(root, 'nodes.json');
  writeFileSync(nodesFile, JSON.stringify({
    minMemAvailableGiB: 1,
    memTimeoutS: 30,
    readyGraceS: 0,
    nodes: {
      fake: {
        host: 'local',
        env: { HF_HOME: path.join(root, 'hf'), BIND: '127.0.0.1', PORT: '18999', REQUIRE_IDLE_GPU: 'false' },
        vars: {},
        gatewayUrl: 'http://127.0.0.1:18999/v1',
        recipes: { 'qwen38-flash': { dir: recipe, start: './start.sh', stop: './stop.sh', container: 'vllm-fn-tp1', dockerArgs: ['-v', `${path.join(root, 'ple')}:/ple`] } },
      },
    },
  }));
  // HF_TOKEN in the caller environment must not reach start.sh (env -i).
  return { root, nodesFile, env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, HF_TOKEN: 'caller_token' } };
}

test('e2e: up datagen, status, smoke and down on a fake local recipe', async () => {
  const fake = fakeNode();
  const stateDir = path.join(fake.root, 'pulse-state');
  const sink = new Writable({ write(_c, _e, cb) { cb(); } });
  const c = capture();
  const deps = {
    ...c.deps,
    env: { PULSE_STATE_DIR: stateDir },
    now: () => NOW,
    pollS: 1,
    retryDelayMs: 0,
    runnerFor: (node: NodeConfig) => new ExecRunner(node, sink, fake.env),
  };
  const recipe = path.join(fake.root, 'recipe');

  const up = await main(['up', 'datagen', '--node', 'fake', '--nodes', fake.nodesFile, '--timeout-s', '60'], deps);
  assert.equal(up, 0, c.err.join('\n') + '\n' + c.out.join('\n'));
  const text = c.out.join('\n');
  assert.match(text, /fake is up: profile datagen, qwen3\.8-flash-next, max_model_len 262144/);
  assert.match(text, /KV cache: 667,808 tokens/);

  // The new .env: the rendered text, then the carried secret line. The secret never reached the output.
  const env = readFileSync(path.join(recipe, '.env'), 'utf8');
  assert.match(env, /^# Rendered by `pulse model up`/);
  assert.match(env, /# pulse-profile: datagen/);
  assert.match(env, /\nHF_TOKEN=hf_secret_value\n$/);
  assert.ok(!c.out.join('\n').includes('hf_secret_value'));
  assert.ok(!c.err.join('\n').includes('hf_secret_value'));
  const backups = readdirSync(recipe).filter((f) => f.startsWith('.env.pulse-bak-'));
  assert.deepEqual(backups, ['.env.pulse-bak-20260923T101500Z']);
  assert.equal(readFileSync(path.join(recipe, backups[0]), 'utf8'), 'IMAGE="old"\nMAX_NUM_SEQS=4\nHF_TOKEN=hf_secret_value\n');
  // start.sh ran in an empty environment plus the .env: it did not see the HF_TOKEN of the caller.
  const log = readdirSync(path.join(recipe, 'logs')).find((f) => f.endsWith('.log'))!;
  assert.match(readFileSync(path.join(recipe, 'logs', log), 'utf8'), /fake start, HF_TOKEN=unset/);

  const st = JSON.parse(readFileSync(path.join(stateDir, 'nodes', 'fake.json'), 'utf8')) as NodeState;
  assert.equal(st.state, 'up');
  assert.equal(st.profile, 'datagen');
  assert.match(st.kv!, /667,808/);
  const frag = JSON.parse(readFileSync(path.join(stateDir, 'gateway-backends.json'), 'utf8'));
  assert.deepEqual(frag.models[0].endpoints, [{ name: 'fake', baseUrl: 'http://127.0.0.1:18999/v1', enabled: true }]);

  // A second up with the same profile finds the server up with this .env: it only verifies.
  const logs = () => readdirSync(path.join(recipe, 'logs')).filter((f) => f.endsWith('.log')).length;
  c.out.length = 0;
  assert.equal(await main(['up', 'datagen', '--node', 'fake', '--nodes', fake.nodesFile, '--timeout-s', '60'], { ...deps, now: () => new Date('2026-09-23T11:00:00Z') }), 0);
  assert.match(c.out.join('\n'), /already runs profile datagen with this \.env/);
  assert.equal(logs(), 1, 'no second start');
  // --restart restarts it, and the .env stays as it is.
  c.out.length = 0;
  assert.equal(await main(['up', 'datagen', '--node', 'fake', '--nodes', fake.nodesFile, '--timeout-s', '60', '--restart'], { ...deps, now: () => new Date('2026-09-23T11:05:00Z') }), 0);
  assert.match(c.out.join('\n'), /\.env unchanged/);
  assert.equal(logs(), 2);
  assert.equal(readdirSync(recipe).filter((f) => f.startsWith('.env.pulse-bak-')).length, 1);

  c.out.length = 0;
  assert.equal(await main(['status', '--node', 'fake', '--nodes', fake.nodesFile, '--json'], deps), 0);
  const status = JSON.parse(c.out.join('\n'));
  assert.equal(status[0].container.running, true);
  assert.equal(status[0].env.header.profile, 'datagen');
  assert.equal(status[0].models.maxModelLen, 262144);
  assert.equal(status[0].health, '200');

  c.out.length = 0;
  assert.equal(await main(['smoke', '--node', 'fake', '--nodes', fake.nodesFile], deps), 0);
  assert.match(c.out.join('\n'), /content {8}"ready"/);

  c.out.length = 0;
  assert.equal(await main(['down', '--node', 'fake', '--nodes', fake.nodesFile], deps), 0);
  assert.ok(!existsSync(path.join(fake.root, 'state', 'running')));
  const down = JSON.parse(readFileSync(path.join(stateDir, 'nodes', 'fake.json'), 'utf8')) as NodeState;
  assert.equal(down.state, 'down');
  // The only node is down, so the fragment keeps it enabled and warns.
  assert.match(c.out.join('\n'), /stays enabled/);
});

test('e2e: a failed preflight changes nothing on the node', async () => {
  const fake = fakeNode();
  const stateDir = path.join(fake.root, 'pulse-state');
  const c = capture();
  const cfg = JSON.parse(readFileSync(fake.nodesFile, 'utf8'));
  cfg.nodes.fake.recipes['qwen38-flash'].dockerArgs = ['-v', '/does/not/exist:/x'];
  writeFileSync(fake.nodesFile, JSON.stringify(cfg));
  const sink = new Writable({ write(_c, _e, cb) { cb(); } });
  const code = await main(['up', 'datagen', '--node', 'fake', '--nodes', fake.nodesFile], {
    ...c.deps, env: { PULSE_STATE_DIR: stateDir }, now: () => NOW, runnerFor: (node) => new ExecRunner(node, sink, fake.env),
  });
  assert.equal(code, 1);
  assert.match(c.out.join('\n'), /FAIL   mount-source \/does\/not\/exist is missing/);
  assert.equal(readFileSync(path.join(fake.root, 'recipe', '.env'), 'utf8'), 'IMAGE="old"\nMAX_NUM_SEQS=4\nHF_TOKEN=hf_secret_value\n');
  assert.ok(!existsSync(stateDir));
});
