// `pulse model ...`: bring a measured vLLM profile up or down on a node.
//
//   pulse model profiles [--json]
//   pulse model up <profile> --node <name> [--overlay <dir>] [--dry-run [--preflight]]
//   pulse model down --node <name> [--dry-run]
//   pulse model status [--node <name>] [--json] [--no-http] [--gpu-probe] [--yes]
//   pulse model smoke --node <name> [--yes]
//   pulse model fragment
//
// See docs/RUNTIME.md.

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  ProfileError, diffMaps, displayEnv, getNode, listProfiles, loadNodes, loadProfile, readOverlay, renderEnv,
} from './profiles.js';
import type { NodeConfig, Rendered } from './profiles.js';
import { planDown, planUp, timestamp } from './plan.js';
import type { Plan, Step } from './plan.js';
import { ExecRunner } from './runner.js';
import type { RunResult, Runner } from './runner.js';
import { first, parseEnvHeader, parseKvLine, parseModels, parseSmoke, parseStatus, recordMap, smokeScript, statusScript } from './status.js';
import type { NodeStatus } from './status.js';
import { fragmentHelp, readNodeState, stateDir, writeFragment, writeNodeState } from './fragment.js';
import type { NodeState } from './fragment.js';
import { DockerArgsError } from './dockerargs.js';
import { EnvFileError } from './envfile.js';

export interface CliDeps {
  out?: (s: string) => void;
  err?: (s: string) => void;
  env?: NodeJS.ProcessEnv;
  now?: () => Date;
  runnerFor?: (node: NodeConfig) => Runner;
  repoRoot?: string;
  /** Pause between retries, in ms. Tests set 0. */
  retryDelayMs?: number;
  /** Seconds between readiness polls. Tests set 1. */
  pollS?: number;
}

const HELP = `pulse model - bring a measured vLLM profile up or down on a node

  pulse model profiles              list the profiles in runtime/qwen38/profiles
  pulse model up <profile> --node N render the recipe .env, stop the server, wait for
                                    memory, start the recipe, wait for /v1/models 200
    --overlay DIR     add DIR/docker-args.txt (overlays/qwen38/build.sh --out DIR)
    --overlay-wins    let overlay mounts replace profile mounts on the same path
    --dry-run         print the rendered .env and every step; run nothing
    --preflight       with --dry-run: also run the read-only checks on the node
    --yes             allow a change on a protected node (spark1)
    --experimental    allow a profile marked experimental
    --timeout-s N     readiness timeout (default READY_TIMEOUT_S 1800 + 300)
    --min-mem-gib N   MemAvailable gate before the start (default: runtime/nodes.json)
    --restart         restart even when the node already runs this exact .env
  pulse model down --node N         stop the server on the node
  pulse model status [--node N]     container, profile, drift, health, KV, memory
    --json --no-http --gpu-probe (starts a GPU container; off by default)
    on a protected node, status sends no HTTP request without --yes, and
    --gpu-probe needs --yes
  pulse model smoke --node N        send one short chat completion on the node
                                    (a protected node needs --yes)
  pulse model fragment              rewrite the gateway backend fragment from the state

  --verbose shows the raw @pulse records. --nodes FILE and --profiles-dir DIR
  replace runtime/nodes.json and runtime/qwen38/profiles.`;

interface Ctx {
  out: (s: string) => void;
  err: (s: string) => void;
  env: NodeJS.ProcessEnv;
  now: () => Date;
  runnerFor: (node: NodeConfig) => Runner;
  root: string;
  nodesFile: string;
  profilesDir: string;
  stateDir: string;
  retryDelayMs: number;
  pollS?: number;
  opts: Opts;
}

interface Opts {
  node?: string;
  overlay?: string;
  'overlay-wins'?: boolean;
  'dry-run'?: boolean;
  preflight?: boolean;
  yes?: boolean;
  experimental?: boolean;
  json?: boolean;
  'no-http'?: boolean;
  'gpu-probe'?: boolean;
  verbose?: boolean;
  'timeout-s'?: string;
  'min-mem-gib'?: string;
  restart?: boolean;
  nodes?: string;
  'profiles-dir'?: string;
  help?: boolean;
}

function defaultRoot(): string {
  // dist/runtime/cli.js -> repo root
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
}

export async function main(argv: string[], deps: CliDeps = {}): Promise<number> {
  const out = deps.out ?? ((s: string) => process.stdout.write(s + '\n'));
  const err = deps.err ?? ((s: string) => process.stderr.write(s + '\n'));
  let parsed;
  try {
    parsed = parseArgs({
      args: argv,
      allowPositionals: true,
      strict: true,
      options: {
        node: { type: 'string' },
        overlay: { type: 'string' },
        'overlay-wins': { type: 'boolean' },
        'dry-run': { type: 'boolean' },
        preflight: { type: 'boolean' },
        yes: { type: 'boolean' },
        experimental: { type: 'boolean' },
        json: { type: 'boolean' },
        'no-http': { type: 'boolean' },
        'gpu-probe': { type: 'boolean' },
        verbose: { type: 'boolean' },
        'timeout-s': { type: 'string' },
        'min-mem-gib': { type: 'string' },
        restart: { type: 'boolean' },
        nodes: { type: 'string' },
        'profiles-dir': { type: 'string' },
        help: { type: 'boolean', short: 'h' },
      },
    });
  } catch (e) {
    err(`pulse model: ${(e as Error).message}`);
    return 2;
  }
  const opts = parsed.values as Opts;
  const [sub, ...rest] = parsed.positionals;
  if (!sub || opts.help || sub === 'help') {
    out(HELP);
    return sub || opts.help ? 0 : 2;
  }
  const env = deps.env ?? process.env;
  const root = deps.repoRoot ?? defaultRoot();
  const ctx: Ctx = {
    out,
    err,
    env,
    now: deps.now ?? (() => new Date()),
    runnerFor: deps.runnerFor ?? ((node) => new ExecRunner(node)),
    root,
    nodesFile: path.resolve(opts.nodes ?? env.PULSE_NODES_FILE ?? path.join(root, 'runtime', 'nodes.json')),
    profilesDir: path.resolve(opts['profiles-dir'] ?? env.PULSE_PROFILES_DIR ?? path.join(root, 'runtime', 'qwen38', 'profiles')),
    stateDir: stateDir(env),
    retryDelayMs: deps.retryDelayMs ?? 10_000,
    pollS: deps.pollS,
    opts,
  };
  if (opts.preflight && !opts['dry-run']) {
    err('pulse model: --preflight works only with --dry-run (a real up always runs the preflight)');
    return 2;
  }
  try {
    switch (sub) {
      case 'profiles': return cmdProfiles(ctx);
      case 'up': return await cmdUp(ctx, rest[0]);
      case 'down': return await cmdDown(ctx);
      case 'status': return await cmdStatus(ctx);
      case 'smoke': return await cmdSmoke(ctx);
      case 'fragment': return cmdFragment(ctx);
      default:
        err(`pulse model: unknown command ${sub}\n`);
        out(HELP);
        return 2;
    }
  } catch (e) {
    if (e instanceof ProfileError || e instanceof DockerArgsError || e instanceof EnvFileError) {
      err(`pulse model: ${e.message}`);
      return 2;
    }
    throw e;
  }
}

function cmdProfiles(ctx: Ctx): number {
  const names = listProfiles(ctx.profilesDir);
  const profiles = names.map((n) => loadProfile(ctx.profilesDir, n));
  const nodes = loadNodes(ctx.nodesFile);
  if (ctx.opts.json) {
    ctx.out(JSON.stringify({
      profiles: profiles.map((p) => ({ name: p.name, recipe: p.recipe, status: p.status, runnable: p.runnable, description: p.description, file: p.file })),
      nodes: Object.values(nodes.nodes).map((n) => ({ name: n.name, host: n.host, ssh: n.ssh, protected: n.protected, recipes: Object.keys(n.recipes) })),
    }, null, 2));
    return 0;
  }
  ctx.out(`profiles in ${path.relative(process.cwd(), ctx.profilesDir) || ctx.profilesDir}:`);
  const w = Math.max(...profiles.map((p) => p.name.length), 4);
  for (const p of profiles) {
    const tag = p.runnable ? p.status : `${p.status}, render only`;
    ctx.out(`  ${p.name.padEnd(w)}  ${p.recipe.padEnd(13)} ${tag.padEnd(26)} ${p.description}`);
  }
  ctx.out('\nnodes:');
  for (const n of Object.values(nodes.nodes)) {
    const how = n.host === 'ssh' ? `ssh ${n.ssh}` : 'local';
    ctx.out(`  ${n.name.padEnd(w)}  ${how}${n.protected ? ', protected (changes need --yes)' : ''}  recipes: ${Object.keys(n.recipes).join(', ')}`);
  }
  return 0;
}

function indent(text: string, pad = '    '): string {
  return text.replace(/\n$/, '').split('\n').map((l) => pad + l).join('\n');
}

function printPlan(ctx: Ctx, plan: Plan): void {
  ctx.out('plan:');
  plan.steps.forEach((s, i) => {
    ctx.out(`  ${i + 1}. [${s.id}] ${s.title}${s.mutates ? '' : ' (read-only)'}`);
    if (s.script) ctx.out(indent(s.script, '       | '));
    if (s.detail) ctx.out(`       ${s.detail}`);
  });
}

async function runStep(ctx: Ctx, runner: Runner, node: NodeConfig, step: Step, stream = true): Promise<RunResult> {
  if (!step.script) throw new Error(`step ${step.id} has no script`);
  for (let attempt = 1; ; attempt++) {
    const r = await runner.run(step.script, { stream, verbose: ctx.opts.verbose, timeoutS: step.timeoutS });
    // ssh exits 255 when the connection fails. Only read-only steps are safe to repeat.
    if (r.code === 255 && node.host === 'ssh' && !step.mutates && attempt < 3) {
      ctx.err(`  ssh to ${node.ssh} failed (exit 255); attempt ${attempt + 1} of 3`);
      if (r.stderr.trim()) ctx.err(indent(r.stderr.trim()));
      await new Promise((res) => setTimeout(res, ctx.retryDelayMs));
      continue;
    }
    return r;
  }
}

function failLines(r: RunResult): string[] {
  return (recordMap(r.records).get('fail') ?? []);
}

interface PreflightReport {
  fails: string[];
  currentProfile?: string;
  unchanged: boolean;
  /**
   * The recipe container runs, and it started after the last change of the
   * .env and after the last change of each mounted file.
   */
  runningCurrent: boolean;
}

function reportPreflight(ctx: Ctx, rendered: Rendered, r: RunResult, plan: Plan): PreflightReport {
  const m = recordMap(r.records);
  const node = rendered.node;
  ctx.out(`preflight on ${node.name}:`);
  for (const ok of m.get('ok') ?? []) ctx.out(`  ok     ${ok}`);
  for (const d of m.get('plan-mkdir') ?? []) ctx.out(`  mkdir  ${d} does not exist; the mkdir step creates it`);
  const fails = failLines(r);
  for (const f of fails) ctx.out(`  FAIL   ${f}`);
  if (r.code !== 0 && !fails.length) {
    ctx.out(`  FAIL   the preflight script exited with ${r.code}`);
    if (r.stderr.trim()) ctx.out(indent(r.stderr.trim()));
    fails.push(`exit ${r.code}`);
  }
  // Hints for missing mount sources.
  for (const f of fails) {
    const mm = /^mount-source (\S+) is missing \((overlay:[^)]*|profile:[^)]*|node:[^)]*)\)/.exec(f);
    if (!mm) continue;
    if (mm[2].startsWith('overlay') && node.host === 'ssh') {
      ctx.out(`         copy it to the node: rsync -a ${mm[1]} ${node.ssh}:${mm[1]}`);
    } else if (!mm[2].startsWith('overlay')) {
      ctx.out(`         this path must exist on ${node.name} (${mm[2]}); pulse does not copy it`);
    }
  }
  const image = first(m, 'image');
  if (image) ctx.out(`  image  ${image}`);
  const container = first(m, 'container') ?? 'unknown';
  ctx.out(`  server ${rendered.recipe.container}: ${container}`);
  const [cstate, cstarted] = container.split(' ');
  const started = cstarted ? Date.parse(cstarted) / 1000 : NaN;
  const running = cstate === 'running' && Number.isFinite(started);
  const envMtime = Number(first(m, 'env-mtime') ?? 'NaN');
  const mountsCtime = Number(first(m, 'mounts-ctime') ?? 'NaN');
  const runningSinceEnv = running && Number.isFinite(envMtime) && started > envMtime;
  const runningSinceMounts = running && Number.isFinite(mountsCtime) && started > mountsCtime;
  const header = parseEnvHeader((m.get('env-header') ?? []).map((l) => l));
  const bodySha = first(m, 'env-body-sha');
  let unchanged = false;
  if (m.has('env-missing')) {
    ctx.out(`  .env   missing in ${rendered.recipe.dir}; the write step creates it`);
  } else {
    if (!header.sha256) ctx.out('  .env   unmanaged (no pulse header)');
    else ctx.out(`  .env   profile ${header.profile ?? '?'}, rendered ${header.rendered ?? '?'}${bodySha === header.sha256 ? '' : ', EDITED after the render'}`);
    unchanged = !!header.sha256 && header.sha256 === rendered.sha256 && bodySha === rendered.sha256
      && header.profile === rendered.profile.name && (header.overlay ?? 'none') === rendered.overlayLine;
    const current: Record<string, string> = {};
    const secrets: string[] = [];
    for (const kv of m.get('env-var') ?? []) {
      const eq = kv.indexOf('=');
      const k = kv.slice(0, eq);
      if (k === 'HF_TOKEN' || k === 'API_KEY') secrets.push(k);
      else current[k] = kv.slice(eq + 1);
    }
    const diff = diffMaps(current, rendered.map);
    if (!diff.length) ctx.out(`  change none: the effective key map of the current .env equals the render (${Object.keys(rendered.map).length} keys)`);
    else {
      ctx.out(`  change ${diff.length} difference(s) between the current .env and the render:`);
      for (const d of diff) ctx.out(`           ${d}`);
    }
    if (secrets.length) ctx.out(`  keep   ${secrets.join(', ')} from the current .env (the value is not shown)`);
    if (header.sha256 && (header.overlay ?? 'none') !== rendered.overlayLine) {
      ctx.out(`  overlay the .env has overlay ${header.overlay ?? 'none'}; the render has ${rendered.overlayLine}`);
    }
    ctx.out(unchanged
      ? '  write  not needed: the .env already has this sha256, profile and overlay'
      : `  write  a new .env; the current one is saved as ${plan.backupPath}`);
  }
  if (running && Number.isFinite(mountsCtime) && !runningSinceMounts) {
    ctx.out('  mounts a mounted file changed after the container started; the running server does not see the new file');
  }
  const clients = Number(first(m, 'clients') ?? '0');
  if (clients > 0) ctx.out(`  WARN   ${clients} open client connection(s) on port ${rendered.port}; they drop when the server stops`);
  const mem = first(m, 'mem-available-gib');
  if (mem) ctx.out(`  memory MemAvailable ${mem} GiB now`);
  return { fails, currentProfile: header.profile, unchanged, runningCurrent: runningSinceEnv && runningSinceMounts };
}

function printRenderSummary(ctx: Ctx, r: Rendered, dryRun: boolean): void {
  const n = r.node;
  ctx.out(`pulse model up ${r.profile.name} --node ${n.name}${dryRun ? ' (dry run)' : ''}`);
  ctx.out(`  profile  ${r.profile.name} (${r.profile.status}${r.profile.runnable ? '' : ', render only'}): ${r.profile.description}`);
  ctx.out(`  node     ${n.name} (${n.host === 'ssh' ? `ssh ${n.ssh}` : 'local'}${n.protected ? ', protected' : ''}), recipe ${r.recipeName} in ${r.recipe.dir}`);
  ctx.out(`  serves   ${r.servedModel}, max_model_len ${r.expectedMaxModelLen ?? '?'}, ${r.bind}:${r.port}, container ${r.recipe.container}`);
  if (r.overlay) ctx.out(`  overlay  ${r.overlay.dir} (overlay sha256 ${r.overlay.sha256.slice(0, 16)}${r.overlay.manifestSha256 ? `, manifest sha256 ${r.overlay.manifestSha256.slice(0, 16)}` : ', no manifest.json'})`);
  for (const line of r.replaced) ctx.out(`  replace  ${line}`);
  ctx.out(`  sha256   ${r.sha256}`);
  for (const note of r.profile.clientNotes) ctx.out(`  client   ${note}`);
}

function restoreHint(ctx: Ctx, plan: Plan, backupWritten: string | undefined, previousProfile: string | undefined, stopRan: boolean): void {
  const yes = plan.node.protected ? ' --yes' : '';
  if (!backupWritten) {
    if (!stopRan) return;
    ctx.err(`the stop ran, so ${plan.node.name} may have no server now. The .env did not change.`);
    if (previousProfile) ctx.err(`to go back: pulse model up ${previousProfile} --node ${plan.node.name}${yes}`);
    else ctx.err(`to go back: run ${plan.recipe.start} in ${plan.recipe.dir}`);
    return;
  }
  ctx.err(`the previous .env is saved as ${backupWritten}.`);
  if (previousProfile) ctx.err(`to go back: pulse model up ${previousProfile} --node ${plan.node.name}${yes}`);
  else ctx.err(`to go back: cp -p ${backupWritten} ${plan.recipe.dir}/.env, then run ${plan.recipe.start} in ${plan.recipe.dir}`);
}

/** Check the verify records against the render. Fills the state; returns the problems. */
function applyVerify(ctx: Ctx, rendered: Rendered, state: NodeState, m: Map<string, string[]>): string[] {
  const models = parseModels(first(m, 'models-body') ?? '');
  const problems: string[] = [];
  if (!models) problems.push('/v1/models returned no model');
  else {
    if (models.id !== rendered.servedModel) problems.push(`/v1/models serves ${models.id}, expected ${rendered.servedModel}`);
    if (rendered.expectedMaxModelLen && models.maxModelLen !== rendered.expectedMaxModelLen) {
      problems.push(`max_model_len is ${models.maxModelLen}, expected ${rendered.expectedMaxModelLen}`);
    }
    ctx.out(`  /v1/models: ${models.id}, max_model_len ${models.maxModelLen}`);
  }
  const kv = parseKvLine(first(m, 'kv') ?? '');
  if (kv) {
    state.kv = kv.line;
    ctx.out(`  KV cache: ${kv.tokens.toLocaleString('en-US')} tokens${kv.maxConcurrency ? ` (${kv.maxConcurrency}x at ${kv.perRequestTokens?.toLocaleString('en-US')} tokens per request)` : ''}`);
  } else ctx.out('  WARN no KV cache line in the container log');
  state.proofs = [];
  for (const p of m.get('proof') ?? []) {
    const ok = p.startsWith('ok ');
    const text = p.replace(/^(ok|missing) /, '');
    state.proofs.push({ text, ok });
    ctx.out(`  proof ${ok ? 'ok     ' : 'MISSING'} ${text}`);
  }
  return problems;
}

async function cmdUp(ctx: Ctx, profileName: string | undefined): Promise<number> {
  if (!profileName) {
    ctx.err('usage: pulse model up <profile> --node <name> [--dry-run]');
    return 2;
  }
  const nodes = loadNodes(ctx.nodesFile);
  const node = getNode(nodes, ctx.opts.node);
  const profile = loadProfile(ctx.profilesDir, profileName);
  const overlay = ctx.opts.overlay ? readOverlay(ctx.opts.overlay) : undefined;
  const now = ctx.now();
  const rendered = renderEnv(profile, node, { overlay, overlayWins: ctx.opts['overlay-wins'], now });
  const timeoutS = ctx.opts['timeout-s'] !== undefined ? Number(ctx.opts['timeout-s']) : undefined;
  if (timeoutS !== undefined && !(timeoutS > 0)) throw new ProfileError('--timeout-s must be a positive number');
  const minMemGiB = ctx.opts['min-mem-gib'] !== undefined ? Number(ctx.opts['min-mem-gib']) : undefined;
  if (minMemGiB !== undefined && !(minMemGiB > 0)) throw new ProfileError('--min-mem-gib must be a positive number');
  const plan = planUp(rendered, nodes, { ts: timestamp(now), yes: ctx.opts.yes, experimental: ctx.opts.experimental, timeoutS, minMemGiB, pollS: ctx.pollS, stateDir: ctx.stateDir });
  const dryRun = !!ctx.opts['dry-run'];

  printRenderSummary(ctx, rendered, dryRun);
  if (dryRun) {
    ctx.out(`\nrendered .env for ${node.name}:${rendered.recipe.dir}/.env:`);
    ctx.out(indent(displayEnv(rendered)));
    ctx.out('');
    if (plan.steps.length) printPlan(ctx, plan);
    for (const reason of plan.refusals) ctx.out(`\na real run stops here: ${reason}`);
    if (ctx.opts.preflight && plan.steps.length) {
      ctx.out('');
      const pre = plan.steps[0];
      const r = await runStep(ctx, ctx.runnerFor(node), node, pre, false);
      const report = reportPreflight(ctx, rendered, r, plan);
      ctx.out(report.fails.length ? `\npreflight: ${report.fails.length} failure(s)` : '\npreflight: clean');
      return report.fails.length ? 1 : 0;
    }
    ctx.out('\ndry run: nothing ran and nothing changed.');
    return 0;
  }
  if (plan.refusals.length) {
    for (const reason of plan.refusals) ctx.err(`pulse model up: ${reason}`);
    return 2;
  }

  const runner = ctx.runnerFor(node);
  const [pre, ...steps] = plan.steps;
  ctx.out(`\n[1/${plan.steps.length}] ${pre.title}`);
  const preRun = await runStep(ctx, runner, node, pre, false);
  const report = reportPreflight(ctx, rendered, preRun, plan);
  if (report.fails.length) {
    ctx.err(`\npulse model up: the preflight failed on ${node.name}; nothing changed.`);
    return 1;
  }

  const alreadyUp = report.unchanged && report.runningCurrent;
  if (alreadyUp && !ctx.opts.restart) {
    ctx.out(`\n${node.name} already runs profile ${profile.name} with this .env; checking it (pass --restart to restart it)`);
  }
  const state: NodeState = {
    node: node.name, state: 'starting', updatedAt: now.toISOString(), profile: profile.name, recipe: rendered.recipeName,
    envSha256: rendered.sha256, servedModel: rendered.servedModel, maxModelLen: rendered.expectedMaxModelLen ?? undefined,
    bind: rendered.bind, port: rendered.port, backup: null, replaced: rendered.replaced,
    overlay: overlay ? { dir: overlay.dir, manifestSha256: overlay.manifestSha256, manifest: overlay.manifest } : null,
  };
  let backupWritten: string | undefined;
  // After the stop step starts, the old server can be gone. From then on, a
  // failure marks the node failed, so the state and the fragment do not keep
  // an endpoint that has no server.
  let stopRan = false;
  const failStep = (step: Step, r: RunResult, idx: number): number => {
    ctx.err(`\npulse model up: step ${idx} [${step.id}] failed (exit ${r.code})`);
    for (const f of failLines(r)) ctx.err(`  ${f}`);
    if (r.stderr.trim() && !failLines(r).length) ctx.err(indent(r.stderr.trim()));
    if (stopRan || ['start', 'wait-ready', 'verify'].includes(step.id)) {
      state.state = 'failed';
      state.error = `${step.id}: ${failLines(r)[0] ?? `exit ${r.code}`}`;
      state.updatedAt = ctx.now().toISOString();
      const file = writeNodeState(ctx.stateDir, state);
      const frag = writeFragment(ctx.stateDir, nodes);
      ctx.err(`state: ${file} (failed)`);
      for (const e of frag.endpoints) ctx.err(`  endpoint ${e.name}: ${e.enabled ? 'enabled' : 'disabled'} (${e.reason})`);
    }
    restoreHint(ctx, plan, backupWritten, report.currentProfile, stopRan);
    return 1;
  };

  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    // A node that already runs this .env only gets the read-only checks.
    if (alreadyUp && !ctx.opts.restart && !['wait-ready', 'verify', 'state'].includes(step.id)) continue;
    const idx = i + 2;
    ctx.out(`\n[${idx}/${plan.steps.length}] ${step.title}`);
    if (step.id === 'state') break;
    if (step.id === 'stop') stopRan = true;
    const r = await runStep(ctx, runner, node, step);
    const m = recordMap(r.records);
    // The backup exists also when a later command of the write step failed.
    if (step.id === 'write-env') {
      backupWritten = first(m, 'env-backup');
      state.backup = backupWritten ?? null;
    }
    if (r.code !== 0) return failStep(step, r, idx);
    if (step.id === 'write-env') {
      if (m.has('env') && first(m, 'env') === 'unchanged') ctx.out('  .env unchanged (same sha256, profile and overlay)');
      else ctx.out(`  .env written${backupWritten ? `; backup ${backupWritten}` : ''}${m.has('env-carried') ? `; kept ${first(m, 'env-carried')} secret line(s)` : ''}`);
    } else if (step.id === 'start') {
      state.startLog = first(m, 'start-log');
      state.updatedAt = ctx.now().toISOString();
      writeNodeState(ctx.stateDir, state);
    } else if (step.id === 'wait-ready') {
      state.readySeconds = Number(first(m, 'ready-s'));
    } else if (step.id === 'verify') {
      const problems = applyVerify(ctx, rendered, state, m);
      if (problems.length) {
        return failStep(step, { ...r, code: 1, records: [...r.records, ...problems.map((p) => `fail verify ${p}`)] }, idx);
      }
    }
  }

  state.state = 'up';
  state.updatedAt = ctx.now().toISOString();
  const stateFile = writeNodeState(ctx.stateDir, state);
  const frag = writeFragment(ctx.stateDir, nodes);
  ctx.out(`\n${node.name} is up: profile ${profile.name}, ${rendered.servedModel}, max_model_len ${rendered.expectedMaxModelLen}${state.readySeconds ? `, ready after ${state.readySeconds} s` : ''}`);
  if (state.proofs?.some((p) => !p.ok)) ctx.out('  WARN at least one proof line is missing from the container log');
  printFragment(ctx, stateFile, frag);
  return 0;
}

function printFragment(ctx: Ctx, stateFile: string | undefined, frag: ReturnType<typeof writeFragment>): void {
  if (stateFile) ctx.out(`state: ${stateFile}`);
  for (const e of frag.endpoints) ctx.out(`  endpoint ${e.name}: ${e.enabled ? 'enabled' : 'disabled'} (${e.reason})`);
  for (const w of frag.warnings) ctx.out(`  WARN ${w}`);
  for (const l of fragmentHelp(frag.file)) ctx.out(l);
}

async function cmdDown(ctx: Ctx): Promise<number> {
  const nodes = loadNodes(ctx.nodesFile);
  const node = getNode(nodes, ctx.opts.node);
  const prev = readNodeState(ctx.stateDir, node.name);
  const recipeName = prev?.recipe ?? 'qwen38-flash';
  if (!node.recipes[recipeName]) throw new ProfileError(`node ${node.name} has no recipe ${recipeName}`);
  const now = ctx.now();
  const plan = planDown(node, recipeName, { ts: timestamp(now), yes: ctx.opts.yes, stateDir: ctx.stateDir });
  ctx.out(`pulse model down --node ${node.name}${ctx.opts['dry-run'] ? ' (dry run)' : ''}: recipe ${recipeName} in ${plan.recipe.dir}, container ${plan.recipe.container}`);
  if (ctx.opts['dry-run']) {
    printPlan(ctx, plan);
    for (const reason of plan.refusals) ctx.out(`\na real run stops here: ${reason}`);
    ctx.out('\ndry run: nothing ran and nothing changed.');
    return 0;
  }
  if (plan.refusals.length) {
    for (const reason of plan.refusals) ctx.err(`pulse model down: ${reason}`);
    return 2;
  }
  const runner = ctx.runnerFor(node);
  const stop = await runStep(ctx, runner, node, plan.steps[0]);
  if (stop.code !== 0) {
    ctx.err(`pulse model down: the stop failed (exit ${stop.code})`);
    for (const f of failLines(stop)) ctx.err(`  ${f}`);
    // The stop ran, so the server can be in any state now. Do not keep an "up" record.
    const failed: NodeState = {
      ...(prev ?? { node: node.name, recipe: recipeName }), node: node.name, state: 'failed',
      updatedAt: ctx.now().toISOString(), error: `stop: ${failLines(stop)[0] ?? `exit ${stop.code}`}`,
    } as NodeState;
    writeNodeState(ctx.stateDir, failed);
    writeFragment(ctx.stateDir, nodes);
    return 1;
  }
  const rep = await runStep(ctx, runner, node, plan.steps[1], false);
  const mem = first(recordMap(rep.records), 'mem-available-gib');
  ctx.out(`${plan.recipe.container} is gone on ${node.name}. MemAvailable ${mem ?? '?'} GiB (the memory comes back over the next minute).`);
  const state: NodeState = { ...(prev ?? { node: node.name, recipe: recipeName }), node: node.name, state: 'down', updatedAt: ctx.now().toISOString() } as NodeState;
  delete state.error;
  const stateFile = writeNodeState(ctx.stateDir, state);
  printFragment(ctx, stateFile, writeFragment(ctx.stateDir, nodes));
  return 0;
}

function printStatus(ctx: Ctx, s: NodeStatus, node: NodeConfig, saved: NodeState | undefined): void {
  ctx.out(`${node.name} (${node.host === 'ssh' ? `ssh ${node.ssh}` : 'local'})`);
  if (!s.reachable) {
    ctx.out(`  unreachable: ${s.error}`);
    return;
  }
  const c = s.container;
  ctx.out(`  container   ${c.exists ? `${c.status}${s.uptime ? `, up ${s.uptime}` : ''}${c.startedAt ? ` (started ${c.startedAt})` : ''}` : 'absent'}${c.oomKilled ? ', OOM-killed' : ''}`);
  if (s.image) ctx.out(`  image       ${s.image}${s.layersSha256 ? ` (layers sha256 ${s.layersSha256})` : ''}`);
  const h = s.env.header;
  if (s.env.missing) ctx.out('  .env        missing');
  else if (!s.env.managed) ctx.out('  .env        unmanaged (no pulse header)');
  else ctx.out(`  .env        profile ${h.profile}, rendered ${h.rendered}, sha256 ${h.sha256?.slice(0, 16)}`);
  ctx.out(`  drift       ${s.env.drift.length ? s.env.drift.join('; ') : 'none'}`);
  if (!s.httpSkipped) {
    ctx.out(`  /health     ${s.health || 'no answer'}`);
    ctx.out(`  /v1/models  ${s.models ? `${s.models.id}, max_model_len ${s.models.maxModelLen}` : 'no model'}`);
  } else ctx.out(`  http        skipped (${s.httpSkipped})`);
  if (s.kv) ctx.out(`  kv cache    ${s.kv.tokens.toLocaleString('en-US')} tokens${s.kv.maxConcurrency ? ` (${s.kv.maxConcurrency}x at ${s.kv.perRequestTokens?.toLocaleString('en-US')} per request)` : ''}${s.kvSource === 'state' ? ' (saved at up; the log rotated)' : ''}`);
  if (s.memAvailableGiB !== undefined) ctx.out(`  memory      MemAvailable ${s.memAvailableGiB} of ${s.memTotalGiB} GiB`);
  ctx.out(`  gpu apps    ${s.gpuApps.length ? s.gpuApps.join('; ') : 'none listed'} (for information only: not a reliable signal on spark2)`);
  if (s.vm) ctx.out(`  vm          ${s.vm}`);
  if (s.thp) ctx.out(`  thp         ${s.thp}`);
  if (s.gpuProbe !== undefined) ctx.out(`  gpu probe   ${s.gpuProbe || 'no result'} (fast state about 200-230 GB/s)`);
  ctx.out(`  pulse state ${saved ? `${saved.state} (profile ${saved.profile ?? '?'}, ${saved.updatedAt})` : 'none'}`);
}

async function cmdStatus(ctx: Ctx): Promise<number> {
  const nodes = loadNodes(ctx.nodesFile);
  const list = ctx.opts.node ? [getNode(nodes, ctx.opts.node)] : Object.values(nodes.nodes);
  // The GPU probe starts a GPU container next to the server. On a protected
  // node that needs --yes, also when the node list comes from nodes.json.
  const guarded = list.filter((n) => n.protected).map((n) => n.name);
  if (ctx.opts['gpu-probe'] && guarded.length && !ctx.opts.yes) {
    ctx.err(`pulse model status: --gpu-probe starts a GPU container on ${guarded.join(', ')}, which is protected (it serves live traffic). Pass --yes, or name another node with --node.`);
    return 2;
  }
  const results: NodeStatus[] = [];
  let bad = 0;
  for (const node of list) {
    const saved = readNodeState(ctx.stateDir, node.name);
    const recipeName = saved?.recipe ?? 'qwen38-flash';
    const recipe = node.recipes[recipeName];
    if (!recipe) throw new ProfileError(`node ${node.name} has no recipe ${recipeName}`);
    if (ctx.opts['gpu-probe'] && !node.gpuProbe) ctx.err(`node ${node.name} has no gpuProbe in ${ctx.nodesFile}`);
    // Requests to the server of a protected node can disturb live traffic and benchmarks.
    let httpSkipped: string | undefined;
    if (ctx.opts['no-http']) httpSkipped = '--no-http';
    else if (node.protected && !ctx.opts.yes) httpSkipped = `${node.name} is protected; pass --yes to send requests to its server`;
    const http = !httpSkipped;
    const script = statusScript(node, recipe, { http, gpuProbe: !!ctx.opts['gpu-probe'] });
    const step: Step = { id: 'report', title: 'status', mutates: false, where: 'node', script, timeoutS: ctx.opts['gpu-probe'] ? 300 : 90 };
    const r = await runStep(ctx, ctx.runnerFor(node), node, step, false);
    let s: NodeStatus;
    if (r.code === 255 && node.host === 'ssh') {
      s = { node: node.name, reachable: false, error: r.stderr.trim() || 'ssh failed', container: { exists: false, status: 'unknown', running: false }, env: { managed: false, missing: false, header: {}, drift: [] }, gpuApps: [] };
      bad++;
    } else {
      s = parseStatus(node.name, r.records, saved?.kv, ctx.now());
    }
    if (httpSkipped) s.httpSkipped = httpSkipped;
    results.push(s);
    if (!ctx.opts.json) {
      printStatus(ctx, s, node, saved);
      if (list.length > 1) ctx.out('');
    }
  }
  if (ctx.opts.json) ctx.out(JSON.stringify(results, null, 2));
  return bad ? 1 : 0;
}

async function cmdSmoke(ctx: Ctx): Promise<number> {
  const nodes = loadNodes(ctx.nodesFile);
  const node = getNode(nodes, ctx.opts.node);
  if (node.protected && !ctx.opts.yes) {
    ctx.err(`pulse model smoke: node ${node.name} is protected (it serves live traffic). Pass --yes to send a request to its server.`);
    return 2;
  }
  const saved = readNodeState(ctx.stateDir, node.name);
  const recipeName = saved?.recipe ?? 'qwen38-flash';
  const recipe = node.recipes[recipeName];
  if (!recipe) throw new ProfileError(`node ${node.name} has no recipe ${recipeName}`);
  const model = saved?.servedModel ?? 'qwen3.8-flash-next';
  const step: Step = { id: 'report', title: 'smoke', mutates: false, where: 'node', script: smokeScript(node, recipe, model), timeoutS: 330 };
  const r = await runStep(ctx, ctx.runnerFor(node), node, step, false);
  const s = parseSmoke(r.records);
  if (ctx.opts.json) ctx.out(JSON.stringify({ node: node.name, model, ...s }, null, 2));
  else {
    ctx.out(`smoke on ${node.name} (${model}): ${s.ok ? 'ok' : 'FAILED'}`);
    if (s.error) ctx.out(`  error          ${s.error}`);
    if (s.content !== undefined) ctx.out(`  content        ${JSON.stringify(s.content)}`);
    if (s.finishReason) ctx.out(`  finish_reason  ${s.finishReason}`);
    if (s.usage) ctx.out(`  usage          ${JSON.stringify(s.usage)}`);
    if (s.latencyMs !== undefined && Number.isFinite(s.latencyMs)) ctx.out(`  latency        ${s.latencyMs} ms`);
  }
  return s.ok ? 0 : 1;
}

function cmdFragment(ctx: Ctx): number {
  const nodes = loadNodes(ctx.nodesFile);
  printFragment(ctx, undefined, writeFragment(ctx.stateDir, nodes));
  return 0;
}

// Allow `node dist/runtime/cli.js model ...` as well as the bin/pulse-cli entry.
if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  const args = process.argv.slice(2);
  main(args[0] === 'model' ? args.slice(1) : args).then((code) => process.exit(code));
}
