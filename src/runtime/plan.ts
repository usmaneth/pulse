// Plans for `pulse model up` and `pulse model down`.
//
// Every function in this file is pure: it returns steps with bash scripts and
// runs nothing. The CLI prints the steps for --dry-run and gives them to a
// Runner for a real run. Each node script runs as `bash -s` on the node, so the
// node login shell (zsh on spark2) never parses it.
//
// Lines that start with "@pulse " are records for the CLI. All other output is
// for the operator.

import type { NodeConfig, NodesFile, Profile, Rendered, RecipeRef } from './profiles.js';

export type StepId =
  | 'preflight' | 'mkdir' | 'write-env' | 'stop' | 'wait-mem' | 'start' | 'wait-ready' | 'verify'
  | 'state' | 'report';

export interface Step {
  id: StepId;
  title: string;
  /** False when the step only reads. --dry-run --preflight runs only read-only preflight. */
  mutates: boolean;
  where: 'node' | 'local';
  script?: string;
  detail?: string;
  timeoutS?: number;
}

export interface Plan {
  action: 'up' | 'down';
  node: NodeConfig;
  recipe: RecipeRef;
  rendered?: Rendered;
  steps: Step[];
  /** Reasons why a real run must not start. --dry-run still prints the plan. */
  refusals: string[];
  ts: string;
  backupPath?: string;
  logPath?: string;
  rcPath?: string;
}

export interface PlanOptions {
  /** UTC time stamp for the backup and log names, for example 20260923T101500Z. */
  ts: string;
  yes?: boolean;
  experimental?: boolean;
  /** Override for the readiness wait in seconds. */
  timeoutS?: number;
  /** Override for the MemAvailable gate in GiB. */
  minMemGiB?: number;
  /** Seconds between readiness polls (default 10). */
  pollS?: number;
  stateDir: string;
}

/** Quote a string for bash with single quotes. */
export function shq(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

export function timestamp(d = new Date()): string {
  return d.toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
}

/** The address that probes on the node use. */
export function probeHost(bind: string): string {
  return bind === '0.0.0.0' || bind === '::' || bind === '' ? '127.0.0.1' : bind;
}

/** The MemAvailable gate: --min-mem-gib, then the node value, then the file value. */
export function minMemGiB(node: NodeConfig, nodes: NodesFile, opts: { minMemGiB?: number }): number {
  return opts.minMemGiB ?? node.minMemAvailableGiB ?? nodes.minMemAvailableGiB;
}

export function refusals(node: NodeConfig, profile: Profile | undefined, opts: { yes?: boolean; experimental?: boolean }): string[] {
  const out: string[] = [];
  if (profile && !profile.runnable) {
    out.push(`profile ${profile.name} is render-only (pulse-runnable: false). Pulse prints it but does not run it.`);
  }
  if (profile && profile.status === 'experimental' && !opts.experimental) {
    out.push(`profile ${profile.name} is experimental. Pass --experimental to use it.`);
  }
  if (node.protected && !opts.yes) {
    out.push(`node ${node.name} is protected (it serves live traffic). Pass --yes to change it.`);
  }
  return out;
}

const SECRET_RE = '^(HF_TOKEN|API_KEY)=';
const BODY_SHA = `grep -vE '^(#|$)' .env | grep -vE '${SECRET_RE}' | sha256sum | cut -d' ' -f1`;

function prelude(recipe: RecipeRef): string {
  return [
    'set -u',
    'export LC_ALL=C',
    `DIR=${shq(recipe.dir)}`,
    'cd -- "$DIR" 2>/dev/null || { echo "@pulse fail recipe-dir $DIR does not exist"; exit 3; }',
  ].join('\n');
}

/** The environment for start.sh and stop.sh. env -i makes the .env the only source of recipe keys. */
function cleanEnv(): string {
  return 'env -i HOME="$HOME" USER="${USER:-$(id -un)}" PATH="$PATH" LANG=C.UTF-8';
}

function hfTarget(dst: string, hfHome: string): string | null {
  const root = '/root/.cache/huggingface/';
  return dst.startsWith(root) ? `${hfHome.replace(/\/+$/, '')}/${dst.slice(root.length)}` : null;
}

/** Read-only checks on the node before anything changes. */
export function preflightScript(r: Rendered): string {
  const { recipe } = r;
  const lines: string[] = [
    prelude(recipe),
    'fails=0',
    'fail() { echo "@pulse fail $*"; fails=$((fails+1)); }',
    'ok() { echo "@pulse ok $*"; }',
    'ok recipe-dir "$DIR"',
  ];
  for (const cmd of [recipe.start, recipe.stop]) {
    const exe = cmd.split(' ')[0];
    lines.push(`if [ -x ${shq(exe)} ]; then ok script ${shq(exe)}; else fail script ${shq(exe)} is missing or not executable; fi`);
  }
  const image = r.map.IMAGE;
  if (image) {
    lines.push(
      `if layers=$(docker image inspect ${shq(image)} --format '{{json .RootFS.Layers}}' 2>/dev/null); then`,
      `  echo "@pulse image ${image} layers=$(printf '%s' "$layers" | tr ',' '\\n' | wc -l) layers-sha256=$(printf '%s' "$layers" | sha256sum | cut -c1-16)"`,
      'else',
      `  fail image ${shq(image)} is not on this node, and start.sh would download it. Pull the image on purpose first`,
      'fi',
    );
  }
  const mkdirs = new Set(r.mkdirs);
  for (const d of r.mkdirs) {
    lines.push(`if [ -d ${shq(d)} ]; then ok dir ${shq(d)}; else echo "@pulse plan-mkdir "${shq(d)}; fi`);
  }
  const hfHome = r.map.HF_HOME;
  for (const m of r.mounts) {
    if (!mkdirs.has(m.src)) {
      lines.push(`if [ -e ${shq(m.src)} ]; then ok mount-source ${shq(m.src)}; else fail mount-source ${shq(m.src)} is missing '(${m.origin})'; fi`);
    }
    const target = hfHome ? hfTarget(m.dst, hfHome) : null;
    if (target) {
      // A missing target inside the HF cache would make docker create an empty
      // root-owned file or directory in the cache.
      lines.push(`if [ -e ${shq(target)} ]; then ok hf-target ${shq(target)}; else fail hf-target ${shq(target)} is missing: the mount over it would create it; fi`);
    }
  }
  for (const key of ['CHAT_TEMPLATE', 'MTP_DRAFT_VOCAB']) {
    const v = r.map[key];
    if (v) {
      const p = v.startsWith('/') ? v : `${recipe.dir}/${v}`;
      lines.push(`if [ -r ${shq(p)} ]; then ok file ${key}; else fail file ${key}=${shq(v)} is not readable; fi`);
    }
  }
  if (hfHome && r.recipeName === 'qwen38-flash') {
    lines.push(`if [ -d ${shq(hfHome + '/hub')} ]; then ok hf-home ${shq(hfHome)}; else fail hf-home ${shq(hfHome + '/hub')} does not exist; fi`);
  }
  lines.push(
    'if ! ps_out=$(docker ps --format \'{{.Names}} {{.Image}}\' 2>&1); then fail docker "docker ps failed: $ps_out"; ps_out=""; fi',
    `others=$(printf '%s\\n' "$ps_out" | awk -v c=${shq(recipe.container)} '$1 != c && $2 ~ /vllm/ {print $1}' | tr '\\n' ' ')`,
    'if [ -n "$others" ]; then fail other-server "another vLLM container runs on this node: $others(one vLLM server per node)"; fi',
    `state=$(docker inspect -f '{{.State.Status}} {{.State.StartedAt}}' ${shq(recipe.container)} 2>/dev/null || echo absent)`,
    'echo "@pulse container $state"',
    'if [ -f .env ]; then',
    '  grep -E \'^# pulse-[a-z0-9-]+:\' .env | sed \'s/^/@pulse env-header /\'',
    `  echo "@pulse env-body-sha $(${BODY_SHA})"`,
    '  echo "@pulse env-mtime $(stat -c %.9Y .env)"',
    // Source the current .env in an empty environment and print the key map.
    // Secret values are masked here, so they never leave the node.
    '  env -i bash -c \'set -a; . ./.env >/dev/null 2>&1; env\' </dev/null | grep -E \'^[A-Za-z_][A-Za-z0-9_]*=\' | grep -vE \'^(PWD|OLDPWD|SHLVL|_)=\' \\',
    '    | while IFS= read -r l; do case "$l" in HF_TOKEN=*|API_KEY=*) echo "@pulse env-var ${l%%=*}=<set>";; *) echo "@pulse env-var $l";; esac; done',
    'else',
    '  echo "@pulse env-missing"',
    'fi',
    `echo "@pulse clients $(ss -Htn state established '( sport = :${r.port} )' 2>/dev/null | wc -l)"`,
    'echo "@pulse mem-available-gib $(awk \'/MemAvailable/{printf "%.1f", $2/1048576}\' /proc/meminfo)"',
    'echo "@pulse preflight-fails $fails"',
    '[ "$fails" -eq 0 ]',
  );
  return lines.join('\n') + '\n';
}

export function mkdirScript(dirs: string[]): string {
  return ['set -eu', ...dirs.map((d) => `mkdir -p -- ${shq(d)} && echo "@pulse ok mkdir "${shq(d)}`)].join('\n') + '\n';
}

/** Write the .env with a backup. Keeps HF_TOKEN and API_KEY lines of the old file without printing them. */
export function writeEnvScript(r: Rendered, backupPath: string): string {
  const eof = 'PULSE_ENV_EOF';
  if (r.text.split('\n').includes(eof)) throw new Error('rendered .env contains the heredoc end marker');
  return [
    prelude(r.recipe),
    'set -e',
    `NEW_SHA=${shq(r.sha256)}`,
    `NEW_PROFILE=${shq(r.profile.name)}`,
    'if [ -f .env ]; then',
    `  cur=$(${BODY_SHA})`,
    '  hdr=$(sed -n \'s/^# pulse-sha256: *//p\' .env | head -n 1)',
    '  prof=$(sed -n \'s/^# pulse-profile: *//p\' .env | head -n 1)',
    '  if [ "$cur" = "$NEW_SHA" ] && [ "$hdr" = "$NEW_SHA" ] && [ "$prof" = "$NEW_PROFILE" ]; then',
    '    echo "@pulse env unchanged"',
    '    exit 0',
    '  fi',
    `  cp -p .env ${shq(backupPath)}`,
    `  echo "@pulse env-backup "${shq(backupPath)}`,
    'fi',
    'tmp=".env.pulse-tmp.$$"',
    `cat > "$tmp" <<'${eof}'`,
    r.text.replace(/\n$/, ''),
    eof,
    `if [ -f .env ] && grep -qE '${SECRET_RE}' .env; then`,
    `  { echo; echo '# pulse-carried: secret lines kept from the previous .env'; grep -E '${SECRET_RE}' .env; } >> "$tmp"`,
    `  echo "@pulse env-carried $(grep -cE '${SECRET_RE}' .env)"`,
    'fi',
    'if [ -f .env ]; then chmod --reference=.env "$tmp"; fi',
    'mv -f "$tmp" .env',
    'echo "@pulse env written sha256=$NEW_SHA"',
  ].join('\n') + '\n';
}

export function stopScript(recipe: RecipeRef): string {
  return [
    prelude(recipe),
    `${cleanEnv()} bash -c ${shq(recipe.stop)} </dev/null`,
    'rc=$?',
    `left=$(docker ps -aq -f ${shq(`name=^${recipe.container}$`)})`,
    `if [ -n "$left" ]; then echo "@pulse fail stop container ${recipe.container} still exists after ${recipe.stop} (exit $rc)"; exit 1; fi`,
    `echo "@pulse ok stopped ${recipe.container} (stop exit $rc)"`,
  ].join('\n') + '\n';
}

export function waitMemScript(minGiB: number, timeoutS: number): string {
  return [
    'set -u',
    `min=${minGiB}; timeout=${timeoutS}; t0=$(date +%s); last=0`,
    'while :; do',
    '  avail=$(awk \'/MemAvailable/{print int($2/1048576)}\' /proc/meminfo)',
    '  now=$(date +%s); el=$((now - t0))',
    '  if [ "$avail" -ge "$min" ]; then echo "@pulse mem-available-gib $avail"; echo "MemAvailable ${avail} GiB after ${el}s"; exit 0; fi',
    '  if [ "$el" -ge "$timeout" ]; then echo "@pulse fail mem MemAvailable is ${avail} GiB, below ${min} GiB after ${el}s"; exit 1; fi',
    '  if [ $((now - last)) -ge 30 ]; then last=$now; echo "waiting for MemAvailable >= ${min} GiB: ${avail} GiB now (${el}s)"; fi',
    '  sleep 5',
    'done',
  ].join('\n') + '\n';
}

/** Start the recipe in its own session, so a dropped ssh connection cannot stop it. */
export function startScript(recipe: RecipeRef, logPath: string, rcPath: string): string {
  const inner = `${recipe.start} > "$1" 2>&1; echo $? > "$2"`;
  return [
    prelude(recipe),
    'mkdir -p logs',
    `rm -f ${shq(rcPath)}`,
    `setsid ${cleanEnv()} bash -c ${shq(inner)} _ ${shq(logPath)} ${shq(rcPath)} </dev/null >/dev/null 2>&1 &`,
    'echo "@pulse start-pid $!"',
    `echo "@pulse start-log $DIR/"${shq(logPath)}`,
    `echo "started ${recipe.start} in the background. Log: $DIR/"${shq(logPath)}`,
  ].join('\n') + '\n';
}

export function waitReadyScript(r: Rendered, logPath: string, rcPath: string, timeoutS: number, pollS = 10): string {
  const host = probeHost(r.bind);
  return [
    prelude(r.recipe),
    `rc=${shq(rcPath)}; log=${shq(logPath)}; c=${shq(r.recipe.container)}`,
    `url=${shq(`http://${host}:${r.port}/v1/models`)}`,
    `timeout=${timeoutS}; t0=$(date +%s); last=0; seen=0`,
    'tails() { echo "--- last 40 lines of $DIR/$log"; tail -n 40 "$log" 2>/dev/null; }',
    'while :; do',
    '  now=$(date +%s); el=$((now - t0))',
    '  if [ -f "$rc" ] && [ "$(cat "$rc")" != "0" ]; then echo "@pulse fail start the start command exited with $(cat "$rc")"; tails; exit 1; fi',
    '  st=$(docker inspect -f \'{{.State.Status}}\' "$c" 2>/dev/null || echo absent)',
    '  [ "$st" = running ] && seen=1',
    '  if [ "$seen" = 1 ] && [ "$st" != running ]; then echo "@pulse fail container $c is $st"; tails; exit 1; fi',
    '  code=$(curl -s -o /dev/null -m 5 -w \'%{http_code}\' "$url" 2>/dev/null); [ -n "$code" ] || code=000',
    '  if [ "$code" = 200 ]; then echo "@pulse ready-s $el"; echo "/v1/models returned 200 after ${el}s"; exit 0; fi',
    '  if [ "$el" -ge "$timeout" ]; then echo "@pulse fail timeout /v1/models did not return 200 in ${el}s"; tails; exit 1; fi',
    '  if [ $((now - last)) -ge 60 ]; then last=$now; echo "waiting for /v1/models: ${el}s, container $st, last HTTP code $code"; fi',
    `  sleep ${pollS}`,
    'done',
  ].join('\n') + '\n';
}

export function verifyScript(r: Rendered): string {
  const host = probeHost(r.bind);
  const lines = [
    prelude(r.recipe),
    `c=${shq(r.recipe.container)}`,
    'logs=$(mktemp); trap \'rm -f "$logs"\' EXIT',
    'docker logs "$c" > "$logs" 2>&1 </dev/null',
    `echo "@pulse models-body $(curl -s -m 10 ${shq(`http://${host}:${r.port}/v1/models`)} | tr -d '\\n')"`,
    'echo "@pulse container $(docker inspect -f \'{{.State.Status}} {{.State.StartedAt}}\' "$c" 2>/dev/null || echo absent)"',
    'echo "@pulse kv $(grep -E \'GPU KV cache size|Available KV cache|Maximum concurrency\' "$logs" | tail -n 1)"',
  ];
  for (const p of r.proofs) {
    lines.push(`if grep -qF -- ${shq(p)} "$logs"; then echo "@pulse proof ok "${shq(p)}; else echo "@pulse proof missing "${shq(p)}; fi`);
  }
  return lines.join('\n') + '\n';
}

export function planUp(r: Rendered, nodes: NodesFile, opts: PlanOptions): Plan {
  const { node, recipe } = r;
  const backupPath = `${recipe.dir}/.env.pulse-bak-${opts.ts}`;
  const logPath = `logs/pulse-up-${opts.ts}.log`;
  const rcPath = `logs/pulse-up-${opts.ts}.rc`;
  const readyTimeout = opts.timeoutS ?? Number(r.map.READY_TIMEOUT_S ?? 1800) + nodes.readyGraceS;
  const minMem = minMemGiB(node, nodes, opts);
  const plan: Plan = {
    action: 'up', node, recipe, rendered: r, steps: [], ts: opts.ts,
    refusals: refusals(node, r.profile, opts), backupPath, logPath, rcPath,
  };
  if (!r.profile.runnable) return plan;
  const where = node.host === 'ssh' ? `on ${node.name} over ssh ${node.ssh}` : `on ${node.name} (local)`;
  plan.steps.push({ id: 'preflight', title: `preflight checks ${where}`, mutates: false, where: 'node', script: preflightScript(r), timeoutS: 120 });
  if (r.mkdirs.length) {
    plan.steps.push({ id: 'mkdir', title: `create directories: ${r.mkdirs.join(', ')}`, mutates: true, where: 'node', script: mkdirScript(r.mkdirs), timeoutS: 60 });
  }
  plan.steps.push({ id: 'write-env', title: `write ${recipe.dir}/.env (backup ${backupPath}; no write when the sha256 is unchanged)`, mutates: true, where: 'node', script: writeEnvScript(r, backupPath), timeoutS: 60 });
  plan.steps.push({ id: 'stop', title: `stop the running server (${recipe.stop}) and check that ${recipe.container} is gone`, mutates: true, where: 'node', script: stopScript(recipe), timeoutS: 180 });
  plan.steps.push({ id: 'wait-mem', title: `wait until MemAvailable >= ${minMem} GiB (timeout ${nodes.memTimeoutS} s)`, mutates: false, where: 'node', script: waitMemScript(minMem, nodes.memTimeoutS), timeoutS: nodes.memTimeoutS + 60 });
  plan.steps.push({ id: 'start', title: `start ${recipe.start} in the background, log ${recipe.dir}/${logPath}`, mutates: true, where: 'node', script: startScript(recipe, logPath, rcPath), timeoutS: 60 });
  plan.steps.push({ id: 'wait-ready', title: `wait until /v1/models returns 200 on ${probeHost(r.bind)}:${r.port} (timeout ${readyTimeout} s)`, mutates: false, where: 'node', script: waitReadyScript(r, logPath, rcPath, readyTimeout, opts.pollS), timeoutS: readyTimeout + 120 });
  plan.steps.push({ id: 'verify', title: `verify: served model ${r.servedModel}, max_model_len ${r.expectedMaxModelLen ?? '?'}, KV line, ${r.proofs.length} proof line(s)`, mutates: false, where: 'node', script: verifyScript(r), timeoutS: 180 });
  plan.steps.push({ id: 'state', title: 'write the node state and the gateway backend fragment', mutates: true, where: 'local', detail: `${opts.stateDir}/nodes/${node.name}.json and ${opts.stateDir}/gateway-backends.json` });
  return plan;
}

export function planDown(node: NodeConfig, recipeName: string, opts: PlanOptions): Plan {
  const recipe = node.recipes[recipeName];
  if (!recipe) throw new Error(`node ${node.name} has no recipe ${recipeName}`);
  const plan: Plan = { action: 'down', node, recipe, steps: [], ts: opts.ts, refusals: refusals(node, undefined, opts) };
  plan.steps.push({ id: 'stop', title: `stop the server (${recipe.stop}) and check that ${recipe.container} is gone`, mutates: true, where: 'node', script: stopScript(recipe), timeoutS: 180 });
  plan.steps.push({
    id: 'report', title: 'report MemAvailable', mutates: false, where: 'node', timeoutS: 30,
    script: 'echo "@pulse mem-available-gib $(awk \'/MemAvailable/{printf "%.1f", $2/1048576}\' /proc/meminfo)"\n',
  });
  plan.steps.push({ id: 'state', title: 'mark the node down and rewrite the gateway backend fragment', mutates: true, where: 'local', detail: `${opts.stateDir}/nodes/${node.name}.json and ${opts.stateDir}/gateway-backends.json` });
  return plan;
}
