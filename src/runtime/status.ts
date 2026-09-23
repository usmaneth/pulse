// `pulse model status` and `pulse model smoke`: scripts and parsers.

import { shq, probeHost } from './plan.js';
import type { NodeConfig, RecipeRef } from './profiles.js';

export interface KvInfo {
  tokens: number;
  perRequestTokens?: number;
  maxConcurrency?: number;
  line: string;
}

/** Parse the vLLM line "GPU KV cache size: 667,808 tokens, Maximum concurrency for 262,144 tokens per request: 2.55x". */
export function parseKvLine(line: string): KvInfo | null {
  const m = /GPU KV cache size:\s*([\d,]+)\s*tokens/.exec(line);
  if (!m) return null;
  const out: KvInfo = { tokens: Number(m[1].replace(/,/g, '')), line: line.trim() };
  const c = /Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x/.exec(line);
  if (c) {
    out.perRequestTokens = Number(c[1].replace(/,/g, ''));
    out.maxConcurrency = Number(c[2]);
  }
  return out;
}

export interface ContainerState {
  exists: boolean;
  status: string;
  running: boolean;
  startedAt?: string;
  exitCode?: number;
  oomKilled?: boolean;
  restartCount?: number;
}

/** Parse `docker inspect -f '{{json .State}}'` output, or "absent". */
export function parseContainerState(text: string): ContainerState {
  const t = text.trim();
  if (!t || t === 'absent') return { exists: false, status: 'absent', running: false };
  const s = JSON.parse(t) as { Status?: string; Running?: boolean; StartedAt?: string; ExitCode?: number; OOMKilled?: boolean; RestartCount?: number };
  return {
    exists: true,
    status: s.Status ?? 'unknown',
    running: s.Running === true,
    startedAt: s.StartedAt,
    exitCode: s.ExitCode,
    oomKilled: s.OOMKilled,
    restartCount: s.RestartCount,
  };
}

export interface ModelsInfo {
  id: string;
  maxModelLen?: number;
}

/** Parse the /v1/models body of vLLM. */
export function parseModels(body: string): ModelsInfo | null {
  try {
    const j = JSON.parse(body) as { data?: { id?: string; max_model_len?: number }[] };
    const m = j.data?.[0];
    if (!m?.id) return null;
    return { id: m.id, maxModelLen: m.max_model_len };
  } catch {
    return null;
  }
}

export interface EnvHeader {
  profile?: string;
  node?: string;
  recipe?: string;
  overlay?: string;
  rendered?: string;
  sha256?: string;
}

/** Parse "# pulse-name: value" lines (or the same lines after "env-header "). */
export function parseEnvHeader(lines: string[]): EnvHeader {
  const out: Record<string, string> = {};
  for (const l of lines) {
    const m = /^#\s*pulse-([a-z0-9-]+):\s*(.*)$/.exec(l.trim());
    if (m && !(m[1] in out)) out[m[1]] = m[2].trim();
  }
  return out as EnvHeader;
}

/** Group @pulse records by their first word. */
export function recordMap(recs: string[]): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const r of recs) {
    const sp = r.indexOf(' ');
    const tag = sp < 0 ? r : r.slice(0, sp);
    const rest = sp < 0 ? '' : r.slice(sp + 1);
    if (!map.has(tag)) map.set(tag, []);
    map.get(tag)!.push(rest);
  }
  return map;
}

export function first(map: Map<string, string[]>, tag: string): string | undefined {
  return map.get(tag)?.[0];
}

export function uptime(startedAt: string | undefined, now = new Date()): string | undefined {
  if (!startedAt || startedAt.startsWith('0001-')) return undefined;
  const ms = now.getTime() - new Date(startedAt).getTime();
  if (!Number.isFinite(ms) || ms < 0) return undefined;
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
}

export interface StatusOptions {
  http: boolean;
  gpuProbe: boolean;
}

export function statusScript(node: NodeConfig, recipe: RecipeRef, opts: StatusOptions): string {
  const port = node.env.PORT ?? '8888';
  const host = probeHost(node.env.BIND ?? '0.0.0.0');
  const lines = [
    'set -u',
    'export LC_ALL=C',
    `DIR=${shq(recipe.dir)}; c=${shq(recipe.container)}`,
    'echo "@pulse state $(docker inspect -f \'{{json .State}}\' "$c" 2>/dev/null || echo absent)"',
    'echo "@pulse image $(docker inspect -f \'{{.Config.Image}}\' "$c" 2>/dev/null)"',
    'img=$(docker inspect -f \'{{.Image}}\' "$c" 2>/dev/null)',
    'if [ -n "$img" ]; then echo "@pulse layers-sha256 $(docker image inspect "$img" --format \'{{json .RootFS.Layers}}\' 2>/dev/null | tr -d \'\\n\' | sha256sum | cut -c1-16)"; fi',
    'if [ -f "$DIR/.env" ]; then',
    '  grep -E \'^# pulse-[a-z0-9-]+:\' "$DIR/.env" | sed \'s/^/@pulse env-header /\'',
    '  echo "@pulse env-body-sha $(grep -vE \'^(#|$)\' "$DIR/.env" | grep -vE \'^(HF_TOKEN|API_KEY)=\' | sha256sum | cut -d\' \' -f1)"',
    '  echo "@pulse env-mtime $(stat -c %.9Y "$DIR/.env")"',
    'else',
    '  echo "@pulse env-missing"',
    'fi',
    'if docker inspect "$c" >/dev/null 2>&1; then',
    '  echo "@pulse kv $(docker logs "$c" 2>&1 </dev/null | grep -E \'GPU KV cache size\' | tail -n 1)"',
    'fi',
  ];
  if (opts.http) {
    lines.push(
      `echo "@pulse health $(curl -s -o /dev/null -m 5 -w '%{http_code}' ${shq(`http://${host}:${port}/health`)} 2>/dev/null)"`,
      `echo "@pulse models-body $(curl -s -m 5 ${shq(`http://${host}:${port}/v1/models`)} 2>/dev/null | tr -d '\\n')"`,
    );
  }
  lines.push(
    'echo "@pulse mem-available-gib $(awk \'/MemAvailable/{printf "%.1f", $2/1048576}\' /proc/meminfo)"',
    'echo "@pulse mem-total-gib $(awk \'/MemTotal/{printf "%.1f", $2/1048576}\' /proc/meminfo)"',
    'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed \'/^$/d; s/^/@pulse gpu-app /\'',
    'echo "@pulse vm compaction_proactiveness=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null) min_free_kbytes=$(cat /proc/sys/vm/min_free_kbytes 2>/dev/null) watermark_scale_factor=$(cat /proc/sys/vm/watermark_scale_factor 2>/dev/null)"',
    'echo "@pulse thp enabled=$(sed -n \'s/.*\\[\\(.*\\)\\].*/\\1/p\' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null) defrag=$(sed -n \'s/.*\\[\\(.*\\)\\].*/\\1/p\' /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null)"',
  );
  if (opts.gpuProbe && node.gpuProbe) {
    lines.push(`echo "@pulse gpu-probe $(${node.gpuProbe} 2>&1 </dev/null | tail -n 1)"`);
  }
  return lines.join('\n') + '\n';
}

export interface NodeStatus {
  node: string;
  reachable: boolean;
  error?: string;
  container: ContainerState;
  uptime?: string;
  image?: string;
  layersSha256?: string;
  env: { managed: boolean; missing: boolean; header: EnvHeader; bodySha?: string; drift: string[] };
  health?: string;
  models?: ModelsInfo | null;
  modelsCode?: string;
  kv?: KvInfo | null;
  kvSource?: 'log' | 'state';
  memAvailableGiB?: number;
  memTotalGiB?: number;
  gpuApps: string[];
  vm?: string;
  thp?: string;
  gpuProbe?: string;
}

export function parseStatus(nodeName: string, recs: string[], savedKv: string | undefined, now = new Date()): NodeStatus {
  const m = recordMap(recs);
  const container = parseContainerState(first(m, 'state') ?? 'absent');
  const headerLines = (m.get('env-header') ?? []);
  const header = parseEnvHeader(headerLines);
  const bodySha = first(m, 'env-body-sha');
  const missing = m.has('env-missing');
  const drift: string[] = [];
  const managed = !!header.sha256;
  if (!missing && !managed) drift.push('unmanaged: the .env has no pulse header');
  if (managed && bodySha && bodySha !== header.sha256) drift.push('the .env was edited after pulse rendered it');
  const mtime = first(m, 'env-mtime');
  if (mtime && container.running && container.startedAt) {
    const started = new Date(container.startedAt).getTime() / 1000;
    if (Number(mtime) > started) drift.push('the .env changed after the container started');
  }
  let kv = parseKvLine(first(m, 'kv') ?? '');
  let kvSource: NodeStatus['kvSource'] = kv ? 'log' : undefined;
  if (!kv && savedKv) {
    kv = parseKvLine(savedKv);
    if (kv) kvSource = 'state';
  }
  const body = first(m, 'models-body');
  const num = (s: string | undefined) => (s === undefined || s === '' ? undefined : Number(s));
  return {
    node: nodeName,
    reachable: true,
    container,
    uptime: container.running ? uptime(container.startedAt, now) : undefined,
    image: first(m, 'image') || undefined,
    layersSha256: first(m, 'layers-sha256'),
    env: { managed, missing, header, bodySha, drift },
    health: first(m, 'health'),
    models: body === undefined ? undefined : parseModels(body),
    kv,
    kvSource,
    memAvailableGiB: num(first(m, 'mem-available-gib')),
    memTotalGiB: num(first(m, 'mem-total-gib')),
    gpuApps: m.get('gpu-app') ?? [],
    vm: first(m, 'vm'),
    thp: first(m, 'thp'),
    gpuProbe: first(m, 'gpu-probe'),
  };
}

export function smokeScript(node: NodeConfig, servedModel: string): string {
  const port = node.env.PORT ?? '8888';
  const host = probeHost(node.env.BIND ?? '0.0.0.0');
  const body = JSON.stringify({
    model: servedModel,
    messages: [{ role: 'user', content: 'Reply with one word: ready' }],
    max_tokens: 16,
    temperature: 0,
    chat_template_kwargs: { enable_thinking: false },
  });
  return [
    'set -u',
    't0=$(date +%s%N)',
    `resp=$(curl -s -m 300 -H 'Content-Type: application/json' -d ${shq(body)} ${shq(`http://${host}:${port}/v1/chat/completions`)} | tr -d '\\n')`,
    't1=$(date +%s%N)',
    'echo "@pulse smoke-ms $(( (t1 - t0) / 1000000 ))"',
    'echo "@pulse smoke-body $resp"',
  ].join('\n') + '\n';
}

export interface SmokeResult {
  ok: boolean;
  content?: string;
  finishReason?: string;
  usage?: unknown;
  latencyMs?: number;
  error?: string;
}

export function parseSmoke(recs: string[]): SmokeResult {
  const m = recordMap(recs);
  const latencyMs = Number(first(m, 'smoke-ms'));
  const body = first(m, 'smoke-body') ?? '';
  try {
    const j = JSON.parse(body) as { choices?: { message?: { content?: string }; finish_reason?: string }[]; usage?: unknown; error?: unknown };
    const c = j.choices?.[0];
    if (!c) return { ok: false, latencyMs, error: `no choices in the reply: ${body.slice(0, 300)}` };
    const content = c.message?.content ?? '';
    return { ok: content.trim().length > 0 && !!c.finish_reason, content, finishReason: c.finish_reason, usage: j.usage, latencyMs };
  } catch {
    return { ok: false, latencyMs, error: `the reply is not JSON: ${body.slice(0, 300) || '(empty)'}` };
  }
}
