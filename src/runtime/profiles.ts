// Nodes, profiles and the rendered recipe .env.
//
// runtime/nodes.json holds the facts of each host: how to reach it, where the
// recipe is, and the node keys (HF_HOME, BIND, PORT, REQUIRE_IDLE_GPU).
// runtime/qwen38/profiles/*.env holds portable profiles. A profile is a
// complete key set for the recipe .env with {{name}} variables for host paths.
// renderEnv() joins a profile, a node and an optional overlay into the .env
// text that `pulse model up` writes on the node.

import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { EnvFileError, KEY_RE, WORD_SPLIT_KEYS, checkValue, formatAssignment, parseEnv } from './envfile.js';
import type { Directive } from './envfile.js';
import { formatDockerArgs, mergeDockerArgs, pairs, parseDockerArgs, tokenize } from './dockerargs.js';
import type { DockerArg, MountArg } from './dockerargs.js';

export class ProfileError extends Error {}

export interface RecipeRef {
  dir: string;
  /** Command that starts the server, run in `dir`, for example ./start.sh. */
  start: string;
  /** Command that stops the server, run in `dir`. */
  stop: string;
  container: string;
  /** Docker arguments that every profile of this recipe gets on this node. */
  dockerArgs?: string[];
  vars?: Record<string, string>;
}

export interface NodeConfig {
  name: string;
  description?: string;
  host: 'local' | 'ssh';
  ssh?: string;
  /** A protected node serves live traffic. Changes need --yes. */
  protected?: boolean;
  /** Node keys of the recipe .env. A profile cannot set them. */
  env: Record<string, string>;
  vars: Record<string, string>;
  /** The URL that the gateway uses for this node. */
  gatewayUrl: string;
  /** Optional GB10 GEMV probe. It starts a GPU container, so it runs only on request. */
  gpuProbe?: string;
  /** Override of the top-level minMemAvailableGiB for this node. */
  minMemAvailableGiB?: number;
  recipes: Record<string, RecipeRef>;
}

export interface NodesFile {
  minMemAvailableGiB: number;
  memTimeoutS: number;
  readyGraceS: number;
  nodes: Record<string, NodeConfig>;
}

/** Keys that always come from the node. */
export const NODE_KEYS = ['HF_HOME', 'BIND', 'PORT', 'REQUIRE_IDLE_GPU'];
/** Keys that a profile must never set. */
export const REFUSED_KEYS: Record<string, string> = {
  HF_TOKEN: 'secrets stay in the node .env; pulse keeps an existing HF_TOKEN line',
  API_KEY: 'secrets stay in the node .env; pulse keeps an existing API_KEY line',
  TP1_CONTAINER_NAME: 'the container name comes from runtime/nodes.json',
};

const SAFE_PATH_RE = /^\/[A-Za-z0-9_.\/@+%,=-]*$/;
const SAFE_CMD_RE = /^[A-Za-z0-9_.\/ =-]+$/;
const SAFE_NAME_RE = /^[A-Za-z0-9_.-]+$/;
const SSH_RE = /^[A-Za-z0-9_.@-]+$/;

function need(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new ProfileError(msg);
}

/** Load and check runtime/nodes.json. */
export function loadNodes(file: string): NodesFile {
  const raw = JSON.parse(readFileSync(file, 'utf8')) as Partial<NodesFile> & { nodes?: Record<string, Partial<NodeConfig>> };
  need(raw.nodes && typeof raw.nodes === 'object', `${file}: "nodes" is missing`);
  const nodes: Record<string, NodeConfig> = {};
  for (const [name, n] of Object.entries(raw.nodes)) {
    const where = `${file}: node ${name}`;
    need(SAFE_NAME_RE.test(name), `${where}: bad node name`);
    need(n.host === 'local' || n.host === 'ssh', `${where}: host must be "local" or "ssh"`);
    if (n.host === 'ssh') need(typeof n.ssh === 'string' && SSH_RE.test(n.ssh), `${where}: ssh target is missing or unsafe`);
    const env = n.env ?? {};
    for (const [k, v] of Object.entries(env)) {
      need(KEY_RE.test(k), `${where}: bad env key ${k}`);
      need(NODE_KEYS.includes(k), `${where}: env key ${k} is not a node key (${NODE_KEYS.join(', ')})`);
      try { checkValue(k, v); } catch (e) { throw new ProfileError(`${where}: ${(e as Error).message}`); }
    }
    need(typeof n.gatewayUrl === 'string', `${where}: gatewayUrl is missing`);
    new URL(n.gatewayUrl);
    need(n.gpuProbe === undefined || SAFE_CMD_RE.test(n.gpuProbe), `${where}: gpuProbe is unsafe`);
    need(n.minMemAvailableGiB === undefined || (typeof n.minMemAvailableGiB === 'number' && n.minMemAvailableGiB > 0), `${where}: minMemAvailableGiB must be a positive number`);
    const recipes: Record<string, RecipeRef> = {};
    for (const [rname, r] of Object.entries(n.recipes ?? {})) {
      const rw = `${where}: recipe ${rname}`;
      need(typeof r.dir === 'string' && SAFE_PATH_RE.test(r.dir), `${rw}: dir must be a safe absolute path`);
      need(typeof r.start === 'string' && SAFE_CMD_RE.test(r.start), `${rw}: start command is missing or unsafe`);
      need(typeof r.stop === 'string' && SAFE_CMD_RE.test(r.stop), `${rw}: stop command is missing or unsafe`);
      need(typeof r.container === 'string' && SAFE_NAME_RE.test(r.container), `${rw}: container name is missing or unsafe`);
      need(r.dockerArgs === undefined || Array.isArray(r.dockerArgs), `${rw}: dockerArgs must be a list of words`);
      // Parse once here so a bad entry fails at load time. {{vars}} resolve at render time.
      parseDockerArgs(tokenize((r.dockerArgs ?? []).join(' ').replace(/\{\{[A-Za-z0-9_]+\}\}/g, 'x'), `${rw}: dockerArgs`), `node:${name}`);
      recipes[rname] = { ...r };
    }
    nodes[name] = {
      name,
      description: n.description,
      host: n.host,
      ssh: n.ssh,
      protected: n.protected === true,
      env,
      vars: n.vars ?? {},
      gatewayUrl: n.gatewayUrl,
      gpuProbe: n.gpuProbe,
      minMemAvailableGiB: n.minMemAvailableGiB,
      recipes,
    };
  }
  return {
    minMemAvailableGiB: raw.minMemAvailableGiB ?? 100,
    memTimeoutS: raw.memTimeoutS ?? 600,
    readyGraceS: raw.readyGraceS ?? 300,
    nodes,
  };
}

export function getNode(nodes: NodesFile, name: string | undefined): NodeConfig {
  if (!name) throw new ProfileError(`--node is required (${Object.keys(nodes.nodes).join(', ')})`);
  const node = nodes.nodes[name];
  if (!node) throw new ProfileError(`unknown node ${name} (known: ${Object.keys(nodes.nodes).join(', ')})`);
  return node;
}

export interface Profile {
  name: string;
  file: string;
  recipe: string;
  status: 'stable' | 'experimental';
  runnable: boolean;
  description: string;
  /** Assignments as written, with {{vars}} not yet resolved. */
  entries: { key: string; value: string }[];
  mkdirs: string[];
  proofs: string[];
  clientNotes: string[];
}

const DIRECTIVES = new Set(['recipe', 'status', 'runnable', 'description', 'mkdir', 'proof', 'client']);

export function listProfiles(dir: string): string[] {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter((f) => f.endsWith('.env')).map((f) => f.slice(0, -4)).sort();
}

export function loadProfile(dir: string, name: string): Profile {
  need(SAFE_NAME_RE.test(name), `bad profile name ${JSON.stringify(name)}`);
  const file = path.join(dir, `${name}.env`);
  if (!existsSync(file)) {
    throw new ProfileError(`no profile ${name} in ${dir} (profiles: ${listProfiles(dir).join(', ') || 'none'})`);
  }
  let parsed;
  try {
    parsed = parseEnv(readFileSync(file, 'utf8'), path.relative(process.cwd(), file) || file);
  } catch (e) {
    if (e instanceof EnvFileError) throw new ProfileError(e.message);
    throw e;
  }
  const one = (d: string): Directive | undefined => {
    const all = parsed.directives.filter((x) => x.name === d);
    need(all.length <= 1, `profile ${name}: pulse-${d} appears more than once`);
    return all[0];
  };
  for (const d of parsed.directives) {
    need(DIRECTIVES.has(d.name), `profile ${name}:${d.line}: unknown directive pulse-${d.name}`);
  }
  const recipe = one('recipe')?.value;
  need(recipe, `profile ${name}: the "# pulse-recipe: <name>" line is missing`);
  const status = one('status')?.value ?? 'stable';
  need(status === 'stable' || status === 'experimental', `profile ${name}: pulse-status must be stable or experimental`);
  const runnable = one('runnable')?.value ?? 'true';
  need(runnable === 'true' || runnable === 'false', `profile ${name}: pulse-runnable must be true or false`);
  const seen = new Set<string>();
  for (const a of parsed.assignments) {
    need(!seen.has(a.key), `profile ${name}:${a.line}: ${a.key} is set twice`);
    seen.add(a.key);
    need(!NODE_KEYS.includes(a.key), `profile ${name}:${a.line}: ${a.key} is a node key. Set it in runtime/nodes.json`);
    need(!(a.key in REFUSED_KEYS), `profile ${name}:${a.line}: ${a.key} is not allowed in a profile: ${REFUSED_KEYS[a.key]}`);
    // pulse sends the key of the node .env (API_KEY) with its requests. A key
    // in EXTRA_VLLM_ARGS would be a secret in the repo, and pulse cannot send it.
    need(!(a.key === 'EXTRA_VLLM_ARGS' && /(^|\s)--api-key(\s|=|$)/.test(a.value)), `profile ${name}:${a.line}: --api-key is not allowed in EXTRA_VLLM_ARGS: ${REFUSED_KEYS.API_KEY}`);
  }
  const many = (d: string) => parsed.directives.filter((x) => x.name === d).map((x) => x.value);
  return {
    name,
    file,
    recipe,
    status,
    runnable: runnable === 'true',
    description: one('description')?.value ?? '',
    entries: parsed.assignments.map(({ key, value }) => ({ key, value })),
    mkdirs: many('mkdir'),
    proofs: many('proof'),
    clientNotes: many('client'),
  };
}

export interface Overlay {
  dir: string;
  words: string[];
  /** manifest.json as written by the overlay build, or null when it is missing. */
  manifest: unknown;
  manifestSha256: string | null;
  /**
   * The sha256 of docker-args.txt and manifest.json. The .env header records
   * it, so a rebuilt overlay gives a different .env identity and `up` restarts
   * the server, also when docker-args.txt did not change.
   */
  sha256: string;
}

/**
 * Read an overlay build output directory. docker-args.txt must exist. It holds
 * -e and -v pairs separated by white space, on one line or more; lines that
 * start with # are comments.
 */
export function readOverlay(dir: string): Overlay {
  const argsFile = path.join(dir, 'docker-args.txt');
  if (!existsSync(argsFile)) {
    throw new ProfileError(`overlay ${dir}: docker-args.txt is missing (build it with overlays/qwen38/build.sh --out ${dir})`);
  }
  const text = readFileSync(argsFile, 'utf8');
  const kept = text.split('\n').filter((l) => !l.trim().startsWith('#')).join(' ');
  const words = tokenize(kept, `overlay ${argsFile}`);
  const manifestFile = path.join(dir, 'manifest.json');
  let manifest: unknown = null;
  let manifestSha256: string | null = null;
  let rawManifest = '';
  if (existsSync(manifestFile)) {
    rawManifest = readFileSync(manifestFile, 'utf8');
    manifest = JSON.parse(rawManifest);
    manifestSha256 = createHash('sha256').update(rawManifest).digest('hex');
  }
  const sha256 = createHash('sha256').update(`docker-args.txt\0${text}\0manifest.json\0${rawManifest}`).digest('hex');
  return { dir: path.resolve(dir), words, manifest, manifestSha256, sha256 };
}

export interface RenderOptions {
  overlay?: Overlay;
  overlayWins?: boolean;
  now?: Date;
}

export interface Rendered {
  profile: Profile;
  node: NodeConfig;
  recipe: RecipeRef;
  recipeName: string;
  /** The final key map, in file order. */
  entries: [string, string][];
  map: Record<string, string>;
  /** Assignment lines only. The sha256 covers exactly these lines. */
  bodyLines: string[];
  sha256: string;
  /**
   * The value of the "# pulse-overlay:" header line: "none", or the overlay
   * directory and its sha256. With the body sha256 and the profile name, it is
   * the identity that decides if the node .env is unchanged.
   */
  overlayLine: string;
  /** The complete file: header, then the body. */
  text: string;
  dockerArgs: DockerArg[];
  mounts: MountArg[];
  replaced: string[];
  mkdirs: string[];
  proofs: string[];
  overlay?: Overlay;
  servedModel: string;
  expectedMaxModelLen: number | null;
  port: string;
  bind: string;
}

const VAR_RE = /\{\{([A-Za-z0-9_]+)\}\}/g;

function resolveVars(value: string, vars: Record<string, string>, where: string): string {
  return value.replace(VAR_RE, (_, name: string) => {
    if (!(name in vars)) {
      throw new ProfileError(`${where}: unknown variable {{${name}}} (defined: ${Object.keys(vars).sort().join(', ')})`);
    }
    return vars[name];
  });
}

/** The sha256 of the assignment lines, as the node computes it with grep and sha256sum. */
export function bodySha(bodyLines: string[]): string {
  return createHash('sha256').update(bodyLines.join('\n') + '\n').digest('hex');
}

/** The context length that /v1/models must report for this .env. */
export function expectedMaxModelLen(recipeName: string, map: Record<string, string>): number | null {
  if (recipeName === 'qwen38-flash') {
    const v = map.YARN === '1' ? map.YARN_MAX_MODEL_LEN : map.MAX_MODEL_LEN;
    return v ? Number(v) : null;
  }
  return map.MAX_MODEL_LEN ? Number(map.MAX_MODEL_LEN) : null;
}

export function renderEnv(profile: Profile, node: NodeConfig, opts: RenderOptions = {}): Rendered {
  const recipe = node.recipes[profile.recipe];
  need(recipe, `node ${node.name} has no recipe ${profile.recipe} (it has: ${Object.keys(node.recipes).join(', ') || 'none'})`);
  const vars: Record<string, string> = { ...node.vars, ...(recipe.vars ?? {}), recipeDir: recipe.dir };
  const where = `profile ${profile.name} on ${node.name}`;

  const values = new Map<string, string>();
  for (const { key, value } of profile.entries) values.set(key, resolveVars(value, vars, `${where}: ${key}`));

  // EXTRA_DOCKER_ARGS: profile words, then node words, then overlay words.
  const layers: DockerArg[][] = [];
  layers.push(parseDockerArgs(tokenize(values.get('EXTRA_DOCKER_ARGS') ?? '', `${where}: EXTRA_DOCKER_ARGS`), `profile:${profile.name}`));
  const nodeWords = (recipe.dockerArgs ?? []).map((w) => resolveVars(w, vars, `node ${node.name}: dockerArgs`));
  layers.push(parseDockerArgs(tokenize(nodeWords.join(' '), `node ${node.name}: dockerArgs`), `node:${node.name}`));
  if (opts.overlay) layers.push(parseDockerArgs(opts.overlay.words, `overlay:${opts.overlay.dir}`));
  const merged = mergeDockerArgs(layers, { overlayWins: opts.overlayWins, recipe: profile.recipe });
  const dockerWords = formatDockerArgs(merged.args);

  const entries: [string, string][] = [];
  let dockerPlaced = false;
  for (const [key, value] of values) {
    if (key === 'EXTRA_DOCKER_ARGS') {
      entries.push([key, dockerWords.join(' ')]);
      dockerPlaced = true;
    } else {
      entries.push([key, WORD_SPLIT_KEYS.has(key) ? value.split(/\s+/).filter(Boolean).join(' ') : value]);
    }
  }
  if (!dockerPlaced && dockerWords.length) entries.push(['EXTRA_DOCKER_ARGS', dockerWords.join(' ')]);
  for (const [key, value] of Object.entries(node.env)) entries.push([key, value]);

  let bodyLines: string[];
  try {
    bodyLines = entries.map(([k, v]) => formatAssignment(k, v));
  } catch (e) {
    throw new ProfileError(`${where}: ${(e as Error).message}`);
  }
  const sha256 = bodySha(bodyLines);
  const map = Object.fromEntries(entries);
  const now = (opts.now ?? new Date()).toISOString().replace(/\.\d+Z$/, 'Z');
  const overlayLine = opts.overlay ? `${opts.overlay.dir} overlay-sha256=${opts.overlay.sha256}` : 'none';
  const header = [
    '# Rendered by `pulse model up`. The next `pulse model up` on this node replaces this file.',
    `# Edit the profile (runtime/qwen38/profiles/${profile.name}.env in the Pulse repo), not this file.`,
    `# pulse-profile: ${profile.name}`,
    `# pulse-node: ${node.name}`,
    `# pulse-recipe: ${profile.recipe}`,
    `# pulse-overlay: ${overlayLine}`,
    `# pulse-rendered: ${now}`,
    `# pulse-sha256: ${sha256}`,
  ];
  const text = [...header, '', ...bodyLines, ''].join('\n');
  return {
    profile,
    node,
    recipe,
    recipeName: profile.recipe,
    entries,
    map,
    bodyLines,
    sha256,
    overlayLine,
    text,
    dockerArgs: merged.args,
    mounts: merged.args.filter((a): a is MountArg => a.kind === 'mount'),
    replaced: merged.replaced,
    mkdirs: profile.mkdirs.map((d) => resolveVars(d, vars, `${where}: pulse-mkdir`)),
    proofs: profile.proofs,
    overlay: opts.overlay,
    servedModel: map.SERVED_MODEL_NAME ?? 'qwen3.8-flash-next',
    expectedMaxModelLen: expectedMaxModelLen(profile.recipe, map),
    port: map.PORT ?? '8888',
    bind: map.BIND ?? '0.0.0.0',
  };
}

/** The rendered .env for a terminal: EXTRA_DOCKER_ARGS with one pair per line. */
export function displayEnv(r: Rendered): string {
  const out: string[] = [];
  for (const line of r.text.split('\n')) {
    if (line.startsWith('EXTRA_DOCKER_ARGS="')) {
      const words = line.slice('EXTRA_DOCKER_ARGS="'.length, -1).split(' ').filter(Boolean);
      out.push('EXTRA_DOCKER_ARGS="');
      for (const p of pairs(words)) out.push(`    ${p}`);
      out.push('"   (one line in the file)');
    } else {
      out.push(line);
    }
  }
  return out.join('\n').replace(/\n+$/, '');
}

/** Compare two key maps. Word-split keys compare as sorted pairs, because docker does not care about their order. */
export function diffMaps(current: Record<string, string>, next: Record<string, string>): string[] {
  const out: string[] = [];
  const keys = [...new Set([...Object.keys(current), ...Object.keys(next)])].sort();
  const norm = (k: string, v: string | undefined) => {
    if (v === undefined) return undefined;
    if (k === 'EXTRA_DOCKER_ARGS') return pairs(v.split(/\s+/).filter(Boolean)).sort().join(' | ');
    if (WORD_SPLIT_KEYS.has(k)) return v.split(/\s+/).filter(Boolean).join(' ');
    return v;
  };
  for (const k of keys) {
    const a = norm(k, current[k]);
    const b = norm(k, next[k]);
    if (a === b) continue;
    if (a === undefined) out.push(`+ ${k}=${next[k]}`);
    else if (b === undefined) out.push(`- ${k}=${current[k]}`);
    else if (k === 'EXTRA_DOCKER_ARGS') {
      const pa = new Set(a.split(' | '));
      const pb = new Set(b.split(' | '));
      for (const p of pa) if (!pb.has(p)) out.push(`- EXTRA_DOCKER_ARGS ${p}`);
      for (const p of pb) if (!pa.has(p)) out.push(`+ EXTRA_DOCKER_ARGS ${p}`);
    } else out.push(`~ ${k}: ${current[k]} -> ${next[k]}`);
  }
  return out;
}
