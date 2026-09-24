// EXTRA_DOCKER_ARGS: parse, merge and check.
//
// start.sh puts $EXTRA_DOCKER_ARGS without quotes into a generated bash script
// (the `docker run` heredoc). The shell splits the value into words there, and
// it expands any shell syntax in it. So Pulse accepts only `-e NAME=VALUE` and
// `-v SRC:DST[:ro|rw]` pairs, and every word must match TOKEN_RE.

export const TOKEN_RE = /^[A-Za-z0-9_.\/:=,@+%-]+$/;
const ENV_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export interface EnvArg {
  kind: 'env';
  name: string;
  value: string;
  origin: string;
}

export interface MountArg {
  kind: 'mount';
  src: string;
  dst: string;
  mode?: 'ro' | 'rw';
  origin: string;
}

export type DockerArg = EnvArg | MountArg;

export class DockerArgsError extends Error {}

/**
 * Mount destinations that the qwen38-flash start.sh sets itself. A second
 * mount on the same destination makes `docker run` fail.
 */
const VLLM_PKG = '/usr/local/lib/python3.12/dist-packages/vllm';
export const RESERVED_MOUNTS: Record<string, string[]> = {
  'qwen38-flash': [
    '/root/.cache/huggingface',
    '/root/.cache/vllm',
    '/root/chat_template.jinja',
    '/root/draft_vocab.txt',
    `${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/ple_layer.py`,
    `${VLLM_PKG}/model_executor/layers/quantization/modelopt.py`,
    `${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/ops/qsa.py`,
    `${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/qsa.py`,
    `${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/mtp.py`,
    `${VLLM_PKG}/model_executor/layers/ple_offload_layer.py`,
    `${VLLM_PKG}/v1/ple_offload/connector.py`,
    `${VLLM_PKG}/v1/ple_offload/worker.py`,
    `${VLLM_PKG}/v1/ple_offload/protocol.py`,
  ],
};

/** Environment names that start.sh sets itself with -e. */
export const RESERVED_ENV: Record<string, string[]> = {
  'qwen38-flash': [
    'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'VLLM_PLE_CPU_OFFLOAD', 'VLLM_PLE_PACKED_TABLE_DIR',
    'VLLM_PLE_OFFLOAD_STEP_TIMEOUT', 'MAX_JOBS', 'FLASHINFER_NVCC_THREADS', 'HF_HOME', 'HF_TOKEN',
    'VLLM_MTP_DRAFT_VOCAB', 'VLLM_QSA_DET_TOPK', 'VLLM_MOE_DET_FINALIZE', 'VLLM_GDN_DECODE_KERNEL',
  ],
};

/** Split a value into words and check each word. */
export function tokenize(value: string, origin = 'EXTRA_DOCKER_ARGS'): string[] {
  const words = value.split(/\s+/).filter(Boolean);
  for (const w of words) {
    if (!TOKEN_RE.test(w)) {
      throw new DockerArgsError(`${origin}: unsafe word ${JSON.stringify(w)} (allowed: letters, digits and _ . / : = , @ + % -)`);
    }
  }
  return words;
}

/** Parse words into -e and -v pairs. */
export function parseDockerArgs(words: string[], origin: string): DockerArg[] {
  const out: DockerArg[] = [];
  for (let i = 0; i < words.length; i += 2) {
    const flag = words[i];
    const arg = words[i + 1];
    if (flag !== '-e' && flag !== '-v') {
      throw new DockerArgsError(`${origin}: unsupported docker argument ${JSON.stringify(flag)} (only -e NAME=VALUE and -v SRC:DST[:ro] pairs)`);
    }
    if (arg === undefined) throw new DockerArgsError(`${origin}: ${flag} has no value`);
    if (!TOKEN_RE.test(arg)) throw new DockerArgsError(`${origin}: unsafe word ${JSON.stringify(arg)}`);
    if (flag === '-e') {
      const eq = arg.indexOf('=');
      const name = eq > 0 ? arg.slice(0, eq) : '';
      if (!ENV_NAME_RE.test(name)) {
        throw new DockerArgsError(`${origin}: -e ${arg}: expected NAME=VALUE (a bare NAME would copy the value from the start.sh environment)`);
      }
      out.push({ kind: 'env', name, value: arg.slice(eq + 1), origin });
    } else {
      const parts = arg.split(':');
      if (parts.length < 2 || parts.length > 3) throw new DockerArgsError(`${origin}: -v ${arg}: expected SRC:DST or SRC:DST:ro`);
      const [src, dst, mode] = parts;
      if (!src.startsWith('/') || !dst.startsWith('/')) {
        throw new DockerArgsError(`${origin}: -v ${arg}: SRC and DST must be absolute paths`);
      }
      if (mode !== undefined && mode !== 'ro' && mode !== 'rw') {
        throw new DockerArgsError(`${origin}: -v ${arg}: the mode must be ro or rw`);
      }
      out.push({ kind: 'mount', src, dst: dst.replace(/\/+$/, '') || '/', mode: mode as MountArg['mode'], origin });
    }
  }
  return out;
}

export interface MergeOptions {
  /** Let an overlay mount or env replace a profile or node entry with the same target. */
  overlayWins?: boolean;
  /** The recipe name, for the reserved destinations. */
  recipe?: string;
}

export interface MergeResult {
  args: DockerArg[];
  /** One line for each entry that an overlay replaced. */
  replaced: string[];
}

function describe(a: DockerArg): string {
  return a.kind === 'mount' ? `${a.src} (${a.origin})` : `${a.name}=${a.value} (${a.origin})`;
}

/** Merge the layers in order. The same mount destination or env name twice is an error. */
export function mergeDockerArgs(layers: DockerArg[][], opts: MergeOptions = {}): MergeResult {
  const reservedMounts = new Set(opts.recipe ? RESERVED_MOUNTS[opts.recipe] ?? [] : []);
  const reservedEnv = new Set(opts.recipe ? RESERVED_ENV[opts.recipe] ?? [] : []);
  const args: DockerArg[] = [];
  const index = new Map<string, number>();
  const replaced: string[] = [];
  for (const layer of layers) {
    for (const a of layer) {
      const id = a.kind === 'mount' ? `mount:${a.dst}` : `env:${a.name}`;
      if (a.kind === 'mount' && reservedMounts.has(a.dst)) {
        throw new DockerArgsError(`mount destination ${a.dst} from ${a.origin}: start.sh of ${opts.recipe} mounts this path itself`);
      }
      if (a.kind === 'env' && reservedEnv.has(a.name)) {
        throw new DockerArgsError(`-e ${a.name} from ${a.origin}: start.sh of ${opts.recipe} sets this variable itself`);
      }
      const at = index.get(id);
      if (at === undefined) {
        index.set(id, args.length);
        args.push(a);
        continue;
      }
      const old = args[at];
      const what = a.kind === 'mount' ? `mount destination ${a.dst}` : `env ${a.name}`;
      if (opts.overlayWins && a.origin.startsWith('overlay') && !old.origin.startsWith('overlay')) {
        args[at] = a;
        replaced.push(`${what}: ${describe(old)} replaced by ${describe(a)}`);
        continue;
      }
      throw new DockerArgsError(`duplicate ${what}: ${describe(old)} and ${describe(a)}`
        + (a.origin.startsWith('overlay') ? '. Pass --overlay-wins to let the overlay replace it.' : ''));
    }
  }
  return { args, replaced };
}

/** Words for EXTRA_DOCKER_ARGS, in merge order. */
export function formatDockerArgs(args: DockerArg[]): string[] {
  const out: string[] = [];
  for (const a of args) {
    if (a.kind === 'env') out.push('-e', `${a.name}=${a.value}`);
    else out.push('-v', `${a.src}:${a.dst}${a.mode ? `:${a.mode}` : ''}`);
  }
  return out;
}

/** Word pairs such as "-v a:b", for display and for order-free comparison. */
export function pairs(words: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < words.length; i += 2) out.push(`${words[i]} ${words[i + 1] ?? ''}`.trim());
  return out;
}
