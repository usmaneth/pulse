// Node state files and the gateway backend fragment.
//
// Pulse keeps its runtime state outside the repo:
//
//   $PULSE_STATE_DIR (default $XDG_STATE_HOME/pulse/runtime or ~/.local/state/pulse/runtime)
//     nodes/<node>.json        the last up or down that pulse did on the node
//     gateway-backends.json    the qwen38 model with one endpoint per node
//
// gateway-backends.json is a partial gateway config. The gateway reads it when
// PULSE_GATEWAY_CONFIG or --config names it. The gateway reads its config only
// at start, and PULSE_QWEN_BACKENDS replaces the endpoints of the file. Pulse
// never restarts or contacts the gateway.

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { DEFAULT_QWEN_MODEL, defaultConfig, validateConfig } from '../gateway/config.js';
import type { GatewayConfig, EndpointConfig } from '../gateway/config.js';
import type { NodeConfig, NodesFile } from './profiles.js';

export type NodeRunState = 'starting' | 'up' | 'failed' | 'down';

export interface NodeState {
  node: string;
  state: NodeRunState;
  updatedAt: string;
  profile?: string;
  recipe?: string;
  envSha256?: string;
  servedModel?: string;
  maxModelLen?: number;
  bind?: string;
  port?: string;
  kv?: string;
  proofs?: { text: string; ok: boolean }[];
  backup?: string | null;
  startLog?: string;
  overlay?: { dir: string; manifestSha256: string | null; manifest: unknown } | null;
  replaced?: string[];
  readySeconds?: number;
  error?: string;
}

export function stateDir(env: NodeJS.ProcessEnv = process.env): string {
  if (env.PULSE_STATE_DIR) return path.resolve(env.PULSE_STATE_DIR);
  const base = env.XDG_STATE_HOME || path.join(env.HOME || os.homedir(), '.local', 'state');
  return path.join(base, 'pulse', 'runtime');
}

export function nodeStatePath(dir: string, node: string): string {
  return path.join(dir, 'nodes', `${node}.json`);
}

export function fragmentPath(dir: string): string {
  return path.join(dir, 'gateway-backends.json');
}

function writeAtomic(file: string, text: string): void {
  mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp-${process.pid}`;
  writeFileSync(tmp, text);
  renameSync(tmp, file);
}

export function readNodeState(dir: string, node: string): NodeState | undefined {
  const file = nodeStatePath(dir, node);
  if (!existsSync(file)) return undefined;
  return JSON.parse(readFileSync(file, 'utf8')) as NodeState;
}

export function writeNodeState(dir: string, state: NodeState): string {
  const file = nodeStatePath(dir, state.node);
  writeAtomic(file, JSON.stringify(state, null, 2) + '\n');
  return file;
}

function isLoopback(bind: string): boolean {
  return bind === '127.0.0.1' || bind === '::1' || bind === 'localhost';
}

/**
 * True when the gateway (which runs on the local node) can reach the node's
 * server: the node is local, or its server does not listen on loopback only.
 */
export function gatewayCanReach(node: NodeConfig, bind: string | undefined): boolean {
  if (node.host === 'local') return true;
  return !isLoopback(bind ?? node.env.BIND ?? '0.0.0.0');
}

export interface FragmentResult {
  fragment: Partial<GatewayConfig>;
  warnings: string[];
  endpoints: { name: string; enabled: boolean; reason: string }[];
}

/** Build the fragment from the nodes file and the node states. */
export function buildFragment(nodes: NodesFile, states: Record<string, NodeState | undefined>): FragmentResult {
  const warnings: string[] = [];
  const endpoints: EndpointConfig[] = [];
  const why: FragmentResult['endpoints'] = [];
  const served = new Set<string>();
  for (const node of Object.values(nodes.nodes)) {
    const st = states[node.name];
    if (st?.servedModel && st.state === 'up') served.add(st.servedModel);
    const reach = gatewayCanReach(node, st?.bind);
    let enabled = reach;
    let reason = reach ? (st ? `state ${st.state}` : 'no pulse state; the gateway health check decides') : `BIND ${st?.bind ?? node.env.BIND} is loopback and the node is remote`;
    if (st && st.state !== 'up') {
      enabled = false;
      reason = `state ${st.state}`;
    }
    endpoints.push({ name: node.name, baseUrl: node.gatewayUrl, enabled });
    why.push({ name: node.name, enabled, reason });
  }
  if (endpoints.length && !endpoints.some((e) => e.enabled)) {
    endpoints[0].enabled = true;
    why[0].enabled = true;
    why[0].reason += '; kept on because the gateway refuses a model with no enabled endpoint';
    warnings.push(`no endpoint is usable; ${endpoints[0].name} stays enabled because the gateway refuses a model with every endpoint disabled`);
  }
  if (served.size > 1) {
    warnings.push(`the nodes serve different model names (${[...served].join(', ')}); the gateway sends one upstream name to every endpoint`);
  }
  const upstreamModel = [...served][0] ?? DEFAULT_QWEN_MODEL;
  const fragment: Partial<GatewayConfig> = {
    models: [{ id: DEFAULT_QWEN_MODEL, upstreamModel, profile: 'qwen38', endpoints }],
  };
  // The same check that the gateway does at start. Validate a copy, because
  // validateConfig normalizes the URLs in place.
  validateConfig({ ...defaultConfig(), ...JSON.parse(JSON.stringify(fragment)) as Partial<GatewayConfig> });
  return { fragment, warnings, endpoints: why };
}

/** Read every node state and write gateway-backends.json. */
export function writeFragment(dir: string, nodes: NodesFile): FragmentResult & { file: string } {
  const states: Record<string, NodeState | undefined> = {};
  for (const name of Object.keys(nodes.nodes)) states[name] = readNodeState(dir, name);
  const result = buildFragment(nodes, states);
  const file = fragmentPath(dir);
  writeAtomic(file, JSON.stringify(result.fragment, null, 2) + '\n');
  return { ...result, file };
}

export function fragmentHelp(file: string): string[] {
  return [
    `gateway fragment: ${file}`,
    'The gateway reads this file only when it starts. To use it:',
    `  1. Set PULSE_GATEWAY_CONFIG=${file} (or start the gateway with --config ${file}).`,
    '  2. Remove PULSE_QWEN_BACKENDS from ~/.config/pulse/qwen38.env. It replaces the endpoints of the file.',
    '  3. Restart the gateway: systemctl --user restart pulse-qwen38. Pulse does not do this for you.',
  ];
}
