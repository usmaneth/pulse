// Run node scripts: locally with `bash -s`, or over ssh with `bash -s` on the node.
//
// The script goes to bash on stdin. The runner wraps it in a function and
// calls the function with stdin from /dev/null. So bash reads the complete
// script before the first command runs, and no command in the script can read
// the rest of the script from stdin.

import { spawn } from 'node:child_process';
import type { NodeConfig } from './profiles.js';

export interface RunResult {
  code: number;
  stdout: string;
  stderr: string;
  /** Lines that start with "@pulse ", without the prefix. */
  records: string[];
}

export interface RunOptions {
  /** Print operator output (all lines except @pulse records) while the script runs. */
  stream?: boolean;
  /** Also print the @pulse records. */
  verbose?: boolean;
  timeoutS?: number;
}

export interface Runner {
  run(script: string, opts?: RunOptions): Promise<RunResult>;
}

export const SSH_OPTIONS = [
  '-o', 'BatchMode=yes',
  '-o', 'ConnectTimeout=10',
  '-o', 'ServerAliveInterval=15',
  '-o', 'ServerAliveCountMax=4',
];

export function wrapScript(script: string): string {
  return `__pulse_main() {\n${script.replace(/\n$/, '')}\n}\n__pulse_main </dev/null\n`;
}

export function records(stdout: string): string[] {
  return stdout.split('\n').filter((l) => l.startsWith('@pulse ')).map((l) => l.slice('@pulse '.length));
}

/** The command and arguments that run a script on the node. */
export function commandFor(node: NodeConfig): [string, string[]] {
  if (node.host === 'local') return ['bash', ['-s']];
  return ['ssh', [...SSH_OPTIONS, node.ssh!, 'bash -s']];
}

export class ExecRunner implements Runner {
  constructor(
    private readonly node: NodeConfig,
    private readonly out: NodeJS.WritableStream = process.stdout,
    private readonly env: NodeJS.ProcessEnv = process.env,
  ) {}

  run(script: string, opts: RunOptions = {}): Promise<RunResult> {
    const [cmd, args] = commandFor(this.node);
    return new Promise((resolve, reject) => {
      const child = spawn(cmd, args, { stdio: ['pipe', 'pipe', 'pipe'], env: this.env });
      let stdout = '';
      let stderr = '';
      let pending = '';
      let timer: NodeJS.Timeout | undefined;
      if (opts.timeoutS) {
        timer = setTimeout(() => {
          stderr += `\npulse: the script did not finish in ${opts.timeoutS} s; the local ${cmd} process was stopped\n`;
          child.kill('SIGTERM');
        }, opts.timeoutS * 1000);
      }
      const emit = (line: string) => {
        if (!opts.stream) return;
        if (line.startsWith('@pulse ') && !opts.verbose) return;
        this.out.write(`    ${line}\n`);
      };
      child.stdout.on('data', (buf: Buffer) => {
        const s = buf.toString();
        stdout += s;
        pending += s;
        const lines = pending.split('\n');
        pending = lines.pop() ?? '';
        lines.forEach(emit);
      });
      child.stderr.on('data', (buf: Buffer) => {
        const s = buf.toString();
        stderr += s;
        if (opts.stream) for (const l of s.split('\n').filter(Boolean)) this.out.write(`    ${l}\n`);
      });
      child.on('error', (e) => {
        if (timer) clearTimeout(timer);
        reject(e);
      });
      child.on('close', (code) => {
        if (timer) clearTimeout(timer);
        if (pending) emit(pending);
        resolve({ code: code ?? 1, stdout, stderr, records: records(stdout) });
      });
      child.stdin.on('error', () => { /* the child exited early; close reports it */ });
      child.stdin.end(wrapScript(script));
    });
  }
}

export interface RecordedCall {
  script: string;
  opts: RunOptions;
}

/** A runner for tests: it records each script and returns canned output. */
export class RecordingRunner implements Runner {
  readonly calls: RecordedCall[] = [];

  constructor(private readonly respond: (script: string, index: number) => { code?: number; stdout?: string } = () => ({})) {}

  async run(script: string, opts: RunOptions = {}): Promise<RunResult> {
    this.calls.push({ script, opts });
    const r = this.respond(script, this.calls.length - 1);
    const stdout = r.stdout ?? '';
    return { code: r.code ?? 0, stdout, stderr: '', records: records(stdout) };
  }
}
