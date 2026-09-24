// Parse and write the recipe .env format.
//
// The recipe start.sh reads its .env with `source .env`, so the file is bash.
// Pulse accepts only a small subset of bash that it can parse without a shell:
//
//   KEY=value                  value from SAFE_VALUE_RE only
//   KEY="value"                no $, backtick, backslash or double quote
//   KEY="                      multi-line value, only for WORD_SPLIT_KEYS
//     word word
//   "
//   # comment                  also after a value, with a space before the #
//   # pulse-name: value        a Pulse directive (profiles only)
//
// The last assignment of a key wins, as in bash.

export const KEY_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const SAFE_VALUE_RE = /^[A-Za-z0-9_.\/:=,@+%-]*$/;
const DIRECTIVE_RE = /^#\s*pulse-([a-z][a-z0-9-]*):\s?(.*)$/;
const UNSAFE_CHARS = ['$', '`', '\\', '"', '\n', '\r', '\0'];

/** start.sh splits these values into words, so Pulse can join lines with one space. */
export const WORD_SPLIT_KEYS = new Set(['EXTRA_DOCKER_ARGS', 'EXTRA_VLLM_ARGS']);

export interface Assignment {
  key: string;
  value: string;
  line: number;
}

export interface Directive {
  name: string;
  value: string;
  line: number;
}

export interface ParsedEnv {
  assignments: Assignment[];
  directives: Directive[];
}

export class EnvFileError extends Error {}

/** Return the first character that is not safe inside bash double quotes, or null. */
export function unsafeChar(value: string): string | null {
  for (const ch of UNSAFE_CHARS) if (value.includes(ch)) return ch;
  return null;
}

function describeChar(ch: string): string {
  return ({ '\n': 'newline', '\r': 'carriage return', '\0': 'NUL' } as Record<string, string>)[ch] ?? `'${ch}'`;
}

/** Check a value that goes into KEY="value". Throws with the key name. */
export function checkValue(key: string, value: string): void {
  const bad = unsafeChar(value);
  if (bad !== null) {
    throw new EnvFileError(`${key}: the value contains ${describeChar(bad)}, which is not safe in the recipe .env`);
  }
}

/** Format one line of the rendered .env. */
export function formatAssignment(key: string, value: string): string {
  if (!KEY_RE.test(key)) throw new EnvFileError(`bad key name: ${JSON.stringify(key)}`);
  checkValue(key, value);
  return `${key}="${value}"`;
}

function normalizeWords(value: string): string {
  return value.split(/\s+/).filter(Boolean).join(' ');
}

function checkTail(tail: string, where: string): void {
  if (tail === '' || /^\s+(#.*)?$/.test(tail)) return;
  throw new EnvFileError(`${where}: unexpected text after the value: ${JSON.stringify(tail.trim())}`);
}

/** Parse the bash subset described at the top of this file. */
export function parseEnv(text: string, source = '.env'): ParsedEnv {
  const lines = text.split('\n');
  const assignments: Assignment[] = [];
  const directives: Directive[] = [];
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i].replace(/\r$/, '');
    const where = `${source}:${i + 1}`;
    const line = raw.trim();
    if (line === '') continue;
    if (line.startsWith('#')) {
      const d = DIRECTIVE_RE.exec(line);
      if (d) directives.push({ name: d[1], value: d[2].trim(), line: i + 1 });
      continue;
    }
    const eq = raw.indexOf('=');
    const key = eq > 0 ? raw.slice(0, eq) : '';
    if (!KEY_RE.test(key)) {
      throw new EnvFileError(`${where}: expected KEY=value, a comment or a blank line`);
    }
    const rest = raw.slice(eq + 1);
    let value: string;
    if (rest.startsWith('"')) {
      const close = rest.indexOf('"', 1);
      if (close >= 0) {
        value = rest.slice(1, close);
        checkTail(rest.slice(close + 1), where);
      } else {
        if (!WORD_SPLIT_KEYS.has(key)) {
          throw new EnvFileError(`${where}: ${key} has no closing double quote (only ${[...WORD_SPLIT_KEYS].join(' and ')} can span lines)`);
        }
        const parts = [rest.slice(1)];
        let j = i + 1;
        for (; j < lines.length; j++) {
          const next = lines[j].replace(/\r$/, '');
          const c = next.indexOf('"');
          if (c >= 0) {
            parts.push(next.slice(0, c));
            checkTail(next.slice(c + 1), `${source}:${j + 1}`);
            break;
          }
          parts.push(next);
        }
        if (j >= lines.length) throw new EnvFileError(`${where}: ${key} has no closing double quote`);
        value = parts.join(' ');
        i = j;
      }
      const bad = unsafeChar(value);
      if (bad !== null) throw new EnvFileError(`${where}: ${key} contains ${describeChar(bad)}`);
    } else {
      const m = /^(\S*)(.*)$/.exec(rest)!;
      value = m[1];
      checkTail(m[2], where);
      if (!SAFE_VALUE_RE.test(value)) {
        throw new EnvFileError(`${where}: ${key}: an unquoted value may use only letters, digits and _ . / : = , @ + % - (quote it with double quotes)`);
      }
    }
    if (WORD_SPLIT_KEYS.has(key)) value = normalizeWords(value);
    assignments.push({ key, value, line: i + 1 });
  }
  return { assignments, directives };
}
