// Structured logs: one JSON object per line on stdout (errors on stderr).
// journald keeps each line as one record, and `jq` can filter them.

export type Level = 'debug' | 'info' | 'warn' | 'error';

const ORDER: Record<Level, number> = { debug: 10, info: 20, warn: 30, error: 40 };

let threshold: Level = (process.env.PULSE_LOG_LEVEL as Level) in ORDER
  ? (process.env.PULSE_LOG_LEVEL as Level)
  : 'info';

export function setLogLevel(level: Level): void {
  if (level in ORDER) threshold = level;
}

export type LogSink = (level: Level, line: string) => void;

const stdSink: LogSink = (level, line) => {
  if (level === 'error' || level === 'warn') process.stderr.write(line + '\n');
  else process.stdout.write(line + '\n');
};

let sink: LogSink = stdSink;

/** Send the log lines to another function, for example in tests. null restores stdout and stderr. */
export function setLogSink(next: LogSink | null): void {
  sink = next ?? stdSink;
}

export function log(level: Level, msg: string, fields: Record<string, unknown> = {}): void {
  if (ORDER[level] < ORDER[threshold]) return;
  sink(level, JSON.stringify({ ts: new Date().toISOString(), level, msg, ...fields }));
}
