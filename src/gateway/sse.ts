// Server-sent events parser for backend chat-completions streams.

/**
 * Yield the `data:` payload of each SSE frame. The parser accepts frames that
 * are split across chunks and across UTF-8 boundaries, CRLF line ends, comment
 * lines and multi-line data fields.
 */
export async function* sseData(
  body: ReadableStream<Uint8Array>,
  maxFrameBytes = 8 * 1024 * 1024,
): AsyncGenerator<string> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let match: RegExpExecArray | null;
      while ((match = /\r?\n\r?\n/.exec(buffer))) {
        const frame = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const lines = frame.split(/\r?\n/).filter((line) => line.startsWith('data:'));
        if (lines.length) yield lines.map((line) => line.slice(5).replace(/^ /, '')).join('\n');
      }
      if (buffer.length > maxFrameBytes) throw new Error('backend SSE frame exceeds the size limit');
      if (done) break;
    }
    // A last frame without the blank line after it is still a frame.
    const lines = buffer.split(/\r?\n/).filter((line) => line.startsWith('data:'));
    if (lines.length) yield lines.map((line) => line.slice(5).replace(/^ /, '')).join('\n');
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
