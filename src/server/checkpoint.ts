/**
 * Slot checkpoint manager.
 *
 * Problem this solves (measured, see bench/RESULTS.md Round 20b/21):
 *   append to context      ->   242 ms  (llama.cpp's own prompt cache handles it)
 *   edit anything earlier  ->  8702 ms  (full re-prefill)
 *
 * llama.cpp's mitigation for divergence, `--cache-reuse`, is silently disabled
 * on this model: it needs K-shifting, and `llama_kv_cache::get_can_shift()`
 * returns false when `n_pos_per_embd() > 1`. Bonsai 2 uses mRoPE
 * (rope.dimension_sections = [11,11,10]), so it is refused.
 *
 * But `POST /slots/:id?action=save|restore` dumps state verbatim without moving
 * positions, so it is unaffected. Measured at 8k: restore 90.8 ms vs an 8702 ms
 * re-prefill, i.e. 96x.
 *
 * Strategy: keep checkpoints keyed by a prefix hash. On a request, find the
 * longest stored checkpoint whose content is a prefix of the incoming prompt and
 * restore it. llama.cpp's normal prompt-cache logic then sees the slot already
 * holds that prefix and prefills only the divergent tail.
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readdirSync, statSync, unlinkSync } from 'node:fs';
import path from 'node:path';

export interface CheckpointEntry {
  /** hash of the prefix text this checkpoint holds */
  hash: string;
  /** the prefix text itself, needed for prefix matching */
  text: string;
  filename: string;
  bytes: number;
  createdAt: number;
  lastUsedAt: number;
}

export interface CheckpointStats {
  entries: number;
  bytes: number;
  saves: number;
  restores: number;
  hits: number;
  misses: number;
  lastRestoreMs: number | null;
  lastSaveMs: number | null;
}

const sha = (s: string) => createHash('sha256').update(s).digest('hex').slice(0, 16);

export class CheckpointManager {
  private entries = new Map<string, CheckpointEntry>();
  private stats: CheckpointStats = {
    entries: 0, bytes: 0, saves: 0, restores: 0, hits: 0, misses: 0,
    lastRestoreMs: null, lastSaveMs: null,
  };

  constructor(
    private backendUrl: string,
    private savePath: string,
    /** don't bother checkpointing below this many characters - the restore costs more than the prefill saves */
    private minChars = 4000,
    /** retention budget; each checkpoint is ~157 MB + 64 KB/token */
    private maxBytes = 20 * 1024 * 1024 * 1024,
  ) {
    try { mkdirSync(savePath, { recursive: true }); } catch { /* already exists */ }
  }

  getStats(): CheckpointStats {
    return { ...this.stats, entries: this.entries.size };
  }

  /** Longest stored checkpoint whose text is a prefix of `prompt`. */
  findBestPrefix(prompt: string): CheckpointEntry | null {
    let best: CheckpointEntry | null = null;
    for (const e of this.entries.values()) {
      if (e.text.length > prompt.length) continue;
      if (!prompt.startsWith(e.text)) continue;
      if (!best || e.text.length > best.text.length) best = e;
    }
    return best;
  }

  private async slotAction(action: 'save' | 'restore', filename: string, idSlot = 0) {
    const res = await fetch(`${this.backendUrl}/slots/${idSlot}?action=${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename }),
    });
    if (!res.ok) throw new Error(`slot ${action} failed: ${res.status} ${await res.text()}`);
    return (await res.json()) as {
      n_saved?: number; n_restored?: number; n_written?: number; n_read?: number;
      timings: { save_ms?: number; restore_ms?: number };
    };
  }

  /**
   * Restore the best matching checkpoint if one exists.
   * Returns the entry restored, or null when nothing matched.
   */
  async restoreBest(prompt: string, idSlot = 0): Promise<CheckpointEntry | null> {
    const best = this.findBestPrefix(prompt);
    if (!best) { this.stats.misses++; return null; }
    try {
      const r = await this.slotAction('restore', best.filename, idSlot);
      this.stats.restores++;
      this.stats.hits++;
      this.stats.lastRestoreMs = r.timings.restore_ms ?? null;
      best.lastUsedAt = Date.now();
      return best;
    } catch {
      // A checkpoint file can vanish underneath us; drop it and fall through.
      this.entries.delete(best.hash);
      this.stats.misses++;
      return null;
    }
  }

  /** Checkpoint the slot's current state as holding `prompt`. */
  async save(prompt: string, idSlot = 0): Promise<CheckpointEntry | null> {
    if (prompt.length < this.minChars) return null;
    const hash = sha(prompt);
    if (this.entries.has(hash)) return this.entries.get(hash)!;
    const filename = `ckpt-${hash}.bin`;
    try {
      const r = await this.slotAction('save', filename, idSlot);
      const entry: CheckpointEntry = {
        hash, text: prompt, filename,
        bytes: r.n_written ?? 0,
        createdAt: Date.now(), lastUsedAt: Date.now(),
      };
      this.entries.set(hash, entry);
      this.stats.saves++;
      this.stats.bytes += entry.bytes;
      this.stats.lastSaveMs = r.timings.save_ms ?? null;
      this.evictIfNeeded();
      return entry;
    } catch {
      return null;
    }
  }

  /** Evict least-recently-used checkpoints until inside the retention budget. */
  private evictIfNeeded(): void {
    if (this.stats.bytes <= this.maxBytes) return;
    const byAge = [...this.entries.values()].sort((a, b) => a.lastUsedAt - b.lastUsedAt);
    for (const e of byAge) {
      if (this.stats.bytes <= this.maxBytes) break;
      this.entries.delete(e.hash);
      this.stats.bytes -= e.bytes;
      try { unlinkSync(path.join(this.savePath, e.filename)); } catch { /* best effort */ }
    }
  }

  /** Remove every checkpoint this manager created. */
  clear(): void {
    for (const e of this.entries.values()) {
      try { unlinkSync(path.join(this.savePath, e.filename)); } catch { /* best effort */ }
    }
    this.entries.clear();
    this.stats.bytes = 0;
  }
}
