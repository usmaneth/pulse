// Bun FFI bindings to the native Pulse CUDA engine (bin/libpulse_engine.so).
//
// HONESTY CONTRACT: this module never returns a performance number it did not
// measure. If the native engine is not loaded, step() throws. There is no
// fallback that substitutes a plausible-looking constant.

// @ts-ignore
import { dlopen, FFIType, ptr } from 'bun:ffi';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export interface NativeStepResult {
  tokens: number[];
  /** Real GPU milliseconds measured by cudaEventElapsedTime inside the engine. */
  stepMs: number;
  /** Derived from the measured stepMs; never a constant. */
  rateToksSec: number;
}

// bin/libpulse_engine.so sits two levels above the compiled dist/server/
// output (repo root, then bin/). PULSE_ENGINE_LIB overrides the path.
const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const LIB_PATH = process.env.PULSE_ENGINE_LIB ?? join(REPO_ROOT, 'bin', 'libpulse_engine.so');

export class NativePulseEngine {
  private lib: ReturnType<typeof dlopen> | null = null;
  private enginePtr: unknown = null;
  private outBuf = new Int32Array(32);
  private stepMsBuf = new Float64Array(1);
  private loadError: string | null = null;

  constructor(format: number = 0, arch: number = 0, k: number = 5) {
    try {
      this.lib = dlopen(LIB_PATH, {
        pulse_engine_create: {
          args: [FFIType.i32, FFIType.i32, FFIType.i32],
          returns: FFIType.ptr,
        },
        pulse_engine_destroy: {
          args: [FFIType.ptr],
          returns: FFIType.void,
        },
        pulse_engine_step: {
          args: [FFIType.ptr, FFIType.i32, FFIType.ptr, FFIType.i32, FFIType.ptr],
          returns: FFIType.i32,
        },
        pulse_engine_last_rate: {
          args: [FFIType.ptr],
          returns: FFIType.f64,
        },
      });

      // format: 0 = PQ2_0 ternary, 1 = NVFP4, 2 = Q4_K_M
      // arch:   0 = dense 27B, 1 = sparse MoE 35B
      const handle = this.lib.symbols.pulse_engine_create(format, arch, k);
      if (!handle) {
        throw new Error('pulse_engine_create returned null (engine initialize() failed)');
      }
      this.enginePtr = handle;
      console.log('[NativePulseEngine] native engine loaded from', LIB_PATH);
    } catch (err) {
      this.loadError = err instanceof Error ? err.message : String(err);
      console.warn('[NativePulseEngine] native engine unavailable:', this.loadError);
      this.lib = null;
      this.enginePtr = null;
    }
  }

  get isAvailable(): boolean {
    return this.enginePtr !== null;
  }

  get error(): string | null {
    return this.loadError;
  }

  step(k: number = 5): NativeStepResult {
    if (!this.enginePtr || !this.lib) {
      throw new Error(
        'Pulse native engine is not loaded; no measurement is available. ' +
          `Build ${LIB_PATH} and ensure the pulse_engine_* symbols are exported. ` +
          `Load error: ${this.loadError ?? 'unknown'}`
      );
    }

    const count = this.lib.symbols.pulse_engine_step(
      this.enginePtr,
      k,
      ptr(this.outBuf),
      this.outBuf.length,
      ptr(this.stepMsBuf)
    );

    const stepMs = this.stepMsBuf[0];
    const rateToksSec = this.lib.symbols.pulse_engine_last_rate(this.enginePtr);

    return {
      tokens: Array.from(this.outBuf.subarray(0, Math.max(0, count))),
      stepMs,
      rateToksSec,
    };
  }

  destroy(): void {
    if (this.enginePtr && this.lib) {
      this.lib.symbols.pulse_engine_destroy(this.enginePtr);
      this.enginePtr = null;
    }
  }
}
