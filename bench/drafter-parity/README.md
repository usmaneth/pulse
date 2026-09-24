# Draft RoPE parity

This experiment changes only the draft GGUF metadata. It does not change target parameters or tensor bytes.

The trainer uses YaRN factor 32, original context 8192, base 10000000, beta_fast 32, and beta_slow 1.
The existing v2 GGUF omits the YaRN metadata.
The loader therefore selects frequency scale 1 and disables the YaRN extrapolation factor.

The variant sets four metadata values:

- `dflash.rope.scaling.type`: `yarn`
- `dflash.rope.scaling.factor`: `32.0`
- `dflash.rope.scaling.original_context_length`: `8192`
- `dflash.rope.scaling.attn_factor`: `1.0`

The last value is a multiplier. The ggml operator already applies `1 + 0.1 * log(32)`.
A metadata value of 1.34657359 would apply that magnitude twice.
The context defaults supply beta_fast 32 and beta_slow 1.
Do not pass global target RoPE overrides.

## CPU checks

Run from this directory:

```sh
make -j1
python3 compare_rope.py --trainer ~/Bonsai-demo/dflash-training/v2/train_dspark_v2.py --probe ./artifacts/rope-probe --output /tmp/rope-comparison.json
PYTHONPATH=~/Bonsai-demo/llama.cpp/gguf-py python3 -m unittest test_metadata.py
```

The probe calls the actual ggml CPU RoPE operator. The Python comparison calls the actual trainer implementation.
The comparison covers positions 0, 511, 4095, 8191, 32767, and 65535.
It uses one deterministic 128-element vector at every position.
It does not test CUDA, attention outputs, model quality, or acceptance.

## Artifact conversion

```sh
python3 metadata_variant.py --gguf-py ~/Bonsai-demo/llama.cpp/gguf-py --source SOURCE.gguf --destination VARIANT.gguf --manifest VARIANT.manifest.json
```

Use a new destination and manifest path.
The utility rejects an existing destination.
It checks the architecture and frequency base.
It preserves each tensor name, shape, type, and raw payload hash.
It rejects metadata changes outside the four keys.
The manifest records complete file hashes and each tensor payload hash.
The CPU test also checks deterministic output and an unchanged source file.

## GPU experiment

The orchestrator must grant an exclusive GPU slot first.
Use the same patched server, target, prompts, sampler, context capacity, and draft depth for both variants.
Use a clean process for each variant.
Record the loaded draft context parameters from the server log.
Measure acceptance with `timings.draft_n_accepted / timings.draft_n`.
Record decode time, total time, output tokens, and the actual task result separately.
Use short tool traces first, then identical 8K and 32K prompts.
Use 64K only when the target capacity includes output and tool turns.
Repeat each prompt three times in alternating variant order.
A RoPE match does not establish an acceptance or speed improvement.

## Bounded live gate

Prepare the three-request gate without GPU use:

```sh
python3 live_acceptance.py --output /tmp/drafter-gate --contexts 3072 --repeats 1
```

After an explicit slot grant, add `--run --exclusive-grant GRANT_ID` and use a new output directory.
The complete schedule uses `--contexts 3072 8192 32768 --repeats 3`.
Run that schedule only after the gate records valid counters and loaded draft parameters.
The launcher checks the draft log for scaling type, frequency scale, original context, and base.
It includes saved target, draft, executable, and shared-library hashes in the run provenance.
It compares both returned token IDs and text against K0.
Each request has a 256-token output limit.
An output-limit stop does not establish task completion, even when both outputs match.
