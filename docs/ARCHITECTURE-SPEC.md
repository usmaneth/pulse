# Bonsai 2 / qwen35 on GB10: the complete inference spec

Everything here was established by running one token through llama.cpp, dumping
all 4,639 intermediate tensors, and checking each stage against them. Nothing is
inferred from documentation, because there is none for this architecture.

Reproduce with `bin/pulse-dumpref <model.gguf> <token> <outdir> [filter]` then
`bin/pulse-engine <model.gguf> <outdir>`.

---

## 1. Model shape, read from the file

| key | value |
|---|---|
| `general.architecture` | **`qwen35`** (not bonsai2, not qwen3next) |
| `block_count` | 64 |
| `embedding_length` | 5120 |
| `feed_forward_length` | 17408 |
| `attention.head_count` / `_kv` | 24 / 4 |
| `attention.key_length` / `value_length` | 256 / 256 |
| `full_attention_interval` | 4 |
| `context_length` | 262144 |
| `rope.dimension_count` | 64 |
| `rope.dimension_sections` | `[11, 11, 10, 0]` |
| `rope.freq_base` | 1e7 |
| `ssm.conv_kernel` / `state_size` | 4 / 128 |
| `ssm.group_count` / `time_step_rank` | 16 / 48 |
| `ssm.inner_size` | 6144 |
| tensors / size | 851 / 6.71 GB, 26.90 B params |

**Layer types.** 16 layers (3, 7, 11 … 63) are full attention and carry
`attn_q/k/v/output`. The other 48 are gated-delta and carry `attn_qkv` plus
`ssm_*`. The naming inverts the obvious reading: `attn_qkv` belongs to the
**SSM** layers, as its own projection.

---

## 2. The Hadamard fold - the thing that breaks naive implementations

401 weights are Hadamard-folded (`prism.hadamard.weight_names`). The fold is
baked into the weights; the matching rotation must be applied to the
**activation** immediately before the matmul. Skip it and output is
uncorrelated with truth - measured cosine **-0.07** on `attn_qkv`.

Order (`llama-graph.cpp:1571`):

```
[optional tiled->grouped head permutation, when perm_rep > 1]
x = x * signs          # sign_mode = explicit
x = WHT(x)             # normalized Sylvester-Walsh, block 1024, scale 1/32
y = W @ x
```

`sign_values` is 28,672 entries; `sign_widths` is `[5120, 6144, 17408]` -
**one sign vector per distinct activation width**: hidden, ssm-inner, ffn.

### Conventions per weight, all measured

| weight | in-width | convention | verified |
|---|---|---|---|
| `output.weight` | 5120 | signs + WHT | argmax match, top-10 10/10 |
| `ffn_gate`, `ffn_up` | 5120 | signs + WHT | cos 0.99999661 (block) |
| `ffn_down` | 17408 | signs + WHT | same |
| `attn_qkv` | 5120 | signs + WHT | cos 0.99999907 |
| `ssm_out` | 6144 | **permute** + signs + WHT | cos 0.99999463 |

`ssm_out` needs the tiled `[hd=128, nk=16, rep=3]` -> grouped `[hd, rep, nk]`
permutation first. Without it: cos **-0.11**. With WHT but no permute: cos
**-0.32**. Both wrong, in different directions.

---

## 3. Head mappings - the two sub-blocks DISAGREE

This is the single most surprising result and the easiest way to get 48 or 16
layers silently wrong.

| path | mapping | evidence |
|---|---|---|
| full attention (24q / 4kv) | **blocked**, `h / 6` | 4.6e-04 vs 2.3e+00 interleaved |
| gated-delta (48v / 16k) | **interleaved**, `h % 16` | 1.3e-07 vs 1.8e+00 blocked |

Four orders of magnitude in both directions. Assuming one convention for the
whole model injects a bug into one half while fixing the other.

Attention is checkable cheaply: with a single token the softmax is over one
element, so each query head's output equals the V of its mapped KV head exactly.

---

## 4. The gated-delta recurrence

**State layout is `S[dv][dk]`** - v-major, k-minor. Settled unambiguously: on a
fresh context the prior state is zero, so `new_state` must equal
`beta * outer(v, k)`. It matches that to **1.342e-07** and `outer(k, v)` to
1.0e+00.

**Gates.** The `alpha` and `beta` tensors are RAW pre-activations
(`ggml-cuda/gated_delta_net.cu:92-114`):

```
beta = sigmoid(beta_raw)
g    = exp( ssm_a[h] * softplus(alpha_raw[h] + ssm_dt_bias[h]) )
```

Using them raw gives rel 4.1e+00. Typical values: g ~ 0.9993 (near-unity
decay), beta ~ 0.077.

**Recurrence**, validated at **9.169e-08**:

```
kv    = S k                  [dv]
delta = (v - g*kv) * beta    [dv]
S_new = g*S + delta (x) k    [dv][dk]
o     = (S_new q) * scale    [dv]
```

The delta term is what makes this not linear attention: it subtracts what the
state already predicts for `k` before writing `v`, so a repeated key overwrites
instead of accumulating.

**State cost: 151.0 MB** (48 layers x 48 heads x 128 x 128 f32) plus ~5.9 MB of
conv state. **Constant in context length** - which is why 3/4 of this model
costs nothing extra as context grows.

**Pre-recurrence chain:** `attn_qkv` -> depthwise conv1d (K=4, validated
2.6e-08) -> SiLU (6.9e-08) -> split q 2048 / k 2048 / v 6144.

**Post-recurrence:** `silu(z) * rmsnorm_per_head(o, ssm_norm)` -> `ssm_out`.
The RMSNorm is per-head over 128 (global over 6144 scores worse: 0.707 vs
0.909). *This stage is not yet fully closed - see Open Items.*

---

## 5. RoPE, and why an engine can beat llama.cpp here

`dimension_sections = [11, 11, 10, 0]` sums to 32 = `rope_dim/2`, fourth slot
unused. Only the first 64 of each head's 256 dims are rotated.

**On the text path every section indexes the same position, so mRoPE collapses
to standard RoPE.** llama.cpp refuses K-shifting for any model with
`n_pos_per_embd() > 1` (`llama-kv-cache.cpp:1176`) - correct for multimodal,
unnecessary for text. That is why `--cache-reuse` is unavailable and a
mid-context edit costs a full re-prefill.

**An engine that owns RoPE can shift positions. llama.cpp will not.** This is a
capability difference, not a percentage - and it is the strongest argument for
owning the stack on this model.

---

## 6. KV cache

16 attention layers x 4 kv-heads x 256 dim = **134.2 MB per 1k tokens** (f32).

| context | f32 | f16 |
|---|---|---|
| 32k | 4.3 GB | 2.1 GB |
| 256k | 34.4 GB | **17.2 GB** |

The f16/256k figure independently confirms the 17 GB number this project quoted
from an unrelated derivation.

---

## 7. Performance envelope on GB10

| quantity | value |
|---|---|
| spec peak (256-bit LPDDR5X @ 8533) | 273 GB/s |
| **achievable** at a 6.70 GB working set | **216 GB/s** |
| llama.cpp decode | 183 GB/s (85% of achievable) |
| best Pulse matvec (v4) | 176 GB/s (96% of llama.cpp) |

Four kernel iterations: 31.3 -> 144.6 -> 169.0 -> 176.0 GB/s. Gains +113, +24,
+7 - asymptotic to llama.cpp, approached from below.

**Splitting the qs/scale arrays measured 12% SLOWER** (155.3 GB/s), bit-identical
output. GGUF's 34-byte interleaving puts each scale in a cache line the warp is
already fetching; separating them creates a second stream. **The layout is a
locality optimisation, not a compromise.**

Conclusion: an engine is buildable and is not where throughput is. Build it for
capability (RoPE/KV ownership, scheduling), not speed.

---

## 8. Method: what actually found these bugs

Five architectural facts above were invisible at op level and fatal in
composition. Every one was caught the same way:

1. **Dump ground truth first.** `cb_eval` gives every named intermediate. A
   forward pass is 64 layers; without per-stage references a wrong answer tells
   you nothing about where.
2. **Run candidates, print all of them.** Every check here is an A/B/C, never an
   assertion. Costs a few lines; caught the Hadamard fold, the ffn width, the
   qkv fold, both head mappings, and the ssm_out permutation.
3. **Scale error by `sum|terms|`, not by the result.** These dot products have
   condition number ~6e3. Relative-to-result flagged a correct kernel as failing
   at 2.3e-05; scaled properly it was 1.2e-08, below fp32 epsilon.
4. **Exploit degenerate cases.** A zero initial state turns the recurrence into
   a pure outer product and pins the layout exactly. One token makes attention
   softmax trivial and exposes the head mapping directly.
5. **Validating against your own reference proves nothing about the model.**
   Step 8's GDN kernel passed its own test while being wrong three ways.

---

## Open items

- **GDN output gate.** Structure is confirmed - `silu(z) * rmsnorm_per_head` -
  but reconstructing the fused op's `output` as `S_new q` reaches only cos
  0.909. The state is not in doubt (9.169e-08). `output` is a view into
  `ggml_gated_delta_net`'s result and is not separately dumped.
- **Full 64-layer loop.** Components validated; not yet composed end to end.
- **Attention output projection + gate.** Mapping confirmed; projection not yet
  checked against `attn_output`.
