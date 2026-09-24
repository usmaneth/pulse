# Multi-token recurrent GDN fixture

This standalone kernel processes 1 through 64 tokens for the Bonsai GDN shape.
It retains recurrent state in registers across the tile.
It is not integrated into the native decoder and does not provide full-model prefill.

The contract uses 128 dimensions, 16 key heads, and 48 value heads.
Queries and keys share explicit head and token strides.
Values have separate head and token strides.
Gates are contiguous `[token][value_head]` arrays: logarithmic decay and activated beta.
Initial state is `[value_head][value_dimension][key_dimension]`.
The caller supplies separate initial state, final state, and token-output buffers.
All inputs and outputs use FP32. One call processes one sequence.

The output includes the upstream `1/sqrt(128)` factor.
The existing single-step native GDN path uses a different output-scale and normalization convention.
Do not replace that path without adapting and testing its following normalization.

Build the fixture:

```sh
make BIN_DIR=build/native build/native/test-gdn-prefill LLAMA_DIR=~/llama.cpp-upstream -j1
```

Run `build/native/test-gdn-prefill` only during an exclusive GPU experiment slot.
The fixture uses the actual upstream CUDA `ggml_gated_delta_net` operator as its reference.
It checks 1, 2, 17, and 64 tokens with contiguous/padded views and zero/nonzero initial state.
It also checks independent-state replay and exact split-tile results.
The fixed reference limits are maximum absolute error 0.00002 and relative L2 error 0.000002.

All 16 fixtures passed on GB10 with upstream commit `999b0a9a6f2fb3e3b60bb3d6bbac4b6b0c0f7fd7`.
The largest output error was 5.588e-9 and the largest final-state error was 4.470e-8.
This fixture does not test full-layer normalization, convolution, matrix projections, attention, or language-model quality.
No kernel or engine speed improvement is claimed.
