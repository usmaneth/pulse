# Native sequential diagnostic

The native diagnostic supports the existing text-only Bonsai 27B PQ2_0 model.
It rejects unsupported dimensions, tensor types, RoPE settings, and Hadamard metadata.
It does not support an arbitrary parent model.

Each decoder owns separate recurrent, convolution, and KV state for each layer.
The decoder advances one position after each complete token step.
A reset clears recurrent and convolution state and resets the position.
The position prevents reads from stale KV entries after a reset.

The path includes token lookup, inverse embedding rotation, all layers, final normalization, output projection, and greedy token selection.
Greedy selection chooses the first maximum logit.
The command accepts raw token IDs and a fixed output count.
It does not contain a tokenizer, EOS policy, HTTP server, speculative decoder, or request scheduler.

## Build and CPU tests

Use the same patched llama.cpp tree for the reference executable and the dequantization library.
The upstream tree on this host includes the Bonsai Hadamard correction.

```sh
make native-cpu-test
make bin/pulse-engine bin/pulse-dumpref LLAMA_DIR=/home/usman/llama.cpp-upstream -j1
```

## Sequential comparison

Run GPU commands only during an exclusive validation slot.
Set `MODEL` to the existing Bonsai GGUF file.
Set `REF` and `NATIVE` to separate output directories.

```sh
bin/pulse-dumpref "$MODEL" --sequence 100,101,102 "$REF"
bin/pulse-engine "$MODEL" --decode 100,101,102 0 "$NATIVE" --reset-check
python3 tests/engine/compare_sequence.py "$REF" "$NATIVE" --output comparison.json
```

The reference command accepts an optional name filter and a greedy continuation count.
An empty filter writes layer outputs, recurrent states, embeddings, and final outputs.
The reference saves the complete forced token list in `inputs.txt`.
Both programs require an empty output directory.

```sh
bin/pulse-dumpref "$MODEL" --sequence 7734,264,12654,709 "$REF" '' 4
bin/pulse-engine "$MODEL" --decode 7734,264,12654,709 5 "$NATIVE" --reset-check
```

The second native command predicts five tokens and evaluates the first four predictions.
Both directories therefore contain eight complete input steps if the greedy choices agree.
The comparison rejects different token IDs before it compares numerical values.
Use the reference `inputs.txt` list with zero new tokens to localize a greedy divergence.

The comparator checks every layer output, recurrent state, and output logit at each step.
Default diagnostic limits are 0.05 maximum relative error and 0.999 minimum cosine.
The relative error is the maximum absolute error divided by the maximum reference magnitude.
The comparator also requires matching greedy token IDs and records both top-two logit margins.
These limits do not establish bitwise equivalence or task quality.
Do not relax the limits after a failure.

The optional `--reset-check` repeats the complete native input sequence after a reset.
It requires identical native logits before and after the reset.
It does not compare reset behavior with llama.cpp.

## Reference layout

The dump header contains an int32 type and four int64 dimensions.
The payload now contains contiguous logical values, with dimension zero first.
The writer gathers scalar tensor views with the original byte strides.
The reader rejects extra bytes from old noncontiguous dumps.
Regenerate old reference directories before a comparison.

## Performance status

The old device timing path shared recurrent state between layers and omitted the output head.
Its timing and comparisons with llama.cpp were invalid. That path was removed.
The sequential diagnostic reports no throughput claim.
Measure complete, matched work only after representative sequence checks pass.

## Complete forced-trace profile

Use one fixed token list for both programs.
Use the same prefill boundary in both programs.
The remaining input tokens form the forced decode trace.
Both programs compute full logits and greedy selection for each forced decode step.
The native path includes its device-to-host logit transfer.
The reference uses batch size one, F32 KV, and disabled flash attention.
The reference has no evaluation callback in profile mode.

```sh
bin/pulse-engine "$MODEL" --decode "$TOKEN_IDS" 0 --profile --prefill-count 17
bin/pulse-dumpref "$MODEL" --profile "$TOKEN_IDS" 17
```

Run these commands sequentially during an exclusive GPU slot.
Run repeated pairs and compare the `forced_prediction` records before throughput results.
Profile mode rejects tensor dumps and reset replay in the native program.
The clocks exclude model load and decoder state allocation.
Each program synchronizes the GPU at the phase boundaries.
The reference reports model load and context initialization separately.
The native program reports model upload separately from the profile.
Sequential prefill is a diagnostic baseline, not an optimized prefill implementation.
Reference context allocation can exceed the token count because llama.cpp pads the context.
Record this allocation difference with the result.
The native matrix-vector kernel uses float activations; the reference can quantize activations internally.
Do not attribute a numerical difference to this detail without an isolated comparison.

The model file contains 7,206,168,928 bytes.
The complete file size does not equal weight traffic per decode step.
A token lookup reads one embedding row, not the complete embedding table.
KV, recurrent state, and activation traffic require separate measurements.

For a normal reference target configuration, append `--production` to the reference profile command.
This option uses F16 KV and enables flash attention. It does not enable a speculative drafter.

The native diagnostic batches Q/K head normalization by default.
Set `PULSE_BATCH_HEAD_NORM=0` to restore individual head launches for an A/B check.
Set `PULSE_PREFETCH_MODEL=1` to populate model file pages before the upload.
Set `PULSE_PIN_MODEL=1` to attempt scoped readonly host registration.
Host registration is off by default and falls back if the CUDA device rejects it.
This GB10 rejects readonly registration. Do not report a registration speedup on this host.

## Experimental full-step CUDA graph

Set `PULSE_CUDA_GRAPH=1` to capture the complete native step once per decoder.
Graph mode remains off by default.
The decoder uploads the current token and position to device control storage before each launch.
The same graph handles every valid sequence position.
The graph includes the final normalization and vocabulary projection.
Host logit transfer and greedy selection remain outside capture and inside complete-step timing.

Diagnostic dumps record the actual graph execution.
The graph copies each layer output into a preallocated snapshot buffer.
After each step, the diagnostic also writes convolution history and valid KV prefixes.

```sh
PULSE_CUDA_GRAPH=0 build/native/pulse-engine "$MODEL" --decode "$TOKEN_IDS" 0 "$STREAMED" --interleave-check
PULSE_CUDA_GRAPH=1 build/native/pulse-engine "$MODEL" --decode "$TOKEN_IDS" 0 "$GRAPH" --interleave-check
python3 tests/engine/compare_native_dumps.py "$STREAMED" "$GRAPH" --output graph-comparison.json
```

Run both GPU commands in an exclusive validation slot.
Use fresh output directories. The comparator requires complete matching tensor sets and exact token manifests.
The interleave check tests two independent sequences, reset replay, context capacity, and invalid token rejection.
A graph result does not establish reference-model equivalence by itself. Retain the llama.cpp numerical checks.

For graph and streamed performance comparisons, add `--warmup` to both native profile commands.
This option evaluates the provided forced trace once, resets state, and then starts the phase clocks.
It requires profile mode. The warmup excludes generated continuation tokens that were not supplied as input IDs.

For a longer forced trace without large dump files, use `--graph-check`.
This option compares streamed and graph logits at every supplied position in one process.
It also compares all layer outputs and persistent state at selected position boundaries and at the last step.
The output lists every state-check position. The check requires byte-identical values.
This mode is separate from profile mode.

```sh
build/native/pulse-engine "$MODEL" --decode "$LONG_TOKEN_IDS" 0 --graph-check
```

A failed step can mutate device state before an error reaches the caller.
The decoder rejects further steps after such an error until a reset succeeds.
Invalid tokens and context overflow fail before device state changes.

## Experimental Q8 activation path

Set `PULSE_Q8_ACTIVATIONS=1` to select the Q8_1 activation and DP4A matrix path.
The default remains the FP32 activation path.
This option changes arithmetic and requires separate full-model reference validation.
The runtime quantizes each shared matrix input after its Hadamard transform.
Q/K/V and FFN gate/up consumers reuse the same prepared buffer.
The runtime allocates that buffer before CUDA graph capture.

Build `build/native/test-pq2-q8` with `BIN_DIR=build/native` and the patched `LLAMA_DIR`.
Run this GPU test only with the experiment's exclusive GPU lease.
The test requires exact quantizer bytes and chunk-dot results against the linked upstream implementation.
Full-model tests remain necessary because the native row reduction differs from upstream.
The upstream source attribution and MIT license are in `src/engine/pq2_q8.cuh`.
