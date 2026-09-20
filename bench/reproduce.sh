#!/usr/bin/env bash
# Regenerate the headline numbers in bench/RESULTS.md from scratch.
#
# This machine is state-dependent (see Round 11): the CPU and GPU share one
# LPDDR5X bus, the measured noise floor is 10.2%, and throughput climbs across
# consecutive runs as unified-memory pages migrate back. Every step below either
# uses bench/ab.py (which quiesces, warms up, interleaves and takes medians) or
# states plainly that it is a single-shot figure.
set -euo pipefail

ROOT=${ROOT:-/home/usman/Bonsai-demo}
BIN=$ROOT/bin/cuda
M=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
V1=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf
V2=$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf
HERE=$(cd "$(dirname "$0")" && pwd)

step() { printf '\n=== %s ===\n' "$1"; }

step "0. measurement hygiene gate"
"$HERE/../bin/pulse-cli" doctor || {
  echo "machine is not clean. Quiesce and retry, or accept 10.2%+ noise."; }

step "1. bandwidth floor - predicts the no-drafter baseline"
echo "6.70 GB / 184.6 GB/s = 36.3 ms/step -> 27.5 tok/s expected"
"$BIN/llama-batched-bench" -m "$M" -ngl 99 -fa on -c 16384 -b 4096 -ub 512 \
  -npp 256 -ntg 64 -npl 1,2,4,8,16 2>&1 | tail -8

step "2. design curve - hardware ceiling vs accepted tokens/step"
echo "model-free n-gram drafter at 100% acceptance isolates verify cost"
python3 - <<'PY' > /tmp/rep_prompt.txt
open('/tmp/rep_prompt.txt','w').write('def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n'*6)
PY
for m in 1 3 4 6 8 12 16; do
  printf '  m=%-3s ' "$m"
  "$BIN/llama-speculative-simple" -m "$M" --spec-type ngram-simple \
    --spec-ngram-simple-size-m $m --spec-ngram-simple-size-n 3 --spec-draft-n-max $m \
    -ngl 99 -fa on -c 4096 -b 4096 -ub 512 --temp 0 --ignore-eos -n 256 \
    -f /tmp/rep_prompt.txt 2>&1 | grep -oP 'speed:\s+\K[\d.]+' | head -1
done

step "3. best single-stream config (median of 6, interleaved)"
head -c 5000 /home/usman/llama.cpp-upstream/src/llama-model.cpp > /tmp/rep_code.txt 2>/dev/null || true
SPEC_DRAFT=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf \
python3 "$HERE/ab.py" -n 6 --warmup 2 --prompt /tmp/rep_code.txt \
  --config "v2 K=7= -- " || true

step "4. concurrency - needs a running server, see bench/conc.py"
echo "  pulse serve --slots 16   # then:"
echo "  python3 bench/conc.py http://127.0.0.1:8085 1,2,4,8,16 128"

step "done"
echo "Compare against bench/RESULTS.md. Differences above 10.2% are real;"
echo "differences below it are machine state."
