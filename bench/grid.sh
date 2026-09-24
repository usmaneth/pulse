#!/usr/bin/env bash
# Measure the full serving policy grid: {drafter} x {parallel slots}.
#
# Speculation and batching are substitutes on a bandwidth-bound machine - the
# drafter buys tokens per weight sweep only while the sweep is under-occupied.
# This maps where each wins so `pulse serve` can pick correctly.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# Ternary Bonsai 2 checkout: a sibling of this repo by default.
ROOT=${ROOT:-"$HERE/../../Bonsai-demo"}
BIN=$ROOT/bin/cuda/llama-server
M=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf
V1=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-Q4_0.gguf
V2=models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-dspark-dflash-v2-Q4_K_M.gguf
SLOTS=${SLOTS:-16}
LEVELS=${LEVELS:-1,2,4,8,16}

kill_servers() { for p in $(pgrep -f 'cuda/llama-server' 2>/dev/null); do kill -9 "$p" 2>/dev/null; done; sleep 3; }
wait_ready() { for _ in $(seq 1 150); do
    curl -s --max-time 2 http://127.0.0.1:8085/health 2>/dev/null | grep -q '"ok"' && return 0; sleep 3; done; return 1; }

run_cfg() {
  local label="$1"; shift
  kill_servers
  ( cd "$ROOT" && nohup "$BIN" -m "$M" -ngl 99 -fa on -c 65536 -np "$SLOTS" \
      -b 4096 -ub 512 --host 127.0.0.1 --port 8085 "$@" > /tmp/grid_srv.log 2>&1 & ) 
  if ! wait_ready; then echo "### $label: SERVER FAILED"; tail -3 /tmp/grid_srv.log; return; fi
  echo ""; echo "### $label"
  python3 "$HERE/conc.py" http://127.0.0.1:8085 "$LEVELS" 128 2>&1 | tail -8
}

echo "serving policy grid: $SLOTS slots, concurrency $LEVELS"
run_cfg "no drafter"
run_cfg "v1 drafter (block 4) K=4" -md "$V1" --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999
run_cfg "v2 drafter (block 7) K=7" -md "$V2" --spec-type draft-dspark --spec-draft-n-max 7 -ngld 999
kill_servers
echo ""; echo "grid done"
