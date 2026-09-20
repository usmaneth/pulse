#!/usr/bin/env bash
# Run one arm of the checkpoint eviction A/B.
# Usage: run_ckpt_arm.sh <label> <PULSE_CKPT_MIN_CHARS> <repeats>
# Kills the proxy by LISTENING PORT, never by pgrep -f (pgrep self-matches).
set -u
LABEL="$1"; MINCHARS="$2"; REPEATS="${3:-3}"
PORT=8000
LOG=/tmp/claude-1001/-home-usman/78db5e84-9418-485d-a8e5-60fdfdd0d656/scratchpad/proxy-$LABEL.log
mkdir -p "$(dirname "$LOG")"

pids=$(ss -tulpnH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u)
[ -n "$pids" ] && { echo "stopping proxy pid(s): $pids"; kill $pids 2>/dev/null; sleep 2; }

rm -f /tmp/slotsave/* 2>/dev/null
cd /home/usman/pulse || exit 1
PULSE_CKPT_PATH=/tmp/slotsave/ PULSE_CKPT_MIN_CHARS="$MINCHARS" \
  nohup bun dist/server/index.js >"$LOG" 2>&1 &
for i in $(seq 1 40); do
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/status" >/dev/null 2>&1 && break
  sleep 1
done
curl -sf --max-time 2 "http://127.0.0.1:$PORT/status" >/dev/null 2>&1 || {
  echo "ARM $LABEL: proxy failed to bind $PORT"; tail -20 "$LOG"; exit 1; }

echo "=== ARM $LABEL (PULSE_CKPT_MIN_CHARS=$MINCHARS) ==="
python3 bench/ckpt_evict.py "http://127.0.0.1:$PORT" "$REPEATS"
