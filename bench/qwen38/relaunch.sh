#!/usr/bin/env bash
# Stop the server, wait until the host has the memory back, start it again.
# Usage: relaunch.sh <log-name>
set -u
cd "${QWEN38_RECIPE_DIR:-/mnt/models/qwen38-flash}"
./stop.sh >/dev/null 2>&1
for i in $(seq 1 60); do
  docker ps --format '{{.Names}}' | grep -q vllm-fn-tp1 || \
  [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 100 ] && break
  sleep 3
done
until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 100 ]; do sleep 3; done
./start.sh > "logs/start-$1.log" 2>&1
echo "exit=$?"
