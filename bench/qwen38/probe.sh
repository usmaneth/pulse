#!/usr/bin/env bash
# GB10 fast/slow state probe in its own container (outside the vLLM cgroup cap).
docker run --rm --gpus all --entrypoint python3 -v "${QWEN38_TUNE_DIR:-/mnt/models/qwen38-tune}/gpuprobe.py:/p.py:ro" \
  vllm/vllm-openai:qwen38-flash-next /p.py 2>&1 | grep -o "gemv_GBps=[0-9]*" | tail -1
