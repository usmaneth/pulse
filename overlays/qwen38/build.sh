#!/usr/bin/env bash
# Build the Qwen3.8-Flash-Next vLLM overlays. See build.py and MANIFEST.md.
#   overlays/qwen38/build.sh --out <dir> --set <comma list of overlay names>
set -euo pipefail
exec python3 "$(dirname "$(readlink -f "$0")")/build.py" "$@"
