#!/usr/bin/env bash
# Submission entry point (kit §6): ./run.sh <video.mp4> <out.json>
# Uses ./.venv if present, else $PYTHON, else python3. Writes a valid JSON or exits 2.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <video.mp4> <out.json>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" -m rewind.cli "$1" "$2" --config "$ROOT/configs/default.yaml"
