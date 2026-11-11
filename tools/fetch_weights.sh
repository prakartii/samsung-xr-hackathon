#!/usr/bin/env bash
# Download every model weight file into weights/ and verify its SHA-256.
# Sources and licenses: ARCHITECTURE.md §10. Weights are gitignored (CLAUDE.md §6.9).
# Re-running is safe: files that already match their hash are skipped.
#
# Usage: tools/fetch_weights.sh          (from the repo root)
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p weights

# name | url | sha256
WEIGHTS=(
  # MediaPipe Hand Landmarker, float16, version 1 (Apache-2.0)
  "hand_landmarker.task|https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task|fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
)

sha256_of() { sha256sum "$1" | cut -d' ' -f1 | tr -d '*'; }

for entry in "${WEIGHTS[@]}"; do
  IFS='|' read -r name url sha <<<"$entry"
  target="weights/$name"
  if [[ -f "$target" && "$(sha256_of "$target")" == "$sha" ]]; then
    echo "ok       $name"
    continue
  fi
  echo "fetching $name"
  curl -sSfL --retry 5 -o "$target.tmp" "$url"
  got="$(sha256_of "$target.tmp")"
  if [[ "$got" != "$sha" ]]; then
    rm -f "$target.tmp"
    echo "SHA-256 mismatch for $name: expected $sha, got $got" >&2
    exit 1
  fi
  mv "$target.tmp" "$target"
  echo "ok       $name"
done
