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
  # CLIP ViT-B/16, OpenAI weights in open_clip format (MIT, OpenAI CLIP), pinned HF revision
  "clip_vit_b16_openai.safetensors|https://huggingface.co/timm/vit_base_patch16_clip_224.openai/resolve/977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d/open_clip_model.safetensors|4b8699299b1e8997753c64b052ba32031449d5d853f55a039148560ee02b820f"
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

# ActionFormer EPIC-100 checkpoints (MIT; trained on EPIC train only, PROGRESS.md §4).
# Released on Google Drive as zips (named .tar.gz) holding config, logs and
# epoch_020.pth.tar; we keep only the checkpoint. Needs `gdown` (requirements-dev.txt).
# name | Google Drive file id | member inside the zip | sha256 of the member
GDRIVE_ZIPS=(
  "actionformer_epic_verb.pth.tar|1Ta4ggKSj2YcszSrDbePlHe1ECF1CFKK4|epic_slowfast_verb_reproduce/epoch_020.pth.tar|88486434a71e1b17ac38dd4a93d9612612026be62d7a93a032a0953a30ac687b"
  "actionformer_epic_noun.pth.tar|1OTlxeiWj8JE9n1-LsRYogHmqgUdsE5PR|epic_slowfast_noun_reproduce/epoch_020.pth.tar|8d313a2dd69562bfb063cb1e3372ad056ee646283d1e32af95ff49318bf6bbf9"
)
PYTHON="${PYTHON:-python}"
# SKIP_GDRIVE=1 fetches only the direct-URL weights above (e.g. the Kaggle feature
# cache job, which needs hands + CLIP but not ActionFormer).
if [[ "${SKIP_GDRIVE:-0}" == "1" ]]; then
  GDRIVE_ZIPS=()
fi

for entry in "${GDRIVE_ZIPS[@]}"; do
  IFS='|' read -r name file_id member sha <<<"$entry"
  target="weights/$name"
  if [[ -f "$target" && "$(sha256_of "$target")" == "$sha" ]]; then
    echo "ok       $name"
    continue
  fi
  echo "fetching $name"
  "$PYTHON" -m gdown "$file_id" -O "$target.zip"
  "$PYTHON" -c "import shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z, z.open(sys.argv[2]) as src, open(sys.argv[3], 'wb') as dst:
    shutil.copyfileobj(src, dst)" "$target.zip" "$member" "$target.tmp"
  rm -f "$target.zip"
  got="$(sha256_of "$target.tmp")"
  if [[ "$got" != "$sha" ]]; then
    rm -f "$target.tmp"
    echo "SHA-256 mismatch for $name: expected $sha, got $got" >&2
    exit 1
  fi
  mv "$target.tmp" "$target"
  echo "ok       $name"
done
