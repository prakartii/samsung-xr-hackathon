#!/usr/bin/env bash
# Download every model weight into weights/ and verify its SHA-256.
# Sources and licenses: ARCHITECTURE.md §10. Weights are gitignored (CLAUDE.md §6.9).
# Re-running is safe: files that already match their hash are skipped.
#
# Each file comes from our GitHub release (weights-v1, PROGRESS.md Decisions
# 2026-11-17, Q12) and, if that fails, from its original source. Hashes are pinned in
# tools/weights.sha256, so either source must deliver the same bytes.
#
# Usage: tools/fetch_weights.sh          (from anywhere)
#   SKIP_GDRIVE=1        only the hand model and CLIP (the Kaggle feature-cache job)
#   RELEASED_FEATURES=1  rebuild weights/released_features from ActionFormer's
#                        original 6.5 GB archive instead of the release
#   WEIGHTS_RELEASE_URL, WEIGHTS_SHA256SUMS  override the release and the hash list
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p weights

PYTHON="${PYTHON:-python}"
RELEASE_URL="${WEIGHTS_RELEASE_URL:-https://github.com/prakartii/samsung-xr-hackathon/releases/download/weights-v1}"
SUMS="${WEIGHTS_SHA256SUMS:-tools/weights.sha256}"

sha256_of() { sha256sum "$1" | cut -d' ' -f1 | tr -d '*'; }
expected() { awk -v n="$1" '$2 == n { print $1 }' "$SUMS"; }
have() { [[ -f "weights/$1" && "$(sha256_of "weights/$1")" == "$(expected "$1")" ]]; }

# Move a downloaded file into weights/ only if its hash is the pinned one.
install_checked() {
  local name=$1 tmp=$2 want got
  want="$(expected "$name")"
  got="$(sha256_of "$tmp")"
  if [[ -z "$want" || "$got" != "$want" ]]; then
    rm -f "$tmp"
    echo "SHA-256 mismatch for $name: expected ${want:-<not in $SUMS>}, got $got" >&2
    return 1
  fi
  mv "$tmp" "weights/$name"
}

# ActionFormer checkpoints are released on Google Drive as zips (named .tar.gz)
# holding config, logs and epoch_020.pth.tar; we keep only the checkpoint.
# Needs `gdown` (requirements-dev.txt).
gdrive_member() {
  local file_id=$1 member=$2 out=$3
  "$PYTHON" -m gdown "$file_id" -O "$out.zip"
  "$PYTHON" -c "import shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z, z.open(sys.argv[2]) as src, open(sys.argv[3], 'wb') as dst:
    shutil.copyfileobj(src, dst)" "$out.zip" "$member" "$out"
  rm -f "$out.zip"
}

# The original source of each file (used when the release can't be reached).
upstream() {
  local name=$1 out=$2
  case "$name" in
    # MediaPipe Hand Landmarker, float16, version 1 (Apache-2.0)
    hand_landmarker.task)
      curl -sSfL --retry 5 -o "$out" "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" ;;
    # CLIP ViT-B/16, OpenAI weights in open_clip format (MIT, OpenAI CLIP), pinned HF revision
    clip_vit_b16_openai.safetensors)
      curl -sSfL --retry 5 -o "$out" "https://huggingface.co/timm/vit_base_patch16_clip_224.openai/resolve/977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d/open_clip_model.safetensors" ;;
    # ActionFormer EPIC-100 checkpoints (MIT; trained on EPIC train only, PROGRESS.md §4)
    actionformer_epic_verb.pth.tar)
      gdrive_member 1Ta4ggKSj2YcszSrDbePlHe1ECF1CFKK4 epic_slowfast_verb_reproduce/epoch_020.pth.tar "$out" ;;
    actionformer_epic_noun.pth.tar)
      gdrive_member 1OTlxeiWj8JE9n1-LsRYogHmqgUdsE5PR epic_slowfast_noun_reproduce/epoch_020.pth.tar "$out" ;;
    # Verb-noun prior: no upstream copy; it is counted from EPIC_100_train.csv
    vn_prior.npy)
      echo "  build it instead: tools/fetch_annotations.sh &&" \
        "python tools/build_vn_prior.py --train-csv data/annotations/epic/EPIC_100_train.csv" >&2
      return 1 ;;
    *) return 1 ;;
  esac
}

fetch() {
  local name=$1
  if have "$name"; then
    echo "ok       $name"
    return
  fi
  echo "fetching $name"
  if curl -sSfL --retry 3 -o "weights/$name.tmp" "$RELEASE_URL/$name"; then
    install_checked "$name" "weights/$name.tmp" || exit 1
  else
    rm -f "weights/$name.tmp"
    echo "  not available from $RELEASE_URL; trying the original source"
    if ! upstream "$name" "weights/$name.tmp"; then
      rm -f "weights/$name.tmp"
      echo "no source for $name" >&2
      exit 1
    fi
    install_checked "$name" "weights/$name.tmp" || exit 1
  fi
  echo "ok       $name"
}

fetch hand_landmarker.task
fetch clip_vit_b16_openai.safetensors
if [[ "${SKIP_GDRIVE:-0}" == "1" ]]; then
  exit 0
fi
fetch actionformer_epic_verb.pth.tar
fetch actionformer_epic_noun.pth.tar
fetch vn_prior.npy

# ActionFormer's released EPIC SlowFast features as a read-only cache in
# weights/released_features (detect.released_features), ~0.8 GB for the 138 manifest
# videos. run.sh finds them by the MP4's content fingerprint, so a clean machine gets
# ActionFormer actions instead of the zero-shot fallback (ARCHITECTURE.md §10.1).
if [[ -d weights/released_features && -n "$(ls -A weights/released_features)" ]]; then
  echo "ok       released_features (present; delete it to fetch again)"
elif [[ "${RELEASED_FEATURES:-0}" == "1" ]]; then
  # Rebuild from the original 6.5 GB archive (Box mirror, MD5-checked); fingerprints
  # come from tools/data/epic_fingerprints.jsonl.
  ARCHIVE_DIR="${ARCHIVE_DIR:-data/actionformer}"
  mkdir -p "$ARCHIVE_DIR"
  "$PYTHON" tools/fetch_released_features.py --out "$ARCHIVE_DIR/epic_kitchens.tar.gz"
  if [[ ! -d "$ARCHIVE_DIR/epic_kitchens/features" ]]; then
    tar -xzf "$ARCHIVE_DIR/epic_kitchens.tar.gz" -C "$ARCHIVE_DIR"
  fi
  "$PYTHON" tools/import_released_features.py \
    --features-dir "$ARCHIVE_DIR/epic_kitchens/features" \
    --cache weights/released_features \
    --fingerprint-logs tools/data/epic_fingerprints.jsonl
else
  fetch released_features.tar
  # Unpack beside weights/ first, so an interrupted run leaves no half-filled cache.
  rm -rf weights/.released_features.unpack
  mkdir weights/.released_features.unpack
  tar -xf weights/released_features.tar -C weights/.released_features.unpack
  rm -rf weights/released_features
  mv weights/.released_features.unpack/released_features weights/released_features
  rm -rf weights/.released_features.unpack weights/released_features.tar
  echo "ok       released_features"
fi
