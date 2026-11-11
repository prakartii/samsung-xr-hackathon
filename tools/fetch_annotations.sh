#!/usr/bin/env bash
# Download the ground-truth annotations used by rewind_eval (never by inference).
# Sources are pinned to exact commits and checked by sha256.
#   EPIC-KITCHENS-100: github.com/epic-kitchens/epic-kitchens-100-annotations (CC BY-NC 4.0)
#   FPVSum:            github.com/azuxmioy/fpvsum (annotation/ + crawler video list)
# Usage: tools/fetch_annotations.sh [dest]   (default: data/annotations)
set -euo pipefail

DEST="${1:-data/annotations}"
EPIC_SHA=ea8b40457a400c3fffa1c7f406ef3dc169cc2522
FPV_SHA=9af31f9027ef2b3f5f34a39bf17df5b120a9cc25
EPIC_URL="https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/$EPIC_SHA"
FPV_URL="https://raw.githubusercontent.com/azuxmioy/fpvsum/$FPV_SHA"

fetch() {  # url  out  sha256
  local url="$1" out="$2" want="$3"
  if [ -f "$out" ] && echo "$want  $out" | sha256sum -c --quiet - 2>/dev/null; then
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  curl -sfL --retry 3 -o "$out.part" "$url"
  echo "$want  $out.part" | sha256sum -c --quiet -
  mv "$out.part" "$out"
  echo "fetched $out"
}

fetch "$EPIC_URL/EPIC_100_train.csv"        "$DEST/epic/EPIC_100_train.csv"        a3a7ef2e397bd8af12bd0496b09d61b1f2c1f3f6e1726fa86af50a27f4855e87
fetch "$EPIC_URL/EPIC_100_validation.csv"   "$DEST/epic/EPIC_100_validation.csv"   35f7932ba0a1127a96cac215a98d35398946f343e3cea9ad6688ed17eee9d75d
fetch "$EPIC_URL/EPIC_100_verb_classes.csv" "$DEST/epic/EPIC_100_verb_classes.csv" aab8bf210d0234b6facb9b191ae402c03a5f8965d123586b4762d3468a85033e
fetch "$EPIC_URL/EPIC_100_noun_classes.csv" "$DEST/epic/EPIC_100_noun_classes.csv" 40f0a81c5bd06b33050df9f0bd17cd3c2fccd759f9581e4e5e04fec7562be893
fetch "$EPIC_URL/EPIC_100_video_info.csv"   "$DEST/epic/EPIC_100_video_info.csv"   75fd040f6662cb407b4ca2eed4811ca94df280a1aba00b6bf8a5cd286deff382
fetch "$EPIC_URL/license.txt"               "$DEST/epic/license.txt"               acb7384c3f73cc96367ac132bc0bd57efa38c4495c17781032a2cf1444b65ebe
fetch "$FPV_URL/annotation/gt_data.mat"     "$DEST/fpvsum/gt_data.mat"             666887c310cadc748e9b2092acd1010b4d32375ce9a01fca1e229b16cd0a784b
fetch "$FPV_URL/crawler/FPVSum_videolist.csv" "$DEST/fpvsum/FPVSum_videolist.csv"  6a95192f0e782ce1f8834a704da4c50b839303e598b3be30b10dc421545546bf
echo "annotations ready in $DEST"
