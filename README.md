# REWIND: coverage-first highlights for first-person video

Samsung XR Hackathon 2026, **Track 4: Action-Aware Video Summarization.**

REWIND turns a first-person video into one JSON file with every detected action
(`actions[]`), a ~15% highlight reel of ≥ 4 s windows of original footage
(`highlights[]`), and the video's category (`label`). The windows are chosen by
**expected-coverage maximization over clustered action proposals**: they go where the
most *distinct* actions happen per second of highlight, not where the video is merely
busy.

```
./run.sh <video.mp4> <out.json>
```

Inference reads only the MP4 (plus `configs/default.yaml` and `weights/`). It never
reads the manifest, annotations or the filename; `video_id` is only echoed into the
JSON. If any stage fails, `run.sh` still writes a valid fallback and exits 0.

## Results

On the videos the organizers score (`report` = 138 EPIC-KITCHENS-100 val + 47 FPVSum;
two Scuba videos left YouTube and are not scored), by our implementation of the kit
metrics under the **official coverage rule**: an action is captured when at least 50%
of it lies inside the highlights (`rewind_eval`, `results/leaderboard.csv`):

| | Weighted | Action coverage (40%) | Action mAP (25%) | Highlight F1 (25%) | Label acc. (10%) |
|---|---|---|---|---|---|
| **REWIND** | **0.289** | 0.281 (46.2% of oracle) | 0.149 | 0.226 | 0.830 |
| Uniform windows | 0.112 | 0.177 (29.0%) | 0 | 0.167 | 0 |
| Oracle (GT-aware) | 0.773 | 0.609 | 1 | 0.717 | 1 |

"% of oracle" divides our coverage by that of a reference that sees the ground truth
and picks windows under the same rule and budget. FPVSum settings are untuned defaults
(see the tuning-split statement).

- **85.7% of visual tokens pruned** (highlight seconds vs video seconds, all 187 outputs).
- **41 s of CPU time per minute of video** (measured 2026-11-21): `run.sh`'s full
  pipeline per video in a fresh process with an empty feature cache, model loading
  included, on a laptop (Intel i7-13700H, 16 GB RAM, Windows, no GPU), 4 videos totalling
  19.6 min (2 EPIC, 2 FPVSum). `tools/latency.py`, `results/latency_2026-11-21.csv`;
  `results/latency.csv` keeps the earlier runs.
- Every component helps on `report`: removing ActionFormer costs 0.048 weighted,
  expected coverage (vs label dedup) 0.006, flow/camera motion 0.015, scene change 0.007,
  hands 0.002, the verb-noun prior 0.0005. Ablations in `results/ablation_report.md` and
  `results/ablation_tune.md` (`tools/ablate.py`); settings were chosen on `tune` only
  (`results/experiments.md`).

## How it works

1. **Signals** (`code/rewind/features/`): one PyAV decode pass → optical flow and
   camera motion (OpenCV Farneback), hand presence (MediaPipe), CLIP ViT-B/16 frame
   embeddings. All cached by the MP4's content fingerprint, never its name.
2. **Route** (`routing.py`): CLIP zero-shot P(cooking) decides EPIC (kitchen) vs
   FPVSum (outdoor activity); an uncertain video still gets actions.
3. **Actions** (`detect/`): ActionFormer's EPIC-100 verb and noun models on SlowFast
   features, paired into (verb, noun) actions; a CLIP zero-shot detector is the
   fallback.
4. **EPIC highlights** (`cluster.py`, `select/epic.py`): proposals are clustered into
   distinct steps; windows are chosen greedily by expected new coverage per second
   within the 15% budget, then post-processed to the kit rules.
5. **FPVSum highlights and label** (`select/importance.py`, `label/`): an importance
   proxy (flow, hands, camera motion, scene change) → best windows per shot;
   CLIP zero-shot label over the kit classes.

Full specification: `ARCHITECTURE.md`. Design rationale: `REWIND_v2.3_plan.md`.

## Setup

Python 3.11, CPU only. In the submitted `TEAM_ID/` package `weights/` is already
filled (`tools/make_submission.py`), so only the `pip install` and `run.sh` lines apply.
The tools, results, tests and docs this README mentions are in the repository, not in
the package.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # pinned; CPU PyTorch wheels
tools/fetch_weights.sh                     # every weight into weights/, SHA-256 checked
./run.sh data/videos/<video_id>.mp4 outputs/<video_id>.json
```

`fetch_weights.sh` downloads from this repository's GitHub release `weights-v1`
(the files exceed GitHub's 100 MB limit for the repository itself) and falls back to
each file's original source; either way the hash must match `tools/weights.sha256`.
The two files with no public upstream can also be rebuilt from scratch:
`tools/fetch_annotations.sh && python tools/build_vn_prior.py --train-csv
data/annotations/epic/EPIC_100_train.csv` (verb-noun prior) and
`RELEASED_FEATURES=1 tools/fetch_weights.sh` (released features, a 6.5 GB download).

Or with Docker (CPU; fill `weights/` first as above):

```bash
docker build -t rewind .
docker run --rm -v "$PWD/data/videos:/in:ro" -v "$PWD/outputs:/out" rewind \
    /in/<video_id>.mp4 /out/<video_id>.json
```

The image is built and smoke-tested in CI (`.github/workflows/docker.yml`).

## Data used

- **EPIC-KITCHENS-100** videos (official download, MD5-checked) and annotations.
  **FPVSum** videos (YouTube) and its ground truth. Neither is redistributed here.
- **Inference** uses only the MP4 and the weights below. Ground truth is read only by
  the offline evaluation package `code/rewind_eval/` and the tools.
- **ActionFormer's published SlowFast features** for EPIC-KITCHENS-100 are used for
  the EPIC videos (we have no GPU to re-extract them). They ship in
  `weights/released_features`, keyed by each MP4's content fingerprint, so `run.sh`
  on the official EPIC files finds them; an EPIC video without them falls back to
  zero-shot actions. `tools/fetch_weights.sh` fetches them from our release; rebuild
  from ActionFormer's archive with `RELEASED_FEATURES=1 tools/fetch_weights.sh`.
  The rebuild reads `tools/data/epic_fingerprints.jsonl`, which pairs each EPIC
  video's ID with the content fingerprint of its MP4. It is an offline import aid,
  used only to file the released features under fingerprints. It is not in the
  submission package, and inference never reads it: `run.sh` computes the MP4's
  fingerprint and looks that up, so it never uses a filename or video ID.
- The verb-noun co-occurrence prior is counted from `EPIC_100_train.csv` only.
- **Same video bytes, same JSON.** Output is deterministic for a given MP4, so a re-run
  reproduces `outputs/` exactly only on byte-identical files. EPIC files from the official
  downloader (MD5-checked) match ours. FPVSum videos come from YouTube: a different
  `yt-dlp` format or re-encode changes the pixels, so highlights and labels can differ
  slightly from `outputs/` even though the code is identical.

## Tuning-split statement

All hyperparameters were chosen on the **`tune` split only**: 20 EPIC-KITCHENS-100
*train* videos (never val). The `report` split (the 138 EPIC val videos and the 49
FPVSum videos) was used only to measure. Two caveats, both recorded in `PROGRESS.md`:

- The `tune` EPIC videos are ActionFormer's own training videos, so detector settings
  were left at their defaults rather than tuned there.
- **No setting comes from the FPVSum annotations.** The organizers asked teams to
  treat them as hidden test labels (2026-11-20). FPVSum has no videos outside the
  scored 49; we had run cross-validation experiments on them earlier, and the only
  values ever adopted from those (two audio importance weights) are set back to 0.
  The FPVSum importance weights and label settings are the untuned defaults.

## Reproduce

```bash
pip install -r requirements-dev.txt
tools/fetch_annotations.sh                       # eval ground truth (sha256-checked)
python tools/run_all.py --manifest kit/manifest.csv --videos data/videos --out outputs/
python -m rewind_eval.validate --dir outputs/    # every kit rule
python -m rewind_eval.score --pred outputs/ --split report --tag "reproduce"
pytest                                           # fast suite (< 60 s)
```

`tools/predict_cached.py --split report` produces the same documents from the
feature cache in minutes; `tools/ablate.py` and `tools/deck_numbers.py` rebuild the
tables and numbers above.

## Deck and demo

Submission deck: [`docs/VITVellore_Nexus.pdf`](docs/VITVellore_Nexus.pdf) (PowerPoint: `docs/VITVellore_Nexus.pptx`,
rebuilt by `tools/make_deck.py`, which needs `pip install python-pptx`).

## Demo tools

Not part of the submission path; both read our output JSON and the original MP4.

```bash
# one page over any number of outputs; open it in a browser
python tools/timeline_html.py --json outputs/*.json --videos data/videos --out demo/timeline.html
# the demo video: problem, live examples, results (about 2.5 min)
python tools/make_demo.py --epic <epic_id> --fpvsum <fpvsum_id> <fpvsum_id> --out demo/rewind_demo.mp4
# captioned highlight reel of one video
python tools/make_reel.py --video data/videos/<video_id>.mp4 \
    --json outputs/<video_id>.json --out demo/reel.mp4
```

The timeline page plays the video with our highlights and detected actions under it:
click to seek, jump between highlights, or play the highlights only.

## Models and licenses

| Component | Source | License |
|---|---|---|
| ActionFormer (code, vendored in `code/third_party/actionformer`) | `happyharrycn/actionformer_release` @ `61ea7eb` | MIT |
| ActionFormer EPIC-100 verb/noun checkpoints | released with ActionFormer (README links) | distributed with the MIT-licensed code; no separate license |
| SlowFast EPIC features | ActionFormer's `epic_kitchens.tar.gz` | EPIC-KITCHENS license |
| CLIP ViT-B/16 (OpenAI weights) via open_clip | HF `timm/vit_base_patch16_clip_224.openai` @ `977e3dd` | MIT |
| MediaPipe Hand Landmarker (library and model) | `mediapipe==1.2.2`, float16 v1 | Apache-2.0 |
| OpenCV, PyAV / FFmpeg | `opencv-contrib-python`, `av` | Apache-2.0; BSD / LGPL |
| EPIC-KITCHENS-100 annotations (eval only) | `epic-kitchens-100-annotations` @ `ea8b404` | CC BY-NC 4.0 |
| FPVSum ground truth (eval only) | `azuxmioy/fpvsum` @ `9af31f9` | research use |

Exact versions, hashes and sources: `ARCHITECTURE.md` §10, `tools/fetch_weights.sh`,
`tools/weights.sha256`.

## For the team

- **`CLAUDE.md`**: how we work (rules, commands, testing standard). Read first.
- **`ARCHITECTURE.md`**: the build spec. **`PROGRESS.md`**: live status, decisions, log.
- `kit/`: the organizers' student kit, unmodified.
