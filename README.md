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

On the kit's 187 videos (`report` = 138 EPIC-KITCHENS-100 val + 49 FPVSum), scored
by our implementation of the kit metrics (`rewind_eval`, `results/leaderboard.csv`):

| | Weighted | Action coverage (40%) | Action mAP (25%) | Highlight F1 (25%) | Label acc. (10%) |
|---|---|---|---|---|---|
| **REWIND** | **0.361** | 0.437 (59.8% of oracle) | 0.149 | 0.263 | 0.837 |
| Uniform windows | 0.176 | 0.338 | 0 | 0.165 | 0 |
| Oracle (GT-aware) | 0.818 | 0.731 | 1 | 0.702 | 1 |

- **85.8% of visual tokens pruned** (highlight seconds vs video seconds, all 187).
- **39.4 s of CPU time per minute of video**, full pipeline from an empty cache on a
  laptop CPU (`tools/latency.py`, `results/latency.csv`). No GPU needed.
- Every component helps; ablations in `results/ablation_report.md` and
  `results/ablation_tune.md` (`tools/ablate.py`).

## How it works

1. **Signals** (`code/rewind/features/`): one PyAV decode pass → optical flow and
   camera motion (OpenCV Farneback), hand presence (MediaPipe), CLIP ViT-B/16 frame
   embeddings; audio loudness and spectral flux on the FPVSum route. All cached by
   the MP4's content fingerprint, never its name.
2. **Route** (`routing.py`): CLIP zero-shot P(cooking) decides EPIC (kitchen) vs
   FPVSum (outdoor activity); an uncertain video still gets actions.
3. **Actions** (`detect/`): ActionFormer's EPIC-100 verb and noun models on SlowFast
   features, paired into (verb, noun) actions; a CLIP zero-shot detector is the
   fallback.
4. **EPIC highlights** (`cluster.py`, `select/epic.py`): proposals are clustered into
   distinct steps; windows are chosen greedily by expected new coverage per second
   within the 15% budget, then post-processed to the kit rules.
5. **FPVSum highlights and label** (`select/importance.py`, `label/`): an importance
   proxy (flow, hands, camera motion, scene change, audio) → best windows per shot;
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
tools/fetch_weights.sh                     # downloaded model weights, SHA-256 checked
RELEASED_FEATURES=1 tools/fetch_weights.sh # + ActionFormer's released EPIC features
tools/fetch_annotations.sh                 # EPIC train CSV for the verb-noun prior below
python tools/build_vn_prior.py --train-csv data/annotations/epic/EPIC_100_train.csv
./run.sh data/videos/<video_id>.mp4 outputs/<video_id>.json
```

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
  zero-shot actions. Rebuild with `RELEASED_FEATURES=1 tools/fetch_weights.sh`.
  The rebuild reads `tools/data/epic_fingerprints.jsonl`, which pairs each EPIC
  video's ID with the content fingerprint of its MP4. It is an offline import aid,
  used only to file the released features under fingerprints. It is not in the
  submission package, and inference never reads it: `run.sh` computes the MP4's
  fingerprint and looks that up, so it never uses a filename or video ID.
- The verb-noun co-occurrence prior is counted from `EPIC_100_train.csv` only.

## Tuning-split statement

All hyperparameters were chosen on the **`tune` split only**: 20 EPIC-KITCHENS-100
*train* videos (never val) and, for FPVSum, 5-fold cross-validation (choose on four
folds, score the fifth). The `report` split (the 138 EPIC val videos and the 49
FPVSum videos) was used only to measure. Two caveats, both recorded in `PROGRESS.md`:
the `tune` EPIC videos are ActionFormer's own training videos, so detector settings
were left at their defaults rather than tuned there; and FPVSum has no videos outside
the scored 49, so only a handful of importance weights were fitted, by
cross-validation.

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

Exact versions, hashes and sources: `ARCHITECTURE.md` §10, `tools/fetch_weights.sh`.

## For the team

- **`CLAUDE.md`**: how we work (rules, commands, testing standard). Read first.
- **`ARCHITECTURE.md`**: the build spec. **`PROGRESS.md`**: live status, decisions, log.
- `kit/`: the organizers' student kit, unmodified.
