# ARCHITECTURE.md — REWIND build spec

This is the living build spec. Code must match it; when the design changes, update
this file in the same PR (see `CLAUDE.md` §5). Rationale for the choices is in
`REWIND_v2.3_plan.md`. Status is in `PROGRESS.md`.

Items marked **[verify]** depend on facts we haven't confirmed yet (kit contents,
organizer answers, checkpoint details). Confirm them before relying on them, then
remove the marker.

---

## 1. System overview

```
                       ┌─────────────────────────── run.sh <video.mp4> <out.json> ───────────────────────────┐
                       │                                                                                       │
 video.mp4 ──► [M0] ingest ──► VideoMeta ──► [M1] features ──► Signals ─┬─► [M2] detect ──► Proposals ─────┐   │
               video_io.py     (id, T,       features/          (flow,   │   detect/                         │   │
               routing.py       dataset)     (cached)            hands,  │                                   ▼   │
                                                                 CLIP,   │                      actions[] ◄── pairing
                                                                 SlowFast)│                                  │   │
                                                                         │   [M3] cluster ◄── Proposals ◄────┘   │
                                                                         │   cluster.py                          │
                                                                         │       │ Clusters                      │
                                                                         ▼       ▼                               │
                                                     [M4] select (EPIC: coverage windows │ FPVSum: importance)    │
                                                     select/                                                    │
                                                                         │ Segments                             │
                                                                         ▼                                      │
                                                     postprocess.py ──► highlights[]                            │
                                                                                                                │
                                                     [M5] label/  ──────► label                                 │
                                                                                                                │
                                                     [M6] output.py + schema.py ──► validate ──► out.json       │
                       └──────────────────── any exception ──► fallback.py (uniform windows) ──► out.json ──────┘
```

Two objectives, two outputs: `actions[]` maximizes mAP (detect everything, ranked);
`highlights[]` maximizes coverage (EPIC) or F1 (FPVSum). They share features but are
computed independently.

---

## 2. Core data types (`code/rewind/types.py`)

All are frozen dataclasses. Times are float seconds.

```python
Segment = tuple[float, float]                      # (start, end), 0 <= start < end <= T

@dataclass(frozen=True)
class VideoMeta:
    video_id: str          # filename stem
    path: Path
    duration_sec: float    # from ffprobe; cross-checked against manifest (±0.5 s)
    fps: float
    width: int
    height: int
    dataset: Literal["epic_kitchens_100", "fpvsum", "unknown"]
    routed_by: Literal["manifest", "signals"]

@dataclass(frozen=True)
class Signals:
    t_dense: np.ndarray        # (N,) timestamps at dense rate (cfg.signals.dense_fps, 2–4 fps)
    flow: np.ndarray           # (N,) residual-motion magnitude, z-scored per video
    global_motion: np.ndarray  # (N,) camera-motion magnitude, z-scored per video
    hands: np.ndarray          # (N,) hand-presence probability in [0, 1]
    t_clip: np.ndarray         # (M,) timestamps at 1 fps
    clip_emb: np.ndarray       # (M, D) L2-normalized CLIP image embeddings
    slowfast: np.ndarray | None  # (K, F) ActionFormer input features (EPIC route only)
    t_slowfast: np.ndarray | None

@dataclass(frozen=True)
class Proposal:
    segment: Segment
    verbs: tuple[tuple[str, float], ...]   # top-k (verb_key, prob), sorted desc
    nouns: tuple[tuple[str, float], ...]   # top-k (noun_key, prob), sorted desc
    pairs: tuple[tuple[str, str, float], ...]  # top-k (verb, noun, score), sorted desc

@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    span: Segment                          # union span of member proposals
    member_idx: tuple[int, ...]
    pair_dist: Mapping[tuple[str, str], float]  # top-k pairs, renormalized to sum 1
    confidence: float                      # max member pair score

@dataclass(frozen=True)
class VideoResult:
    team_id: str
    video_id: str
    duration_sec: float
    label: tuple[str, float]               # (name, score)
    actions: tuple[tuple[Segment, str, str, float], ...]
    highlights: tuple[Segment, ...]
```

Verb and noun names are always the **kit keys** from `kit/supported_classes.json`
(EPIC-100 canonical keys, e.g. `turn-on`, `board:chopping`). Map model class indices
to keys once, in `rewind/vocab.py`.

---

## 3. Package map

### 3.1 `code/rewind/` — inference (what `run.sh` runs)

| Module | Responsibility | Pure? |
|---|---|---|
| `constants.py` | Kit facts: `MIN_SEG=4.0`, `MIN_SEG_EMIT=4.05`, `SCORING_BUDGET=0.15`, `DURATION_TOL=0.5`, `MAX_ACTIONS=2000`, `SHORT_VIDEO_T=MIN_SEG/SCORING_BUDGET` (26.67 s) | — |
| `config.py` | Load and validate `configs/default.yaml` into typed dataclasses; `config_hash()` | yes |
| `types.py` | §2 types | — |
| `vocab.py` | Load kit classes; index ↔ key maps; verb–noun co-occurrence prior | load at edge, lookups pure |
| `video_io.py` | ffprobe metadata; frame decoding at a target fps; frame ↔ seconds | no |
| `routing.py` | Dataset routing: manifest lookup, else signal fallback (§6.1) | yes, given inputs |
| `features/flow.py` | Farneback flow at low resolution → residual + global motion | no |
| `features/hands.py` | MediaPipe hand presence | no |
| `features/clip.py` | CLIP image embeddings @ 1 fps; text embeddings for prompts | no |
| `features/slowfast.py` | ActionFormer input features, exactly matching its EPIC config | no |
| `features/cache.py` | Versioned on-disk cache (§8) | no |
| `detect/actionformer.py` | Run vendored ActionFormer → raw verb/noun proposals | no |
| `detect/pairing.py` | Raw proposals → `Proposal` with paired, scored (verb, noun) (§5.1) | yes |
| `detect/fallback_zeroshot.py` | Sliding-window CLIP zero-shot detector (kill-rule fallback) | no |
| `cluster.py` | Proposals → Clusters (§5.2) | yes |
| `select/candidates.py` | Candidate windows (§5.3) | yes |
| `select/coverage.py` | Expected-coverage gain + lazy greedy (§5.4) | yes |
| `select/hedge.py` | Hedge windows (§5.5) | yes |
| `select/importance.py` | FPVSum importance proxy + segment selection (§6.2) | yes |
| `postprocess.py` | Sort, merge, enforce ≥ 4 s, budget trim, clamp (§5.6) | yes |
| `label/` | EPIC → `cooking`; FPVSum → CLIP zero-shot (§6.3) | scoring pure |
| `schema.py` | `validate(result_dict, manifest_duration) -> (errors, warnings)` per §7 | yes |
| `output.py` | `VideoResult` → JSON dict (rounded to 3 decimals, sorted) | yes |
| `fallback.py` | Safe output: uniform windows, empty actions, default label | yes |
| `pipeline.py` | Wires M0–M6 for one video; catches exceptions → fallback | no |
| `cli.py` | `python -m rewind.cli <video> <out>`; called by `run.sh` | no |
| `utils.py` | `seed_all`, logging, timing | — |

### 3.2 `code/rewind_eval/` — evaluation (never imported by `rewind`)

| Module | Responsibility |
|---|---|
| `gt_epic.py` | Load EPIC-100 annotation CSVs → GT actions `(segment, verb_key, noun_key)` per video |
| `gt_fpvsum.py` | Load FPVSum frame-level importance → per-frame mean score; top-15% mask |
| `splits.py` | `tune` and `report` video lists (fixed, committed in `configs/splits.yaml`) |
| `coverage.py` | Our coverage metric (§9.1) |
| `map.py` | mAP over (verb, noun) pairs at tIoU {0.1…0.5} (§9.2) |
| `f1.py` | Frame-level F1 (§9.3) |
| `label_acc.py` | Label accuracy |
| `oracle.py` | Greedy max coverage using GT spans + same window rules (§9.5) |
| `baselines.py` | Uniform windows; random windows (seeded) |
| `score.py` | CLI: score a prediction dir on a split; append to `results/leaderboard.csv` |
| `validate.py` | CLI wrapper around `rewind.schema.validate` for a directory |

### 3.3 `tools/`

`fetch_weights.sh`, `fetch_annotations.sh`, `cache_features.py` (batch, longest
videos first, resumable), `run_all.py`, `tune.py` (grid/random search on `tune` only),
`ablate.py`, `latency.py`, `make_reel.py` (ffmpeg reel from a JSON, optional padding
and crossfades for the demo only), `timeline_html.py` (stretch).

---

## 4. `run.sh` contract and pipeline flow

```
./run.sh <video.mp4> <out.json>
```

1. Activate the bundled environment if present, then
   `python -m rewind.cli "$1" "$2" --config configs/default.yaml`.
2. `cli` seeds everything, builds `VideoMeta` (ffprobe), routes the dataset.
3. Features: read from `cache/` if a valid versioned entry exists; else compute.
   (On a clean machine nothing is cached; everything is computed from the MP4.)
4. EPIC route: detect → pairing → `actions[]`; cluster → coverage windows + hedge
   → postprocess → `highlights[]`; label = `cooking`.
   FPVSum route: importance proxy → segments → postprocess → `highlights[]`;
   `actions[]` = `[]` **[verify Q6]**; label = CLIP zero-shot.
   Unknown route: run both label scorers, pick by CLIP P(cooking) ≥ threshold.
5. `output.py` builds the dict → `schema.validate` → on errors, log them and write
   `fallback.py` output instead (which is validated too).
6. Any exception at any step → fallback output, error to stderr, exit code 0.
7. Write JSON atomically (temp file + rename).

`TEAM_ID` comes from the `REWIND_TEAM_ID` env var, defaulting to the registered
`CollegeName_TeamName` set in `configs/default.yaml`.

---

## 5. EPIC route algorithms

### 5.1 Pairing (`detect/pairing.py`)

ActionFormer's EPIC models produce verb proposals and noun proposals separately.
For each verb proposal, match noun proposals with tIoU ≥ `cfg.pairing.match_tiou`.
For each matched pair of top-k verbs and top-k nouns:

```
score(v, n) = p(v) · p(n) · prior(v, n)^α
prior(v, n) = (count_train(v, n) + ε) / Σ_n' (count_train(v, n') + ε)
```

`count_train` comes from `EPIC_100_train.csv` only, and is precomputed into
`weights/vn_prior.npy` by a tool, so inference doesn't read annotations. Keep pairs
whose verb and noun are both in the kit vocabulary. Rescale scores to [0, 1] per
video using a fixed monotone map (not per-video min-max, which would break
cross-video ranking). Emit up to `cfg.actions.max_per_video` (≤ 2000) sorted by score.

### 5.2 Clustering (`cluster.py`)

Proposals are clustered so the selector reasons about *distinct actions*, not raw
detections:

1. Sort proposals by start time.
2. Link two proposals if tIoU ≥ `cfg.cluster.tiou` **and** their top-1 pairs match or
   the cosine similarity of the mean CLIP embedding over their spans ≥ `cfg.cluster.sim`.
3. Connected components → clusters. `pair_dist` = score-weighted sum of member
   top-k pair distributions, truncated to top `cfg.cluster.k` (default 5),
   renormalized to 1, then temperature-scaled with `cfg.calibration.T` (default 1.0).

### 5.3 Candidate windows (`select/candidates.py`)

For lengths `L ∈ cfg.select.window_lengths` (default `[4.05, 5.0, 6.0]`) and starts on
a grid of `cfg.select.stride` (default 0.5 s), keep windows inside `[0, T]` that
overlap ≥ 1 cluster span. Precompute a sparse matrix `cap[w, c] ∈ [0, 1]`:

- `any_overlap` rule: `cap = 1` if the window and the cluster span overlap.
- `fraction` rule: `cap = sigmoid((overlap / span_len − θ) / τ)` with θ, τ in config.

The rule is `cfg.select.capture_rule` **[verify Q4]**; the default is `any_overlap`.

### 5.4 Expected-coverage selection (`select/coverage.py`)

```
P_miss[pair]      = Π over selected windows w, clusters c:  (1 − cap[w,c] · p_c(pair))
gain(w | S)       = Σ_pair P_miss[pair] · (1 − Π_c (1 − cap[w,c] · p_c(pair)))
cost(w | S)       = seconds of w not already covered by ∪S
priority(w | S)   = gain / max(cost, ε)        (cost == 0 → skip)
```

Lazy greedy with a max-heap of priorities. Before accepting `w`, simulate the union
and reject `w` if it would create any fragment < `MIN_SEG`, or push the total over
`(1 − h) · B` where `B = cfg.select.budget_frac · T` (default 0.145) and `h` is the
hedge share. Stop when no window fits. `gain` is monotone submodular in `S`, so
greedy keeps the (1 − 1/e) guarantee for the unit-cost case. Tests check monotonicity
and diminishing returns (`CLAUDE.md` §7.2).

Complexity: windows ≈ 3 · T / 0.5, so ~12k for a 33-minute video; clusters per
window are few. With lazy evaluation this runs in well under a second; it must stay
under 2 s per video.

### 5.5 Hedge windows (`select/hedge.py`)

Activity score `a(t) = z(flow)·w_f + hands·w_h` on the dense grid. Mask out times
inside selected windows (± `cfg.hedge.gap`) and inside clusters with
`confidence ≥ cfg.hedge.strong`. Take peaks of the smoothed `a(t)` spaced
≥ `cfg.hedge.min_spacing`; place `MIN_SEG_EMIT` windows centered on them, highest
first, until the hedge budget `h · B` is used. The default `h` is 1/8, tuned later.

### 5.6 Post-processing (`postprocess.py`), shared by both routes

Input: any list of segments. Output satisfies every highlight rule in §7:

1. Clamp to `[0, T]`; drop empty segments.
2. Sort by start; merge segments that overlap or have a gap < `cfg.post.merge_gap` (0.5 s).
3. Drop segments shorter than `MIN_SEG`. (Selectors should not produce them; this
   is a safety net.)
4. While total > `cfg.select.budget_frac · T`: shrink the lowest-value segment
   (value passed in by the caller) from its lower-value end, never below
   `MIN_SEG_EMIT`; if it is already at the minimum, drop it.
5. **Short video** (`T < SHORT_VIDEO_T`): return exactly one segment of length
   `min(MIN_SEG_EMIT, T)` at the best-value position. This deliberately exceeds the
   15% budget (soft penalty) because a < 4 s segment scores nothing.
6. If the result is empty and `T ≥ MIN_SEG`: one `MIN_SEG_EMIT` window at the
   highest-activity position.
7. Round to 3 decimals *inside* the bounds (round start up and end down so that
   lengths stay ≥ `MIN_SEG`).

---

## 6. FPVSum route

### 6.1 Routing (`routing.py`)

1. If `video_id` is in `kit/manifest.csv` → use its `dataset` column
   (`routed_by="manifest"`) **[verify Q9]**.
2. Otherwise use signals: CLIP P(cooking-scene prompts) averaged over keyframes ≥
   `cfg.routing.cooking_thr` → EPIC, else FPVSum (`routed_by="signals"`).

Test both paths. The signal router must reach ≥ 95% agreement with the manifest on
all 187 videos.

### 6.2 Importance and segments (`select/importance.py`)

Per dense timestep:
`imp(t) = w1·z(flow) + w2·hands + w3·z(global_motion) + w4·z(scene_change)`, then a
Gaussian smoothing of σ = `cfg.importance.sigma` s. Scene change = 1 − cosine
similarity of consecutive CLIP embeddings, interpolated to the dense grid.
Weights are tuned on FPVSum cross-validation folds only.

Segments: find shots from CLIP change points (`cfg.importance.shot_thr`). Within
each shot, slide windows of 4.05–10 s and score each by mean `imp`. Greedily pick the
best non-overlapping windows until budget `B`, then postprocess (§5.6).

Learned option (only if it beats the proxy on CV): ridge regression from CLIP
embeddings to per-second importance, 5-fold CV over videos.

### 6.3 Label (`label/`)

- EPIC route → `("cooking", 1.0)`.
- FPVSum route → CLIP zero-shot over the 14 FPVSum categories with a prompt ensemble
  (`cfg.label.prompts`), averaged over `cfg.label.n_keyframes` (8) evenly spaced
  frames; the score is the softmax probability of the argmax. `other` only on the
  unknown route below a confidence threshold.

---

## 7. Output schema and validation rules (`schema.py`)

Source: the kit's `TEAM_README.md` as reported second-hand **[verify against `kit/`]**.

```json
{
  "team_id": "CollegeName_TeamName",
  "video_id": "P01_11",
  "duration_sec": 561.53,
  "label": { "name": "cooking", "score": 1.0 },
  "actions": [ { "segment": [12.4, 15.9], "verb": "take", "noun": "knife", "score": 0.82 } ],
  "highlights": [ { "segment": [40.2, 44.3] } ]
}
```

| # | Rule | Severity |
|---|---|---|
| R1 | All six keys present | error |
| R2 | `video_id` non-empty string, equals the output filename stem | error |
| R3 | `duration_sec` positive finite number, within ±0.5 s of the manifest when listed | error |
| R4 | `label` = `{name: str, score: [0,1]}`; name in kit labels | error / warning if unknown name |
| R5 | `actions` list, ≤ 2000 items | error |
| R6 | Each action: `segment` = 2 finite numbers with `0 ≤ s < e ≤ duration + 0.5` | error |
| R7 | Each action: `verb`, `noun` strings in the kit vocabulary | error / warning if out of vocabulary |
| R8 | Each action: `score` finite in [0, 1] | error |
| R9 | `highlights` non-empty list; each `segment` valid as in R6 | error |
| R10 | Highlights sorted by start and non-overlapping | error |
| R11 | At least one highlight ≥ 4.0 s | error |
| R12 | Highlights < 4.0 s are ignored in scoring | warning (we treat it as error) |
| R13 | Total highlight duration ≤ 15% of `duration_sec` | warning (soft penalty) |

Our validator treats R12 as an error, and R13 as an error except for short videos,
so that we never ship something that loses points silently.

**Conservative assumptions until organizers answer** (see `PROGRESS.md` → Open
questions): any-overlap capture for selection, reporting both capture variants;
the < 4 s rule applies to coverage and F1; manifest routing with a working fallback.

---

## 8. Feature cache (`features/cache.py`)

```
cache/<feature>/<video_id>.npz    # arrays
cache/<feature>/meta.json         # {"version": "<extractor>-<params-hash>", "created": ..., "videos": {...}}
```

Features: `flow`, `hands`, `clip`, `slowfast`, `proposals` (detector output, so
selector work doesn't rerun models). The version string includes the model name,
weights hash, fps and resolution. A version mismatch → recompute. Writes are atomic.
`tools/cache_features.py` processes videos longest first and is resumable.

---

## 9. Evaluation definitions (`rewind_eval/`)

These are **our approximations** of the official metrics, which haven't been
published. Keep each one parameterized where the official definition is unknown.

### 9.1 Action coverage
For a video: let `G` be the set of distinct GT (verb, noun) pairs. A pair is
captured if any of its GT instances is captured by any highlight of length ≥ 4.0 s,
under the capture rule (`any_overlap` or `fraction ≥ θ`).
`coverage = |captured pairs| / |G|`. Apply an over-budget penalty:
`× max(0, 1 − λ · max(0, total/T − 0.15) / 0.15)` with λ configurable (default 1.0)
**[verify]**. Report the mean over videos. Always report both capture variants.

### 9.2 Action mAP
ActivityNet-style, with classes = (verb, noun) pairs present in GT. For each tIoU in
{0.1, 0.2, 0.3, 0.4, 0.5}: per class, sort predictions by score and greedily match
each to the unmatched GT of the same class in the same video with the highest tIoU
≥ threshold; AP = area under the interpolated precision–recall curve. mAP = mean
over classes, then over thresholds. Computed over all EPIC videos jointly.

### 9.3 Highlight F1 (FPVSum)
Sample at `cfg.eval.f1_fps` (default 2 fps). GT positives = the frames in the top 15%
of mean human importance. Predicted positives = frames inside highlights ≥ 4.0 s.
F1 per video, mean over videos.

### 9.4 Label accuracy
Exact match of `label.name` on FPVSum videos.

### 9.5 Oracle
The same window rules (≥ 4.0 s, budget 0.145·T, non-overlapping) with clusters
built from GT actions (one-hot pair distributions), selected with the same greedy
as §5.4. This gives an upper reference for coverage. Report
`system / oracle` as "% of oracle".

### 9.6 Weighted total
`0.40·coverage + 0.25·mAP + 0.25·F1 + 0.10·label_acc`. Computed per split for the
leaderboard; it is our estimate, not the official score.

### 9.7 Splits (`configs/splits.yaml`)
- `tune`: ~20 EPIC-100 **train** videos (chosen to match the val duration
  distribution) + FPVSum 5-fold CV.
- `report`: the 138 EPIC val + 49 FPVSum videos in `kit/manifest.csv`.

---

## 10. Models, weights and licenses

Fill each row in before the model is used **[verify all]**.

| Use | Model / repo | Weights source | License | Notes |
|---|---|---|---|---|
| Action detection | ActionFormer (vendored in `code/third_party/actionformer`) | Released EPIC-100 verb/noun checkpoints | | Must be trained on EPIC train only; record the evidence |
| ActionFormer input | Feature extractor matching ActionFormer's EPIC config | | | Match fps, clip length and stride from its config exactly |
| CLIP | open_clip ViT-B/16 (OpenAI weights) | | | Label, scene change, routing, clustering similarity |
| Hands | MediaPipe Hands | | | CPU |
| Optical flow | OpenCV Farneback | — | Apache-2.0 (OpenCV) | Low resolution (160 px) |
| Verb–noun prior | Counts from `EPIC_100_train.csv` | Built by tool | EPIC-KITCHENS license | Train split only |

---

## 11. Configuration (`configs/default.yaml`)

Top-level sections mirror the modules: `team`, `signals`, `features`, `actions`,
`pairing`, `cluster`, `calibration`, `select`, `hedge`, `post`, `importance`,
`label`, `routing`, `eval`. Each key has a comment. `config_hash()` (sha1 of the
canonical YAML) is written to every leaderboard row so results are reproducible.

---

## 12. Performance targets

| Stage | Target |
|---|---|
| Full pipeline, 10-minute video, one consumer GPU | ≤ 3 min |
| Full pipeline, 10-minute video, CPU only | ≤ 20 min (it must work, even slowly) |
| Selector + postprocess, any video | ≤ 2 s |
| Default test suite | ≤ 60 s |

Report measured latency per minute of video in the deck (`tools/latency.py`).

---

## 13. Submission package layout (from the kit)

```
TEAM_ID/
├── run.sh
├── code/            # rewind/, rewind_eval/, third_party/
├── weights/         # all weights needed to run offline
├── requirements.txt
├── outputs/         # one <video_id>.json per manifest video
└── README.md        # setup, data used, how to reproduce, tuning split statement
```

Plus a Dockerfile in the repo, the demo video link and the deck
(`CollegeName_TeamName`).
