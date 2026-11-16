# ARCHITECTURE.md — REWIND build spec

> **Status: LOCKED — v1.1, 2026-11-11.** Build against this spec.
> v1.1: rules checked against the official kit in `kit/`; inference routes on video
> signals only (never `manifest.csv`, never the filename); one output schema; 2-decimal
> timestamps. See `PROGRESS.md` → Decisions.

This is the build spec. Code must match it. Rationale for the choices is in
`REWIND_v2.3_plan.md`. Status is in `PROGRESS.md`.

**Change control after the lock**
- *Resolving a `[verify]` item* (kit facts, organizer answers, checkpoint details) is
  not a design change: update this file in the PR that resolves it, and log it in
  `PROGRESS.md`.
- *Kill-rule outcomes* listed in `PROGRESS.md` (fallback detector, label-dedup
  selector) are pre-approved: switch, update this file, log the decision.
- *Anything else* (a new module, a changed algorithm, a new output behavior) needs a
  `PROGRESS.md` Decisions entry agreed by both A and B **before** the code PR, and
  this file updated in that PR. Bump the version (v1.1, v1.2 …) in the status line.
- Stretch work never changes the MVP design.

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

All are frozen dataclasses. Times are float seconds. `Signals` uses `eq=False`
(numpy arrays don't compare with `==`). Build `Cluster.pair_dist` as a read-only
mapping (`types.MappingProxyType`) so a frozen cluster can't be mutated through it.

```python
Segment = tuple[float, float]                      # (start, end), 0 <= start < end <= T

@dataclass(frozen=True)
class VideoMeta:
    video_id: str          # filename stem; echoed into the JSON, never used for any decision
    path: Path
    duration_sec: float    # from ffprobe (the eval validator checks it against the manifest)
    fps: float
    width: int
    height: int
    dataset: Literal["epic_kitchens_100", "fpvsum", "unknown"]
    route_conf: float      # signal-router margin; below cfg.routing.min_conf = uncertain

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
| `constants.py` | Kit facts: `MIN_SEG=4.0`, `MIN_SEG_EMIT=4.05`, `SCORING_BUDGET=0.15`, `DURATION_TOL=0.5`, `MAX_ACTIONS=2000`, `SHORT_VIDEO_T=MIN_SEG_EMIT/SCORING_BUDGET` (27.0 s: the shortest video whose budget fits one emitted 4.05 s window), `TIME_DECIMALS=2`, `END_TOL=0.5`, `REQUIRED_KEYS`, dataset names. Confirmed against `kit/TEAM_README.md`; every value cites its kit section in a comment | — |
| `config.py` | Load and validate `configs/default.yaml` into typed dataclasses; `config_hash()` | yes |
| `types.py` | §2 types | — |
| `vocab.py` | Kit classes from the bundled copy `rewind/resources/classes.json` (byte-identical to `kit/supported_classes.json`, so `run.sh` runs without `kit/`); index ↔ key maps; verb–noun co-occurrence prior | load at edge, lookups pure |
| `video_io.py` | Metadata: `ffprobe_meta` (used by `run.sh`) or PyAV `probe_video`. Frames: PyAV decode of the first video stream at a target fps by presentation time, relative to the stream start (D1); frame ↔ seconds | no |
| `routing.py` | Dataset routing from video signals only (§6.1) | yes, given inputs |
| `features/flow.py` | Farneback flow at low resolution → residual + global motion | no |
| `features/hands.py` | MediaPipe hand presence | no |
| `features/clip.py` | CLIP image embeddings @ 1 fps; text embeddings for prompts | no |
| `features/slowfast.py` | ActionFormer input features (§10.1): read from the cache (where `tools/import_released_features.py` puts ActionFormer's released EPIC-100 features); otherwise extract with the matching SlowFast extractor | no |
| `features/cache.py` | Versioned on-disk cache (§8) | no |
| `features/extract.py` | `load_signals`: flow, hands and CLIP for one video from the cache, else one decoding pass for the missing ones, stored back (§4 step 3); `tools/cache_features.py` batches it | no |
| `detect/actionformer.py` | Run vendored ActionFormer → raw verb/noun proposals | no |
| `detect/pairing.py` | Raw proposals → `Proposal` with paired, scored (verb, noun) (§5.1) | yes |
| `detect/fallback_zeroshot.py` | Sliding-window CLIP zero-shot detector (kill-rule fallback) | no |
| `cluster.py` | Proposals → Clusters (§5.2) | yes |
| `select/candidates.py` | Candidate windows (§5.3) | yes |
| `select/coverage.py` | Expected-coverage gain + lazy greedy (§5.4) | yes |
| `select/hedge.py` | Hedge windows (§5.5) | yes |
| `select/epic.py` | `select_epic(clusters, T, cfg, mode, profile)`: expected coverage on `(1 − h)·B`, hedges on whatever budget is left; `mode="dedup"` = label-dedup baseline (top-1 pair per cluster) | yes |
| `select/importance.py` | FPVSum importance proxy + segment selection (§6.2) | yes |
| `postprocess.py` | Sort, merge, enforce ≥ 4 s, budget trim, clamp (§5.6) | yes |
| `label/` | EPIC → `cooking`; FPVSum → CLIP zero-shot (§6.3) | scoring pure |
| `schema.py` | `validate(doc, vocab, *, filename_stem, expected_duration, strict) -> (errors, warnings)` per §7; `run.sh` passes `strict=True` and no expected duration, the eval CLI passes the manifest duration | yes |
| `output.py` | `VideoResult` → JSON dict in the kit schema (rounded to 2 decimals, sorted) | yes |
| `fallback.py` | Safe output: uniform windows, empty actions, default label | yes |
| `analyze.py` | §4 step 4. `summarize` (pure): route (§6.1) → actions (ActionFormer detections if given, else zero-shot; skipped on the FPVSum route) → label (§6.3) → highlights (EPIC selector when the label is `cooking`, else the importance proxy) → postprocess. `load_models` / `analyze_video`: the edge (weights from `cfg.detect` and `cfg.features`, feature cache, SlowFast cache lookup) | `summarize` yes |
| `pipeline.py` | `run_video(video, out, cfg, *, team_id, analyze)`: probe → `analyze` (default `rewind.analyze`, models loaded once per process) → `to_document` → strict validation → atomic write; any exception or validation error → validated fallback | no |
| `cli.py` | `python -m rewind.cli <video> <out> [--config]`; called by `run.sh`. Exit 0 = valid JSON written (maybe the fallback); exit 2 = no readable duration, nothing written | no |
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
| `validate.py` | CLI over `rewind.schema.validate` for a directory: durations against `kit/manifest.csv`, one file per manifest video, same `team_id` everywhere; `--kit-only` for kit severities |

### 3.3 `tools/`

`fetch_weights.sh`, `fetch_annotations.sh`, `cache_features.py` (batch, longest
videos first, resumable), `run_all.py`, `tune.py` (grid/random search on `tune` only),
`ablate.py`, `latency.py`, `make_reel.py` (ffmpeg reel from a JSON, optional padding
and crossfades for the demo only), `timeline_html.py` (demo viewer: one static
HTML page over output JSONs, template in `timeline_template.html`; demo only).

---

## 4. `run.sh` contract and pipeline flow

```
./run.sh <video.mp4> <out.json>
```

1. Use `./.venv/bin/python` if present, else `$PYTHON`, else `python3`; set
   `PYTHONPATH=code`; run `python -m rewind.cli "$1" "$2" --config configs/default.yaml`
   (or `$REWIND_CONFIG` when set; the contract tests use a config without weights).
2. `cli` seeds everything, builds `VideoMeta` (ffprobe), routes the dataset.
   `video_id` is the **output** file's stem (R2 requires it to match the JSON name);
   the input file name is not used at all.
3. Features: read from `cache/` if a valid versioned entry exists; else compute.
   Prompt text embeddings are cached too (`cache/text/`, versioned by the CLIP
   weights): `run.sh` is one process per video and encoding them costs ~30 s on CPU.
   (On a clean machine nothing is cached; everything is computed from the MP4.)
4. EPIC route: detect → pairing → `actions[]`; cluster → coverage windows + hedge
   → postprocess → `highlights[]`; label = `cooking`.
   FPVSum route: importance proxy → segments → postprocess → `highlights[]`;
   `actions[]` = `[]` (the kit allows it; mAP is scored on EPIC only); label = CLIP zero-shot.
   Uncertain route (§6.1 confidence below threshold): **always emit actions** (EPIC
   detector if SlowFast features are available, else the zero-shot fallback
   detector), pick the label by CLIP P(cooking) ≥ threshold, and select highlights
   with the route that label implies. A misrouted EPIC video must never lose its
   actions.
   ActionFormer runs only when the video's SlowFast features are in the cache. Until
   the MP4 SlowFast extractor exists (§10.1 item 2, a P4 open item), an EPIC video
   without cached features gets its actions from the zero-shot detector too.
5. `output.py` builds the dict (action times clamped into `[0, T]`, non-finite actions
   dropped) → strict `schema.validate`. On errors, **salvage**: drop actions with
   out-of-vocabulary names, replace an unknown label with the fallback label, replace
   highlights only if they are invalid on their own; validate again. Only if that
   still fails, write the full `fallback.py` output (validated too).
6. Any exception at any step → fallback output, error to stderr, exit code 0. The
   one exception: a video with no readable duration (missing, corrupt) gets no JSON
   and exit code 2, because R3 can't be satisfied without a duration.
7. Write JSON atomically (temp file + rename). The validated fallback is written
   **before** step 3, so a hard kill during analysis (SIGKILL, OOM killer, native
   crash: no `except` runs) still leaves a valid file; a successful run replaces it.

`TEAM_ID` comes from the `REWIND_TEAM_ID` env var, defaulting to `team.team_id` in
`configs/default.yaml`. It must be identical in every file (kit). The kit examples use
`T07`; use the ID the organizers assigned us **[verify Q11]**.

**Inference inputs.** `run.sh` gets only the MP4 path and the output path. The
organizers re-run it on sample videos in an isolated environment, and hardcoded
video IDs mean disqualification (kit §6). So inference never reads `kit/manifest.csv`
and never derives the route, label or anything else from the filename; `video_id` is
only echoed into the JSON. A contract test renames the input and checks that only
`video_id` changes.

---

## 5. EPIC route algorithms

### 5.1 Pairing (`detect/pairing.py`)

ActionFormer's EPIC models produce verb proposals and noun proposals separately.
Greedy anchors: visit verb proposals by (probability desc, start, key); a verb anchors
one proposal if it has a noun with tIoU ≥ `cfg.pairing.match_tiou` and doesn't overlap
(same threshold) an anchor already emitted. So duplicate labels on one span give one
proposal, and a confident verb without a matching noun never blocks an overlapping verb
that has one. Each anchor's group is the verbs and nouns overlapping it; for each pair
of its top-k verbs and top-k nouns:

```
score(v, n) = p(v) · p(n) · prior(v, n)^α
prior(v, n) = (count_train(v, n) + ε) / Σ_n' (count_train(v, n') + ε)
```

`count_train` comes from `EPIC_100_train.csv` only, and is precomputed into
`weights/vn_prior.npy` by a tool, so inference doesn't read annotations. Keep pairs
whose verb and noun are both in the kit vocabulary. Rescale scores to [0, 1] per
video using a fixed monotone map (not per-video min-max, which would break
cross-video ranking). Emit up to `cfg.actions.max_per_video` (≤ 2000) sorted by score.

α differs by use (PROGRESS Decisions 2026-11-13): the proposals given to clustering and
the selector use α = `cfg.pairing.alpha` (the prior helps coverage), while the emitted
`actions[]` are paired again with α = `cfg.actions.prior_alpha` (0: the prior lowered
mAP on `tune`, 0.527 → 0.653 without it).

### 5.2 Clustering (`cluster.py`)

Proposals are clustered so the selector reasons about *distinct actions*, not raw
detections:

1. Sort proposals by start time.
2. Link two proposals if tIoU ≥ `cfg.cluster.tiou` **and** their top-1 pairs match or
   the cosine similarity of the mean CLIP embedding over their spans ≥ `cfg.cluster.sim`
   (embeddings optional; without them only the pair rule applies). Links are
   transitive (union-find).
3. Connected components → clusters. `pair_dist` = score-weighted sum of member
   top-k pair distributions, truncated to top `cfg.cluster.k` (default 5),
   renormalized to 1, then temperature-scaled with `cfg.calibration.temperature` (default 1.0).

### 5.3 Candidate windows (`select/candidates.py`)

For lengths `L ∈ cfg.select.window_lengths` (default `[4.05, 5.0, 6.0]`) and starts on
a grid of `cfg.select.stride` (default 1.0 s; tuned on `tune`, PROGRESS Decisions 2026-11-13), keep windows inside `[0, T]` that
overlap ≥ 1 cluster span (window ends clamped to `≤ T`). A cluster's capture
probability is computed from how many seconds of its span the union covers:

- `any_overlap` rule: `cap = 1` if the union overlaps the span at all.
- `fraction` rule: `cap = sigmoid((covered / span_len − θ) / τ)` with θ, τ in config.

The rule is `cfg.select.capture_rule` **[verify Q4]**; the default is `any_overlap`.

### 5.4 Expected-coverage selection (`select/coverage.py`)

State is the union `U` of selected windows, merged exactly as post-processing
merges highlights (gaps < `max(merge_gap, 0.01 s)` filled), so the model scores what
is emitted. Each cluster is one action whose pair is drawn from `p_c`; `cap_c(U)` is
the capture probability given how much of its span `U` covers (§5.3).

```
P_miss[pair]      = Π_c (1 − p_c(pair) · cap_c(U))
E(U)              = Σ_pair (1 − P_miss[pair])          expected distinct pairs captured
gain(w | U)       = E(U + w) − E(U)
cost(w | U)       = |U + w| − |U|                      new seconds, incl. filled gaps
priority(w | U)   = gain / cost                        (gain or cost ≈ 0 → skip)
```

(v1.0 multiplied a miss factor per selected *window*, so two overlapping windows over
one cluster counted as two independent chances to catch the same action. Corrected;
see `PROGRESS.md` Decisions.)

Lazy greedy with a max-heap of priorities; gains and costs are rounded (1e-12, 1e-9)
so equal choices tie exactly and the earliest window wins. After each pick, the windows
next to the changed component are re-scored at once: their cost can drop (overlap) and
their gain rise (filled gaps), which plain lazy evaluation misses (a review
counterexample is a pinned test). Every union component
contains a whole window, so it is ≥ 4.05 s (no fragment check needed). A window that
would push the total over `(1 − h) · B` (`B = cfg.select.budget_frac · T`, `h` the
hedge share) is dropped. Stop when the heap is empty. Under `any_overlap`, E as a
function of a set of segments is monotone submodular, and tests check monotonicity and
diminishing returns (`CLAUDE.md` §7.2). Under `fraction` it is not (the sigmoid is convex
below θ), and union merging makes the window-level objective slightly non-submodular. **We claim no
approximation guarantee for the full procedure:** cost-ratio greedy under a budget,
the ≥ 4 s and fragment rules, the hedge reserve and post-processing all break the
conditions for one. Quality is measured empirically as **% of oracle coverage** (§9.5);
that is the number we report in the deck.

Complexity: windows ≈ 3 · T / 0.5, so ~12k for a 33-minute video; clusters per
window are few. Each evaluation only looks at union components within one span length
of the change. Measured: 1.4 s for 2000 random proposals with top-5 pairs on a 2000 s
video (the actions cap; `slow` test), 0.6 s on GT-derived clusters; must stay under 2 s.

### 5.5 Hedge windows (`select/hedge.py`)

Activity score `a(t) = z(flow)·w_f + hands·w_h` on the dense grid. Mask out times
inside selected windows (± `cfg.hedge.gap`) and inside clusters with
`confidence ≥ cfg.hedge.strong`. Take peaks of the smoothed `a(t)` spaced
≥ `cfg.hedge.min_spacing`; place `MIN_SEG_EMIT` windows centered on them, highest
first, until the hedge budget is used. Without an activity profile, farthest-point
sampling: each hedge goes where the distance to the nearest selected highlight, video end
or earlier hedge is largest, so hedges spread over the uncovered stretches. Hedges stay
≥ one window + `max(hedge.gap, merge_gap)` from each other and from selected highlights.
The hedge budget is `B − |U|`: the reserve `h · B` plus any coverage budget the selector
didn't use. Default `h = 0`: on simulated proposals a reserve of 1/8 cost ~0.05 coverage
(no activity signals yet); re-tune once P3's flow/hands profile exists.

**Short videos** (`B < 4.05 s`, i.e. T < 27.93 s at 14.5%): `select_epic` picks the single
best `min(4.05, T)` window with the same expected-coverage gain instead of leaving it to
post-processing, which would centre it on the middle of the video.

### 5.6 Post-processing (`postprocess.py`), shared by both routes

`postprocess(segments, values, T, *, budget_frac, merge_gap, short_video_mode,
profile=None)`. Input: any list of segments with one value each (higher = keep), and
optionally a value profile over time `(t, v)`. Output satisfies every highlight rule
in §7:

1. Clamp to `[0, T]`; drop empty and non-finite segments.
2. Sort by start; merge segments that overlap, touch, or have a gap <
   `max(cfg.post.merge_gap, 0.01 s)` (a smaller gap could round to touching). A merged
   segment's value is the max of its parts.
3. Drop segments shorter than `MIN_SEG_EMIT` (not just `MIN_SEG`: 4.01 s could round
   to 3.99 s). Selectors should not produce them; this is a safety net.
4. While total > `cfg.select.budget_frac · T′` (T′ = T floored to 2 decimals, the
   duration as written, so rounding can't break R13): shrink the lowest-value segment
   (ties: the later one) from its lower-value end per the profile (the start if there
   is no profile), never below `MIN_SEG_EMIT`; if it is already at the minimum, drop it.
5. **Short video** (`T < SHORT_VIDEO_T` = 4.05 / 0.15 = 27.0 s; 3 EPIC videos:
   11.13 s, 15.28 s, 25.16 s):
   R11 (≥ 1 highlight of 4 s) and R13 (≤ 15%) cannot both hold. The kit makes R11
   pass/fail and R13 a warning with a soft coverage penalty, so we emit one segment of
   `min(MIN_SEG_EMIT, T)` at the best-value position (`cfg.post.short_video_mode:
   min_seg`). The alternative `within_budget` (one segment of `SCORING_BUDGET · T`)
   fails R11 and stays only as a flag for tests; never ship it.
6. If the result is empty and `T ≥ MIN_SEG`: one `MIN_SEG_EMIT` window centred on
   the best input segment from before step 4, else on the profile's peak, else on
   `T/2`. This also covers `T` between 27.0 s and 27.93 s, where `budget_frac · T` is
   below 4.05 s but the kit's 15% still fits one window.
7. Round to 2 decimals (kit) *inside* the bounds (round start up and end down so that
   lengths stay ≥ `MIN_SEG`).

---

## 6. FPVSum route

### 6.1 Routing (`routing.py`)

Routing uses the video's content only (see "Inference inputs" in §4). Never the
manifest, never the filename: FPVSum filenames contain the category
(`Ski_-Z5a4UQmZLs`) and EPIC ids have their own pattern (`P01_11`), so either would
be a hardcoded shortcut.

1. CLIP P(cooking-scene prompts) averaged over keyframes ≥ `cfg.routing.cooking_thr`
   → EPIC, else FPVSum.
2. The route confidence is the margin from the threshold. Below
   `cfg.routing.min_conf` the route is **uncertain** and §4 step 4's uncertain-route
   rule applies.

Offline, `rewind_eval` measures router agreement with the manifest's `dataset`
column on all 187 videos; the target is ≥ 95%, and P6 tunes the threshold on `tune`.

### 6.2 Importance and segments (`select/importance.py`)

Per dense timestep:
`imp(t) = w1·z(flow) + w2·hands + w3·z(global_motion) + w4·z(scene_change)
+ w5·z(log(rms + 1e-4)) + w6·z(flux)`, then a Gaussian smoothing of σ =
`cfg.importance.sigma` s. Scene change = 1 − cosine similarity of consecutive CLIP
embeddings, interpolated to the dense grid. The audio terms (w5 = `w_audio_rms`,
w6 = `w_audio_flux`) use loudness and spectral flux of the mono track resampled to
`cfg.features.audio.sample_rate`, each summarised over `cfg.features.audio.window` s
around every dense timestep (`features/audio.py`); they are decoded only on this
route, cached by fingerprint, and left out when the video has no audio stream.
Weights are tuned on FPVSum cross-validation folds only.

Segments: find shots from CLIP change points (change > `cfg.importance.shot_thr`).
Within each shot, slide windows of `min_window`…`max_window` s (lengths every
`importance.len_step` plus `max_window` itself; starts every `importance.stride`) and
score each by mean `imp`. If no shot is long enough for a `min_window` window, windows
may span shots. Greedily pick the best non-overlapping windows until budget `B` (or the
single best window when `B < min_window`), then postprocess (§5.6).

Ablation candidate (P6): add audio terms (RMS loudness, spectral flux; extracted
with ffmpeg at 16 kHz) to `imp(t)`. Keep only if it wins on CV. Many FPVSum videos are
YouTube uploads with music overlays, which may make audio misleading.

Learned option (only if it beats the proxy on CV): ridge regression from CLIP
embeddings to per-second importance, 5-fold CV over videos.

### 6.3 Label (`label/`)

- EPIC route → `("cooking", 1.0)`.
- Names are exactly the keys in `kit/supported_classes.json` (`ski`, not `skiing`;
  the kit's `skiing_03.json` example uses `skiing`, which is not in the vocabulary
  **[verify Q10]**).
- FPVSum route → CLIP zero-shot over the 14 FPVSum categories with a prompt ensemble
  (`cfg.label.prompts`), averaged over `cfg.label.n_keyframes` (8) evenly spaced
  frames; the score is the softmax probability of the argmax. `other` only on the
  unknown route below a confidence threshold.

---

## 7. Output schema and validation rules (`schema.py`)

Source: the official kit, `kit/TEAM_README.md` §3–§4 (received 2026-11-11). The kit
lists a `validate.py`, but it was missing from our package **[verify Q8]**; until it
arrives, our validator is the reference.

```json
{
  "team_id": "T07",
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
| R3 | `duration_sec` positive finite number, within ±0.5 s of the manifest (checked offline; `run.sh` uses ffprobe) | error |
| R4 | `label` = `{name: str, score: [0,1]}`; name in kit labels | error / warning if unknown name |
| R5 | `actions` list, ≤ 2000 items | error |
| R6 | Each action: `segment` = 2 finite numbers with `0 ≤ s < e ≤ duration + 0.5` | error |
| R7 | Each action: `verb`, `noun` strings in the kit vocabulary | error / warning if out of vocabulary |
| R8 | Each action: `score` finite in [0, 1] | error |
| R9 | `highlights` non-empty list; each `segment` valid as in R6 | error |
| R10 | Highlights sorted by start and strictly non-overlapping (touching, `end == next start`, counts as overlap) | error |
| R11 | At least one highlight ≥ 4.0 s | error |
| R12 | Highlights < 4.0 s are ignored in scoring | warning (we treat it as error) |
| R13 | Total highlight duration ≤ 15% of `duration_sec` | warning (soft penalty) |
| R14 | Timestamps have at most 2 decimal places (kit: "should") | warning |

`strict=True` (what `run.sh` uses) also rejects what would lose points silently: R12,
R14, unknown names under R4/R7, and R13, except for a short video (`T < SHORT_VIDEO_T`)
whose only highlight is at most `MIN_SEG_EMIT` long; it also compares lengths and
bounds exactly (no float slack), as a plain `end - start >= 4.0` check would.
`strict=False` follows the kit's severities with a 1e-9 slack, except that unexpected
keys are always errors (conservative: the kit only requires the six to be present).

Timestamps have at most 2 decimal places (kit). Any invalid or missing file scores
0 on every metric for that video (kit §4).

**Scoring facts from the kit (§5):** mAP and coverage are scored on EPIC only; F1 and
label accuracy on FPVSum only. Coverage counts distinct GT (verb, noun) pairs captured
by highlights, with a soft penalty over 15%. Only a hidden subset of the 187 videos is
scored, so every output matters.

**Still open** (see `PROGRESS.md` → Open questions): what counts as "captured" (Q4;
we select with any-overlap and report both capture variants), and the shape of the
over-budget penalty (Q8).

---

## 8. Feature cache (`features/cache.py`)

```
cache/<feature>/<fingerprint>.npz   # arrays
cache/<feature>/meta.json           # {"version": "<extractor>-<params-hash>", "created": ..., "videos": {...}}
```

Entries are keyed by a **content fingerprint** of the MP4 (sha1 of its size plus its
first and last 1 MB), not by `video_id`, so inference never uses the filename (§4).
`tools/import_released_features.py` maps each released feature file to its video's
fingerprint once, offline. A clean machine has no cache and computes everything.

Features: `flow`, `hands`, `clip`, `slowfast`, `proposals` (detector output, so
selector work doesn't rerun models). The version string includes the model name,
weights hash, fps and resolution. A version mismatch → recompute. Writes are atomic.
`tools/cache_features.py` processes videos longest first and is resumable.

---

## 9. Evaluation definitions (`rewind_eval/`)

These are **our approximations** of the official metrics; the kit gives only one-line
definitions (`kit/TEAM_README.md` §5). Keep each one parameterized where the official definition is unknown.

### 9.1 Action coverage
For a video: let `G` be the set of distinct GT (verb, noun) pairs. A pair is
captured if any of its GT instances is captured by any highlight of length ≥ 4.0 s,
under the capture rule: `any_overlap` (overlap > 0; touching doesn't count) or
`fraction` (the valid highlights together cover ≥ θ of the instance's span, default
θ = 0.5). `coverage = |captured pairs| / |G|`; a video with no GT actions is left out
of the mean. Highlights under 4 s capture nothing but still count toward the budget.
Apply an over-budget penalty:
`× max(0, 1 − λ · max(0, total/T − 0.15) / 0.15)` with λ configurable (default 1.0)
**[verify Q4, Q8]**. Report the mean over EPIC videos. Always report both capture variants.

### 9.2 Action mAP
ActivityNet-style, with classes = (verb, noun) pairs present in GT. For each tIoU in
{0.1, 0.2, 0.3, 0.4, 0.5}: per class, sort predictions by score and greedily match
each to the unmatched GT of the same class in the same video with the highest tIoU
≥ threshold; AP = area under the interpolated precision–recall curve (all-point,
precision made monotone). Equal scores keep input order. A GT class with no
predictions has AP 0; predictions of classes absent from the GT are ignored. mAP =
mean over classes, then over thresholds. Computed over all EPIC videos jointly.

### 9.3 Highlight F1 (FPVSum)
Evaluated at the GT's native frames (frame `i` of `N` at `i · T / N` s, T from the
manifest). GT positives = FPVSum's own top-15% highlight frames (`Binary_score` in
`gt_data.mat`; 15.1–27.1% of frames because of ties, > 18% on 6 videos, so recall is
capped below 1 there at a 14.5% budget), the closest match to the kit's "top
15% of human-annotated consensus importance frames". `top_frames(score)` gives an
exact-count variant. Predicted positives = frames with `start ≤ t < end` for a
highlight ≥ 4.0 s. F1 per video (0 if no true positive), mean over videos.

### 9.4 Label accuracy
Exact match of `label.name` on FPVSum videos; a missing prediction counts as wrong.

### 9.5 Oracle
Greedy over candidate windows built from the GT (`rewind_eval/oracle.py`), using
`cfg.select.window_lengths` and `stride` like the selector; repeatedly add the window
with the most new gain per second that fits the budget (0.145·T, or one window for
short videos) and is ≥ 0.01 s from every chosen window. Coverage oracle: gain =
distinct GT pairs newly captured (any-overlap); the result is whichever of the greedy
windows and the uniform windows captures more, because greedy alone can lose to a
fixed layout (a counterexample is a pinned test). F1 oracle: gain = GT highlight frames
covered. A strong reference, not a proven optimum: the selector, which merges windows,
can beat it (101% on clean simulated proposals). Report `system / oracle` as "% of
oracle". Measured ceilings (report split, 4.05/5/6 s windows): coverage 0.737, F1 0.718.

### 9.6 Weighted total
`0.40·coverage + 0.25·mAP + 0.25·F1 + 0.10·label_acc`. Computed per split for the
leaderboard; it is our estimate, not the official score.

The fraction capture threshold θ of the metric is `cfg.eval.capture_theta` (0.5), not
the selector's `select.fraction_theta`, so tuning the selector never changes what a
leaderboard column measures. `--pred` files that fail kit-severity validation score as
empty (0), like a missing file.

### 9.7 Splits (`configs/splits.yaml`)
- `tune`: 20 EPIC-100 **train** videos (chosen to match the val duration
  distribution) + FPVSum 5-fold CV. `score --split tune` includes FPVSum only with
  `--fold k` (the held-out fold); without it the FPVSum columns are blank.
- `report`: the 138 EPIC val + 49 FPVSum videos in `kit/manifest.csv`.

---

## 10. Models, weights and licenses

Fill each row in before the model is used **[verify all]**.

| Use | Model / repo | Weights source | License | Notes |
|---|---|---|---|---|
| Action detection | ActionFormer (`happyharrycn/actionformer_release`, vendored in `code/third_party/actionformer`) | Released EPIC-100 verb and noun checkpoints (Google Drive links in its README) | MIT (code, vendored at `61ea7eb`; see `code/third_party/actionformer/VENDORED.md`) | Checkpoints `epoch_020.pth.tar` (verb 97 classes, noun 300; kit adds `other` → 98 / 301), SHA-256 pinned in `tools/fetch_weights.sh`. **Leakage check passed (2026-11-11):** both released configs have `train_split: ['training']`, `val_split: ['validation']`; their logs evaluate on 138 videos (= our EPIC val set) with avg mAP verb 23.51, noun 21.88 |
| ActionFormer input | ActionFormer's released `epic_kitchens.tar.gz` (Box / Google Drive / BaiduYun; md5 `add9803756afd9a023bc9a9c547e0229`) | — | EPIC-KITCHENS license | SlowFast pretrained on EPIC-100 train; 32-frame clips at 30 fps, stride 16 → one feature per 16/30 ≈ 0.533 s |
| Raw-MP4 feature extraction | — (**cut** 2026-11-14, PROGRESS Decisions) | | | The release covers all 158 EPIC videos we use; features ship in `weights/released_features`. A video without them gets zero-shot actions |
| CLIP | open_clip ViT-B/16 (OpenAI weights; `open_clip_torch==4.2.2`, CPU torch) | HF `timm/vit_base_patch16_clip_224.openai` `open_clip_model.safetensors` at revision `977e3dd`, SHA-256 pinned in `tools/fetch_weights.sh` | MIT (OpenAI CLIP weights; the HF repo is tagged Apache-2.0); open_clip MIT | Label, scene change, routing, clustering similarity. Loaded from the local file, no network at inference |
| Hands | MediaPipe Hand Landmarker (`mediapipe==1.2.2`, Tasks API, VIDEO mode) | `hand_landmarker.task` float16 v1 from storage.googleapis.com/mediapipe-models, SHA-256 pinned in `tools/fetch_weights.sh` | Apache-2.0 (library and model) | CPU. Presence = max handedness score of detected hands, 0 if none |
| Optical flow | OpenCV Farneback | — | Apache-2.0 (OpenCV) | Low resolution (160 px) |
| Verb–noun prior | Counts from `EPIC_100_train.csv` | Built by tool | EPIC-KITCHENS license | Train split only |
| GT annotations (eval only) | `epic-kitchens-100-annotations` @ `ea8b404`, `azuxmioy/fpvsum` @ `9af31f9` | `tools/fetch_annotations.sh` (sha256-checked) | EPIC: CC BY-NC 4.0; FPVSum: research use (no license file) | Never read by `rewind/` |

### 10.1 Feature strategy (CPU-first; neither team member has a GPU)

1. **Development and submission outputs for EPIC:** use ActionFormer's released
   SlowFast features, imported into the cache by fingerprint (§8). This removes the
   only heavy GPU job.
   P4 first checks that every EPIC `video_id` in `kit/manifest.csv` and every `tune`
   video is present.
   **Shipped with the submission** (2026-11-12): the features for the 138 manifest EPIC
   videos (~0.8 GB) live in a read-only cache under `weights/released_features`
   (`detect.released_features`). `analyze_video` looks for SlowFast features in the
   local cache, then there, always by the MP4's content fingerprint, so `run.sh` on the
   organizers' clean machine reproduces our ActionFormer actions instead of falling
   back to zero-shot. Rebuilt by `RELEASED_FEATURES=1 tools/fetch_weights.sh`.
2. **`run.sh` on a clean machine:** finds the shipped released features by the MP4's
   content fingerprint (the official EPIC files; ours are MD5-checked against
   `md5.csv`). Extraction from the MP4 was **cut** (PROGRESS Decisions 2026-11-14): an
   EPIC video whose features are not found gets the zero-shot detector instead.
3. **Gaps:** none (158/158 EPIC videos, 0 alignment warnings).
4. **README transparency:** state that submitted EPIC outputs used the published
   features from the same extractor, and how to regenerate them from raw MP4.
5. Everything else (decode, flow, hands, CLIP, ActionFormer inference, selection)
   runs on CPU.
6. **Data size:** the 138 EPIC videos are ~184 GB (13.2 h of video; kit §2), FPVSum
   49 videos ~2.4 h from YouTube via `yt-dlp`. Keep only what is being processed on
   disk if space is short; cached features are small.

---

## 11. Configuration (`configs/default.yaml`)

Top-level sections mirror the modules: `run` (seed), `team`, `signals`, `features`,
`actions`, `pairing`, `cluster`, `calibration`, `select`, `hedge`, `post`,
`importance`, `label`, `routing`, `eval`. Each key has a comment. The loader is
strict: unknown keys, missing keys and wrong types are errors, and range checks tie
values to the kit constants (e.g. `budget_frac ≤ 0.15`, every window ≥ 4.05 s).
Duplicate keys and non-finite numbers (NaN, inf) are errors too.
`config_hash(cfg)` (sha1 of the *loaded* `Config` as canonical JSON, so comments, key
order and `1` vs `1.0` don't matter) is written to every leaderboard row so results
are reproducible. `rewind_eval.score` must hash the same `Config` object that ran.

---

## 12. Performance targets

| Stage | Target |
|---|---|
| Full pipeline, 10-minute video, CPU, released SlowFast features | ≤ 10 min |
| Full pipeline, 10-minute video, CPU, extracting SlowFast from the MP4 | must complete; measure and report |
| Light signals (decode at low fps, flow, hands, CLIP), all 187 videos, one laptop | ≤ ~5 h (estimate; P3 measures it) |
| Selector + postprocess, any video | ≤ 2 s |
| Default test suite | ≤ 60 s |

Report measured latency per minute of video in the deck (`tools/latency.py`).

---

## 13. Submission package layout (`kit/TEAM_README.md` §6)

```
TEAM_ID/
├── run.sh
├── code/            # rewind/, rewind_eval/, third_party/
├── configs/         # default.yaml only (run.sh reads it; splits.yaml stays in the repo)
├── weights/         # exactly the paths configs/default.yaml names under weights/
├── requirements.txt
├── outputs/         # one <video_id>.json per manifest video
├── README.md        # setup, data used, how to reproduce, tuning split statement
└── Dockerfile       # + .dockerignore
```

Additions to the kit's list: `configs/` (run.sh needs the tuned config) and the
Dockerfile.
`tools/make_submission.py` builds the folder from committed code and refuses to while
any of these holds: `team.team_id` is a placeholder or differs from an output's; an
output is missing, fails strict validation or is the fallback (label `other`); a
configured weight is missing; a shipped path has uncommitted changes; or a shipped
file names a manifest video ID (case-insensitive substring; kit §6 scans for hardcoded
IDs; a test keeps the repo clean). `--check-video` re-runs the packaged run.sh from an
empty cache in a scratch directory and requires a byte-identical JSON with no stage
falling back; a package that fails it is removed. Weights are copied, not linked. No
tests, tools, kit, data or cache are shipped.

Plus the demo video link and the deck (`CollegeName_TeamName`).
