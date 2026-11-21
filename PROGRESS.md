# PROGRESS.md — REWIND live status

**Update this file in every PR** (rules: `CLAUDE.md` §5). Tick checklist items as
they land, add a Log line at the top of §5, and fill in the phase's Result block
when the phase is done. Never delete log entries; correct them with a new one.

Status values: `not started` · `in progress` · `blocked (reason)` · `done` · `cut`.
Owners: **A** = ML & perception · **B** = algorithms & infra.
Who is A and who is B: **A = prakartii**, **B = Daksh Arora (DA)**.

---

## 1. Phase overview

| Phase | Name | Owner | Planned | Status | Exit check |
|---|---|---|---|---|---|
| P0 | Kit & rules | B | Nov 10 | in progress | Kit committed in `kit/`; kit rules in `ARCHITECTURE.md` §7 confirmed or corrected; questions emailed |
| P1 | Skeleton & submission contract | B | Nov 10–11 | done | `run.sh` writes valid fallback JSON for all 187 videos; CI green |
| P2 | Evaluation harness | B | Nov 11–12 | done | Leaderboard has uniform + oracle rows on `tune` and `report` |
| P3 | Video I/O, signals, feature cache | A | Nov 10–12 | done | Flow, hands and CLIP cached for all 187 + `tune` videos |
| P4 | Action detection → `actions[]` | A | Nov 11–13 | done | mAP measured on `tune` and `report`; beats the zero-shot fallback |
| P5 | Clustering + EPIC coverage selector | B | Nov 12–14 | done | EPIC coverage > uniform; % of oracle reported; ablation done |
| P6 | FPVSum route: importance, segments, label | A | Nov 13–15 | done | F1 > uniform on FPVSum CV; label accuracy reported |
| P7 | Integration, tuning, ablations | A + B | Nov 15–16 | in progress | All 187 outputs from raw MP4 pass validation; tuned config committed; floors set |
| P8 | Packaging, demo, deck | B (pkg) + A (demo) | Nov 16–21 | in progress | Fresh clone + Docker reproduces one video; demo video + deck done |
| P9 | Submission | A + B | Nov 22 | in progress | Public GitHub repo with everything, submitted via the Google Form before 22 Nov 11:59 PM IST |

Critical path: P0 → P1 → P2 (scorer) and P3 → P4 (detector) run in parallel; P5
starts on GT-derived proposals before P4 lands, so neither person waits.

---

## 2. Phases

### P0 — Kit & rules
**Owner:** B · **Planned:** Nov 10 · **Status:** in progress (started 2026-11-11) · **Depends on:** —

- [x] Commit the official student kit unmodified to `kit/` (received 2026-11-11 as
      `Student package_track04.zip`: `TEAM_README.md`, `manifest.csv`,
      `supported_classes.json`, `P02_109.json`, `skiing_03.json`; files are flat, not in `examples/`)
- [x] Diff the kit against `ARCHITECTURE.md` §7 (schema, R1–R13): all rules match; `ARCHITECTURE.md` updated to v1.1
- [x] Check whether the official `validate.py` is in our kit: **no**, although `TEAM_README.md` lists it (asked in Q8)
- [ ] Email the organizers with what is still open, the penalty curve (Q8) and weights hosting (Q12) (draft in §3); record the date sent. Q4 and Q10 were answered in the 2026-11-20 mail. Since the 2026-11-17 mail, queries go to m.shankar@samsung.com and prakhar.c@samsung.com (not prism@)
- [x] Organizer mail of 2026-11-17 (deadline extension + final instructions) checked: the re-sent `Student package_track04.zip` is byte-identical to `kit/` (still no `validate.py`); the submission deck template is in `docs/organizers/`; consequences recorded in P8, P9, Open questions (Q7, Q12) and Decisions
- [x] Organizer mail of 2026-11-20 (answers to teams' questions, a reissued `TEAM_README.md` and `skiing_03.json`, sent as .docx "due to limitations", with a request to convert them): converted into `kit/` with the text unchanged (Markdown and JSON rebuilt from the .docx; the docx's "To-Do Problem Statement" title line dropped). The reissue drops `validate.py`, renumbers sections (references updated), states the capture rule and adds an FAQ; `skiing_03.json` now says `ski`. Answers recorded in §3 (Q4, Q8, Q10, Q13, Q14)
- [x] Confirm our assigned team ID and set `team.team_id` in config (Q11) — **decided 2026-11-17: our registered team name**, `VITVellore_Nexus`, set in `configs/default.yaml`
- [x] Output schema: the kit's nested six-key schema; the flat brochure variant is dropped
- [x] Kit constants confirmed: 4.0 s minimum, 15% budget (warning), ±0.5 s duration, 0–2000 actions, `end ≤ duration + 0.5`, 2 decimals, 16/98/301 classes, 187 videos (138 EPIC + 49 FPVSum)
- [x] Short videos: kit makes ≥ 4 s pass/fail and the 15% budget a warning, so `short_video_mode: min_seg` is the default (re-check with the official validator when it arrives)

**Exit check:** `kit/` committed; every `[verify]` in `ARCHITECTURE.md` §7 resolved
or listed in §3 below.
**Tests required:** none (documentation phase).

**Result (partial, 2026-11-11):** The kit confirmed every second-hand rule (R1–R13),
the nested schema and the class counts. New from the kit: organizers re-run `run.sh`
in an isolated environment and disqualify hardcoded video IDs → inference no longer
reads `manifest.csv` or uses the filename (Decisions); only a hidden subset of the
187 videos is scored; timestamps use 2 decimals; Q6 answered. Missing from the
package: `validate.py`. Remaining: send the email, set the team ID.

---

### P1 — Skeleton & submission contract
**Owner:** B · **Planned:** Nov 10–11 · **Status:** done (started 2026-11-11, finished 2026-11-15) · **Depends on:** P0 (schema confirmed)

- [x] `pyproject.toml` (ruff, pytest with `pythonpath = ["code"]`, markers `slow`, `data`, `gpu`), `requirements*.txt`, `.gitignore`
- [x] `configs/default.yaml` + `rewind/config.py` with `config_hash()` (strict loader)
- [x] `rewind/constants.py`, `rewind/types.py`
- [x] `rewind/vocab.py` (kit classes, bundled copy for runs without `kit/`)
- [x] `python -m rewind…` / `rewind_eval…` work outside pytest: `run.sh` and the docs set `PYTHONPATH=code` (or install the package)
- [x] `rewind/schema.py` implementing R1–R14 (strict and kit severities) + `rewind_eval/validate.py` CLI
- [x] `rewind/postprocess.py` (§5.6 of `ARCHITECTURE.md`)
- [x] `rewind/fallback.py` (uniform ≥ 4 s windows at 14.5%)
- [x] `rewind/output.py` (kit schema, 2-decimal rounding); `rewind/pipeline.py` (fallback-only for now), `rewind/cli.py`, `run.sh`
- [x] `post.short_video_mode` flag (`min_seg` / `within_budget`) in postprocess
- [x] `video_io.ffprobe_meta` (minimal; A extends it in P3)
- [x] GitHub Actions: ruff + `pytest` on every PR
- [x] Run on all 187 videos → `python -m rewind_eval.validate --dir outputs/` passes (blocked on A's downloads; stand-in test: the fallback at all 187 manifest durations passes strict directory validation) — **done 2026-11-15**: the full pipeline's 187 report outputs (`tools/predict_cached.py`, same `summarize` as run.sh) pass strict validation, ours and kit-only severities: 0 errors, 3 warnings (P03_26, P26_30, P28_23 are < 27 s, so the one 4.05 s `min_seg` window exceeds 15% by design, §5.6)

**Exit check:** all 187 dummy outputs pass our validator (and the official one if available); CI green on `main`.
**Tests required:**
- unit: one failing case per rule R1–R13; postprocess hand cases (merge, drop < 4 s, budget trim, short video, rounding keeps ≥ 4 s)
- property: postprocess and fallback outputs always satisfy R9–R13 for T in [5, 2000] s
- contract: `run.sh` on 30 s and 12 s synthetic videos → valid JSON; forced exception → fallback JSON, exit 0
- contract: `run.sh` works with no `kit/` directory present; renaming the input file changes only `video_id`
- contract: a `sys.addaudithook` during the `run.sh` test fails on any open of a manifest, annotation or `data/` GT path (covers `third_party/` and config-driven paths)
- import boundary test (`rewind` never imports `rewind_eval`, never opens `manifest.csv` or annotation paths)

**Result (2026-11-16, DA):** exit check met. All 187 full-pipeline outputs pass strict
validation, ours and kit-only severities: 0 errors, 3 budget warnings on the < 27 s
videos, by design (§5.6). CI (ruff + `pytest`) green on `main`, and the Docker job
builds the image and smoke-tests `run.sh` in it (#41). Still open, tracked elsewhere:
the official `validate.py` (Q8) and the team ID (Q11, P0).
Deviations from ARCHITECTURE.md: none outstanding (salvage before fallback, §4 step 5;
fallback written before analysis, §4).

---

### P2 — Evaluation harness
**Owner:** B · **Planned:** Nov 11–12 · **Status:** done (2026-11-11, pending PR review) · **Depends on:** P1

- [x] `tools/fetch_annotations.sh` (EPIC-100 train/val CSVs + class CSVs; FPVSum importance), documenting exact sources (pinned commits, sha256-checked)
- [x] `rewind_eval/gt_epic.py`, `gt_fpvsum.py` (class index → kit key mapping; all 138 EPIC and 49 FPVSum manifest videos have GT)
- [x] `configs/splits.yaml`: `tune` (~20 EPIC train videos matching the val duration spread) and `report` (generated by `tools/make_splits.py`)
- [x] `coverage.py` (both capture rules, < 4 s exclusion, penalty λ), `map.py`, `f1.py`, `label_acc.py`
- [x] `baselines.py` (uniform, random), `oracle.py`
- [x] `score.py` CLI → `results/leaderboard.csv` (SHA, tag, config hash, all metrics, split)
- [x] Leaderboard rows: uniform and oracle on `tune` and `report` (plus random)

**Exit check:** uniform + oracle rows exist; oracle coverage > uniform on every video.
**Tests required:**
- unit: every metric on hand-computed cases (arithmetic in comments); AP with ties and duplicate detections; each tIoU threshold; F1 at a known overlap
- property: metrics in [0, 1]; perfect prediction → 1.0; empty → 0.0; coverage never decreases when a highlight is added (within budget)
- unit: GT loaders on a 5-row CSV fixture (class keys mapped correctly)

**Result (2026-11-11; rows regenerated 2026-11-12 after review):**

| Split | Run | Cov (any) | Cov (frac) | % oracle | mAP | F1 | Label | Weighted |
|---|---|---|---|---|---|---|---|---|
| tune | uniform | 0.358 | 0.178 | 49.4% | 0 | — | — | — |
| tune | oracle | 0.725 | 0.361 | 100% | 1 | — | — | — |
| report | uniform | 0.338 | 0.177 | 46.7% | 0 | 0.163 | 0 | 0.176 |
| report | random | 0.312 | 0.184 | 43.1% | 0 | 0.158 | 0 | 0.164 |
| report | oracle | 0.724 | 0.330 | 100% | 1 | 0.708 | 1 | 0.817 |

(Tune rows score FPVSum only per CV fold, `--fold k`; without one the FPVSum columns
are blank.)

- **Ceilings:** with 4.05 s windows at 14.5%, even perfect action knowledge captures
  only ~72% of distinct pairs (any-overlap) and reaches F1 ~0.71. Uniform already gets
  ~47% of the coverage ceiling; the selector's job is the gap between 0.34 and 0.72.
- **Exit check:** oracle ≥ uniform on every video of both splits; 2 exact ties (short
  videos, one window each), so "≥", not strictly ">". Property-tested on random GT.
- **Deviations from ARCHITECTURE.md:** F1 GT = FPVSum's `Binary_score` at native frames
  (§9.3 updated); the oracle is a standalone greedy, not the §5.4 selector (§9.5 updated);
  FPVSum `tune` = CV folds over the same 49 videos (no separate tune videos exist).
- Scoring all 187 videos takes 1–2.5 s.
- **Short videos:** with the documented penalty, a single 4.05 s window scores 0 coverage
  on P03_26 (T = 11.13 s) and at most 0.233 on P26_30, whatever the system does (Q4/Q8).

---

### P3 — Video I/O, signals, feature cache
**Owner:** A (prakartii) · **Planned:** Nov 10–12 · **Status:** done (started 2026-11-11, finished 2026-11-14) · **Depends on:** — (P1 types; coordinate with B)

> **Coordination note (resolved 2026-11-12):** P3 started before P1 and carried minimal
> scaffolding; P1 has since merged and this branch was reconciled with `main` (P1 owns
> probing via `ffprobe_meta`; P3 adds PyAV decoding).

- [x] Download the 187 videos into `data/videos/` + the `tune` videos — **features for all 187 + 20 `tune` done**: FPVSum 49 downloaded locally; EPIC 138 downloaded and processed on Kaggle (videos deleted there, features kept). `tune` (2026-11-13): 20/20 EPIC train videos (111 min) processed on Kaggle (`--split tune`, 0 errors; flow, hands, CLIP collected) and their released SlowFast features imported by logged fingerprint (20/20, 0 alignment warnings)
- [x] `video_io.py`: decoding at a target fps, frame ↔ seconds (also ffprobe parsing, so P1's `ffprobe_meta` item is covered)
- [x] `features/flow.py` (residual + global motion)
- [x] `features/hands.py` (MediaPipe Hand Landmarker), `features/clip.py` (open_clip ViT-B/16, prompt ensemble, zero-shot); weights via `tools/fetch_weights.sh` (SHA-256 pinned)
- [x] `features/cache.py` (versioned, atomic)
- [x] `tools/cache_features.py` (longest first, resumable; logs `cache/throughput.csv`)
- [x] `routing.py` (video signals only; no manifest, no filename): pure decision from per-keyframe CLIP P(cooking); the CLIP prompts that produce it come with `features/clip.py`
- [x] Start caching on Nov 10; record throughput (seconds of compute per minute of video): flow + hands + CLIP in one pass ≈ 43–63 s per video minute (laptop, 8 torch threads) and ≈ 63 s on a 4-core Kaggle CPU notebook

**Exit check:** caches complete for all 187 + `tune`; throughput recorded below.
**Tests required:**
- unit: frame ↔ seconds conversion at 25/30/59.94 fps; cache version mismatch → recompute; atomic write leaves no partial file; cache key is the content fingerprint (renamed file → cache hit)
- unit: routing on fake CLIP scores (clear EPIC, clear FPVSum, uncertain margin)
- integration (`data`): signals on one real video have the expected shapes, and the timestamps span `[0, T]`

**Notes so far (2026-11-11, PK):**
- Decoding uses PyAV (FFmpeg libraries bundled in the wheel): first video stream only,
  threaded, each sample gets the last frame with presentation time <= t on the shared
  `sample_times` grid; a stream that ends early raises `DecodeError`. (OpenCV was
  used first and silently truncated EPIC files; see D1.) No ffmpeg on PATH needed.
- Global motion = |median flow vector| (camera ego-motion); residual = mean |flow − median|
  (hands/objects). Frame 0 copies frame 1's value. Both z-scored per video.
- Routing returns `unknown` when |P(cooking) − thr| < `min_conf` and keeps the mean
  P(cooking) so the uncertain-route rule (ARCHITECTURE.md §4) can still pick a side.
- Videos: `python tools/download_videos.py --root <data root> --epic-scripts <clone of
  epic-kitchens-download-scripts>`. EPIC comes straight from data.bris (URLs and MD5s
  read from the official scripts' CSVs; resumable; MD5-verified). FPVSum comes from
  YouTube via yt-dlp, which now needs a JS runtime (pip `deno`, pinned in
  `requirements-dev.txt`); 360p progressive MP4. **FPVSum YouTube ids come from the
  official `FPVSum_videolist.csv`, not the name:** 4 manifest ids
  (`BikePolo_*_1/_3`) are YouTube videos with other ids.
- Dependencies: `mediapipe` requires `opencv-contrib-python` by name, so we use that
  instead of `opencv-python-headless` (two packages owning `cv2` break each other);
  a Linux image needs `libgl1`. Torch is the CPU build from the PyTorch index.
- Hands: presence = max handedness score of detected hands (0 if none), MediaPipe
  VIDEO mode on the shared time grid.
- CLIP smoke check (2026-11-11, first-pass prompts in `configs/default.yaml` →
  `prompts`): mean P(cooking) ≤ 0.006 on 9 FPVSum videos (routing threshold 0.5);
  zero-shot label top-1 right on 5 of 9 (Bike→motor, BikePolo→bike, Jumping→plane,
  Plane→horse). Phrase tuning is P6 work, on CV folds only.
- **Provisional throughput (2026-11-11, i7-13700H on battery, Balanced plan, so
  throttled; re-measure on AC):** compute seconds per minute of video: flow 2.0,
  hands 10.1, CLIP 45.2 (377 s FPVSum video). CLIP forward ≈ 0.4–0.8 s/image
  (6 threads best; 14 threads slower). At these rates EPIC's 13.2 h of video would
  need ~10 h for CLIP alone vs the ≤ 5 h target (ARCHITECTURE.md §12).
- **EPIC download speed:** data.bris gives ~0.5 MiB/s per connection; 6 parallel
  connections reach ~2.3 MiB/s on our link (≈ 20 h for 184 GB).
- Local dev here runs Python 3.12 (CLAUDE.md says 3.11); `pyproject` targets ≥ 3.11 and
  ruff checks against 3.11 syntax.

**Result (2026-11-14):** exit check met: flow, hands and CLIP cached for all 187 manifest videos + the 20 `tune` videos; SlowFast (released) for all 158 EPIC; audio for the 49 FPVSum.
- Throughput: Kaggle free CPU, one decode pass for all three features, ~60 s per video-minute (158 EPIC videos timed); laptop full pipeline 39.4 s per video-minute (`results/latency.csv`).
- Cache size: flow 6.5 MB, hands 4.5 MB, CLIP 124 MB, SlowFast 894 MB (+ 785 MB shipped in `weights/released_features`), audio 0.7 MB.
- Decode failures: 9 EPIC videos failed the first pass (P04_28, P08_09, P13_03, P18_05, P18_08, P22_02, P26_41, P29_05, P30_08): streams ending up to 0.25 s early (`_END_SLACK_SEC`) and genuine bad packets (≤ 60 skipped, `frames_skipping_bad_packets`); all succeeded on retry. OpenCV truncated GoPro files, so decoding moved to PyAV (cache v2).
- Deviations from ARCHITECTURE.md: none outstanding (PyAV decoding in §3, the fingerprint cache in §8).

---

### P4 — Action detection → `actions[]`
**Owner:** A (prakartii) · **Planned:** Nov 11–13 · **Status:** done (started 2026-11-11, finished 2026-11-14) · **Depends on:** P3 (frames), P2 (mAP scorer)

> **Coordination (A → B):** pairing needed `rewind/types.py` and `rewind/constants.py`
> (P1 items). Both are transcribed verbatim from ARCHITECTURE.md §2 / §3.1 on this branch;
> keep or replace them in P1. Also: kit key *i* = EPIC class id *i* for verbs and
> nouns (`other` last), checked against `EPIC_100_{verb,noun}_classes.csv`; `vocab.py`
> can rely on it.
>
> Note: the vendoring commit (`fd0d477`) landed on the P3 branch before `p4/actionformer`
> was cut; it is P4 work. Rewriting the pushed P3 branch to move it was not worth the risk.

- [x] Vendor ActionFormer (`happyharrycn/actionformer_release`) into `code/third_party/actionformer` with its LICENSE (commit `61ea7eb`, MIT); make sure the boundary tests cover the code paths we call (its dataset loaders read annotation JSONs) — inference imports only `libs.core` and `libs.modeling`, never `libs.datasets` (`tests/contract/test_actionformer_boundary.py`)
- [x] Download the released `epic_kitchens.tar.gz` (check md5) and the verb/noun checkpoints into `weights/`; script it in `tools/fetch_weights.sh` — checkpoints done (SHA-256 pinned); features archive via `tools/fetch_released_features.py` (Box mirror, parallel byte ranges, resumable, MD5-checked; Google Drive throttled to ~25 KiB/s). **MD5 ok (2026-11-11);** 700 videos, `feats` (K, 2304) float32 (+ `v_prob`, `n_prob`)
- [x] **Coverage check:** every EPIC `video_id` in `kit/manifest.csv` and every `tune` video has released features; list any gaps here — **138/138 manifest EPIC videos covered, no gaps** (2026-11-11); `tune` ids pending B's `configs/splits.yaml` (the archive has 700 videos, train included)
- [x] **Leakage check:** confirm from configs/logs that the checkpoints were trained on EPIC train only; record the evidence in §4 Decisions (passed, see Decisions 2026-11-11)
- [x] Run ActionFormer inference on CPU; record seconds per video — **all 158 split videos: 1.32 s per video minute** (verb + noun + pairing, laptop CPU, `tools/predict_epic_actions.py`); `detect/actionformer.py` loads the EMA checkpoint on CPU (strict) and is deterministic; smoke run on random features: 13.6 s for a 267 s video (≈ 3 s per video minute, incl. soft-NMS). Real features (P01_12, 174 s): verb + noun models 4.1 s total (≈ 1.4 s per video minute), 2000 + 2000 detections → 788 paired actions; top: close/open cupboard, open fridge, open drawer. `tests/integration/test_epic_actions.py` passes
- [x] `tools/import_released_features.py` (released features → cache by content fingerprint); `features/slowfast.py` reads the cache, else extracts from the MP4 — importer done: 138/138 imported (fingerprints from Kaggle logs, 0 alignment warnings); extraction from MP4 not yet written; extraction from the MP4 **cut** 2026-11-14 (Decisions): released features ship in `weights/released_features`
- [x] Extraction check on 1–2 videos (Kaggle/Colab if needed): our features match the released ones (cosine ≥ 0.95) — **cut** 2026-11-14 with SlowFast-from-MP4 (Decisions)
- [x] Extract any gap videos on Kaggle/Colab, one at a time — not needed: released features cover 138/138 manifest + 20/20 `tune` videos, 0 alignment warnings
- [x] `tools/build_vn_prior.py` → `weights/vn_prior.npy` (train split only): 67,217 train segments, 3,568 distinct pairs; refuses CSVs containing manifest videos. Source: `epic-kitchens-100-annotations` @ `ea8b404`
- [x] `detect/actionformer.py`, `detect/pairing.py`; cache proposals — `pairing.py` done (§5.1; proposals anchored on the group's top verb, scores `(p_v·p_n·prior^α)^power`); `actionformer.py` wired in #27; detections memoized per video in `tools/predict_cached.Predictor` for tuning/ablation (inference runs once per video, nothing to cache)
- [x] Wire into `pipeline.py` (EPIC route → `actions[]`) — `analyze.summarize`: ActionFormer when SlowFast features are found, else zero-shot (#27)
- [x] Tune `max_per_video` and the minimum score on `tune` — **decided 2026-11-17: kept untuned** (B approved A's proposal; Decisions). Was blocked: `tune` is contaminated for the detector** (EPIC train videos = ActionFormer's training set; mAP 0.527 there vs 0.117 on `report`). Defaults (2000, 0.0) kept untuned; see Decisions 2026-11-12
- [x] **Kill rule (end of Nov 13):** not triggered — ActionFormer mAP 0.117 vs zero-shot 0.0013 on `report`. Fallback kept for the uncertain route. Was: if no working mAP, switch to `detect/fallback_zeroshot.py` — fallback implemented (2 s windows / 1 s stride over 1 fps CLIP, top-k verbs and nouns per window, paired by `pairing.py`); also serves the uncertain route (ARCHITECTURE.md §4)

**Exit check:** mAP rows on `tune` and `report`; mAP > the zero-shot fallback's mAP (or the fallback is shipped and logged).
**Tests required:**
- unit: pairing on hand-made verb/noun proposals (matching tIoU, prior applied, out-of-vocabulary pairs dropped, score in [0, 1], sorted, ≤ 2000)
- unit: score rescaling is monotone and identical across videos
- integration (`data`): one cached EPIC video → ≥ 1 valid action on CPU; mAP floor recorded

**Result (partial, 2026-11-12):** exit check met: mAP measured on both splits and above the zero-shot fallback.

| Date | Split | Tag | mAP (avg tIoU 0.1–0.5) |
|---|---|---|---|
| 2026-11-12 | report (138 EPIC val) | ActionFormer + pairing, default cfg | **0.1169** |
| 2026-11-12 | report | zero-shot CLIP fallback | 0.0013 |
| 2026-11-12 | tune (20 EPIC train) | ActionFormer + pairing, default cfg | 0.5266 (inflated: AF trained on these videos) |

Checkpoint: released `epic_slowfast_{verb,noun}_reproduce/epoch_020` (leakage evidence in Decisions 2026-11-11).
**Final (2026-11-14):** wired into the pipeline (#27); output actions scored without the verb-noun prior (Decisions 2026-11-13). Full pipeline on `report`: **mAP 0.1493** (was 0.1169), zero-shot 0.0015; `tune` 0.6532 (inflated). SlowFast extraction from the MP4 cut (Decisions 2026-11-14); `max_per_video` / `min_score` stay untuned (no clean split).

---

### P5 — Clustering + EPIC coverage selector
**Owner:** B · **Planned:** Nov 12–14 · **Status:** done (started 2026-11-11, finished 2026-11-13) · **Depends on:** P2; starts on GT-derived proposals, switches to P4's proposals when they land

- [x] `cluster.py`
- [x] `select/candidates.py` (both capture rules)
- [x] `select/coverage.py` (expected-coverage lazy greedy, new-seconds cost; no fragment check needed, every component holds a whole window)
- [x] `select/coverage.py` (expected-coverage lazy greedy, new-seconds cost, fragment check)
- [x] `select/hedge.py`
- [x] Label-dedup baseline selector (for the ablation and the kill rule) — `select_epic(mode="dedup")`
- [x] Wire into `pipeline.py` (EPIC route → `highlights[]`) (P7 wiring PR)
- [x] Tune `window_lengths`, `stride`, `h`, `cluster.*` on `tune` (first pass on simulated proposals: mixed lengths kept, `h = 0`; redo on P4 proposals) — redone on real P4 proposals 2026-11-13 (`tools/tune_epic_selector.py`): only `select.stride` 0.5 → 1.0 adopted; temperature 2026-11-16 (S1), no change
- [x] **Kill rule (end of Nov 14):** if expected coverage doesn't beat label dedup, ship label dedup — 2026-11-13, real ActionFormer proposals on `tune` (`tools/ablate.py`, `select.mode`): expected coverage 0.4883 vs label dedup 0.4299 (% of oracle 67.6 vs 59.5) → **keep expected coverage**

**Exit check:** EPIC coverage > uniform on `report`; % of oracle recorded; ablation table below.
**Tests required:**
- unit: gain values on 2–3 hand-made clusters; one-hot distributions reduce to plain distinct counting
- property: gain is monotone and has diminishing returns; selector output always passes R9–R13; total ≤ 0.145·T (except short videos)
- property: on GT-derived clusters, oracle coverage ≥ selector coverage ≥ uniform coverage (allow a small tolerance on the second inequality and log failures)
- unit: hedge windows never overlap selected windows and respect spacing
- performance: selector on a synthetic 2000 s video with 300 clusters < 2 s

**Result (final, 2026-11-16, DA; real ActionFormer proposals):** exit check met.

| Split | Selector | Coverage (any) | % of oracle |
|---|---|---|---|
| tune | expected coverage | 0.4883 | 67.6% |
| tune | label dedup | 0.4299 | 59.5% |
| report | expected coverage (full pipeline) | 0.4368 | 59.8% |
| report | label dedup | 0.4079 | 55.8% |
| report | uniform | 0.3380 | 46.3% |

- Kill rule (decided on `tune` on 2026-11-13, rule set in advance): keep expected
  coverage. `report` agrees (+0.029 coverage, +0.012 weighted).
- Tuned on `tune` only: `select.stride` 0.5 → 1.0; temperature (S1) stays 1.0;
  `hedge.share` 0 (Decisions).
- On `report` the run without the verb-noun prior in the selector scores higher
  (coverage 0.4464 vs 0.4368, `results/ablation_report.md`), while on `tune` the
  prior helps (0.4883 vs 0.4691). Not changed: `report` is not used for choices
  (CLAUDE.md §6.2).
- Deviations from ARCHITECTURE.md: none outstanding (the §5.4 formula fix is in
  Decisions, 2026-11-11).

**Result (partial, 2026-11-11, simulated proposals on `tune`):** `tools/eval_selector.py`;
clean = GT, noisy = 30% missed, ±1 s jitter, top-1 40% / top-5, 1 false positive per
minute (`rewind_eval/sim.py`). Oracle (4.05/5/6 s windows) = 0.740; uniform = 0.358.

| Variant | clean cov | noisy cov | noisy % oracle |
|---|---|---|---|
| expected, h = 1/8 | 0.668 | 0.447 | 60.3% |
| dedup, h = 1/8 | 0.668 | 0.443 | 59.8% |
| expected, h = 0 | 0.744 | 0.499 | 67.4% |
| expected, h = 0, T = 2 | 0.744 | 0.502 | 67.9% |
| expected, h = 0, 4.05 s only | 0.721 | 0.495 | 66.9% |
| dedup, h = 0 | 0.744 | 0.510 | 68.8% |

(Re-run 2026-11-12 after review: farthest-point hedges, short-video selection, simulator
edge fix. Hedges improved from 0.422 to 0.447 but still trail h = 0.)

- Hedging without activity signals costs ~0.05 → `h = 0` for now.
- Mixed window lengths beat 4.05 s only (+0.02 clean, +0.009 noisy).
- Expected vs dedup: within noise on simulated proposals (one seed, 20 videos).
  **Kill-rule decision waits for P4's real proposals** (calibrated top-k scores are
  what expected coverage needs; the simulator's are not calibrated).

---

### P6 — FPVSum route: importance, segments, label
**Owner:** A (prakartii) · **Planned:** Nov 13–15 · **Status:** done (started 2026-11-11, finished 2026-11-14) · **Depends on:** P3, P2 (F1 scorer)

- [x] `select/importance.py` (proxy, shots, windows of 4.05–10 s); unit + Hypothesis property tests (`tests/property/`)
- [x] `label/` (EPIC → cooking; FPVSum CLIP zero-shot with a prompt ensemble); uncertain route per ARCHITECTURE.md §4
- [x] Wire the FPVSum route into `pipeline.py` — importance proxy highlights, zero-shot label, no actions (#27)
- [x] Tune importance weights on FPVSum CV folds (never on all 49 at once) — done, **no gain: kept the current weights**. `tools/tune_fpvsum.py --what importance` (243 configs: each weight in {0, 0.5, 1}, σ in {1, 2, 4} s): held-out CV F1 0.199 for the search vs 0.221 for the current config; the all-folds best (σ = 1 s) is 0.232 in-sample only. Label phrases likewise (`--what label`, 49 phrase sets for the 5 confused classes): held-out accuracy 0.775 vs 0.835 current; kept
- [x] Check signal-router agreement with the manifest (target ≥ 95%); tune `routing.min_conf` — **agreement 99.5% (186/187), 0 misrouted** (2026-11-12, `tools/eval_router.py`, 8 keyframes): EPIC P(cooking) 0.569–1.000 (median 0.980), FPVSum 0.000–0.020; 1 uncertain (EPIC P03_22, 0.569, still gets actions on the uncertain route). `min_conf` not tuned (measured on `report` only)
- [x] Check FPVSum GT timing — `tools/check_fpvsum_gt.py`: **3/49 mistimed** (implied GT fps fits no standard rate: `Horse_2otBDeLWCow` 21.3, `Plane_QsL_gCDHWHA` 39.6, `Surfing_G7lPf1gF-W4` 21.0; the other 46 are 24–30 fps). The scorer now detects this itself (`gt_fpvsum.gt_rate_suspect`, `FpvGT.timing_ok`) and leaves those videos out of F1 only; label accuracy still counts them. No video ids hard-coded
- [x] Uncertain route always emits actions (EPIC detector or zero-shot fallback) — `test_summarize_uncertain_route_always_emits_actions` (#27)
- [x] Ablation: audio terms (RMS, spectral flux) in the importance proxy; keep only if CV improves — **kept** 2026-11-13 (`tools/ablate_audio.py`, 46 timing-ok videos): CV F1 0.2236 → 0.2477 (weights 0–1 grid) / 0.2405 (0–4 grid); chosen w_audio_rms 1.0, w_audio_flux 0.25, better on 27 videos, worse on 17. In the pipeline: fold-mean F1 0.2636 (in-sample), weighted 0.476
- [x] Optional: ridge on CLIP if it beats the proxy on CV — **cut** 2026-11-14 (Decisions: would train on the scored videos' GT)

**Exit check:** F1 > uniform on FPVSum (CV); label accuracy reported; router agreement recorded.
**Tests required:**
- unit: importance on synthetic signals (a spike → selected); segments obey ≥ 4 s and budget
- unit: label picks the argmax prompt with fake CLIP embeddings; score in [0, 1]
- property: FPVSum route output always passes R9–R13

**Result (partial, 2026-11-12, default config, untuned):** exit check met: F1 > uniform on every CV fold; label accuracy reported.

| Split | F1 (ours) | F1 (uniform) | Label acc |
|---|---|---|---|
| report (49 FPVSum) | **0.2219** | 0.1626 | **0.837** (41/49) |
| fold 0 (11) | 0.2064 | 0.2029 | 0.909 |
| fold 1 (10) | 0.2847 | 0.1406 | 0.800 |
| fold 2 (10) | 0.2513 | 0.1730 | 0.800 |
| fold 3 (9) | 0.1737 | 0.1436 | 1.000 |
| fold 4 (9) | 0.1864 | 0.1454 | 0.667 |

**Final (2026-11-14):** audio terms added (Decisions 2026-11-13). Full pipeline on `report` (49 FPVSum, 46 with usable GT timing): **F1 0.2626** vs uniform 0.1645, label **0.8367**; held-out CV F1 0.24–0.25 vs 0.1636. Router 99.5% agreement. CLIP ridge cut (Decisions 2026-11-14).

**Corrected (2026-11-12, mistimed GT left out of F1, 46 videos):** report F1 **0.2255** vs uniform 0.1645; CV folds 0–4: 0.2076 / 0.2847 / 0.2727 / 0.1737 / 0.1792 vs uniform 0.2184 / 0.1406 / 0.1629 / 0.1436 / 0.1523 → **CV mean 0.2236 vs 0.1636; above uniform on 4 of 5 folds** (fold 0 slightly below). The table above used the uncorrected GT.

Label errors (8): jumping -> plane (3), bikepolo -> bike (2), bike -> motor, longboarding -> motor, plane -> horse. `tools/predict_fpvsum.py` writes the documents; scored with `rewind_eval.score`. Not yet: weights/phrases tuned on CV folds (train on 4, test on the held-out one), audio ablation, pipeline wiring, GT timing check.

---

### P7 — Integration, tuning, ablations
**Owner:** A + B · **Planned:** Nov 15–16 · **Status:** in progress (started 2026-11-12 by B) · **Depends on:** P4, P5, P6

- [x] Wire every stage into `run.sh`: `rewind/analyze.py` (route → actions → label → highlights) and `features/extract.py` (signals from the cache or one decoding pass); fake-model tests end to end
- [x] Confirm MediaPipe hands run inside the P8 Docker image (killed at creation on DA's laptop; fine on Kaggle). If it fails there, decide: drop hands (weights 0) or pin another mediapipe — **done 2026-11-15** (`.github/workflows/docker.yml`): first CI run failed `libEGL.so.1: cannot open shared object file`; the image now installs libegl1 + libgles2 and hands run
- [x] SlowFast extraction from the MP4 (P4 open item): until then an EPIC video without cached features gets zero-shot actions, also under `run.sh` on a clean machine — **cut** 2026-11-14 (Decisions): the released features ship by content fingerprint (`weights-v1` release); an unknown EPIC file falls back to zero-shot actions
- [ ] Clean-run all 187 videos from raw MP4 with an empty cache (time it) — **FPVSum part done 2026-11-17 in Docker** (`rewind` image, one fresh container per video, empty cache, `--network none`, laptop CPU, Linux): 48 of 49 videos (Scuba_KD74Ywuy9ng is now a private YouTube video and can't be downloaded), all exit 0, no degraded stage in any log, all pass strict validation at the manifest duration. 138.3 min of video in 69.8 min wall time: **30.3 s per video minute** incl. container start and model load (max 172 s per video). Scored on `report` (`--no-write`, Scuba and EPIC missing count as empty): F1 0.2602, label 0.8163 (40/49), against the full run's 0.2626 / 0.8367 (41/49); both gaps equal one video's share (label: 1/49; F1: 0.117/49), consistent with Scuba alone. EPIC part: only 2 EPIC videos are on this laptop (see P8)
- [x] Final tuning on `tune` only; commit the tuned `configs/default.yaml` — EPIC selector done 2026-11-13 (`tools/tune_epic_selector.py`, real ActionFormer proposals, coordinate search over cluster/select/hedge/pairing/post, adopt rule: mean gain ≥ 0.005 and more wins than losses): only `select.stride` 0.5 → 1.0 adopted (coverage 0.4687 → 0.4883). FPVSum importance CV-tuned (weights unchanged; audio terms added, see P6); prior split (`actions.prior_alpha` 0) and S1 temperature (no change). Tuned config is `configs/default.yaml` on main
- [x] Score on `report`; full ablation table (`tools/ablate.py`) — 2026-11-13, `results/ablation_{tune,report}.md`. **Report, full pipeline: weighted 0.3614** (uniform 0.1763, oracle 0.8178): coverage 0.4368 (59.8% of oracle), mAP 0.1493, F1 0.2626, label 0.8367. Every component helps on both splits; largest: ActionFormer (−0.053 weighted without it on report), expected-coverage selector (−0.012), audio (−0.010)
- [x] `tools/latency.py`: seconds per minute of video on CPU (add a GPU row only if a free Kaggle/Colab run is handy) — 2026-11-12, laptop CPU (Windows), empty cache, full CLI per video incl. model load: **39.4 s per video minute** over 4 videos (EPIC 47.3, FPVSum 26.3; slowest P01_11 50.9). Runs in `results/latency.csv`; rerun on more videos / in Docker at P8
- [x] Set metric floors in `tests/integration/floors.yaml` — 2026-11-13 from the full pipeline on `tune` (`tools/predict_cached.py`): EPIC coverage 0.4687 (uniform 0.3583, oracle 0.7401), 63.3% of oracle (uniform 48.4%), mAP 0.5266 (optimistic: AF train videos); FPVSum CV F1 0.2228 (uniform 0.1636), label 0.835. `tests/integration/test_floors.py` (slow, data) regenerates and checks them, ~4 min
- [x] Determinism check: two runs → byte-identical JSON — 2026-11-12, CLI twice on FPVSum Bike_lhTFVa0h9Rk and EPIC P01_11 (1820 actions, ActionFormer): identical apart from `video_id` (taken from the output name). **Repeated inside Docker 2026-11-17**: `tests/integration/test_docker_image.py`, two fresh containers on the shortest local FPVSum video → byte-identical JSON

**Exit check:** all 187 outputs pass validation; leaderboard has the final `report` row; floors committed.
**Tests required:** integration floors; determinism test on 2 videos (`data`).

**Result:** _fill in: final leaderboard row, ablation table, latency numbers._

---

### P8 — Packaging, demo, deck
**Owner:** B (packaging) + A (demo, numbers) · **Planned:** Nov 16–21 · **Status:** in progress (Dockerfile and README done 2026-11-15) · **Depends on:** P7

- [x] `Dockerfile` (CPU; the pipeline needs no GPU); `tools/fetch_weights.sh` works on a clean machine — Dockerfile **done 2026-11-15** (#41), built and smoke-tested on every relevant PR by `.github/workflows/docker.yml` (deps import, hands run, run.sh fallback writes valid JSON with no large weights). Not built locally: this network's TLS interception (weak-signature CA) breaks pip's download.pytorch.org. `fetch_weights.sh` on a clean machine still to check end to end. **2026-11-17:** every weight (incl. `vn_prior.npy` and the released features as `released_features.tar`) uploaded to the GitHub release `weights-v1`; `fetch_weights.sh` now takes each file from the release first, then its original source, against hashes pinned in `tools/weights.sha256` (contract tests on a fake `file://` release)
- [x] Submission `README.md`: setup, data used, how to reproduce, the tuning-split statement, licenses — **done 2026-11-15**: README rewritten for the organizers (results, how it works, setup incl. Docker, data used incl. the released-features statement, tuning-split statement, reproduce, licenses); `tools/run_all.py` added (was referenced in CLAUDE.md but missing)
- [ ] Fresh clone → Docker build → `run.sh` on one video → identical JSON to ours — **without Docker, done 2026-11-15**: fresh `git clone` + copied weights + empty cache, `run.sh` on EPIC P03_26 (80 s; 85 ActionFormer actions, so the shipped released features were found by fingerprint) and FPVSum Bike_lhTFVa0h9Rk (254 s, audio computed fresh): both JSONs **byte-identical** to the cached runs. Docker leg: image smoke-tested in CI; **full-weights image done locally 2026-11-17** (`docker build -t rewind:full .` with every weight incl. `weights/released_features`, 7.5 GB). `tests/integration/test_docker_image.py` (slow, data; `REWIND_IMAGE` picks the image): in a fresh container with no network, the shortest local EPIC video gets valid JSON, label cooking, ActionFormer actions and no degraded stage (seen failing on the image without released features: "no SlowFast features cached: zero-shot detector"); the shortest FPVSum video gets valid JSON, no actions, a non-cooking label, byte-identical across two containers. Equality to a non-Docker run isn't checked here (MediaPipe crashes natively on this laptop); scores agree (P7)
- [x] `tools/make_reel.py`: demo reel with padding/crossfades (demo only, not outputs) — 2026-11-17, PyAV (no ffmpeg CLI), captions: label, source clock, top actions; P03_26 smoke run OK; writes to `demo/` (ignored)
- [ ] Demo video (**max 3 min**, YouTube or Drive link; Q7) uploaded; link recorded here — **rendered 2026-11-17** by `tools/make_demo.py` (2:27, 1280×720, no audio): title → problem → `run.sh` in/out → kitchen example (input fast-forwarded with highlights on the timeline, 8 highlights with live action captions, the real JSON) → boxing + rock climbing (predicted category) → pipeline → results vs uniform/oracle + the three numbers. Every number read from our outputs/results files. Still to do: upload and record the link
- [x] Deck `VITVellore_Nexus` (`CollegeName_TeamName`; PPT or PDF) on the organizers' template `docs/organizers/Submission_Final_forGithub.pptx`, 9 slides: title (track no., team name, college, members) · the problem · solution · architecture · track final output (refer to the track slides) · differentiation · demo & local setup · brownie slides (extras: S4 viewer, reel) · thank you. Content: problem, architecture diagram, tech stack, innovation, results, limitations; the three numbers (% of oracle, tokens pruned, latency) — numbers ready (`tools/deck_numbers.py`, report): **59.8% of oracle coverage, 85.8% of visual tokens pruned, 39.4 s CPU per video minute** — **built 2026-11-19**: `docs/VITVellore_Nexus.pptx` + `.pdf` on the organizers' template, 9 slides, by `tools/make_deck.py` (numbers from `results/ablation_report.md`, team ID from config, images from the demo video); checked in PowerPoint. Members line: confirm the full names
- [ ] Repo **public** on GitHub (required by the 2026-11-17 mail: code, resources, document, demo, ppt, package and README all in the public repo). It is owned by prakartii, so she switches visibility. Weights and released features don't fit in git (files > 100 MB): see Q12

**Exit check:** reproduction from a fresh clone works; demo video and deck links recorded.
**Tests required:** contract test runs inside the Docker image in CI (or documented manual run).

**Result:** _fill in: links to demo video and deck; reproduction notes._

---

### P9 — Submission
**Owner:** A + B · **Planned:** Nov 22 · **Status:** in progress (packaging tool started 2026-11-16 by B) · **Depends on:** P8

- [ ] Freeze `main`; tag `submission-v1`
- [ ] Regenerate `outputs/` from the tagged commit; validate (ours + official); commit `outputs/`
- [ ] Assemble `TEAM_ID/` exactly as `ARCHITECTURE.md` §13 — tool ready 2026-11-16: `python tools/make_submission.py --outputs outputs/ --dest dist/ --check-video <one EPIC mp4> --check-video <one FPVSum mp4> --zip`. Team ID set 2026-11-17 (`VITVellore_Nexus`), so it runs once `outputs/` is regenerated with it (outputs made before carry `TBD_Q11` and are refused)
- [x] Refresh the README results table (and the deck's numbers, A) from the final outputs under the official coverage rule (≥ 50% inside, Q4) with the two unscored Scuba videos left out (Q14): `rewind_eval.score` does both since 2026-11-20. A: `tools/ablate.py` `METRICS`, `make_deck.py` (six-column table, "Coverage" = `coverage_any`, "all 187 videos") and `make_demo.py` ("Coverage 40%" = `coverage_any`) still show any-overlap coverage next to a weighted total that now uses `coverage_frac` — **done 2026-11-21 (PK)**: README table, ablation tables (`coverage_frac` first), deck and demo generators on the official rule; `pct_oracle` fixed to the official rule too (Decisions 2026-11-21). Report: weighted 0.2789, coverage 0.2555 (43.1% of oracle), mAP 0.1493, F1 0.2256, label 0.8298
- [ ] Put the deck, the demo link and the `TEAM_ID/` package in the public repo. The weights (and anything > 100 MB) are GitHub Release assets on the repo, SHA-256-pinned in `tools/weights.sha256` — **release `weights-v1` uploaded 2026-11-17**, fetched by `tools/fetch_weights.sh` (anonymous downloads work once the repo is public; until then the script falls back to the original sources)
- [ ] A switches the repo to **public** at the freeze (B's call 2026-11-17: private until then)
- [ ] Submit the Google Form https://forms.gle/86xENrtqhx988hBt8 (public repo link); screenshot the confirmation
- [ ] No new features today. Fixes only, and only for validation failures

**Exit check:** confirmation received before 11:59 PM IST, Nov 22 (extended from Nov 18 by the organizers' mail of 2026-11-17).

**Result:** _fill in: submission time, tag, confirmation._

---

### Stretch (only after P7 is done)
- [x] S1 Temperature calibration of cluster distributions (fit on `tune`) — 2026-11-16, `tools/tune_epic_selector.py --only calibration.temperature`: no value clears the adopt rule, T stays 1.0 (Decisions)
- [ ] S2 Learned FPVSum importance (ridge on CLIP)
- [ ] S3 ONNX export + CPU latency benchmark
- [x] S4 HTML timeline viewer for the demo — 2026-11-17, `tools/timeline_html.py` (+ `timeline_template.html`): one static page over output JSONs, plays the MP4 with highlights and action lanes, click to seek, "Highlights only" playback; no libraries or network
- [ ] S5 Downstream QA demo (local VLM: full video vs highlights)

---

## 3. Open questions

| # | Question | Asked (date, how) | Answer (date, source) | Changes made |
|---|---|---|---|---|
| Q4 | How is a GT action "captured" by a highlight (any overlap, tIoU, fraction)? | | Kit §5 says only "distinct GT action pairs captured within the selected highlights". **2026-11-20, organizers' mail (and kit reissue §6, §9): captured if at least 50% of its duration falls inside the union of all highlights; verb and noun don't need to match** | Our `fraction` rule with θ = 0.5 (`eval.capture_theta`) is the official one; ARCHITECTURE §5.3, §7, §9.1. Weighted total now on `coverage_frac`; `tune_epic_selector` optimizes it; selector re-tuned against it: no change (Decisions 2026-11-20) |
| Q6 | Is highlight F1 scored on EPIC? Is mAP scored on FPVSum? | — | 2026-11-11, `kit/TEAM_README.md` §5: mAP + coverage on EPIC only; F1 + label on FPVSum only; FPVSum `actions` may be `[]` | `ARCHITECTURE.md` §4, §7 |
| Q7 | Demo video max 3 or 5 min? | | 2026-11-17, organizers' mail (consolidated slides V3, slide 24 "Registrations & Submissions"): **max 3 minutes**, YouTube or Drive link (slides 23 and 26 still say 5 min; we stay under 3, which satisfies both) | P8 demo item; our cut is 2:27 |
| Q8 | `validate.py` is listed in `TEAM_README.md` but missing from our package: please send it. What is the over-budget penalty curve? | | Partly, kit §4–§5: < 4 s segments are ignored in scoring; ≥ 4 s is pass/fail; the 15% budget is a warning with a soft coverage penalty (so short videos keep one 4 s window). **2026-11-20, organizers' mail: `validate.py` won't be provided; teams write their own and check the edge cases** (the reissue lists the checks in §5, all covered by R1–R14). Penalty curve still unknown | `ARCHITECTURE.md` §5.6, §7 |
| Q9 | May `run.sh` read `manifest.csv`? Scoring only on the 187 videos? | — | 2026-11-11, kit §6: a hidden subset of the 187 is scored; re-runs happen in an isolated environment | Closed by decision: inference doesn't read the manifest |
| Q10 | The example `skiing_03.json` uses label `skiing`, but `supported_classes.json` has `ski`. Which is correct? | | **2026-11-20, organizers' mail: `supported_classes.json` (`ski`); the example was illustration only** (the reissued example says `ski`) | We use `ski` (the vocabulary file); no change |
| Q11 | What is our team ID? The kit examples use `T07`. | | Still open after the 2026-11-17 mail. The slides name teams `CollegeName_TeamName` (registration) but give no ID. 2026-11-17, team decision (DA): use the registered team name, **`VITVellore_Nexus`** | `team.team_id` = `VITVellore_Nexus`; the deck is `VITVellore_Nexus` too |
| Q12 | The 2026-11-17 mail wants the package in the public GitHub repo, but `weights/` (CLIP 571 MB, ActionFormer 2 × ~460 MB, released features) can't be committed (GitHub's 100 MB file limit). Are GitHub Release assets (≤ 2 GB each), fetched by `tools/fetch_weights.sh`, acceptable? | | 2026-11-17, team decision (DA): yes, GitHub Release assets on our repo; still asked in the email in case the organizers want something else | P9 release item |
| Q13 | (Asked by another team) Are there runtime or hardware limits when the organizers re-run `run.sh`? May the public FPVSum annotations be used? | — | 2026-11-20, organizers' mail: **no hardware or runtime limits**; FPVSum annotations: **"advised to treat them as hidden test labels"** | Inference never reads them (CLAUDE.md §6.1). We used them offline for 5-fold CV and adopted only the two audio importance weights from it (Decisions 2026-11-13); every other CV search kept the defaults. **2026-11-21, PK: reverted** (both weights 0, Decisions 2026-11-21); README tuning statement says no setting comes from FPVSum GT |
| Q14 | (Asked by another team) Two manifest FPVSum videos, Scuba_KD74Ywuy9ng and Scuba_Um4kqB8bQBg, are gone from YouTube | — | 2026-11-20, organizers' mail: they are **excluded from final scoring** | We still write all 187 files (kit §5); `report` scoring leaves them out (`report.unscored_fpvsum` in `configs/splits.yaml`), so FPVSum metrics there are over 47 |
| K2 | The track slides (consolidated V3, slide 22) show a flat JSON (`label` a string, actions with `start`/`end` and no score, `importance` on highlights) that differs from `kit/TEAM_README.md` | — | Kit files outrank slides (CLAUDE.md §2): we keep the kit's six-key schema, as decided for the brochure variant | none |
| D1 | **A → B:** switch video metadata + decoding from ffprobe/OpenCV to PyAV (`av==18.0.0`, the newest with Python 3.11 wheels)? OpenCV silently stopped after 44 of 10,418 frames on EPIC GoPro files (audio + 3 data streams); PyAV decodes them fully, 2.5× faster (threaded), by presentation time, and needs no ffmpeg binary. Implemented in `video_io.iter_frames` / `probe_video`; `ffprobe_meta` kept | 2026-11-11, PROGRESS (PK) | 2026-11-12, DA: **approved** (decoding with PyAV; `run.sh` metadata stays on ffprobe) | ARCHITECTURE.md §3.1 `video_io` row updated in #17 |
| K1 | Do the second-hand kit rules (`ARCHITECTURE.md` §7) match our own kit copy? | — | 2026-11-11: yes, all of R1–R13 | `ARCHITECTURE.md` §7 now cites `kit/` |

**Email draft (to m.shankar@samsung.com and prakhar.c@samsung.com, subject "Track 4 — questions on the student package"; Q7 answered 2026-11-17, Q4, Q8's validator part and Q10 answered 2026-11-20, so their lines are dropped):**

> Hello, we are a Track 4 team. Thank you for the answers of 20 November. Three questions remain:
> 1. What is the exact soft penalty when highlights exceed 15% of the duration?
> 2. What team ID should we use in `team_id`? The examples show `T07`; we plan to use our registered team name, `VITVellore_Nexus`.
> 3. The final instructions ask for the package in a public GitHub repo. Our model weights are larger than GitHub's 100 MB file limit. May we host them as GitHub Release assets downloaded by a script in the repo?
>
> Thank you.

---

## 4. Decisions log

Newest first. Record anything that changes `ARCHITECTURE.md` or a kill-rule outcome.

| Date | Decision | Why | By |
|---|---|---|---|
| 2026-11-21 | X1: confidence-weighted expected coverage, `select.confidence_power` 1.0 | On 150 EPIC-100 train videos (`tools/eval_selector_train.py`; tune and report excluded; no CLIP): coverage 0.3563 → 0.4223 (112 better, 22 worse); tune (full pipeline) +0.0104. X2 (median spans) and X3 (budget 0.147–0.149) rejected on tune; X0 not confirmed on the train benchmark (neutral). Details in `results/experiments.md` | PK |
| 2026-11-21 | X0: EPIC-only merge gap `post.merge_gap_epic` = 2.0 (`post.merge_gap` 0.5 kept for FPVSum and the fallback) | tune coverage_frac 0.3302 -> 0.3388 (better 2, worse 1, same 17); neighbouring gaps 1.5/3.0 give 0.3351/0.3372 (plateau). With 0.5 the new code reproduces main exactly (69/69 tune outputs); FPVSum outputs identical. Report not run for selection; outputs/ not regenerated yet. Details in `results/experiments.md` | PK |
| 2026-11-21 | E1: selector re-tuned under the official rule with the real cached pipeline: `cluster.tiou` 0.3 → 0.2, `select.stride` 1.0 → 0.25, `pairing.score_power` 0.5 → 0.25, `select.fraction_theta` 0.5 → 0.55 | `tools/tune_epic_selector.py --passes 2` on `tune`, adopt rule as before: coverage_frac 0.2970 → 0.3302 (details and search path in `results/experiments.md`). `post.merge_gap` 2.0 also passed (+0.0086) but is **left out**: it changes FPVSum highlights, which can't be validated without the hidden labels. Hedge windows lost in every setting, so `hedge.share` stays 0. Report, measured after the decision: weighted 0.2844 → **0.2893**, coverage 0.2692 → 0.2815; FPVSum output and labels unchanged. Floors: coverage_frac and pct_oracle raised; coverage_any lowered 0.4883 → 0.4429 because the selector no longer optimizes that (non-official) rule | PK |
| 2026-11-21 | Selector models the official capture rule: `select.capture_rule` any_overlap → **fraction** (θ 0.5, τ 0.05) | Decided on `tune` with the real cached pipeline (CLIP clustering, released ActionFormer features), `tools/tune_epic_selector.py --only select.capture_rule`, adopt rule as before: coverage_frac 0.2883 → **0.2970**, ADOPT. The 2026-11-20 run that kept any_overlap was an approximation without CLIP (see that row). Report, measured after the decision (same cache, same scorer): coverage_frac 0.2555 → 0.2692 (67 videos better, 35 worse, 36 same), % of oracle 43.1 → 45.4, weighted **0.2789 → 0.2844**; mAP, F1, label, actions and all FPVSum highlights unchanged. Config only, selector code untouched; two tests whose hand-computed windows assumed any_overlap now pin it; new test shows the selector skips a grazing window under fraction. Revert: set the key back to any_overlap (baseline outputs kept in `results/preds/baseline_any_overlap_report`) | PK |
| 2026-11-21 | `pct_oracle` and the coverage oracle on the official rule | #58 moved coverage to the fraction rule but the oracle still picked and scored its windows with any-overlap, and `pct_oracle` divided any-overlap coverage by it, so the deck's "% of oracle" overstated us (59.8%). Now the oracle picks windows under the fraction rule (θ = 0.5) and `pct_oracle` = `coverage_frac` / that oracle: **report 43.1%** (uniform 29.8%), **tune 48.2%** (uniform 29.7%; floor lowered 0.6760 → 0.4820 for this reason). Oracle baseline on report: coverage_frac 0.5925, weighted 0.7620. Rows before 2026-11-21 used the old definition. Touches B's finished P2 scorer; B to confirm | PK |
| 2026-11-21 | Audio importance terms off: `w_audio_rms` 1.0 → 0, `w_audio_flux` 0.25 → 0 (Q13) | The organizers (2026-11-20) advised treating the FPVSum annotations as hidden test labels; these two weights were the only values ever adopted from FPVSum CV, so they go. Cost on `report` (official rule, 47 scored FPVSum): F1 0.2626 → 0.2256 (label 0.8298 on 47), weighted 0.2789; EPIC metrics unchanged. With both weights 0, `summarize` no longer decodes audio. `tests/integration/floors.yaml` FPVSum CV f1 floor lowered 0.2636 → 0.2228 (folds 0.2076/0.2847/0.2704/0.1723/0.1792; label 0.8352 unchanged) for this reason. Every other FPVSum setting is an untuned default (the label and importance CV searches adopted nothing). Decided by PK (P6 owner) on the user's go-ahead; B to confirm | PK |
| 2026-11-20 | The leaderboard's weighted total uses `coverage_frac` (fraction rule, θ = 0.5) instead of `coverage_any`; `tools/ablate.py` takes its weights from `rewind_eval.score`; `tools/tune_epic_selector.py` optimizes `coverage_frac` and can switch `select.capture_rule` | The organizers confirmed the fraction rule as the official one (Q4). Weighted values in rows before this date used `coverage_any` and are not comparable; `coverage_frac` is in every row, so they can be recomputed | DA (B) |
| 2026-11-20 | Selector config unchanged after re-tuning against the official rule; `select.capture_rule` stays `any_overlap` | Coordinate search over cluster, select, hedge, pairing, post and calibration keys plus `select.capture_rule`, `fraction_theta` (0.4–0.7), `fraction_tau` (0.02–0.1) and `budget_frac`, objective `coverage_frac` on the 20 `tune` videos (all of these keys are in the tool's `GRID` since this PR), adopt rule as before (mean gain ≥ 0.005 and more wins than losses): **nothing adopted**. Current config 0.2585; `capture_rule: fraction` 0.2598 (9 better, 5 worse; gain 0.0013); best single moves temperature 1.5 (0.2687) and k 10 (0.2630) also failed the rule. Approximation: no local feature cache, so ActionFormer ran on the released SlowFast features with no CLIP embeddings in clustering (`cluster.sim` inert, left out) and no hedge activity profile (`hedge.share` is 0); the absolute numbers are lower than the cached pipeline's (any-overlap 0.42 vs 0.49) | DA (B) |
| 2026-11-17 | `actions.max_per_video` = 2000 and `actions.min_score` = 0.0 stay untuned (A's proposal of 2026-11-12, now approved) | No clean split to tune them on (see that row) | DA (B) |
| 2026-11-17 | `team.team_id` = our registered team name (`CollegeName_TeamName`) unless the organizers assign an ID (Q11) | The organizers gave no ID; the kit only needs a non-empty string identical in every file | DA (B) |
| 2026-11-17 | Weights and other files > 100 MB are hosted as GitHub Release assets on our repo, pinned by SHA-256 in `tools/fetch_weights.sh` (Q12) | The public repo must hold everything, but GitHub rejects files > 100 MB; release assets allow 2 GB each with no quota, unlike LFS | DA (B) |
| 2026-11-17 | `tools/data/epic_fingerprints.jsonl` stays in the repo, explained in the README | Offline import aid only; not shipped, never read at inference (lookup is by the MP4's fingerprint) | DA (B) |
| 2026-11-17 | The repo stays private until the freeze; A then makes it public | Only A (owner) can change visibility | DA (B) |
| 2026-11-17 | Deadline moves to 22 Nov 2026, 11:59 PM IST; P8 planned Nov 16–21, P9 Nov 22. The rule "no new features on submission day" now applies to Nov 22 | Organizers' mail of 2026-11-17 (dates: submission 22 Nov, top-12 29 Nov, final demo 9 Dec, results 12 Dec) | DA (B), from the organizers |
| 2026-11-16 | Each person reviews and merges their own PRs; the cross-review agent loop (`tools/pr_watch.py`) is removed. Never merge or push to the other person's PR | Requested by B; in practice A already self-merged #24–#43 | DA (B) |
| 2026-11-18 | FPVSum label: keep 8 keyframes, mean of per-frame probabilities | `tools/tune_label.py`, nested 5-fold CV over the FPVSum folds, structural grid only (keyframes 8/16/32/64/all × prob/logprob/embedding; no phrase or prompt edits, those would fit the scored labels): every setting 39–42/49; selection on 4 folds scored on the 5th gives 0.816 held out vs 0.837 for the current setting → no change. Remaining errors are class confusions (jumping→plane, bikepolo→bike, bike→motor), not frame sampling | PK |
| 2026-11-16 | S1: cluster-distribution temperature stays 1.0 | Swept on `tune` with real ActionFormer proposals (adopt rule: mean gain ≥ 0.005 and more wins than losses): T 0.5 → 0.4706, 0.75 → 0.4893, 1.0 → 0.4883, 1.5 → 0.4804, 2.0 → 0.4894, 3.0 → 0.4829. Flat within noise; no value adopted. Caveat: tune detections are over-confident (ActionFormer's training videos), the case where a temperature would matter most, so report-side calibration remains unmeasured by design | PK |
| 2026-11-14 | Cut SlowFast extraction from the MP4 (P4) | Its only use is a clean machine whose EPIC files differ from ours. The organizers re-run on the official EPIC-KITCHENS files; ours are those files (MD5-checked against the official md5.csv), so content fingerprints match and the shipped released features are found. Cost: ~10–15 min CPU per video plus a cosine check. Residual risk: a re-encoded copy falls back to zero-shot actions (report mAP 0.0015 vs 0.1493) | PK |
| 2026-11-14 | Cut the CLIP ridge for FPVSum importance (P6 optional) | FPVSum has no videos outside the scored 49; a 512-d regressor fit on their frame GT trains on the evaluation set, and the CV gain would not carry to the organizers' score honestly. The 6-weight proxy (CV-tuned) stays | PK |
| 2026-11-13 | Output actions scored without the verb-noun prior (`actions.prior_alpha` 0); selector proposals keep it (`pairing.alpha` 1) | Ablation on `tune`: without the prior mAP 0.5266 → 0.6532 but coverage 0.4883 → 0.4691; splitting keeps both (weighted 0.4764 → 0.5080). The prior comes from train counts, so it should help most on these train videos, yet it lowers mAP. Chosen by PK before any report run. Report afterwards (measurement only): mAP 0.1493 (was 0.117 with the prior). Observed, not acted on: on report the selector does slightly better without the prior too (coverage 0.4464 vs 0.4368), the opposite of tune | PK |
| 2026-11-13 | P5 kill rule: keep the expected-coverage selector | Real proposals on `tune`: coverage 0.4883 vs label dedup 0.4299 (rule set in advance; `select.mode` switch added for it) | PK |
| 2026-11-13 | FPVSum importance gets audio terms: `w_audio_rms` 1.0 (z of log loudness), `w_audio_flux` 0.25 (z of spectral flux) | P6 ablation on FPVSum CV (46 videos with usable GT timing, pick on 4 folds, score the 5th): F1 0.2236 → 0.2477 (0.2405 with a wider weight grid); better on 27 videos, worse on 17; folds 0, 3, 4 up, 1, 2 down. Decoded only on the FPVSum route; no audio stream → terms left out. ARCHITECTURE §6.2 updated | PK |
| 2026-11-13 | `select.stride` 0.5 → 1.0 (P7 tuning on `tune`) | Real ActionFormer proposals on tune: coverage 0.4883 vs 0.4687 (better on more videos than worse); B's noisy simulation (3 seeds) 0.5217 vs 0.5112; clean simulation prefers 0.5 (0.7438 vs 0.7305). Report detections are far noisier than tune ones (mAP 0.117 vs 0.527), so the noisy case decides. Every other selector knob: no change cleared the adopt rule (best near-misses cluster.k 10 +0.0044, pairing.alpha 0.5 +0.0015) | PK |
| 2026-11-12 | Keep the current FPVSum importance weights and label phrases (no CV tuning adopted) | 5-fold CV on the 49 videos: grid-tuned settings score lower on held-out folds (F1 0.199 vs 0.221; label 0.775 vs 0.835) than the current ones; with 49 videos the grids overfit | PK |
| 2026-11-12 | **Proposed (needs B):** keep `actions.max_per_video` = 2000 and `actions.min_score` = 0.0 untuned | No clean set to tune them on: `tune` = EPIC train = ActionFormer's training data (mAP 0.527 vs 0.117 on `report`); `report` is off-limits (CLAUDE.md §6.2); all 138 EPIC val videos are in the manifest. Note: more detections can lower mAP (global ranking across videos), so this is a choice, not an optimum | PK |
| 2026-11-11 | `SHORT_VIDEO_T` = `MIN_SEG_EMIT / SCORING_BUDGET` = 27.0 s (was 4.0 / 0.15 = 26.67 s) | We emit 4.05 s windows; between 26.67 and 27.0 s one window is over 15% but wasn't treated as short, so strict validation would fail. No manifest video is in that range | DA (bug fix, no design change) |
| 2026-11-12 | D1 approved: frames decoded with PyAV (`av==18.0.0`), metadata for `run.sh` stays on ffprobe; `signals.dense_fps` 3.0 → 4.0 | OpenCV silently truncated EPIC GoPro files after < 1 s; PyAV decodes them fully and faster. 4 fps gives finer motion/hand peaks and the P3 caches already use it | DA (B), on A's proposal |
| 2026-11-12 | Invalid stage output is salvaged part by part before the full fallback (§4 step 5) | Review of #10: one out-of-vocabulary verb or one bad highlight discarded a video's whole result | DA (review fix) |
| 2026-11-11 | `hedge.share` 0.125 → 0 | Simulated proposals on `tune`: the reserve cost ~0.08 coverage without an activity profile. Re-tune when P3's flow/hands exist | DA (tuning on `tune`) |
| 2026-11-11 | Expected-coverage formula corrected (§5.4): `P_miss` uses each cluster's capture by the merged union, not a factor per window | The v1.0 formula counted overlapping windows over one cluster as independent chances to catch the same action, inflating gains. The fix also scores exactly what post-processing emits | DA (prakartii to confirm) |
| 2026-11-11 | F1 ground truth = FPVSum's own top-15% frames (`Binary_score`), evaluated at native frame times; `eval.f1_fps` removed | Closest to the kit's "top 15% of consensus importance frames"; no resampling error. Exact-count `top_frames(score)` kept as a variant | DA |
| 2026-11-11 | **ActionFormer checkpoints accepted (leakage check passed).** Released `epic_slowfast_{verb,noun}_reproduce` configs: `train_split: ['training']`, `val_split: ['validation']`; logs evaluate on 138 val videos (avg mAP verb 23.51, noun 21.88). Classes 97 verbs / 300 nouns map to the kit's 98 / 301 with `other` | P4 leakage check (ARCHITECTURE.md §10) | PK |
| 2026-11-11 | **ARCHITECTURE.md v1.1.** Inference never reads `manifest.csv` and never derives the route, label or anything else from the filename; routing uses video signals only. Feature cache keyed by content fingerprint, not `video_id`. Supersedes "Keep manifest routing as primary" below | Kit §6: re-runs in an isolated environment; hardcoded video IDs disqualify. FPVSum filenames contain the category (`Ski_…`) | DA + prakartii (confirmed 2026-11-11) |
| 2026-11-11 | Drop the flat brochure serializer (`output.schema`); one schema | Kit confirms the nested six-key schema | DA (kit fact) |
| 2026-11-11 | Remove `REWIND_v2_plan.md` and `REWIND_v2.2_plan.md` (still in git history); mark the superseded parts of the v2.3 plan in its banner | Stale; `ARCHITECTURE.md` is the spec | DA |
| 2026-11-11 | **Architecture locked (ARCHITECTURE.md v1.0).** Design changes now need a Decisions entry agreed by A and B first; `[verify]` resolutions and pre-approved kill rules excepted | Build starts; avoid design churn with 7 days left | A + B |
| 2026-11-11 | Withdraw the (1 − 1/e) guarantee claim (also made in the frozen v2.3 plan §4 and §15); report % of oracle coverage instead | Cost-ratio greedy under a budget, min-length and fragment rules break the conditions | A + B |
| 2026-11-11 | Short videos: `post.short_video_mode` flag; default `min_seg` until P0 tests both against the official validator | R11 and R13 conflict for T < 26.7 s; under known rules only `min_seg` validates | A + B |
| 2026-11-11 | Serializer supports both schemas (`output.schema`); P0 picks | Nested schema is second-hand; brochure shows a flat one | A + B |
| 2026-11-11 | FPVSum: keep `actions: []` when confidently routed; uncertain route always emits actions | Kitchen actions on FPVSum videos would be false positives if mAP is pooled; a misrouted EPIC video must not lose its actions | A + B |
| 2026-11-11 | Keep manifest routing as primary (pending Q9) | Manifest is a kit file, not GT; signal router kept working as a fallback | A + B |
| 2026-11-11 | Rejected: fixed rare-action budget reserve (70/20/10) | Coverage counts distinct pairs per video; expected coverage already avoids repeats | A + B |
| 2026-11-11 | Use ActionFormer's released EPIC-100 SlowFast features + checkpoints; Kaggle/Colab only for gaps | No GPU on the team; removes the only heavy GPU job | A + B |
| 2026-11-10 | Adopt REWIND v2.3: ≥ 4 s distinct-action windows, expected coverage, ActionFormer, no per-frame HOI | Kit rules: < 4 s highlights ignored; ~8 windows per median video | A + B |

---

## 5. Log

Newest first. Format: `YYYY-MM-DD · initials · phase · what changed · PR #`.

- 2026-11-20 · DA · P2/P7 · Official coverage rule in scoring: weighted total on `coverage_frac` (Q4), `report` leaves out the two unscored Scuba videos (Q14, `report.unscored_fpvsum` in splits.yaml, not in shipped code), `tune_epic_selector` optimizes the official rule; selector re-tuned against it, nothing adopted (Decisions). README notes the official-rule numbers (coverage 0.256, weighted 0.289); P9 item to refresh the table. Tests: weighted uses `coverage_frac` (hand-computed case where the rules differ), ablate's weights, unscored exclusion; each seen failing · this PR
- 2026-11-20 · DA · P0 · Organizers' 2026-11-20 answers recorded (Q4 capture = ≥ 50% of the action inside the highlights' union, Q8 no official validator, Q10 `ski`, new Q13 no runtime limits + FPVSum annotations advised as hidden labels, Q14 two Scuba videos unscored); reissued `TEAM_README.md` and `skiing_03.json` converted from .docx into `kit/`; kit section references renumbered; kit example test now expects no warnings (seen failing on the old file); ARCHITECTURE §5.3, §6.3, §7, §9.1 and CLAUDE.md §4 updated · this PR
- 2026-11-21 · PK · P8 · Final presentation: ablation table regenerated under the final config (full pipeline 0.2985 / coverage 0.3045 / 50.0% of oracle); deck (PPTX + PDF) and demo re-rendered with the final numbers; README rounding fixed (full precision 0.298475 / 0.304458); ARCHITECTURE no longer reads as if hedges or audio are active · this PR
- 2026-11-21 · PK · P9 · Final outputs regenerated from main c357288 (X0 + X1): 124 kitchen highlight sets changed, actions/labels/FPVSum identical; both validators pass 187/187; run.sh byte-identical on P02_13 and Bike_lhTFVa0h9Rk. Report measured once on the frozen code: weighted 0.2893 → 0.2985, coverage 0.2815 → 0.3045 (50.0% of oracle) · this PR
- 2026-11-21 · PK · P5 · X0: `post.merge_gap_epic` (2.0) separates the EPIC merge gap from FPVSum; tune coverage 0.3302 → 0.3388; FPVSum unchanged · this PR
- 2026-11-21 · PK · P9 · `outputs/` committed (187 JSONs from the E1 config, stacked on #62/#63; `/outputs/` no longer ignored): validated by our validator and an independent check (0 problems), `run.sh` byte-identical on P02_13 and Bike_lhTFVa0h9Rk; package built with `tools/make_submission.py` (see PR) · this PR
- 2026-11-21 · PK · P8 · Presentation matches the code (stacked on #62): deck no longer claims hedge windows (off) or audio (off) or a submodularity guarantee (doesn't hold under the 50% rule); demo pipeline card drops audio; README/deck/demo numbers from the E1 report run (weighted 0.289, coverage 0.281 = 46.2% of oracle); latency re-measured 2026-11-21 (41.0 s CPU per video minute, empty cache, i7-13700H, 4 videos, `results/latency_2026-11-21.csv`); ablation tables regenerated under the E1 config; README notes that identical JSON needs identical video bytes (YouTube encodes for FPVSum) · this PR
- 2026-11-21 · PK · P5/P7 · E1 selector re-tune on the official rule (4 EPIC-only keys), tune 0.2970 → 0.3302, report weighted 0.2844 → 0.2893; experiment log `results/experiments.md` · this PR
- 2026-11-21 · PK · P5/P7 · `select.capture_rule: fraction` (official ≥ 50% rule in the selector's own objective), adopted on tune 0.2883 → 0.2970; report weighted 0.2789 → 0.2844 (measurement) · this PR
- 2026-11-21 · PK · P2/P8/P9 · Official capture rule everywhere we report: oracle and `pct_oracle` (scorer, tests first), `tools/ablate.py` columns (coverage_frac first, audio variant dropped), `make_deck.py` / `make_demo.py` / `deck_numbers.py` read official coverage (and the deck computes frames pruned and CPU/min instead of hard-coding them); ablation tables, README results, deck regenerated · this PR
- 2026-11-21 · PK · P6/Q13 · Audio importance weights back to 0 (FPVSum GT is hidden test labels); audio not decoded when off; README tuning statement and ARCHITECTURE §6.2 updated; report rescored under the official rule: weighted 0.2789 (F1 0.2256) · this PR
- 2026-11-19 · PK · P8 · Submission deck `docs/VITVellore_Nexus.pptx` / `.pdf` (organizers' template, 9 slides) and its generator `tools/make_deck.py` · this PR
- 2026-11-18 · PK · P7 · Spec sync: ARCHITECTURE §3.3 names the three tuning tools that replaced `tune.py`; P7 SlowFast-from-MP4 item closed as cut · this PR
- 2026-11-18 · PK · P6 · Label refinement checked by nested CV (`tools/tune_label.py`, `fpvsum_label(aggregate=...)` option, default unchanged): no setting beats the current one held out; config unchanged · this PR
- 2026-11-17 · DA · P8/P9 · Weights hosted as GitHub release `weights-v1` (6 assets + SHA256SUMS, 2.3 GB). `tools/fetch_weights.sh` rewritten: release first, original source as fallback, hashes in `tools/weights.sha256`; it now also fetches `vn_prior.npy` and the released features (before, both needed annotations or a 6.5 GB archive). Contract tests run it against a fake `file://` release: every configured weight installed, a corrupt file refused, a re-run downloads nothing, hash list = configured weights (6 breakages caught). README, ARCHITECTURE §10 updated · this PR
- 2026-11-17 · DA · P7/P8 · `tests/integration/test_docker_image.py` (slow, data): the submission image on real videos with full weights (strict validity, no degraded stage, EPIC ActionFormer actions via released features, byte-identical JSON across two fresh containers). Released features imported locally (138/138 EPIC manifest videos, 785 MB); full image built. Clean Docker run of 48/49 FPVSum videos: 30.3 s per video minute, F1 0.2602 / label 0.8163 (Scuba now private on YouTube, scored as empty) · this PR
- 2026-11-17 · DA · P0/P9 · `team.team_id` = `VITVellore_Nexus` (registered team name; Q11 closed by team decision). The make_submission placeholder test now injects its config (it relied on the real one holding `TBD_Q11`); new test pins the shipped ID · this PR
- 2026-11-17 · DA · P4/P9 · Team decisions recorded: untuned `max_per_video` / `min_score` approved; team ID = registered team name (Q11, exact string pending); weights as GitHub Release assets (Q12); fingerprint file kept and explained in the README; repo public at the freeze · this PR
- 2026-11-17 · DA · P0/P8/P9 · Organizer mail recorded: deadline 22 Nov; everything (code, deck, demo, package, README) goes in a public GitHub repo; demo max 3 min (Q7 closed); queries to m.shankar@ / prakhar.c@; re-sent kit identical to `kit/`; deck template committed to `docs/organizers/`; new Q12 (weights > 100 MB in a public repo), K2 (slides' flat JSON vs kit) · this PR
- 2026-11-16 · DA · P9 · Self-review fixes to `make_submission.py`: `--check-video` ran the repo's code when started from `code/` (cwd on sys.path; now a scratch cwd, regression test first); fallback outputs (label `other`) and fallback re-runs are refused; uncommitted shipped paths refused and the commit SHA printed; ID scan is case-insensitive substring and also covers weight text files; weights copied, not hard-linked; timeouts and missing JSON reported; a package failing the re-run is removed; `/dist/` gitignored · #44
- 2026-11-16 · DA · P9/P1/P5 · `tools/make_submission.py` assembles `TEAM_ID/` (§13 + `configs/default.yaml`) and refuses on a placeholder team ID, an output whose `team_id` differs from the config, a missing or invalid output, a missing configured weight, or a manifest video ID in any shipped text file; `--check-video` re-runs the packaged run.sh from an empty cache and requires byte-identical JSON. Tests first (9 deliberate breakages caught). The new ID scan found 10 mentions (README, Dockerfile, comments in `video_io.py` and `gt_fpvsum.py`), all reworded. README setup now builds `weights/vn_prior.npy` (a fresh clone silently used a uniform prior). P1 and P5 closed with Result blocks · —
- 2026-11-16 · DA · process · Review-agent loop removed (supersedes the #7 entry): each person reviews and merges their own PRs and never merges the other's; `tools/pr_watch.py` and its tests deleted; CLAUDE.md §4–§5 updated · #45
- 2026-11-17 · PK · P8 · `tools/make_demo.py`: the full demo video rendered from our outputs and the MP4s (2:27) · this PR
- 2026-11-17 · PK · S4/P8 · `tools/timeline_html.py` demo viewer (static HTML: video + highlight/action timeline, highlights-only playback; videos on another drive linked by file URI); `make_reel.py` merged · #46, this PR
- 2026-11-16 · PK · S1 · Temperature added to the selector tuner's grid (`--only` to sweep one key); swept on tune, no change (T = 1.0) · —
- 2026-11-15 · PK · P8 · Fresh-clone check: empty cache, run.sh on 1 EPIC + 1 FPVSum video → JSON byte-identical to the cached runs (features recomputed from the MP4 match the cache; released SlowFast features found by fingerprint) · —
- 2026-11-15 · PK · P8 · Submission README; `tools/run_all.py` (batch `run_video`, models loaded once, `--skip-existing`; tests first; 2-video run validates); ARCHITECTURE §10/§10.1 updated for the cut MP4 extraction · —
- 2026-11-15 · PK · P8/P7/P1 · Submission `Dockerfile` (python:3.11-slim, CPU) + CI job building and smoke-testing it; CI caught MediaPipe needing libEGL in the image (fixed). All 187 full-pipeline outputs pass strict validation (0 errors) · #41
- 2026-11-14 · PK · P3/P4/P6 · Phases closed: status done, Result blocks (throughput, cache sizes, decode failures; final report mAP 0.1493, F1 0.2626, label 0.8367). Cut SlowFast-from-MP4 and the CLIP ridge (Decisions). Floors test re-run after the mAP raise: passes · —
- 2026-11-13 · PK · P7/P8 · Output actions without the verb-noun prior (`actions.prior_alpha`, tests first; Decisions): tune weighted 0.5080. First full-pipeline **report** score: weighted 0.3614 vs uniform 0.1763 (leaderboard). Ablation tables for tune and report. `tools/deck_numbers.py`: 59.8% of oracle, 85.8% tokens pruned, 39.4 s/min. mAP floor raised · —
- 2026-11-13 · PK · P7 · `tools/ablate.py` (variants through `predict_cached.Predictor`, AF detections memoized; `select.mode` switch; tests first). `tune` table in `results/ablation_tune.md`: full 0.4764; without ActionFormer 0.3057, label dedup 0.4530, no audio 0.4662, no flow 0.4734, no hands 0.4746, no scene change 0.4760; without the verb-noun prior 0.5004 (mAP 0.6532 vs 0.5266, coverage 0.4691 vs 0.4883): open question. P5 kill rule: keep expected coverage · —
- 2026-11-13 · PK · P6 · Audio terms in the FPVSum importance proxy (`features/audio.py`, `tools/ablate_audio.py`, tests first): CV F1 0.2236 → 0.24–0.25, kept (Decisions). `summarize` takes an `audio` loader (FPVSum route only, cached, failure-safe). Tune fold-mean F1 0.2636, weighted 0.476 (was 0.466); EPIC unchanged. Floors raised · —
- 2026-11-13 · PK · P7 · EPIC selector tuned on `tune` with real proposals (`tools/tune_epic_selector.py`, tests first): `select.stride` 1.0 adopted (Decisions). Tune: coverage 0.4883 (was 0.4687), 67.6% of oracle, weighted fold mean 0.466 (was 0.458); FPVSum unchanged. Floors raised; leaderboard rows · —
- 2026-11-13 · PK · P7 · `tools/predict_cached.py`: full pipeline (`analyze.summarize`) for a split from the feature cache alone, 69 tune videos in 168 s. First full tune run: weighted 0.458 vs uniform 0.184 (fold mean); EPIC coverage 0.4687 vs 0.3583, 63.3% of oracle. Floors set in `tests/integration/floors.yaml` with `test_floors.py` (raised floors seen failing). Leaderboard rows for full, uniform, oracle and the 5 folds · —
- 2026-11-13 · PK · P3/P4 · `tune` split fully cached: Kaggle job 20/20 EPIC train videos (flow, hands, CLIP; 0 errors); `import_released_features.py` now also imports logged non-manifest ids (tests first) → 20/20 SlowFast features, 0 alignment warnings. The pipeline can now run on `tune` with ActionFormer (caveat: these are ActionFormer training videos, so its detections here are optimistic: unusable for P4 tuning (Decisions), and selector tuning on them should be checked against `report`-free CV where possible) · —
- 2026-11-12 · PK · P7 · `tools/latency.py` (cold cache, full CLI per video, tests first): 39.4 s per minute of video on a laptop CPU (EPIC 47.3, FPVSum 26.3; 2+2 videos) · —
- 2026-11-12 · PK · P3/P7 · CLI runs without the ffmpeg CLI and on Windows: `ffprobe_meta` falls back to PyAV when the binary is missing; `_write_atomic` uses `os.chmod` by path (POSIX-only `os.fchmod` crashed and wrote no JSON). Tests first; the `run.sh` contract tests skip without ffmpeg, which hid this. Determinism check passed on 2 videos · —
- 2026-11-12 · PK · P4/P6 · Ticked route wiring (EPIC, FPVSum) and uncertain-route actions: done in `analyze.py` with tests in #27 · —
- 2026-11-12 · PK · P6 · FPVSum GT timing check: 3/49 videos' GT covers a different cut (non-standard implied fps); scorer leaves them out of F1 (data-driven, tests first). Corrected F1 0.2255 vs uniform 0.1645; CV mean 0.2236 vs 0.1636, 4/5 folds above uniform · —
- 2026-11-12 · PK · P3 · Kaggle job for the 20 `tune` EPIC train videos launched (`kaggle_launch.py --split tune`; `shard_ids` keeps non-manifest ids, per-run log tag `_tune`) · —
- 2026-11-12 · PK · P4 · Released SlowFast features ship in `weights/released_features` (785 MB, 138/138, `detect.released_features`); `analyze_video` checks the local cache then weights/ via `find_slowfast` (tests first). Clean-cache run on EPIC P01_12: routed EPIC, ActionFormer used (851 actions, top close/open cupboard), 61 s. Rebuild: `RELEASED_FEATURES=1 tools/fetch_weights.sh` with `tools/data/epic_fingerprints.jsonl`. README (P8) must say submitted EPIC outputs used these published features (§10.1.4) · —
- 2026-11-12 · PK · P6 · CV tuning (`tools/tune_fpvsum.py`, tests first): importance weights and label phrases both score lower held-out than the current config, so nothing adopted (Decisions) · —
- 2026-11-12 · PK · P4 · Pairing memory fix (open item from #21): `match_lists` gives each detection its tIoU neighbours from overlapping segments only, identical to the dense matrix (Hypothesis-checked); 6000 + 6000 zero-shot detections over 2000 s now 1.3 s and 3.3 MiB peak (was ~8 s, ~1 GB). Regression test caps pairing at 100 MB · —
- 2026-11-12 · PK · P6 · FPVSum route measured (default cfg): F1 0.222 vs uniform 0.163 on `report`, above uniform on all 5 CV folds; label accuracy 0.837; `tools/predict_fpvsum.py` · —
- 2026-11-12 · PK · P6 · Router agreement with the manifest: 99.5% (186/187), 0 misrouted, 1 uncertain (P03_22); `tools/eval_router.py` · —
- 2026-11-12 · PK · P4 · mAP measured: **0.117 on `report`** (ActionFormer + pairing) vs 0.0013 for the zero-shot fallback; `tune` 0.527 is inflated (AF trained on EPIC train). `tools/predict_epic_actions.py` (both detectors), `Vocab.verb_keys/noun_keys`, `class_phrase`; leaderboard rows added · —
- 2026-11-12 · PK · P4 · Released SlowFast features imported for **138/138** manifest EPIC videos (0 alignment warnings): `import_released_features.py --fingerprint-logs` takes fingerprints and durations from the Kaggle job logs for videos never stored locally · —
- 2026-11-12 · DA · P7 · Review fixes on #21: an unwritable cache (read-only mount, full disk) no longer sends every video to the fallback (cache writes are best-effort); text-cache key keeps `label.classes` order and a `TEXT_VERSION`; ActionFormer load or runtime errors and a wrong-shape VN prior degrade to zero-shot / uniform instead of the fallback; frames always decode at the widest feature width, so flow no longer depends on which features were cached (was up to 3.5 z-units); flow/hands grid mismatch fails loudly; tests for the selector choice per label, the FPVSum label excluding cooking and CLIP subsampling. Open (P4 code): zero-shot pairing on a 2000 s video takes ~8 s and ~1 GB (dense tIoU matrices) · #21
- 2026-11-12 · DA · P7 · Real-model smoke run (CLIP ViT-B/16 + PyAV + flow; hands stubbed) on 45 s and 12 s synthetic videos: valid JSON, routed FPVSum, cached re-run 0.1 s. Fixes it found: (1) **`run_video` now writes the validated fallback before analysis**, so a hard kill (SIGKILL, OOM, native crash) still leaves a valid JSON; regression test first. (2) Prompt text embeddings (~450 prompts, ~30 s CPU per process) are cached in `cache/text/`; a second `run.sh` process skipped it (3.5 s for a 12 s video). (3) Contract tests take the fallback via `REWIND_CONFIG` (new: run.sh/CLI config override), so the default suite never loads real models even when `weights/` exists. **Known local issue:** MediaPipe 1.2.2 Hand Landmarker is SIGKILLed at creation on DA's laptop (Intel Core Ultra 5 125U, kernel 7.2; not memory, not seccomp); it ran on Kaggle for 129 videos. P8 Docker run must confirm hands work on a clean Linux image · #21
- 2026-11-12 · DA · P7 · Pipeline wired end to end: `run.sh` now routes from CLIP P(cooking), emits actions on the EPIC and uncertain routes (ActionFormer when SlowFast features are cached, else zero-shot), labels, and selects highlights (expected coverage for `cooking`, importance proxy otherwise). Signals come from the cache or one decoding pass (`features/extract.py`, shared with `tools/cache_features.py`). New `detect:` config section (ActionFormer files, VN prior). Missing CLIP/hand weights still mean the fallback. Removed a stale duplicate P4 row from the phase table · #21
- 2026-11-12 · PK · P3 · **Feature cache complete: 187/187 manifest videos** (138 EPIC + 49 FPVSum) verified at the current cache versions, each signal reaching the end of its video. Kaggle retry: 9/9 done (incl. both bad-packet videos); 2 FPVSum videos the reaped local run had missed were cached locally · #22
- 2026-11-12 · DA · P3 · Review fixes (B took over on request): end-to-end test that a real file with one undecodable frame (zeroed MJPEG packet, same `InvalidDataError`) decodes with only that sample repeated, so `iter_frames` is covered and not just the helper; `--only-ids` rejects ids not in the kit manifest instead of silently retrying nothing; `iter_frames` docstring states the skip and the 0.25 s end slack; P3 log lines cite #20 · #20
- 2026-11-12 · PK · P3 · Kaggle retry launched for the 9 failed EPIC videos with both decode fixes (`--only-ids`); first attempt failed at startup because the code dataset lacked `requirements.txt` (now packed) · #20
- 2026-11-12 · PK · P3 · Kaggle EPIC cache: **129/138 videos done**; 9 failed: 7 on the end-slack bug (fixed below), 2 (P29_05, P30_08; MD5-verified files) on a packet FFmpeg rejects (`Invalid data found when processing input`). `iter_frames` now decodes packet by packet and skips up to 60 rejected packets (~1 s at 60 fps) before raising `DecodeError`; regression tests with fake packets. The 9 are re-run on Kaggle · #20
- 2026-11-12 · PK · P3 · **Bug fixed** (seen on Kaggle, EPIC P08_09): `iter_frames` raised `DecodeError` when the container duration (with audio) ran 0.03 s past the last video frame. Samples up to 0.25 s past the last frame now get that frame; truncation (seconds to minutes short) still raises. Regression test first · #20
- 2026-11-12 · DA · P6 · Review fixes (B took over on request): Kaggle download crashes always report back (no 12 h hang), Kaggle installs the requirements.txt pins and logs library versions per record, `--collect` replaces stale-version cache entries; scene-change term, `shots`/`other_thr` boundaries and `max_window` (now reachable) tested; property test over 5–2000 s; ARCHITECTURE §6.2 updated; P6 log lines cite #19 · #19
- 2026-11-12 · DA · P4 · Review fixes (B took over on request): pairing uses greedy anchors so a confident verb without a matching noun no longer drops an overlapping verb's action (two regression tests; ARCHITECTURE §5.1); zero-shot windows end exactly at T; P4 log lines now cite #18 · #18
- 2026-11-12 · PK · P3 · CI fixes: PyAV pinned to 18.0.0 (19.x has no Python 3.11 wheels); `video_io` imports OpenCV lazily so `run.sh`'s fallback path loads no OpenCV; `test_run_sh_never_opens_manifest_or_ground_truth` (P1) now ignores files inside the Python installation, which tripped it with library names (`torch/jit/annotations.py`, `cv2/data/`), while still checking every other path · —
- 2026-11-12 · PK · P3 · Merged `main` (P1) into `p3/video-io-signals`: P1 owns `video_io` probing (`ProbeInfo`, `ffprobe_meta`, `parse_probe`), P3 adds frame↔seconds, `sample_times`, PyAV `probe_video`/`iter_frames`; config schema extended (`features.flow/hands/clip` blocks replace `features.flow_width`, new `label.classes` mapping; `signals.dense_fps` = 4.0, see D1 note) · —
- 2026-11-12 · DA · P3 · Review fixes (B took over on request): `iter_frames` samples relative to the stream start time (offset-video test); contract test that `run.sh` still writes valid fallback JSON when cv2 can't load; D1 (PyAV) and `dense_fps` 4.0 approved; P3 log lines now cite #17 · #17
- 2026-11-12 · PK · P3 · CI fixes: PyAV pinned to 18.0.0 (19.x has no Python 3.11 wheels); `video_io` imports OpenCV lazily so `run.sh`'s fallback path loads no OpenCV; `test_run_sh_never_opens_manifest_or_ground_truth` (P1) now ignores files inside the Python installation, which tripped it with library names (`torch/jit/annotations.py`, `cv2/data/`), while still checking every other path · #17
- 2026-11-12 · PK · P3 · Merged `main` (P1) into `p3/video-io-signals`: P1 owns `video_io` probing (`ProbeInfo`, `ffprobe_meta`, `parse_probe`), P3 adds frame↔seconds, `sample_times`, PyAV `probe_video`/`iter_frames`; config schema extended (`features.flow/hands/clip` blocks replace `features.flow_width`, new `label.classes` mapping; `signals.dense_fps` = 4.0, see D1 note) · #17
- 2026-11-12 · DA · P1 · Review fixes on the pipeline: salvage valid parts before the full fallback, output files 0644, infinite durations rejected; tests for each · #10
- 2026-11-12 · DA · P5 · Review fixes on hedge/selector entry: farthest-point hedges (they bunched), exact hedge count, short videos get the best single window, simulator can't invert segments, eval rows carry a config hash and use `eval.capture_theta`; `select_epic` property test; sim rows regenerated · #16
- 2026-11-12 · DA · P1 · Review fixes on post-processing/output: budget on the written (2-decimal) duration, action times clamped into [0, T] and non-finite actions dropped, 4.0 s highlight for 3.5–4 s videos; tests for best-value placement, trim tie-break, merged value · #9
- 2026-11-12 · DA · P2 · Review fixes on scoring: oracle = better of greedy and uniform (counterexample pinned), tune scores FPVSum only per `--fold`, invalid `--pred` files score 0, metric θ moved to `eval.capture_theta`, random baseline seeded by video id; leaderboard rows regenerated · #14
- 2026-11-12 · DA · P5 · Review fixes on the selector: windows next to each pick re-scored (counterexample pinned), coverage looked up only in nearby components (2000 proposals: 11.2 s → 1.4 s), candidate ends clamped to T; tests for filled gaps, confidence, values; perf tests marked `slow`. Correction to the #15 log line: the 0.742 coverage was measured on `report` (a check, no tuning); the tune numbers are in the P5 result block · #15
- 2026-11-12 · DA · P1 · Review fixes on the validator: strict mode compares exactly (no float slack), huge integers and non-UTF-8 files no longer crash it; tests for short videos with two highlights, negative label score, the skiing example · #8
- 2026-11-11 · DA · P1 · `run.sh`, `rewind.cli`, `rewind.pipeline` (validate-or-fallback, atomic write), `video_io.ffprobe_meta`, `utils.seed_all`; contract tests on synthetic 30 s/12 s videos (validity, no `kit/`, filename independence, determinism, audit hook on file opens, failure → fallback, env team ID); ffmpeg in CI · #10
- 2026-11-11 · DA · P1 · Post-processing, fallback and `output.py` (`to_document`, deterministic `dumps`); `SHORT_VIDEO_T` corrected to 4.05 / 0.15 = 27.0 s; unit + Hypothesis property tests (any input → valid highlights; fallback passes strict validation for T in 5–2000 s) · #9
- 2026-11-11 · DA · P1 · Validator: `rewind/schema.py` (R1–R14, strict + kit severities), bundled class list (`rewind/vocab.py`), `rewind_eval.validate` directory CLI (manifest durations, missing files, team_id); 63 new tests, 21 of 22 deliberate bugs caught (1 harmless rewrite) · #8
- 2026-11-11 · DA · process · Review agents post every review on the PR from their person's account (request-changes / comment / approve, with suggested diffs); the author's agent replies on the PR with what it fixed (CLAUDE.md §5) · #11
- 2026-11-12 · DA · P2 · Review fixes on GT/splits: FPVSum CV folds keep clips of one YouTube source together (splits.yaml regenerated), integration tests skip unless both GT files exist and check FPVSum labels, highlight-frame share documented as 15–27%; P6 item to verify 3 FPVSum videos' timing · #13
- 2026-11-12 · DA · P2 · Review fixes on metrics: tests for score ranking in AP, tIoU = threshold, exactly-4 s highlights, short highlights in the budget penalty, empty/perfect coverage and empty F1 · #12
- 2026-11-11 · DA · P5 · Hedge windows, `select_epic` (expected / dedup), proposal simulator, `tools/eval_selector.py` with 14 sim rows + oracle rows using config window lengths; `hedge.share` → 0 · #16
- 2026-11-11 · DA · P5 · Clustering (union-find on tIoU + pair/embedding match; weighted top-k pair dists, temperature), candidate windows + capture model, expected-coverage selector on the merged union (formula corrected); GT-derived clusters on all 138 val videos: coverage 0.742, slowest 0.6 s · #15
- 2026-11-11 · DA · P2 · Baselines (uniform, random), GT oracle (coverage + F1), `rewind_eval.score` CLI and the first 6 leaderboard rows; P2 done: coverage ceiling 0.724, uniform 0.338 (46.7% of oracle) on report · #14
- 2026-11-11 · DA · P2 · Ground truth: `tools/fetch_annotations.sh` (pinned, hashed), EPIC loader (class id → kit key; raw verbs ignored), FPVSum loader (native frames, `Binary_score`, `Jumping__` name quirk), `configs/splits.yaml` via `tools/make_splits.py` (20 duration-matched EPIC train videos; 5 FPVSum folds); F1 takes a GT mask; scipy for eval · #13
- 2026-11-11 · DA · P2 · Metrics as pure functions: coverage (any-overlap / fraction, < 4 s ignored, budget penalty), mAP (highest-tIoU greedy matching, interpolated AP, 5 thresholds), frame F1 (exact top 15%), label accuracy; 31 hand-computed + property tests, 19/19 deliberate bugs caught · #12
- 2026-11-11 · DA · process · Review agents: every session runs a background review sub-agent that polls PRs (`tools/pr_watch.py`), reviews the other person's PRs and reports to the main agent, which posts the review and merges (CLAUDE.md §5); approved by prakartii · #7
- 2026-11-11 · DA · P1 · Skeleton: pyproject, pinned requirements, CI, constants, types, strict config loader + default.yaml, import-boundary tests (source scan, config scan, runtime import check); 69 tests, checked against deliberate bugs; review fixes: NaN/inf and duplicate keys rejected, more range checks, hash of the loaded config · #6
- 2026-11-11 · PK · P3 · EPIC download stalled at 11/138: data.bris intermittently serves a SHA-1-signed CA, rejected by OpenSSL's default security level (`CA signature digest algorithm too weak`). `download_videos.py` now uses a verified TLS context at security level 1 (chain + hostname still checked; files still MD5-checked) and backs off between retries · —
- 2026-11-12 · PK · P4 · Merged P3 (with `main`/P1) into `p4/actionformer`: P1's `types.py`/`constants.py` replace the stand-ins; pairing uses P1's `pairing.top_k`/`eps` (values from P1: alpha 1.0, eps 1.0, max 2000 actions) + new `pairing.score_power` and `zeroshot` config; `tests/contract/test_actionformer_boundary.py` keeps inference off ActionFormer's dataset loaders · —
- 2026-11-11 · PK · P4 · Released features in (MD5 ok); coverage 138/138; imported the 10 downloaded videos; end-to-end action chain runs on real features (data-tier tests pass) · —
- 2026-11-11 · PK · P3 · Home link (~2.5 MiB/s) needs ~20 h for the 184 GB of EPIC video, so the EPIC feature cache now runs on Kaggle: `tools/kaggle_launch.py` uploads the code as a private dataset and starts 5 free CPU notebooks (`tools/kaggle_cache_epic.py`, one shard each: parallel MD5-checked downloads, single-pass flow/hands/CLIP, videos deleted). Results come back with `--collect` (~100 MB, keyed by content fingerprint like a local run) · —
- 2026-11-12 · PK · P6 · Merged P4 (with `main`) into `p6/fpvsum-importance`: importance uses P1's config names and values (`w_global_motion`, `w_scene_change`, `min_window`, `max_window`) + new `importance.len_step`/`stride`; the label uses `label.prompts` + `label.classes`; `tests/strategies.py` merged (P1's + `dense_signal`) · #19
- 2026-11-11 · PK · P6 · `select/importance.py`, `label/`, `importance`/`label`/`select.budget_frac` config, shared Hypothesis strategies · #19
- 2026-11-11 · PK · P6 · Started P6 on `p6/fpvsum-importance` (P4 is blocked on the features download) · #19
- 2026-11-11 · PK · P4 · `tools/fetch_released_features.py`: Google Drive throttled the 6.5 GB archive to ~25 KiB/s; switched to the README's Box mirror with parallel ranges · —
- 2026-11-11 · PK · P4 · `detect/fallback_zeroshot.py` + `zeroshot` config · —
- 2026-11-12 · PK · P4 · Merged P3 (with `main`/P1) into `p4/actionformer`: P1's `types.py`/`constants.py` replace the stand-ins; pairing uses P1's `pairing.top_k`/`eps` (values from P1: alpha 1.0, eps 1.0, max 2000 actions) + new `pairing.score_power` and `zeroshot` config; `tests/contract/test_actionformer_boundary.py` keeps inference off ActionFormer's dataset loaders · #18
- 2026-11-11 · PK · P4 · Released features in (MD5 ok); coverage 138/138; imported the 10 downloaded videos; end-to-end action chain runs on real features (data-tier tests pass) · #18
- 2026-11-11 · PK · P4 · `tools/fetch_released_features.py`: Google Drive throttled the 6.5 GB archive to ~25 KiB/s; switched to the README's Box mirror with parallel ranges · #18
- 2026-11-11 · PK · P4 · `detect/fallback_zeroshot.py` + `zeroshot` config · #18
- 2026-11-11 · PK · P3/P4 · **Bug fixed:** OpenCV decoded < 1 s of EPIC GoPro files without error (regression test `tests/integration/test_decode_epic.py`). Decoding moved to PyAV (open question D1 for B), `iter_frames` raises `DecodeError` on early end, flow/hands/clip cache versions bumped to v2 (old entries ignored), and caching now decodes each video once for all three features: 63 s compute per video minute for flow+hands+CLIP on one EPIC video · —
- 2026-11-11 · PK · P4 · `detect/actionformer.py` (CPU, EMA weights, released feature timing); ActionFormer utils import lazily (no pandas at runtime); EPIC video download paused so the 6.5 GB features archive gets the bandwidth · #18
- 2026-11-11 · PK · P4 · `detect/pairing.py` (+ `types.py`, `constants.py` from the spec), `tools/build_vn_prior.py` (train-only, leakage guard); NMS numpy port so ActionFormer runs without a compiler · #18
- 2026-11-11 · PK · P4 · Checkpoints fetched (`tools/fetch_weights.sh`, SHA-256 pinned); leakage check passed · #18
- 2026-11-11 · PK · P4 · Started P4 on `p4/actionformer`; vendored ActionFormer `61ea7eb` (MIT); fetching released EPIC checkpoints · #18
- 2026-11-11 · PK · P3 · **Bug fixed:** OpenCV decoded < 1 s of EPIC GoPro files without error (regression test `tests/integration/test_decode_epic.py`). Decoding moved to PyAV (open question D1), `iter_frames` raises `DecodeError` on early end, flow/hands/clip cache versions bumped to v2, caching decodes each video once for all three features (54–63 s compute per video minute for flow+hands+CLIP on EPIC). Commits cherry-picked from `p4/actionformer`, where they were first made · —
- 2026-11-11 · PK · P3 · `tools/cache_features.py`; first throughput numbers (provisional, on battery); EPIC download parallelized (6 connections) · —
- 2026-11-11 · PK · P3 · `features/hands.py` (MediaPipe) and `features/clip.py` (open_clip); `tools/fetch_weights.sh`; licenses in ARCHITECTURE §10; data-tier tests for both models; FPVSum 49/49 downloaded, durations within ±0.5 s · —
- 2026-11-11 · PK · P3 · `tools/download_videos.py` (EPIC direct + resumable + MD5; FPVSum via the official id list); downloads of the 187 videos started · —
- 2026-11-11 · PK · P3 · `video_io` (frame ↔ seconds, ffprobe parsing, decoding at a target fps), versioned fingerprint-keyed `features/cache`, Farneback `features/flow` (global vs residual motion), signal-only `routing`; 70 tests, each mutation-checked · —
- 2026-11-11 · PK · P3 · Started P3 on `p3/video-io-signals`; minimal scaffolding note for B · —
- 2026-11-11 · PK · P3 · EPIC download stalled at 11/138: data.bris intermittently serves a SHA-1-signed CA, rejected by OpenSSL's default security level (`CA signature digest algorithm too weak`). `download_videos.py` now uses a verified TLS context at security level 1 (chain + hostname still checked; files still MD5-checked) and backs off between retries · #17
- 2026-11-11 · PK · P3 · **Bug fixed:** OpenCV decoded < 1 s of EPIC GoPro files without error (regression test `tests/integration/test_decode_epic.py`). Decoding moved to PyAV (open question D1), `iter_frames` raises `DecodeError` on early end, flow/hands/clip cache versions bumped to v2, caching decodes each video once for all three features (54–63 s compute per video minute for flow+hands+CLIP on EPIC). Commits cherry-picked from `p4/actionformer`, where they were first made · #17
- 2026-11-11 · PK · P3 · `tools/cache_features.py`; first throughput numbers (provisional, on battery); EPIC download parallelized (6 connections) · #17
- 2026-11-11 · PK · P3 · `features/hands.py` (MediaPipe) and `features/clip.py` (open_clip); `tools/fetch_weights.sh`; licenses in ARCHITECTURE §10; data-tier tests for both models; FPVSum 49/49 downloaded, durations within ±0.5 s · #17
- 2026-11-11 · PK · P3 · `tools/download_videos.py` (EPIC direct + resumable + MD5; FPVSum via the official id list); downloads of the 187 videos started · #17
- 2026-11-11 · PK · P3 · `video_io` (frame ↔ seconds, ffprobe parsing, decoding at a target fps), versioned fingerprint-keyed `features/cache`, Farneback `features/flow` (global vs residual motion), signal-only `routing`; 70 tests, each mutation-checked · #17
- 2026-11-11 · PK · P3 · Started P3 on `p3/video-io-signals`; minimal scaffolding note for B · #17
- 2026-11-11 · DA · P0 · Recorded roles (A = prakartii, B = DA); removed leftover GPU assumptions (P4 test marker, P7 latency, P8 Dockerfile) · #5
- 2026-11-11 · DA · P0 · Teammate (prakartii) confirmed the v1.1 routing decision · #4
- 2026-11-11 · DA · P0 · Committed the official kit to `kit/`; ARCHITECTURE v1.1 (signal-only routing, cache keyed by content fingerprint, one schema, 2 decimals, kit facts); answered Q6, Q9, K1; added Q10, Q11 and the email draft; removed superseded plans · #3
- 2026-11-11 · — · design · Recorded the lock in PROGRESS.md and README (follow-up to #1) · #2
- 2026-11-11 · — · design · Locked ARCHITECTURE.md v1.0 with change-control rules · #1
- 2026-11-11 · — · design · Applied external review: short-video flag, dual schema serializer, uncertain-route actions, routing confidence, no approximation claim, provisional constants, audio ablation; CPU-first feature strategy using ActionFormer's released features · #1
- 2026-11-10 · — · setup · Created CLAUDE.md, ARCHITECTURE.md, PROGRESS.md · —

---

## 6. Leaderboard summary

Copy the key rows from `results/leaderboard.csv` when a phase finishes.

| Date | Phase | Split | Tag | Coverage (any / 50%) | % oracle | mAP | F1 | Label | Weighted |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | P2 | report | uniform baseline | 0.338 / 0.177 | 46.7% | 0 | 0.163 | 0 | 0.176 |
| 2026-11-11 | P2 | report | oracle baseline | 0.724 / 0.330 | 100% | 1 | 0.708 | 1 | 0.817 |
| 2026-11-11 | P2 | tune | uniform baseline | 0.358 / 0.178 | 49.4% | 0 | 0.163 | 0 | 0.184 |
| 2026-11-11 | P2 | tune | oracle baseline | 0.725 / 0.361 | 100% | 1 | 0.708 | 1 | 0.817 |
