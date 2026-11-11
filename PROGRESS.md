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
| P1 | Skeleton & submission contract | B | Nov 10–11 | not started | `run.sh` writes valid fallback JSON for all 187 videos; CI green |
| P2 | Evaluation harness | B | Nov 11–12 | not started | Leaderboard has uniform + oracle rows on `tune` and `report` |
| P3 | Video I/O, signals, feature cache | A | Nov 10–12 | in progress | Flow, hands and CLIP cached for all 187 + `tune` videos |
| P4 | Action detection → `actions[]` | A | Nov 11–13 | in progress | mAP measured on `tune` and `report`; beats the zero-shot fallback |
| P5 | Clustering + EPIC coverage selector | B | Nov 12–14 | not started | EPIC coverage > uniform; % of oracle reported; ablation done |
| P6 | FPVSum route: importance, segments, label | A | Nov 13–15 | not started | F1 > uniform on FPVSum CV; label accuracy reported |
| P7 | Integration, tuning, ablations | A + B | Nov 15–16 | not started | All 187 outputs from raw MP4 pass validation; tuned config committed; floors set |
| P8 | Packaging, demo, deck | B (pkg) + A (demo) | Nov 16–17 | not started | Fresh clone + Docker reproduces one video; demo video + deck done |
| P9 | Submission | A + B | Nov 18 | not started | Submitted via Google Form before 11:59 PM IST |

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
- [ ] Email prism@samsung.com with Q4, Q7, Q8, Q10, Q11 (draft in §3); record the date sent
- [ ] Confirm our assigned team ID and set `team.team_id` in config (Q11)
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
**Owner:** B · **Planned:** Nov 10–11 · **Status:** not started · **Depends on:** P0 (schema confirmed; can start on the second-hand rules)

- [ ] `pyproject.toml` (ruff, pytest with `pythonpath = ["code"]`, markers `slow`, `data`, `gpu`), `requirements*.txt`, `.gitignore`
- [ ] `configs/default.yaml` + `rewind/config.py` with `config_hash()`
- [ ] `rewind/constants.py`, `rewind/types.py`, `rewind/vocab.py` (kit classes)
- [ ] `rewind/schema.py` implementing R1–R13 + `rewind_eval/validate.py` CLI
- [ ] `rewind/postprocess.py` (§5.6 of `ARCHITECTURE.md`)
- [ ] `rewind/fallback.py` (uniform ≥ 4 s windows at 14.5%)
- [ ] `rewind/output.py` (kit schema, 2-decimal rounding); `rewind/pipeline.py` (fallback-only for now), `rewind/cli.py`, `run.sh`
- [ ] `post.short_video_mode` flag (`min_seg` / `within_budget`) in postprocess
- [ ] `video_io.ffprobe_meta` (minimal; A extends it in P3)
- [ ] GitHub Actions: ruff + `pytest` on every PR
- [ ] Run on all 187 videos → `python -m rewind_eval.validate --dir outputs/` passes

**Exit check:** all 187 dummy outputs pass our validator (and the official one if available); CI green on `main`.
**Tests required:**
- unit: one failing case per rule R1–R13; postprocess hand cases (merge, drop < 4 s, budget trim, short video, rounding keeps ≥ 4 s)
- property: postprocess and fallback outputs always satisfy R9–R13 for T in [5, 2000] s
- contract: `run.sh` on 30 s and 12 s synthetic videos → valid JSON; forced exception → fallback JSON, exit 0
- contract: `run.sh` works with no `kit/` directory present; renaming the input file changes only `video_id`
- import boundary test (`rewind` never imports `rewind_eval`, never opens `manifest.csv` or annotation paths)

**Result:** _fill in when done._

---

### P2 — Evaluation harness
**Owner:** B · **Planned:** Nov 11–12 · **Status:** not started · **Depends on:** P1

- [ ] `tools/fetch_annotations.sh` (EPIC-100 train/val CSVs + class CSVs; FPVSum importance), documenting exact sources
- [ ] `rewind_eval/gt_epic.py`, `gt_fpvsum.py` (class index → kit key mapping)
- [ ] `configs/splits.yaml`: `tune` (~20 EPIC train videos matching the val duration spread) and `report`
- [ ] `coverage.py` (both capture rules, < 4 s exclusion, penalty λ), `map.py`, `f1.py`, `label_acc.py`
- [ ] `baselines.py` (uniform, random), `oracle.py`
- [ ] `score.py` CLI → `results/leaderboard.csv` (SHA, tag, config hash, all metrics, split)
- [ ] Leaderboard rows: uniform and oracle on `tune` and `report`

**Exit check:** uniform + oracle rows exist; oracle coverage > uniform on every video.
**Tests required:**
- unit: every metric on hand-computed cases (arithmetic in comments); AP with ties and duplicate detections; each tIoU threshold; F1 at a known overlap
- property: metrics in [0, 1]; perfect prediction → 1.0; empty → 0.0; coverage never decreases when a highlight is added (within budget)
- unit: GT loaders on a 5-row CSV fixture (class keys mapped correctly)

**Result:** _fill in: uniform and oracle numbers; note the oracle's coverage ceiling._

---

### P3 — Video I/O, signals, feature cache
**Owner:** A (prakartii) · **Planned:** Nov 10–12 · **Status:** in progress (started 2026-11-11, branch `p3/video-io-signals`) · **Depends on:** — (P1 types; coordinate with B)

> **Coordination note (A → B):** P1 hasn't started, but P3 needs a test harness. This
> branch adds the minimum P1 scaffolding P3 needs (`pyproject.toml`, `.gitignore`,
> `requirements-dev.txt`, package `__init__.py` files, and only the `signals`,
> `features` and `routing` sections of `configs/default.yaml`), each in its own commit.
> B owns these files: keep or replace them in P1; P3 code depends only on the
> layout in `CLAUDE.md` §3.

- [ ] Download the 187 videos into `data/videos/` + the `tune` videos — in progress (2026-11-11) with `tools/download_videos.py`; `tune` videos wait for B's `configs/splits.yaml`
- [x] `video_io.py`: decoding at a target fps, frame ↔ seconds (also ffprobe parsing, so P1's `ffprobe_meta` item is covered)
- [x] `features/flow.py` (residual + global motion)
- [x] `features/hands.py` (MediaPipe Hand Landmarker), `features/clip.py` (open_clip ViT-B/16, prompt ensemble, zero-shot); weights via `tools/fetch_weights.sh` (SHA-256 pinned)
- [x] `features/cache.py` (versioned, atomic)
- [x] `tools/cache_features.py` (longest first, resumable; logs `cache/throughput.csv`)
- [x] `routing.py` (video signals only; no manifest, no filename): pure decision from per-keyframe CLIP P(cooking); the CLIP prompts that produce it come with `features/clip.py`
- [ ] Start caching on Nov 10; record throughput (seconds of compute per minute of video)

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

**Result:** _fill in: throughput, total cache size, any videos that failed to decode._

---

### P4 — Action detection → `actions[]`
**Owner:** A (prakartii) · **Planned:** Nov 11–13 · **Status:** in progress (started 2026-11-11, branch `p4/actionformer`, stacked on `p3/video-io-signals`) · **Depends on:** P3 (frames), P2 (mAP scorer)

> **Coordination (A → B):** pairing needed `rewind/types.py` and `rewind/constants.py`
> (P1 items). Both are transcribed verbatim from ARCHITECTURE.md §2 / §3.1 on this branch;
> keep or replace them in P1. Also: kit key *i* = EPIC class id *i* for verbs and
> nouns (`other` last), checked against `EPIC_100_{verb,noun}_classes.csv`; `vocab.py`
> can rely on it.
>
> Note: the vendoring commit (`fd0d477`) landed on the P3 branch before `p4/actionformer`
> was cut; it is P4 work. Rewriting the pushed P3 branch to move it was not worth the risk.

- [x] Vendor ActionFormer (`happyharrycn/actionformer_release`) into `code/third_party/actionformer` with its LICENSE (commit `61ea7eb`, MIT)
- [x] Download the released `epic_kitchens.tar.gz` (check md5) and the verb/noun checkpoints into `weights/`; script it in `tools/fetch_weights.sh` — checkpoints done (SHA-256 pinned); features archive via `tools/fetch_released_features.py` (Box mirror, parallel byte ranges, resumable, MD5-checked; Google Drive throttled to ~25 KiB/s). **MD5 ok (2026-11-11);** 700 videos, `feats` (K, 2304) float32 (+ `v_prob`, `n_prob`)
- [x] **Coverage check:** every EPIC `video_id` in `kit/manifest.csv` and every `tune` video has released features; list any gaps here — **138/138 manifest EPIC videos covered, no gaps** (2026-11-11); `tune` ids pending B's `configs/splits.yaml` (the archive has 700 videos, train included)
- [x] **Leakage check:** confirm from configs/logs that the checkpoints were trained on EPIC train only; record the evidence in §4 Decisions (passed, see Decisions 2026-11-11)
- [ ] Run ActionFormer inference on CPU; record seconds per video — `detect/actionformer.py` loads the EMA checkpoint on CPU (strict) and is deterministic; smoke run on random features: 13.6 s for a 267 s video (≈ 3 s per video minute, incl. soft-NMS). Real features (P01_12, 174 s): verb + noun models 4.1 s total (≈ 1.4 s per video minute), 2000 + 2000 detections → 788 paired actions; top: close/open cupboard, open fridge, open drawer. `tests/integration/test_epic_actions.py` passes
- [ ] `tools/import_released_features.py` (released features → cache by content fingerprint); `features/slowfast.py` reads the cache, else extracts from the MP4 — importer done (10 downloaded videos imported, 0 alignment warnings; rest import as videos land); extraction from MP4 not yet written
- [ ] Extraction check on 1–2 videos (Kaggle/Colab if needed): our features match the released ones (cosine ≥ 0.95)
- [ ] Extract any gap videos on Kaggle/Colab, one at a time
- [x] `tools/build_vn_prior.py` → `weights/vn_prior.npy` (train split only): 67,217 train segments, 3,568 distinct pairs; refuses CSVs containing manifest videos. Source: `epic-kitchens-100-annotations` @ `ea8b404`
- [ ] `detect/actionformer.py`, `detect/pairing.py`; cache proposals — `pairing.py` done (§5.1; proposals anchored on the group's top verb, scores `(p_v·p_n·prior^α)^power`)
- [ ] Wire into `pipeline.py` (EPIC route → `actions[]`)
- [ ] Tune `max_per_video` and the minimum score on `tune`
- [ ] **Kill rule (end of Nov 13):** if no working mAP, switch to `detect/fallback_zeroshot.py` — fallback implemented (2 s windows / 1 s stride over 1 fps CLIP, top-k verbs and nouns per window, paired by `pairing.py`); also serves the uncertain route (ARCHITECTURE.md §4)

**Exit check:** mAP rows on `tune` and `report`; mAP > the zero-shot fallback's mAP (or the fallback is shipped and logged).
**Tests required:**
- unit: pairing on hand-made verb/noun proposals (matching tIoU, prior applied, out-of-vocabulary pairs dropped, score in [0, 1], sorted, ≤ 2000)
- unit: score rescaling is monotone and identical across videos
- integration (`data`): one cached EPIC video → ≥ 1 valid action on CPU; mAP floor recorded

**Result:** _fill in: mAP per tIoU, checkpoint used, leakage evidence._

---

### P5 — Clustering + EPIC coverage selector
**Owner:** B · **Planned:** Nov 12–14 · **Status:** not started · **Depends on:** P2; starts on GT-derived proposals, switches to P4's proposals when they land

- [ ] `cluster.py`
- [ ] `select/candidates.py` (both capture rules)
- [ ] `select/coverage.py` (expected-coverage lazy greedy, new-seconds cost, fragment check)
- [ ] `select/hedge.py`
- [ ] Label-dedup baseline selector (for the ablation and the kill rule)
- [ ] Wire into `pipeline.py` (EPIC route → `highlights[]`)
- [ ] Tune `window_lengths`, `stride`, `h`, `cluster.*` on `tune`
- [ ] **Kill rule (end of Nov 14):** if expected coverage doesn't beat label dedup, ship label dedup

**Exit check:** EPIC coverage > uniform on `report`; % of oracle recorded; ablation table below.
**Tests required:**
- unit: gain values on 2–3 hand-made clusters; one-hot distributions reduce to plain distinct counting
- property: gain is monotone and has diminishing returns; selector output always passes R9–R13; total ≤ 0.145·T (except short videos)
- property: on GT-derived clusters, oracle coverage ≥ selector coverage ≥ uniform coverage (allow a small tolerance on the second inequality and log failures)
- unit: hedge windows never overlap selected windows and respect spacing
- performance: selector on a synthetic 2000 s video with 300 clusters < 2 s

**Result:** _fill in: coverage (both capture rules), % of oracle, ablation table (label dedup vs expected; 4 s only vs mixed lengths; hedge on/off)._

---

### P6 — FPVSum route: importance, segments, label
**Owner:** A · **Planned:** Nov 13–15 · **Status:** not started · **Depends on:** P3, P2 (F1 scorer)

- [ ] `select/importance.py` (proxy, shots, windows of 4.05–10 s)
- [ ] `label/` (EPIC → cooking; FPVSum CLIP zero-shot with a prompt ensemble)
- [ ] Wire the FPVSum route into `pipeline.py`
- [ ] Tune importance weights on FPVSum CV folds (never on all 49 at once)
- [ ] Check signal-router agreement with the manifest (target ≥ 95%); tune `routing.min_conf`
- [ ] Uncertain route always emits actions (EPIC detector or zero-shot fallback)
- [ ] Ablation: audio terms (RMS, spectral flux) in the importance proxy; keep only if CV improves
- [ ] Optional: ridge on CLIP if it beats the proxy on CV

**Exit check:** F1 > uniform on FPVSum (CV); label accuracy reported; router agreement recorded.
**Tests required:**
- unit: importance on synthetic signals (a spike → selected); segments obey ≥ 4 s and budget
- unit: label picks the argmax prompt with fake CLIP embeddings; score in [0, 1]
- property: FPVSum route output always passes R9–R13

**Result:** _fill in: F1 per fold, label accuracy, confusion of the worst categories, router agreement._

---

### P7 — Integration, tuning, ablations
**Owner:** A + B · **Planned:** Nov 15–16 · **Status:** not started · **Depends on:** P4, P5, P6

- [ ] Clean-run all 187 videos from raw MP4 with an empty cache (time it)
- [ ] Final tuning on `tune` only; commit the tuned `configs/default.yaml`
- [ ] Score on `report`; full ablation table (`tools/ablate.py`)
- [ ] `tools/latency.py`: seconds per minute of video on CPU (add a GPU row only if a free Kaggle/Colab run is handy)
- [ ] Set metric floors in `tests/integration/floors.yaml`
- [ ] Determinism check: two runs → byte-identical JSON

**Exit check:** all 187 outputs pass validation; leaderboard has the final `report` row; floors committed.
**Tests required:** integration floors; determinism test on 2 videos (`data`).

**Result:** _fill in: final leaderboard row, ablation table, latency numbers._

---

### P8 — Packaging, demo, deck
**Owner:** B (packaging) + A (demo, numbers) · **Planned:** Nov 16–17 · **Status:** not started · **Depends on:** P7

- [ ] `Dockerfile` (CPU; the pipeline needs no GPU); `tools/fetch_weights.sh` works on a clean machine
- [ ] Submission `README.md`: setup, data used, how to reproduce, the tuning-split statement, licenses
- [ ] Fresh clone → Docker build → `run.sh` on one video → identical JSON to ours
- [ ] `tools/make_reel.py`: demo reel with padding/crossfades (demo only, not outputs)
- [ ] Demo video (length per Q7; default ≤ 3 min) uploaded; link recorded here
- [ ] Deck `CollegeName_TeamName`: problem, architecture diagram, tech stack, innovation, results, limitations; the three numbers (% of oracle, tokens pruned, latency)
- [ ] Repo shared with the organizers (public, or invite as required)

**Exit check:** reproduction from a fresh clone works; demo video and deck links recorded.
**Tests required:** contract test runs inside the Docker image in CI (or documented manual run).

**Result:** _fill in: links to demo video and deck; reproduction notes._

---

### P9 — Submission
**Owner:** A + B · **Planned:** Nov 18 · **Status:** not started · **Depends on:** P8

- [ ] Freeze `main`; tag `submission-v1`
- [ ] Regenerate `outputs/` from the tagged commit; validate (ours + official); commit `outputs/`
- [ ] Assemble `TEAM_ID/` exactly as `ARCHITECTURE.md` §13
- [ ] Submit the Google Form (repo link, demo video, deck); screenshot the confirmation
- [ ] No new features today. Fixes only, and only for validation failures

**Exit check:** confirmation received before 11:59 PM IST, Nov 18.

**Result:** _fill in: submission time, tag, confirmation._

---

### Stretch (only after P7 is done)
- [ ] S1 Temperature calibration of cluster distributions (fit on `tune`)
- [ ] S2 Learned FPVSum importance (ridge on CLIP)
- [ ] S3 ONNX export + CPU latency benchmark
- [ ] S4 HTML timeline viewer for the demo
- [ ] S5 Downstream QA demo (local VLM: full video vs highlights)

---

## 3. Open questions

| # | Question | Asked (date, how) | Answer (date, source) | Changes made |
|---|---|---|---|---|
| Q4 | How is a GT action "captured" by a highlight (any overlap, tIoU, fraction)? | | Kit §5 says only "distinct GT action pairs captured within the selected highlights" | Still open; select with any-overlap, report both |
| Q6 | Is highlight F1 scored on EPIC? Is mAP scored on FPVSum? | — | 2026-11-11, `kit/TEAM_README.md` §5: mAP + coverage on EPIC only; F1 + label on FPVSum only; FPVSum `actions` may be `[]` | `ARCHITECTURE.md` §4, §7 |
| Q7 | Demo video max 3 or 5 min? | | | |
| Q8 | `validate.py` is listed in `TEAM_README.md` but missing from our package: please send it. What is the over-budget penalty curve? | | Partly, kit §4–§5: < 4 s segments are ignored in scoring; ≥ 4 s is pass/fail; the 15% budget is a warning with a soft coverage penalty (so short videos keep one 4 s window) | `ARCHITECTURE.md` §5.6, §7 |
| Q9 | May `run.sh` read `manifest.csv`? Scoring only on the 187 videos? | — | 2026-11-11, kit §6: a hidden subset of the 187 is scored; re-runs happen in an isolated environment | Closed by decision: inference doesn't read the manifest |
| Q10 | The example `skiing_03.json` uses label `skiing`, but `supported_classes.json` has `ski`. Which is correct? | | | We use `ski` (the vocabulary file) |
| Q11 | What is our team ID? The kit examples use `T07`. | | | |
| D1 | **A → B:** switch video metadata + decoding from ffprobe/OpenCV to PyAV (`av==19.0.2`)? OpenCV silently stopped after 44 of 10,418 frames on EPIC GoPro files (audio + 3 data streams); PyAV decodes them fully, 2.5× faster (threaded), by presentation time, and needs no ffmpeg binary. Implemented in `video_io.iter_frames` / `probe_video`; `ffprobe_meta` kept | 2026-11-11, PROGRESS (PK) | | If B agrees: ARCHITECTURE.md §3.1 `video_io` row and §4 ("ffprobe") updated in the merge PR |
| K1 | Do the second-hand kit rules (`ARCHITECTURE.md` §7) match our own kit copy? | — | 2026-11-11: yes, all of R1–R13 | `ARCHITECTURE.md` §7 now cites `kit/` |

**Email draft (to prism@samsung.com, subject "Track 4 — questions on the student package"):**

> Hello, we are a Track 4 team. Thank you for the student package. A few questions:
> 1. `TEAM_README.md` lists `validate.py`, but it wasn't in the zip we received. Could you send it?
> 2. Action coverage: when does a highlight "capture" a ground-truth action: any overlap, a minimum tIoU, or a fraction of the action inside the highlight?
> 3. What is the exact soft penalty when highlights exceed 15% of the duration?
> 4. `skiing_03.json` uses the label `skiing`, but `supported_classes.json` has `ski`. Should we use `ski`?
> 5. What team ID should we use in `team_id`? The examples show `T07`.
> 6. Is the demo video limit 3 or 5 minutes? The brochure mentions both.
>
> Thank you.

---

## 4. Decisions log

Newest first. Record anything that changes `ARCHITECTURE.md` or a kill-rule outcome.

| Date | Decision | Why | By |
|---|---|---|---|
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

- 2026-11-11 · PK · P4 · Released features in (MD5 ok); coverage 138/138; imported the 10 downloaded videos; end-to-end action chain runs on real features (data-tier tests pass) · —
- 2026-11-11 · PK · P4 · `tools/fetch_released_features.py`: Google Drive throttled the 6.5 GB archive to ~25 KiB/s; switched to the README's Box mirror with parallel ranges · —
- 2026-11-11 · PK · P4 · `detect/fallback_zeroshot.py` + `zeroshot` config · —
- 2026-11-11 · PK · P3/P4 · **Bug fixed:** OpenCV decoded < 1 s of EPIC GoPro files without error (regression test `tests/integration/test_decode_epic.py`). Decoding moved to PyAV (open question D1 for B), `iter_frames` raises `DecodeError` on early end, flow/hands/clip cache versions bumped to v2 (old entries ignored), and caching now decodes each video once for all three features: 63 s compute per video minute for flow+hands+CLIP on one EPIC video · —
- 2026-11-11 · PK · P4 · `detect/actionformer.py` (CPU, EMA weights, released feature timing); ActionFormer utils import lazily (no pandas at runtime); EPIC video download paused so the 6.5 GB features archive gets the bandwidth · —
- 2026-11-11 · PK · P4 · `detect/pairing.py` (+ `types.py`, `constants.py` from the spec), `tools/build_vn_prior.py` (train-only, leakage guard); NMS numpy port so ActionFormer runs without a compiler · —
- 2026-11-11 · PK · P4 · Checkpoints fetched (`tools/fetch_weights.sh`, SHA-256 pinned); leakage check passed · —
- 2026-11-11 · PK · P4 · Started P4 on `p4/actionformer`; vendored ActionFormer `61ea7eb` (MIT); fetching released EPIC checkpoints · —
- 2026-11-11 · PK · P3 · **Bug fixed:** OpenCV decoded < 1 s of EPIC GoPro files without error (regression test `tests/integration/test_decode_epic.py`). Decoding moved to PyAV (open question D1), `iter_frames` raises `DecodeError` on early end, flow/hands/clip cache versions bumped to v2, caching decodes each video once for all three features (54–63 s compute per video minute for flow+hands+CLIP on EPIC). Commits cherry-picked from `p4/actionformer`, where they were first made · —
- 2026-11-11 · PK · P3 · `tools/cache_features.py`; first throughput numbers (provisional, on battery); EPIC download parallelized (6 connections) · —
- 2026-11-11 · PK · P3 · `features/hands.py` (MediaPipe) and `features/clip.py` (open_clip); `tools/fetch_weights.sh`; licenses in ARCHITECTURE §10; data-tier tests for both models; FPVSum 49/49 downloaded, durations within ±0.5 s · —
- 2026-11-11 · PK · P3 · `tools/download_videos.py` (EPIC direct + resumable + MD5; FPVSum via the official id list); downloads of the 187 videos started · —
- 2026-11-11 · PK · P3 · `video_io` (frame ↔ seconds, ffprobe parsing, decoding at a target fps), versioned fingerprint-keyed `features/cache`, Farneback `features/flow` (global vs residual motion), signal-only `routing`; 70 tests, each mutation-checked · —
- 2026-11-11 · PK · P3 · Started P3 on `p3/video-io-signals`; minimal scaffolding note for B · —
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
| | | | | | | | | | |
