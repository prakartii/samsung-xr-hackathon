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
| P1 | Skeleton & submission contract | B | Nov 10–11 | in progress | `run.sh` writes valid fallback JSON for all 187 videos; CI green |
| P2 | Evaluation harness | B | Nov 11–12 | done | Leaderboard has uniform + oracle rows on `tune` and `report` |
| P3 | Video I/O, signals, feature cache | A | Nov 10–12 | in progress | Flow, hands and CLIP cached for all 187 + `tune` videos |
| P4 | Action detection → `actions[]` | A | Nov 11–13 | not started | mAP measured on `tune` and `report`; beats the zero-shot fallback |
| P5 | Clustering + EPIC coverage selector | B | Nov 12–14 | in progress | EPIC coverage > uniform; % of oracle reported; ablation done |
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
**Owner:** B · **Planned:** Nov 10–11 · **Status:** in progress (started 2026-11-11) · **Depends on:** P0 (schema confirmed)

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
- [ ] Run on all 187 videos → `python -m rewind_eval.validate --dir outputs/` passes (blocked on A's downloads; stand-in test: the fallback at all 187 manifest durations passes strict directory validation)

**Exit check:** all 187 dummy outputs pass our validator (and the official one if available); CI green on `main`.
**Tests required:**
- unit: one failing case per rule R1–R13; postprocess hand cases (merge, drop < 4 s, budget trim, short video, rounding keeps ≥ 4 s)
- property: postprocess and fallback outputs always satisfy R9–R13 for T in [5, 2000] s
- contract: `run.sh` on 30 s and 12 s synthetic videos → valid JSON; forced exception → fallback JSON, exit 0
- contract: `run.sh` works with no `kit/` directory present; renaming the input file changes only `video_id`
- contract: a `sys.addaudithook` during the `run.sh` test fails on any open of a manifest, annotation or `data/` GT path (covers `third_party/` and config-driven paths)
- import boundary test (`rewind` never imports `rewind_eval`, never opens `manifest.csv` or annotation paths)

**Result:** _fill in when done._

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
**Owner:** A (prakartii) · **Planned:** Nov 10–12 · **Status:** in progress (started 2026-11-11, branch `p3/video-io-signals`) · **Depends on:** — (P1 types; coordinate with B)

> **Coordination note (resolved 2026-11-12):** P3 started before P1 and carried minimal
> scaffolding; P1 has since merged and this branch was reconciled with `main` (P1 owns
> probing via `ffprobe_meta`; P3 adds PyAV decoding).

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
**Owner:** A · **Planned:** Nov 11–13 · **Status:** not started · **Depends on:** P3 (frames), P2 (mAP scorer)

- [ ] Vendor ActionFormer (`happyharrycn/actionformer_release`) into `code/third_party/actionformer` with its LICENSE; make sure the boundary tests cover the code paths we call (its dataset loaders read annotation JSONs)
- [ ] Download the released `epic_kitchens.tar.gz` (check md5) and the verb/noun checkpoints into `weights/`; script it in `tools/fetch_weights.sh`
- [ ] **Coverage check:** every EPIC `video_id` in `kit/manifest.csv` and every `tune` video has released features; list any gaps here
- [ ] **Leakage check:** confirm from configs/logs that the checkpoints were trained on EPIC train only; record the evidence in §4 Decisions
- [ ] Run ActionFormer inference on CPU; record seconds per video
- [ ] `tools/import_released_features.py` (released features → cache by content fingerprint); `features/slowfast.py` reads the cache, else extracts from the MP4
- [ ] Extraction check on 1–2 videos (Kaggle/Colab if needed): our features match the released ones (cosine ≥ 0.95)
- [ ] Extract any gap videos on Kaggle/Colab, one at a time
- [ ] `tools/build_vn_prior.py` → `weights/vn_prior.npy` (train split only)
- [ ] `detect/actionformer.py`, `detect/pairing.py`; cache proposals
- [ ] Wire into `pipeline.py` (EPIC route → `actions[]`)
- [ ] Tune `max_per_video` and the minimum score on `tune`
- [ ] **Kill rule (end of Nov 13):** if no working mAP, switch to `detect/fallback_zeroshot.py`

**Exit check:** mAP rows on `tune` and `report`; mAP > the zero-shot fallback's mAP (or the fallback is shipped and logged).
**Tests required:**
- unit: pairing on hand-made verb/noun proposals (matching tIoU, prior applied, out-of-vocabulary pairs dropped, score in [0, 1], sorted, ≤ 2000)
- unit: score rescaling is monotone and identical across videos
- integration (`data`): one cached EPIC video → ≥ 1 valid action on CPU; mAP floor recorded

**Result:** _fill in: mAP per tIoU, checkpoint used, leakage evidence._

---

### P5 — Clustering + EPIC coverage selector
**Owner:** B · **Planned:** Nov 12–14 · **Status:** in progress (started 2026-11-11) · **Depends on:** P2; starts on GT-derived proposals, switches to P4's proposals when they land

- [x] `cluster.py`
- [x] `select/candidates.py` (both capture rules)
- [x] `select/coverage.py` (expected-coverage lazy greedy, new-seconds cost; no fragment check needed, every component holds a whole window)
- [ ] `select/hedge.py`
- [ ] Label-dedup baseline selector (for the ablation and the kill rule)
- [x] `select/coverage.py` (expected-coverage lazy greedy, new-seconds cost, fragment check)
- [x] `select/hedge.py`
- [x] Label-dedup baseline selector (for the ablation and the kill rule) — `select_epic(mode="dedup")`
- [ ] Wire into `pipeline.py` (EPIC route → `highlights[]`)
- [ ] Tune `window_lengths`, `stride`, `h`, `cluster.*` on `tune` (first pass on simulated proposals: mixed lengths kept, `h = 0`; redo on P4 proposals)
- [ ] **Kill rule (end of Nov 14):** if expected coverage doesn't beat label dedup, ship label dedup

**Exit check:** EPIC coverage > uniform on `report`; % of oracle recorded; ablation table below.
**Tests required:**
- unit: gain values on 2–3 hand-made clusters; one-hot distributions reduce to plain distinct counting
- property: gain is monotone and has diminishing returns; selector output always passes R9–R13; total ≤ 0.145·T (except short videos)
- property: on GT-derived clusters, oracle coverage ≥ selector coverage ≥ uniform coverage (allow a small tolerance on the second inequality and log failures)
- unit: hedge windows never overlap selected windows and respect spacing
- performance: selector on a synthetic 2000 s video with 300 clusters < 2 s

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
**Owner:** A · **Planned:** Nov 13–15 · **Status:** not started · **Depends on:** P3, P2 (F1 scorer)

- [ ] `select/importance.py` (proxy, shots, windows of 4.05–10 s)
- [ ] `label/` (EPIC → cooking; FPVSum CLIP zero-shot with a prompt ensemble)
- [ ] Wire the FPVSum route into `pipeline.py`
- [ ] Tune importance weights on FPVSum CV folds (never on all 49 at once)
- [ ] Check signal-router agreement with the manifest (target ≥ 95%); tune `routing.min_conf`
- [ ] Check FPVSum GT timing: GT frame count vs ffprobe frame count per downloaded video. `Horse_2otBDeLWCow` (21.3 fps implied), `Plane_QsL_gCDHWHA` (39.6) and `Surfing_G7lPf1gF-W4` (21.0) look like a different cut than the annotated one; exclude or re-align mismatches (`rewind_eval/gt_fpvsum.py` stretches frames linearly over T)
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
| D1 | **A → B:** switch video metadata + decoding from ffprobe/OpenCV to PyAV (`av==18.0.0`, the newest with Python 3.11 wheels)? OpenCV silently stopped after 44 of 10,418 frames on EPIC GoPro files (audio + 3 data streams); PyAV decodes them fully, 2.5× faster (threaded), by presentation time, and needs no ffmpeg binary. Implemented in `video_io.iter_frames` / `probe_video`; `ffprobe_meta` kept | 2026-11-11, PROGRESS (PK) | 2026-11-12, DA: **approved** (decoding with PyAV; `run.sh` metadata stays on ffprobe) | ARCHITECTURE.md §3.1 `video_io` row updated in #17 |
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
| 2026-11-11 | `SHORT_VIDEO_T` = `MIN_SEG_EMIT / SCORING_BUDGET` = 27.0 s (was 4.0 / 0.15 = 26.67 s) | We emit 4.05 s windows; between 26.67 and 27.0 s one window is over 15% but wasn't treated as short, so strict validation would fail. No manifest video is in that range | DA (bug fix, no design change) |
| 2026-11-12 | D1 approved: frames decoded with PyAV (`av==18.0.0`), metadata for `run.sh` stays on ffprobe; `signals.dense_fps` 3.0 → 4.0 | OpenCV silently truncated EPIC GoPro files after < 1 s; PyAV decodes them fully and faster. 4 fps gives finer motion/hand peaks and the P3 caches already use it | DA (B), on A's proposal |
| 2026-11-12 | Invalid stage output is salvaged part by part before the full fallback (§4 step 5) | Review of #10: one out-of-vocabulary verb or one bad highlight discarded a video's whole result | DA (review fix) |
| 2026-11-11 | `hedge.share` 0.125 → 0 | Simulated proposals on `tune`: the reserve cost ~0.08 coverage without an activity profile. Re-tune when P3's flow/hands exist | DA (tuning on `tune`) |
| 2026-11-11 | Expected-coverage formula corrected (§5.4): `P_miss` uses each cluster's capture by the merged union, not a factor per window | The v1.0 formula counted overlapping windows over one cluster as independent chances to catch the same action, inflating gains. The fix also scores exactly what post-processing emits | DA (prakartii to confirm) |
| 2026-11-11 | F1 ground truth = FPVSum's own top-15% frames (`Binary_score`), evaluated at native frame times; `eval.f1_fps` removed | Closest to the kit's "top 15% of consensus importance frames"; no resampling error. Exact-count `top_frames(score)` kept as a variant | DA |
| 2026-11-11 | Expected-coverage formula corrected (§5.4): `P_miss` uses each cluster's capture by the merged union, not a factor per window | The v1.0 formula counted overlapping windows over one cluster as independent chances to catch the same action, inflating gains. The fix also scores exactly what post-processing emits | DA (prakartii to confirm) |
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
- 2026-11-11 · DA · P5 · Clustering (union-find on tIoU + pair/embedding match; weighted top-k pair dists, temperature), candidate windows + capture model, expected-coverage selector on the merged union (formula corrected); GT-derived clusters on all 138 val videos: coverage 0.742, slowest 0.6 s · #15
- 2026-11-11 · DA · process · Review agents: every session runs a background review sub-agent that polls PRs (`tools/pr_watch.py`), reviews the other person's PRs and reports to the main agent, which posts the review and merges (CLAUDE.md §5); approved by prakartii · #7
- 2026-11-11 · DA · P1 · Skeleton: pyproject, pinned requirements, CI, constants, types, strict config loader + default.yaml, import-boundary tests (source scan, config scan, runtime import check); 69 tests, checked against deliberate bugs; review fixes: NaN/inf and duplicate keys rejected, more range checks, hash of the loaded config · #6
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
