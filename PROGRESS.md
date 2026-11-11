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
| P2 | Evaluation harness | B | Nov 11–12 | in progress | Leaderboard has uniform + oracle rows on `tune` and `report` |
| P3 | Video I/O, signals, feature cache | A | Nov 10–12 | not started | Flow, hands and CLIP cached for all 187 + `tune` videos |
| P4 | Action detection → `actions[]` | A | Nov 11–13 | not started | mAP measured on `tune` and `report`; beats the zero-shot fallback |
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
**Owner:** B · **Planned:** Nov 11–12 · **Status:** in progress (started 2026-11-11) · **Depends on:** P1

- [ ] `tools/fetch_annotations.sh` (EPIC-100 train/val CSVs + class CSVs; FPVSum importance), documenting exact sources
- [ ] `rewind_eval/gt_epic.py`, `gt_fpvsum.py` (class index → kit key mapping)
- [ ] `configs/splits.yaml`: `tune` (~20 EPIC train videos matching the val duration spread) and `report`
- [x] `coverage.py` (both capture rules, < 4 s exclusion, penalty λ), `map.py`, `f1.py`, `label_acc.py`
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
**Owner:** A · **Planned:** Nov 10–12 · **Status:** not started · **Depends on:** — (P1 types; coordinate with B)

- [ ] Download the 187 videos into `data/videos/` + the `tune` videos
- [ ] `video_io.py`: decoding at a target fps, frame ↔ seconds
- [ ] `features/flow.py` (residual + global motion), `features/hands.py`, `features/clip.py`
- [ ] `features/cache.py` (versioned, atomic) + `tools/cache_features.py` (longest first, resumable)
- [ ] `routing.py` (video signals only; no manifest, no filename)
- [ ] Start caching on Nov 10; record throughput (seconds of compute per minute of video)

**Exit check:** caches complete for all 187 + `tune`; throughput recorded below.
**Tests required:**
- unit: frame ↔ seconds conversion at 25/30/59.94 fps; cache version mismatch → recompute; atomic write leaves no partial file; cache key is the content fingerprint (renamed file → cache hit)
- unit: routing on fake CLIP scores (clear EPIC, clear FPVSum, uncertain margin)
- integration (`data`): signals on one real video have the expected shapes, and the timestamps span `[0, T]`

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
| 2026-11-12 | Invalid stage output is salvaged part by part before the full fallback (§4 step 5) | Review of #10: one out-of-vocabulary verb or one bad highlight discarded a video's whole result | DA (review fix) |
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

- 2026-11-12 · DA · P1 · Review fixes on the pipeline: salvage valid parts before the full fallback, output files 0644, infinite durations rejected; tests for each · #10
- 2026-11-12 · DA · P1 · Review fixes on post-processing/output: budget on the written (2-decimal) duration, action times clamped into [0, T] and non-finite actions dropped, 4.0 s highlight for 3.5–4 s videos; tests for best-value placement, trim tie-break, merged value · #9
- 2026-11-12 · DA · P1 · Review fixes on the validator: strict mode compares exactly (no float slack), huge integers and non-UTF-8 files no longer crash it; tests for short videos with two highlights, negative label score, the skiing example · #8
- 2026-11-11 · DA · P1 · `run.sh`, `rewind.cli`, `rewind.pipeline` (validate-or-fallback, atomic write), `video_io.ffprobe_meta`, `utils.seed_all`; contract tests on synthetic 30 s/12 s videos (validity, no `kit/`, filename independence, determinism, audit hook on file opens, failure → fallback, env team ID); ffmpeg in CI · #10
- 2026-11-11 · DA · P1 · Post-processing, fallback and `output.py` (`to_document`, deterministic `dumps`); `SHORT_VIDEO_T` corrected to 4.05 / 0.15 = 27.0 s; unit + Hypothesis property tests (any input → valid highlights; fallback passes strict validation for T in 5–2000 s) · #9
- 2026-11-11 · DA · P1 · Validator: `rewind/schema.py` (R1–R14, strict + kit severities), bundled class list (`rewind/vocab.py`), `rewind_eval.validate` directory CLI (manifest durations, missing files, team_id); 63 new tests, 21 of 22 deliberate bugs caught (1 harmless rewrite) · #8
- 2026-11-11 · DA · process · Review agents post every review on the PR from their person's account (request-changes / comment / approve, with suggested diffs); the author's agent replies on the PR with what it fixed (CLAUDE.md §5) · #11
- 2026-11-11 · DA · P2 · Metrics as pure functions: coverage (any-overlap / fraction, < 4 s ignored, budget penalty), mAP (highest-tIoU greedy matching, interpolated AP, 5 thresholds), frame F1 (exact top 15%), label accuracy; 31 hand-computed + property tests, 19/19 deliberate bugs caught · #12
- 2026-11-11 · DA · process · Review agents: every session runs a background review sub-agent that polls PRs (`tools/pr_watch.py`), reviews the other person's PRs and reports to the main agent, which posts the review and merges (CLAUDE.md §5); approved by prakartii · #7
- 2026-11-11 · DA · P1 · Skeleton: pyproject, pinned requirements, CI, constants, types, strict config loader + default.yaml, import-boundary tests (source scan, config scan, runtime import check); 69 tests, checked against deliberate bugs; review fixes: NaN/inf and duplicate keys rejected, more range checks, hash of the loaded config · #6
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
