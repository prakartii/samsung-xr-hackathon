# REWIND v2.3 — Distinct-Action Window Selection for Egocentric Video

**Track 4 — Action-Aware Video Summarization · Samsung XR Hackathon 2026**
*The most distinct steps per second of highlight.*

> **Status: design rationale only, frozen.** The build spec is `ARCHITECTURE.md`;
> where the two differ, `ARCHITECTURE.md` wins. Superseded here:
> - Routing by `manifest.csv` (§4, §5, §6.4): inference now routes on video signals
>   only and never reads the manifest or the filename (`ARCHITECTURE.md` §6.1).
> - The (1 − 1/e) / near-optimality guarantee (§4, §15): withdrawn; we report % of
>   oracle coverage.
> - Timeline and four team roles (§12, §13): we are two people; phases, owners and
>   dates live in `PROGRESS.md`.
> - "Second-hand" kit rules (§3.1, §10): confirmed against the official kit in `kit/`.
> - Open questions (§3.2): tracked in `PROGRESS.md` §3.
> - Feature extraction on a GPU: replaced by the CPU-first plan (`ARCHITECTURE.md` §10.1).

---

## 1. One-line positioning

REWIND turns a first-person video into a ~15% highlight reel of short windows of
original footage, each placed where the most **distinct** actions happen in the
fewest seconds. Selection is budget-constrained expected-coverage maximization over
clustered action proposals. For the Samsung jury it is an **on-device perception
pre-filter**: ~85% of visual tokens pruned, as many distinct steps kept as the
budget allows, measured against the oracle maximum.

---

## 2. Scoring — design everything to this

| Metric | Weight | Lives in | What wins it |
|---|---|---|---|
| **Action coverage** | **40%** | `highlights[]` | Overlap as many **distinct** GT (verb, noun) pairs as possible with ≥4 s windows inside 15% → pack windows where distinct actions are dense; expected-coverage selection; hedge windows for detector misses |
| **Action mAP** | 25% | `actions[]` | Ranked detections (required `score`) across the whole video at tIoU 0.1–0.5. Loose tIoU → recall and ranking matter more than boundaries |
| **Highlight F1** | 25% | `highlights[]` (FPVSum) | Frame-level match to top-15% human-importance frames → contiguous ≥4 s segments ranked by importance |
| **Label accuracy** | 10% | `label` (FPVSum) | 14 FPVSum categories → CLIP zero-shot |

**Structural rules**
- `actions[]` and `highlights[]` are two outputs with two objectives. Detect for mAP;
  select for coverage / F1.
- Coverage is measured against **GT time spans**. A window does not need the right
  label to score, only to overlap distinct GT actions. Labels matter to the selector
  only for deciding what is *distinct*.
- **The budget is tight.** At 14.5% and 4 s per window, the median EPIC video
  (216 s) gets ~**8 windows**; the median FPVSum video (159 s) ~6. Every window must
  earn its place.
- EPIC-KITCHENS-100 (138 val videos, 11 s – 33 min, 13.2 h total): optimize
  coverage + mAP. FPVSum (49 videos, 37 s – 6.3 min, 2.4 h total): optimize
  highlight F1 + label.

---

## 3. Kit rules

### 3.1 Known (second-hand — verify against our own kit copy)

Source: another Track 4 team's public repo (`emberwing-dash/Vistara`), which
committed the kit's `manifest.csv` and `supported_classes.json` and re-implemented
the validator from the kit's `TEAM_README.md`. They report the official
`validate.py` was **missing** from the kit package. We used only rule details from
that repo, not their approach.

| Rule | Detail |
|---|---|
| Schema | Keys: `team_id`, `video_id`, `duration_sec`, `label`, `actions`, `highlights` (§10) |
| `duration_sec` | Must be within **±0.5 s** of `manifest.csv` |
| `video_id` | Must equal the output filename stem |
| `label` | `{name, score ∈ [0,1]}`; name from 16 labels: 14 FPVSum categories + `cooking` + `other` |
| `actions[]` | ≤ 2000 items; each `{segment: [s, e], verb, noun, score ∈ [0,1]}`; `0 ≤ s < e ≤ duration + 0.5`; verb/noun from `supported_classes.json` (98 verbs / 301 nouns, EPIC keys). May be `[]` |
| `highlights[]` | ≥ 1 item; each `{segment: [s, e]}`; **sorted, non-overlapping**; **segments < 4.0 s are ignored in scoring**; at least one segment ≥ 4.0 s |
| Budget | Over 15% is a **warning** (soft scoring penalty), not a validation failure |
| Routing | `manifest.csv` columns: `video_id, duration_sec, dataset` (`epic_kitchens_100` / `fpvsum`). IDs differ in form (`P01_11` vs `Bike_lhTFVa0h9Rk`) |
| Edge case | 3 EPIC videos are < 26.7 s, so 15% < 4 s: cannot satisfy both rules |

### 3.2 Still open (ask prism@samsung.com on Nov 10)

| # | Question | What it changes |
|---|---|---|
| Q4 | How is a GT action "captured" by a highlight? Any overlap, a tIoU threshold, or a fraction of the action? | Window placement and the capture model in the scorer (§6.2, §9) |
| Q6 | Is highlight F1 scored on EPIC too, or FPVSum only? Is mAP scored on FPVSum? | Whether EPIC windows need importance; whether FPVSum needs `actions[]` |
| Q7 | Demo video: max 3 min or 5 min? (brochure says both) | Demo length; default ≤ 3 min |
| Q8 | Is the official `validate.py` coming? Does "< 4 s ignored" apply to both coverage and F1? What is the over-budget penalty curve? | Final validation; short-video handling (§6.5) |
| Q9 | May `run.sh` read the kit's `manifest.csv` to route by dataset? Will scoring use only the 187 listed videos? | Routing (§6.4); whether the fallback router matters |

Until answered: **any-overlap capture** in the selector with a **≥50%-of-action**
variant in the scorer (report both), the < 4 s rule applies everywhere, manifest
routing with a signal-based fallback.

---

## 4. Key decisions & rationale (don't re-litigate)

1. **Distinct-action windows.** Highlights under 4 s don't score, and the budget
   allows only ~8 windows on a median video. Each window should sit where the most
   *distinct* GT actions overlap it. EPIC actions are short (a few seconds) and dense,
   so a well-placed 4 s window can overlap 2–3 distinct actions; a badly placed one
   overlaps one repeated action. Window placement is the main coverage lever.
2. **Pay only for new seconds.** A window's cost is the seconds it adds to the union
   of selected highlights, so extending a selected window by 1–2 s to reach an
   adjacent distinct action is cheap. This keeps selection, merging and the
   non-overlap rule in one mechanism.
3. **Expected coverage, not label dedup.** Top-1 labels are often wrong. Deduping on
   them drops genuinely distinct actions. Cluster proposals, keep top-k label
   distributions, and maximize the expected number of distinct pairs covered (§6.2).
   This is monotone submodular, so greedy keeps its (1−1/e) guarantee.
4. **Calibration is useful for coverage, not for mAP.** Temperature scaling is
   monotonic, so it cannot change mAP rankings. Expected coverage multiplies
   probabilities, so calibrated top-k distributions improve it. Fit one temperature
   on the tuning split. Not a Day 1–4 task.
5. **Hedge windows.** GT actions the detector missed still count. Reserve a small
   share of the budget (≈1 window per 8, tuned) for ≥4 s windows on active
   (high-motion, hands-visible) stretches with no detections.
6. **One selector, two modes**, routed by `manifest.csv`. EPIC mode = distinct-action
   windows by expected coverage. FPVSum mode = contiguous segments by importance.
   Same code; gain function and segment-length prior switch.
7. **Commit the stack; cut HOI detection.** A per-frame hand–object detector
   (100DOH/EgoHOS) is slow and redundant with action proposals on EPIC. Use
   MediaPipe hands + optical-flow magnitude as a cheap activity signal.
8. **Scorer before selector.** The official `validate.py` reportedly isn't even in
   the kit, and no scorer is. Every tuning decision needs our own coverage / mAP / F1
   scorer, plus the oracle.
9. **Clean tuning split.** EPIC-100 val labels are public, and the 138 val videos are
   what we are scored on. Tune hyperparameters on ~20 EPIC **train** videos; report on
   val. State this in the README and deck.
10. **Timebox the UI.** It scores zero. The demo reel and timeline are generated from
    the output JSON with ffmpeg + one static HTML page.

---

## 5. Pipeline — 6 modules

```
MP4 (+ video_id from filename, dataset + duration from manifest.csv)
 │
 ├─[M1] Features (cached)       SlowFast clip features @ ActionFormer's stride
 │                              + CLIP frame embeddings @ 1 fps
 │                              + flow magnitude + MediaPipe hand presence @ 2–4 fps
 │
 ├─[M2] Detect                  ActionFormer EPIC-100 (verb head, noun head)
 │                              → proposals {segment, top-k verbs, top-k nouns}
 │                              → pair via EPIC-train verb–noun co-occurrence prior,
 │                                restricted to supported_classes.json
 │                              ──────────────────────────────► actions[] (with score)
 │
 ├─[M3] Cluster                 merge proposals by temporal overlap + CLIP similarity
 │                              → clusters with a top-k (verb,noun) distribution
 │
 ├─[M4] Select                  EPIC: ≥4 s distinct-action windows, expected-coverage
 │                                    greedy on gain per new second, hedge windows
 │                              FPVSum: contiguous ≥4 s segments by importance
 │                              → sort, merge, cap at 14.5% ─────► highlights[]
 │
 ├─[M5] Label                   EPIC → "cooking"; FPVSum → CLIP zero-shot ─► label
 │
 └─[M6] Write + check           schema JSON (§10) → in-process validator (§3.1 rules)
                                → out.json
```

**Stage 0 (Day 1, before any ML):** `run.sh <video.mp4> <out.json>` writes JSON in
the §10 schema (uniform ≥4 s highlights at 14.5%, empty actions, `cooking` /
`other` label) that passes **our** validator implementing every §3.1 rule (and the
official `validate.py` if it arrives). Non-compliance = DQ.

`duration_sec` comes from ffprobe and is checked against the manifest (±0.5 s).
Feature caching is for development speed; `run.sh` still computes everything from
the raw MP4 on a clean machine.

---

## 6. Highlight selection

### 6.1 Candidate windows (EPIC mode)

Slide windows of length `L ∈ {4.0, 5.0, 6.0}` s (stored as 4.05 s minimum for float
safety) with a 0.5 s stride over the video. Keep only windows overlapping at least
one proposal, plus hedge candidates (§6.3).

### 6.2 Expected-coverage gain per new second

Each cluster `c` carries a distribution `p_c(pair)` over its top-k (verb, noun) pairs
(k = 5, calibrated, renormalized), and a span. `cap(w, c)` ∈ [0, 1] is the
probability window `w` captures cluster `c`'s action under the capture rule (Q4):
any-overlap → 1 if they overlap; fraction rule → overlap fraction vs threshold,
softened. With `S` the selected windows:

```
P(pair covered | S) = 1 − Π_{w ∈ S} Π_c (1 − cap(w, c) · p_c(pair))
gain(w | S)         = Σ_pair [P(pair covered | S ∪ {w}) − P(pair covered | S)]
cost(w | S)         = |w \ ∪S|            # only seconds not already selected
pick argmax gain/cost (lazy greedy) until |∪S| ≥ 0.145 · T
```

Overlapping or adjacent picks merge into one longer highlight automatically.
Selecting a window that would leave a fragment < 4 s after merging is disallowed.

### 6.3 Hedge windows

Before greedy, reserve `h · B` of the budget (`h` tuned; start at 1/8). After greedy,
fill it with 4 s windows centered on peaks of `hand_presence × flow` in stretches
with no selected window and no strong proposal, spaced ≥ 3 s apart.

### 6.4 Mode routing and FPVSum mode

Route by `dataset` in `manifest.csv` (looked up by `video_id`). Fallback if the id is
missing: input signals (resolution, fps, duration, CLIP P(cooking)).

FPVSum mode: segment the video into shots (CLIP embedding change points), score each
second by the importance proxy (§8), and pick contiguous segments of 4–10 s by mean
importance until the budget. Merge touching segments.

### 6.5 Post-processing (both modes)

Sort → merge segments with gaps < 0.5 s → drop any fragment < 4.0 s → trim the
lowest gain-per-second segment ends until total ≤ **0.145 · T** (never below 4.0 s
per segment) → clamp to [0, T] → validate.

**Short videos (T < 26.7 s, 3 EPIC videos):** emit one 4.0 s window at the best
gain. A slight budget overrun (soft penalty) beats a segment that is ignored.

**Selector growth rule:** the MVP gain is exactly §6.2 (+ hedge). Add a term
(temporal diversity, visual diversity, importance blending) only when an ablation on
the tuning split shows it helps.

---

## 7. Action detection → `actions[]` (mAP)

- **Model:** ActionFormer's released EPIC-KITCHENS-100 checkpoints (verb and noun),
  run on **exactly the features they were trained on** (the SlowFast features used in
  the ActionFormer EPIC config). Do not swap backbones: that means retraining.
- **Leakage check (Day 1):** confirm the checkpoint was trained on EPIC train only,
  not train+val. If unclear, document it and prefer a train-only checkpoint.
- **Verb–noun pairing:** `score = p(verb) · p(noun) · prior(verb, noun)`, where the prior
  is the EPIC train co-occurrence table; keep only pairs in `supported_classes.json`;
  rescale to [0, 1].
- **Recall-first, ranked:** the required `score` field means AP can rank our output.
  Emit up to ~1000 proposals per video (limit 2000), sorted by score; tune the count
  and minimum score on the split.
- **FPVSum:** `actions: []` unless Q6 says mAP is scored there.
- **Fallback detector:** sliding windows (2 s, stride 1 s) over CLIP/EgoVLP features,
  zero-shot against verb and noun text prompts, NMS. Used if ActionFormer isn't
  producing mAP by end of Nov 13.

---

## 8. FPVSum importance and label

- **Importance proxy (MVP):** per-second `z(flow) · z(hand_presence) · z(scene change)`,
  smoothed. FPVSum categories are mostly sports (bike, ski, surfing, scuba…), so test
  global camera motion as a substitute for hand presence on those. No training on 49
  videos.
- **Learned option (only if the proxy loses on CV):** ridge regression on CLIP
  embeddings, 5-fold CV across FPVSum videos.
- **Label:** EPIC → `{name: "cooking", score: 1.0}`. FPVSum → CLIP ViT-B/16 zero-shot
  over the 14 FPVSum categories on 8 evenly spaced keyframes, mean of probabilities,
  prompt ensemble ("a first-person video of …"). `other` only if routing is uncertain.

---

## 9. Evaluation harness (Day 1 deliverable)

- `eval/validate_ours.py`: every §3.1 rule; run on all outputs in CI.
- `eval/score.py`: coverage (capture rule as a parameter: any-overlap and ≥50%-of-action,
  segments < 4 s dropped, over-budget penalty as a parameter), mAP at tIoU
  {0.1, 0.2, 0.3, 0.4, 0.5} on (verb, noun) pairs, frame-level F1 vs top-15%
  importance, label accuracy. Weighted total.
- **Splits:** `tune` = ~20 EPIC train videos + 5-fold CV on FPVSum; `report` = 138 EPIC
  val + all FPVSum (CV folds).
- **Reference lines on every run:**
  - *Uniform*: evenly spaced 4 s windows at 14.5% (floor).
  - *Oracle coverage*: greedy max coverage using GT action spans and ≥4 s windows
    (the ceiling; headline chart = "REWIND reaches X% of oracle").
  - *Current system.*
- **Ablations:** label dedup vs expected coverage; 4 s only vs {4, 5, 6} s windows;
  new-seconds cost vs fixed cost; hedge on/off and `h`; calibration on/off;
  importance proxy vs learned.
- `results/leaderboard.csv`: one row per run (git SHA, config, all metrics).

---

## 10. Output schema (from the kit's TEAM_README, second-hand — verify)

```json
{
  "team_id": "CollegeName_TeamName",
  "video_id": "P01_11",
  "duration_sec": 561.53,
  "label": { "name": "cooking", "score": 1.0 },
  "actions": [
    { "segment": [12.4, 15.9], "verb": "take", "noun": "knife", "score": 0.82 }
  ],
  "highlights": [
    { "segment": [40.2, 44.3] },
    { "segment": [71.0, 76.5] }
  ]
}
```

The brochure's simpler schema (`start`/`end`, string `label`, `importance`) is **not**
what the kit specifies. The kit README wins.

---

## 11. MVP vs stretch vs not building

**MVP:** run.sh + schema-valid JSON for all 187 videos; our validator; local scorer +
oracle; cached features; ActionFormer → `actions[]`; clustering; expected-coverage
≥4 s window selection + hedge; FPVSum importance-proxy mode; label; Docker; README;
demo video; deck.

**Stretch (only after a full scored run exists):** temperature calibration;
learned FPVSum importance; extra selector terms that win ablations; ONNX export +
CPU latency benchmark; HTML timeline viewer; "downstream QA" demo (a small local VLM
answers "what did I do?" from the full video vs from the highlights).

**Not building:** per-frame HOI detector, 3D reconstruction, live XR tracking,
LLM-based selection, generative video, conversational assistant, cloud inference,
mobile/headset deployment, training any detector from scratch.

---

## 12. Timeline (today Nov 10 → submit Nov 18, 11:59 PM)

| Date | Milestone | Exit check |
|---|---|---|
| **Nov 10** | Get our own kit copy and diff it against §3.1; email Q4/Q6–Q9; Stage 0 `run.sh` + our validator; start feature caching; lock ActionFormer checkpoint + features | All 187 dummy outputs pass our validator |
| Nov 11 | `eval/score.py`; uniform + oracle numbers (expect Nov 10 to spill here) | Leaderboard has 2 rows |
| Nov 11–12 | Features cached for all 187 + tune split; ActionFormer → `actions[]`; pairing; mAP measured | mAP row on leaderboard |
| Nov 13–14 | Clustering; window candidates; expected-coverage greedy with new-seconds cost; hedge; post-processing | Coverage beats uniform; % of oracle reported |
| Nov 15 | FPVSum mode (shots + importance proxy); FPVSum label | All four metrics on leaderboard |
| Nov 16 | Full end-to-end run on 187 videos from raw MP4; tuning; ablations; latency numbers | All outputs pass validation |
| Nov 17 | Dockerfile; README (setup, data, tuning split); demo video (ffmpeg reel from JSON); deck | Fresh-clone Docker run reproduces one video |
| **Nov 18** | Buffer only: final validate, package `TEAM_ID/`, submit via Google Form | Submitted |

**Kill rules:** if ActionFormer yields no mAP by end of Nov 13, switch to the §7
fallback. If expected coverage doesn't beat label dedup by Nov 14, ship label dedup
and move on.

**Compute note:** EPIC is 13.2 h of video (17 videos over 10 min, the longest 33 min).
Start caching on Nov 10 and process the longest videos first.

---

## 13. Team roles (4)

- **ML lead:** M1 SlowFast features, M2 ActionFormer + pairing, mAP, fallback detector.
- **Algorithms lead:** M3 clustering, M4 selector (windows, expected coverage, hedge,
  post-processing), oracle, ablations.
- **Perception lead:** flow + MediaPipe hands, FPVSum mode + importance proxy,
  M5 label, F1/label eval.
- **Product/infra lead:** kit diff, Stage 0, our validator, `eval/score.py` +
  leaderboard, `run.sh`, Docker, README, demo video, deck.

---

## 14. Risks & mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Second-hand kit rules are wrong | DQ or wasted work | Diff our own kit copy on Nov 10; validator rules in one file |
| Coverage needs a fraction of the action, not any overlap (Q4) | Window placement changes | Capture rule is a scorer + selector parameter; report both variants |
| Very few windows per video (~8 median) | Low coverage ceiling | Place windows on distinct-action density; extend windows cheaply; oracle shows the ceiling is shared by all teams |
| ActionFormer features/checkpoint mismatch | No `actions[]` | Use the exact training features; fallback detector by Nov 13 |
| Checkpoint trained on EPIC val | Leakage / credibility | Verify Day 1; document; prefer train-only |
| Compute for 15.6 h of video | Slow iteration | Cache features from Nov 10, longest first; CPU path for light signals |
| Short videos (< 26.7 s) | Can't meet 4 s and 15% | One 4 s window; accept soft penalty |
| `duration_sec` mismatch | Validation failure | ffprobe + manifest cross-check in M6 |
| Overfitting FPVSum (49 vids) | ↓ F1 | Unsupervised proxy first; CV for anything learned |
| UI gold-plating | Deadline | Demo reel via ffmpeg; HTML timeline is stretch |

---

## 15. Jury story

**Problem:** Egocentric video is thousands of visual tokens, mostly idle or
repetitive. Existing summarizers keep visually busy moments without knowing what the
person did.

**Solution:** REWIND detects actions, groups them into distinct steps, and spends its
15% budget on the few seconds where the most *different* steps happen, chosen by an
expected-coverage optimizer with a near-optimality guarantee. Hedge windows catch
steps the detector missed. 4-second windows also make a watchable reel.

**Three numbers for the deck:**
1. **X% of oracle coverage** in a 15% budget.
2. **~85% of visual tokens pruned** (frames kept vs total).
3. **Y s of CPU time per minute of video** (ONNX, no cloud).

*REWIND: the most distinct steps per second of highlight.*
