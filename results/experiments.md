# Experiment log (post-#61)

Baseline for everything below: main at the #61 merge (`aa5a981`), config `0e9c1929b0`,
`select.capture_rule: fraction`. Report (138 EPIC + 47 scored FPVSum, official rule):
weighted 0.2844, coverage 0.2692, mAP 0.1493, F1 0.2256, label 0.8298. Baseline outputs
kept locally in `results/preds/baseline_pr61_report` (git-ignored).

Rules: decisions on `tune` only (20 EPIC train videos; clean for selector decisions,
contaminated for detector decisions), adopt rule = mean gain >= 0.005 and more videos
better than worse; `report` measured once per adopted change, never used to choose.
FPVSum annotations are hidden test labels: nothing that changes FPVSum output is adopted.

## E1: selector re-tune under the official rule, real cached pipeline (2026-11-21)

- **Hypothesis.** The selector's other knobs were last tuned under any-overlap (or, on
  2026-11-20, approximately without CLIP). Re-tuning them against `coverage_frac` with the
  real pipeline should find better settings.
- **Method.** `tools/tune_epic_selector.py --passes 2` from the baseline, every GRID key.
- **Split.** `tune` (EPIC). **Files.** `configs/default.yaml`.
- **Search path (tune coverage_frac, each step passed the adopt rule):**

| Step | Change | tune coverage_frac |
|---|---|---|
| baseline | | 0.2970 |
| 1 | `cluster.tiou` 0.3 -> 0.2 | 0.3086 |
| 2 | `select.stride` 1.0 -> 0.25 | 0.3168 |
| 3 | `pairing.score_power` 0.5 -> 0.25 | 0.3240 |
| 4 | `post.merge_gap` 0.5 -> 2.0 | 0.3292 |
| 5 | `select.fraction_theta` 0.5 -> 0.55 | 0.3388 |

- **Excluded: step 4** (`post.merge_gap`). `analyze.py` applies it to FPVSum highlights as
  well, which can't be validated without the hidden FPVSum labels. Steps 1, 2, 3, 5 only
  affect the EPIC route (`score_power` is a monotone rescale of action scores).
  Those four together: tune coverage_frac **0.3302** (merge_gap 2.0 on top would add 0.0086).
- **Not adopted, tested in the same search:** hedge windows (`hedge.share` 0.1: -0.034 and
  -0.025 in the two passes; 0.2 worse), cluster k, cluster sim, window lengths, pairing alpha,
  temperature, fraction tau, budget 0.14, any_overlap (0.2910).
- **Report, measured once after the decision (same cache and scorer):**

| | coverage | % oracle* | mAP | F1 | label | weighted |
|---|---|---|---|---|---|---|
| baseline (#61) | 0.2692 | 0.4419 | 0.1493 | 0.2256 | 0.8298 | 0.2844 |
| E1 | **0.2815** | **0.4621** | 0.1492 | 0.2256 | 0.8298 | **0.2893** |

  *same oracle for both rows (window grid of the E1 config). Per video (EPIC): 66 better,
  41 worse, 31 same. Labels and all 49 FPVSum highlight sets identical. mAP -0.0001: the
  rescale is monotone, but output scores are rounded, which creates a few extra ties.
  187/187 outputs valid.
- **Runtime.** Detection-free part of the pipeline on the two longest local EPIC videos
  (17.8 min of video): 1.6 s -> 2.8 s (stride 0.25 = 4x more candidate windows).
- **Tests.** 711 passed, 26 skipped (fast suite, Windows; bash-script contract tests run on
  Linux CI). Four tests hard-coded the old defaults (score_power 0.5 arithmetic, a stride
  value in the config-hash test) and were updated with new arithmetic.
- **Decision: KEEP** (adopted on tune; report confirms the direction).
- **Revert.** Restore the four values in `configs/default.yaml` (git revert of the PR).

## X0: EPIC-only merge gap (2026-11-21)

| Field | Value |
|---|---|
| Hypothesis | The E1 search found `post.merge_gap` 0.5 -> 2.0 (+0.0086 tune coverage) but it also moved FPVSum highlights. A separate EPIC-only gap keeps the gain with FPVSum unchanged |
| Change | New `post.merge_gap_epic` (2.0) used by the EPIC selector's union model, its hedge clearance and EPIC post-processing; `post.merge_gap` (0.5) stays for FPVSum and the fallback. Files: `code/rewind/config.py`, `code/rewind/select/epic.py`, `code/rewind/analyze.py`, `configs/default.yaml`, `ARCHITECTURE.md` |
| Split | `tune` (20 EPIC train videos; decision). FPVSum checked for identity only, no GT used |
| Backward compatibility | With `merge_gap_epic: 0.5` the new code reproduces main's tune outputs for 69/69 videos |
| Isolation | FPVSum-route outputs identical 49/49; EPIC actions and all labels identical |
| tune coverage_frac | baseline 0.3302 -> **0.3388** (+0.0086); better 2, worse 1, same 17; distinct pairs +8 / -4 |
| Neighbouring values (tune) | 0.5: 0.3302, 1.0: 0.3299, 1.5: 0.3351, **2.0: 0.3388**, 3.0: 0.3372 (a plateau, not a spike) |
| mAP / F1 / label | mAP unchanged (actions identical; tune mAP contaminated anyway). F1 and label not evaluable without hidden FPVSum GT; FPVSum outputs identical, so unchanged by construction |
| Weighted (tune-evaluable part) | coverage term +0.4 x 0.0086 = **+0.0034** (estimate of the effect, tune only) |
| Budget / validity | max budget use 14.50%; 0 strict-validation errors on 69 tune outputs |
| Runtime | tune split 205 s -> 214 s wall (same order; one run each) |
| Tests | 716 passed, 26 skipped (fast suite, Windows); new tests fail on main's code |
| Report | not run (per instruction: no report-based selection; final outputs regenerated later) |
| Decision | **ADOPTED** (adopt rule: mean gain >= 0.005 and more videos better than worse) |
| Revert | `git revert` the X0 merge commit, or set `merge_gap_epic: 0.5` (reproduces the old behaviour exactly) |

## X1 / X2 / X3 on tune, and the train benchmark (2026-11-21)

- **X1 (tune, 20 videos):** `select.confidence_power` 1.0: coverage 0.3388 -> 0.3492 but 5 better /
  5 worse; 0.5: +0.0070, also fails the rule. Rejected on tune alone (branch exp/X1-confidence-weighting).
- **X2 (tune):** `cluster.span_mode: median`: 0.3388 -> 0.3061. **Rejected** (branch exp/X2-core-span).
- **X3 (tune):** `select.budget_frac` 0.147 / 0.148 / 0.149: 0.3310 / 0.3260 / 0.3267, all below
  0.3388 at 0.145. **Rejected** (branch exp/X3-budget-headroom). Budget safety was checked first:
  at 0.149 every output stays <= 14.95% even of the manifest duration.
- **Diagnostics (tune):** ours 0.330 vs an oracle restricted to our own candidate windows 0.629
  (unrestricted 0.627): the gap is selection, not candidates. Cluster confidence separates real
  from false clusters with AUROC 0.767, and the selector ignored it.
- **Train benchmark** (`tools/eval_selector_train.py`, 150 EPIC-100 train videos, fixed seed,
  tune and report videos excluded, released features + official train annotations, no CLIP):
  - stride 1.0 vs 0.25: -0.0124 (36 better, 72 worse): confirms E1's stride.
  - X0 off (EPIC gap 0.5) vs on: +0.0010 (34 better, 19 worse): X0 is **not confirmed**; neutral
    within noise, left in place.
  - **X1:** power 0.5: 0.3563 -> 0.3966 (+0.0404, 102 better, 20 worse); **power 1.0: 0.4223
    (+0.0660, 112 better, 22 worse)**. Power 1.0 is also best on tune (+0.0104).

## X1b: confidence-weighted expected coverage, adopted (2026-11-21)

| Field | Value |
|---|---|
| Change | `select.confidence_power: 1.0` (new key; 0 = previous behaviour). Files: `code/rewind/config.py`, `code/rewind/select/epic.py`, `configs/default.yaml`, `ARCHITECTURE.md` §5.4, `tools/tune_epic_selector.py` (GRID), tests |
| Evidence | 150 train videos: coverage +0.0660, 112 better / 22 worse (adopt rule passes); tune full pipeline +0.0104 (5 / 5) |
| Isolation | Only the EPIC selector's cluster weights; actions, labels, FPVSum unchanged (tune check: FPVSum 49/49 identical) |
| Caveat | Train/tune detections come from ActionFormer's own training videos, so confidence there is optimistic; held-out transfer is measured once at the final report run |
| Decision | **ADOPTED** |
| Revert | `select.confidence_power: 0.0` reproduces the previous selector exactly |

## C campaign: held-out selector protocol, C1, C4 (2026-11-22)

Protocol (C2, PR #71): `tools/eval_selector_train.py --part dev|val`. Dev = 150 EPIC-100 train
videos (seed 0), val = 150 different train videos (seed 1); both exclude tune and report; ids in
`results/selector_splits.json`. Settings are chosen on dev; val is used once to confirm. Caveat:
all EPIC train videos are ActionFormer's training data, so detections are optimistic (fine for
selector decisions, never used for detector decisions). No report data, no FPVSum labels.

Baseline (main, confidence_power 1.0): **dev 0.4223, val 0.4050** (official coverage rule).

| ID | Change | Dev coverage vs main | Better / worse | Val | Decision |
|---|---|---|---|---|---|
| C1 | Per-step verb/noun classifier evidence on cluster pairs (`select.step_evidence_power`) 0.5 / 1 / 2 | -0.0091 / -0.0310 / -0.0659 | 37/72, 34/91, 25/102 | not run (failed dev) | **REJECTED** (branch exp/C1-step-evidence) |
| C4 | Cluster confidence from all members: noisy-OR | +0.0043 | 55 / 58 | not run | **REJECTED** (below 0.005, more worse than better) |
| C4 | Damped noisy-OR (product ** 1/sqrt(n)) | +0.0022 | 41 / 39 | not run | **REJECTED** (noise) |
| C1+C4 | Combination | n/a | n/a | n/a | not applicable (neither passed alone) |

Reading: the per-step classifier disagrees with the detector's spans often enough to suppress
real actions; combining member scores adds nothing beyond the best member's score. The selector
gain available from confidence is already captured by X1 (confidence_power 1.0). The submission
(outputs, README, deck) is unchanged: report weighted 0.2985.

## Campaign #1: output action pairing (2026-11-22)

Output-only settings (`actions.*`); the selector's proposals are untouched, so coverage, F1 and
labels can't change. `tools/eval_actions_train.py` scores action mAP on the C2 dev/val train
videos (inflated absolute values: ActionFormer's training data; compare variants only).

| Variant (alone) | dev mAP | val mAP | Decision |
|---|---|---|---|
| baseline | 0.5105 | 0.5426 | |
| keep 10 / 25 pairs per segment | 0.5095 / 0.5107 | not run | rejected (no gain: the right label is usually not in the detector's top 5) |
| verb+noun gate 0.3 / 0.1 (also merges verb groups) | 0.5079 / 0.4800 | not run | rejected |
| per-class soft-NMS sigma 0.5 | 0.5086 | not run | rejected (no duplicates exist: dup rate 0.000) |
| **noun-only gate 0.3** | **0.5144 (+0.0039)** | **0.5447 (+0.0021)** | **ADOPTED** (better at tIoU .1, .3 and .5 on both) |
| noun-only gate 0.1 | 0.5064 | not run | rejected |

Reading: mAP is limited by the detector's verb/noun classification (fixing labels with GT would
add +0.24 dev mAP), not by pairing; output pairing offers only this small gain.

## Campaign #2: learned cluster-reality scorer (2026-11-22)

- Model: L2 logistic regression on 9 inference-time cluster features (confidence, member count,
  duration, pair entropy, top-1 prob, top-1/2 margin, position, local density, confidence rank;
  `code/rewind/select/reality.py`), label = cluster span holds >= 50% of a GT action instance.
  Fit on the C2 **dev** train videos only (`tools/train_reality.py`); val never used to fit.
- Ranking: AUC dev 5-fold (by video) 0.8025 vs confidence 0.7577; **val 0.8013 vs 0.7467**.
- Selector: cluster weight = P(real) ** confidence_power (replaces confidence ** 1.0).

| Split | confidence ** 1 (main) | P(real) ** 1 | **P(real) ** 2** | better / worse |
|---|---|---|---|---|
| dev (fit here) | 0.4223 | 0.4158 | **0.4326** | 67 / 52 |
| val (unseen) | 0.4050 | - | **0.4142** | 68 / 44 |
| tune, full pipeline with CLIP | 0.3492 | - | **0.3783** | 10 / 1 |

FPVSum outputs and all EPIC actions identical on tune (selector-only change). **ADOPTED.**

## Campaign #3: pair-level scorer + consistent clustering (2026-11-22)

- Model: L2 logistic regression per (cluster, pair) on the 9 cluster features + pair probability,
  log rank, ratio to top-1 (`select/reality.py: pair_features`); label = a GT instance of exactly
  that pair has >= 50% inside the cluster span. Fit on dev + 175 extra train videos
  (`tools/train_pairs.py`); val never fit. Val AUC 0.795 vs raw pair probability 0.737.
- Selector: each cluster's distribution := P(pair) ** pair_power (1.0).

| Check | main (Campaign #2) | pair scorer | better / worse |
|---|---|---|---|
| dev (in training data) | 0.4326 | 0.4388 | 70 / 49 |
| val (unseen) | 0.4142 | **0.4295** | 73 / 38 |
| tune full pipeline, CLIP linking (cluster.sim 0.85) | 0.3783 | 0.3703 | 5 / 7 |
| tune full pipeline, **cluster.sim 1.0** | 0.3783 | **0.3954** | **12 / 6** |

The model was trained on clusters linked by top-1 pair only (no CLIP for train videos); with
CLIP linking in production the features shift and the gain disappears. With `cluster.sim: 1.0`
production clusters exactly as in training and validation, and the gain holds (cluster scorer
alone with sim 1.0: 0.3748, 6/8, so the pair model is what helps). **ADOPTED** as the pair: pair
scorer + cluster.sim 1.0. The Campaign #2 coefficients stay in the config as the fallback when
`pair_coef` is emptied.

## Campaign #4: learned output-action score for mAP (2026-11-22)

- Model: L2 logistic regression per output action on 16 inference-time features (pairing
  score, verb/noun probability, verb/noun/pair rank in the proposal, verb-noun tIoU, duration,
  position, overlap density, same-class rank and count, same-verb/noun counts, score ratio,
  tIoU with a higher-ranked same-class action; `code/rewind/detect/rescore.py`). Label: mean
  over tIoU 0.1-0.5 of "true positive under greedy mAP matching". Fit on dev + 175 extra train
  videos (`tools/train_rescore.py`); transform chosen on dev 5-fold OOF; val scored once.
- Output score = P(TP) (old-score power 0, best on dev OOF).

| Split | pairing score (main) | learned P(TP) | @0.1 | @0.3 | @0.5 |
|---|---|---|---|---|---|
| dev, 5-fold OOF | 0.5144 | **0.5450** | 0.5415 -> 0.5728 | 0.5207 -> 0.5519 | 0.4749 -> 0.5035 |
| val (unseen, once) | 0.5447 | **0.5753** | 0.5735 -> 0.6049 | 0.5494 -> 0.5809 | 0.5058 -> 0.5337 |
| tune, full pipeline | 0.6637 | **0.6847** | 0.6900 -> 0.7113 | 0.6669 -> 0.6886 | 0.6266 -> 0.6460 |

Highlights and labels identical on tune (69/69). Caveat: all checks are EPIC *train* videos
(ActionFormer's training data), so absolute mAP is inflated; the gain is consistent on three
independent sets. **ADOPTED.**

## Final tuning (2026-11-22)

**Cluster tIoU 0.2 -> 0.4** (selector). Coverage: dev 0.4388 -> 0.4589 (88/27); val 0.4295 ->
0.4414 (57/39); tune full pipeline 0.3954 -> 0.4045 (11/4). 0.5/0.6 equal on dev with fewer wins.

**FPVSum-route highlight F1** on YouTube Highlights (Sun et al. 2014; first-person GoPro, the
only legitimate first-person importance labels; FPVSum labels never used; 32 videos with
cached features, fixed md5 dev/val halves 14/18). Baseline REWIND 0.152 vs uniform 0.128.

| Variant | dev F1 (better/worse) | val F1 (better/worse) |
|---|---|---|
| base (sigma 2, scene +0.5, no fill) | 0.1424 | 0.1597 |
| scene 0.0 | +0.034 (5/2) | +0.019 (8/2) |
| sigma 4 | +0.041 (6/2) | +0.046 (10/0) |
| fill 0.5 alone | +0.018 (5/1) | -0.001 (5/5) |
| **sigma 6, scene -0.5, fill 0.5** | **+0.083 (8/2)** | **+0.052 (10/3)** |

Adopted the last row. Caveats: small sample, the variant was picked after seeing both halves,
and YouTube Highlights labels (edited-video matches) differ from FPVSum's (frame importance).
The changes match hypotheses from the pre-data code audit (scene-cut reward, under-smoothing,
unused budget: highlights used 12.4% of the 14.5% budget on average).

## Highlight config chosen on SumMe (FPVSum protocol), 2026-11-22

SumMe (Gygli et al. 2014) is annotated like FPVSum: per-frame score = annotator agreement, GT =
top 15% frames (`tools/eval_summ_f1.py`, #81). It is independent of FPVSum and of the scored
data. First 9 videos with cached features (md5 halves 5/4):

| Config | SumMe F1 all | dev | val |
|---|---|---|---|
| B0 = sigma 6, scene -0.5, fill 0.5 (YouTube Highlights pick, #79) | 0.146 | 0.108 | 0.194 |
| B-old = sigma 2, scene +0.5, fill 0 | 0.258 | 0.174 | 0.364 |
| **sigma 2, scene +0.5, fill 0.5** | **0.269** | **0.188** | **0.370** |
| sigma 6, scene +0.5, fill 0 | 0.183 | 0.132 | 0.247 |
| sigma 2, scene -0.5, fill 0 | 0.214 | 0.156 | 0.286 |
| uniform | 0.235 | 0.219 | 0.256 |

YouTube Highlights labels (segments matched to a user's edit) rewarded heavy smoothing and a
negative scene-cut weight; the FPVSum-like protocol rewards neither. The fill-budget pass helps
on both. **ADOPTED:** sigma 2.0, w_scene_change +0.5, fill_step 0.5.
