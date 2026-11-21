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
