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
