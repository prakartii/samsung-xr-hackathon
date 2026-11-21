Ablation on `tune` (tools/ablate.py, config 6dddc3af60).

| variant | weighted | coverage_frac | pct_oracle | map | f1 | label_acc | coverage_any |
|---|---|---|---|---|---|---|---|
| full pipeline | 0.4346 | 0.3302 | 0.5269 | 0.6532 | 0.2228 | 0.8352 | 0.4429 |
| zero-shot actions (no ActionFormer) | 0.2371 | 0.2420 | 0.3861 | 0.0043 | 0.2228 | 0.8352 | 0.2567 |
| label-dedup selector (no expected coverage) | 0.4253 | 0.3069 | 0.4898 | 0.6532 | 0.2228 | 0.8352 | 0.4262 |
| no verb-noun prior (selector too) | 0.4315 | 0.3224 | 0.5145 | 0.6532 | 0.2228 | 0.8352 | 0.4382 |
| no hand signal | 0.4328 | 0.3302 | 0.5269 | 0.6532 | 0.2154 | 0.8352 | 0.4429 |
| no flow / camera motion | 0.4203 | 0.3302 | 0.5269 | 0.6532 | 0.1655 | 0.8352 | 0.4429 |
| no scene change (FPVSum) | 0.4281 | 0.3302 | 0.5269 | 0.6532 | 0.1968 | 0.8352 | 0.4429 |
| uniform baseline | 0.1119 | 0.1776 | 0.2834 | 0.0000 | 0.1636 | 0.0000 | 0.3583 |
| oracle baseline | 0.7797 | 0.6267 | 1.0000 | 1.0000 | 0.7160 | 1.0000 | 0.6415 |
