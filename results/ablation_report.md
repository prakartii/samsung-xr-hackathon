Ablation on `report` (tools/ablate.py, config 6dddc3af60).

| variant | weighted | coverage_frac | pct_oracle | map | f1 | label_acc | coverage_any |
|---|---|---|---|---|---|---|---|
| full pipeline | 0.2893 | 0.2815 | 0.4621 | 0.1492 | 0.2256 | 0.8298 | 0.4098 |
| zero-shot actions (no ActionFormer) | 0.2411 | 0.2534 | 0.4160 | 0.0015 | 0.2256 | 0.8298 | 0.2716 |
| label-dedup selector (no expected coverage) | 0.2831 | 0.2661 | 0.4368 | 0.1492 | 0.2256 | 0.8298 | 0.3992 |
| no verb-noun prior (selector too) | 0.2888 | 0.2802 | 0.4601 | 0.1492 | 0.2256 | 0.8298 | 0.4050 |
| no hand signal | 0.2872 | 0.2815 | 0.4621 | 0.1492 | 0.2173 | 0.8298 | 0.4098 |
| no flow / camera motion | 0.2748 | 0.2815 | 0.4621 | 0.1492 | 0.1678 | 0.8298 | 0.4098 |
| no scene change (FPVSum) | 0.2819 | 0.2815 | 0.4621 | 0.1492 | 0.1963 | 0.8298 | 0.4098 |
| uniform baseline | 0.1124 | 0.1768 | 0.2902 | 0.0000 | 0.1669 | 0.0000 | 0.3380 |
| oracle baseline | 0.7728 | 0.6091 | 1.0000 | 1.0000 | 0.7167 | 1.0000 | 0.6286 |
