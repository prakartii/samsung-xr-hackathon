Ablation on `report` (tools/ablate.py, config 79d5cb9bc1).

| variant | weighted | coverage_frac | pct_oracle | map | f1 | label_acc | coverage_any |
|---|---|---|---|---|---|---|---|
| full pipeline | 0.2985 | 0.3045 | 0.4999 | 0.1492 | 0.2256 | 0.8298 | 0.4163 |
| zero-shot actions (no ActionFormer) | 0.2411 | 0.2532 | 0.4157 | 0.0015 | 0.2256 | 0.8298 | 0.2716 |
| label-dedup selector (no expected coverage) | 0.2832 | 0.2662 | 0.4370 | 0.1492 | 0.2256 | 0.8298 | 0.3990 |
| no verb-noun prior (selector too) | 0.3005 | 0.3094 | 0.5080 | 0.1492 | 0.2256 | 0.8298 | 0.4266 |
| no hand signal | 0.2964 | 0.3045 | 0.4999 | 0.1492 | 0.2173 | 0.8298 | 0.4163 |
| no flow / camera motion | 0.2840 | 0.3045 | 0.4999 | 0.1492 | 0.1678 | 0.8298 | 0.4163 |
| no scene change (FPVSum) | 0.2911 | 0.3045 | 0.4999 | 0.1492 | 0.1963 | 0.8298 | 0.4163 |
| uniform baseline | 0.1124 | 0.1768 | 0.2902 | 0.0000 | 0.1669 | 0.0000 | 0.3380 |
| oracle baseline | 0.7728 | 0.6091 | 1.0000 | 1.0000 | 0.7167 | 1.0000 | 0.6286 |
