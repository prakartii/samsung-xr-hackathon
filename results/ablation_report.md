Ablation on `report` (tools/ablate.py, config 33603bb882).

| variant | weighted | coverage_frac | pct_oracle | map | f1 | label_acc | coverage_any |
|---|---|---|---|---|---|---|---|
| full pipeline | 0.2789 | 0.2555 | 0.4312 | 0.1493 | 0.2256 | 0.8298 | 0.4368 |
| zero-shot actions (no ActionFormer) | 0.2318 | 0.2301 | 0.3884 | 0.0015 | 0.2256 | 0.8298 | 0.3977 |
| label-dedup selector (no expected coverage) | 0.2751 | 0.2461 | 0.4153 | 0.1493 | 0.2256 | 0.8298 | 0.4079 |
| no verb-noun prior (selector too) | 0.2821 | 0.2636 | 0.4449 | 0.1493 | 0.2256 | 0.8298 | 0.4464 |
| no hand signal | 0.2768 | 0.2555 | 0.4312 | 0.1493 | 0.2173 | 0.8298 | 0.4368 |
| no flow / camera motion | 0.2645 | 0.2555 | 0.4312 | 0.1493 | 0.1678 | 0.8298 | 0.4368 |
| no scene change (FPVSum) | 0.2716 | 0.2555 | 0.4312 | 0.1493 | 0.1963 | 0.8298 | 0.4368 |
| uniform baseline | 0.1124 | 0.1768 | 0.2983 | 0.0000 | 0.1669 | 0.0000 | 0.3380 |
| oracle baseline | 0.7620 | 0.5925 | 1.0000 | 1.0000 | 0.6999 | 1.0000 | 0.6159 |
