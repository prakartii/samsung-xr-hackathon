Ablation on `tune` (tools/ablate.py, config 33603bb882).

| variant | weighted | coverage_frac | pct_oracle | map | f1 | label_acc | coverage_any |
|---|---|---|---|---|---|---|---|
| full pipeline | 0.4178 | 0.2883 | 0.4820 | 0.6532 | 0.2228 | 0.8352 | 0.4883 |
| zero-shot actions (no ActionFormer) | 0.2290 | 0.2217 | 0.3706 | 0.0043 | 0.2228 | 0.8352 | 0.3882 |
| label-dedup selector (no expected coverage) | 0.4064 | 0.2596 | 0.4339 | 0.6532 | 0.2228 | 0.8352 | 0.4299 |
| no verb-noun prior (selector too) | 0.4186 | 0.2903 | 0.4854 | 0.6532 | 0.2228 | 0.8352 | 0.4691 |
| no hand signal | 0.4160 | 0.2883 | 0.4820 | 0.6532 | 0.2154 | 0.8352 | 0.4883 |
| no flow / camera motion | 0.4035 | 0.2883 | 0.4820 | 0.6532 | 0.1655 | 0.8352 | 0.4883 |
| no scene change (FPVSum) | 0.4113 | 0.2883 | 0.4820 | 0.6532 | 0.1968 | 0.8352 | 0.4883 |
| uniform baseline | 0.1119 | 0.1776 | 0.2969 | 0.0000 | 0.1636 | 0.0000 | 0.3583 |
| oracle baseline | 0.7644 | 0.5981 | 1.0000 | 1.0000 | 0.7007 | 1.0000 | 0.6079 |
