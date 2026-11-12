Ablation on `tune` (tools/ablate.py, config 315e074808).

| variant | weighted | coverage_any | pct_oracle | map | f1 | label_acc |
|---|---|---|---|---|---|---|
| full pipeline | 0.5080 | 0.4883 | 0.6760 | 0.6532 | 0.2636 | 0.8352 |
| zero-shot actions (no ActionFormer) | 0.3058 | 0.3882 | 0.5375 | 0.0043 | 0.2636 | 0.8352 |
| label-dedup selector (no expected coverage) | 0.4847 | 0.4299 | 0.5951 | 0.6532 | 0.2636 | 0.8352 |
| no verb-noun prior (selector too) | 0.5004 | 0.4691 | 0.6495 | 0.6532 | 0.2636 | 0.8352 |
| no audio terms (FPVSum) | 0.4978 | 0.4883 | 0.6760 | 0.6532 | 0.2228 | 0.8352 |
| no hand signal | 0.5062 | 0.4883 | 0.6760 | 0.6532 | 0.2564 | 0.8352 |
| no flow / camera motion | 0.5050 | 0.4883 | 0.6760 | 0.6532 | 0.2516 | 0.8352 |
| no scene change (FPVSum) | 0.5077 | 0.4883 | 0.6760 | 0.6532 | 0.2622 | 0.8352 |
| uniform baseline | 0.1842 | 0.3583 | 0.4961 | 0.0000 | 0.1636 | 0.0000 |
| oracle baseline | 0.8141 | 0.7223 | 1.0000 | 1.0000 | 0.7007 | 1.0000 |
