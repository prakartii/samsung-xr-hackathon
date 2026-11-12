Ablation on `report` (tools/ablate.py, config 315e074808).

| variant | weighted | coverage_any | pct_oracle | map | f1 | label_acc |
|---|---|---|---|---|---|---|
| full pipeline | 0.3614 | 0.4368 | 0.5976 | 0.1493 | 0.2626 | 0.8367 |
| zero-shot actions (no ActionFormer) | 0.3088 | 0.3977 | 0.5442 | 0.0015 | 0.2626 | 0.8367 |
| label-dedup selector (no expected coverage) | 0.3498 | 0.4079 | 0.5581 | 0.1493 | 0.2626 | 0.8367 |
| no verb-noun prior (selector too) | 0.3652 | 0.4464 | 0.6108 | 0.1493 | 0.2626 | 0.8367 |
| no audio terms (FPVSum) | 0.3519 | 0.4368 | 0.5976 | 0.1493 | 0.2248 | 0.8367 |
| no hand signal | 0.3595 | 0.4368 | 0.5976 | 0.1493 | 0.2551 | 0.8367 |
| no flow / camera motion | 0.3584 | 0.4368 | 0.5976 | 0.1493 | 0.2508 | 0.8367 |
| no scene change (FPVSum) | 0.3609 | 0.4368 | 0.5976 | 0.1493 | 0.2608 | 0.8367 |
| uniform baseline | 0.1763 | 0.3380 | 0.4625 | 0.0000 | 0.1645 | 0.0000 |
| oracle baseline | 0.8178 | 0.7309 | 1.0000 | 1.0000 | 0.7018 | 1.0000 |
