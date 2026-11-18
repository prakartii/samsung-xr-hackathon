"""FPVSum label: choose how keyframes are combined, by nested CV (P6 refinement).

    python tools/tune_label.py

The zero-shot label (rewind.label.fpvsum_label) averages per-frame class probabilities
over `label.n_keyframes` evenly spaced 1 fps CLIP embeddings. This tool scores a small
structural grid, never class phrases or prompts (those would be fitted to the scored
videos' labels):

- `n_keyframes`: 8, 16, 32, 64, or every cached frame (0),
- `aggregate`: `prob` (mean of per-frame softmax, current), `logprob` (mean of
  log-probabilities, a product of experts) or `embedding` (one softmax of the mean
  embedding).

Nested 5-fold CV over the FPVSum folds (CLAUDE.md §6.2): for each held-out fold the
setting with the best accuracy on the other four is chosen (ties: current setting,
then fewer frames), then scored on the held-out fold. Prints per-setting accuracy and
the held-out accuracy of the procedure against the current setting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

N_KEYFRAMES = [8, 16, 32, 64, 0]
AGGREGATES = ["prob", "logprob", "embedding"]
Setting = tuple[int, str]


def choose(acc_by_setting: dict[Setting, float], current: Setting) -> Setting:
    """Best accuracy; ties go to the current setting, then to fewer frames (0 = all,
    the most), then to the grid order of AGGREGATES."""
    best = max(acc_by_setting.values())
    tied = [s for s, a in acc_by_setting.items() if a >= best - 1e-12]
    if current in tied:
        return current
    return min(tied, key=lambda s: (s[0] if s[0] > 0 else 10**9, AGGREGATES.index(s[1])))


def nested_cv(
    correct: dict[Setting, dict[str, bool]], folds: dict[str, int], current: Setting
) -> tuple[float, float, list[Setting]]:
    """(held-out accuracy of the selection procedure, held-out accuracy of ``current``,
    setting chosen per fold). ``correct[s][v]``: video v labelled right under s."""
    hits_sel = hits_cur = 0
    chosen = []
    for k in sorted(set(folds.values())):
        train = [v for v, f in folds.items() if f != k]
        test = [v for v, f in folds.items() if f == k]
        acc = {s: sum(c[v] for v in train) / len(train) for s, c in correct.items()}
        s = choose(acc, current)
        chosen.append(s)
        hits_sel += sum(correct[s][v] for v in test)
        hits_cur += sum(correct[current][v] for v in test)
    return hits_sel / len(folds), hits_cur / len(folds), chosen


def main() -> None:
    from import_released_features import load_fingerprint_logs
    from predict_cached import _local, default_logs, video_index

    from rewind.analyze import COOKING, load_models
    from rewind.config import load_config
    from rewind.features.extract import signal_caches
    from rewind.label import fpvsum_label
    from rewind_eval.score import load_eval_data
    from rewind_eval.splits import load_splits

    cfg = load_config(REPO / "configs" / "default.yaml")
    folds = dict(load_splits().fpvsum_folds)
    gt = {}
    for k in sorted(set(folds.values())):
        gt.update({v: g.label for v, g in load_eval_data("tune", k).fpv_gt.items()})
    index, missing = video_index(sorted(folds), load_fingerprint_logs(default_logs()),
                                 REPO / "data" / "videos", _local)  # fmt: skip
    if missing:
        raise SystemExit(f"no fingerprint: {missing}")
    models = load_models(cfg, REPO)
    clip_cache = signal_caches(models.cache_root, cfg, models.signals)["clip"]
    text = models.text(cfg)
    idx = [i for i, k in enumerate(text.label_keys) if k != COOKING]
    keys = [text.label_keys[i] for i in idx]
    embs = {v: np.asarray(clip_cache.get(index[v][0])["clip_emb"]) for v in folds}

    current: Setting = (cfg.label.n_keyframes, "prob")
    correct: dict[Setting, dict[str, bool]] = {}
    for n in N_KEYFRAMES:
        for agg in AGGREGATES:
            pred = {v: fpvsum_label(e, text.label[idx], keys, text.logit_scale, n or len(e), agg)[0]
                    for v, e in embs.items()}  # fmt: skip
            correct[(n, agg)] = {v: pred[v] == gt[v] for v in folds}
    for s, c in correct.items():
        mark = "  (current)" if s == current else ""
        print(f"n_keyframes={s[0]:>2} aggregate={s[1]:<9} acc {sum(c.values())}/{len(c)}{mark}")
    sel, cur, chosen = nested_cv(correct, folds, current)
    print(f"\nnested CV held-out accuracy: selection {sel:.4f} vs current {cur:.4f}")
    print(f"chosen per fold: {chosen}")


if __name__ == "__main__":
    main()
