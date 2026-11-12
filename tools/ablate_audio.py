"""P6 ablation: do audio terms help the FPVSum importance proxy? (CV folds only)

    python tools/ablate_audio.py

For each FPVSum video (local MP4 + cached flow/hands/CLIP): loudness z(log RMS) and
z(spectral flux) on the dense grid (features/audio.py), added to the importance
signal with weights w_rms, w_flux before the same Gaussian smoothing:

    imp' = smooth(raw + w_rms * z(log rms) + w_flux * z(flux))

then shots → windows → post-processing → frame F1, exactly as the FPVSum route.
Scored on the videos whose GT timing is usable (`FpvGT.timing_ok`) and compared by
`tune_fpvsum.cross_validate` (pick on 4 folds, score the 5th). PROGRESS P6: keep audio
only if the CV mean beats the current config (weights 0, 0). Audio is decoded once
and kept in --audio-cache (outside the repo cache; not part of inference).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code"), str(REPO / "tools")]

WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
SR = 8000


def main() -> None:
    from tune_fpvsum import cross_validate

    from rewind.analyze import load_models
    from rewind.config import load_config
    from rewind.features.audio import RMS_FLOOR, decode_mono, rms_at, spectral_flux_at
    from rewind.features.cache import content_fingerprint
    from rewind.features.extract import FEATURES, signal_caches
    from rewind.features.flow import zscore
    from rewind.postprocess import postprocess
    from rewind.select.importance import (
        ImportanceParams,
        gaussian_smooth,
        importance_signal,
        select_segments,
        shots,
    )
    from rewind_eval.f1 import frame_f1
    from rewind_eval.gt_fpvsum import load_fpvsum
    from rewind_eval.splits import load_splits

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "default.yaml")
    ap.add_argument("--audio-cache", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = ImportanceParams(**asdict(cfg.importance))
    folds_all = load_splits().fpvsum_folds
    with (REPO / "kit" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        dur = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}
    gt = load_fpvsum(
        REPO / "data" / "annotations" / "fpvsum" / "gt_data.mat", {v: dur[v] for v in folds_all}
    )
    folds = {v: k for v, k in folds_all.items() if gt[v].timing_ok}
    models = load_models(cfg, REPO)
    caches = signal_caches(models.cache_root, cfg, models.signals)
    args.audio_cache.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for vid in sorted(folds):
        mp4 = args.videos / f"{vid}.mp4"
        sig = {f: caches[f].get(content_fingerprint(mp4)) for f in FEATURES}
        t = np.asarray(sig["flow"]["t"])
        npz = args.audio_cache / f"{vid}.npz"
        if npz.exists():
            with np.load(npz) as a:
                rms, flux = a["rms"], a["flux"]
        else:
            x = decode_mono(mp4, SR)
            if x is None:
                raise SystemExit(f"{vid}: no audio stream")
            win = 2.0 / cfg.signals.dense_fps
            rms, flux = rms_at(x, SR, t, win), spectral_flux_at(x, SR, t, win)
            np.savez(npz, rms=rms, flux=flux)
        base = importance_signal(t, sig["flow"]["flow"], sig["hands"]["hands"],
                                 sig["flow"]["global_motion"], sig["clip"]["t"],
                                 sig["clip"]["clip_emb"], p)  # fmt: skip
        spans = shots(sig["clip"]["t"], sig["clip"]["clip_emb"], dur[vid], p.shot_thr)
        rate = 1.0 / np.median(np.diff(t))
        audio = (zscore(np.log(rms + RMS_FLOOR)), zscore(flux))
        prepared[vid] = (t, base, spans, audio, rate)
        print(f"prepared {vid}", flush=True)

    scores: dict[str, dict[str, float]] = {}
    for w_rms, w_flux in itertools.product(WEIGHTS, WEIGHTS):  # (0, 0) first = current
        name = f"rms{w_rms}_flux{w_flux}"
        scores[name] = {}
        for vid, (t, base, spans, (a_rms, a_flux), rate) in prepared.items():
            T = dur[vid]
            imp = base + gaussian_smooth(w_rms * a_rms + w_flux * a_flux, p.sigma * rate)
            segs = select_segments(t, imp, spans, T, cfg.select.budget_frac * T, p)
            hl = postprocess(
                segs, [0.0] * len(segs), T, budget_frac=cfg.select.budget_frac,
                merge_gap=cfg.post.merge_gap, short_video_mode=cfg.post.short_video_mode,
            )  # fmt: skip
            scores[name][vid] = frame_f1(gt[vid].times, gt[vid].highlight, hl)

    cv = cross_validate(scores, folds)
    current = cross_validate({"rms0.0_flux0.0": scores["rms0.0_flux0.0"]}, folds)
    print(json.dumps({
        "videos": len(folds),
        "current_cv_mean": round(current["cv_mean"], 4),
        "audio_cv_mean": round(cv["cv_mean"], 4),
        "per_fold": {k: [c, round(s, 4)] for k, (c, s) in cv["per_fold"].items()},
        "chosen_all_folds": cv["chosen"],
        "chosen_vs_current_videos": {
            "better": sum(scores[cv["chosen"]][v] > scores["rms0.0_flux0.0"][v] + 1e-9
                          for v in folds),
            "worse": sum(scores[cv["chosen"]][v] < scores["rms0.0_flux0.0"][v] - 1e-9
                         for v in folds),
        },
        "all_folds_mean": {n: round(float(np.mean(list(s.values()))), 4)
                           for n, s in scores.items()},
    }, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
