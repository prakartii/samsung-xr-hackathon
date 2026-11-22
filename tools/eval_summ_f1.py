"""Highlight F1 of REWIND outputs on TVSum and SumMe, scored like FPVSum.

    PYTHONPATH=code python tools/eval_summ_f1.py --dataset tvsum --pred-dir outputs/tvsum \
        --videos-dir D:/rewind-data/external/tvsum/videos \
        --anno D:/rewind-data/external/tvsum/data/ydata-tvsum50-anno.tsv
    PYTHONPATH=code python tools/eval_summ_f1.py --dataset summe --pred-dir outputs/summe \
        --videos-dir D:/rewind-data/external/summe/videos \
        --anno D:/rewind-data/external/summe/eccv16_dataset_summe_google_pool5.h5

Why: YouTube Highlights labels "matched in the owner's edit" intervals, which is not
what FPVSum scores. TVSum (Song et al., CVPR 2015; 20 annotators rate every 2 s shot
1-5) and SumMe (Gygli et al., ECCV 2014; 15-18 users each pick a 5-15% summary) carry
per-frame importance, the same kind of label as FPVSum's ``Score``.

GT (mirrors ``rewind_eval.gt_fpvsum`` + ``rewind_eval.f1``):
- per-frame importance = mean over annotators (TVSum: mean 1-5 rating per frame;
  SumMe: fraction of users whose summary contains the frame);
- GT positives = ``rewind_eval.f1.top_frames(importance, 0.15)``: the top-15% frames,
  exact count, stable on ties (FPVSum's ``Binary_score`` is its top-15% frames);
- frame i sits at ``i * T / N`` s, T = MP4 duration (as ``load_fpvsum`` does);
- videos whose N / T fits no standard frame rate are skipped (``gt_rate_suspect``),
  exactly the FPVSum ``timing_ok`` rule;
- F1 = ``rewind_eval.f1.frame_f1`` (predicted positives: frames inside a highlight
  >= 4 s, end exclusive).

Annotation sources: TVSum ``ydata-tvsum50-anno.tsv`` (video_id, category, comma list of
per-frame scores; one row per annotator). SumMe: the eccv16 ``*_summe_*.h5`` (per video
group: ``video_name``, ``user_summary`` (users, n_frames) binary, full frame rate).
Reads container metadata only (PyAV), never decodes frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code")]

from rewind_eval.f1 import frame_f1, top_frames  # noqa: E402
from rewind_eval.gt_fpvsum import gt_rate_suspect  # noqa: E402

GT_TOP_FRAC = 0.15  # FPVSum Binary_score = top-15% frames
Interval = tuple[float, float]


def tvsum_scores(rows: Sequence[Sequence[str]]) -> dict[str, np.ndarray]:
    """TVSum anno rows (vid, category, "s1,s2,...") -> vid -> mean score per frame."""
    per: dict[str, list[np.ndarray]] = {}
    for vid, _cat, scores in rows:
        per.setdefault(vid, []).append(np.array([float(x) for x in scores.split(",")]))
    out = {}
    for vid, users in per.items():
        if len({len(u) for u in users}) != 1:
            raise ValueError(f"{vid}: annotators disagree on frame count")
        out[vid] = np.mean(users, axis=0)
    return out


def load_tvsum(anno_tsv: Path) -> dict[str, np.ndarray]:
    with open(anno_tsv, encoding="utf-8", newline="") as f:
        return tvsum_scores([r for r in csv.reader(f, delimiter="\t") if r])


def summe_scores(user_summary: np.ndarray) -> np.ndarray:
    """(users, n_frames) 0/1 selections -> fraction of users selecting each frame."""
    return np.asarray(user_summary, dtype=float).mean(axis=0)


def load_summe(h5_path: Path) -> dict[str, np.ndarray]:
    import h5py  # eval-only dependency, not in requirements: pip install h5py

    out = {}
    with h5py.File(h5_path, "r") as f:
        for key in f:
            g = f[key]
            name = g["video_name"][()]
            name = name.decode() if isinstance(name, bytes) else str(name)
            out[name.strip().replace(" ", "_")] = summe_scores(g["user_summary"][()])
    return out


def gt_mask(score: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """(frame times i*T/N, top-15% bool mask) for per-frame importance ``score``."""
    n = len(score)
    return np.arange(n, dtype=float) * duration / n, top_frames(score, GT_TOP_FRAC)


def score_video(score: np.ndarray, duration: float, highlights: Sequence[Interval]) -> float:
    times, gt = gt_mask(score, duration)
    return frame_f1(times, gt, highlights)


def _duration(path: Path) -> float:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        if s.duration is not None:
            return float(s.duration * s.time_base)
        return c.duration / av.time_base


def _read_pred(path: Path) -> list[Interval]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [(float(h["segment"][0]), float(h["segment"][1])) for h in doc.get("highlights", [])]


def load_scores(dataset: str, anno: Path) -> dict[str, np.ndarray]:
    return load_tvsum(anno) if dataset == "tvsum" else load_summe(anno)


def main(argv: Sequence[str] | None = None) -> None:
    from rewind_eval.baselines import uniform_highlights

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=["tvsum", "summe"], required=True)
    ap.add_argument("--pred-dir", type=Path, required=True, help="REWIND outputs <vid>.json")
    ap.add_argument("--videos-dir", type=Path, required=True, help="<vid>.mp4 files")
    ap.add_argument("--anno", type=Path, required=True, help="TVSum anno .tsv / SumMe .h5")
    ap.add_argument("--budget", type=float, default=0.145, help="uniform baseline budget")
    args = ap.parse_args(argv)

    ours, base, skipped = [], [], {}
    for vid, score in sorted(load_scores(args.dataset, args.anno).items()):
        video, pred = args.videos_dir / f"{vid}.mp4", args.pred_dir / f"{vid}.json"
        reason = None
        if not video.exists():
            reason = "missing video"
        elif not pred.exists():
            reason = "missing pred"
        else:
            duration = _duration(video)
            if gt_rate_suspect(len(score), duration):
                reason = "frame count / duration fits no standard fps"
        if reason:
            skipped.setdefault(reason, []).append(vid)
            continue
        ours.append(score_video(score, duration, _read_pred(pred)))
        base.append(score_video(score, duration, uniform_highlights(duration, args.budget)))
    if ours:
        print(
            f"{args.dataset}: n={len(ours)} REWIND F1 {np.mean(ours):.4f} "
            f"uniform F1 {np.mean(base):.4f}"
        )
    for reason, vids in sorted(skipped.items()):
        print(f"  skipped ({reason}): {len(vids)} {vids[:5]}")


if __name__ == "__main__":
    main()
