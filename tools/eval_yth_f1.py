"""Highlight F1 of REWIND outputs on the YouTube Highlights dataset (Sun et al., ECCV 2014).

    PYTHONPATH=code python tools/eval_yth_f1.py --pred-dir outputs/yth \
        --videos-dir data/external/yth_videos --labels-dir data/external/yth

FPVSum's highlight labels are hidden and never used; YTH is the only legitimate
first-person highlight label set we have, so it is our offline proxy for the
Highlight F1 metric.

GT: ``<labels-dir>/<domain>/<vid>/match_label.json`` = ``[clips, labels]``. ``clips``
are ``[start_frame, end_frame]`` in the frame indices of the original video (~100-frame
clips with ~50-frame stride, so they overlap); a label of 1 means the clip was
*matched* in the owner's edited video. -1 (unmatched) and 0 (borderline) are both
negative here. GT positives = union of the matched clips, converted with the video's
fps.

Difference from ``rewind_eval.f1.frame_f1`` (FPVSum): there the samples are the GT's
native frames and the GT positives are the top-15% importance frames. YTH has no
per-frame importance, so we sample a uniform time grid (``step`` s) over
``[0, duration)`` and a sample is GT-positive if it falls inside a matched interval.
Predicted positives use the very same ``highlight_mask`` (highlights >= 4.0 s, end
exclusive). Note YTH positives are not capped at 15%: some videos are mostly matched,
which bounds recall for a 15% budget.

Videos are skipped (and counted) when the prediction or the MP4 is missing, when the
label span (last clip end / fps) disagrees with the MP4 duration (an incomplete
download or a re-encode at a different frame rate would misplace the labels), or when
no clip is matched. Reads container metadata only (PyAV), never decodes frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "code")]

from rewind_eval.f1 import highlight_mask  # noqa: E402

UNIFORM_BUDGET = 0.145
#: Max |label span − video duration| before a video is treated as mismatched (s, frac).
SPAN_TOL_SEC = 2.0
SPAN_TOL_FRAC = 0.05

Interval = tuple[float, float]


def positive_intervals(
    clips: Sequence[Sequence[float]], labels: Sequence[float], fps: float
) -> list[Interval]:
    """Union of the clips labelled 1, in seconds (frame / fps), overlaps merged."""
    if len(clips) != len(labels):
        raise ValueError(f"{len(clips)} clips but {len(labels)} labels")
    pos = sorted((s / fps, e / fps) for (s, e), lab in zip(clips, labels, strict=True) if lab == 1)
    merged: list[list[float]] = []
    for s, e in pos:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def frame_f1(
    pred: Sequence[Interval], gt: Sequence[Interval], duration: float, step: float = 0.1
) -> float:
    """F1 on a ``step`` s time grid over [0, duration); 0 if no true positive.

    Predicted positive: inside a highlight >= 4.0 s (``highlight_mask``, end exclusive).
    GT positive: inside a positive interval (end exclusive).
    """
    times = np.arange(0.0, duration - 1e-9, step)
    pred_mask = highlight_mask(times, pred)
    gt_mask = np.zeros(len(times), dtype=bool)
    for s, e in gt:
        gt_mask |= (times >= s) & (times < e)
    tp = int(np.sum(pred_mask & gt_mask))
    if tp == 0:
        return 0.0
    precision = tp / int(np.sum(pred_mask))
    recall = tp / int(np.sum(gt_mask))
    return 2 * precision * recall / (precision + recall)


def span_matches(last_end_frame: float, fps: float, duration: float) -> bool:
    """Whether the labels' span (last clip end / fps) agrees with the video duration."""
    gap = abs(last_end_frame / fps - duration)
    return gap <= max(SPAN_TOL_SEC, SPAN_TOL_FRAC * duration)


def _video_meta(path: Path) -> tuple[float, float]:
    """(fps, duration s) from container metadata; no frames are decoded."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = container.duration / av.time_base
    return fps, duration


def _read_pred(path: Path) -> list[Interval]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [(float(h["segment"][0]), float(h["segment"][1])) for h in doc.get("highlights", [])]


def _list_videos(labels_dir: Path) -> list[tuple[str, str]]:
    return sorted(
        (p.parent.parent.name, p.parent.name) for p in labels_dir.glob("*/*/match_label.json")
    )


def main(argv: Sequence[str] | None = None) -> None:
    from rewind_eval.baselines import uniform_highlights

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True, help="REWIND outputs <vid>.json")
    ap.add_argument("--videos-dir", type=Path, required=True, help="<vid>.mp4 files")
    ap.add_argument("--labels-dir", type=Path, required=True, help="YTH repo root")
    ap.add_argument("--step", type=float, default=0.1, help="time grid step (s)")
    args = ap.parse_args(argv)

    ours: dict[str, list[float]] = defaultdict(list)
    base: dict[str, list[float]] = defaultdict(list)
    skipped: dict[str, list[str]] = defaultdict(list)
    for domain, vid in _list_videos(args.labels_dir):
        pred_path = args.pred_dir / f"{vid}.json"
        video_path = args.videos_dir / f"{vid}.mp4"
        if not video_path.exists():
            skipped["missing video"].append(vid)
            continue
        if not pred_path.exists():
            skipped["missing pred"].append(vid)
            continue
        label_path = args.labels_dir / domain / vid / "match_label.json"
        clips, labels = json.loads(label_path.read_text(encoding="utf-8"))
        try:
            fps, duration = _video_meta(video_path)
        except Exception as exc:
            print(f"  {vid}: cannot read metadata ({exc})", file=sys.stderr)
            skipped["unreadable video"].append(vid)
            continue
        if not clips or not span_matches(clips[-1][1], fps, duration):
            skipped["label/video span mismatch"].append(vid)
            continue
        gt = positive_intervals(clips, labels, fps)
        if not gt:
            skipped["no matched clip"].append(vid)
            continue
        ours[domain].append(frame_f1(_read_pred(pred_path), gt, duration, args.step))
        base[domain].append(
            frame_f1(uniform_highlights(duration, UNIFORM_BUDGET), gt, duration, args.step)
        )

    print(f"{'domain':<12} {'n':>4} {'REWIND F1':>10} {'uniform F1':>11}")
    for domain in sorted(ours):
        print(
            f"{domain:<12} {len(ours[domain]):>4} {np.mean(ours[domain]):>10.4f} "
            f"{np.mean(base[domain]):>11.4f}"
        )
    all_ours = [f for v in ours.values() for f in v]
    all_base = [f for v in base.values() for f in v]
    if all_ours:
        print(
            f"{'overall':<12} {len(all_ours):>4} {np.mean(all_ours):>10.4f} "
            f"{np.mean(all_base):>11.4f}"
        )
    print(f"videos used: {len(all_ours)}; skipped: {sum(len(v) for v in skipped.values())}")
    for reason, vids in sorted(skipped.items()):
        print(f"  {reason}: {len(vids)}")


if __name__ == "__main__":
    main()
