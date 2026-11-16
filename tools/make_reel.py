"""Demo reel: cut a video's highlights from our JSON into one captioned MP4 (P8).

    python tools/make_reel.py --video data/videos/P01_11.mp4 --json outputs/P01_11.json \
        --out demo/P01_11_reel.mp4

For the demo video only, never for outputs/: each highlight is padded by --pad
seconds (merged where padded clips overlap) and fades in and out over --fade
seconds. Each frame is captioned with the clip's place in the source ("12:04 / 21:30")
and the top actions active at that moment from the JSON. Video only (no audio).
Uses PyAV (the FFmpeg libraries the pipeline decodes with), so no ffmpeg CLI is needed.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

Segment = tuple[float, float]


def reel_plan(highlights: list[Segment], duration: float, pad: float) -> list[Segment]:
    """Highlights padded by ``pad`` on both sides, clipped to [0, duration], merged."""
    out: list[list[float]] = []
    for s, e in sorted(highlights):
        s, e = max(0.0, s - pad), min(duration, e + pad)
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def captions_at(t: float, actions: list[dict], top_k: int) -> list[str]:
    """ "verb noun" of the actions whose segment contains t (start <= t < end), highest
    score first, each phrase once, at most ``top_k``."""
    active = sorted(
        (a for a in actions if a["segment"][0] <= t < a["segment"][1]),
        key=lambda a: -a["score"],
    )
    out: list[str] = []
    for a in active:
        phrase = f"{a['verb']} {a['noun']}"
        if phrase not in out:
            out.append(phrase)
        if len(out) == top_k:
            break
    return out


def fade_gain(t: float, clip_len: float, fade: float) -> float:
    """Brightness factor at time t into a clip: linear 0 → 1 over the first ``fade``
    seconds and 1 → 0 over the last ``fade``."""
    if fade <= 0:
        return 1.0
    return float(min(1.0, t / fade, (clip_len - t) / fade))


def _clock(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _draw(frame: np.ndarray, lines: list[tuple[str, float]], cv2) -> None:
    """White text with a dark outline; lines are (text, y as a fraction of height)."""
    h = frame.shape[0]
    scale = h / 720
    for text, y in lines:
        org = (int(24 * scale), int(y * h))
        for color, thick in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, color,
                        max(1, round(thick * scale)), cv2.LINE_AA)  # fmt: skip


def main() -> None:
    import av
    import cv2

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pad", type=float, default=0.5, help="seconds added around each clip")
    ap.add_argument("--fade", type=float, default=0.3, help="fade in/out per clip, seconds")
    ap.add_argument("--fps", type=float, default=30.0, help="output frame rate")
    ap.add_argument("--width", type=int, default=1280, help="output width (height keeps aspect)")
    ap.add_argument("--captions", type=int, default=2, help="actions shown per frame")
    args = ap.parse_args()

    doc = json.loads(args.json.read_text(encoding="utf-8"))
    duration = float(doc["duration_sec"])
    plan = reel_plan([tuple(h["segment"]) for h in doc["highlights"]], duration, args.pad)
    actions, label = doc.get("actions", []), doc["label"]["name"]
    total = sum(e - s for s, e in plan)
    print(f"{len(plan)} clips, {total:.1f} s of {duration:.1f} s ({100 * total / duration:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rate = Fraction(args.fps).limit_denominator(1001)
    with av.open(str(args.video)) as src, av.open(str(args.out), "w") as dst:
        sv = src.streams.video[0]
        sv.thread_type = "AUTO"
        w = args.width - args.width % 2
        h = round(sv.codec_context.height * w / sv.codec_context.width / 2) * 2
        ov = dst.add_stream("libx264", rate=rate)
        ov.width, ov.height, ov.pix_fmt = w, h, "yuv420p"
        ov.options = {"crf": "20", "preset": "medium"}
        clip_i, next_t, written = 0, plan[0][0] if plan else duration, 0
        for frame in src.decode(sv):
            if clip_i >= len(plan) or frame.time is None:
                break
            t = float(frame.time)
            while clip_i < len(plan) and t >= plan[clip_i][1]:  # past this clip
                clip_i += 1
                next_t = plan[clip_i][0] if clip_i < len(plan) else duration
            if clip_i >= len(plan) or t < next_t:
                continue
            s, e = plan[clip_i]
            next_t += 1.0 / args.fps  # resample to the output rate by source time
            img = cv2.resize(frame.to_ndarray(format="bgr24"), (w, h), interpolation=cv2.INTER_AREA)
            gain = fade_gain(t - s, e - s, args.fade)
            lines = [(f"REWIND  {label}  |  {_clock(t)} / {_clock(duration)}"
                      f"  |  clip {clip_i + 1}/{len(plan)}", 0.07)]  # fmt: skip
            caps = captions_at(t, actions, args.captions)
            lines += [(c, 0.95 - 0.07 * (len(caps) - 1 - i)) for i, c in enumerate(caps)]
            _draw(img, lines, cv2)
            if gain < 1.0:
                img = (img.astype(np.float32) * gain).astype(np.uint8)
            out = av.VideoFrame.from_ndarray(img, format="bgr24")
            out.pts, out.time_base = written, 1 / rate
            written += 1
            for packet in ov.encode(out):
                dst.mux(packet)
        for packet in ov.encode():
            dst.mux(packet)
    print(f"wrote {args.out}: {written} frames, {written / args.fps:.1f} s")


if __name__ == "__main__":
    main()
