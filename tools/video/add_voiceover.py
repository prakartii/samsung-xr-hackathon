# ruff: noqa: E501
"""Add a neural voiceover to demo/rewind_demo.mp4 (video stream copied, audio re-mixed).

    python tools/video/add_voiceover.py --tts <path to edge-tts.exe>

Voice: Microsoft neural TTS via the `edge-tts` CLI (only the narration text below is sent).
Lines are placed at the scene starts of tools/video/make_final_demo.py; the five action
phrases land when each predicted label appears in the montage. Music is ducked under speech.
Writes demo/rewind_demo_voiceover.mp4; the captions-only cut stays in demo/rewind_demo.mp4.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEMO = Path(__file__).resolve().parents[2] / "demo"
import make_final_demo as R  # noqa: E402

SR = 48000
VOICE = "en-US-AndrewNeural"

LINES = {  # scene name -> (offset s, text)
    "Hook": (
        0.8,
        "Head-mounted cameras record everything. But only a few moments really matter. This is REWIND.",
    ),
    "Problem": (
        0.6,
        "Here's over nine minutes in a real kitchen. Finding what actually happened means watching all of it.",
    ),
    "InOut": (
        0.6,
        "REWIND takes one video and writes one file: every detected action with its time and confidence, the highlights to keep, and a label.",
    ),
    "Pipeline": (
        0.8,
        "First, CLIP decides what kind of video it is. In kitchens, an action detector finds verbs and nouns in time. We score which actions are real, and keep the windows that cover the most distinct ones. Sports footage is ranked by motion instead.",
    ),
    "Summary": (
        0.6,
        "These are the fourteen windows REWIND kept. Nine minutes of video becomes one minute twenty-one, and that minute is where the actions are.",
    ),
    "Sports": (
        0.5,
        "The same system works on outdoor first-person video: skiing, motocross, riding, diving, climbing and surfing, each recognised and cut to its strongest moments.",
    ),
    "JsonView": (
        0.6,
        "Everything lands in one plain JSON file per video: timestamped actions, highlight segments, and a label.",
    ),
    "Results": (
        0.6,
        "On the official metrics: thirty-six percent action coverage, fifteen percent action mAP, twenty-three percent highlight F1, and eighty-five percent label accuracy.",
    ),
    "End": (0.4, "REWIND. Turn long recordings into action-aware summaries."),
}
MONTAGE_SAY = [
    "Turn on the tap.",
    "Take the sponge.",
    "Pour washing liquid.",
    "Wash the knife.",
    "Open the bin.",
]


def decode(out: Path) -> np.ndarray:
    c = av.open(str(out))
    rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
    chunks = [f2.to_ndarray()[0] for f in c.decode(audio=0) for f2 in rs.resample(f)]
    c.close()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def tts(exe: str, text: str, out: Path, rate: int) -> np.ndarray:
    for _attempt in range(4):  # the TTS service occasionally drops a request
        try:
            r = subprocess.run(
                [
                    exe,
                    "--voice",
                    VOICE,
                    f"--rate={rate:+d}%",
                    "--text",
                    text,
                    "--write-media",
                    str(out),
                ],
                capture_output=True,
                timeout=40,
            )
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0:
            break
    else:
        raise RuntimeError(r.stderr.decode(errors="replace")[-300:])
    c = av.open(str(out))
    rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
    chunks = [f2.to_ndarray()[0] for f in c.decode(audio=0) for f2 in rs.resample(f)]
    c.close()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def fit(exe: str, text: str, out: Path, room: float) -> np.ndarray:
    if out.exists() and out.stat().st_size > 0:  # reuse a clip generated earlier
        a = decode(out)
        if 0 < len(a) / SR <= room + 0.3:
            return a
    a = np.zeros(0, np.float32)
    for rate in (0, 6, 12, 18, 25):
        for variant in (text, text.rstrip("."), "Now: " + text):
            try:
                a = tts(exe, variant, out, rate)
                break
            except RuntimeError:
                continue
        if len(a) and len(a) / SR <= room:
            return a
    print(f"  ! still long ({len(a) / SR:.1f} s > {room:.1f} s): {text[:40]}")
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", required=True)
    ap.add_argument("--src", type=Path, default=DEMO / "rewind_demo.mp4")
    ap.add_argument("--out", type=Path, default=DEMO / "rewind_demo_voiceover.mp4")
    ap.add_argument("--work", type=Path, default=DEMO / "rewind_assets" / "vo")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    scenes = [
        R.Hook(),
        R.Problem(),
        R.InOut(),
        R.Pipeline(),
        R.Montage(),
        R.Summary(),
        R.Sports(),
        R.JsonView(),
        R.Results(),
        R.End(),
    ]
    starts = np.cumsum([0] + [s.dur for s in scenes])
    total = float(starts[-1])
    speech = np.zeros(int(total * SR) + SR, np.float32)
    placed = []
    for i, sc in enumerate(scenes):
        name = type(sc).__name__
        if name == "Montage":
            for k, phrase in enumerate(MONTAGE_SAY):
                t0 = starts[i] + sc.starts[k] + sc.PRE
                room = (
                    (sc.starts[k + 1] if k + 1 < len(sc.starts) else sc.dur) - sc.starts[k] - sc.PRE
                )
                a = fit(args.tts, phrase, args.work / f"m{k}.mp3", room)
                placed.append((t0, a, phrase))
            continue
        off, text = LINES[name]
        a = fit(args.tts, text, args.work / f"{name}.mp3", sc.dur - off - 0.3)
        placed.append((starts[i] + off, a, text))
    for t0, a, txt in placed:
        j = int(t0 * SR)
        speech[j : j + len(a)] += a[: len(speech) - j]
        print(f"  {t0:6.2f}-{t0 + len(a) / SR:6.2f} s  {txt[:60]}")

    ic = av.open(str(args.src))
    rs = av.AudioResampler(format="fltp", layout="stereo", rate=SR)
    music = np.concatenate(
        [f2.to_ndarray() for f in ic.decode(audio=0) for f2 in rs.resample(f)], axis=1
    )
    ic.close()
    n = music.shape[1]
    sp = speech[:n] if len(speech) >= n else np.pad(speech, (0, n - len(speech)))
    act = np.convolve(
        (np.abs(sp) > 0.01).astype(np.float32), np.ones(SR // 4) / (SR // 4), mode="same"
    )
    duck = 1 - 0.65 * np.clip(act * 3, 0, 1)
    mix = music * duck + 0.95 * sp[None, :]
    mix = (mix / max(1e-6, np.abs(mix).max()) * 0.92).astype(np.float32)

    ic = av.open(str(args.src))
    tmp = args.out.with_name(args.out.stem + ".partial.mp4")
    oc = av.open(str(tmp), "w", format="mp4")
    iv = ic.streams.video[0]
    ov = oc.add_stream("libx264", rate=30)
    ov.width, ov.height, ov.pix_fmt = iv.width, iv.height, "yuv420p"
    ov.options = {"crf": "18", "preset": "medium"}
    oa = oc.add_stream("aac", rate=SR)
    oa.layout = "stereo"
    for f in ic.decode(iv):
        nf = av.VideoFrame.from_ndarray(f.to_ndarray(format="rgb24"), format="rgb24")
        for p in ov.encode(nf):
            oc.mux(p)
    for p in ov.encode():
        oc.mux(p)
    for p0 in range(0, n, 1024):
        seg = np.ascontiguousarray(mix[:, p0 : p0 + 1024])
        af = av.AudioFrame.from_ndarray(seg, format="fltp", layout="stereo")
        af.sample_rate, af.pts, af.time_base = SR, p0, Fraction(1, SR)
        for p in oa.encode(af):
            oc.mux(p)
    for p in oa.encode():
        oc.mux(p)
    oc.close()
    ic.close()
    tmp.replace(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
