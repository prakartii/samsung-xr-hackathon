"""Demo video: one MP4 that shows REWIND working end to end (P8, demo only).

    python tools/make_demo.py --pred results/preds/full_report --videos data/videos \
        --epic <epic_id> --fpvsum <fpvsum_id> [<fpvsum_id> ...] --out demo/rewind_demo.mp4

Story (~2.5 min at the defaults): title → the problem → what the program does (video
in, JSON out) → a kitchen video: the whole input fast-forwarded with our highlights
marked on its timeline, then our highlights played back with the detected actions
captioned live, then the JSON we wrote → sports videos the same way, with the predicted
category → how it works → results vs baselines and the three deck numbers → end.

Everything shown is read from our output JSONs, `results/ablation_report.md`,
`results/latency.csv` and the original MP4s; nothing is typed in by hand except the
narration text. Video only (no audio). PyAV + Pillow, no ffmpeg CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from make_reel import captions_at, reel_plan  # noqa: E402

W, H, FPS = 1280, 720, 30
HEADER, FOOTER = 64, 112
BG = (16, 18, 22)
INK = (236, 236, 232)
MUTED = (150, 152, 158)
ACCENT = (240, 140, 50)
ACT = (110, 165, 225)
TRACK = (52, 55, 62)

Segment = tuple[float, float]


# --- pure helpers -------------------------------------------------------------------


def pick_evenly(items: list, n: int) -> list:
    """``n`` items spread evenly over ``items`` (first and last kept), in order."""
    if n >= len(items):
        return list(items)
    if n <= 1:
        return list(items[:n])
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]


def demo_clips(highlights: list[Segment], duration: float, n: int, max_len: float) -> list[Segment]:
    """Up to ``n`` highlights (evenly spread), each cut to its first ``max_len`` s."""
    plan = reel_plan(highlights, duration, 0.0)
    return [(s, min(e, s + max_len)) for s, e in pick_evenly(plan, n)]


def fit_box(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int]:
    """Largest even (w, h) with the source aspect ratio that fits in the box."""
    scale = min(box_w / src_w, box_h / src_h)
    return max(2, int(src_w * scale) // 2 * 2), max(2, int(src_h * scale) // 2 * 2)


def sample_times(start: float, end: float, n: int) -> list[float]:
    """``n`` times evenly spaced over [start, end), starting at ``start``."""
    return [start + (end - start) * i / n for i in range(n)]


def parse_ablation(md: str) -> dict[str, dict[str, float]]:
    """Rows of the markdown table in results/ablation_<split>.md: variant -> metrics."""
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    out = {}
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        out[cells[0]] = {k: float(v) for k, v in zip(header[1:], cells[1:], strict=False) if v}
    return out


def clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# --- drawing ------------------------------------------------------------------------


def _font(size: int, bold: bool = False, mono: bool = False):
    from PIL import ImageFont

    names = (
        ["consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"]
        if mono
        else (["segoeuib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
              else ["segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    )  # fmt: skip
    dirs = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"),
            Path("/usr/share/fonts/truetype/liberation"), Path("/Library/Fonts")]  # fmt: skip
    for name in names:
        for d in dirs:
            if (d / name).exists():
                return ImageFont.truetype(str(d / name), size)
    return ImageFont.load_default(size=size)


@dataclass
class Fonts:
    def __post_init__(self) -> None:
        self.huge = _font(84, bold=True)
        self.title = _font(46, bold=True)
        self.big = _font(34)
        self.body = _font(26)
        self.small = _font(20)
        self.head = _font(24, bold=True)
        self.mono = _font(19, mono=True)


class Canvas:
    """A 1280x720 RGB frame with Pillow drawing."""

    def __init__(self) -> None:
        from PIL import Image, ImageDraw

        self.img = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.img)

    def text(self, xy, s, font, fill=INK, anchor="la") -> None:
        self.d.text(xy, s, font=font, fill=fill, anchor=anchor)

    def width(self, s, font) -> float:
        return self.d.textlength(s, font=font)

    def paste(self, arr: np.ndarray, x: int, y: int) -> None:
        from PIL import Image

        self.img.paste(Image.fromarray(arr), (x, y))

    def array(self) -> np.ndarray:
        return np.asarray(self.img)


def draw_timeline(c: Canvas, f: Fonts, duration: float, hls: list[Segment], t: float,
                  active: int | None) -> None:  # fmt: skip
    """Bottom bar: the whole video, our highlights in orange, the playhead."""
    x0, x1, y, h = 40, W - 40, H - 34, 14
    X = lambda s: x0 + (x1 - x0) * s / duration  # noqa: E731
    c.d.rounded_rectangle([x0, y, x1, y + h], 4, fill=TRACK)
    for i, (s, e) in enumerate(hls):
        col = (255, 196, 120) if i == active else ACCENT
        c.d.rectangle([X(s), y, max(X(s) + 2, X(e)), y + h], fill=col)
    px = X(t)
    c.d.polygon([(px - 7, y - 10), (px + 7, y - 10), (px, y - 1)], fill=INK)
    c.d.rectangle([px - 1, y - 2, px + 1, y + h + 2], fill=INK)
    c.text((x0, y - 8), "0:00", f.small, MUTED, "lb")
    c.text((x1, y - 8), clock(duration), f.small, MUTED, "rb")


def chips(c: Canvas, f: Fonts, words: list[str], y: int, color=ACT) -> None:
    x = 40
    for w in words:
        tw = c.width(w, f.body)
        c.d.rounded_rectangle([x, y, x + tw + 28, y + 40], 20, fill=color)
        c.text((x + 14, y + 20), w, f.body, (12, 14, 18), "lm")
        x += tw + 40


# --- video I/O ----------------------------------------------------------------------


def frames_at(path: Path, times: list[float], size: tuple[int, int]):
    """Yield an RGB array of the source frame at each time (ascending), resized to
    ``size``. Seeks for jumps over 3 s, otherwise decodes forward."""
    import av

    with av.open(str(path)) as cont:
        s = cont.streams.video[0]
        s.thread_type = "AUTO"
        it, prev, arr = None, None, None
        for t in times:
            if it is None or prev is None or t < prev.time - 1e-6 or t - prev.time > 3.0:
                cont.seek(max(0, int(t / s.time_base)), stream=s)
                it, prev = cont.decode(s), None
            moved = False
            while prev is None or prev.time < t - 1e-6:
                fr = next(it, None)
                if fr is None:
                    break
                if fr.time is not None:
                    prev, moved = fr, True
            if prev is None:
                continue
            if moved or arr is None:
                arr = prev.reformat(width=size[0], height=size[1], format="rgb24").to_ndarray()
            yield t, arr


def video_size(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as cont:
        cc = cont.streams.video[0].codec_context
        return cc.width, cc.height


class Writer:
    def __init__(self, out: Path) -> None:
        import av

        out.parent.mkdir(parents=True, exist_ok=True)
        self.cont = av.open(str(out), "w")
        self.s = self.cont.add_stream("libx264", rate=Fraction(FPS))
        self.s.width, self.s.height, self.s.pix_fmt = W, H, "yuv420p"
        self.s.options = {"crf": "20", "preset": "medium"}
        self.n = 0

    def put(self, arr: np.ndarray, repeat: int = 1) -> None:
        import av

        for _ in range(repeat):
            fr = av.VideoFrame.from_ndarray(arr, format="rgb24")
            fr.pts, fr.time_base = self.n, Fraction(1, FPS)
            self.n += 1
            for p in self.s.encode(fr):
                self.cont.mux(p)

    def close(self) -> None:
        for p in self.s.encode():
            self.cont.mux(p)
        self.cont.close()


def fade(arr: np.ndarray, i: int, n: int, k: int = 9) -> np.ndarray:
    g = min(1.0, (i + 1) / k, (n - i) / k)
    return arr if g >= 1.0 else (arr.astype(np.float32) * g).astype(np.uint8)


def hold(w: Writer, arr: np.ndarray, seconds: float) -> None:
    n = round(seconds * FPS)
    for i in range(n):
        w.put(fade(arr, i, n))


# --- scenes -------------------------------------------------------------------------


def card(f: Fonts, title: str, lines: list[str], kicker: str = "") -> np.ndarray:
    c = Canvas()
    y = 200 if lines else 300
    if kicker:
        c.text((90, y - 56), kicker.upper(), f.head, ACCENT)
    c.text((90, y), title, f.title)
    for i, ln in enumerate(lines):
        c.text((90, y + 90 + 50 * i), ln, f.big if not ln.startswith("  ") else f.body,
               INK if not ln.startswith("  ") else MUTED)  # fmt: skip
    return c.array()


def title_card(f: Fonts) -> np.ndarray:
    c = Canvas()
    c.text((W // 2, 280), "REWIND", f.huge, ACCENT, "mm")
    c.text((W // 2, 370), "the most distinct steps per second of highlight", f.big, INK, "mm")
    c.text((W // 2, 450), "Samsung XR Hackathon 2026  ·  Track 4: Action-Aware Video Summarization",
           f.small, MUTED, "mm")  # fmt: skip
    return c.array()


def json_card(f: Fonts, doc: dict, n_act: int = 4, n_hl: int = 3) -> np.ndarray:
    c = Canvas()
    c.text((90, 60), "What it wrote: " + f"{doc['video_id']}.json", f.head, ACCENT)
    acts = sorted(doc["actions"], key=lambda a: -a["score"])[:n_act]
    lines = [
        "{",
        f'  "video_id": "{doc["video_id"]}",',
        f'  "duration_sec": {doc["duration_sec"]},',
        f'  "label": {{"name": "{doc["label"]["name"]}", "score": {doc["label"]["score"]}}},',
        '  "actions": [',
    ]
    lines += [f'    {{"segment": [{a["segment"][0]:.2f}, {a["segment"][1]:.2f}], "verb": '
              f'"{a["verb"]}", "noun": "{a["noun"]}", "score": {a["score"]:.2f}}},'
              for a in acts]  # fmt: skip
    lines += [f"    ... {len(doc['actions']) - n_act} more actions", "  ],", '  "highlights": [']
    lines += [f'    {{"segment": [{h["segment"][0]:.2f}, {h["segment"][1]:.2f}]}},'
              for h in doc["highlights"][:n_hl]]  # fmt: skip
    lines += [f"    ... {len(doc['highlights']) - n_hl} more", "  ]", "}"]
    for i, ln in enumerate(lines):
        c.text((90, 120 + 28 * i), ln, f.mono, INK if "..." not in ln else MUTED)
    return c.array()


def video_scene(w: Writer, f: Fonts, path: Path, doc: dict, times: list[float],
                heading: str, sub: str, clips: list[Segment] | None,
                show_label: bool) -> None:  # fmt: skip
    """Frames at ``times`` with header, live action captions and the timeline."""
    duration = float(doc["duration_sec"])
    hls = [tuple(h["segment"]) for h in doc["highlights"]]
    box = fit_box(*video_size(path), W, H - HEADER - FOOTER)
    x0, y0 = (W - box[0]) // 2, HEADER + (H - HEADER - FOOTER - box[1]) // 2
    n = len(times)
    for i, (t, arr) in enumerate(frames_at(path, times, box)):
        c = Canvas()
        c.paste(arr, x0, y0)
        c.text((40, HEADER // 2), heading, f.head, ACCENT, "lm")
        c.text((40 + c.width(heading, f.head) + 18, HEADER // 2), sub, f.body, MUTED, "lm")
        c.text((W - 40, HEADER // 2), f"{clock(t)} / {clock(duration)}", f.head, INK, "rm")
        active = next((k for k, (s, e) in enumerate(hls) if s <= t < e), None)
        if clips is not None:
            k = next((k for k, (s, e) in enumerate(clips) if s <= t < e), None)
            tag = f"highlight {k + 1}/{len(clips)} of {len(hls)}" if k is not None else ""
            c.text((W - 40, H - FOOTER + 26), tag, f.small, MUTED, "rm")
        caps = captions_at(t, doc.get("actions", []), 3)
        if show_label and not caps:
            caps = [f"category: {doc['label']['name']}"]
            chips(c, f, caps, H - FOOTER + 6, ACCENT)
        else:
            chips(c, f, caps, H - FOOTER + 6)
        draw_timeline(c, f, duration, hls, t, active)
        w.put(fade(c.array(), i, n))


def example(w: Writer, f: Fonts, path: Path, doc: dict, args, kind: str) -> None:
    duration = float(doc["duration_sec"])
    hls = [tuple(h["segment"]) for h in doc["highlights"]]
    kept = sum(e - s for s, e in hls)
    is_epic = kind == "kitchen"
    hold(w, card(f, f"Example: {kind} video", [
        f"{clock(duration)} of first-person video in.",
        f"REWIND keeps {len(hls)} windows, {clock(kept)} in total ({100 * kept / duration:.0f}%).",
        (f"  {len(doc['actions'])} actions detected (verb + noun, with times)" if is_epic
         else f"  predicted category: {doc['label']['name']} ({100 * doc['label']['score']:.0f}%)"),
    ], kicker="input"), 4.5)  # fmt: skip
    # 1) the input, fast-forwarded, highlights marked on the timeline
    ff = sample_times(0.0, duration, round(args.ff_seconds * FPS / 2))
    ff2 = [t for t in ff for _ in range(2)]
    video_scene(w, f, path, doc, ff2, "INPUT", f"whole video at {duration / args.ff_seconds:.0f}x",
                None, show_label=False)  # fmt: skip
    # 2) the highlights at normal speed
    clips = demo_clips(hls, duration, args.clips_epic if is_epic else args.clips_fpv, args.clip_len)
    for s, e in clips:
        video_scene(w, f, path, doc, sample_times(s, e, round((e - s) * FPS)), "REWIND",
                    "highlights at normal speed", clips, show_label=not is_epic)  # fmt: skip
    if is_epic:
        hold(w, json_card(f, doc), 6.0)


def how_card(f: Fonts) -> np.ndarray:
    return card(f, "How it works", [
        "1  Signals: motion, hands, CLIP frame embeddings, audio",
        "2  Route: kitchen (EPIC) or outdoor activity (FPVSum)",
        "3  Actions: ActionFormer verb + noun detections, paired",
        "4  Cluster actions into distinct steps",
        "5  Pick windows by expected new steps per second (15% budget)",
        "  CPU only. Same video in → byte-identical JSON out.",
    ], kicker="pipeline")  # fmt: skip


def results_card(f: Fonts, abl: dict, tokens: float, latency: float) -> np.ndarray:
    c = Canvas()
    c.text((90, 70), "RESULTS", f.head, ACCENT)
    c.text((90, 100), "185 scored videos (138 EPIC-KITCHENS + 47 FPVSum), official metrics", f.body,
           MUTED)  # fmt: skip
    cols = [("weighted", "Score"), ("coverage_frac", "Coverage 40%"), ("map", "mAP 25%"),
            ("f1", "F1 25%"), ("label_acc", "Label 10%")]  # fmt: skip
    rows = [("REWIND", "full pipeline"), ("Uniform windows", "uniform baseline"),
            ("Oracle (sees GT)", "oracle baseline")]  # fmt: skip
    xs = [90, 420, 600, 760, 900, 1050]
    for x, (_, name) in zip(xs[1:], cols, strict=True):
        c.text((x, 170), name, f.small, MUTED)
    for r, (name, key) in enumerate(rows):
        y = 210 + 50 * r
        col = ACCENT if r == 0 else INK
        c.text((xs[0], y), name, f.big if r == 0 else f.body, col)
        for x, (k, _) in zip(xs[1:], cols, strict=True):
            c.text((x, y), f"{abl[key][k]:.3f}", f.big if r == 0 else f.body, col)
    nums = [(f"{100 * abl['full pipeline']['pct_oracle']:.0f}%", "of oracle action coverage"),
            (f"{100 * tokens:.0f}%", "of video frames pruned"),
            (f"{latency:.0f} s", "CPU time per video minute")]  # fmt: skip
    for i, (big, small) in enumerate(nums):
        x = 90 + 380 * i
        c.text((x, 470), big, f.huge, ACCENT)
        c.text((x, 570), small, f.body, INK)
    return c.array()


def main() -> None:
    import csv

    from deck_numbers import sec_per_video_minute, tokens_pruned

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred", type=Path, default=REPO / "results" / "preds" / "full_report")
    ap.add_argument("--videos", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--epic", required=True, help="kitchen video id")
    ap.add_argument("--fpvsum", nargs="+", required=True, help="outdoor video id(s)")
    ap.add_argument("--out", type=Path, default=REPO / "demo" / "rewind_demo.mp4")
    ap.add_argument("--ff-seconds", type=float, default=8.0, help="fast-forward length per video")
    ap.add_argument("--clips-epic", type=int, default=8, help="highlights shown, kitchen")
    ap.add_argument("--clips-fpv", type=int, default=4, help="highlights shown per outdoor video")
    ap.add_argument("--clip-len", type=float, default=4.5, help="max seconds shown per highlight")
    args = ap.parse_args()

    abl = parse_ablation((REPO / "results" / "ablation_report.md").read_text(encoding="utf-8"))
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.pred.glob("*.json"))]
    with (REPO / "results" / "latency.csv").open(newline="", encoding="utf-8") as fh:
        latency = sec_per_video_minute(list(csv.DictReader(fh)))
    tokens = tokens_pruned(docs)

    f = Fonts()
    w = Writer(args.out)
    hold(w, title_card(f), 4.0)
    hold(w, card(f, "First-person video is long, and mostly idle.", [
        "Smart glasses record minutes of footage for every few seconds that matter.",
        "Which 15% should you keep, so that every distinct step is still there?",
    ], kicker="the problem"), 6.0)  # fmt: skip
    hold(w, card(f, "./run.sh video.mp4 out.json", [
        "One command per video, CPU only. It writes one JSON with:",
        "  actions: every detected action (verb, noun, start, end, score)",
        "  highlights: >= 4 s windows of the original footage, <= 15% in total",
        "  label: what kind of video it is",
    ], kicker="what REWIND does"), 6.5)  # fmt: skip
    for vid, kind in [(args.epic, "kitchen")] + [(v, "sport") for v in args.fpvsum]:
        doc = json.loads((args.pred / f"{vid}.json").read_text(encoding="utf-8"))
        example(w, f, args.videos / f"{vid}.mp4", doc, args, kind)
        print(f"{vid}: done ({w.n / FPS:.0f} s so far)", flush=True)
    hold(w, how_card(f), 8.0)
    hold(w, results_card(f, abl, tokens, latency), 9.0)
    end = Canvas()
    end.text((W // 2, 320), "REWIND", f.huge, ACCENT, "mm")
    end.text((W // 2, 410), "every distinct step, in 15% of the video", f.big, INK, "mm")
    hold(w, end.array(), 4.0)
    w.close()
    print(f"wrote {args.out}: {w.n / FPS:.1f} s")


if __name__ == "__main__":
    main()
