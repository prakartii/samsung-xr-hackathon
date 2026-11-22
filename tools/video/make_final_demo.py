# ruff: noqa: E501
"""REWIND final demo video (demo tooling only; never touches inference or outputs).

    python tools/video/make_final_demo.py [--out demo/rewind_demo.mp4] [--preview SECS]

Every label, timestamp, score, highlight window and duration shown is read at render time
from the frozen outputs (outputs/<video_id>.json, commit ea1e5d8) and the original MP4s in
data/videos. The only typed-in numbers are the five final benchmark results (report split,
measured once at the final freeze; README / results/ablation_report.md). The sound bed is
synthesised here (original, no third-party music). 1920x1080, 30 fps, H.264 + AAC.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs"
VID = REPO / "data" / "videos"
W, H, FPS = 1920, 1080, 30
SR = 48000

BG = (8, 9, 12)
INK = (242, 241, 236)
MUTED = (138, 142, 150)
DIM = (70, 74, 82)
AMBER = (139, 146, 255)  # accent: indigo/violet
BLUE = (103, 222, 245)
GREEN = (120, 210, 150)

FONTS = Path("C:/Windows/Fonts")


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / name), size)


F = {
    "hero": font("segoeuib.ttf", 150),
    "h1": font("segoeuisl.ttf", 76),
    "h1b": font("segoeuib.ttf", 76),
    "h2": font("segoeuisl.ttf", 52),
    "h2b": font("segoeuib.ttf", 52),
    "h3": font("segoeui.ttf", 36),
    "h3b": font("segoeuib.ttf", 36),
    "body": font("segoeui.ttf", 30),
    "small": font("segoeui.ttf", 24),
    "kicker": font("segoeuib.ttf", 24),
    "num": font("bahnschrift.ttf", 96),
    "numm": font("bahnschrift.ttf", 60),
    "mono": font("consola.ttf", 30),
    "monos": font("consola.ttf", 24),
    "verb": font("segoeuib.ttf", 92),
    "noun": font("segoeuisl.ttf", 54),
}


# ---------------------------------------------------------------- helpers
def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ramp(t: float, a: float, b: float) -> float:
    return ease((t - a) / (b - a)) if b > a else float(t >= a)


def clock(t: float, ms: bool = False) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:04.1f}" if ms else f"{int(m)}:{int(s):02d}"


def noun_txt(n: str) -> str:
    """EPIC nouns are 'head:modifier' (e.g. liquid:washing) -> 'washing liquid'."""
    if ":" in n:
        a, b = n.split(":", 1)
        return f"{b} {a}"
    return n


def text(
    img: Image.Image,
    xy,
    s: str,
    f,
    fill=INK,
    alpha: float = 1.0,
    anchor: str = "la",
    shadow: bool = False,
):
    if alpha <= 0.01 or not s:
        return
    d = ImageDraw.Draw(img, "RGBA")
    if shadow:
        for dx, dy in ((2, 2), (0, 3), (3, 0)):
            d.text(
                (xy[0] + dx, xy[1] + dy),
                s,
                font=f,
                fill=(0, 0, 0, int(170 * min(1.0, alpha))),
                anchor=anchor,
            )
    d.text(xy, s, font=f, fill=(*fill, int(255 * min(1.0, alpha))), anchor=anchor)


def rect(img, box, fill, alpha=1.0, r=0):
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(img, "RGBA")
    col = (*fill, int(255 * min(1.0, alpha)))
    if r:
        d.rounded_rectangle(box, r, fill=col)
    else:
        d.rectangle(box, fill=col)


def fit_cover(im: Image.Image, w: int, h: int) -> Image.Image:
    sw, sh = im.size
    s = max(w / sw, h / sh)
    im = im.resize((max(w, int(sw * s + 0.5)), max(h, int(sh * s + 0.5))), Image.LANCZOS)
    x, y = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x, y, x + w, y + h))


def darken(im: Image.Image, k: float) -> Image.Image:
    a = np.asarray(im, dtype=np.float32) * k
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


_GRAD = None


def bottom_grad(im: Image.Image, strength: float = 0.85) -> Image.Image:
    global _GRAD
    if _GRAD is None:
        y = np.linspace(0, 1, H, dtype=np.float32)
        g = np.clip((y - 0.45) / 0.55, 0, 1) ** 1.6
        top = np.clip((0.18 - y) / 0.18, 0, 1) ** 1.5 * 0.6
        _GRAD = (1 - np.maximum(g, top))[:, None, None]
    a = np.asarray(im, dtype=np.float32)
    k = 1 - strength * (1 - _GRAD)
    return Image.fromarray((a * k).clip(0, 255).astype(np.uint8))


_BGIMG = None


def _bg() -> Image.Image:
    """Near-black with a soft indigo glow at the top and a faint grid fading out (Linear-like)."""
    global _BGIMG
    if _BGIMG is None:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        base = np.zeros((H, W, 3), np.float32) + np.array(BG, np.float32)
        g1 = np.exp(-(((xx - W * 0.5) / (W * 0.42)) ** 2 + ((yy + H * 0.12) / (H * 0.55)) ** 2))
        g2 = np.exp(-(((xx - W * 0.9) / (W * 0.3)) ** 2 + ((yy - H * 1.05) / (H * 0.45)) ** 2))
        base += g1[..., None] * np.array([46, 40, 110], np.float32) * 0.55
        base += g2[..., None] * np.array([20, 60, 90], np.float32) * 0.35
        grid = ((xx % 80) < 1) | ((yy % 80) < 1)
        fade = np.clip(1 - np.hypot((xx - W / 2) / (W * 0.6), (yy - H * 0.3) / (H * 0.8)), 0, 1)
        base += (grid * fade * 7.0)[..., None]
        _BGIMG = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    return _BGIMG


def canvas(c=BG) -> Image.Image:
    return _bg().copy()


def card(img, box, alpha=1.0, r=16, accent=None):
    """Glass card: faint white fill, 1 px hairline border, optional accent top line."""
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle(
        box,
        r,
        fill=(255, 255, 255, int(10 * alpha)),
        outline=(255, 255, 255, int(34 * alpha)),
        width=1,
    )
    if accent:
        x0, y0, x1, _ = box
        d.line([(x0 + r, y0), (x1 - r, y0)], fill=(*accent, int(220 * alpha)), width=2)


def glass(img, box, alpha=1.0, r=22):
    """Frosted panel over footage: blurred + darkened backdrop, hairline border."""
    if alpha <= 0.01:
        return
    x0, y0, x1, y1 = [int(v) for v in box]
    reg = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(22))
    reg = darken(reg, 0.55)
    m = Image.new("L", reg.size, 0)
    ImageDraw.Draw(m).rounded_rectangle(
        (0, 0, reg.size[0] - 1, reg.size[1] - 1), r, fill=int(255 * alpha)
    )
    img.paste(reg, (x0, y0), m)
    ImageDraw.Draw(img, "RGBA").rounded_rectangle(
        (x0, y0, x1, y1), r, outline=(255, 255, 255, int(46 * alpha)), width=1
    )


def grad_text(img, xy, s, f, alpha=1.0, anchor="mm", top=(255, 255, 255), bot=(150, 156, 175)):
    """Text filled with a vertical white-to-grey gradient (hero titles)."""
    if alpha <= 0.01:
        return
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).text(xy, s, font=f, fill=int(255 * alpha), anchor=anchor)
    bb = m.getbbox()
    if not bb:
        return
    h = bb[3] - bb[1]
    col = np.zeros((h, 1, 3), np.float32)
    k = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    col[:, 0, :] = np.array(top) * (1 - k) + np.array(bot) * k
    grad = Image.fromarray(np.repeat(col, bb[2] - bb[0], axis=1).astype(np.uint8))
    img.paste(grad, bb[:2], m.crop(bb))


# ---------------------------------------------------------------- video access
class Source:
    """Sequential frame access at arbitrary times (seek once per contiguous request)."""

    def __init__(self, vid: str):
        self.path = VID / f"{vid}.mp4"
        self.c = av.open(str(self.path))
        self.s = self.c.streams.video[0]
        self.s.thread_type = "AUTO"
        self._it = None
        self._last = None

    def frame_at(self, t: float) -> Image.Image:
        if self._last is None or t < self._last[0] - 1e-3 or t > self._last[0] + 2.0:
            self.c.seek(max(0, int((t - 0.5) / self.s.time_base)), stream=self.s)
            self._it = self.c.decode(self.s)
            self._last = None
        while self._last is None or self._last[0] + 1e-3 < t:
            try:
                f = next(self._it)
            except StopIteration:
                break
            self._last = (float(f.time), f)
            if f.time >= t - 1e-3:
                break
        return self._last[1].to_image()


_SRC: dict[str, Source] = {}


def src(vid: str) -> Source:
    if vid not in _SRC:
        _SRC[vid] = Source(vid)
    return _SRC[vid]


def doc(vid: str) -> dict:
    return json.loads((OUT / f"{vid}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- data (all from outputs/)
EPIC = "P01_11"
D = doc(EPIC)
T_EPIC = float(D["duration_sec"])
HLS = [tuple(h["segment"]) for h in D["highlights"]]
HL_TOTAL = sum(b - a for a, b in HLS)


def find_action(verb: str, noun: str, near: float) -> dict:
    cands = [a for a in D["actions"] if a["verb"] == verb and a["noun"] == noun]
    return min(cands, key=lambda a: abs(a["segment"][0] - near))


# Five predictions, frames checked by eye against the footage (demo/rewind_demo_qc.md).
MONTAGE = [
    find_action("turn-on", "tap", 121.3),
    find_action("take", "sponge", 123.9),
    find_action("pour", "liquid:washing", 127.4),
    find_action("wash", "knife", 249.4),
    find_action("open", "bin", 418.2),
]
SPORTS = [  # (video, highlight index shown); label/score read from the JSON
    ("Ski_-Z5a4UQmZLs", 2),
    ("Motor_0JZi_gJ7Jfw", 1),
    ("Horse_7ZzWccM_5cM", 2),
    ("Scuba__JnCVbCL5Wg", 3),
    ("RockClimbing_mDUcIkDzpVI", 1),
    ("Surfing_BKwkLQr9ai8", 1),
]
HOOK = ("Surfing_BKwkLQr9ai8", 0)
RESULTS = [  # final freeze, report split (README, results/ablation_report.md)
    ("Action coverage", 0.3638, "40%"),
    ("Action mAP", 0.1522, "25%"),
    ("Highlight F1", 0.2325, "25%"),
    ("Label accuracy", 0.8511, "10%"),
]
WEIGHTED = 0.3268


# ---------------------------------------------------------------- shared overlays
def caption(img, t, a, b, line1, line2="", y=900):
    al = ramp(t, a, a + 0.5) * (1 - ramp(t, b - 0.5, b))
    text(img, (W // 2, y), line1, F["h2"], INK, al, "mm", shadow=True)
    if line2:
        text(img, (W // 2, y + 64), line2, F["h3"], (205, 207, 212), al, "mm", shadow=True)


def kicker(img, s, alpha=1.0, xy=(80, 64)):
    """Pill label: hairline border, accent dot, light text."""
    if alpha <= 0.01:
        return
    f = F["kicker"].font_variant(size=22)
    d = ImageDraw.Draw(img, "RGBA")
    tw = d.textlength(s, font=f)
    x, y = xy
    d.rounded_rectangle(
        (x, y - 6, x + tw + 58, y + 34),
        20,
        fill=(20, 21, 30, int(170 * alpha)),
        outline=(255, 255, 255, int(40 * alpha)),
        width=1,
    )
    d.ellipse((x + 18, y + 9, x + 28, y + 19), fill=(*AMBER, int(255 * alpha)))
    d.text((x + 40, y + 14), s, font=f, fill=(225, 226, 232, int(255 * alpha)), anchor="lm")


def timeline(img, y, x0, x1, dur, hls, t_cursor=None, lit=None, alpha=1.0, h=14, label=True):
    rect(img, (x0, y, x1, y + h), (44, 47, 54), alpha, r=h // 2)
    sc = (x1 - x0) / dur
    for i, (a, b) in enumerate(hls):
        k = 1.0 if lit is None else lit(i)
        if k > 0:
            rect(
                img,
                (x0 + a * sc, y - 4, x0 + max(b * sc, a * sc + 3), y + h + 4),
                AMBER,
                alpha * k,
                r=3,
            )
    if t_cursor is not None:
        cx = x0 + t_cursor * sc
        rect(img, (cx - 2, y - 14, cx + 2, y + h + 14), INK, alpha)
    if label:
        text(img, (x0, y + h + 16), "0:00", F["small"], MUTED, alpha)
        text(img, (x1, y + h + 16), clock(dur), F["small"], MUTED, alpha, "ra")


# ---------------------------------------------------------------- scenes
class Scene:
    dur: float

    def frame(self, t: float) -> Image.Image:  # pragma: no cover
        raise NotImplementedError


class Hook(Scene):
    """0:00 surf barrel, real highlight window, then the title."""

    dur = 9.0

    def __init__(self):
        v, i = HOOK
        self.vid = v
        self.seg = doc(v)["highlights"][i]["segment"]

    def frame(self, t):
        a, b = self.seg
        st = a + (b - a - 0.05) * t / self.dur  # whole highlight, slow motion
        im = fit_cover(src(self.vid).frame_at(st), W, H)
        k = 1 - 0.55 * ramp(t, 4.2, 5.4)
        im = darken(im, k * min(1, 0.25 + ramp(t, 0, 0.8)))
        caption(im, t, 0.6, 4.4, "Hours of first-person video.", "Only some moments matter.", y=930)
        al = ramp(t, 5.0, 5.8) * (1 - ramp(t, 8.4, 9.0))
        grad_text(im, (W // 2, 470), "REWIND", F["hero"], al)
        text(im, (W // 2, 580), "Action-aware video summarization", F["h2"], AMBER, al, "mm")
        return im


class Problem(Scene):
    """A 9:21 kitchen recording fast-forwarded; the clock runs."""

    dur = 9.0

    def frame(self, t):
        u = ramp(t, 0.3, self.dur - 0.6)
        st = 2.0 + u * (T_EPIC - 4.0)
        im = bottom_grad(fit_cover(src(EPIC).frame_at(st), W, H), 0.9)
        im = darken(im, min(1, ramp(t, 0, 0.6) + 0.15))
        kicker(im, "The problem", ramp(t, 0.2, 0.8))
        text(im, (80, 104), f"{EPIC}  ·  EPIC-KITCHENS-100", F["h3"], INK, ramp(t, 0.2, 0.8))
        text(im, (W - 80, 80), clock(st), F["num"], INK, ramp(t, 0.2, 0.8), "ra")
        speed = (T_EPIC - 4.0) / (self.dur - 0.9)
        text(
            im,
            (W - 80, 182),
            f"of {clock(T_EPIC)}   ·   fast-forward {speed:.0f}x",
            F["small"],
            MUTED,
            ramp(t, 0.2, 0.8),
            "ra",
        )
        timeline(im, 1000, 80, W - 80, T_EPIC, [], t_cursor=st, alpha=ramp(t, 0.2, 0.8))
        caption(im, t, 0.8, 4.6, "Continuous activity. Every second recorded.", "", y=880)
        caption(im, t, 4.8, 8.8, "Finding the actions means watching all of it.", "", y=880)
        return im


class InOut(Scene):
    """Input MP4 -> run.sh -> one JSON (real counts; precomputed by the frozen pipeline)."""

    dur = 10.0

    def __init__(self):
        self.thumb = fit_cover(src(EPIC).frame_at(127.9), 560, 315)
        s = src(EPIC).s
        self.res = f"{s.width}×{s.height}"

    def frame(self, t):
        im = canvas()
        kicker(im, "What REWIND does", ramp(t, 0, 0.5))
        al = ramp(t, 0.3, 1.0)
        x, y = 90, 330
        card(im, (x - 8, y - 8, x + 568, y + 323), al, r=14)
        if al > 0.01:
            im.paste(darken(self.thumb, al), (x, y))
        text(im, (x, y + 345), f"{EPIC}.mp4", F["h3b"], INK, al)
        text(
            im,
            (x, y + 395),
            f"{clock(T_EPIC)}  ·  {self.res}  ·  one MP4, nothing else",
            F["body"],
            MUTED,
            al,
        )
        a2 = ramp(t, 1.4, 2.2)
        cx = 990
        text(
            im,
            (cx, 430),
            "./run.sh P01_11.mp4 P01_11.json",
            F["mono"].font_variant(size=27),
            INK,
            a2,
            "mm",
        )
        prog = ramp(t, 2.2, 4.0)
        rect(im, (cx - 220, 480, cx + 220, 488), (44, 47, 54), a2, r=4)
        rect(im, (cx - 220, 480, cx - 220 + 440 * prog, 488), AMBER, a2, r=4)
        text(im, (cx, 530), "frozen pipeline · results precomputed", F["small"], MUTED, a2, "mm")
        a3 = ramp(t, 4.0, 4.8)
        x3 = 1330
        n_act = len(D["actions"])
        rows = [
            (f"{n_act:,}", "timestamped actions  (verb, noun, score)"),
            (f"{len(HLS)}", f"highlight windows  ·  {clock(HL_TOTAL)} of {clock(T_EPIC)}"),
            (D["label"]["name"], f"video label  ·  score {D['label']['score']:.2f}"),
        ]
        text(im, (x3, 300), f"{EPIC}.json", F["h3b"], AMBER, a3)
        for i, (big, small) in enumerate(rows):
            ai = ramp(t, 4.4 + 0.45 * i, 5.1 + 0.45 * i)
            text(im, (x3, 370 + 130 * i), big, F["numm"], INK, ai)
            text(im, (x3, 440 + 130 * i), small, F["small"], MUTED, ai)
        caption(im, t, 5.6, 9.9, "One video in. One structured summary out.", "", y=930)
        return im


STAGES_EPIC = [
    ("Features", "SlowFast", "clip features"),
    ("Detection", "ActionFormer", "verb + noun"),
    ("Pair + score", "learned", "P(true action)"),
    ("Cluster", "candidate", "action events"),
    ("Select", "max expected", "action coverage"),
    ("Highlights", "≥ 4 s windows", "15% budget"),
]
STAGES_FPV = [
    ("Signals", "motion, hands,", "scene changes"),
    ("Importance", "smoothed", "per-second score"),
    ("Windows", "best per shot,", "fill the budget"),
]


class Pipeline(Scene):
    dur = 17.0

    def frame(self, t):
        im = canvas()
        kicker(im, "How it works", ramp(t, 0, 0.5))
        text(
            im,
            (80, 104),
            "Two routes, chosen from the pixels by CLIP",
            F["h2"],
            INK,
            ramp(t, 0.1, 0.7),
        )
        # route node
        ar = ramp(t, 0.8, 1.5)
        card(im, (80, 470, 330, 610), ar, r=16, accent=AMBER)
        text(im, (205, 512), "CLIP router", F["h3b"], INK, ar, "mm")
        text(im, (205, 562), "cooking or not?", F["small"], MUTED, ar, "mm")
        bw, gap, x0 = 222, 22, 410
        for _lane, (stages, y, col, start, name) in enumerate(
            [
                (STAGES_EPIC, 330, AMBER, 1.6, "kitchen → actions"),
                (STAGES_FPV, 700, BLUE, 9.5, "sports → importance"),
            ]
        ):
            al = ramp(t, start - 0.3, start + 0.2)
            text(im, (x0, y - 46), name.upper(), F["kicker"], col, al)
            d = ImageDraw.Draw(im, "RGBA")
            if al > 0.01:
                d.line(
                    [(330, 540), (365, 540), (365, y + 70), (x0, y + 70)],
                    fill=(*col, int(200 * al)),
                    width=3,
                )
            for i, (a, b, b2) in enumerate(stages):
                ai = ramp(t, start + 0.9 * i, start + 0.9 * i + 0.6)
                bx = x0 + i * (bw + gap)
                card(im, (bx, y, bx + bw, y + 140), ai, r=14, accent=col)
                text(im, (bx + 18, y + 26), a, F["h3b"].font_variant(size=30), INK, ai)
                text(im, (bx + 18, y + 76), b, F["small"], MUTED, ai)
                text(im, (bx + 18, y + 104), b2, F["small"], MUTED, ai)
                if i and ai > 0.01:
                    d.line(
                        [(bx - gap + 4, y + 70), (bx - 4, y + 70)],
                        fill=(*col, int(255 * ai)),
                        width=3,
                    )
        caption(
            im,
            t,
            7.4,
            9.4,
            "Kitchens: find every (verb, noun) action, then cover as many distinct ones as possible.",
            "",
            y=960,
        )
        caption(
            im,
            t,
            12.6,
            16.9,
            "Outdoor sports: keep the moments with the most purposeful motion.",
            "",
            y=960,
        )
        return im


class Montage(Scene):
    """Real predictions over the footage they describe (P01_11, real speed)."""

    PRE, POST = 0.7, 0.8

    def __init__(self):
        self.clips = []
        for a in MONTAGE:
            s, e = a["segment"]
            self.clips.append((a, s - self.PRE, max(e + self.POST, s + 2.6)))
        self.dur = sum(b - a for _, a, b in self.clips) + 0.0
        self.starts = np.cumsum([0] + [b - a for _, a, b in self.clips])

    def frame(self, t):
        i = int(np.searchsorted(self.starts, t, side="right") - 1)
        i = min(max(i, 0), len(self.clips) - 1)
        a, c0, c1 = self.clips[i]
        lt = t - self.starts[i]
        st = c0 + lt
        im = bottom_grad(fit_cover(src(EPIC).frame_at(st), W, H), 0.9)
        cl = c1 - c0
        edge = min(ramp(lt, 0, 0.25), 1 - ramp(lt, cl - 0.25, cl))
        if edge < 1:
            im = darken(im, 0.35 + 0.65 * edge)
        kicker(im, "Real predictions  ·  P01_11", 1.0)
        text(im, (W - 80, 64), clock(st, ms=True), F["h3"], INK, 1.0, "ra")
        s, e = a["segment"]
        on = ramp(st, s - 0.15, s + 0.2)
        live = s - 0.05 <= st <= e + 0.05
        glass(im, (70, 742, 70 + max(560, 40 + 60 * len(a["verb"])), 1000), on)
        text(im, (96, 760), a["verb"].upper(), F["verb"], INK, on)
        text(im, (100, 860), noun_txt(a["noun"]), F["noun"], AMBER, on)
        text(
            im,
            (100, 940),
            f"{clock(s, True)} – {clock(e, True)}   ·   score {a['score']:.2f}",
            F["mono"],
            MUTED,
            on,
        )
        if live:
            rect(im, (80, 770, 88, 900), AMBER, on)
        timeline(
            im, 1020, 1080, W - 80, T_EPIC, [(s, e)], t_cursor=st, alpha=0.9, h=10, label=False
        )
        text(im, (1080, 990), f"position in the {clock(T_EPIC)} video", F["small"], MUTED, 0.9)
        return im


class Summary(Scene):
    """Original vs REWIND summary: windows light up on the real timeline; highlights play."""

    dur = 12.0

    def __init__(self):
        self.order = sorted(range(len(HLS)), key=lambda i: HLS[i][0])
        self.play = self.dur - 0.8
        self.per = self.play / len(HLS)

    def frame(self, t):
        im = canvas()
        kicker(im, "Original  →  REWIND summary", ramp(t, 0, 0.5))
        k = ramp(t, 0.6, 7.0)
        n_lit = k * len(HLS)
        timeline(
            im,
            250,
            80,
            W - 80,
            T_EPIC,
            HLS,
            lit=lambda i: min(1, max(0, n_lit - self.order.index(i))),
            alpha=ramp(t, 0.2, 0.8),
            h=22,
        )
        # playback panel: highlight clips back-to-back, 2.5x
        pt = min(max(0.0, t - 0.6), self.play - 1e-3)
        j = min(int(pt / self.per), len(HLS) - 1)
        a, b = HLS[self.order[j]]
        frac = (pt - j * self.per) / self.per
        st = a + frac * (b - a)
        pw, ph = 960, 540
        fr = fit_cover(src(EPIC).frame_at(st), pw, ph)
        alv = ramp(t, 0.4, 1.0)
        im.paste(darken(fr, alv), (80, 380))
        text(
            im,
            (80, 940),
            f"highlight {j + 1} of {len(HLS)}  ·  {clock(a, True)} – {clock(b, True)}",
            F["body"],
            MUTED,
            alv,
        )
        # numbers
        x = 1160
        cnt = ramp(t, 1.0, 7.0)
        text(im, (x, 400), "ORIGINAL", F["kicker"], MUTED, ramp(t, 0.6, 1.2))
        text(im, (x, 440), clock(T_EPIC), F["num"], INK, ramp(t, 0.6, 1.2))
        text(im, (x, 590), "REWIND SUMMARY", F["kicker"], AMBER, ramp(t, 1.0, 1.6))
        text(im, (x, 630), clock(HL_TOTAL * cnt), F["num"], AMBER, ramp(t, 1.0, 1.6))
        pct = 100 * HL_TOTAL / T_EPIC
        text(
            im,
            (x, 780),
            f"{len(HLS)} windows · {pct:.1f}% of the video",
            F["h3"],
            INK,
            ramp(t, 7.0, 7.6),
        )
        text(
            im,
            (x, 832),
            "placed where the most distinct actions happen",
            F["body"],
            MUTED,
            ramp(t, 7.4, 8.0),
        )
        caption(
            im,
            t,
            8.4,
            11.9,
            f"{clock(T_EPIC)} of footage  →  {clock(HL_TOTAL)} that keeps the actions.",
            "",
            y=1010,
        )
        return im


class Sports(Scene):
    """Same system on outdoor first-person video: highlight + predicted label."""

    def __init__(self):
        self.items = []
        for v, i in SPORTS:
            d = doc(v)
            self.items.append((v, d, tuple(d["highlights"][i]["segment"])))
        self.per = 2.4
        self.dur = self.per * len(self.items) + 0.6

    def frame(self, t):
        j = min(int(t / self.per), len(self.items) - 1)
        v, d, (a, b) = self.items[j]
        lt = t - j * self.per
        st = a + min(lt, b - a - 0.05)
        im = bottom_grad(fit_cover(src(v).frame_at(st), W, H), 0.85)
        edge = (
            min(ramp(lt, 0, 0.2), 1 - ramp(lt, self.per - 0.2, self.per))
            if j < len(self.items) - 1
            else ramp(lt, 0, 0.2)
        )
        if edge < 1:
            im = darken(im, 0.3 + 0.7 * edge)
        kicker(im, "Outdoor first-person  ·  FPVSum", 1.0)
        lab = d["label"]
        glass(im, (70, 782, 70 + max(640, 40 + 62 * len(lab["name"])), 1000), 1.0)
        text(im, (96, 800), lab["name"].upper(), F["verb"], INK, 1.0)
        text(im, (100, 905), f"predicted label  ·  score {lab['score']:.2f}", F["body"], AMBER, 1.0)
        text(
            im,
            (100, 950),
            f"highlight {clock(a, True)} – {clock(b, True)} of {clock(d['duration_sec'])}",
            F["mono"].font_variant(size=26),
            MUTED,
            1.0,
        )
        hl = [tuple(h["segment"]) for h in d["highlights"]]
        timeline(
            im, 1020, 1080, W - 80, d["duration_sec"], hl, t_cursor=st, alpha=0.9, h=10, label=False
        )
        caption(im, t, self.dur - 2.0, self.dur, "", "")
        return im


class JsonView(Scene):
    dur = 10.0

    def __init__(self):
        shown = MONTAGE[2:4]
        self.lines = [
            ("{", INK),
            (f'  "video_id": "{D["video_id"]}",', INK),
            (f'  "duration_sec": {D["duration_sec"]},', INK),
            (
                f'  "label": {{"name": "{D["label"]["name"]}", "score": {D["label"]["score"]}}},',
                BLUE,
            ),
            ('  "actions": [', INK),
        ]
        for a in shown:
            s, e = a["segment"]
            self.lines.append(
                (
                    f'    {{"segment": [{s}, {e}], "verb": "{a["verb"]}", "noun": "{a["noun"]}", "score": {a["score"]}}},',
                    AMBER,
                )
            )
        self.lines.append((f"    ...  {len(D['actions']) - len(shown):,} more", MUTED))
        self.lines.append(("  ],", INK))
        self.lines.append(('  "highlights": [', INK))
        for a, b in HLS[:3]:
            self.lines.append((f'    {{"segment": [{a}, {b}]}},', GREEN))
        self.lines.append((f"    ...  {len(HLS) - 3} more", MUTED))
        self.lines.append(("  ]", INK))
        self.lines.append(("}", INK))

    def frame(self, t):
        im = canvas()
        kicker(im, f"Actual output  ·  outputs/{EPIC}.json", ramp(t, 0, 0.5))
        card(im, (80, 150, W - 80, 880), ramp(t, 0, 0.6), r=18)
        y = 190
        for i, (s, col) in enumerate(self.lines):
            al = ramp(t, 0.4 + 0.12 * i, 0.8 + 0.12 * i)
            text(im, (130, y + i * 42), s, F["mono"], col, al)
        caption(
            im,
            t,
            3.6,
            9.9,
            "Structured, timestamped output.",
            "Every action and every highlight, in one file per video.",
            y=950,
        )
        return im


class Results(Scene):
    dur = 9.5

    def frame(self, t):
        im = canvas()
        kicker(
            im,
            "Final benchmark results  ·  official metrics  ·  EPIC-KITCHENS-100 val + FPVSum",
            ramp(t, 0, 0.5),
        )
        x0, x1 = 520, 1500
        for i, (name, v, wgt) in enumerate(RESULTS):
            y = 300 + i * 150
            ai = ramp(t, 0.4 + 0.35 * i, 1.0 + 0.35 * i)
            text(im, (80, y), name, F["h3b"], INK, ai)
            text(im, (80, y + 48), f"weight {wgt}", F["small"], MUTED, ai)
            card(im, (x0, y + 10, x1, y + 46), ai, r=18)
            g = ramp(t, 0.6 + 0.35 * i, 2.2 + 0.35 * i)
            rect(im, (x0, y + 10, x0 + (x1 - x0) * v * g, y + 46), AMBER, ai, r=18)
            text(im, (x1 + 40, y + 28), f"{100 * v * g:.2f}%", F["h2b"], INK, ai, "lm")
        return im


class End(Scene):
    dur = 6.0

    def __init__(self):
        _v, _d, (_a, _b) = "Ski_-Z5a4UQmZLs", None, (0, 0)
        self.bg = Image.fromarray(
            np.asarray(fit_cover(src("Ski_-Z5a4UQmZLs").frame_at(18.0), W, H))
        )
        self.bg = darken(self.bg.filter(ImageFilter.GaussianBlur(18)), 0.32)

    def frame(self, t):
        z = 1.0 + 0.04 * t / self.dur  # slow push-in
        bw, bh = int(W * z), int(H * z)
        im = self.bg.resize((bw, bh), Image.BILINEAR).crop(
            ((bw - W) // 2, (bh - H) // 2, (bw - W) // 2 + W, (bh - H) // 2 + H)
        )
        if t < 0.8:
            im = darken(im, 0.4 + 0.6 * ramp(t, 0, 0.8))
        a1 = ramp(t, 0.3, 1.1) * (1 - ramp(t, 5.4, 6.0) * 0.0)
        grad_text(im, (W // 2, 440), "REWIND", F["hero"], a1)
        text(
            im,
            (W // 2, 560),
            "Turn long recordings into action-aware summaries.",
            F["h2"],
            INK,
            ramp(t, 0.9, 1.7),
            "mm",
        )
        text(im, (W // 2, 700), "VITVellore_Nexus", F["h3b"], AMBER, ramp(t, 1.6, 2.4), "mm")
        text(
            im,
            (W // 2, 750),
            "Samsung XR Hackathon 2026  ·  Track 04",
            F["body"],
            MUTED,
            ramp(t, 1.9, 2.7),
            "mm",
        )
        fade = ramp(t, self.dur - 0.7, self.dur)
        return darken(im, 1 - fade) if fade > 0 else im


# ---------------------------------------------------------------- audio (original sound bed)
def sound_bed(total: float, cuts: list[float]) -> np.ndarray:
    n = int(total * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros((2, n), dtype=np.float32)
    chord = [110.0, 164.81, 220.0, 277.18]  # A2 E3 A3 C#4: soft A major pad
    for k, f0 in enumerate(chord):
        for det, pan in ((-0.6, 0.35), (0.6, 0.65)):
            ph = 2 * np.pi * (f0 + det) * t
            lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.05 + 0.013 * k) * t + k)
            w = (np.sin(ph) + 0.18 * np.sin(2 * ph)) * lfo * (0.05 / (1 + 0.35 * k))
            out[0] += w * (1 - pan)
            out[1] += w * pan
    env = np.minimum(1, t / 3.0) * np.minimum(1, (total - t) / 3.0)
    out *= env
    rng = np.random.default_rng(7)
    for c in cuts:  # soft airy whoosh at each scene change
        i0, ln = int(max(0, c - 0.35) * SR), int(0.7 * SR)
        if i0 + ln >= n:
            continue
        nz = rng.standard_normal(ln).astype(np.float32)
        nz = np.convolve(nz, np.ones(24) / 24, mode="same")
        e = np.sin(np.linspace(0, np.pi, ln)) ** 2
        out[:, i0 : i0 + ln] += 0.05 * nz * e
    peak = np.abs(out).max()
    return (out / peak * 0.32) if peak > 0 else out


# ---------------------------------------------------------------- render
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "demo" / "rewind_demo.mp4")
    ap.add_argument("--preview", type=float, default=0.0, help="render only the first N seconds")
    ap.add_argument(
        "--stills", type=str, default="", help="comma list of times (s): write PNG frames only"
    )
    ap.add_argument("--stills-dir", type=Path, default=REPO / "demo" / "rewind_assets" / "stills")
    args = ap.parse_args()

    scenes = [
        Hook(),
        Problem(),
        InOut(),
        Pipeline(),
        Montage(),
        Summary(),
        Sports(),
        JsonView(),
        Results(),
        End(),
    ]
    if args.stills:
        args.stills_dir.mkdir(parents=True, exist_ok=True)
        starts = np.cumsum([0] + [s.dur for s in scenes])
        for ts in [float(x) for x in args.stills.split(",")]:
            j = int(np.searchsorted(starts, ts, side="right") - 1)
            j = min(j, len(scenes) - 1)
            scenes[j].frame(ts - starts[j]).save(args.stills_dir / f"t{ts:06.1f}.png")
        print(f"stills -> {args.stills_dir}")
        return
    total = sum(s.dur for s in scenes)
    cuts = list(np.cumsum([s.dur for s in scenes])[:-1])
    if args.preview:
        total = min(total, args.preview)
    print(f"scenes: {[(type(s).__name__, round(s.dur, 2)) for s in scenes]}  total {total:.2f} s")

    audio = sound_bed(total, cuts)
    tmp = args.out.with_name(
        args.out.stem + ".partial.mp4"
    )  # replace the deliverable only when complete
    oc = av.open(str(tmp), "w", format="mp4")
    vs = oc.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    vs.options = {"crf": "18", "preset": "medium"}
    as_ = oc.add_stream("aac", rate=SR)
    as_.layout = "stereo"
    a_pos, a_chunk = 0, 1024
    n_frames = round(total * FPS)
    t0 = 0.0
    k = 0
    for sc in scenes:
        nf = round(sc.dur * FPS)
        for i in range(nf):
            if k >= n_frames:
                break
            img = sc.frame(i / FPS)
            vf = av.VideoFrame.from_image(img)
            for p in vs.encode(vf):
                oc.mux(p)
            k += 1
            while a_pos < min(audio.shape[1], int(k / FPS * SR)):
                seg = np.ascontiguousarray(audio[:, a_pos : a_pos + a_chunk])
                af = av.AudioFrame.from_ndarray(seg, format="fltp", layout="stereo")
                af.sample_rate = SR
                af.pts = a_pos
                af.time_base = Fraction(1, SR)
                for p in as_.encode(af):
                    oc.mux(p)
                a_pos += seg.shape[1]
            if k % 150 == 0:
                print(f"  {k / FPS:6.1f} s rendered", flush=True)
        t0 += sc.dur
    for p in vs.encode():
        oc.mux(p)
    for p in as_.encode():
        oc.mux(p)
    oc.close()
    tmp.replace(args.out)
    print(f"wrote {args.out} ({k / FPS:.1f} s)")


if __name__ == "__main__":
    main()
