# ruff: noqa: E501
"""Render demo/rewind_reproducibility.mp4 from the real captured run (demo/repro_capture.json).

    python tools/video/repro_render.py

Commands appear as typed; output lines appear in the order and relative timing they were
captured, with long waits compressed and labelled ("ran 41.3 s, shown faster"). No output
text is invented. 1920x1080, 30 fps, H.264, silent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[2] / "demo"
OUT = HERE / "rewind_reproducibility.mp4"
W, H, FPS = 1920, 1080, 30
BG, TERM, BAR = (12, 13, 16), (18, 20, 24), (30, 33, 39)
INK, MUTED, AMBER, GREEN, BLUE, RED = (
    (232, 232, 228),
    (140, 144, 152),
    (255, 168, 64),
    (120, 210, 150),
    (112, 172, 240),
    (240, 110, 100),
)
FD = Path("C:/Windows/Fonts")
MONO = ImageFont.truetype(str(FD / "consola.ttf"), 29)
MONOB = ImageFont.truetype(str(FD / "consolab.ttf"), 29)
H1 = ImageFont.truetype(str(FD / "segoeuib.ttf"), 120)
H2 = ImageFont.truetype(str(FD / "segoeuisl.ttf"), 48)
SEC = ImageFont.truetype(str(FD / "segoeuib.ttf"), 30)
CAP = ImageFont.truetype(str(FD / "segoeui.ttf"), 28)
LH = 38
TX0, TY0, TX1, TY1 = 80, 150, W - 80, 960
ROWS = (TY1 - TY0 - 70) // LH
COLS = (TX1 - TX0 - 48) // 16


def wrap(s: str) -> list[str]:
    s = s.replace("\t", "    ")
    return [s[i : i + COLS] for i in range(0, max(len(s), 1), COLS)] or [""]


TOK = re.compile(r'("[^"]*")(\s*:)?|(-?\b\d+(?:\.\d+)?\b)|\b(true|false|null)\b')
JSONISH = re.compile(r'^\s*([{}\[\],]|"|-?\d)')


def draw_line(d, x, y, s, col):
    """Commands: green prompt + white command; JSON: keys blue, strings green, numbers amber."""
    if col == BLUE and s.startswith("$ "):
        d.text((x, y), "$", font=MONOB, fill=GREEN)
        d.text((x + 2 * d.textlength(" ", font=MONO), y), s[2:], font=MONOB, fill=INK)
        return
    if col == INK and JSONISH.match(s):
        pos = 0
        for m in TOK.finditer(s):
            if m.start() > pos:
                d.text(
                    (x + d.textlength(s[:pos], font=MONO), y),
                    s[pos : m.start()],
                    font=MONO,
                    fill=MUTED,
                )
            tok, c = m.group(0), INK
            if m.group(1):
                c = BLUE if m.group(2) else GREEN
            elif m.group(3) or m.group(4):
                c = AMBER
            d.text((x + d.textlength(s[: m.start()], font=MONO), y), tok, font=MONO, fill=c)
            pos = m.end()
        if pos < len(s):
            d.text((x + d.textlength(s[:pos], font=MONO), y), s[pos:], font=MONO, fill=MUTED)
        return
    d.text((x, y), s, font=MONOB if col == GREEN else MONO, fill=col)


STEPS = ["Repository", "Environment", "Inference", "Output", "Validation", "Results"]


class Term:
    def __init__(self):
        self.lines: list[tuple[str, tuple]] = []

    def add(self, s, col=INK):
        for w in wrap(s):
            self.lines.append((w, col))

    def draw(self, section, caption, cursor=False):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((TX0, 62), section, font=SEC, fill=AMBER)
        cur = int(section[0]) if section[:1].isdigit() else 0
        x = TX1
        for i in range(len(STEPS), 0, -1):
            lab = f"{i} {STEPS[i - 1]}"
            tw = d.textlength(lab, font=CAP.font_variant(size=22)) + 28
            x -= tw
            on = i == cur
            d.rounded_rectangle((x, 60, x + tw - 8, 100), 18, fill=AMBER if on else BAR)
            d.text(
                (x + (tw - 8) / 2, 80),
                lab,
                font=CAP.font_variant(size=22),
                fill=BG if on else (MUTED if i > cur else INK),
                anchor="mm",
            )
        d.rounded_rectangle((TX0, TY0, TX1, TY1), 14, fill=TERM)
        d.rounded_rectangle((TX0, TY0, TX1, TY0 + 44), 14, fill=BAR)
        d.rectangle((TX0, TY0 + 30, TX1, TY0 + 44), fill=BAR)
        for i, c in enumerate(((240, 95, 85), (245, 190, 80), (100, 200, 110))):
            d.ellipse((TX0 + 20 + i * 28, TY0 + 14, TX0 + 36 + i * 28, TY0 + 30), fill=c)
        d.text(
            ((TX0 + TX1) // 2, TY0 + 22),
            "bash  ·  rewind",
            font=CAP.font_variant(size=22),
            fill=MUTED,
            anchor="mm",
        )
        vis = self.lines[-ROWS:]
        for i, (s, col) in enumerate(vis):
            draw_line(d, TX0 + 24, TY0 + 64 + i * LH, s, col)
        if cursor and vis:
            x = TX0 + 24 + int(d.textlength(vis[-1][0], font=MONO))
            d.rectangle(
                (x + 4, TY0 + 64 + (len(vis) - 1) * LH + 4, x + 16, TY0 + 64 + len(vis) * LH - 4),
                fill=INK,
            )
        if caption:
            d.text((W // 2, 1015), caption, font=CAP, fill=MUTED, anchor="mm")
        return im


class Writer:
    def __init__(self, path: Path):
        self.tmp = path.with_name(path.stem + ".partial.mp4")
        self.path = path
        self.c = av.open(str(self.tmp), "w", format="mp4")
        self.s = self.c.add_stream("libx264", rate=FPS)
        self.s.width, self.s.height, self.s.pix_fmt = W, H, "yuv420p"
        self.s.options = {"crf": "20", "preset": "medium"}
        self.n = 0

    def put(self, im: Image.Image, secs: float):
        vf = av.VideoFrame.from_image(im)
        for _ in range(max(1, round(secs * FPS))):
            for p in self.s.encode(vf):
                self.c.mux(p)
            self.n += 1

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
        self.tmp.replace(self.path)


def title(w: Writer, big: str, sub: str, lines: list[str], secs: float):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 380), big, font=H1, fill=INK, anchor="mm")
    d.text((W // 2, 490), sub, font=H2, fill=AMBER, anchor="mm")
    for i, s in enumerate(lines):
        d.text((W // 2, 600 + i * 46), s, font=CAP, fill=MUTED, anchor="mm")
    w.put(im, secs)


CAPTIONS = {
    "1": "Anyone can clone the public repository and read the documented setup; weights come from a public GitHub release.",
    "2": "Prepared machine: pip confirms the pinned requirements; fetch_weights.sh SHA-256-checks every weight (downloads any that are missing).",
    "3": "run.sh reads only the MP4 (plus config and weights) and always writes a valid JSON.",
    "4": "The fresh output matches the file we submitted, byte for byte.",
    "5": "All 187 submitted files pass every kit rule; the unit, property and contract tests pass.",
    "6": "Re-scoring needs the hidden FPVSum labels, so the final scores are shown as previously measured at the freeze.",
}


def main() -> None:
    log = json.loads((HERE / "repro_capture.json").read_text(encoding="utf-8"))
    w = Writer(OUT)
    title(
        w,
        "REWIND",
        "Reproducibility & execution",
        [
            "A real run, captured command by command.",
            "Long waits are shortened and labelled; no output is edited.",
        ],
        6.0,
    )
    term, section = Term(), None
    for step in log:
        if step["section"] != section:
            if section is not None:  # short fade to the next section
                last = term.draw(section, CAPTIONS.get(section[0], ""))
                for k in range(8):
                    w.put(Image.blend(last, Image.new("RGB", (W, H), BG), (k + 1) / 8), 1 / FPS)
            section = step["section"]
            term = Term()
        cap = CAPTIONS.get(section[0], "")
        prompt = "$ "
        cmd = step["cmd"]
        for k in range(0, len(cmd) + 1, 2):  # typing
            t2 = Term()
            t2.lines = term.lines + [(w2, BLUE) for w2 in wrap(prompt + cmd[:k])]
            w.put(t2.draw(section, cap, cursor=True), 1 / FPS)
        term.add(prompt + cmd, BLUE)
        w.put(term.draw(section, cap), 0.3)
        lines = step["lines"]
        budget = min(7.0, 2.0 + 0.3 * len(lines))
        last = 0.0
        tmax = max([t for t, _ in lines] + [0.01])
        for t, s in lines:
            gap = (t - last) / tmax * budget
            last = t
            col = (
                RED
                if "error" in s.lower() and "0 error" not in s.lower()
                else (
                    GREEN
                    if any(k in s for k in ("identical", "passed", "0 error", "HTTP 200"))
                    else INK
                )
            )
            if gap > 1 / FPS:
                w.put(term.draw(section, cap), gap)
            term.add(s, col)
        if step["seconds"] > 4:
            term.add(
                f"# ran {step['seconds']:.1f} s on CPU (shown faster)  ·  exit code {step['rc']}",
                MUTED,
            )
        else:
            term.add(f"# exit code {step['rc']}", MUTED)
        w.put(term.draw(section, cap), 6.0 if len(lines) > 6 else 4.5)
    title(
        w,
        "REWIND",
        "Setup. Run. Reproduce.",
        [
            "github.com/prakartii/samsung-xr-hackathon",
            "Action-aware video summarization  ·  VITVellore_Nexus",
        ],
        6.0,
    )
    w.close()
    print(f"wrote {OUT} ({w.n / FPS:.1f} s)")


if __name__ == "__main__":
    main()
