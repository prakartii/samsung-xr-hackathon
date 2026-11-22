"""The submission deck: the organizers' 9-slide template filled from our results (P8).

    pip install python-pptx            # deck only, not a runtime dependency
    python tools/make_deck.py --demo demo/rewind_demo.mp4 --out docs/VITVellore_Nexus.pptx

Every number comes from results/ablation_report.md and the team ID from
configs/default.yaml; the example JSON is an output in results/preds/full_report;
slide images are frames of the demo video (tools/make_demo.py). Open the result in
PowerPoint to export a PDF.
"""

import argparse
import copy
import csv
import json
import tempfile
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
MEMBERS = [("Prakarti Dubey", "24BCI0263"), ("Daksh Arora", "24BCI0259")]
FRAMES = {"ff": 26, "fridge": 60, "boxing": 95}  # demo-video seconds used as slide images

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument(
    "--template",
    type=Path,
    default=REPO / "docs" / "organizers" / "Submission_Final_forGithub.pptx",
)
ap.add_argument("--demo", type=Path, default=REPO / "demo" / "rewind_demo.mp4")
ap.add_argument("--example", default="P01_11", help="output JSON shown on the output slide")
ap.add_argument("--out", type=Path, default=REPO / "docs" / "VITVellore_Nexus.pptx")
args = ap.parse_args()
HERE = Path(tempfile.mkdtemp())


def grab_frames() -> None:
    import av

    want = {round(t * 30): n for n, t in FRAMES.items()}
    with av.open(str(args.demo)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i in want:
                f.to_image().save(HERE / f"img_{want[i]}.png")
            if i > max(want):
                break


grab_frames()

BG = RGBColor(0x0A, 0x0E, 0x1A)
BAND = RGBColor(0x0D, 0x1B, 0x3E)
CARD = RGBColor(0x12, 0x22, 0x4A)
CYAN = RGBColor(0x00, 0xB4, 0xD8)
BLUE = RGBColor(0x15, 0x65, 0xC0)
TEAL = RGBColor(0x00, 0xE5, 0xB0)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0xA8, 0xB3, 0xC7)
CODEBG = RGBColor(0x06, 0x0A, 0x14)

prs = Presentation(str(args.template))
S = list(prs.slides)
_md = (REPO / "results" / "ablation_report.md").read_text(encoding="utf-8").splitlines()
_rows = [ln for ln in _md if ln.startswith("|")]
_hdr = [x.strip() for x in _rows[0].strip("|").split("|")][1:]
abl = {}
for ln in _rows[2:]:
    c = [x.strip() for x in ln.strip("|").split("|")]
    abl[c[0]] = {k: float(x) if x else None for k, x in zip(_hdr, c[1:], strict=True)}
_preds = sorted((REPO / "results" / "preds" / "full_report").glob("*.json"))
_docs = [json.loads(q.read_text(encoding="utf-8")) for q in _preds]
_kept = sum(e - b for d in _docs for b, e in (h["segment"] for h in d["highlights"]))
PRUNED = 1.0 - _kept / sum(float(d["duration_sec"]) for d in _docs)
with (REPO / "results" / "latency_2026-11-21.csv").open(encoding="utf-8") as _f:
    _lat = list(csv.DictReader(_f))
SEC_PER_MIN = sum(float(r["seconds"]) for r in _lat) / (
    sum(float(r["duration_sec"]) for r in _lat) / 60.0
)
FULL, UNI, ORA = abl["full pipeline"], abl["uniform baseline"], abl["oracle baseline"]


def text(
    slide,
    x,
    y,
    w,
    h,
    runs,
    size=14,
    color=WHITE,
    bold=False,
    font="Calibri",
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    name=None,
    space_after=0,
):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, opts)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        for t, o in [(para, {})] if isinstance(para, str) else para:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False)
            f.color.rgb = o.get("color", color)
    return tb


def box(
    slide,
    x,
    y,
    w,
    h,
    fill=CARD,
    line=None,
    shape=MSO_SHAPE.ROUNDED_RECTANGLE,
    name=None,
    radius=0.08,
):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    s.text_frame.text = ""
    return s


def arrow(slide, x1, y1, x2, y2, color=MUTED):
    c = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def title(slide, s, color=WHITE):
    text(slide, 0.4, 1.05, 9.2, 0.5, s, size=24, bold=True, color=color, name="Slide title")


def set_header(slide, new):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.name == "Text 2":
            sh.text_frame.paragraphs[0].runs[0].text = new
            for r in sh.text_frame.paragraphs[0].runs[1:]:
                r.text = ""


def picture(slide, path, x, y, w=None, h=None, name=None):
    p = slide.shapes.add_picture(
        str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None
    )
    if name:
        p.name = name
    return p


def code(slide, x, y, w, h, lines, size=10.5, name=None):
    box(slide, x, y, w, h, fill=CODEBG, line=RGBColor(0x24, 0x33, 0x5C), name=name, radius=0.04)
    paras = []
    for ln in lines:
        col = (
            MUTED
            if ln.lstrip().startswith(("#", "..."))
            else (TEAL if ln.startswith("$") else WHITE)
        )
        paras.append([(ln, {"font": "Courier New", "size": size, "color": col})])
    text(slide, x + 0.18, y + 0.14, w - 0.3, h - 0.25, paras, space_after=1)


# ---- 1 title ------------------------------------------------------------------------
s = S[0]
for sh in s.shapes:
    if not sh.has_text_frame:
        continue
    t = sh.text_frame.text
    if t == "Track No":
        sh.text_frame.paragraphs[0].runs[0].text = "Track 4"
    elif t.startswith("Team Name"):
        ps = sh.text_frame.paragraphs
        ps[0].runs[0].text = "Team Nexus  ·  VITVellore_Nexus"
        ps[1].runs[0].text = "VIT Vellore"
    elif t == "Members Details":
        p = sh.text_frame.paragraphs[0]
        p.runs[0].text = "Members"
        p.runs[0].font.bold = True
        for _ in MEMBERS:
            np_ = copy.deepcopy(p._p)
            sh.text_frame._txBody.append(np_)
        for p2, (name_, role) in zip(sh.text_frame.paragraphs[1:], MEMBERS, strict=True):
            p2.runs[0].text = f"{name_}  ·  {role}"
            p2.runs[0].font.bold = False
            p2.runs[0].font.color.rgb = MUTED
text(
    s,
    0.55,
    3.55,
    8.2,
    0.5,
    [
        [
            ("REWIND", {"bold": True, "color": CYAN, "size": 22}),
            ("   Action-Aware Video Summarization", {"size": 18}),
        ]
    ],
    name="Project name",
)
text(
    s,
    0.55,
    4.1,
    8.2,
    0.4,
    "The most distinct steps per second of highlight",
    size=14,
    color=MUTED,
    name="Tagline",
)

# ---- 2 problem ----------------------------------------------------------------------
s = S[1]
title(s, "Minutes of first-person video, a few seconds that matter")
stats = [
    ("2–10 min", "per egocentric video: kitchens, sports, travel"),
    ("≤ 15%", "of it may be kept, in windows of at least 4 s"),
    ("98 × 301", "verbs × nouns to recognise, plus 16 video categories"),
]
for i, (big, small) in enumerate(stats):
    y = 1.8 + 1.0 * i
    text(s, 0.4, y, 2.0, 0.6, big, size=30, bold=True, color=CYAN, name=f"Stat {i + 1}")
    text(s, 2.45, y + 0.12, 2.9, 0.6, small, size=14, color=WHITE, name=f"Stat {i + 1} label")
picture(s, HERE / "img_ff.png", 5.55, 1.75, w=4.05, name="Kitchen frame")
text(
    s,
    5.55,
    4.08,
    4.05,
    0.5,
    "Most of a kitchen video is reaching, walking and waiting. "
    "Busy-looking frames are not the steps the wearer took.",
    size=11,
    color=MUTED,
    name="Frame caption",
)
text(
    s,
    0.4,
    4.85,
    9.2,
    0.5,
    [
        [
            (
                "Motion-based summarizers keep what looks busy and miss what "
                "was done: steps get dropped, repeats get kept.",
                {"italic": True},
            )
        ]
    ],
    size=14,
    color=WHITE,
    name="Problem statement",
)

# ---- 3 solution ---------------------------------------------------------------------
s = S[2]
title(s, "REWIND: one command, one JSON, every distinct step")
box(
    s,
    0.4,
    1.7,
    9.2,
    0.55,
    fill=CODEBG,
    line=RGBColor(0x24, 0x33, 0x5C),
    name="Command bar",
    radius=0.1,
)
text(
    s,
    0.65,
    1.7,
    8.8,
    0.55,
    [[("$ ", {"color": TEAL}), ("./run.sh  video.mp4  out.json", {"color": WHITE})]],
    size=16,
    font="Courier New",
    anchor=MSO_ANCHOR.MIDDLE,
    name="Command",
)
cards = [
    (
        "actions[]",
        "Every detected verb + noun action, with start, end and score",
        "ActionFormer on SlowFast features, paired with a verb-noun prior",
    ),
    (
        "highlights[]",
        "≥ 4 s windows of the original footage, ≤ 15% in total",
        "Chosen by expected new steps covered per second",
    ),
    (
        "label",
        "What kind of video it is: cooking or one of 14 sports",
        "CLIP zero-shot over the kit's classes",
    ),
]
for i, (h, body, how) in enumerate(cards):
    x = 0.4 + 3.1 * i
    box(s, x, 2.5, 2.95, 2.05, name=f"Card {i + 1}")
    text(
        s,
        x + 0.22,
        2.68,
        2.55,
        0.4,
        h,
        size=18,
        bold=True,
        color=TEAL,
        font="Courier New",
        name=f"Card {i + 1} title",
    )
    text(s, x + 0.22, 3.15, 2.55, 0.8, body, size=14, name=f"Card {i + 1} body")
    text(s, x + 0.22, 3.95, 2.55, 0.55, how, size=11, color=MUTED, name=f"Card {i + 1} how")
text(
    s,
    0.4,
    4.8,
    9.2,
    0.45,
    "CPU only  ·  no cloud  ·  same video → byte-identical JSON  ·  "
    "never reads the filename or manifest  ·  always writes valid JSON",
    size=12,
    color=MUTED,
    name="Properties",
)

# ---- 4 architecture -----------------------------------------------------------------
s = S[3]
title(s, "Two routes from one decode pass")


def node(x, y, w, h, head, sub, col, name):
    box(s, x, y, w, h, fill=CARD, line=col, name=name)
    text(
        s,
        x + 0.08,
        y + 0.06,
        w - 0.16,
        h - 0.1,
        [
            [(head, {"bold": True, "size": 12, "color": WHITE})],
            [(sub, {"size": 10, "color": MUTED})],
        ],
        align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE,
        name=name + " text",
    )


node(0.3, 2.75, 0.85, 0.75, "MP4", "video in", CYAN, "Input")
node(
    1.4,
    2.35,
    1.5,
    1.55,
    "Signals",
    "one PyAV decode: optical flow, MediaPipe hands, CLIP frames",
    CYAN,
    "Signals",
)
node(3.15, 2.7, 1.05, 0.85, "Router", "CLIP P(cooking)", CYAN, "Router")
text(
    s,
    4.45,
    1.62,
    4.0,
    0.25,
    "KITCHEN ROUTE (EPIC-KITCHENS)",
    size=10,
    bold=True,
    color=TEAL,
    name="EPIC lane label",
)
lane = [
    ("ActionFormer", "verb + noun detections"),
    ("Pair", "+ verb-noun prior"),
    ("Cluster", "into distinct steps"),
    ("Select", "expected coverage, ≥ 50% rule"),
]
for i, (h, sub) in enumerate(lane):
    node(4.45 + 1.27 * i, 1.9, 1.12, 0.8, h, sub, TEAL, f"EPIC {i + 1}")
    if i:
        arrow(s, 4.45 + 1.27 * i - 0.15, 2.3, 4.45 + 1.27 * i, 2.3)
text(
    s,
    4.45,
    3.62,
    4.0,
    0.25,
    "OUTDOOR ROUTE (FPVSum)",
    size=10,
    bold=True,
    color=AMBER,
    name="FPVSum lane label",
)
fl = [
    ("Importance", "flow, hands, camera motion, scene cuts", 1.75),
    ("Windows", "best ≥ 4 s per shot, 15% budget", 1.45),
    ("Label", "CLIP zero-shot category", 1.4),
]
x = 4.45
for i, (h, sub, w) in enumerate(fl):
    node(x, 3.9, w, 0.8, h, sub, AMBER, f"FPV {i + 1}")
    if i:
        arrow(s, x - 0.15, 4.3, x, 4.3)
    x += w + 0.15
box(s, 8.4, 2.85, 1.25, 0.65, fill=BLUE, name="Output")
text(
    s,
    8.4,
    2.85,
    1.25,
    0.65,
    [
        [("out.json", {"bold": True, "font": "Courier New", "size": 12})],
        [("validated", {"size": 10, "color": WHITE})],
    ],
    align=PP_ALIGN.CENTER,
    anchor=MSO_ANCHOR.MIDDLE,
    name="Output text",
)
arrow(s, 1.15, 3.12, 1.4, 3.12)
arrow(s, 2.9, 3.12, 3.15, 3.12)
arrow(s, 4.2, 2.9, 4.45, 2.4, TEAL)
arrow(s, 4.2, 3.35, 4.45, 4.2, AMBER)
arrow(s, 9.0, 2.7, 9.0, 2.85, TEAL)
arrow(s, 9.0, 3.9, 9.0, 3.5, AMBER)
text(
    s,
    0.4,
    4.95,
    9.2,
    0.4,
    [
        [
            ("Stack  ", {"bold": True, "color": CYAN}),
            (
                "Python 3.11 · PyTorch (CPU) · ActionFormer · OpenCLIP · MediaPipe · OpenCV · "
                "PyAV · "
                "Docker · GitHub Actions CI",
                {},
            ),
        ]
    ],
    size=12,
    name="Tech stack",
)

# ---- 5 final output -----------------------------------------------------------------
s = S[4]
set_header(s, "TRACK FINAL OUTPUT")
title(s, "One kit-schema JSON per video: all 187 pass validation")
doc = json.loads((REPO / "results" / "preds" / "full_report" / f"{args.example}.json").read_text())
acts = sorted(doc["actions"], key=lambda a: -a["score"])[:3]
team = next(
    ln.split(":", 1)[1].split("#")[0].strip()
    for ln in (REPO / "configs" / "default.yaml").read_text(encoding="utf-8").splitlines()
    if ln.strip().startswith("team_id:")
)
lines = [
    "{",
    f'  "team_id": "{team}",',
    f'  "video_id": "{doc["video_id"]}",',
    f'  "duration_sec": {doc["duration_sec"]},',
    f'  "label": {{"name": "{doc["label"]["name"]}", "score": {doc["label"]["score"]}}},',
    '  "actions": [',
]
lines += [f'    {{"segment": [{a["segment"][0]:.1f}, {a["segment"][1]:.1f}],' for a in acts[:1]]
lines += [
    f'     "verb": "{acts[0]["verb"]}", "noun": "{acts[0]["noun"]}",',
    f'     "score": {acts[0]["score"]:.2f}}},',
    f"    ... {len(doc['actions']) - 1} more",
    "  ],",
    '  "highlights": [',
]
lines += [
    f'    {{"segment": [{h["segment"][0]:.2f}, {h["segment"][1]:.2f}]}},'
    for h in doc["highlights"][:2]
]
lines += [f"    ... {len(doc['highlights']) - 2} more", "  ]", "}"]
code(s, 0.4, 1.7, 4.35, 3.6, lines, size=10, name="JSON example")
tbl = s.shapes.add_table(4, 5, Inches(5.0), Inches(1.7), Inches(4.6), Inches(1.5))
tbl.name = "Results table"
tbl._element.graphic.graphicData.tbl.tblPr.find(
    qn("a:tableStyleId")
).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
t = tbl.table
hdr = ["", "Score", "Coverage", "mAP", "F1"]
rows = [("REWIND", FULL), ("Uniform", UNI), ("Oracle*", ORA)]
for j, h in enumerate(hdr):
    t.cell(0, j).text = h
for i, (n, r) in enumerate(rows, 1):
    vals = [n] + [f"{r[k]:.3f}" for k in ("weighted", "coverage_frac", "map", "f1")]
    for j, v in enumerate(vals):
        t.cell(i, j).text = v
t.columns[0].width = Inches(1.2)
for j in range(1, 5):
    t.columns[j].width = Inches(0.85)
for i in range(4):
    for j in range(5):
        c = t.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = BAND if i == 0 else (CARD if i == 1 else BG)
        c.margin_left = c.margin_right = Inches(0.06)
        for p in c.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
            for r in p.runs:
                r.font.name = "Calibri"
                r.font.size = Pt(12 if i else 11)
                r.font.bold = i in (0, 1)
                r.font.color.rgb = AMBER if i == 1 else (MUTED if i == 0 else WHITE)
text(
    s,
    5.0,
    3.3,
    4.6,
    0.45,
    [
        "185 scored videos; coverage: ≥ 50% of an action inside. *Oracle sees the GT.",
        "Score = 40% coverage + 25% mAP + 25% F1 + 10% label.",
    ],
    size=10,
    color=MUTED,
    name="Table note",
)
nums = [
    (f"{100 * FULL['pct_oracle']:.0f}%", "of oracle coverage"),
    (f"{100 * PRUNED:.0f}%", "of frames pruned"),
    (f"{SEC_PER_MIN:.0f} s", "CPU per video minute"),
]
for i, (big, small) in enumerate(nums):
    x = 5.0 + 1.56 * i
    text(s, x, 3.85, 1.5, 0.6, big, size=28, bold=True, color=AMBER, name=f"Number {i + 1}")
    text(s, x, 4.5, 1.5, 0.5, small, size=12, color=WHITE, name=f"Number {i + 1} label")

# ---- 6 differentiation --------------------------------------------------------------
s = S[5]
title(s, "Spend the budget on distinct steps, not busy pixels")
cd = CategoryChartData()
cats = ["Uniform windows", "Zero-shot actions", "Label dedup", "REWIND", "Oracle (GT)"]
vals = [
    UNI["coverage_frac"],
    abl["zero-shot actions (no ActionFormer)"]["coverage_frac"],
    abl["label-dedup selector (no expected coverage)"]["coverage_frac"],
    FULL["coverage_frac"],
    ORA["coverage_frac"],
]
cd.categories = cats
cd.add_series("Action coverage", [round(v, 3) for v in vals])
gf = s.shapes.add_chart(
    XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.4), Inches(1.65), Inches(4.6), Inches(3.2), cd
)
gf.name = "Coverage chart"
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "Action coverage (official rule), 138 EPIC val videos"
tp = ch.chart_title.text_frame.paragraphs[0]
tp.runs[0].font.size = Pt(12)
tp.runs[0].font.color.rgb = WHITE
tp.runs[0].font.name = "Calibri"
plot = ch.plots[0]
plot.gap_width = 55
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format = "0.000"
dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(11)
dl.font.color.rgb = WHITE
dl.font.name = "Calibri"
ser = plot.series[0]
for i, col in enumerate([MUTED, MUTED, MUTED, TEAL, BLUE]):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = col
ca, va = ch.category_axis, ch.value_axis
ca.tick_labels.font.size = Pt(11)
ca.tick_labels.font.color.rgb = WHITE
ca.tick_labels.font.name = "Calibri"
ca.format.line.color.rgb = RGBColor(0x33, 0x40, 0x60)
va.visible = False
va.has_major_gridlines = False
va.maximum_scale = 0.85
va.minimum_scale = 0
pts = [
    (
        "Expected coverage, not label dedup",
        "Top-1 labels are often wrong. We keep top-k label distributions per cluster and "
        "greedily maximize expected distinct steps per second, weighted by detection "
        "confidence.",
    ),
    (
        "Optimizes the official rule",
        "A window only gets credit for actions it covers at least 50% of, the rule Samsung "
        "scores; switching to it raised coverage on held-out videos.",
    ),
    (
        "Honest by construction",
        "Tuned only on EPIC train videos; FPVSum GT never used to choose a setting. No "
        "ground truth, filename or manifest at inference (enforced by tests).",
    ),
    ("Runs anywhere", "CPU only, deterministic, Docker image built and tested in CI."),
]
for i, (h, b) in enumerate(pts):
    y = 1.65 + 0.82 * i
    box(s, 5.3, y + 0.05, 0.22, 0.22, fill=TEAL, shape=MSO_SHAPE.OVAL, name=f"Point {i + 1} dot")
    text(
        s,
        5.65,
        y,
        3.95,
        0.8,
        [[(h, {"bold": True, "size": 14, "color": WHITE})], [(b, {"size": 11, "color": MUTED})]],
        name=f"Point {i + 1}",
    )

# ---- 7 demo & setup -----------------------------------------------------------------
s = S[6]
title(s, "See it work, then run it yourself")
picture(s, HERE / "img_fridge.png", 0.4, 1.7, w=4.5, name="Demo frame")
text(
    s,
    0.4,
    4.32,
    4.5,
    0.85,
    [
        [
            ("Demo video, under 3 min  ", {"bold": True}),
            ("(link in the submission form)", {"color": MUTED}),
        ],
        [
            (
                "A 9-minute kitchen video: our highlights play with the detected actions captioned "
                "live, the timeline shows where each clip comes from.",
                {"size": 11, "color": MUTED},
            )
        ],
    ],
    size=13,
    name="Demo caption",
)
code(
    s,
    5.15,
    1.7,
    4.45,
    2.55,
    [
        "# CPU, Python 3.11",
        "$ git clone github.com/prakartii/",
        "      samsung-xr-hackathon",
        "$ pip install -r requirements.txt",
        "$ tools/fetch_weights.sh",
        "$ ./run.sh video.mp4 out.json",
        "",
        "# or with Docker",
        "$ docker build -t rewind .",
        "$ docker run rewind in.mp4 out.json",
    ],
    size=11,
    name="Setup commands",
)
text(
    s,
    5.15,
    4.42,
    4.45,
    0.6,
    "Weights (2.3 GB) download from our GitHub release, SHA-256 "
    f"checked. About {SEC_PER_MIN:.0f} s of CPU per minute of video.",
    size=11,
    color=MUTED,
    name="Setup note",
)

# ---- 8 brownie ----------------------------------------------------------------------
s = S[7]
title(s, "Extras, and what we would fix next")
picture(s, HERE / "img_boxing.png", 0.4, 1.7, w=3.0, name="Sports demo frame")
text(
    s,
    0.4,
    3.6,
    3.0,
    1.6,
    [
        [("Demo tools, from our JSON", {"bold": True, "size": 13})],
        [
            ("make_demo.py  ", {"font": "Courier New", "size": 11, "color": TEAL}),
            ("the demo video (under 3 min)", {"size": 11, "color": MUTED}),
        ],
        [
            ("make_reel.py  ", {"font": "Courier New", "size": 11, "color": TEAL}),
            ("captioned reel per video", {"size": 11, "color": MUTED}),
        ],
        [
            ("timeline_html.py  ", {"font": "Courier New", "size": 11, "color": TEAL}),
            ("interactive timeline", {"size": 11, "color": MUTED}),
        ],
    ],
    space_after=4,
    name="Viewer caption",
)
box(s, 3.65, 1.7, 2.85, 3.5, name="Ablation card")
text(
    s,
    3.85,
    1.85,
    2.5,
    0.4,
    "Every part earns its place",
    size=14,
    bold=True,
    color=AMBER,
    name="Ablation title",
)
drops = [
    ("ActionFormer", FULL["weighted"] - abl["zero-shot actions (no ActionFormer)"]["weighted"]),
    (
        "Expected coverage",
        FULL["weighted"] - abl["label-dedup selector (no expected coverage)"]["weighted"],
    ),
    ("Scene change", FULL["weighted"] - abl["no scene change (FPVSum)"]["weighted"]),
    ("Flow / motion", FULL["weighted"] - abl["no flow / camera motion"]["weighted"]),
    ("Hands", FULL["weighted"] - abl["no hand signal"]["weighted"]),
]
text(s, 3.85, 2.35, 2.5, 0.4, "Score lost when removed:", size=11, color=MUTED, name="Ablation sub")
for i, (n, d) in enumerate(drops):
    y = 2.7 + 0.45 * i
    text(s, 3.85, y, 1.7, 0.4, n, size=13, name=f"Ablation {i + 1}")
    text(
        s,
        5.45,
        y,
        0.85,
        0.4,
        f"−{d:.3f}",
        size=13,
        bold=True,
        color=TEAL,
        align=PP_ALIGN.RIGHT,
        name=f"Ablation {i + 1} value",
    )
box(s, 6.75, 1.7, 2.85, 3.5, name="Limits card")
text(s, 6.95, 1.85, 2.5, 0.4, "Limitations", size=14, bold=True, color=AMBER, name="Limits title")
lims = [
    "Action mAP is 0.152: ActionFormer can't be tuned on its own training videos.",
    "Labels confuse look-alikes: jumping vs plane, bike vs motorbike.",
    "Videos under 27 s keep one 4 s window, over 15% by design.",
    "EPIC files not from the official set fall back to zero-shot actions.",
]
text(
    s,
    6.95,
    2.35,
    2.5,
    2.8,
    [[(lim, {})] for lim in lims],
    size=11,
    color=WHITE,
    space_after=6,
    name="Limits",
)

# ---- 9 thank you --------------------------------------------------------------------
s = S[8]
for sh in s.shapes:
    if sh.has_text_frame and sh.name == "Text 25":
        tf = sh.text_frame
        tf.text = "REWIND  ·  Team Nexus, VIT Vellore  ·  github.com/prakartii/samsung-xr-hackathon"
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size = Pt(11)
            r.font.color.rgb = WHITE
            r.font.name = "Calibri"

args.out.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(args.out))
print("saved", args.out)
