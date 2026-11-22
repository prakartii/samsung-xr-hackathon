# ruff: noqa: E501
"""Run the reproducibility workflow for real and record every command's output with timings.

    python tools/video/repro_capture.py

Each step runs in Git Bash; stdout+stderr lines are stored with their arrival time in
demo/repro_capture.json, which tools/video/repro_render.py replays as the video. Nothing is
typed in by hand. Home-directory paths are redacted to "~" (privacy), nothing else.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASH = r"C:\Program Files\Git\bin\bash.exe"
CLEAN = Path(tempfile.mkdtemp(prefix="rewind_clean_"))
URL = "https://github.com/prakartii/samsung-xr-hackathon"
HOME_RE = re.compile(r"(?i)(c:[\\/]users[\\/]|/c/users/)prakarti\b")

STEPS = [  # (section, cwd, display command, actual command)
    (
        "1 · Public repository (fresh, anonymous clone)",
        CLEAN,
        f"git clone --depth 1 {URL} rewind",
        f"GIT_TERMINAL_PROMPT=0 git -c credential.helper= clone --depth 1 {URL} rewind 2>&1",
    ),
    (
        "1 · Public repository (fresh, anonymous clone)",
        CLEAN / "rewind",
        "git log -1 --format='%h  %s' && ls",
        "git log -1 --format='%h  %s' && ls",
    ),
    (
        "1 · Public repository (fresh, anonymous clone)",
        CLEAN / "rewind",
        "sed -n 78,83p README.md      # the documented setup",
        "sed -n 78,83p README.md",
    ),
    (
        "1 · Public repository (fresh, anonymous clone)",
        CLEAN / "rewind",
        "curl -sIL -o /dev/null -w '%{http_code}\\n' <release>/weights-v1/vn_prior.npy",
        f"curl -sIL -o /dev/null -w 'HTTP %{{http_code}}  (weights release, no login)\\n' {URL}/releases/download/weights-v1/vn_prior.npy",
    ),
    (
        "2 · Environment (prepared machine: Python 3.12, CPU)",
        REPO,
        "python --version && pip install -r requirements.txt",
        ".venv/Scripts/python --version && .venv/Scripts/python -m pip install -r requirements.txt 2>&1 | tail -n 6",
    ),
    (
        "2 · Environment (prepared machine: Python 3.12, CPU)",
        REPO,
        "tools/fetch_weights.sh       # SHA-256 checks every weight",
        "PYTHON=.venv/Scripts/python bash tools/fetch_weights.sh 2>&1 | tail -n 8",
    ),
    (
        "3 · Inference on one video",
        REPO,
        "./run.sh data/videos/P02_14.mp4 demo/repro_out/P02_14.json",
        "mkdir -p demo/repro_out && PYTHON=.venv/Scripts/python ./run.sh data/videos/P02_14.mp4 demo/repro_out/P02_14.json 2>&1 | tail -n 12",
    ),
    (
        "4 · The output JSON",
        REPO,
        "python -m json.tool demo/repro_out/P02_14.json | head -n 32",
        ".venv/Scripts/python -m json.tool demo/repro_out/P02_14.json | head -n 32",
    ),
    (
        "4 · The output JSON",
        REPO,
        "cmp demo/repro_out/P02_14.json outputs/P02_14.json && echo identical to the submitted file",
        "cmp demo/repro_out/P02_14.json outputs/P02_14.json && echo 'byte-identical to the submitted outputs/P02_14.json'",
    ),
    (
        "5 · Validation and tests",
        REPO,
        "python -m rewind_eval.validate --dir outputs/",
        "PYTHONPATH=code .venv/Scripts/python -m rewind_eval.validate --dir outputs/ 2>&1 | tail -n 4",
    ),
    (
        "5 · Validation and tests",
        REPO,
        "pytest -q                    # fast suite",
        ".venv/Scripts/python -m pytest -q -p no:cacheprovider 2>&1 | tail -n 3",
    ),
    (
        "6 · Benchmark results (previously measured, not re-scored here)",
        REPO,
        "tail -n 1 results/leaderboard.csv",
        ".venv/Scripts/python -c \"import csv;r=list(csv.DictReader(open('results/leaderboard.csv')))[-1];"
        "print('  '.join(f'{k}={r[k]}' for k in ('git_sha','split','coverage_frac','map','f1','label_acc')))\"",
    ),
]


def main() -> None:
    log = []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for section, cwd, shown, cmd in STEPS:
        t0 = time.time()
        p = subprocess.Popen(
            [BASH, "-lc", cmd],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        lines = []
        for line in p.stdout:
            lines.append([round(time.time() - t0, 2), HOME_RE.sub("~", line.rstrip("\n"))])
        rc = p.wait()
        dt = round(time.time() - t0, 2)
        print(f"[{rc}] {dt:6.1f}s  {shown}")
        log.append({"section": section, "cmd": shown, "rc": rc, "seconds": dt, "lines": lines})
    (REPO / "demo" / "repro_capture.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("wrote demo/repro_capture.json")


if __name__ == "__main__":
    main()
