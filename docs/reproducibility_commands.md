# Reproducibility video — exact commands (run 2026-11-22, Git Bash on Windows 11, CPU)

Captured by `demo/repro/capture.py` (all output in `demo/repro/capture.json`), rendered by
`demo/repro/render.py`. Home-directory paths are redacted to `~`; nothing else is edited.

| # | Where | Command shown | Exit | Time |
|---|---|---|---|---|
| 1 | fresh temp dir | `git clone --depth 1 https://github.com/prakartii/samsung-xr-hackathon rewind` (run with credential helper off, no prompt: anonymous) | 0 | 6.5 s |
| 2 | clone | `git log -1 --format='%h  %s' && ls` → c165bb9 (main) | 0 | 0.9 s |
| 3 | clone | `sed -n 78,83p README.md` (documented setup) | 0 | 0.8 s |
| 4 | clone | `curl -sIL ... /releases/download/weights-v1/vn_prior.npy` → HTTP 200, no login | 0 | 2.0 s |
| 5 | prepared repo | `python --version && pip install -q -r requirements.txt` → Python 3.12.3, requirements satisfied (an earlier run of the same command installed 3 pinned packages: PyYAML, numpy, opencv) | 0 | 4.3 s |
| 6 | prepared repo | `tools/fetch_weights.sh` → every weight SHA-256 `ok` | 0 | 25.1 s |
| 7 | prepared repo | `./run.sh data/videos/P02_14.mp4 demo/repro_out/P02_14.json` (kitchen) | 0 | 61.2 s |
| 8 | prepared repo | `python -m json.tool demo/repro_out/P02_14.json \| head -n 32` | 0 | 1.9 s |
| 9 | prepared repo | `cmp demo/repro_out/P02_14.json outputs/P02_14.json` → byte-identical | 0 | 1.9 s |
| 10 | prepared repo | `./run.sh data/videos/Skate_el49s-RlIa4.mp4 ...` (outdoor, label skate) | 0 | 55.6 s |
| 11 | prepared repo | JSON content compare with `outputs/Skate_el49s-RlIa4.json` → True (bytes differ only in CRLF line endings on Windows) | 0 | 0.4 s |
| 12 | prepared repo | `./run.sh data/videos/P03_21.mp4 ...` (kitchen) | 0 | 39.0 s |
| 13 | prepared repo | `cmp demo/repro_out/P03_21.json outputs/P03_21.json` → byte-identical | 0 | 0.5 s |
| 14 | prepared repo | `python -m rewind_eval.validate --dir outputs/` → 187 files, 0 errors, 3 known short-video warnings | 0 | 8.0 s |
| 15 | prepared repo | `tail -n 1 results/leaderboard.csv` → final scores, previously measured (git 9533198, report split) | 0 | 0.7 s |

Source videos are not in the repository (the kit does not redistribute them); `data/videos/`
holds the MP4s downloaded per `kit/manifest.csv`.
