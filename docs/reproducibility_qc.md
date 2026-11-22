# Reproducibility video — QC

File: `demo/rewind_reproducibility.mp4` · 1920×1080 · 30 fps · H.264 · silent.
Method: no screen recorder is installed (no OBS, no ffmpeg CLI), so every command was run for
real by `demo/repro/capture.py`, its stdout/stderr captured line by line with timestamps, and
`demo/repro/render.py` replays that capture as a terminal. Long waits are compressed and
labelled ("ran 61.2 s on CPU (shown faster)"); exit codes are shown; no output text is edited
except home-directory paths redacted to `~`.

## What it proves
- Public access: anonymous `git clone` of github.com/prakartii/samsung-xr-hackathon works; the
  weights release answers HTTP 200 without login.
- Setup: `pip install -r requirements.txt` succeeds; `tools/fetch_weights.sh` verifies every
  weight by SHA-256.
- Inference: `./run.sh` ran on three real videos (2 kitchen, 1 outdoor), exit 0 each.
- Reproducibility: P02_14 and P03_21 outputs are byte-identical to the submitted `outputs/`
  files; Skate_el49s-RlIa4 has identical JSON content (Windows writes CRLF line endings there;
  the repo copy is LF).
- Validity: all 187 submitted files pass the validator (0 errors).

## Limitations (stated honestly)
- Not a fully clean machine: the clone is fresh, but inference ran in the prepared repo where
  the Python environment, weights and source MP4s already exist (a 2.3 GB weight download and
  a full dependency install from scratch would take far longer than the video). The prepared
  environment runs Python 3.12.3 (README says 3.11).
- Metrics are not re-scored in the video: scoring the highlight F1 needs the hidden FPVSum
  labels, which we do not use. The final scores are shown as previously measured at the freeze
  (`results/leaderboard.csv`, git 9533198).
- The local `pytest` run was left out: 11 tests in `tests/contract/test_make_submission.py`
  fail on Windows only (bash/exec-bit assumptions); the same suite passes in Linux CI on main.
- No secrets appear: the anonymous clone disables the credential helper; no tokens are printed.
