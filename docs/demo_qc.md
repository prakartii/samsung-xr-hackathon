# REWIND demo — QC (final render, 2026-11-22)

File: `demo/rewind_demo.mp4` (87.6 MB) · generator `demo/rewind_demo_source.py` · storyboard
`demo/rewind_storyboard.md` · narration script `demo/rewind_voiceover.md`.

## Automated checks
| Check | Result |
|---|---|
| Opens and decodes (PyAV) | yes, 3,501 frames decoded, no errors |
| Duration | 116.70 s (1:56.7; target 1:30–2:00) |
| Video | H.264, 1920×1080, 30 fps, yuv420p |
| Audio | AAC, 48 kHz stereo, peak 0.32 (no clipping); original synthesised pad + transition whooshes |
| Black frames | only fade-in frames (4) |
| Frozen footage | none: longest still run in footage scenes = Hook 2, Problem 12 (idle stretch in the source), Montage 0, Summary 13, Sports 1 frames. Long holds only on text/graphic cards (In/Out, Pipeline, JSON, Results) and the slow end-card push-in, by design |
| First render | Hook froze 3.5 s and Summary froze 3.4 s after the clip ran out; fixed (slow-motion hook, playback spread over the scene) and re-rendered |

## Content checks (against the frozen outputs, commit ea1e5d8)
- Every label, timestamp, score, highlight window and duration is read from `outputs/*.json` at render time; nothing is typed in except the five benchmark numbers.
- The five montage actions (turn-on tap, take sponge, pour washing liquid, wash knife, open bin) were checked by eye: a frame at each predicted segment's midpoint shows that action.
- Sports labels (ski, motor, horse, scuba, rockclimbing, surfing) were checked by eye against their highlight frames. The ski clip moved from highlight 1 to highlight 3 because the first carries the uploader's burned-in watermark.
- Summary: 14 windows, 1:21 of 9:21 = 14.5%, from `P01_11.json`.
- Results card: 36.38 / 15.22 / 23.25 / 85.11, weighted 32.68 (README, `results/ablation_report.md`). No claim about ranking.
- In/Out scene is labelled "frozen pipeline · results precomputed": no fake live processing.
- Stills reviewed for text clipping, overlap and contrast at 2.5, 7, 13, 24, 40, 50, 58, 72, 84, 94, 105 and 112 s; fixed: missing arrow glyph, overflowing pipeline boxes, command overlapping the thumbnail, low-contrast captions over bright footage.

## Known limitations
- No voice: the only local TTS voices (SAPI David/Hazel/Zira) are robotic. Captions carry the story; record `demo/rewind_voiceover.md` over it if wanted.
- `demo/` is gitignored: upload the MP4 (YouTube unlisted or Drive) and put the link in the form / README.
- The previous demo is kept at `demo/rewind_demo_v1_backup.mp4`.
