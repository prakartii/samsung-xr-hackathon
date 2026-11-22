# REWIND demo — storyboard (final render, 1:56.7, 1920×1080, 30 fps)

Generator: `demo/rewind_demo_source.py`. Every label, timestamp, score and window on
screen is read at render time from the frozen outputs (`outputs/*.json`, commit ea1e5d8)
and the original MP4s in `data/videos/`. The only typed numbers are the final benchmark
results (README / `results/ablation_report.md`).

| # | Time | Scene | Footage (real) | On screen |
|---|---|---|---|---|
| 1 | 0:00–0:09 | Hook | `Surfing_BKwkLQr9ai8`, REWIND highlight 41.5–45.55 s (inside the wave) | "Hours of first-person video. / Only some moments matter." → **REWIND** · *Action-aware video summarization* |
| 2 | 0:09–0:18 | The problem | `P01_11` (EPIC-KITCHENS-100) fast-forwarded ~62×, 0:02 → 9:19 | running clock "x:xx of 9:21", position bar; "Continuous activity. Every second recorded." / "Finding the actions means watching all of it." |
| 3 | 0:18–0:28 | Video in, JSON out | frame of `P01_11` at 2:07.9 | `P01_11.mp4 · 9:21 · 1920×1080` → `./run.sh P01_11.mp4 P01_11.json` (labelled "frozen pipeline · results precomputed") → `P01_11.json`: 2,000 actions · 14 highlight windows · 1:21 of 9:21 · label cooking 1.00 |
| 4 | 0:28–0:45 | How it works | — (motion graphic) | CLIP router → kitchen lane: Features (SlowFast) → Detection (ActionFormer verb + noun) → Pair + score (learned P(true action)) → Cluster → Select (max expected action coverage) → Highlights (≥ 4 s, 15% budget); sports lane: Signals (motion, hands, scene changes) → Importance → Windows |
| 5 | 0:45–1:04 | Real predictions | `P01_11` at real speed, each clip = predicted segment − 0.7 s … + 0.8 s | **TURN-ON tap** 02:01.3–02:03.3 (0.76) · **TAKE sponge** 02:03.9–02:05.5 (0.71) · **POUR washing liquid** 02:07.4–02:08.9 (0.85) · **WASH knife** 04:09.4–04:13.8 (0.91) · **OPEN bin** 06:58.2–06:59.1 (0.83); label appears when the predicted segment starts; live clock + position in the 9:21 video |
| 6 | 1:04–1:16 | Original → summary | the 14 real `P01_11` highlight windows played back to back | timeline 0–9:21 with the 14 windows lighting up in order; ORIGINAL 9:21 → REWIND SUMMARY 1:21 (counting up); "14 windows · 14.5% of the video" |
| 7 | 1:16–1:31 | Outdoor first-person | one real REWIND highlight each: `Ski_-Z5a4UQmZLs` 82.5–88.05, `Motor_0JZi_gJ7Jfw` 21.0–25.05, `Horse_7ZzWccM_5cM` 98.5–103.05, `Scuba__JnCVbCL5Wg` 133.0–137.55, `RockClimbing_mDUcIkDzpVI` 21.0–29.55, `Surfing_BKwkLQr9ai8` 49.5–53.55 | predicted label + score (SKI 0.92, MOTOR 0.97, HORSE 0.93, SCUBA 0.99, ROCKCLIMBING 0.94, SURFING 0.76), highlight range, the video's highlight timeline |
| 8 | 1:31–1:41 | Actual output | `outputs/P01_11.json` excerpt | video_id, duration, label; the pour / wash actions shown earlier (exact segments and scores); first 3 highlight windows; "Structured, timestamped output." |
| 9 | 1:41–1:50 | Results | — | Coverage 36.38% · mAP 15.22% · Highlight F1 23.25% · Label 85.11% · **Weighted 32.68%** with weights |
| 10 | 1:50–1:56.7 | End | blurred `Ski_-Z5a4UQmZLs` frame | **REWIND** · "Turn long recordings into action-aware summaries." · VITVellore_Nexus · Samsung XR Hackathon 2026 · Track 04 |

Audio: an original synthesised ambient pad (A major, slow LFO) with soft air whooshes on scene cuts;
no voice. The voiceover script for a human recording is in `demo/rewind_voiceover.md`.
