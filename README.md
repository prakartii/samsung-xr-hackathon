# Samsung XR Hackathon 2026 — Track 4 (Action-Aware Video Summarization)

**Project:** REWIND — Coverage-First Egocentric Memory Summarizer.
**Team track:** Track 4. **Final submission:** 18 Nov 2026 (no extension).

## Start here
- **`REWIND_v2.2_plan.md`** — the current canonical build spec. Read it first.
- `REWIND_v2_plan.md` — earlier version, **superseded** by v2.2 (kept for history).

## TL;DR of what we're building
A system that turns a 2–10 min first-person video into an ~15% highlight reel that
preserves the **maximum number of distinct meaningful actions** within the budget,
treated as **budget-constrained coverage optimization** over action-grounded,
hand–object-aware segments. Outputs one schema-valid JSON per video
(`actions[]` + `highlights[]` + `label`).

## The 5 decisions that matter most
1. **Coverage-first selection** is the crown jewel (targets the 40% metric).
2. **Recall-first detection, separate from selection** — but keep proposals confidence-ranked (for mAP).
3. **Confidence calibration = stretch, not core** (temperature scaling is monotonic → can't change mAP).
4. **Start the selector at 3 terms, grow only via ablation** (don't overfit 49 FPVSum videos).
5. **Stage 0 = submission contract** (`run.sh` + `validate.py` passing) before any ML; feature caching is a day-1 task.

## Scoring (design to this)
Action coverage 40% · Action mAP 25% · Highlight F1 25% · Label accuracy 10%.

## Status
Planning complete (v2.2). Next: scaffold the repo (run.sh + schema writer passing
validate.py), lock pretrained checkpoints + licenses, start feature caching.
Work continues on another agent.
