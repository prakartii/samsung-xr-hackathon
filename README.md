# Samsung XR Hackathon 2026 — Track 4 (Action-Aware Video Summarization)

**Project:** REWIND — Coverage-First Egocentric Memory Summarizer.
**Team track:** Track 4. **Final submission:** 18 Nov 2026 (no extension).

## Start here
- **`CLAUDE.md`** — how we work: rules, commands, testing standard. Read first.
- **`ARCHITECTURE.md`** — the build spec: modules, interfaces, algorithms, schema.
- **`PROGRESS.md`** — live status: phases, owners, checklists, results. Updated in every PR.
- `REWIND_v2.3_plan.md` — design rationale (frozen).
- `REWIND_v2.2_plan.md` — previous version, **superseded** by v2.3 (kept for history).
- `REWIND_v2_plan.md` — earlier version, **superseded** (kept for history).

## TL;DR of what we're building
A system that turns a first-person video into an ~15% highlight reel of **≥4-second
windows placed where the most distinct actions happen**, chosen by budget-constrained
**expected-coverage** optimization over clustered action proposals. Outputs one JSON
per video in the kit's schema (`team_id`, `video_id`, `duration_sec`, `label`,
`actions[]`, `highlights[]`; see plan §10).

## The 5 decisions that matter most
1. **Distinct-action windows:** highlights under 4 s are ignored in scoring and a median video gets only ~8 windows, so window placement is the main lever on coverage (40%).
2. **Expected coverage over clustered proposals**, not top-1 label dedup, with cost = new seconds added; plus hedge windows for detector misses.
3. **Committed stack:** ActionFormer EPIC-100 checkpoint on its own training features, ranked by the required `score`; no per-frame HOI detector (MediaPipe hands + optical flow instead).
4. **Scorer before selector:** our own validator, coverage/mAP/F1 scorer and oracle-coverage ceiling first; tune on EPIC *train* videos, report on val.
5. **Kit rules first:** the rules in plan §3.1 are second-hand; diff them against our own kit copy on day 1 and email the open questions (§3.2).

## Scoring (design to this)
Action coverage 40% · Action mAP 25% · Highlight F1 25% · Label accuracy 10%.

## Status
Architecture **locked** (`ARCHITECTURE.md` v1.0, 2026-11-11). Build starts: P0 (kit
check, email open questions), then P1 (B) and P3 (A) in parallel. Live status in
`PROGRESS.md`.
