# REWIND v2.2 — Coverage-First Egocentric Memory Summarizer

**Track 4 — Action-Aware Video Summarization · Samsung XR Hackathon 2026**
*Remember what matters. Rewind the rest.*

> **Status:** This is the current canonical build spec (supersedes `REWIND_v2_plan.md`).
> It merges the detailed v2.1 design with the agreed scope-discipline decisions
> (see "Key decisions & rationale"). Read "Scope discipline" before building — the
> main risk is over-engineering for a 10-day window, not the approach.

---

## 1. One-line positioning

REWIND turns a 2–10 min first-person video into an ~15% highlight reel that preserves
the **maximum number of distinct meaningful actions** within the duration budget. It
treats summarization as a **budget-constrained coverage-optimization** over
action-grounded, hand–object-aware segments, producing a compact, explainable
procedural memory. Framed for a Samsung XR jury as an **on-device perception
pre-filter** that prunes thousands of egocentric frames to the 15% that matter.

---

## 2. Scoring — design everything to this

| Metric | Weight | Lives in | What wins it |
|---|---|---|---|
| **Action coverage** | **40%** | `highlights[]` | Capture the most **distinct** (verb,noun) pairs within 15% → coverage-first marginal-gain selection, dedupe repeats, spend budget on rare actions |
| **Action mAP** | 25% | `actions[]` | High-**recall**, confidence-ranked detection across the whole video at tIoU 0.1–0.5 |
| **Highlight F1** | 25% | `highlights[]` (FPVSum) | Match top-15% human-importance frames |
| **Label accuracy** | 10% | `label` (FPVSum) | Video-level category — CLIP zero-shot |

**Structural rule:** `actions[]` and `highlights[]` are two outputs with two objectives.
Detect **everything** (recall) for mAP; **select** a diverse 15% for coverage.

**Dataset split (via input-derived signals only — never filename/GT):**
- **EPIC-KITCHENS-100 (138 videos):** optimize coverage + mAP.
- **FPVSum (49 videos):** optimize highlight F1 + label. (Frame-level importance GT lives here, not on EPIC.)

---

## 3. Key decisions & rationale (so the next agent doesn't re-litigate)

1. **Coverage-first selection is the core.** Coverage is 40% and rewards *distinct*
   actions, so naive top-k importance wastes budget repeating frequent actions.
   Model selection as budget-constrained maximum coverage (submodular, greedy).
2. **Recall-first detection, separate from selection.** But recall must stay
   **confidence-ranked** — mAP is average precision, so unranked false positives
   tank it.
3. **Soft HOI gating, not a hard filter.** Hand–object evidence raises action scores
   and flags idle footage for redundancy removal; it must not clip the start/end of
   a real action.
4. **Confidence calibration = STRETCH, not core.** Precise reason: temperature
   scaling is **monotonic → it cannot change mAP** (ranking is unchanged), and
   coverage dedups by **class identity**, not confidence magnitude. Its only real use
   is cross-class comparability inside the selector — marginal. Do not spend days 1–5
   on it.
5. **Start the selector at 3 terms, grow via ablation.** 9 weighted terms can't be
   tuned on 49 FPVSum videos without overfitting. Begin with
   `coverage_gain + importance + confidence − redundancy`; add a term only when an
   ablation shows it helps. Build the ablation harness early.
6. **Lighter importance model.** A GBM (LightGBM/XGBoost) on 49 videos overfits.
   Prefer pretrained features + a linear head, or an **unsupervised proxy**
   (motion × HOI × action-density) as a strong, safe baseline.
7. **Timebox the UI.** It scores **zero** (metrics are automated); it only helps the
   3-min demo. A clean split-screen (original vs 15% highlight) + coverage bar is
   enough. Do not gold-plate.
8. **Feature caching is a day-1 task.** Compute over 187 videos (some 10 min) is the
   real bottleneck. Precompute and cache all features once.
9. **Keep the on-device / efficiency framing** for the Samsung jury (report latency;
   no cloud dependency) — cheap tiebreaker.

---

## 4. Pipeline (12 stages; MVP vs stretch annotated)

```
First-person MP4
  → Preprocessing + feature extraction (adaptive sampling)        [MVP, cache day-1]
  → Hand–object interaction evidence (soft)                       [MVP]
  → High-recall temporal action detection (confidence-ranked)     [MVP]
  → Verb–noun recognition (constrained to supported_classes)      [MVP]
  → Confidence calibration + uncertainty                          [STRETCH]
  → Human-importance prediction (FPVSum)                          [MVP, keep light]
  → Candidate highlight generation                                [MVP]
  → Coverage-critical, redundancy-aware selection                 [MVP — crown jewel]
  → Boundary trimming (budget efficiency)                         [MVP]
  → Video-level label (CLIP zero-shot)                            [MVP]
  → Structured JSON output (schema-valid)                         [MVP — stage zero]
  → Interactive timeline + highlight reel                         [MVP-lite, timeboxed]
```

**Stage 0 (do first): submission contract.** `run.sh <video.mp4> <out.json>` + a
schema-valid JSON writer that passes the official `validate.py` on dummy data, day 1.
Non-compliant format = disqualification.

---

## 5. Highlight-selection algorithm

Greedy, budget-constrained, **efficiency-ranked** (gain per second, so short
information-dense segments beat long repetitive ones).

**Start minimal (MVP):**
```
gain(seg) =  coverage_gain(new distinct verb–noun pairs)
           + importance(seg)            # FPVSum weight
           + confidence(seg)
           − redundancy(seg vs selected)
efficiency(seg) = gain(seg) / duration(seg)
select argmax efficiency until Σ duration ≤ 0.15 · T
then: merge nearby intervals, remove overlaps, trim low-value boundaries, validate budget.
```

**Grow only via ablation** (candidates to add if they help): action novelty,
temporal diversity, visual/semantic diversity, uncertainty penalty, duration
efficiency weighting.

Keep the 15% budget **conservative** (coverage has a soft penalty for exceeding it).

---

## 6. Output schema (unified across datasets)

```json
{
  "video_id": "P01_01",
  "label": "category",
  "actions":   [ { "verb": "take", "noun": "knife", "start": 12.4, "end": 15.9 } ],
  "highlights":[ { "start": 40.2, "end": 58.0, "importance": 0.93 } ]
}
```

---

## 7. Concrete components (stand on pretrained — do NOT train from scratch)

*Verify licenses day 0.*
- **Egocentric features / recognition:** VideoMAE-v2 or InternVideo2 features; LaViLa / EgoVLP for verb–noun.
- **Temporal action localization:** ActionFormer (feature-based, fast to stand up).
- **Hand–object interaction:** 100DOH or EgoHOS.
- **Label / zero-shot category:** CLIP.
- **Importance (FPVSum):** pretrained features + linear head, or unsupervised proxy (motion × HOI × action-density). Avoid heavy GBM.
- **Eval tooling:** official `validate.py` + student kit (`manifest.csv`, `supported_classes.json`, `examples/`).

**Stack:** Python 3.11, PyTorch, OpenCV, FFmpeg, NumPy, Pandas, scikit-learn,
ONNX Runtime, CLIP, pretrained EPIC action-detection + HOI models, lightweight web timeline/viewer.

---

## 8. Evaluation strategy

Official metrics: coverage 40 / mAP 25 / highlight-F1 25 / label 10.

**Baselines to beat:** uniform 15% selection, motion-saliency, scene-change,
action-only, importance-only.

**Ablations (also justify selector complexity):** action detection, HOI evidence,
importance, redundancy suppression, coverage-aware selection, (calibration),
boundary trimming.

Strict dev/eval separation to prevent leakage.

---

## 9. MVP vs not-to-build

**MVP (must ship):** run.sh + schema-valid JSON for every video; preprocessing +
cached features; high-recall detection → `actions[]`; verb–noun; light importance;
coverage-critical selection → `highlights[]`; redundancy suppression; boundary
trimming; CLIP label; validation vs schema + budget; minimal interactive timeline.

**Stretch (only if MVP solid):** confidence calibration/uncertainty; extra selector
terms; richer UI; on-device latency benchmark; procedural task-graph.

**Not building:** 3D reconstruction, live XR tracking, LLM-based selection,
generative video, conversational assistant, graph DB, mobile/headset deployment,
cloud inference.

---

## 10. Timeline (Nov 8 launch → Nov 18 final submission)

| Day | Milestone |
|---|---|
| Day 0 | Register; scaffold repo; `run.sh` + JSON writer + `validate.py` passing on dummy; **start feature caching**; verify model licenses |
| Day 1–2 | Preprocessing, feature extraction (cached), high-recall detection baseline → measure mAP; integrate soft HOI |
| Day 3–4 | Coverage-critical selector (3-term) + redundancy suppression → measure coverage; build ablation harness |
| Day 5 | FPVSum importance predictor (light) + CLIP label |
| Day 6 | Minimal explainable timeline + highlight viewer (timeboxed) |
| Day 7 | Full-dataset eval, threshold tuning, ablations (grow selector only if they help) |
| Day 8 | Reproducible code, weights, README, Docker, demo video; efficiency/latency numbers |
| Day 9 | Deck + final submission package (`CollegeName_TeamName`) |
| Day 10 | Final validation, buffer, submit (no extension) |

---

## 11. Team roles (4)

- **ML lead:** action detection, verb–noun, (calibration), EPIC eval.
- **Algorithms lead:** coverage-critical selection, redundancy, boundary trimming, ablations.
- **Perception lead:** HOI evidence, importance, label, FPVSum eval.
- **Product/infra lead:** run.sh, JSON validation, reproducibility, timeboxed UI, demo video, deck.

---

## 12. Risks & mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Detector underperforms | ↓ mAP & coverage | Pretrained features, high-recall proposals, sliding-window fallback |
| Importance overfits (49 vids) | ↓ highlight F1 | Light/linear or unsupervised proxy; regularize; CV |
| Over-engineered selector | Deadline risk | Start 3-term; grow only via ablation |
| Calibration time-sink | Deadline risk | Treat as stretch; raw scores are fine for mAP |
| Compute/GPU bottleneck | Slow dev | Cache features day-1; CPU-compatible inference |
| JSON/schema fail | DQ | Validation-first; run `validate.py` in CI |
| Budget exceeded (>15%) | Coverage penalty | Conservative target + strict post-processing |
| UI gold-plating | Deadline risk | Timebox; split-screen + coverage bar only |

---

## 13. Pitch (for the deck)

Egocentric video records valuable human experience, but most footage is idle,
repetitive, or hard to review, and existing summarizers pick visually active moments
without understanding what the person did. REWIND preserves the compact set of
original moments needed to understand and reconstruct the key activity: it detects
meaningful hand–object interactions, recognizes verb–noun actions, suppresses
redundancy, and selects a diverse highlight that preserves the most distinct
procedural steps within the budget. It doesn't just make a shorter video — it builds
an **explainable, searchable, reconstructable action memory**, and it runs as a
local on-device perception pre-filter for downstream XR.

*REWIND: Remember what matters. Rewind the rest.*

---

## 14. Open questions for the next agent

- Confirm exactly how the organizers compute **Action coverage** (highlight-span vs
  GT-action-span overlap, and the soft over-budget penalty curve) from the student kit.
- Confirm whether **Highlight F1** is evaluated on EPIC at all, or FPVSum-only.
- Lock the specific pretrained checkpoints + licenses (ActionFormer features, HOI detector, egocentric VLM).
- Decide the unsupervised-vs-learned importance baseline after seeing FPVSum size/quality.
