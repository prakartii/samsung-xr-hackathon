# REWIND v2 — Coverage-First Egocentric Memory Summarizer

**Track 4 — Action-Aware Video Summarization · Samsung XR Hackathon 2026**
*Remember what matters. Rewind the rest.*

---

## 0. One-line positioning (the reframe)

REWIND turns a 2–10 min first-person video into a ~15% highlight reel that **captures the maximum number of distinct actions the budget allows** — framed as a **budget-constrained coverage optimization** over hand–object-grounded action segments, with a reconstructable **procedural task-graph** as the memory output.

> Old framing led with "uncertainty-aware boundary refinement." That targets boundary precision — but **mAP is scored at tIoU 0.1–0.5 (loose overlap), so boundary precision barely scores.** The real lever is **coverage (40%)**. v2 leads with the coverage optimizer; boundary trimming is demoted to a budget-efficiency step.

---

## 1. Where the points actually are (design to the rubric)

| Metric | Weight | Lives in | What wins it |
|---|---|---|---|
| **Action coverage** | **40%** | `highlights[]` | Capture the most **distinct** (verb,noun) pairs within 15% → submodular max-coverage, dedupe repeats, spend budget on rare actions |
| **Action mAP** | 25% | `actions[]` | **High-recall** detection across the *whole* video at loose tIoU — recall > boundary precision |
| **Highlight F1** | 25% | `highlights[]` (FPVSum) | Match top-15% human importance frames — importance predictor on FPVSum |
| **Label accuracy** | 10% | `label` (FPVSum) | Video-level category — cheap CLIP zero-shot / small classifier |

**Key structural insight:** `actions[]` and `highlights[]` are **two different outputs with two different objectives.**
- `actions[]` → detect **everything** (maximize recall for mAP).
- `highlights[]` → **select** a diverse 15% subset (maximize distinct coverage + importance).

**Dataset-adaptive scoring** (unified schema, different objective per source):
- **EPIC-KITCHENS-100 (138 videos):** optimize coverage + mAP.
- **FPVSum (49 videos):** optimize highlight F1 + label.

---

## 2. Architecture — 7 stages (fixes applied)

**Stage 0 — Submission contract (DO THIS FIRST).**
`run.sh <video.mp4> <out.json>` + a schema-valid JSON writer that passes `validate.py --dir outputs/` on dummy data **on day 1**. Brief says non-compliance = **direct disqualification**. Lock the contract before any ML.

**Stage 1 — HOI idle-gating** *(re-added from v1 loss).*
Hand–object interaction detector flags frames with active manipulation; drop no-contact/idle frames early. Cheap, directly serves "drop idle footage," and anchors the **XR token-pruning** narrative.

**Stage 2 — High-recall action detection → `actions[]`** *(targets 25% mAP).*
Pretrained egocentric features + temporal action localization → (verb, noun, start, end) with confidences. Tune for **recall** at tIoU 0.1–0.5, not tight boundaries.

**Stage 3 — Importance prediction → FPVSum** *(targets 25% F1).*
Light regressor on frozen features (or use FPVSum's provided importance) to score per-segment human-perceived importance.

**Stage 4 — Budget-constrained submodular coverage selection → `highlights[]`** *(the crown jewel, targets 40%).*
Greedy submodular maximization of marginal gain:

```
gain(seg) = α·(new distinct verb–noun pairs covered)
          + β·(human importance)      # FPVSum weight
          + γ·(action confidence)
          − δ·(visual/temporal redundancy vs already-selected)
subject to  Σ duration(selected) ≤ 0.15 · T
```

Greedy with lazy evaluation; dedupe repeated actions; prioritize the long tail of rare pairs. Provable (1−1/e) near-optimality → credibility.

**Stage 5 — Scene/category classifier → `label`** *(targets 10%).*
CLIP zero-shot over FPVSum categories, or a tiny head on frozen features. Cheap points.

**Stage 6 — Boundary trimming** *(demoted: budget efficiency only).*
Trim segments to the high-activity core so each action costs less of the 15% budget → **more distinct actions fit → higher coverage.** Justified by budget, not mAP.

---

## 3. Differentiators (pick to stand out — don't build all)

- **Procedural task-graph (recommended demo hook):** emit a graph of sub-actions / object-state transitions — the literal "procedural memory" the branding promises. Reasoning about *distinct steps* also helps coverage. Big "wow" in the live round.
- **On-device / efficiency tiebreaker (recommended for this jury):** report measured latency/throughput (ideally Galaxy-class). Samsung is an on-device company — few teams will bother.
- **Coverage-optimality analysis:** chart showing the selector reaches within X% of the **oracle** max coverage achievable in 15% (submodular bound). One chart that earns jury trust.

---

## 4. Concrete components (close the "name real models" gap)

*All licenses to be verified before use — see Day 0 tasks.*

- **Egocentric features / recognition:** VideoMAE-v2 or InternVideo2 features; **LaViLa / EgoVLP** for verb–noun semantics (strong on EPIC).
- **Temporal action localization:** **ActionFormer** (feature-based, fast to stand up) on the above features.
- **Hand–object interaction:** **100DOH** hand-object detector or **EgoHOS** (egocentric HOI).
- **Video-LLM (optional, for semantics/label):** Qwen2-VL / LLaVA-Video / InternVideo2-Chat.
- **Label / zero-shot category:** CLIP.
- **Eval tooling:** official `validate.py` + the student kit (`manifest.csv`, `supported_classes.json`, `examples/`).

**Build-vs-train stance:** stand on pretrained weights; do **not** train a detector from scratch in 10 days. Leverage is in **selection + engineering + narrative**, not recognition SOTA.

---

## 5. 10-day build plan (Nov 8 → Nov 18 submission)

- **Day 0 (TODAY, Nov 8):** Register team before 11:59 PM. Scaffold repo, `run.sh` + schema writer passing `validate.py` on dummy output. Pull datasets + student kit. Verify model licenses.
- **Day 1–2:** Pretrained action detection → `actions[]` on EPIC; measure mAP. Integrate HOI idle-gating.
- **Day 3–4:** Submodular coverage selector → `highlights[]`; tune α/β/γ/δ and 15% budget. Measure coverage.
- **Day 5:** FPVSum importance predictor (F1) + CLIP label (10%).
- **Day 6:** Task-graph differentiator + explainable timeline / original-vs-summary UI.
- **Day 7:** Efficiency pass — latency measurement + optimization, on-device framing.
- **Day 8:** Full run on all 187 videos; `validate.py` clean; ablations (naive top-k vs submodular; oracle-coverage bound).
- **Day 9:** 3-min demo video + deck (`CollegeName_TeamName`) + README with reproducible setup.
- **Day 10 (buffer):** Final checks, submit via Google Form. **No extension — do not use this day for building.**

---

## 6. Roles (team of up to 4)

- **A — ML lead:** action detection + features (Stages 2, 3).
- **B — Algorithms:** submodular selection + boundary trimming + coverage analysis (Stages 4, 6, differentiator #3).
- **C — Perception/FPVSum:** HOI gating, importance, label (Stages 1, 5).
- **D — Infra/demo:** submission contract, `run.sh`, `validate.py` compliance, demo UI, deck, video (Stage 0, differentiator #1 viz).

---

## 7. MVP vs stretch

- **MVP (must ship):** compliant submission; pretrained detection → `actions[]`; submodular selection → `highlights[]`; CLIP label; basic FPVSum importance; clean run on all 187 videos.
- **Stretch (if MVP solid):** procedural task-graph; on-device latency; calibration/uncertainty polish; richer UI.

> Calibration/uncertainty (v1's headline) = **stretch only.** Not directly scored; don't let it eat MVP time.

---

## 8. Risks & mitigations

- **Format DQ** → Stage 0 on day 1; re-run `validate.py` in CI.
- **Model/weights integration pain** → feature-based (ActionFormer) with a CLIP-zero-shot fallback path.
- **Runtime on 187 videos** → precompute features; cap per-video compute; measure early.
- **FPVSum tiny (49 vids)** → prefer zero-shot/light heads; don't overfit.
- **Scope creep** → freeze MVP by Day 5; differentiators only after.

---

## 9. Jury narrative (how to tell it)

"Egocentric video is thousands of visual tokens, most of them idle or redundant. REWIND is an **on-device perception pre-filter**: HOI-gate the idle, detect every action, then solve a **budget-constrained coverage optimization** to keep the 15% that captures the most *distinct* steps — output as a reconstructable **procedural memory graph**. Within X% of the theoretical coverage maximum, at Y ms/frame on a Galaxy device."
