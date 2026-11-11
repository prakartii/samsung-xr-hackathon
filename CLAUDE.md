# CLAUDE.md — REWIND (Samsung XR Hackathon 2026, Track 4)

Instructions for anyone working in this repo: the two team members and any coding
agent they use. Read this file fully before your first change. It is short on
purpose; the detail lives in the files it points to.

---

## 1. What this project is

REWIND is our Track 4 entry ("Action-Aware Video Summarization"). For each
first-person video it writes one JSON file containing:

- `actions[]`: every detected (verb, noun) action with a time span and a score,
- `highlights[]`: ≥ 4 s windows of original footage totalling ~15% of the video,
  placed where the most *distinct* actions happen,
- `label`: the video category.

Scoring: action coverage 40% · action mAP 25% · highlight F1 25% · label accuracy 10%.
Datasets: EPIC-KITCHENS-100 (138 val videos) and FPVSum (49 videos).
**Final submission: 18 Nov 2026, 11:59 PM IST. No extension.**

## 2. Which document to read for what

| File | Role | Who edits it |
|---|---|---|
| `CLAUDE.md` (this file) | How we work: rules, commands, testing standard | Rarely; both agree first |
| `ARCHITECTURE.md` | **The build spec.** Modules, data types, interfaces, algorithms, output schema, scorer definitions | Whenever the design changes, in the same PR as the code |
| `PROGRESS.md` | **The live status.** Phases, checklists, owners, results, log, open questions | Every PR (see §5) |
| `REWIND_v2.3_plan.md` | Rationale: why we chose this design (its banner lists what `ARCHITECTURE.md` has since superseded) | Frozen; only the banner may change |
| `kit/` | The organizers' student kit (`TEAM_README.md`, `manifest.csv`, `supported_classes.json`, two example JSONs), committed unmodified | Never edit; replace wholesale if they reissue it |

**Order of authority when sources disagree:** official kit files in `kit/` >
answers from the organizers (recorded in `PROGRESS.md` → Open questions) >
`ARCHITECTURE.md` > `REWIND_v2.3_plan.md` > the brochure PDF. If you find a
conflict, fix the lower document (or log it in `PROGRESS.md` if it needs a team
decision). Never silently code around it.

## 3. Repository layout

```
run.sh                    # submission entry point: ./run.sh <video.mp4> <out.json>
requirements.txt          # runtime deps (pinned) — part of the submission
requirements-dev.txt      # pytest, hypothesis, ruff
pyproject.toml            # ruff + pytest config (pythonpath = code)
configs/default.yaml      # every tunable number lives here, nowhere else
code/
  rewind/                 # INFERENCE package (what run.sh runs)
  rewind_eval/            # EVAL package: GT loaders, scorer, oracle, leaderboard
  third_party/            # vendored code (e.g. ActionFormer) with its LICENSE
tools/                    # CLI scripts: cache features, batch runs, tuning, reels
tests/                    # unit/, property/, contract/, integration/, fixtures/
kit/                      # student kit, unmodified
weights/                  # model weights (gitignored; fetched by tools/fetch_weights.sh)
data/                     # videos + annotations (gitignored)
cache/                    # cached features (gitignored)
outputs/                  # one <video_id>.json per video (committed only at Phase 9)
results/leaderboard.csv   # one row per scored run (committed)
```

Full module map and interfaces: `ARCHITECTURE.md` §3–§5.

## 4. Commands

```bash
# setup (Python 3.11, ffmpeg on PATH)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH=code                      # for `python -m rewind…` / `rewind_eval…` (run.sh sets it itself)
tools/fetch_weights.sh                      # downloads into weights/

# checks — run all three before every push
ruff check . && ruff format --check .
pytest                                      # fast suite: unit + property + contract, < 60 s
pytest -m "slow or data or gpu"             # heavy suite: needs data/, cache/ or a GPU

# run one video / everything
./run.sh data/videos/P01_11.mp4 outputs/P01_11.json
python tools/run_all.py --manifest kit/manifest.csv --videos data/videos --out outputs/

# validate + score
python -m rewind_eval.validate --dir outputs/          # our validator (all kit rules)
python kit/validate.py --dir outputs/                  # official one, once the organizers send it (missing from our kit)
python -m rewind_eval.score --pred outputs/ --split report --tag "<short description>"

# review agent (§5): PRs needing attention; mark one handled
python tools/pr_watch.py
python tools/pr_watch.py --done <key>
```

`rewind_eval.score` appends a row to `results/leaderboard.csv` (git SHA, tag,
config hash, every metric). Paste the row into `PROGRESS.md` when a phase finishes.

## 5. How we work (two people)

### Ownership
`PROGRESS.md` assigns every phase an owner: **A (ML & perception)** or
**B (algorithms & infra)**. The owner drives it; the other person reviews its PRs.
Ask before working inside the other person's in-progress phase.

### Branches and PRs
- Branch from `main` as `p<phase>/<short-name>`, e.g. `p5/expected-coverage`.
- Small PRs (a few hundred lines at most), merged the same day when possible.
- The other person reviews, through their review agent (below). Merge only when CI checks
  (ruff + `pytest`) pass.
- Never push directly to `main`, except a `PROGRESS.md`-only log line.
- Commit messages: imperative and specific ("Add 4 s minimum to postprocess").
  **No AI co-author trailers or "generated with" lines anywhere** (commits, PRs,
  code comments, docs).

### Review agents (always on, both sides)

Every coding-agent session on this repo, A's or B's, runs a **review sub-agent** in the
background for the whole session. It writes no production code. It watches PRs,
reviews the other person's PRs, and reports to the **main agent**, which decides what
happens. So when A opens a PR, B's review agent reviews it and B's main agent acts on
the review, and vice versa.

**Start it first**, before any other work. Brief it with this section, your GitHub
login (A = `prakartii`, B = `dakshcodez`) and the path to the venv. If it stops,
restart it.

**Its loop**
1. About every 10 minutes (never a tight loop), run `python tools/pr_watch.py`. Each
   output line is one item:
   - `review:<n>:<sha>`: a new PR, or new commits, from the other person → review it.
   - `own:<n>:…`: one of our PRs got a review, a comment or failing CI → summarize it
     for the main agent.
2. After reporting an item, run `python tools/pr_watch.py --done <key>`. Each head
   commit is reviewed once; new commits produce a new key and a new review.
3. Nothing new → wait and poll again, until the session ends.

**How it reviews** (one PR at a time)
- Check the PR out in its own worktree outside the repo
  (`git worktree add ../rewind-review-<n> <sha>`); never touch the main agent's working
  tree. Remove the worktree when done.
- Review the code itself, not the PR description's claims: read the diff and the code
  around it, and check it against `ARCHITECTURE.md` (interfaces, algorithms), §6 (rules)
  and §7 (tests).
- Run `ruff check`, `ruff format --check` and `pytest` in the worktree.
- Spot-check the new tests: break 2–3 lines of the new code and confirm a test fails
  (§7.3 rule 4).
- Always check: anything that could produce invalid output (R1–R13); any use of ground
  truth, the manifest or the filename at inference; tuning on `report`; a score change
  without a leaderboard row; `PROGRESS.md` not updated; AI attribution; PR size.

**Its report to the main agent**
- A verdict: approve, approve with nits, or request changes.
- Findings, most severe first: severity (blocker / should-fix / nit), `file:line`,
  what's wrong, a concrete input that fails, and a suggested fix (a unified diff when
  small).
- What it ran, and the results.

The review agent never posts to GitHub, pushes, merges or edits a branch itself.

**The main agent decides**, then tells its person what it did:
- **Every review is posted on the PR, from its person's GitHub account** (`gh` is logged
  in as that person: A's agent posts as `prakartii`, B's as `dakshcodez`). Nothing found
  stays private between the agents.
  - Findings or fixes → `gh pr review <n> --request-changes` (any blocker or
    should-fix) or `--comment` (nits only), with `--body-file <file>`. The body lists
    every finding with severity, `file:line`, the failing input and the suggested fix as
    a ```` ```diff ```` block. Findings the main agent disagrees with stay in the
    comment, marked "not requiring a change" with the reason.
  - No findings → `gh pr review <n> --approve` with a one-line summary of what was
    checked (commands run, deliberate breakages caught).
  - Start the body with `Review by <A|B>'s review agent at <short sha>` so the author
    knows which commit was reviewed.
- Don't push to the other person's branch unless they ask. Suggested fixes go in the
  review; the author's agent applies them.
- Merge only when the PR is approved, CI is green on the head commit, there are no open
  blockers and `PROGRESS.md` is updated. Squash-merge and delete the branch.
- For `own:` items: fix and push, then **reply on the PR from its person's account**
  (`gh pr comment <n> --body-file <file>`) listing each finding and what was done: fixed
  in `<sha>`, or not changed and why. That reply plus the new commits triggers a fresh
  review. Tell its person when CI fails.
- Anything that changes the design or a recorded decision goes to its person first
  (`ARCHITECTURE.md` change control).

### PROGRESS.md is updated in every PR — this is mandatory
A PR that changes code or design **must** also update `PROGRESS.md`:
1. **Starting a phase:** set its status to `in progress`, fill in the owner and start date.
2. **Each PR:** tick the checklist items it completes; add a log line at the top of
   the Log (date, initials, what changed, PR number).
3. **Finishing a phase:** set status `done` and fill in the phase's **Result** block:
   the leaderboard row(s), what met or missed the exit check, and any
   **deviations from ARCHITECTURE.md**. If the design changed, update
   `ARCHITECTURE.md` in the same PR.
4. **Blocked:** set status `blocked`, say on what, and add it to Open questions if
   it needs the organizers or a team decision.
5. **Organizer answers:** record them under Open questions with the date and source,
   then update `ARCHITECTURE.md` and the code to match.

Reviewers: reject a PR that skips this. Never delete log entries; correct them with
a new entry.

### Definition of done (any PR)
- Code follows `ARCHITECTURE.md` interfaces, or `ARCHITECTURE.md` is updated to match.
- Tests per §7 are written, and each new test was seen failing against broken code.
- `ruff check`, `ruff format --check` and `pytest` pass locally.
- `PROGRESS.md` is updated (above).
- If behavior affecting scores changed: a scored run on the `tune` split, row in the
  leaderboard, numbers in the PR description.

## 6. Engineering rules (non-negotiable)

1. **No ground truth at inference.** `code/rewind/` must never import
   `rewind_eval`, read annotation files, or branch on GT. A test enforces this.
   Inference reads only the MP4 (plus config and weights). It never reads
   `kit/manifest.csv` and never derives the route, label or anything else from the
   filename: organizers re-run `run.sh` in an isolated environment, and hardcoded
   video IDs disqualify (kit §6). FPVSum filenames even contain the category
   (`Ski_…`). `video_id` is only echoed into the JSON. Tests enforce this.
2. **No tuning on the report split.** Hyperparameters are chosen on `tune`
   (~20 EPIC *train* videos + FPVSum cross-validation folds). `report` (138 EPIC val +
   FPVSum) is for measuring only. Record any exception in `PROGRESS.md`.
3. **Every tunable number lives in `configs/default.yaml`**, with a comment saying
   what it controls. No magic numbers in code. Kit rules (4.0 s minimum, 15% budget,
   ±0.5 s duration tolerance, 2000 action cap) live in `rewind/constants.py`, not in
   config: they're facts, not knobs.
4. **Time is float seconds everywhere.** Segments are `(start, end)` with
   `0 ≤ start < end ≤ duration`. Convert frames ↔ seconds only in `rewind/video_io.py`.
5. **Pure core, thin edges.** Selection, post-processing, pairing, scoring and
   validation are pure functions on plain data (dataclasses, numpy arrays): no file or
   model access inside. I/O and models live at the edges (`video_io`, `features`,
   `detect`, CLI). This is what makes them testable.
6. **Deterministic.** Seed everything (`rewind.utils.seed_all`). Same input + same
   config + same weights → byte-identical JSON. Sort lists before writing.
7. **Never emit invalid output.** `run.sh` validates before writing. If any stage
   fails, it writes the safe fallback (uniform ≥ 4 s windows, empty actions, a
   default label), logs the error to stderr, and exits 0. A missing JSON costs more
   than a weak one.
8. **Caches are versioned.** Each feature cache directory has a `meta.json` with an
   extractor version string. Change the extractor → bump the version → old caches are
   ignored. Never hand-edit caches.
9. **Don't commit large or licensed data.** Videos, annotations, cache and weights are
   gitignored. `tools/fetch_weights.sh` and `tools/fetch_annotations.sh` document the
   exact sources and versions.
10. **Licenses.** Record every pretrained model and vendored repo with its license in
    `ARCHITECTURE.md` §10 before using it.

## 7. Writing high-quality tests

Tests are how two people change the same pipeline in parallel without breaking each
other's scores. The scorer and validator matter most: every tuning decision trusts
the scorer, and a validator miss means disqualification.

### 7.1 Test tiers

| Tier | Location | Marker | Runs in default `pytest` | What belongs here |
|---|---|---|---|---|
| Unit | `tests/unit/` | — | yes | One pure function, hand-computed expected values |
| Property | `tests/property/` | — | yes | Invariants over many generated inputs (Hypothesis) |
| Contract | `tests/contract/` | — | yes | `run.sh` on tiny synthetic videos → valid JSON; schema shape; import boundaries |
| Integration | `tests/integration/` | `slow`, `data`, `gpu` | no | Real models on 1–3 real videos; metric regression floors |

Default suite: under 60 s on a laptop CPU, no network, no GPU, no files from
`data/`. Tests needing those are marked and **skip with a clear reason** when the
resource is missing.

### 7.2 What must be tested (minimum)

- **Validator** (`rewind/schema.py`): one passing case, plus one failing case
  **per kit rule** in `ARCHITECTURE.md` §7 (missing key, segment out of range,
  overlap, unsorted, no ≥ 4 s segment, bad score range, duration mismatch > 0.5 s,
  > 2000 actions, `video_id` ≠ filename).
- **Post-processing and selector** — property tests for every output, any input:
  sorted; non-overlapping; every segment ≥ 4.0 s; inside `[0, T]`; total ≤ 0.145·T;
  except when T < 27.0 s (`SHORT_VIDEO_T`), which gives exactly one segment shaped by
  `post.short_video_mode` (test both modes). Plus unit tests on small hand-made cases
  with the exact expected windows.
- **Expected-coverage gain**: hand-computed values on 2–3 clusters; monotone (adding
  a window never lowers covered probability); diminishing returns (gain of a window
  never increases as the selected set grows).
- **Scorer** (`rewind_eval/`): every metric on tiny hand-computed cases written out in
  the test's comment:
  - coverage under both capture rules,
  - the < 4 s exclusion,
  - over-budget penalty,
  - AP with ties and duplicate detections,
  - each tIoU threshold,
  - frame F1 at a known overlap,
  - label accuracy.
  Plus properties: every metric in [0, 1]; perfect prediction → 1.0; empty
  prediction → 0.0.
- **Oracle**: on GT-as-proposals, oracle coverage ≥ system coverage, and ≥ uniform.
- **Pairing, routing (including the uncertain route always emitting actions), time
  conversion, cache versioning**: unit tests.
- **Filename independence**: the same synthetic video under two filenames gives
  identical JSON apart from `video_id`, and `run.sh` works with no `kit/` present.
- **Import boundary**: a test that parses every module in `code/rewind/` and fails
  if any imports `rewind_eval` or reads an annotation path or `manifest.csv`.
- **`run.sh` contract**: on 30 s and 12 s synthetic videos generated in a fixture
  (`ffmpeg -f lavfi -i testsrc`), produces JSON that passes the validator, including
  through the fallback path (force a detector failure).

### 7.3 Rules for every test

1. **Test behavior, not implementation.** Assert on outputs a caller can see:
   returned segments, JSON content, metric values. Not on private helpers, call
   counts or internal state.
2. **Hand-compute the expected value** and show the arithmetic in a comment.
   Never paste the code's current output in as the expected value.
3. **One behavior per test**, named `test_<unit>_<condition>_<expected>`, e.g.
   `test_postprocess_fragment_under_4s_is_dropped`. Arrange, act, assert.
4. **Prove it can fail.** Before committing a test, break the code it covers (flip a
   comparison, drop a line) and watch it fail. A test that can't fail is deleted, not
   kept.
5. **Every bug gets a regression test first**: write the failing test, then fix.
6. **Mock only at the edges.** Fake the detector, CLIP and video decoding with small
   deterministic fakes (fixed proposals, fixed embeddings). Never mock our own pure
   functions. Never call a real model in the default suite.
7. **Floats:** use `pytest.approx` with an explicit tolerance. Never compare floats
   with `==`.
8. **Property tests (Hypothesis):** generate durations from 5 s to 2000 s and 0–300
   proposals with random spans and label distributions. Keep strategies in
   `tests/strategies.py`. When Hypothesis finds a failure, add that input as an
   explicit `@example`.
9. **Fixtures are built in code**, are small (no committed binary > 1 MB) and live in
   `tests/fixtures/` or `conftest.py`. Golden JSON files are reviewed by a human when
   they change; never regenerate them blindly to make a test pass.
10. **No sleeps, no network, no wall-clock dependence, no order dependence.** Each
    test sets its own seed.
11. **Don't weaken a test to make it pass.** If behavior changed on purpose, update
    the test and explain why in the PR description.
12. **Metric regression floors** (integration tier): after each phase, record the
    `tune`-split metrics as floors in `tests/integration/floors.yaml`. A drop below a
    floor minus tolerance fails. Raising a floor is a normal part of a PR; lowering
    one needs a reason in `PROGRESS.md`.

## 8. When unsure

- About a kit rule: check `kit/` first, then `PROGRESS.md` → Open questions. If it's
  still unanswered, code to the conservative assumption in `ARCHITECTURE.md` §7 and log it.
- About design: `ARCHITECTURE.md`, then `REWIND_v2.3_plan.md` for the reasoning.
- About priority: the current phase's exit check in `PROGRESS.md`. Don't start stretch
  work while any MVP phase is unfinished.
