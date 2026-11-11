# Vendored: ActionFormer

- Source: https://github.com/happyharrycn/actionformer_release
- Commit: `61ea7eb9308a568b0cf45e3804830836e30061de` (main, fetched 2026-11-11)
- License: MIT (`LICENSE`, recorded in ARCHITECTURE.md §10)
- Copied: `libs/`, `eval.py` (reference only), `configs/epic_slowfast_{verb,noun}.yaml`, `LICENSE`.
  Not copied: training script, other datasets' configs, tools, docs, images.
- Local modifications (each with its reason):
  1. **Added `libs/utils/nms_1d_py.py`**: an exact numpy port of the C++ 1D NMS /
     soft-NMS extension (`libs/utils/csrc/nms_cpu.cpp`). The organizers re-run
     `run.sh` in an isolated environment that may have no C++ compiler. Verified
     against a literal transcription of the C++ code in
     `tests/unit/test_actionformer_nms.py` (hand cases + randomized inputs with ties).
  2. **`libs/utils/nms.py`**: `import nms_1d_cpu` now falls back to that port when
     the extension isn't built (3-line change at the import).
  3. **`libs/utils/__init__.py`**: only `batched_nms` is imported eagerly; the other
     exports (metrics, training utilities, post-processing) load lazily on first use.
     The model imports `libs.utils`, and the eager imports required pandas and joblib,
     which inference doesn't need and the runtime doesn't ship.
