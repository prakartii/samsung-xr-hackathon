# Vendored: ActionFormer

- Source: https://github.com/happyharrycn/actionformer_release
- Commit: `61ea7eb9308a568b0cf45e3804830836e30061de` (main, fetched 2026-11-11)
- License: MIT (`LICENSE`, recorded in ARCHITECTURE.md §10)
- Copied: `libs/`, `eval.py` (reference only), `configs/epic_slowfast_{verb,noun}.yaml`, `LICENSE`.
  Not copied: training script, other datasets' configs, tools, docs, images.
- Local modifications: **none yet**. Any change must be listed here with its reason.
- Note: `libs/utils/nms.py` imports a C++ extension (`libs/utils/csrc/nms_cpu.cpp`,
  built by `libs/utils/setup.py`). P4 decides whether to build it or replace NMS with
  a pure-PyTorch version (no compiler needed in the organizers' environment).
