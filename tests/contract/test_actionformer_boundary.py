"""Boundary for the vendored ActionFormer (PROGRESS.md P4, B's review note).

ActionFormer's dataset loaders (`libs.datasets`) read annotation JSONs, and its
metrics/training utilities need ground truth. Inference must reach only the model
code: `libs.core` (config) and `libs.modeling` (network), plus `libs.utils.nms` via
the model. This test scans every module in code/rewind for imports of the vendored
`libs` package and fails on anything else.
"""

import ast
from pathlib import Path

REWIND = Path(__file__).resolve().parents[2] / "code" / "rewind"
ALLOWED = {"libs.core", "libs.modeling"}


def _vendored_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "libs":
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.split(".")[0] == "libs"}
    return found


def test_inference_imports_only_actionformer_model_code():
    bad = {
        str(p.relative_to(REWIND)): sorted(mods - ALLOWED)
        for p in REWIND.rglob("*.py")
        if (mods := _vendored_imports(p)) - ALLOWED
    }
    assert bad == {}, f"inference imports ActionFormer modules outside {sorted(ALLOWED)}: {bad}"


def test_detector_does_import_the_model_code():
    # Guards the scan itself: if the detector moved, the test above would pass vacuously.
    mods = _vendored_imports(REWIND / "detect" / "actionformer.py")
    assert mods == ALLOWED
