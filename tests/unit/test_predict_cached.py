"""tools/predict_cached.py: which fingerprint and duration each video id uses."""

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools" / "predict_cached.py"
_spec = importlib.util.spec_from_file_location("predict_cached", _PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["predict_cached"] = tool
_spec.loader.exec_module(tool)


def test_video_index_prefers_logs_then_local_mp4_else_missing(tmp_path):
    # A: in the logs (fingerprint fa, 60 s) -> taken from the logs, MP4 never read.
    # B: not logged but its MP4 is here -> local fingerprint/duration ("fp:B.mp4", 9.0).
    # C: neither -> missing.
    (tmp_path / "B.mp4").write_bytes(b"x")
    read = []

    def local(mp4):
        read.append(mp4.name)
        return f"fp:{mp4.name}", 9.0

    index, missing = tool.video_index(["A", "B", "C"], {"A": ("fa", 60.0)}, tmp_path, local)
    assert index == {"A": ("fa", 60.0), "B": ("fp:B.mp4", 9.0)}
    assert missing == ["C"]
    assert read == ["B.mp4"]
