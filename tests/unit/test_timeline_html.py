"""tools/timeline_html.py: data handed to the page, video paths, script-safe JSON."""

import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools" / "timeline_html.py"
_spec = importlib.util.spec_from_file_location("timeline_html", _PATH)
tool = importlib.util.module_from_spec(_spec)
sys.modules["timeline_html"] = tool
_spec.loader.exec_module(tool)


def _doc():
    return {
        "video_id": "v1",
        "duration_sec": 60.0,
        "label": {"name": "cooking", "score": 0.91234},
        "highlights": [{"segment": [40.0, 44.05]}, {"segment": [10.0, 14.05]}],
        "actions": [
            {"verb": "take", "noun": "cup", "segment": [1.0, 2.0], "score": 0.2},
            {"verb": "open", "noun": "fridge", "segment": [5.0, 7.0], "score": 0.9},
            {"verb": "wash", "noun": "plate", "segment": [3.0, 4.0], "score": 0.9},
        ],
    }


def test_page_entry_highlights_in_time_order():
    e = tool.page_entry(_doc(), None, max_actions=10)
    assert e["highlights"] == [[10.0, 14.05], [40.0, 44.05]]


def test_page_entry_keeps_top_actions_by_score_then_start():
    # scores 0.9 (start 5), 0.9 (start 3), 0.2 -> top 2 = wash (3.0) then open (5.0)
    e = tool.page_entry(_doc(), None, max_actions=2)
    assert [(a["v"], a["n"]) for a in e["actions"]] == [("wash", "plate"), ("open", "fridge")]
    assert e["n_actions"] == 3


def test_page_entry_label_and_duration():
    e = tool.page_entry(_doc(), "x.mp4", max_actions=10)
    assert e["label"] == "cooking"
    assert e["label_conf"] == 0.912
    assert e["duration"] == 60.0
    assert e["src"] == "x.mp4"


def test_video_src_is_relative_to_the_page(tmp_path):
    (tmp_path / "data" / "videos").mkdir(parents=True)
    (tmp_path / "data" / "videos" / "v1.mp4").write_bytes(b"")
    out = tmp_path / "demo" / "timeline.html"
    out.parent.mkdir()
    assert tool.video_src(tmp_path / "data" / "videos", "v1", out) == "../data/videos/v1.mp4"


def test_video_src_missing_video_is_none(tmp_path):
    assert tool.video_src(tmp_path, "nope", tmp_path / "t.html") is None


def test_script_json_cannot_close_the_script_tag():
    s = tool.script_json({"x": "</script><!--"})
    assert "</" not in s and "<!--" not in s
    assert json.loads(s) == {"x": "</script><!--"}


def test_render_inlines_the_data():
    page = tool.render([tool.page_entry(_doc(), None, 10)])
    assert "__DATA__" not in page
    assert '"video_id":"v1"' in page


def test_video_src_on_another_drive_falls_back_to_a_file_uri(tmp_path, monkeypatch):
    # Windows: relpath between C: and D: raises ValueError; the page then needs file:///.
    (tmp_path / "v1.mp4").write_bytes(b"")

    def no_relpath(*_a, **_k):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(tool.os.path, "relpath", no_relpath)
    src = tool.video_src(tmp_path, "v1", tmp_path / "t.html")
    assert src == (tmp_path / "v1.mp4").resolve().as_uri()
