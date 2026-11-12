"""tools/check_fpvsum_gt.py: does the GT frame count fit the video's duration?"""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_fpvsum_gt.py"
_spec = importlib.util.spec_from_file_location("check_fpvsum_gt", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


@pytest.mark.parametrize(
    ("n_gt", "duration", "expect"),
    [
        (3000, 100.0, False),  # 30.0 fps: standard
        (2997, 100.0, False),  # 29.97 fps: NTSC
        (2500, 100.0, False),  # 25 fps
        (2130, 100.0, True),  # 21.3 fps: no standard rate -> different cut (Horse_2otBDeLWCow)
        (3960, 100.0, True),  # 39.6 fps (Plane_QsL_gCDHWHA)
    ],
)
def test_gt_rate_suspect_flags_non_standard_implied_fps(n_gt, duration, expect):
    assert tool.gt_rate_suspect(n_gt, duration, tol=0.02) is expect


def test_gt_rate_suspect_tolerance_is_relative():
    # 30.5 fps is 1.7% off 30: within 2%, outside 1%
    assert tool.gt_rate_suspect(3050, 100.0, tol=0.02) is False
    assert tool.gt_rate_suspect(3050, 100.0, tol=0.01) is True
