"""python -m rewind.cli <video.mp4> <out.json> [--config FILE]   (called by run.sh)

Exit codes: 0 = a valid JSON was written (possibly the fallback); 2 = the video has
no readable duration, so no valid JSON can exist.
"""

import argparse
import os
import sys
from pathlib import Path

from rewind.config import load_config
from rewind.pipeline import run_video
from rewind.utils import get_logger
from rewind.video_io import VideoError

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="REWIND: summarize one video into JSON.")
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    default = Path(os.environ.get("REWIND_CONFIG") or DEFAULT_CONFIG)
    parser.add_argument("--config", type=Path, default=default)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    team_id = os.environ.get("REWIND_TEAM_ID") or cfg.team.team_id
    try:
        run_video(args.video, args.out, cfg, team_id=team_id)
    except VideoError as exc:
        get_logger("rewind.cli").error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
