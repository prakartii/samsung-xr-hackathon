"""Run one video end to end and write its JSON. Never writes invalid output.

ARCHITECTURE.md §4. ``analyze`` is the model part (features → detect → select →
label); in P1 it is the fallback itself, and later phases plug in the real stages.
Whatever it returns or raises, the file written is either its validated result or
the validated fallback.
"""

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from rewind.config import Config
from rewind.fallback import fallback_result
from rewind.output import dumps, to_document
from rewind.schema import validate
from rewind.types import VideoResult
from rewind.utils import get_logger, seed_all
from rewind.video_io import ProbeInfo, ffprobe_meta
from rewind.vocab import load_vocab

Analyze = Callable[[Path, ProbeInfo, str, Config], VideoResult]
log = get_logger("rewind.pipeline")


def default_analyze(video: Path, probe: ProbeInfo, video_id: str, cfg: Config) -> VideoResult:
    """P1: no models yet, so the result is the fallback."""
    return _fallback(video_id, probe.duration_sec, cfg)


def run_video(
    video: Path, out: Path, cfg: Config, *, team_id: str, analyze: Analyze = default_analyze
) -> None:
    """Analyze ``video`` and write ``out`` atomically. Raises only if the video has no
    readable duration (then no valid JSON can exist: R3 needs the duration)."""
    seed_all(cfg.run.seed)
    # R2: video_id must equal the output file name. The input name is not used.
    video_id = out.stem
    probe = ffprobe_meta(video)
    cfg = _with_team(cfg, team_id)

    try:
        result = analyze(video, probe, video_id, cfg)
        doc = to_document(result)
        errors, _ = validate(doc, load_vocab(), filename_stem=video_id, strict=True)
        if errors:
            raise ValueError(f"{len(errors)} validation error(s): {errors[:3]}")
    except Exception as exc:  # any stage failure → fallback (CLAUDE.md §6.7)
        log.error("analysis failed for %s, writing fallback: %s", video, exc)
        doc = to_document(_fallback(video_id, probe.duration_sec, cfg))
        errors, _ = validate(doc, load_vocab(), filename_stem=video_id, strict=True)
        if errors:  # should be impossible (property-tested); still write it
            log.error("fallback failed validation: %s", errors)
    _write_atomic(out, dumps(doc))


def _with_team(cfg: Config, team_id: str) -> Config:
    from dataclasses import replace

    return replace(cfg, team=replace(cfg.team, team_id=team_id))


def _fallback(video_id: str, duration: float, cfg: Config) -> VideoResult:
    return fallback_result(
        cfg.team.team_id,
        video_id,
        duration,
        budget_frac=cfg.select.budget_frac,
        merge_gap=cfg.post.merge_gap,
        short_video_mode=cfg.post.short_video_mode,
    )


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
