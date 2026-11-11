"""Load and validate configs/default.yaml into typed, frozen dataclasses.

Rules (CLAUDE.md §6.3): every tunable number lives in the YAML, so the dataclasses
have no defaults. Unknown keys, missing keys and wrong types are errors, never
silently ignored. Kit rules live in ``rewind.constants``, not here.
"""

import hashlib
import json
import typing
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from rewind.constants import MAX_ACTIONS, MIN_SEG_EMIT, SCORING_BUDGET


class ConfigError(ValueError):
    """The config file is invalid. The message names the offending key path."""


@dataclass(frozen=True)
class RunCfg:
    seed: int


@dataclass(frozen=True)
class TeamCfg:
    team_id: str


@dataclass(frozen=True)
class SignalsCfg:
    dense_fps: float
    clip_fps: float


@dataclass(frozen=True)
class FeaturesCfg:
    cache_dir: str
    flow_width: int


@dataclass(frozen=True)
class ActionsCfg:
    max_per_video: int
    min_score: float


@dataclass(frozen=True)
class PairingCfg:
    match_tiou: float
    top_k: int
    alpha: float
    eps: float


@dataclass(frozen=True)
class ClusterCfg:
    tiou: float
    sim: float
    k: int


@dataclass(frozen=True)
class CalibrationCfg:
    temperature: float


@dataclass(frozen=True)
class SelectCfg:
    window_lengths: tuple[float, ...]
    stride: float
    budget_frac: float
    capture_rule: Literal["any_overlap", "fraction"]
    fraction_theta: float
    fraction_tau: float


@dataclass(frozen=True)
class HedgeCfg:
    share: float
    gap: float
    strong: float
    min_spacing: float
    w_flow: float
    w_hands: float


@dataclass(frozen=True)
class PostCfg:
    merge_gap: float
    short_video_mode: Literal["min_seg", "within_budget"]


@dataclass(frozen=True)
class ImportanceCfg:
    sigma: float
    shot_thr: float
    w_flow: float
    w_hands: float
    w_global_motion: float
    w_scene_change: float
    min_window: float
    max_window: float


@dataclass(frozen=True)
class LabelCfg:
    prompts: tuple[str, ...]
    n_keyframes: int
    other_thr: float


@dataclass(frozen=True)
class RoutingCfg:
    cooking_thr: float
    min_conf: float


@dataclass(frozen=True)
class EvalCfg:
    f1_fps: float
    coverage_lambda: float


@dataclass(frozen=True)
class Config:
    run: RunCfg
    team: TeamCfg
    signals: SignalsCfg
    features: FeaturesCfg
    actions: ActionsCfg
    pairing: PairingCfg
    cluster: ClusterCfg
    calibration: CalibrationCfg
    select: SelectCfg
    hedge: HedgeCfg
    post: PostCfg
    importance: ImportanceCfg
    label: LabelCfg
    routing: RoutingCfg
    eval: EvalCfg


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated ``Config``."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return config_from_dict(data)


def config_from_dict(data: Any) -> Config:
    """Validate a plain dict (as parsed from YAML) and return a ``Config``."""
    cfg = _build(Config, data, "")
    _check_ranges(cfg)
    return cfg


def config_hash(data: Any) -> str:
    """sha1 of the canonical JSON form of a parsed config dict.

    Comments, key order and YAML formatting don't change the hash; any value does.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _build(cls: type, data: Any, path: str) -> Any:
    where = path or "<root>"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    missing = [n for n in names if n not in data]
    if missing:
        raise ConfigError(f"{where}: missing key(s) {missing}")
    values = {n: _coerce(hints[n], data[n], f"{path}.{n}" if path else n) for n in names}
    return cls(**values)


def _coerce(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if is_dataclass(tp):
        return _build(tp, value, path)
    if origin is Literal:
        allowed = typing.get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{path}: {value!r} is not one of {list(allowed)}")
        return value
    if origin is tuple:
        (item_tp, _ellipsis) = typing.get_args(tp)
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return tuple(_coerce(item_tp, v, f"{path}[{i}]") for i, v in enumerate(value))
    if tp is float:
        # bool is a subclass of int; `true` must never pass as a number.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{path}: expected a non-empty string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported field type {tp!r}")


def _check_ranges(cfg: Config) -> None:
    """Cross-field and range checks that types alone can't express."""
    s = cfg.select
    if not 0.0 < s.budget_frac <= SCORING_BUDGET:
        raise ConfigError(
            f"select.budget_frac: {s.budget_frac} must be in (0, {SCORING_BUDGET}] (kit budget)"
        )
    if not s.window_lengths:
        raise ConfigError("select.window_lengths: must not be empty")
    short = [w for w in s.window_lengths if w < MIN_SEG_EMIT]
    if short:
        raise ConfigError(
            f"select.window_lengths: {short} below MIN_SEG_EMIT={MIN_SEG_EMIT} "
            "(windows under 4 s are ignored in scoring)"
        )
    if s.stride <= 0:
        raise ConfigError(f"select.stride: {s.stride} must be > 0")
    if not 1 <= cfg.actions.max_per_video <= MAX_ACTIONS:
        raise ConfigError(
            f"actions.max_per_video: {cfg.actions.max_per_video} must be in [1, {MAX_ACTIONS}]"
        )
    if not 0.0 <= cfg.hedge.share < 1.0:
        raise ConfigError(f"hedge.share: {cfg.hedge.share} must be in [0, 1)")
    imp = cfg.importance
    if not MIN_SEG_EMIT <= imp.min_window <= imp.max_window:
        raise ConfigError(
            f"importance.min_window/max_window: need {MIN_SEG_EMIT} <= "
            f"{imp.min_window} <= {imp.max_window}"
        )
    unit_interval = {
        "actions.min_score": cfg.actions.min_score,
        "pairing.match_tiou": cfg.pairing.match_tiou,
        "cluster.tiou": cfg.cluster.tiou,
        "cluster.sim": cfg.cluster.sim,
        "hedge.strong": cfg.hedge.strong,
        "label.other_thr": cfg.label.other_thr,
        "routing.cooking_thr": cfg.routing.cooking_thr,
        "routing.min_conf": cfg.routing.min_conf,
        "select.fraction_theta": s.fraction_theta,
    }
    for key, v in unit_interval.items():
        if not 0.0 <= v <= 1.0:
            raise ConfigError(f"{key}: {v} must be in [0, 1]")
    positive = {
        "signals.dense_fps": cfg.signals.dense_fps,
        "signals.clip_fps": cfg.signals.clip_fps,
        "calibration.temperature": cfg.calibration.temperature,
        "select.fraction_tau": s.fraction_tau,
        "importance.sigma": imp.sigma,
        "eval.f1_fps": cfg.eval.f1_fps,
        "features.flow_width": cfg.features.flow_width,
        "pairing.top_k": cfg.pairing.top_k,
        "cluster.k": cfg.cluster.k,
        "label.n_keyframes": cfg.label.n_keyframes,
    }
    for key, v in positive.items():
        if v <= 0:
            raise ConfigError(f"{key}: {v} must be > 0")
