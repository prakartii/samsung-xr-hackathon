"""Load and validate configs/default.yaml into typed, frozen dataclasses.

Rules (CLAUDE.md §6.3): every tunable number lives in the YAML, so the dataclasses
have no defaults. Unknown keys, missing keys and wrong types are errors, never
silently ignored. Kit rules live in ``rewind.constants``, not here.
"""

import hashlib
import json
import math
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
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
class FlowCfg:
    """Farneback parameters (features/flow.py)."""

    width: int
    pyr_scale: float
    levels: int
    winsize: int
    iterations: int
    poly_n: int
    poly_sigma: float


@dataclass(frozen=True)
class HandsCfg:
    """MediaPipe Hand Landmarker (features/hands.py)."""

    model: str
    width: int
    num_hands: int
    min_detection_conf: float
    min_presence_conf: float
    min_tracking_conf: float


@dataclass(frozen=True)
class ClipCfg:
    """open_clip image/text encoder (features/clip.py)."""

    model: str
    weights: str
    width: int
    batch_size: int


@dataclass(frozen=True)
class FeaturesCfg:
    cache_dir: str
    flow: FlowCfg
    hands: HandsCfg
    clip: ClipCfg


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
    score_power: float


@dataclass(frozen=True)
class ZeroShotCfg:
    """Sliding-window CLIP zero-shot detector (detect/fallback_zeroshot.py)."""

    window_sec: float
    stride_sec: float
    top_k: int
    verb_template: str
    noun_template: str


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
    len_step: float
    stride: float


@dataclass(frozen=True)
class LabelCfg:
    prompts: tuple[str, ...]
    classes: dict[str, str]  # kit label -> CLIP phrase (routing + FPVSum label)
    n_keyframes: int
    other_thr: float


@dataclass(frozen=True)
class RoutingCfg:
    cooking_thr: float
    min_conf: float


@dataclass(frozen=True)
class EvalCfg:
    coverage_lambda: float
    capture_theta: float


@dataclass(frozen=True)
class Config:
    run: RunCfg
    team: TeamCfg
    signals: SignalsCfg
    features: FeaturesCfg
    actions: ActionsCfg
    pairing: PairingCfg
    zeroshot: ZeroShotCfg
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
        data = yaml.load(f, Loader=_StrictLoader)  # a SafeLoader subclass
    return config_from_dict(data)


def config_from_dict(data: Any) -> Config:
    """Validate a plain dict (as parsed from YAML) and return a ``Config``."""
    cfg = _build(Config, data, "")
    _check_ranges(cfg)
    return cfg


def config_hash(cfg: Config) -> str:
    """sha1 of the loaded config as canonical JSON.

    Hashes what actually ran, not the file: comments, key order, YAML formatting and
    ``1`` vs ``1.0`` for a float field don't change it; any value does.
    """
    canonical = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys instead of silently keeping the last."""


def _no_duplicate_keys(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict:
    seen: set[Any] = set()
    for key_node, _value_node in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            line = key_node.start_mark.line + 1
            raise ConfigError(f"duplicate key {key!r} at line {line}")
        seen.add(key)
    return loader.construct_mapping(node)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def _build(cls: type, data: Any, path: str) -> Any:
    where = path or "<root>"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(data) - set(names), key=str)
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
    if origin is dict:
        key_tp, val_tp = typing.get_args(tp)
        if not isinstance(value, dict) or not value:
            raise ConfigError(f"{path}: expected a non-empty mapping, got {value!r}")
        return {
            _coerce(key_tp, k, f"{path}.<key>"): _coerce(val_tp, v, f"{path}.{k}")
            for k, v in value.items()
        }
    if origin is tuple:
        args = typing.get_args(tp)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise ConfigError(f"{path}: only variable-length tuple[X, ...] fields are supported")
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        return tuple(_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
    if tp is float:
        # bool is a subclass of int; `true` must never pass as a number.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        if not math.isfinite(value):
            raise ConfigError(f"{path}: expected a finite number, got {value!r}")
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
    if not 1 <= cfg.actions.max_per_video <= MAX_ACTIONS:
        raise ConfigError(
            f"actions.max_per_video: {cfg.actions.max_per_video} must be in [1, {MAX_ACTIONS}]"
        )
    if not 0 <= cfg.run.seed <= 2**32 - 1:
        raise ConfigError(f"run.seed: {cfg.run.seed} must be in [0, 2**32 - 1] (numpy seed range)")
    if not 0.0 <= cfg.hedge.share < 1.0:
        raise ConfigError(f"hedge.share: {cfg.hedge.share} must be in [0, 1)")
    imp = cfg.importance
    if not MIN_SEG_EMIT <= imp.min_window <= imp.max_window:
        raise ConfigError(
            f"importance.min_window/max_window: need {MIN_SEG_EMIT} <= "
            f"{imp.min_window} <= {imp.max_window}"
        )
    if "cooking" not in cfg.label.classes:
        raise ConfigError("label.classes: must include 'cooking' (routing scores P(cooking))")
    for key, tmpl in (
        ("zeroshot.verb_template", cfg.zeroshot.verb_template),
        ("zeroshot.noun_template", cfg.zeroshot.noun_template),
    ):
        if "{}" not in tmpl:
            raise ConfigError(f"{key}: {tmpl!r} has no '{{}}' slot for the class name")
    if not cfg.label.prompts:
        raise ConfigError("label.prompts: must not be empty")
    no_slot = [p for p in cfg.label.prompts if "{}" not in p]
    if no_slot:
        raise ConfigError(f"label.prompts: {no_slot} have no '{{}}' slot for the category name")
    unit_interval = {
        "actions.min_score": cfg.actions.min_score,
        "pairing.match_tiou": cfg.pairing.match_tiou,
        "cluster.tiou": cfg.cluster.tiou,
        "cluster.sim": cfg.cluster.sim,
        "hedge.strong": cfg.hedge.strong,
        "label.other_thr": cfg.label.other_thr,
        "routing.cooking_thr": cfg.routing.cooking_thr,
        "routing.min_conf": cfg.routing.min_conf,
        "features.hands.min_detection_conf": cfg.features.hands.min_detection_conf,
        "features.hands.min_presence_conf": cfg.features.hands.min_presence_conf,
        "features.hands.min_tracking_conf": cfg.features.hands.min_tracking_conf,
        "select.fraction_theta": s.fraction_theta,
        "eval.capture_theta": cfg.eval.capture_theta,
    }
    for key, v in unit_interval.items():
        if not 0.0 <= v <= 1.0:
            raise ConfigError(f"{key}: {v} must be in [0, 1]")
    positive = {
        "signals.dense_fps": cfg.signals.dense_fps,
        "signals.clip_fps": cfg.signals.clip_fps,
        "calibration.temperature": cfg.calibration.temperature,
        "select.stride": s.stride,
        "select.fraction_tau": s.fraction_tau,
        "importance.sigma": imp.sigma,
        "importance.len_step": imp.len_step,
        "importance.stride": imp.stride,
        "features.flow.width": cfg.features.flow.width,
        "features.flow.pyr_scale": cfg.features.flow.pyr_scale,
        "features.flow.levels": cfg.features.flow.levels,
        "features.flow.winsize": cfg.features.flow.winsize,
        "features.flow.iterations": cfg.features.flow.iterations,
        "features.flow.poly_n": cfg.features.flow.poly_n,
        "features.flow.poly_sigma": cfg.features.flow.poly_sigma,
        "features.hands.width": cfg.features.hands.width,
        "features.hands.num_hands": cfg.features.hands.num_hands,
        "features.clip.width": cfg.features.clip.width,
        "features.clip.batch_size": cfg.features.clip.batch_size,
        "pairing.top_k": cfg.pairing.top_k,
        "pairing.eps": cfg.pairing.eps,
        "pairing.score_power": cfg.pairing.score_power,
        "zeroshot.window_sec": cfg.zeroshot.window_sec,
        "zeroshot.stride_sec": cfg.zeroshot.stride_sec,
        "zeroshot.top_k": cfg.zeroshot.top_k,
        "cluster.k": cfg.cluster.k,
        "label.n_keyframes": cfg.label.n_keyframes,
    }
    for key, v in positive.items():
        if v <= 0:
            raise ConfigError(f"{key}: {v} must be > 0")
    non_negative = {
        "pairing.alpha": cfg.pairing.alpha,
        "hedge.gap": cfg.hedge.gap,
        "hedge.min_spacing": cfg.hedge.min_spacing,
        "post.merge_gap": cfg.post.merge_gap,
        "eval.coverage_lambda": cfg.eval.coverage_lambda,
        "importance.shot_thr": imp.shot_thr,
    }
    for key, v in non_negative.items():
        if v < 0:
            raise ConfigError(f"{key}: {v} must be >= 0")
