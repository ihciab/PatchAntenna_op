from __future__ import annotations

"""Multi-stage Bayesian optimization helpers.

This module is intentionally pipeline-facing only: it defines the stage
schedule and high-level scale variables without touching CST construction,
parameterization, S11 parsing, or the objective factory.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bayesian_optimization.geometry.primitive_mutator import DesignVariable


GLOBAL_SCALE_X = "global_scale_x"
GLOBAL_SCALE_Y = "global_scale_y"
PORT_WIDTH_SCALE = "port_width_scale"
GLOBAL_STAGE_VARIABLE_NAMES = {GLOBAL_SCALE_X, GLOBAL_SCALE_Y, PORT_WIDTH_SCALE}
STAGE4_DELTA_X = "stage4_delta_x"
STAGE4_DELTA_Y = "stage4_delta_y"
STAGE4_POINT_INDEX = "stage4_point_index"


class OptimizationStage(str, Enum):
    STAGE1_FREQUENCY_LOCK = "STAGE1_FREQUENCY_LOCK"
    STAGE2_SHAPE_OPT = "STAGE2_SHAPE_OPT"
    STAGE3_FINE_TUNING = "STAGE3_FINE_TUNING"
    STAGE4_TOPOLOGY_EXPLORATION = "STAGE4_TOPOLOGY_EXPLORATION"

    @property
    def label(self) -> str:
        if self is OptimizationStage.STAGE1_FREQUENCY_LOCK:
            return "Stage1 Frequency Lock"
        if self is OptimizationStage.STAGE2_SHAPE_OPT:
            return "Stage2 Shape Optimization"
        if self is OptimizationStage.STAGE3_FINE_TUNING:
            return "Stage3 Joint Fine Tuning"
        return "Stage4 Topology Exploration"


@dataclass
class Stage1Best:
    evaluation: int
    scale_x: float
    scale_y: float
    port_scale: float
    loss: float
    frequency_ghz: Optional[float] = None
    bandwidth_ghz: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageObjectiveTerms:
    loss: float
    eres: float
    ebw: float
    weights: Dict[str, float]
    source_total: Optional[float]
    failure_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StageManager:
    """Own the trial-to-stage schedule and Stage 1 scale cache."""

    def __init__(
        self,
        *,
        enabled: bool,
        total_trials: int,
        stage1_trials: int,
        stage3_trials: int,
        stage1_scale_range: Tuple[float, float],
        stage3_scale_delta: float,
        stage4_trials: int = 0,
        stage4_delta_px: float = 7.0,
        stage4_point_count: int = 0,
    ) -> None:
        self.enabled = bool(enabled)
        self.total_trials = int(total_trials)
        self.stage1_trials = int(stage1_trials)
        self.stage3_trials = int(stage3_trials)
        self.stage4_trials = int(stage4_trials)
        self.stage1_scale_range = (float(stage1_scale_range[0]), float(stage1_scale_range[1]))
        self.stage3_scale_delta = float(stage3_scale_delta)
        self.stage4_delta_px = float(stage4_delta_px)
        self.stage4_point_count = max(0, int(stage4_point_count))
        self.stage1_best: Optional[Stage1Best] = None

    @property
    def stage2_start_evaluation(self) -> int:
        return self.stage1_trials + 1

    @property
    def stage3_start_evaluation(self) -> int:
        return max(1, self.total_trials - self.stage4_trials - self.stage3_trials + 1)

    @property
    def stage4_start_evaluation(self) -> int:
        if self.stage4_trials <= 0:
            return self.total_trials + 1
        return max(1, self.total_trials - self.stage4_trials + 1)

    def set_stage4_point_count(self, count: int) -> None:
        self.stage4_point_count = max(0, int(count))

    def stage_for_evaluation(self, evaluation: int) -> OptimizationStage:
        if not self.enabled:
            return OptimizationStage.STAGE2_SHAPE_OPT
        trial_index = max(0, int(evaluation) - 1)
        if trial_index < self.stage1_trials:
            return OptimizationStage.STAGE1_FREQUENCY_LOCK
        if self.stage4_trials > 0 and int(evaluation) >= self.stage4_start_evaluation:
            return OptimizationStage.STAGE4_TOPOLOGY_EXPLORATION
        if int(evaluation) >= self.stage3_start_evaluation:
            return OptimizationStage.STAGE3_FINE_TUNING
        return OptimizationStage.STAGE2_SHAPE_OPT

    def high_level_variables(self, stage: OptimizationStage) -> List[DesignVariable]:
        if not self.enabled:
            return []
        if stage is OptimizationStage.STAGE2_SHAPE_OPT:
            return []
        if stage is OptimizationStage.STAGE4_TOPOLOGY_EXPLORATION:
            return []
        low, high = self.stage1_scale_range
        if stage is OptimizationStage.STAGE3_FINE_TUNING:
            low_x, high_x = self._stage3_bounds(self.best_scale_x)
            low_y, high_y = self._stage3_bounds(self.best_scale_y)
            low_p, high_p = self._stage3_bounds(self.best_port_scale)
        else:
            low_x = low_y = low_p = low
            high_x = high_y = high_p = high
        return [
            DesignVariable(
                name=GLOBAL_SCALE_X,
                lower=low_x,
                upper=high_x,
                default=self.best_scale_x if stage is OptimizationStage.STAGE3_FINE_TUNING else 1.0,
                description="High-level conductor X-axis scale about the conductor geometry center.",
            ),
            DesignVariable(
                name=GLOBAL_SCALE_Y,
                lower=low_y,
                upper=high_y,
                default=self.best_scale_y if stage is OptimizationStage.STAGE3_FINE_TUNING else 1.0,
                description="High-level conductor Y-axis scale about the conductor geometry center.",
            ),
            DesignVariable(
                name=PORT_WIDTH_SCALE,
                lower=low_p,
                upper=high_p,
                default=self.best_port_scale if stage is OptimizationStage.STAGE3_FINE_TUNING else 1.0,
                description="High-level feed-port width scale.",
            ),
        ]

    def variables_for_stage(
        self,
        stage: OptimizationStage,
        local_variables: Sequence[DesignVariable],
        evaluation: Optional[int] = None,
    ) -> List[DesignVariable]:
        if not self.enabled:
            return list(local_variables)
        if stage is OptimizationStage.STAGE1_FREQUENCY_LOCK:
            return self.high_level_variables(stage)
        if stage is OptimizationStage.STAGE2_SHAPE_OPT:
            return list(local_variables)
        if stage is OptimizationStage.STAGE4_TOPOLOGY_EXPLORATION:
            delta = max(0.0, float(self.stage4_delta_px))
            point_index = self.stage4_point_index_for_evaluation(evaluation or self.stage4_start_evaluation)
            return [
                DesignVariable(
                    name=STAGE4_DELTA_X,
                    lower=-delta,
                    upper=delta,
                    default=0.0,
                    description=f"Stage4 local escape X move for Point_{point_index}.",
                ),
                DesignVariable(
                    name=STAGE4_DELTA_Y,
                    lower=-delta,
                    upper=delta,
                    default=0.0,
                    description=f"Stage4 local escape Y move for Point_{point_index}.",
                ),
            ]
        return self.high_level_variables(stage) + list(local_variables)

    def applied_values(
        self,
        stage: OptimizationStage,
        sampled_values: Dict[str, float],
        evaluation: Optional[int] = None,
    ) -> Dict[str, float]:
        values = {name: float(value) for name, value in sampled_values.items()}
        if not self.enabled:
            return values
        if stage is OptimizationStage.STAGE2_SHAPE_OPT:
            values[GLOBAL_SCALE_X] = self.best_scale_x
            values[GLOBAL_SCALE_Y] = self.best_scale_y
            values[PORT_WIDTH_SCALE] = self.best_port_scale
        elif stage is OptimizationStage.STAGE4_TOPOLOGY_EXPLORATION:
            values[STAGE4_POINT_INDEX] = float(
                self.stage4_point_index_for_evaluation(evaluation or self.stage4_start_evaluation)
            )
        elif stage is OptimizationStage.STAGE1_FREQUENCY_LOCK:
            values[GLOBAL_SCALE_X] = values.get(GLOBAL_SCALE_X, 1.0)
            values[GLOBAL_SCALE_Y] = values.get(GLOBAL_SCALE_Y, 1.0)
            values[PORT_WIDTH_SCALE] = values.get(PORT_WIDTH_SCALE, 1.0)
        else:
            values.setdefault(GLOBAL_SCALE_X, self.best_scale_x)
            values.setdefault(GLOBAL_SCALE_Y, self.best_scale_y)
            values.setdefault(PORT_WIDTH_SCALE, self.best_port_scale)
        return values

    def update_stage1_best(
        self,
        *,
        evaluation: int,
        values: Dict[str, float],
        loss: float,
        s11_metrics: Optional[Dict[str, Any]],
    ) -> None:
        if not math.isfinite(float(loss)):
            return
        if self.stage1_best is not None and float(loss) >= self.stage1_best.loss:
            return
        metrics = s11_metrics or {}
        self.stage1_best = Stage1Best(
            evaluation=int(evaluation),
            scale_x=float(values.get(GLOBAL_SCALE_X, 1.0)),
            scale_y=float(values.get(GLOBAL_SCALE_Y, 1.0)),
            port_scale=float(values.get(PORT_WIDTH_SCALE, 1.0)),
            loss=float(loss),
            frequency_ghz=_optional_float(metrics.get("resonant_frequency_ghz")),
            bandwidth_ghz=_optional_float(metrics.get("bandwidth_ghz")),
        )

    @property
    def best_scale_x(self) -> float:
        return self.stage1_best.scale_x if self.stage1_best is not None else 1.0

    @property
    def best_scale_y(self) -> float:
        return self.stage1_best.scale_y if self.stage1_best is not None else 1.0

    @property
    def best_port_scale(self) -> float:
        return self.stage1_best.port_scale if self.stage1_best is not None else 1.0

    def _stage3_bounds(self, center: float) -> Tuple[float, float]:
        delta = max(0.0, float(self.stage3_scale_delta))
        low = float(center) * (1.0 - delta)
        high = float(center) * (1.0 + delta)
        return max(1e-6, low), max(1e-6, high)

    def stage4_point_index_for_evaluation(self, evaluation: int) -> int:
        if self.stage4_point_count <= 0:
            return 0
        offset = max(0, int(evaluation) - self.stage4_start_evaluation)
        return offset % self.stage4_point_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "total_trials": self.total_trials,
            "stage1_trials": self.stage1_trials,
            "stage2_start_evaluation": self.stage2_start_evaluation,
            "stage3_trials": self.stage3_trials,
            "stage3_start_evaluation": self.stage3_start_evaluation,
            "stage4_trials": self.stage4_trials,
            "stage4_start_evaluation": self.stage4_start_evaluation,
            "stage4_delta_px": self.stage4_delta_px,
            "stage4_point_count": self.stage4_point_count,
            "stage1_scale_range": list(self.stage1_scale_range),
            "stage3_scale_delta": self.stage3_scale_delta,
            "stage1_best": self.stage1_best.to_dict() if self.stage1_best else None,
            "stage3_scale_ranges": {
                GLOBAL_SCALE_X: list(self._stage3_bounds(self.best_scale_x)),
                GLOBAL_SCALE_Y: list(self._stage3_bounds(self.best_scale_y)),
                PORT_WIDTH_SCALE: list(self._stage3_bounds(self.best_port_scale)),
            },
        }


def compute_stage_objective_terms(
    stage: OptimizationStage,
    objective_breakdown: Optional[Dict[str, Any]],
    stage_loss_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> StageObjectiveTerms:
    breakdown = objective_breakdown or {}
    failure_penalty = _finite_float(breakdown.get("failure_penalty"), 0.0)
    source_total = _optional_float(breakdown.get("total"))
    if failure_penalty > 0.0:
        return StageObjectiveTerms(
            loss=failure_penalty,
            eres=0.0,
            ebw=0.0,
            weights={"failure": 1.0},
            source_total=source_total,
            failure_penalty=failure_penalty,
        )

    normalized = breakdown.get("normalized_errors") or {}
    eres = _finite_float(normalized.get("resonance"), 0.0)
    ebw = _finite_float(
        normalized.get("bandwidth_compliance", normalized.get("bandwidth")),
        0.0,
    )
    configured_weights = _weights_for_stage(stage, stage_loss_weights)
    if stage is OptimizationStage.STAGE1_FREQUENCY_LOCK:
        weights = {
            "ERES": _configured_weight(configured_weights, "ERES", 1.0),
            "EBW": _configured_weight(configured_weights, "EBW", 1.0),
        }
        loss = weights["ERES"] * eres + weights["EBW"] * ebw
    elif stage is OptimizationStage.STAGE2_SHAPE_OPT:
        weights = {
            "ERES": _configured_weight(configured_weights, "ERES", 0.2),
            "EBW": _configured_weight(configured_weights, "EBW", 1.0),
        }
        loss = weights["ERES"] * eres + weights["EBW"] * ebw
    else:
        weights = {
            "full_objective": _configured_weight(configured_weights, "full_objective", 1.0),
        }
        loss = weights["full_objective"] * _finite_float(source_total, 0.0)
    return StageObjectiveTerms(
        loss=max(0.0, float(loss)),
        eres=eres,
        ebw=ebw,
        weights=weights,
        source_total=source_total,
    )


def _weights_for_stage(
    stage: OptimizationStage,
    stage_loss_weights: Optional[Dict[str, Dict[str, float]]],
) -> Dict[str, float]:
    if not stage_loss_weights:
        return {}
    aliases = {
        OptimizationStage.STAGE1_FREQUENCY_LOCK: ("stage1", stage.value.lower()),
        OptimizationStage.STAGE2_SHAPE_OPT: ("stage2", stage.value.lower()),
        OptimizationStage.STAGE3_FINE_TUNING: ("stage3", stage.value.lower()),
        OptimizationStage.STAGE4_TOPOLOGY_EXPLORATION: ("stage4", stage.value.lower()),
    }
    accepted = set(aliases.get(stage, (stage.value.lower(),)))
    for key, value in stage_loss_weights.items():
        if str(key).lower() not in accepted:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _configured_weight(weights: Dict[str, float], name: str, default: float) -> float:
    for key in (name, name.lower(), name.upper()):
        if key in weights:
            return _finite_float(weights.get(key), default)
    return float(default)


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _optional_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
