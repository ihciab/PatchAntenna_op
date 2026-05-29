from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from geometry_validator import ValidationReport
from s11_parser import S11Metrics


@dataclass(frozen=True)
class ObjectiveWeights:
    resonance: float = 1.0
    target_s11: float = 0.05
    bandwidth_reward: float = 0.02
    complexity: float = 0.001
    curvature: float = 0.002
    tiny_segments: float = 0.05
    topology_instability: float = 5.0
    invalid_geometry: float = 100.0
    cst_failure: float = 80.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveBreakdown:
    total: float
    primary_frequency_error: float
    target_s11_penalty: float
    bandwidth_reward: float
    complexity_penalty: float
    curvature_penalty: float
    tiny_segment_penalty: float
    topology_penalty: float
    failure_penalty: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_objective(
    s11_metrics: Optional[S11Metrics],
    geometry_metrics: Dict[str, Any],
    validation: ValidationReport,
    target_frequency_ghz: float,
    target_s11_db: float,
    weights: Optional[ObjectiveWeights] = None,
    cst_failed: bool = False,
) -> ObjectiveBreakdown:
    """优化目标函数。

    Primary:
    - 谐振频率匹配
    - 目标频点 S11 越低越好

    Secondary penalties:
    - 几何复杂度
    - spline 曲率过大
    - tiny segments
    - 拓扑不稳定 / CST 失败
    """

    w = weights or ObjectiveWeights()
    if not validation.valid:
        failure_penalty = w.invalid_geometry
        return ObjectiveBreakdown(
            total=failure_penalty,
            primary_frequency_error=0.0,
            target_s11_penalty=0.0,
            bandwidth_reward=0.0,
            complexity_penalty=0.0,
            curvature_penalty=0.0,
            tiny_segment_penalty=0.0,
            topology_penalty=w.topology_instability,
            failure_penalty=failure_penalty,
        )

    if cst_failed or s11_metrics is None:
        failure_penalty = w.cst_failure
        return ObjectiveBreakdown(
            total=failure_penalty,
            primary_frequency_error=0.0,
            target_s11_penalty=0.0,
            bandwidth_reward=0.0,
            complexity_penalty=0.0,
            curvature_penalty=0.0,
            tiny_segment_penalty=0.0,
            topology_penalty=0.0,
            failure_penalty=failure_penalty,
        )

    frequency_error = abs(s11_metrics.resonant_frequency_ghz - target_frequency_ghz)
    target_s11_penalty = max(0.0, s11_metrics.s11_at_target_db - target_s11_db) * w.target_s11
    bandwidth_reward = -max(0.0, s11_metrics.bandwidth_ghz) * w.bandwidth_reward
    complexity_penalty = float(geometry_metrics.get("primitive_count", 0)) * w.complexity
    curvature_penalty = float(geometry_metrics.get("spline_curvature_proxy", 0.0)) * w.curvature
    tiny_segment_penalty = float(geometry_metrics.get("tiny_segment_count", 0)) * w.tiny_segments
    topology_penalty = 0.0
    if validation.warnings:
        topology_penalty = len(validation.warnings) * 0.25

    total = (
        frequency_error * w.resonance
        + target_s11_penalty
        + bandwidth_reward
        + complexity_penalty
        + curvature_penalty
        + tiny_segment_penalty
        + topology_penalty
    )
    return ObjectiveBreakdown(
        total=max(0.0, total),
        primary_frequency_error=frequency_error,
        target_s11_penalty=target_s11_penalty,
        bandwidth_reward=bandwidth_reward,
        complexity_penalty=complexity_penalty,
        curvature_penalty=curvature_penalty,
        tiny_segment_penalty=tiny_segment_penalty,
        topology_penalty=topology_penalty,
        failure_penalty=0.0,
    )
