from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from bayesian_optimization.geometry.geometry_validator import ValidationReport
from bayesian_optimization.optimization.objective_factory import (
    WidebandPatchObjective,
    create_objective_profile_from_instance,
)
from bayesian_optimization.optimization.s11_parser import S11Metrics


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for normalized paper-reconstruction objective terms."""

    resonance: float = 4.0
    target_s11: float = 2.0
    bandwidth_reward: float = 3.0
    gain: float = 1.0
    complexity: float = 0.0
    curvature: float = 0.0
    tiny_segments: float = 0.0
    topology_instability: float = 0.0
    invalid_geometry: float = 100.0
    cst_failure: float = 80.0

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-serializable weight dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """Detailed objective values stored with each evaluation record."""

    total: float
    primary_frequency_error: float = 0.0
    target_s11_penalty: float = 0.0
    bandwidth_reward: float = 0.0
    gain_penalty: float = 0.0
    complexity_penalty: float = 0.0
    curvature_penalty: float = 0.0
    tiny_segment_penalty: float = 0.0
    topology_penalty: float = 0.0
    failure_penalty: float = 0.0
    objective_profile: str = ""
    normalized_errors: Dict[str, Optional[float]] = field(default_factory=dict)
    target_metrics: Dict[str, Any] = field(default_factory=dict)
    actual_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable objective breakdown."""

        return asdict(self)


_CURRENT_OBJECTIVE_PROFILE: Optional[WidebandPatchObjective] = None


def set_current_objective_profile(profile: Optional[WidebandPatchObjective]) -> None:
    """Set the profile used by evaluate_objective without changing its signature."""

    global _CURRENT_OBJECTIVE_PROFILE
    _CURRENT_OBJECTIVE_PROFILE = profile


def get_current_objective_profile(
    target_frequency_ghz: float,
    target_s11_db: float,
) -> WidebandPatchObjective:
    """Return the configured profile or a legacy-target fallback profile."""

    if _CURRENT_OBJECTIVE_PROFILE is not None:
        return _CURRENT_OBJECTIVE_PROFILE
    return create_objective_profile_from_instance(
        {},
        fallback_target_frequency_ghz=target_frequency_ghz,
        fallback_target_s11_db=target_s11_db,
        source="fallback:evaluate_objective",
    )


def evaluate_objective(
    s11_metrics: Optional[S11Metrics],
    geometry_metrics: Dict[str, Any],
    validation: ValidationReport,
    target_frequency_ghz: float,
    target_s11_db: float,
    weights: Optional[ObjectiveWeights] = None,
    cst_failed: bool = False,
) -> ObjectiveBreakdown:
    """Evaluate the paper-reconstruction objective with legacy-call compatibility."""

    w = weights or ObjectiveWeights()
    profile = get_current_objective_profile(target_frequency_ghz, target_s11_db)

    if not validation.valid:
        return _failure_breakdown(
            total=float(w.invalid_geometry),
            profile_name=profile.name,
            reason="invalid_geometry",
        )

    if cst_failed or s11_metrics is None:
        return _failure_breakdown(
            total=float(w.cst_failure),
            profile_name=profile.name,
            reason="cst_failure",
        )

    result = profile.evaluate(s11_metrics, geometry_metrics, w)

    # Continuous loss uses only normalized paper metric errors:
    # 4*E_res + 3*E_bw + 2*E_s11 + 1*E_gain.
    return ObjectiveBreakdown(
        total=max(0.0, float(result.total)),
        primary_frequency_error=result.weighted_terms.get("resonance", 0.0),
        target_s11_penalty=result.weighted_terms.get("s11", 0.0),
        bandwidth_reward=result.weighted_terms.get("bandwidth", 0.0),
        gain_penalty=result.weighted_terms.get("gain", 0.0),
        complexity_penalty=0.0,
        curvature_penalty=0.0,
        tiny_segment_penalty=0.0,
        topology_penalty=0.0,
        failure_penalty=0.0,
        objective_profile=result.profile_name,
        normalized_errors=result.normalized_errors,
        target_metrics=result.targets,
        actual_metrics=result.actuals,
    )


def _failure_breakdown(total: float, profile_name: str, reason: str) -> ObjectiveBreakdown:
    """Create a hard-failure objective breakdown without continuous terms."""

    return ObjectiveBreakdown(
        total=float(total),
        failure_penalty=float(total),
        objective_profile=profile_name,
        normalized_errors={},
        target_metrics={"failure_reason": reason},
        actual_metrics={},
    )
