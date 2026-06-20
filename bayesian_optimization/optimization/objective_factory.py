from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, Optional, Sequence

from bayesian_optimization.optimization.s11_parser import S11Metrics


OBJECTIVE_EPSILON = 1e-9


@dataclass(frozen=True)
class ObjectiveTargets:
    """Paper-level target metrics used by a reconstruction objective."""

    resonance_ghz: Optional[float]
    bandwidth_ghz: Optional[float]
    bandwidth_start_ghz: Optional[float]
    bandwidth_end_ghz: Optional[float]
    gain_dbi: Optional[float]
    s11_db: float
    resonance_count: Optional[int] = None
    resonance_frequencies_ghz: tuple[float, ...] = field(default_factory=tuple)
    antenna_type: Dict[str, Any] = field(default_factory=dict)
    source: str = "fallback"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable target summary."""

        return asdict(self)


@dataclass(frozen=True)
class ObjectiveProfileResult:
    """Weighted objective terms and raw normalized errors from one profile."""

    profile_name: str
    total: float
    weighted_terms: Dict[str, float]
    normalized_errors: Dict[str, Optional[float]]
    targets: Dict[str, Any]
    actuals: Dict[str, Any]


@dataclass(frozen=True)
class BandwidthComplianceResult:
    """Full paper-band S11 compliance diagnostics."""

    error: Optional[float]
    target_start_ghz: Optional[float]
    target_end_ghz: Optional[float]
    evaluated_sample_count: int


class WidebandPatchObjective:
    """Paper reconstruction objective for wideband patch-like antennas."""

    name = "WidebandPatchObjective"

    def __init__(self, targets: ObjectiveTargets) -> None:
        """Store the paper targets used by this profile."""

        self.targets = targets

    def evaluate(
        self,
        s11_metrics: S11Metrics,
        geometry_metrics: Dict[str, Any],
        weights: Any,
    ) -> ObjectiveProfileResult:
        """Evaluate normalized relative errors against paper targets."""

        actual_gain = _extract_actual_gain(geometry_metrics)
        target_resonances = self.targets.resonance_frequencies_ghz
        qualified_actual_resonances = tuple(
            float(value)
            for value in getattr(s11_metrics, "resonant_frequencies_ghz", ())
            if value is not None
        )
        actual_resonances = qualified_actual_resonances
        if not actual_resonances:
            actual_resonances = (float(s11_metrics.resonant_frequency_ghz),)

        # Normalized relative error: abs(actual - target) / max(abs(target), epsilon).
        # A target resonance of 0.0 means the paper parser did not extract a
        # usable resonance, so the resonance loss should not steer BO.
        e_res = multi_resonance_error_optional(actual_resonances, target_resonances)
        # e_res_count = resonance_count_error_optional(
        #     actual_count=len(qualified_actual_resonances),
        #     target_count=self.targets.resonance_count,
        # )
        e_res_count=0.0
        bw_compliance = bandwidth_compliance_error_optional(
            s11_samples=getattr(s11_metrics, "s11_samples", ()),
            target_start=self.targets.bandwidth_start_ghz,
            target_end=self.targets.bandwidth_end_ghz,
        )
        e_bw = bw_compliance.error
        if e_bw is None:
            e_bw = normalized_relative_error_optional(
                s11_metrics.bandwidth_ghz,
                self.targets.bandwidth_ghz,
            )
        e_gain = normalized_relative_error_optional(actual_gain, self.targets.gain_dbi)
        e_s11 = normalized_s11_threshold_error(
            s11_metrics.s11_at_target_db,
            self.targets.s11_db,
        )

        normalized_errors = {
            "resonance": e_res,
            "resonance_count": e_res_count,
            "bandwidth": e_bw,
            "bandwidth_compliance": e_bw,
            "gain": e_gain,
            "s11": e_s11,
        }
        weighted_terms = {
            "resonance": float(getattr(weights, "resonance", 0.0)) * (e_res or 0.0),
            "resonance_count": (
                float(getattr(weights, "resonance_count", 0.0)) * (e_res_count or 0.0)
            ),
            "bandwidth": float(getattr(weights, "bandwidth_reward", 0.0)) * (e_bw or 0.0),
            "s11": float(getattr(weights, "target_s11", 0.0)) * e_s11,
            "gain": float(getattr(weights, "gain", 0.0)) * (e_gain or 0.0),
        }
        total = sum(weighted_terms.values())

        return ObjectiveProfileResult(
            profile_name=self.name,
            total=float(total),
            weighted_terms=weighted_terms,
            normalized_errors=normalized_errors,
            targets=self.targets.to_dict(),
            actuals={
                "resonance_ghz": float(s11_metrics.resonant_frequency_ghz),
                "resonance_frequencies_ghz": list(actual_resonances),
                "qualified_resonance_frequencies_ghz": list(qualified_actual_resonances),
                "resonance_count": len(qualified_actual_resonances),
                "bandwidth_ghz": float(s11_metrics.bandwidth_ghz),
                "bandwidth_start_ghz": s11_metrics.bandwidth_start_ghz,
                "bandwidth_end_ghz": s11_metrics.bandwidth_end_ghz,
                "target_start_ghz": bw_compliance.target_start_ghz,
                "target_end_ghz": bw_compliance.target_end_ghz,
                "evaluated_sample_count": bw_compliance.evaluated_sample_count,
                "bandwidth_compliance_error": e_bw,
                "gain_dbi": actual_gain,
                "s11_at_target_db": float(s11_metrics.s11_at_target_db),
                "minimum_s11_db": float(s11_metrics.minimum_s11_db),
            },
        )


def create_objective_profile_from_instance_path(
    instance_json_path: Optional[Path | str],
    fallback_target_frequency_ghz: float,
    fallback_target_s11_db: float,
) -> WidebandPatchObjective:
    """Create an objective profile from an instance JSON path."""

    if instance_json_path is None:
        return create_objective_profile_from_instance(
            {},
            fallback_target_frequency_ghz=fallback_target_frequency_ghz,
            fallback_target_s11_db=fallback_target_s11_db,
            source="fallback:no_instance_json",
        )

    path = Path(instance_json_path)
    if not path.exists():
        return create_objective_profile_from_instance(
            {},
            fallback_target_frequency_ghz=fallback_target_frequency_ghz,
            fallback_target_s11_db=fallback_target_s11_db,
            source=f"fallback:missing_instance_json:{path}",
        )

    with path.open("r", encoding="utf-8") as file:
        instance = json.load(file)
    if not isinstance(instance, dict):
        instance = {}
    return create_objective_profile_from_instance(
        instance,
        fallback_target_frequency_ghz=fallback_target_frequency_ghz,
        fallback_target_s11_db=fallback_target_s11_db,
        source=str(path),
    )


def create_objective_profile_from_instance(
    instance: Dict[str, Any],
    fallback_target_frequency_ghz: float,
    fallback_target_s11_db: float,
    source: str = "instance",
) -> WidebandPatchObjective:
    """Select an objective profile from Antenna_Type and parsed paper targets."""

    targets = extract_objective_targets(
        instance,
        fallback_target_frequency_ghz=fallback_target_frequency_ghz,
        fallback_target_s11_db=fallback_target_s11_db,
        source=source,
    )
    profile_name = select_objective_profile_name(instance)
    if profile_name != WidebandPatchObjective.name:
        targets = ObjectiveTargets(
            resonance_ghz=targets.resonance_ghz,
            resonance_count=targets.resonance_count,
            resonance_frequencies_ghz=targets.resonance_frequencies_ghz,
            bandwidth_ghz=targets.bandwidth_ghz,
            bandwidth_start_ghz=targets.bandwidth_start_ghz,
            bandwidth_end_ghz=targets.bandwidth_end_ghz,
            gain_dbi=targets.gain_dbi,
            s11_db=targets.s11_db,
            antenna_type=targets.antenna_type,
            source=f"{targets.source}; fallback_profile={profile_name}->WidebandPatchObjective",
        )
    return WidebandPatchObjective(targets)


def extract_objective_targets(
    instance: Dict[str, Any],
    fallback_target_frequency_ghz: float,
    fallback_target_s11_db: float,
    source: str = "instance",
) -> ObjectiveTargets:
    """Extract paper reconstruction targets from the new-format instance JSON."""

    paper = instance.get("Paper_Performance", {})
    if not isinstance(paper, dict):
        paper = {}
    antenna_type = instance.get("Antenna_Type", {})
    if not isinstance(antenna_type, dict):
        antenna_type = {}

    resonance_frequencies, resonance = _extract_resonance_targets(
        paper,
        fallback_target_frequency_ghz=fallback_target_frequency_ghz,
    )
    resonance_count = _extract_resonance_count_target(paper, resonance_frequencies)

    bandwidth, bandwidth_start, bandwidth_end = _extract_bandwidth_target(paper)
    gain = _number_or_none(paper.get("Peak_Gain_dBi"))
    s11_target = _extract_s11_target(paper, fallback_target_s11_db)

    return ObjectiveTargets(
        resonance_ghz=resonance,
        resonance_count=resonance_count,
        resonance_frequencies_ghz=resonance_frequencies,
        bandwidth_ghz=bandwidth,
        bandwidth_start_ghz=bandwidth_start,
        bandwidth_end_ghz=bandwidth_end,
        gain_dbi=gain,
        s11_db=float(s11_target),
        antenna_type=dict(antenna_type),
        source=source,
    )


def select_objective_profile_name(instance: Dict[str, Any]) -> str:
    """Choose an objective profile name from Antenna_Type descriptors."""

    antenna_type = instance.get("Antenna_Type", {})
    if not isinstance(antenna_type, dict):
        antenna_type = {}
    text = " ".join(str(value) for value in antenna_type.values()).lower()

    if "array" in text:
        return "ArrayObjective"
    if "fss" in text or "frequency selective" in text:
        return "FSSObjective"
    if "multi" in text or "dual" in text or "triple" in text:
        return "MultibandPatchObjective"
    if "wideband" in text or "patch" in text:
        return WidebandPatchObjective.name
    return WidebandPatchObjective.name


def normalized_relative_error(actual: float, target: float, epsilon: float = OBJECTIVE_EPSILON) -> float:
    """Compute abs(actual-target)/max(abs(target), epsilon)."""

    denominator = max(abs(float(target)), float(epsilon))
    return abs(float(actual) - float(target)) / denominator


def normalized_s11_threshold_error(actual_s11_db: float, target_s11_db: float) -> float:
    """Compute S11 threshold loss.

    S11 is already good enough when it is at or below the target, for example
    actual=-13 dB and target=-10 dB. Only worse values are penalized:

        e_s11 = 0, if s11 <= target
        e_s11 = (s11 - target) / abs(target), otherwise
    """

    actual = float(actual_s11_db)
    target = float(target_s11_db)
    if actual <= target:
        return 0.0
    return (actual - target) / max(abs(target), OBJECTIVE_EPSILON)


def normalized_relative_error_optional(
    actual: Optional[float],
    target: Optional[float],
    epsilon: float = OBJECTIVE_EPSILON,
) -> Optional[float]:
    """Compute normalized error when both actual and target are available."""

    if actual is None or target is None:
        return None
    return normalized_relative_error(float(actual), float(target), epsilon=epsilon)


def multi_resonance_error_optional(
    actual: Sequence[float],
    target: Sequence[float],
    epsilon: float = OBJECTIVE_EPSILON,
) -> Optional[float]:
    """Compute an averaged resonance loss for one or more target resonances.

    The target list is the source of truth. When a paper specifies two
    resonance frequencies, the simulated curve is expected to provide two
    resonances as well. Missing simulated resonances receive a full penalty of
    1.0 so BO is steered toward recovering the absent dip.
    """

    target_values = sorted(
        float(value)
        for value in target
        if value is not None and float(value) > 0.0
    )
    if not target_values:
        return None
    actual_values = sorted(
        float(value)
        for value in actual
        if value is not None and float(value) > 0.0
    )
    errors = []
    for index, target_value in enumerate(target_values):
        if index < len(actual_values):
            errors.append(
                normalized_relative_error(
                    actual_values[index],
                    target_value,
                    epsilon=epsilon,
                )
            )
        else:
            errors.append(1.0)
    return sum(errors) / len(errors)


def resonance_count_error_optional(
    actual_count: int,
    target_count: Optional[int],
) -> Optional[float]:
    """Return a hard loss when the simulated modal count misses the target."""

    if target_count is None or target_count <= 1:
        return None
    actual_value = max(0, int(actual_count))
    target_value = max(1, int(target_count))
    if actual_value == target_value:
        return 0.0
    count_error = abs(actual_value - target_value) / target_value
    return max(1.0, count_error)


def bandwidth_compliance_error_optional(
    s11_samples: Sequence[Sequence[float]],
    target_start: Optional[float],
    target_end: Optional[float],
    threshold_db: float = -10.0,
    epsilon: float = OBJECTIVE_EPSILON,
) -> BandwidthComplianceResult:
    """Measure whether the whole paper bandwidth satisfies the -10 dB limit.

    Edge-only bandwidth comparison is insufficient for wideband reconstruction:
    a simulated curve can cross -10 dB near the paper start/end frequencies but
    still rise above -10 dB in the middle of the band. That design would look
    acceptable to an edge loss while failing the actual wideband return-loss
    requirement.

    This objective instead evaluates every sampled S11 point inside the paper
    interval. Full-band compliance better matches the antenna goal: every
    frequency that the paper claims as usable bandwidth should satisfy the
    return-loss threshold, not only the two endpoints.

    The prompt specifies violations as max(0, s11_db + 10). The per-sample
    violation is normalized by the 10 dB return-loss scale, keeping E_bw in the
    same rough 0..1 range as the other normalized objective terms: -9 dB gives
    0.1, while 0 dB gives 1.0.
    """

    if target_start is None or target_end is None:
        return BandwidthComplianceResult(
            error=None,
            target_start_ghz=target_start,
            target_end_ghz=target_end,
            evaluated_sample_count=0,
        )

    start = float(target_start)
    end = float(target_end)
    if end <= start:
        return BandwidthComplianceResult(
            error=None,
            target_start_ghz=start,
            target_end_ghz=end,
            evaluated_sample_count=0,
        )

    in_band_samples = [
        (float(freq), float(s11_db))
        for freq, s11_db in s11_samples
        if start <= float(freq) <= end
    ]
    if not in_band_samples:
        return BandwidthComplianceResult(
            error=1.0,
            target_start_ghz=start,
            target_end_ghz=end,
            evaluated_sample_count=0,
        )

    normalizer = max(abs(float(threshold_db)), epsilon)
    violations = [
        max(0.0, s11_db - float(threshold_db)) / normalizer
        for _freq, s11_db in in_band_samples
    ]
    return BandwidthComplianceResult(
        error=sum(violations) / len(violations),
        target_start_ghz=start,
        target_end_ghz=end,
        evaluated_sample_count=len(in_band_samples),
    )


def _extract_bandwidth_target(paper: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract 10 dB bandwidth width and edge targets from Paper_Performance."""

    bandwidth = paper.get("Bandwidth_10dB")
    if isinstance(bandwidth, dict):
        direct = _number_or_none(bandwidth.get("Bandwidth_GHz"))
        if direct is not None:
            width = direct
        else:
            width = None
        start = _number_or_none(bandwidth.get("Start_GHz"))
        end = _number_or_none(bandwidth.get("End_GHz"))
        if width is None and start is not None and end is not None:
            width = abs(float(end) - float(start))
        return width, start, end
    return _number_or_none(paper.get("Bandwidth_GHz")), None, None


def _extract_resonance_targets(
    paper: Dict[str, Any],
    fallback_target_frequency_ghz: float,
) -> tuple[tuple[float, ...], Optional[float]]:
    """Extract one or more paper resonance frequencies.

    `Target_Resonances_GHz` may be a scalar or a list. Explicit non-positive
    values like `[0.0]` disable the resonance loss instead of falling back.
    """

    resonance_values = _positive_numbers(paper.get("Target_Resonances_GHz"))
    if not resonance_values:
        legacy_resonance = _number_or_none(paper.get("Target_Resonance_GHz"))
        if legacy_resonance is not None and legacy_resonance > 0.0:
            resonance_values = (float(legacy_resonance),)
    if (
        not resonance_values
        and "Target_Resonances_GHz" not in paper
        and "Target_Resonance_GHz" not in paper
    ):
        resonance_values = (float(fallback_target_frequency_ghz),)
    return resonance_values, (resonance_values[0] if resonance_values else None)


def _extract_resonance_count_target(
    paper: Dict[str, Any],
    resonance_values: Sequence[float],
) -> Optional[int]:
    """Extract the intended number of S11 modes when the paper specifies it."""

    explicit_count = _number_or_none(paper.get("Resonance_Count"))
    if explicit_count is not None and explicit_count > 0.0:
        return max(1, int(round(explicit_count)))
    if len(resonance_values) > 1:
        return len(resonance_values)
    return None


def _extract_s11_target(paper: Dict[str, Any], fallback_target_s11_db: float) -> float:
    """Extract an S11 target in dB, defaulting to the 10 dB return-loss target."""

    for key in ("Target_S11_dB", "S11_Target_dB", "Minimum_S11_dB", "Return_Loss_dB"):
        value = _number_or_none(paper.get(key))
        if value is not None:
            return float(value)
    if isinstance(paper.get("Bandwidth_10dB"), dict):
        return -10.0
    return float(fallback_target_s11_db)


def _extract_actual_gain(geometry_metrics: Dict[str, Any]) -> Optional[float]:
    """Read a simulated gain value from metrics when a future exporter provides one."""

    for key in ("gain_dbi", "peak_gain_dbi", "simulated_gain_dbi"):
        value = _number_or_none(geometry_metrics.get(key))
        if value is not None:
            return value
    return None


def _first_number(value: Any) -> Optional[float]:
    """Return the first numeric value from a list-like object."""

    if isinstance(value, (list, tuple)) and value:
        return _number_or_none(value[0])
    return _number_or_none(value)


def _first_positive_number(value: Any) -> Optional[float]:
    """Return the first positive numeric value from a list-like object."""

    values = value if isinstance(value, (list, tuple)) else (value,)
    for item in values:
        number = _number_or_none(item)
        if number is not None and number > 0.0:
            return number
    return None


def _positive_numbers(value: Any) -> tuple[float, ...]:
    """Return all positive numeric values from a scalar or list-like object."""

    values = value if isinstance(value, (list, tuple)) else (value,)
    numbers = []
    for item in values:
        number = _number_or_none(item)
        if number is not None and number > 0.0:
            numbers.append(float(number))
    return tuple(numbers)



def _number_or_none(value: Any) -> Optional[float]:
    """Convert a value to float, returning None when conversion is not possible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
