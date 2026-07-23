"""Experiment 1 single-band patch antenna objective for design-agent BO.

This objective is separate from the existing ``bayesian_optimization``
objective code.  It is tailored to the design-agent slot-tuning stage for the
single-band 2.45 GHz patch antenna experiment.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from bayesian_optimization.optimization.s11_parser import interpolate_s11, read_s11_rows


Point = Tuple[float, float]

TARGET_FREQUENCY_GHZ = 2.45
TARGET_S11_DB = -15.0
TARGET_GAIN_DBI = 6.0
FREQUENCY_WEIGHT = 5.0
MATCHING_WEIGHT = 3.0
GAIN_WEIGHT = 1.0


@dataclass(frozen=True)
class Experiment1ObjectiveResult:
    """Serializable objective breakdown for one BO evaluation."""

    loss: float
    weighted_losses: Dict[str, float]
    normalized_losses: Dict[str, float]
    metrics: Dict[str, Optional[float]]
    targets: Dict[str, float]
    weights: Dict[str, float]
    status: str
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_experiment1_objective_from_files(
    s11_path: Path | str,
    simulation_summary_path: Optional[Path | str] = None,
    *,
    gain_dbi: Optional[float] = None,
    output_path: Optional[Path | str] = None,
) -> Experiment1ObjectiveResult:
    """Evaluate the Experiment 1 objective from S11 and optional gain summary."""

    s11_file = Path(s11_path)
    rows = read_s11_rows(s11_file)
    if not rows:
        raise ValueError(f"S11 file has no parseable rows: {s11_file}")

    summary = _load_optional_json(simulation_summary_path)
    gain = _number_or_none(gain_dbi)
    if gain is None:
        gain = _extract_gain(summary)

    result = evaluate_experiment1_objective(rows, gain_dbi=gain)
    payload = result.to_dict()
    payload["source_files"] = {
        "s11_path": str(s11_file.resolve()),
        "simulation_summary_path": (
            str(Path(simulation_summary_path).resolve()) if simulation_summary_path is not None else None
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def evaluate_experiment1_objective(
    s11_rows: Sequence[Point],
    *,
    gain_dbi: Optional[float] = None,
) -> Experiment1ObjectiveResult:
    """Compute normalized goal-oriented loss for Experiment 1.

    Loss terms:
    - Lf = ((fr - 2.45) / 2.45) ** 2
    - Lmatch = max(0, (s11_at_2.45 - (-15)) / 15)
    - Lgain = max(0, (6 - gain) / 6)
    """

    rows = _clean_rows(s11_rows)
    if not rows:
        raise ValueError("S11 rows are empty after numeric cleanup.")

    resonant_frequency, minimum_s11 = min(rows, key=lambda item: item[1])
    s11_at_target = interpolate_s11(rows, TARGET_FREQUENCY_GHZ)

    frequency_loss = ((resonant_frequency - TARGET_FREQUENCY_GHZ) / TARGET_FREQUENCY_GHZ) ** 2
    matching_loss = max(0.0, (s11_at_target - TARGET_S11_DB) / abs(TARGET_S11_DB))

    warnings = []
    gain = _number_or_none(gain_dbi)
    if gain is None:
        gain_loss = 1.0
        warnings.append("realized gain missing; gain loss set to 1.0")
    else:
        gain_loss = max(0.0, (TARGET_GAIN_DBI - gain) / TARGET_GAIN_DBI)

    weighted = {
        "frequency": FREQUENCY_WEIGHT * frequency_loss,
        "matching": MATCHING_WEIGHT * matching_loss,
        "gain": GAIN_WEIGHT * gain_loss,
    }
    total = sum(weighted.values())

    return Experiment1ObjectiveResult(
        loss=float(total),
        weighted_losses=weighted,
        normalized_losses={
            "frequency": float(frequency_loss),
            "matching": float(matching_loss),
            "gain": float(gain_loss),
        },
        metrics={
            "resonant_frequency_ghz": float(resonant_frequency),
            "minimum_s11_db": float(minimum_s11),
            "s11_at_2p45_ghz_db": float(s11_at_target),
            "realized_gain_dbi": None if gain is None else float(gain),
        },
        targets={
            "resonant_frequency_ghz": TARGET_FREQUENCY_GHZ,
            "s11_at_2p45_ghz_db_max": TARGET_S11_DB,
            "realized_gain_dbi_min": TARGET_GAIN_DBI,
        },
        weights={
            "frequency": FREQUENCY_WEIGHT,
            "matching": MATCHING_WEIGHT,
            "gain": GAIN_WEIGHT,
        },
        status="ok",
        warnings=tuple(warnings),
    )


def _clean_rows(rows: Sequence[Point]) -> Tuple[Point, ...]:
    clean = []
    for freq, value in rows:
        try:
            f = float(freq)
            s11 = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and math.isfinite(s11):
            clean.append((f, s11))
    return tuple(sorted(clean, key=lambda item: item[0]))


def _load_optional_json(path: Optional[Path | str]) -> Dict[str, Any]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.exists() or json_path.stat().st_size == 0:
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _extract_gain(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("gain", "gain_dbi", "realized_gain_dbi", "peak_gain_dbi"):
        value = _number_or_none(payload.get(key))
        if value is not None:
            return value
    current = payload.get("current") if isinstance(payload.get("current"), dict) else {}
    for key in ("gain_dbi", "realized_gain_dbi", "peak_gain_dbi"):
        value = _number_or_none(current.get(key))
        if value is not None:
            return value
    return _find_nested_number(payload, {"gain", "gain_dbi", "realized_gain_dbi", "peak_gain_dbi"})


def _find_nested_number(value: Any, keys: set[str]) -> Optional[float]:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                number = _number_or_none(item)
                if number is not None:
                    return number
        for item in value.values():
            number = _find_nested_number(item, keys)
            if number is not None:
                return number
    elif isinstance(value, list):
        for item in value:
            number = _find_nested_number(item, keys)
            if number is not None:
                return number
    return None


def _number_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "Experiment1ObjectiveResult",
    "evaluate_experiment1_objective",
    "evaluate_experiment1_objective_from_files",
]
