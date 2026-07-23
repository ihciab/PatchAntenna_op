"""Summarize CST antenna simulation results for design-agent feedback."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bayesian_optimization.optimization.s11_parser import (
    find_latest_s11_file,
    interpolate_s11,
    read_s11_rows,
    resonant_frequencies_from_rows,
)


Point = Tuple[float, float]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_INPUTS_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"
SIMULATION_SUMMARY_FILENAME = "simulation_summary.json"


@dataclass(frozen=True)
class SimulationSummary:
    """Compact simulation summary intended for LLM design feedback."""

    target: Dict[str, Optional[float]]
    current: Dict[str, Optional[float]]
    gap_to_target: Dict[str, Optional[float]]
    passed: Dict[str, Optional[bool]]
    resonance: List[float]
    target_resonance: float
    frequency_error: Optional[float]
    bandwidth: float
    target_bandwidth: Optional[float]
    target_s11: float
    peak_s11: float
    s11_at_target: float
    s11_error: float
    target_gain: float
    gain: Optional[float]
    gain_error: Optional[float]
    bandwidth_start: Optional[float]
    bandwidth_end: Optional[float]
    s11_threshold: float
    point_count: int
    s11_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


class SimulationSummaryBuilder:
    """Build ``simulation_summary.json`` from an exported S11 curve."""

    def __init__(
        self,
        target_resonance: float = 2.45,
        target_bandwidth: Optional[float] = None,
        target_s11: float = -15.0,
        target_gain: float = 6.0,
        s11_threshold: Optional[float] = None,
        resonance_limit: int = 3,
    ) -> None:
        """Configure target metrics and S11 interpretation.

        Frequencies and bandwidths are interpreted in the same unit used by the
        S11 curve. CST exports in this project are normally in GHz.
        """

        self.target_resonance = float(target_resonance)
        self.target_bandwidth = None if target_bandwidth is None else float(target_bandwidth)
        self.target_s11 = float(target_s11)
        self.target_gain = float(target_gain)
        self.s11_threshold = self.target_s11 if s11_threshold is None else float(s11_threshold)
        self.resonance_limit = int(resonance_limit)

    def from_s11_file(self, s11_path: Path | str, gain: Optional[float] = None) -> SimulationSummary:
        """Parse an S11 file and return a compact simulation summary."""

        path = Path(s11_path)
        rows = read_s11_rows(path)
        if not rows:
            raise ValueError("S11 curve has no parseable data: {0}".format(path))
        return self.from_rows(rows=rows, s11_path=path, gain=gain)

    def from_rows(
        self,
        rows: Sequence[Point],
        s11_path: Path | str,
        gain: Optional[float] = None,
    ) -> SimulationSummary:
        """Build a summary from already parsed S11 rows."""

        clean_rows = sorted(
            [(float(freq), float(value)) for freq, value in rows if math.isfinite(freq) and math.isfinite(value)],
            key=lambda item: item[0],
        )
        if not clean_rows:
            raise ValueError("S11 rows are empty after numeric cleanup.")

        peak_frequency, peak_s11 = min(clean_rows, key=lambda item: item[1])
        resonances = list(
            resonant_frequencies_from_rows(
                clean_rows,
                limit=self.resonance_limit,
                threshold_db=self.s11_threshold,
            )
        )
        if not resonances:
            resonances = [float(peak_frequency)]

        selected_resonance = closest_resonance(resonances, self.target_resonance)
        frequency_error = (
            round_float(selected_resonance - self.target_resonance)
            if selected_resonance is not None
            else None
        )

        bandwidth_start, bandwidth_end = interpolated_bandwidth_edges_below_threshold(
            clean_rows,
            self.s11_threshold,
        )
        bandwidth = (
            round_float(bandwidth_end - bandwidth_start)
            if bandwidth_start is not None and bandwidth_end is not None
            else 0.0
        )
        s11_at_target = round_float(interpolate_s11(clean_rows, self.target_resonance))
        s11_error = round_float(s11_at_target - self.target_s11)
        gain_value = None if gain is None else round_float(gain)
        gain_error = None if gain_value is None else round_float(self.target_gain - gain_value)
        bandwidth_error = (
            None
            if self.target_bandwidth is None
            else round_float(self.target_bandwidth - bandwidth)
        )
        current = {
            "f0_ghz": None if selected_resonance is None else round_float(selected_resonance),
            "s11_at_target_db": s11_at_target,
            "peak_s11_db": round_float(peak_s11),
            "gain_dbi": gain_value,
            "bandwidth_ghz": bandwidth,
        }
        target = {
            "f0_ghz": round_float(self.target_resonance),
            "s11_db_max": round_float(self.target_s11),
            "gain_dbi_min": round_float(self.target_gain),
            "bandwidth_ghz_min": None if self.target_bandwidth is None else round_float(self.target_bandwidth),
        }
        gap_to_target = {
            "frequency_error_ghz": frequency_error,
            "s11_error_db": s11_error,
            "gain_error_dbi": gain_error,
            "bandwidth_error_ghz": bandwidth_error,
        }
        passed = {
            "frequency": None,
            "s11": s11_at_target <= self.target_s11,
            "gain": None if gain_value is None else gain_value >= self.target_gain,
            "bandwidth": None if self.target_bandwidth is None else bandwidth >= self.target_bandwidth,
        }
        passed["overall"] = overall_pass_status(passed)

        return SimulationSummary(
            target=target,
            current=current,
            gap_to_target=gap_to_target,
            passed=passed,
            resonance=[round_float(value) for value in resonances],
            target_resonance=round_float(self.target_resonance),
            frequency_error=frequency_error,
            bandwidth=bandwidth,
            target_bandwidth=None if self.target_bandwidth is None else round_float(self.target_bandwidth),
            target_s11=round_float(self.target_s11),
            peak_s11=round_float(peak_s11),
            s11_at_target=s11_at_target,
            s11_error=s11_error,
            target_gain=round_float(self.target_gain),
            gain=gain_value,
            gain_error=gain_error,
            bandwidth_start=None if bandwidth_start is None else round_float(bandwidth_start),
            bandwidth_end=None if bandwidth_end is None else round_float(bandwidth_end),
            s11_threshold=round_float(self.s11_threshold),
            point_count=len(clean_rows),
            s11_path=str(Path(s11_path).resolve()),
        )


def build_simulation_summary(
    s11_path: Path | str,
    target_resonance: float = 2.45,
    target_bandwidth: Optional[float] = None,
    target_s11: float = -15.0,
    target_gain: float = 6.0,
    s11_threshold: Optional[float] = None,
    gain: Optional[float] = None,
    resonance_limit: int = 3,
) -> Dict[str, Any]:
    """Convenience function returning the summary dictionary."""

    builder = SimulationSummaryBuilder(
        target_resonance=target_resonance,
        target_bandwidth=target_bandwidth,
        target_s11=target_s11,
        target_gain=target_gain,
        s11_threshold=s11_threshold,
        resonance_limit=resonance_limit,
    )
    return builder.from_s11_file(s11_path=s11_path, gain=gain).to_dict()


def write_simulation_summary(
    output_path: Optional[Path | str],
    s11_path: Path | str,
    target_resonance: float = 2.45,
    target_bandwidth: Optional[float] = None,
    target_s11: float = -15.0,
    target_gain: float = 6.0,
    s11_threshold: Optional[float] = None,
    gain: Optional[float] = None,
    resonance_limit: int = 3,
) -> Path:
    """Build and write ``simulation_summary.json`` to the shared agent-input folder by default."""

    summary = build_simulation_summary(
        s11_path=s11_path,
        target_resonance=target_resonance,
        target_bandwidth=target_bandwidth,
        target_s11=target_s11,
        target_gain=target_gain,
        s11_threshold=s11_threshold,
        gain=gain,
        resonance_limit=resonance_limit,
    )
    path = Path(output_path) if output_path is not None else default_simulation_summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def default_simulation_summary_path() -> Path:
    """Return the shared simulation summary path consumed by later agents."""

    return AGENT_INPUTS_DIR / SIMULATION_SUMMARY_FILENAME


def resolve_s11_path(s11_path: Optional[Path | str] = None, search_dir: Optional[Path | str] = None) -> Path:
    """Resolve an explicit S11 file or find the newest S11 file below a directory."""

    if s11_path is not None:
        path = Path(s11_path)
        if not path.exists():
            raise FileNotFoundError("S11 file does not exist: {0}".format(path))
        return path
    if search_dir is None:
        raise ValueError("Pass either s11_path or search_dir.")
    return find_latest_s11_file(Path(search_dir))


def load_gain_from_json(path: Path | str, key: str = "gain") -> Optional[float]:
    """Load a gain value from a JSON file using a flexible key lookup."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    value = find_nested_number(payload, preferred_key=key)
    return None if value is None else float(value)


def interpolated_bandwidth_edges_below_threshold(
    rows: Sequence[Point],
    threshold_db: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Return the widest below-threshold band with linear edge interpolation."""

    if not rows:
        return None, None
    clean_rows = sorted(rows, key=lambda item: item[0])
    bands: List[Tuple[float, float]] = []
    start: Optional[float] = None
    previous_freq, previous_value = clean_rows[0]

    if previous_value <= threshold_db:
        start = previous_freq

    for freq, value in clean_rows[1:]:
        previous_below = previous_value <= threshold_db
        current_below = value <= threshold_db

        if not previous_below and current_below:
            start = threshold_crossing_frequency(
                previous_freq,
                previous_value,
                freq,
                value,
                threshold_db,
            )
        elif previous_below and not current_below:
            end = threshold_crossing_frequency(
                previous_freq,
                previous_value,
                freq,
                value,
                threshold_db,
            )
            if start is not None:
                bands.append((start, end))
            start = None

        previous_freq, previous_value = freq, value

    if start is not None:
        bands.append((start, clean_rows[-1][0]))
    if not bands:
        return None, None
    return max(bands, key=lambda band: band[1] - band[0])


def threshold_crossing_frequency(
    f0: float,
    v0: float,
    f1: float,
    v1: float,
    threshold_db: float,
) -> float:
    """Linearly interpolate the frequency where S11 crosses a threshold."""

    if abs(v1 - v0) <= 1e-12:
        return float(f0)
    ratio = (float(threshold_db) - float(v0)) / (float(v1) - float(v0))
    ratio = max(0.0, min(1.0, ratio))
    return float(f0) + ratio * (float(f1) - float(f0))


def find_nested_number(value: Any, preferred_key: str = "gain") -> Optional[float]:
    """Find a numeric gain-like value in nested JSON data."""

    preferred = preferred_key.lower()
    aliases = {
        preferred,
        "gain",
        "gain_dbi",
        "peak_gain_dbi",
        "realized_gain_dbi",
        "simulated_gain_dbi",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in aliases:
                number = number_or_none(item)
                if number is not None:
                    return number
        for item in value.values():
            number = find_nested_number(item, preferred_key=preferred_key)
            if number is not None:
                return number
    elif isinstance(value, list):
        for item in value:
            number = find_nested_number(item, preferred_key=preferred_key)
            if number is not None:
                return number
    return None


def closest_resonance(resonances: Sequence[float], target: float) -> Optional[float]:
    """Return the resonance nearest to the target frequency."""

    if not resonances:
        return None
    return float(min(resonances, key=lambda value: abs(float(value) - float(target))))


def number_or_none(value: Any) -> Optional[float]:
    """Return a finite float or ``None``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def overall_pass_status(passed: Dict[str, Optional[bool]]) -> Optional[bool]:
    """Return overall pass status while preserving unknown required metrics."""

    required = (passed.get("s11"), passed.get("gain"))
    optional = (passed.get("bandwidth"),)
    if any(value is False for value in required + optional):
        return False
    if any(value is None for value in required):
        return None
    return True


def round_float(value: float, digits: int = 6) -> float:
    """Round while avoiding noisy floating-point tails in JSON."""

    return round(float(value), digits)
