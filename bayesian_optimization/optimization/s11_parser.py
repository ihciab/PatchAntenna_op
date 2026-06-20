from __future__ import annotations

import ast
import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class S11Metrics:
    s11_path: Path
    resonant_frequency_ghz: float
    minimum_s11_db: float
    s11_at_target_db: float
    bandwidth_ghz: float
    bandwidth_start_ghz: Optional[float]
    bandwidth_end_ghz: Optional[float]
    point_count: int
    resonant_frequencies_ghz: Tuple[float, ...] = ()
    s11_samples: Tuple[Point, ...] = ()

    def to_dict(self):
        data = asdict(self)
        data["s11_path"] = str(self.s11_path)
        return data


def parse_s11_file(path: Path, target_frequency_ghz: float = 2.4, threshold_db: float = -10.0) -> S11Metrics:
    rows = read_s11_rows(path)
    if not rows:
        raise ValueError(f"S11 文件没有可解析数据: {path}")
    resonance_frequencies = resonant_frequencies_from_rows(
        rows,
        limit=2,
        threshold_db=threshold_db,
    )
    resonance_freq, min_s11 = min(rows, key=lambda item: item[1])
    target_s11 = interpolate_s11(rows, target_frequency_ghz)
    bandwidth_start, bandwidth_end = bandwidth_edges_below_threshold(rows, threshold_db)
    bandwidth = (
        float(bandwidth_end - bandwidth_start)
        if bandwidth_start is not None and bandwidth_end is not None
        else 0.0
    )
    return S11Metrics(
        s11_path=path,
        resonant_frequency_ghz=resonance_freq,
        minimum_s11_db=min_s11,
        s11_at_target_db=target_s11,
        bandwidth_ghz=bandwidth,
        bandwidth_start_ghz=bandwidth_start,
        bandwidth_end_ghz=bandwidth_end,
        point_count=len(rows),
        resonant_frequencies_ghz=resonance_frequencies,
        s11_samples=tuple(rows),
    )


def read_s11_rows(path: Path) -> List[Point]:
    if not path.exists():
        raise FileNotFoundError(f"S11 文件不存在: {path}")
    text = path.read_text(encoding="utf-8-sig", errors="ignore")

    rows = _read_python_tuple_series(text)
    if not rows:
        rows = _read_csv_like(text)
    if not rows:
        rows = _read_regex_pairs(text)
    rows = [(freq, value) for freq, value in rows if math.isfinite(freq) and math.isfinite(value)]
    rows.sort(key=lambda item: item[0])
    return rows

def resonant_frequencies_from_rows(
    rows: Sequence[Point],
    limit: int = 2,
    threshold_db: float = -10.0,
) -> Tuple[float, ...]:
    """Return up to ``limit`` resonance frequencies from local S11 minima.

    The legacy resonance is the single global minimum. For dual-band targets we
    need the next meaningful dip as well. Only minima at or below the return
    loss threshold count as resonances for modal-count matching.
    """

    if not rows or limit <= 0:
        return tuple()
    minima = _local_minima(rows)
    if not minima:
        minima = [min(rows, key=lambda item: item[1])]
    qualified = [point for point in minima if point[1] <= threshold_db]
    if not qualified:
        global_minimum = min(rows, key=lambda item: item[1])
        if global_minimum[1] <= threshold_db:
            qualified = [global_minimum]
    if not qualified:
        return tuple()
    deepest = sorted(qualified, key=lambda item: item[1])[:limit]
    return tuple(freq for freq, _value in sorted(deepest, key=lambda item: item[0]))


def find_latest_s11_file(search_root: Path) -> Path:
    candidates: List[Path] = []
    for pattern in ("s11.csv", "*s11*.csv", "*S11*.csv", "*s11*.txt", "*S11*.txt"):
        candidates.extend(search_root.rglob(pattern))
    files = [path for path in candidates if path.is_file()]
    if not files:
        raise FileNotFoundError(f"未在 {search_root} 下找到 S11 导出文件")
    return max(files, key=lambda path: path.stat().st_mtime)


def interpolate_s11(rows: Sequence[Point], target_frequency_ghz: float) -> float:
    if not rows:
        raise ValueError("empty S11 rows")
    if target_frequency_ghz <= rows[0][0]:
        return rows[0][1]
    if target_frequency_ghz >= rows[-1][0]:
        return rows[-1][1]
    for index in range(1, len(rows)):
        f0, s0 = rows[index - 1]
        f1, s1 = rows[index]
        if f0 <= target_frequency_ghz <= f1:
            if abs(f1 - f0) <= 1e-12:
                return s0
            t = (target_frequency_ghz - f0) / (f1 - f0)
            return s0 + (s1 - s0) * t
    return min(rows, key=lambda item: abs(item[0] - target_frequency_ghz))[1]


def bandwidth_below_threshold(rows: Sequence[Point], threshold_db: float = -10.0) -> float:
    start, end = bandwidth_edges_below_threshold(rows, threshold_db)
    if start is None or end is None:
        return 0.0
    return float(end - start)


def bandwidth_edges_below_threshold(
    rows: Sequence[Point],
    threshold_db: float = -10.0,
) -> Tuple[Optional[float], Optional[float]]:
    """Return the widest contiguous sampled band whose S11 is below threshold.

    The objective compares these simulated band edges against the paper's
    Start_GHz/End_GHz targets. This keeps the legacy bandwidth_ghz value
    available while exposing the extra information needed by the loss.
    """

    bands = _below_threshold_bands(rows, threshold_db)
    if not bands:
        return None, None
    return max(bands, key=lambda band: band[1] - band[0])


def _local_minima(rows: Sequence[Point]) -> List[Point]:
    """Find sampled local minima, including simple flat-bottom dips."""

    if len(rows) < 3:
        return list(rows)

    minima: List[Point] = []
    index = 1
    last_index = len(rows) - 1
    while index < last_index:
        left_value = rows[index - 1][1]
        current_value = rows[index][1]
        right_index = index + 1
        while right_index < len(rows) and rows[right_index][1] == current_value:
            right_index += 1
        if right_index >= len(rows):
            break
        right_value = rows[right_index][1]
        if current_value <= left_value and current_value <= right_value and (
            current_value < left_value or current_value < right_value
        ):
            plateau = rows[index:right_index]
            minima.append(plateau[len(plateau) // 2])
        index = right_index
    return minima


def _below_threshold_bands(
    rows: Sequence[Point],
    threshold_db: float,
) -> List[Tuple[float, float]]:
    """Return sampled contiguous runs whose values stay below threshold."""

    bands: List[Tuple[float, float]] = []
    start: Optional[float] = None
    end: Optional[float] = None
    for freq, value in rows:
        if value <= threshold_db:
            if start is None:
                start = freq
            end = freq
            continue
        if start is not None and end is not None:
            bands.append((start, end))
        start = None
        end = None
    if start is not None and end is not None:
        bands.append((start, end))
    return bands


def _read_python_tuple_series(text: str) -> List[Point]:
    """Read CST exports shaped like [(freq, complex_s11, impedance), ...].

    In this format the second item is complex S11, not dB. The parser converts
    it to dB using 20*log10(abs(S11)) before objective evaluation and plotting.
    """

    stripped = text.strip()
    if not stripped.startswith("["):
        return []
    try:
        items = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(items, list):
        return []

    rows: List[Point] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            freq = float(item[0])
        except (TypeError, ValueError):
            continue
        value = _s11_value_to_db(item[1])
        if value is not None:
            rows.append((freq, value))
    return rows


def _s11_value_to_db(value: object) -> Optional[float]:
    """Convert a raw S11 value to dB when complex, or pass scalar dB through."""

    if isinstance(value, complex):
        magnitude = abs(value)
        if magnitude <= 0.0:
            return -math.inf
        return 20.0 * math.log10(magnitude)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv_like(text: str) -> List[Point]:
    rows: List[Point] = []
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(text.splitlines(), dialect)
    for row_index, row in enumerate(reader, start=1):
        if len(row) < 2:
            continue
        first = row[0].strip()
        second = row[1].strip()
        if row_index == 1 and first.lower() in {"frequency", "freq", "f"}:
            continue
        try:
            rows.append((float(first), float(second)))
        except ValueError:
            continue
    return rows


def _read_regex_pairs(text: str) -> List[Point]:
    rows: List[Point] = []
    pattern = re.compile(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    for line in text.splitlines():
        numbers = pattern.findall(line)
        if len(numbers) < 2:
            continue
        try:
            rows.append((float(numbers[0]), float(numbers[1])))
        except ValueError:
            continue
    return rows
