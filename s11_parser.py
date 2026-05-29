from __future__ import annotations

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
    point_count: int

    def to_dict(self):
        data = asdict(self)
        data["s11_path"] = str(self.s11_path)
        return data


def parse_s11_file(path: Path, target_frequency_ghz: float = 2.4, threshold_db: float = -10.0) -> S11Metrics:
    rows = read_s11_rows(path)
    if not rows:
        raise ValueError(f"S11 文件没有可解析数据: {path}")

    resonance_freq, min_s11 = min(rows, key=lambda item: item[1])
    target_s11 = interpolate_s11(rows, target_frequency_ghz)
    bandwidth = bandwidth_below_threshold(rows, threshold_db)
    return S11Metrics(
        s11_path=path,
        resonant_frequency_ghz=resonance_freq,
        minimum_s11_db=min_s11,
        s11_at_target_db=target_s11,
        bandwidth_ghz=bandwidth,
        point_count=len(rows),
    )


def read_s11_rows(path: Path) -> List[Point]:
    if not path.exists():
        raise FileNotFoundError(f"S11 文件不存在: {path}")
    text = path.read_text(encoding="utf-8-sig", errors="ignore")

    rows = _read_csv_like(text)
    if not rows:
        rows = _read_regex_pairs(text)
    rows = [(freq, value) for freq, value in rows if math.isfinite(freq) and math.isfinite(value)]
    rows.sort(key=lambda item: item[0])
    return rows


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
    below = [freq for freq, value in rows if value <= threshold_db]
    if not below:
        return 0.0
    return max(below) - min(below)


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
