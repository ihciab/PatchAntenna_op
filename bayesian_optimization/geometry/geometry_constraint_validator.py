from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesian_optimization.geometry.geometry_validation import (
    GeometryValidationConfig,
    ValidationReport,
    distance_2d,
    extract_component_points,
    validate_geometry,
)


@dataclass
class GeometryConstraintReport:
    valid: bool
    repair_applied: bool
    repair_operations: List[str]
    errors: List[str]
    warnings: List[str]
    geometry_robustness_score: float
    base_validation: Dict[str, Any]
    curvature_max_angle_deg: float
    minimum_gap: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GeometryConstraintValidator:
    """控制点形变后的几何约束验证器。"""

    def __init__(
        self,
        min_gap: float = 0.01,
        max_curvature_angle_deg: float = 160.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.min_gap = min_gap
        self.max_curvature_angle_deg = max_curvature_angle_deg
        self.logger = logger or logging.getLogger(__name__)
        self.base_config = GeometryValidationConfig()

    def validate_and_repair(
        self,
        payload: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], GeometryConstraintReport]:
        repaired, base_report = validate_geometry(payload, output_dir=output_dir, config=self.base_config, logger=self.logger)
        report = self._extend_report(repaired, base_report, output_dir)
        return repaired, report

    def validate(self, payload: Dict[str, Any], output_dir: Optional[Path] = None) -> GeometryConstraintReport:
        _, base_report = validate_geometry(payload, output_dir=output_dir, config=self.base_config, logger=self.logger)
        return self._extend_report(payload, base_report, output_dir)

    def _extend_report(
        self,
        payload: Dict[str, Any],
        base_report: ValidationReport,
        output_dir: Optional[Path],
    ) -> GeometryConstraintReport:
        errors = list(base_report.errors)
        warnings = list(base_report.warnings)
        max_angle = max_curvature_angle(payload)
        if max_angle > self.max_curvature_angle_deg:
            errors.append(f"curvature spike detected: max_angle={max_angle:.6f} deg")

        minimum_gap = minimum_non_adjacent_gap(payload)
        if minimum_gap is not None and minimum_gap < self.min_gap:
            errors.append(f"minimum gap too small: gap={minimum_gap:.12g}")

        score = compute_geometry_robustness_score(
            base_score=base_report.robustness_score,
            max_angle=max_angle,
            max_angle_allowed=self.max_curvature_angle_deg,
            minimum_gap=minimum_gap,
            min_gap_allowed=self.min_gap,
            error_count=len(errors),
        )
        report = GeometryConstraintReport(
            valid=not errors,
            repair_applied=base_report.repair_applied,
            repair_operations=base_report.repair_operations,
            errors=errors,
            warnings=warnings,
            geometry_robustness_score=score,
            base_validation=base_report.to_dict(),
            curvature_max_angle_deg=max_angle,
            minimum_gap=minimum_gap,
        )

        if output_dir is not None:
            debug_dir = Path(output_dir) / "deformation_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            with (debug_dir / "constraint_report.json").open("w", encoding="utf-8") as file:
                json.dump(report.to_dict(), file, ensure_ascii=False, indent=2)
            plot_deformation_debug(payload, debug_dir / "deformed_geometry.png", title="Deformed Geometry")
        return report


def max_curvature_angle(payload: Dict[str, Any]) -> float:
    max_angle = 0.0
    for component in payload.get("components", []) or []:
        points = extract_component_points(component)
        for index in range(1, len(points) - 1):
            angle = turning_angle_deg(points[index - 1], points[index], points[index + 1])
            max_angle = max(max_angle, angle)
    return max_angle


def turning_angle_deg(a, b, c) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 180.0
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(dot))


def minimum_non_adjacent_gap(payload: Dict[str, Any]) -> Optional[float]:
    all_points: List[Tuple[int, int, Any]] = []
    component_lengths: Dict[int, int] = {}
    for component_index, component in enumerate(payload.get("components", []) or []):
        points = extract_component_points(component)
        component_lengths[component_index] = len(points)
        for point_index, point in enumerate(points):
            all_points.append((component_index, point_index, point))
    if len(all_points) < 4:
        return None
    best: Optional[float] = None
    for i in range(len(all_points)):
        ci, pi, p = all_points[i]
        for j in range(i + 1, len(all_points)):
            cj, pj, q = all_points[j]
            if ci == cj and abs(pi - pj) <= 1:
                continue
            if ci == cj and {pi, pj} == {0, component_lengths.get(ci, 0) - 1}:
                continue
            gap = distance_2d(p, q)
            if best is None or gap < best:
                best = gap
    return best


def compute_geometry_robustness_score(
    base_score: float,
    max_angle: float,
    max_angle_allowed: float,
    minimum_gap: Optional[float],
    min_gap_allowed: float,
    error_count: int,
) -> float:
    score = base_score
    if max_angle > max_angle_allowed:
        score -= min(0.25, (max_angle - max_angle_allowed) / 180.0)
    if minimum_gap is not None and minimum_gap < min_gap_allowed:
        score -= 0.20
    score -= min(0.5, error_count * 0.12)
    return max(0.0, min(1.0, score))


def plot_deformation_debug(payload: Dict[str, Any], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for component_index, component in enumerate(payload.get("components", []) or []):
        points = extract_component_points(component)
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, linewidth=1.4, label=f"component {component_index}")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
