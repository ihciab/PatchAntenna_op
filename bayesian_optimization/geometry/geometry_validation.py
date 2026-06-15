from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from shapely.geometry import LineString, Polygon
except Exception:  # pragma: no cover - shapely 是推荐依赖，但保留纯 Python fallback。
    LineString = None
    Polygon = None


Point = Tuple[float, float, float]


@dataclass
class GeometryValidationConfig:
    """几何验证和保守修复参数。单位跟输入 JSON 坐标一致。"""

    closure_tolerance: float = 1e-6
    closure_repair_tolerance: float = 0.01
    connectivity_repair_tolerance: float = 0.01
    planarity_tolerance: float = 1e-9
    planarity_repair_tolerance: float = 1e-6
    min_edge_length: float = 0.01
    area_epsilon: float = 1e-12
    duplicate_tolerance: float = 1e-9
    enable_plots: bool = True


@dataclass
class LoopValidation:
    component_index: int
    closed: bool
    gap_distance: float
    planar: bool
    max_z_deviation: float
    self_intersection: bool
    minimum_edge_length: float
    area: float
    duplicate_point_count: int
    broken_segments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    valid: bool
    repair_applied: bool
    repair_operations: List[str]
    errors: List[str]
    warnings: List[str]
    robustness_score: float
    loops: List[LoopValidation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "repair_applied": self.repair_applied,
            "repair_operations": self.repair_operations,
            "errors": self.errors,
            "warnings": self.warnings,
            "robustness_score": self.robustness_score,
            "loops": [loop.to_dict() for loop in self.loops],
        }


class GeometryRepairer:
    """只做保守修复：snap、去重、删除退化短边、强制小 gap 闭合。"""

    def __init__(self, config: Optional[GeometryValidationConfig] = None) -> None:
        self.config = config or GeometryValidationConfig()
        self.repair_operations: List[str] = []

    def repair_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        repaired = copy.deepcopy(payload)
        for component_index, component in enumerate(repaired.get("components", []) or []):
            points = extract_component_points(component)
            if not points:
                continue

            segment_count = len(component.get("segments") or component.get("primitives") or [])
            explicit_closure = bool(component.get("closed", False)) and segment_count > 0 and len(points) == segment_count + 1
            points = self._repair_points(
                points,
                component_index,
                bool(component.get("closed", False)),
                explicit_closure=explicit_closure,
            )
            write_component_points(component, points)
        return repaired

    def _repair_points(
        self,
        points: List[Point],
        component_index: int,
        closed: bool,
        explicit_closure: bool = False,
    ) -> List[Point]:
        before_count = len(points)
        points = remove_consecutive_duplicates(points, self.config.duplicate_tolerance)
        removed = before_count - len(points)
        if removed > 0:
            self.repair_operations.append(
                f"component {component_index}: remove {removed} consecutive duplicate points"
            )

        before_count = len(points)
        points = remove_degenerate_edges(points, self.config.min_edge_length, closed)
        removed = before_count - len(points)
        if removed > 0:
            self.repair_operations.append(
                f"component {component_index}: remove {removed} degenerate tiny-edge points"
            )

        if len(points) >= 3:
            z_values = [point[2] for point in points]
            mean_z = sum(z_values) / len(z_values)
            max_dev = max(abs(z - mean_z) for z in z_values)
            if self.config.planarity_tolerance < max_dev < self.config.planarity_repair_tolerance:
                points = [(x, y, mean_z) for x, y, _ in points]
                self.repair_operations.append(
                    f"component {component_index}: flatten z to mean plane {mean_z:.12g}"
                )

        if closed and len(points) >= 3:
            gap = distance_2d(points[0], points[-1])
            if gap > self.config.closure_tolerance:
                if explicit_closure:
                    points[-1] = points[0]
                    self.repair_operations.append(
                        f"component {component_index}: snap explicit closure point to first, gap={gap:.12g}"
                    )
                else:
                    points.append(points[0])
                    self.repair_operations.append(
                        f"component {component_index}: append explicit closure point, gap={gap:.12g}"
                    )
            else:
                points[-1] = points[0]

        return points


class GeometryValidator:
    """CST 前几何验证层，完全在 Python 内运行，不依赖 CST。"""

    def __init__(
        self,
        config: Optional[GeometryValidationConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or GeometryValidationConfig()
        self.logger = logger or logging.getLogger(__name__)

    def validate_and_repair(
        self,
        payload: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], ValidationReport]:
        output_dir = Path(output_dir) if output_dir else None
        validation_logs = output_dir / "validation_logs" if output_dir else None
        debug_plots = output_dir / "geometry_debug_plots" if output_dir else None
        if validation_logs:
            validation_logs.mkdir(parents=True, exist_ok=True)
        if debug_plots:
            debug_plots.mkdir(parents=True, exist_ok=True)

        repairer = GeometryRepairer(self.config)
        repaired_payload = repairer.repair_payload(payload)
        report = self.validate(repaired_payload, repairer.repair_operations)

        if validation_logs:
            write_json(validation_logs / "validation_report.json", report.to_dict())
            write_json(validation_logs / "repaired_geometry.json", repaired_payload)
            if not report.valid:
                write_json(validation_logs / "invalid_geometry.json", repaired_payload)
            (validation_logs / "repair_history.txt").write_text(
                "\n".join(report.repair_operations) or "no repair applied",
                encoding="utf-8",
            )

        if debug_plots and self.config.enable_plots:
            self._plot_geometry(payload, debug_plots / "original_geometry.png", title="Original Geometry")
            self._plot_geometry(
                repaired_payload,
                debug_plots / "repaired_geometry.png",
                title="Repaired Geometry",
                report=report,
            )
            self._plot_invalid_features(
                repaired_payload,
                report,
                debug_plots / "invalid_features.png",
            )

        if report.errors:
            self.logger.warning("geometry validation failed: %s", report.errors)
        elif report.repair_applied:
            self.logger.info("geometry repaired before CST: %s", report.repair_operations)
        return repaired_payload, report

    def validate(
        self,
        payload: Dict[str, Any],
        repair_operations: Optional[List[str]] = None,
    ) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []
        loop_reports: List[LoopValidation] = []

        for component_index, component in enumerate(payload.get("components", []) or []):
            points = extract_component_points(component)
            if len(points) < 3:
                errors.append(f"component {component_index}: less than 3 points")
                continue

            closed_expected = bool(component.get("closed", False))
            gap = distance_2d(points[0], points[-1])
            closed = gap < self.config.closure_tolerance
            if closed_expected and not closed:
                errors.append(f"component {component_index}: open loop gap={gap:.12g}")
            if not closed_expected:
                errors.append(f"component {component_index}: component is not marked closed for CST extrusion")

            broken_segments = validate_segment_connectivity(component, self.config)
            for broken in broken_segments:
                warnings.append(
                    f"component {component_index}: broken segment {broken['segment_index']} gap={broken['gap_distance']:.12g}"
                )

            planar, max_z_deviation = validate_planarity(points, self.config)
            if not planar:
                errors.append(f"component {component_index}: non-planar max_z_deviation={max_z_deviation:.12g}")

            duplicate_count = count_consecutive_duplicates(points, self.config.duplicate_tolerance)
            if duplicate_count > 0:
                warnings.append(f"component {component_index}: duplicate consecutive points={duplicate_count}")

            min_edge = minimum_edge_length(points, closed=closed_expected)
            if min_edge < self.config.min_edge_length:
                errors.append(f"component {component_index}: minimum edge too short={min_edge:.12g}")

            area = polygon_area(points)
            if area <= self.config.area_epsilon:
                errors.append(f"component {component_index}: zero or tiny polygon area={area:.12g}")

            self_intersection = has_self_intersection(points)
            if self_intersection:
                errors.append(f"component {component_index}: self intersection detected")

            loop_reports.append(
                LoopValidation(
                    component_index=component_index,
                    closed=closed,
                    gap_distance=gap,
                    planar=planar,
                    max_z_deviation=max_z_deviation,
                    self_intersection=self_intersection,
                    minimum_edge_length=min_edge,
                    area=area,
                    duplicate_point_count=duplicate_count,
                    broken_segments=broken_segments,
                )
            )

        repair_ops = repair_operations or []
        robustness_score = compute_robustness_score(loop_reports, errors, warnings, self.config)
        return ValidationReport(
            valid=not errors,
            repair_applied=bool(repair_ops),
            repair_operations=repair_ops,
            errors=errors,
            warnings=warnings,
            robustness_score=robustness_score,
            loops=loop_reports,
        )

    def _plot_geometry(
        self,
        payload: Dict[str, Any],
        path: Path,
        title: str,
        report: Optional[ValidationReport] = None,
    ) -> None:
        try:
            fig, ax = plt.subplots(figsize=(7, 7))
            for component_index, component in enumerate(payload.get("components", []) or []):
                points = extract_component_points(component)
                if not points:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                ax.plot(xs, ys, linewidth=1.5, label=f"component {component_index}")
                ax.scatter([xs[0]], [ys[0]], s=18, color="#16a34a")
                ax.scatter([xs[-1]], [ys[-1]], s=18, color="#dc2626")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
        except Exception as exc:
            self.logger.warning("geometry plot failed: %s", exc)

    def _plot_invalid_features(self, payload: Dict[str, Any], report: ValidationReport, path: Path) -> None:
        try:
            fig, ax = plt.subplots(figsize=(7, 7))
            for loop in report.loops:
                component = payload.get("components", [])[loop.component_index]
                points = extract_component_points(component)
                if not points:
                    continue
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                color = "#dc2626" if loop.self_intersection or not loop.closed else "#2563eb"
                ax.plot(xs, ys, linewidth=1.5, color=color)
                if loop.gap_distance >= self.config.closure_tolerance:
                    ax.plot([xs[-1], xs[0]], [ys[-1], ys[0]], "--", color="#f97316", linewidth=1.2)
                for index in range(1, len(points)):
                    if distance_2d(points[index - 1], points[index]) < self.config.min_edge_length:
                        ax.scatter([points[index][0]], [points[index][1]], color="#eab308", s=30)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Invalid Edges / Open Gaps / Self Intersections")
            ax.grid(True, alpha=0.25)
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
        except Exception as exc:
            self.logger.warning("invalid feature plot failed: %s", exc)


def validate_geometry(
    payload: Dict[str, Any],
    output_dir: Optional[Path] = None,
    config: Optional[GeometryValidationConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Any], ValidationReport]:
    """便捷入口：返回 repaired_payload 和 ValidationReport。"""

    return GeometryValidator(config=config, logger=logger).validate_and_repair(payload, output_dir)


def extract_component_points(component: Dict[str, Any]) -> List[Point]:
    raw_points = component.get("resampled_points") or component.get("fallback_points") or component.get("points") or []
    points = parse_points(raw_points)
    if points:
        return points

    primitive_points: List[Point] = []
    for primitive in (component.get("primitives") or []) + (component.get("segments") or []):
        primitive_points.extend(parse_points(primitive.get("control_points") or primitive.get("points") or []))
        for key in ("start", "end", "center"):
            value = primitive.get(key)
            if is_point(value):
                primitive_points.append(to_point(value))
    return primitive_points


def write_component_points(component: Dict[str, Any], points: List[Point]) -> None:
    json_points = [point_to_json(point) for point in points]
    if "resampled_points" in component:
        component["resampled_points"] = json_points
    if "fallback_points" in component:
        component["fallback_points"] = json_points
    if "points" in component and "resampled_points" not in component:
        component["points"] = json_points


def validate_segment_connectivity(component: Dict[str, Any], config: GeometryValidationConfig) -> List[Dict[str, Any]]:
    segments = component.get("segments") or component.get("primitives") or []
    broken: List[Dict[str, Any]] = []
    if len(segments) < 2:
        return broken

    for index in range(len(segments) - 1):
        end_point = segment_end_point(segments[index])
        start_point = segment_start_point(segments[index + 1])
        if end_point is None or start_point is None:
            continue
        gap = distance_2d(end_point, start_point)
        if gap > config.closure_tolerance:
            broken.append({"segment_index": index, "gap_distance": gap})
    return broken


def segment_start_point(segment: Dict[str, Any]) -> Optional[Point]:
    if is_point(segment.get("start")):
        return to_point(segment["start"])
    points = parse_points(segment.get("control_points") or segment.get("points") or [])
    return points[0] if points else None


def segment_end_point(segment: Dict[str, Any]) -> Optional[Point]:
    if is_point(segment.get("end")):
        return to_point(segment["end"])
    points = parse_points(segment.get("control_points") or segment.get("points") or [])
    return points[-1] if points else None


def parse_points(raw_points: Any) -> List[Point]:
    if not isinstance(raw_points, list):
        return []
    return [to_point(point) for point in raw_points if is_point(point)]


def is_point(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        float(value[0])
        float(value[1])
        if len(value) >= 3:
            float(value[2])
    except (TypeError, ValueError):
        return False
    return True


def to_point(value: Sequence[Any]) -> Point:
    z = float(value[2]) if len(value) >= 3 else 0.0
    return float(value[0]), float(value[1]), z


def point_to_json(point: Point) -> List[float]:
    if abs(point[2]) <= 1e-15:
        return [float(point[0]), float(point[1])]
    return [float(point[0]), float(point[1]), float(point[2])]


def remove_consecutive_duplicates(points: List[Point], tolerance: float) -> List[Point]:
    if not points:
        return []
    cleaned = [points[0]]
    for point in points[1:]:
        if distance_3d(cleaned[-1], point) > tolerance:
            cleaned.append(point)
    return cleaned


def remove_degenerate_edges(points: List[Point], threshold: float, closed: bool) -> List[Point]:
    if len(points) <= 3:
        return points
    cleaned = [points[0]]
    for point in points[1:]:
        if distance_2d(cleaned[-1], point) >= threshold:
            cleaned.append(point)
    return cleaned


def count_consecutive_duplicates(points: List[Point], tolerance: float) -> int:
    return sum(1 for index in range(1, len(points)) if distance_3d(points[index - 1], points[index]) <= tolerance)


def validate_planarity(points: List[Point], config: GeometryValidationConfig) -> Tuple[bool, float]:
    if not points:
        return False, math.inf
    z_ref = points[0][2]
    max_dev = max(abs(point[2] - z_ref) for point in points)
    return max_dev < config.planarity_tolerance, max_dev


def minimum_edge_length(points: List[Point], closed: bool) -> float:
    if len(points) < 2:
        return 0.0
    lengths = [distance_2d(points[index - 1], points[index]) for index in range(1, len(points))]
    if closed and len(points) >= 3 and distance_2d(points[0], points[-1]) > 1e-12:
        lengths.append(distance_2d(points[0], points[-1]))
    return min(lengths) if lengths else 0.0


def polygon_area(points: List[Point]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    xy = [(point[0], point[1]) for point in points]
    for index in range(len(xy)):
        x1, y1 = xy[index]
        x2, y2 = xy[(index + 1) % len(xy)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def has_self_intersection(points: List[Point]) -> bool:
    xy = [(point[0], point[1]) for point in points]
    if len(xy) < 4:
        return False
    if Polygon is not None:
        polygon = Polygon(xy)
        return not polygon.is_valid
    return pure_python_self_intersection(xy)


def pure_python_self_intersection(points: List[Tuple[float, float]]) -> bool:
    edges = list(zip(points[:-1], points[1:]))
    if points[0] != points[-1]:
        edges.append((points[-1], points[0]))
    for i, edge_a in enumerate(edges):
        for j, edge_b in enumerate(edges):
            if j <= i + 1:
                continue
            if i == 0 and j == len(edges) - 1:
                continue
            if segments_intersect(edge_a[0], edge_a[1], edge_b[0], edge_b[1]):
                return True
    return False


def segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def distance_2d(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_3d(a: Point, b: Point) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def compute_robustness_score(
    loops: List[LoopValidation],
    errors: List[str],
    warnings: List[str],
    config: GeometryValidationConfig,
) -> float:
    if not loops:
        return 0.0
    score = 1.0
    score -= min(0.7, 0.15 * len(errors))
    score -= min(0.2, 0.03 * len(warnings))
    for loop in loops:
        if loop.gap_distance > config.closure_tolerance:
            score -= min(0.2, loop.gap_distance / max(config.closure_repair_tolerance, 1e-12) * 0.1)
        if loop.minimum_edge_length < config.min_edge_length:
            score -= 0.1
        if loop.self_intersection:
            score -= 0.4
        if not loop.planar:
            score -= 0.2
    return max(0.0, min(1.0, score))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
