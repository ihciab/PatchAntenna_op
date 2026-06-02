from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bayesian_optimization.geometry.primitive_mutator import collect_component_cache_points, parse_points


Point = Tuple[float, float]


@dataclass(frozen=True)
class TopologySignature:
    component_count: int
    closed_flags: List[bool]
    node_count: int
    edge_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    valid: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class GeometryValidationConfig:
    min_edge_length_px: float = 0.20
    max_outside_canvas_ratio: float = 0.15
    allow_reference_component_mismatch: bool = False


def make_topology_signature(payload: Dict[str, Any]) -> TopologySignature:
    components = payload.get("components", []) or []
    return TopologySignature(
        component_count=len(components),
        closed_flags=[bool(component.get("closed", False)) for component in components],
        node_count=len(payload.get("nodes", []) or []),
        edge_count=len(payload.get("edges", []) or []),
    )


def validate_geometry(
    payload: Dict[str, Any],
    reference_signature: Optional[TopologySignature] = None,
    config: Optional[GeometryValidationConfig] = None,
) -> ValidationReport:
    """CST 构建前几何验证。

    验证策略偏保守：拓扑正确性和 CST 可重建性优先于几何平滑。
    """

    cfg = config or GeometryValidationConfig()
    report = ValidationReport(valid=True)
    components = payload.get("components", []) or []
    signature = make_topology_signature(payload)

    if reference_signature is not None:
        if signature.component_count != reference_signature.component_count:
            report.reasons.append(
                f"component count changed: {signature.component_count} != {reference_signature.component_count}"
            )
        if signature.closed_flags != reference_signature.closed_flags:
            report.reasons.append("component closed/open topology changed")
        if signature.node_count != reference_signature.node_count:
            report.warnings.append(
                f"node count changed: {signature.node_count} != {reference_signature.node_count}"
            )
        if signature.edge_count != reference_signature.edge_count:
            report.warnings.append(
                f"edge count changed: {signature.edge_count} != {reference_signature.edge_count}"
            )

    all_points = collect_component_cache_points(payload)
    if not all_points:
        report.reasons.append("no reconstructable component points")

    finite_points = [point for point in all_points if _is_finite_point(point)]
    if len(finite_points) != len(all_points):
        report.reasons.append("non-finite coordinate detected")

    tiny_edges = 0
    self_intersections = 0
    open_too_short = 0
    for component_index, component in enumerate(components):
        points = _component_points(component)
        closed = bool(component.get("closed", False))
        min_points = 3 if closed else 2
        if len(points) < min_points:
            report.reasons.append(f"component {component_index} has too few points")
            continue
        tiny_edges += _count_tiny_edges(points, cfg.min_edge_length_px, closed)
        self_intersections += _count_self_intersections(points, closed)
        if not closed and _polyline_length(points) < cfg.min_edge_length_px:
            open_too_short += 1

    if tiny_edges > 0:
        report.reasons.append(f"tiny edge count {tiny_edges} exceeds CST-safe threshold")
    if self_intersections > 0:
        report.reasons.append(f"self intersection count {self_intersections} detected")
    if open_too_short > 0:
        report.reasons.append(f"{open_too_short} open components are too short")

    canvas = payload.get("canvas") or {}
    outside_ratio = _outside_canvas_ratio(
        finite_points,
        float(canvas.get("width", 0) or 0),
        float(canvas.get("height", 0) or 0),
    )
    if outside_ratio > cfg.max_outside_canvas_ratio:
        report.reasons.append(f"too many points outside canvas: {outside_ratio:.3f}")

    report.metrics.update(
        {
            "component_count": len(components),
            "point_count": len(all_points),
            "tiny_edge_count": tiny_edges,
            "self_intersection_count": self_intersections,
            "outside_canvas_ratio": outside_ratio,
            "topology_signature": signature.to_dict(),
        }
    )
    report.valid = not report.reasons
    return report


def geometry_complexity_metrics(payload: Dict[str, Any], validation: Optional[ValidationReport] = None) -> Dict[str, Any]:
    components = payload.get("components", []) or []
    primitive_count = 0
    spline_count = 0
    tiny_segments = 0
    curvature_proxy = 0.0
    for component in components:
        primitives = (component.get("primitives") or []) + (component.get("segments") or [])
        primitive_count += len(primitives)
        for primitive in primitives:
            kind = str(primitive.get("type") or primitive.get("kind") or "").lower()
            if "spline" in kind or kind not in ("line", "arc"):
                spline_count += 1
                curvature_proxy += _spline_curvature_proxy(parse_points(primitive.get("control_points") or primitive.get("points")))
            points = parse_points(primitive.get("points") or [])
            tiny_segments += _count_tiny_edges(points, 0.20, False)

    metrics = {
        "component_count": len(components),
        "primitive_count": primitive_count,
        "spline_count": spline_count,
        "spline_curvature_proxy": curvature_proxy,
        "tiny_segment_count": tiny_segments,
    }
    if validation:
        metrics.update(validation.metrics)
    return metrics


def _component_points(component: Dict[str, Any]) -> List[Point]:
    for key in ("resampled_points", "fallback_points", "points", "sampled_points"):
        points = parse_points(component.get(key))
        if points:
            return points
    return []


def _is_finite_point(point: Point) -> bool:
    return math.isfinite(point[0]) and math.isfinite(point[1])


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(points: Sequence[Point]) -> float:
    return sum(_distance(points[index - 1], points[index]) for index in range(1, len(points)))


def _count_tiny_edges(points: Sequence[Point], min_edge_length: float, closed: bool) -> int:
    if len(points) < 2:
        return 0
    edges = list(zip(points[:-1], points[1:]))
    if closed and _distance(points[0], points[-1]) > 1e-9:
        edges.append((points[-1], points[0]))
    return sum(1 for a, b in edges if _distance(a, b) < min_edge_length)


def _count_self_intersections(points: Sequence[Point], closed: bool) -> int:
    if len(points) < 4:
        return 0
    edges = list(zip(points[:-1], points[1:]))
    if closed and _distance(points[0], points[-1]) > 1e-9:
        edges.append((points[-1], points[0]))

    count = 0
    edge_count = len(edges)
    for i in range(edge_count):
        for j in range(i + 1, edge_count):
            if abs(i - j) <= 1:
                continue
            if closed and {i, j} == {0, edge_count - 1}:
                continue
            if _segments_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                count += 1
    return count


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) <= 1e-9 and on_segment(a, c, b):
        return True
    if abs(o2) <= 1e-9 and on_segment(a, d, b):
        return True
    if abs(o3) <= 1e-9 and on_segment(c, a, d):
        return True
    if abs(o4) <= 1e-9 and on_segment(c, b, d):
        return True
    return False


def _outside_canvas_ratio(points: Sequence[Point], width: float, height: float) -> float:
    if not points or width <= 0 or height <= 0:
        return 0.0
    margin_x = width * 0.05
    margin_y = height * 0.05
    outside = 0
    for x, y in points:
        if x < -margin_x or x > width + margin_x or y < -margin_y or y > height + margin_y:
            outside += 1
    return outside / max(1, len(points))


def _spline_curvature_proxy(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for index in range(1, len(points) - 1):
        a, b, c = points[index - 1], points[index], points[index + 1]
        ab = math.atan2(b[1] - a[1], b[0] - a[0])
        bc = math.atan2(c[1] - b[1], c[0] - b[0])
        delta = abs((bc - ab + math.pi) % (2 * math.pi) - math.pi)
        total += delta
    return total
