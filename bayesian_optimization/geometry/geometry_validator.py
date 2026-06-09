from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bayesian_optimization.geometry.primitive_analyzer import analyze_primitives
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
    enable_substrate_edge_clearance: bool = True
    substrate_edge_clearance_ratio: float = 0.02
    min_substrate_edge_clearance_px: float = 10.0
    allow_feedline_port_corridor_clearance: bool = True


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
    port_summary: Optional[Dict[str, Any]] = None,
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

    substrate_clearance = validate_substrate_edge_clearance(payload, cfg, port_summary=port_summary)
    if substrate_clearance:
        report.metrics["substrate_edge_clearance"] = substrate_clearance
        if not bool(substrate_clearance.get("valid", True)):
            for violation in substrate_clearance.get("violations", [])[:8]:
                report.reasons.append(
                    "non-port metal too close to substrate edge: "
                    f"{violation.get('primitive_id')} clearance={violation.get('clearance_px'):.3f}px "
                    f"< required={substrate_clearance.get('required_clearance_px'):.3f}px"
                )
            extra = len(substrate_clearance.get("violations", [])) - 8
            if extra > 0:
                report.reasons.append(f"{extra} additional substrate edge clearance violations")

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


def validate_substrate_edge_clearance(
    payload: Dict[str, Any],
    config: GeometryValidationConfig,
    port_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate that non-port metal stays away from substrate edges.

    Input: parameterization payload, validation config, and optional port summary.
    Output: JSON-safe clearance report.
    Algorithm purpose: prevent BO from moving non-port patch/slot/line geometry
    onto the dielectric board boundary while still allowing the explicit port
    feed corridor to approach the board edge.
    """

    if not config.enable_substrate_edge_clearance:
        return {}
    substrate_bbox = substrate_bbox_from_payload(payload)
    if substrate_bbox is None:
        return {
            "valid": True,
            "enabled": True,
            "skipped": True,
            "reason": "canvas bbox unavailable; substrate edge clearance not evaluated",
        }

    min_x, min_y, max_x, max_y = substrate_bbox
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0.0 or height <= 0.0:
        return {
            "valid": True,
            "enabled": True,
            "skipped": True,
            "reason": "invalid substrate bbox",
            "substrate_bbox": list(substrate_bbox),
        }

    required = max(
        float(config.min_substrate_edge_clearance_px),
        float(config.substrate_edge_clearance_ratio) * min(width, height),
    )
    try:
        analysis = analyze_primitives(payload, port_summary=port_summary)
    except Exception as exc:
        return {
            "valid": True,
            "enabled": True,
            "skipped": True,
            "reason": f"primitive analysis failed: {exc}",
            "substrate_bbox": list(substrate_bbox),
            "required_clearance_px": required,
        }

    port_context = analysis.get("summary", {}).get("port_context", {}) or {}
    min_clearance = None
    violations: List[Dict[str, Any]] = []
    checked_count = 0
    exempt_count = 0
    for primitive in analysis.get("primitives", []) or []:
        points = [tuple(point) for point in primitive.get("points", []) if len(point) >= 2]
        if not points:
            continue
        clearance = min(point_substrate_edge_clearance((float(x), float(y)), substrate_bbox) for x, y in points)
        min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
        if primitive.get("role") == "PORT":
            exempt_count += 1
            continue
        if is_feedline_port_corridor_exempt(primitive, port_context, config):
            exempt_count += 1
            continue
        checked_count += 1
        if clearance < required:
            violations.append(
                {
                    "primitive_id": primitive.get("primitive_id"),
                    "type": primitive.get("type"),
                    "role": primitive.get("role"),
                    "clearance_px": float(clearance),
                    "bbox": primitive.get("bbox"),
                    "points": [[float(x), float(y)] for x, y in points],
                }
            )

    return {
        "valid": not violations,
        "enabled": True,
        "substrate_bbox": list(substrate_bbox),
        "required_clearance_px": float(required),
        "minimum_clearance_px": float(min_clearance) if min_clearance is not None else None,
        "checked_non_port_primitive_count": checked_count,
        "exempt_port_or_feed_corridor_count": exempt_count,
        "violation_count": len(violations),
        "violations": violations,
        "policy": "PORT primitives are exempt; FEEDLINE primitives in the inferred port corridor are exempt.",
    }


def substrate_bbox_from_payload(payload: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Return substrate/canvas bbox in parameterization coordinates."""

    canvas = payload.get("canvas") or {}
    width = _finite_positive(canvas.get("width"))
    height = _finite_positive(canvas.get("height"))
    if width is not None and height is not None:
        return 0.0, 0.0, width, height
    return None


def point_substrate_edge_clearance(point: Point, bbox: Sequence[float]) -> float:
    """Return signed distance from a point to the nearest substrate bbox edge."""

    x, y = point
    min_x, min_y, max_x, max_y = [float(value) for value in bbox[:4]]
    return min(x - min_x, max_x - x, y - min_y, max_y - y)


def is_feedline_port_corridor_exempt(
    primitive: Dict[str, Any],
    port_context: Dict[str, Any],
    config: GeometryValidationConfig,
) -> bool:
    """Allow the explicit feed corridor near the excitation port."""

    if not config.allow_feedline_port_corridor_clearance:
        return False
    if primitive.get("role") != "FEEDLINE":
        return False
    center = _primitive_center(primitive)
    if center is None:
        return False
    for key in ("core_bbox", "neighbor_bbox"):
        bbox = port_context.get(key)
        if isinstance(bbox, list) and len(bbox) >= 4 and _point_in_bbox(center, bbox):
            return True
    return False


def _primitive_center(primitive: Dict[str, Any]) -> Optional[Point]:
    points = [tuple(point) for point in primitive.get("points", []) if len(point) >= 2]
    if not points:
        return None
    return (
        sum(float(point[0]) for point in points) / len(points),
        sum(float(point[1]) for point in points) / len(points),
    )


def _point_in_bbox(point: Point, bbox: Sequence[float]) -> bool:
    if len(bbox) < 4:
        return False
    return float(bbox[0]) <= point[0] <= float(bbox[2]) and float(bbox[1]) <= point[1] <= float(bbox[3])


def _finite_positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number) and number > 0.0:
        return number
    return None


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
