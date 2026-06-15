from __future__ import annotations

"""Helpers for consuming image-based port summary metadata."""

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
VALID_PORT_SIDES = {"left", "right", "top", "bottom"}
DEFAULT_INWARD_PORT_OFFSET_PX = 1.0
DEFAULT_PORT_CONNECTION_STEP_PX = 0.2
DEFAULT_PORT_CONNECTION_TOLERANCE_PX = 0.15
DEFAULT_PORT_CONNECTION_MAX_SHIFT_PX = 120.0
DEFAULT_FINAL_FREE_NORMAL_INWARD_PX = 2.0


def find_port_summary(payload: Dict[str, Any], port_summary: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if isinstance(port_summary, dict):
        return port_summary
    for key in ("port_summary", "patch_port_summary", "patch_port_detection"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if looks_like_port_summary(payload):
        return payload
    return None


def looks_like_port_summary(value: Dict[str, Any]) -> bool:
    schema = str(value.get("schema_version", "")).lower()
    if "port_summary" in schema:
        return True
    return any(
        key in value
        for key in (
            "selected_port",
            "patch_port_detection",
            "closest_border_sides",
            "closest_edge",
        )
    )


def selected_port_candidate(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return None
    selected = summary.get("selected_port")
    if isinstance(selected, dict):
        return selected
    ports = summary.get("ports")
    if isinstance(ports, list):
        for item in ports:
            if isinstance(item, dict):
                return item
    detection = summary.get("patch_port_detection")
    if isinstance(detection, dict):
        selected = detection.get("selected_port")
        if isinstance(selected, dict):
            return selected
        ports = detection.get("ports")
        if isinstance(ports, list):
            for item in ports:
                if isinstance(item, dict):
                    return item
    return None


def resolve_port_side(payload: Dict[str, Any], port_summary: Optional[Dict[str, Any]] = None) -> Optional[str]:
    summary = find_port_summary(payload, port_summary)
    candidate = selected_port_candidate(summary)
    for source in (candidate, summary):
        if not isinstance(source, dict):
            continue
        for key in ("direction", "port_side", "side"):
            side = normalize_port_side(source.get(key))
            if side is not None:
                return side
    if isinstance(summary, dict):
        border_sides = summary.get("closest_border_sides")
        if isinstance(border_sides, list):
            for item in border_sides:
                side = normalize_port_side(item)
                if side is not None:
                    return side
    return None


def resolve_port_point(
    payload: Dict[str, Any],
    port_summary: Optional[Dict[str, Any]] = None,
    *,
    inward_offset_px: float = DEFAULT_INWARD_PORT_OFFSET_PX,
) -> Optional[Point]:
    summary = find_port_summary(payload, port_summary)
    candidate = selected_port_candidate(summary)
    side = resolve_port_side(payload, summary)
    for source in (candidate, summary):
        if not isinstance(source, dict):
            continue
        for key in ("point", "cst_contact_point", "raw_endpoint", "center"):
            point = parse_point(source.get(key))
            if point is not None:
                return shift_point_inward(point, side, inward_offset_px)
    if isinstance(summary, dict):
        edge = summary.get("closest_edge")
        if isinstance(edge, list) and len(edge) >= 2:
            points = [parse_point(item) for item in edge[:2]]
            points = [point for point in points if point is not None]
            if len(points) == 2:
                midpoint = (
                    0.5 * (points[0][0] + points[1][0]),
                    0.5 * (points[0][1] + points[1][1]),
                )
                return shift_point_inward(midpoint, side, inward_offset_px)
    return None


def shift_point_inward(
    point: Point,
    side: Optional[str],
    offset_px: float = DEFAULT_INWARD_PORT_OFFSET_PX,
) -> Point:
    """Move a detected boundary port point inward along the port normal.

    The image coordinate system uses y growing downward.  For a bottom port,
    inward therefore means y decreases; for a right port, inward means x
    decreases.  This small final offset keeps the BO/CST handoff from placing
    the feed exactly on a raster boundary where reconstruction can leave a gap.
    """

    normalized = normalize_port_side(side)
    try:
        offset = float(offset_px)
    except (TypeError, ValueError):
        offset = DEFAULT_INWARD_PORT_OFFSET_PX
    if not math.isfinite(offset) or offset <= 0.0 or normalized is None:
        return point

    x, y = point
    if normalized == "left":
        return x + offset, y
    if normalized == "right":
        return x - offset, y
    if normalized == "top":
        return x, y + offset
    if normalized == "bottom":
        return x, y - offset
    return point


def resolve_port_width(payload: Dict[str, Any], port_summary: Optional[Dict[str, Any]] = None) -> Optional[float]:
    summary = find_port_summary(payload, port_summary)
    candidate = selected_port_candidate(summary)
    for source in (candidate, summary):
        if not isinstance(source, dict):
            continue
        for key in ("local_width", "feed_width", "width", "port_width"):
            width = finite_positive_float(source.get(key))
            if width is not None:
                return width
    if isinstance(summary, dict):
        edge = summary.get("closest_edge")
        if isinstance(edge, list) and len(edge) >= 2:
            p1 = parse_point(edge[0])
            p2 = parse_point(edge[1])
            if p1 is not None and p2 is not None:
                return max(1.0, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return None


def ensure_port_summary_connected_to_geometry(
    payload: Dict[str, Any],
    port_summary: Optional[Dict[str, Any]],
    *,
    step_px: float = DEFAULT_PORT_CONNECTION_STEP_PX,
    tolerance_px: float = DEFAULT_PORT_CONNECTION_TOLERANCE_PX,
    max_shift_px: float = DEFAULT_PORT_CONNECTION_MAX_SHIFT_PX,
    final_free_normal_inward_px: float = DEFAULT_FINAL_FREE_NORMAL_INWARD_PX,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return a per-evaluation port summary whose port reaches the curve.

    If the selected port is not touching the current parameterized geometry,
    walk it inward along the port normal in ``step_px`` increments. This helper
    writes the curve contact point into the per-evaluation summary; the CST
    builder then applies the final free-normal inward push when it creates the
    actual free port plane.
    """

    summary = find_port_summary(payload, port_summary)
    report: Dict[str, Any] = {
        "enabled": True,
        "step_px": float(step_px),
        "tolerance_px": float(tolerance_px),
        "max_shift_px": float(max_shift_px),
        "final_free_normal_inward_px": float(final_free_normal_inward_px),
        "connected_before": False,
        "connected_after": False,
        "final_free_normal_shift_applied": False,
        "shift_applied_px": 0.0,
        "iterations": 0,
        "status": "not_started",
    }
    if not isinstance(summary, dict):
        report["status"] = "skipped_no_port_summary"
        return port_summary, report

    side = resolve_port_side(payload, summary)
    normal = inward_normal_for_side(side)
    if normal is None:
        report["status"] = "failed_unknown_port_side"
        report["side"] = side
        return copy.deepcopy(summary), report
    report["side"] = side
    report["inward_normal"] = [float(normal[0]), float(normal[1])]

    source_point = resolve_raw_port_point(payload, summary)
    if source_point is None:
        report["status"] = "failed_no_port_point"
        return copy.deepcopy(summary), report
    report["original_point"] = [float(source_point[0]), float(source_point[1])]

    polylines = geometry_polylines(payload)
    if not polylines:
        report["status"] = "failed_no_geometry_polylines"
        return copy.deepcopy(summary), report

    step = finite_positive_or_default(step_px, DEFAULT_PORT_CONNECTION_STEP_PX)
    tolerance = finite_positive_or_default(tolerance_px, DEFAULT_PORT_CONNECTION_TOLERANCE_PX)
    max_shift = finite_positive_or_default(max_shift_px, DEFAULT_PORT_CONNECTION_MAX_SHIFT_PX)
    final_push = nonnegative_finite_or_default(final_free_normal_inward_px, DEFAULT_FINAL_FREE_NORMAL_INWARD_PX)
    max_iterations = max(0, int(math.ceil(max_shift / step)))

    distance = distance_to_polylines(source_point, polylines)
    report["minimum_distance_before_px"] = float(distance)
    if distance <= tolerance:
        connected_summary = copy.deepcopy(summary)
        report["connected_before"] = True
        report["connected_after"] = True
        report["geometry_contact_point"] = [float(source_point[0]), float(source_point[1])]
        report["connected_point"] = [float(source_point[0]), float(source_point[1])]
        report["minimum_distance_after_px"] = float(distance)
        report["status"] = "already_connected"
        return connected_summary, report

    candidate = source_point
    for iteration in range(1, max_iterations + 1):
        shift = step * iteration
        candidate = (
            source_point[0] + normal[0] * shift,
            source_point[1] + normal[1] * shift,
        )
        distance = distance_to_polylines(candidate, polylines)
        if distance <= tolerance:
            connected_summary = copy.deepcopy(summary)
            write_port_point_to_summary(connected_summary, candidate, final_free_normal_inward_px=final_push)
            report["connected_after"] = True
            report["geometry_contact_point"] = [float(candidate[0]), float(candidate[1])]
            report["connected_point"] = [float(candidate[0]), float(candidate[1])]
            report["minimum_distance_after_px"] = float(distance)
            report["shift_applied_px"] = float(shift)
            report["final_free_normal_shift_applied"] = False
            report["final_free_normal_deferred_to_cst_builder"] = final_push > 0.0
            report["iterations"] = int(iteration)
            report["status"] = "connected_by_inward_shift"
            return connected_summary, report

    failed_summary = copy.deepcopy(summary)
    write_port_point_to_summary(failed_summary, candidate, final_free_normal_inward_px=final_push)
    report["connected_point"] = [float(candidate[0]), float(candidate[1])]
    report["minimum_distance_after_px"] = float(distance_to_polylines(candidate, polylines))
    report["shift_applied_px"] = float(step * max_iterations)
    report["final_free_normal_shift_applied"] = False
    report["final_free_normal_deferred_to_cst_builder"] = final_push > 0.0
    report["iterations"] = int(max_iterations)
    report["status"] = "failed_no_curve_contact_within_max_shift"
    return failed_summary, report


def resolve_raw_port_point(payload: Dict[str, Any], port_summary: Optional[Dict[str, Any]] = None) -> Optional[Point]:
    """Return the selected port point without applying any inward offset."""

    summary = find_port_summary(payload, port_summary)
    candidate = selected_port_candidate(summary)
    for source in (candidate, summary):
        if not isinstance(source, dict):
            continue
        for key in ("point", "cst_contact_point", "raw_endpoint", "center"):
            point = parse_point(source.get(key))
            if point is not None:
                return point
    if isinstance(summary, dict):
        edge = summary.get("closest_edge")
        if isinstance(edge, list) and len(edge) >= 2:
            points = [parse_point(item) for item in edge[:2]]
            points = [point for point in points if point is not None]
            if len(points) == 2:
                return (
                    0.5 * (points[0][0] + points[1][0]),
                    0.5 * (points[0][1] + points[1][1]),
                )
    return None


def inward_normal_for_side(side: Optional[str]) -> Optional[Point]:
    normalized = normalize_port_side(side)
    if normalized == "left":
        return 1.0, 0.0
    if normalized == "right":
        return -1.0, 0.0
    if normalized == "top":
        return 0.0, 1.0
    if normalized == "bottom":
        return 0.0, -1.0
    return None


def geometry_polylines(payload: Dict[str, Any]) -> List[List[Point]]:
    polylines: List[List[Point]] = []
    for component in payload.get("components", []) or []:
        points = parse_points(
            component.get("resampled_points")
            or component.get("fallback_points")
            or component.get("sampled_points")
            or component.get("points")
            or []
        )
        if len(points) >= 2:
            if bool(component.get("closed", False)) and points[0] != points[-1]:
                points = list(points) + [points[0]]
            polylines.append(points)
        for hole in component.get("holes", []) or []:
            hole_points = parse_points(
                hole.get("resampled_points")
                or hole.get("fallback_points")
                or hole.get("sampled_points")
                or hole.get("points")
                or []
            )
            if len(hole_points) >= 2:
                if hole_points[0] != hole_points[-1]:
                    hole_points = list(hole_points) + [hole_points[0]]
                polylines.append(hole_points)
    return polylines


def parse_points(value: Any) -> List[Point]:
    if not isinstance(value, list):
        return []
    points: List[Point] = []
    for item in value:
        point = parse_point(item)
        if point is not None:
            points.append(point)
    return points


def distance_to_polylines(point: Point, polylines: Sequence[Sequence[Point]]) -> float:
    distances = []
    for polyline in polylines:
        for index in range(1, len(polyline)):
            distances.append(point_segment_distance(point, polyline[index - 1], polyline[index]))
    return min(distances) if distances else math.inf


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    closest = (ax + t * dx, ay + t * dy)
    return math.hypot(px - closest[0], py - closest[1])


def write_port_point_to_summary(
    summary: Dict[str, Any],
    point: Point,
    *,
    final_free_normal_inward_px: float = DEFAULT_FINAL_FREE_NORMAL_INWARD_PX,
) -> None:
    updated = [float(point[0]), float(point[1])]
    for candidate in candidate_port_dicts(summary):
        for key in ("point", "cst_contact_point", "raw_endpoint", "center"):
            if key in candidate:
                candidate[key] = list(updated)
        if "point" not in candidate:
            candidate["point"] = list(updated)
    summary["bo_port_connection_adjustment"] = {
        "point": list(updated),
        "connected_point": list(updated),
        "strategy": "inward_normal_step_until_curve_contact_then_free_normal_push",
        "step_px": DEFAULT_PORT_CONNECTION_STEP_PX,
        "final_free_normal_inward_px": float(final_free_normal_inward_px),
        "final_free_normal_applied_by": "cst_builder",
    }


def candidate_port_dicts(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    selected = summary.get("selected_port")
    if isinstance(selected, dict):
        candidates.append(selected)
    ports = summary.get("ports")
    if isinstance(ports, list):
        candidates.extend(item for item in ports if isinstance(item, dict))
    detection = summary.get("patch_port_detection")
    if isinstance(detection, dict):
        selected_detection = detection.get("selected_port")
        if isinstance(selected_detection, dict):
            candidates.append(selected_detection)
        detection_ports = detection.get("ports")
        if isinstance(detection_ports, list):
            candidates.extend(item for item in detection_ports if isinstance(item, dict))
    unique: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        marker = id(candidate)
        if marker not in seen:
            unique.append(candidate)
            seen.add(marker)
    return unique


def finite_positive_or_default(value: Any, default: float) -> float:
    result = finite_positive_float(value)
    return float(result) if result is not None else float(default)


def nonnegative_finite_or_default(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result) or result < 0.0:
        return float(default)
    return result


def normalize_port_side(value: Any) -> Optional[str]:
    if value is None:
        return None
    side = str(value).strip().lower()
    aliases = {
        "xmin": "left",
        "x_min": "left",
        "xmax": "right",
        "x_max": "right",
        "ymin": "top",
        "y_min": "top",
        "ymax": "bottom",
        "y_max": "bottom",
    }
    side = aliases.get(side, side)
    return side if side in VALID_PORT_SIDES else None


def parse_point(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def finite_positive_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result
