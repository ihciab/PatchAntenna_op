from __future__ import annotations

"""Junction graph and synchronization for primitive-aware mutation."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Point = Tuple[float, float]


def build_junction_graph(
    payload: Dict[str, Any],
    analysis: Dict[str, Any],
    tolerance: float = 1.0,
) -> Dict[str, Any]:
    """Build a graph of shared primitive endpoints.

    Input: geometry payload, primitive analysis dictionary, and endpoint tolerance.
    Output: junction graph dictionary.
    Algorithm purpose: identify line-spline and curve-curve joins that must be
    synchronized after mutation to preserve C0 topology.
    """

    endpoint_refs: List[Dict[str, Any]] = []
    for primitive in analysis.get("primitives", []) or []:
        for role in ("start", "end"):
            point = primitive_endpoint_from_payload(payload, primitive, role)
            if point is None:
                continue
            endpoint_refs.append(
                {
                    "primitive_id": primitive.get("primitive_id"),
                    "endpoint": role,
                    "ref": f"{primitive.get('primitive_id')}.{role}",
                    "point": list(point),
                    "component_index": primitive.get("component_index"),
                    "primitive_index": primitive.get("primitive_index"),
                    "source_key": primitive.get("source_key", "segments"),
                }
            )

    clusters: List[List[Dict[str, Any]]] = []
    for endpoint in endpoint_refs:
        placed = False
        for cluster in clusters:
            center = average_points([tuple(item["point"]) for item in cluster])
            if distance(tuple(endpoint["point"]), center) <= tolerance:
                cluster.append(endpoint)
                placed = True
                break
        if not placed:
            clusters.append([endpoint])

    junctions: Dict[str, Any] = {}
    junction_index = 1
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        shared = average_points([tuple(item["point"]) for item in cluster])
        junction_id = f"junction{junction_index:03d}"
        junctions[junction_id] = {
            "shared_point": list(shared),
            "connected": [item["ref"] for item in cluster],
            "endpoints": cluster,
        }
        junction_index += 1
    return {"tolerance": float(tolerance), "junctions": junctions}


def synchronize_junctions(
    payload: Dict[str, Any],
    graph: Dict[str, Any],
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Synchronize all endpoints in each junction.

    Input: mutable geometry payload, junction graph, and primitive analysis.
    Output: synchronization report.
    Algorithm purpose: eliminate mutation-created cracks by snapping connected
    primitive endpoints to their shared junction point before validation.
    """

    fixed: List[Dict[str, Any]] = []
    primitive_by_id = {primitive.get("primitive_id"): primitive for primitive in analysis.get("primitives", []) or []}
    for junction_id, junction in (graph.get("junctions") or {}).items():
        endpoints = junction.get("endpoints", []) or []
        current_points: List[Point] = []
        port_points: List[Point] = []
        for endpoint in endpoints:
            primitive = primitive_by_id.get(endpoint.get("primitive_id"))
            point = primitive_endpoint_from_payload(payload, primitive, endpoint.get("endpoint")) if primitive else None
            if point is not None:
                current_points.append(point)
                if primitive and primitive.get("role") == "PORT":
                    port_points.append(point)
        if not current_points:
            continue
        target = average_points(port_points or current_points)
        # Junction synchronization core: all connected primitive endpoint fields
        # and dependent sampled-cache endpoints are snapped to one shared point.
        for endpoint in endpoints:
            primitive = primitive_by_id.get(endpoint.get("primitive_id"))
            if primitive is None:
                continue
            before = primitive_endpoint_from_payload(payload, primitive, endpoint.get("endpoint"))
            set_primitive_endpoint(payload, primitive, endpoint.get("endpoint"), target)
            fixed.append(
                {
                    "junction_id": junction_id,
                    "endpoint": endpoint.get("ref"),
                    "before": list(before) if before is not None else None,
                    "after": list(target),
                }
            )
        junction["shared_point"] = list(target)
    return {"junction_fixed": fixed, "fixed_count": len(fixed)}


def validate_junctions(
    payload: Dict[str, Any],
    graph: Dict[str, Any],
    analysis: Dict[str, Any],
    tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate that all junction endpoints remain connected.

    Input: geometry payload, junction graph, analysis, and optional tolerance.
    Output: validation dictionary with valid flag and broken junction list.
    Algorithm purpose: reject mutation when synchronization cannot restore
    topology within tolerance.
    """

    tol = float(tolerance if tolerance is not None else graph.get("tolerance", 1.0))
    primitive_by_id = {primitive.get("primitive_id"): primitive for primitive in analysis.get("primitives", []) or []}
    broken: List[Dict[str, Any]] = []
    valid_junctions: List[str] = []
    for junction_id, junction in (graph.get("junctions") or {}).items():
        points: List[Point] = []
        refs: List[str] = []
        for endpoint in junction.get("endpoints", []) or []:
            primitive = primitive_by_id.get(endpoint.get("primitive_id"))
            point = primitive_endpoint_from_payload(payload, primitive, endpoint.get("endpoint")) if primitive else None
            if point is not None:
                points.append(point)
                refs.append(str(endpoint.get("ref")))
        if len(points) < 2:
            continue
        shared = average_points(points)
        max_gap = max(distance(point, shared) for point in points)
        if max_gap > tol:
            broken.append(
                {
                    "junction_id": junction_id,
                    "shared_point": list(shared),
                    "max_gap": max_gap,
                    "connected": refs,
                }
            )
        else:
            valid_junctions.append(junction_id)
    return {
        "valid": not broken,
        "tolerance": tol,
        "valid_junctions": valid_junctions,
        "broken_junctions": broken,
        "junction_count": len(graph.get("junctions") or {}),
    }


def primitive_endpoint_from_payload(
    payload: Dict[str, Any],
    primitive: Optional[Dict[str, Any]],
    endpoint: str,
) -> Optional[Point]:
    """Read a primitive endpoint from the mutable payload.

    Input: payload, primitive analysis record, and endpoint name start/end.
    Output: endpoint point or None.
    Algorithm purpose: use current mutated geometry rather than stale analysis
    points when validating and synchronizing junctions.
    """

    if primitive is None:
        return None
    primitive_obj = get_primitive_object(payload, primitive)
    if primitive_obj is None:
        return None
    kind = primitive.get("type")
    if kind == "LINE":
        key = "start" if endpoint == "start" else "end"
        return parse_point(primitive_obj.get(key))
    if kind == "BSPLINE":
        controls = parse_points(primitive_obj.get("control_points"))
        if controls:
            return controls[0] if endpoint == "start" else controls[-1]
    sampled = sample_endpoint(payload, primitive, endpoint)
    if sampled is not None:
        return sampled
    points = parse_points(primitive_obj.get("points"))
    if points:
        return points[0] if endpoint == "start" else points[-1]
    return None


def set_primitive_endpoint(
    payload: Dict[str, Any],
    primitive: Dict[str, Any],
    endpoint: str,
    point: Point,
) -> None:
    """Set a primitive endpoint in both parameters and sampled caches.

    Input: mutable payload, primitive analysis record, endpoint name, and target point.
    Output: in-place payload update.
    Algorithm purpose: keep line/spline/curve parameter fields and dependent
    sampled geometry synchronized at junctions.
    """

    primitive_obj = get_primitive_object(payload, primitive)
    if primitive_obj is None:
        return
    point_list = [float(point[0]), float(point[1])]
    kind = primitive.get("type")
    if kind == "LINE":
        primitive_obj["start" if endpoint == "start" else "end"] = point_list
    elif kind == "BSPLINE":
        controls = primitive_obj.get("control_points")
        if isinstance(controls, list) and controls:
            controls[0 if endpoint == "start" else len(controls) - 1] = point_list

    component = get_component(payload, primitive)
    if component is None:
        return
    index = primitive.get("start_idx") if endpoint == "start" else primitive.get("end_idx")
    if isinstance(index, int):
        for key in ("resampled_points", "fallback_points", "sampled_points", "points"):
            values = component.get(key)
            if isinstance(values, list) and 0 <= index < len(values):
                values[index] = point_list


def get_primitive_object(payload: Dict[str, Any], primitive: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the live primitive object referenced by an analysis record.

    Input: payload and primitive analysis record.
    Output: primitive dictionary or None.
    Algorithm purpose: bridge derived analysis metadata back to mutable JSON.
    """

    component = get_component(payload, primitive)
    if component is None:
        return None
    source_key = str(primitive.get("source_key", "segments"))
    primitive_index = primitive.get("primitive_index")
    items = component.get(source_key, []) or []
    if isinstance(primitive_index, int) and 0 <= primitive_index < len(items):
        item = items[primitive_index]
        return item if isinstance(item, dict) else None
    return None


def get_component(payload: Dict[str, Any], primitive: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the live component referenced by an analysis record.

    Input: payload and primitive analysis record.
    Output: component dictionary or None.
    Algorithm purpose: centralize component lookup for mutation helpers.
    """

    component_index = primitive.get("component_index")
    components = payload.get("components", []) or []
    if isinstance(component_index, int) and 0 <= component_index < len(components):
        component = components[component_index]
        return component if isinstance(component, dict) else None
    return None


def sample_endpoint(payload: Dict[str, Any], primitive: Dict[str, Any], endpoint: str) -> Optional[Point]:
    """Read an endpoint from component sampled arrays.

    Input: payload, primitive analysis record, and endpoint name.
    Output: sampled endpoint or None.
    Algorithm purpose: support arcs/curves that store only index spans.
    """

    component = get_component(payload, primitive)
    if component is None:
        return None
    index = primitive.get("start_idx") if endpoint == "start" else primitive.get("end_idx")
    if not isinstance(index, int):
        return None
    for key in ("resampled_points", "fallback_points", "sampled_points", "points"):
        values = component.get(key)
        if isinstance(values, list) and 0 <= index < len(values):
            return parse_point(values[index])
    return None


def parse_point(value: Any) -> Optional[Point]:
    """Parse one 2D point.

    Input: arbitrary JSON value.
    Output: point tuple or None.
    Algorithm purpose: guard junction math against malformed coordinates.
    """

    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def parse_points(value: Any) -> List[Point]:
    """Parse a point list.

    Input: arbitrary JSON value.
    Output: list of valid points.
    Algorithm purpose: normalize control-point arrays for endpoint lookup.
    """

    if not isinstance(value, list):
        return []
    points: List[Point] = []
    for item in value:
        point = parse_point(item)
        if point is not None:
            points.append(point)
    return points


def average_points(points: Sequence[Point]) -> Point:
    """Average a point collection.

    Input: sequence of points.
    Output: centroid point.
    Algorithm purpose: compute shared junction snap location.
    """

    if not points:
        return 0.0, 0.0
    return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)


def distance(a: Point, b: Point) -> float:
    """Calculate Euclidean distance.

    Input: two points.
    Output: scalar distance.
    Algorithm purpose: measure junction gaps.
    """

    return math.hypot(a[0] - b[0], a[1] - b[1])


def write_junction_validation(path: Union[str, Path], report: Dict[str, Any]) -> None:
    """Write a junction validation report.

    Input: output path and report dictionary.
    Output: JSON file.
    Algorithm purpose: make C0 continuity decisions auditable per evaluation.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def plot_junction_debug(
    payload: Dict[str, Any],
    graph: Dict[str, Any],
    validation: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """Plot valid and broken junctions.

    Input: payload, junction graph, validation report, and output path.
    Output: PNG image with green valid and red broken junctions.
    Algorithm purpose: show where topology-preserving constraints were applied.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    broken_ids = {item.get("junction_id") for item in validation.get("broken_junctions", []) or []}
    fig, ax = plt.subplots(figsize=(8, 8))
    for component in payload.get("components", []) or []:
        points = parse_points(component.get("resampled_points") or component.get("fallback_points"))
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], color="#c8c8c8", linewidth=0.8)
    for junction_id, junction in (graph.get("junctions") or {}).items():
        point = tuple(junction.get("shared_point", [0.0, 0.0]))
        color = "#d62728" if junction_id in broken_ids else "#2ca02c"
        ax.scatter([point[0]], [point[1]], color=color, s=25)
        ax.text(point[0], point[1], junction_id, fontsize=7, color=color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Junction Debug")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
