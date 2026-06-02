from __future__ import annotations

"""Shape-quality regularization for primitive-aware BO mutations."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Point = Tuple[float, float]


def validate_shape_quality(
    before: Dict[str, Any],
    after: Dict[str, Any],
    analysis: Dict[str, Any],
    max_curvature_delta: float = 0.30,
    max_energy_ratio: float = 0.18,
) -> Dict[str, Any]:
    """Validate curvature, smoothness energy, and spline continuity.

    Input: pre-mutation payload, post-mutation payload, primitive analysis, and thresholds.
    Output: shape quality report with valid flag and score.
    Algorithm purpose: reject sharp spikes, local dents, and spline continuity
    failures before existing GeometryValidation and CST handoff.
    """

    component_reports: List[Dict[str, Any]] = []
    max_delta = 0.0
    max_energy = 0.0
    errors: List[str] = []
    warnings: List[str] = []
    for component_index, (before_points, after_points) in enumerate(zip(component_point_sets(before), component_point_sets(after))):
        if len(before_points) != len(after_points) or len(before_points) < 3:
            continue
        before_curv = curvature_profile(before_points)
        after_curv = curvature_profile(after_points)
        deltas = [abs(a - b) for a, b in zip(after_curv, before_curv)]
        component_max_delta = max(deltas) if deltas else 0.0
        energy = deformation_energy(before_points, after_points)
        scale = max(polyline_length(before_points), 1e-9)
        energy_ratio = energy / (scale * scale)
        max_delta = max(max_delta, component_max_delta)
        max_energy = max(max_energy, energy_ratio)
        component_reports.append(
            {
                "component_index": component_index,
                "max_curvature_delta": component_max_delta,
                "smoothness_energy": energy,
                "smoothness_energy_ratio": energy_ratio,
            }
        )
        if component_max_delta > max_curvature_delta:
            errors.append(f"component {component_index} curvature delta {component_max_delta:.4f} exceeds {max_curvature_delta:.4f}")
        if energy_ratio > max_energy_ratio:
            errors.append(f"component {component_index} deformation energy ratio {energy_ratio:.4f} exceeds {max_energy_ratio:.4f}")

    continuity = validate_spline_continuity(after, analysis)
    errors.extend(continuity.get("errors", []))
    warnings.extend(continuity.get("warnings", []))
    curvature_score = max(0.0, 1.0 - max_delta / max(max_curvature_delta, 1e-9))
    energy_score = max(0.0, 1.0 - max_energy / max(max_energy_ratio, 1e-9))
    continuity_score = 1.0 if continuity.get("valid", True) else 0.0
    score = 0.45 * curvature_score + 0.35 * energy_score + 0.20 * continuity_score
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "shape_quality_score": float(score),
        "max_curvature_delta": float(max_delta),
        "max_smoothness_energy_ratio": float(max_energy),
        "component_reports": component_reports,
        "spline_continuity": continuity,
    }


def curvature_profile(points: Sequence[Point]) -> List[float]:
    """Compute discrete curvature along a polyline.

    Input: ordered 2D polyline points.
    Output: curvature value per interior point.
    Algorithm purpose: detect sharp spikes introduced by mutation.
    """

    curvatures: List[float] = []
    for index in range(1, len(points) - 1):
        a = points[index - 1]
        b = points[index]
        c = points[index + 1]
        ab = (b[0] - a[0], b[1] - a[1])
        bc = (c[0] - b[0], c[1] - b[1])
        la = math.hypot(ab[0], ab[1])
        lb = math.hypot(bc[0], bc[1])
        if la <= 1e-12 or lb <= 1e-12:
            curvatures.append(0.0)
            continue
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        dot = ab[0] * bc[0] + ab[1] * bc[1]
        angle = abs(math.atan2(cross, dot))
        curvatures.append(angle / max(0.5 * (la + lb), 1e-9))
    return curvatures


def deformation_energy(before: Sequence[Point], after: Sequence[Point]) -> float:
    """Compute smoothness energy from displacement second differences.

    Input: before and after point sequences of equal length.
    Output: nonnegative deformation energy.
    Algorithm purpose: penalize localized violent deformation even when absolute
    point motion is small.
    """

    if len(before) != len(after) or len(before) < 3:
        return 0.0
    displacements = [(a[0] - b[0], a[1] - b[1]) for b, a in zip(before, after)]
    energy = 0.0
    for index in range(1, len(displacements) - 1):
        ddx = displacements[index - 1][0] - 2.0 * displacements[index][0] + displacements[index + 1][0]
        ddy = displacements[index - 1][1] - 2.0 * displacements[index][1] + displacements[index + 1][1]
        energy += ddx * ddx + ddy * ddy
    return energy


def validate_spline_continuity(payload: Dict[str, Any], analysis: Dict[str, Any], max_angle_deg: float = 70.0) -> Dict[str, Any]:
    """Validate C0/C1-style continuity around spline endpoints.

    Input: mutated payload, primitive analysis, and angle threshold in degrees.
    Output: continuity report.
    Algorithm purpose: catch line-B-spline and curve-curve tangent breaks that
    GeometryValidation may not classify as invalid topology.
    """

    warnings: List[str] = []
    errors: List[str] = []
    splines = [primitive for primitive in analysis.get("primitives", []) or [] if primitive.get("type") == "BSPLINE"]
    all_primitives = list(analysis.get("primitives", []) or [])
    for spline in splines:
        for endpoint in ("start", "end"):
            tangent = primitive_tangent(payload, spline, endpoint)
            junction_point = primitive_endpoint(payload, spline, endpoint)
            if tangent is None or junction_point is None:
                continue
            for other in all_primitives:
                if other is spline:
                    continue
                for other_endpoint in ("start", "end"):
                    other_point = primitive_endpoint(payload, other, other_endpoint)
                    if other_point is None or distance(junction_point, other_point) > 1.0:
                        continue
                    other_tangent = primitive_tangent(payload, other, other_endpoint)
                    if other_tangent is None:
                        continue
                    angle = angle_between(tangent, other_tangent)
                    if angle > max_angle_deg:
                        warnings.append(
                            f"{spline.get('primitive_id')}.{endpoint} tangent angle {angle:.1f} deg with {other.get('primitive_id')}.{other_endpoint}"
                        )
    return {"valid": not errors, "errors": errors, "warnings": warnings, "max_angle_deg": max_angle_deg}


def component_point_sets(payload: Dict[str, Any]) -> List[List[Point]]:
    """Extract component point lists.

    Input: geometry payload.
    Output: list of point arrays.
    Algorithm purpose: provide regularizer inputs from unchanged schema fields.
    """

    sets: List[List[Point]] = []
    for component in payload.get("components", []) or []:
        sets.append(parse_points(component.get("resampled_points") or component.get("fallback_points") or component.get("points")))
    return sets


def primitive_endpoint(payload: Dict[str, Any], primitive: Dict[str, Any], endpoint: str) -> Union[Point, None]:
    """Read a primitive endpoint from payload.

    Input: payload, primitive analysis record, and endpoint name.
    Output: endpoint point or None.
    Algorithm purpose: support continuity checks without importing mutation code.
    """

    points = primitive_points_from_payload(payload, primitive)
    if not points:
        return None
    return points[0] if endpoint == "start" else points[-1]


def primitive_tangent(payload: Dict[str, Any], primitive: Dict[str, Any], endpoint: str) -> Union[Point, None]:
    """Estimate endpoint tangent for a primitive.

    Input: payload, primitive analysis record, and endpoint name.
    Output: unit tangent vector or None.
    Algorithm purpose: approximate C1 continuity at primitive junctions.
    """

    points = primitive_points_from_payload(payload, primitive)
    if len(points) < 2:
        return None
    if endpoint == "start":
        vec = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    else:
        vec = (points[-2][0] - points[-1][0], points[-2][1] - points[-1][1])
    length = math.hypot(vec[0], vec[1])
    if length <= 1e-12:
        return None
    return vec[0] / length, vec[1] / length


def primitive_points_from_payload(payload: Dict[str, Any], primitive: Dict[str, Any]) -> List[Point]:
    """Return current points for a primitive.

    Input: payload and primitive analysis record.
    Output: point list from primitive parameters or component sample span.
    Algorithm purpose: measure continuity and curvature on mutated geometry.
    """

    component_index = primitive.get("component_index")
    primitive_index = primitive.get("primitive_index")
    source_key = primitive.get("source_key", "segments")
    components = payload.get("components", []) or []
    if not isinstance(component_index, int) or not (0 <= component_index < len(components)):
        return []
    component = components[component_index]
    items = component.get(source_key, []) or []
    primitive_obj = items[primitive_index] if isinstance(primitive_index, int) and 0 <= primitive_index < len(items) else {}
    if primitive.get("type") == "LINE":
        points = [parse_point(primitive_obj.get("start")), parse_point(primitive_obj.get("end"))]
        return [point for point in points if point is not None]
    if primitive.get("type") == "BSPLINE":
        controls = parse_points(primitive_obj.get("control_points"))
        if controls:
            return controls
    samples = parse_points(component.get("resampled_points") or component.get("fallback_points") or component.get("points"))
    start_idx = primitive.get("start_idx")
    end_idx = primitive.get("end_idx")
    if isinstance(start_idx, int) and isinstance(end_idx, int) and samples:
        lo = max(0, min(start_idx, end_idx))
        hi = min(len(samples) - 1, max(start_idx, end_idx))
        return samples[lo : hi + 1]
    return samples


def parse_point(value: Any) -> Union[Point, None]:
    """Parse one coordinate.

    Input: arbitrary value.
    Output: point or None.
    Algorithm purpose: normalize JSON coordinates for regularization.
    """

    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def parse_points(value: Any) -> List[Point]:
    """Parse a list of coordinates.

    Input: arbitrary value.
    Output: valid 2D points.
    Algorithm purpose: keep geometry checks robust to optional fields.
    """

    if not isinstance(value, list):
        return []
    points: List[Point] = []
    for item in value:
        point = parse_point(item)
        if point is not None:
            points.append(point)
    return points


def polyline_length(points: Sequence[Point]) -> float:
    """Calculate polyline length.

    Input: ordered points.
    Output: total segment length.
    Algorithm purpose: normalize deformation energy by geometry scale.
    """

    return sum(distance(points[index - 1], points[index]) for index in range(1, len(points)))


def distance(a: Point, b: Point) -> float:
    """Calculate Euclidean distance.

    Input: two points.
    Output: distance.
    Algorithm purpose: shared regularizer metric.
    """

    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_between(a: Point, b: Point) -> float:
    """Calculate unsigned angle between two unit-like vectors.

    Input: two vectors.
    Output: angle in degrees.
    Algorithm purpose: estimate C1 discontinuity at junctions.
    """

    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    return math.degrees(math.acos(abs(dot)))


def write_shape_quality_report(path: Union[str, Path], report: Dict[str, Any]) -> None:
    """Write shape-quality JSON.

    Input: path and report dictionary.
    Output: JSON report on disk.
    Algorithm purpose: persist regularizer decisions for each evaluation.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def plot_curvature_change_map(before: Dict[str, Any], after: Dict[str, Any], path: Union[str, Path]) -> None:
    """Plot curvature change before and after mutation.

    Input: before payload, after payload, and output path.
    Output: PNG image.
    Algorithm purpose: visualize where curvature changed most strongly.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    for before_points, after_points in zip(component_point_sets(before), component_point_sets(after)):
        if len(before_points) != len(after_points) or len(after_points) < 3:
            continue
        before_curv = curvature_profile(before_points)
        after_curv = curvature_profile(after_points)
        deltas = [abs(a - b) for a, b in zip(after_curv, before_curv)]
        xs = [point[0] for point in after_points[1:-1]]
        ys = [point[1] for point in after_points[1:-1]]
        scatter = ax.scatter(xs, ys, c=deltas, s=8, cmap="magma")
        ax.plot([p[0] for p in after_points], [p[1] for p in after_points], color="#b0b0b0", linewidth=0.6)
        fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Curvature Change Map")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
