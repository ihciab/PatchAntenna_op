from __future__ import annotations

"""Primitive analysis for Bayesian geometry mutation.

This module reads an existing curve_parameterization payload and builds a
derived primitive inventory. The derived files are BO diagnostics only; the
input parameterization schema is not modified.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesian_optimization.geometry.port_summary_utils import find_port_summary, resolve_port_side


Point = Tuple[float, float]
CURVE_PRIMITIVE_TYPES = {"ARC", "CURVE", "BSPLINE"}
CURVE_PARAMETERIZATION_NATIVE = "native"
CURVE_PARAMETERIZATION_LINEARIZED = "linearized"
DEFAULT_CURVE_PARAMETERIZATION_MODE = CURVE_PARAMETERIZATION_NATIVE


@dataclass(frozen=True)
class PrimitiveRecord:
    """Analyzed primitive metadata used by primitive-aware BO mutation."""

    primitive_id: str
    type: str
    role: str
    points: List[Point]
    optimization_mode: str
    component_index: int
    primitive_index: int
    source_key: str
    original_type: Optional[str] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    point_roles: List[str] = field(default_factory=list)
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary.

        Input: self.
        Output: JSON-serializable primitive metadata.
        Algorithm purpose: preserve a stable diagnostic representation for BO.
        """

        data = asdict(self)
        data["points"] = [[float(x), float(y)] for x, y in self.points]
        return data


def load_parameterization(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a curve_parameterization JSON file.

    Input: filesystem path to an existing parameterization file.
    Output: decoded JSON dictionary.
    Algorithm purpose: provide a small IO helper for analyzer entry points.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_primitives(
    payload_or_path: Union[Dict[str, Any], str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    port_summary: Optional[Dict[str, Any]] = None,
    curve_parameterization_mode: str = DEFAULT_CURVE_PARAMETERIZATION_MODE,
) -> Dict[str, Any]:
    """Analyze primitive type, role, points, and optimization mode.

    Input: parameterization payload or path, plus optional output directory.
    Output: primitive analysis dictionary with summary and primitive records.
    Algorithm purpose: convert raw line/arc/spline/feed/port geometry into a
    BO-friendly inventory without changing the original schema.
    """

    payload = load_parameterization(payload_or_path) if isinstance(payload_or_path, (str, Path)) else payload_or_path
    curve_mode = normalize_curve_parameterization_mode(curve_parameterization_mode)
    all_points = collect_payload_points(payload)
    payload_bbox = point_bbox(all_points) if all_points else (0.0, 0.0, 1.0, 1.0)
    port_context = infer_port_context(payload, payload_bbox, port_summary)

    records: List[PrimitiveRecord] = []
    for component_index, component in enumerate(payload.get("components", []) or []):
        for source_key, primitive_index, primitive in iter_component_primitives(component):
            original_type = classify_primitive_type(primitive)
            primitive_type = effective_primitive_type(original_type, curve_mode)
            points = extract_primitive_points(component, primitive, original_type)
            if primitive_type == "LINE" and original_type in CURVE_PRIMITIVE_TYPES:
                points = line_endpoints(points)
            if not points:
                continue
            primitive_id = make_primitive_id(component, component_index, primitive, primitive_index)
            role = infer_primitive_role(primitive_type, points, port_context, payload_bbox)
            mode = choose_optimization_mode(primitive_type, role)
            point_roles = classify_point_roles(primitive_type, points)
            records.append(
                PrimitiveRecord(
                    primitive_id=primitive_id,
                    type=primitive_type,
                    role=role,
                    points=points,
                    optimization_mode=mode,
                    component_index=component_index,
                    primitive_index=primitive_index,
                    source_key=source_key,
                    original_type=original_type if original_type != primitive_type else None,
                    start_idx=safe_int(primitive.get("start_idx")),
                    end_idx=safe_int(primitive.get("end_idx")),
                    point_roles=point_roles,
                    bbox=point_bbox(points),
                )
            )

    summary = {
        "primitive_count": len(records),
        "line_count": sum(1 for record in records if record.type == "LINE"),
        "bspline_count": sum(1 for record in records if record.type == "BSPLINE"),
        "curve_count": sum(1 for record in records if record.type in {"ARC", "CURVE"}),
        "linearized_curve_count": sum(1 for record in records if record.original_type in CURVE_PRIMITIVE_TYPES),
        "port_count": sum(1 for record in records if record.role == "PORT"),
        "feedline_count": sum(1 for record in records if record.role == "FEEDLINE"),
        "bbox": list(payload_bbox),
        "port_context": port_context,
        "curve_parameterization_mode": curve_mode,
    }
    analysis = {
        "summary": summary,
        "primitives": [record.to_dict() for record in records],
    }
    if output_dir is not None:
        write_json(Path(output_dir) / "primitive_analysis.json", analysis)
        debug_dir = Path(output_dir) / "primitive_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        plot_primitive_classification(payload, analysis, debug_dir / "primitive_classification.png")
    return analysis


def iter_component_primitives(component: Dict[str, Any]) -> Iterable[Tuple[str, int, Dict[str, Any]]]:
    """Iterate primitive-like entries in a component.

    Input: one component dictionary from curve_parameterization.json.
    Output: tuples of source key, index, and primitive dictionary.
    Algorithm purpose: support both `segments` and `primitives` while keeping
    the original schema untouched.
    """

    source_keys = ("segments",) if component.get("segments") else ("primitives",)
    for source_key in source_keys:
        for index, primitive in enumerate(component.get(source_key, []) or []):
            if isinstance(primitive, dict):
                yield source_key, index, primitive


def classify_primitive_type(primitive: Dict[str, Any]) -> str:
    """Normalize a raw primitive kind into LINE, BSPLINE, ARC, CURVE, or UNKNOWN.

    Input: primitive dictionary.
    Output: normalized uppercase primitive type.
    Algorithm purpose: route each primitive to an appropriate mutation strategy.
    """

    raw = str(primitive.get("type") or primitive.get("kind") or primitive.get("primitive_type") or "").lower()
    if "spline" in raw or "bspline" in raw:
        return "BSPLINE"
    if "line" in raw:
        return "LINE"
    if "arc" in raw:
        return "ARC"
    if "curve" in raw:
        return "CURVE"
    if "port" in raw:
        return "PORT"
    if "feed" in raw:
        return "FEEDLINE"
    return "UNKNOWN"


def normalize_curve_parameterization_mode(mode: str) -> str:
    """Normalize curve handling mode.

    Input: raw mode name.
    Output: `native` or `linearized`.
    Algorithm purpose: keep the old primitive-aware curve/spline behavior as
    the default while allowing BO to treat every curve primitive as a line.
    """

    value = str(mode or CURVE_PARAMETERIZATION_NATIVE).lower().strip()
    aliases = {
        "native": CURVE_PARAMETERIZATION_NATIVE,
        "current": CURVE_PARAMETERIZATION_NATIVE,
        "original": CURVE_PARAMETERIZATION_NATIVE,
        "preserve": CURVE_PARAMETERIZATION_NATIVE,
        "linearized": CURVE_PARAMETERIZATION_LINEARIZED,
        "linear": CURVE_PARAMETERIZATION_LINEARIZED,
        "line": CURVE_PARAMETERIZATION_LINEARIZED,
        "lines": CURVE_PARAMETERIZATION_LINEARIZED,
        "curves_as_lines": CURVE_PARAMETERIZATION_LINEARIZED,
        "all_lines": CURVE_PARAMETERIZATION_LINEARIZED,
    }
    if value not in aliases:
        raise ValueError(
            "curve_parameterization_mode must be one of: "
            "'native' or 'linearized'."
        )
    return aliases[value]


def effective_primitive_type(original_type: str, curve_mode: str) -> str:
    """Return the type used for BO analysis under the selected curve mode.

    Input: schema-native primitive type and normalized curve mode.
    Output: effective primitive type for role and variable generation.
    Algorithm purpose: optionally weaken curve parameterization by routing
    arcs, curves, and splines through the line mutation path.
    """

    if curve_mode == CURVE_PARAMETERIZATION_LINEARIZED and original_type in CURVE_PRIMITIVE_TYPES:
        return "LINE"
    return original_type


def extract_primitive_points(
    component: Dict[str, Any],
    primitive: Dict[str, Any],
    primitive_type: str,
) -> List[Point]:
    """Extract representative points for one primitive.

    Input: component dictionary, primitive dictionary, and normalized type.
    Output: representative 2D points.
    Algorithm purpose: use true primitive parameters where available and fall
    back to indexed component samples for diagnostics and constraints.
    """

    points: List[Point] = []
    if primitive_type == "LINE":
        for key in ("start", "end"):
            point = parse_point(primitive.get(key))
            if point is not None:
                points.append(point)
    elif primitive_type == "BSPLINE":
        points.extend(parse_points(primitive.get("control_points")))
    elif primitive_type in {"ARC", "CURVE"}:
        for key in ("start", "center", "end"):
            point = parse_point(primitive.get(key))
            if point is not None:
                points.append(point)

    if not points:
        points.extend(sample_component_range(component, primitive))
    if len(points) == 1:
        sampled = sample_component_range(component, primitive)
        if len(sampled) >= 2:
            points = [sampled[0], sampled[-1]]
    return points


def line_endpoints(points: Sequence[Point]) -> List[Point]:
    """Reduce representative curve points to a straight line span.

    Input: representative primitive points.
    Output: first and last point only.
    Algorithm purpose: let linearized curve mode keep the same source schema
    while exposing every curve/spline primitive as a line-like BO primitive.
    """

    if len(points) <= 2:
        return list(points)
    return [points[0], points[-1]]


def sample_component_range(component: Dict[str, Any], primitive: Dict[str, Any]) -> List[Point]:
    """Read the sampled points covered by a primitive index range.

    Input: component dictionary and primitive dictionary containing start/end indexes.
    Output: sampled point list for that primitive span.
    Algorithm purpose: provide endpoint and curvature context for primitives
    whose compact parameters omit explicit start/end fields.
    """

    samples = parse_points(
        component.get("resampled_points")
        or component.get("fallback_points")
        or component.get("sampled_points")
        or component.get("points")
    )
    if not samples:
        return []
    start_idx = safe_int(primitive.get("start_idx"))
    end_idx = safe_int(primitive.get("end_idx"))
    if start_idx is None or end_idx is None:
        return samples
    lo = max(0, min(start_idx, end_idx))
    hi = min(len(samples) - 1, max(start_idx, end_idx))
    if hi < lo:
        return []
    return samples[lo : hi + 1]


def infer_port_context(
    payload: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    port_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Infer port/feed direction and protected regions from geometry.

    Input: full payload and global geometry bbox.
    Output: dictionary describing port side, core bbox, neighbor bbox, and axes.
    Algorithm purpose: protect the port contact area when explicit port
    annotations are absent from the parameterization schema.
    """

    width = max(1e-9, bbox[2] - bbox[0])
    height = max(1e-9, bbox[3] - bbox[1])
    summary = find_port_summary(payload, port_summary)
    side = resolve_port_side(payload, summary)
    if side is None:
        axis = "x" if width >= height else "y"
        margin = 0.08 * (width if axis == "x" else height)
        if axis == "x":
            side = "left"
            core_bbox = (bbox[0] - 1e-6, bbox[1], bbox[0] + margin, bbox[3])
            neighbor_bbox = (bbox[0] + margin, bbox[1], bbox[0] + 3.0 * margin, bbox[3])
            propagation = (1.0, 0.0)
            transverse = (0.0, 1.0)
        else:
            side = "bottom"
            core_bbox = (bbox[0], bbox[1] - 1e-6, bbox[2], bbox[1] + margin)
            neighbor_bbox = (bbox[0], bbox[1] + margin, bbox[2], bbox[1] + 3.0 * margin)
            propagation = (0.0, 1.0)
            transverse = (1.0, 0.0)
    elif side in {"left", "right"}:
        axis = "x"
        margin = 0.08 * width
        if side == "left":
            core_bbox = (bbox[0] - 1e-6, bbox[1], bbox[0] + margin, bbox[3])
            neighbor_bbox = (bbox[0] + margin, bbox[1], bbox[0] + 3.0 * margin, bbox[3])
            propagation = (1.0, 0.0)
        else:
            core_bbox = (bbox[2] - margin, bbox[1], bbox[2] + 1e-6, bbox[3])
            neighbor_bbox = (bbox[2] - 3.0 * margin, bbox[1], bbox[2] - margin, bbox[3])
            propagation = (-1.0, 0.0)
        transverse = (0.0, 1.0)
    elif side == "top":
        axis = "y"
        margin = 0.08 * height
        core_bbox = (bbox[0], bbox[1] - 1e-6, bbox[2], bbox[1] + margin)
        neighbor_bbox = (bbox[0], bbox[1] + margin, bbox[2], bbox[1] + 3.0 * margin)
        propagation = (0.0, 1.0)
        transverse = (1.0, 0.0)
    else:
        axis = "y"
        margin = 0.08 * height
        core_bbox = (bbox[0], bbox[3] - margin, bbox[2], bbox[3] + 1e-6)
        neighbor_bbox = (bbox[0], bbox[3] - 3.0 * margin, bbox[2], bbox[3] - margin)
        propagation = (0.0, -1.0)
        transverse = (1.0, 0.0)

    explicit = summary or payload.get("port") or payload.get("ports")
    return {
        "axis": axis,
        "side": side,
        "core_bbox": list(core_bbox),
        "neighbor_bbox": list(neighbor_bbox),
        "propagation_direction": list(propagation),
        "transverse_direction": list(transverse),
        "explicit_port_metadata_present": bool(explicit),
    }


def infer_primitive_role(
    primitive_type: str,
    points: Sequence[Point],
    port_context: Dict[str, Any],
    payload_bbox: Tuple[float, float, float, float],
) -> str:
    """Assign PORT, FEEDLINE, RESONANT_SLOT, or STRUCTURAL role.

    Input: primitive type, primitive points, inferred port context, and bbox.
    Output: semantic role string.
    Algorithm purpose: drive port freezing, feedline direction constraints, and
    physical BO variable naming.
    """

    center = centroid(points)
    core_bbox = tuple(float(v) for v in port_context.get("core_bbox", []))
    neighbor_bbox = tuple(float(v) for v in port_context.get("neighbor_bbox", []))
    if len(core_bbox) == 4 and point_in_bbox(center, core_bbox):
        return "PORT"
    if len(neighbor_bbox) == 4 and point_in_bbox(center, neighbor_bbox):
        return "FEEDLINE"
    if primitive_type == "BSPLINE":
        return "RESONANT_SLOT"
    if primitive_type == "LINE" and is_long_axis_aligned(points, payload_bbox):
        return "FEEDLINE"
    return "STRUCTURAL"


def choose_optimization_mode(primitive_type: str, role: str) -> str:
    """Choose the mutation mode for a primitive.

    Input: normalized primitive type and semantic role.
    Output: optimization mode name.
    Algorithm purpose: prevent single endpoint/control-point drift by mapping
    primitives to constrained deformation families.
    """

    if role == "PORT":
        return "FROZEN"
    if role == "FEEDLINE":
        return "FEED_DIRECTION_ONLY"
    if primitive_type == "LINE":
        return "RIGID_LINE_OFFSET"
    if primitive_type == "BSPLINE":
        return "SMOOTH_DEFORM"
    if primitive_type in {"ARC", "CURVE"}:
        return "CURVE_SMOOTH_OFFSET"
    return "DEPENDENT_CACHE_ONLY"


def classify_point_roles(primitive_type: str, points: Sequence[Point]) -> List[str]:
    """Classify points as ENDPOINT, HANDLE_POINT, or INTERNAL_POINT.

    Input: primitive type and representative points.
    Output: role label per point.
    Algorithm purpose: freeze spline endpoints, limit handles, and allow only
    smooth internal deformation.
    """

    if not points:
        return []
    if primitive_type != "BSPLINE":
        return ["ENDPOINT" if index in (0, len(points) - 1) else "INTERNAL_POINT" for index in range(len(points))]
    roles: List[str] = []
    for index in range(len(points)):
        if index == 0 or index == len(points) - 1:
            roles.append("ENDPOINT")
        elif index == 1 or index == len(points) - 2:
            roles.append("HANDLE_POINT")
        else:
            roles.append("INTERNAL_POINT")
    return roles


def make_primitive_id(
    component: Dict[str, Any],
    component_index: int,
    primitive: Dict[str, Any],
    primitive_index: int,
) -> str:
    """Create a stable primitive id for reports and variable names.

    Input: component, component index, primitive, and primitive index.
    Output: string id such as c000_s003.
    Algorithm purpose: keep BO variables traceable across reports.
    """

    component_id = component.get("component_id", component_index)
    segment_id = primitive.get("segment_id", primitive_index)
    return f"c{component_id}_s{segment_id}"


def collect_payload_points(payload: Dict[str, Any]) -> List[Point]:
    """Collect all available geometry points from a payload.

    Input: parameterization payload.
    Output: list of 2D points.
    Algorithm purpose: estimate global bbox and port regions.
    """

    points: List[Point] = []
    for component in payload.get("components", []) or []:
        for key in ("resampled_points", "fallback_points", "sampled_points", "points"):
            points.extend(parse_points(component.get(key)))
        for _, _, primitive in iter_component_primitives(component):
            points.extend(parse_points(primitive.get("control_points")))
            for key in ("start", "end", "center"):
                point = parse_point(primitive.get(key))
                if point is not None:
                    points.append(point)
    return points


def parse_points(value: Any) -> List[Point]:
    """Parse a list of [x, y] values.

    Input: arbitrary JSON value.
    Output: valid finite 2D points.
    Algorithm purpose: safely normalize coordinate arrays used by analysis.
    """

    if not isinstance(value, list):
        return []
    points: List[Point] = []
    for item in value:
        point = parse_point(item)
        if point is not None:
            points.append(point)
    return points


def parse_point(value: Any) -> Optional[Point]:
    """Parse one [x, y] value.

    Input: arbitrary JSON value.
    Output: finite point tuple or None.
    Algorithm purpose: filter malformed coordinates before geometry math.
    """

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


def point_bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Calculate a bounding box.

    Input: sequence of 2D points.
    Output: (min_x, min_y, max_x, max_y).
    Algorithm purpose: support role inference and debug visualization.
    """

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def centroid(points: Sequence[Point]) -> Point:
    """Calculate point centroid.

    Input: sequence of 2D points.
    Output: average x/y coordinate.
    Algorithm purpose: compact role classification signal.
    """

    if not points:
        return 0.0, 0.0
    return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)


def point_in_bbox(point: Point, bbox: Tuple[float, float, float, float]) -> bool:
    """Check whether a point lies inside a bbox.

    Input: point and bbox.
    Output: boolean.
    Algorithm purpose: classify port core and neighbor regions.
    """

    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def is_long_axis_aligned(points: Sequence[Point], payload_bbox: Tuple[float, float, float, float]) -> bool:
    """Detect feed-like long straight primitives.

    Input: primitive points and global bbox.
    Output: boolean.
    Algorithm purpose: mark likely feedlines when no explicit feed metadata is present.
    """

    if len(points) < 2:
        return False
    length = distance(points[0], points[-1])
    diag = max(1e-9, distance((payload_bbox[0], payload_bbox[1]), (payload_bbox[2], payload_bbox[3])))
    dx = abs(points[-1][0] - points[0][0])
    dy = abs(points[-1][1] - points[0][1])
    return length > 0.20 * diag and (dx < 0.15 * length or dy < 0.15 * length)


def distance(a: Point, b: Point) -> float:
    """Calculate Euclidean distance between two points.

    Input: two 2D points.
    Output: distance.
    Algorithm purpose: shared primitive geometry measurement.
    """

    return math.hypot(a[0] - b[0], a[1] - b[1])


def safe_int(value: Any) -> Optional[int]:
    """Convert a value to int when possible.

    Input: arbitrary value.
    Output: integer or None.
    Algorithm purpose: read optional segment index fields safely.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_json(path: Union[str, Path], payload: Any) -> None:
    """Write a JSON diagnostic file.

    Input: output path and JSON-serializable payload.
    Output: file on disk.
    Algorithm purpose: persist primitive analysis reports for each evaluation.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def plot_primitive_classification(payload: Dict[str, Any], analysis: Dict[str, Any], path: Union[str, Path]) -> None:
    """Plot primitive classes and roles.

    Input: original payload, primitive analysis dictionary, and output path.
    Output: PNG diagnostic image.
    Algorithm purpose: make LINE/BSPLINE/FEED/PORT classification auditable.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {
        "LINE": "#2f6fed",
        "BSPLINE": "#d95f02",
        "ARC": "#7570b3",
        "CURVE": "#1b9e77",
        "PORT": "#d62728",
        "FEEDLINE": "#17becf",
        "UNKNOWN": "#7f7f7f",
    }
    for component in payload.get("components", []) or []:
        samples = parse_points(component.get("resampled_points") or component.get("fallback_points"))
        if samples:
            ax.plot([p[0] for p in samples], [p[1] for p in samples], color="#d0d0d0", linewidth=0.8)
    for primitive in analysis.get("primitives", []) or []:
        points = [tuple(point) for point in primitive.get("points", []) if len(point) >= 2]
        if not points:
            continue
        color_key = primitive.get("role") if primitive.get("role") in {"PORT", "FEEDLINE"} else primitive.get("type")
        color = colors.get(str(color_key), "#7f7f7f")
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=2.5, linewidth=1.5, color=color)
        cx, cy = centroid(points)
        ax.text(cx, cy, str(primitive.get("primitive_id", "")), fontsize=6, color=color)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Primitive Classification")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
