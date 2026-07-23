"""Summarize Geometry Engine antenna JSON for LLM design feedback."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
C0_M_PER_S = 299_792_458.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_INPUTS_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"
GEOMETRY_SUMMARY_FILENAME = "geometry_summary.json"


@dataclass(frozen=True)
class GeometrySummaryConfig:
    """Configuration for antenna geometry summaries."""

    target_frequency_ghz: float = 2.45
    epsilon_r: Optional[float] = None


@dataclass(frozen=True)
class GeometrySummary:
    """Compact antenna geometry summary."""

    patch: Dict[str, Any]
    feed: Dict[str, Any]
    slots: List[Dict[str, Any]]
    symmetry: str
    electrical_size: Dict[str, Any]
    source: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


class GeometrySummaryBuilder:
    """Build a geometry summary from ``geometry_engine_geometry_v1`` JSON."""

    def __init__(self, config: Optional[GeometrySummaryConfig] = None) -> None:
        """Create a summary builder."""

        self.config = config or GeometrySummaryConfig()

    def from_file(self, geometry_json_path: Path | str) -> GeometrySummary:
        """Load Geometry JSON and summarize it."""

        path = Path(geometry_json_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object: {0}".format(path))
        return self.from_dict(payload, source_path=path)

    def from_dict(self, payload: Dict[str, Any], source_path: Optional[Path] = None) -> GeometrySummary:
        """Summarize an already loaded Geometry JSON object."""

        if payload.get("schema_version") != "geometry_engine_geometry_v1":
            raise ValueError("Expected schema_version='geometry_engine_geometry_v1'.")
        geometry = first_planar_geometry(payload)
        outer_vertices = loop_vertices(geometry.get("outer_boundary"))
        feed_metadata = geometry.get("metadata", {}).get("feed")
        feed = summarize_feed(feed_metadata if isinstance(feed_metadata, dict) else None)
        patch_bbox = infer_patch_bbox(outer_vertices, feed)
        patch = summarize_patch(patch_bbox, outer_vertices)
        feed = add_feed_offset(feed, patch)
        slots = [
            summarize_slot(hole, index=index + 1)
            for index, hole in enumerate(geometry.get("holes", []) or [])
            if isinstance(hole, dict)
        ]
        symmetry = infer_symmetry(outer_vertices, [slot["vertices"] for slot in slots])
        electrical_size = summarize_electrical_size(
            patch=patch,
            target_frequency_ghz=self.config.target_frequency_ghz,
            epsilon_r=self.config.epsilon_r,
        )
        return GeometrySummary(
            patch=patch,
            feed=feed,
            slots=slots,
            symmetry=symmetry,
            electrical_size=electrical_size,
            source={
                "schema_version": payload.get("schema_version"),
                "geometry_id": geometry.get("id"),
                "path": None if source_path is None else str(source_path.resolve()),
            },
        )


def build_geometry_summary(
    geometry_json_path: Path | str,
    target_frequency_ghz: float = 2.45,
    epsilon_r: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience function returning a summary dictionary."""

    return GeometrySummaryBuilder(
        GeometrySummaryConfig(
            target_frequency_ghz=target_frequency_ghz,
            epsilon_r=epsilon_r,
        )
    ).from_file(geometry_json_path).to_dict()


def write_geometry_summary(
    geometry_json_path: Path | str,
    output_path: Optional[Path | str] = None,
    target_frequency_ghz: float = 2.45,
    epsilon_r: Optional[float] = None,
) -> Path:
    """Write ``geometry_summary.json`` to the shared agent-input folder by default."""

    summary = build_geometry_summary(
        geometry_json_path=geometry_json_path,
        target_frequency_ghz=target_frequency_ghz,
        epsilon_r=epsilon_r,
    )
    path = Path(output_path) if output_path is not None else default_geometry_summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def default_geometry_summary_path() -> Path:
    """Return the shared geometry summary path consumed by later agents."""

    return AGENT_INPUTS_DIR / GEOMETRY_SUMMARY_FILENAME


def first_planar_geometry(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the first planar conductor geometry object."""

    geometries = payload.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise ValueError("Geometry JSON must contain non-empty `geometries`.")
    for geometry in geometries:
        if isinstance(geometry, dict) and geometry.get("type") == "planar_conductor":
            return geometry
    raise ValueError("No planar_conductor geometry found.")


def summarize_patch(patch_bbox: Tuple[float, float, float, float], outer_vertices: Sequence[Point]) -> Dict[str, Any]:
    """Summarize the radiating patch body."""

    min_x, min_y, max_x, max_y = patch_bbox
    width = max_x - min_x
    length = max_y - min_y
    return {
        "type": "rectangle" if is_axis_aligned_rectangle_bbox(patch_bbox, outer_vertices) else "polygon",
        "length_mm": round_float(length),
        "width_mm": round_float(width),
        "center_x_mm": round_float((min_x + max_x) / 2.0),
        "center_y_mm": round_float((min_y + max_y) / 2.0),
        "bbox_mm": [round_float(value) for value in patch_bbox],
    }


def summarize_feed(feed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize feed metadata exported by Geometry Engine."""

    if not feed:
        return {"type": "unknown"}
    direction = str(feed.get("direction", "unknown"))
    feed_type = "edge" if feed.get("type") == "edge_feed" else str(feed.get("type", "unknown"))
    patch_edge = point_list(feed.get("patch_edge"))
    terminal_edge = point_list(feed.get("terminal_edge"))
    summary = {
        "type": feed_type,
        "direction": direction,
        "x_mm": number_or_none(feed.get("x")),
        "y_mm": number_or_none(feed.get("y")),
        "width_mm": number_or_none(feed.get("width")),
        "length_mm": number_or_none(feed.get("length")),
        "patch_edge_mm": patch_edge,
        "terminal_edge_mm": terminal_edge,
    }
    return {key: round_nested(value) for key, value in summary.items() if value is not None}


def add_feed_offset(feed: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Add feed offset relative to patch center."""

    if feed.get("type") == "unknown":
        return feed
    direction = str(feed.get("direction", "")).lower()
    if direction in {"bottom", "top"}:
        x = number_or_none(feed.get("x_mm"))
        center = number_or_none(patch.get("center_x_mm"))
    elif direction in {"left", "right"}:
        x = number_or_none(feed.get("y_mm"))
        center = number_or_none(patch.get("center_y_mm"))
    else:
        x = None
        center = None
    if x is not None and center is not None:
        feed = dict(feed)
        feed["offset_mm"] = round_float(x - center)
    return feed


def summarize_slot(hole: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Summarize a Geometry JSON hole as a slot."""

    vertices = loop_vertices(hole)
    min_x, min_y, max_x, max_y = bbox(vertices)
    width = max_x - min_x
    height = max_y - min_y
    return {
        "id": str(hole.get("id", "slot_{0:03d}".format(index))),
        "type": "rectangle" if is_rectangle_loop(vertices) else "polygon",
        "center_x_mm": round_float((min_x + max_x) / 2.0),
        "center_y_mm": round_float((min_y + max_y) / 2.0),
        "width_mm": round_float(width),
        "height_mm": round_float(height),
        "area_mm2": round_float(abs(polygon_area(vertices))),
        "vertices": [[round_float(x), round_float(y)] for x, y in vertices],
    }


def infer_patch_bbox(outer_vertices: Sequence[Point], feed: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Infer patch bbox while excluding an edge-feed protrusion when possible."""

    min_x, min_y, max_x, max_y = bbox(outer_vertices)
    direction = str(feed.get("direction", "")).lower()
    x = number_or_none(feed.get("x_mm"))
    y = number_or_none(feed.get("y_mm"))
    if direction == "bottom" and y is not None:
        min_y = y
    elif direction == "top" and y is not None:
        max_y = y
    elif direction == "left" and x is not None:
        min_x = x
    elif direction == "right" and x is not None:
        max_x = x
    return min_x, min_y, max_x, max_y


def infer_symmetry(outer_vertices: Sequence[Point], hole_vertex_sets: Sequence[Sequence[Point]]) -> str:
    """Infer approximate mirror symmetry of outer boundary and holes."""

    all_vertices: List[Point] = list(outer_vertices)
    for vertices in hole_vertex_sets:
        all_vertices.extend(vertices)
    min_x, min_y, max_x, max_y = bbox(all_vertices)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    symmetric_x_axis = is_mirror_symmetric(all_vertices, axis="x", center=center_y)
    symmetric_y_axis = is_mirror_symmetric(all_vertices, axis="y", center=center_x)
    if symmetric_x_axis and symmetric_y_axis:
        return "xy"
    if symmetric_x_axis:
        return "x"
    if symmetric_y_axis:
        return "y"
    return "none"


def summarize_electrical_size(
    patch: Dict[str, Any],
    target_frequency_ghz: float,
    epsilon_r: Optional[float],
) -> Dict[str, Any]:
    """Compute patch dimensions relative to free-space and optional guided wavelength."""

    lambda0_mm = C0_M_PER_S / (float(target_frequency_ghz) * 1e9) * 1000.0
    length = float(patch["length_mm"])
    width = float(patch["width_mm"])
    result: Dict[str, Any] = {
        "target_frequency_ghz": round_float(target_frequency_ghz),
        "lambda0_mm": round_float(lambda0_mm),
        "length_lambda0": round_float(length / lambda0_mm),
        "width_lambda0": round_float(width / lambda0_mm),
    }
    if epsilon_r is not None and epsilon_r > 0.0:
        lambda_g_mm = lambda0_mm / math.sqrt(float(epsilon_r))
        result.update(
            {
                "epsilon_r": round_float(epsilon_r),
                "lambda_g_mm": round_float(lambda_g_mm),
                "length_lambda_g": round_float(length / lambda_g_mm),
                "width_lambda_g": round_float(width / lambda_g_mm),
            }
        )
    return result


def loop_vertices(loop: Any) -> List[Point]:
    """Extract XY vertices from a boundary or hole loop."""

    if isinstance(loop, dict):
        raw_vertices = loop.get("vertices")
    else:
        raw_vertices = None
    if not isinstance(raw_vertices, list):
        return []
    vertices: List[Point] = []
    for vertex in raw_vertices:
        if isinstance(vertex, dict):
            vertices.append((float(vertex["x"]), float(vertex["y"])))
        elif isinstance(vertex, Sequence) and len(vertex) >= 2:
            vertices.append((float(vertex[0]), float(vertex[1])))
    if len(vertices) >= 2 and distance(vertices[0], vertices[-1]) <= 1e-9:
        return vertices[:-1]
    return vertices


def point_list(value: Any) -> Optional[List[List[float]]]:
    """Normalize a list of JSON points."""

    if not isinstance(value, list):
        return None
    points: List[List[float]] = []
    for item in value:
        if isinstance(item, dict):
            points.append([float(item["x"]), float(item["y"])])
        elif isinstance(item, Sequence) and len(item) >= 2:
            points.append([float(item[0]), float(item[1])])
    return points or None


def is_axis_aligned_rectangle_bbox(
    patch_bbox: Tuple[float, float, float, float],
    outer_vertices: Sequence[Point],
) -> bool:
    """Return whether the inferred patch body is a rectangle."""

    min_x, min_y, max_x, max_y = patch_bbox
    body_points = [
        (x, y)
        for x, y in outer_vertices
        if min_x - 1e-7 <= x <= max_x + 1e-7 and min_y - 1e-7 <= y <= max_y + 1e-7
    ]
    bbox_corners = {
        rounded_point((min_x, min_y)),
        rounded_point((max_x, min_y)),
        rounded_point((max_x, max_y)),
        rounded_point((min_x, max_y)),
    }
    return bbox_corners.issubset({rounded_point(point) for point in body_points})


def is_rectangle_loop(vertices: Sequence[Point]) -> bool:
    """Return whether a loop is an axis-aligned rectangle."""

    if len(vertices) != 4:
        return False
    min_x, min_y, max_x, max_y = bbox(vertices)
    expected = {
        rounded_point((min_x, min_y)),
        rounded_point((max_x, min_y)),
        rounded_point((max_x, max_y)),
        rounded_point((min_x, max_y)),
    }
    return {rounded_point(point) for point in vertices} == expected


def is_mirror_symmetric(vertices: Sequence[Point], axis: str, center: float, tolerance: float = 1e-6) -> bool:
    """Return whether vertices are mirrored about an x or y axis."""

    rounded = {rounded_point(point, digits=6) for point in vertices}
    for x, y in vertices:
        mirror = (x, 2.0 * center - y) if axis == "x" else (2.0 * center - x, y)
        if rounded_point(mirror, digits=6) not in rounded:
            return False
    return True


def bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Return min x, min y, max x, max y."""

    if not points:
        raise ValueError("Cannot compute bbox from empty point set.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(vertices: Sequence[Point]) -> float:
    """Return signed polygon area."""

    if len(vertices) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(vertices):
        x2, y2 = vertices[(index + 1) % len(vertices)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def distance(p1: Point, p2: Point) -> float:
    """Return Euclidean distance."""

    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def number_or_none(value: Any) -> Optional[float]:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def round_nested(value: Any) -> Any:
    """Round nested numeric values for clean JSON."""

    if isinstance(value, float):
        return round_float(value)
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    return value


def rounded_point(point: Point, digits: int = 9) -> Tuple[float, float]:
    """Return rounded point tuple."""

    return round(float(point[0]), digits), round(float(point[1]), digits)


def round_float(value: float, digits: int = 6) -> float:
    """Round numeric values for stable JSON."""

    return round(float(value), digits)
