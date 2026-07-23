"""Adapter from BO curve-parameterization JSON back to Geometry Engine JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geometry_engine.boundary import BoundaryLoop, GeometryBoundary
from geometry_engine.exporter import GeometryJSONExporter


Point = Tuple[float, float]


def convert_bo_parameterization_to_geometry_engine(
    parameterization_json_path: str | Path,
    output_path: str | Path,
    *,
    source_geometry_json_path: str | Path | None = None,
    optimization_record: Optional[Dict[str, Any]] = None,
    port_summary_path: str | Path | None = None,
) -> Path:
    """Write a Geometry Engine JSON from a BO-optimized parameterization."""

    parameterization_path = Path(parameterization_json_path)
    parameterization = _load_json_object(parameterization_path)
    source_geometry = (
        _load_json_object(Path(source_geometry_json_path))
        if source_geometry_json_path is not None and Path(source_geometry_json_path).exists()
        else None
    )
    port_summary = (
        _load_json_object(Path(port_summary_path))
        if port_summary_path is not None and Path(port_summary_path).exists()
        else None
    )
    payload = build_geometry_engine_payload(
        parameterization=parameterization,
        source_geometry=source_geometry,
        source_parameterization_path=parameterization_path,
        optimization_record=optimization_record,
        port_summary=port_summary,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def build_geometry_engine_payload(
    *,
    parameterization: Dict[str, Any],
    source_geometry: Optional[Dict[str, Any]] = None,
    source_parameterization_path: Optional[Path] = None,
    optimization_record: Optional[Dict[str, Any]] = None,
    port_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a loaded BO parameterization to ``geometry_engine_geometry_v1``."""

    components = parameterization.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("BO parameterization must contain non-empty `components`.")

    source_metadata = _first_geometry_metadata(source_geometry)
    feed_metadata = _build_feed_metadata(
        source_geometry=source_geometry,
        parameterization=parameterization,
        optimization_record=optimization_record,
        port_summary=port_summary,
    )
    geometries: List[Dict[str, Any]] = []
    exporter = GeometryJSONExporter()
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        outer_points = _component_points(component)
        if len(outer_points) < 3:
            continue
        outer = BoundaryLoop(
            id=str(component.get("source_geometry_id") or component.get("component_id") or f"geometry_{index}"),
            role="outer",
            vertices=[_bo_to_geometry_point(point, parameterization) for point in outer_points],
        )
        holes: List[BoundaryLoop] = []
        for hole_index, hole in enumerate(component.get("holes", []) or [], start=1):
            if not isinstance(hole, dict):
                continue
            hole_points = _component_points(hole)
            if len(hole_points) < 3:
                continue
            holes.append(
                BoundaryLoop(
                    id=str(hole.get("id") or f"slot_{hole_index:03d}"),
                    role=str(hole.get("role") or "hole"),
                    vertices=[_bo_to_geometry_point(point, parameterization) for point in hole_points],
                )
            )
        metadata = dict(source_metadata if index == 0 else {})
        metadata.setdefault("semantic_type", "patch_with_feed" if feed_metadata else "patch_conductor")
        metadata.setdefault("layer", "top")
        metadata["bo_reverse_adapter"] = {
            "source_parameterization_json": (
                None if source_parameterization_path is None else str(source_parameterization_path.resolve())
            ),
            "source_backend": parameterization.get("actual_backend") or parameterization.get("backend"),
            "optimization_evaluation": (optimization_record or {}).get("evaluation"),
        }
        if index == 0 and feed_metadata is not None:
            metadata["feed"] = feed_metadata
        boundary = GeometryBoundary(
            id=str(component.get("source_geometry_id") or f"bo_component_{index}"),
            unit="mm",
            plane="XY",
            outer=outer,
            holes=holes,
            metadata=metadata,
        )
        geometries.append(exporter.boundary_to_dict(boundary)["geometries"][0])

    if not geometries:
        raise ValueError("No valid BO components could be converted to Geometry Engine JSON.")

    return {
        "schema_version": "geometry_engine_geometry_v1",
        "generator": "design_agent_bo_reverse_adapter",
        "unit": "mm",
        "coordinate_system": {
            "plane": "XY",
            "x_axis": "right",
            "y_axis": "up",
            "orientation": "right_handed",
        },
        "geometries": geometries,
        "export_rules": {
            "closed_boundary": True,
            "continuous_boundary": True,
            "ordered_vertices": True,
            "orientation": "counter_clockwise",
            "duplicated_vertices": False,
            "self_intersection": False,
        },
    }


def _build_feed_metadata(
    *,
    source_geometry: Optional[Dict[str, Any]],
    parameterization: Dict[str, Any],
    optimization_record: Optional[Dict[str, Any]],
    port_summary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    source_feed = _first_feed_metadata(source_geometry)
    variables = (optimization_record or {}).get("variables")
    variables = variables if isinstance(variables, dict) else {}
    scale_x = _finite_float(variables.get("global_scale_x"), 1.0)
    scale_y = _finite_float(variables.get("global_scale_y"), 1.0)

    if source_feed:
        direction = _normalize_direction(source_feed.get("direction")) or "bottom"
        patch_point = _scaled_source_geometry_point(
            [source_feed.get("x"), source_feed.get("y")],
            parameterization,
            scale_x,
            scale_y,
        )
        terminal = source_feed.get("terminal_point")
        if isinstance(terminal, dict) and terminal.get("x") is not None and terminal.get("y") is not None:
            terminal_point = _scaled_source_geometry_point(
                [terminal.get("x"), terminal.get("y")],
                parameterization,
                scale_x,
                scale_y,
            )
        else:
            length = _finite_float(source_feed.get("length"), 0.0)
            terminal_point = _terminal_from_patch_point(patch_point, direction, length, scale_x, scale_y)
        width_scale = scale_x if direction in {"top", "bottom"} else scale_y
        width = max(1e-9, _finite_float(source_feed.get("width"), 3.0) * width_scale)
        length = _feed_length(patch_point, terminal_point, direction)
        return {
            "type": "edge_feed",
            "x": patch_point[0],
            "y": patch_point[1],
            "width": width,
            "length": length,
            "direction": direction,
            "patch_edge": _edge_points(patch_point, width, direction),
            "terminal_point": {"x": terminal_point[0], "y": terminal_point[1], "z": 0.0},
            "terminal_edge": _edge_points(terminal_point, width, direction),
            "source": "bo_reverse_adapter_scaled_source_feed",
        }

    selected = _selected_port(port_summary)
    if selected is None:
        return None
    direction = _normalize_direction(selected.get("direction") or selected.get("port_side") or selected.get("side")) or "bottom"
    point_value = selected.get("point") or selected.get("center") or selected.get("raw_endpoint")
    if not isinstance(point_value, Sequence) or isinstance(point_value, (str, bytes)) or len(point_value) < 2:
        return None
    terminal_point = _bo_to_geometry_point([point_value[0], point_value[1]], parameterization)
    width = _finite_float(
        selected.get("port_width") or selected.get("feed_width") or selected.get("width"),
        3.0,
    )
    return {
        "type": "edge_feed",
        "x": terminal_point[0],
        "y": terminal_point[1],
        "width": width,
        "length": 0.0,
        "direction": direction,
        "patch_edge": _edge_points(terminal_point, width, direction),
        "terminal_point": {"x": terminal_point[0], "y": terminal_point[1], "z": 0.0},
        "terminal_edge": _edge_points(terminal_point, width, direction),
        "source": "bo_reverse_adapter_port_summary_fallback",
    }


def _scaled_source_geometry_point(
    point: Sequence[Any],
    parameterization: Dict[str, Any],
    scale_x: float,
    scale_y: float,
) -> Point:
    bo_point = _geometry_to_bo_point(point, parameterization)
    canvas = parameterization.get("canvas") if isinstance(parameterization.get("canvas"), dict) else {}
    center_x = _finite_float(canvas.get("width"), 0.0) / 2.0
    center_y = _finite_float(canvas.get("height"), 0.0) / 2.0
    scaled = [
        center_x + (bo_point[0] - center_x) * scale_x,
        center_y + (bo_point[1] - center_y) * scale_y,
    ]
    return _bo_to_geometry_point(scaled, parameterization)


def _component_points(component: Dict[str, Any]) -> List[List[float]]:
    raw_points = (
        component.get("points")
        or component.get("resampled_points")
        or component.get("fallback_points")
        or []
    )
    return _remove_trailing_closure(_points_from_value(raw_points))


def _bo_to_geometry_point(point: Sequence[Any], parameterization: Dict[str, Any]) -> Point:
    canvas = parameterization.get("canvas") if isinstance(parameterization.get("canvas"), dict) else {}
    origin = canvas.get("source_origin") if isinstance(canvas.get("source_origin"), list) else [0.0, 0.0]
    origin_x = _finite_float(origin[0], 0.0) if len(origin) >= 1 else 0.0
    origin_y = _finite_float(origin[1], 0.0) if len(origin) >= 2 else 0.0
    height = _finite_float(canvas.get("height"), 0.0)
    return (
        _finite_float(point[0], 0.0) + origin_x,
        height - _finite_float(point[1], 0.0) + origin_y,
    )


def _geometry_to_bo_point(point: Sequence[Any], parameterization: Dict[str, Any]) -> Point:
    canvas = parameterization.get("canvas") if isinstance(parameterization.get("canvas"), dict) else {}
    origin = canvas.get("source_origin") if isinstance(canvas.get("source_origin"), list) else [0.0, 0.0]
    origin_x = _finite_float(origin[0], 0.0) if len(origin) >= 1 else 0.0
    origin_y = _finite_float(origin[1], 0.0) if len(origin) >= 2 else 0.0
    height = _finite_float(canvas.get("height"), 0.0)
    return (
        _finite_float(point[0], 0.0) - origin_x,
        height - (_finite_float(point[1], 0.0) - origin_y),
    )


def _terminal_from_patch_point(point: Point, direction: str, length: float, scale_x: float, scale_y: float) -> Point:
    if direction == "bottom":
        return point[0], point[1] - length * scale_y
    if direction == "top":
        return point[0], point[1] + length * scale_y
    if direction == "left":
        return point[0] - length * scale_x, point[1]
    return point[0] + length * scale_x, point[1]


def _feed_length(patch_point: Point, terminal_point: Point, direction: str) -> float:
    if direction in {"top", "bottom"}:
        return abs(float(terminal_point[1]) - float(patch_point[1]))
    return abs(float(terminal_point[0]) - float(patch_point[0]))


def _edge_points(point: Point, width: float, direction: str) -> List[Dict[str, float]]:
    half = float(width) / 2.0
    if direction in {"top", "bottom"}:
        return [
            {"x": point[0] - half, "y": point[1], "z": 0.0},
            {"x": point[0] + half, "y": point[1], "z": 0.0},
        ]
    return [
        {"x": point[0], "y": point[1] - half, "z": 0.0},
        {"x": point[0], "y": point[1] + half, "z": 0.0},
    ]


def _selected_port(port_summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(port_summary, dict):
        return None
    selected = port_summary.get("selected_port")
    if isinstance(selected, dict):
        return selected
    ports = port_summary.get("ports")
    if isinstance(ports, list) and ports and isinstance(ports[0], dict):
        return ports[0]
    detection = port_summary.get("patch_port_detection")
    if isinstance(detection, dict):
        ports = detection.get("ports")
        if isinstance(ports, list) and ports and isinstance(ports[0], dict):
            return ports[0]
    return None


def _first_geometry_metadata(source_geometry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    geometry = _first_planar_geometry(source_geometry)
    metadata = geometry.get("metadata") if isinstance(geometry, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _first_feed_metadata(source_geometry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    metadata = _first_geometry_metadata(source_geometry)
    feed = metadata.get("feed")
    return dict(feed) if isinstance(feed, dict) else None


def _first_planar_geometry(source_geometry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(source_geometry, dict):
        return {}
    for geometry in source_geometry.get("geometries", []) or []:
        if isinstance(geometry, dict) and geometry.get("type") == "planar_conductor":
            return geometry
    return {}


def _points_from_value(value: Any) -> List[List[float]]:
    points: List[List[float]] = []
    if not isinstance(value, list):
        return points
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            points.append([_finite_float(item[0], 0.0), _finite_float(item[1], 0.0)])
    return points


def _remove_trailing_closure(points: Sequence[Sequence[float]]) -> List[List[float]]:
    clean = [[float(point[0]), float(point[1])] for point in points]
    while len(clean) >= 2 and math.isclose(clean[0][0], clean[-1][0], abs_tol=1e-9) and math.isclose(
        clean[0][1],
        clean[-1][1],
        abs_tol=1e-9,
    ):
        clean = clean[:-1]
    return clean


def _normalize_direction(value: Any) -> Optional[str]:
    if value is None:
        return None
    direction = str(value).strip().lower()
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
    direction = aliases.get(direction, direction)
    return direction if direction in {"left", "right", "top", "bottom"} else None


def _finite_float(value: Any, default: Optional[float] = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        if default is None:
            raise
        return float(default)
    if not math.isfinite(number):
        if default is None:
            raise ValueError(f"Expected finite float, got {value!r}")
        return float(default)
    return number


def _load_json_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


__all__ = [
    "build_geometry_engine_payload",
    "convert_bo_parameterization_to_geometry_engine",
]
