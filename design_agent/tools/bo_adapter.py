"""Adapter from Geometry Engine JSON to BO curve-parameterization JSON.

The Geometry Engine owns explicit polygon geometry in millimeters.  The
Bayesian optimization pipeline expects a curve-parameterization style payload
with components, nodes, line primitives, and a BO-compatible port summary.  This
module performs that schema bridge without calling CST, CadQuery, or the
Geometry Engine.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


def convert_geometry_engine_to_bo(
    geometry_json_path: str | Path,
    output_dir: str | Path,
    *,
    stackup_path: str | Path | None = None,
    parameterization_filename: str = "curve_parameterization.json",
    port_summary_filename: str = "patch_port_summary.json",
    connect_port: bool = False,
    include_primitive_analysis: bool = True,
) -> Dict[str, Path]:
    """Write BO-ready parameterization and port summary files.

    Returns paths for:
    - ``parameterization_json``
    - ``port_summary_json``
    - ``metadata_json``
    """

    geometry_path = Path(geometry_json_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    geometry = _load_json_object(geometry_path)
    stackup = _load_json_object(Path(stackup_path)) if stackup_path is not None else None
    parameterization = build_bo_parameterization(geometry, geometry_path=geometry_path, stackup=stackup)

    parameterization_path = output_path / parameterization_filename
    _write_json(parameterization_path, parameterization)

    port_summary = build_bo_port_summary(
        geometry,
        parameterization,
        parameterization_path=parameterization_path,
    )

    port_connection_report: Dict[str, Any] = {"enabled": False, "status": "skipped"}
    if connect_port:
        try:
            from bayesian_optimization.geometry.port_summary_utils import (
                DEFAULT_FINAL_FREE_NORMAL_INWARD_PX,
                DEFAULT_FINAL_PORT_WIDTH_SCALE,
                ensure_port_summary_connected_to_geometry,
            )

            connected, report = ensure_port_summary_connected_to_geometry(
                parameterization,
                port_summary,
                final_free_normal_inward_px=DEFAULT_FINAL_FREE_NORMAL_INWARD_PX,
                final_port_width_scale=DEFAULT_FINAL_PORT_WIDTH_SCALE,
            )
            if connected is not None:
                port_summary = connected
            port_connection_report = report
        except Exception as exc:  # pragma: no cover - optional BO helper path
            port_connection_report = {
                "enabled": True,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

    port_summary_path = output_path / port_summary_filename
    _write_json(port_summary_path, port_summary)

    primitive_analysis: Optional[Dict[str, Any]] = None
    if include_primitive_analysis:
        primitive_analysis = analyze_bo_primitives(parameterization, port_summary)
        if primitive_analysis is not None:
            _write_json(output_path / "primitive_analysis.json", primitive_analysis)

    metadata_path = output_path / "geometry_engine_bo_adapter_metadata.json"
    _write_json(
        metadata_path,
        {
            "schema_version": "geometry_engine_bo_adapter_metadata_v1",
            "source_geometry_json": str(geometry_path.resolve()),
            "parameterization_json": str(parameterization_path.resolve()),
            "port_summary_json": str(port_summary_path.resolve()),
            "stackup_json": str(Path(stackup_path).resolve()) if stackup_path is not None else None,
            "component_count": len(parameterization.get("components", []) or []),
            "node_count": len(parameterization.get("nodes", []) or []),
            "primitive_count": _count_primitives(parameterization),
            "port_connection_report": port_connection_report,
            "primitive_analysis_summary": (primitive_analysis or {}).get("summary"),
        },
    )

    return {
        "parameterization_json": parameterization_path,
        "port_summary_json": port_summary_path,
        "metadata_json": metadata_path,
    }


def build_bo_parameterization(
    geometry: Dict[str, Any],
    *,
    geometry_path: str | Path | None = None,
    stackup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a decoded Geometry Engine payload to BO curve parameterization."""

    if geometry.get("schema_version") != "geometry_engine_geometry_v1":
        raise ValueError("Expected schema_version='geometry_engine_geometry_v1'.")

    geometries = geometry.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise ValueError("Geometry Engine JSON must contain non-empty `geometries`.")

    canvas = _raw_canvas_from_stackup_or_geometry(stackup, geometries, geometry)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    constraints: List[Dict[str, Any]] = []
    components: List[Dict[str, Any]] = []
    node_lookup: Dict[Tuple[float, float], int] = {}
    edge_id = 1

    for geometry_index, item in enumerate(geometries):
        if not isinstance(item, dict):
            continue
        outer = item.get("outer_boundary")
        if not isinstance(outer, dict):
            continue
        raw_points = _vertices_to_points(outer.get("vertices"))
        points = _transform_points_to_bo(raw_points, canvas)
        if len(points) < 3:
            raise ValueError(f"Geometry {geometry_index} outer boundary has fewer than 3 vertices.")

        component_id = len(components) + 1
        outer_edge_id = edge_id
        outer_nodes = _nodes_for_points(
            points,
            nodes=nodes,
            node_lookup=node_lookup,
            component_index=component_id,
            kind="outer",
        )
        primitives = _line_primitives_from_points(
            points,
            closed=True,
            label_prefix=f"component-{component_id}-outer",
            node_ids=outer_nodes,
            source_edge_id=outer_edge_id,
        )
        edges.append(
            _edge_record(
                edge_id=outer_edge_id,
                start_node=outer_nodes[0],
                end_node=outer_nodes[0],
                points=_closed_points(points),
                source_component_index=component_id,
                closed=True,
            )
        )
        constraints.append(
            {
                "type": "edge_endpoint_lock",
                "edge_id": outer_edge_id,
                "start_node": outer_nodes[0],
                "end_node": outer_nodes[0],
            }
        )
        edge_id += 1

        holes: List[Dict[str, Any]] = []
        for hole_index, hole in enumerate(item.get("holes", []) or [], start=1):
            adapted = _adapt_bo_hole(
                hole,
                hole_index=hole_index,
                component_id=component_id,
                nodes=nodes,
                node_lookup=node_lookup,
                edge_id=edge_id,
                canvas=canvas,
            )
            if adapted is None:
                continue
            hole_payload, hole_edge, hole_constraint = adapted
            holes.append(hole_payload)
            edges.append(hole_edge)
            constraints.append(hole_constraint)
            edge_id += 1

        hole_primitive_count = sum(len(hole.get("primitives", []) or []) for hole in holes)
        primitive_count = len(primitives) + hole_primitive_count
        component = {
            "component_id": component_id,
            "source_geometry_index": geometry_index,
            "source_geometry_id": item.get("id", f"geometry_{geometry_index}"),
            "source_edge_id": outer_edge_id,
            "closed": True,
            "topology": "solid_with_holes" if holes else "solid",
            "start_node": outer_nodes[0],
            "end_node": outer_nodes[0],
            "bbox": list(_bbox(points)),
            "area_px": abs(_polygon_area(points)),
            "area_mm2": abs(_polygon_area(points)),
            "sampled_point_count": len(_closed_points(points)),
            "point_count": len(_closed_points(points)),
            "points": _closed_points(points),
            "fallback_points": _closed_points(points),
            "resampled_points": _closed_points(points),
            "holes": holes,
            "primitives": primitives,
            "segments": primitives,
            "metrics": {
                "sampled_point_count": len(_closed_points(points)),
                "primitive_count": primitive_count,
                "primitive_by_type": {"line": primitive_count},
                "parameter_count": primitive_count * 4,
                "mean_error_px": 0.0,
                "max_error_px": 0.0,
                "compression_ratio": float((primitive_count * 4) / max(1, len(points) * 2)),
                "hole_count": len(holes),
                "hole_primitive_count": hole_primitive_count,
            },
            "metadata": {
                "source": "geometry_engine_geometry_v1",
                "geometry_id": item.get("id", f"geometry_{geometry_index}"),
                "semantic_type": (item.get("metadata") or {}).get("semantic_type"),
                "unit": item.get("unit") or geometry.get("unit") or "mm",
            },
        }
        components.append(component)

    if not components:
        raise ValueError("No modelable planar conductor geometry was found.")

    _update_node_degrees(nodes, components)
    metrics = _aggregate_metrics(components)

    return {
        "schema_version": "3.0",
        "backend": "graph_local_lines",
        "actual_backend": "geometry_engine_bo_adapter",
        "source_geometry_json": str(Path(geometry_path).resolve()) if geometry_path is not None else None,
        "coordinate_system": geometry.get("coordinate_system", {}),
        "canvas": canvas,
        "nodes": nodes,
        "edges": edges,
        "components": components,
        "constraints": constraints,
        "metadata": {
            "source": "geometry_engine_geometry_v1",
            "adapter": "design_agent.tools.bo_adapter",
            "line_only_parameterization": True,
            "units_are_physical_mm": True,
            "coordinate_transform": {
                "from": "geometry_engine_xy_y_up_mm",
                "to": "bo_image_xy_y_down_mm",
                "formula": "x_bo=x-origin_x; y_bo=canvas_height-(y-origin_y)",
                "origin": canvas.get("source_origin", [0.0, 0.0]),
                "canvas_height": canvas.get("height"),
            },
        },
        "metrics": metrics,
    }


def build_bo_port_summary(
    geometry: Dict[str, Any],
    parameterization: Dict[str, Any],
    *,
    parameterization_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Build a BO-compatible patch_port_summary from Geometry Engine feed metadata."""

    feed = _find_feed_metadata(geometry)
    canvas = parameterization.get("canvas", {}) if isinstance(parameterization.get("canvas"), dict) else {}
    canvas_width = _finite_float(canvas.get("width"))
    canvas_height = _finite_float(canvas.get("height"))

    if feed is not None:
        direction = _normalize_port_side(feed.get("direction")) or "bottom"
        feed_width = max(1e-6, _finite_float(feed.get("width"), 2.0))
        terminal_point = feed.get("terminal_point")
        if isinstance(terminal_point, dict):
            raw_port_x = _finite_float(terminal_point.get("x"), _finite_float(feed.get("x"), 0.0))
            raw_port_y = _finite_float(terminal_point.get("y"), _finite_float(feed.get("y"), 0.0))
        else:
            raw_port_x = _finite_float(feed.get("x"), 0.0)
            raw_port_y = _finite_float(feed.get("y"), 0.0)
        port_x, port_y = _transform_point_to_bo([raw_port_x, raw_port_y], canvas)
        raw_edge_points = _parse_optional_edge(feed.get("terminal_edge"))
        edge_points = _transform_points_to_bo(raw_edge_points, canvas) if raw_edge_points is not None else None
        if edge_points is None:
            edge_points = _edge_from_port_point(port_x, port_y, feed_width, direction)
        source = "geometry_engine_feed_metadata"
        feed_length = _finite_float(feed.get("length"), None)
    else:
        port_x, port_y, direction, feed_width, edge_points = _fallback_port_from_geometry(parameterization)
        source = "synthetic_from_geometry_bbox"
        feed_length = None

    candidate = {
        "id": 0,
        "point": [port_x, port_y],
        "raw_endpoint": [port_x, port_y],
        "cst_contact_point": [port_x, port_y],
        "center": [port_x, port_y],
        "direction": direction,
        "port_side": direction,
        "side": direction,
        "local_width": feed_width,
        "feed_width": feed_width,
        "width": feed_width,
        "port_width": feed_width,
        "score": 1.0,
        "confidence": 1.0,
        "source": source,
        "touches_border": _touches_canvas_border(port_x, port_y, direction, canvas_width, canvas_height),
        "connected_to_main_patch": True,
    }
    if feed_length is not None:
        candidate["feed_length"] = feed_length

    summary = {
        "schema_version": "patch_port_summary_v1",
        "name": "geometry_engine_edge_feed",
        "parameterization_json": str(Path(parameterization_path).resolve()) if parameterization_path is not None else None,
        "source": source,
        "border_contact_mode": "geometry_engine_edge_feed",
        "closest_edge": edge_points,
        "closest_border_sides": [direction],
        "ports": [copy.deepcopy(candidate)],
        "selected_port": copy.deepcopy(candidate),
        "port_geometries": [
            {
                "port_id": 0,
                "endpoint": [port_x, port_y],
                "raw_endpoint": [port_x, port_y],
                "center": [port_x, port_y],
                "cst_contact_point": [port_x, port_y],
                "direction": direction,
                "width": feed_width,
                "feed_width": feed_width,
                "closest_edge": edge_points,
                "source": source,
            }
        ],
        "debug_metadata": {
            "source": source,
            "geometry_engine_feed": feed,
            "canvas": canvas,
            "adapter": "design_agent.tools.bo_adapter",
        },
        "patch_port_detection": {
            "enabled": True,
            "source": source,
            "ports": [copy.deepcopy(candidate)],
            "selected_port": copy.deepcopy(candidate),
            "metadata": {
                "closest_edge": edge_points,
                "closest_border_sides": [direction],
            },
        },
        "bo_port_connection_adjustment": {
            "point": [port_x, port_y],
            "connected_point": [port_x, port_y],
            "final_free_normal_inward_px": 0.0,
            "source": source,
        },
    }
    return summary


def analyze_bo_primitives(
    parameterization: Dict[str, Any],
    port_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run BO primitive analyzer when available."""

    try:
        from bayesian_optimization.geometry.primitive_analyzer import analyze_primitives

        return analyze_primitives(parameterization, port_summary=port_summary)
    except Exception:
        return None


def _adapt_bo_hole(
    hole: Any,
    *,
    hole_index: int,
    component_id: int,
    nodes: List[Dict[str, Any]],
    node_lookup: Dict[Tuple[float, float], int],
    edge_id: int,
    canvas: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    if not isinstance(hole, dict):
        return None
    raw_points = _vertices_to_points(hole.get("vertices"))
    points = _transform_points_to_bo(raw_points, canvas)
    if len(points) < 3:
        return None
    node_ids = _nodes_for_points(
        points,
        nodes=nodes,
        node_lookup=node_lookup,
        component_index=component_id,
        kind=f"hole_{hole_index}",
    )
    primitives = _line_primitives_from_points(
        points,
        closed=True,
        label_prefix=f"component-{component_id}-hole-{hole_index}",
        node_ids=node_ids,
        source_edge_id=edge_id,
    )
    payload = {
        "hole_id": hole_index,
        "id": hole.get("id", f"hole_{hole_index:03d}"),
        "role": hole.get("role", "hole"),
        "closed": True,
        "bbox": list(_bbox(points)),
        "area_px": abs(_polygon_area(points)),
        "area_mm2": abs(_polygon_area(points)),
        "point_count": len(_closed_points(points)),
        "points": _closed_points(points),
        "fallback_points": _closed_points(points),
        "resampled_points": _closed_points(points),
        "start_node": node_ids[0],
        "end_node": node_ids[0],
        "primitives": primitives,
        "segments": primitives,
        "metadata": {"source": "geometry_engine_geometry_v1", "role": hole.get("role", "hole")},
    }
    edge = _edge_record(
        edge_id=edge_id,
        start_node=node_ids[0],
        end_node=node_ids[0],
        points=_closed_points(points),
        source_component_index=component_id,
        closed=True,
    )
    constraint = {
        "type": "edge_endpoint_lock",
        "edge_id": edge_id,
        "start_node": node_ids[0],
        "end_node": node_ids[0],
    }
    return payload, edge, constraint


def _line_primitives_from_points(
    points: Sequence[Sequence[float]],
    *,
    closed: bool,
    label_prefix: str,
    node_ids: Sequence[int],
    source_edge_id: int,
) -> List[Dict[str, Any]]:
    clean = _remove_trailing_closure([_point_list(point) for point in points])
    if len(clean) < 2:
        return []
    pairs = [(idx, idx + 1) for idx in range(len(clean) - 1)]
    if closed and len(clean) >= 3:
        pairs.append((len(clean) - 1, 0))

    primitives: List[Dict[str, Any]] = []
    for segment_id, (start_idx, end_idx) in enumerate(pairs, start=1):
        start = clean[start_idx]
        end = clean[end_idx]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        direction = [dx / length, dy / length] if length > 1e-12 else [0.0, 0.0]
        primitive = {
            "type": "line",
            "kind": "line",
            "primitive_type": "line",
            "fit_method": "geometry_engine_polygon_edge",
            "line_only_parameterization": True,
            "start": start,
            "end": end,
            "points": [start, end],
            "fallback_points": [start, end],
            "direction": direction,
            "length": length,
            "max_error": 0.0,
            "mean_error": 0.0,
            "effective_params": 4,
            "parameter_count": 4,
            "source_point_count": 2,
            "source_edge_id": int(source_edge_id),
            "segment_id": int(segment_id),
            "source_start_index": int(start_idx),
            "source_end_index": int(end_idx),
            "start_node": int(node_ids[start_idx]),
            "end_node": int(node_ids[end_idx]),
            "visual_label": f"#{segment_id} LINE {label_prefix}",
        }
        primitive["parameters"] = {
            "start": primitive["start"],
            "end": primitive["end"],
            "direction": primitive["direction"],
        }
        primitives.append(primitive)
    return primitives


def _nodes_for_points(
    points: Sequence[Sequence[float]],
    *,
    nodes: List[Dict[str, Any]],
    node_lookup: Dict[Tuple[float, float], int],
    component_index: int,
    kind: str,
) -> List[int]:
    node_ids: List[int] = []
    for point_index, point in enumerate(_remove_trailing_closure([_point_list(point) for point in points])):
        key = (round(point[0], 9), round(point[1], 9))
        node_id = node_lookup.get(key)
        if node_id is None:
            node_id = len(nodes) + 1
            node_lookup[key] = node_id
            nodes.append(
                {
                    "id": node_id,
                    "x": point[0],
                    "y": point[1],
                    "degree": 0,
                    "type": "junction",
                    "source_refs": [],
                }
            )
        nodes[node_id - 1].setdefault("source_refs", []).append(
            {
                "component_index": int(component_index),
                "point_index": int(point_index),
                "kind": kind,
            }
        )
        node_ids.append(node_id)
    return node_ids


def _update_node_degrees(nodes: List[Dict[str, Any]], components: Sequence[Dict[str, Any]]) -> None:
    degree_by_node = {int(node["id"]): 0 for node in nodes}
    for component in components:
        for primitive in component.get("primitives", []) or []:
            degree_by_node[int(primitive.get("start_node", 0))] = degree_by_node.get(int(primitive.get("start_node", 0)), 0) + 1
            degree_by_node[int(primitive.get("end_node", 0))] = degree_by_node.get(int(primitive.get("end_node", 0)), 0) + 1
        for hole in component.get("holes", []) or []:
            for primitive in hole.get("primitives", []) or []:
                degree_by_node[int(primitive.get("start_node", 0))] = degree_by_node.get(int(primitive.get("start_node", 0)), 0) + 1
                degree_by_node[int(primitive.get("end_node", 0))] = degree_by_node.get(int(primitive.get("end_node", 0)), 0) + 1
    for node in nodes:
        node["degree"] = int(degree_by_node.get(int(node["id"]), 0))
        node["type"] = "endpoint" if node["degree"] <= 1 else "junction"


def _edge_record(
    *,
    edge_id: int,
    start_node: int,
    end_node: int,
    points: Sequence[Sequence[float]],
    source_component_index: int,
    closed: bool,
) -> Dict[str, Any]:
    clean = [_point_list(point) for point in points]
    return {
        "id": int(edge_id),
        "start_node": int(start_node),
        "end_node": int(end_node),
        "ordered_points": clean,
        "length": _polyline_length(clean),
        "local_curvature": [],
        "local_width_estimate": None,
        "is_closed_loop": bool(closed),
        "source_component_index": int(source_component_index),
    }


def _raw_canvas_from_stackup_or_geometry(
    stackup: Optional[Dict[str, Any]],
    geometries: Sequence[Any],
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    unit = str(geometry.get("unit") or "mm")
    substrate = stackup.get("substrate") if isinstance(stackup, dict) else None
    if isinstance(substrate, dict):
        width = _finite_float(substrate.get("width"), None)
        height = _finite_float(substrate.get("length"), None)
        if width is not None and width > 0 and height is not None and height > 0:
            return {"width": width, "height": height, "unit": unit, "source_origin": [0.0, 0.0]}

    all_points: List[List[float]] = []
    for item in geometries:
        if not isinstance(item, dict):
            continue
        outer = item.get("outer_boundary")
        if isinstance(outer, dict):
            all_points.extend(_vertices_to_points(outer.get("vertices")))
        for hole in item.get("holes", []) or []:
            if isinstance(hole, dict):
                all_points.extend(_vertices_to_points(hole.get("vertices")))
    min_x, min_y, max_x, max_y = _bbox(all_points)
    return {
        "width": max_x - min_x,
        "height": max_y - min_y,
        "unit": unit,
        "source_origin": [min_x, min_y],
        "note": "derived_from_geometry_bbox_no_stackup",
    }


def _transform_points_to_bo(points: Optional[Sequence[Sequence[float]]], canvas: Dict[str, Any]) -> List[List[float]]:
    if points is None:
        return []
    return [_transform_point_to_bo(point, canvas) for point in points]


def _transform_point_to_bo(point: Sequence[float], canvas: Dict[str, Any]) -> List[float]:
    origin = canvas.get("source_origin") if isinstance(canvas.get("source_origin"), list) else [0.0, 0.0]
    origin_x = _finite_float(origin[0], 0.0) if len(origin) >= 1 else 0.0
    origin_y = _finite_float(origin[1], 0.0) if len(origin) >= 2 else 0.0
    height = _finite_float(canvas.get("height"), 0.0)
    x = _finite_float(point[0], 0.0) - origin_x
    y = height - (_finite_float(point[1], 0.0) - origin_y)
    return [x, y]


def _aggregate_metrics(components: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    primitive_count = _count_primitives({"components": list(components)})
    hole_count = sum(len(component.get("holes", []) or []) for component in components)
    return {
        "component_count": len(components),
        "primitive_count": primitive_count,
        "primitive_by_type": {"line": primitive_count},
        "hole_count": hole_count,
        "line_only_parameterization": True,
    }


def _count_primitives(parameterization: Dict[str, Any]) -> int:
    total = 0
    for component in parameterization.get("components", []) or []:
        total += len(component.get("primitives", []) or component.get("segments", []) or [])
        for hole in component.get("holes", []) or []:
            total += len(hole.get("primitives", []) or hole.get("segments", []) or [])
    return total


def _find_feed_metadata(geometry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in geometry.get("geometries", []) or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        feed = metadata.get("feed") if isinstance(metadata, dict) else None
        if isinstance(feed, dict) and feed.get("x") is not None and feed.get("y") is not None:
            return feed
    return None


def _fallback_port_from_geometry(parameterization: Dict[str, Any]) -> Tuple[float, float, str, float, List[List[float]]]:
    component = (parameterization.get("components") or [{}])[0]
    points = component.get("points") or component.get("fallback_points") or [[0.0, 0.0], [1.0, 1.0]]
    min_x, min_y, max_x, _max_y = _bbox(points)
    width = max(1.0, (max_x - min_x) * 0.08)
    x = 0.5 * (min_x + max_x)
    y = min_y
    edge = _edge_from_port_point(x, y, width, "bottom")
    return x, y, "bottom", width, edge


def _edge_from_port_point(x: float, y: float, width: float, direction: str) -> List[List[float]]:
    half = width / 2.0
    if direction in {"top", "bottom"}:
        return [[x - half, y], [x + half, y]]
    return [[x, y - half], [x, y + half]]


def _parse_optional_edge(value: Any) -> Optional[List[List[float]]]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    points: List[List[float]] = []
    for item in value[:2]:
        if isinstance(item, dict):
            points.append([_finite_float(item.get("x"), 0.0), _finite_float(item.get("y"), 0.0)])
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 2:
            points.append([_finite_float(item[0], 0.0), _finite_float(item[1], 0.0)])
        else:
            return None
    return points


def _vertices_to_points(vertices: Any) -> List[List[float]]:
    if not isinstance(vertices, list):
        return []
    points: List[List[float]] = []
    for vertex in vertices:
        if isinstance(vertex, dict):
            points.append([_finite_float(vertex.get("x"), 0.0), _finite_float(vertex.get("y"), 0.0)])
        elif isinstance(vertex, Sequence) and not isinstance(vertex, (str, bytes)) and len(vertex) >= 2:
            points.append([_finite_float(vertex[0], 0.0), _finite_float(vertex[1], 0.0)])
    return _remove_trailing_closure(points)


def _closed_points(points: Sequence[Sequence[float]]) -> List[List[float]]:
    clean = _remove_trailing_closure([_point_list(point) for point in points])
    if len(clean) >= 3:
        clean.append(list(clean[0]))
    return clean


def _remove_trailing_closure(points: Sequence[Sequence[float]]) -> List[List[float]]:
    clean = [_point_list(point) for point in points]
    while len(clean) >= 2 and math.isclose(clean[0][0], clean[-1][0], abs_tol=1e-9) and math.isclose(clean[0][1], clean[-1][1], abs_tol=1e-9):
        clean = clean[:-1]
    return clean


def _point_list(point: Sequence[float]) -> List[float]:
    return [_finite_float(point[0], 0.0), _finite_float(point[1], 0.0)]


def _bbox(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    xs = [_finite_float(point[0], 0.0) for point in points]
    ys = [_finite_float(point[1], 0.0) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    clean = _remove_trailing_closure(points)
    if len(clean) < 3:
        return 0.0
    total = 0.0
    for index, start in enumerate(clean):
        end = clean[(index + 1) % len(clean)]
        total += start[0] * end[1] - end[0] * start[1]
    return 0.5 * total


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for index in range(1, len(points)):
        total += math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
    return total


def _touches_canvas_border(
    x: float,
    y: float,
    direction: str,
    canvas_width: Optional[float],
    canvas_height: Optional[float],
) -> bool:
    tolerance = 1e-6
    if direction == "left":
        return math.isclose(x, 0.0, abs_tol=tolerance)
    if direction == "right" and canvas_width is not None:
        return math.isclose(x, canvas_width, abs_tol=tolerance)
    if direction == "top":
        return math.isclose(y, 0.0, abs_tol=tolerance)
    if direction == "bottom" and canvas_height is not None:
        return math.isclose(y, canvas_height, abs_tol=tolerance)
    return False


def _normalize_port_side(value: Any) -> Optional[str]:
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
    return side if side in {"left", "right", "top", "bottom"} else None


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
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
