from __future__ import annotations

"""Helpers for consuming image-based port summary metadata."""

import math
from typing import Any, Dict, Optional, Sequence, Tuple


Point = Tuple[float, float]
VALID_PORT_SIDES = {"left", "right", "top", "bottom"}


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


def resolve_port_point(payload: Dict[str, Any], port_summary: Optional[Dict[str, Any]] = None) -> Optional[Point]:
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
