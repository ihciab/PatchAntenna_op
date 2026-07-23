"""Variable selection for design-agent Bayesian optimization handoff.

This module is intentionally separate from the existing
``bayesian_optimization`` implementation.  It describes the BO variables that
the design agent wants after an LLM geometry-edit step, without changing the
legacy optimizer's extraction or mutation functions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

STRATEGY_NAME = "llm_slot_and_model_size"


def build_llm_slot_model_size_variable_plan_from_files(
    parameterization_json: Path | str,
    primitive_analysis_json: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    *,
    scale_min: float = 0.5,
    scale_max: float = 1.5,
    global_scale_min: float = 0.8,
    global_scale_max: float = 1.5,
    board_margin: float = 0.0,
) -> Dict[str, Any]:
    """Build and optionally write the LLM-post-edit BO variable plan."""

    parameterization_path = Path(parameterization_json)
    parameterization = _load_json_object(parameterization_path)
    primitive_analysis = (
        _load_json_object(Path(primitive_analysis_json))
        if primitive_analysis_json is not None and Path(primitive_analysis_json).exists()
        else None
    )
    plan = build_llm_slot_model_size_variable_plan(
        parameterization,
        primitive_analysis=primitive_analysis,
        scale_min=scale_min,
        scale_max=scale_max,
        global_scale_min=global_scale_min,
        global_scale_max=global_scale_max,
        board_margin=board_margin,
    )
    plan["source_files"] = {
        "parameterization_json": str(parameterization_path.resolve()),
        "primitive_analysis_json": (
            str(Path(primitive_analysis_json).resolve()) if primitive_analysis_json is not None else None
        ),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def build_llm_slot_model_size_variable_plan(
    parameterization: Dict[str, Any],
    primitive_analysis: Optional[Dict[str, Any]] = None,
    *,
    scale_min: float = 0.5,
    scale_max: float = 1.5,
    global_scale_min: float = 0.8,
    global_scale_max: float = 1.5,
    board_margin: float = 0.0,
) -> Dict[str, Any]:
    """Select only LLM-added slot variables and whole-model size variables.

    Variables:
    - ``global_scale_x`` / ``global_scale_y``: whole-board/model anisotropic
      scale variables, bounded by ``global_scale_min`` and
      ``global_scale_max``.
    - ``slot_<id>_dx/dy``: rigid slot translation. Bounds keep the current
      slot bbox inside the dielectric-board bbox.
    - ``slot_<id>_length/width``: absolute slot length and width. Bounds are
      the current slot dimension multiplied by
      ``scale_min`` and ``scale_max``.
    """

    _validate_scale_range(scale_min, scale_max)
    _validate_scale_range(global_scale_min, global_scale_max)
    board_bbox = _board_bbox(parameterization)
    slots = _slot_records(parameterization, primitive_analysis)

    variables: List[Dict[str, Any]] = [
        {
            "name": "global_scale_x",
            "variable_type": "global_scale_x",
            "target": "model",
            "lower": float(global_scale_min),
            "upper": float(global_scale_max),
            "default": 1.0,
            "description": "Scale the whole model along x after the LLM geometry edit.",
        },
        {
            "name": "global_scale_y",
            "variable_type": "global_scale_y",
            "target": "model",
            "lower": float(global_scale_min),
            "upper": float(global_scale_max),
            "default": 1.0,
            "description": "Scale the whole model along y after the LLM geometry edit.",
        }
    ]

    for slot in slots:
        slot_id = _safe_name(str(slot.get("slot_id") or slot.get("primitive_id") or "slot"))
        bbox = tuple(float(value) for value in slot["bbox"])
        translation_bbox = _bbox_from_value(slot.get("parent_component_bbox")) or board_bbox
        translate_x = _translation_bounds(translation_bbox, bbox, axis="x", margin=board_margin)
        translate_y = _translation_bounds(translation_bbox, bbox, axis="y", margin=board_margin)
        dimensions = _slot_length_width_dimensions(bbox)
        variables.extend(
            [
                {
                    "name": f"{slot_id}_dx",
                    "variable_type": "slot_dx",
                    "target": "slot",
                    "slot_id": slot.get("slot_id"),
                    "primitive_id": slot.get("primitive_id"),
                    "lower": translate_x[0],
                    "upper": translate_x[1],
                    "default": 0.0,
                    "description": "Rigidly translate the LLM-added slot along x while keeping it inside the parent conductor.",
                },
                {
                    "name": f"{slot_id}_dy",
                    "variable_type": "slot_dy",
                    "target": "slot",
                    "slot_id": slot.get("slot_id"),
                    "primitive_id": slot.get("primitive_id"),
                    "lower": translate_y[0],
                    "upper": translate_y[1],
                    "default": 0.0,
                    "description": "Rigidly translate the LLM-added slot along y while keeping it inside the parent conductor.",
                },
                {
                    "name": f"{slot_id}_length",
                    "variable_type": "slot_length",
                    "target": "slot",
                    "slot_id": slot.get("slot_id"),
                    "primitive_id": slot.get("primitive_id"),
                    "axis": dimensions["length_axis"],
                    "lower": dimensions["length"] * float(scale_min),
                    "upper": dimensions["length"] * float(scale_max),
                    "default": dimensions["length"],
                    "description": "Set the LLM-added slot length while preserving its center.",
                },
                {
                    "name": f"{slot_id}_width",
                    "variable_type": "slot_width",
                    "target": "slot",
                    "slot_id": slot.get("slot_id"),
                    "primitive_id": slot.get("primitive_id"),
                    "axis": dimensions["width_axis"],
                    "lower": dimensions["width"] * float(scale_min),
                    "upper": dimensions["width"] * float(scale_max),
                    "default": dimensions["width"],
                    "description": "Set the LLM-added slot width while preserving its center.",
                },
            ]
        )

    return {
        "schema_version": "design_agent_bo_variable_plan_v1",
        "strategy_name": STRATEGY_NAME,
        "description": "Optimize global x/y scale plus each LLM-added slot dx, dy, length, and width.",
        "board_bbox": list(board_bbox),
        "board_margin": float(board_margin),
        "slot_dimension_scale_range": [float(scale_min), float(scale_max)],
        "global_scale_range": [float(global_scale_min), float(global_scale_max)],
        "slot_count": len(slots),
        "expected_variable_count": 2 + 4 * len(slots),
        "slots": slots,
        "variables": variables,
        "notes": [
            "This file only defines the design-agent variable selection plan.",
            "It does not modify bayesian_optimization extraction or mutation functions.",
            "The variable count is 2 + 4 * slot_count.",
            "Translation bounds are computed from the current slot bbox and parent conductor bbox so the slot remains visible in CST.",
            "The runtime adapter clamps dx/dy again after global scaling and slot dimension changes.",
        ],
    }


def _slot_records(
    parameterization: Dict[str, Any],
    primitive_analysis: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    records = _slot_records_from_analysis(primitive_analysis)
    if records:
        return _attach_parent_component_bboxes(parameterization, records)
    return _attach_parent_component_bboxes(parameterization, _slot_records_from_parameterization(parameterization))


def _slot_records_from_analysis(primitive_analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(primitive_analysis, dict):
        return []
    slots: List[Dict[str, Any]] = []
    for primitive in primitive_analysis.get("primitives", []) or []:
        if not isinstance(primitive, dict):
            continue
        if primitive.get("type") != "HOLE" and primitive.get("role") != "SLOT":
            continue
        bbox = _bbox_from_value(primitive.get("bbox")) or _points_bbox(_points_from_value(primitive.get("points")))
        if bbox is None:
            continue
        primitive_id = str(primitive.get("primitive_id") or f"slot_{len(slots) + 1}")
        slots.append(
            {
                "slot_id": primitive_id,
                "primitive_id": primitive_id,
                "component_index": primitive.get("component_index"),
                "primitive_index": primitive.get("primitive_index"),
                "source_key": primitive.get("source_key"),
                "bbox": list(bbox),
                "center": list(_bbox_center(bbox)),
                "dimensions": _slot_length_width_dimensions(bbox),
                "source": "primitive_analysis",
            }
        )
    return slots


def _slot_records_from_parameterization(parameterization: Dict[str, Any]) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    components = parameterization.get("components", []) or []
    for component_index, component in enumerate(components):
        if not isinstance(component, dict):
            continue
        for hole_index, hole in enumerate(component.get("holes", []) or []):
            if not isinstance(hole, dict):
                continue
            bbox = _bbox_from_value(hole.get("bbox")) or _points_bbox(
                _points_from_value(hole.get("resampled_points") or hole.get("fallback_points") or hole.get("points"))
            )
            if bbox is None:
                continue
            slot_id = str(hole.get("id") or f"component_{component_index + 1}_hole_{hole_index + 1}")
            slots.append(
                {
                    "slot_id": slot_id,
                    "primitive_id": None,
                    "component_index": component_index,
                    "primitive_index": hole_index,
                    "source_key": "holes",
                    "bbox": list(bbox),
                    "center": list(_bbox_center(bbox)),
                    "dimensions": _slot_length_width_dimensions(bbox),
                    "source": "curve_parameterization",
                }
            )
    return slots


def _board_bbox(parameterization: Dict[str, Any]) -> BBox:
    canvas = parameterization.get("canvas") if isinstance(parameterization.get("canvas"), dict) else {}
    width = _number_or_none(canvas.get("width"))
    height = _number_or_none(canvas.get("height"))
    if width is not None and height is not None and width > 0.0 and height > 0.0:
        return (0.0, 0.0, float(width), float(height))

    points: List[Point] = []
    for component in parameterization.get("components", []) or []:
        if not isinstance(component, dict):
            continue
        points.extend(
            _points_from_value(component.get("resampled_points") or component.get("fallback_points") or component.get("points"))
        )
    bbox = _points_bbox(points)
    if bbox is None:
        raise ValueError("Cannot infer dielectric board bbox from parameterization canvas or component points.")
    return bbox


def _attach_parent_component_bboxes(
    parameterization: Dict[str, Any],
    slots: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    components = parameterization.get("components", []) or []
    for slot in slots:
        component_index = slot.get("component_index")
        if not isinstance(component_index, int) or component_index < 0 or component_index >= len(components):
            continue
        component = components[component_index]
        if not isinstance(component, dict):
            continue
        parent_bbox = _bbox_from_value(component.get("bbox")) or _points_bbox(
            _points_from_value(component.get("resampled_points") or component.get("fallback_points") or component.get("points"))
        )
        if parent_bbox is not None:
            slot["parent_component_bbox"] = list(parent_bbox)
    return slots


def _translation_bounds(board: BBox, slot: BBox, *, axis: str, margin: float) -> Tuple[float, float]:
    if axis == "x":
        lower = board[0] + margin - slot[0]
        upper = board[2] - margin - slot[2]
    elif axis == "y":
        lower = board[1] + margin - slot[1]
        upper = board[3] - margin - slot[3]
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    if lower > upper:
        midpoint = 0.5 * (lower + upper)
        return (float(midpoint), float(midpoint))
    return (float(lower), float(upper))


def _bbox_from_value(value: Any) -> Optional[BBox]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    numbers = [_number_or_none(item) for item in value[:4]]
    if any(number is None for number in numbers):
        return None
    x0, y0, x1, y1 = [float(number) for number in numbers]
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _points_from_value(value: Any) -> List[Point]:
    points: List[Point] = []
    if not isinstance(value, list):
        return points
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            continue
        x = _number_or_none(item[0])
        y = _number_or_none(item[1])
        if x is not None and y is not None:
            points.append((float(x), float(y)))
    return points


def _points_bbox(points: Sequence[Point]) -> Optional[BBox]:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_center(bbox: BBox) -> Point:
    return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


def _slot_length_width_dimensions(bbox: BBox) -> Dict[str, Any]:
    x_size = max(1e-9, float(bbox[2]) - float(bbox[0]))
    y_size = max(1e-9, float(bbox[3]) - float(bbox[1]))
    if x_size >= y_size:
        return {
            "length": float(x_size),
            "width": float(y_size),
            "length_axis": "x",
            "width_axis": "y",
        }
    return {
        "length": float(y_size),
        "width": float(x_size),
        "length_axis": "y",
        "width_axis": "x",
    }


def _number_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_scale_range(scale_min: float, scale_max: float) -> None:
    if not math.isfinite(float(scale_min)) or not math.isfinite(float(scale_max)):
        raise ValueError("scale_min and scale_max must be finite.")
    if float(scale_min) <= 0.0 or float(scale_max) <= float(scale_min):
        raise ValueError("scale range must be positive and increasing.")


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    return safe.strip("_") or "slot"


def _load_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


__all__ = [
    "STRATEGY_NAME",
    "build_llm_slot_model_size_variable_plan",
    "build_llm_slot_model_size_variable_plan_from_files",
]
