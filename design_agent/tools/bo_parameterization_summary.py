"""Summarize BO parameterization and variable plan for LLM feedback."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_INPUTS_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"
BO_PARAMETERIZATION_SUMMARY_FILENAME = "bo_parameterization_summary.json"


@dataclass(frozen=True)
class BOParameterizationSummary:
    """Compact BO parameterization summary for LLM consumers."""

    optimization_surface: Dict[str, Any]
    model: Dict[str, Any]
    port: Dict[str, Any]
    slots: List[Dict[str, Any]]
    variables: List[Dict[str, Any]]
    variable_groups: Dict[str, Any]
    source: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


class BOParameterizationSummaryBuilder:
    """Build a compact summary from BO parameterization artifacts."""

    def from_files(
        self,
        *,
        parameterization_json: Path | str,
        variable_plan_json: Optional[Path | str] = None,
        port_summary_json: Optional[Path | str] = None,
        primitive_analysis_json: Optional[Path | str] = None,
        variable_plan: Optional[Dict[str, Any]] = None,
    ) -> BOParameterizationSummary:
        """Load BO files and summarize the parameterization surface."""

        parameterization_path = Path(parameterization_json)
        parameterization = load_json_object(parameterization_path)
        plan_path = Path(variable_plan_json) if variable_plan_json is not None else None
        port_path = Path(port_summary_json) if port_summary_json is not None else None
        primitive_path = Path(primitive_analysis_json) if primitive_analysis_json is not None else None
        plan = variable_plan or (load_json_object(plan_path) if plan_path is not None and plan_path.exists() else None)
        port_summary = load_json_object(port_path) if port_path is not None and port_path.exists() else None
        primitive_analysis = (
            load_json_object(primitive_path) if primitive_path is not None and primitive_path.exists() else None
        )
        return self.from_dicts(
            parameterization=parameterization,
            variable_plan=plan,
            port_summary=port_summary,
            primitive_analysis=primitive_analysis,
            source_paths={
                "parameterization_json": parameterization_path,
                "variable_plan_json": plan_path,
                "port_summary_json": port_path,
                "primitive_analysis_json": primitive_path,
            },
        )

    def from_dicts(
        self,
        *,
        parameterization: Dict[str, Any],
        variable_plan: Optional[Dict[str, Any]] = None,
        port_summary: Optional[Dict[str, Any]] = None,
        primitive_analysis: Optional[Dict[str, Any]] = None,
        source_paths: Optional[Dict[str, Optional[Path]]] = None,
    ) -> BOParameterizationSummary:
        """Summarize already loaded BO parameterization artifacts."""

        plan = variable_plan if isinstance(variable_plan, dict) else {}
        slots = summarize_slots(parameterization, plan)
        variables = summarize_variables(plan)
        variable_groups = summarize_variable_groups(variables)
        model = summarize_model(parameterization, plan)
        port = summarize_port(port_summary)
        optimization_surface = {
            "strategy_name": plan.get("strategy_name"),
            "description": plan.get("description"),
            "slot_count": len(slots),
            "variable_count": len(variables),
            "expected_variable_count": plan.get("expected_variable_count"),
            "global_scale_range": round_nested(plan.get("global_scale_range")),
            "slot_dimension_scale_range": round_nested(plan.get("slot_dimension_scale_range")),
            "board_margin": round_nested(plan.get("board_margin")),
            "notes": plan.get("notes", []),
        }
        primitive_summary = (
            primitive_analysis.get("summary")
            if isinstance(primitive_analysis, dict) and isinstance(primitive_analysis.get("summary"), dict)
            else None
        )
        source = {
            "schema_version": "design_agent_bo_parameterization_summary_v1",
            "parameterization_schema_version": parameterization.get("schema_version"),
            "parameterization_backend": parameterization.get("backend"),
            "actual_backend": parameterization.get("actual_backend"),
            "primitive_analysis_summary": primitive_summary,
            "paths": {
                key: None if path is None else str(path.resolve())
                for key, path in (source_paths or {}).items()
            },
        }
        return BOParameterizationSummary(
            optimization_surface=optimization_surface,
            model=model,
            port=port,
            slots=slots,
            variables=variables,
            variable_groups=variable_groups,
            source=source,
        )


def build_bo_parameterization_summary(
    parameterization_json: Path | str,
    variable_plan_json: Optional[Path | str] = None,
    port_summary_json: Optional[Path | str] = None,
    primitive_analysis_json: Optional[Path | str] = None,
    variable_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience function returning a summary dictionary."""

    return BOParameterizationSummaryBuilder().from_files(
        parameterization_json=parameterization_json,
        variable_plan_json=variable_plan_json,
        port_summary_json=port_summary_json,
        primitive_analysis_json=primitive_analysis_json,
        variable_plan=variable_plan,
    ).to_dict()


def write_bo_parameterization_summary(
    parameterization_json: Path | str,
    output_path: Optional[Path | str] = None,
    variable_plan_json: Optional[Path | str] = None,
    port_summary_json: Optional[Path | str] = None,
    primitive_analysis_json: Optional[Path | str] = None,
    variable_plan: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``bo_parameterization_summary.json`` to the agent-input folder by default."""

    summary = build_bo_parameterization_summary(
        parameterization_json=parameterization_json,
        variable_plan_json=variable_plan_json,
        port_summary_json=port_summary_json,
        primitive_analysis_json=primitive_analysis_json,
        variable_plan=variable_plan,
    )
    path = Path(output_path) if output_path is not None else default_bo_parameterization_summary_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def default_bo_parameterization_summary_path() -> Path:
    """Return the shared BO parameterization summary path for LLM agents."""

    return AGENT_INPUTS_DIR / BO_PARAMETERIZATION_SUMMARY_FILENAME


def summarize_model(parameterization: Dict[str, Any], variable_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize BO model-level geometry and metrics."""

    canvas = parameterization.get("canvas") if isinstance(parameterization.get("canvas"), dict) else {}
    metrics = parameterization.get("metrics") if isinstance(parameterization.get("metrics"), dict) else {}
    board_bbox = bbox_from_value(variable_plan.get("board_bbox")) or canvas_bbox(canvas) or components_bbox(parameterization)
    return {
        "canvas": round_nested(
            {
                "width": number_or_none(canvas.get("width")),
                "height": number_or_none(canvas.get("height")),
                "unit": canvas.get("unit"),
                "source_origin": canvas.get("source_origin"),
            }
        ),
        "board_bbox": None if board_bbox is None else [round_float(value) for value in board_bbox],
        "component_count": int_or_none(metrics.get("component_count")) or count_components(parameterization),
        "primitive_count": int_or_none(metrics.get("primitive_count")) or count_primitives(parameterization),
        "hole_count": int_or_none(metrics.get("hole_count")) or count_holes(parameterization),
        "line_only_parameterization": bool(metrics.get("line_only_parameterization", False)),
    }


def summarize_port(port_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize the selected BO port."""

    if not isinstance(port_summary, dict) or not port_summary:
        return {"available": False}
    selected = port_summary.get("selected_port")
    if not isinstance(selected, dict):
        ports = port_summary.get("ports")
        selected = ports[0] if isinstance(ports, list) and ports and isinstance(ports[0], dict) else {}
    return round_nested(
        {
            "available": bool(selected),
            "direction": selected.get("direction") or selected.get("port_side") or selected.get("side"),
            "point": selected.get("point") or selected.get("center") or selected.get("raw_endpoint"),
            "width": selected.get("port_width") or selected.get("feed_width") or selected.get("width"),
            "source": selected.get("source") or port_summary.get("source"),
            "touches_border": selected.get("touches_border"),
            "connected_to_main_patch": selected.get("connected_to_main_patch"),
        }
    )


def summarize_slots(parameterization: Dict[str, Any], variable_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Summarize slots selected by the BO variable plan."""

    plan_slots = variable_plan.get("slots") if isinstance(variable_plan.get("slots"), list) else []
    if plan_slots:
        return [summarize_plan_slot(slot) for slot in plan_slots if isinstance(slot, dict)]
    slots: List[Dict[str, Any]] = []
    for component_index, component in enumerate(parameterization.get("components", []) or []):
        if not isinstance(component, dict):
            continue
        for hole_index, hole in enumerate(component.get("holes", []) or []):
            if isinstance(hole, dict):
                slot = {
                    "slot_id": hole.get("id") or "component_{0}_hole_{1}".format(component_index + 1, hole_index + 1),
                    "component_index": component_index,
                    "primitive_index": hole_index,
                    "bbox": hole.get("bbox"),
                    "center": center_from_bbox(bbox_from_value(hole.get("bbox"))),
                    "dimensions": dimensions_from_bbox(bbox_from_value(hole.get("bbox"))),
                    "source": "curve_parameterization",
                }
                slots.append(round_nested(slot))
    return slots


def summarize_plan_slot(slot: Dict[str, Any]) -> Dict[str, Any]:
    """Return the LLM-relevant fields for one BO slot."""

    bbox = bbox_from_value(slot.get("bbox"))
    parent_bbox = bbox_from_value(slot.get("parent_component_bbox"))
    return round_nested(
        {
            "slot_id": slot.get("slot_id"),
            "primitive_id": slot.get("primitive_id"),
            "component_index": slot.get("component_index"),
            "primitive_index": slot.get("primitive_index"),
            "source_key": slot.get("source_key"),
            "source": slot.get("source"),
            "bbox": None if bbox is None else list(bbox),
            "center": slot.get("center") or center_from_bbox(bbox),
            "dimensions": slot.get("dimensions") or dimensions_from_bbox(bbox),
            "parent_component_bbox": None if parent_bbox is None else list(parent_bbox),
        }
    )


def summarize_variables(variable_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Summarize BO variable bounds and intent."""

    variables: List[Dict[str, Any]] = []
    for variable in variable_plan.get("variables", []) or []:
        if not isinstance(variable, dict):
            continue
        variables.append(
            round_nested(
                {
                    "name": variable.get("name"),
                    "variable_type": variable.get("variable_type"),
                    "target": variable.get("target"),
                    "slot_id": variable.get("slot_id"),
                    "axis": variable.get("axis"),
                    "lower": variable.get("lower"),
                    "upper": variable.get("upper"),
                    "default": variable.get("default"),
                    "description": variable.get("description"),
                }
            )
        )
    return variables


def summarize_variable_groups(variables: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Group BO variables by target and variable type."""

    by_target: Dict[str, List[str]] = {}
    by_type: Dict[str, List[str]] = {}
    by_slot: Dict[str, List[str]] = {}
    for variable in variables:
        name = str(variable.get("name"))
        target = str(variable.get("target") or "unknown")
        variable_type = str(variable.get("variable_type") or "unknown")
        by_target.setdefault(target, []).append(name)
        by_type.setdefault(variable_type, []).append(name)
        slot_id = variable.get("slot_id")
        if slot_id is not None:
            by_slot.setdefault(str(slot_id), []).append(name)
    return {
        "by_target": by_target,
        "by_type": by_type,
        "by_slot": by_slot,
    }


def count_components(parameterization: Dict[str, Any]) -> int:
    """Return component count."""

    return len(parameterization.get("components", []) or [])


def count_holes(parameterization: Dict[str, Any]) -> int:
    """Return total hole count."""

    return sum(
        len(component.get("holes", []) or [])
        for component in parameterization.get("components", []) or []
        if isinstance(component, dict)
    )


def count_primitives(parameterization: Dict[str, Any]) -> int:
    """Return total primitive count."""

    total = 0
    for component in parameterization.get("components", []) or []:
        if not isinstance(component, dict):
            continue
        total += len(component.get("primitives", []) or component.get("segments", []) or [])
        for hole in component.get("holes", []) or []:
            if isinstance(hole, dict):
                total += len(hole.get("primitives", []) or hole.get("segments", []) or [])
    return total


def components_bbox(parameterization: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Infer bbox from all component points."""

    points: List[Point] = []
    for component in parameterization.get("components", []) or []:
        if isinstance(component, dict):
            points.extend(points_from_value(component.get("points") or component.get("fallback_points")))
    return points_bbox(points)


def canvas_bbox(canvas: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Return bbox from canvas dimensions."""

    width = number_or_none(canvas.get("width"))
    height = number_or_none(canvas.get("height"))
    if width is None or height is None or width <= 0.0 or height <= 0.0:
        return None
    return (0.0, 0.0, width, height)


def bbox_from_value(value: Any) -> Optional[Tuple[float, float, float, float]]:
    """Normalize a bbox-like list."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 4:
        return None
    numbers = [number_or_none(item) for item in value[:4]]
    if any(number is None for number in numbers):
        return None
    x0, y0, x1, y1 = [float(number) for number in numbers]
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def center_from_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[List[float]]:
    """Return bbox center."""

    if bbox is None:
        return None
    return [0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])]


def dimensions_from_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Dict[str, Any]]:
    """Return length/width dimensions from bbox."""

    if bbox is None:
        return None
    x_size = max(1e-9, bbox[2] - bbox[0])
    y_size = max(1e-9, bbox[3] - bbox[1])
    if x_size >= y_size:
        return {"length": x_size, "width": y_size, "length_axis": "x", "width_axis": "y"}
    return {"length": y_size, "width": x_size, "length_axis": "y", "width_axis": "x"}


def points_from_value(value: Any) -> List[Point]:
    """Extract point tuples from a JSON list."""

    if not isinstance(value, list):
        return []
    points: List[Point] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 2:
            continue
        x = number_or_none(item[0])
        y = number_or_none(item[1])
        if x is not None and y is not None:
            points.append((x, y))
    return points


def points_bbox(points: Sequence[Point]) -> Optional[Tuple[float, float, float, float]]:
    """Return bbox for points."""

    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def load_json_object(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object: {0}".format(path))
    return payload


def number_or_none(value: Any) -> Optional[float]:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def int_or_none(value: Any) -> Optional[int]:
    """Return an int or None."""

    number = number_or_none(value)
    return None if number is None else int(number)


def round_nested(value: Any) -> Any:
    """Round nested numeric values for clean JSON."""

    if isinstance(value, float):
        return round_float(value)
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    return value


def round_float(value: float, digits: int = 6) -> float:
    """Round numeric values for stable JSON."""

    return round(float(value), digits)


__all__ = [
    "BOParameterizationSummary",
    "BOParameterizationSummaryBuilder",
    "build_bo_parameterization_summary",
    "default_bo_parameterization_summary_path",
    "write_bo_parameterization_summary",
]
