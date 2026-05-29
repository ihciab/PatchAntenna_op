from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from constrained_shape_optimizer import build_control_point_optimization_plan
from deformation_engine import ControlPointDeformer


Point = Tuple[float, float]
PROJECT_ROOT = Path(__file__).resolve().parent
CONTROL_POINT_CONSTRAINTS_PATH = PROJECT_ROOT / "control_point_constraints.json"


@dataclass(frozen=True)
class DesignVariable:
    """优化安全设计变量。

    第一版只启用 topology-preserving 的全局等比例缩放。变量来源是
    line / arc / spline primitive 的几何包围盒，而不是 sampled points。
    """

    name: str
    lower: float
    upper: float
    default: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveInventory:
    line_count: int
    arc_count: int
    spline_count: int
    component_count: int
    bbox: Tuple[float, float, float, float]
    center: Point
    deformation_plan: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_design_variables(payload: Dict[str, Any]) -> Tuple[List[DesignVariable], PrimitiveInventory]:
    """从 primitive 层提取优化安全变量。

    重要原则：
    - 不把 resampled_points / sampled_points 当作独立优化变量。
    - 只从 line / arc / spline primitive 参数提取几何尺度。
    - 第一版使用全局等比例缩放，优先保证拓扑和 CST 可重建性。
    """

    primitives = list(iter_primitives(payload))
    line_count = sum(1 for primitive in primitives if primitive_kind(primitive) == "line")
    arc_count = sum(1 for primitive in primitives if primitive_kind(primitive) == "arc")
    spline_count = sum(1 for primitive in primitives if primitive_kind(primitive) == "spline")

    points = collect_primitive_points(primitives)
    if not points:
        points = collect_component_cache_points(payload)
    if not points:
        raise ValueError("无法从 curve_parameterization.json 提取任何几何点")

    bbox = point_bbox(points)
    cx = 0.5 * (bbox[0] + bbox[2])
    cy = 0.5 * (bbox[1] + bbox[3])

    deformation_plan = None
    try:
        cp_plan = build_control_point_optimization_plan(payload, CONTROL_POINT_CONSTRAINTS_PATH)
        if cp_plan.enabled and cp_plan.variables:
            variables = [
                DesignVariable(
                    name=variable.name,
                    lower=variable.lower,
                    upper=variable.upper,
                    default=variable.default,
                    description=variable.description,
                )
                for variable in cp_plan.variables
            ]
            deformation_plan = cp_plan.to_dict()
            inventory = PrimitiveInventory(
                line_count=line_count,
                arc_count=arc_count,
                spline_count=spline_count,
                component_count=len(payload.get("components", []) or []),
                bbox=bbox,
                center=(cx, cy),
                deformation_plan=deformation_plan,
            )
            return variables, inventory
    except Exception:
        deformation_plan = None

    variables = [
        DesignVariable(
            name="global_scale",
            lower=0.92,
            upper=1.08,
            default=1.0,
            description=(
                "Topology-preserving uniform scale extracted from line/arc/spline primitive bbox. "
                "Cached reconstruction points are updated only as dependent geometry."
            ),
        )
    ]

    inventory = PrimitiveInventory(
        line_count=line_count,
        arc_count=arc_count,
        spline_count=spline_count,
        component_count=len(payload.get("components", []) or []),
        bbox=bbox,
        center=(cx, cy),
        deformation_plan=deformation_plan,
    )
    return variables, inventory


def mutate_geometry(
    payload: Dict[str, Any],
    variable_values: Dict[str, float],
    inventory: Optional[PrimitiveInventory] = None,
) -> Dict[str, Any]:
    """按优化变量生成新的几何 JSON。

    本函数会同步更新 primitive 参数和 CST builder 使用的缓存点列。
    这些点列不是优化变量，只是由 primitive-level 变换派生出的重建缓存。
    """

    mutated = copy.deepcopy(payload)
    if inventory is None:
        _, inventory = extract_design_variables(payload)

    if inventory.deformation_plan:
        cp_plan = build_control_point_optimization_plan(payload, CONTROL_POINT_CONSTRAINTS_PATH)
        deformer = ControlPointDeformer(cp_plan)
        mutated, deformation_report = deformer.apply_offsets(mutated, variable_values, run_validation=False)
        metadata = mutated.setdefault("optimization_metadata", {})
        metadata["mutation"] = {
            "variables": dict(variable_values),
            "strategy": "constrained_control_point_offsets",
            "raw_sampled_points_optimized": False,
            "geometry_robustness_score": deformation_report.geometry_robustness_score,
            "deformation_validation": deformation_report.to_dict(),
        }
        return mutated

    scale = float(variable_values.get("global_scale", 1.0))
    center = inventory.center

    def transform_point(point: Sequence[Any]) -> List[float]:
        x = float(point[0])
        y = float(point[1])
        return [
            center[0] + (x - center[0]) * scale,
            center[1] + (y - center[1]) * scale,
        ]

    _transform_payload_coordinates(mutated, transform_point, scale)
    metadata = mutated.setdefault("optimization_metadata", {})
    metadata["mutation"] = {
        "variables": dict(variable_values),
        "strategy": "primitive_safe_uniform_scale",
        "raw_sampled_points_optimized": False,
    }
    return mutated


def iter_primitives(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for component in payload.get("components", []) or []:
        for primitive in component.get("primitives", []) or []:
            if isinstance(primitive, dict):
                yield primitive
        for segment in component.get("segments", []) or []:
            if isinstance(segment, dict):
                yield segment


def primitive_kind(primitive: Dict[str, Any]) -> str:
    kind = str(
        primitive.get("type")
        or primitive.get("kind")
        or primitive.get("primitive_type")
        or "spline"
    ).lower()
    if "line" in kind:
        return "line"
    if "arc" in kind:
        return "arc"
    return "spline"


def collect_primitive_points(primitives: Iterable[Dict[str, Any]]) -> List[Point]:
    points: List[Point] = []
    for primitive in primitives:
        for key in ("start", "end", "center"):
            point = primitive.get(key)
            if is_point(point):
                points.append((float(point[0]), float(point[1])))
        for key in ("control_points", "points", "fallback_points"):
            points.extend(parse_points(primitive.get(key)))
        params = primitive.get("parameters")
        if isinstance(params, dict):
            for key in ("start", "end", "center"):
                point = params.get(key)
                if is_point(point):
                    points.append((float(point[0]), float(point[1])))
            for key in ("control_points", "points"):
                points.extend(parse_points(params.get(key)))
    return points


def collect_component_cache_points(payload: Dict[str, Any]) -> List[Point]:
    points: List[Point] = []
    for component in payload.get("components", []) or []:
        for key in ("resampled_points", "fallback_points", "sampled_points", "points"):
            points.extend(parse_points(component.get(key)))
    return points


def parse_points(value: Any) -> List[Point]:
    if not isinstance(value, list):
        return []
    points: List[Point] = []
    for item in value:
        if is_point(item):
            points.append((float(item[0]), float(item[1])))
    return points


def is_point(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y)


def point_bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _transform_payload_coordinates(
    value: Any,
    transform_point: Any,
    uniform_scale: float,
    parent_key: str = "",
) -> Any:
    point_keys = {
        "start",
        "end",
        "center",
        "point",
        "origin",
    }
    point_list_keys = {
        "points",
        "control_points",
        "fallback_points",
        "resampled_points",
        "sampled_points",
        "smoothed_points",
    }
    skip_keys = {
        "direction",
        "direction_histogram",
        "source_refs",
        "source",
        "features",
        "metrics",
    }

    if isinstance(value, dict):
        if {"x", "y"}.issubset(value.keys()) and _is_number(value.get("x")) and _is_number(value.get("y")):
            x, y = transform_point([value["x"], value["y"]])
            value["x"] = x
            value["y"] = y

        for key, child in list(value.items()):
            if key in skip_keys:
                continue
            if key == "bbox" and isinstance(child, list) and len(child) >= 4:
                value[key] = _transform_bbox(child, transform_point)
            elif key == "radius" and _is_number(child):
                value[key] = float(child) * uniform_scale
            elif key in point_keys and is_point(child):
                value[key] = transform_point(child)
            elif key in point_list_keys and isinstance(child, list):
                value[key] = [
                    transform_point(item) if is_point(item) else item
                    for item in child
                ]
            else:
                _transform_payload_coordinates(child, transform_point, uniform_scale, key)
    elif isinstance(value, list):
        for item in value:
            _transform_payload_coordinates(item, transform_point, uniform_scale, parent_key)
    return value


def _transform_bbox(value: Sequence[Any], transform_point: Any) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in value[:4]]
    corners = [
        transform_point([x1, y1]),
        transform_point([x1, y2]),
        transform_point([x2, y1]),
        transform_point([x2, y2]),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
