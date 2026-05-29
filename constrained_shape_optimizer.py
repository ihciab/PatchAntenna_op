from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class ControlPointConstraint:
    point_id: str
    component_index: int
    point_index: int
    original_point: Point2D
    movable: bool
    dx_range: Tuple[float, float]
    dy_range: Tuple[float, float]
    priority: str = "medium"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlPointDesignVariable:
    name: str
    lower: float
    upper: float
    default: float
    point_id: str
    axis: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlPointOptimizationPlan:
    constraints: List[ControlPointConstraint]
    variables: List[ControlPointDesignVariable]
    source: str
    enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "variables": [variable.to_dict() for variable in self.variables],
            "source": self.source,
            "enabled": self.enabled,
        }


def load_constraint_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return default_constraint_config()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"control point constraint config must be a JSON object: {path}")
    return {**default_constraint_config(), **data}


def default_constraint_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "auto_select": True,
        "max_movable_points": 8,
        "max_dimensions": 20,
        "default_dx_range": [-1.0, 1.0],
        "default_dy_range": [-1.0, 1.0],
        "freeze_topology_endpoints": True,
        "selection_strategy": {
            "prefer_high_curvature": True,
            "prefer_internal_points": True,
            "min_endpoint_distance_index": 2,
        },
        "point_constraints": {},
    }


def build_control_point_optimization_plan(
    payload: Dict[str, Any],
    constraint_path: Path,
) -> ControlPointOptimizationPlan:
    config = load_constraint_config(constraint_path)
    if not bool(config.get("enabled", True)):
        return ControlPointOptimizationPlan([], [], str(constraint_path), enabled=False)

    explicit_constraints = config.get("point_constraints") or {}
    candidates = extract_candidate_control_points(payload, config)
    constraints = apply_constraint_overrides(candidates, explicit_constraints, config)

    movable_constraints = [constraint for constraint in constraints if constraint.movable]
    max_points = int(config.get("max_movable_points", 8))
    max_dimensions = int(config.get("max_dimensions", 20))
    movable_constraints = movable_constraints[: max(0, min(max_points, max_dimensions // 2))]

    variables: List[ControlPointDesignVariable] = []
    for constraint in movable_constraints:
        variables.append(
            ControlPointDesignVariable(
                name=f"{constraint.point_id}_dx",
                lower=float(constraint.dx_range[0]),
                upper=float(constraint.dx_range[1]),
                default=0.0,
                point_id=constraint.point_id,
                axis="dx",
                description=f"Constrained local x-offset for {constraint.point_id}",
            )
        )
        variables.append(
            ControlPointDesignVariable(
                name=f"{constraint.point_id}_dy",
                lower=float(constraint.dy_range[0]),
                upper=float(constraint.dy_range[1]),
                default=0.0,
                point_id=constraint.point_id,
                axis="dy",
                description=f"Constrained local y-offset for {constraint.point_id}",
            )
        )

    return ControlPointOptimizationPlan(
        constraints=movable_constraints,
        variables=variables,
        source=str(constraint_path),
        enabled=bool(variables),
    )


def extract_candidate_control_points(payload: Dict[str, Any], config: Dict[str, Any]) -> List[ControlPointConstraint]:
    default_dx = tuple(float(v) for v in config.get("default_dx_range", [-1.0, 1.0]))
    default_dy = tuple(float(v) for v in config.get("default_dy_range", [-1.0, 1.0]))
    min_endpoint_distance = int((config.get("selection_strategy") or {}).get("min_endpoint_distance_index", 2))
    freeze_endpoints = bool(config.get("freeze_topology_endpoints", True))

    candidates: List[Tuple[float, ControlPointConstraint]] = []
    for component_index, component in enumerate(payload.get("components", []) or []):
        points = component_points(component)
        if len(points) < 5:
            continue
        for point_index in range(1, len(points) - 1):
            point_id = make_point_id(component_index, point_index)
            is_near_endpoint = point_index < min_endpoint_distance or point_index >= len(points) - min_endpoint_distance
            if freeze_endpoints and is_near_endpoint:
                continue

            curvature = local_curvature_score(points, point_index)
            if curvature <= 1e-6:
                continue
            priority = "high" if curvature > 0.50 else "medium"
            candidates.append(
                (
                    curvature,
                    ControlPointConstraint(
                        point_id=point_id,
                        component_index=component_index,
                        point_index=point_index,
                        original_point=points[point_index],
                        movable=True,
                        dx_range=(default_dx[0], default_dx[1]),
                        dy_range=(default_dy[0], default_dy[1]),
                        priority=priority,
                        reason=f"auto high-curvature point, curvature={curvature:.6f}",
                    ),
                )
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [constraint for _, constraint in candidates]


def apply_constraint_overrides(
    candidates: List[ControlPointConstraint],
    explicit_constraints: Dict[str, Any],
    config: Dict[str, Any],
) -> List[ControlPointConstraint]:
    by_id = {candidate.point_id: candidate for candidate in candidates}
    default_dx = tuple(float(v) for v in config.get("default_dx_range", [-1.0, 1.0]))
    default_dy = tuple(float(v) for v in config.get("default_dy_range", [-1.0, 1.0]))

    for point_id, override in explicit_constraints.items():
        if not isinstance(override, dict):
            continue
        base = by_id.get(point_id)
        if base is None:
            parsed = parse_point_id(point_id)
            if parsed is None:
                continue
            component_index, point_index = parsed
            base = ControlPointConstraint(
                point_id=point_id,
                component_index=component_index,
                point_index=point_index,
                original_point=(0.0, 0.0),
                movable=False,
                dx_range=default_dx,
                dy_range=default_dy,
                priority="manual",
                reason="manual override placeholder",
            )
        by_id[point_id] = ControlPointConstraint(
            point_id=point_id,
            component_index=base.component_index,
            point_index=base.point_index,
            original_point=base.original_point,
            movable=bool(override.get("movable", base.movable)),
            dx_range=tuple(float(v) for v in override.get("dx_range", base.dx_range)),
            dy_range=tuple(float(v) for v in override.get("dy_range", base.dy_range)),
            priority=str(override.get("priority", base.priority)),
            reason=str(override.get("reason", base.reason or "manual override")),
        )

    return [constraint for constraint in by_id.values() if constraint.movable]


def make_point_id(component_index: int, point_index: int) -> str:
    return f"c{component_index:03d}_p{point_index:03d}"


def parse_point_id(point_id: str) -> Optional[Tuple[int, int]]:
    try:
        c_part, p_part = point_id.split("_")
        return int(c_part[1:]), int(p_part[1:])
    except Exception:
        return None


def component_points(component: Dict[str, Any]) -> List[Point2D]:
    raw_points = component.get("resampled_points") or component.get("fallback_points") or component.get("points") or []
    points: List[Point2D] = []
    for point in raw_points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def local_curvature_score(points: Sequence[Point2D], index: int) -> float:
    a = points[index - 1]
    b = points[index]
    c = points[index + 1]
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return abs(math.acos(dot))
