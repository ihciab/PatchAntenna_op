from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from constrained_shape_optimizer import ControlPointOptimizationPlan
from geometry_constraint_validator import GeometryConstraintValidator, GeometryConstraintReport


@dataclass
class DeformationRecord:
    iteration: int
    moved_points: List[str]
    offsets: Dict[str, float]
    validation_result: Dict[str, Any]
    repair_actions: List[str]
    objective_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ControlPointDeformer:
    """受约束控制点位移变形器。

    只应用 P' = P + delta 的局部形变，不生成新拓扑。
    """

    def __init__(
        self,
        plan: ControlPointOptimizationPlan,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.plan = plan
        self.logger = logger or logging.getLogger(__name__)
        self.validator = GeometryConstraintValidator(logger=self.logger)
        self.history: List[DeformationRecord] = []

    def apply_offsets(
        self,
        payload: Dict[str, Any],
        offsets: Dict[str, float],
        iteration: int = 0,
        output_dir: Optional[Path] = None,
        objective_value: Optional[float] = None,
        run_validation: bool = True,
    ) -> Tuple[Dict[str, Any], GeometryConstraintReport]:
        deformed = copy.deepcopy(payload)
        moved_points: List[str] = []

        for constraint in self.plan.constraints:
            dx = clamp(
                float(offsets.get(f"{constraint.point_id}_dx", 0.0)),
                constraint.dx_range[0],
                constraint.dx_range[1],
            )
            dy = clamp(
                float(offsets.get(f"{constraint.point_id}_dy", 0.0)),
                constraint.dy_range[0],
                constraint.dy_range[1],
            )
            if abs(dx) <= 1e-15 and abs(dy) <= 1e-15:
                continue
            self._move_control_point(deformed, constraint.component_index, constraint.point_index, dx, dy)
            moved_points.append(constraint.point_id)

        metadata = deformed.setdefault("optimization_metadata", {})
        metadata["control_point_deformation"] = {
            "enabled": True,
            "moved_points": moved_points,
            "offsets": dict(offsets),
            "topology_preserving": True,
        }

        if run_validation:
            repaired, report = self.repair_geometry(deformed, output_dir=output_dir)
        else:
            repaired = deformed
            report = GeometryConstraintReport(
                valid=True,
                repair_applied=False,
                repair_operations=[],
                errors=[],
                warnings=["validation deferred until CST handoff geometry is built"],
                geometry_robustness_score=1.0,
                base_validation={},
                curvature_max_angle_deg=0.0,
                minimum_gap=None,
            )
        self._record_history(iteration, moved_points, offsets, report, objective_value, output_dir)
        return repaired, report

    def validate_deformation(
        self,
        payload: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> GeometryConstraintReport:
        return self.validator.validate(payload, output_dir=output_dir)

    def repair_geometry(
        self,
        payload: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], GeometryConstraintReport]:
        return self.validator.validate_and_repair(payload, output_dir=output_dir)

    def _move_control_point(
        self,
        payload: Dict[str, Any],
        component_index: int,
        point_index: int,
        dx: float,
        dy: float,
    ) -> None:
        components = payload.get("components", []) or []
        if component_index >= len(components):
            return
        component = components[component_index]

        for key in ("resampled_points", "fallback_points", "points"):
            points = component.get(key)
            if isinstance(points, list) and 0 <= point_index < len(points):
                points[point_index] = move_json_point(points[point_index], dx, dy)

        self._move_matching_primitive_points(component, point_index, dx, dy)

    def _move_matching_primitive_points(
        self,
        component: Dict[str, Any],
        point_index: int,
        dx: float,
        dy: float,
    ) -> None:
        # Line / arc / spline primitive 数据也同步保守更新；CST handoff 仍以缓存点列为主。
        for primitive in (component.get("primitives") or []) + (component.get("segments") or []):
            for key in ("points", "control_points", "fallback_points"):
                points = primitive.get(key)
                if isinstance(points, list) and 0 <= point_index < len(points):
                    points[point_index] = move_json_point(points[point_index], dx, dy)
            for key in ("start", "end"):
                value = primitive.get(key)
                if isinstance(value, list):
                    points = component.get("resampled_points") or component.get("fallback_points") or []
                    if point_index == 0 and key == "start":
                        primitive[key] = move_json_point(value, dx, dy)
                    elif points and point_index == len(points) - 1 and key == "end":
                        primitive[key] = move_json_point(value, dx, dy)

    def _record_history(
        self,
        iteration: int,
        moved_points: List[str],
        offsets: Dict[str, float],
        report: GeometryConstraintReport,
        objective_value: Optional[float],
        output_dir: Optional[Path],
    ) -> None:
        record = DeformationRecord(
            iteration=iteration,
            moved_points=moved_points,
            offsets=dict(offsets),
            validation_result=report.to_dict(),
            repair_actions=report.repair_operations,
            objective_value=objective_value,
        )
        self.history.append(record)
        if output_dir is not None:
            path = Path(output_dir) / "deformation_history.json"
            with path.open("w", encoding="utf-8") as file:
                json.dump([item.to_dict() for item in self.history], file, ensure_ascii=False, indent=2)


def move_json_point(point: Any, dx: float, dy: float) -> Any:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return point
    moved = list(point)
    moved[0] = float(moved[0]) + dx
    moved[1] = float(moved[1]) + dy
    return moved


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
