"""Design-agent adapter around the existing Bayesian optimization pipeline.

The existing ``bayesian_optimization`` package remains unchanged.  This adapter
lets the design-agent run use its own variable selection plan while mapping the
sampled values back to variable names already understood by the legacy mutator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bayesian_optimization.geometry.primitive_mutator import DesignVariable
from bayesian_optimization.pipelines.optimization_pipeline import (
    EvaluationRecord,
    OptimizationConfig,
    OptimizationPipeline,
    OptimizationStage,
    write_json,
)
from design_agent.tools.experiment1_objective import evaluate_experiment1_objective_from_files


@dataclass(frozen=True)
class VariableMapping:
    """Mapping from design-agent BO variable to legacy mutator variable."""

    external_name: str
    internal_name: str
    transform: str
    offset: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "external_name": self.external_name,
            "internal_name": self.internal_name,
            "transform": self.transform,
            "offset": self.offset,
        }


class DesignAgentOptimizationPipeline(OptimizationPipeline):
    """OptimizationPipeline variant using design-agent variable selection."""

    def __init__(self, config: OptimizationConfig, variable_plan: Dict[str, Any]) -> None:
        self.design_agent_variable_plan = variable_plan
        self.design_agent_variable_mappings: List[VariableMapping] = []
        super().__init__(config)
        self.original_pipeline_variables = list(self.variables)
        self.variables, self.design_agent_variable_mappings = self._variables_and_mappings_from_plan(variable_plan)

    def evaluate(
        self,
        evaluation: int,
        variables: Dict[str, float],
        output_dir_name: Optional[str] = None,
        append_history: bool = True,
        stage: Optional[OptimizationStage] = None,
    ) -> EvaluationRecord:
        external_variables = {key: float(value) for key, value in variables.items()}
        internal_variables = self._map_external_variables_to_internal(external_variables)
        record = super().evaluate(
            evaluation=evaluation,
            variables=internal_variables,
            output_dir_name=output_dir_name,
            append_history=append_history,
            stage=stage,
        )
        record.variables = external_variables
        eval_dir = self.state.valid_designs_dir / (output_dir_name or f"eval_{evaluation:03d}")
        if eval_dir.exists():
            write_json(
                eval_dir / "design_agent_variable_mapping.json",
                {
                    "external_variables": external_variables,
                    "internal_variables": internal_variables,
                    "mappings": [mapping.to_dict() for mapping in self.design_agent_variable_mappings],
                },
            )
        self._apply_design_agent_objective(record, eval_dir)
        return record

    def _write_run_metadata(self) -> None:
        super()._write_run_metadata()
        metadata_path = self.state.run_dir / "run_metadata.json"
        payload = {}
        if metadata_path.exists():
            import json

            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {}
        payload["design_agent_variable_plan"] = self.design_agent_variable_plan
        payload["design_agent_variable_mappings"] = [
            mapping.to_dict() for mapping in self.design_agent_variable_mappings
        ]
        payload["original_pipeline_design_variables"] = [
            variable.to_dict() for variable in self.original_pipeline_variables
        ]
        write_json(metadata_path, payload)

    def _variables_and_mappings_from_plan(
        self,
        variable_plan: Dict[str, Any],
    ) -> Tuple[List[DesignVariable], List[VariableMapping]]:
        variables: List[DesignVariable] = []
        mappings: List[VariableMapping] = []
        slots_by_id = {
            str(slot.get("slot_id")): slot
            for slot in variable_plan.get("slots", []) or []
            if isinstance(slot, dict) and slot.get("slot_id") is not None
        }

        for item in variable_plan.get("variables", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name"))
            variable_type = str(item.get("variable_type"))
            variables.append(
                DesignVariable(
                    name=name,
                    lower=float(item.get("lower")),
                    upper=float(item.get("upper")),
                    default=float(item.get("default")),
                    description=str(item.get("description", "")),
                )
            )
            mappings.append(self._mapping_for_variable(item, slots_by_id))
        if not variables:
            raise ValueError("Design-agent variable plan did not produce any BO variables.")
        return variables, mappings

    def _mapping_for_variable(
        self,
        variable: Dict[str, Any],
        slots_by_id: Dict[str, Dict[str, Any]],
    ) -> VariableMapping:
        name = str(variable.get("name"))
        variable_type = str(variable.get("variable_type"))
        slot_id = str(variable.get("slot_id")) if variable.get("slot_id") is not None else ""

        if variable_type in {"global_scale_x", "global_scale_y"}:
            return VariableMapping(name, variable_type, "identity")
        if variable_type == "slot_dx":
            return VariableMapping(name, f"{slot_id}_slot_translate_x", "identity")
        if variable_type == "slot_dy":
            return VariableMapping(name, f"{slot_id}_slot_translate_y", "identity")
        if variable_type in {"slot_length", "slot_width"}:
            slot = slots_by_id.get(slot_id, {})
            dimensions = slot.get("dimensions") if isinstance(slot.get("dimensions"), dict) else {}
            axis = str(variable.get("axis") or dimensions.get("length_axis" if variable_type == "slot_length" else "width_axis"))
            baseline = float(dimensions.get("length" if variable_type == "slot_length" else "width", variable.get("default")))
            internal_name = f"{slot_id}_slot_width_delta" if axis == "x" else f"{slot_id}_slot_height_delta"
            return VariableMapping(name, internal_name, "absolute_to_scaled_delta", offset=baseline)
        raise ValueError(f"Unsupported design-agent BO variable type: {variable_type}")

    def _map_external_variables_to_internal(self, variables: Dict[str, float]) -> Dict[str, float]:
        internal: Dict[str, float] = {
            "global_scale_x": float(variables.get("global_scale_x", 1.0)),
            "global_scale_y": float(variables.get("global_scale_y", 1.0)),
        }
        self._map_slot_variables_to_internal(variables, internal)
        return internal

    def _map_slot_variables_to_internal(self, variables: Dict[str, float], internal: Dict[str, float]) -> None:
        slots = self.design_agent_variable_plan.get("slots", []) or []
        variable_defs = self.design_agent_variable_plan.get("variables", []) or []
        variables_by_type = {
            (str(item.get("slot_id")), str(item.get("variable_type"))): item
            for item in variable_defs
            if isinstance(item, dict) and item.get("slot_id") is not None
        }
        scale_x = float(internal.get("global_scale_x", 1.0))
        scale_y = float(internal.get("global_scale_y", 1.0))
        scale_center = self._global_scale_center()
        for slot in slots:
            if not isinstance(slot, dict) or slot.get("slot_id") is None:
                continue
            slot_id = str(slot["slot_id"])
            bbox = self._bbox_from_slot(slot, "bbox")
            parent_bbox = self._bbox_from_slot(slot, "parent_component_bbox")
            dimensions = slot.get("dimensions") if isinstance(slot.get("dimensions"), dict) else {}
            if bbox is None or parent_bbox is None:
                self._map_slot_variables_without_clamp(slot_id, variables, internal)
                continue

            scaled_slot_bbox = self._scale_bbox(bbox, scale_center, scale_x, scale_y)
            scaled_parent_bbox = self._scale_bbox(parent_bbox, scale_center, scale_x, scale_y)
            scaled_center = self._bbox_center(scaled_slot_bbox)
            current_x_size = max(1e-9, scaled_slot_bbox[2] - scaled_slot_bbox[0])
            current_y_size = max(1e-9, scaled_slot_bbox[3] - scaled_slot_bbox[1])

            length_default = float(dimensions.get("length", max(current_x_size, current_y_size)))
            width_default = float(dimensions.get("width", min(current_x_size, current_y_size)))
            length = self._external_value(variables, variables_by_type, slot_id, "slot_length", length_default)
            width = self._external_value(variables, variables_by_type, slot_id, "slot_width", width_default)
            length_axis = str(dimensions.get("length_axis", "x"))
            width_axis = str(dimensions.get("width_axis", "y"))
            desired_x_size = float(length if length_axis == "x" else width)
            desired_y_size = float(width if width_axis == "y" else length)

            raw_dx = self._external_value(variables, variables_by_type, slot_id, "slot_dx", 0.0)
            raw_dy = self._external_value(variables, variables_by_type, slot_id, "slot_dy", 0.0)
            dx = self._clamp_translation(raw_dx, scaled_center[0], desired_x_size, scaled_parent_bbox[0], scaled_parent_bbox[2])
            dy = self._clamp_translation(raw_dy, scaled_center[1], desired_y_size, scaled_parent_bbox[1], scaled_parent_bbox[3])

            internal[f"{slot_id}_slot_translate_x"] = dx
            internal[f"{slot_id}_slot_translate_y"] = dy
            internal[f"{slot_id}_slot_width_delta"] = desired_x_size - current_x_size
            internal[f"{slot_id}_slot_height_delta"] = desired_y_size - current_y_size

    def _map_slot_variables_without_clamp(
        self,
        slot_id: str,
        variables: Dict[str, float],
        internal: Dict[str, float],
    ) -> None:
        mapping_by_name = {mapping.external_name: mapping for mapping in self.design_agent_variable_mappings}
        for name, value in variables.items():
            mapping = mapping_by_name.get(name)
            if mapping is None or not mapping.internal_name.startswith(f"{slot_id}_"):
                continue
            if mapping.transform == "identity":
                internal[mapping.internal_name] = float(value)
            elif mapping.transform == "absolute_to_delta":
                internal[mapping.internal_name] = float(value) - float(mapping.offset)
            elif mapping.transform == "absolute_to_scaled_delta":
                internal[mapping.internal_name] = float(value) - float(mapping.offset)
            else:
                raise ValueError(f"Unsupported variable transform: {mapping.transform}")

    def _external_value(
        self,
        variables: Dict[str, float],
        variables_by_type: Dict[Tuple[str, str], Dict[str, Any]],
        slot_id: str,
        variable_type: str,
        default: float,
    ) -> float:
        variable_def = variables_by_type.get((slot_id, variable_type))
        if not variable_def:
            return float(default)
        return float(variables.get(str(variable_def.get("name")), variable_def.get("default", default)))

    def _global_scale_center(self) -> Tuple[float, float]:
        inventory = getattr(self, "inventory", None)
        center = getattr(inventory, "center", None)
        if isinstance(center, tuple) and len(center) >= 2:
            return (float(center[0]), float(center[1]))
        if isinstance(center, list) and len(center) >= 2:
            return (float(center[0]), float(center[1]))
        plan_bbox = self.design_agent_variable_plan.get("board_bbox")
        bbox = self._bbox_from_value(plan_bbox)
        return self._bbox_center(bbox) if bbox else (0.0, 0.0)

    def _bbox_from_slot(self, slot: Dict[str, Any], key: str) -> Optional[Tuple[float, float, float, float]]:
        return self._bbox_from_value(slot.get(key))

    @staticmethod
    def _bbox_from_value(value: Any) -> Optional[Tuple[float, float, float, float]]:
        if not isinstance(value, list) or len(value) < 4:
            return None
        try:
            x0, y0, x1, y1 = [float(item) for item in value[:4]]
        except (TypeError, ValueError):
            return None
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @staticmethod
    def _scale_bbox(
        bbox: Tuple[float, float, float, float],
        center: Tuple[float, float],
        scale_x: float,
        scale_y: float,
    ) -> Tuple[float, float, float, float]:
        points = [
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
        ]
        scaled = [
            (
                center[0] + (point[0] - center[0]) * scale_x,
                center[1] + (point[1] - center[1]) * scale_y,
            )
            for point in points
        ]
        xs = [point[0] for point in scaled]
        ys = [point[1] for point in scaled]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    @staticmethod
    def _clamp_translation(value: float, center: float, size: float, lower: float, upper: float) -> float:
        half = 0.5 * max(1e-9, float(size))
        min_delta = float(lower) + half - float(center)
        max_delta = float(upper) - half - float(center)
        if min_delta > max_delta:
            midpoint = 0.5 * (min_delta + max_delta)
            return midpoint
        return max(min_delta, min(float(value), max_delta))

    def _apply_design_agent_objective(self, record: EvaluationRecord, eval_dir: Path) -> None:
        s11_metrics = record.s11_metrics if isinstance(record.s11_metrics, dict) else {}
        s11_path = s11_metrics.get("s11_path")
        if not s11_path:
            return
        try:
            result = evaluate_experiment1_objective_from_files(s11_path)
        except Exception as exc:
            self.state.logger.warning("design-agent Experiment 1 objective skipped: %s", exc)
            return
        breakdown = result.to_dict()
        breakdown["objective_source"] = (
            "design_agent.tools.experiment1_objective.evaluate_experiment1_objective_from_files"
        )
        record.objective = float(result.loss)
        record.objective_breakdown = breakdown
        if eval_dir.exists():
            write_json(eval_dir / "objective_breakdown.json", breakdown)
            write_json(eval_dir / "design_agent_objective_breakdown.json", breakdown)


__all__ = ["DesignAgentOptimizationPipeline", "VariableMapping"]
