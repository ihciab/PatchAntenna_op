from __future__ import annotations

"""优化变量提取与几何变异适配层。

本文件是 optimization_pipeline 与具体形变实现之间的桥：
1. 优先调用 constrained_shape_optimizer 生成控制点 offset 变量；
2. 如果控制点计划不可用，则保留旧的 global_scale 兜底变量；
3. 根据 BO 给出的变量值生成 mutated geometry JSON。

注意：这里不改 parameterization schema，不调用 CST，不计算 objective。
"""

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesian_optimization.geometry.feature_shape_optimizer import FeatureConstrainedDeformer, build_feature_shape_plan
from bayesian_optimization.geometry.junction_constraint_manager import (
    build_junction_graph,
    plot_junction_debug,
    synchronize_junctions,
    validate_junctions,
    write_junction_validation,
)
from bayesian_optimization.geometry.primitive_analyzer import (
    DEFAULT_CURVE_PARAMETERIZATION_MODE,
    analyze_primitives,
)
from bayesian_optimization.geometry.primitive_variable_generator import generate_primitive_variables
from bayesian_optimization.geometry.shape_regularizer import (
    plot_curvature_change_map,
    validate_shape_quality,
    write_shape_quality_report,
)
from bayesian_optimization.geometry.spline_deformation import smooth_control_cage


Point = Tuple[float, float]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROL_POINT_CONSTRAINTS_PATH = PROJECT_ROOT / "control_point_constraints.json"
FEATURE_SHAPE_CONSTRAINTS_PATH = CONTROL_POINT_CONSTRAINTS_PATH


# =============================================================================
# 数据结构：BO 变量与 primitive inventory
# =============================================================================


@dataclass(frozen=True)
class DesignVariable:
    """BO 后端可直接采样的一维变量描述。"""
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
        """转换为 JSON 友好的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveInventory:
    """当前几何的 primitive 统计信息与可选控制点计划。"""
    line_count: int
    arc_count: int
    spline_count: int
    component_count: int
    bbox: Tuple[float, float, float, float]
    center: Point
    deformation_plan: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 友好的字典。"""
        return asdict(self)


def extract_design_variables(
    payload: Dict[str, Any],
    port_summary: Optional[Dict[str, Any]] = None,
    curve_parameterization_mode: str = DEFAULT_CURVE_PARAMETERIZATION_MODE,
) -> Tuple[List[DesignVariable], PrimitiveInventory]:
    # 【关键函数】优先提取控制点 offset 变量；失败时回退到 global_scale。
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
        primitive_analysis = analyze_primitives(
            payload,
            port_summary=port_summary,
            curve_parameterization_mode=curve_parameterization_mode,
        )
        primitive_variables = generate_primitive_variables(primitive_analysis)
        if primitive_variables:
            primitive_summary = primitive_analysis.get("summary", {}) or {}
            variables = [
                DesignVariable(
                    name=variable.name,
                    lower=variable.lower,
                    upper=variable.upper,
                    default=variable.default,
                    description=variable.description,
                )
                for variable in primitive_variables
            ]
            deformation_plan = {
                "mode": "primitive_aware_shape_optimization",
                "analysis": primitive_analysis,
                "variables": [variable.to_dict() for variable in primitive_variables],
                "all_parameterized_lines_optimized": True,
                "line_normal_offsets_only": False,
                "non_port_line_normal_offsets_enabled": True,
                "port_width_expansion_enabled": True,
                "port_propagation_shift_enabled": True,
                "raw_sampled_points_optimized": False,
                "single_control_point_offsets_enabled": False,
                "curve_parameterization_mode": primitive_summary.get("curve_parameterization_mode", curve_parameterization_mode),
            }
            inventory = PrimitiveInventory(
                line_count=int(primitive_summary.get("line_count", line_count)),
                arc_count=int(primitive_summary.get("curve_count", arc_count)),
                spline_count=int(primitive_summary.get("bspline_count", spline_count)),
                component_count=len(payload.get("components", []) or []),
                bbox=bbox,
                center=(cx, cy),
                deformation_plan=deformation_plan,
            )
            return variables, inventory
    except Exception:
        deformation_plan = None

    try:
        shape_plan = build_feature_shape_plan(payload, FEATURE_SHAPE_CONSTRAINTS_PATH, port_summary=port_summary)
        if shape_plan.enabled and shape_plan.variables:
            variables = [
                DesignVariable(
                    name=variable.name,
                    lower=variable.lower,
                    upper=variable.upper,
                    default=variable.default,
                    description=variable.description,
                )
                for variable in shape_plan.variables
            ]
            deformation_plan = shape_plan.to_dict()
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
    output_dir: Optional[Path] = None,
    iteration: int = 0,
    port_summary: Optional[Dict[str, Any]] = None,
    curve_parameterization_mode: str = DEFAULT_CURVE_PARAMETERIZATION_MODE,
) -> Dict[str, Any]:
    # 【关键函数】把 BO 采样值应用到几何 JSON，供后续 validation/CST 使用。
    """按优化变量生成新的几何 JSON。

    本函数会同步更新 primitive 参数和 CST builder 使用的缓存点列。
    这些点列不是优化变量，只是由 primitive-level 变换派生出的重建缓存。
    """

    mutated = copy.deepcopy(payload)
    if inventory is None:
        _, inventory = extract_design_variables(
            payload,
            port_summary=port_summary,
            curve_parameterization_mode=curve_parameterization_mode,
        )

    if inventory.deformation_plan and inventory.deformation_plan.get("mode") == "primitive_aware_shape_optimization":
        mutated, primitive_report = apply_primitive_aware_mutation(
            payload,
            mutated,
            variable_values,
            inventory.deformation_plan,
            output_dir=output_dir,
            iteration=iteration,
            port_summary=port_summary,
            curve_parameterization_mode=curve_parameterization_mode,
        )
        metadata = mutated.setdefault("optimization_metadata", {})
        metadata["mutation"] = {
            "variables": dict(variable_values),
            "strategy": "primitive_aware_feature_constrained_topology_preserving_shape_optimization",
            "raw_sampled_points_optimized": False,
            "single_control_point_offsets_enabled": False,
            "primitive_aware": True,
            "shape_quality_score": primitive_report.get("shape_quality_score", 0.0),
            "primitive_validation": primitive_report,
        }
        metadata["primitive_aware_mutation"] = primitive_report
        return mutated

    if inventory.deformation_plan:
        shape_plan = build_feature_shape_plan(payload, FEATURE_SHAPE_CONSTRAINTS_PATH, port_summary=port_summary)
        deformer = FeatureConstrainedDeformer(shape_plan)
        mutated, deformation_report = deformer.apply(
            mutated,
            variable_values,
            iteration=iteration,
            output_dir=output_dir,
        )
        metadata = mutated.setdefault("optimization_metadata", {})
        metadata["mutation"] = {
            "variables": dict(variable_values),
            "strategy": "feature_constrained_topology_preserving_shape_optimization",
            "raw_sampled_points_optimized": False,
            "single_control_point_offsets_enabled": False,
            "point_group_normal_offsets_enabled": True,
            "geometry_robustness_score": deformation_report.geometry_robustness_score,
            "deformation_validation": deformation_report.to_dict(),
        }
        return mutated

    scale = float(variable_values.get("global_scale", 1.0))
    center = inventory.center

    def transform_point(point: Sequence[Any]) -> List[float]:
        """围绕 inventory.center 做全局等比例缩放。"""
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


def apply_primitive_aware_mutation(
    original: Dict[str, Any],
    mutated: Dict[str, Any],
    variable_values: Dict[str, float],
    deformation_plan: Dict[str, Any],
    output_dir: Optional[Path] = None,
    iteration: int = 0,
    port_summary: Optional[Dict[str, Any]] = None,
    curve_parameterization_mode: str = DEFAULT_CURVE_PARAMETERIZATION_MODE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply primitive-aware mutation and engineering constraints.

    Input: original payload, mutable copied payload, BO values, primitive plan,
    optional output directory, and evaluation index.
    Output: mutated payload plus primitive mutation report.
    Algorithm purpose: run PrimitiveAnalyzer -> primitive variables -> mutation
    -> junction synchronization -> shape regularizer before existing validation.
    """

    output_path = Path(output_dir) if output_dir is not None else None
    plan_curve_mode = str(deformation_plan.get("curve_parameterization_mode") or curve_parameterization_mode)
    analysis = analyze_primitives(
        original,
        output_path,
        port_summary=port_summary,
        curve_parameterization_mode=plan_curve_mode,
    )
    variable_defs = deformation_plan.get("variables") or generate_primitive_variables(analysis)
    variables_by_name = {variable.get("name"): variable for variable in variable_defs}
    primitive_by_id = {primitive.get("primitive_id"): primitive for primitive in analysis.get("primitives", []) or []}
    mutation_report: Dict[str, Any] = {
        "iteration": int(iteration),
        "variables_used": [],
        "line_mutations": [],
        "port_mutations": [],
        "spline_mutations": [],
        "curve_mutations": [],
        "junction_fixed": [],
        "port_constraints": {},
        "shape_quality_score": 1.0,
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    before_for_debug = copy.deepcopy(mutated)

    for name, value in variable_values.items():
        variable = variables_by_name.get(name)
        if variable is None:
            continue
        sampled_value = float(value)
        if abs(sampled_value) <= 1e-15:
            continue
        mutation_report["variables_used"].append({"name": name, "value": sampled_value, "type": variable.get("variable_type")})
        variable_type = variable.get("variable_type")
        primitive_id = str(variable.get("primitive_id"))
        if variable_type == "parallel_line_spacing":
            apply_parallel_line_spacing(mutated, primitive_by_id, primitive_id.split("|"), sampled_value, mutation_report)
            continue
        primitive = primitive_by_id.get(primitive_id)
        if primitive is None:
            continue
        if variable_type in {"line_normal_offset", "line_length_delta", "feed_length"}:
            apply_line_mutation(mutated, primitive, variable_type, sampled_value, analysis, mutation_report)
        elif variable_type in {"port_width_delta", "port_propagation_shift"}:
            apply_port_mutation(mutated, primitive, variable_type, sampled_value, analysis, mutation_report)
        elif variable_type in {"spline_bulge", "spline_smooth_offset"}:
            apply_spline_mutation(mutated, primitive, sampled_value, mutation_report)
        elif variable_type in {"curve_smooth_offset", "primitive_smooth_offset"}:
            apply_curve_mutation(mutated, primitive, sampled_value, mutation_report)

    sync_closed_component_point_sequences(mutated)
    update_component_bboxes(mutated)
    graph = build_junction_graph(original, analysis, tolerance=1.0)
    sync_report = synchronize_junctions(mutated, graph, analysis)
    sync_line_counterparts(mutated, analysis)
    sync_closed_component_point_sequences(mutated)
    update_component_bboxes(mutated)
    junction_report = validate_junctions(mutated, graph, analysis, tolerance=1.0)
    mutation_report["junction_fixed"] = sync_report.get("junction_fixed", [])
    mutation_report["junction_validation"] = junction_report

    shape_report = validate_shape_quality(original, mutated, analysis)
    mutation_report["shape_quality"] = shape_report
    mutation_report["shape_quality_score"] = shape_report.get("shape_quality_score", 0.0)
    mutation_report["valid"] = bool(junction_report.get("valid", True) and shape_report.get("valid", True))
    mutation_report["errors"].extend(junction_error_messages(junction_report))
    mutation_report["errors"].extend(shape_report.get("errors", []))
    mutation_report["warnings"].extend(shape_report.get("warnings", []))
    mutation_report["port_constraints"] = build_port_constraint_report(analysis, variable_values)

    metadata = mutated.setdefault("optimization_metadata", {})
    metadata["primitive_analysis"] = analysis
    metadata["primitive_aware_constraints"] = {
        "junction_validation": junction_report,
        "shape_quality": shape_report,
        "port_constraint_report": mutation_report["port_constraints"],
    }

    if output_path is not None:
        write_primitive_debug_outputs(
            original,
            before_for_debug,
            mutated,
            analysis,
            variable_defs,
            variable_values,
            graph,
            junction_report,
            mutation_report,
            output_path,
        )
    return mutated, mutation_report


def apply_line_mutation(
    payload: Dict[str, Any],
    primitive: Dict[str, Any],
    variable_type: str,
    value: float,
    analysis: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """Apply a constrained line/feedline mutation.

    Input: mutable payload, primitive record, variable type, value, full
    analysis, and mutation report.
    Output: in-place geometry update and appended report entry.
    Algorithm purpose: move an entire line as a rigid feature, or move feed
    neighbors only along propagation direction while preserving port width.
    """

    primitive_obj = primitive_object(payload, primitive)
    if primitive_obj is None:
        return
    start = parse_point(primitive_obj.get("start")) or primitive_endpoint_from_samples(payload, primitive, "start")
    end = parse_point(primitive_obj.get("end")) or primitive_endpoint_from_samples(payload, primitive, "end")
    if start is None or end is None:
        return
    direction = unit((end[0] - start[0], end[1] - start[1]))
    normal = (-direction[1], direction[0])
    displacement_start = (0.0, 0.0)
    displacement_end = (0.0, 0.0)
    if variable_type == "line_normal_offset":
        displacement_start = (normal[0] * value, normal[1] * value)
        displacement_end = displacement_start
    elif variable_type == "line_length_delta":
        displacement_start = (-0.5 * direction[0] * value, -0.5 * direction[1] * value)
        displacement_end = (0.5 * direction[0] * value, 0.5 * direction[1] * value)
    elif variable_type == "feed_length":
        port_context = analysis.get("summary", {}).get("port_context", {})
        propagation = tuple(float(v) for v in port_context.get("propagation_direction", direction))
        displacement_start = (0.0, 0.0)
        displacement_end = (propagation[0] * value, propagation[1] * value)

    new_start = (start[0] + displacement_start[0], start[1] + displacement_start[1])
    new_end = (end[0] + displacement_end[0], end[1] + displacement_end[1])
    set_line_primitive_endpoints(payload, primitive, new_start, new_end)
    apply_range_linear_endpoint_motion(payload, primitive, start, end, new_start, new_end, analysis)
    report["line_mutations"].append(
        {
            "primitive_id": primitive.get("primitive_id"),
            "variable_type": variable_type,
            "value": value,
            "start": list(new_start),
            "end": list(new_end),
        }
    )


def apply_port_mutation(
    payload: Dict[str, Any],
    primitive: Dict[str, Any],
    variable_type: str,
    value: float,
    analysis: Dict[str, Any],
    report: Dict[str, Any],
) -> None:
    """Apply explicit port width/alignment mutations.

    Input: mutable payload, PORT primitive record, variable type/value, full
    analysis, and mutation report.
    Output: in-place port line update and appended report entry.
    Algorithm purpose: increase port width or shift the port along propagation
    direction while keeping the port represented as a straight line.
    """

    primitive_obj = primitive_object(payload, primitive)
    if primitive_obj is None:
        return
    start = parse_point(primitive_obj.get("start")) or primitive_endpoint_from_samples(payload, primitive, "start")
    end = parse_point(primitive_obj.get("end")) or primitive_endpoint_from_samples(payload, primitive, "end")
    if start is None or end is None:
        return

    direction = unit((end[0] - start[0], end[1] - start[1]))
    if math.hypot(direction[0], direction[1]) <= 1e-12:
        return

    port_context = analysis.get("summary", {}).get("port_context", {}) or {}
    fallback_normal = (-direction[1], direction[0])
    propagation = unit(tuple(float(v) for v in port_context.get("propagation_direction", fallback_normal)))
    if math.hypot(propagation[0], propagation[1]) <= 1e-12:
        propagation = fallback_normal

    if variable_type == "port_width_delta":
        delta = max(0.0, float(value))
        new_start = (start[0] - 0.5 * direction[0] * delta, start[1] - 0.5 * direction[1] * delta)
        new_end = (end[0] + 0.5 * direction[0] * delta, end[1] + 0.5 * direction[1] * delta)
    elif variable_type == "port_propagation_shift":
        shift = float(value)
        offset = (propagation[0] * shift, propagation[1] * shift)
        new_start = (start[0] + offset[0], start[1] + offset[1])
        new_end = (end[0] + offset[0], end[1] + offset[1])
    else:
        return

    set_line_primitive_endpoints(payload, primitive, new_start, new_end)
    apply_range_linear_endpoint_motion(
        payload,
        primitive,
        start,
        end,
        new_start,
        new_end,
        analysis,
        freeze_port_core=False,
    )
    report["port_mutations"].append(
        {
            "primitive_id": primitive.get("primitive_id"),
            "variable_type": variable_type,
            "value": value,
            "start": list(new_start),
            "end": list(new_end),
            "propagation_direction": list(propagation),
        }
    )


def apply_parallel_line_spacing(
    payload: Dict[str, Any],
    primitive_by_id: Dict[str, Dict[str, Any]],
    primitive_ids: Sequence[str],
    value: float,
    report: Dict[str, Any],
) -> None:
    """Apply paired parallel-line spacing mutation.

    Input: payload, primitive lookup, two primitive ids, spacing delta, and report.
    Output: in-place paired line offset.
    Algorithm purpose: change slot/feed width while keeping both boundary lines
    parallel and avoiding endpoint drift.
    """

    if len(primitive_ids) < 2:
        return
    left = primitive_by_id.get(primitive_ids[0])
    right = primitive_by_id.get(primitive_ids[1])
    if left is None or right is None:
        return
    for primitive, sign in ((left, -0.5), (right, 0.5)):
        primitive_obj = primitive_object(payload, primitive)
        if primitive_obj is None:
            continue
        start = parse_point(primitive_obj.get("start"))
        end = parse_point(primitive_obj.get("end"))
        if start is None or end is None:
            continue
        direction = unit((end[0] - start[0], end[1] - start[1]))
        normal = (-direction[1], direction[0])
        delta = (normal[0] * value * sign, normal[1] * value * sign)
        set_line_primitive_endpoints(
            payload,
            primitive,
            (start[0] + delta[0], start[1] + delta[1]),
            (end[0] + delta[0], end[1] + delta[1]),
        )
        apply_range_translation(payload, primitive, delta)
    report["line_mutations"].append({"primitive_id": list(primitive_ids[:2]), "variable_type": "parallel_line_spacing", "value": value})


def apply_spline_mutation(
    payload: Dict[str, Any],
    primitive: Dict[str, Any],
    value: float,
    report: Dict[str, Any],
) -> None:
    """Apply Gaussian smooth B-spline deformation.

    Input: mutable payload, spline primitive record, sampled scalar value, and report.
    Output: in-place control cage/sample updates and report entry.
    Algorithm purpose: deform the spline as a smooth cage with frozen endpoints
    and damped handles, preserving smooth curvature and C1-style behavior.
    """

    primitive_obj = primitive_object(payload, primitive)
    if primitive_obj is None:
        return
    controls = parse_points(primitive_obj.get("control_points"))
    if len(controls) < 3:
        return
    normal = primitive_normal(controls)
    deformation = smooth_control_cage(controls, delta=(normal[0] * value, normal[1] * value), endpoint_mode="freeze")
    primitive_obj["control_points"] = deformation["points"]

    samples = primitive_sample_points(payload, primitive)
    if len(samples) >= 3:
        sample_deformation = smooth_control_cage(samples, delta=(normal[0] * value, normal[1] * value), endpoint_mode="freeze")
        set_primitive_sample_points(payload, primitive, sample_deformation["points"])

    entry = {
        "primitive_id": primitive.get("primitive_id"),
        "value": value,
        "center_index": deformation.get("center_index"),
        "sigma": deformation.get("sigma"),
        "point_roles": deformation.get("roles"),
        "frozen_indices": deformation.get("frozen_indices"),
        "weights": deformation.get("weights"),
    }
    report["spline_mutations"].append(entry)


def apply_curve_mutation(
    payload: Dict[str, Any],
    primitive: Dict[str, Any],
    value: float,
    report: Dict[str, Any],
) -> None:
    """Apply smooth offset to arc/curve sampled span.

    Input: mutable payload, curve primitive record, scalar value, and report.
    Output: in-place dependent sampled span update.
    Algorithm purpose: provide curve-level motion without independent sampled
    point mutation.
    """

    samples = primitive_sample_points(payload, primitive)
    if len(samples) < 3:
        return
    normal = primitive_normal(samples)
    deformation = smooth_control_cage(samples, delta=(normal[0] * value, normal[1] * value), endpoint_mode="freeze")
    set_primitive_sample_points(payload, primitive, deformation["points"])
    report["curve_mutations"].append({"primitive_id": primitive.get("primitive_id"), "value": value, "weights": deformation.get("weights")})


def apply_range_linear_endpoint_motion(
    payload: Dict[str, Any],
    primitive: Dict[str, Any],
    old_start: Point,
    old_end: Point,
    new_start: Point,
    new_end: Point,
    analysis: Dict[str, Any],
    freeze_port_core: bool = True,
) -> None:
    """Interpolate endpoint motion across a primitive's sampled span.

    Input: payload, primitive record, old/new endpoints, and analysis context.
    Output: in-place component sample update.
    Algorithm purpose: keep dependent cached line points consistent with rigid
    line offset/length changes while respecting port core freezing.
    """

    component = component_object(payload, primitive)
    if component is None:
        return
    start_idx = primitive.get("start_idx")
    end_idx = primitive.get("end_idx")
    if not isinstance(start_idx, int) or not isinstance(end_idx, int):
        return
    lo = max(0, min(start_idx, end_idx))
    hi = max(start_idx, end_idx)
    port_context = analysis.get("summary", {}).get("port_context", {})
    core_bbox = tuple(float(v) for v in port_context.get("core_bbox", []))
    for key in ("resampled_points", "fallback_points", "sampled_points", "points"):
        values = component.get(key)
        if not isinstance(values, list):
            continue
        hi_clamped = min(hi, len(values) - 1)
        span = max(1, hi_clamped - lo)
        for index in range(lo, hi_clamped + 1):
            point = parse_point(values[index])
            if point is None:
                continue
            # Port constraint: points inside PORT_CORE are completely frozen;
            # feed-neighbor motion is generated only along propagation upstream.
            if freeze_port_core and len(core_bbox) == 4 and point_in_bbox(point, core_bbox):
                continue
            t = (index - lo) / span
            dx = (new_start[0] - old_start[0]) * (1.0 - t) + (new_end[0] - old_end[0]) * t
            dy = (new_start[1] - old_start[1]) * (1.0 - t) + (new_end[1] - old_end[1]) * t
            values[index] = [point[0] + dx, point[1] + dy]


def set_line_primitive_endpoints(
    payload: Dict[str, Any],
    primitive: Dict[str, Any],
    start: Point,
    end: Point,
) -> None:
    """Set line endpoints in both `segments` and `primitives` when present.

    Input: payload, analysis primitive record, and replacement endpoints.
    Output: in-place schema-native line endpoint updates.
    Algorithm purpose: keep BO mutation visible to both diagnostic segment
    records and CST builder compact primitive reconstruction.
    """

    component = component_object(payload, primitive)
    if component is None:
        return
    index = primitive.get("primitive_index")
    if not isinstance(index, int):
        return
    source_key = str(primitive.get("source_key", "segments"))
    keys = [source_key]
    for key in ("segments", "primitives"):
        if key not in keys:
            keys.append(key)

    for key in keys:
        items = component.get(key)
        if not isinstance(items, list) or not (0 <= index < len(items)):
            continue
        item = items[index]
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or item.get("kind") or item.get("primitive_type") or "").lower()
        if "line" not in kind and key != source_key:
            continue
        item["start"] = [float(start[0]), float(start[1])]
        item["end"] = [float(end[0]), float(end[1])]
        points = item.get("points")
        if isinstance(points, list) and len(points) >= 2:
            points[0] = [float(start[0]), float(start[1])]
            points[-1] = [float(end[0]), float(end[1])]


def sync_line_counterparts(payload: Dict[str, Any], analysis: Dict[str, Any]) -> None:
    """Copy mutated line endpoints from the analyzed source to its counterpart.

    Input: mutable payload and primitive analysis.
    Output: in-place synchronization between `segments` and `primitives`.
    Algorithm purpose: keep CST compact primitive reconstruction consistent
    after junction synchronization, which updates the analyzed source entries.
    """

    for primitive in analysis.get("primitives", []) or []:
        if primitive.get("type") != "LINE":
            continue
        source = primitive_object(payload, primitive)
        if source is None:
            continue
        start = parse_point(source.get("start"))
        end = parse_point(source.get("end"))
        if start is None or end is None:
            continue
        set_line_primitive_endpoints(payload, primitive, start, end)


def apply_range_translation(payload: Dict[str, Any], primitive: Dict[str, Any], delta: Point) -> None:
    """Translate a primitive sampled span.

    Input: mutable payload, primitive record, and delta vector.
    Output: in-place sample update.
    Algorithm purpose: keep line-pair spacing variables as rigid parallel moves.
    """

    samples = primitive_sample_points(payload, primitive)
    if not samples:
        return
    translated = [[point[0] + delta[0], point[1] + delta[1]] for point in samples]
    set_primitive_sample_points(payload, primitive, translated)


def primitive_object(payload: Dict[str, Any], primitive: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a live primitive dictionary from an analysis record.

    Input: payload and primitive analysis record.
    Output: mutable primitive object or None.
    Algorithm purpose: locate the schema-native primitive for parameter updates.
    """

    component = component_object(payload, primitive)
    if component is None:
        return None
    items = component.get(str(primitive.get("source_key", "segments")), []) or []
    index = primitive.get("primitive_index")
    if isinstance(index, int) and 0 <= index < len(items) and isinstance(items[index], dict):
        return items[index]
    return None


def component_object(payload: Dict[str, Any], primitive: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a live component dictionary from an analysis record.

    Input: payload and primitive analysis record.
    Output: mutable component object or None.
    Algorithm purpose: locate component-level sampled caches for updates.
    """

    components = payload.get("components", []) or []
    index = primitive.get("component_index")
    if isinstance(index, int) and 0 <= index < len(components) and isinstance(components[index], dict):
        return components[index]
    return None


def primitive_endpoint_from_samples(payload: Dict[str, Any], primitive: Dict[str, Any], endpoint: str) -> Optional[Point]:
    """Read an endpoint from component sampled caches.

    Input: payload, primitive record, and endpoint name.
    Output: point or None.
    Algorithm purpose: support line records with missing explicit start/end.
    """

    samples = primitive_sample_points(payload, primitive)
    if not samples:
        return None
    return samples[0] if endpoint == "start" else samples[-1]


def primitive_sample_points(payload: Dict[str, Any], primitive: Dict[str, Any]) -> List[Point]:
    """Read sampled points for a primitive span.

    Input: payload and primitive analysis record.
    Output: current sampled point span.
    Algorithm purpose: update dependent caches for spline/curve/line mutation.
    """

    component = component_object(payload, primitive)
    if component is None:
        return []
    points = parse_points(component.get("resampled_points") or component.get("fallback_points") or component.get("sampled_points") or component.get("points"))
    start_idx = primitive.get("start_idx")
    end_idx = primitive.get("end_idx")
    if not isinstance(start_idx, int) or not isinstance(end_idx, int) or not points:
        return []
    lo = max(0, min(start_idx, end_idx))
    hi = min(len(points) - 1, max(start_idx, end_idx))
    return points[lo : hi + 1]


def set_primitive_sample_points(payload: Dict[str, Any], primitive: Dict[str, Any], points: Sequence[Sequence[float]]) -> None:
    """Write sampled points back to a primitive span.

    Input: payload, primitive record, and replacement point span.
    Output: in-place cache update.
    Algorithm purpose: keep CST-facing cached point arrays aligned with
    primitive-level variables while preserving the original schema.
    """

    component = component_object(payload, primitive)
    if component is None:
        return
    start_idx = primitive.get("start_idx")
    end_idx = primitive.get("end_idx")
    if not isinstance(start_idx, int) or not isinstance(end_idx, int):
        return
    lo = max(0, min(start_idx, end_idx))
    for key in ("resampled_points", "fallback_points", "sampled_points", "points"):
        values = component.get(key)
        if not isinstance(values, list):
            continue
        for offset, point in enumerate(points):
            index = lo + offset
            if index >= len(values):
                break
            values[index] = [float(point[0]), float(point[1])]


def update_component_bboxes(payload: Dict[str, Any]) -> None:
    """Update component bbox fields after mutation.

    Input: mutable payload.
    Output: in-place bbox updates.
    Algorithm purpose: keep derived component metadata consistent for validators.
    """

    for component in payload.get("components", []) or []:
        points = parse_points(component.get("resampled_points") or component.get("fallback_points") or component.get("points"))
        if points:
            component["bbox"] = list(point_bbox(points))


def sync_closed_component_point_sequences(payload: Dict[str, Any], tolerance: float = 1e-7) -> None:
    """Keep cached point arrays explicitly closed for closed components.

    Input: mutable parameterization payload.
    Output: in-place update of resampled/fallback point arrays.
    Algorithm purpose: candidate mutations may move the duplicated closure
    endpoint independently from the first point; optimization validators read
    these cached arrays directly, so closed loops must keep last == first.
    """

    for component in payload.get("components", []) or []:
        if not bool(component.get("closed", False)):
            continue
        for key in ("resampled_points", "fallback_points", "sampled_points", "points"):
            values = component.get(key)
            if not isinstance(values, list) or len(values) < 3:
                continue
            first = values[0]
            last = values[-1]
            if not is_point(first):
                continue
            segment_count = len(component.get("segments") or component.get("primitives") or [])
            explicit_closure_length = segment_count > 0 and len(values) == segment_count + 1
            if explicit_closure_length:
                values[-1] = [float(first[0]), float(first[1])]
            elif not is_point(last) or point_distance_2d(first, last) > tolerance:
                values.append([float(first[0]), float(first[1])])
            else:
                values[-1] = [float(first[0]), float(first[1])]
        points = parse_points(component.get("resampled_points") or component.get("fallback_points") or component.get("points"))
        if points:
            component["sampled_point_count"] = len(points)
            metrics = component.get("metrics")
            if isinstance(metrics, dict):
                metrics["sampled_point_count"] = len(points)


def point_distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def build_port_constraint_report(analysis: Dict[str, Any], variable_values: Dict[str, float]) -> Dict[str, Any]:
    """Build the port constraint report.

    Input: primitive analysis and sampled variable values.
    Output: report for PORT_CORE, PORT_NEIGHBOR, and NORMAL_REGION behavior.
    Algorithm purpose: document that port core is frozen and feed-neighbor
    motion is restricted to feed propagation direction.
    """

    primitives = analysis.get("primitives", []) or []
    return {
        "zones": {
            "PORT_CORE": "explicit port width expansion and propagation shift allowed; raw point drift remains disabled",
            "PORT_NEIGHBOR": "feed_propagation_direction_only",
            "NORMAL_REGION": "primitive_mode_specific",
        },
        "port_primitives": [primitive.get("primitive_id") for primitive in primitives if primitive.get("role") == "PORT"],
        "feed_primitives": [primitive.get("primitive_id") for primitive in primitives if primitive.get("role") == "FEEDLINE"],
        "allowed_port_variables": [
            name
            for name in variable_values
            if "port_width_delta" in name.lower() or "port_propagation_shift" in name.lower()
        ],
        "blocked_variables": [
            name
            for name in variable_values
            if "port_contact" in name.lower() or "port_alignment" in name.lower()
        ],
        "port_context": analysis.get("summary", {}).get("port_context", {}),
    }


def junction_error_messages(junction_report: Dict[str, Any]) -> List[str]:
    """Convert broken junctions into validation error messages.

    Input: junction validation report.
    Output: list of error strings.
    Algorithm purpose: propagate topology failures into BO rejection metadata.
    """

    return [
        f"{item.get('junction_id')} gap {item.get('max_gap'):.6f} exceeds tolerance"
        for item in junction_report.get("broken_junctions", []) or []
    ]


def write_primitive_debug_outputs(
    original: Dict[str, Any],
    before_debug: Dict[str, Any],
    mutated: Dict[str, Any],
    analysis: Dict[str, Any],
    variable_defs: Sequence[Dict[str, Any]],
    variable_values: Dict[str, float],
    graph: Dict[str, Any],
    junction_report: Dict[str, Any],
    mutation_report: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write primitive-aware JSON and visualization diagnostics.

    Input: original/mutated payloads, analysis, variables, junction graph/report,
    mutation report, and evaluation directory.
    Output: debug JSON files and PNG images under the evaluation directory.
    Algorithm purpose: satisfy per-evaluation traceability for primitive-aware
    BO mutation decisions.
    """

    debug_dir = output_dir / "primitive_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    write_json_file(output_dir / "primitive_mutation_report.json", mutation_report)
    write_json_file(debug_dir / "spline_constraint_report.json", {"spline_mutations": mutation_report.get("spline_mutations", [])})
    write_json_file(debug_dir / "port_constraint_report.json", mutation_report.get("port_constraints", {}))
    write_junction_validation(debug_dir / "junction_validation.json", junction_report)
    write_shape_quality_report(debug_dir / "shape_quality_report.json", mutation_report.get("shape_quality", {}))
    plot_junction_debug(mutated, graph, junction_report, debug_dir / "junction_debug.png")
    plot_curvature_change_map(original, mutated, debug_dir / "curvature_change_map.png")
    plot_spline_deformation_debug(before_debug, mutated, mutation_report, debug_dir / "spline_deformation_debug.png")
    plot_variable_effect_overlay(original, analysis, variable_defs, variable_values, debug_dir / "variable_effect_overlay.png")
    selected_report = build_selected_optimization_points(original, mutated, analysis, variable_defs, variable_values)
    write_json_file(debug_dir / "selected_optimization_points.json", selected_report)
    plot_optimization_before_after_comparison(
        original,
        mutated,
        analysis,
        selected_report,
        debug_dir / "optimization_before_after_comparison.png",
    )
    plot_selected_points_overlay(
        original,
        mutated,
        selected_report,
        debug_dir / "selected_points_overlay.png",
    )


def plot_spline_deformation_debug(
    before: Dict[str, Any],
    after: Dict[str, Any],
    report: Dict[str, Any],
    path: Path,
) -> None:
    """Plot before/after spline deformation and influence weights.

    Input: before payload, after payload, mutation report, and output path.
    Output: PNG diagnostic image.
    Algorithm purpose: visualize control cage movement and Gaussian influence.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_payload_points(ax, before, "#bdbdbd", "before")
    plot_payload_points(ax, after, "#1f77b4", "after")
    for mutation in report.get("spline_mutations", []) or []:
        weights = mutation.get("weights") or []
        ax.text(0.02, 0.98, f"{mutation.get('primitive_id')} weights={len(weights)}", transform=ax.transAxes, fontsize=8, va="top")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Spline Deformation Debug")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_variable_effect_overlay(
    payload: Dict[str, Any],
    analysis: Dict[str, Any],
    variable_defs: Sequence[Dict[str, Any]],
    variable_values: Dict[str, float],
    path: Path,
) -> None:
    """Plot BO variable influence regions.

    Input: payload, analysis, variable definitions, sampled values, and path.
    Output: PNG overlay.
    Algorithm purpose: show which primitive region each physical BO variable affects.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_payload_points(ax, payload, "#cccccc", "geometry")
    primitive_by_id = {primitive.get("primitive_id"): primitive for primitive in analysis.get("primitives", []) or []}
    for variable in variable_defs:
        name = variable.get("name")
        if name not in variable_values:
            continue
        primitive_ids = str(variable.get("primitive_id", "")).split("|")
        for primitive_id in primitive_ids:
            primitive = primitive_by_id.get(primitive_id)
            if not primitive:
                continue
            points = [tuple(point) for point in primitive.get("points", []) if len(point) >= 2]
            if not points:
                continue
            ax.plot([p[0] for p in points], [p[1] for p in points], linewidth=2.0, marker="o", markersize=2.5)
            cx, cy = sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)
            ax.text(cx, cy, str(name), fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Variable Effect Overlay")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_selected_optimization_points(
    original: Dict[str, Any],
    mutated: Dict[str, Any],
    analysis: Dict[str, Any],
    variable_defs: Sequence[Dict[str, Any]],
    variable_values: Dict[str, float],
) -> Dict[str, Any]:
    """Build a report of optimized primitives and selected points.

    Input: original/mutated payloads, primitive analysis, BO variable
    definitions, and sampled variable values.
    Output: JSON-safe report listing affected curve segments and chosen points.
    Algorithm purpose: make the visual comparison traceable by connecting each
    physical BO variable to its primitive span and control/sample points.
    """

    primitive_by_id = {primitive.get("primitive_id"): primitive for primitive in analysis.get("primitives", []) or []}
    selected: List[Dict[str, Any]] = []
    for variable in variable_defs:
        name = variable.get("name")
        value = float(variable_values.get(str(name), 0.0)) if name in variable_values else 0.0
        if name not in variable_values or abs(value) <= 1e-15:
            continue
        for primitive_id in str(variable.get("primitive_id", "")).split("|"):
            primitive = primitive_by_id.get(primitive_id)
            if primitive is None:
                continue
            before_points = current_primitive_points(original, primitive)
            after_points = current_primitive_points(mutated, primitive)
            selected_points = choose_display_points(before_points)
            selected.append(
                {
                    "variable": name,
                    "value": value,
                    "variable_type": variable.get("variable_type"),
                    "primitive_id": primitive_id,
                    "primitive_type": primitive.get("type"),
                    "role": primitive.get("role"),
                    "before_points": [[x, y] for x, y in before_points],
                    "after_points": [[x, y] for x, y in after_points],
                    "selected_points": [
                        {"index": index, "point": [point[0], point[1]], "label": f"P{order + 1}"}
                        for order, (index, point) in enumerate(selected_points)
                    ],
                }
            )
    return {
        "optimized_segment_count": len(selected),
        "optimized_segments": selected,
        "legend": {
            "gray": "original full geometry",
            "blue": "optimized full geometry",
            "orange": "original optimized segment",
            "red": "optimized segment",
            "black_points": "selected/control points used by BO primitive variable",
        },
    }


def plot_optimization_before_after_comparison(
    original: Dict[str, Any],
    mutated: Dict[str, Any],
    analysis: Dict[str, Any],
    selected_report: Dict[str, Any],
    path: Path,
) -> None:
    """Plot original and optimized geometry with optimized segments highlighted.

    Input: original payload, mutated payload, analysis, selected point report,
    and output path.
    Output: PNG comparison image.
    Algorithm purpose: provide a direct visual answer to what changed, which
    curve segment was optimized, and which points drove the mutation.
    """

    fig, ax = plt.subplots(figsize=(10, 9))
    plot_payload_points(ax, original, "#a8a8a8", "original")
    plot_payload_points(ax, mutated, "#1f77b4", "optimized")
    for segment in selected_report.get("optimized_segments", []) or []:
        before_points = [tuple(point) for point in segment.get("before_points", [])]
        after_points = [tuple(point) for point in segment.get("after_points", [])]
        if before_points:
            ax.plot(
                [p[0] for p in before_points],
                [p[1] for p in before_points],
                color="#f59e0b",
                linestyle="--",
                linewidth=2.2,
                label="optimized segment before",
            )
        if after_points:
            ax.plot(
                [p[0] for p in after_points],
                [p[1] for p in after_points],
                color="#dc2626",
                linewidth=2.6,
                label="optimized segment after",
            )
            mid = after_points[len(after_points) // 2]
            ax.text(mid[0], mid[1], str(segment.get("variable")), fontsize=8, color="#991b1b")
        draw_selected_points(ax, segment)
        draw_motion_arrows(ax, before_points, after_points)
    dedupe_legend(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Optimization Before/After Comparison")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_selected_points_overlay(
    original: Dict[str, Any],
    mutated: Dict[str, Any],
    selected_report: Dict[str, Any],
    path: Path,
) -> None:
    """Plot selected optimization points and their moved locations.

    Input: original payload, mutated payload, selected point report, and output path.
    Output: PNG selected-point overlay.
    Algorithm purpose: make chosen BO primitive/control/sample points visible
    without the visual clutter of all debug layers.
    """

    fig, ax = plt.subplots(figsize=(10, 9))
    plot_payload_points(ax, original, "#d1d5db", "original")
    plot_payload_points(ax, mutated, "#60a5fa", "optimized")
    for segment in selected_report.get("optimized_segments", []) or []:
        before_points = [tuple(point) for point in segment.get("before_points", [])]
        after_points = [tuple(point) for point in segment.get("after_points", [])]
        if after_points:
            ax.plot([p[0] for p in after_points], [p[1] for p in after_points], color="#ef4444", linewidth=2.5)
        draw_selected_points(ax, segment)
        draw_motion_arrows(ax, before_points, after_points)
    dedupe_legend(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Selected Optimization Points")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def current_primitive_points(payload: Dict[str, Any], primitive: Dict[str, Any]) -> List[Point]:
    """Read current representative points for a primitive.

    Input: payload and primitive analysis record.
    Output: current sampled/control/endpoint points.
    Algorithm purpose: draw before/after primitive spans from live geometry
    rather than stale analysis coordinates.
    """

    if primitive.get("type") == "BSPLINE":
        obj = primitive_object(payload, primitive)
        controls = parse_points(obj.get("control_points") if obj else None)
        if controls:
            return controls
    if primitive.get("type") == "LINE":
        obj = primitive_object(payload, primitive)
        if obj:
            start = parse_point(obj.get("start"))
            end = parse_point(obj.get("end"))
            if start is not None and end is not None:
                return [start, end]
    sampled = primitive_sample_points(payload, primitive)
    if sampled:
        return sampled
    return [tuple(point) for point in primitive.get("points", []) if len(point) >= 2]


def choose_display_points(points: Sequence[Point], max_points: int = 7) -> List[Tuple[int, Point]]:
    """Select readable point markers for a primitive span.

    Input: primitive points and maximum display count.
    Output: list of original point indexes and points.
    Algorithm purpose: mark endpoints, handles, and representative internal
    points without overcrowding the comparison image.
    """

    if not points:
        return []
    if len(points) <= max_points:
        return [(index, points[index]) for index in range(len(points))]
    indexes = {0, 1, len(points) // 2, len(points) - 2, len(points) - 1}
    step = max(1, len(points) // max_points)
    for index in range(0, len(points), step):
        indexes.add(index)
        if len(indexes) >= max_points:
            break
    return [(index, points[index]) for index in sorted(indexes) if 0 <= index < len(points)]


def draw_selected_points(ax: Any, segment: Dict[str, Any]) -> None:
    """Draw selected point markers and labels.

    Input: matplotlib axis and one selected segment report.
    Output: marker artists on the axis.
    Algorithm purpose: consistently identify which points were selected for BO
    control or primitive influence.
    """

    for selected in segment.get("selected_points", []) or []:
        point = selected.get("point", [])
        if len(point) < 2:
            continue
        ax.scatter([point[0]], [point[1]], s=38, facecolor="#ffffff", edgecolor="#111827", linewidth=1.2, zorder=5)
        ax.text(point[0], point[1], selected.get("label", ""), fontsize=7, color="#111827", zorder=6)


def draw_motion_arrows(ax: Any, before_points: Sequence[Point], after_points: Sequence[Point]) -> None:
    """Draw sparse arrows from original points to optimized points.

    Input: matplotlib axis plus before/after point sequences.
    Output: arrow artists on the axis.
    Algorithm purpose: make deformation direction and magnitude visible.
    """

    if len(before_points) != len(after_points) or not before_points:
        return
    for index, before_point in choose_display_points(before_points, max_points=5):
        after_point = after_points[index]
        dx = after_point[0] - before_point[0]
        dy = after_point[1] - before_point[1]
        if math.hypot(dx, dy) <= 1e-9:
            continue
        ax.arrow(
            before_point[0],
            before_point[1],
            dx,
            dy,
            color="#111827",
            width=0.15,
            head_width=3.0,
            length_includes_head=True,
            alpha=0.75,
            zorder=4,
        )


def dedupe_legend(ax: Any) -> None:
    """Remove duplicate legend entries.

    Input: matplotlib axis.
    Output: compact legend on the axis.
    Algorithm purpose: keep comparison figures readable when many segments are
    highlighted with repeated labels.
    """

    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)


def plot_payload_points(ax: Any, payload: Dict[str, Any], color: str, label: str) -> None:
    """Plot component sampled points on an axis.

    Input: matplotlib axis, payload, color, and label.
    Output: plotted geometry.
    Algorithm purpose: share debug plotting code for primitive overlays.
    """

    used_label = False
    for component in payload.get("components", []) or []:
        points = parse_points(component.get("resampled_points") or component.get("fallback_points") or component.get("points"))
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=0.9, label=label if not used_label else None)
            used_label = True


def parse_point(value: Any) -> Optional[Point]:
    """Parse a single 2D point.

    Input: arbitrary JSON value.
    Output: finite point tuple or None.
    Algorithm purpose: normalize coordinates for primitive-aware mutation math.
    """

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


def unit(vector: Point) -> Point:
    """Normalize a vector.

    Input: 2D vector.
    Output: unit vector or zero vector.
    Algorithm purpose: compute line, feed, and normal deformation directions.
    """

    length = math.hypot(vector[0], vector[1])
    if length <= 1e-12:
        return 0.0, 0.0
    return vector[0] / length, vector[1] / length


def primitive_normal(points: Sequence[Point]) -> Point:
    """Estimate a primitive normal from its endpoint chord.

    Input: ordered primitive points.
    Output: unit normal vector.
    Algorithm purpose: orient line/curve/spline offset variables consistently.
    """

    if len(points) < 2:
        return 0.0, 1.0
    direction = unit((points[-1][0] - points[0][0], points[-1][1] - points[0][1]))
    normal = (-direction[1], direction[0])
    if math.hypot(normal[0], normal[1]) <= 1e-12:
        return 0.0, 1.0
    return normal


def point_in_bbox(point: Point, bbox: Sequence[float]) -> bool:
    """Check whether a point is inside a bbox.

    Input: point and bbox sequence.
    Output: boolean.
    Algorithm purpose: enforce PORT_CORE freeze during sample updates.
    """

    if len(bbox) < 4:
        return False
    return float(bbox[0]) <= point[0] <= float(bbox[2]) and float(bbox[1]) <= point[1] <= float(bbox[3])


def write_json_file(path: Path, payload: Any) -> None:
    """Write a JSON file.

    Input: path and JSON payload.
    Output: JSON file on disk.
    Algorithm purpose: local helper for primitive mutation reports.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def iter_primitives(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """遍历 payload 中的 primitives/segments。"""
    for component in payload.get("components", []) or []:
        for primitive in component.get("primitives", []) or []:
            if isinstance(primitive, dict):
                yield primitive
        for segment in component.get("segments", []) or []:
            if isinstance(segment, dict):
                yield segment


def primitive_kind(primitive: Dict[str, Any]) -> str:
    """归一化 primitive 类型名：line/arc/spline。"""
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
    """从 primitive 参数中收集几何点。"""
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
    """从 component 缓存点列中收集几何点。"""
    points: List[Point] = []
    for component in payload.get("components", []) or []:
        for key in ("resampled_points", "fallback_points", "sampled_points", "points"):
            points.extend(parse_points(component.get(key)))
    return points


def parse_points(value: Any) -> List[Point]:
    """从列表中解析二维点列表。"""
    if not isinstance(value, list):
        return []
    points: List[Point] = []
    for item in value:
        if is_point(item):
            points.append((float(item[0]), float(item[1])))
    return points


def is_point(value: Any) -> bool:
    """判断 value 是否形如 [x, y] 或 [x, y, z]。"""
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return False
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and math.isfinite(y)


def point_bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    """计算二维点集包围盒。"""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _transform_payload_coordinates(
    value: Any,
    transform_point: Any,
    uniform_scale: float,
    parent_key: str = "",
) -> Any:
    """递归更新 payload 中的几何坐标字段，用于 global_scale 兜底路径。"""
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
    """对 bbox 的四个角点做同样坐标变换。"""
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
    """判断值是否可转换为有限浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
