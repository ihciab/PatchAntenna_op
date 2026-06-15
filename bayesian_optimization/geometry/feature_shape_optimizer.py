from __future__ import annotations

"""Feature-constrained BO geometry layer.

This module replaces single-control-point BO motion with feature variables and
continuous point-group normal offsets. It stays between parameterization JSON and
the CST/objective pipeline: it does not change the input schema, CST builder,
simulation flow, S11 parsing, or objective definition.
"""

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesian_optimization.geometry.port_summary_utils import (
    resolve_port_point,
    resolve_port_side,
    resolve_port_width,
)


Point = Tuple[float, float]
PointKey = Tuple[int, int]
NON_FEED_PORT_RANGE_MULTIPLIER = 1.25

PORT_CORE = "PORT_CORE"
PORT_NEIGHBOR = "PORT_NEIGHBOR"
NORMAL_REGION = "NORMAL_REGION"


@dataclass(frozen=True)
class ShapeDesignVariable:
    name: str
    lower: float
    upper: float
    default: float
    level: int
    category: str
    target_id: str
    deformation_mode: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureRegion:
    feature_id: str
    kind: str
    component_index: int
    point_indices: List[int]
    bbox: Tuple[float, float, float, float]
    confidence: float
    variables: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PointGroup:
    group_id: str
    component_index: int
    point_indices: List[int]
    point_ids: List[str]
    normal: Point
    max_offset: float
    source_feature: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortModel:
    axis: str
    port_side: str
    propagation_direction: Point
    transverse_direction: Point
    centerline: float
    estimated_width: float
    core_depth: float
    neighbor_depth: float
    bbox: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureShapePlan:
    variables: List[ShapeDesignVariable]
    features: List[FeatureRegion]
    groups: List[PointGroup]
    point_zones: Dict[str, str]
    port_model: PortModel
    port_constraint_report: Dict[str, Any]
    manufacturability_baseline: Dict[str, Any]
    enabled: bool
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variables": [variable.to_dict() for variable in self.variables],
            "features": [feature.to_dict() for feature in self.features],
            "groups": [group.to_dict() for group in self.groups],
            "point_zones": dict(self.point_zones),
            "port_model": self.port_model.to_dict(),
            "port_constraint_report": self.port_constraint_report,
            "manufacturability_baseline": self.manufacturability_baseline,
            "enabled": self.enabled,
            "source": self.source,
        }


@dataclass
class FeatureDeformationReport:
    valid: bool
    errors: List[str]
    warnings: List[str]
    variables: Dict[str, float]
    moved_points: List[str]
    group_offsets: Dict[str, float]
    feature_offsets: Dict[str, float]
    port_constraint_report: Dict[str, Any]
    manufacturability_report: Dict[str, Any]
    geometry_robustness_score: float
    deformation_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_feature_shape_plan(
    payload: Dict[str, Any],
    source: Path,
    port_summary: Optional[Dict[str, Any]] = None,
) -> FeatureShapePlan:
    port_model = detect_port_model(payload, port_summary)
    point_zones = classify_port_zones(payload, port_model)
    features = detect_feature_regions(payload, point_zones, port_model)
    groups = build_point_groups(payload, point_zones, features, port_model)
    variables = build_shape_variables(features, groups, port_model)
    port_report = build_port_constraint_report(payload, point_zones, port_model)
    baseline = validate_manufacturability(payload)
    return FeatureShapePlan(
        variables=variables,
        features=features,
        groups=groups,
        point_zones=point_zones,
        port_model=port_model,
        port_constraint_report=port_report,
        manufacturability_baseline=baseline,
        enabled=bool(variables),
        source=str(source),
    )


class FeatureConstrainedDeformer:
    def __init__(self, plan: FeatureShapePlan) -> None:
        self.plan = plan

    def apply(
        self,
        payload: Dict[str, Any],
        variable_values: Dict[str, float],
        iteration: int = 0,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], FeatureDeformationReport]:
        deformed = copy.deepcopy(payload)
        displacements: Dict[PointKey, Point] = {}
        feature_offsets: Dict[str, float] = {}
        group_offsets: Dict[str, float] = {}

        variable_by_name = {variable.name: variable for variable in self.plan.variables}
        for name, variable in variable_by_name.items():
            value = clamp(float(variable_values.get(name, variable.default)), variable.lower, variable.upper)
            if abs(value) <= 1e-15:
                continue
            if variable.category == "feature":
                self._apply_feature_variable(deformed, variable, value, displacements)
                feature_offsets[name] = value
            elif variable.category == "group":
                self._apply_group_variable(deformed, variable, value, displacements)
                group_offsets[name] = value

        moved_points = apply_displacements(deformed, displacements, self.plan.point_zones, self.plan.port_model)
        update_component_bboxes(deformed)
        manufacturability_report = validate_manufacturability(deformed)
        robustness = manufacturability_report.get("robustness_score", 1.0)
        report = FeatureDeformationReport(
            valid=bool(manufacturability_report.get("valid", True)),
            errors=list(manufacturability_report.get("errors", [])),
            warnings=list(manufacturability_report.get("warnings", [])),
            variables=dict(variable_values),
            moved_points=moved_points,
            group_offsets=group_offsets,
            feature_offsets=feature_offsets,
            port_constraint_report=self.plan.port_constraint_report,
            manufacturability_report=manufacturability_report,
            geometry_robustness_score=float(robustness),
            deformation_mode=("feature_variables" if feature_offsets else "group_normal_offset"),
        )

        metadata = deformed.setdefault("optimization_metadata", {})
        metadata["feature_constrained_deformation"] = report.to_dict()
        metadata["manufacturability_validator"] = manufacturability_report
        metadata["port_constraint_report"] = self.plan.port_constraint_report

        if output_dir is not None:
            self._write_debug_outputs(payload, deformed, variable_values, report, iteration, Path(output_dir))
        return deformed, report

    def _apply_feature_variable(
        self,
        payload: Dict[str, Any],
        variable: ShapeDesignVariable,
        value: float,
        displacements: Dict[PointKey, Point],
    ) -> None:
        feature = find_feature(self.plan.features, variable.target_id)
        if feature is None:
            return
        points = component_points(payload.get("components", [])[feature.component_index])
        if not points:
            return
        if feature.kind == "feedline":
            self._apply_feedline_variable(feature, variable.name, value, points, displacements)
        elif feature.kind == "slot":
            self._apply_slot_variable(feature, variable.name, value, points, displacements)
        elif feature.kind == "patch":
            self._apply_patch_edge_variable(feature, value, points, displacements)

    def _apply_feedline_variable(
        self,
        feature: FeatureRegion,
        name: str,
        value: float,
        points: Sequence[Point],
        displacements: Dict[PointKey, Point],
    ) -> None:
        pm = self.plan.port_model
        for index in feature.point_indices:
            point_id = make_point_id(feature.component_index, index)
            zone = self.plan.point_zones.get(point_id, NORMAL_REGION)
            if zone == PORT_CORE:
                continue
            point = points[index]
            if name == "feed_width":
                if zone != NORMAL_REGION:
                    continue
                transverse_value = point[0] if pm.axis == "vertical" else point[1]
                side = 1.0 if transverse_value >= pm.centerline else -1.0
                vector = scale_point(pm.transverse_direction, 0.5 * value * side)
            else:
                vector = scale_point(pm.propagation_direction, value)
            add_displacement(displacements, (feature.component_index, index), vector)

    def _apply_slot_variable(
        self,
        feature: FeatureRegion,
        name: str,
        value: float,
        points: Sequence[Point],
        displacements: Dict[PointKey, Point],
    ) -> None:
        x1, y1, x2, y2 = feature.bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        for index in feature.point_indices:
            point_id = make_point_id(feature.component_index, index)
            if self.plan.point_zones.get(point_id) == PORT_CORE:
                continue
            x, y = points[index]
            if name == "slot_length":
                direction = (1.0 if x >= cx else -1.0, 0.0)
                vector = scale_point(direction, 0.5 * value)
            elif name == "slot_width":
                direction = (0.0, 1.0 if y >= cy else -1.0)
                vector = scale_point(direction, 0.5 * value)
            else:
                vector = (value, 0.0)
            add_displacement(displacements, (feature.component_index, index), vector)

    def _apply_patch_edge_variable(
        self,
        feature: FeatureRegion,
        value: float,
        points: Sequence[Point],
        displacements: Dict[PointKey, Point],
    ) -> None:
        for index in feature.point_indices:
            point_id = make_point_id(feature.component_index, index)
            if self.plan.point_zones.get(point_id) != NORMAL_REGION:
                continue
            normal = point_normal(points, index)
            add_displacement(displacements, (feature.component_index, index), scale_point(normal, value))

    def _apply_group_variable(
        self,
        payload: Dict[str, Any],
        variable: ShapeDesignVariable,
        value: float,
        displacements: Dict[PointKey, Point],
    ) -> None:
        group = find_group(self.plan.groups, variable.target_id)
        if group is None:
            return
        n = max(1, len(group.point_indices) - 1)
        for local_index, point_index in enumerate(group.point_indices):
            point_id = make_point_id(group.component_index, point_index)
            if self.plan.point_zones.get(point_id) == PORT_CORE:
                continue
            weight = 0.5 - 0.5 * math.cos(math.pi * local_index / n)
            if local_index > n / 2:
                weight = 0.5 - 0.5 * math.cos(math.pi * (n - local_index) / n)
            vector = scale_point(group.normal, value * max(0.25, weight))
            add_displacement(displacements, (group.component_index, point_index), vector)

    def _write_debug_outputs(
        self,
        original: Dict[str, Any],
        deformed: Dict[str, Any],
        variables: Dict[str, float],
        report: FeatureDeformationReport,
        iteration: int,
        output_dir: Path,
    ) -> None:
        debug_dir = output_dir / "deformation_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_json(debug_dir / "feature_regions.json", [feature.to_dict() for feature in self.plan.features])
        write_json(debug_dir / "point_groups.json", [group.to_dict() for group in self.plan.groups])
        write_json(debug_dir / "normal_offset_debug.json", normal_offset_debug(self.plan, variables))
        write_json(debug_dir / "port_constraint_report.json", self.plan.port_constraint_report)
        write_json(debug_dir / "port_constraints_report.json", self.plan.port_constraint_report)
        write_json(debug_dir / "manufacturability_report.json", report.manufacturability_report)
        write_json(debug_dir / "evaluation_summary.json", report.to_dict())
        write_json(output_dir / "optimization_variable_report.json", optimization_variable_report(self.plan, variables, report))

        plot_group_overlay(original, self.plan, debug_dir / "group_overlay.png")
        plot_normal_offset_overlay(original, self.plan, variables, debug_dir / "normal_offset_overlay.png")
        plot_port_constraint_overlay(original, self.plan, debug_dir / "port_constraint_overlay.png")
        plot_feature_regions(original, self.plan, debug_dir / "feature_regions.png")
        plot_before_after(original, deformed, debug_dir / "geometry_before_after.png")
        plot_deformation_vectors(original, self.plan, variables, debug_dir / "deformation_vectors.png")


def build_shape_variables(
    features: Sequence[FeatureRegion],
    groups: Sequence[PointGroup],
    port_model: PortModel,
    max_dimensions: int = 10,
) -> List[ShapeDesignVariable]:
    variables: List[ShapeDesignVariable] = []
    feedline = first_feature(features, "feedline")
    if feedline is not None:
        width_limit = max(0.5, 0.20 * port_model.estimated_width)
        length_limit = max(0.75, 0.08 * max(port_model.bbox[2] - port_model.bbox[0], port_model.bbox[3] - port_model.bbox[1]))
        variables.extend(
            [
                ShapeDesignVariable("feed_width", -width_limit, width_limit, 0.0, 1, "feature", feedline.feature_id, "symmetric_feed_width", "Feedline width offset outside PORT_CORE/PORT_NEIGHBOR."),
                ShapeDesignVariable("feed_length", -length_limit, length_limit, 0.0, 1, "feature", feedline.feature_id, "feed_axis_translation", "Feedline length offset along propagation direction."),
                ShapeDesignVariable("inset_depth", -length_limit, length_limit, 0.0, 1, "feature", feedline.feature_id, "feed_axis_translation", "Inset depth offset along feedline propagation direction."),
            ]
        )

    slot = first_feature(features, "slot")
    if slot is not None and len(variables) + 3 <= max_dimensions:
        sx = max(0.625, 0.125 * (slot.bbox[2] - slot.bbox[0]))
        sy = max(0.625, 0.125 * (slot.bbox[3] - slot.bbox[1]))
        variables.extend(
            [
                ShapeDesignVariable("slot_length", -sx, sx, 0.0, 2, "feature", slot.feature_id, "symmetric_slot_length", "Slot length offset."),
                ShapeDesignVariable("slot_width", -sy, sy, 0.0, 2, "feature", slot.feature_id, "symmetric_slot_width", "Slot width offset."),
                ShapeDesignVariable("slot_position", -sx, sx, 0.0, 2, "feature", slot.feature_id, "slot_translation", "Slot lateral position offset."),
            ]
        )

    patch = first_feature(features, "patch")
    if patch is not None and len(variables) < max_dimensions:
        limit = max(0.625, 0.01875 * math.hypot(patch.bbox[2] - patch.bbox[0], patch.bbox[3] - patch.bbox[1]))
        variables.append(
            ShapeDesignVariable("patch_edge_offset", -limit, limit, 0.0, 3, "feature", patch.feature_id, "boundary_normal_offset", "Patch edge normal offset outside port constraints.")
        )

    if not variables:
        for group in groups[:max_dimensions]:
            variables.append(
                ShapeDesignVariable(
                    f"{group.group_id}_offset_distance",
                    -group.max_offset,
                    group.max_offset,
                    0.0,
                    4,
                    "group",
                    group.group_id,
                    "boundary_normal_offset",
                    f"Normal offset distance for {group.group_id}.",
                )
            )
    return variables[:max_dimensions]


def detect_feature_regions(
    payload: Dict[str, Any],
    point_zones: Dict[str, str],
    port_model: PortModel,
) -> List[FeatureRegion]:
    components = payload.get("components", []) or []
    component_infos = []
    for component_index, component in enumerate(components):
        points = component_points(component)
        if len(points) < 3:
            continue
        bbox = point_bbox(points)
        area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        component_infos.append((area, component_index, points, bbox))
    if not component_infos:
        return []

    component_infos.sort(reverse=True)
    largest_area, patch_component, patch_points, patch_bbox = component_infos[0]
    features: List[FeatureRegion] = []
    patch_indices = [
        index for index in range(len(patch_points))
        if point_zones.get(make_point_id(patch_component, index)) == NORMAL_REGION
    ]
    features.append(FeatureRegion("patch_001", "patch", patch_component, patch_indices, patch_bbox, 0.75, ["patch_edge_offset"]))

    feed_indices = feedline_indices(patch_points, patch_component, point_zones, port_model)
    if len(feed_indices) >= 3:
        features.append(FeatureRegion("feedline_001", "feedline", patch_component, feed_indices, point_bbox([patch_points[i] for i in feed_indices]), 0.80, ["feed_width", "feed_length", "inset_depth"]))

    slot_count = 1
    for area, component_index, points, bbox in component_infos[1:]:
        if area <= 0.65 * largest_area:
            indices = list(range(len(points)))
            features.append(FeatureRegion(f"slot_{slot_count:03d}", "slot", component_index, indices, bbox, 0.65, ["slot_length", "slot_width", "slot_position"]))
            slot_count += 1
            break

    meander_indices = high_turn_indices(patch_points)
    if len(meander_indices) >= 6:
        features.append(FeatureRegion("meander_001", "meander", patch_component, meander_indices, point_bbox([patch_points[i] for i in meander_indices]), 0.45, []))
    return features


def build_point_groups(
    payload: Dict[str, Any],
    point_zones: Dict[str, str],
    features: Sequence[FeatureRegion],
    port_model: PortModel,
    group_size: int = 5,
    max_groups: int = 12,
) -> List[PointGroup]:
    groups: List[PointGroup] = []
    group_index = 1
    for component_index, component in enumerate(payload.get("components", []) or []):
        points = component_points(component)
        if len(points) < group_size:
            continue
        run: List[int] = []
        for point_index in range(len(points)):
            point_id = make_point_id(component_index, point_index)
            if point_zones.get(point_id) == NORMAL_REGION:
                run.append(point_index)
            else:
                if len(run) >= group_size:
                    group_index = append_groups(groups, group_index, component_index, run, points, features, max_groups)
                run = []
        if len(run) >= group_size:
            group_index = append_groups(groups, group_index, component_index, run, points, features, max_groups)
        if len(groups) >= max_groups:
            break
    return groups[:max_groups]


def append_groups(
    groups: List[PointGroup],
    group_index: int,
    component_index: int,
    run: Sequence[int],
    points: Sequence[Point],
    features: Sequence[FeatureRegion],
    max_groups: int,
) -> int:
    step = max(3, len(run) // max(1, math.ceil(len(run) / 5)))
    for start in range(0, len(run) - 4, step):
        if len(groups) >= max_groups:
            break
        indices = list(run[start:start + 5])
        normals = [point_normal(points, index) for index in indices]
        normal = normalize((sum(n[0] for n in normals), sum(n[1] for n in normals)))
        local_lengths = [distance(points[indices[i - 1]], points[indices[i]]) for i in range(1, len(indices))]
        max_offset = max(0.3125, min(3.75, 0.3125 * (sum(local_lengths) / max(1, len(local_lengths)))))
        source_feature = feature_for_indices(features, component_index, indices)
        groups.append(
            PointGroup(
                group_id=f"group_{group_index:03d}",
                component_index=component_index,
                point_indices=indices,
                point_ids=[make_point_id(component_index, index) for index in indices],
                normal=normal,
                max_offset=max_offset,
                source_feature=source_feature,
            )
        )
        group_index += 1
    return group_index


def detect_port_model(payload: Dict[str, Any], port_summary: Optional[Dict[str, Any]] = None) -> PortModel:
    points = collect_payload_points(payload)
    bbox = point_bbox(points) if points else (0.0, 0.0, 1.0, 1.0)
    x1, y1, x2, y2 = bbox
    width = max(1e-9, x2 - x1)
    height = max(1e-9, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    port_side = resolve_port_side(payload, port_summary)
    if port_side is None:
        side_scores = {
            "bottom": sum(1 for x, y in points if y >= y1 + 0.72 * height and abs(x - cx) <= 0.28 * width),
            "top": sum(1 for x, y in points if y <= y1 + 0.28 * height and abs(x - cx) <= 0.28 * width),
            "left": sum(1 for x, y in points if x <= x1 + 0.28 * width and abs(y - cy) <= 0.28 * height),
            "right": sum(1 for x, y in points if x >= x1 + 0.72 * width and abs(y - cy) <= 0.28 * height),
        }
        port_side = max(side_scores, key=side_scores.get)
        if side_scores[port_side] == 0:
            port_side = "bottom"

    port_point = resolve_port_point(payload, port_summary)
    summary_width = resolve_port_width(payload, port_summary)

    if port_side in {"bottom", "top"}:
        axis = "vertical"
        propagation = (0.0, -1.0 if port_side == "bottom" else 1.0)
        transverse = (1.0, 0.0)
        side_points = [p for p in points if (p[1] >= y1 + 0.65 * height if port_side == "bottom" else p[1] <= y1 + 0.35 * height)]
        centered = [p[0] for p in side_points if abs(p[0] - cx) <= 0.35 * width]
        centerline = port_point[0] if port_point is not None else (median(centered) if centered else cx)
        estimated_width = max(1.0, summary_width if summary_width is not None else (percentile_width(centered) if len(centered) >= 2 else 0.12 * width))
        span = height
    else:
        axis = "horizontal"
        propagation = (1.0 if port_side == "left" else -1.0, 0.0)
        transverse = (0.0, 1.0)
        side_points = [p for p in points if (p[0] <= x1 + 0.35 * width if port_side == "left" else p[0] >= x1 + 0.65 * width)]
        centered = [p[1] for p in side_points if abs(p[1] - cy) <= 0.35 * height]
        centerline = port_point[1] if port_point is not None else (median(centered) if centered else cy)
        estimated_width = max(1.0, summary_width if summary_width is not None else (percentile_width(centered) if len(centered) >= 2 else 0.12 * height))
        span = width

    return PortModel(
        axis=axis,
        port_side=port_side,
        propagation_direction=propagation,
        transverse_direction=transverse,
        centerline=centerline,
        estimated_width=estimated_width,
        core_depth=max(1.0, 0.06 * span),
        neighbor_depth=max(2.0, 0.16 * span),
        bbox=bbox,
    )


def classify_port_zones(payload: Dict[str, Any], port_model: PortModel) -> Dict[str, str]:
    zones: Dict[str, str] = {}
    for component_index, component in enumerate(payload.get("components", []) or []):
        points = component_points(component)
        for point_index, point in enumerate(points):
            point_id = make_point_id(component_index, point_index)
            axial_distance = distance_from_port_side(point, port_model)
            transverse_distance = abs((point[0] if port_model.axis == "vertical" else point[1]) - port_model.centerline)
            in_feed_band = transverse_distance <= max(1.5 * port_model.estimated_width, port_model.estimated_width + 2.0)
            if in_feed_band and axial_distance <= port_model.core_depth:
                zones[point_id] = PORT_CORE
            elif in_feed_band and axial_distance <= port_model.neighbor_depth:
                zones[point_id] = PORT_NEIGHBOR
            else:
                zones[point_id] = NORMAL_REGION
    return zones


def build_port_constraint_report(
    payload: Dict[str, Any],
    point_zones: Dict[str, str],
    port_model: PortModel,
) -> Dict[str, Any]:
    by_zone = {PORT_CORE: [], PORT_NEIGHBOR: [], NORMAL_REGION: []}
    for point_id, zone in point_zones.items():
        by_zone.setdefault(zone, []).append(point_id)
    if port_model.axis == "vertical":
        allowed = "vertical/feed propagation direction"
        forbidden = "horizontal/transverse feed-width direction"
    else:
        allowed = "horizontal/feed propagation direction"
        forbidden = "vertical/transverse feed-width direction"
    return {
        "zones": by_zone,
        "feedline_axis": port_model.axis,
        "port_side": port_model.port_side,
        "propagation_direction": list(port_model.propagation_direction),
        "estimated_feed_width": port_model.estimated_width,
        "PORT_CORE": {
            "rule": "frozen; dx=0 and dy=0",
            "point_count": len(by_zone[PORT_CORE]),
        },
        "PORT_NEIGHBOR": {
            "rule": "project motion onto feed propagation direction only; preserve feed width, contact area, and port boundary length",
            "allowed_motion": allowed,
            "forbidden_motion": forbidden,
            "point_count": len(by_zone[PORT_NEIGHBOR]),
        },
        "NORMAL_REGION": {
            "rule": "feature-level or group normal offsets allowed",
            "point_count": len(by_zone[NORMAL_REGION]),
        },
    }


def validate_manufacturability(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    min_edge: Optional[float] = None
    spike_count = 0
    sharp_angle_count = 0
    curvature_jump_count = 0
    for component_index, component in enumerate(payload.get("components", []) or []):
        points = component_points(component)
        if len(points) < 3:
            continue
        for index in range(1, len(points)):
            edge = distance(points[index - 1], points[index])
            min_edge = edge if min_edge is None else min(min_edge, edge)
        previous_angle: Optional[float] = None
        for index in range(1, len(points) - 1):
            angle = turning_angle(points[index - 1], points[index], points[index + 1])
            if angle < 8.0:
                sharp_angle_count += 1
                if min(distance(points[index - 1], points[index]), distance(points[index], points[index + 1])) < 1.0:
                    spike_count += 1
            if previous_angle is not None and abs(angle - previous_angle) > 145.0:
                curvature_jump_count += 1
            previous_angle = angle
    if min_edge is not None and min_edge < 1e-6:
        errors.append(f"minimum edge length collapsed: {min_edge:.12g}")
    if spike_count:
        errors.append(f"spike detector found {spike_count} narrow spike(s)")
    if sharp_angle_count:
        warnings.append(f"sharp local angle count: {sharp_angle_count}")
    if curvature_jump_count:
        warnings.append(f"curvature jump count: {curvature_jump_count}")
    penalty = 0.25 * spike_count + 0.02 * sharp_angle_count + 0.02 * curvature_jump_count
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "minimum_edge_length": min_edge,
        "minimum_radius_proxy": min_edge,
        "sharp_angle_count": sharp_angle_count,
        "curvature_jump_count": curvature_jump_count,
        "spike_count": spike_count,
        "penalty": penalty,
        "robustness_score": max(0.0, min(1.0, 1.0 - penalty)),
    }


def optimization_variable_report(
    plan: FeatureShapePlan,
    variables: Dict[str, float],
    report: FeatureDeformationReport,
) -> Dict[str, Any]:
    feature_vars = [variable.to_dict() for variable in plan.variables if variable.category == "feature"]
    group_vars = [variable.to_dict() for variable in plan.variables if variable.category == "group"]
    port_vars = [variable.to_dict() for variable in plan.variables if variable.name in {"feed_width", "feed_length", "inset_depth"}]
    frozen = plan.port_constraint_report.get("zones", {}).get(PORT_CORE, [])
    max_offset = 0.0
    for variable in plan.variables:
        max_offset = max(max_offset, abs(variable.lower), abs(variable.upper))
    return {
        "feature_variables": feature_vars,
        "group_variables": group_vars,
        "port_variables": port_vars,
        "frozen_variables": frozen,
        "deformation_mode": report.deformation_mode,
        "max_offset": max_offset,
        "sampled_values": dict(variables),
        "manufacturability": report.manufacturability_report,
    }


def normal_offset_debug(plan: FeatureShapePlan, variables: Dict[str, float]) -> Dict[str, Any]:
    return {
        "mode": "Boundary Normal Offset",
        "groups": [
            {
                **group.to_dict(),
                "sampled_offset": float(variables.get(f"{group.group_id}_offset_distance", 0.0)),
            }
            for group in plan.groups
        ],
        "feature_normal_variables": [
            variable.to_dict()
            for variable in plan.variables
            if variable.deformation_mode == "boundary_normal_offset"
        ],
    }


def feedline_indices(
    points: Sequence[Point],
    component_index: int,
    point_zones: Dict[str, str],
    port_model: PortModel,
) -> List[int]:
    result: List[int] = []
    x1, y1, x2, y2 = port_model.bbox
    span = (y2 - y1) if port_model.axis == "vertical" else (x2 - x1)
    for index, point in enumerate(points):
        point_id = make_point_id(component_index, index)
        zone = point_zones.get(point_id, NORMAL_REGION)
        transverse = point[0] if port_model.axis == "vertical" else point[1]
        axial = distance_from_port_side(point, port_model)
        if abs(transverse - port_model.centerline) <= max(2.0 * port_model.estimated_width, 0.18 * span):
            if axial <= max(port_model.neighbor_depth + 0.35 * span, 0.25 * span):
                if zone != PORT_CORE:
                    result.append(index)
    return result


def high_turn_indices(points: Sequence[Point]) -> List[int]:
    indices = []
    for index in range(1, len(points) - 1):
        angle = turning_angle(points[index - 1], points[index], points[index + 1])
        if angle < 105.0:
            indices.append(index)
    return indices


def apply_displacements(
    payload: Dict[str, Any],
    displacements: Dict[PointKey, Point],
    point_zones: Dict[str, str],
    port_model: PortModel,
) -> List[str]:
    moved: List[str] = []
    for (component_index, point_index), vector in displacements.items():
        point_id = make_point_id(component_index, point_index)
        zone = point_zones.get(point_id, NORMAL_REGION)
        dx, dy = vector
        if zone == PORT_CORE:
            dx, dy = 0.0, 0.0
        elif zone == PORT_NEIGHBOR:
            dot = dx * port_model.propagation_direction[0] + dy * port_model.propagation_direction[1]
            dx, dy = scale_point(port_model.propagation_direction, dot)
        if abs(dx) <= 1e-15 and abs(dy) <= 1e-15:
            continue
        move_component_point(payload, component_index, point_index, dx, dy)
        moved.append(point_id)
    return moved


def add_displacement(displacements: Dict[PointKey, Point], key: PointKey, vector: Point) -> None:
    old = displacements.get(key, (0.0, 0.0))
    displacements[key] = (old[0] + vector[0], old[1] + vector[1])


def move_component_point(payload: Dict[str, Any], component_index: int, point_index: int, dx: float, dy: float) -> None:
    components = payload.get("components", []) or []
    if component_index >= len(components):
        return
    component = components[component_index]
    for key in ("resampled_points", "fallback_points", "points", "sampled_points", "smoothed_points"):
        raw = component.get(key)
        if isinstance(raw, list) and 0 <= point_index < len(raw):
            raw[point_index] = move_json_point(raw[point_index], dx, dy)
    for primitive in (component.get("primitives") or []) + (component.get("segments") or []):
        for key in ("points", "control_points", "fallback_points"):
            raw = primitive.get(key)
            if isinstance(raw, list) and 0 <= point_index < len(raw):
                raw[point_index] = move_json_point(raw[point_index], dx, dy)
    sync_closed_component_points(component)


def sync_closed_component_points(component: Dict[str, Any], tolerance: float = 1e-7) -> None:
    if not bool(component.get("closed", False)):
        return
    segment_count = len(component.get("segments") or component.get("primitives") or [])
    for key in ("resampled_points", "fallback_points", "points", "sampled_points", "smoothed_points"):
        raw = component.get(key)
        if not isinstance(raw, list) or len(raw) < 3:
            continue
        first = raw[0]
        last = raw[-1]
        if not isinstance(first, (list, tuple)) or len(first) < 2:
            continue
        explicit_closure_length = segment_count > 0 and len(raw) == segment_count + 1
        if explicit_closure_length:
            raw[-1] = [float(first[0]), float(first[1])]
        elif not isinstance(last, (list, tuple)) or len(last) < 2 or math.hypot(float(first[0]) - float(last[0]), float(first[1]) - float(last[1])) > tolerance:
            raw.append([float(first[0]), float(first[1])])
        else:
            raw[-1] = [float(first[0]), float(first[1])]


def update_component_bboxes(payload: Dict[str, Any]) -> None:
    for component in payload.get("components", []) or []:
        points = component_points(component)
        if points:
            component["bbox"] = list(point_bbox(points))


def move_json_point(point: Any, dx: float, dy: float) -> Any:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return point
    moved = list(point)
    moved[0] = float(moved[0]) + dx
    moved[1] = float(moved[1]) + dy
    return moved


def component_points(component: Dict[str, Any]) -> List[Point]:
    for key in ("resampled_points", "fallback_points", "points", "sampled_points"):
        points = parse_points(component.get(key))
        if points:
            return points
    return []


def collect_payload_points(payload: Dict[str, Any]) -> List[Point]:
    points: List[Point] = []
    for component in payload.get("components", []) or []:
        points.extend(component_points(component))
    return points


def parse_points(value: Any) -> List[Point]:
    if not isinstance(value, list):
        return []
    points: List[Point] = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                x = float(item[0])
                y = float(item[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
    return points


def point_bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def point_normal(points: Sequence[Point], index: int) -> Point:
    if len(points) < 2:
        return (0.0, 1.0)
    prev_point = points[max(0, index - 1)]
    next_point = points[min(len(points) - 1, index + 1)]
    tangent = normalize((next_point[0] - prev_point[0], next_point[1] - prev_point[1]))
    return normalize((-tangent[1], tangent[0]))


def distance_from_port_side(point: Point, port_model: PortModel) -> float:
    x1, y1, x2, y2 = port_model.bbox
    x, y = point
    if port_model.port_side == "bottom":
        return max(0.0, y2 - y)
    if port_model.port_side == "top":
        return max(0.0, y - y1)
    if port_model.port_side == "left":
        return max(0.0, x - x1)
    return max(0.0, x2 - x)


def turning_angle(a: Point, b: Point, c: Point) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(v1[0], v1[1])
    n2 = math.hypot(v2[0], v2[1])
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 180.0
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(dot))


def normalize(vector: Point) -> Point:
    norm = math.hypot(vector[0], vector[1])
    if norm <= 1e-12:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def scale_point(vector: Point, scale: float) -> Point:
    return (vector[0] * scale, vector[1] * scale)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def percentile_width(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if len(ordered) < 2:
        return 0.0
    low = ordered[int(0.1 * (len(ordered) - 1))]
    high = ordered[int(0.9 * (len(ordered) - 1))]
    return max(0.0, high - low)


def make_point_id(component_index: int, point_index: int) -> str:
    return f"c{component_index:03d}_p{point_index:03d}"


def first_feature(features: Sequence[FeatureRegion], kind: str) -> Optional[FeatureRegion]:
    for feature in features:
        if feature.kind == kind:
            return feature
    return None


def find_feature(features: Sequence[FeatureRegion], feature_id: str) -> Optional[FeatureRegion]:
    for feature in features:
        if feature.feature_id == feature_id:
            return feature
    return None


def find_group(groups: Sequence[PointGroup], group_id: str) -> Optional[PointGroup]:
    for group in groups:
        if group.group_id == group_id:
            return group
    return None


def feature_for_indices(features: Sequence[FeatureRegion], component_index: int, indices: Sequence[int]) -> str:
    index_set = set(indices)
    for feature in features:
        if feature.component_index == component_index and index_set.intersection(feature.point_indices):
            return feature.feature_id
    return "unassigned_boundary"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def plot_payload(ax: Any, payload: Dict[str, Any], color: str, linewidth: float, label: Optional[str] = None) -> None:
    first = True
    for component in payload.get("components", []) or []:
        points = component_points(component)
        if len(points) < 2:
            continue
        ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linewidth=linewidth, label=label if first else None)
        first = False


def finish_plot(ax: Any, title: str) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()


def plot_group_overlay(payload: Dict[str, Any], plan: FeatureShapePlan, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_payload(ax, payload, "#d1d5db", 0.9)
    cmap = plt.cm.get_cmap("tab20", max(1, len(plan.groups)))
    for idx, group in enumerate(plan.groups):
        points = component_points(payload.get("components", [])[group.component_index])
        xy = [points[i] for i in group.point_indices if i < len(points)]
        if not xy:
            continue
        ax.scatter([p[0] for p in xy], [p[1] for p in xy], s=26, color=cmap(idx), label=group.group_id)
    if plan.groups:
        ax.legend(fontsize=6, loc="best")
    finish_plot(ax, "Point Groups")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_normal_offset_overlay(payload: Dict[str, Any], plan: FeatureShapePlan, variables: Dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_payload(ax, payload, "#cbd5e1", 0.9)
    for group in plan.groups:
        points = component_points(payload.get("components", [])[group.component_index])
        xy = [points[i] for i in group.point_indices if i < len(points)]
        if not xy:
            continue
        cx = sum(p[0] for p in xy) / len(xy)
        cy = sum(p[1] for p in xy) / len(xy)
        sampled = float(variables.get(f"{group.group_id}_offset_distance", group.max_offset))
        scale = sampled if abs(sampled) > 1e-12 else group.max_offset
        ax.arrow(cx, cy, group.normal[0] * scale, group.normal[1] * scale, head_width=0.35, color="#dc2626", length_includes_head=True)
    finish_plot(ax, "Normal Offset Regions")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_port_constraint_overlay(payload: Dict[str, Any], plan: FeatureShapePlan, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_payload(ax, payload, "#e5e7eb", 0.9)
    colors = {PORT_CORE: "#dc2626", PORT_NEIGHBOR: "#f59e0b", NORMAL_REGION: "#2563eb"}
    labels = set()
    for component_index, component in enumerate(payload.get("components", []) or []):
        points = component_points(component)
        for point_index, point in enumerate(points):
            zone = plan.point_zones.get(make_point_id(component_index, point_index), NORMAL_REGION)
            label = zone if zone not in labels else None
            ax.scatter([point[0]], [point[1]], s=18, color=colors.get(zone, "#6b7280"), label=label, alpha=0.85)
            labels.add(zone)
    ax.legend(fontsize=8, loc="best")
    finish_plot(ax, "Port Constraint Zones")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_feature_regions(payload: Dict[str, Any], plan: FeatureShapePlan, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_payload(ax, payload, "#d1d5db", 0.9)
    colors = {"feedline": "#7c3aed", "patch": "#059669", "slot": "#dc2626", "branch": "#0284c7", "meander": "#f97316"}
    plotted = set()
    for feature in plan.features:
        points = component_points(payload.get("components", [])[feature.component_index])
        xy = [points[i] for i in feature.point_indices if i < len(points)]
        if not xy:
            continue
        label = feature.kind if feature.kind not in plotted else None
        ax.scatter([p[0] for p in xy], [p[1] for p in xy], s=22, color=colors.get(feature.kind, "#111827"), label=label, alpha=0.85)
        plotted.add(feature.kind)
    if plotted:
        ax.legend(fontsize=8, loc="best")
    finish_plot(ax, "Feature Regions")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_before_after(original: Dict[str, Any], deformed: Dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_payload(axes[0], original, "#6b7280", 1.1)
    finish_plot(axes[0], "Original Geometry")
    plot_payload(axes[1], deformed, "#2563eb", 1.1)
    finish_plot(axes[1], "Mutated Geometry")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_deformation_vectors(payload: Dict[str, Any], plan: FeatureShapePlan, variables: Dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_payload(ax, payload, "#cbd5e1", 0.9)
    for variable in plan.variables:
        value = float(variables.get(variable.name, 0.0))
        if abs(value) <= 1e-15:
            continue
        if variable.category == "group":
            group = find_group(plan.groups, variable.target_id)
            if group is None:
                continue
            points = component_points(payload.get("components", [])[group.component_index])
            xy = [points[i] for i in group.point_indices if i < len(points)]
            if not xy:
                continue
            cx = sum(p[0] for p in xy) / len(xy)
            cy = sum(p[1] for p in xy) / len(xy)
            vector = scale_point(group.normal, value)
        else:
            feature = find_feature(plan.features, variable.target_id)
            if feature is None:
                continue
            x1, y1, x2, y2 = feature.bbox
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            if variable.name in {"feed_length", "inset_depth"}:
                vector = scale_point(plan.port_model.propagation_direction, value)
            elif variable.name == "feed_width":
                vector = scale_point(plan.port_model.transverse_direction, value)
            else:
                vector = (value, 0.0)
        ax.arrow(cx, cy, vector[0], vector[1], head_width=0.35, color="#dc2626", length_includes_head=True)
        ax.text(cx + vector[0], cy + vector[1], variable.name, fontsize=7)
    finish_plot(ax, "Region Deformation Vectors")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
