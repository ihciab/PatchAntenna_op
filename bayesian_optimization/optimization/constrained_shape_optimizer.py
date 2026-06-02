from __future__ import annotations

"""受约束控制点选择与 BO 变量生成模块。

本文件只负责：
1. 从 parameterization JSON 中枚举控制点；
2. 给控制点分类、评分、冻结 PORT 点；
3. 按 quota 和空间分散约束选择 BO 可动点；
4. 为每个可动点生成 dx/dy 搜索范围。

注意：这里不调用 CST、不计算目标函数、不修改原始 parameterization schema。
"""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point2D = Tuple[float, float]
CLASS_PORT = "PORT"
CLASS_FEEDLINE = "FEEDLINE"
CLASS_RESONANT = "RESONANT"
CLASS_STRUCTURAL = "STRUCTURAL"
NON_FEED_PORT_RANGE_MULTIPLIER = 1.25


# =============================================================================
# 数据结构：约束、变量、优化计划
# =============================================================================


@dataclass(frozen=True)
class ControlPointConstraint:
    """单个控制点的移动约束。"""
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
        """转换为 JSON 友好的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class ControlPointDesignVariable:
    """传递给 BO 后端的一维设计变量，例如 c010_p012_dx。"""
    name: str
    lower: float
    upper: float
    default: float
    point_id: str
    axis: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 友好的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class ControlPointOptimizationPlan:
    """一次 BO 运行使用的控制点选择计划和调试报告集合。"""
    constraints: List[ControlPointConstraint]
    variables: List[ControlPointDesignVariable]
    source: str
    enabled: bool
    point_classification: Dict[str, str] = None
    selection_scores: Dict[str, Dict[str, float]] = None
    port_constraints_report: Dict[str, Any] = None
    feedline_groups: Dict[str, Any] = None
    symmetry_groups: List[Dict[str, Any]] = None
    frozen_points: List[str] = None
    selection_quota_report: Dict[str, Any] = None
    point_distribution_report: Dict[str, Any] = None
    feedline_selection_report: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 友好的优化计划字典。"""
        return {
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "variables": [variable.to_dict() for variable in self.variables],
            "source": self.source,
            "enabled": self.enabled,
            "point_classification": self.point_classification or {},
            "selection_scores": self.selection_scores or {},
            "port_constraints_report": self.port_constraints_report or {},
            "feedline_groups": self.feedline_groups or {},
            "symmetry_groups": self.symmetry_groups or [],
            "frozen_points": self.frozen_points or [],
            "selection_quota_report": self.selection_quota_report or {},
            "point_distribution_report": self.point_distribution_report or {},
            "feedline_selection_report": self.feedline_selection_report or {},
        }


def load_constraint_config(path: Path) -> Dict[str, Any]:
    """读取控制点约束配置；文件不存在时使用默认配置。"""
    if not path.exists():
        return default_constraint_config()
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"control point constraint config must be a JSON object: {path}")
    return {**default_constraint_config(), **data}


def default_constraint_config() -> Dict[str, Any]:
    """默认控制点策略配置，外部 JSON 可以覆盖这些字段。"""
    return {
        "enabled": True,
        "auto_select": True,
        "max_movable_points": 12,
        "max_dimensions": 24,
        "default_dx_range": [-1.0, 1.0],
        "default_dy_range": [-1.0, 1.0],
        "class_ranges": {
            CLASS_PORT: {"dx_range": [0.0, 0.0], "dy_range": [0.0, 0.0]},
            CLASS_STRUCTURAL: {"dx_range": [-2.0, 2.0], "dy_range": [-2.0, 2.0]},
            CLASS_FEEDLINE: {"dx_range": [-4.0, 4.0], "dy_range": [-4.0, 4.0]},
            CLASS_RESONANT: {"dx_range": [-6.0, 6.0], "dy_range": [-6.0, 6.0]},
        },
        "selection_quota": {
            CLASS_RESONANT: 4,
            CLASS_FEEDLINE: 3,
            CLASS_STRUCTURAL: 3,
            "EXPLORATION": 2,
        },
        "minimum_point_spacing": 20.0,
        "displacement_ratio": 0.2,
        "global_feature_scale_ratio": 0.12,
        "resonance_weight": 4.0,
        "random_seed": 42,
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
    """【关键函数】从几何 JSON 构建完整控制点 BO 计划。

    主流程：
    1. 读取配置；
    2. 枚举控制点；
    3. 分类、评分、识别 symmetry/feedline/port；
    4. 按 quota 和 spacing 选择可动点；
    5. 生成 dx/dy 变量。
    """
    config = load_constraint_config(constraint_path)
    if not bool(config.get("enabled", True)):
        return ControlPointOptimizationPlan([], [], str(constraint_path), enabled=False)

    explicit_constraints = config.get("point_constraints") or {}
    point_map = enumerate_control_points(payload)
    point_classification = classify_control_points(payload, point_map)
    selection_scores = compute_selection_scores(payload, point_map, point_classification, config)
    symmetry_groups = detect_symmetry_groups(point_map)
    feedline_groups = detect_feedline_groups(point_map, point_classification)
    port_report = build_port_constraints_report(point_classification)

    candidates = extract_candidate_control_points(payload, config, point_classification, selection_scores)
    constraints = apply_constraint_overrides(candidates, explicit_constraints, config)

    port_ids = {point_id for point_id, cls in point_classification.items() if cls == CLASS_PORT}
    removed_port_points = [constraint.point_id for constraint in constraints if constraint.point_id in port_ids and constraint.movable]
    constraints = [
        constraint for constraint in constraints
        if not (constraint.point_id in port_ids and constraint.movable)
    ]
    port_report["removed_from_bo_variables"] = removed_port_points
    port_report["warnings"] = [
        f"{point_id} classified as PORT and removed from BO variables"
        for point_id in removed_port_points
    ]

    selected_constraints, quota_report, distribution_report, feedline_selection_report = select_by_quota(
        constraints=constraints,
        point_classification=point_classification,
        selection_scores=selection_scores,
        config=config,
    )
    max_points = int(config.get("max_movable_points", 12))
    max_dimensions = int(config.get("max_dimensions", 24))
    movable_constraints = selected_constraints[: max(0, min(max_points, max_dimensions // 2))]

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
        point_classification=point_classification,
        selection_scores=selection_scores,
        port_constraints_report=port_report,
        feedline_groups=feedline_groups,
        symmetry_groups=symmetry_groups,
        frozen_points=[point_id for point_id, cls in point_classification.items() if cls == CLASS_PORT],
        selection_quota_report=quota_report,
        point_distribution_report=distribution_report,
        feedline_selection_report=feedline_selection_report,
    )


def extract_candidate_control_points(
    payload: Dict[str, Any],
    config: Dict[str, Any],
    point_classification: Dict[str, str],
    selection_scores: Dict[str, Dict[str, float]],
) -> List[ControlPointConstraint]:
    """从所有控制点中提取“理论上可移动”的候选点。

    这里先过滤 PORT 点和拓扑端点，再根据类别与局部尺度生成 dx/dy 范围。
    真正进入 BO 的点会在后续 select_by_quota 中再次筛选。
    """
    default_dx = tuple(float(v) for v in config.get("default_dx_range", [-1.0, 1.0]))
    default_dy = tuple(float(v) for v in config.get("default_dy_range", [-1.0, 1.0]))
    min_endpoint_distance = int((config.get("selection_strategy") or {}).get("min_endpoint_distance_index", 2))
    freeze_endpoints = bool(config.get("freeze_topology_endpoints", True))
    global_bbox = payload_bbox(payload)
    global_feature_floor = float(config.get("global_feature_scale_ratio", 0.04)) * math.hypot(
        global_bbox[2] - global_bbox[0],
        global_bbox[3] - global_bbox[1],
    )

    candidates: List[Tuple[float, ControlPointConstraint]] = []
    for component_index, component in enumerate(payload.get("components", []) or []):
        points = component_points(component)
        if len(points) < 5:
            continue
        for point_index in range(1, len(points) - 1):
            point_id = make_point_id(component_index, point_index)
            if point_classification.get(point_id) == CLASS_PORT:
                continue
            is_near_endpoint = point_index < min_endpoint_distance or point_index >= len(points) - min_endpoint_distance
            if freeze_endpoints and is_near_endpoint:
                continue

            score_info = selection_scores.get(point_id, {})
            score = float(score_info.get("score", 0.0))
            if score <= 1e-6:
                continue
            priority = "high" if score >= 4.0 else "medium"
            point_class = point_classification.get(point_id, CLASS_STRUCTURAL)
            feature_scale = local_feature_scale(points, point_index)
            if point_class in {CLASS_RESONANT, CLASS_FEEDLINE}:
                feature_scale = max(feature_scale, global_feature_floor)
            elif point_class == CLASS_STRUCTURAL:
                feature_scale = max(feature_scale, 0.5 * global_feature_floor)
            dx_range, dy_range = point_ranges(
                point_class,
                feature_scale,
                config,
            )
            candidates.append(
                (
                    score,
                    ControlPointConstraint(
                        point_id=point_id,
                        component_index=component_index,
                        point_index=point_index,
                        original_point=points[point_index],
                        movable=True,
                        dx_range=dx_range,
                        dy_range=dy_range,
                        priority=priority,
                        reason=f"auto selected point, score={score:.6f}",
                    ),
                )
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [constraint for _, constraint in candidates]


def select_by_quota(
    constraints: List[ControlPointConstraint],
    point_classification: Dict[str, str],
    selection_scores: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> Tuple[List[ControlPointConstraint], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """【关键函数】按类别 quota 选择最终 BO 控制点。

    该函数替代旧的纯 Top-K selection，避免所有点集中到同一条边。
    """
    quota = config.get("selection_quota") or {}
    spacing = float(config.get("minimum_point_spacing", 20.0))
    rng = random.Random(int(config.get("random_seed", 42)))
    by_id = {constraint.point_id: constraint for constraint in constraints}
    selected: List[ControlPointConstraint] = []
    selected_ids = set()

    def sorted_class(cls: str) -> List[ControlPointConstraint]:
        """返回某一类别内按 score 降序排列的候选点。"""
        items = [
            constraint for constraint in constraints
            if point_classification.get(constraint.point_id) == cls and constraint.movable
        ]
        items.sort(key=lambda item: selection_scores.get(item.point_id, {}).get("score", 0.0), reverse=True)
        return items

    quota_result: Dict[str, Any] = {"requested": quota, "selected": {}, "spacing": spacing}
    for cls in (CLASS_RESONANT, CLASS_FEEDLINE, CLASS_STRUCTURAL):
        requested = int(quota.get(cls, 0))
        picked = pick_spaced(sorted_class(cls), selected, selected_ids, selection_scores, requested, spacing)
        selected.extend(picked)
        selected_ids.update(item.point_id for item in picked)
        quota_result["selected"][cls] = [item.point_id for item in picked]

    remaining = [item for item in constraints if item.point_id not in selected_ids and item.movable]
    remaining.sort(key=lambda item: selection_scores.get(item.point_id, {}).get("score", 0.0), reverse=True)
    exploration_count = int(quota.get("EXPLORATION", 0))
    exploration_pool = remaining[: max(exploration_count * 4, exploration_count)]
    rng.shuffle(exploration_pool)
    exploration = pick_spaced(exploration_pool, selected, selected_ids, selection_scores, exploration_count, spacing)
    selected.extend(exploration)
    selected_ids.update(item.point_id for item in exploration)
    quota_result["selected"]["EXPLORATION"] = [item.point_id for item in exploration]

    distribution = point_distribution_report(selected, spacing)
    feedline_selected = [item.point_id for item in selected if point_classification.get(item.point_id) == CLASS_FEEDLINE]
    feedline_report = {
        "minimum_required": int(quota.get(CLASS_FEEDLINE, 0)),
        "selected_count": len(feedline_selected),
        "selected_points": feedline_selected,
        "satisfied": len(feedline_selected) >= int(quota.get(CLASS_FEEDLINE, 0)),
    }
    if not feedline_report["satisfied"]:
        feedline_report["warning"] = "Feedline quota not satisfied; not enough safe feedline candidates."
    return selected, quota_result, distribution, feedline_report


def pick_spaced(
    candidates: List[ControlPointConstraint],
    selected: List[ControlPointConstraint],
    selected_ids: set,
    selection_scores: Dict[str, Dict[str, float]],
    count: int,
    spacing: float,
) -> List[ControlPointConstraint]:
    """在候选点中优先挑选满足最小间距的点；不足时保守补齐。"""
    picked: List[ControlPointConstraint] = []
    for candidate in candidates:
        if len(picked) >= count:
            break
        if candidate.point_id in selected_ids:
            continue
        if any(point_distance(candidate.original_point, item.original_point) < spacing for item in selected + picked):
            continue
        picked.append(candidate)
    if len(picked) < count:
        for candidate in candidates:
            if len(picked) >= count:
                break
            if candidate.point_id in selected_ids or candidate in picked:
                continue
            picked.append(candidate)
    return picked


def point_distribution_report(selected: List[ControlPointConstraint], spacing: float) -> Dict[str, Any]:
    """生成已选点空间分布报告，用于检查变量是否过度聚集。"""
    distances = []
    for i, item in enumerate(selected):
        for other in selected[i + 1:]:
            distances.append(point_distance(item.original_point, other.original_point))
    min_distance = min(distances) if distances else None
    xs = [item.original_point[0] for item in selected]
    ys = [item.original_point[1] for item in selected]
    return {
        "minimum_point_spacing": spacing,
        "selected_points": [item.point_id for item in selected],
        "min_pairwise_distance": min_distance,
        "spacing_satisfied": min_distance is None or min_distance >= spacing,
        "coverage_bbox": [min(xs), min(ys), max(xs), max(ys)] if xs and ys else None,
        "selected_count": len(selected),
    }


def enumerate_control_points(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """枚举每个 component 中的缓存点，并生成稳定 point_id。"""
    points: Dict[str, Dict[str, Any]] = {}
    for component_index, component in enumerate(payload.get("components", []) or []):
        component_pts = component_points(component)
        for point_index, point in enumerate(component_pts):
            point_id = make_point_id(component_index, point_index)
            points[point_id] = {
                "component_index": component_index,
                "point_index": point_index,
                "point": point,
                "component_point_count": len(component_pts),
                "start_node": component.get("start_node"),
                "end_node": component.get("end_node"),
            }
    return points


def classify_control_points(payload: Dict[str, Any], point_map: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """【关键函数】按几何启发式给控制点分类。

    分类结果包括 PORT、FEEDLINE、RESONANT、STRUCTURAL。
    这是变量选择层的输入，不修改几何本身。
    """
    classification: Dict[str, str] = {}
    bbox = payload_bbox(payload)
    width = max(1e-9, bbox[2] - bbox[0])
    height = max(1e-9, bbox[3] - bbox[1])
    for point_id, info in point_map.items():
        point_index = int(info["point_index"])
        count = int(info["component_point_count"])
        x, y = info["point"]
        component = payload.get("components", [])[int(info["component_index"])]
        points = component_points(component)
        curvature = local_curvature_score(points, point_index) if 0 < point_index < len(points) - 1 else 0.0
        near_endpoint = point_index <= 1 or point_index >= count - 2
        near_bottom_feed = y > bbox[1] + 0.72 * height and abs(x - 0.5 * (bbox[0] + bbox[2])) < 0.28 * width
        near_outer_boundary = (
            abs(x - bbox[0]) < 0.08 * width
            or abs(x - bbox[2]) < 0.08 * width
            or abs(y - bbox[1]) < 0.08 * height
            or abs(y - bbox[3]) < 0.08 * height
        )
        internal_turn = curvature > 0.35 and not near_outer_boundary
        edge_corner_or_slot = curvature > 0.22 and near_outer_boundary

        if near_endpoint:
            classification[point_id] = CLASS_PORT if near_bottom_feed else CLASS_STRUCTURAL
        elif near_bottom_feed:
            classification[point_id] = CLASS_FEEDLINE
        elif internal_turn or edge_corner_or_slot:
            classification[point_id] = CLASS_RESONANT
        else:
            classification[point_id] = CLASS_STRUCTURAL
    return classification


def compute_selection_scores(
    payload: Dict[str, Any],
    point_map: Dict[str, Dict[str, Any]],
    point_classification: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """【关键函数】计算控制点选择分数。

    score = curvature + resonance + symmetry - topology_risk。
    分数只用于排序和报告，PORT 点仍由冻结规则强制排除。
    """
    scores: Dict[str, Dict[str, float]] = {}
    symmetry_groups = detect_symmetry_groups(point_map)
    symmetry_points = {item for group in symmetry_groups for item in group.get("points", [])}
    bbox = payload_bbox(payload)
    width = max(1e-9, bbox[2] - bbox[0])
    height = max(1e-9, bbox[3] - bbox[1])
    resonance_weight = float(config.get("resonance_weight", 4.0))
    for point_id, info in point_map.items():
        component = payload.get("components", [])[info["component_index"]]
        points = component_points(component)
        point_index = info["point_index"]
        curvature = local_curvature_score(points, point_index) if 0 < point_index < len(points) - 1 else 0.0
        cls = point_classification.get(point_id, CLASS_STRUCTURAL)
        resonance = 0.0
        x, y = info["point"]
        center_x = 0.5 * (bbox[0] + bbox[2])
        near_feed_junction = (
            bbox[1] + 0.55 * height <= y <= bbox[1] + 0.80 * height
            and abs(x - center_x) < 0.22 * width
        )
        local_scale = local_feature_scale(points, point_index) if points else 0.0
        compact_feature_bonus = 1.0 if 0.0 < local_scale < 0.08 * max(width, height) else 0.0
        if cls == CLASS_RESONANT:
            resonance = resonance_weight + 1.5 * min(1.0, curvature) + compact_feature_bonus
        elif cls == CLASS_FEEDLINE:
            resonance = 0.55 * resonance_weight + (1.0 if near_feed_junction else 0.0)
        elif near_feed_junction and curvature > 0.12:
            resonance = 0.75 * resonance_weight
        symmetry = 1.5 if point_id in symmetry_points else 0.0
        risk = 0.0
        if cls == CLASS_PORT:
            risk += 100.0
        if point_index <= 2 or point_index >= len(points) - 3:
            risk += 2.0
        if cls == CLASS_FEEDLINE:
            risk += 0.8
        score = curvature + resonance + symmetry - risk
        scores[point_id] = {
            "score": score,
            "curvature": curvature,
            "resonance": resonance,
            "symmetry": symmetry,
            "risk": risk,
            "local_feature_scale": local_scale,
        }
    return scores


def detect_symmetry_groups(point_map: Dict[str, Dict[str, Any]], tolerance: float = 1.5) -> List[Dict[str, Any]]:
    """检测左右镜像控制点组，供评分和调试可视化使用。"""
    if not point_map:
        return []
    xs = [info["point"][0] for info in point_map.values()]
    axis_x = 0.5 * (min(xs) + max(xs))
    groups: List[Dict[str, Any]] = []
    used = set()
    for point_id, info in point_map.items():
        if point_id in used:
            continue
        x, y = info["point"]
        mirror_x = 2 * axis_x - x
        best_id = None
        best_dist = tolerance
        for other_id, other in point_map.items():
            if other_id == point_id or other_id in used:
                continue
            ox, oy = other["point"]
            dist = math.hypot(ox - mirror_x, oy - y)
            if dist < best_dist:
                best_id = other_id
                best_dist = dist
        if best_id:
            groups.append({"group_id": f"sym_{len(groups):03d}", "axis_x": axis_x, "points": [point_id, best_id]})
            used.add(point_id)
            used.add(best_id)
    return groups


def detect_feedline_groups(point_map: Dict[str, Dict[str, Any]], classification: Dict[str, str]) -> Dict[str, Any]:
    """根据分类结果生成 feedline group 报告。"""
    feed_points = [point_id for point_id, cls in classification.items() if cls == CLASS_FEEDLINE]
    return {
        "feedline_group_000": {
            "points": feed_points,
            "left_edge_points": feed_points[::2],
            "right_edge_points": feed_points[1::2],
            "parameters": ["feed_width", "feed_length"],
            "constraint": "keep centerline continuity; avoid one-sided random displacement",
        }
    } if feed_points else {}


def build_port_constraints_report(classification: Dict[str, str]) -> Dict[str, Any]:
    """生成 PORT 点冻结报告。"""
    port_points = [point_id for point_id, cls in classification.items() if cls == CLASS_PORT]
    return {
        "port_points": port_points,
        "frozen": port_points,
        "rule": "PORT points are frozen and must not enter BO variable space.",
        "removed_from_bo_variables": [],
        "warnings": [],
    }


def payload_bbox(payload: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """计算整个几何 payload 的二维包围盒。"""
    all_points: List[Point2D] = []
    for component in payload.get("components", []) or []:
        all_points.extend(component_points(component))
    xs = [point[0] for point in all_points] or [0.0]
    ys = [point[1] for point in all_points] or [0.0]
    return min(xs), min(ys), max(xs), max(ys)


def apply_constraint_overrides(
    candidates: List[ControlPointConstraint],
    explicit_constraints: Dict[str, Any],
    config: Dict[str, Any],
) -> List[ControlPointConstraint]:
    """应用 control_point_constraints.json 中的人工覆盖配置。"""
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
    """把 component/point 下标转换成稳定控制点 ID。"""
    return f"c{component_index:03d}_p{point_index:03d}"


def parse_point_id(point_id: str) -> Optional[Tuple[int, int]]:
    """把 c000_p001 格式的 point_id 解析回下标。"""
    try:
        c_part, p_part = point_id.split("_")
        return int(c_part[1:]), int(p_part[1:])
    except Exception:
        return None


def component_points(component: Dict[str, Any]) -> List[Point2D]:
    """从 component 中读取 CST/调试使用的二维缓存点。"""
    raw_points = component.get("resampled_points") or component.get("fallback_points") or component.get("points") or []
    points: List[Point2D] = []
    for point in raw_points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    return points


def point_distance(a: Point2D, b: Point2D) -> float:
    """计算两个二维点之间的欧氏距离。"""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def local_feature_scale(points: Sequence[Point2D], index: int) -> float:
    """计算控制点附近的局部几何尺度，用于约束最大位移。

    注意：parameterization 输出常常是密集采样点，直接使用相邻采样点距离会把
    BO 范围压得过小。因此这里同时考虑局部窗口 chord 与组件 bbox 尺度。
    """
    lengths: List[float] = []
    if 0 < index < len(points):
        lengths.append(point_distance(points[index], points[index - 1]))
    if 0 <= index < len(points) - 1:
        lengths.append(point_distance(points[index], points[index + 1]))
    valid = [length for length in lengths if length > 1e-12]

    window = min(5, max(1, len(points) // 12))
    chord_lengths: List[float] = []
    if index - window >= 0:
        chord_lengths.append(point_distance(points[index], points[index - window]))
    if index + window < len(points):
        chord_lengths.append(point_distance(points[index], points[index + window]))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox_scale = 0.06 * math.hypot(max(xs) - min(xs), max(ys) - min(ys)) if xs and ys else 0.0
    candidates = valid + chord_lengths + [bbox_scale]
    candidates = [value for value in candidates if value > 1e-12]
    return max(candidates) if candidates else 0.0


def point_ranges(
    point_class: str,
    feature_scale: float,
    config: Dict[str, Any],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """【关键函数】按控制点类别与局部尺度共同确定 dx/dy 搜索范围。

    类别范围提供全局上限；局部尺度上限用于避免小特征被 BO 一次拉坏。
    """
    if point_class == CLASS_PORT:
        return (0.0, 0.0), (0.0, 0.0)

    class_ranges = config.get("class_ranges") or {}
    range_config = class_ranges.get(point_class) or class_ranges.get(CLASS_STRUCTURAL) or {}
    base_dx = tuple(float(v) for v in range_config.get("dx_range", config.get("default_dx_range", [-1.0, 1.0])))
    base_dy = tuple(float(v) for v in range_config.get("dy_range", config.get("default_dy_range", [-1.0, 1.0])))

    ratio = float(config.get("displacement_ratio", 0.2))
    adaptive_limit = abs(feature_scale) * ratio if feature_scale > 1e-12 else max(abs(base_dx[0]), abs(base_dx[1]), abs(base_dy[0]), abs(base_dy[1]))
    if point_class not in {CLASS_PORT, CLASS_FEEDLINE}:
        adaptive_limit *= NON_FEED_PORT_RANGE_MULTIPLIER
    # 保留一个很小的下限，避免过密采样点导致变量范围被压成近似 0。
    adaptive_limit = max(0.5, adaptive_limit)

    def bounded_range(base: Tuple[float, float]) -> Tuple[float, float]:
        """把类别范围限制在自适应位移上限内。"""
        lower, upper = min(base), max(base)
        limit = min(max(abs(lower), abs(upper)), adaptive_limit)
        return -limit, limit

    return bounded_range(base_dx), bounded_range(base_dy)


def local_curvature_score(points: Sequence[Point2D], index: int) -> float:
    """计算局部转角曲率分数，拐角越尖分数越高。"""
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
