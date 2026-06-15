from __future__ import annotations

"""控制点 offset 应用、变形记录与调试可视化模块。

本文件只负责局部形变和调试输出：
1. 读取 ControlPointOptimizationPlan 中选出的控制点；
2. 将 BO 给出的 dx/dy 应用到几何缓存点；
3. 记录每轮 deformation history；
4. 输出点分类、位移、覆盖范围等调试图。

注意：这里不改变 BO objective，也不直接调用 CST。
"""

import copy
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bayesian_optimization.geometry.geometry_constraint_validator import (
    GeometryConstraintReport,
    GeometryConstraintValidator,
)
from bayesian_optimization.optimization.constrained_shape_optimizer import ControlPointOptimizationPlan


# =============================================================================
# 数据结构：单轮变形记录
# =============================================================================


@dataclass
class DeformationRecord:
    """单轮 evaluation 中控制点位移与验证结果的记录。"""
    iteration: int
    moved_points: List[str]
    offsets: Dict[str, float]
    validation_result: Dict[str, Any]
    repair_actions: List[str]
    objective_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 友好的字典。"""
        return asdict(self)


class ControlPointDeformer:
    """受约束控制点位移变形器。

    只应用 P' = P + delta 的局部形变，不生成新拓扑。
    """

    # 【关键类】受约束控制点位移变形器。
    # 核心原则：只应用 P' = P + delta 的局部形变，不生成新拓扑。
    def __init__(
        self,
        plan: ControlPointOptimizationPlan,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """保存控制点计划并初始化几何约束验证器。"""
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
        """【关键函数】将 BO 输出的 dx/dy 应用到控制点。"""
        deformed = copy.deepcopy(payload)
        moved_points: List[str] = []
        applied_offsets: Dict[str, Tuple[float, float]] = {}

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
            applied_offsets[constraint.point_id] = (dx, dy)

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
        if output_dir is not None:
            self._write_debug_outputs(payload, repaired, moved_points, applied_offsets, report, iteration, output_dir)
        return repaired, report

    def validate_deformation(
        self,
        payload: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> GeometryConstraintReport:
        """对变形后的几何执行约束验证。"""
        return self.validator.validate(payload, output_dir=output_dir)

    def repair_geometry(
        self,
        payload: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Tuple[Dict[str, Any], GeometryConstraintReport]:
        """执行保守几何修复，例如 snap、去重、闭合等。"""
        return self.validator.validate_and_repair(payload, output_dir=output_dir)

    def _move_control_point(
        self,
        payload: Dict[str, Any],
        component_index: int,
        point_index: int,
        dx: float,
        dy: float,
    ) -> None:
        """同步移动 component 缓存点和 primitive 内相关点。"""
        components = payload.get("components", []) or []
        if component_index >= len(components):
            return
        component = components[component_index]

        for key in ("resampled_points", "fallback_points", "points"):
            points = component.get(key)
            if isinstance(points, list) and 0 <= point_index < len(points):
                points[point_index] = move_json_point(points[point_index], dx, dy)

        self._move_matching_primitive_points(component, point_index, dx, dy)
        self._sync_closed_component_points(component)

    def _move_matching_primitive_points(
        self,
        component: Dict[str, Any],
        point_index: int,
        dx: float,
        dy: float,
    ) -> None:
        """保守同步 line/arc/spline primitive 中的点字段。"""
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

    @staticmethod
    def _sync_closed_component_points(component: Dict[str, Any], tolerance: float = 1e-7) -> None:
        """Keep duplicated closure endpoints locked after point movement."""

        if not bool(component.get("closed", False)):
            return
        segment_count = len(component.get("segments") or component.get("primitives") or [])
        for key in ("resampled_points", "fallback_points", "points"):
            points = component.get(key)
            if not isinstance(points, list) or len(points) < 3:
                continue
            first = points[0]
            last = points[-1]
            if not isinstance(first, (list, tuple)) or len(first) < 2:
                continue
            explicit_closure_length = segment_count > 0 and len(points) == segment_count + 1
            if explicit_closure_length:
                points[-1] = [float(first[0]), float(first[1])]
            elif not isinstance(last, (list, tuple)) or len(last) < 2 or math.hypot(float(first[0]) - float(last[0]), float(first[1]) - float(last[1])) > tolerance:
                points.append([float(first[0]), float(first[1])])
            else:
                points[-1] = [float(first[0]), float(first[1])]

    def _record_history(
        self,
        iteration: int,
        moved_points: List[str],
        offsets: Dict[str, float],
        report: GeometryConstraintReport,
        objective_value: Optional[float],
        output_dir: Optional[Path],
    ) -> None:
        """写入 deformation_history.json。"""
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

    def _write_debug_outputs(
        self,
        original: Dict[str, Any],
        deformed: Dict[str, Any],
        moved_points: List[str],
        applied_offsets: Dict[str, Tuple[float, float]],
        report: GeometryConstraintReport,
        iteration: int,
        output_dir: Path,
    ) -> None:
        """【关键函数】输出每轮控制点调试 JSON 与 PNG。"""
        debug_dir = Path(output_dir) / "deformation_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_json(debug_dir / "point_classification.json", self.plan.point_classification or {})
        write_json(debug_dir / "point_selection_scores.json", self.plan.selection_scores or {})
        write_json(debug_dir / "port_constraints_report.json", self.plan.port_constraints_report or {})
        write_json(debug_dir / "feedline_groups.json", self.plan.feedline_groups or {})
        write_json(debug_dir / "symmetry_groups.json", self.plan.symmetry_groups or [])
        write_json(debug_dir / "selection_quota_report.json", self.plan.selection_quota_report or {})
        write_json(debug_dir / "point_distribution_report.json", self.plan.point_distribution_report or {})
        write_json(debug_dir / "feedline_selection_report.json", self.plan.feedline_selection_report or {})

        summary = self._evaluation_summary(iteration, moved_points, applied_offsets, report)
        write_json(debug_dir / "evaluation_summary.json", summary)
        write_json(debug_dir / "deformation_statistics.json", self._deformation_statistics(iteration, applied_offsets))

        self._plot_moved_points_overlay(original, deformed, moved_points, debug_dir / "moved_points_overlay.png")
        self._plot_displacement_vectors(original, applied_offsets, debug_dir / "displacement_vectors.png")
        self._plot_point_ids(original, debug_dir / "point_id_overlay.png")
        self._plot_top_selected_points(original, debug_dir / "top_selected_points.png")
        self._plot_before_after(original, deformed, debug_dir / "geometry_before_after.png")
        self._plot_symmetry_debug(original, debug_dir / "symmetry_debug.png")
        self._plot_selected_points_heatmap(original, debug_dir / "selected_points_heatmap.png")
        self._plot_selection_category_overlay(original, debug_dir / "selection_category_overlay.png")
        self._plot_coverage_map(original, debug_dir / "coverage_map.png")
        self._plot_displacement_histogram(applied_offsets, debug_dir / "displacement_histogram.png")

    def _evaluation_summary(
        self,
        iteration: int,
        moved_points: List[str],
        applied_offsets: Dict[str, Tuple[float, float]],
        report: GeometryConstraintReport,
    ) -> Dict[str, Any]:
        """生成 evaluation_summary.json 的内容。"""
        displacements = [math.hypot(dx, dy) for dx, dy in applied_offsets.values()]
        classification = self.plan.point_classification or {}
        return {
            "evaluation_id": iteration,
            "selected_points": [constraint.point_id for constraint in self.plan.constraints],
            "moved_points": moved_points,
            "frozen_points": self.plan.frozen_points or [],
            "port_points": [point_id for point_id, cls in classification.items() if cls == "PORT"],
            "feedline_points": [point_id for point_id, cls in classification.items() if cls == "FEEDLINE"],
            "symmetry_groups": self.plan.symmetry_groups or [],
            "max_displacement": max(displacements) if displacements else 0.0,
            "mean_displacement": sum(displacements) / len(displacements) if displacements else 0.0,
            "validation_result": report.to_dict(),
        }

    def _deformation_statistics(
        self,
        iteration: int,
        applied_offsets: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Any]:
        """生成 deformation_statistics.json 的真实位移统计。"""
        displacements = [math.hypot(dx, dy) for dx, dy in applied_offsets.values()]
        classification = self.plan.point_classification or {}
        selected = [constraint.point_id for constraint in self.plan.constraints]
        return {
            "evaluation_id": iteration,
            "selected_points": len(selected),
            "mean_displacement": sum(displacements) / len(displacements) if displacements else 0.0,
            "max_displacement": max(displacements) if displacements else 0.0,
            "min_displacement": min(displacements) if displacements else 0.0,
            "resonant_points": sum(1 for point_id in selected if classification.get(point_id) == "RESONANT"),
            "feedline_points": sum(1 for point_id in selected if classification.get(point_id) == "FEEDLINE"),
            "structural_points": sum(1 for point_id in selected if classification.get(point_id) == "STRUCTURAL"),
            "moved_points": len(applied_offsets),
        }

    def _plot_moved_points_overlay(self, original, deformed, moved_points, path: Path) -> None:
        """绘制原始/变形轮廓与已移动控制点。"""
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_payload(ax, original, color="#9ca3af", linewidth=1.0, label="original")
        plot_payload(ax, deformed, color="#2563eb", linewidth=1.3, label="deformed")
        classification = self.plan.point_classification or {}
        for constraint in self.plan.constraints:
            point = get_component_point(deformed, constraint.component_index, constraint.point_index)
            if point is None:
                continue
            cls = classification.get(constraint.point_id, "STRUCTURAL")
            marker = "s" if cls in {"PORT", "FEEDLINE"} else "o"
            color = "#facc15" if cls == "PORT" else "#a855f7" if cls == "FEEDLINE" else "#22c55e"
            if constraint.point_id in moved_points:
                color = "#dc2626"
            ax.scatter([point[0]], [point[1]], s=34, marker=marker, color=color, zorder=4)
        finish_plot(ax, "Moved Points Overlay")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_displacement_vectors(self, original, applied_offsets, path: Path) -> None:
        """绘制控制点位移向量。"""
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_payload(ax, original, color="#9ca3af", linewidth=1.0)
        by_id = {constraint.point_id: constraint for constraint in self.plan.constraints}
        for point_id, (dx, dy) in applied_offsets.items():
            constraint = by_id.get(point_id)
            if not constraint:
                continue
            point = get_component_point(original, constraint.component_index, constraint.point_index)
            if point is None:
                continue
            ax.arrow(point[0], point[1], dx, dy, head_width=0.18, color="#dc2626", length_includes_head=True)
            ax.text(point[0] + dx, point[1] + dy, f"{dx:.2f},{dy:.2f}", fontsize=7)
        finish_plot(ax, "Displacement Vectors")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_point_ids(self, payload, path: Path) -> None:
        """绘制已选控制点 ID，方便人工定位。"""
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_payload(ax, payload, color="#6b7280", linewidth=1.0)
        for constraint in self.plan.constraints:
            point = get_component_point(payload, constraint.component_index, constraint.point_index)
            if point is not None:
                ax.text(point[0], point[1], constraint.point_id, fontsize=6, color="#111827")
        finish_plot(ax, "Point ID Overlay")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_top_selected_points(self, payload, path: Path) -> None:
        """绘制最终进入 BO 的控制点及其 selection score。"""
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_payload(ax, payload, color="#9ca3af", linewidth=1.0)
        scores = self.plan.selection_scores or {}
        selected = [constraint.point_id for constraint in self.plan.constraints]
        values = [float(scores.get(point_id, {}).get("score", 0.0)) for point_id in selected]
        max_score = max(values) if values else 1.0
        for constraint, score in zip(self.plan.constraints, values):
            point = get_component_point(payload, constraint.component_index, constraint.point_index)
            if point is None:
                continue
            ax.scatter([point[0]], [point[1]], s=42, color=plt.cm.viridis(score / max_score if max_score else 0.0))
        finish_plot(ax, "Top Selected Points")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_before_after(self, original, deformed, path: Path) -> None:
        """并排绘制变形前后几何。"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_payload(axes[0], original, color="#6b7280", linewidth=1.1)
        finish_plot(axes[0], "Original Geometry")
        plot_payload(axes[1], deformed, color="#2563eb", linewidth=1.1)
        finish_plot(axes[1], "Mutated Geometry")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_symmetry_debug(self, payload, path: Path) -> None:
        """绘制镜像控制点配对关系。"""
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_payload(ax, payload, color="#9ca3af", linewidth=1.0)
        constraints = {constraint.point_id: constraint for constraint in self.plan.constraints}
        all_constraints = constraints
        for group in self.plan.symmetry_groups or []:
            points = []
            for point_id in group.get("points", []):
                constraint = all_constraints.get(point_id)
                if constraint:
                    point = get_component_point(payload, constraint.component_index, constraint.point_index)
                    if point:
                        points.append(point)
            if len(points) == 2:
                ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], "--", color="#f97316")
        finish_plot(ax, "Symmetry Debug")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_selected_points_heatmap(self, payload, path: Path) -> None:
        """绘制全部候选控制点的 selection score 热力图。"""
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_payload(ax, payload, color="#d1d5db", linewidth=0.9)
        scores = self.plan.selection_scores or {}
        xs: List[float] = []
        ys: List[float] = []
        values: List[float] = []
        for point_id, score_info in scores.items():
            point = point_by_id(payload, point_id)
            if point is None:
                continue
            xs.append(point[0])
            ys.append(point[1])
            values.append(float(score_info.get("score", 0.0)))
        if xs:
            scatter = ax.scatter(xs, ys, c=values, s=22, cmap="viridis", alpha=0.9)
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="selection score")
        finish_plot(ax, "Selected Points Heatmap")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_selection_category_overlay(self, payload, path: Path) -> None:
        """按 PORT/FEEDLINE/RESONANT/STRUCTURAL 分类绘制控制点。"""
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_payload(ax, payload, color="#d1d5db", linewidth=0.9)
        style = {
            "PORT": ("#facc15", "s"),
            "FEEDLINE": ("#a855f7", "s"),
            "RESONANT": ("#ef4444", "o"),
            "STRUCTURAL": ("#2563eb", "o"),
        }
        plotted = set()
        for point_id, cls in (self.plan.point_classification or {}).items():
            point = point_by_id(payload, point_id)
            if point is None:
                continue
            color, marker = style.get(cls, ("#6b7280", "."))
            label = cls if cls not in plotted else None
            ax.scatter([point[0]], [point[1]], s=26, marker=marker, color=color, label=label, alpha=0.85)
            plotted.add(cls)
        if plotted:
            ax.legend(loc="best", fontsize=8)
        finish_plot(ax, "Selection Category Overlay")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_coverage_map(self, payload, path: Path) -> None:
        """绘制已选优化点的空间覆盖范围。"""
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_payload(ax, payload, color="#d1d5db", linewidth=0.9)
        selected = [constraint.point_id for constraint in self.plan.constraints]
        spacing = float((self.plan.point_distribution_report or {}).get("minimum_point_spacing", 20.0))
        for constraint in self.plan.constraints:
            point = get_component_point(payload, constraint.component_index, constraint.point_index)
            if point is None:
                continue
            circle = plt.Circle(point, spacing, fill=False, color="#10b981", alpha=0.18, linewidth=1.0)
            ax.add_patch(circle)
            ax.scatter([point[0]], [point[1]], s=45, color="#059669", zorder=4)
        ax.text(
            0.02,
            0.98,
            f"selected={len(selected)}, spacing={spacing:g}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
        )
        finish_plot(ax, "Coverage Map")
        fig.savefig(path, dpi=180)
        plt.close(fig)

    def _plot_displacement_histogram(self, applied_offsets, path: Path) -> None:
        """绘制本轮实际位移量分布。"""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        values = [math.hypot(dx, dy) for dx, dy in applied_offsets.values()]
        if values:
            ax.hist(values, bins=min(12, max(3, len(values))), color="#2563eb", edgecolor="#111827", alpha=0.85)
        else:
            ax.text(0.5, 0.5, "No non-zero displacement", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlabel("Displacement")
        ax.set_ylabel("Count")
        ax.set_title("Displacement Histogram")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def move_json_point(point: Any, dx: float, dy: float) -> Any:
    """移动 JSON 中的二维/三维点，保留原列表长度。"""
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return point
    moved = list(point)
    moved[0] = float(moved[0]) + dx
    moved[1] = float(moved[1]) + dy
    return moved


def clamp(value: float, lower: float, upper: float) -> float:
    """把数值限制在闭区间 [lower, upper]。"""
    return max(lower, min(upper, value))


def write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 写入 JSON 调试文件。"""
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def get_component_point(payload: Dict[str, Any], component_index: int, point_index: int) -> Optional[Tuple[float, float]]:
    """按 component/point 下标读取二维点。"""
    components = payload.get("components", []) or []
    if component_index >= len(components):
        return None
    points = components[component_index].get("resampled_points") or components[component_index].get("fallback_points") or []
    if not isinstance(points, list) or point_index >= len(points):
        return None
    point = points[point_index]
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    return float(point[0]), float(point[1])


def point_by_id(payload: Dict[str, Any], point_id: str) -> Optional[Tuple[float, float]]:
    """按 c000_p001 格式的 point_id 读取二维点。"""
    try:
        component_part, point_part = point_id.split("_")
        component_index = int(component_part[1:])
        point_index = int(point_part[1:])
    except Exception:
        return None
    return get_component_point(payload, component_index, point_index)


def plot_payload(ax: Any, payload: Dict[str, Any], color: str, linewidth: float, label: Optional[str] = None) -> None:
    """绘制 payload 中所有 component 的二维折线。"""
    first = True
    for component in payload.get("components", []) or []:
        points = component.get("resampled_points") or component.get("fallback_points") or []
        xy = [(float(p[0]), float(p[1])) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(xy) < 2:
            continue
        ax.plot([p[0] for p in xy], [p[1] for p in xy], color=color, linewidth=linewidth, label=label if first else None)
        first = False


def finish_plot(ax: Any, title: str) -> None:
    """统一设置调试图坐标、标题和网格样式。"""
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.invert_yaxis()
