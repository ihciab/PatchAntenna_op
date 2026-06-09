from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesian_optimization.simulation.cst_library_path import ensure_cst_library_path

ensure_cst_library_path()

import Simulink.handle as ch
import cst.results


Point = Tuple[float, float]


@dataclass
class CSTParametricConfig:
    project_folder: Path
    project_name: str = "Parameterized_FSS"
    unit: str = "mm"
    frequency_unit: str = "GHz"
    size_x: float = 36.0
    size_y: float = 36.0
    f0: float = 6.0
    f1: float = 14.0
    component: str = "layer0"
    metal_material: str = "PEC"
    metal_thickness: float = 0.035
    substrate_material: Optional[str] = "Rogers RT-duroid 5880 (loss free)"
    substrate_thickness: float = 0.6
    add_ground: bool = True
    close_project: bool = False
    save_project: bool = True
    run_solver: bool = True
    port_summary_path: Optional[Path] = PROJECT_ROOT / "port_summary.json"
    curve_method: str = "polygon"
    simplify_tolerance_px: float = 0.0
    geometry_frame: str = "svg"

    @property
    def cst_path(self) -> Path:
        name = self.project_name
        if not name.lower().endswith(".cst"):
            name = f"{name}.cst"
        return self.project_folder / name


class ParameterizedJsonCSTBuilder:
    """Build CST geometry from NewParams / VTracer parameterization JSON.

    The first integration path is intentionally conservative: each closed
    component is drawn from its `resampled_points`, then extruded as a metal
    sheet. Segment metadata remains available for future native line/arc/spline
    drawing, but the sampled path is the most robust CST handoff format.
    """

    def __init__(self, json_path: Path | str, config: CSTParametricConfig):
        self.json_path = Path(json_path)
        self.config = config
        self.payload = self._load_json(self.json_path)
        self.modeler = None
        self.design_environment = None
        self.cst_project = None
        self._project_closed = False

    def build(self) -> Path:
        self._open_or_create_project()
        self._load_materials()

        components = self._components()
        if not components:
            raise ValueError(f"No components found in parameterization JSON: {self.json_path}")

        bbox = self._geometry_bbox(components)
        print(f"[ParameterizedJsonCSTBuilder] geometry frame: {self.config.geometry_frame}, bbox={bbox}")
        self._draw_substrate_and_ground()

        for index, component in enumerate(components):
            closed = self._is_closed_component(component)
            points = self._prepare_component_points(component, closed=closed)
            min_points = 3 if closed else 2
            if len(points) < min_points:
                print(f"[ParameterizedJsonCSTBuilder] skip component {index}: less than {min_points} points")
                continue

            cst_points = self._map_points_to_cst(points, bbox)
            curve_name = f"param_curve_{index:03d}"
            solid_name = f"param_solid_{index:03d}"

            draw_cmd = self._component_curve_command(
                name=curve_name,
                curve=self.config.component,
                component=component,
                fallback_points=cst_points,
                bbox=bbox,
                closed=closed,
            )
            self.modeler.add_to_history(f"draw {curve_name}", draw_cmd)

            extrude_cmd = ch.cst_extrudecurve(
                name=solid_name,
                curve=f"{self.config.component}:{curve_name}",
                component=self.config.component,
                material=self.config.metal_material,
                thickness=self.config.metal_thickness,
            )
            self.modeler.add_to_history(f"extrude {solid_name}", extrude_cmd)

            holes = self._component_holes(component)
            if holes:
                print(
                    "[ParameterizedJsonCSTBuilder] subtract holes: "
                    f"{solid_name}, holes={len(holes)}"
                )
            for hole_index, hole in enumerate(holes, start=1):
                hole_points = self._prepare_component_points(hole, closed=True)
                if len(hole_points) < 3:
                    print(
                        "[ParameterizedJsonCSTBuilder] skip hole "
                        f"{index}:{hole_index}: less than 3 points"
                    )
                    continue

                hole_cst_points = self._map_points_to_cst(hole_points, bbox)
                hole_curve_name = f"param_hole_curve_{index:03d}_{hole_index:03d}"
                hole_solid_name = f"param_hole_solid_{index:03d}_{hole_index:03d}"
                hole_draw_cmd = self._component_curve_command(
                    name=hole_curve_name,
                    curve=self.config.component,
                    component=hole,
                    fallback_points=hole_cst_points,
                    bbox=bbox,
                    closed=True,
                )
                self.modeler.add_to_history(f"draw {hole_curve_name}", hole_draw_cmd)

                hole_extrude_cmd = ch.cst_extrudecurve(
                    name=hole_solid_name,
                    curve=f"{self.config.component}:{hole_curve_name}",
                    component=self.config.component,
                    material=self.config.metal_material,
                    thickness=self.config.metal_thickness,
                )
                self.modeler.add_to_history(f"extrude {hole_solid_name}", hole_extrude_cmd)
                subtract_cmd = ch.cst_del_subtract(
                    self.config.component,
                    solid_name,
                    self.config.component,
                    hole_solid_name,
                )
                self.modeler.add_to_history(f"subtract {hole_solid_name} from {solid_name}", subtract_cmd)

        self.modeler.add_to_history("delete parameter curves", ch.cst_del_curves(self.config.component))

        if self.config.run_solver:
            self._add_waveguide_port(bbox)
            self._run_solver()

        if self.config.save_project and not self._project_closed:
            self.cst_project.save()
        if self.config.close_project and not self._project_closed:
            ch.cst_close_project(self.design_environment, self.cst_project, save_flag=self.config.save_project)
            self._project_closed = True

        return self.config.cst_path

    def _open_or_create_project(self) -> None:
        self.config.project_folder.mkdir(parents=True, exist_ok=True)
        self._validate_frequency_range()
        ch.cst_create_project(str(self.config.cst_path))
        self.design_environment, self.cst_project, _ = ch.cst_open_project(str(self.config.cst_path))
        self.modeler = self.cst_project.modeler
        ch.cst_auto_init(self.modeler, self.config.f0, self.config.f1)

    def _validate_frequency_range(self) -> None:
        """Validate the CST solver frequency range before project initialization."""

        if not math.isfinite(float(self.config.f0)) or not math.isfinite(float(self.config.f1)):
            raise ValueError(
                f"CST frequency range must be finite, got f0={self.config.f0}, f1={self.config.f1}"
            )
        if float(self.config.f1) <= float(self.config.f0):
            raise ValueError(
                f"CST frequency range requires f1 > f0, got f0={self.config.f0}, f1={self.config.f1}"
            )

    def _load_materials(self) -> None:
        if self.config.substrate_material and self.config.substrate_material != "PEC":
            command = ch.cst_load_material(self.config.substrate_material)
            if command:
                self.modeler.add_to_history(f"load {self.config.substrate_material}", command)

    def _add_waveguide_port(self, bbox: Tuple[float, float, float, float]) -> None:
        if self.config.port_summary_path is None:
            raise ValueError("run_solver=True requires a port summary path.")
        port_summary_path = Path(self.config.port_summary_path)
        if not port_summary_path.exists():
            raise FileNotFoundError(
                f"Port summary JSON does not exist: {port_summary_path}. "
                "Run with --build-only if you only want CST geometry."
            )

        ports = self._load_json(port_summary_path)
        if ports.get("border_contact_mode") == "separate":
            print(
                "Warning: detected PEC edge does not touch the image border; "
                "the auto-generated waveguide port may be unstable."
            )

        if self._add_patch_topology_port_if_available(ports, bbox):
            return

        edge = ports.get("closest_edge")
        if not isinstance(edge, list) or len(edge) < 2:
            raise ValueError(f"Invalid closest_edge in port summary: {port_summary_path}")

        p1 = (float(edge[0][0]), float(edge[0][1]))
        p2 = (float(edge[1][0]), float(edge[1][1]))
        cst_p1, cst_p2 = self._map_points_to_cst([p1, p2], bbox)

        x1, x2 = sorted([cst_p1[0], cst_p2[0]])
        y1, y2 = sorted([cst_p1[1], cst_p2[1]])
        orientation = self._resolve_port_orientation(ports, p1, p2)
        padding = max(self.config.substrate_thickness, self.config.metal_thickness, 0.1) * 3.0
        z1 = -max(self.config.substrate_thickness, self.config.metal_thickness) * 1.5
        z2 = max(self.config.substrate_thickness, self.config.metal_thickness) * 2.5

        if abs(p2[0] - p1[0]) < abs(p2[1] - p1[1]):
            port_x1 = self._floor_value(x1)
            port_x2 = self._ceil_value(x2)
            port_y1 = self._floor_value(y1 - padding)
            port_y2 = self._ceil_value(y2 + padding)
            set_port = ch.cst_waveguide_port(
                orientation,
                port_x1,
                port_x2,
                port_y1,
                port_y2,
                self._floor_value(z1),
                self._ceil_value(z2),
            )
        else:
            port_x1 = self._floor_value(x1 - padding)
            port_x2 = self._ceil_value(x2 + padding)
            port_y1 = self._floor_value(y1)
            port_y2 = self._ceil_value(y2)
            set_port = ch.cst_waveguide_port(
                orientation,
                port_x1,
                port_x2,
                port_y1,
                port_y2,
                self._floor_value(z1),
                self._ceil_value(z2),
            )
        print(
            "[ParameterizedJsonCSTBuilder] add waveguide port: "
            f"orientation={orientation}, "
            f"x=({port_x1}, {port_x2}), "
            f"y=({port_y1}, {port_y2}), "
            f"z=({self._floor_value(z1)}, {self._ceil_value(z2)})"
        )
        self.modeler.add_to_history("set waveguide port", set_port)

    def _add_patch_topology_port_if_available(
        self,
        ports: Dict[str, Any],
        bbox: Tuple[float, float, float, float],
    ) -> bool:
        patch_detection = ports.get("patch_port_detection")
        if not isinstance(patch_detection, dict):
            return False

        candidates = patch_detection.get("ports")
        if not isinstance(candidates, list) or not candidates:
            return False

        candidate = candidates[0]
        if not isinstance(candidate, dict):
            return False

        point = candidate.get("point")
        direction = str(candidate.get("direction", "")).lower()
        if not isinstance(point, list) or len(point) != 2:
            return False
        if direction not in {"left", "right", "top", "bottom"}:
            return False

        x = float(point[0])
        y = float(point[1])
        local_width_px = max(2.0, float(candidate.get("local_width", 2.0)))
        half_width = max(1.0, local_width_px * 0.5)
        raw_x, raw_y = x, y
        snapped_point = self._snap_patch_port_to_reconstructed_geometry(
            point=(x, y),
            direction=direction,
            local_width_px=local_width_px,
        )
        snap_applied = False
        if snapped_point is not None:
            x, y = snapped_point
            snap_applied = not (
                math.isclose(x, raw_x, abs_tol=1e-6)
                and math.isclose(y, raw_y, abs_tol=1e-6)
            )

        # 拓扑端口表示的是馈线入口点。这里用入口点附近、垂直于馈线方向的
        # 一小段截面来生成 CST waveguide port，避免继续使用上边界 closest_edge。
        if direction in {"top", "bottom"}:
            image_p1 = (x - half_width, y)
            image_p2 = (x + half_width, y)
            cst_p1, cst_p2 = self._map_points_to_cst([image_p1, image_p2], bbox)
            center = self._map_points_to_cst([(x, y)], bbox)[0]
            x1, x2 = sorted([cst_p1[0], cst_p2[0]])
            y1 = y2 = center[1]
        else:
            image_p1 = (x, y - half_width)
            image_p2 = (x, y + half_width)
            cst_p1, cst_p2 = self._map_points_to_cst([image_p1, image_p2], bbox)
            center = self._map_points_to_cst([(x, y)], bbox)[0]
            x1 = x2 = center[0]
            y1, y2 = sorted([cst_p1[1], cst_p2[1]])

        orientation = self._orientation_from_patch_direction(direction)
        min_port_span = max(self.config.substrate_thickness, self.config.metal_thickness, 0.1) * 2.0
        if direction in {"top", "bottom"} and abs(x2 - x1) < min_port_span:
            mid = (x1 + x2) / 2.0
            x1 = mid - min_port_span / 2.0
            x2 = mid + min_port_span / 2.0
        if direction in {"left", "right"} and abs(y2 - y1) < min_port_span:
            mid = (y1 + y2) / 2.0
            y1 = mid - min_port_span / 2.0
            y2 = mid + min_port_span / 2.0

        z1 = -max(self.config.substrate_thickness, self.config.metal_thickness) * 1.5
        z2 = max(self.config.substrate_thickness, self.config.metal_thickness) * 2.5
        port_x1 = self._fmt(x1)
        port_x2 = self._fmt(x2)
        port_y1 = self._fmt(y1)
        port_y2 = self._fmt(y2)
        set_port = ch.cst_waveguide_port(
            orientation,
            port_x1,
            port_x2,
            port_y1,
            port_y2,
            self._floor_value(z1),
            self._ceil_value(z2),
        )
        print(
            "[ParameterizedJsonCSTBuilder] add topology patch port: "
            f"orientation={orientation}, "
            f"point=({x:.2f}, {y:.2f}), "
            f"raw_point=({raw_x:.2f}, {raw_y:.2f}), "
            f"snapped_to_reconstructed_geometry={snap_applied}, "
            f"local_width_px={local_width_px:.2f}, "
            f"x=({port_x1}, {port_x2}), "
            f"y=({port_y1}, {port_y2}), "
            f"z=({self._floor_value(z1)}, {self._ceil_value(z2)})"
        )
        self.modeler.add_to_history("set topology patch waveguide port", set_port)
        return True

    def _snap_patch_port_to_reconstructed_geometry(
        self,
        *,
        point: Point,
        direction: str,
        local_width_px: float,
    ) -> Optional[Point]:
        """Snap the image-detected port to the reconstructed PEC terminal.

        端口检测使用原图/掩膜坐标，CST 金属来自参数化 primitives。
        两者端面可能相差少量像素；如果不处理，port 在 CST 里会看起来
        与馈线断开。这里只沿馈线传播方向吸附到重建几何端面，
        不改变求解器、边界、介质或金属几何。
        """

        if not hasattr(self, "payload"):
            return None
        try:
            components = self._components()
        except Exception:
            return None

        points: list[Point] = []
        for component in components:
            points.extend(self._component_points(component))
        if not points:
            return None

        x, y = point
        transverse_half = max(6.0, local_width_px * 0.8)
        max_axis_gap = max(12.0, local_width_px * 2.0)

        if direction in {"top", "bottom"}:
            candidates = [
                (px, py)
                for px, py in points
                if abs(px - x) <= transverse_half and abs(py - y) <= max_axis_gap
            ]
            if not candidates:
                return None
            terminal_y = max(py for _, py in candidates) if direction == "bottom" else min(py for _, py in candidates)
            terminal_band = [(px, py) for px, py in candidates if abs(py - terminal_y) <= 1.5]
            snapped_x = sum(px for px, _ in terminal_band) / max(1, len(terminal_band))
            return float(snapped_x), float(terminal_y)

        candidates = [
            (px, py)
            for px, py in points
            if abs(py - y) <= transverse_half and abs(px - x) <= max_axis_gap
        ]
        if not candidates:
            return None
        terminal_x = max(px for px, _ in candidates) if direction == "right" else min(px for px, _ in candidates)
        terminal_band = [(px, py) for px, py in candidates if abs(px - terminal_x) <= 1.5]
        snapped_y = sum(py for _, py in terminal_band) / max(1, len(terminal_band))
        return float(terminal_x), float(snapped_y)

    @staticmethod
    def _orientation_from_patch_direction(direction: str) -> str:
        orientation_map = {
            "left": "xmin",
            "right": "xmax",
            "top": "ymax",
            "bottom": "ymin",
        }
        return orientation_map[direction]

    def _run_solver(self) -> None:
        print("[ParameterizedJsonCSTBuilder] save project before solver")
        self.cst_project.save()

        print("[ParameterizedJsonCSTBuilder] start CST solver")
        solver = self.modeler.run_solver()
        if not solver:
            raise ValueError("CST solver failed. Please inspect the project geometry, port and mesh settings.")

        print("[ParameterizedJsonCSTBuilder] solver finished")
        self.modeler.add_to_history(
            f"ExportImage{self.config.cst_path.stem}",
            ch.cst_export_pic(str(self.config.project_folder), self.config.cst_path.stem),
        )
        print("[ParameterizedJsonCSTBuilder] save and close project before reading results")
        ch.cst_close_project(self.design_environment, self.cst_project, save_flag=True)
        self._project_closed = True
        self._export_s11()

    def _export_s11(self) -> None:
        try:
            project = cst.results.ProjectFile(str(self.config.cst_path), allow_interactive=True)
            s11 = project.get_3d().get_result_item(r"1D Results\S-Parameters\S1,1").get_data()
        except Exception as exc:
            print(f"[ParameterizedJsonCSTBuilder] S11 export skipped: {exc}")
            return

        s11_path = self.config.project_folder / f"{self.config.cst_path.stem}_s11.txt"
        with s11_path.open("w", encoding="utf-8") as file:
            print(s11, file=file)
        print(f"[ParameterizedJsonCSTBuilder] S11 saved: {s11_path}")

    def _resolve_port_orientation(self, ports: Dict[str, Any], p1: Point, p2: Point) -> str:
        border_sides = ports.get("closest_border_sides", [])
        axis = "x" if abs(p2[0] - p1[0]) < abs(p2[1] - p1[1]) else "y"

        # Image y grows downward, while _map_points_to_cst flips y.
        orientation_map = {
            ("x", "left"): "xmin",
            ("x", "right"): "xmax",
            ("y", "top"): "ymax",
            ("y", "bottom"): "ymin",
        }
        for side in border_sides:
            orientation = orientation_map.get((axis, side))
            if orientation is not None:
                return orientation

        return f"{axis}max"

    def _draw_substrate_and_ground(self) -> None:
        if self.config.substrate_material and self.config.substrate_thickness > 0:
            self.modeler.add_to_history(
                "draw substrate",
                self._create_brick_range(
                    name="substrate",
                    x1=-self.config.size_x / 2,
                    x2=self.config.size_x / 2,
                    y1=-self.config.size_y / 2,
                    y2=self.config.size_y / 2,
                    z1=-self.config.substrate_thickness,
                    z2=0.0,
                    component=self.config.component,
                    material=self.config.substrate_material,
                ),
            )

        if self.config.add_ground:
            self.modeler.add_to_history(
                "draw ground",
                self._create_brick_range(
                    name="ground",
                    x1=-self.config.size_x / 2,
                    x2=self.config.size_x / 2,
                    y1=-self.config.size_y / 2,
                    y2=self.config.size_y / 2,
                    z1=-(self.config.substrate_thickness + self.config.metal_thickness),
                    z2=-self.config.substrate_thickness,
                    component=self.config.component,
                    material=self.config.metal_material,
                ),
            )

    @staticmethod
    def _create_brick_range(
        name: str,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
        z1: float,
        z2: float,
        component: str,
        material: str,
    ) -> str:
        return "\n".join(
            [
                "With Brick",
                ".Reset",
                f'.Name "{name}"',
                f'.Component "{component}"',
                f'.Material "{material}"',
                f'.Xrange "{x1}", "{x2}"',
                f'.Yrange "{y1}", "{y2}"',
                f'.Zrange "{z1}", "{z2}"',
                ".Create",
                "End With",
            ]
        )

    def _curve_command(self, name: str, curve: str, points: Sequence[Point], closed: bool) -> str:
        if closed:
            return self._closed_polygon_command(name=name, curve=curve, points=points)
        if self.config.curve_method == "spline":
            return ch.cst_spline_curves(name=name, curve=curve, contour=points)
        return ch.cst_curves(name=name, curve=curve, contour=points)

    def _component_curve_command(
        self,
        name: str,
        curve: str,
        component: Dict[str, Any],
        fallback_points: Sequence[Point],
        bbox: Tuple[float, float, float, float],
        closed: bool,
    ) -> str:
        """Prefer compact primitives, fallback to sampled polygon.

        中文说明：
        新 JSON 可能包含 compact primitives。这里优先把 primitives 转成 CST
        polygon/spline 可接受的点序列；如果 primitive 数据不完整或不闭合，则回退到
        resampled_points/fallback_points，保持原 pipeline 的稳健性。
        """
        primitives = component.get("primitives") or []
        if primitives:
            try:
                primitive_points = self._points_from_primitives(primitives, bbox=bbox, closed=closed)
                primitive_points = self._remove_consecutive_duplicates(primitive_points)
                if closed:
                    primitive_points = self._remove_trailing_closure(primitive_points)
                    if len(primitive_points) >= 3:
                        primitive_points.append(primitive_points[0])
                min_points = 3 if closed else 2
                if len(primitive_points) >= min_points:
                    print(
                        "[ParameterizedJsonCSTBuilder] use compact primitives: "
                        f"{name}, primitives={len(primitives)}, points={len(primitive_points)}"
                    )
                    return self._curve_command(name=name, curve=curve, points=primitive_points, closed=closed)
            except Exception as exc:
                print(
                    "[ParameterizedJsonCSTBuilder] primitive reconstruction failed; "
                    f"fallback to sampled points for {name}: {exc}"
                )

        return self._curve_command(name=name, curve=curve, points=fallback_points, closed=closed)

    def _points_from_primitives(
        self,
        primitives: Sequence[Dict[str, Any]],
        bbox: Tuple[float, float, float, float],
        closed: bool,
    ) -> List[Point]:
        points: List[Point] = []
        for primitive in primitives:
            primitive_type = str(primitive.get("type", primitive.get("kind", "spline"))).lower()
            if primitive_type == "line":
                raw_points = self._line_primitive_points(primitive)
            elif primitive_type == "arc":
                raw_points = self._arc_primitive_points(primitive)
            else:
                raw_points = self._spline_primitive_points(primitive)
            if len(raw_points) < 2:
                continue
            mapped = self._map_points_to_cst(raw_points, bbox)
            if points and mapped:
                if self._distance(points[-1], mapped[0]) <= 1e-7:
                    points.extend(mapped[1:])
                else:
                    points.extend(mapped)
            else:
                points.extend(mapped)

        if closed and len(points) >= 3 and self._distance(points[0], points[-1]) > 1e-7:
            points.append(points[0])
        return points

    @staticmethod
    def _line_primitive_points(primitive: Dict[str, Any]) -> List[Point]:
        start = primitive.get("start")
        end = primitive.get("end")
        if start is not None and end is not None:
            return [(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]
        points = primitive.get("points") or []
        return ParameterizedJsonCSTBuilder._parse_points(points)

    @staticmethod
    def _spline_primitive_points(primitive: Dict[str, Any]) -> List[Point]:
        points = primitive.get("control_points") or primitive.get("points") or []
        return ParameterizedJsonCSTBuilder._parse_points(points)

    @staticmethod
    def _arc_primitive_points(primitive: Dict[str, Any], max_step_deg: float = 8.0) -> List[Point]:
        center = primitive.get("center")
        start = primitive.get("start")
        end = primitive.get("end")
        radius = primitive.get("radius")
        if center is None or start is None or end is None or radius is None:
            return ParameterizedJsonCSTBuilder._parse_points(primitive.get("points") or [])

        cx, cy = float(center[0]), float(center[1])
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        r = float(radius)
        a0 = math.atan2(sy - cy, sx - cx)
        a1 = math.atan2(ey - cy, ex - cx)
        clockwise = bool(primitive.get("clockwise", False))
        if clockwise and a1 > a0:
            a1 -= 2.0 * math.pi
        if not clockwise and a1 < a0:
            a1 += 2.0 * math.pi

        sweep = a1 - a0
        count = max(3, int(math.ceil(abs(math.degrees(sweep)) / max(1.0, max_step_deg))) + 1)
        points = []
        for index in range(count):
            t = index / max(1, count - 1)
            angle = a0 + sweep * t
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        return points

    def _closed_polygon_command(self, name: str, curve: str, points: Sequence[Point]) -> str:
        if len(points) < 4:
            raise ValueError(f"Closed polygon needs at least 3 unique points plus closure: {name}")
        if self._distance(points[0], points[-1]) > 1e-9:
            raise ValueError(f"Closed polygon points are not exactly closed: {name}")

        first_x, first_y = points[0]
        command = [
            "With Polygon",
            ".Reset",
            f'.Name "{name}"',
            f'.Curve "{curve}"',
            f'.Point "{self._fmt(first_x)}", "{self._fmt(first_y)}"',
        ]
        for x, y in points[1:]:
            command.append(f'.LineTo "{self._fmt(x)}", "{self._fmt(y)}"')
        command.extend([".Create", "End With"])
        return "\n".join(command)

    def _components(self) -> List[Dict[str, Any]]:
        components = self.payload.get("components", [])
        if not isinstance(components, list):
            raise ValueError("Invalid parameterization JSON: `components` must be a list.")
        return components

    def _component_points(self, component: Dict[str, Any]) -> List[Point]:
        points = component.get("resampled_points") or component.get("fallback_points") or component.get("points") or []
        return self._parse_points(points)

    @staticmethod
    def _component_holes(component: Dict[str, Any]) -> List[Dict[str, Any]]:
        holes = component.get("holes", [])
        if not isinstance(holes, list):
            return []
        return [hole for hole in holes if isinstance(hole, dict)]

    def _prepare_component_points(self, component: Dict[str, Any], closed: bool) -> List[Point]:
        points = self._component_points(component)
        points = self._remove_consecutive_duplicates(points)
        if len(points) < 3:
            return points

        if closed:
            points = self._remove_trailing_closure(points)

        points = self._simplify_points(points, self.config.simplify_tolerance_px)
        points = self._remove_consecutive_duplicates(points)

        if closed:
            points = self._remove_trailing_closure(points)
            if len(points) < 3:
                return points
            points.append(points[0])

        return points

    def _payload_bbox(self, components: Iterable[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        all_points: List[Point] = []
        for component in components:
            all_points.extend(self._component_points(component))
        if not all_points:
            raise ValueError("Cannot compute bbox: no points found in components.")

        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        if math.isclose(min_x, max_x) or math.isclose(min_y, max_y):
            raise ValueError(f"Invalid point bbox: {(min_x, min_y, max_x, max_y)}")
        return min_x, min_y, max_x, max_y

    def _geometry_bbox(self, components: Iterable[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        if self.config.geometry_frame == "component":
            return self._payload_bbox(components)

        canvas_bbox = self._canvas_bbox_from_payload()
        if canvas_bbox is not None:
            return canvas_bbox

        print(
            "[ParameterizedJsonCSTBuilder] SVG/image canvas size was not found; "
            "falling back to component bbox."
        )
        return self._payload_bbox(components)

    def _canvas_bbox_from_payload(self) -> Optional[Tuple[float, float, float, float]]:
        svg_path = self._resolve_payload_path(self.payload.get("svg_path"))
        if svg_path is not None and svg_path.exists():
            bbox = self._svg_viewbox_bbox(svg_path)
            if bbox is not None:
                return bbox

        trace_image_path = self._resolve_payload_path(self.payload.get("trace_image_path"))
        if trace_image_path is not None and trace_image_path.exists():
            image_size = self._image_size(trace_image_path)
            if image_size is not None:
                width, height = image_size
                return 0.0, 0.0, float(width), float(height)

        return None

    def _resolve_payload_path(self, raw_path: Any) -> Optional[Path]:
        if not raw_path:
            return None
        path = Path(str(raw_path))
        if path.is_absolute():
            return path
        return (self.json_path.parent / path).resolve()

    @staticmethod
    def _svg_viewbox_bbox(svg_path: Path) -> Optional[Tuple[float, float, float, float]]:
        text = svg_path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r'viewBox\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            values = [float(item) for item in re.split(r"[\s,]+", match.group(1).strip()) if item]
            if len(values) == 4:
                min_x, min_y, width, height = values
                if width > 0 and height > 0:
                    return min_x, min_y, min_x + width, min_y + height

        width_match = re.search(r'width\s*=\s*["\']([0-9.]+)', text)
        height_match = re.search(r'height\s*=\s*["\']([0-9.]+)', text)
        if width_match and height_match:
            width = float(width_match.group(1))
            height = float(height_match.group(1))
            if width > 0 and height > 0:
                return 0.0, 0.0, width, height

        return None

    @staticmethod
    def _image_size(image_path: Path) -> Optional[Tuple[int, int]]:
        try:
            import cv2

            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                height, width = img.shape[:2]
                return int(width), int(height)
        except Exception:
            return None
        return None

    def _map_points_to_cst(
        self,
        points: Sequence[Point],
        bbox: Tuple[float, float, float, float],
    ) -> List[Point]:
        min_x, min_y, max_x, max_y = bbox
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        scale = min(self.config.size_x / bbox_w, self.config.size_y / bbox_h)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        mapped = []
        for x, y in points:
            mapped.append(((x - cx) * scale, -(y - cy) * scale))
        return mapped

    def _simplify_points(self, points: List[Point], tolerance: float) -> List[Point]:
        if tolerance <= 0 or len(points) <= 3:
            return points
        return self._rdp(points, tolerance)

    def _rdp(self, points: List[Point], epsilon: float) -> List[Point]:
        if len(points) < 3:
            return points

        first = points[0]
        last = points[-1]
        max_dist = -1.0
        max_index = 0
        for index, point in enumerate(points[1:-1], start=1):
            dist = self._point_line_distance(point, first, last)
            if dist > max_dist:
                max_dist = dist
                max_index = index

        if max_dist > epsilon:
            left = self._rdp(points[: max_index + 1], epsilon)
            right = self._rdp(points[max_index:], epsilon)
            return left[:-1] + right
        return [first, last]

    @staticmethod
    def _point_line_distance(point: Point, start: Point, end: Point) -> float:
        sx, sy = start
        ex, ey = end
        px, py = point
        dx = ex - sx
        dy = ey - sy
        denom = math.hypot(dx, dy)
        if denom == 0:
            return math.hypot(px - sx, py - sy)
        return abs(dy * px - dx * py + ex * sy - ey * sx) / denom

    @staticmethod
    def _parse_points(raw_points: Any) -> List[Point]:
        points = []
        for item in raw_points:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            points.append((float(item[0]), float(item[1])))
        return points

    def _remove_consecutive_duplicates(self, points: Sequence[Point], tolerance: float = 1e-9) -> List[Point]:
        cleaned: List[Point] = []
        for point in points:
            if not cleaned or self._distance(cleaned[-1], point) > tolerance:
                cleaned.append(point)
        return cleaned

    def _remove_trailing_closure(self, points: Sequence[Point], tolerance: float = 1e-9) -> List[Point]:
        points = list(points)
        while len(points) > 1 and self._distance(points[0], points[-1]) <= tolerance:
            points.pop()
        return points

    @staticmethod
    def _is_closed_component(component: Dict[str, Any]) -> bool:
        return bool(component.get("closed", False))

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{float(value):.9g}"

    @staticmethod
    def _floor_value(value: float) -> float:
        return float(math.floor(float(value)))

    @staticmethod
    def _ceil_value(value: float) -> float:
        return float(math.ceil(float(value)))

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Parameterization JSON does not exist: {path}")
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSON payload: {path}")
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CST geometry from NewParams parameterization JSON.")
    parser.add_argument(
        "--json",
        default=str(PROJECT_ROOT / "Rebuild" / "param48_output" / "curve_parameterization.json"),
        help="Path to NewParams / VTracer parameterization JSON.",
    )
    parser.add_argument(
        "--instance-json",
        default=str(PROJECT_ROOT / "pipeline_test_instance.json"),
        help="Simulation-style Instance_dict JSON used for CST project and simulation settings.",
    )
    parser.add_argument(
        "--layer",
        default="layer0",
        help="Layer key in instance JSON whose substrate, ground and material settings are used.",
    )
    parser.add_argument(
        "--project-folder",
        default=None,
        help="Folder where the CST project will be created.",
    )
    parser.add_argument("--project-name", default=None, help="CST project name.")
    parser.add_argument(
        "--reuse-project-name",
        action="store_true",
        help="Use the project name from instance JSON exactly instead of adding a timestamp suffix.",
    )
    parser.add_argument("--size-x", type=float, default=None, help="Physical model size in x direction.")
    parser.add_argument("--size-y", type=float, default=None, help="Physical model size in y direction.")
    parser.add_argument("--f0", type=float, default=None, help="Start frequency.")
    parser.add_argument("--f1", type=float, default=None, help="Stop frequency.")
    parser.add_argument("--metal-thickness", type=float, default=None, help="Metal extrusion thickness.")
    parser.add_argument("--substrate-thickness", type=float, default=None, help="Substrate thickness.")
    parser.add_argument(
        "--substrate-material",
        default=None,
        help="Substrate material name. Use empty string to skip substrate.",
    )
    parser.add_argument("--no-ground", action="store_true", help="Do not create a PEC ground plane.")
    parser.add_argument(
        "--port-summary",
        default=str(PROJECT_ROOT / "port_summary.json"),
        help="Port summary JSON generated by the image-based port analyzer.",
    )
    parser.add_argument("--build-only", action="store_true", help="Only build CST geometry; do not add port or run solver.")
    parser.add_argument("--close-project", action="store_true", help="Save and close CST after building.")
    parser.add_argument(
        "--curve-method",
        choices=["polygon", "spline"],
        default="polygon",
        help="CST curve creation method for sampled paths.",
    )
    parser.add_argument(
        "--simplify-tolerance-px",
        type=float,
        default=0.0,
        help="Optional RDP simplification tolerance in input pixel units.",
    )
    parser.add_argument(
        "--geometry-frame",
        choices=["svg", "component"],
        default="svg",
        help="Map geometry using the SVG/image canvas or only the component bbox.",
    )
    return parser.parse_args()


def load_instance_config(instance_json_path: Path | str, layer_name: str) -> CSTParametricConfig:
    instance_path = Path(instance_json_path)
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance JSON does not exist: {instance_path}")

    with instance_path.open("r", encoding="utf-8") as file:
        instance = json.load(file)

    if not isinstance(instance, dict):
        raise ValueError(f"Invalid instance JSON payload: {instance_path}")

    package = _resolve_package_config(instance)
    layers = instance.get("layers", {})
    if layer_name not in layers:
        available = ", ".join(str(name) for name in layers.keys())
        raise KeyError(f"Layer `{layer_name}` not found in {instance_path}. Available layers: {available}")

    layer = layers[layer_name]
    col_mats = layer.get("col_mats", {})

    return CSTParametricConfig(
        project_folder=Path(instance.get("Folder_path", r"D:\CST2023proj\autocst_parametric_json")),
        project_name=str(instance.get("Instance", "Parameterized_FSS")),
        unit=_list_get(instance.get("Units", []), 0, "mm"),
        frequency_unit=_list_get(instance.get("Units", []), 1, "GHz"),
        size_x=float(package.get("X", 36.0)),
        size_y=float(package.get("Y", 36.0)),
        f0=float(package.get("f0", 6.0)),
        f1=float(package.get("f1", 14.0)),
        component=layer_name,
        metal_material=_find_material_by_name(col_mats, "PEC") or "PEC",
        metal_thickness=float(instance.get("Metal_thickness", 0.035)),
        substrate_material=_find_first_non_pec_material(col_mats),
        substrate_thickness=float(layer.get("substrate", 0.0)),
        add_ground=bool(layer.get("gnd", False)),
    )


def apply_cli_overrides(config: CSTParametricConfig, args: argparse.Namespace) -> CSTParametricConfig:
    if args.project_folder is not None:
        config.project_folder = Path(args.project_folder)
    if args.project_name is not None:
        config.project_name = args.project_name
    if args.size_x is not None:
        config.size_x = args.size_x
    if args.size_y is not None:
        config.size_y = args.size_y
    if args.f0 is not None:
        config.f0 = args.f0
    if args.f1 is not None:
        config.f1 = args.f1
    if args.metal_thickness is not None:
        config.metal_thickness = args.metal_thickness
    if args.substrate_thickness is not None:
        config.substrate_thickness = args.substrate_thickness
    if args.substrate_material is not None:
        config.substrate_material = args.substrate_material.strip() or None

    if args.no_ground:
        config.add_ground = False
    config.run_solver = not args.build_only
    config.port_summary_path = Path(args.port_summary) if args.port_summary else None
    config.close_project = args.close_project
    config.curve_method = args.curve_method
    config.simplify_tolerance_px = args.simplify_tolerance_px
    config.geometry_frame = args.geometry_frame
    return config


def _find_material_by_name(col_mats: Dict[str, str], target_material: str) -> Optional[str]:
    for material in col_mats.values():
        if material == target_material:
            return material
    return None


def _find_first_non_pec_material(col_mats: Dict[str, str]) -> Optional[str]:
    for material in col_mats.values():
        if material != "PEC":
            return material
    return None


def _resolve_package_config(instance: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("Antenna_package", "FSS_package"):
        package = instance.get(key)
        if isinstance(package, dict):
            return package
    return {}


def _list_get(values: Sequence[Any], index: int, default: Any) -> Any:
    try:
        return values[index]
    except Exception:
        return default


def make_unique_project_name(project_name: str) -> str:
    suffix = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = project_name[:-4] if project_name.lower().endswith(".cst") else project_name
    return f"{stem}_{suffix}"


def main() -> None:
    args = parse_args()
    config = load_instance_config(args.instance_json, args.layer)
    config = apply_cli_overrides(config, args)
    if args.project_name is None and not args.reuse_project_name:
        config.project_name = make_unique_project_name(config.project_name)

    print(f"Parameterization JSON: {Path(args.json).resolve()}")
    print(f"Instance JSON:         {Path(args.instance_json).resolve()}")
    print(f"CST project:           {config.cst_path}")
    print(
        "Simulation settings:  "
        f"size=({config.size_x}, {config.size_y}) {config.unit}, "
        f"freq=({config.f0}, {config.f1}) {config.frequency_unit}, "
        f"substrate={config.substrate_material}, "
        f"substrate_thickness={config.substrate_thickness}, "
        f"ground={config.add_ground}"
    )
    print(
        "Run settings:         "
        f"run_solver={config.run_solver}, "
        f"port_summary={config.port_summary_path}, "
        f"curve_method={config.curve_method}, "
        f"simplify_tolerance_px={config.simplify_tolerance_px}, "
        f"geometry_frame={config.geometry_frame}"
    )

    builder = ParameterizedJsonCSTBuilder(args.json, config)
    cst_path = builder.build()
    print(f"CST project built from parameterization JSON: {cst_path}")


if __name__ == "__main__":
    main()
