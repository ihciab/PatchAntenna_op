"""Build and simulate CST from design-agent JSON artifacts.

This script consumes the JSON files produced under ``design_agent_runs``:

    design_trace.json
    stackup.json
    patch.json

It adapts the LLM patch primitives into the parameterization JSON shape used by
``ParameterizedJsonCSTBuilder`` and then starts CST build/simulation.

Examples:

    python -m design_agent.scripts.run_design_agent_cst
    python -m design_agent.scripts.run_design_agent_cst --run-dir design_agent_runs/initial_design_test
    python -m design_agent.scripts.run_design_agent_cst --run-dir design_agent_runs/initial_design_test --build-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesian_optimization.pipelines.run_parameterization_only import ParameterizationOnlyRunner
from bayesian_optimization.simulation.parameterized_json_to_cst import (
    CSTParametricConfig,
    ParameterizedJsonCSTBuilder,
    make_unique_project_name,
)


Point = Tuple[float, float]


MATERIAL_NAME_MAP = {
    "Rogers RT5880": "Rogers RT-duroid 5880 (lossy)",
    "Rogers RT/duroid 5880": "Rogers RT-duroid 5880 (lossy)",
    "RT5880": "Rogers RT-duroid 5880 (lossy)",
}


class DesignAgentCSTRunner(ParameterizationOnlyRunner):
    """Run CST directly from design-agent JSON artifacts.

    The class intentionally reuses the logging and JSON helper style from
    ``ParameterizationOnlyRunner`` while bypassing image parameterization because
    the LLM has already produced geometry primitives.
    """

    def __init__(
        self,
        run_dir: Path | str,
        output_root: Path | str,
        cst_project_folder: Optional[Path | str] = None,
        project_name: str = "DesignAgent_Antenna",
        layer_name: str = "layer0",
        run_solver: bool = True,
        f0_ghz: Optional[float] = None,
        f1_ghz: Optional[float] = None,
        geometry_frame: str = "svg",
        simplify_tolerance_px: float = 0.0,
        close_project: bool = False,
    ) -> None:
        """Initialize a CST run from a design-agent artifact directory."""

        self.source_run_dir = Path(run_dir)
        self.output_root = Path(output_root)
        self.layer_name = str(layer_name)
        self.run_solver = bool(run_solver)
        self.f0_ghz = f0_ghz
        self.f1_ghz = f1_ghz
        self.geometry_frame = str(geometry_frame)
        self.simplify_tolerance_px = float(simplify_tolerance_px)
        self.close_project = bool(close_project)
        self.project_name = project_name

        self.stackup_path = self.source_run_dir / "stackup.json"
        self.patch_path = self.source_run_dir / "patch.json"
        self.trace_path = self.source_run_dir / "design_trace.json"

        self.run_name = self.source_run_dir.name + "_cst"
        self.run_dir = self.output_root / self.run_name
        self.adapter_dir = self.run_dir / "01_adapter"
        self.cst_dir = Path(cst_project_folder) if cst_project_folder else self.run_dir / "02_cst"
        self.metadata_path = self.run_dir / "design_agent_cst_metadata.json"

    def run(self) -> Path:
        """Create intermediate CST inputs, build the project, and optionally solve."""

        self._prepare_dirs()
        stackup = self._load_required_json(self.stackup_path)
        patch = self._load_required_json(self.patch_path)
        trace = self._load_optional_json(self.trace_path)

        self._log_header("1. Load Design-Agent Artifacts")
        self._log(f"source_run_dir: {self.source_run_dir.resolve()}")
        self._log(f"stackup_json:   {self.stackup_path.resolve()}")
        self._log(f"patch_json:     {self.patch_path.resolve()}")
        if self.trace_path.exists():
            self._log(f"trace_json:     {self.trace_path.resolve()}")

        self._log_header("2. Adapt Patch JSON")
        parameterization_json = self._build_parameterization_json(stackup, patch)
        parameterization_path = self.adapter_dir / "parameterization_from_agent.json"
        self._write_json(parameterization_path, parameterization_json)
        self._log(f"parameterization_json: {parameterization_path}")

        port_summary = self._build_port_summary_from_patch(patch, parameterization_json)
        port_summary_path = self.adapter_dir / "patch_port_summary.json"
        self._write_json(port_summary_path, port_summary)
        self._log(f"port_summary_json:     {port_summary_path}")

        self._log_header("3. Start CST")
        config = self._build_cst_config(stackup, trace, port_summary_path)
        self._log(f"cst_project: {config.cst_path}")
        self._log(f"run_solver:  {config.run_solver}")
        self._log(
            "settings:    "
            f"size=({config.size_x}, {config.size_y}) {config.unit}, "
            f"freq=({config.f0}, {config.f1}) {config.frequency_unit}, "
            f"substrate={config.substrate_material}, "
            f"h={config.substrate_thickness}, ground={config.add_ground}"
        )

        builder = ParameterizedJsonCSTBuilder(parameterization_path, config)
        cst_path = builder.build()

        metadata = {
            "source_run_dir": str(self.source_run_dir.resolve()),
            "stackup_json": str(self.stackup_path.resolve()),
            "patch_json": str(self.patch_path.resolve()),
            "trace_json": str(self.trace_path.resolve()) if self.trace_path.exists() else None,
            "run_dir": str(self.run_dir.resolve()),
            "parameterization_json": str(parameterization_path.resolve()),
            "port_summary_json": str(port_summary_path.resolve()),
            "cst_project": str(cst_path.resolve()),
            "run_solver": config.run_solver,
            "geometry_frame": config.geometry_frame,
            "simplify_tolerance_px": config.simplify_tolerance_px,
        }
        self._write_json(self.metadata_path, metadata)
        self._log(f"metadata:    {self.metadata_path}")
        return cst_path

    def _prepare_dirs(self) -> None:
        for path in (self.run_dir, self.adapter_dir, self.cst_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _build_parameterization_json(self, stackup: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        substrate = self._substrate(stackup)
        width = self._positive_float(substrate.get("width"), "substrate.width")
        length = self._positive_float(substrate.get("length"), "substrate.length")
        primitive_groups = self._raw_primitive_groups(patch)
        components: List[Dict[str, Any]] = []
        for component_id, raw_primitives in enumerate(primitive_groups):
            primitives = self._adapt_line_primitives(raw_primitives)
            self._validate_primitive_chain(primitives)
            points = self._points_from_primitives(primitives)
            if len(points) < 4:
                raise ValueError("Patch primitives must form at least one closed polygon.")
            if self._distance(points[0], points[-1]) > 1e-9:
                points.append(points[0])
            self._validate_modelable_polygon(points, canvas_width=width, canvas_height=length)
            bbox = self._bbox(points)
            components.append(
                {
                    "component_id": component_id,
                    "closed": True,
                    "resampled_points": [[x, y] for x, y in points],
                    "fallback_points": [[x, y] for x, y in points],
                    "points": [[x, y] for x, y in points],
                    "bbox": list(bbox),
                    "primitives": primitives,
                    "metadata": {
                        "source": "design_agent.patch_json",
                        "material": self._patch_material(patch),
                    },
                }
            )
        self._validate_feed_patch_connection(components)

        return {
            "schema_version": "design_agent_cst_adapter_v1",
            "backend": "llm_patch_primitives",
            "source_patch_json": str(self.patch_path.resolve()),
            "source_stackup_json": str(self.stackup_path.resolve()),
            "coordinate_system": {
                "unit": "mm",
                "origin": "lower_left_or_prompt_defined_canvas",
                "cst_mapping": "canvas-centered; y axis flipped by ParameterizedJsonCSTBuilder",
            },
            "canvas": {
                "width": width,
                "height": length,
                "unit": "mm",
            },
            "components": components,
        }

    def _adapt_line_primitives(self, raw_primitives: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_primitives, list) or not raw_primitives:
            raise ValueError("patch.json must contain a non-empty `primitives` list.")

        adapted: List[Dict[str, Any]] = []
        for index, primitive in enumerate(raw_primitives):
            if not isinstance(primitive, dict):
                raise ValueError(f"Invalid primitive at index {index}: expected object.")
            primitive_type = str(primitive.get("type", "line")).lower()
            if primitive_type != "line":
                raise ValueError(f"Only line primitives are supported for this first CST adapter, got {primitive_type}.")
            start = self._point_from_primitive_endpoint(primitive, "p1", "start")
            end = self._point_from_primitive_endpoint(primitive, "p2", "end")
            adapted.append(
                {
                    "id": primitive.get("id", f"L{index + 1}"),
                    "type": "line",
                    "kind": "line",
                    "start": [start[0], start[1]],
                    "end": [end[0], end[1]],
                    "points": [[start[0], start[1]], [end[0], end[1]]],
                    "material": primitive.get("material", "PEC"),
                    "role": primitive.get("role", ""),
                    "parameter_name": primitive.get("parameter_name", ""),
                    "source": "design_agent.patch_json",
                }
            )
        return adapted

    def _build_port_summary(self, parameterization_json: Dict[str, Any]) -> Dict[str, Any]:
        component = parameterization_json["components"][0]
        points = [(float(x), float(y)) for x, y in component["points"]]
        min_x, min_y, max_x, max_y = self._bbox(points)
        port_x = (min_x + max_x) / 2.0
        port_y = max_y
        local_width = max(2.0, min(max_x - min_x, max_y - min_y) * 0.12)

        return {
            "source": "design_agent_cst_adapter",
            "border_contact_mode": "synthetic_patch_edge_port",
            "closest_edge": [[min_x, max_y], [max_x, max_y]],
            "closest_border_sides": ["bottom"],
            "patch_port_detection": {
                "ports": [
                    {
                        "point": [port_x, port_y],
                        "direction": "bottom",
                        "local_width": local_width,
                        "source": "synthetic_bottom_edge_center",
                    }
                ]
            },
            "bo_port_connection_adjustment": {
                "connected_point": [port_x, port_y],
                "final_free_normal_inward_px": 0.0,
                "source": "synthetic_no_shift",
            },
        }

    def _build_port_summary_from_patch(
        self,
        patch: Dict[str, Any],
        parameterization_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        port = patch.get("port")
        if not isinstance(port, dict):
            return self._build_port_summary(parameterization_json)

        point = port.get("point")
        direction = str(port.get("direction", "bottom")).lower()
        if not isinstance(point, dict) or direction not in {"left", "right", "top", "bottom"}:
            return self._build_port_summary(parameterization_json)

        try:
            port_x = float(point["x"])
            port_y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            return self._build_port_summary(parameterization_json)

        feed_width = self._number_or_none(port.get("feed_width_mm")) or 2.0
        edge_points = self._port_edge_points(port.get("edge"))
        if edge_points is None:
            if direction in {"top", "bottom"}:
                edge_points = [[port_x - feed_width / 2.0, port_y], [port_x + feed_width / 2.0, port_y]]
            else:
                edge_points = [[port_x, port_y - feed_width / 2.0], [port_x, port_y + feed_width / 2.0]]

        return {
            "source": "design_agent_patch_port",
            "border_contact_mode": "design_agent_explicit_port",
            "closest_edge": edge_points,
            "closest_border_sides": [direction],
            "patch_port_detection": {
                "ports": [
                    {
                        "point": [port_x, port_y],
                        "direction": direction,
                        "local_width": max(2.0, float(feed_width)),
                        "source": "patch_json.port",
                    }
                ]
            },
            "bo_port_connection_adjustment": {
                "connected_point": [port_x, port_y],
                "final_free_normal_inward_px": 0.0,
                "source": "patch_json.port",
            },
        }

    def _build_cst_config(
        self,
        stackup: Dict[str, Any],
        trace: Dict[str, Any],
        port_summary_path: Path,
    ) -> CSTParametricConfig:
        substrate = self._substrate(stackup)
        ground = stackup.get("ground", {}) if isinstance(stackup.get("ground", {}), dict) else {}
        f0, f1 = self._frequency_range_ghz(trace)
        material = self._normalize_material_name(str(substrate.get("material", "Rogers RT5880")))

        config = CSTParametricConfig(
            project_folder=self.cst_dir,
            project_name=make_unique_project_name(self.project_name),
            unit="mm",
            frequency_unit="GHz",
            size_x=self._positive_float(substrate.get("width"), "substrate.width"),
            size_y=self._positive_float(substrate.get("length"), "substrate.length"),
            f0=f0,
            f1=f1,
            component=self.layer_name,
            metal_material=str(ground.get("material", "PEC") or "PEC"),
            metal_thickness=float(ground.get("thickness", 0.035) or 0.035),
            substrate_material=material,
            substrate_thickness=self._positive_float(substrate.get("thickness"), "substrate.thickness"),
            add_ground=True,
            close_project=self.close_project,
            save_project=True,
            run_solver=self.run_solver,
            port_summary_path=port_summary_path,
            result_export_folder=self.run_dir / "03_results",
            curve_method="polygon",
            simplify_tolerance_px=self.simplify_tolerance_px,
            geometry_frame=self.geometry_frame,
        )
        return config

    def _frequency_range_ghz(self, trace: Dict[str, Any]) -> Tuple[float, float]:
        if self.f0_ghz is not None and self.f1_ghz is not None:
            return float(self.f0_ghz), float(self.f1_ghz)
        if self.f0_ghz is not None or self.f1_ghz is not None:
            raise ValueError("Pass both --f0 and --f1, or neither.")

        summary = trace.get("input_summary", {}) if isinstance(trace, dict) else {}
        center = self._number_or_none(summary.get("center_frequency"))
        bandwidth = self._number_or_none(summary.get("bandwidth"))

        if center is None:
            return 2.0, 3.0
        center_ghz = center / 1e9 if center > 1e6 else center
        if bandwidth is None:
            span_ghz = max(0.2, center_ghz * 0.2)
        else:
            span_ghz = bandwidth / 1e9 if bandwidth > 1e6 else bandwidth
            span_ghz = max(0.05, span_ghz)
        f0 = max(0.01, center_ghz - span_ghz / 2.0)
        f1 = center_ghz + span_ghz / 2.0
        if f1 <= f0:
            raise ValueError(f"Invalid inferred frequency range: f0={f0}, f1={f1}")
        return f0, f1

    @staticmethod
    def _load_required_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Required JSON file does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSON object: {path}")
        return payload

    @staticmethod
    def _load_optional_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _substrate(stackup: Dict[str, Any]) -> Dict[str, Any]:
        substrate = stackup.get("substrate")
        if not isinstance(substrate, dict):
            raise ValueError("stackup.json must contain a `substrate` object.")
        return substrate

    @staticmethod
    def _patch_material(patch: Dict[str, Any]) -> str:
        groups = DesignAgentCSTRunner._raw_primitive_groups(patch)
        if groups and groups[0] and isinstance(groups[0][0], dict):
            return str(groups[0][0].get("material", "PEC"))
        return "PEC"

    @staticmethod
    def _raw_primitives(patch: Dict[str, Any]) -> Any:
        groups = DesignAgentCSTRunner._raw_primitive_groups(patch)
        if groups:
            return groups[0]
        return []

    @staticmethod
    def _raw_primitive_groups(patch: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        conductor = patch.get("conductor")
        if isinstance(conductor, dict):
            components = conductor.get("components")
            if isinstance(components, list):
                groups: List[List[Dict[str, Any]]] = []
                for component in components:
                    if not isinstance(component, dict):
                        continue
                    primitives = component.get("primitives")
                    if isinstance(primitives, list) and primitives:
                        groups.append(primitives)
                        continue
                    vertices = component.get("polygon_vertices")
                    derived = DesignAgentCSTRunner._primitives_from_polygon_vertices(vertices, component.get("name", "component"))
                    if derived:
                        groups.append(derived)
                if groups:
                    return groups

            boundaries = conductor.get("boundaries")
            if isinstance(boundaries, list):
                groups = []
                for boundary in boundaries:
                    if isinstance(boundary, dict):
                        primitives = boundary.get("primitives")
                        if isinstance(primitives, list) and primitives:
                            groups.append(primitives)
                if groups:
                    return groups

            primitives = conductor.get("primitives")
            if isinstance(primitives, list) and primitives:
                role_groups = DesignAgentCSTRunner._split_primitives_by_role_if_closed(primitives)
                if role_groups:
                    return role_groups
                return [primitives]

            vertices = conductor.get("polygon_vertices")
            derived = DesignAgentCSTRunner._primitives_from_polygon_vertices(vertices, "conductor")
            if derived:
                return [derived]

        primitives = patch.get("primitives")
        if isinstance(primitives, list) and primitives:
            role_groups = DesignAgentCSTRunner._split_primitives_by_role_if_closed(primitives)
            if role_groups:
                return role_groups
            return [primitives]
        return []

    @staticmethod
    def _split_primitives_by_role_if_closed(primitives: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for primitive in primitives:
            if not isinstance(primitive, dict):
                return []
            role = str(primitive.get("role", "")).lower()
            if "feed" in role:
                key = "feed"
            elif "patch" in role:
                key = "patch"
            else:
                return []
            grouped.setdefault(key, []).append(primitive)

        if len(grouped) < 2:
            return []

        ordered_groups: List[List[Dict[str, Any]]] = []
        for key in ("patch", "feed"):
            group = grouped.get(key, [])
            if len(group) < 3:
                return []
            try:
                adapted = DesignAgentCSTRunner._adapt_static_line_primitives(group)
                DesignAgentCSTRunner._validate_primitive_chain(adapted)
            except Exception:
                return []
            ordered_groups.append(group)
        return ordered_groups

    @staticmethod
    def _adapt_static_line_primitives(raw_primitives: Any) -> List[Dict[str, Any]]:
        runner = object.__new__(DesignAgentCSTRunner)
        return DesignAgentCSTRunner._adapt_line_primitives(runner, raw_primitives)

    @staticmethod
    def _primitives_from_polygon_vertices(vertices: Any, prefix: Any) -> List[Dict[str, Any]]:
        if not isinstance(vertices, list) or len(vertices) < 4:
            return []
        points: List[Dict[str, float]] = []
        for vertex in vertices:
            if isinstance(vertex, dict):
                try:
                    points.append(
                        {
                            "x": float(vertex["x"]),
                            "y": float(vertex["y"]),
                            "z": float(vertex.get("z", 0.0)),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    return []
            elif isinstance(vertex, Sequence) and len(vertex) >= 2:
                try:
                    points.append(
                        {
                            "x": float(vertex[0]),
                            "y": float(vertex[1]),
                            "z": float(vertex[2]) if len(vertex) >= 3 else 0.0,
                        }
                    )
                except (TypeError, ValueError):
                    return []
            else:
                return []
        primitives: List[Dict[str, Any]] = []
        for index, (p1, p2) in enumerate(zip(points[:-1], points[1:]), start=1):
            primitives.append(
                {
                    "type": "line",
                    "id": f"{prefix}_{index}",
                    "p1": p1,
                    "p2": p2,
                    "layer": "top",
                    "material": "PEC",
                    "role": f"{prefix}_boundary",
                    "parameter_name": str(prefix),
                }
            )
        return primitives

    @staticmethod
    def _port_edge_points(edge: Any) -> Optional[List[List[float]]]:
        if not isinstance(edge, list) or len(edge) < 2:
            return None
        points: List[List[float]] = []
        for item in edge[:2]:
            if isinstance(item, dict):
                try:
                    points.append([float(item["x"]), float(item["y"])])
                except (KeyError, TypeError, ValueError):
                    return None
            elif isinstance(item, Sequence) and len(item) >= 2:
                try:
                    points.append([float(item[0]), float(item[1])])
                except (TypeError, ValueError):
                    return None
            else:
                return None
        return points

    @staticmethod
    def _point_from_primitive_endpoint(primitive: Dict[str, Any], object_key: str, list_key: str) -> Point:
        value = primitive.get(object_key)
        if isinstance(value, dict):
            return float(value["x"]), float(value["y"])
        value = primitive.get(list_key)
        if isinstance(value, Sequence) and len(value) >= 2:
            return float(value[0]), float(value[1])
        raise ValueError(f"Primitive missing endpoint `{object_key}` or `{list_key}`: {primitive}")

    @staticmethod
    def _points_from_primitives(primitives: List[Dict[str, Any]]) -> List[Point]:
        points: List[Point] = []
        for primitive in primitives:
            start = (float(primitive["start"][0]), float(primitive["start"][1]))
            end = (float(primitive["end"][0]), float(primitive["end"][1]))
            if not points:
                points.append(start)
            elif DesignAgentCSTRunner._distance(points[-1], start) > 1e-9:
                points.append(start)
            points.append(end)
        return points

    @classmethod
    def _validate_primitive_chain(cls, primitives: List[Dict[str, Any]]) -> None:
        if len(primitives) < 3:
            raise ValueError("A closed CST polygon requires at least three line primitives.")
        for index, primitive in enumerate(primitives):
            start = (float(primitive["start"][0]), float(primitive["start"][1]))
            end = (float(primitive["end"][0]), float(primitive["end"][1]))
            if cls._distance(start, end) <= 1e-9:
                raise ValueError(f"Primitive {index} has zero length: {primitive}")
            next_primitive = primitives[(index + 1) % len(primitives)]
            next_start = (float(next_primitive["start"][0]), float(next_primitive["start"][1]))
            if cls._distance(end, next_start) > 1e-7:
                raise ValueError(
                    "Patch primitives are not one continuous closed boundary. "
                    f"Primitive {index} ends at {end}, but primitive {(index + 1) % len(primitives)} "
                    f"starts at {next_start}."
                )

    @classmethod
    def _validate_modelable_polygon(
        cls,
        points: List[Point],
        canvas_width: float,
        canvas_height: float,
    ) -> None:
        if len(points) < 4:
            raise ValueError("Closed polygon needs at least 3 unique vertices plus closure.")
        if cls._distance(points[0], points[-1]) > 1e-7:
            raise ValueError("Polygon is not closed: final point must equal first point.")

        open_points = points[:-1]
        rounded = [cls._rounded_point(point) for point in open_points]
        seen: Dict[Tuple[float, float], int] = {}
        duplicates: List[Dict[str, Any]] = []
        for index, point in enumerate(rounded):
            if point in seen:
                duplicates.append({"first_index": seen[point], "duplicate_index": index, "point": point})
            else:
                seen[point] = index
        if duplicates:
            raise ValueError(
                "CST polygon contains duplicate vertices before closure. "
                "Trace the outer conductor boundary exactly once. "
                f"Duplicates: {duplicates[:5]}"
            )

        outside = [
            {"index": index, "point": point}
            for index, point in enumerate(open_points)
            if point[0] < -1e-7 or point[0] > canvas_width + 1e-7 or point[1] < -1e-7 or point[1] > canvas_height + 1e-7
        ]
        if outside:
            raise ValueError(
                "CST polygon contains vertices outside the substrate canvas. "
                f"Canvas=({canvas_width}, {canvas_height}), outside={outside[:5]}"
            )

        repeated_edges = cls._repeated_edges(open_points)
        if repeated_edges:
            raise ValueError(f"CST polygon contains repeated/backtracking edges: {repeated_edges[:5]}")

        intersections = cls._self_intersections(points)
        if intersections:
            raise ValueError(
                "CST polygon self-intersects or contains internal seams. "
                f"Intersections: {intersections[:5]}"
            )

    @classmethod
    def _validate_feed_patch_connection(cls, components: List[Dict[str, Any]]) -> None:
        patch_components = [
            component
            for component in components
            if cls._component_has_role(component, "patch")
        ]
        feed_components = [
            component
            for component in components
            if cls._component_has_role(component, "feed")
        ]
        if not patch_components or not feed_components:
            return

        for feed in feed_components:
            feed_bbox = tuple(float(value) for value in feed["bbox"])
            for patch in patch_components:
                patch_bbox = tuple(float(value) for value in patch["bbox"])
                if cls._rectangles_have_line_or_area_contact(feed_bbox, patch_bbox):
                    return
        raise ValueError(
            "Feed line is not robustly connected to the patch. "
            "It must share a non-zero-length edge segment with the patch or overlap "
            "into the patch by a small amount. Point contact is not enough for CST simulation."
        )

    @staticmethod
    def _component_has_role(component: Dict[str, Any], role_keyword: str) -> bool:
        for primitive in component.get("primitives", []):
            role = str(primitive.get("role", "")).lower()
            if role_keyword in role:
                return True
        return False

    @staticmethod
    def _rectangles_have_line_or_area_contact(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
        tolerance: float = 1e-7,
    ) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        overlap_x = min(ax2, bx2) - max(ax1, bx1)
        overlap_y = min(ay2, by2) - max(ay1, by1)
        if overlap_x > tolerance and overlap_y > tolerance:
            return True
        shared_horizontal_edge = (
            (abs(ay2 - by1) <= tolerance or abs(by2 - ay1) <= tolerance)
            and overlap_x > tolerance
        )
        shared_vertical_edge = (
            (abs(ax2 - bx1) <= tolerance or abs(bx2 - ax1) <= tolerance)
            and overlap_y > tolerance
        )
        return shared_horizontal_edge or shared_vertical_edge

    @classmethod
    def _repeated_edges(cls, open_points: List[Point]) -> List[Dict[str, Any]]:
        edges: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}
        repeated: List[Dict[str, Any]] = []
        point_count = len(open_points)
        for index in range(point_count):
            p1 = cls._rounded_point(open_points[index])
            p2 = cls._rounded_point(open_points[(index + 1) % point_count])
            edge = tuple(sorted((p1, p2)))
            if edge in edges:
                repeated.append({"first_index": edges[edge], "duplicate_index": index, "edge": edge})
            else:
                edges[edge] = index
        return repeated

    @classmethod
    def _self_intersections(cls, points: List[Point]) -> List[Dict[str, Any]]:
        intersections: List[Dict[str, Any]] = []
        edge_count = len(points) - 1
        for i in range(edge_count):
            a1, a2 = points[i], points[i + 1]
            for j in range(i + 1, edge_count):
                if abs(i - j) <= 1:
                    continue
                if i == 0 and j == edge_count - 1:
                    continue
                b1, b2 = points[j], points[j + 1]
                if cls._segments_intersect(a1, a2, b1, b2):
                    intersections.append({"edge_a": i, "edge_b": j})
        return intersections

    @classmethod
    def _segments_intersect(cls, a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
        def orient(p: Point, q: Point, r: Point) -> float:
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p: Point, q: Point, r: Point) -> bool:
            return (
                min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
                and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
            )

        o1 = orient(a1, a2, b1)
        o2 = orient(a1, a2, b2)
        o3 = orient(b1, b2, a1)
        o4 = orient(b1, b2, a2)

        if o1 * o2 < -1e-12 and o3 * o4 < -1e-12:
            return True
        if abs(o1) <= 1e-12 and on_segment(a1, b1, a2):
            return True
        if abs(o2) <= 1e-12 and on_segment(a1, b2, a2):
            return True
        if abs(o3) <= 1e-12 and on_segment(b1, a1, b2):
            return True
        if abs(o4) <= 1e-12 and on_segment(b1, a2, b2):
            return True
        return False

    @staticmethod
    def _rounded_point(point: Point) -> Tuple[float, float]:
        return round(float(point[0]), 9), round(float(point[1]), 9)

    @staticmethod
    def _bbox(points: Sequence[Point]) -> Tuple[float, float, float, float]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _distance(p1: Point, p2: Point) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be a positive finite number, got {value!r}")
        return number

    @staticmethod
    def _number_or_none(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _normalize_material_name(material: str) -> str:
        return MATERIAL_NAME_MAP.get(material, material)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the CST runner."""

    parser = argparse.ArgumentParser(description="Run CST from design_agent_runs JSON artifacts.")
    parser.add_argument(
        "--run-dir",
        default=str(PROJECT_ROOT / "design_agent_runs" / "initial_design_test"),
        help="Directory containing design_trace.json, stackup.json and patch.json.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "design_agent_runs"),
        help="Root directory where CST adapter outputs are written.",
    )
    parser.add_argument(
        "--cst-project-folder",
        default=None,
        help="Optional CST project folder. Default: <output-root>/<run-name>_cst/02_cst.",
    )
    parser.add_argument("--project-name", default="DesignAgent_Antenna", help="Base CST project name.")
    parser.add_argument("--layer", default="layer0", help="CST component/layer name.")
    parser.add_argument("--f0", type=float, default=None, help="Start frequency in GHz.")
    parser.add_argument("--f1", type=float, default=None, help="Stop frequency in GHz.")
    parser.add_argument("--build-only", action="store_true", help="Build geometry but do not run CST solver.")
    parser.add_argument("--close-project", action="store_true", help="Close CST project after build/simulation.")
    parser.add_argument(
        "--geometry-frame",
        choices=["svg", "component"],
        default="svg",
        help="Use canvas frame or component bbox for CST geometry mapping.",
    )
    parser.add_argument(
        "--simplify-tolerance-px",
        type=float,
        default=0.0,
        help="RDP simplification tolerance passed to the CST builder.",
    )
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_arg_parser().parse_args()
    runner = DesignAgentCSTRunner(
        run_dir=args.run_dir,
        output_root=args.output_root,
        cst_project_folder=args.cst_project_folder,
        project_name=args.project_name,
        layer_name=args.layer,
        run_solver=not args.build_only,
        f0_ghz=args.f0,
        f1_ghz=args.f1,
        geometry_frame=args.geometry_frame,
        simplify_tolerance_px=args.simplify_tolerance_px,
        close_project=args.close_project,
    )
    cst_path = runner.run()
    print(f"\n[DesignAgentCSTRunner] DONE: {cst_path}")


if __name__ == "__main__":
    main()
