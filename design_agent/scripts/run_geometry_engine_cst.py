"""Build and simulate CST from Geometry Engine JSON.

Example:

    python -m design_agent.scripts.run_geometry_engine_cst
    python -m design_agent.scripts.run_geometry_engine_cst --geometry-json design_agent_runs/redesign_test/geometry_engine_redesign.json
    python -m design_agent.scripts.run_geometry_engine_cst --build-only
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

from bayesian_optimization.simulation.parameterized_json_to_cst import (
    CSTParametricConfig,
    ParameterizedJsonCSTBuilder,
    make_unique_project_name,
)
from design_agent.scripts.run_design_agent_cst import DesignAgentCSTRunner


DEFAULT_GEOMETRY_JSON = PROJECT_ROOT / "design_agent_runs" / "redesign_test" / "geometry_engine_redesign.json"
DEFAULT_SOURCE_RUN_DIR = PROJECT_ROOT / "design_agent_runs" / "initial_design_test"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "design_agent_runs"


Point = Tuple[float, float]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CST from Geometry Engine geometry JSON.")
    parser.add_argument(
        "--geometry-json",
        default=str(DEFAULT_GEOMETRY_JSON),
        help="Path to geometry_engine_geometry_v1 JSON.",
    )
    parser.add_argument(
        "--source-run-dir",
        default=str(DEFAULT_SOURCE_RUN_DIR),
        help="Directory containing stackup.json and design_trace.json for CST settings.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where CST adapter outputs are written.",
    )
    parser.add_argument(
        "--cst-project-folder",
        default=None,
        help="Optional CST project folder. Default: <output-root>/<geometry-name>_cst/02_cst.",
    )
    parser.add_argument("--project-name", default="GeometryEngine_Redesign", help="Base CST project name.")
    parser.add_argument("--layer", default="layer0", help="CST component/layer name.")
    parser.add_argument("--f0", type=float, default=None, help="Start frequency in GHz.")
    parser.add_argument("--f1", type=float, default=None, help="Stop frequency in GHz.")
    parser.add_argument("--build-only", action="store_true", help="Build geometry but do not run CST solver.")
    parser.add_argument("--close-project", action="store_true", help="Close CST project after build/simulation.")
    parser.add_argument(
        "--simplify-tolerance-px",
        type=float,
        default=0.0,
        help="RDP simplification tolerance passed to the CST builder.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runner = GeometryEngineCSTRunner(
        geometry_json_path=args.geometry_json,
        source_run_dir=args.source_run_dir,
        output_root=args.output_root,
        cst_project_folder=args.cst_project_folder,
        project_name=args.project_name,
        layer_name=args.layer,
        run_solver=not args.build_only,
        f0_ghz=args.f0,
        f1_ghz=args.f1,
        simplify_tolerance_px=args.simplify_tolerance_px,
        close_project=args.close_project,
    )
    cst_path = runner.run()
    print(f"\n[GeometryEngineCSTRunner] DONE: {cst_path}")
    return 0


class GeometryEngineCSTRunner:
    """Adapt Geometry Engine JSON to ParameterizedJsonCSTBuilder input."""

    def __init__(
        self,
        geometry_json_path: Path | str,
        source_run_dir: Path | str,
        output_root: Path | str,
        cst_project_folder: Optional[Path | str] = None,
        project_name: str = "GeometryEngine_Redesign",
        layer_name: str = "layer0",
        run_solver: bool = True,
        f0_ghz: Optional[float] = None,
        f1_ghz: Optional[float] = None,
        simplify_tolerance_px: float = 0.0,
        close_project: bool = False,
    ) -> None:
        self.geometry_json_path = Path(geometry_json_path)
        self.source_run_dir = Path(source_run_dir)
        self.output_root = Path(output_root)
        self.project_name = str(project_name)
        self.layer_name = str(layer_name)
        self.run_solver = bool(run_solver)
        self.f0_ghz = f0_ghz
        self.f1_ghz = f1_ghz
        self.simplify_tolerance_px = float(simplify_tolerance_px)
        self.close_project = bool(close_project)

        self.stackup_path = self.source_run_dir / "stackup.json"
        self.trace_path = self.source_run_dir / "design_trace.json"

        self.run_name = self.geometry_json_path.stem + "_cst"
        self.run_dir = self.output_root / self.run_name
        self.adapter_dir = self.run_dir / "01_adapter"
        self.cst_dir = Path(cst_project_folder) if cst_project_folder else self.run_dir / "02_cst"
        self.metadata_path = self.run_dir / "geometry_engine_cst_metadata.json"

    def run(self) -> Path:
        self._prepare_dirs()
        geometry = self._load_required_json(self.geometry_json_path)
        stackup = self._load_required_json(self.stackup_path)
        trace = self._load_optional_json(self.trace_path)

        print("\n==================================================")
        print("1. Load Geometry Engine JSON")
        print("==================================================")
        print(f"geometry_json: {self.geometry_json_path.resolve()}")
        print(f"stackup_json:  {self.stackup_path.resolve()}")
        if self.trace_path.exists():
            print(f"trace_json:    {self.trace_path.resolve()}")

        print("\n==================================================")
        print("2. Adapt Geometry JSON")
        print("==================================================")
        parameterization = self._build_parameterization_json(geometry, stackup)
        parameterization_path = self.adapter_dir / "parameterization_from_geometry_engine.json"
        self._write_json(parameterization_path, parameterization)
        print(f"parameterization_json: {parameterization_path}")

        port_summary = self._build_port_summary(geometry, parameterization)
        port_summary_path = self.adapter_dir / "patch_port_summary.json"
        self._write_json(port_summary_path, port_summary)
        print(f"port_summary_json:     {port_summary_path}")

        print("\n==================================================")
        print("3. Start CST")
        print("==================================================")
        config = self._build_cst_config(stackup, trace, port_summary_path)
        print(f"cst_project: {config.cst_path}")
        print(f"run_solver:  {config.run_solver}")
        print(
            "settings:    "
            f"size=({config.size_x}, {config.size_y}) {config.unit}, "
            f"freq=({config.f0}, {config.f1}) {config.frequency_unit}, "
            f"substrate={config.substrate_material}, "
            f"h={config.substrate_thickness}, ground={config.add_ground}"
        )

        builder = ParameterizedJsonCSTBuilder(parameterization_path, config)
        cst_path = builder.build()

        self._write_json(
            self.metadata_path,
            {
                "geometry_json": str(self.geometry_json_path.resolve()),
                "source_run_dir": str(self.source_run_dir.resolve()),
                "parameterization_json": str(parameterization_path.resolve()),
                "port_summary_json": str(port_summary_path.resolve()),
                "cst_project": str(cst_path.resolve()),
                "run_solver": config.run_solver,
            },
        )
        print(f"metadata:    {self.metadata_path}")
        return cst_path

    def _prepare_dirs(self) -> None:
        for path in (self.run_dir, self.adapter_dir, self.cst_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _build_parameterization_json(self, geometry: Dict[str, Any], stackup: Dict[str, Any]) -> Dict[str, Any]:
        if geometry.get("schema_version") != "geometry_engine_geometry_v1":
            raise ValueError("Expected schema_version='geometry_engine_geometry_v1'.")

        substrate = self._substrate(stackup)
        width = self._positive_float(substrate.get("width"), "substrate.width")
        length = self._positive_float(substrate.get("length"), "substrate.length")
        components: List[Dict[str, Any]] = []

        geometries = geometry.get("geometries")
        if not isinstance(geometries, list) or not geometries:
            raise ValueError("Geometry Engine JSON must contain non-empty `geometries`.")

        for index, item in enumerate(geometries):
            if not isinstance(item, dict):
                continue
            outer = item.get("outer_boundary")
            if not isinstance(outer, dict):
                continue
            points = self._vertices_to_points(outer.get("vertices"))
            if len(points) < 3:
                raise ValueError(f"Geometry {index} outer boundary has fewer than 3 vertices.")
            holes = [self._adapt_hole(hole, hole_index) for hole_index, hole in enumerate(item.get("holes", []) or [])]
            components.append(
                {
                    "component_id": index,
                    "closed": True,
                    "points": points,
                    "resampled_points": points,
                    "fallback_points": points,
                    "bbox": list(self._bbox(points)),
                    "holes": [hole for hole in holes if hole is not None],
                    "metadata": {
                        "source": "geometry_engine_geometry_v1",
                        "geometry_id": item.get("id", f"geometry_{index}"),
                        "semantic_type": (item.get("metadata") or {}).get("semantic_type"),
                    },
                }
            )

        if not components:
            raise ValueError("No modelable planar conductor geometry was found.")

        return {
            "schema_version": "geometry_engine_cst_adapter_v1",
            "backend": "geometry_engine_geometry_v1",
            "source_geometry_json": str(self.geometry_json_path.resolve()),
            "coordinate_system": geometry.get("coordinate_system", {}),
            "canvas": {
                "width": width,
                "height": length,
                "unit": "mm",
            },
            "components": components,
        }

    def _adapt_hole(self, hole: Any, index: int) -> Optional[Dict[str, Any]]:
        if not isinstance(hole, dict):
            return None
        points = self._vertices_to_points(hole.get("vertices"))
        if len(points) < 3:
            return None
        return {
            "id": hole.get("id", f"hole_{index:03d}"),
            "closed": True,
            "points": points,
            "resampled_points": points,
            "fallback_points": points,
            "bbox": list(self._bbox(points)),
            "metadata": {"role": hole.get("role", "hole")},
        }

    def _build_port_summary(self, geometry: Dict[str, Any], parameterization: Dict[str, Any]) -> Dict[str, Any]:
        feed = self._find_feed_metadata(geometry)
        if feed is not None:
            direction = str(feed.get("direction", "bottom")).lower()
            feed_width = max(2.0, float(feed.get("width", 2.0)))
            terminal_point = feed.get("terminal_point")
            terminal_edge = feed.get("terminal_edge")
            if isinstance(terminal_point, dict):
                port_x = float(terminal_point["x"])
                port_y = float(terminal_point["y"])
            else:
                port_x = float(feed["x"])
                port_y = float(feed["y"])
            edge_points = self._parse_optional_edge(terminal_edge)
            if edge_points is None and direction in {"top", "bottom"}:
                edge_points = [[port_x - feed_width / 2.0, port_y], [port_x + feed_width / 2.0, port_y]]
            elif edge_points is None:
                edge_points = [[port_x, port_y - feed_width / 2.0], [port_x, port_y + feed_width / 2.0]]
            source = "geometry_engine_feed_metadata"
        else:
            component = parameterization["components"][0]
            min_x, min_y, max_x, _max_y = self._bbox([(float(x), float(y)) for x, y in component["points"]])
            feed_width = max(2.0, (max_x - min_x) * 0.08)
            port_x = (min_x + max_x) / 2.0
            port_y = min_y
            direction = "bottom"
            edge_points = [[port_x - feed_width / 2.0, port_y], [port_x + feed_width / 2.0, port_y]]
            source = "synthetic_bottom_edge_center"

        return {
            "source": source,
            "border_contact_mode": "geometry_engine_edge_feed",
            "closest_edge": edge_points,
            "closest_border_sides": [direction],
            "patch_port_detection": {
                "ports": [
                    {
                        "point": [port_x, port_y],
                        "direction": direction,
                        "local_width": feed_width,
                        "source": source,
                    }
                ]
            },
            "bo_port_connection_adjustment": {
                "connected_point": [port_x, port_y],
                "final_free_normal_inward_px": 0.0,
                "source": source,
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
        material = DesignAgentCSTRunner._normalize_material_name(str(substrate.get("material", "Rogers RT5880")))
        return CSTParametricConfig(
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
            geometry_frame="svg",
        )

    def _frequency_range_ghz(self, trace: Dict[str, Any]) -> Tuple[float, float]:
        if self.f0_ghz is not None and self.f1_ghz is not None:
            return float(self.f0_ghz), float(self.f1_ghz)
        if self.f0_ghz is not None or self.f1_ghz is not None:
            raise ValueError("Pass both --f0 and --f1, or neither.")
        summary = trace.get("input_summary", {}) if isinstance(trace, dict) else {}
        center = self._number_or_none(summary.get("center_frequency"))
        bandwidth = self._number_or_none(summary.get("bandwidth"))
        if center is None:
            return 2.4, 2.5
        center_ghz = center / 1e9 if center > 1e6 else center
        span_ghz = 0.1 if bandwidth is None else bandwidth / 1e9 if bandwidth > 1e6 else bandwidth
        span_ghz = max(0.05, float(span_ghz))
        return max(0.01, center_ghz - span_ghz / 2.0), center_ghz + span_ghz / 2.0

    @staticmethod
    def _find_feed_metadata(geometry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for item in geometry.get("geometries", []) or []:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {})
            feed = metadata.get("feed") if isinstance(metadata, dict) else None
            if isinstance(feed, dict) and feed.get("x") is not None and feed.get("y") is not None:
                return feed
        return None

    @staticmethod
    def _parse_optional_edge(value: Any) -> Optional[List[List[float]]]:
        if not isinstance(value, list) or len(value) < 2:
            return None
        points: List[List[float]] = []
        for item in value[:2]:
            if isinstance(item, dict):
                points.append([float(item["x"]), float(item["y"])])
            elif isinstance(item, Sequence) and len(item) >= 2:
                points.append([float(item[0]), float(item[1])])
            else:
                return None
        return points

    @staticmethod
    def _vertices_to_points(vertices: Any) -> List[List[float]]:
        if not isinstance(vertices, list):
            return []
        points: List[List[float]] = []
        for vertex in vertices:
            if isinstance(vertex, dict):
                points.append([float(vertex["x"]), float(vertex["y"])])
            elif isinstance(vertex, Sequence) and len(vertex) >= 2:
                points.append([float(vertex[0]), float(vertex[1])])
        if len(points) >= 2 and math.isclose(points[0][0], points[-1][0]) and math.isclose(points[0][1], points[-1][1]):
            points = points[:-1]
        return points

    @staticmethod
    def _bbox(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _substrate(stackup: Dict[str, Any]) -> Dict[str, Any]:
        substrate = stackup.get("substrate")
        if not isinstance(substrate, dict):
            raise ValueError("stackup.json must contain a `substrate` object.")
        return substrate

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
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

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


if __name__ == "__main__":
    raise SystemExit(main())
