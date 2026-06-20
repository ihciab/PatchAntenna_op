from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REBUILD_DIR = PROJECT_ROOT / "Rebuild"

for path in (PROJECT_ROOT, REBUILD_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from bayesian_optimization.pipelines.fss_simulation_pipeline import FSSImagePreprocessor, write_instance_dict
from bayesian_optimization.simulation.parameterized_json_to_cst import (
    generate_cst_length_annotated_svg,
    load_instance_config,
)


class ParameterizationOnlyRunner:
    """Run FSS image repair and parameterization without CST build/simulation."""

    def __init__(
        self,
        instance_json: Path | str,
        output_root: Path | str,
        layer_name: str = "layer0",
        run_name: Optional[str] = None,
        parameterization_mode: str = "graph_local_primitives",
        line_triplet_merge_distance_px: float = 3.0,
        line_triplet_merge_max_angle_deg: float = 35.0,
        skip_fss_cleanup: bool = False,
        honor_instance_skip: bool = True,
    ):
        self.instance_json = Path(instance_json)
        self.output_root = Path(output_root)
        self.layer_name = str(layer_name)
        self.parameterization_mode = str(parameterization_mode).lower().strip()
        self.line_triplet_merge_distance_px = float(line_triplet_merge_distance_px)
        self.line_triplet_merge_max_angle_deg = float(line_triplet_merge_max_angle_deg)
        if self.parameterization_mode not in ("standard", "geometry_primitives", "graph_local_primitives", "graph_local_lines"):
            raise ValueError(
                "parameterization_mode must be one of: standard, geometry_primitives, graph_local_primitives, graph_local_lines"
            )
        self.skip_fss_cleanup = bool(skip_fss_cleanup)
        self.honor_instance_skip = bool(honor_instance_skip)

        self.run_name = run_name or self._default_run_name()
        self.run_dir = self.output_root / self.run_name
        self.clean_dir = self.run_dir / "01_fss_clean"
        self.param_dir = self.run_dir / "02_parameterization"
        self.metadata_path = self.run_dir / "parameterization_only_metadata.json"

    def run(self) -> Path:
        self._prepare_dirs()
        instance = self._load_instance()
        layer_cfg = self._layer_config(instance)

        self._log_header("1. Load Instance JSON")
        self._log(f"instance_json: {self.instance_json.resolve()}")
        self._log(f"layer:         {self.layer_name}")
        self._log(f"run_dir:       {self.run_dir.resolve()}")
        self._write_json(self.run_dir / "input_instance.json", instance)

        self._log_header("2. Image Preparation")
        image_path, image_status = self._prepare_layer_image(instance, layer_cfg)
        prepared_instance = self._prepared_instance(instance, image_path, image_status)
        prepared_instance_path = self.run_dir / "prepared_instance.json"
        write_instance_dict(prepared_instance, prepared_instance_path)
        self._log(f"parameterization_image: {image_path}")
        self._log(f"prepared_json:          {prepared_instance_path}")

        self._log_header("3. Parameterization Only")
        self._log(f"parameterization_mode: {self.parameterization_mode}")
        actual_mode = self.parameterization_mode
        status: Dict[str, Any] = {}
        try:
            if self.parameterization_mode in {"graph_local_primitives", "graph_local_lines"}:
                json_path, status = self._parameterize_via_graph_local_primitives(
                    image_path,
                    force_line_primitives=self.parameterization_mode == "graph_local_lines",
                )
                if status.get("fallback"):
                    if self.parameterization_mode == "graph_local_lines":
                        raise ValueError("graph_local_lines does not allow fallback to non-line parameterization")
                    actual_mode = f"{self.parameterization_mode}_internal_fallback"
            elif self.parameterization_mode == "geometry_primitives":
                json_path, status = self._parameterize_via_geometry_primitives(image_path)
                if status.get("fallback"):
                    actual_mode = "geometry_primitives_standard_topology_fallback"
            else:
                json_path = self._parameterize_standard(image_path)
        except Exception as exc:
            if self.parameterization_mode == "standard":
                raise
            self._log(f"{self.parameterization_mode} failed: {exc}")
            if self.parameterization_mode in {"graph_local_primitives", "graph_local_lines"}:
                if self.parameterization_mode == "graph_local_lines":
                    raise
                self._log(f"fallback chain: {self.parameterization_mode} -> geometry_primitives")
                try:
                    json_path, status = self._parameterize_via_geometry_primitives(image_path)
                    actual_mode = f"{self.parameterization_mode}_geometry_primitives_fallback"
                    if status.get("fallback"):
                        actual_mode = f"{self.parameterization_mode}_geometry_primitives_standard_topology_fallback"
                except Exception as second_exc:
                    self._log(f"geometry_primitives failed: {second_exc}")
                    self._log("fallback chain: geometry_primitives -> standard")
                    json_path = self._parameterize_standard(image_path)
                    actual_mode = f"{self.parameterization_mode}_standard_fallback"
            else:
                self._log("fallback chain: geometry_primitives -> standard")
                json_path = self._parameterize_standard(image_path)
                actual_mode = "geometry_primitives_standard_fallback"

        length_overlay = self._write_cst_length_overlay(json_path, prepared_instance_path)

        metadata = {
            "mode": "parameterization_only",
            "instance_json": str(self.instance_json.resolve()),
            "run_dir": str(self.run_dir.resolve()),
            "layer": self.layer_name,
            "parameterization_image": str(image_path),
            "prepared_instance_json": str(prepared_instance_path),
            "parameterization_json": str(json_path),
            "cst_length_overlay_svg": length_overlay.get("annotated_svg"),
            "cst_length_overlay_json": length_overlay.get("report_json"),
            "parameterization_mode": self.parameterization_mode,
            "actual_parameterization_mode": actual_mode,
            "image_preparation": image_status,
            "parameterization_status": status,
            "honor_instance_skip": self.honor_instance_skip,
            "cst_build_skipped": True,
            "cst_solver_skipped": True,
        }
        self._write_json(self.metadata_path, metadata)
        self._log(f"param_json: {json_path}")
        self._log(f"metadata:   {self.metadata_path}")
        self._log("DONE: parameterization completed; CST build/simulation skipped.")
        return json_path

    def _prepare_layer_image(self, instance: Dict[str, Any], layer_cfg: Dict[str, Any]) -> tuple[Path, Dict[str, Any]]:
        if self._should_skip_fss_cleanup(instance, layer_cfg):
            image_path = Path(layer_cfg["img_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Layer image does not exist: {image_path}")
            status = {
                "stage": "direct_input_image",
                "skip_fss_cleanup": True,
                "source_image": str(image_path),
                "reason": "skip_fss_cleanup flag is enabled",
            }
            self._write_json(self.run_dir / "image_preparation.json", status)
            self._log("FSS cleanup skipped; using layer img_path directly.")
            return image_path, status

        preprocessor = FSSImagePreprocessor(output_root=self.clean_dir, result_name="repair_fig.png")
        try:
            repair_path = Path(
                preprocessor.process_image(
                    image_path=layer_cfg["img_path"],
                    layer_name=self.layer_name,
                    col_mats=layer_cfg["col_mats"],
                    result_index=layer_cfg.get("detector_result_index", 0),
                )
            )
        except ModuleNotFoundError as exc:
            image_path = Path(layer_cfg["img_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Layer image does not exist: {image_path}") from exc
            status = {
                "stage": "direct_input_image",
                "skip_fss_cleanup": True,
                "source_image": str(image_path),
                "reason": "fss_detector_dependency_missing",
                "dependency_error": str(exc),
                "honor_instance_skip": self.honor_instance_skip,
                "configured_skip_flags": self._configured_skip_flags(instance, layer_cfg),
            }
            self._write_json(self.run_dir / "image_preparation.json", status)
            self._log(
                "FSS cleanup skipped because detector dependency is missing; "
                f"{exc}. using layer img_path directly."
            )
            return image_path, status
        if not repair_path.exists():
            raise FileNotFoundError(f"FSS repair image was not created: {repair_path}")
        preprocessor_status = getattr(preprocessor, "last_status", {}) or {}
        detector_passthrough = bool(preprocessor_status.get("skip_fss_processing", False))
        status = {
            "stage": "detector_passthrough" if detector_passthrough else "fss_repair",
            "skip_fss_cleanup": detector_passthrough,
            "honor_instance_skip": self.honor_instance_skip,
            "ignored_instance_skip_flags": self._configured_skip_flags(instance, layer_cfg),
            "source_image": str(layer_cfg["img_path"]),
            "repair_fig": str(repair_path),
            "detector_passthrough": detector_passthrough,
            "preprocessor_status": preprocessor_status,
        }
        if detector_passthrough:
            status["reason"] = preprocessor_status.get("skip_reason", "detector_passthrough")
            self._log(
                "FSS cleanup skipped by detector; "
                f"reason={status['reason']}, using passthrough repair_fig."
            )
        self._write_json(self.run_dir / "image_preparation.json", status)
        return repair_path, status

    def _parameterize_standard(self, image_path: Path) -> Path:
        from Rebuild.NewParams import NewParams

        params = NewParams(image_path, save_dir=self.param_dir, edge_mode="canny")
        parameterizer = params.parameterize(save_dir=self.param_dir)
        json_path = parameterizer.save_json(self.param_dir / "curve_parameterization.json")
        visual_path = parameterizer.visualize(self.param_dir / "curve_parameterization.png")
        metrics = parameterizer.metrics()
        self._log(
            "standard summary: "
            f"components={len(parameterizer.results())}, "
            f"segments={metrics.get('total_semantic_segments')}, "
            f"mean_error={metrics.get('mean_error_px', metrics.get('mean_component_error_px'))}"
        )
        self._log(f"visualization: {visual_path}")
        return json_path

    def _parameterize_via_geometry_primitives(self, image_path: Path) -> tuple[Path, Dict[str, Any]]:
        from bayesian_optimization.geometry.geometry_driven_parameterizer import GeometryDrivenParameterizer

        parameterizer = GeometryDrivenParameterizer(
            image_path=image_path,
            save_dir=self.param_dir,
            line_tolerance_px=1.6,
            arc_tolerance_px=1.5,
            residual_spline_tolerance_px=2.2,
            resample_step_px=2.5,
            bspline_smoothing=8.0,
            arc_min_sweep_deg=28.0,
            arc_min_error_improvement_ratio=0.55,
            arc_max_radius_to_chord_ratio=4.0,
            arc_min_source_points=24,
            max_centerline_components_for_geometry=20,
            max_bspline_length_shrink_ratio=0.08,
            max_bspline_rms_error_px=2.0,
            close_parallel_edge_distance_px=3,
            close_parallel_edge_ratio_threshold=0.35,
            close_parallel_merge_distance_px=2,
            max_stroke_mask_foreground_ratio=0.08,
            max_stroke_mask_to_canny_ratio=3.0,
            max_sparse_auto_to_canny_ratio=0.45,
            min_merged_canny_retention_ratio=0.50,
            max_decompose_depth=12,
            min_segment_points=8,
            min_component_length_px=10.0,
        )
        json_path = parameterizer.run()
        status = getattr(parameterizer, "last_status", {})
        self._log(f"geometry_primitives json: {json_path}")
        return json_path, status

    def _parameterize_via_graph_local_primitives(
        self,
        image_path: Path,
        force_line_primitives: bool = False,
    ) -> tuple[Path, Dict[str, Any]]:
        from bayesian_optimization.geometry.geometry_graph_parameterizer import GraphBasedLocalSplineParameterizer

        parameterizer = GraphBasedLocalSplineParameterizer(
            image_path=image_path,
            save_dir=self.param_dir,
            force_line_primitives=force_line_primitives,
            line_tolerance_px=1.6,
            arc_tolerance_px=1.5,
            residual_spline_tolerance_px=2.2,
            resample_step_px=2.5,
            arc_min_sweep_deg=28.0,
            arc_min_error_improvement_ratio=0.55,
            arc_max_radius_to_chord_ratio=4.0,
            arc_min_source_points=24,
            max_centerline_components_for_geometry=30,
            close_parallel_edge_distance_px=3,
            close_parallel_edge_ratio_threshold=0.35,
            close_parallel_merge_distance_px=2,
            max_stroke_mask_foreground_ratio=0.08,
            max_stroke_mask_to_canny_ratio=3.0,
            max_sparse_auto_to_canny_ratio=0.45,
            min_merged_canny_retention_ratio=0.50,
            max_decompose_depth=12,
            min_segment_points=8,
            min_component_length_px=10.0,
            graph_node_merge_tolerance_px=3.0,
            graph_endpoint_snap_tolerance_px=4.0,
            graph_corner_angle_threshold_deg=38.0,
            graph_curvature_split_percentile=92.0,
            max_local_spline_rms_error_px=2.0,
            max_local_spline_length_shrink_ratio=0.04,
            line_triplet_merge_distance_px=self.line_triplet_merge_distance_px,
            line_triplet_merge_max_angle_deg=self.line_triplet_merge_max_angle_deg,
        )
        json_path = parameterizer.run()
        status = getattr(parameterizer, "last_status", {})
        mode_name = "graph_local_lines" if force_line_primitives else "graph_local_primitives"
        self._log(f"{mode_name} json: {json_path}")
        return json_path, status

    def _should_skip_fss_cleanup(self, instance: Dict[str, Any], layer_cfg: Dict[str, Any]) -> bool:
        if self.skip_fss_cleanup:
            return True
        if not self.honor_instance_skip:
            return False

        for key in ("skip_fss_cleanup", "skip_fss_repair", "use_raw_image_directly"):
            if key in layer_cfg:
                return self._as_bool(layer_cfg.get(key))

        preprocess_cfg = instance.get("preprocess", {})
        if isinstance(preprocess_cfg, dict):
            for key in ("skip_fss_cleanup", "skip_fss_repair", "use_raw_image_directly"):
                if key in preprocess_cfg:
                    return self._as_bool(preprocess_cfg.get(key))

        for key in ("skip_fss_cleanup", "skip_fss_repair", "use_raw_image_directly"):
            if key in instance:
                return self._as_bool(instance.get(key))

        return False

    @staticmethod
    def _configured_skip_flags(instance: Dict[str, Any], layer_cfg: Dict[str, Any]) -> Dict[str, Any]:
        flags: Dict[str, Any] = {}
        for key in ("skip_fss_cleanup", "skip_fss_repair", "use_raw_image_directly"):
            if key in layer_cfg:
                flags[f"layer.{key}"] = layer_cfg.get(key)

        preprocess_cfg = instance.get("preprocess", {})
        if isinstance(preprocess_cfg, dict):
            for key in ("skip_fss_cleanup", "skip_fss_repair", "use_raw_image_directly"):
                if key in preprocess_cfg:
                    flags[f"preprocess.{key}"] = preprocess_cfg.get(key)

        for key in ("skip_fss_cleanup", "skip_fss_repair", "use_raw_image_directly"):
            if key in instance:
                flags[f"instance.{key}"] = instance.get(key)
        return flags

    def _prepared_instance(
        self,
        instance: Dict[str, Any],
        image_path: Path,
        image_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        prepared = copy.deepcopy(instance)
        layer_cfg = prepared["layers"][self.layer_name]
        layer_cfg["raw_img_path"] = layer_cfg["img_path"]
        layer_cfg["img_path"] = str(image_path)
        layer_cfg["image_preparation"] = image_status
        return prepared

    def _load_instance(self) -> Dict[str, Any]:
        if not self.instance_json.exists():
            raise FileNotFoundError(f"Instance JSON does not exist: {self.instance_json}")
        with self.instance_json.open("r", encoding="utf-8") as file:
            instance = json.load(file)
        if not isinstance(instance, dict):
            raise ValueError(f"Invalid instance JSON payload: {self.instance_json}")
        return instance

    def _write_cst_length_overlay(self, json_path: Path, prepared_instance_path: Path) -> Dict[str, Any]:
        config = load_instance_config(prepared_instance_path, self.layer_name)
        config.geometry_frame = "svg"
        report = generate_cst_length_annotated_svg(json_path, config)
        self._log(f"cst_length_overlay: {report.get('annotated_svg')}")
        return report

    def _layer_config(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        layers = instance.get("layers", {})
        if self.layer_name not in layers:
            available = ", ".join(str(name) for name in layers.keys())
            raise KeyError(f"Layer `{self.layer_name}` not found. Available layers: {available}")
        layer_cfg = layers[self.layer_name]
        for key in ("img_path", "col_mats"):
            if key not in layer_cfg:
                raise KeyError(f"Layer `{self.layer_name}` missing required key `{key}`")
        return layer_cfg

    def _prepare_dirs(self) -> None:
        for path in (self.run_dir, self.param_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _default_run_name() -> str:
        return "param_only_" + _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _log_header(text: str) -> None:
        print("\n" + "=" * 78)
        print(f"[ParameterizationOnlyRunner] {text}")
        print("=" * 78)

    @staticmethod
    def _log(text: str) -> None:
        print(f"[ParameterizationOnlyRunner] {text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FSS repair and antenna parameterization; skip CST build and solver."
    )
    parser.add_argument(
        "--instance-json",
        default=str(PROJECT_ROOT / "pipeline_test_instance.json"),
        help="Simulation-style instance JSON. Default: pipeline_test_instance.json.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "pipeline_runs"),
        help="Root folder for parameterization-only runs.",
    )
    parser.add_argument("--layer", default="layer0", help="Layer key to parameterize.")
    parser.add_argument("--run-name", default=None, help="Optional fixed run directory name.")
    parser.add_argument(
        "--parameterization-mode",
        choices=["standard", "geometry_primitives", "graph_local_primitives", "graph_local_lines"],
        default="graph_local_primitives",
        help="Parameterization backend to test.",
    )
    parser.add_argument(
        "--line-triplet-merge-distance-px",
        type=float,
        default=3.0,
        help="graph_local_lines only: merge the middle point of close same-trend triplets within this pixel distance.",
    )
    parser.add_argument(
        "--line-triplet-merge-max-angle-deg",
        type=float,
        default=35.0,
        help="graph_local_lines only: maximum local angle change allowed when simplifying close triplets.",
    )
    parser.add_argument(
        "--skip-fss-cleanup",
        action="store_true",
        help="Explicitly use layer img_path directly instead of running FSS repair.",
    )
    parser.add_argument(
        "--honor-instance-skip",
        action="store_true",
        default=True,
        help="Honor skip_fss_cleanup/skip_fss_repair/use_raw_image_directly flags from the instance JSON.",
    )
    parser.add_argument(
        "--ignore-instance-skip",
        action="store_false",
        dest="honor_instance_skip",
        help="Ignore skip flags from the instance JSON and run detector/FSS preprocessing unless --skip-fss-cleanup is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = ParameterizationOnlyRunner(
        instance_json=args.instance_json,
        output_root=args.output_root,
        layer_name=args.layer,
        run_name=args.run_name,
        parameterization_mode=args.parameterization_mode,
        line_triplet_merge_distance_px=args.line_triplet_merge_distance_px,
        line_triplet_merge_max_angle_deg=args.line_triplet_merge_max_angle_deg,
        skip_fss_cleanup=args.skip_fss_cleanup,
        honor_instance_skip=args.honor_instance_skip,
    )
    json_path = runner.run()
    print(f"\n[ParameterizationOnlyRunner] PARAMETERIZATION JSON: {json_path}")


if __name__ == "__main__":
    main()
