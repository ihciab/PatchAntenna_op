from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Windows + Anaconda + OpenCV / torch / sklearn / matplotlib can load more
# than one Intel OpenMP runtime in the same process.  The conflict commonly
# appears when PortSearch is imported after FSS/YOLO stages.  Set this before
# heavy native libraries are imported so the full pipeline can continue.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REBUILD_DIR = PROJECT_ROOT / "Rebuild"
PDF_ANALYSIS_AGENT_ROOT = next(
    (
        path
        for path in (
            PROJECT_ROOT / "PDF_analy_agent",
            PROJECT_ROOT / "PDFanalyagent",
            PROJECT_ROOT / "test3",
        )
        if (path / "agent").exists()
    ),
    PROJECT_ROOT / "PDF_analy_agent",
)

for path in (PROJECT_ROOT, REBUILD_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from bayesian_optimization.pipelines.fss_simulation_pipeline import FSSImagePreprocessor, write_instance_dict
from bayesian_optimization.geometry.port_summary_utils import (
    DEFAULT_FINAL_FREE_NORMAL_INWARD_PX,
    DEFAULT_FINAL_PORT_WIDTH_SCALE,
    ensure_port_summary_connected_to_geometry,
)
from bayesian_optimization.simulation.parameterized_json_to_cst import (
    ParameterizedJsonCSTBuilder,
    generate_cst_length_annotated_svg,
    load_instance_config,
    make_unique_project_name,
)


# 你可以在这里直接修改调试用 Instance_dict。
# You can edit this inline Instance_dict directly when debugging without
# touching an external JSON file.
DEFAULT_INSTANCE_DICT: Dict[str, Any] = {
    "Folder_path": r"D:\CST2023proj\autocst_MA\202605161643",
    "Instance": "Microstrip_Antenna",
    "mode": "S",
    "Units": ["mm", "GHz"],
    "Antenna_package": {
        "X": 36,
        "Y": 36,
        "f0": 8,
        "f1": 12,
    },
    "layers": {
        "layer0": {
            "img_path": r"D:\cst2py_box\Auto_py2cst_v0.71\test\test51.png",
            "substrate": 0.6,
            "gnd": True,
            "col_mats": {
                "white": "Rogers RT-duroid 5880 (lossy)",
                "gray": "PEC",
            },
        },
    },
}


# 编辑器直接运行配置区：你主要改这里。
# Editor-run configuration: edit this block, then run this .py file directly.
#
# 说明：
# 1. RUN_WITH_INLINE_INSTANCE = False 时，读取 INSTANCE_JSON_PATH 指向的外部 JSON。
# 2. RUN_WITH_INLINE_INSTANCE = True 时，读取上面的 DEFAULT_INSTANCE_DICT。
# 3. BUILD_ONLY = True 只建模不仿真；False 会继续运行 CST solver。
# 4. OUTPUT_ROOT 是所有中间文件的统一输出根目录。
# Notes:
# 1. RUN_WITH_INLINE_INSTANCE=False reads the external JSON at INSTANCE_JSON_PATH.
# 2. RUN_WITH_INLINE_INSTANCE=True reads DEFAULT_INSTANCE_DICT above.
# 3. BUILD_ONLY=True builds geometry only; False also starts CST solver.
# 4. OUTPUT_ROOT stores all intermediate files from this pipeline.
EDITOR_RUN_CONFIG: Dict[str, Any] = {
    "INSTANCE_JSON_PATH": PROJECT_ROOT / "pipeline_test_instance3.json",
    "RUN_WITH_INLINE_INSTANCE": False,
    "OUTPUT_ROOT": PROJECT_ROOT / "pipeline_runs",
    "LAYER_NAME": "layer0",
    "RUN_NAME": None,
    "BUILD_ONLY": False,
    "SIMPLIFY_TOLERANCE_PX": 1.0,
    "GEOMETRY_FRAME": "svg",
    "SKIP_FSS_CLEANUP": False,
    "HONOR_INSTANCE_SKIP": True,
    # 参数化模式 / Parameterization mode:
    # "standard": 当前稳定保底流程，repair_fig.png -> NewParams -> VTracer -> JSON。
    # "optimized_bs_seed": 上一版实验流程，repair_fig.png -> optimized_bs -> VTracer seed -> JSON。
    # "geometry_primitives": 新几何驱动流程，B-spline 只做中间层，最终输出 line/arc/spline primitives。
    # "graph_local_lines": graph-local 拓扑流程，但最终只输出 line primitives。
    #"PARAMETERIZATION_MODE": "standard",
    "PARAMETERIZATION_MODE": "graph_local_lines",
    "LINE_TRIPLET_MERGE_DISTANCE_PX": 3.0,
    "LINE_TRIPLET_MERGE_MAX_ANGLE_DEG": 35.0,
    "REUSE_PROJECT_FOLDER": False,
    "REUSE_PROJECT_NAME": False,
    # 流程展示用：先运行 prompt/model 抽取脚本，只展示控制台输出。
    # 该步骤结束后会继续执行本文件原本的 CST pipeline，不读取抽取结果。
    "RUN_PRE_PROMPT_EXTRACTION": True,
    "PRE_PROMPT_PYTHON_EXE": Path(r"D:\Python\python.exe"),
    "PRE_PROMPT_SCRIPT": PROJECT_ROOT / "design_agent" / "scripts" / "run_antenna_agent.py",
    "PRE_PROMPT_ROOT": PDF_ANALYSIS_AGENT_ROOT
    / "Antenna PDF"
    / "Single-Layer Line-Fed Broadband Microstrip Patch Antenna on",
}


def run_pre_prompt_extraction(
    *,
    enabled: bool,
    python_exe: Path | str,
    script_path: Path | str,
    paper_root: Path | str,
) -> None:
    """Run the prompt extraction demo step and stream its console output."""
    if not enabled:
        print("[FSSParameterizedCSTPipeline] pre prompt extraction skipped")
        return

    python_exe = Path(python_exe)
    script_path = Path(script_path)
    paper_root = Path(paper_root)
    command = [str(python_exe), str(script_path), "--root", str(paper_root)]

    print("\n" + "=" * 78)
    print("[FSSParameterizedCSTPipeline] 0. Pre Prompt Extraction Demo")
    print("=" * 78)
    print(f"[FSSParameterizedCSTPipeline] command: {' '.join(command)}")
    print("[FSSParameterizedCSTPipeline] streaming prompt extraction output below...")

    try:
        result = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    except OSError as exc:
        print(f"[FSSParameterizedCSTPipeline] pre prompt extraction could not start: {exc}")
        print("[FSSParameterizedCSTPipeline] continuing with original CST pipeline.")
        return

    print(
        "[FSSParameterizedCSTPipeline] pre prompt extraction finished "
        f"with returncode={result.returncode}; continuing with original CST pipeline."
    )


class FSSParameterizedCSTPipeline:
    """End-to-end pipeline from instance JSON to CST simulation.

    从 Instance_dict/JSON 开始，串起图片清洗、参数化和 CST 建模/仿真。

    Pipeline stages:
    1. Read a Simulation-style Instance_dict JSON.
    2. Clean the layer image through FSSfigDetector and keep repair_fig.png.
    3. Parameterize repair_fig.png through NewParams / VTracer and save JSON.
    4. Build and optionally solve a CST project from the parameterization JSON.

    Each run writes all intermediate files under one run directory so that the
    inputs, derived images, JSON files and CST outputs can be inspected together.
    """

    def __init__(
        self,
        instance_json: Path | str | None,
        output_root: Path | str,
        layer_name: str = "layer0",
        run_name: Optional[str] = None,
        build_only: bool = False,
        simplify_tolerance_px: float = 1.0,
        geometry_frame: str = "svg",
        parameterization_mode: str = "standard",
        line_triplet_merge_distance_px: float = 3.0,
        line_triplet_merge_max_angle_deg: float = 35.0,
        skip_fss_cleanup: bool = False,
        honor_instance_skip: bool = True,
        reuse_project_folder: bool = False,
        reuse_project_name: bool = False,
        inline_instance: Optional[Dict[str, Any]] = None,
    ):
        # instance_json 和 inline_instance 二选一：前者适合正式运行，后者适合在本文件内快速调参。
        # Use either instance_json for normal runs or inline_instance for quick
        # debugging directly in this file.
        self.instance_json = Path(instance_json) if instance_json is not None else None
        self.inline_instance = copy.deepcopy(inline_instance) if inline_instance is not None else None
        self.output_root = Path(output_root)
        self.layer_name = layer_name
        self.build_only = bool(build_only)
        self.simplify_tolerance_px = float(simplify_tolerance_px)
        self.geometry_frame = geometry_frame
        self.skip_fss_cleanup = bool(skip_fss_cleanup)
        self.honor_instance_skip = bool(honor_instance_skip)
        self.parameterization_mode = str(parameterization_mode).lower().strip()
        self.line_triplet_merge_distance_px = float(line_triplet_merge_distance_px)
        self.line_triplet_merge_max_angle_deg = float(line_triplet_merge_max_angle_deg)
        if self.parameterization_mode not in ("standard", "optimized_bs_seed", "geometry_primitives", "graph_local_primitives", "graph_local_lines"):
            raise ValueError(
                "parameterization_mode must be one of: 'standard', 'optimized_bs_seed', "
                "'geometry_primitives', 'graph_local_primitives', 'graph_local_lines'."
            )
        self.reuse_project_folder = bool(reuse_project_folder)
        self.reuse_project_name = bool(reuse_project_name)
        self._last_image_preprocessor_status: Dict[str, Any] = {}

        self.run_name = run_name or self._default_run_name()
        self.run_dir = self.output_root / self.run_name
        self.clean_dir = self.run_dir / "01_fss_clean"
        self.param_dir = self.run_dir / "02_parameterization"
        self.cst_dir = self.run_dir / "03_cst"
        self.metadata_path = self.run_dir / "pipeline_metadata.json"

    def run(self) -> Path:
        self._prepare_dirs()
        instance = self._load_instance()
        layer_cfg = self._layer_config(instance)

        self._log_header("1. Load Instance JSON")
        if self.instance_json is not None:
            self._log(f"instance_json: {self.instance_json.resolve()}")
        else:
            self._log("instance_json: <inline DEFAULT_INSTANCE_DICT>")
        self._log(f"layer:         {self.layer_name}")
        self._log(f"run_dir:       {self.run_dir.resolve()}")
        self._write_json(self.run_dir / "input_instance.json", instance)

        self._log_header("2. Image Preparation")
        repair_path, image_prep_status = self._prepare_layer_image(instance, layer_cfg)
        prepared_instance = self._prepared_instance(instance, repair_path, image_prep_status)
        prepared_instance_path = self.run_dir / "prepared_instance.json"
        write_instance_dict(prepared_instance, prepared_instance_path)
        if image_prep_status.get("skip_fss_cleanup"):
            self._log(f"direct_image:  {repair_path}")
        else:
            self._log(f"repair_fig:    {repair_path}")
        self._log(f"prepared_json: {prepared_instance_path}")

        self._log_header("3. Parameterization")
        self._log(f"parameterization_mode: {self.parameterization_mode}")
        actual_parameterization_mode = self.parameterization_mode
        parameterization_status: Dict[str, Any] = {}
        try:
            if self.parameterization_mode in {"graph_local_primitives", "graph_local_lines"}:
                parameter_json_path, parameterization_status = self._parameterize_repair_image_via_graph_local_primitives(
                    repair_path,
                    force_line_primitives=self.parameterization_mode == "graph_local_lines",
                )
                if parameterization_status.get("fallback"):
                    if self.parameterization_mode == "graph_local_lines":
                        raise ValueError("graph_local_lines does not allow fallback to non-line parameterization")
                    actual_parameterization_mode = f"{self.parameterization_mode}_internal_fallback"
            elif self.parameterization_mode == "geometry_primitives":
                parameter_json_path, parameterization_status = self._parameterize_repair_image_via_geometry_primitives(repair_path)
                if parameterization_status.get("fallback"):
                    actual_parameterization_mode = "geometry_primitives_standard_topology_fallback"
            elif self.parameterization_mode == "optimized_bs_seed":

                parameter_json_path = self._parameterize_repair_image_via_optimized_bs(repair_path)
            else:
                parameter_json_path = self._parameterize_repair_image(repair_path)
        except Exception as exc:
            if self.parameterization_mode == "standard":
                raise
            self._log(
                "experimental parameterization failed; "
                f"error={exc}"
            )
            if self.parameterization_mode in {"graph_local_primitives", "graph_local_lines"}:
                if self.parameterization_mode == "graph_local_lines":
                    raise
                self._log(f"fallback chain: {self.parameterization_mode} -> geometry_primitives")
                try:
                    parameter_json_path, parameterization_status = self._parameterize_repair_image_via_geometry_primitives(repair_path)
                    actual_parameterization_mode = f"{self.parameterization_mode}_geometry_primitives_fallback"
                    if parameterization_status.get("fallback"):
                        actual_parameterization_mode = f"{self.parameterization_mode}_geometry_primitives_standard_topology_fallback"
                except Exception as second_exc:
                    self._log(
                        "geometry_primitives fallback failed; "
                        f"fallback to standard pipeline. error={second_exc}"
                    )
                    actual_parameterization_mode = f"{self.parameterization_mode}_standard_fallback"
                    parameter_json_path = self._parameterize_repair_image(repair_path)
            else:
                self._log("fallback to standard pipeline.")
                actual_parameterization_mode = "standard_fallback"
                parameter_json_path = self._parameterize_repair_image(repair_path)
        self._log(f"param_json:    {parameter_json_path}")
        cst_length_overlay = self._write_cst_length_overlay(parameter_json_path, prepared_instance_path)

        self._log_header("4. Patch Port Summary")
        raw_port_summary_path = self._create_port_summary(repair_path, layer_cfg, parameter_json_path=parameter_json_path)
        port_summary_path, port_connection_report = self._connect_port_summary_to_parameterization(
            parameter_json_path,
            raw_port_summary_path,
        )
        self._log(f"patch_port_summary_raw:       {raw_port_summary_path}")
        self._log(f"patch_port_summary_for_cst:   {port_summary_path}")

        self._log_header("5. CST Build / Simulation")
        cst_path = self._build_cst(parameter_json_path, port_summary_path)
        self._log(f"cst_project:   {cst_path}")

        metadata = {
            "instance_json": str(self.instance_json.resolve()) if self.instance_json else "<inline DEFAULT_INSTANCE_DICT>",
            "run_dir": str(self.run_dir.resolve()),
            "layer": self.layer_name,
            "repair_fig": str(repair_path),
            "parameterization_image": str(repair_path),
            "fss_cleanup_skipped": bool(image_prep_status.get("skip_fss_cleanup")),
            "image_preparation": image_prep_status,
            "prepared_instance_json": str(prepared_instance_path),
            "patch_port_summary_raw_json": str(raw_port_summary_path),
            "patch_port_summary_json": str(port_summary_path),
            "port_summary_json": str(port_summary_path),
            "port_connection_report": port_connection_report,
            "parameterization_json": str(parameter_json_path),
            "cst_length_overlay_svg": cst_length_overlay.get("annotated_svg"),
            "cst_length_overlay_json": cst_length_overlay.get("report_json"),
            "cst_project": str(cst_path),
            "build_only": self.build_only,
            "simplify_tolerance_px": self.simplify_tolerance_px,
            "geometry_frame": self.geometry_frame,
            "parameterization_mode": self.parameterization_mode,
            "actual_parameterization_mode": actual_parameterization_mode,
            "parameterization_status": parameterization_status,
            "honor_instance_skip": self.honor_instance_skip,
        }
        self._write_json(self.metadata_path, metadata)
        self._log(f"metadata:      {self.metadata_path}")
        return cst_path

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
                "honor_instance_skip": self.honor_instance_skip,
                "configured_skip_flags": self._configured_skip_flags(instance, layer_cfg),
            }
            self._write_json(self.run_dir / "image_preparation.json", status)
            self._log("FSS cleanup skipped; using layer img_path directly.")
            return image_path, status

        repair_path = self._clean_layer_image(layer_cfg)
        preprocessor_status = self._last_image_preprocessor_status or {}
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

    def _clean_layer_image(self, layer_cfg: Dict[str, Any]) -> Path:
        # 图片清洗阶段：复用 FSSfigDetector，最终只使用 repair_fig.png。
        # Cleanup stage: use FSSfigDetector and keep repair_fig.png only.
        from Rebuild.FssDetector_2 import FSSfigDetector as FSSfigDetectorV2
        from Rebuild.fssdetector_ocr import TextSystemOCRAdapter

        ocr = TextSystemOCRAdapter(project_dir=str(PROJECT_ROOT))
        detector = FSSfigDetectorV2(
            max_k=6,
            min_color_diff=30,
            ocr_engine=ocr,
            yolo_model_path=str(PROJECT_ROOT / "models" / "bestyolo.pt"),
        )
        self._log("FSS cleanup detector: Rebuild.FssDetector_2.FSSfigDetector")
        preprocessor = FSSImagePreprocessor(
            output_root=self.clean_dir,
            result_name="repair_fig.png",
            detector=detector,
            normalize_for_simulation=False,
        )
        repair_path = Path(
            preprocessor.process_image(
                image_path=layer_cfg["img_path"],
                layer_name=self.layer_name,
                col_mats=layer_cfg["col_mats"],
                result_index=layer_cfg.get("detector_result_index", 0),
            )
        )
        self._last_image_preprocessor_status = getattr(preprocessor, "last_status", {}) or {}
        if not repair_path.exists():
            raise FileNotFoundError(f"FSS repair image was not created: {repair_path}")
        return repair_path

    def _create_port_summary(
        self,
        image_path: Path,
        layer_cfg: Dict[str, Any],
        parameter_json_path: Optional[Path | str] = None,
    ) -> Path:
        from Rebuild.PortSearch import SubjectEdgeAnalyzer

        # 端口分析阶段：根据 PEC 颜色寻找最接近边界的边，后续映射到 CST 端口。
        # Port stage: locate the PEC edge closest to the image border for CST port setup.
        pec_color = self._pec_color_name(layer_cfg["col_mats"])
        if pec_color is None:
            raise ValueError(f"No PEC color found in layer col_mats: {layer_cfg['col_mats']}")

        analyzer = SubjectEdgeAnalyzer(min_component_area=500, approx_epsilon_ratio=0.0025)
        result = analyzer.analyze(str(image_path), subject_color=pec_color)

        # The old nearest-edge visualization is no longer emitted.  Port debug
        # artifacts now live under 03_port_detection/ and 03_port_geometry/.

        # 贴片天线端口检测是新增的可选增强阶段：
        # - 不替换旧的 closest_edge 字段，避免影响已有 CST waveguide port fallback。
        # - 单独写入 03_port_detection，便于检查 skeleton/endpoints/selected_ports。
        port_detection_dir = self.run_dir / "03_port_detection"
        port_detection_dir = self.run_dir / "03_port_detection"
        patch_port_result = analyzer.detect_patch_ports(
            result,
            border_distance_px=8,
            parameterization_path=parameter_json_path,
            debug_dir=port_detection_dir,
        )
        self._log(
            "patch_port_detection: "
            f"ports={len(patch_port_result.ports)}, "
            f"parameterization_ref={parameter_json_path or '<none>'}, "
            f"debug_dir={port_detection_dir}"
        )

        ports = [port.__dict__ for port in patch_port_result.ports]
        summary = {
            "schema_version": "patch_port_summary_v1",
            "name": f"{self.layer_name}2PEC",
            "image_path": str(image_path),
            "subject_color": pec_color,
            "resolved_subject_color": result.subject_color,
            "debug_dir": str(port_detection_dir),
            "parameterization_json": str(parameter_json_path) if parameter_json_path is not None else None,
            "ports": ports,
            "selected_port": ports[0] if ports else None,
            "port_geometries": patch_port_result.debug_metadata.get("port_geometries", []),
            "debug_metadata": patch_port_result.debug_metadata,
            "patch_port_detection": {
                "enabled": True,
                "source_mask": patch_port_result.debug_metadata.get("source_mask"),
                "fallback_from_subject_mask": bool(patch_port_result.debug_metadata.get("fallback_from_subject_mask", False)),
                "debug_dir": str(port_detection_dir),
                "ports": ports,
                "metadata": patch_port_result.debug_metadata,
            },
        }

        summary_path = self.run_dir / "patch_port_summary.json"
        self._write_json(summary_path, summary)
        return summary_path

    def _connect_port_summary_to_parameterization(
        self,
        parameter_json_path: Path,
        port_summary_path: Path,
    ) -> tuple[Path, Dict[str, Any]]:
        """Create the CST-facing port summary after geometry-aware port adjustment."""

        with Path(parameter_json_path).open("r", encoding="utf-8") as file:
            parameter_payload = json.load(file)
        with Path(port_summary_path).open("r", encoding="utf-8") as file:
            port_summary = json.load(file)

        connected_summary, report = ensure_port_summary_connected_to_geometry(
            parameter_payload,
            port_summary,
            final_free_normal_inward_px=DEFAULT_FINAL_FREE_NORMAL_INWARD_PX,
            final_port_width_scale=DEFAULT_FINAL_PORT_WIDTH_SCALE,
        )

        report_path = self.run_dir / "port_connection_report.json"
        self._write_json(report_path, report)
        self._log(
            "port_connection: "
            f"status={report.get('status')}, "
            f"connected_after={report.get('connected_after')}, "
            f"final_inward_px={report.get('final_free_normal_inward_px')}, "
            f"width_scale={report.get('final_port_width_scale')}, "
            f"report={report_path}"
        )

        if connected_summary is None:
            return port_summary_path, report

        connected_path = self.run_dir / "patch_port_summary_connected.json"
        self._write_json(connected_path, connected_summary)
        return connected_path, report

    def _parameterize_repair_image(self, repair_path: Path) -> Path:
        from Rebuild.NewParams import NewParams

        # 参数化阶段：NewParams 会输出边缘图、SVG、metrics 和 curve_parameterization.json。
        # Parameterization stage: NewParams writes edge image, SVG, metrics and JSON.
        params = NewParams(repair_path, save_dir=self.param_dir, edge_mode="canny")
        parameterizer = params.parameterize(save_dir=self.param_dir)

        json_path = parameterizer.save_json(self.param_dir / "curve_parameterization.json")
        visual_path = parameterizer.visualize(self.param_dir / "curve_parameterization.png")

        metrics = parameterizer.metrics()
        self._log(
            "parameterization: "
            f"components={len(parameterizer.results())}, "
            f"segments={metrics.get('total_semantic_segments')}, "
            f"mean_error={metrics.get('mean_error_px', metrics.get('mean_component_error_px'))}"
        )
        self._log(f"visualization: {visual_path}")
        return json_path

    def _parameterize_repair_image_via_graph_local_primitives(
        self,
        repair_path: Path,
        force_line_primitives: bool = False,
    ) -> tuple[Path, Dict[str, Any]]:
        """Run topology-aware graph local primitive parameterization."""
        from bayesian_optimization.geometry.geometry_graph_parameterizer import GraphBasedLocalSplineParameterizer

        parameterizer = GraphBasedLocalSplineParameterizer(
            image_path=repair_path,
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
        if status.get("fallback"):
            self._log(
                f"{mode_name} fallback: "
                f"{status.get('fallback_reason', '')}"
            )
        return json_path, status

    def _parameterize_repair_image_via_geometry_primitives(self, repair_path: Path) -> tuple[Path, Dict[str, Any]]:
        """Run the new geometry-driven primitive pipeline.

        中文说明：
        这是新的实验主线：先用 B-spline 得到连续、平滑、拓扑更稳定的中间曲线，
        再分解为 compact line / arc / residual spline primitives。原 standard
        流程仍然保留为保底。

        English notes:
        B-spline is only an intermediate representation here. The final JSON is
        compact primitives plus fallback sampled points for CST compatibility.
        """
        from bayesian_optimization.geometry.geometry_driven_parameterizer import GeometryDrivenParameterizer

        parameterizer = GeometryDrivenParameterizer(
            image_path=repair_path,
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
        if status.get("fallback"):
            self._log(
                "geometry_primitives fallback: "
                f"{status.get('fallback_reason', '')}"
            )
        return json_path, status

    def _parameterize_repair_image_via_optimized_bs(self, repair_path: Path) -> Path:
        """Experimental parameterization path using optimized B-spline as VTracer seed.

        实验流程 / Experimental flow:
        1. repair_fig.png -> ImageInitializer，得到居中图和边缘图。
        2. OptimizedBSplineFitter 拟合轮廓，输出 fitting['points']。
        3. 将每条有效拟合折线保存为 .npy seed，交给 VTracerPython 的 centerline_seed_npy_path。
        4. 汇总 VTracer 的 semantic_debug.json，保存为 CST 兼容的 curve_parameterization.json。

        This does not replace the stable NewParams path. It is selected only
        when parameterization_mode == "optimized_bs_seed".
        """
        import shutil

        import cv2
        import numpy as np

        from core.geometry.optimized_bspline_fitter import OptimizedBSplineFitter
        from core.image.initializer import ImageInitializer
        from bayesian_optimization.tools.vtracer_python import TraceConfig, VTracerPython

        exp_dir = self.param_dir / "optimized_bs_seed"
        seeds_dir = exp_dir / "seeds"
        vtracer_dir = exp_dir / "vtracer"
        if exp_dir.exists():
            shutil.rmtree(exp_dir, ignore_errors=True)
        for path in (exp_dir, seeds_dir, vtracer_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._log("optimized_bs: preparing centered repair image and edges")
        image_init = ImageInitializer(str(repair_path), show=False, save="")
        centered_img = image_init.centered_img()
        edges = image_init.edges()
        centered_image_path = exp_dir / "repair_fig_centered.png"
        edges_path = exp_dir / "repair_fig_centered_edges.png"
        self._write_image(centered_image_path, centered_img)
        self._write_image(edges_path, edges)

        self._log("optimized_bs: fitting contours")
        fitter = OptimizedBSplineFitter(
            img=centered_img,
            edges=edges,
            line_threshold=2.0,
            arc_threshold=2.0,
            curvature_threshold=0.15,
            spline_degree=3,
            show=False,
            save="",
        )
        contours_dict = fitter.get_contours_dict()
        seed_records = self._extract_optimized_bs_seed_records(contours_dict)
        if not seed_records:
            raise ValueError("optimized_bs did not produce any valid fitting['points'] seed.")

        contour_summary_path = exp_dir / "optimized_bs_contours_summary.json"
        self._write_json(
            contour_summary_path,
            {
                "repair_path": str(repair_path),
                "centered_image_path": str(centered_image_path),
                "edges_path": str(edges_path),
                "contour_count": len(contours_dict),
                "valid_seed_count": len(seed_records),
                "seeds": [
                    {
                        "source_contour_id": record["source_contour_id"],
                        "point_count": int(len(record["points"])),
                        "path_length_px": float(record["path_length_px"]),
                        "closed_guess": bool(record["closed_guess"]),
                    }
                    for record in seed_records
                ],
            },
        )
        self._log(f"optimized_bs: valid seeds={len(seed_records)}, summary={contour_summary_path}")

        components: List[Dict[str, Any]] = []
        metrics_list: List[Dict[str, Any]] = []
        svg_paths: List[str] = []

        for component_index, record in enumerate(seed_records, start=1):
            points = record["points"]
            seed_path = seeds_dir / f"optimized_bs_seed_{component_index:03d}.npy"
            np.save(str(seed_path), points.astype(np.float64))

            component_dir = vtracer_dir / f"seed_{component_index:03d}"
            intermediate_dir = component_dir / "intermediates"
            svg_path = component_dir / f"optimized_bs_seed_{component_index:03d}.svg"
            metrics_path = component_dir / f"optimized_bs_seed_{component_index:03d}_metrics.json"
            component_dir.mkdir(parents=True, exist_ok=True)
            intermediate_dir.mkdir(parents=True, exist_ok=True)

            cfg = TraceConfig(
                image_path=str(centered_image_path),
                color_mode="bw",
                trace_style="centerline",
                mode="spline",
                metrics_path=str(metrics_path),
                save_intermediates=str(intermediate_dir),
                fit_tolerance=1.5,
                resample_step=3.0,
                filter_speckle=4,
                median_ksize=1,
            )
            cfg.centerline_seed_npy_path = str(seed_path)
            cfg.gaussian_sigma = 1.05
            cfg.semantic_window_size = 12
            cfg.keypoint_angle_threshold_deg = 32.0
            cfg.keypoint_refine_radius = 5
            cfg.keypoint_use_model_guided = True
            cfg.keypoint_model_multiscale_votes = False
            cfg.dp_complexity_weight = 1.02
            cfg.dp_max_segment_points = 78
            cfg.line_merge_angle_deg = 13.5
            cfg.arc_radius_rel_tol = 0.22
            cfg.arc_center_tol = 2.2
            cfg.arc_min_sweep_deg = 17.0
            cfg.spline_ctrl_penalty = 0.11

            self._log(
                "vtracer seed: "
                f"component={component_index}, source={record['source_contour_id']}, "
                f"points={len(points)}, length_px={record['path_length_px']:.2f}"
            )
            tracer = VTracerPython(cfg)
            tracer.to_svg(str(svg_path))
            svg_paths.append(str(svg_path))

            metrics = self._load_json(metrics_path, default={})
            if metrics:
                metrics["source_contour_id"] = record["source_contour_id"]
                metrics["seed_path"] = str(seed_path)
                metrics_list.append(metrics)

            component = self._load_seed_component(
                intermediate_dir=intermediate_dir,
                component_id=component_index,
                source_contour_id=str(record["source_contour_id"]),
                seed_path=seed_path,
            )
            components.append(component)

        aggregate_metrics = self._aggregate_seed_metrics(metrics_list)
        json_path = self.param_dir / "curve_parameterization.json"
        payload = {
            "backend": "vtracer_python_optimized_bs_seed",
            "trace_image_path": str(centered_image_path),
            "svg_path": svg_paths[0] if svg_paths else "",
            "svg_paths": svg_paths,
            "metrics_path": str(exp_dir / "optimized_bs_seed_metrics.json"),
            "metrics": aggregate_metrics,
            "repair_path": str(repair_path),
            "centered_image_path": str(centered_image_path),
            "edges_path": str(edges_path),
            "optimized_bs_summary_path": str(contour_summary_path),
            "components": components,
        }
        self._write_json(json_path, self._to_jsonable(payload))
        self._write_json(exp_dir / "optimized_bs_seed_metrics.json", self._to_jsonable(aggregate_metrics))
        self._write_seed_visualization(centered_img, components, exp_dir / "optimized_bs_seed_visualization.png")

        self._log(
            "optimized_bs_seed summary: "
            f"components={len(components)}, "
            f"segments={aggregate_metrics.get('total_semantic_segments')}, "
            f"mean_error={aggregate_metrics.get('mean_error_px')}"
        )
        return json_path

    def _build_cst(self, parameter_json_path: Path, port_summary_path: Path) -> Path:
        # CST 阶段：几何来自参数化 JSON，仿真参数来自 prepared_instance.json。
        # CST stage: geometry comes from parameterization JSON, simulation
        # settings come from prepared_instance.json.
        config = load_instance_config(self.run_dir / "prepared_instance.json", self.layer_name)
        if not self.reuse_project_folder:
            config.project_folder = self.cst_dir
        if not self.reuse_project_name:
            config.project_name = make_unique_project_name(config.project_name)

        # These flags match the important knobs from parameterized_json_to_cst.py,
        # but the total pipeline owns their values so a single command can run
        # the whole process reproducibly.
        config.run_solver = not self.build_only
        config.port_summary_path = port_summary_path
        config.simplify_tolerance_px = self.simplify_tolerance_px
        config.geometry_frame = self.geometry_frame
        config.close_project = False

        self._log(f"cst_folder:    {config.project_folder}")
        self._log(f"cst_name:      {config.project_name}")
        self._log(f"run_solver:    {config.run_solver}")
        self._log(f"geometry_frame:{config.geometry_frame}")
        self._log(
            "substrate:    "
            f"{config.substrate_material}, thickness={config.substrate_thickness}, "
            f"ground={config.add_ground}"
        )

        builder = ParameterizedJsonCSTBuilder(parameter_json_path, config)
        return builder.build()

    def _write_cst_length_overlay(self, parameter_json_path: Path, instance_json_path: Path) -> Dict[str, Any]:
        config = load_instance_config(instance_json_path, self.layer_name)
        config.geometry_frame = self.geometry_frame
        config.simplify_tolerance_px = self.simplify_tolerance_px
        report = generate_cst_length_annotated_svg(parameter_json_path, config)
        self._log(f"cst_length_overlay: {report.get('annotated_svg')}")
        return report

    def _prepared_instance(
        self,
        instance: Dict[str, Any],
        repair_path: Path,
        image_prep_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 保存一份回填 repair_fig 路径后的 Instance_dict，方便复现本次运行。
        # Keep a copy of Instance_dict with img_path replaced by repair_fig
        # for reproducibility.
        prepared = copy.deepcopy(instance)
        layer_cfg = prepared["layers"][self.layer_name]
        layer_cfg["raw_img_path"] = layer_cfg["img_path"]
        layer_cfg["img_path"] = str(repair_path)
        if image_prep_status is not None:
            layer_cfg["image_preparation"] = image_prep_status
        return prepared

    def _load_instance(self) -> Dict[str, Any]:
        if self.inline_instance is not None:
            return self._normalize_instance_schema(copy.deepcopy(self.inline_instance))

        if self.instance_json is None:
            raise ValueError("Either instance_json or inline_instance must be provided.")
        if not self.instance_json.exists():
            raise FileNotFoundError(f"Instance JSON does not exist: {self.instance_json}")
        with self.instance_json.open("r", encoding="utf-8") as file:
            instance = json.load(file)
        if not isinstance(instance, dict):
            raise ValueError(f"Invalid instance JSON payload: {self.instance_json}")
        return self._normalize_instance_schema(instance)

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

    @staticmethod
    def _pec_color_name(col_mats: Dict[str, str]) -> Optional[str]:
        for color_name, material in col_mats.items():
            if material == "PEC":
                return color_name
        return None

    @staticmethod
    def _normalize_instance_schema(instance: Dict[str, Any]) -> Dict[str, Any]:
        if "Antenna_package" not in instance and "FSS_package" in instance:
            instance["Antenna_package"] = copy.deepcopy(instance["FSS_package"])
        return instance

    def _extract_optimized_bs_seed_records(self, contours_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract valid fitted polylines from OptimizedBSplineFitter output.

        从 optimized_bs 的 contours_dict 中提取可交给 VTracer seed 模式的折线。
        这里保留所有有效轮廓，并按长度从大到小排序，方便后续调试每个 component。
        """
        import numpy as np

        records: List[Dict[str, Any]] = []
        for contour_id, contour_data in contours_dict.items():
            if not isinstance(contour_data, dict):
                continue
            fitting = contour_data.get("fitting", {})
            raw_points = fitting.get("points")
            if raw_points is None:
                continue

            points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
            points = points[np.isfinite(points).all(axis=1)]
            points = self._remove_duplicate_array_points(points)
            if len(points) < 4:
                continue

            path_length = self._polyline_length(points)
            if path_length < 8.0:
                continue

            # findContours 得到的轮廓本质上是闭合轮廓；补上首点可以让
            # VTracer seed 模式稳定识别 closed=True。
            # The source is a contour, so append the first point to make the
            # downstream closed-loop detection explicit.
            endpoint_gap = float(np.linalg.norm(points[0] - points[-1]))
            if endpoint_gap > 1e-9:
                points = np.vstack([points, points[0]])

            records.append(
                {
                    "source_contour_id": str(contour_id),
                    "points": points,
                    "path_length_px": self._polyline_length(points),
                    "closed_guess": True,
                }
            )

        records.sort(key=lambda item: float(item["path_length_px"]), reverse=True)
        return records

    def _load_seed_component(
        self,
        intermediate_dir: Path,
        component_id: int,
        source_contour_id: str,
        seed_path: Path,
    ) -> Dict[str, Any]:
        """Load VTracer semantic debug and convert it to CurveParameterizer JSON shape.

        读取 VTracer 写出的 semantic_debug.json，并转换为 parameterized_json_to_cst.py
        已经支持的 components[*] 格式。
        """
        component_dirs = sorted(
            path
            for path in intermediate_dir.iterdir()
            if path.is_dir() and path.name.startswith("component_")
        )
        if not component_dirs:
            raise FileNotFoundError(f"No VTracer component debug directory found: {intermediate_dir}")

        debug_path = component_dirs[0] / "semantic_debug.json"
        if not debug_path.exists():
            raise FileNotFoundError(f"Missing semantic_debug.json: {debug_path}")

        debug = self._load_json(debug_path, default={})
        primitives = debug.get("primitives") or debug.get("final_segments") or []
        resampled_points = debug.get("resampled_points") or []

        component = {
            "component_id": int(component_id),
            "component_dir": str(component_dirs[0]),
            "debug_path": str(debug_path),
            "source_contour_id": str(source_contour_id),
            "optimized_bs_seed_path": str(seed_path),
            "closed": bool(debug.get("closed", False)),
            "sampled_point_count": int(len(resampled_points)),
            "resampled_points": resampled_points,
            "smoothed_points": debug.get("smoothed_points", []),
            "raw_keypoints": debug.get("raw_keypoints", []),
            "refined_keypoints": debug.get("refined_keypoints", []),
            "keypoints": debug.get("keypoints", []),
            "global_lines": debug.get("global_lines", []),
            "collapsed_full_loop_arc": bool(debug.get("collapsed_full_loop_arc", False)),
            "segments": [
                self._normalize_seed_segment(segment, segment_id)
                for segment_id, segment in enumerate(primitives)
            ],
            "debug": {
                "initial_segments": debug.get("initial_segments", []),
                "dp_segments": debug.get("dp_segments", []),
                "merged_segments": debug.get("merged_segments", []),
                "final_segments": debug.get("final_segments", []),
                "boundary_field_weight": debug.get("boundary_field_weight", 0.0),
                "boundary_field_mean": debug.get("boundary_field_mean", 0.0),
                "boundary_field_max": debug.get("boundary_field_max", 0.0),
            },
        }
        self._log(
            f"seed component {component_id}: closed={component['closed']}, "
            f"points={component['sampled_point_count']}, segments={len(component['segments'])}"
        )
        return component

    @staticmethod
    def _normalize_seed_segment(segment: Dict[str, Any], segment_id: int) -> Dict[str, Any]:
        kind = str(segment.get("kind", segment.get("type", "spline"))).lower()
        if kind == "circle":
            kind = "arc"
        if kind in ("bspline", "curve"):
            kind = "spline"

        out = dict(segment)
        out["segment_id"] = int(segment_id)
        out["kind"] = kind
        out["type"] = kind
        out["start_idx"] = int(out.get("start_idx", 0) or 0)
        out["end_idx"] = int(out.get("end_idx", 0) or 0)
        out["effective_params"] = int(out.get("effective_params", 0) or 0)
        out["max_error"] = float(out.get("max_error", 0.0) or 0.0)
        out["mean_error"] = float(out.get("mean_error", 0.0) or 0.0)
        return out

    @staticmethod
    def _aggregate_seed_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics_list:
            return {
                "component_count": 0,
                "total_semantic_segments": 0,
                "components": [],
            }

        flat_components: List[Dict[str, Any]] = []
        for metrics in metrics_list:
            for component in metrics.get("components", []) or []:
                if isinstance(component, dict):
                    item = dict(component)
                    item["source_contour_id"] = metrics.get("source_contour_id", "")
                    item["seed_path"] = metrics.get("seed_path", "")
                    flat_components.append(item)

        def _sum(key: str) -> int:
            return int(sum(int(metrics.get(key, 0) or 0) for metrics in metrics_list))

        mean_errors = [
            float(component.get("mean_error_px", 0.0) or 0.0)
            for component in flat_components
        ]
        rmse_values = [
            float(component.get("rmse_px", 0.0) or 0.0)
            for component in flat_components
        ]
        max_errors = [
            float(component.get("max_error_px", 0.0) or 0.0)
            for component in flat_components
        ]
        reductions = [
            float(component.get("reduction_ratio", 0.0) or 0.0)
            for component in flat_components
        ]

        return {
            "component_count": len(flat_components),
            "mean_error_px": sum(mean_errors) / max(1, len(mean_errors)),
            "mean_rmse_px": sum(rmse_values) / max(1, len(rmse_values)),
            "max_component_error_px": max(max_errors) if max_errors else 0.0,
            "mean_reduction_ratio": sum(reductions) / max(1, len(reductions)),
            "total_line_segments": _sum("total_line_segments"),
            "total_arc_segments": _sum("total_arc_segments"),
            "total_curve_segments": _sum("total_curve_segments"),
            "total_semantic_segments": _sum("total_semantic_segments"),
            "total_semantic_line": _sum("total_semantic_line"),
            "total_semantic_arc": _sum("total_semantic_arc"),
            "total_semantic_spline": _sum("total_semantic_spline"),
            "components": flat_components,
        }

    def _write_seed_visualization(self, base_img: Any, components: List[Dict[str, Any]], output_path: Path) -> Path:
        """Write a quick overlay for the experimental optimized_bs_seed path.

        写一张快速调试图：左侧是居中 repair 图，右侧是 VTracer seed 语义分段结果。
        """
        import cv2
        import numpy as np

        if base_img.ndim == 2:
            base = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            base = base_img.copy()

        overlay = np.full_like(base, 255)
        colors = {
            "line": (30, 144, 255),
            "arc": (0, 165, 255),
            "spline": (50, 205, 50),
        }
        height, width = overlay.shape[:2]
        for component in components:
            points = np.asarray(component.get("resampled_points", []), dtype=np.float64).reshape(-1, 2)
            if len(points) < 2:
                continue
            clipped = self._clip_array_points(points, width, height)
            cv2.polylines(
                overlay,
                [clipped.reshape(-1, 1, 2)],
                bool(component.get("closed", False)),
                (210, 210, 210),
                1,
                lineType=cv2.LINE_AA,
            )
            for segment in component.get("segments", []):
                segment_points = self._slice_array_points(
                    points,
                    int(segment.get("start_idx", 0)),
                    int(segment.get("end_idx", 0)),
                    bool(component.get("closed", False)),
                )
                if len(segment_points) < 2:
                    continue
                segment_poly = self._clip_array_points(segment_points, width, height)
                kind = str(segment.get("kind", "spline"))
                cv2.polylines(
                    overlay,
                    [segment_poly.reshape(-1, 1, 2)],
                    False,
                    colors.get(kind, (80, 80, 80)),
                    2,
                    lineType=cv2.LINE_AA,
                )

        panel = np.hstack([base, overlay])
        self._write_image(output_path, panel)
        self._log(f"optimized_bs_seed visualization: {output_path}")
        return output_path

    @staticmethod
    def _remove_duplicate_array_points(points: Any, tolerance: float = 1e-9):
        import numpy as np

        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(arr) <= 1:
            return arr
        keep = [arr[0]]
        for point in arr[1:]:
            if float(np.linalg.norm(point - keep[-1])) > tolerance:
                keep.append(point)
        return np.asarray(keep, dtype=np.float64)

    @staticmethod
    def _polyline_length(points: Any) -> float:
        import numpy as np

        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(arr) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))

    @staticmethod
    def _slice_array_points(points: Any, start_idx: int, end_idx: int, closed: bool):
        import numpy as np

        arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(arr) == 0:
            return arr
        n = len(arr)
        start = int(start_idx) % n
        end = int(end_idx) % n
        if closed:
            if start <= end:
                return arr[start:end + 1]
            return np.vstack([arr[start:], arr[:end + 1]])
        start = max(0, min(n - 1, start))
        end = max(0, min(n - 1, end))
        if start <= end:
            return arr[start:end + 1]
        return arr[end:start + 1]

    @staticmethod
    def _clip_array_points(points: Any, width: int, height: int):
        import numpy as np

        arr = np.round(np.asarray(points, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)
        return arr

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_image(path: Path, image: Any) -> None:
        import cv2
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix or ".png"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            raise ValueError(f"Failed to encode image as {suffix}: {path}")
        encoded.tofile(str(path))

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        try:
            import numpy as np
        except Exception:
            np = None

        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        if np is not None:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        return value

    def _prepare_dirs(self) -> None:
        # 每次运行的所有中间文件都写入同一个 run_dir，方便排查。
        # All intermediate files from one run live under the same run_dir.
        for path in (self.run_dir, self.param_dir, self.cst_dir):
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
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _default_run_name() -> str:
        return "run_" + _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _log_header(text: str) -> None:
        print("\n" + "=" * 78)
        print(f"[FSSParameterizedCSTPipeline] {text}")
        print("=" * 78)

    @staticmethod
    def _log(text: str) -> None:
        print(f"[FSSParameterizedCSTPipeline] {text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end FSS cleanup, parameterization and CST simulation pipeline."
    )
    parser.add_argument(
        "--instance-json",
        default=None,
        help="Path to Simulation-style instance JSON, for example pipeline_test_instance.json.",
    )
    parser.add_argument(
        "--use-inline-instance",
        action="store_true",
        help="Use DEFAULT_INSTANCE_DICT defined in this py file instead of --instance-json.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "pipeline_runs"),
        help="Root folder for all intermediate and output files.",
    )
    parser.add_argument("--layer", default="layer0", help="Layer key to process from the instance JSON.")
    parser.add_argument("--run-name", default=None, help="Optional fixed run directory name.")
    parser.add_argument("--build-only", action="store_true", help="Build CST geometry but do not run solver.")
    parser.add_argument(
        "--simplify-tolerance-px",
        type=float,
        default=1.0,
        help="RDP simplification tolerance for CST geometry in parameterization pixel units.",
    )
    parser.add_argument(
        "--geometry-frame",
        choices=["svg", "component"],
        default="svg",
        help="Map CST geometry using the SVG/image canvas or only the component bbox.",
    )
    parser.add_argument(
        "--parameterization-mode",
        choices=["standard", "optimized_bs_seed", "geometry_primitives", "graph_local_primitives", "graph_local_lines"],
        default="standard",
        help="Choose standard NewParams, optimized_bs seed, geometry-driven, graph-local primitive, or graph-local line-only flow.",
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
        help="Use layer img_path directly; do not run FSSfigDetector repair.",
    )
    parser.add_argument(
        "--honor-instance-skip",
        dest="honor_instance_skip",
        action="store_true",
        default=True,
        help="Honor skip_fss_cleanup/skip_fss_repair/use_raw_image_directly flags from the instance JSON.",
    )
    parser.add_argument(
        "--ignore-instance-skip",
        dest="honor_instance_skip",
        action="store_false",
        help="Ignore skip_fss_cleanup/skip_fss_repair/use_raw_image_directly flags from the instance JSON.",
    )
    parser.add_argument(
        "--reuse-project-folder",
        action="store_true",
        help="Use Folder_path from instance JSON instead of the run directory's 03_cst folder.",
    )
    parser.add_argument(
        "--reuse-project-name",
        action="store_true",
        help="Use Instance from instance JSON exactly instead of appending a timestamp.",
    )
    parser.add_argument(
        "--skip-pre-prompt-extraction",
        action="store_true",
        help="Skip the demonstration pre-run of design_agent/scripts/run_antenna_agent.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_inline_instance:
        instance_json = None
        inline_instance = DEFAULT_INSTANCE_DICT
    else:
        if args.instance_json is None:
            raise ValueError("Please pass --instance-json or use --use-inline-instance.")
        instance_json = args.instance_json
        inline_instance = None

    run_pre_prompt_extraction(
        enabled=not args.skip_pre_prompt_extraction,
        python_exe=Path(r"D:\Python\python.exe"),
        script_path=PROJECT_ROOT / "design_agent" / "scripts" / "run_antenna_agent.py",
        paper_root=PDF_ANALYSIS_AGENT_ROOT
        / "Antenna PDF"
        / "Single-Layer Line-Fed Broadband Microstrip Patch Antenna on",
    )

    pipeline = FSSParameterizedCSTPipeline(
        instance_json=instance_json,
        output_root=args.output_root,
        layer_name=args.layer,
        run_name=args.run_name,
        build_only=args.build_only,
        simplify_tolerance_px=args.simplify_tolerance_px,
        geometry_frame=args.geometry_frame,
        parameterization_mode=args.parameterization_mode,
        line_triplet_merge_distance_px=args.line_triplet_merge_distance_px,
        line_triplet_merge_max_angle_deg=args.line_triplet_merge_max_angle_deg,
        skip_fss_cleanup=args.skip_fss_cleanup,
        honor_instance_skip=args.honor_instance_skip,
        reuse_project_folder=args.reuse_project_folder,
        reuse_project_name=args.reuse_project_name,
        inline_instance=inline_instance,
    )
    cst_path = pipeline.run()
    print(f"\n[FSSParameterizedCSTPipeline] DONE: {cst_path}")


def run_from_editor_config() -> None:
    """Run pipeline from EDITOR_RUN_CONFIG.

    从文件内 EDITOR_RUN_CONFIG 读取配置并运行，适合直接在编辑器里点运行。
    """
    use_inline = bool(EDITOR_RUN_CONFIG["RUN_WITH_INLINE_INSTANCE"])
    instance_json = None if use_inline else EDITOR_RUN_CONFIG["INSTANCE_JSON_PATH"]
    inline_instance = DEFAULT_INSTANCE_DICT if use_inline else None

    run_pre_prompt_extraction(
        enabled=bool(EDITOR_RUN_CONFIG.get("RUN_PRE_PROMPT_EXTRACTION", True)),
        python_exe=EDITOR_RUN_CONFIG.get("PRE_PROMPT_PYTHON_EXE", r"D:\Python\python.exe"),
        script_path=EDITOR_RUN_CONFIG.get(
            "PRE_PROMPT_SCRIPT",
            PROJECT_ROOT / "design_agent" / "scripts" / "run_antenna_agent.py",
        ),
        paper_root=EDITOR_RUN_CONFIG.get(
            "PRE_PROMPT_ROOT",
            PDF_ANALYSIS_AGENT_ROOT
            / "Antenna PDF"
            / "Single-Layer Line-Fed Broadband Microstrip Patch Antenna on",
        ),
    )

    pipeline = FSSParameterizedCSTPipeline(
        instance_json=instance_json,
        output_root=EDITOR_RUN_CONFIG["OUTPUT_ROOT"],
        layer_name=EDITOR_RUN_CONFIG["LAYER_NAME"],
        run_name=EDITOR_RUN_CONFIG["RUN_NAME"],
        build_only=EDITOR_RUN_CONFIG["BUILD_ONLY"],
        simplify_tolerance_px=EDITOR_RUN_CONFIG["SIMPLIFY_TOLERANCE_PX"],
        geometry_frame=EDITOR_RUN_CONFIG["GEOMETRY_FRAME"],
        parameterization_mode=EDITOR_RUN_CONFIG["PARAMETERIZATION_MODE"],
        line_triplet_merge_distance_px=EDITOR_RUN_CONFIG.get("LINE_TRIPLET_MERGE_DISTANCE_PX", 3.0),
        line_triplet_merge_max_angle_deg=EDITOR_RUN_CONFIG.get("LINE_TRIPLET_MERGE_MAX_ANGLE_DEG", 35.0),
        skip_fss_cleanup=EDITOR_RUN_CONFIG.get("SKIP_FSS_CLEANUP", False),
        honor_instance_skip=EDITOR_RUN_CONFIG.get("HONOR_INSTANCE_SKIP", False),
        reuse_project_folder=EDITOR_RUN_CONFIG["REUSE_PROJECT_FOLDER"],
        reuse_project_name=EDITOR_RUN_CONFIG["REUSE_PROJECT_NAME"],
        inline_instance=inline_instance,
    )
    cst_path = pipeline.run()
    print(f"\n[FSSParameterizedCSTPipeline] DONE: {cst_path}")


if __name__ == "__main__":
    # 没有命令行参数时，默认走文件内配置；有命令行参数时，仍保留 CLI 调试能力。
    # With no CLI arguments, use the editor config. CLI remains available
    # for occasional scripted runs.
    if len(sys.argv) == 1:
        run_from_editor_config()
    else:
        main()
