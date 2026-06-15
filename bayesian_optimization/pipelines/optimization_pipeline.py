from __future__ import annotations

"""外挂式 Bayesian Optimization 主流程。

本文件负责把已有 parameterization JSON 接入优化循环：
curve_parameterization.json -> 变量采样 -> 几何变异 -> Python 几何验证
-> CST build/simulation -> S11 解析 -> objective 记录 -> 下一轮采样。

重要边界：
- 不修改 CST Builder；
- 不修改已有 parameterization backend；
- 不修改 objective 定义；
- 只在已有流程之后/之前做外挂式优化调度。
"""

import argparse
import datetime as _datetime
import json
import logging
import math
import os
import shutil
import sys
import time
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
VERSIONED_LOCAL_PACKAGES_DIR = PROJECT_ROOT / f"local_packages_py{sys.version_info.major}{sys.version_info.minor}"
if VERSIONED_LOCAL_PACKAGES_DIR.exists():
    local_packages_text = str(VERSIONED_LOCAL_PACKAGES_DIR)
    if local_packages_text not in sys.path:
        sys.path.append(local_packages_text)

MAX_ALLOWED_EVALUATIONS = 30
from bayesian_optimization.geometry.primitive_analyzer import DEFAULT_CURVE_PARAMETERIZATION_MODE  # noqa: E402


# ============================================================================
# 编辑器直接运行配置区
# ============================================================================
# 用法：
# 1. 在 IDE 中直接运行 optimization_pipeline.py 时，如果没有命令行参数，
#    会自动使用这里的配置。
# 2. 命令行传入 --parameter-json 等参数时，仍然走原来的 argparse 流程。
# 3. 调试建议：如果只想测试几何/CST 建模，可临时设置 BUILD_ONLY=True。
#    如果要看谐振频率和 S11 优化结果，必须设置 BUILD_ONLY=False。
EDITOR_RUN_CONFIG: Dict[str, Any] = {
    "RUN_WITH_EDITOR_CONFIG": True,
    "BASE_RUN_DIR": PROJECT_ROOT / "pipeline_runs" / "run_20260609_181608",
    "INSTANCE_JSON_PATH": PROJECT_ROOT / "pipeline_test_instance2.json",
    "OUTPUT_ROOT": PROJECT_ROOT / "optimization_runs",
    "RUN_NAME": "bo_editor_full30_213238",
    "LAYER_NAME": "layer0",
    "MAX_EVALUATIONS": 30,
    "TARGET_FREQUENCY_GHZ": 10.0,
    "TARGET_S11_DB": -10.0,
    "CONVERGENCE_THRESHOLD": 0.02,
    "NO_IMPROVEMENT_PATIENCE": 8,
    "MAX_INVALID_RATIO": 0.50,
    "MAX_CONSECUTIVE_CST_FAILURES": 5,
    "BUILD_ONLY": False,
    "SIMPLIFY_TOLERANCE_PX": 1.0,
    "GEOMETRY_FRAME": "svg",
    "CURVE_PARAMETERIZATION_MODE": DEFAULT_CURVE_PARAMETERIZATION_MODE,
    "RANDOM_STATE": 42,
    # "auto": 优先 Optuna，失败则回退 skopt。
    # "optuna": 强制使用 Optuna TPESampler。
    # "skopt": 强制使用 scikit-optimize GP/EI。
    "OPTIMIZER_BACKEND": "optuna",
}

from bayesian_optimization.geometry.geometry_validator import (  # noqa: E402
    GeometryValidationConfig,
    TopologySignature,
    geometry_complexity_metrics,
    make_topology_signature,
    validate_geometry,
)
from bayesian_optimization.geometry.geometry_validation import validate_geometry as validate_and_repair_cst_geometry  # noqa: E402
from bayesian_optimization.geometry.port_summary_utils import ensure_port_summary_connected_to_geometry  # noqa: E402
from bayesian_optimization.geometry.primitive_mutator import (  # noqa: E402
    DesignVariable,
    PrimitiveInventory,
    extract_design_variables,
    mutate_geometry,
)
from bayesian_optimization.optimization.optimization_objectives import (  # noqa: E402
    ObjectiveBreakdown,
    ObjectiveWeights,
    evaluate_objective,
    set_current_objective_profile,
)
from bayesian_optimization.optimization.objective_factory import create_objective_profile_from_instance_path  # noqa: E402
from bayesian_optimization.optimization.s11_parser import S11Metrics, find_latest_s11_file, parse_s11_file  # noqa: E402


# =============================================================================
# CST handoff helper：把图结构 open edges 临时合并成 CST 可拉伸闭合 loop
# =============================================================================


def prepare_cst_handoff_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # 【关键函数】仅在 BO->CST 交接前做临时 closed-loop handoff，不改原始 JSON schema。
    """把优化后的参数化 JSON 转成更适合 CST solid extrusion 的临时 handoff JSON。

    graph_local_primitives 常把一个闭合导体外轮廓拆成多条 open edge
    component。现有 CST builder 会逐 component 执行 ExtrudeCurve，而 CST
    要求被 ExtrudeCurve 的曲线必须闭合。这里在优化层内把首尾相连的
    open edges 合并回 closed loop；不修改原始 pipeline，也不改 CST builder。
    """

    components = payload.get("components", []) or []
    if not components:
        return payload
    if any(bool(component.get("closed", False)) for component in components):
        return payload
    if not all(component.get("start_node") is not None and component.get("end_node") is not None for component in components):
        return payload

    loops = _merge_open_edge_components_to_closed_loops(components)
    if not loops:
        return payload

    handoff = copy.deepcopy(payload)
    handoff["components"] = loops
    metadata = handoff.setdefault("optimization_metadata", {})
    metadata["cst_handoff"] = {
        "strategy": "merge_graph_open_edges_to_closed_loops",
        "source_component_count": len(components),
        "handoff_component_count": len(loops),
        "reason": "CST ExtrudeCurve requires closed planar profiles.",
    }
    return handoff


def _merge_open_edge_components_to_closed_loops(components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把首尾相接的 open-edge components 合并成闭合 loop components。"""
    unused = set(range(len(components)))
    loops: List[Dict[str, Any]] = []
    loop_index = 0

    while unused:
        first_index = min(unused)
        first = components[first_index]
        start_node = first.get("start_node")
        current_node = first.get("end_node")
        ordered_indices = [first_index]
        unused.remove(first_index)

        while current_node != start_node:
            next_index = _find_next_component_index(components, unused, current_node)
            if next_index is None:
                return []
            next_component = components[next_index]
            ordered_indices.append(next_index)
            unused.remove(next_index)
            current_node = next_component.get("end_node")

        points: List[List[float]] = []
        source_ids: List[Any] = []
        for order, component_index in enumerate(ordered_indices):
            component = components[component_index]
            component_points = _component_handoff_points(component)
            if len(component_points) < 2:
                return []
            if order > 0 and points and _same_point(points[-1], component_points[0]):
                points.extend(component_points[1:])
            else:
                points.extend(component_points)
            source_ids.append(component.get("component_id", component_index))

        if len(points) < 3:
            return []
        if not _same_point(points[0], points[-1]):
            points.append(list(points[0]))

        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        loops.append(
            {
                "component_id": f"cst_handoff_loop_{loop_index}",
                "closed": True,
                "source_component_ids": source_ids,
                "resampled_points": points,
                "fallback_points": points,
                "primitives": [],
                "segments": [],
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "optimization_handoff": True,
            }
        )
        loop_index += 1

    return loops


def _find_next_component_index(
    components: List[Dict[str, Any]],
    unused: set,
    current_node: Any,
) -> Optional[int]:
    """根据当前 loop 末端节点寻找下一条可连接 component。"""
    for index in sorted(unused):
        if components[index].get("start_node") == current_node:
            return index
    return None


def _component_handoff_points(component: Dict[str, Any]) -> List[List[float]]:
    """读取 component 中用于 handoff 的二维点列。"""
    raw_points = (
        component.get("resampled_points")
        or component.get("fallback_points")
        or component.get("points")
        or []
    )
    points: List[List[float]] = []
    for point in raw_points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append([float(point[0]), float(point[1])])
    return points


def _same_point(a: List[float], b: List[float], tolerance: float = 1e-6) -> bool:
    """按 tolerance 判断两个二维点是否相同。"""
    return abs(float(a[0]) - float(b[0])) <= tolerance and abs(float(a[1]) - float(b[1])) <= tolerance


# =============================================================================
# Optimizer backend：Optuna 优先，skopt 作为可选回退
# =============================================================================


class OptunaBackend:
    """Optuna TPESampler 后端封装，向主循环提供 ask/tell 接口。"""
    name = "optuna_tpe"

    def __init__(self, random_state: int, max_evaluations: int) -> None:
        """初始化 Optuna study。"""
        import optuna

        sampler = optuna.samplers.TPESampler(
            seed=random_state,
            n_startup_trials=min(5, max(1, max_evaluations // 3)),
            multivariate=False,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

    def ask(self, variables: List[DesignVariable]):
        """根据当前变量空间请求下一组采样值。"""
        trial = self.study.ask()
        values = {
            variable.name: float(trial.suggest_float(variable.name, variable.lower, variable.upper))
            for variable in variables
        }
        return trial, values

    def tell(self, token: Any, objective: float) -> None:
        """把当前 evaluation 的 objective 回传给 Optuna。"""
        self.study.tell(token, float(objective))

    def trials(self) -> List[Dict[str, Any]]:
        """导出 Optuna trial 历史，写入 optimizer_trials.json。"""
        return [
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": dict(trial.params),
            }
            for trial in self.study.trials
        ]

    def param_importances(self) -> Dict[str, float]:
        """计算 Optuna 参数重要性；样本不足时可能抛异常，由调用方捕获。"""
        import optuna

        return {
            key: float(value)
            for key, value in optuna.importance.get_param_importances(self.study).items()
        }


class SkoptBackend:
    """scikit-optimize GP/EI 后端封装，作为 Optuna 不可用时的回退。"""
    name = "skopt_gp_ei"

    def __init__(self, random_state: int) -> None:
        """初始化 skopt 依赖，并延迟创建 Optimizer 实例。"""
        from skopt import Optimizer
        from skopt.space import Real

        self._optimizer_cls = Optimizer
        self._real_cls = Real
        self.random_state = random_state
        self.optimizer = None
        self._trials: List[Dict[str, Any]] = []

    def ask(self, variables: List[DesignVariable]):
        """根据变量空间请求下一组 skopt 采样值。"""
        if self.optimizer is None:
            dimensions = [
                self._real_cls(variable.lower, variable.upper, name=variable.name)
                for variable in variables
            ]

#--------------------------------------------------------------------------------------------------------------------------------------
            self.optimizer = self._optimizer_cls(
                dimensions=dimensions,
                base_estimator="GP",
                acq_func="EI",
                random_state=self.random_state,
            )
        point = self.optimizer.ask()
        values = {
            variable.name: float(point[index])
            for index, variable in enumerate(variables)
        }
        return point, values

    def tell(self, token: Any, objective: float) -> None:
        """把当前 evaluation 的 objective 回传给 skopt。"""
        self.optimizer.tell(token, float(objective))
        self._trials.append(
            {
                "number": len(self._trials),
                "state": "COMPLETE",
                "value": float(objective),
                "params": list(token),
            }
        )

    def trials(self) -> List[Dict[str, Any]]:
        """导出 skopt 已评估点与目标值。"""
        return list(self._trials)

    def param_importances(self) -> Dict[str, float]:
        """skopt 后端暂无内置重要性，返回空字典。"""
        return {}


@dataclass
class OptimizationConfig:
    """一次优化运行的配置参数。"""
    parameter_json: Path
    output_root: Path = PROJECT_ROOT / "optimization_runs"
    run_name: Optional[str] = None
    instance_json: Optional[Path] = None
    port_summary: Optional[Path] = None
    layer_name: str = "layer0"
    max_evaluations: int = MAX_ALLOWED_EVALUATIONS
    target_frequency_ghz: float = 2.4
    target_s11_db: float = -10.0
    convergence_threshold: float = 0.02
    no_improvement_patience: int = 8
    max_invalid_ratio: float = 0.50
    max_consecutive_cst_failures: int = 5
    run_solver: bool = True
    simplify_tolerance_px: float = 1.0
    geometry_frame: str = "svg"
    curve_parameterization_mode: str = DEFAULT_CURVE_PARAMETERIZATION_MODE
    random_state: int = 42
    optimizer_backend: str = "auto"
    port_connection_step_px: float = 0.2
    port_connection_tolerance_px: float = 0.15
    port_connection_max_shift_px: float = 120.0
    port_connection_final_free_normal_inward_px: float = 2.0


@dataclass
class EvaluationRecord:
    """单轮 BO evaluation 的完整记录。"""
    evaluation: int
    variables: Dict[str, float]
    status: str
    objective: Optional[float]
    objective_breakdown: Optional[Dict[str, Any]]
    validation: Dict[str, Any]
    geometry_metrics: Dict[str, Any]
    design_json: str
    cst_project: Optional[str] = None
    s11_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为可写入 JSON 的字典。"""
        return asdict(self)


@dataclass
class RunState:
    """运行目录、日志器、history 与 best record 的状态容器。"""
    run_dir: Path
    valid_designs_dir: Path
    invalid_designs_dir: Path
    best_design_dir: Path
    plots_dir: Path
    logs_dir: Path
    history_path: Path
    logger: logging.Logger
    history: List[EvaluationRecord] = field(default_factory=list)
    best_record: Optional[EvaluationRecord] = None


def load_json(path: Path) -> Dict[str, Any]:
    """读取 UTF-8 JSON 文件。"""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 写入 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def setup_run(config: OptimizationConfig) -> RunState:
    """创建 optimization_runs/<run_name>/ 目录结构和日志系统。"""
    run_name = config.run_name or "opt"
    run_dir = make_unique_run_dir(config.output_root, run_name)
    valid_designs_dir = run_dir / "valid_designs"
    invalid_designs_dir = run_dir / "invalid_designs"
    best_design_dir = run_dir / "best_designs"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"
    for directory in (valid_designs_dir, invalid_designs_dir, best_design_dir, plots_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"optimization_pipeline.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(logs_dir / "optimization.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return RunState(
        run_dir=run_dir,
        valid_designs_dir=valid_designs_dir,
        invalid_designs_dir=invalid_designs_dir,
        best_design_dir=best_design_dir,
        plots_dir=plots_dir,
        logs_dir=logs_dir,
        history_path=run_dir / "optimization_history.json",
        logger=logger,
    )


def make_unique_run_dir(output_root: Path, run_name: str) -> Path:
    """Create a unique per-run output directory.

    Input: output root and requested logical run name.
    Output: a newly created directory path that never reuses an existing run.
    Algorithm purpose: prevent BO artifacts from overwriting previous records
    when editor config or CLI uses the same run name repeatedly.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_run_name(run_name)
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{safe_name}_{timestamp}"
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = output_root / f"{safe_name}_{timestamp}_{counter:02d}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def sanitize_run_name(run_name: str) -> str:
    """Sanitize a logical run name for use as a folder name.

    Input: requested run name.
    Output: filesystem-safe non-empty run name.
    Algorithm purpose: keep automatic unique folders portable and readable.
    """

    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(run_name).strip())
    return safe.strip("_") or "opt"


def close_logger(logger: logging.Logger) -> None:
    """关闭并移除当前 run logger 的 handler。"""
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


# =============================================================================
# 主优化流程：ask -> mutate -> validate -> CST -> S11 -> objective -> tell
# =============================================================================


class OptimizationPipeline:
    """curve_parameterization.json 后置优化层。

    不修改参数化 backend，不修改 CST reconstruction，只在二者之间新增
    primitive-safe mutation、validation、CST build/simulation 和目标函数评估。
    """

    def __init__(self, config: OptimizationConfig) -> None:
        """加载输入 JSON、提取变量、创建优化器和参考拓扑。"""
        self.config = config
        self.state = setup_run(config)
        self.payload = load_json(config.parameter_json)
        self.port_summary = load_json(config.port_summary) if config.port_summary is not None else None
        self.reference_signature = make_topology_signature(self.payload)

        self.validation_config = GeometryValidationConfig()

        self.variables, self.inventory = extract_design_variables(
            self.payload,
            port_summary=self.port_summary,
            curve_parameterization_mode=self.config.curve_parameterization_mode,
        )
        self.objective_weights = ObjectiveWeights()
        self.objective_profile = self._configure_objective_profile()
        self.optimizer = self._create_optimizer_backend()

    def _configure_objective_profile(self):
        """Load the objective profile from the new-format instance JSON."""

        profile = create_objective_profile_from_instance_path(
            self.config.instance_json,
            fallback_target_frequency_ghz=self.config.target_frequency_ghz,
            fallback_target_s11_db=self.config.target_s11_db,
        )
        if profile.targets.resonance_ghz is not None:
            self.config.target_frequency_ghz = float(profile.targets.resonance_ghz)
        self.config.target_s11_db = float(profile.targets.s11_db)
        set_current_objective_profile(profile)
        return profile

    def run(self) -> Path:
        """【关键函数】执行完整 BO 循环，并返回 run_dir。"""
        logger = self.state.logger
        try:
            self._write_run_metadata()
            reference_report = validate_geometry(
                self.payload,
                reference_signature=self.reference_signature,
                config=self.validation_config,
                port_summary=self.port_summary,
            )
            if not reference_report.valid:
                logger.warning("初始参数化 JSON 已存在几何风险: %s", reference_report.reasons)

            logger.info("开始优化: %s", self.config.parameter_json.resolve())
            logger.info("输出目录: %s", self.state.run_dir.resolve())
            logger.info("backend: %s", self.payload.get("backend", "unknown"))
            logger.info("设计变量: %s", [variable.to_dict() for variable in self.variables])
            logger.info("primitive inventory: %s", self.inventory.to_dict())

            self._evaluate_initial_design()

            no_improvement_count = 0
            consecutive_cst_failures = 0
            best_objective = None

            for evaluation in range(1, self.config.max_evaluations + 1):
                token, values = self.optimizer.ask(self.variables)
                record = self.evaluate(evaluation, values)
                geometry_rejected = self._is_geometry_rejection(record)
                objective_for_optimizer = record.objective
                if objective_for_optimizer is None:
                    objective_for_optimizer = self.objective_weights.cst_failure
                self.optimizer.tell(token, float(objective_for_optimizer))

                if record.status == "cst_failed":
                    consecutive_cst_failures += 1
                else:
                    consecutive_cst_failures = 0

                if geometry_rejected:
                    logger.info(
                        "evaluation=%03d geometry rejected; rollback applied, patience unchanged",
                        record.evaluation,
                    )
                elif record.objective is not None and (best_objective is None or record.objective < best_objective):
                    best_objective = record.objective
                    no_improvement_count = 0
                    self._save_best(record)
                else:
                    no_improvement_count += 1

                self._save_history()
                self._save_plots()

                if self._should_stop(record, no_improvement_count, consecutive_cst_failures):
                    break

            logger.info("优化结束，history: %s", self.state.history_path.resolve())
            self._save_parameter_sensitivity()
            self._save_optimization_animation()
            return self.state.run_dir
        finally:
            close_logger(logger)

    def _evaluate_initial_design(self) -> Optional[EvaluationRecord]:
        """Evaluate and store the unperturbed initial design as S11 baseline.

        The baseline is not a BO trial: it is not appended to optimization
        history and is not sent to the optimizer. It exists only as a fixed
        reference curve for plots such as plots/best_s11_curve.png.
        """

        initial_dir = self.state.valid_designs_dir / "initial_design"
        if (initial_dir / "initial_design_record.json").exists():
            return None

        default_values = {variable.name: float(variable.default) for variable in self.variables}
        self.state.logger.info("evaluating initial design S11 baseline")
        record = self.evaluate(
            evaluation=0,
            variables=default_values,
            output_dir_name="initial_design",
            append_history=False,
        )
        write_json(initial_dir / "initial_design_record.json", record.to_dict())
        if record.s11_metrics:
            write_json(initial_dir / "initial_s11_metrics.json", record.s11_metrics)
        return record

    def evaluate(
        self,
        evaluation: int,
        variables: Dict[str, float],
        output_dir_name: Optional[str] = None,
        append_history: bool = True,
    ) -> EvaluationRecord:
        """【关键函数】执行单轮 evaluation。

        单轮流程：mutate -> validation -> CST build/simulation -> S11 parse -> objective。
        任何失败都会转成 EvaluationRecord，不让整个优化程序崩溃。
        """
        logger = self.state.logger
        start = time.perf_counter()
        logger.info("evaluation=%03d variables=%s", evaluation, variables)

        record_name = output_dir_name or f"eval_{evaluation:03d}"
        eval_dir = self.state.valid_designs_dir / record_name
        eval_dir.mkdir(parents=True, exist_ok=True)
        mutated = mutate_geometry(
            self.payload,
            variables,
            self.inventory,
            output_dir=eval_dir,
            iteration=evaluation,
            port_summary=self.port_summary,
            curve_parameterization_mode=self.config.curve_parameterization_mode,
        )
        validation = validate_geometry(
            mutated,
            self.reference_signature,
            self.validation_config,
            port_summary=self.port_summary,
        )
        geometry_metrics = geometry_complexity_metrics(mutated, validation)
        deformation_meta = (mutated.get("optimization_metadata") or {}).get("feature_constrained_deformation") or {}
        manufacturability = deformation_meta.get("manufacturability_report") or {}
        if manufacturability:
            geometry_metrics["manufacturability"] = manufacturability
            geometry_metrics["spike_count"] = manufacturability.get("spike_count", 0)
            geometry_metrics["manufacturability_penalty"] = manufacturability.get("penalty", 0.0)
            if not bool(manufacturability.get("valid", True)):
                validation.valid = False
                validation.reasons.extend(
                    [f"manufacturability: {error}" for error in manufacturability.get("errors", [])]
                )
        primitive_meta = (mutated.get("optimization_metadata") or {}).get("primitive_aware_mutation") or {}
        if primitive_meta:
            geometry_metrics["primitive_shape_quality_score"] = primitive_meta.get("shape_quality_score", 0.0)
            geometry_metrics["primitive_junction_valid"] = (
                primitive_meta.get("junction_validation", {}).get("valid", True)
            )
            geometry_metrics["primitive_aware_valid"] = primitive_meta.get("valid", True)
            if not bool(primitive_meta.get("valid", True)):
                validation.valid = False
                validation.reasons.extend(
                    [f"primitive-aware constraint: {error}" for error in primitive_meta.get("errors", [])]
                )

        if not validation.valid:
            design_path = self.state.invalid_designs_dir / f"{record_name}.json"
            report_path = self.state.invalid_designs_dir / f"{record_name}_validation.json"
            write_json(design_path, mutated)
            write_json(report_path, validation.to_dict())
            objective = evaluate_objective(
                None,
                geometry_metrics,
                validation,
                self.config.target_frequency_ghz,
                self.config.target_s11_db,
                self.objective_weights,
            )
            write_json(
                self.state.invalid_designs_dir / f"{record_name}_objective_breakdown.json",
                objective.to_dict(),
            )
            record = EvaluationRecord(
                evaluation=evaluation,
                variables=variables,
                status="invalid_geometry",
                objective=objective.total,
                objective_breakdown=objective.to_dict(),
                validation=validation.to_dict(),
                geometry_metrics=geometry_metrics,
                design_json=str(design_path),
                error_message="; ".join(validation.reasons),
                elapsed_seconds=time.perf_counter() - start,
            )
            if append_history:
                self.state.history.append(record)
            logger.warning("rejected invalid geometry eval=%03d reasons=%s", evaluation, validation.reasons)
            return record

        design_path = eval_dir / "curve_parameterization.json"
        cst_handoff_payload = prepare_cst_handoff_payload(mutated)
        if cst_handoff_payload is not mutated:
            write_json(eval_dir / "mutation_raw_curve_parameterization.json", mutated)
            handoff_report = validate_geometry(
                cst_handoff_payload,
                None,
                self.validation_config,
                port_summary=self.port_summary,
            )
            write_json(eval_dir / "cst_handoff_validation.json", handoff_report.to_dict())

        repaired_payload, cst_geometry_report = validate_and_repair_cst_geometry(
            cst_handoff_payload,
            output_dir=eval_dir,
            logger=logger,
        )
        port_summary_for_eval = self.config.port_summary
        connected_port_summary_path = None
        if self.port_summary is not None:
            connected_port_summary, port_connection_report = ensure_port_summary_connected_to_geometry(
                repaired_payload,
                self.port_summary,
                step_px=self.config.port_connection_step_px,
                tolerance_px=self.config.port_connection_tolerance_px,
                max_shift_px=self.config.port_connection_max_shift_px,
                final_free_normal_inward_px=self.config.port_connection_final_free_normal_inward_px,
            )
            write_json(eval_dir / "port_connection_report.json", port_connection_report)
            metadata = repaired_payload.setdefault("optimization_metadata", {})
            metadata["port_connection"] = port_connection_report
            if connected_port_summary is not None:
                connected_port_summary_path = eval_dir / "port_summary_connected.json"
                write_json(connected_port_summary_path, connected_port_summary)
                port_summary_for_eval = connected_port_summary_path
            if not bool(port_connection_report.get("connected_after", False)):
                cst_geometry_report.valid = False
                cst_geometry_report.errors.append(
                    "port is not connected to parameterized feedline after inward normal stepping"
                )
        write_json(design_path, repaired_payload)
        geometry_metrics["cst_robustness_score"] = cst_geometry_report.robustness_score

        if not cst_geometry_report.valid:
            objective_value = self.objective_weights.invalid_geometry
            objective_breakdown = {
                "total": objective_value,
                "failure_penalty": objective_value,
                "reason": "cst_geometry_validation_failed",
                "normalized_errors": {},
            }
            write_json(eval_dir / "objective_breakdown.json", objective_breakdown)
            record = EvaluationRecord(
                evaluation=evaluation,
                variables=variables,
                status="invalid_geometry",
                objective=objective_value,
                objective_breakdown=objective_breakdown,
                validation={
                    "topology_validation": validation.to_dict(),
                    "cst_geometry_validation": cst_geometry_report.to_dict(),
                },
                geometry_metrics=geometry_metrics,
                design_json=str(design_path),
                error_message="; ".join(cst_geometry_report.errors),
                elapsed_seconds=time.perf_counter() - start,
            )
            if append_history:
                self.state.history.append(record)
            logger.warning(
                "rejected before CST eval=%03d geometry errors=%s",
                evaluation,
                cst_geometry_report.errors,
            )
            return record

        cst_project = None
        s11_metrics = None
        cst_failed = False
        error_message = None
        try:
            cst_project = self._build_and_simulate(design_path, eval_dir, port_summary_path=port_summary_for_eval)
            if self.config.run_solver:
                s11_path = find_latest_s11_file(eval_dir)
                s11_metrics = parse_s11_file(s11_path, target_frequency_ghz=self.config.target_frequency_ghz)
            else:
                cst_failed = False
                error_message = "solver disabled; no S11 available for objective evaluation"
        except Exception as exc:
            cst_failed = True
            error_message = str(exc)
            logger.exception("CST failure eval=%03d: %s", evaluation, exc)

        objective = evaluate_objective(
            s11_metrics,
            geometry_metrics,
            validation,
            self.config.target_frequency_ghz,
            self.config.target_s11_db,
            self.objective_weights,
            cst_failed=cst_failed,
        )
        write_json(eval_dir / "objective_breakdown.json", objective.to_dict())

        status = "completed"
        if not self.config.run_solver and not cst_failed:
            status = "build_completed"
        elif cst_failed:
            status = "cst_failed"
        record = EvaluationRecord(
            evaluation=evaluation,
            variables=variables,
            status=status,
            objective=objective.total,
            objective_breakdown=objective.to_dict(),
            validation=validation.to_dict(),
            geometry_metrics=geometry_metrics,
            design_json=str(design_path),
            cst_project=str(cst_project) if cst_project else None,
            s11_metrics=s11_metrics.to_dict() if s11_metrics else None,
            error_message=error_message,
            elapsed_seconds=time.perf_counter() - start,
        )
        if append_history:
            self.state.history.append(record)
        logger.info(
            "evaluation=%03d status=%s objective=%.6f elapsed=%.2fs",
            evaluation,
            status,
            objective.total,
            record.elapsed_seconds,
        )
        return record

    def _build_and_simulate(
        self,
        design_json: Path,
        eval_dir: Path,
        port_summary_path: Optional[Path] = None,
    ) -> Path:
        """调用已有 ParameterizedJsonCSTBuilder 完成 CST 建模和可选仿真。"""
        from bayesian_optimization.simulation.parameterized_json_to_cst import (
            CSTParametricConfig,
            ParameterizedJsonCSTBuilder,
            load_instance_config,
        )

        if self.config.instance_json is not None:
            cst_config = load_instance_config(self.config.instance_json, self.config.layer_name)
        else:
            cst_config = CSTParametricConfig(project_folder=eval_dir / "cst")

        cst_config.project_folder = eval_dir / "cst"
        unique_suffix = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cst_config.project_name = f"optimization_eval_{design_json.parent.name}_{unique_suffix}"
        cst_config.run_solver = bool(self.config.run_solver)
        cst_config.port_summary_path = port_summary_path if port_summary_path is not None else self.config.port_summary
        cst_config.simplify_tolerance_px = self.config.simplify_tolerance_px
        cst_config.geometry_frame = self.config.geometry_frame
        cst_config.close_project = True
        self._validate_cst_frequency_range(cst_config.f0, cst_config.f1)
        self.state.logger.info(
            "CST S11 simulation frequency range: %.6g-%.6g %s",
            cst_config.f0,
            cst_config.f1,
            cst_config.frequency_unit,
        )

        if cst_config.run_solver and cst_config.port_summary_path is None:
            raise ValueError("run_solver=True requires --port-summary for CST port reconstruction")

        cst_config.project_folder.mkdir(parents=True, exist_ok=True)
        builder = ParameterizedJsonCSTBuilder(design_json, cst_config)
        return builder.build()

    @staticmethod
    def _validate_cst_frequency_range(f0: float, f1: float) -> None:
        """Validate the CST S11 simulation frequency range before building."""

        if not math.isfinite(float(f0)) or not math.isfinite(float(f1)):
            raise ValueError(f"CST frequency range must be finite, got f0={f0}, f1={f1}")
        if float(f1) <= float(f0):
            raise ValueError(f"CST frequency range requires f1 > f0, got f0={f0}, f1={f1}")

    def _cst_frequency_range_metadata(self) -> Dict[str, Any]:
        """Return the CST S11 frequency range resolved from instance JSON."""

        if self.config.instance_json is None:
            return {
                "source": "default_CSTParametricConfig",
                "f0": 6.0,
                "f1": 14.0,
                "unit": "GHz",
            }

        from bayesian_optimization.simulation.parameterized_json_to_cst import load_instance_config

        cst_config = load_instance_config(self.config.instance_json, self.config.layer_name)
        self._validate_cst_frequency_range(cst_config.f0, cst_config.f1)
        return {
            "source": str(self.config.instance_json),
            "f0": float(cst_config.f0),
            "f1": float(cst_config.f1),
            "unit": cst_config.frequency_unit,
        }

    def _create_optimizer_backend(self):
        """根据配置创建 Optuna/skopt/auto 优化后端。"""
        backend = self.config.optimizer_backend.lower()
        if backend not in {"auto", "optuna", "skopt"}:
            raise ValueError("optimizer_backend must be one of: auto, optuna, skopt")

        if backend in {"auto", "optuna"}:
            try:
                return OptunaBackend(
                    random_state=self.config.random_state,
                    max_evaluations=self.config.max_evaluations,
                )
            except Exception as exc:
                if backend == "optuna":
                    raise
                self.state.logger.warning("Optuna unavailable, fallback to skopt GP/EI: %s", exc)

        try:
            return SkoptBackend(random_state=self.config.random_state)
        except Exception as exc:
            raise ImportError(
                "Optuna 和 scikit-optimize 都不可用。请检查 local_packages_py38 或运行 pip install -r requirements.txt。"
            ) from exc

    def _should_stop(
        self,
        record: EvaluationRecord,
        no_improvement_count: int,
        consecutive_cst_failures: int,
    ) -> bool:
        """检查 convergence、patience、invalid ratio、CST failure 等停止条件。"""
        logger = self.state.logger
        if record.objective is not None and record.objective <= self.config.convergence_threshold:
            logger.info(
                "停止条件触发: objective %.6f <= convergence_threshold %.6f",
                record.objective,
                self.config.convergence_threshold,
            )
            return True
        if no_improvement_count >= self.config.no_improvement_patience:
            logger.info("停止条件触发: no-improvement patience=%d", no_improvement_count)
            return True
        if consecutive_cst_failures >= self.config.max_consecutive_cst_failures:
            logger.warning("停止条件触发: consecutive CST failures=%d", consecutive_cst_failures)
            return True
        return False

    def _is_geometry_rejection(self, record: EvaluationRecord) -> bool:
        """判断当前 evaluation 是否为几何非法样本。

        几何非法样本采用 rollback 语义：记录并反馈惩罚给优化器，但不更新 best，
        也不消耗 no-improvement patience，以便 BO 尽量探索完整 30 轮空间。
        """
        return record.status == "invalid_geometry"

    def _save_best(self, record: EvaluationRecord) -> None:
        """保存当前最优设计 JSON、record 和 S11 文件。"""
        self.state.best_record = record
        snapshot_dir = self.state.best_design_dir / f"eval_{record.evaluation:03d}"
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        source = Path(record.design_json)
        target = snapshot_dir / "curve_parameterization.json"
        shutil.copy2(source, target)
        write_json(snapshot_dir / "best_record.json", record.to_dict())
        self._copy_best_visuals(source.parent, snapshot_dir)
        write_json(
            self.state.best_design_dir / "best_index.json",
            {
                "latest_best_evaluation": record.evaluation,
                "latest_best_objective": record.objective,
                "latest_best_snapshot": str(snapshot_dir),
                "policy": "each improved best design is saved in a new eval_xxx snapshot folder",
            },
        )

        if record.s11_metrics:
            s11_path = Path(record.s11_metrics["s11_path"])
            if s11_path.exists():
                shutil.copy2(s11_path, snapshot_dir / s11_path.name)

    def _copy_best_visuals(self, eval_dir: Path, snapshot_dir: Path) -> None:
        """Copy high-signal visual comparison artifacts into a best snapshot.

        Input: source evaluation directory and destination best snapshot directory.
        Output: copied PNG/JSON visual artifacts when present.
        Algorithm purpose: make the final best-design folder directly show the
        original-vs-optimized difference and selected optimization points.
        """

        debug_dir = eval_dir / "primitive_debug"
        if not debug_dir.exists():
            return
        visual_dir = snapshot_dir / "visual_comparison"
        visual_dir.mkdir(parents=True, exist_ok=True)
        artifact_names = [
            "optimization_before_after_comparison.png",
            "selected_points_overlay.png",
            "variable_effect_overlay.png",
            "curvature_change_map.png",
            "selected_optimization_points.json",
        ]
        for name in artifact_names:
            source = debug_dir / name
            if source.exists():
                shutil.copy2(source, visual_dir / name)

    def _save_history(self) -> None:
        """写入 optimization_history.json。"""
        write_json(
            self.state.history_path,
            {
                "records": [record.to_dict() for record in self.state.history],
                "best_record": self.state.best_record.to_dict() if self.state.best_record else None,
            },
        )
        self._save_optimizer_trials()

    def _write_run_metadata(self) -> None:
        """写入 run_metadata.json，记录输入、变量、目标和优化器配置。"""
        metadata = {
            "config": {
                "parameter_json": str(self.config.parameter_json),
                "instance_json": str(self.config.instance_json) if self.config.instance_json else None,
                "port_summary": str(self.config.port_summary) if self.config.port_summary else None,
                "max_evaluations": self.config.max_evaluations,
                "target_frequency_ghz": self.config.target_frequency_ghz,
                "target_s11_db": self.config.target_s11_db,
                "cst_s11_frequency_range": self._cst_frequency_range_metadata(),
                "run_solver": self.config.run_solver,
                "geometry_frame": self.config.geometry_frame,
                "simplify_tolerance_px": self.config.simplify_tolerance_px,
                "curve_parameterization_mode": self.config.curve_parameterization_mode,
                "optimizer_backend": self.config.optimizer_backend,
                "storage_policy": {
                    "run_directory": "always create a timestamped unique run folder",
                    "best_designs": "save each improved best in best_designs/eval_xxx without overwriting old snapshots",
                    "latest_best_index": "best_designs/best_index.json points to the latest best snapshot",
                },
                "invalid_geometry_policy": {
                    "action": "rollback_and_continue",
                    "optimizer_feedback": "large_penalty",
                    "skip_cst": True,
                    "counts_toward_no_improvement_patience": False,
                    "counts_toward_early_stop_invalid_ratio": False,
                    "max_invalid_ratio_retained_for_compatibility": self.config.max_invalid_ratio,
                },
                "substrate_edge_clearance_policy": {
                    "enabled": self.validation_config.enable_substrate_edge_clearance,
                    "clearance_ratio_of_min_canvas_dimension": self.validation_config.substrate_edge_clearance_ratio,
                    "minimum_clearance_px": self.validation_config.min_substrate_edge_clearance_px,
                    "allow_feedline_port_corridor": self.validation_config.allow_feedline_port_corridor_clearance,
                    "action": "reject_non_port_geometry_before_cst",
                },
                "port_connection_policy": {
                    "enabled": self.config.port_summary is not None,
                    "action": "shift_port_inward_until_parameterized_curve_contact_before_cst",
                    "step_px": self.config.port_connection_step_px,
                    "tolerance_px": self.config.port_connection_tolerance_px,
                    "max_shift_px": self.config.port_connection_max_shift_px,
                    "final_free_normal_inward_px": self.config.port_connection_final_free_normal_inward_px,
                    "failure_action": "reject_before_cst",
                    "per_evaluation_outputs": [
                        "port_connection_report.json",
                        "port_summary_connected.json",
                    ],
                },
            },
            "design_variables": [variable.to_dict() for variable in self.variables],
            "primitive_inventory": self.inventory.to_dict(),
            "reference_topology": self.reference_signature.to_dict(),
            "objective_weights": self.objective_weights.to_dict(),
            "objective_profile": {
                "name": self.objective_profile.name,
                "targets": self.objective_profile.targets.to_dict(),
                "source": self.objective_profile.targets.source,
            },
            "optimizer": {
                "backend": self.optimizer.name,
                "requested_backend": self.config.optimizer_backend,
                "direction": "minimize",
                "note": "auto tries Optuna TPESampler first, then falls back to scikit-optimize GP/EI.",
            },
        }
        write_json(self.state.run_dir / "run_metadata.json", metadata)
        self._save_optimizer_trials()

    def _save_plots(self) -> None:
        """【关键函数】保存 objective、状态、谐振频率、S11 等优化曲线。"""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.state.logger.warning("plot skipped: %s", exc)
            return

        completed = [record for record in self.state.history if record.objective is not None]
        if not completed:
            return

        evaluations = [record.evaluation for record in completed]
        objectives = [float(record.objective) for record in completed]
        primary_variable_name = self.variables[0].name if self.variables else "design_variable"
        primary_variable_values = [
            float(record.variables.get(primary_variable_name, 0.0))
            for record in completed
        ]
        colors = ["#dc2626" if record.status != "completed" else "#2563eb" for record in completed]

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(evaluations, objectives, color="#111827", linewidth=1.2, alpha=0.6)
        ax.scatter(evaluations, objectives, c=colors, s=48)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Weighted Normalized Error")
        ax.set_title("Objective Function History")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "objective_history.png", dpi=180)
        plt.close(fig)

        self._save_objective_error_plot(plt, completed)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(primary_variable_values, objectives, c=colors, s=52)
        ax.set_xlabel(primary_variable_name)
        ax.set_ylabel("Weighted Normalized Error")
        ax.set_title("Design Variable vs Objective Function")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "variable_objective.png", dpi=180)
        plt.close(fig)

        status_counts: Dict[str, int] = {}
        for record in self.state.history:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar(list(status_counts.keys()), list(status_counts.values()), color="#0f766e")
        ax.set_xlabel("Evaluation Status")
        ax.set_ylabel("Count")
        ax.set_title("Evaluation Status Distribution")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "evaluation_status.png", dpi=180)
        plt.close(fig)

        self._save_optimizer_importance_plot(plt)

        resonance_records = [
            record
            for record in completed
            if record.s11_metrics and record.s11_metrics.get("resonant_frequency_ghz") is not None
        ]
        if resonance_records:
            resonance_evaluations = [record.evaluation for record in resonance_records]
            resonance_values = [
                float(record.s11_metrics["resonant_frequency_ghz"])
                for record in resonance_records
            ]
            min_s11_values = [
                float(record.s11_metrics.get("minimum_s11_db", 0.0))
                for record in resonance_records
            ]

            fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
            axes[0].plot(resonance_evaluations, resonance_values, marker="o", color="#059669", linewidth=1.6)
            axes[0].axhline(self.config.target_frequency_ghz, color="#dc2626", linestyle="--", linewidth=1.2)
            axes[0].set_ylabel("Resonance (GHz)")
            axes[0].set_title("Resonance Frequency History")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(resonance_evaluations, min_s11_values, marker="o", color="#7c3aed", linewidth=1.6)
            axes[1].set_xlabel("Evaluation")
            axes[1].set_ylabel("Minimum S11 (dB)")
            axes[1].ticklabel_format(axis="y", useOffset=False)
            axes[1].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(self.state.plots_dir / "resonance_history.png", dpi=180)
            plt.close(fig)

        s11_records = [
            record
            for record in completed
            if record.s11_metrics and Path(record.s11_metrics["s11_path"]).exists()
        ]
        if not s11_records:
            return
        best = min(s11_records, key=lambda item: float(item.objective or 1e9))
        from bayesian_optimization.optimization.s11_parser import read_s11_rows

        best_s11_path = Path(best.s11_metrics["s11_path"])
        rows = read_s11_rows(best_s11_path)
        if not rows:
            return
        initial_s11_path = self._initial_s11_path()
        initial_rows = read_s11_rows(initial_s11_path) if initial_s11_path is not None else []

        fig, ax = plt.subplots(figsize=(8, 4.8))
        if initial_rows:
            ax.plot(
                [row[0] for row in initial_rows],
                [row[1] for row in initial_rows],
                color="#6b7280",
                linestyle="--",
                linewidth=1.3,
                label="Initial design",
            )
        ax.plot(
            [row[0] for row in rows],
            [row[1] for row in rows],
            color="#7c3aed",
            linewidth=1.6,
            label=f"Best eval {best.evaluation:03d}",
        )
        ax.axvline(self.config.target_frequency_ghz, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("S11 (dB)")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_title("Initial vs Best S11 Curve")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "best_s11_curve.png", dpi=180)
        plt.close(fig)
        write_json(
            self.state.plots_dir / "s11_curve_comparison.json",
            {
                "initial_s11_path": str(initial_s11_path) if initial_s11_path else None,
                "best_s11_path": str(best_s11_path),
                "best_evaluation": best.evaluation,
                "best_objective": best.objective,
                "target_frequency_ghz": self.config.target_frequency_ghz,
            },
        )

    def _initial_s11_path(self) -> Optional[Path]:
        """Return the stored initial-design S11 file when it exists."""

        initial_dir = self.state.valid_designs_dir / "initial_design"
        if not initial_dir.exists():
            return None
        try:
            return find_latest_s11_file(initial_dir)
        except FileNotFoundError:
            return None

    def _save_optimizer_importance_plot(self, plt: Any) -> None:
        """保存优化器自身提供的参数重要性图。"""
        try:
            importances = self.optimizer.param_importances()
        except Exception as exc:
            self.state.logger.info("parameter importance plot skipped: %s", exc)
            return
        if not importances:
            return

        names = list(importances.keys())
        values = [float(importances[name]) for name in names]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(names, values, color="#2563eb")
        ax.set_xlabel("Importance")
        ax.set_title("Optimizer Parameter Importance")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "optimizer_param_importance.png", dpi=180)
        plt.close(fig)

    def _save_objective_error_plot(self, plt: Any, records: List[EvaluationRecord]) -> None:
        """Save normalized objective-error histories for each BO evaluation."""

        rows: List[Dict[str, Any]] = []
        for record in records:
            breakdown = record.objective_breakdown or {}
            normalized = breakdown.get("normalized_errors") or {}
            if not normalized:
                continue
            rows.append(
                {
                    "evaluation": record.evaluation,
                    "normalized_errors": dict(normalized),
                    "weighted_terms": {
                        "resonance": float(breakdown.get("primary_frequency_error", 0.0)),
                        "resonance_count": float(breakdown.get("resonance_count_penalty", 0.0)),
                        "bandwidth": float(breakdown.get("bandwidth_reward", 0.0)),
                        "bandwidth_edges": float(breakdown.get("bandwidth_reward", 0.0)),
                        "s11": float(breakdown.get("target_s11_penalty", 0.0)),
                        "gain": float(breakdown.get("gain_penalty", 0.0)),
                    },
                    "total": record.objective,
                }
            )
        write_json(self.state.plots_dir / "objective_error_history.json", rows)
        if not rows:
            return

        evaluations = [int(row["evaluation"]) for row in rows]
        error_keys = [
            ("resonance", "E_res"),
            ("resonance_count", "E_mode_count"),
            ("bandwidth_edges", "E_edge"),
            ("bandwidth_width_shortfall", "E_bw_shortfall"),
            ("s11", "E_s11"),
            ("gain", "E_gain"),
        ]
        weighted_keys = [
            ("resonance", "w_res*E_res"),
            ("resonance_count", "w_count*E_mode_count"),
            ("bandwidth", "w_bw*E_bw"),
            ("s11", "w_s11*E_s11"),
            ("gain", "w_gain*E_gain"),
        ]

        fig, axes = plt.subplots(2, 1, figsize=(9, 7.2), sharex=True)
        for key, label in error_keys:
            values = []
            for row in rows:
                value = row["normalized_errors"].get(key)
                if value is None and key == "bandwidth_edges":
                    value = row["normalized_errors"].get("bandwidth")
                values.append(float(value) if value is not None else math.nan)
            axes[0].plot(evaluations, values, marker="o", linewidth=1.4, label=label)
        axes[0].set_ylabel("Normalized Error")
        axes[0].set_title("Objective Error Terms")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)

        for key, label in weighted_keys:
            values = [float(row["weighted_terms"].get(key, 0.0)) for row in rows]
            axes[1].plot(evaluations, values, marker="o", linewidth=1.4, label=label)
        axes[1].set_xlabel("Evaluation")
        axes[1].set_ylabel("Weighted Term")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "objective_error_history.png", dpi=180)
        plt.close(fig)

    def _save_optimizer_trials(self) -> None:
        """保存优化器 trial 历史。"""
        write_json(self.state.run_dir / "optimizer_trials.json", self.optimizer.trials())

    def _save_parameter_sensitivity(self) -> None:
        """优化结束后做轻量变量敏感度统计，不改变 BO 目标函数。"""
        records = [record for record in self.state.history if record.objective is not None]
        if len(records) < 2:
            self.state.logger.info("parameter sensitivity skipped: fewer than 2 records")
            return

        variable_names = [variable.name for variable in self.variables]
        rows: List[Dict[str, Any]] = []
        for name in variable_names:
            objective_corr = pearson_correlation(
                [float(record.variables.get(name, 0.0)) for record in records],
                [float(record.objective or 0.0) for record in records],
            )
            resonance_pairs = [
                (
                    float(record.variables.get(name, 0.0)),
                    float(record.s11_metrics["resonant_frequency_ghz"]),
                )
                for record in records
                if record.s11_metrics and record.s11_metrics.get("resonant_frequency_ghz") is not None
            ]
            s11_pairs = [
                (
                    float(record.variables.get(name, 0.0)),
                    float(record.s11_metrics.get("minimum_s11_db", 0.0)),
                )
                for record in records
                if record.s11_metrics and record.s11_metrics.get("minimum_s11_db") is not None
            ]
            resonance_corr = pearson_correlation(
                [item[0] for item in resonance_pairs],
                [item[1] for item in resonance_pairs],
            )
            s11_corr = pearson_correlation(
                [item[0] for item in s11_pairs],
                [item[1] for item in s11_pairs],
            )
            sensitivity_score = sum(
                abs(value)
                for value in (objective_corr, resonance_corr, s11_corr)
                if value is not None
            )
            rows.append(
                {
                    "variable": name,
                    "objective_correlation": objective_corr,
                    "resonant_frequency_correlation": resonance_corr,
                    "minimum_s11_correlation": s11_corr,
                    "sensitivity_score": sensitivity_score,
                    "sample_count": len(records),
                    "resonance_sample_count": len(resonance_pairs),
                }
            )

        rows.sort(key=lambda item: float(item["sensitivity_score"]), reverse=True)
        write_json(self.state.run_dir / "most_sensitive_variables.json", rows)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.state.logger.info("parameter sensitivity heatmap skipped: %s", exc)
            return

        top_rows = rows[: min(30, len(rows))]
        if not top_rows:
            return
        metrics = [
            ("objective_correlation", "Objective"),
            ("resonant_frequency_correlation", "f_res"),
            ("minimum_s11_correlation", "min S11"),
        ]
        matrix = [
            [float(row[key]) if row.get(key) is not None else 0.0 for key, _ in metrics]
            for row in top_rows
        ]
        fig_height = max(4.8, 0.32 * len(top_rows) + 1.5)
        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_xticks(list(range(len(metrics))))
        ax.set_xticklabels([label for _, label in metrics])
        ax.set_yticks(list(range(len(top_rows))))
        ax.set_yticklabels([row["variable"] for row in top_rows], fontsize=7)
        ax.set_title("Parameter Sensitivity Heatmap")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "parameter_sensitivity_heatmap.png", dpi=180)
        plt.close(fig)

    def _save_optimization_animation(self) -> None:
        """把每轮 geometry_before_after.png 合成为优化过程 GIF。"""
        try:
            import imageio.v2 as imageio
        except Exception as exc:
            self.state.logger.info("optimization animation skipped: %s", exc)
            return

        frames = []
        for record in self.state.history:
            frame_path = (
                self.state.valid_designs_dir
                / f"eval_{record.evaluation:03d}"
                / "deformation_debug"
                / "geometry_before_after.png"
            )
            if frame_path.exists():
                frames.append(imageio.imread(frame_path))
        if not frames:
            return
        output_path = self.state.plots_dir / "optimization_animation.gif"
        imageio.mimsave(output_path, frames, duration=0.8)
        self.state.logger.info("optimization animation saved: %s", output_path)


def pearson_correlation(xs: List[float], ys: List[float]) -> Optional[float]:
    """计算轻量 Pearson 相关系数，用于变量敏感度诊断。"""
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    x_values = [item[0] for item in pairs]
    y_values = [item[1] for item in pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    if denominator <= 1e-15:
        return None
    return numerator / denominator


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Additive optimization layer between curve_parameterization.json and CST simulation."
    )
    parser.add_argument("--parameter-json", required=True, type=Path, help="Input curve_parameterization.json.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "optimization_runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--instance-json", type=Path, default=None, help="Prepared/instance JSON for CST settings.")
    parser.add_argument("--port-summary", type=Path, default=None, help="port_summary.json required when solver runs.")
    parser.add_argument("--layer", default="layer0")
    parser.add_argument("--max-evaluations", type=int, default=MAX_ALLOWED_EVALUATIONS)
    parser.add_argument("--target-frequency-ghz", type=float, default=2.4)
    parser.add_argument("--target-s11-db", type=float, default=-10.0)
    parser.add_argument("--convergence-threshold", type=float, default=0.02)
    parser.add_argument("--no-improvement-patience", type=int, default=8)
    parser.add_argument("--max-invalid-ratio", type=float, default=0.50)
    parser.add_argument("--max-consecutive-cst-failures", type=int, default=5)
    parser.add_argument("--build-only", action="store_true", help="Build CST only; objective receives CST failure penalty.")
    parser.add_argument("--simplify-tolerance-px", type=float, default=1.0)
    parser.add_argument("--geometry-frame", choices=["svg", "component"], default="svg")
    parser.add_argument(
        "--curve-parameterization-mode",
        choices=["linearized", "native"],
        default=DEFAULT_CURVE_PARAMETERIZATION_MODE,
        help="linearized treats arc/spline/curve primitives as straight line spans for BO; native keeps old curve-aware rules.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--optimizer-backend",
        choices=["auto", "optuna", "skopt"],
        default="auto",
        help="auto tries Optuna first and falls back to skopt GP/EI.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OptimizationConfig:
    """把 argparse 结果转换为 OptimizationConfig。"""
    return OptimizationConfig(
        parameter_json=args.parameter_json,
        output_root=args.output_root,
        run_name=args.run_name,
        instance_json=args.instance_json,
        port_summary=args.port_summary,
        layer_name=args.layer,
        max_evaluations=args.max_evaluations,
        target_frequency_ghz=args.target_frequency_ghz,
        target_s11_db=args.target_s11_db,
        convergence_threshold=args.convergence_threshold,
        no_improvement_patience=args.no_improvement_patience,
        max_invalid_ratio=args.max_invalid_ratio,
        max_consecutive_cst_failures=args.max_consecutive_cst_failures,
        run_solver=not args.build_only,
        simplify_tolerance_px=args.simplify_tolerance_px,
        geometry_frame=args.geometry_frame,
        curve_parameterization_mode=args.curve_parameterization_mode,
        random_state=args.random_state,
        optimizer_backend=args.optimizer_backend,
    )


def build_config_from_editor_config(editor_config: Dict[str, Any]) -> OptimizationConfig:
    """把 EDITOR_RUN_CONFIG 转换为 OptimizationConfig，方便 IDE 直接运行。"""
    base_run_dir = Path(editor_config["BASE_RUN_DIR"])
    parameter_json = base_run_dir / "02_parameterization" / "curve_parameterization.json"
    instance_json = Path(editor_config.get("INSTANCE_JSON_PATH") or (base_run_dir / "prepared_instance.json"))
    port_summary = base_run_dir / "patch_port_summary.json"
    if not port_summary.exists():
        port_summary = base_run_dir / "port_summary.json"

    return OptimizationConfig(
        parameter_json=parameter_json,
        output_root=Path(editor_config["OUTPUT_ROOT"]),
        run_name=str(editor_config["RUN_NAME"]) if editor_config.get("RUN_NAME") else None,
        instance_json=instance_json,
        port_summary=port_summary,
        layer_name=str(editor_config["LAYER_NAME"]),
        max_evaluations=int(editor_config["MAX_EVALUATIONS"]),
        target_frequency_ghz=float(editor_config["TARGET_FREQUENCY_GHZ"]),
        target_s11_db=float(editor_config["TARGET_S11_DB"]),
        convergence_threshold=float(editor_config["CONVERGENCE_THRESHOLD"]),
        no_improvement_patience=int(editor_config["NO_IMPROVEMENT_PATIENCE"]),
        max_invalid_ratio=float(editor_config["MAX_INVALID_RATIO"]),
        max_consecutive_cst_failures=int(editor_config["MAX_CONSECUTIVE_CST_FAILURES"]),
        run_solver=not bool(editor_config["BUILD_ONLY"]),
        simplify_tolerance_px=float(editor_config["SIMPLIFY_TOLERANCE_PX"]),
        geometry_frame=str(editor_config["GEOMETRY_FRAME"]),
        curve_parameterization_mode=str(editor_config.get("CURVE_PARAMETERIZATION_MODE", DEFAULT_CURVE_PARAMETERIZATION_MODE)),
        random_state=int(editor_config["RANDOM_STATE"]),
        optimizer_backend=str(editor_config["OPTIMIZER_BACKEND"]),
    )


def validate_config(config: OptimizationConfig) -> None:
    """检查输入路径、最大轮数和必要 CST 参数。"""
    if not config.parameter_json.exists():
        raise FileNotFoundError(f"parameter json not found: {config.parameter_json}")
    if config.instance_json is not None and not config.instance_json.exists():
        raise FileNotFoundError(f"instance json not found: {config.instance_json}")
    if config.port_summary is not None and not config.port_summary.exists():
        raise FileNotFoundError(f"port summary not found: {config.port_summary}")
    if config.max_evaluations <= 0:
        raise ValueError("max_evaluations must be > 0")
    if config.max_evaluations > MAX_ALLOWED_EVALUATIONS:
        raise ValueError(f"max_evaluations must be <= {MAX_ALLOWED_EVALUATIONS}")
    if config.no_improvement_patience <= 0:
        raise ValueError("no_improvement_patience must be > 0")
    if not 0.0 < config.max_invalid_ratio <= 1.0:
        raise ValueError("max_invalid_ratio must be in (0, 1]")
    if config.optimizer_backend not in {"auto", "optuna", "skopt"}:
        raise ValueError("optimizer_backend must be one of: auto, optuna, skopt")
    if config.curve_parameterization_mode not in {"linearized", "native"}:
        raise ValueError("curve_parameterization_mode must be one of: linearized, native")


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    config = build_config(args)
    validate_config(config)
    pipeline = OptimizationPipeline(config)
    run_dir = pipeline.run()
    print(f"[OptimizationPipeline] DONE: {run_dir}")


def run_from_editor_config() -> None:
    """IDE 直接运行入口，使用 EDITOR_RUN_CONFIG。"""
    config = build_config_from_editor_config(EDITOR_RUN_CONFIG)
    validate_config(config)
    print("[OptimizationPipeline] RUN_WITH_EDITOR_CONFIG=True")
    print(f"[OptimizationPipeline] parameter_json: {config.parameter_json}")
    print(f"[OptimizationPipeline] instance_json:   {config.instance_json}")
    print(f"[OptimizationPipeline] port_summary:    {config.port_summary}")
    print(f"[OptimizationPipeline] run_name:        {config.run_name}")
    print(f"[OptimizationPipeline] build_only:      {not config.run_solver}")
    print(f"[OptimizationPipeline] optimizer:       {config.optimizer_backend}")

    ###关键点提取在OptimizationPipeline的初始化中
    pipeline = OptimizationPipeline(config)
    run_dir = pipeline.run()
    print(f"[OptimizationPipeline] DONE: {run_dir}")


if __name__ == "__main__":
    if len(sys.argv) == 1 and bool(EDITOR_RUN_CONFIG.get("RUN_WITH_EDITOR_CONFIG", False)):
        run_from_editor_config()
    else:
        main()
