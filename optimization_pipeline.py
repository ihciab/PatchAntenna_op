from __future__ import annotations

import argparse
import datetime as _datetime
import json
import logging
import os
import shutil
import sys
import time
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
VERSIONED_LOCAL_PACKAGES_DIR = PROJECT_ROOT / f"local_packages_py{sys.version_info.major}{sys.version_info.minor}"
if VERSIONED_LOCAL_PACKAGES_DIR.exists():
    local_packages_text = str(VERSIONED_LOCAL_PACKAGES_DIR)
    if local_packages_text not in sys.path:
        sys.path.insert(0, local_packages_text)

MAX_ALLOWED_EVALUATIONS = 30


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
    "BASE_RUN_DIR": PROJECT_ROOT / "pipeline_runs" / "run_20260528_213437",
    "OUTPUT_ROOT": PROJECT_ROOT / "optimization_runs",
    "RUN_NAME": "bo_editor_full30_213437",
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
    "RANDOM_STATE": 42,
    # "auto": 优先 Optuna，失败则回退 skopt。
    # "optuna": 强制使用 Optuna TPESampler。
    # "skopt": 强制使用 scikit-optimize GP/EI。
    "OPTIMIZER_BACKEND": "optuna",
}

from geometry_validator import (  # noqa: E402
    GeometryValidationConfig,
    TopologySignature,
    geometry_complexity_metrics,
    make_topology_signature,
    validate_geometry,
)
from geometry_validation import validate_geometry as validate_and_repair_cst_geometry  # noqa: E402
from optimization_objectives import ObjectiveBreakdown, ObjectiveWeights, evaluate_objective  # noqa: E402
from primitive_mutator import DesignVariable, PrimitiveInventory, extract_design_variables, mutate_geometry  # noqa: E402
from s11_parser import S11Metrics, find_latest_s11_file, parse_s11_file  # noqa: E402


def prepare_cst_handoff_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    for index in sorted(unused):
        if components[index].get("start_node") == current_node:
            return index
    return None


def _component_handoff_points(component: Dict[str, Any]) -> List[List[float]]:
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
    return abs(float(a[0]) - float(b[0])) <= tolerance and abs(float(a[1]) - float(b[1])) <= tolerance


class OptunaBackend:
    name = "optuna_tpe"

    def __init__(self, random_state: int, max_evaluations: int) -> None:
        import optuna

        sampler = optuna.samplers.TPESampler(
            seed=random_state,
            n_startup_trials=min(5, max(1, max_evaluations // 3)),
            multivariate=False,
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(direction="minimize", sampler=sampler)

    def ask(self, variables: List[DesignVariable]):
        trial = self.study.ask()
        values = {
            variable.name: float(trial.suggest_float(variable.name, variable.lower, variable.upper))
            for variable in variables
        }
        return trial, values

    def tell(self, token: Any, objective: float) -> None:
        self.study.tell(token, float(objective))

    def trials(self) -> List[Dict[str, Any]]:
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
        import optuna

        return {
            key: float(value)
            for key, value in optuna.importance.get_param_importances(self.study).items()
        }


class SkoptBackend:
    name = "skopt_gp_ei"

    def __init__(self, random_state: int) -> None:
        from skopt import Optimizer
        from skopt.space import Real

        self._optimizer_cls = Optimizer
        self._real_cls = Real
        self.random_state = random_state
        self.optimizer = None
        self._trials: List[Dict[str, Any]] = []

    def ask(self, variables: List[DesignVariable]):
        if self.optimizer is None:
            dimensions = [
                self._real_cls(variable.lower, variable.upper, name=variable.name)
                for variable in variables
            ]
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
        return list(self._trials)

    def param_importances(self) -> Dict[str, float]:
        return {}


@dataclass
class OptimizationConfig:
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
    random_state: int = 42
    optimizer_backend: str = "auto"


@dataclass
class EvaluationRecord:
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
        return asdict(self)


@dataclass
class RunState:
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
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def setup_run(config: OptimizationConfig) -> RunState:
    run_name = config.run_name or _datetime.datetime.now().strftime("opt_%Y%m%d_%H%M%S")
    run_dir = config.output_root / run_name
    valid_designs_dir = run_dir / "valid_designs"
    invalid_designs_dir = run_dir / "invalid_designs"
    best_design_dir = run_dir / "best_design"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"
    for directory in (valid_designs_dir, invalid_designs_dir, best_design_dir, plots_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"optimization_pipeline.{run_name}")
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


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


class OptimizationPipeline:
    """curve_parameterization.json 后置优化层。

    不修改参数化 backend，不修改 CST reconstruction，只在二者之间新增
    primitive-safe mutation、validation、CST build/simulation 和目标函数评估。
    """

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config
        self.state = setup_run(config)
        self.payload = load_json(config.parameter_json)
        self.reference_signature = make_topology_signature(self.payload)
        self.validation_config = GeometryValidationConfig()
        self.variables, self.inventory = extract_design_variables(self.payload)
        self.objective_weights = ObjectiveWeights()
        self.optimizer = self._create_optimizer_backend()

    def run(self) -> Path:
        logger = self.state.logger
        try:
            self._write_run_metadata()
            reference_report = validate_geometry(
                self.payload,
                reference_signature=self.reference_signature,
                config=self.validation_config,
            )
            if not reference_report.valid:
                logger.warning("初始参数化 JSON 已存在几何风险: %s", reference_report.reasons)

            logger.info("开始优化: %s", self.config.parameter_json.resolve())
            logger.info("输出目录: %s", self.state.run_dir.resolve())
            logger.info("backend: %s", self.payload.get("backend", "unknown"))
            logger.info("设计变量: %s", [variable.to_dict() for variable in self.variables])
            logger.info("primitive inventory: %s", self.inventory.to_dict())

            no_improvement_count = 0
            consecutive_cst_failures = 0
            best_objective = None

            for evaluation in range(1, self.config.max_evaluations + 1):
                token, values = self.optimizer.ask(self.variables)
                record = self.evaluate(evaluation, values)
                objective_for_optimizer = record.objective
                if objective_for_optimizer is None:
                    objective_for_optimizer = self.objective_weights.cst_failure
                self.optimizer.tell(token, float(objective_for_optimizer))

                if record.status == "cst_failed":
                    consecutive_cst_failures += 1
                else:
                    consecutive_cst_failures = 0

                if record.objective is not None and (best_objective is None or record.objective < best_objective):
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
            return self.state.run_dir
        finally:
            close_logger(logger)

    def evaluate(self, evaluation: int, variables: Dict[str, float]) -> EvaluationRecord:
        logger = self.state.logger
        start = time.perf_counter()
        logger.info("evaluation=%03d variables=%s", evaluation, variables)

        mutated = mutate_geometry(self.payload, variables, self.inventory)
        validation = validate_geometry(mutated, self.reference_signature, self.validation_config)
        geometry_metrics = geometry_complexity_metrics(mutated, validation)

        if not validation.valid:
            design_path = self.state.invalid_designs_dir / f"eval_{evaluation:03d}.json"
            report_path = self.state.invalid_designs_dir / f"eval_{evaluation:03d}_validation.json"
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
            self.state.history.append(record)
            logger.warning("rejected invalid geometry eval=%03d reasons=%s", evaluation, validation.reasons)
            return record

        eval_dir = self.state.valid_designs_dir / f"eval_{evaluation:03d}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        design_path = eval_dir / "curve_parameterization.json"
        cst_handoff_payload = prepare_cst_handoff_payload(mutated)
        if cst_handoff_payload is not mutated:
            write_json(eval_dir / "mutation_raw_curve_parameterization.json", mutated)
            handoff_report = validate_geometry(cst_handoff_payload, None, self.validation_config)
            write_json(eval_dir / "cst_handoff_validation.json", handoff_report.to_dict())

        repaired_payload, cst_geometry_report = validate_and_repair_cst_geometry(
            cst_handoff_payload,
            output_dir=eval_dir,
            logger=logger,
        )
        write_json(design_path, repaired_payload)
        geometry_metrics["cst_robustness_score"] = cst_geometry_report.robustness_score

        if not cst_geometry_report.valid:
            objective_value = self.objective_weights.invalid_geometry
            record = EvaluationRecord(
                evaluation=evaluation,
                variables=variables,
                status="invalid_geometry",
                objective=objective_value,
                objective_breakdown={
                    "total": objective_value,
                    "failure_penalty": objective_value,
                    "reason": "cst_geometry_validation_failed",
                },
                validation={
                    "topology_validation": validation.to_dict(),
                    "cst_geometry_validation": cst_geometry_report.to_dict(),
                },
                geometry_metrics=geometry_metrics,
                design_json=str(design_path),
                error_message="; ".join(cst_geometry_report.errors),
                elapsed_seconds=time.perf_counter() - start,
            )
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
            cst_project = self._build_and_simulate(design_path, eval_dir)
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
        self.state.history.append(record)
        logger.info(
            "evaluation=%03d status=%s objective=%.6f elapsed=%.2fs",
            evaluation,
            status,
            objective.total,
            record.elapsed_seconds,
        )
        return record

    def _build_and_simulate(self, design_json: Path, eval_dir: Path) -> Path:
        from parameterized_json_to_cst import CSTParametricConfig, ParameterizedJsonCSTBuilder, load_instance_config

        if self.config.instance_json is not None:
            cst_config = load_instance_config(self.config.instance_json, self.config.layer_name)
        else:
            cst_config = CSTParametricConfig(project_folder=eval_dir / "cst")

        cst_config.project_folder = eval_dir / "cst"
        unique_suffix = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cst_config.project_name = f"optimization_eval_{design_json.parent.name}_{unique_suffix}"
        cst_config.run_solver = bool(self.config.run_solver)
        cst_config.port_summary_path = self.config.port_summary
        cst_config.simplify_tolerance_px = self.config.simplify_tolerance_px
        cst_config.geometry_frame = self.config.geometry_frame
        cst_config.close_project = True

        if cst_config.run_solver and cst_config.port_summary_path is None:
            raise ValueError("run_solver=True requires --port-summary for CST port reconstruction")

        cst_config.project_folder.mkdir(parents=True, exist_ok=True)
        builder = ParameterizedJsonCSTBuilder(design_json, cst_config)
        return builder.build()

    def _create_optimizer_backend(self):
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
        invalid_count = sum(1 for item in self.state.history if item.status == "invalid_geometry")
        invalid_ratio = invalid_count / max(1, len(self.state.history))
        if len(self.state.history) >= 5 and invalid_ratio > self.config.max_invalid_ratio:
            logger.warning("停止条件触发: invalid geometry ratio %.3f", invalid_ratio)
            return True
        if consecutive_cst_failures >= self.config.max_consecutive_cst_failures:
            logger.warning("停止条件触发: consecutive CST failures=%d", consecutive_cst_failures)
            return True
        return False

    def _save_best(self, record: EvaluationRecord) -> None:
        self.state.best_record = record
        self.state.best_design_dir.mkdir(parents=True, exist_ok=True)
        source = Path(record.design_json)
        target = self.state.best_design_dir / "curve_parameterization.json"
        shutil.copy2(source, target)
        write_json(self.state.best_design_dir / "best_record.json", record.to_dict())

        if record.s11_metrics:
            s11_path = Path(record.s11_metrics["s11_path"])
            if s11_path.exists():
                shutil.copy2(s11_path, self.state.best_design_dir / s11_path.name)

    def _save_history(self) -> None:
        write_json(
            self.state.history_path,
            {
                "records": [record.to_dict() for record in self.state.history],
                "best_record": self.state.best_record.to_dict() if self.state.best_record else None,
            },
        )
        self._save_optimizer_trials()

    def _write_run_metadata(self) -> None:
        metadata = {
            "config": {
                "parameter_json": str(self.config.parameter_json),
                "instance_json": str(self.config.instance_json) if self.config.instance_json else None,
                "port_summary": str(self.config.port_summary) if self.config.port_summary else None,
                "max_evaluations": self.config.max_evaluations,
                "target_frequency_ghz": self.config.target_frequency_ghz,
                "target_s11_db": self.config.target_s11_db,
                "run_solver": self.config.run_solver,
                "geometry_frame": self.config.geometry_frame,
                "simplify_tolerance_px": self.config.simplify_tolerance_px,
                "optimizer_backend": self.config.optimizer_backend,
            },
            "design_variables": [variable.to_dict() for variable in self.variables],
            "primitive_inventory": self.inventory.to_dict(),
            "reference_topology": self.reference_signature.to_dict(),
            "objective_weights": self.objective_weights.to_dict(),
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
        scales = [float(record.variables.get("global_scale", 1.0)) for record in completed]
        colors = ["#dc2626" if record.status != "completed" else "#2563eb" for record in completed]

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(evaluations, objectives, color="#111827", linewidth=1.2, alpha=0.6)
        ax.scatter(evaluations, objectives, c=colors, s=48)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel("Objective")
        ax.set_title("Optimization Objective History")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "objective_history.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(scales, objectives, c=colors, s=52)
        ax.set_xlabel("global_scale")
        ax.set_ylabel("Objective")
        ax.set_title("Design Variable vs Objective")
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
        from s11_parser import read_s11_rows

        rows = read_s11_rows(Path(best.s11_metrics["s11_path"]))
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot([row[0] for row in rows], [row[1] for row in rows], color="#7c3aed", linewidth=1.5)
        ax.axvline(self.config.target_frequency_ghz, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("S11 (dB)")
        ax.set_title("Best S11 Curve")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.state.plots_dir / "best_s11_curve.png", dpi=180)
        plt.close(fig)

    def _save_optimizer_importance_plot(self, plt: Any) -> None:
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

    def _save_optimizer_trials(self) -> None:
        write_json(self.state.run_dir / "optimizer_trials.json", self.optimizer.trials())


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--optimizer-backend",
        choices=["auto", "optuna", "skopt"],
        default="auto",
        help="auto tries Optuna first and falls back to skopt GP/EI.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OptimizationConfig:
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
        random_state=args.random_state,
        optimizer_backend=args.optimizer_backend,
    )


def build_config_from_editor_config(editor_config: Dict[str, Any]) -> OptimizationConfig:
    base_run_dir = Path(editor_config["BASE_RUN_DIR"])
    parameter_json = base_run_dir / "02_parameterization" / "curve_parameterization.json"
    instance_json = base_run_dir / "prepared_instance.json"
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
        random_state=int(editor_config["RANDOM_STATE"]),
        optimizer_backend=str(editor_config["OPTIMIZER_BACKEND"]),
    )


def validate_config(config: OptimizationConfig) -> None:
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


def main() -> None:
    args = parse_args()
    config = build_config(args)
    validate_config(config)
    pipeline = OptimizationPipeline(config)
    run_dir = pipeline.run()
    print(f"[OptimizationPipeline] DONE: {run_dir}")


def run_from_editor_config() -> None:
    config = build_config_from_editor_config(EDITOR_RUN_CONFIG)
    validate_config(config)
    print("[OptimizationPipeline] RUN_WITH_EDITOR_CONFIG=True")
    print(f"[OptimizationPipeline] parameter_json: {config.parameter_json}")
    print(f"[OptimizationPipeline] instance_json:   {config.instance_json}")
    print(f"[OptimizationPipeline] port_summary:    {config.port_summary}")
    print(f"[OptimizationPipeline] run_name:        {config.run_name}")
    print(f"[OptimizationPipeline] build_only:      {not config.run_solver}")
    print(f"[OptimizationPipeline] optimizer:       {config.optimizer_backend}")
    pipeline = OptimizationPipeline(config)
    run_dir = pipeline.run()
    print(f"[OptimizationPipeline] DONE: {run_dir}")


if __name__ == "__main__":
    if len(sys.argv) == 1 and bool(EDITOR_RUN_CONFIG.get("RUN_WITH_EDITOR_CONFIG", False)):
        run_from_editor_config()
    else:
        main()
