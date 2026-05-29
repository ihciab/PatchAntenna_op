from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
VERSIONED_LOCAL_PACKAGES_DIR = PROJECT_ROOT / f"local_packages_py{sys.version_info.major}{sys.version_info.minor}"
if VERSIONED_LOCAL_PACKAGES_DIR.exists():
    local_packages_text = str(VERSIONED_LOCAL_PACKAGES_DIR)
    if local_packages_text not in sys.path:
        sys.path.insert(0, local_packages_text)

MAX_ALLOWED_ITERATIONS = 30


@dataclass(frozen=True)
class BOConfig:
    """实验性 BO 配置。第一阶段只优化 patch_length。"""

    initial_json: Path
    results_dir: Path = PROJECT_ROOT / "results"
    pipeline_callable: str = "fss_parameterized_cst_pipeline:run_existing_pipeline"
    target_frequency_ghz: float = 2.4
    patch_length_min_mm: float = 15.0
    patch_length_max_mm: float = 40.0
    max_iterations: int = MAX_ALLOWED_ITERATIONS
    success_tolerance_ghz: float = 0.05
    failed_objective_threshold_ghz: float = 0.5
    failed_objective_penalty: float = 10.0
    random_state: int = 42
    s11_filename: str = "s11.csv"
    allow_missing_patch_length: bool = False


@dataclass
class SimulationResult:
    """单次 CST 仿真和 S11 提取结果。"""

    iteration: int
    patch_length: float
    f_resonance: Optional[float] = None
    min_s11: Optional[float] = None
    objective_value: Optional[float] = None
    status: str = "pending"
    simulation_time: float = 0.0
    json_path: Optional[Path] = None
    s11_path: Optional[Path] = None
    pipeline_output: Optional[Path] = None
    error_message: Optional[str] = None

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "patch_length": f"{self.patch_length:.6f}",
            "f_resonance": "" if self.f_resonance is None else f"{self.f_resonance:.9f}",
            "min_s11": "" if self.min_s11 is None else f"{self.min_s11:.9f}",
            "objective_value": "" if self.objective_value is None else f"{self.objective_value:.9f}",
            "status": self.status,
            "simulation_time": f"{self.simulation_time:.3f}",
        }


@dataclass
class ResultsStore:
    """负责结果目录、CSV、JSON 和日志文件。"""

    results_dir: Path
    logger: logging.Logger
    history_csv: Path = field(init=False)
    success_json: Path = field(init=False)
    failed_json: Path = field(init=False)
    progress_plot: Path = field(init=False)
    patch_objective_plot: Path = field(init=False)
    best_s11_plot: Path = field(init=False)
    iteration_dir: Path = field(init=False)
    history_records: List[SimulationResult] = field(default_factory=list)
    success_cases: List[Dict[str, Any]] = field(default_factory=list)
    failed_cases: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_dir = self.results_dir / "iterations"
        self.iteration_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv = self.results_dir / "optimization_history.csv"
        self.success_json = self.results_dir / "success_cases.json"
        self.failed_json = self.results_dir / "failed_cases.json"
        self.progress_plot = self.results_dir / "optimization_progress.png"
        self.patch_objective_plot = self.results_dir / "patch_length_objective.png"
        self.best_s11_plot = self.results_dir / "best_s11_curve.png"
        self._init_history_csv()
        self._init_json_files()

    def _init_history_csv(self) -> None:
        with self.history_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "iteration",
                    "patch_length",
                    "f_resonance",
                    "min_s11",
                    "objective_value",
                    "status",
                    "simulation_time",
                ],
            )
            writer.writeheader()

    def _init_json_files(self) -> None:
        self._write_json(self.success_json, self.success_cases)
        self._write_json(self.failed_json, self.failed_cases)

    def append_history(self, result: SimulationResult) -> None:
        self.history_records.append(result)
        with self.history_csv.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(result.to_csv_row().keys()))
            writer.writerow(result.to_csv_row())

    def append_success(self, result: SimulationResult) -> None:
        item = {
            "iteration": result.iteration,
            "parameters": {"patch_length": result.patch_length},
            "f_resonance": result.f_resonance,
            "min_s11": result.min_s11,
            "objective_value": result.objective_value,
            "json_path": str(result.json_path) if result.json_path else None,
            "s11_path": str(result.s11_path) if result.s11_path else None,
        }
        self.success_cases.append(item)
        self._write_json(self.success_json, self.success_cases)

    def append_failed(self, result: SimulationResult) -> None:
        item = {
            "iteration": result.iteration,
            "parameters": {"patch_length": result.patch_length},
            "error_message": result.error_message or result.status,
            "objective_value": result.objective_value,
            "json_path": str(result.json_path) if result.json_path else None,
            "s11_path": str(result.s11_path) if result.s11_path else None,
        }
        self.failed_cases.append(item)
        self._write_json(self.failed_json, self.failed_cases)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def save_visualizations(self, target_frequency_ghz: float) -> None:
        """保存优化过程可视化，每次迭代后都会覆盖为最新版本。"""

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.logger.warning("可视化跳过：matplotlib 不可用，错误: %s", exc)
            return

        completed = [
            record
            for record in self.history_records
            if record.objective_value is not None and record.f_resonance is not None
        ]
        if not completed:
            return

        iterations = [record.iteration for record in completed]
        objectives = [float(record.objective_value) for record in completed]
        frequencies = [float(record.f_resonance) for record in completed]
        patch_lengths = [float(record.patch_length) for record in completed]

        best_record = min(completed, key=lambda record: float(record.objective_value))
        best_iteration = int(best_record.iteration)
        best_objective = float(best_record.objective_value or 0.0)

        # 图 1：目标函数和谐振频率随迭代变化。
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(iterations, objectives, marker="o", linewidth=1.8, color="#2563eb")
        axes[0].scatter([best_iteration], [best_objective], color="#dc2626", zorder=3, label="best")
        axes[0].set_ylabel("Objective |f_res - 2.4| (GHz)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(iterations, frequencies, marker="o", linewidth=1.8, color="#059669")
        axes[1].axhline(target_frequency_ghz, color="#dc2626", linestyle="--", linewidth=1.4)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Resonance Frequency (GHz)")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("Bayesian Optimization Progress")
        fig.tight_layout()
        fig.savefig(self.progress_plot, dpi=180)
        plt.close(fig)

        # 图 2：patch_length 与 objective 的关系，颜色表示迭代顺序。
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            patch_lengths,
            objectives,
            c=iterations,
            cmap="viridis",
            s=54,
            edgecolors="#111827",
            linewidths=0.4,
        )
        ax.scatter(
            [float(best_record.patch_length)],
            [best_objective],
            color="#dc2626",
            marker="*",
            s=160,
            label="best",
            zorder=4,
        )
        ax.set_xlabel("patch_length (mm)")
        ax.set_ylabel("Objective |f_res - 2.4| (GHz)")
        ax.set_title("Patch Length vs Objective")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.colorbar(scatter, ax=ax, label="Iteration")
        fig.tight_layout()
        fig.savefig(self.patch_objective_plot, dpi=180)
        plt.close(fig)

        # 图 3：当前最佳设计的 S11 曲线。
        if best_record.s11_path and best_record.s11_path.exists():
            try:
                s11_rows = read_s11_csv(best_record.s11_path)
            except Exception as exc:
                self.logger.warning("最佳 S11 曲线绘制失败: %s", exc)
                return

            s11_freq = [row[0] for row in s11_rows]
            s11_values = [row[1] for row in s11_rows]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(s11_freq, s11_values, linewidth=1.8, color="#7c3aed")
            if best_record.f_resonance is not None and best_record.min_s11 is not None:
                ax.scatter(
                    [best_record.f_resonance],
                    [best_record.min_s11],
                    color="#dc2626",
                    zorder=3,
                    label=f"min S11 @ {best_record.f_resonance:.4f} GHz",
                )
            ax.axvline(target_frequency_ghz, color="#111827", linestyle="--", linewidth=1.2)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("S11 (dB)")
            ax.set_title("Best S11 Curve")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.best_s11_plot, dpi=180)
            plt.close(fig)


def setup_logger(results_dir: Path) -> logging.Logger:
    """建立同时输出到控制台和 optimization_log.txt 的日志系统。"""

    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("experimental_bo_optimizer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = results_dir / "optimization_log.txt"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """关闭日志句柄，避免 Windows 上日志文件被占用。"""

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def import_pipeline_callable(callable_path: str) -> Callable[[Path], Any]:
    """按 module:function 形式加载已有 pipeline 入口，不在本模块重写旧流程。"""

    if ":" not in callable_path:
        raise ValueError(
            "pipeline callable 必须是 'module:function' 格式，"
            f"当前值为 {callable_path!r}"
        )

    module_name, function_name = callable_path.split(":", 1)
    module = importlib.import_module(module_name)
    pipeline_func = getattr(module, function_name)
    if not callable(pipeline_func):
        raise TypeError(f"{callable_path} 不是可调用对象")
    return pipeline_func


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def update_patch_length_json(
    source_json: Path,
    destination_json: Path,
    patch_length: float,
    allow_missing_patch_length: bool = False,
) -> Path:
    """只修改 patch_length，其余 JSON 字段保持不变。"""

    data = load_json(source_json)
    if "patch_length" not in data and not allow_missing_patch_length:
        raise KeyError(
            "输入 JSON 中没有 patch_length。为避免误改旧格式，默认不自动新增该字段；"
            "如确需新增，请使用 --allow-missing-patch-length。"
        )
    data["patch_length"] = float(patch_length)
    write_json(destination_json, data)
    return destination_json


def read_s11_csv(s11_path: Path) -> List[Tuple[float, float]]:
    """读取 frequency,S11 格式 CSV，返回 [(frequency_GHz, s11_dB), ...]。"""

    if not s11_path.exists():
        raise FileNotFoundError(f"S11 文件不存在: {s11_path}")

    rows: List[Tuple[float, float]] = []
    with s11_path.open("r", newline="", encoding="utf-8-sig") as file:
        sample = file.read(2048)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.excel
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(file, dialect)
        for row_index, row in enumerate(reader, start=1):
            if not row or len(row) < 2:
                continue

            first = row[0].strip()
            second = row[1].strip()
            if row_index == 1 and first.lower() in {"frequency", "freq", "f"}:
                continue

            try:
                frequency = float(first)
                s11_value = float(second)
            except ValueError:
                continue
            rows.append((frequency, s11_value))

    if not rows:
        raise ValueError(f"S11 CSV 中没有可解析数据: {s11_path}")
    return rows


def extract_resonance_from_s11(s11_path: Path) -> Tuple[float, float]:
    """找到最小 S11 对应的谐振频率。"""

    rows = read_s11_csv(s11_path)
    frequency, min_s11 = min(rows, key=lambda item: item[1])
    return frequency, min_s11


def calculate_objective(f_resonance: float, target_frequency_ghz: float) -> float:
    """目标函数：abs(f_resonance - 2.4GHz)。"""

    return abs(float(f_resonance) - float(target_frequency_ghz))


def normalize_pipeline_output(output: Any) -> Optional[Path]:
    """尽量从已有 pipeline 的返回值中提取路径，兼容 Path、str 和 dict。"""

    if output is None:
        return None
    if isinstance(output, Path):
        return output
    if isinstance(output, str):
        return Path(output)
    if isinstance(output, dict):
        for key in ("s11_path", "s11_csv", "output_dir", "run_dir", "cst_path", "project_dir"):
            value = output.get(key)
            if value:
                return Path(value)
    return None


def candidate_search_roots(
    pipeline_output: Optional[Path],
    json_path: Path,
    results_dir: Path,
) -> List[Path]:
    """构造 S11 搜索范围，优先围绕本次 pipeline 输出。"""

    roots: List[Path] = []
    if pipeline_output is not None:
        roots.append(pipeline_output if pipeline_output.is_dir() else pipeline_output.parent)
    roots.extend([json_path.parent, results_dir, PROJECT_ROOT])

    unique_roots: List[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved not in seen and resolved.exists():
            unique_roots.append(resolved)
            seen.add(resolved)
    return unique_roots


def find_s11_file(
    pipeline_output: Optional[Path],
    json_path: Path,
    results_dir: Path,
    s11_filename: str,
) -> Path:
    """自动定位 CST 导出的 s11.csv。"""

    if pipeline_output is not None and pipeline_output.is_file():
        if pipeline_output.name.lower() == s11_filename.lower():
            return pipeline_output

    for root in candidate_search_roots(pipeline_output, json_path, results_dir):
        direct = root / s11_filename
        if direct.exists():
            return direct

    matches: List[Path] = []
    for root in candidate_search_roots(pipeline_output, json_path, results_dir):
        matches.extend(root.rglob(s11_filename))
        matches.extend(root.rglob("*s11*.csv"))
        matches.extend(root.rglob("*S11*.csv"))

    existing_matches = [path for path in matches if path.is_file()]
    if not existing_matches:
        raise FileNotFoundError(
            f"未找到 {s11_filename}。请确认已有 pipeline 已导出 frequency,S11 格式 CSV。"
        )
    return max(existing_matches, key=lambda path: path.stat().st_mtime)


def run_existing_fss_pipeline_adapter(json_path: Path) -> Path:
    """可选适配器：调用当前工程已有 FSSParameterizedCSTPipeline。

    注意：这个函数不改动旧 pipeline，只是在本实验模块内调用它。
    对于已经提供 run_existing_pipeline(json_path) 的工程，可继续使用默认入口。
    """

    from fss_parameterized_cst_pipeline import FSSParameterizedCSTPipeline

    json_path = Path(json_path)
    run_name = f"bo_iteration_{json_path.stem}"
    pipeline = FSSParameterizedCSTPipeline(
        instance_json=json_path,
        output_root=PROJECT_ROOT / "pipeline_runs",
        run_name=run_name,
        build_only=False,
        parameterization_mode="standard",
        skip_fss_cleanup=False,
        honor_instance_skip=True,
        reuse_project_folder=False,
        reuse_project_name=False,
    )
    pipeline.run()
    return pipeline.run_dir


class ExperimentalBOOptimizer:
    """外挂式 Bayesian Optimization 优化器，使用 Optuna TPESampler。"""

    def __init__(self, config: BOConfig) -> None:
        self.config = config
        self.logger = setup_logger(config.results_dir)
        self.results = ResultsStore(config.results_dir, self.logger)
        self.pipeline_func = import_pipeline_callable(config.pipeline_callable)
        self.optimizer_backend = "optuna_tpe"
        self.study = None
        self.skopt_optimizer = None
        try:
            import optuna
            sampler = optuna.samplers.TPESampler(
                seed=config.random_state,
                n_startup_trials=min(5, max(1, config.max_iterations // 3)),
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.study = optuna.create_study(direction="minimize", sampler=sampler)
        except ImportError as exc:
            self.logger.warning("Optuna 不可用，回退到 scikit-optimize GP/EI: %s", exc)
            try:
                from skopt import Optimizer
                from skopt.space import Real
            except ImportError as skopt_exc:
                raise ImportError(
                    "Optuna 和 scikit-optimize 都不可用，请运行 pip install -r requirements.txt。"
                ) from skopt_exc
            self.optimizer_backend = "skopt_gp_ei"
            self.skopt_optimizer = Optimizer(
                dimensions=[
                    Real(
                        config.patch_length_min_mm,
                        config.patch_length_max_mm,
                        name="patch_length",
                    )
                ],
                base_estimator="GP",
                acq_func="EI",
                random_state=config.random_state,
            )

    def run(self) -> None:
        try:
            self.logger.info("开始实验性 Bayesian Optimization 模块")
            self.logger.info("初始 JSON: %s", self.config.initial_json.resolve())
            self.logger.info("结果目录: %s", self.config.results_dir.resolve())
            self.logger.info("目标频率: %.6f GHz", self.config.target_frequency_ghz)
            self.logger.info(
                "优化参数: patch_length, 范围 %.3f mm 到 %.3f mm",
                self.config.patch_length_min_mm,
                self.config.patch_length_max_mm,
            )

            for iteration in range(1, self.config.max_iterations + 1):
                if self.optimizer_backend == "optuna_tpe":
                    trial = self.study.ask()
                    patch_length = float(
                        trial.suggest_float(
                            "patch_length",
                            self.config.patch_length_min_mm,
                            self.config.patch_length_max_mm,
                        )
                    )
                    optimizer_token = trial
                else:
                    optimizer_token = self.skopt_optimizer.ask()
                    patch_length = float(optimizer_token[0])
                result = self.evaluate_iteration(iteration, patch_length)

                objective_for_optimizer = (
                    result.objective_value
                    if result.objective_value is not None
                    else self.config.failed_objective_penalty
                )
                if result.status == "failed":
                    objective_for_optimizer = max(
                        float(objective_for_optimizer),
                        self.config.failed_objective_penalty,
                    )
                if self.optimizer_backend == "optuna_tpe":
                    self.study.tell(optimizer_token, float(objective_for_optimizer))
                else:
                    self.skopt_optimizer.tell(optimizer_token, float(objective_for_optimizer))

                if result.objective_value is not None and result.objective_value < self.config.success_tolerance_ghz:
                    self.logger.info(
                        "达到 success case: iteration=%d, patch_length=%.6f, f_resonance=%.6f GHz, objective=%.6f",
                        iteration,
                        patch_length,
                        result.f_resonance,
                        result.objective_value,
                    )

            self.logger.info("BO 优化结束，共执行 %d 次迭代", self.config.max_iterations)
            self.logger.info("history: %s", self.results.history_csv.resolve())
            self.logger.info("success_cases: %s", self.results.success_json.resolve())
            self.logger.info("failed_cases: %s", self.results.failed_json.resolve())
        finally:
            close_logger(self.logger)

    def evaluate_iteration(self, iteration: int, patch_length: float) -> SimulationResult:
        result = SimulationResult(iteration=iteration, patch_length=patch_length)
        iteration_json = self.results.iteration_dir / f"iteration_{iteration:03d}.json"
        result.json_path = iteration_json

        self.logger.info(
            "iteration=%d 开始，patch_length=%.6f mm",
            iteration,
            patch_length,
        )

        start_time = time.perf_counter()
        try:
            update_patch_length_json(
                self.config.initial_json,
                iteration_json,
                patch_length,
                allow_missing_patch_length=self.config.allow_missing_patch_length,
            )
            pipeline_output_raw = self.pipeline_func(iteration_json)
            result.pipeline_output = normalize_pipeline_output(pipeline_output_raw)

            s11_path = find_s11_file(
                result.pipeline_output,
                iteration_json,
                self.config.results_dir,
                self.config.s11_filename,
            )
            result.s11_path = self._snapshot_s11(iteration, s11_path)

            f_resonance, min_s11 = extract_resonance_from_s11(result.s11_path)
            objective_value = calculate_objective(
                f_resonance,
                self.config.target_frequency_ghz,
            )

            result.f_resonance = f_resonance
            result.min_s11 = min_s11
            result.objective_value = objective_value

            if objective_value > self.config.failed_objective_threshold_ghz:
                result.status = "failed"
                result.error_message = (
                    "objective > "
                    f"{self.config.failed_objective_threshold_ghz:.3f} GHz"
                )
            else:
                result.status = "success" if objective_value < self.config.success_tolerance_ghz else "completed"

        except Exception as exc:
            result.status = "failed"
            result.error_message = str(exc)
            self.logger.exception(
                "iteration=%d 失败，patch_length=%.6f mm，错误: %s",
                iteration,
                patch_length,
                exc,
            )
        finally:
            result.simulation_time = time.perf_counter() - start_time
            self._record_result(result)

        return result

    def _snapshot_s11(self, iteration: int, s11_path: Path) -> Path:
        """复制每次迭代的 S11，避免下一次 CST 导出覆盖历史结果。"""

        snapshot_path = self.results.iteration_dir / f"iteration_{iteration:03d}_s11.csv"
        if s11_path.resolve() != snapshot_path.resolve():
            shutil.copy2(s11_path, snapshot_path)
        return snapshot_path

    def _record_result(self, result: SimulationResult) -> None:
        self.results.append_history(result)
        if result.status == "success":
            self.results.append_success(result)
        if result.status == "failed":
            self.results.append_failed(result)

        self.results.save_visualizations(self.config.target_frequency_ghz)
        self.logger.info(
            "iteration=%d 结束，status=%s, f_resonance=%s, min_s11=%s, objective=%s, time=%.3fs",
            result.iteration,
            result.status,
            _format_optional_float(result.f_resonance),
            _format_optional_float(result.min_s11),
            _format_optional_float(result.objective_value),
            result.simulation_time,
        )


def _format_optional_float(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.9f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="外挂式实验 Bayesian Optimization 模块：只优化 patch_length。"
    )
    parser.add_argument(
        "--initial-json",
        required=True,
        type=Path,
        help="初始 geometry/参数 JSON，只会读取并生成迭代副本，不会覆盖原文件。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="优化结果目录，默认 ./results。",
    )
    parser.add_argument(
        "--pipeline-callable",
        default="fss_parameterized_cst_pipeline:run_existing_pipeline",
        help="已有 pipeline 入口，格式为 module:function，函数签名应为 func(json_path)。",
    )
    parser.add_argument("--target-frequency-ghz", type=float, default=2.4)
    parser.add_argument("--patch-length-min-mm", type=float, default=15.0)
    parser.add_argument("--patch-length-max-mm", type=float, default=40.0)
    parser.add_argument("--max-iterations", type=int, default=MAX_ALLOWED_ITERATIONS)
    parser.add_argument("--success-tolerance-ghz", type=float, default=0.05)
    parser.add_argument("--failed-objective-threshold-ghz", type=float, default=0.5)
    parser.add_argument("--failed-objective-penalty", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--s11-filename",
        default="s11.csv",
        help="已有 pipeline 导出的 S 参数 CSV 文件名，默认 s11.csv。",
    )
    parser.add_argument(
        "--allow-missing-patch-length",
        action="store_true",
        help="允许在初始 JSON 缺少 patch_length 时新增该字段。",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BOConfig:
    return BOConfig(
        initial_json=args.initial_json,
        results_dir=args.results_dir,
        pipeline_callable=args.pipeline_callable,
        target_frequency_ghz=args.target_frequency_ghz,
        patch_length_min_mm=args.patch_length_min_mm,
        patch_length_max_mm=args.patch_length_max_mm,
        max_iterations=args.max_iterations,
        success_tolerance_ghz=args.success_tolerance_ghz,
        failed_objective_threshold_ghz=args.failed_objective_threshold_ghz,
        failed_objective_penalty=args.failed_objective_penalty,
        random_state=args.random_state,
        s11_filename=args.s11_filename,
        allow_missing_patch_length=args.allow_missing_patch_length,
    )


def validate_config(config: BOConfig) -> None:
    if not config.initial_json.exists():
        raise FileNotFoundError(f"初始 JSON 不存在: {config.initial_json}")
    if config.patch_length_min_mm >= config.patch_length_max_mm:
        raise ValueError("patch_length 最小值必须小于最大值")
    if config.max_iterations <= 0:
        raise ValueError("max_iterations 必须大于 0")
    if config.max_iterations > MAX_ALLOWED_ITERATIONS:
        raise ValueError(f"max_iterations 最多只能为 {MAX_ALLOWED_ITERATIONS}")
    if config.success_tolerance_ghz <= 0:
        raise ValueError("success_tolerance_ghz 必须大于 0")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    validate_config(config)
    optimizer = ExperimentalBOOptimizer(config)
    optimizer.run()


if __name__ == "__main__":
    main()
