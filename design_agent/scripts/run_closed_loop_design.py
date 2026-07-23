"""Run the lightweight design -> geometry -> CST -> summary feedback loop.

The loop uses summarized agent inputs, keeps per-iteration artifacts under
``design_agent_runs/schem``, and refreshes ``design_agent_runs/agents_inputs``
for the next iteration.

Examples:

    python -m design_agent.scripts.run_closed_loop_design --iterations 1 --build-only
    python -m design_agent.scripts.run_closed_loop_design --iterations 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from design_agent.llm.client import LLMClient, OpenAICompatibleLLMClient
from design_agent.skills.lightweight_design import LightweightDesignSkill
from design_agent.tools.bayesian_optimization_runner import BayesianOptimizationAgentRunner
from design_agent.tools.bo_adapter import convert_geometry_engine_to_bo
from design_agent.tools.bo_reverse_adapter import convert_bo_parameterization_to_geometry_engine
from design_agent.tools.geometry_summary import write_geometry_summary
from design_agent.tools.history import (
    append_history_record,
    build_closed_loop_history_record,
    load_history,
    next_iteration_number,
    write_history,
)
from design_agent.tools.simulation_summary import resolve_s11_path, write_simulation_summary
from geometry_engine.cadquery_backend import CadQueryPlanarModel
from geometry_engine.context import GeometryContext
from geometry_engine.engine import GeometryEngine
from geometry_engine.geometry.feed import Feed
from geometry_engine.geometry.patch import Patch
from geometry_engine.geometry.slot import Slot
from design_agent.tools.geometry_summary import (
    bbox,
    first_planar_geometry,
    infer_patch_bbox,
    is_axis_aligned_rectangle_bbox,
    is_rectangle_loop,
    loop_vertices,
    summarize_feed,
)
from design_agent.scripts.run_geometry_engine_cst import GeometryEngineCSTRunner


AGENT_INPUTS_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "design_agent_runs" / "schem"
DEFAULT_SOURCE_RUN_DIR = PROJECT_ROOT / "design_agent_runs" / "initial_design_test"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description="Run the closed-loop lightweight antenna design workflow.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of closed-loop iterations to run.")
    parser.add_argument("--input-dir", default=str(AGENT_INPUTS_DIR), help="Folder containing target.md and summaries.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Folder for per-iteration artifacts.")
    parser.add_argument("--source-run-dir", default=str(DEFAULT_SOURCE_RUN_DIR), help="Initial run dir with stackup.json.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.json"), help="LLM config file.")
    parser.add_argument("--geometry-only", action="store_true", help="Skip CST and only test plan -> geometry update.")
    parser.add_argument("--build-only", action="store_true", help="Build CST model but do not run the solver.")
    parser.add_argument("--close-project", action="store_true", help="Close CST project after each CST run.")
    parser.add_argument("--f0", type=float, default=None, help="CST start frequency in GHz.")
    parser.add_argument("--f1", type=float, default=None, help="CST stop frequency in GHz.")
    parser.add_argument("--target-frequency", type=float, default=2.45, help="Target frequency in GHz.")
    parser.add_argument("--target-s11", type=float, default=-15.0, help="Target S11 in dB.")
    parser.add_argument("--target-gain", type=float, default=6.0, help="Target gain in dBi.")
    parser.add_argument("--target-bandwidth", type=float, default=None, help="Optional target bandwidth in GHz.")
    parser.add_argument("--s11-threshold", type=float, default=None, help="Threshold used for bandwidth calculation.")
    parser.add_argument("--gain", type=float, default=None, help="Optional gain value to include in summaries.")
    parser.add_argument("--initial-geometry-json", default=None, help="Optional current Geometry Engine JSON path.")
    parser.add_argument("--prepare-bo", action="store_true", help="Prepare BO inputs inside each closed-loop iteration.")
    parser.add_argument("--execute-bo", action="store_true", help="Run BO inside each closed-loop iteration.")
    parser.add_argument("--bo-output-root", default=None, help="Root directory for per-iteration BO runs.")
    parser.add_argument("--bo-run-name", default="design_agent_bo", help="Base BO run name.")
    parser.add_argument("--bo-max-evaluations", type=int, default=None, help="BO evaluations per closed-loop iteration.")
    parser.add_argument("--bo-build-only", action="store_true", help="Build BO CST projects without solver.")
    parser.add_argument("--bo-instance-json", default=None, help="Optional BO instance JSON.")
    parser.add_argument("--bo-target-json", default=str(PROJECT_ROOT / "target.json"), help="BO target JSON.")
    parser.add_argument("--bo-f0", type=float, default=None, help="BO CST start frequency in GHz.")
    parser.add_argument("--bo-f1", type=float, default=None, help="BO CST stop frequency in GHz.")
    parser.add_argument("--optimizer-backend", default="optuna", help="BO optimizer backend.")
    parser.add_argument("--enable-multistage-optimization", action="store_true", help="Enable BO multistage mode.")
    parser.add_argument(
        "--max-geometry-repair-attempts",
        type=int,
        default=5,
        help="Maximum LLM repair attempts after Geometry Engine validation failure.",
    )
    return parser


def main() -> int:
    """CLI entry point."""

    args = build_arg_parser().parse_args()
    runner = ClosedLoopDesignRunner(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        source_run_dir=Path(args.source_run_dir),
        config_path=Path(args.config),
        iterations=args.iterations,
        build_only=args.build_only,
        geometry_only=args.geometry_only,
        close_project=args.close_project,
        f0_ghz=args.f0,
        f1_ghz=args.f1,
        target_frequency_ghz=args.target_frequency,
        target_s11=args.target_s11,
        target_gain=args.target_gain,
        target_bandwidth=args.target_bandwidth,
        s11_threshold=args.s11_threshold,
        gain=args.gain,
        initial_geometry_json=Path(args.initial_geometry_json) if args.initial_geometry_json else None,
        max_geometry_repair_attempts=args.max_geometry_repair_attempts,
        prepare_bo=args.prepare_bo,
        execute_bo=args.execute_bo,
        bo_output_root=Path(args.bo_output_root) if args.bo_output_root else None,
        bo_run_name=args.bo_run_name,
        bo_max_evaluations=args.bo_max_evaluations,
        bo_build_only=True if args.bo_build_only else None,
        bo_instance_json=Path(args.bo_instance_json) if args.bo_instance_json else None,
        bo_target_json=Path(args.bo_target_json),
        bo_f0_ghz=args.bo_f0,
        bo_f1_ghz=args.bo_f1,
        optimizer_backend=args.optimizer_backend,
        enable_multistage_optimization=args.enable_multistage_optimization,
    )
    runner.run()
    return 0


class ClosedLoopDesignRunner:
    """Coordinate one or more lightweight closed-loop design iterations."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        source_run_dir: Path,
        config_path: Path,
        iterations: int,
        build_only: bool = False,
        geometry_only: bool = False,
        close_project: bool = False,
        f0_ghz: Optional[float] = None,
        f1_ghz: Optional[float] = None,
        target_frequency_ghz: float = 2.45,
        target_s11: float = -15.0,
        target_gain: float = 6.0,
        target_bandwidth: Optional[float] = None,
        s11_threshold: Optional[float] = None,
        gain: Optional[float] = None,
        initial_geometry_json: Optional[Path] = None,
        llm_client: Optional[LLMClient] = None,
        max_geometry_repair_attempts: int = 5,
        prepare_bo: bool = False,
        execute_bo: bool = False,
        bo_output_root: Optional[Path] = None,
        bo_run_name: str = "design_agent_bo",
        bo_max_evaluations: Optional[int] = None,
        bo_build_only: Optional[bool] = None,
        bo_instance_json: Optional[Path] = None,
        bo_target_json: Path = PROJECT_ROOT / "target.json",
        bo_f0_ghz: Optional[float] = None,
        bo_f1_ghz: Optional[float] = None,
        optimizer_backend: str = "optuna",
        enable_multistage_optimization: bool = False,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.source_run_dir = source_run_dir
        self.config_path = config_path
        self.iterations = int(iterations)
        self.build_only = bool(build_only)
        self.geometry_only = bool(geometry_only)
        self.close_project = bool(close_project)
        self.f0_ghz = f0_ghz
        self.f1_ghz = f1_ghz
        self.target_frequency_ghz = float(target_frequency_ghz)
        self.target_s11 = float(target_s11)
        self.target_gain = float(target_gain)
        self.target_bandwidth = target_bandwidth
        self.s11_threshold = s11_threshold
        self.gain = gain
        self.initial_geometry_json = initial_geometry_json
        self.llm_client = llm_client
        self.max_geometry_repair_attempts = int(max_geometry_repair_attempts)
        self.prepare_bo = bool(prepare_bo)
        self.execute_bo = bool(execute_bo)
        self.bo_output_root = Path(bo_output_root) if bo_output_root is not None else self.output_dir / "02_bayesian_optimization"
        self.bo_run_name = str(bo_run_name)
        self.bo_max_evaluations = bo_max_evaluations
        self.bo_build_only = bo_build_only
        self.bo_instance_json = bo_instance_json
        self.bo_target_json = Path(bo_target_json)
        self.bo_f0_ghz = bo_f0_ghz
        self.bo_f1_ghz = bo_f1_ghz
        self.optimizer_backend = str(optimizer_backend)
        self.enable_multistage_optimization = bool(enable_multistage_optimization)

        self.target_path = self.input_dir / "target.md"
        self.geometry_summary_path = self.input_dir / "geometry_summary.json"
        self.simulation_summary_path = self.input_dir / "simulation_summary.json"
        self.history_path = self.input_dir / "history.json"
        self.bo_parameterization_summary_path = self.input_dir / "bo_parameterization_summary.json"

    def run(self) -> List[Dict[str, Any]]:
        """Run all configured iterations."""

        if self.iterations < 1:
            raise ValueError("--iterations must be >= 1.")
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        client = self.llm_client or OpenAICompatibleLLMClient.from_config_file(str(self.config_path))
        skill = LightweightDesignSkill(llm_client=client)

        history = load_history(self.history_path)
        current_geometry_json = self._resolve_current_geometry_json()
        iteration_outputs: List[Dict[str, Any]] = []

        for offset in range(self.iterations):
            iteration = next_iteration_number(history)
            iteration_dir = self.output_dir / "iter_{0:03d}".format(iteration)
            iteration_dir.mkdir(parents=True, exist_ok=True)
            print("\n==================================================")
            print("Closed-loop iteration {0}".format(iteration))
            print("==================================================")

            llm_inputs = {
                "target": load_target_object(self.target_path),
                "geometry_summary": load_json_object(self.geometry_summary_path),
                "simulation_summary": load_json_object(self.simulation_summary_path),
                "history": history,
            }
            bo_summary = load_optional_json_object(self.bo_parameterization_summary_path)
            if bo_summary is not None:
                llm_inputs["bo_parameterization_summary"] = bo_summary
            trace = skill.run_with_trace(llm_inputs)
            operation_plan = remove_disallowed_early_delete_operations(
                trace["result"],
                llm_inputs["geometry_summary"],
            )
            write_json(iteration_dir / "diagnosis.json", trace["diagnosis"])
            write_json(iteration_dir / "plan.json", trace["plan"])
            before_simulation_summary = load_json_object(self.simulation_summary_path)

            operation_plan, geometry_result = self._apply_geometry_with_llm_repair(
                skill=skill,
                llm_inputs=llm_inputs,
                diagnosis=trace["diagnosis"],
                plan=trace["plan"],
                initial_operation_plan=operation_plan,
                current_geometry_json=current_geometry_json,
                iteration_dir=iteration_dir,
            )
            write_json(iteration_dir / "operation_plan.json", operation_plan)
            write_json(self.output_dir / "operation_plan.json", operation_plan)
            print("operation_plan:", iteration_dir / "operation_plan.json")
            write_json(iteration_dir / "geometry_engine_execution.json", geometry_result)
            current_geometry_json = Path(geometry_result["geometry_json"])
            print("geometry_json:", current_geometry_json)

            geometry_summary_out = write_geometry_summary(
                geometry_json_path=current_geometry_json,
                output_path=iteration_dir / "geometry_summary.json",
                target_frequency_ghz=self.target_frequency_ghz,
                epsilon_r=self._epsilon_r(),
            )
            copy_json(geometry_summary_out, self.geometry_summary_path)

            bo_result = (
                self._run_bo_for_iteration(
                    skill=skill,
                    history=history,
                    iteration=iteration,
                    iteration_dir=iteration_dir,
                    geometry_json=current_geometry_json,
                )
                if self.prepare_bo
                else {"enabled": False, "reason": "per-iteration BO is disabled."}
            )
            write_json(iteration_dir / "bo_result.json", bo_result)

            if bo_result.get("best_geometry_engine_json"):
                current_geometry_json = Path(str(bo_result["best_geometry_engine_json"]))
                copy_json(current_geometry_json, self.input_dir / "geometry_engine_geometry.json")
                geometry_summary_out = write_geometry_summary(
                    geometry_json_path=current_geometry_json,
                    output_path=iteration_dir / "bo_geometry_summary.json",
                    target_frequency_ghz=self.target_frequency_ghz,
                    epsilon_r=self._epsilon_r(),
                )
                copy_json(geometry_summary_out, self.geometry_summary_path)

            simulation_summary_out: Optional[Path] = (
                Path(str(bo_result["simulation_summary"]))
                if bo_result.get("simulation_summary")
                else None
            )
            after_simulation_summary: Optional[Dict[str, Any]] = (
                load_json_object(simulation_summary_out) if simulation_summary_out is not None else None
            )
            if simulation_summary_out is not None:
                copy_json(simulation_summary_out, self.simulation_summary_path)
                cst_result = {
                    "handled_by_bo": True,
                    "bo_run_dir": bo_result.get("run_dir"),
                    "best_record": bo_result.get("best_record"),
                    "run_solver": not bool(self.bo_build_only),
                }
                print("simulation_summary:", simulation_summary_out)
            else:
                cst_result = (
                    self._skip_cst_result()
                    if self.geometry_only
                    else self._run_cst(iteration=iteration, geometry_json=current_geometry_json)
                )
                if not self.build_only and not self.geometry_only:
                    s11_path = resolve_s11_path(search_dir=Path(cst_result["results_dir"]))
                    simulation_summary_out = write_simulation_summary(
                        output_path=iteration_dir / "simulation_summary.json",
                        s11_path=s11_path,
                        target_resonance=self.target_frequency_ghz,
                        target_bandwidth=self.target_bandwidth,
                        target_s11=self.target_s11,
                        target_gain=self.target_gain,
                        s11_threshold=self.s11_threshold,
                        gain=self.gain,
                    )
                    copy_json(simulation_summary_out, self.simulation_summary_path)
                    after_simulation_summary = load_json_object(simulation_summary_out)
                    print("simulation_summary:", simulation_summary_out)
                else:
                    print("CST skipped: simulation_summary was not refreshed")
            write_json(iteration_dir / "cst_result.json", cst_result)

            history_record = build_closed_loop_history_record(
                iteration=iteration,
                iteration_dir=iteration_dir,
                operation_plan=operation_plan,
                geometry_result=geometry_result,
                cst_result=cst_result,
                geometry_summary_path=geometry_summary_out,
                simulation_summary_path=simulation_summary_out,
                before_simulation_summary=before_simulation_summary,
                after_simulation_summary=after_simulation_summary,
            )
            history_record["bo"] = bo_result
            if bo_result.get("best_geometry_engine_json"):
                history_record.setdefault("geometry_engine", {})["post_bo_geometry_json"] = bo_result[
                    "best_geometry_engine_json"
                ]
            history_record["llm_effect_summary"] = self._build_llm_effect_summary(
                skill=skill,
                llm_inputs=llm_inputs,
                history_record=history_record,
                operation_plan=operation_plan,
                before_simulation_summary=before_simulation_summary,
                after_simulation_summary=after_simulation_summary,
                geometry_summary=load_json_object(geometry_summary_out),
                iteration_dir=iteration_dir,
            )
            history = append_history_record(history, history_record)
            if isinstance(bo_result.get("bo_effect_summary"), dict):
                history["bo_effect_summary"] = bo_result["bo_effect_summary"]
            write_history(self.history_path, history)
            write_json(iteration_dir / "history_after_iteration.json", history)
            print("history:", self.history_path)
            iteration_outputs.append(
                {
                    "iteration": iteration,
                    "iteration_dir": str(iteration_dir.resolve()),
                    "geometry_json": str(current_geometry_json.resolve()),
                    "geometry_summary": str(geometry_summary_out.resolve()),
                    "simulation_summary": None
                    if simulation_summary_out is None
                    else str(simulation_summary_out.resolve()),
                    "bo": bo_result,
                }
            )

        return iteration_outputs

    def _apply_geometry_with_llm_repair(
        self,
        skill: LightweightDesignSkill,
        llm_inputs: Dict[str, Dict[str, Any]],
        diagnosis: Dict[str, Any],
        plan: Dict[str, Any],
        initial_operation_plan: Dict[str, Any],
        current_geometry_json: Path,
        iteration_dir: Path,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply geometry operations, asking the LLM to repair invalid plans."""

        operation_plan = initial_operation_plan
        repair_attempts: List[Dict[str, Any]] = []
        max_attempts = max(0, self.max_geometry_repair_attempts)

        for attempt in range(max_attempts + 1):
            attempt_plan_path = iteration_dir / "operation_plan_attempt_{0:02d}.json".format(attempt)
            write_json(attempt_plan_path, operation_plan)
            try:
                geometry_result = self._apply_geometry_operations(
                    operation_plan=operation_plan,
                    current_geometry_json=current_geometry_json,
                    output_geometry_json=iteration_dir / "geometry_engine_geometry.json",
                )
                geometry_result["repair_attempts"] = repair_attempts
                return operation_plan, geometry_result
            except Exception as exc:
                error_payload = {
                    "attempt": attempt,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "failed_operation_plan": operation_plan,
                    "operation_plan_path": str(attempt_plan_path.resolve()),
                }
                repair_attempts.append(error_payload)
                write_json(
                    iteration_dir / "geometry_error_attempt_{0:02d}.json".format(attempt),
                    error_payload,
                )
                if attempt >= max_attempts:
                    raise
                print(
                    "Geometry failed on attempt {0}; asking LLM for repair: {1}".format(
                        attempt,
                        exc,
                    )
                )
                operation_plan = remove_disallowed_early_delete_operations(
                    skill.repair_operation_plan(
                        inputs=llm_inputs,
                        diagnosis=diagnosis,
                        plan=plan,
                        failed_operation_plan=operation_plan,
                        geometry_error=error_payload,
                        repair_attempt=attempt + 1,
                    ),
                    llm_inputs["geometry_summary"],
                )

        raise RuntimeError("Geometry repair loop ended unexpectedly.")

    def _run_bo_for_iteration(
        self,
        *,
        skill: LightweightDesignSkill,
        history: Dict[str, Any],
        iteration: int,
        iteration_dir: Path,
        geometry_json: Path,
    ) -> Dict[str, Any]:
        """Prepare and optionally execute BO for the current LLM-produced geometry."""

        stackup_path = self.source_run_dir / "stackup.json"
        adapter_paths = convert_geometry_engine_to_bo(
            geometry_json_path=geometry_json,
            output_dir=self.input_dir,
            stackup_path=stackup_path if stackup_path.exists() else None,
            connect_port=True,
            include_primitive_analysis=True,
        )
        run_name = "{0}_iter_{1:03d}".format(self.bo_run_name, iteration)
        preparation = BayesianOptimizationAgentRunner(input_dir=self.input_dir).prepare(
            output_root=self.bo_output_root,
            run_name=run_name,
            max_evaluations=self.bo_max_evaluations,
            build_only=self.bo_build_only,
            instance_json=self.bo_instance_json,
            target_json=self.bo_target_json,
            simulation_f0_ghz=self.bo_f0_ghz,
            simulation_f1_ghz=self.bo_f1_ghz,
            optimizer_backend=self.optimizer_backend,
            enable_multistage_optimization=self.enable_multistage_optimization,
            execute=self.execute_bo,
        )
        result: Dict[str, Any] = {
            "enabled": True,
            "iteration": iteration,
            "source_geometry_json": str(geometry_json.resolve()),
            "adapter_outputs": {key: str(path.resolve()) for key, path in adapter_paths.items()},
            "manifest_path": preparation.manifest_path,
            "ready_to_run": preparation.ready_to_run,
            "missing_inputs": preparation.missing_inputs,
            "bo_parameterization_summary": preparation.config_preview.get("bo_parameterization_summary"),
            "run_dir": preparation.config_preview.get("run_dir"),
            "execute_bo": self.execute_bo,
        }
        if not self.execute_bo or not preparation.config_preview.get("run_dir"):
            result["bo_effect_summary"] = {
                "available": False,
                "reason": "BO was prepared for this closed-loop iteration but not executed.",
            }
            write_json(iteration_dir / "bo_effect_summary.json", result["bo_effect_summary"])
            write_json(self.input_dir / "bo_effect_summary.json", result["bo_effect_summary"])
            return result

        bo_run_dir = Path(str(preparation.config_preview["run_dir"]))
        optimization_history_path = bo_run_dir / "optimization_history.json"
        if not optimization_history_path.exists():
            result["bo_effect_summary"] = {
                "available": False,
                "reason": "optimization_history.json was not found after BO execution.",
                "bo_run_dir": str(bo_run_dir.resolve()),
            }
            write_json(iteration_dir / "bo_effect_summary.json", result["bo_effect_summary"])
            write_json(self.input_dir / "bo_effect_summary.json", result["bo_effect_summary"])
            return result

        optimization_history = load_json_object(optimization_history_path)
        best_record = optimization_history.get("best_record")
        if isinstance(best_record, dict):
            result["best_record"] = best_record
            best_design_json_value = best_record.get("design_json")
            best_design_json = Path(str(best_design_json_value)) if best_design_json_value else None
            if best_design_json is not None and best_design_json.exists():
                reverse_output = iteration_dir / "bo_best_geometry_engine_geometry.json"
                port_summary_path = self._best_bo_port_summary_path(best_design_json.parent)
                try:
                    convert_bo_parameterization_to_geometry_engine(
                        best_design_json,
                        reverse_output,
                        source_geometry_json_path=geometry_json,
                        optimization_record=best_record,
                        port_summary_path=port_summary_path,
                    )
                    result["best_geometry_engine_json"] = str(reverse_output.resolve())
                except Exception as exc:
                    result["best_geometry_engine_json_error"] = {
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    }

            s11_metrics = best_record.get("s11_metrics") if isinstance(best_record.get("s11_metrics"), dict) else {}
            s11_path_value = s11_metrics.get("s11_path")
            if s11_path_value and Path(str(s11_path_value)).exists():
                simulation_summary_path = write_simulation_summary(
                    output_path=iteration_dir / "bo_simulation_summary.json",
                    s11_path=Path(str(s11_path_value)),
                    target_resonance=self.target_frequency_ghz,
                    target_bandwidth=self.target_bandwidth,
                    target_s11=self.target_s11,
                    target_gain=self.target_gain,
                    s11_threshold=self.s11_threshold,
                    gain=self.gain,
                )
                result["simulation_summary"] = str(simulation_summary_path.resolve())

        bo_parameterization_summary_path = self.input_dir / "bo_parameterization_summary.json"
        bo_parameterization_summary = (
            load_json_object(bo_parameterization_summary_path)
            if bo_parameterization_summary_path.exists()
            else None
        )
        try:
            bo_effect_summary = skill.reflect_bo_effect(
                target=load_target_object(self.target_path),
                history=history,
                optimization_history=optimization_history,
                bo_parameterization_summary=bo_parameterization_summary,
            )
        except Exception as exc:
            bo_effect_summary = {
                "available": False,
                "reason": "LLM BO reflection failed.",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "bo_run_dir": str(bo_run_dir.resolve()),
            }
        result["bo_effect_summary"] = bo_effect_summary
        write_json(iteration_dir / "bo_effect_summary.json", bo_effect_summary)
        write_json(self.input_dir / "bo_effect_summary.json", bo_effect_summary)
        return result

    @staticmethod
    def _best_bo_port_summary_path(eval_dir: Path) -> Optional[Path]:
        """Return the per-evaluation BO port summary most consistent with CST input."""

        for name in (
            "port_summary_connected.json",
            "port_summary_stage4_frozen.json",
            "port_connection_report.json",
        ):
            candidate = eval_dir / name
            if candidate.exists() and name != "port_connection_report.json":
                return candidate
        return None

    def _apply_geometry_operations(
        self,
        operation_plan: Dict[str, Any],
        current_geometry_json: Path,
        output_geometry_json: Path,
    ) -> Dict[str, Any]:
        """Execute operation_plan.json through Geometry Engine and export JSON."""

        geometry_summary = load_json_object(self.geometry_summary_path)
        patch = patch_from_geometry_json(current_geometry_json)
        engine = GeometryEngine(context=GeometryContext(patch=patch))
        dsl_commands = operations_to_dsl(operation_plan.get("operations", []), geometry_summary)
        if not dsl_commands:
            raise ValueError("operation_plan.json contains no geometry operations.")
        if dsl_commands[-1] != "Validate()":
            dsl_commands.append("Validate()")

        execution_log: List[Dict[str, Any]] = []
        for index, command in enumerate(dsl_commands):
            result = engine.execute(command)
            validation = engine.validate()
            execution_log.append(
                {
                    "index": index,
                    "command": command,
                    "result": str(result),
                    "valid": validation.valid,
                    "errors": list(validation.errors),
                }
            )
            if not validation.valid:
                raise ValueError("Geometry validation failed after {0}: {1}".format(command, validation.errors))

        self._center_patch_on_substrate(engine.context.patch)
        exported = engine.export_json(output_geometry_json)
        return {
            "input_geometry_json": str(current_geometry_json.resolve()),
            "geometry_json": str(exported.resolve()),
            "dsl_commands": dsl_commands,
            "execution_log": execution_log,
        }

    def _run_cst(self, iteration: int, geometry_json: Path) -> Dict[str, Any]:
        """Build or simulate CST for the current Geometry JSON."""

        runner = GeometryEngineCSTRunner(
            geometry_json_path=geometry_json,
            source_run_dir=self.source_run_dir,
            output_root=self.output_dir,
            project_name="ClosedLoop_Iter_{0:03d}".format(iteration),
            run_solver=not self.build_only,
            f0_ghz=self.f0_ghz,
            f1_ghz=self.f1_ghz,
            close_project=self.close_project,
        )
        cst_path = runner.run()
        return {
            "cst_project": str(cst_path.resolve()),
            "run_dir": str(runner.run_dir.resolve()),
            "results_dir": str((runner.run_dir / "03_results").resolve()),
            "metadata_path": str(runner.metadata_path.resolve()),
            "run_solver": not self.build_only,
        }

    def _build_llm_effect_summary(
        self,
        *,
        skill: LightweightDesignSkill,
        llm_inputs: Dict[str, Dict[str, Any]],
        history_record: Dict[str, Any],
        operation_plan: Dict[str, Any],
        before_simulation_summary: Dict[str, Any],
        after_simulation_summary: Optional[Dict[str, Any]],
        geometry_summary: Dict[str, Any],
        iteration_dir: Path,
    ) -> Dict[str, Any]:
        """Summarize this iteration's simulated effect for future planning."""

        if after_simulation_summary is None:
            summary = {
                "available": False,
                "reason": "No new simulation summary was generated; LLM effect reflection skipped.",
            }
            write_json(iteration_dir / "llm_effect_summary.json", summary)
            return summary

        try:
            summary = skill.reflect_iteration_effect(
                inputs=llm_inputs,
                operation_plan=operation_plan,
                before_simulation_summary=before_simulation_summary,
                after_simulation_summary=after_simulation_summary,
                geometry_summary=geometry_summary,
                effect=history_record.get("effect", {}),
            )
        except Exception as exc:
            summary = {
                "available": False,
                "reason": "LLM effect reflection failed.",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
        write_json(iteration_dir / "llm_effect_summary.json", summary)
        return summary

    @staticmethod
    def _skip_cst_result() -> Dict[str, Any]:
        """Return a structured CST skip marker for geometry-only tests."""

        return {
            "skipped": True,
            "reason": "--geometry-only was used.",
            "run_solver": False,
        }

    def _resolve_current_geometry_json(self) -> Path:
        if self.initial_geometry_json is not None:
            path = self.initial_geometry_json.resolve()
            if path.exists():
                write_geometry_summary(
                    geometry_json_path=path,
                    output_path=self.geometry_summary_path,
                    target_frequency_ghz=self.target_frequency_ghz,
                    epsilon_r=self._epsilon_r(),
                )
                return path
        summary = load_json_object(self.geometry_summary_path)
        source = summary.get("source", {}) if isinstance(summary.get("source"), dict) else {}
        source_path = source.get("path")
        if source_path:
            path = Path(source_path)
            if not path.is_absolute():
                path = self.geometry_summary_path.parent / path
            if path.exists():
                return path.resolve()
        raise FileNotFoundError(
            "Could not resolve current Geometry Engine JSON. Provide --initial-geometry-json, "
            "or make geometry_summary.source.path point to an existing geometry_engine_geometry_v1 file: {0}".format(
                self.geometry_summary_path
            )
        )

    def _epsilon_r(self) -> Optional[float]:
        stackup_path = self.source_run_dir / "stackup.json"
        if not stackup_path.exists():
            return None
        stackup = load_json_object(stackup_path)
        substrate = stackup.get("substrate", {}) if isinstance(stackup.get("substrate"), dict) else {}
        value = substrate.get("epsilon_r")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _center_patch_on_substrate(self, patch: Patch) -> None:
        """Keep the patch centered on the substrate and the feed on the patch centerline."""

        stackup_path = self.source_run_dir / "stackup.json"
        if not stackup_path.exists():
            return
        stackup = load_json_object(stackup_path)
        substrate = stackup.get("substrate", {}) if isinstance(stackup.get("substrate"), dict) else {}
        width = optional_number(substrate.get("width"))
        length = optional_number(substrate.get("length"))
        if width is None or length is None:
            return
        target_x = width / 2.0
        target_y = length / 2.0
        feed_direction = str(getattr(patch.feed, "direction", "bottom"))
        feed_offset = 0.0
        dx = target_x - patch.center_x
        dy = target_y - patch.center_y
        if not (math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9)):
            patch.translate(dx=dx, dy=dy)
        self._reattach_feed_to_substrate_edge(
            patch=patch,
            substrate_width=width,
            substrate_length=length,
            feed_direction=feed_direction,
            feed_offset=feed_offset,
        )

    @staticmethod
    def _reattach_feed_to_substrate_edge(
        *,
        patch: Patch,
        substrate_width: float,
        substrate_length: float,
        feed_direction: str,
        feed_offset: float,
    ) -> None:
        """After patch recentering, keep the feed connected to the substrate boundary."""

        if feed_direction in {"top", "bottom"}:
            patch.feed.x = patch.center_x + feed_offset
        else:
            patch.feed.y = patch.center_y + feed_offset
        patch.attach_feed_to_edge(feed_direction)
        if feed_direction == "bottom":
            patch.feed.length = max(0.0, patch.bottom)
        elif feed_direction == "top":
            patch.feed.length = max(0.0, substrate_length - patch.top)
        elif feed_direction == "left":
            patch.feed.length = max(0.0, patch.left)
        elif feed_direction == "right":
            patch.feed.length = max(0.0, substrate_width - patch.right)
        patch.rebuild_model()

def operations_to_dsl(operations: Any, geometry_summary: Dict[str, Any]) -> List[str]:
    """Convert operation_plan operations into Geometry Engine DSL strings."""

    if not isinstance(operations, list):
        raise ValueError("operation_plan.operations must be a list.")

    commands: List[str] = []
    patch = geometry_summary.get("patch", {}) if isinstance(geometry_summary.get("patch"), dict) else {}
    current_length = float(patch.get("length_mm"))
    current_width = float(patch.get("width_mm"))

    for index, item in enumerate(operations):
        if not isinstance(item, dict):
            raise ValueError("Operation {0} must be an object.".format(index))
        name = str(item.get("operation", "")).strip()
        params = item.get("parameters", {})
        if not isinstance(params, dict):
            raise ValueError("Operation {0} parameters must be an object.".format(index))

        if name == "ResizePatch":
            length = params.get("length_mm", params.get("absolute_length"))
            width = params.get("width_mm", params.get("absolute_width"))
            if length is None and params.get("length") is not None:
                length = current_length + float(params["length"])
            if width is None and params.get("width") is not None:
                width = current_width + float(params["width"])
            args = []
            if length is not None:
                args.append("length={0}".format(format_number(length)))
                current_length = float(length)
            if width is not None:
                args.append("width={0}".format(format_number(width)))
                current_width = float(width)
            if not args:
                raise ValueError("ResizePatch requires length/width delta or absolute length_mm/width_mm.")
            commands.append("ResizePatch({0})".format(", ".join(args)))
        elif name == "MoveFeed":
            raise ValueError(
                "MoveFeed is disabled in the closed-loop workflow. Use ResizePatch or AddSlot instead."
            )
        elif name == "AddSlot":
            shape = str(params.get("shape", "rectangle"))
            x = required_number_any(params, ("x", "center_x", "center_x_mm"))
            y = required_number_any(params, ("y", "center_y", "center_y_mm"))
            width = required_number_any(params, ("width", "width_mm"))
            height = required_number_any(params, ("height", "height_mm"))
            slot_id = params.get("id")
            args = [
                "shape={0}".format(repr(shape)),
                "x={0}".format(format_number(x)),
                "y={0}".format(format_number(y)),
                "width={0}".format(format_number(width)),
                "height={0}".format(format_number(height)),
            ]
            if slot_id is not None:
                args.append("id={0}".format(repr(str(slot_id))))
            commands.append("AddSlot({0})".format(", ".join(args)))
        elif name == "DeleteSlot":
            slot_count = current_slot_count(geometry_summary)
            if slot_count < 4:
                raise ValueError(
                    "DeleteSlot is disabled until the current geometry has at least four slots. "
                    "Current slot count: {0}.".format(slot_count)
                )
            slot_id = params.get("id")
            if slot_id is None:
                raise ValueError("DeleteSlot requires parameters.id.")
            commands.append("DeleteSlot(id={0})".format(repr(str(slot_id))))
        elif name == "Validate":
            commands.append("Validate()")
        else:
            raise ValueError("Unsupported operation: {0}".format(name))

    return commands


def remove_disallowed_early_delete_operations(
    operation_plan: Dict[str, Any],
    geometry_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Remove DeleteSlot operations before the design has enough slot history."""

    if current_slot_count(geometry_summary) >= 4:
        return operation_plan
    if not isinstance(operation_plan, dict):
        return operation_plan
    operations = operation_plan.get("operations")
    if not isinstance(operations, list):
        return operation_plan

    filtered = [
        operation
        for operation in operations
        if not (
            isinstance(operation, dict)
            and str(operation.get("operation", "")).strip() == "DeleteSlot"
        )
    ]
    if len(filtered) == len(operations):
        return operation_plan

    cleaned = dict(operation_plan)
    cleaned["operations"] = filtered
    return cleaned


def current_slot_count(geometry_summary: Optional[Dict[str, Any]]) -> int:
    """Return the number of summarized slots in the current geometry."""

    if not isinstance(geometry_summary, dict):
        return 0
    slots = geometry_summary.get("slots")
    return len(slots) if isinstance(slots, list) else 0


def patch_from_geometry_json(geometry_json: Path | str | Dict[str, Any]) -> Patch:
    """Build a Geometry Engine Patch object from geometry_engine_geometry_v1 JSON."""

    if isinstance(geometry_json, (str, Path)):
        payload = load_json_object(Path(geometry_json))
    else:
        payload = geometry_json
    if payload.get("schema_version") != "geometry_engine_geometry_v1":
        raise ValueError("Expected schema_version='geometry_engine_geometry_v1'.")

    geometry = first_planar_geometry(payload)
    outer_vertices = loop_vertices(geometry.get("outer_boundary"))
    if len(outer_vertices) < 3:
        raise ValueError("Geometry JSON outer_boundary must contain at least three vertices.")

    metadata = geometry.get("metadata") if isinstance(geometry.get("metadata"), dict) else {}
    feed_metadata = metadata.get("feed")
    feed_summary = summarize_feed(feed_metadata if isinstance(feed_metadata, dict) else None)
    patch_bbox = infer_patch_bbox(outer_vertices, feed_summary)
    left, bottom, right, top = [float(value) for value in patch_bbox]

    feed = Feed(
        x=float(feed_summary.get("x_mm", (left + right) / 2.0)),
        y=float(feed_summary.get("y_mm", bottom)),
        width=float(feed_summary.get("width_mm", 3.0)),
        length=float(feed_summary.get("length_mm", 0.0)),
        direction=str(feed_summary.get("direction", "bottom")),
    )

    rectangular_slots: List[Slot] = []
    polygon_holes: List[List[tuple[float, float]]] = []
    for index, hole in enumerate(geometry.get("holes", []) or [], start=1):
        if not isinstance(hole, dict):
            continue
        vertices = loop_vertices(hole)
        if len(vertices) < 3:
            continue
        if is_rectangle_loop(vertices):
            min_x, min_y, max_x, max_y = bbox(vertices)
            rectangular_slots.append(
                Slot(
                    id=str(hole.get("id", "slot_{0:03d}".format(index))),
                    shape="rectangle",
                    x=(min_x + max_x) / 2.0,
                    y=(min_y + max_y) / 2.0,
                    width=max_x - min_x,
                    height=max_y - min_y,
                )
            )
        else:
            polygon_holes.append(vertices)

    patch = Patch(
        width=right - left,
        length=top - bottom,
        center_x=(left + right) / 2.0,
        center_y=(bottom + top) / 2.0,
        layer=str(metadata.get("layer", "top")),
        feed=feed,
        slots=rectangular_slots,
    )

    if not is_axis_aligned_rectangle_bbox(patch_bbox, outer_vertices):
        patch.set_polygon(outer_vertices)
        patch.feed = feed
        patch.layer = str(metadata.get("layer", "top"))
        patch.slots = rectangular_slots
        for slot in rectangular_slots:
            patch.boolean_difference(
                CadQueryPlanarModel.rectangle(
                    width=slot.width,
                    height=slot.height,
                    center_x=slot.x,
                    center_y=slot.y,
                    z=patch.z,
                    thickness=patch.thickness * 3.0,
                )
            )

    for vertices in polygon_holes:
        patch.boolean_difference(
            CadQueryPlanarModel.polygon(
                points=vertices,
                z=patch.z,
                thickness=patch.thickness * 3.0,
            )
        )

    return patch


def patch_from_geometry_summary(summary: Dict[str, Any]) -> Patch:
    """Rebuild a Geometry Engine Patch object from compact geometry_summary.json."""

    patch_summary = summary.get("patch", {}) if isinstance(summary.get("patch"), dict) else {}
    feed_summary = summary.get("feed", {}) if isinstance(summary.get("feed"), dict) else {}
    bbox = patch_summary.get("bbox_mm")
    if not isinstance(bbox, list) or len(bbox) < 4:
        raise ValueError("geometry_summary.patch.bbox_mm is required.")

    left, bottom, right, top = [float(value) for value in bbox[:4]]
    feed = Feed(
        x=float(feed_summary.get("x_mm", (left + right) / 2.0)),
        y=float(feed_summary.get("y_mm", bottom)),
        width=float(feed_summary.get("width_mm", 3.0)),
        length=float(feed_summary.get("length_mm", 0.0)),
        direction=str(feed_summary.get("direction", "bottom")),
    )
    slots = []
    for index, slot_summary in enumerate(summary.get("slots", []) or [], start=1):
        if not isinstance(slot_summary, dict):
            continue
        slots.append(
            Slot(
                id=str(slot_summary.get("id", "slot_{0:03d}".format(index))),
                shape=str(slot_summary.get("type", "rectangle")),
                x=float(slot_summary.get("center_x_mm")),
                y=float(slot_summary.get("center_y_mm")),
                width=float(slot_summary.get("width_mm")),
                height=float(slot_summary.get("height_mm")),
            )
        )
    return Patch(
        width=right - left,
        length=top - bottom,
        center_x=(left + right) / 2.0,
        center_y=(bottom + top) / 2.0,
        feed=feed,
        slots=slots,
    )


def load_target_object(path: Path) -> Dict[str, Any]:
    """Load target.md as structured prompt input."""

    return {
        "format": "markdown",
        "source": str(path.resolve()),
        "content": path.read_text(encoding="utf-8"),
    }


def load_json_object(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object in {0}".format(path))
    return payload


def load_optional_json_object(path: Path) -> Optional[Dict[str, Any]]:
    """Load an optional JSON object, returning None when absent."""

    if not path.exists() or path.stat().st_size == 0:
        return None
    return load_json_object(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_json(source: Path, destination: Path) -> None:
    """Copy one JSON artifact via parse/write for stable formatting."""

    write_json(destination, load_json_object(source))


def required_number(params: Dict[str, Any], key: str) -> float:
    """Read a required finite number from an operation parameter dictionary."""

    if params.get(key) is None:
        raise ValueError("Missing required parameter: {0}".format(key))
    number = float(params[key])
    if not math.isfinite(number):
        raise ValueError("Parameter {0} must be finite.".format(key))
    return number


def optional_number(value: Any) -> Optional[float]:
    """Return a finite number or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def required_number_any(params: Dict[str, Any], keys: tuple[str, ...]) -> float:
    """Read the first available finite number from a set of parameter names."""

    for key in keys:
        if params.get(key) is not None:
            return required_number(params, key)
    raise ValueError("Missing one of required parameters: {0}".format(", ".join(keys)))


def format_number(value: Any) -> str:
    """Format a numeric DSL literal."""

    number = float(value)
    if not math.isfinite(number):
        raise ValueError("DSL numeric value must be finite.")
    return "{0:.9g}".format(number)


if __name__ == "__main__":
    raise SystemExit(main())
