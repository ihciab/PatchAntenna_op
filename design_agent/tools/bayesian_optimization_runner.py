"""Design-agent bridge for the Bayesian optimization pipeline.

This module keeps the design-agent side thin: it reads the shared
``design_agent_runs/agents_inputs`` files, reads targets from the project
``target.json``, records the still-open optimization choices, and calls the
existing BO pipeline when explicitly requested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from design_agent.tools.bo_variable_selection import (
    STRATEGY_NAME as LLM_SLOT_MODEL_SIZE_STRATEGY,
    build_llm_slot_model_size_variable_plan_from_files,
)
from design_agent.tools.bo_parameterization_summary import write_bo_parameterization_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_INPUTS_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "optimization_runs"
DEFAULT_TARGET_JSON = PROJECT_ROOT / "target.json"


@dataclass
class BayesianOptimizationPreparation:
    """Serializable summary of a design-agent BO handoff."""

    ready_to_run: bool
    manifest_path: str
    config_preview: Dict[str, Any]
    input_files: Dict[str, Optional[str]]
    optional_context: Dict[str, Any]
    placeholders: Dict[str, Any]
    missing_inputs: List[str]


class BayesianOptimizationAgentRunner:
    """Prepare and optionally execute BO from design-agent JSON artifacts."""

    def __init__(self, input_dir: Path | str = AGENT_INPUTS_DIR) -> None:
        self.input_dir = Path(input_dir)

    def prepare(
        self,
        *,
        output_root: Path | str = DEFAULT_OUTPUT_ROOT,
        run_name: str = "design_agent_bo",
        max_evaluations: Optional[int] = None,
        build_only: Optional[bool] = None,
        instance_json: Optional[Path | str] = None,
        target_json: Path | str = DEFAULT_TARGET_JSON,
        layer_name: str = "layer0",
        simulation_f0_ghz: Optional[float] = None,
        simulation_f1_ghz: Optional[float] = None,
        simulation_frequency_unit: str = "GHz",
        optimizer_backend: str = "optuna",
        enable_multistage_optimization: bool = False,
        variable_strategy: Optional[Dict[str, Any]] = None,
        objective_strategy: Optional[Dict[str, Any]] = None,
        execute: bool = False,
    ) -> BayesianOptimizationPreparation:
        """Build a BO handoff manifest from ``agents_inputs``.

        ``execute=False`` is the safe default: the method only checks files and
        writes a manifest.  When ``execute=True`` all required runtime values
        must be present, then the existing ``OptimizationPipeline`` is called.
        """

        input_files = self._resolve_input_files(instance_json=instance_json, target_json=target_json)
        target_payload = _load_optional_json(input_files["target_json"])
        simulation_summary = _load_optional_json(input_files["simulation_summary"])
        primitive_analysis = _load_optional_json(input_files["primitive_analysis"])
        history = _load_optional_json(input_files["history"])
        metadata = _load_optional_json(input_files["metadata"])
        target_config = _extract_target_config(target_payload)
        variable_selection_plan = self._build_variable_selection_plan(input_files, variable_strategy)
        bo_parameterization_summary_path = self._write_bo_parameterization_summary(
            input_files=input_files,
            variable_selection_plan=variable_selection_plan,
        )

        config_preview: Dict[str, Any] = {
            "parameter_json": _path_text(input_files["parameter_json"]),
            "port_summary": _path_text(input_files["port_summary"]),
            "instance_json": _path_text(input_files["instance_json"]),
            "target_json": _path_text(input_files["target_json"]),
            "output_root": str(Path(output_root).resolve()),
            "run_name": run_name,
            "layer_name": layer_name,
            "max_evaluations": max_evaluations,
            "build_only": build_only,
            "run_solver": None if build_only is None else not bool(build_only),
            "target_frequency_ghz": target_config.get("target_frequency_ghz"),
            "target_s11_db": target_config.get("target_s11_db"),
            "target_gain_dbi": target_config.get("target_gain_dbi"),
            "target_bandwidth_reference_db": target_config.get("target_bandwidth_reference_db"),
            "simulation_f0_ghz": simulation_f0_ghz,
            "simulation_f1_ghz": simulation_f1_ghz,
            "simulation_frequency_unit": simulation_frequency_unit,
            "optimizer_backend": optimizer_backend,
            "enable_multistage_optimization": enable_multistage_optimization,
            "variable_strategy_name": variable_selection_plan.get("strategy_name"),
            "bo_parameterization_summary": _path_text(bo_parameterization_summary_path),
        }
        placeholders = {
            "design_variable_function": (
                "design_agent.tools.bo_variable_selection."
                "build_llm_slot_model_size_variable_plan_from_files"
            ),
            "variable_selection_strategy": variable_selection_plan,
            "custom_objective_function": (
                "design_agent.tools.experiment1_objective."
                "evaluate_experiment1_objective_from_files"
            ),
            "objective_strategy": objective_strategy
            or {
                "name": "experiment1_single_band_patch",
                "target_frequency_ghz": target_config.get("target_frequency_ghz"),
                "target_s11_db": target_config.get("target_s11_db"),
                "target_gain_dbi": target_config.get("target_gain_dbi"),
                "weights": {"frequency": 5.0, "matching": 3.0, "gain": 1.0},
                "target_source": _path_text(input_files["target_json"]),
            },
            "notes": (
                "The design-agent variable selection plan is generated separately from the existing "
                "bayesian_optimization variable extractor. It records the intended LLM-post-edit BO "
                "surface: slots plus whole-model size."
            ),
        }
        missing_inputs = self._missing_inputs(
            input_files=input_files,
            config_preview=config_preview,
            execute=execute,
        )
        manifest_path = self._write_manifest(
            output_root=Path(output_root),
            run_name=run_name,
            input_files=input_files,
            config_preview=config_preview,
            optional_context={
                "metadata_summary": _metadata_summary(metadata),
                "primitive_analysis_summary": primitive_analysis.get("summary")
                if isinstance(primitive_analysis, dict)
                else None,
                "history_loaded": bool(history),
                "history_empty_or_missing": not bool(history),
                "target": target_config,
            },
            placeholders=placeholders,
            missing_inputs=missing_inputs,
        )

        preparation = BayesianOptimizationPreparation(
            ready_to_run=not missing_inputs,
            manifest_path=str(manifest_path.resolve()),
            config_preview=config_preview,
            input_files={key: _path_text(value) for key, value in input_files.items()},
            optional_context={
                "metadata_summary": _metadata_summary(metadata),
                "primitive_analysis_summary": primitive_analysis.get("summary")
                if isinstance(primitive_analysis, dict)
                else None,
                "history_loaded": bool(history),
                "history_empty_or_missing": not bool(history),
                "target": target_config,
            },
            placeholders=placeholders,
            missing_inputs=missing_inputs,
        )

        if execute:
            if missing_inputs:
                raise ValueError("Cannot run Bayesian optimization; missing inputs: {0}".format(missing_inputs))
            run_dir = self._run_pipeline(preparation)
            preparation.config_preview["run_dir"] = str(run_dir.resolve())
        return preparation

    def _resolve_input_files(
        self,
        *,
        instance_json: Optional[Path | str],
        target_json: Path | str,
    ) -> Dict[str, Optional[Path]]:
        return {
            "parameter_json": self.input_dir / "curve_parameterization.json",
            "port_summary": self.input_dir / "patch_port_summary.json",
            "primitive_analysis": self.input_dir / "primitive_analysis.json",
            "history": self.input_dir / "history.json",
                "simulation_summary": self.input_dir / "simulation_summary.json",
            "bo_parameterization_summary": self.input_dir / "bo_parameterization_summary.json",
            "metadata": self.input_dir / "geometry_engine_bo_adapter_metadata.json",
            "instance_json": Path(instance_json) if instance_json is not None else None,
            "target_json": Path(target_json),
        }

    def _missing_inputs(
        self,
        *,
        input_files: Dict[str, Optional[Path]],
        config_preview: Dict[str, Any],
        execute: bool,
    ) -> List[str]:
        missing: List[str] = []
        for key in ("parameter_json", "port_summary"):
            path = input_files.get(key)
            if path is None or not path.exists():
                missing.append(key)
        target_path = input_files.get("target_json")
        if target_path is None or not target_path.exists():
            missing.append("target_json")
        instance_path = input_files.get("instance_json")
        if instance_path is not None and not instance_path.exists():
            missing.append("instance_json")
        if config_preview.get("target_frequency_ghz") is None:
            missing.append("target.f0_ghz")
        if config_preview.get("target_s11_db") is None:
            missing.append("target.s11.threshold_db")
        if execute:
            if config_preview.get("max_evaluations") is None:
                missing.append("max_evaluations")
            if config_preview.get("build_only") is None:
                missing.append("build_only")
            if config_preview.get("run_solver"):
                if config_preview.get("simulation_f0_ghz") is None:
                    missing.append("simulation_f0_ghz")
                if config_preview.get("simulation_f1_ghz") is None:
                    missing.append("simulation_f1_ghz")
        return sorted(set(missing))

    def _build_variable_selection_plan(
        self,
        input_files: Dict[str, Optional[Path]],
        variable_strategy: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        strategy = dict(variable_strategy or {})
        strategy_name = str(strategy.get("name") or LLM_SLOT_MODEL_SIZE_STRATEGY)
        if strategy_name != LLM_SLOT_MODEL_SIZE_STRATEGY:
            strategy.setdefault("strategy_name", strategy_name)
            return strategy

        parameter_json = input_files.get("parameter_json")
        if parameter_json is None or not parameter_json.exists():
            return {
                "strategy_name": LLM_SLOT_MODEL_SIZE_STRATEGY,
                "status": "unavailable",
                "reason": "parameter_json missing",
            }

        primitive_analysis = input_files.get("primitive_analysis")
        return build_llm_slot_model_size_variable_plan_from_files(
            parameterization_json=parameter_json,
            primitive_analysis_json=primitive_analysis if primitive_analysis and primitive_analysis.exists() else None,
            output_path=self.input_dir / "bo_variable_plan.json",
            scale_min=float(strategy.get("scale_min", 0.5)),
            scale_max=float(strategy.get("scale_max", 1.5)),
            global_scale_min=float(strategy.get("global_scale_min", 0.8)),
            global_scale_max=float(strategy.get("global_scale_max", 1.5)),
            board_margin=float(strategy.get("board_margin", 0.0)),
        )

    def _write_bo_parameterization_summary(
        self,
        *,
        input_files: Dict[str, Optional[Path]],
        variable_selection_plan: Dict[str, Any],
    ) -> Optional[Path]:
        parameter_json = input_files.get("parameter_json")
        if parameter_json is None or not parameter_json.exists():
            return None
        return write_bo_parameterization_summary(
            parameterization_json=parameter_json,
            output_path=self.input_dir / "bo_parameterization_summary.json",
            port_summary_json=input_files.get("port_summary"),
            primitive_analysis_json=input_files.get("primitive_analysis"),
            variable_plan=variable_selection_plan,
        )

    def _write_manifest(
        self,
        *,
        output_root: Path,
        run_name: str,
        input_files: Dict[str, Optional[Path]],
        config_preview: Dict[str, Any],
        optional_context: Dict[str, Any],
        placeholders: Dict[str, Any],
        missing_inputs: List[str],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_dir = output_root / "design_agent_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{_safe_name(run_name)}_{timestamp}.json"
        payload = {
            "schema_version": "design_agent_bayesian_optimization_manifest_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_dir": str(self.input_dir.resolve()),
            "input_files": {key: _path_text(value) for key, value in input_files.items()},
            "config_preview": config_preview,
            "optional_context": optional_context,
            "placeholders": placeholders,
            "missing_inputs": missing_inputs,
            "ready_to_run": not missing_inputs,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest_path

    def _run_pipeline(self, preparation: BayesianOptimizationPreparation) -> Path:
        from bayesian_optimization.pipelines.optimization_pipeline import (
            OptimizationConfig,
            validate_config,
        )
        from design_agent.tools.design_agent_bo_pipeline import DesignAgentOptimizationPipeline

        preview = preparation.config_preview
        config = OptimizationConfig(
            parameter_json=Path(str(preview["parameter_json"])),
            output_root=Path(str(preview["output_root"])),
            run_name=str(preview["run_name"]),
            instance_json=Path(str(preview["instance_json"])) if preview.get("instance_json") else None,
            port_summary=Path(str(preview["port_summary"])) if preview.get("port_summary") else None,
            layer_name=str(preview["layer_name"]),
            max_evaluations=int(preview["max_evaluations"]),
            simulation_f0_ghz=float(preview["simulation_f0_ghz"]) if preview.get("simulation_f0_ghz") is not None else None,
            simulation_f1_ghz=float(preview["simulation_f1_ghz"]) if preview.get("simulation_f1_ghz") is not None else None,
            simulation_frequency_unit=str(preview.get("simulation_frequency_unit") or "GHz"),
            target_frequency_ghz=float(preview["target_frequency_ghz"]),
            target_s11_db=float(preview["target_s11_db"]),
            run_solver=bool(preview["run_solver"]),
            optimizer_backend=str(preview.get("optimizer_backend") or "optuna"),
            enable_multistage_optimization=bool(preview.get("enable_multistage_optimization", False)),
        )
        validate_config(config)
        variable_plan = preparation.placeholders.get("variable_selection_strategy") or {}
        return DesignAgentOptimizationPipeline(config, variable_plan).run()


def prepare_bayesian_optimization_from_files(**kwargs: Any) -> BayesianOptimizationPreparation:
    """Convenience function for callers that do not need an agent instance."""

    input_dir = kwargs.pop("input_dir", AGENT_INPUTS_DIR)
    return BayesianOptimizationAgentRunner(input_dir=input_dir).prepare(**kwargs)


def _load_optional_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _nested_get(payload: Dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _extract_target_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    s11 = target.get("s11") if isinstance(target.get("s11"), dict) else {}
    gain = target.get("gain") if isinstance(target.get("gain"), dict) else {}
    bandwidth = target.get("bandwidth") if isinstance(target.get("bandwidth"), dict) else {}
    return {
        "experiment": payload.get("experiment"),
        "target_frequency_ghz": _number_or_none(target.get("f0_ghz")),
        "target_s11_db": _number_or_none(s11.get("threshold_db")),
        "target_s11_goal": s11.get("goal"),
        "target_gain_dbi": _number_or_none(gain.get("threshold_dbi")),
        "target_gain_goal": gain.get("goal"),
        "target_bandwidth_reference_db": _number_or_none(bandwidth.get("reference_db")),
        "target_bandwidth_goal": bandwidth.get("goal"),
    }


def _number_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metadata_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not metadata:
        return {}
    return {
        "schema_version": metadata.get("schema_version"),
        "source_geometry_json": metadata.get("source_geometry_json"),
        "stackup_json": metadata.get("stackup_json"),
        "component_count": metadata.get("component_count"),
        "node_count": metadata.get("node_count"),
        "primitive_count": metadata.get("primitive_count"),
        "port_connection_status": _nested_get(metadata, "port_connection_report", "status"),
    }


def _path_text(path: Optional[Path]) -> Optional[str]:
    return None if path is None else str(path.resolve())


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value))
    return safe.strip("_") or "design_agent_bo"


__all__ = [
    "BayesianOptimizationAgentRunner",
    "BayesianOptimizationPreparation",
    "prepare_bayesian_optimization_from_files",
]
