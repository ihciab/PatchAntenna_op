"""Top-level design-agent pipeline orchestration."""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from design_agent.llm.client import LLMClient
from design_agent.skills.lightweight_design import LightweightDesignSkill
from design_agent.scripts.run_initial_design_test import run_initial_design_test
from design_agent.scripts.run_closed_loop_design import ClosedLoopDesignRunner, load_json_object
from design_agent.tools.geometry_summary import write_geometry_summary
from design_agent.tools.bayesian_optimization_runner import BayesianOptimizationAgentRunner
from design_agent.tools.bo_adapter import convert_geometry_engine_to_bo
from design_agent.tools.history import empty_history, load_history, refresh_history_knowledge, write_history
from geometry_engine.context import GeometryContext
from geometry_engine.engine import GeometryEngine
from geometry_engine.importer import ParameterizationImporter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "design_agent_runs"
DEFAULT_RUN_PREFIX = "degsin_test"
DEFAULT_SEED_INPUT_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"
DEFAULT_SOURCE_RUN_DIR = PROJECT_ROOT / "design_agent_runs" / "initial_design_test"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_TARGET_JSON = PROJECT_ROOT / "target.json"
DEFAULT_AGENT_CONFIG_PATH = PROJECT_ROOT / "design_agent" / "agent_config.json"

AGENT_INPUT_FILENAMES = (
    "target.md",
    "geometry_engine_geometry.json",
    "geometry_summary.json",
    "simulation_summary.json",
    "history.json",
    "bo_parameterization_summary.json",
    "bo_variable_plan.json",
    "curve_parameterization.json",
    "patch_port_summary.json",
    "primitive_analysis.json",
    "geometry_engine_bo_adapter_metadata.json",
)

CONFIG_SECTION_FIELDS = {
    "run_folder": ("run_root", "run_prefix", "run_index"),
    "input_paths": (
        "seed_input_dir",
        "source_run_dir",
        "target_json",
        "config_path",
        "initial_geometry_json",
    ),
    "closed_loop": (
        "iterations",
        "geometry_only",
        "build_only",
        "close_project",
        "f0_ghz",
        "f1_ghz",
        "target_frequency_ghz",
        "target_s11",
        "target_gain",
        "target_bandwidth",
        "s11_threshold",
        "gain",
        "max_geometry_repair_attempts",
    ),
    "bo": (
        "prepare_bo",
        "execute_bo",
        "bo_run_name",
        "bo_max_evaluations",
        "bo_build_only",
        "bo_instance_json",
        "bo_target_json",
        "bo_f0_ghz",
        "bo_f1_ghz",
        "optimizer_backend",
        "enable_multistage_optimization",
    ),
}

PATH_FIELDS = {
    "run_root",
    "seed_input_dir",
    "source_run_dir",
    "target_json",
    "config_path",
    "initial_geometry_json",
    "bo_instance_json",
    "bo_target_json",
}


@dataclass(frozen=True)
class DesignAgentPipelineConfig:
    """Configuration for one top-level design-agent pipeline run."""

    run_root: Path = DEFAULT_RUN_ROOT
    run_prefix: str = DEFAULT_RUN_PREFIX
    run_index: Optional[int] = None
    seed_input_dir: Path = DEFAULT_SEED_INPUT_DIR
    source_run_dir: Path = DEFAULT_SOURCE_RUN_DIR
    target_json: Path = DEFAULT_TARGET_JSON
    config_path: Path = DEFAULT_CONFIG_PATH
    iterations: int = 1
    geometry_only: bool = False
    build_only: bool = False
    close_project: bool = False
    f0_ghz: Optional[float] = None
    f1_ghz: Optional[float] = None
    target_frequency_ghz: float = 2.45
    target_s11: float = -15.0
    target_gain: float = 6.0
    target_bandwidth: Optional[float] = None
    s11_threshold: Optional[float] = None
    gain: Optional[float] = None
    initial_geometry_json: Optional[Path] = None
    max_geometry_repair_attempts: int = 5
    prepare_bo: bool = True
    execute_bo: bool = False
    bo_run_name: str = "design_agent_bo"
    bo_max_evaluations: Optional[int] = None
    bo_build_only: Optional[bool] = None
    bo_instance_json: Optional[Path] = None
    bo_target_json: Path = DEFAULT_TARGET_JSON
    bo_f0_ghz: Optional[float] = None
    bo_f1_ghz: Optional[float] = None
    optimizer_backend: str = "optuna"
    enable_multistage_optimization: bool = False


@dataclass(frozen=True)
class DesignAgentPipelineResult:
    """Paths produced by one top-level pipeline run."""

    run_dir: str
    agents_inputs_dir: str
    closed_loop_dir: str
    manifest_path: str
    stage_outputs: Dict[str, Any]


def load_pipeline_config(config_path: Path | str = DEFAULT_AGENT_CONFIG_PATH) -> DesignAgentPipelineConfig:
    """Load top-level pipeline config from ``agent_config.json``."""

    path = Path(config_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object: {0}".format(path))

    values: Dict[str, Any] = {}
    for section_name, field_names in CONFIG_SECTION_FIELDS.items():
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError("Config section `{0}` must be a JSON object.".format(section_name))
        for field_name in field_names:
            if field_name in section:
                values[field_name] = _config_value(field_name, section[field_name])

    # Flat overrides are accepted for small ad hoc config files.
    for field_name in DesignAgentPipelineConfig.__dataclass_fields__:
        if field_name in payload:
            values[field_name] = _config_value(field_name, payload[field_name])

    return DesignAgentPipelineConfig(**values)


class DesignAgentPipelineRunner:
    """Run the current design-agent workflow in numbered ``degsin_test_N`` folders."""

    def __init__(
        self,
        config: Optional[DesignAgentPipelineConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.config = self._normalize_config(config or DesignAgentPipelineConfig())
        self.llm_client = llm_client

    def run(self) -> DesignAgentPipelineResult:
        """Create a numbered run folder and execute the orchestrated stages."""

        run_dir = self._allocate_run_dir()
        agents_inputs_dir = run_dir / "agents_inputs"
        closed_loop_dir = run_dir / "01_closed_loop"
        manifests_dir = run_dir / "manifests"
        for path in (agents_inputs_dir, closed_loop_dir, manifests_dir):
            path.mkdir(parents=True, exist_ok=True)

        stage_outputs: Dict[str, Any] = {}
        stage_outputs["00_prepare_run_folder"] = {
            "run_dir": str(run_dir.resolve()),
            "agents_inputs_dir": str(agents_inputs_dir.resolve()),
            "closed_loop_dir": str(closed_loop_dir.resolve()),
        }
        stage_outputs["01_bootstrap_from_target"] = self._bootstrap_from_target_json(
            run_dir=run_dir,
            agents_inputs_dir=agents_inputs_dir,
        )
        effective_source_run_dir = Path(stage_outputs["01_bootstrap_from_target"]["initial_design_dir"])
        target_values = stage_outputs["01_bootstrap_from_target"]["target_values"]

        closed_loop = ClosedLoopDesignRunner(
            input_dir=agents_inputs_dir,
            output_dir=closed_loop_dir,
            source_run_dir=effective_source_run_dir,
            config_path=self.config.config_path,
            iterations=self.config.iterations,
            build_only=self.config.build_only,
            geometry_only=self.config.geometry_only,
            close_project=self.config.close_project,
            f0_ghz=self.config.f0_ghz,
            f1_ghz=self.config.f1_ghz,
            target_frequency_ghz=_value_or_default(
                target_values.get("target_frequency_ghz"),
                self.config.target_frequency_ghz,
            ),
            target_s11=_value_or_default(target_values.get("target_s11"), self.config.target_s11),
            target_gain=_value_or_default(target_values.get("target_gain"), self.config.target_gain),
            target_bandwidth=_value_or_default(
                target_values.get("target_bandwidth"),
                self.config.target_bandwidth,
            ),
            s11_threshold=_value_or_default(target_values.get("s11_threshold"), self.config.s11_threshold),
            gain=self.config.gain,
            initial_geometry_json=agents_inputs_dir / "geometry_engine_geometry.json",
            llm_client=self.llm_client,
            max_geometry_repair_attempts=self.config.max_geometry_repair_attempts,
            prepare_bo=self.config.prepare_bo,
            execute_bo=self.config.execute_bo,
            bo_output_root=run_dir / "02_bayesian_optimization",
            bo_run_name=self.config.bo_run_name,
            bo_max_evaluations=self.config.bo_max_evaluations,
            bo_build_only=self.config.bo_build_only,
            bo_instance_json=self.config.bo_instance_json,
            bo_target_json=self.config.bo_target_json,
            bo_f0_ghz=self.config.bo_f0_ghz,
            bo_f1_ghz=self.config.bo_f1_ghz,
            optimizer_backend=self.config.optimizer_backend,
            enable_multistage_optimization=self.config.enable_multistage_optimization,
        )
        closed_loop_iterations = closed_loop.run()
        stage_outputs["02_closed_loop"] = {
            "input_dir": str(agents_inputs_dir.resolve()),
            "output_dir": str(closed_loop_dir.resolve()),
            "latest_geometry_summary": str((agents_inputs_dir / "geometry_summary.json").resolve()),
            "latest_simulation_summary": str((agents_inputs_dir / "simulation_summary.json").resolve()),
            "history": str((agents_inputs_dir / "history.json").resolve()),
            "bo_mode": "inside_each_closed_loop_iteration" if self.config.prepare_bo else "disabled",
            "iterations": closed_loop_iterations,
        }

        manifest_path = self._write_manifest(run_dir, stage_outputs)
        result = DesignAgentPipelineResult(
            run_dir=str(run_dir.resolve()),
            agents_inputs_dir=str(agents_inputs_dir.resolve()),
            closed_loop_dir=str(closed_loop_dir.resolve()),
            manifest_path=str(manifest_path.resolve()),
            stage_outputs=stage_outputs,
        )
        return result

    def _allocate_run_dir(self) -> Path:
        run_root = self.config.run_root
        run_root.mkdir(parents=True, exist_ok=True)
        if self.config.run_index is not None:
            run_dir = run_root / "{0}_{1}".format(self.config.run_prefix, int(self.config.run_index))
            if run_dir.exists():
                raise FileExistsError("Run folder already exists: {0}".format(run_dir))
            run_dir.mkdir(parents=True)
            return run_dir

        next_index = self._next_run_index(run_root, self.config.run_prefix)
        while True:
            run_dir = run_root / "{0}_{1}".format(self.config.run_prefix, next_index)
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_dir
            except FileExistsError:
                next_index += 1

    def _seed_agents_inputs(self, destination: Path) -> Dict[str, Any]:
        copied: Dict[str, str] = {}
        missing: List[str] = []
        for filename in AGENT_INPUT_FILENAMES:
            source = self.config.seed_input_dir / filename
            target = destination / filename
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied[filename] = str(target.resolve())
            else:
                missing.append(filename)

        history_path = destination / "history.json"
        if not history_path.exists():
            write_history(history_path, empty_history())
            copied["history.json"] = str(history_path.resolve())
            if "history.json" in missing:
                missing.remove("history.json")

        required = ("target.md", "geometry_summary.json", "simulation_summary.json")
        geometry_seed = self._seed_geometry_source(destination)
        for generated in ("geometry_engine_geometry.json", "geometry_summary.json"):
            if generated in missing:
                missing.remove(generated)
        missing_required = [filename for filename in required if not (destination / filename).exists()]
        if missing_required:
            raise FileNotFoundError(
                "Missing required agent input files after seeding: {0}. Seed input dir: {1}".format(
                    ", ".join(missing_required),
                    self.config.seed_input_dir,
                )
            )

        return {
            "source_dir": str(self.config.seed_input_dir.resolve()),
            "destination_dir": str(destination.resolve()),
            "copied": copied,
            "missing_optional": sorted(missing),
            "geometry_source": geometry_seed,
        }

    def _bootstrap_from_target_json(self, run_dir: Path, agents_inputs_dir: Path) -> Dict[str, Any]:
        """Build this run's initial design and seed files from ``target.json``."""

        target_payload = load_json_object(self.config.target_json)
        target_values = _target_values(target_payload)
        initial_design_dir = run_dir / "00_initial_design"
        saved_initial_paths = run_initial_design_test(
            output_dir=initial_design_dir,
            target_json_path=self.config.target_json,
            client=self.llm_client,
            config_path=self.config.config_path,
        )

        target_md_path = agents_inputs_dir / "target.md"
        target_md_path.write_text(_target_markdown(target_payload), encoding="utf-8")

        geometry_json_path = self._export_initial_geometry_json(
            patch_json_path=initial_design_dir / "patch.json",
            stackup_json_path=initial_design_dir / "stackup.json",
            output_path=agents_inputs_dir / "geometry_engine_geometry.json",
        )
        geometry_summary_path = write_geometry_summary(
            geometry_json_path=geometry_json_path,
            output_path=agents_inputs_dir / "geometry_summary.json",
            target_frequency_ghz=_value_or_default(
                target_values.get("target_frequency_ghz"),
                self.config.target_frequency_ghz,
            ),
            epsilon_r=self._epsilon_r(initial_design_dir),
        )
        simulation_summary_path = self._write_initial_simulation_summary(
            output_path=agents_inputs_dir / "simulation_summary.json",
            target_values=target_values,
        )
        history_path = agents_inputs_dir / "history.json"
        write_history(history_path, empty_history())

        return {
            "target_json": str(self.config.target_json.resolve()),
            "target_md": str(target_md_path.resolve()),
            "initial_design_dir": str(initial_design_dir.resolve()),
            "initial_design_outputs": [str(path.resolve()) for path in saved_initial_paths],
            "patch_json": str((initial_design_dir / "patch.json").resolve()),
            "stackup_json": str((initial_design_dir / "stackup.json").resolve()),
            "geometry_engine_geometry": str(geometry_json_path.resolve()),
            "geometry_summary": str(geometry_summary_path.resolve()),
            "simulation_summary": str(simulation_summary_path.resolve()),
            "history": str(history_path.resolve()),
            "target_values": target_values,
        }

    @staticmethod
    def _export_initial_geometry_json(patch_json_path: Path, stackup_json_path: Path, output_path: Path) -> Path:
        """Convert initial design ``patch.json`` into Geometry Engine JSON."""

        patch = ParameterizationImporter().from_file(patch_json_path)
        _ensure_stackup_clearance(patch, stackup_json_path)
        _center_patch_on_stackup(patch, stackup_json_path)
        engine = GeometryEngine(context=GeometryContext(patch=patch))
        validation = engine.validate()
        if not validation.valid:
            raise ValueError("Initial Geometry Engine patch is invalid: {0}".format(validation.errors))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return engine.export_json(output_path)

    @staticmethod
    def _write_initial_simulation_summary(output_path: Path, target_values: Dict[str, Any]) -> Path:
        """Write the no-simulation-yet summary required by the lightweight skill."""

        payload = {
            "status": "not_simulated",
            "reason": "Initial target bootstrap has not run CST yet.",
            "target": {
                "f0_ghz": target_values.get("target_frequency_ghz"),
                "s11_db_max": target_values.get("target_s11"),
                "gain_dbi_min": target_values.get("target_gain"),
                "bandwidth_ghz_min": target_values.get("target_bandwidth"),
            },
            "current": {
                "f0_ghz": None,
                "s11_at_target_db": None,
                "peak_s11_db": None,
                "gain_dbi": None,
                "bandwidth_ghz": None,
            },
            "gap_to_target": {
                "frequency_error_ghz": None,
                "s11_error_db": None,
                "gain_error_dbi": None,
                "bandwidth_error_ghz": None,
            },
            "passed": {
                "frequency": None,
                "s11": None,
                "gain": None,
                "bandwidth": None,
                "overall": None,
            },
            "resonance": [],
            "target_resonance": target_values.get("target_frequency_ghz"),
            "frequency_error": None,
            "bandwidth": None,
            "target_bandwidth": target_values.get("target_bandwidth"),
            "target_s11": target_values.get("target_s11"),
            "peak_s11": None,
            "s11_at_target": None,
            "s11_error": None,
            "target_gain": target_values.get("target_gain"),
            "gain": None,
            "gain_error": None,
            "bandwidth_start": None,
            "bandwidth_end": None,
            "s11_threshold": target_values.get("s11_threshold"),
            "point_count": 0,
            "s11_path": None,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def _seed_geometry_source(self, destination: Path) -> Dict[str, str]:
        """Copy the canonical Geometry Engine JSON into this run and rebuild its summary."""

        target = destination / "geometry_engine_geometry.json"
        source = self._resolve_initial_geometry_source(destination)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)

        summary_path = write_geometry_summary(
            geometry_json_path=target,
            output_path=destination / "geometry_summary.json",
            target_frequency_ghz=self.config.target_frequency_ghz,
            epsilon_r=self._epsilon_r(),
        )
        return {
            "source": str(source.resolve()),
            "run_geometry_json": str(target.resolve()),
            "geometry_summary": str(summary_path.resolve()),
        }

    def _resolve_initial_geometry_source(self, destination: Path) -> Path:
        if self.config.initial_geometry_json is not None:
            path = self.config.initial_geometry_json.resolve()
            if not path.exists():
                raise FileNotFoundError("Configured initial_geometry_json does not exist: {0}".format(path))
            return path

        copied_geometry = destination / "geometry_engine_geometry.json"
        if copied_geometry.exists():
            return copied_geometry

        summary_path = destination / "geometry_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                "A real Geometry Engine JSON is required to seed the run, but neither "
                "initial_geometry_json nor geometry_engine_geometry.json was provided. "
                "Seed input dir: {0}".format(self.config.seed_input_dir)
            )

        summary = load_json_object(summary_path)
        source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
        source_path = source.get("path")
        if source_path:
            path = Path(source_path)
            if not path.is_absolute():
                path = summary_path.parent / path
            if path.exists():
                return path.resolve()
            raise FileNotFoundError(
                "geometry_summary.json points to a missing Geometry Engine JSON: {0}. "
                "Provide input_paths.initial_geometry_json or include geometry_engine_geometry.json "
                "in the seed input directory.".format(path)
            )

        raise FileNotFoundError(
            "geometry_summary.json does not contain source.path. Provide input_paths.initial_geometry_json "
            "or include geometry_engine_geometry.json in the seed input directory."
        )

    def _epsilon_r(self, source_run_dir: Optional[Path] = None) -> Optional[float]:
        stackup_path = (source_run_dir or self.config.source_run_dir) / "stackup.json"
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

    def _prepare_bo_handoff(self, run_dir: Path, agents_inputs_dir: Path, source_run_dir: Path) -> Dict[str, Any]:
        geometry_json = self._latest_geometry_json(agents_inputs_dir)
        stackup_path = source_run_dir / "stackup.json"
        adapter_paths = convert_geometry_engine_to_bo(
            geometry_json_path=geometry_json,
            output_dir=agents_inputs_dir,
            stackup_path=stackup_path if stackup_path.exists() else None,
            connect_port=True,
            include_primitive_analysis=True,
        )
        preparation = BayesianOptimizationAgentRunner(input_dir=agents_inputs_dir).prepare(
            output_root=run_dir / "02_bayesian_optimization",
            run_name=self.config.bo_run_name,
            max_evaluations=self.config.bo_max_evaluations,
            build_only=self.config.bo_build_only,
            instance_json=self.config.bo_instance_json,
            target_json=self.config.bo_target_json,
            simulation_f0_ghz=self.config.bo_f0_ghz,
            simulation_f1_ghz=self.config.bo_f1_ghz,
            optimizer_backend=self.config.optimizer_backend,
            enable_multistage_optimization=self.config.enable_multistage_optimization,
            execute=self.config.execute_bo,
        )
        return {
            "source_geometry_json": str(geometry_json.resolve()),
            "adapter_outputs": {key: str(path.resolve()) for key, path in adapter_paths.items()},
            "manifest_path": preparation.manifest_path,
            "ready_to_run": preparation.ready_to_run,
            "missing_inputs": preparation.missing_inputs,
            "bo_parameterization_summary": preparation.config_preview.get("bo_parameterization_summary"),
            "run_dir": preparation.config_preview.get("run_dir"),
            "execute_bo": self.config.execute_bo,
        }

    def _append_bo_effect_summary(self, agents_inputs_dir: Path, bo_handoff: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize a completed BO run and refresh shared history knowledge."""

        history_path = agents_inputs_dir / "history.json"
        history = load_history(history_path)
        bo_run_dir_value = bo_handoff.get("run_dir")
        if not bo_handoff.get("execute_bo") or not bo_run_dir_value:
            summary = {
                "available": False,
                "reason": "BO was not executed in this pipeline run.",
            }
            (agents_inputs_dir / "bo_effect_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            history["bo_effect_summary"] = summary
            refresh_history_knowledge(history)
            write_history(history_path, history)
            return summary

        bo_run_dir = Path(bo_run_dir_value)
        optimization_history_path = bo_run_dir / "optimization_history.json"
        if not optimization_history_path.exists():
            summary = {
                "available": False,
                "reason": "optimization_history.json was not found after BO execution.",
                "bo_run_dir": str(bo_run_dir.resolve()),
            }
            (agents_inputs_dir / "bo_effect_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            history["bo_effect_summary"] = summary
            refresh_history_knowledge(history)
            write_history(history_path, history)
            return summary

        optimization_history = load_json_object(optimization_history_path)
        target = self._load_target_markdown(agents_inputs_dir / "target.md")
        bo_parameterization_summary_path = agents_inputs_dir / "bo_parameterization_summary.json"
        bo_parameterization_summary = (
            load_json_object(bo_parameterization_summary_path) if bo_parameterization_summary_path.exists() else None
        )
        skill = LightweightDesignSkill(llm_client=self.llm_client)
        try:
            summary = skill.reflect_bo_effect(
                target=target,
                history=history,
                optimization_history=optimization_history,
                bo_parameterization_summary=bo_parameterization_summary,
            )
        except Exception as exc:
            summary = {
                "available": False,
                "reason": "LLM BO reflection failed.",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "bo_run_dir": str(bo_run_dir.resolve()),
            }

        (agents_inputs_dir / "bo_effect_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        history["bo_effect_summary"] = summary
        refresh_history_knowledge(history)
        write_history(history_path, history)
        return summary

    @staticmethod
    def _latest_geometry_json(agents_inputs_dir: Path) -> Path:
        summary_path = agents_inputs_dir / "geometry_summary.json"
        summary = load_json_object(summary_path)
        source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
        source_path = source.get("path")
        if source_path:
            path = Path(source_path)
            if not path.is_absolute():
                path = summary_path.parent / path
            if path.exists():
                return path.resolve()
        raise FileNotFoundError(
            "Could not resolve latest Geometry Engine JSON from {0}".format(summary_path)
        )

    @staticmethod
    def _load_target_markdown(target_path: Path) -> Dict[str, Any]:
        return {
            "format": "markdown",
            "source": str(target_path.resolve()),
            "content": target_path.read_text(encoding="utf-8"),
        }

    def _write_manifest(self, run_dir: Path, stage_outputs: Dict[str, Any]) -> Path:
        manifest_path = run_dir / "pipeline_manifest.json"
        payload = {
            "schema_version": "design_agent_top_level_pipeline_manifest_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": self._config_payload(),
            "stage_outputs": stage_outputs,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest_path

    def _config_payload(self) -> Dict[str, Any]:
        payload = asdict(self.config)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value.resolve())
        return payload

    @staticmethod
    def _next_run_index(run_root: Path, run_prefix: str) -> int:
        pattern = re.compile(r"^{0}_(\d+)$".format(re.escape(run_prefix)))
        indexes: List[int] = []
        for path in run_root.iterdir():
            if not path.is_dir():
                continue
            match = pattern.match(path.name)
            if match:
                indexes.append(int(match.group(1)))
        return max(indexes, default=0) + 1

    @staticmethod
    def _normalize_config(config: DesignAgentPipelineConfig) -> DesignAgentPipelineConfig:
        return replace(
            config,
            run_root=Path(config.run_root),
            seed_input_dir=Path(config.seed_input_dir),
            source_run_dir=Path(config.source_run_dir),
            target_json=Path(config.target_json),
            config_path=Path(config.config_path),
            initial_geometry_json=(
                Path(config.initial_geometry_json) if config.initial_geometry_json is not None else None
            ),
            bo_instance_json=Path(config.bo_instance_json) if config.bo_instance_json is not None else None,
            bo_target_json=Path(config.bo_target_json),
        )


def _config_value(field_name: str, value: Any) -> Any:
    if field_name not in PATH_FIELDS or value in (None, ""):
        return value
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _target_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pipeline target values from project ``target.json``."""

    target = payload.get("target", {}) if isinstance(payload.get("target"), dict) else {}
    s11 = target.get("s11", {}) if isinstance(target.get("s11"), dict) else {}
    gain = target.get("gain", {}) if isinstance(target.get("gain"), dict) else {}
    bandwidth = target.get("bandwidth", {}) if isinstance(target.get("bandwidth"), dict) else {}
    return {
        "target_frequency_ghz": _number_or_none(target.get("f0_ghz")),
        "target_s11": _number_or_none(s11.get("threshold_db")),
        "target_gain": _number_or_none(gain.get("threshold_dbi")),
        "target_bandwidth": _number_or_none(
            bandwidth.get("threshold_ghz")
            if bandwidth.get("threshold_ghz") is not None
            else bandwidth.get("threshold")
        ),
        "s11_threshold": _number_or_none(s11.get("threshold_db")),
        "target_bandwidth_reference_db": _number_or_none(bandwidth.get("reference_db")),
    }


def _target_markdown(payload: Dict[str, Any]) -> str:
    """Render project ``target.json`` as the Markdown prompt input expected downstream."""

    values = _target_values(payload)
    experiment = payload.get("experiment", "antenna_design")
    lines = [
        "# Design Target",
        "",
        "Source: target.json",
        "",
        "- Experiment: {0}".format(experiment),
        "- Center frequency: {0} GHz".format(_display_value(values.get("target_frequency_ghz"))),
        "- S11 target: less than {0} dB".format(_display_value(values.get("target_s11"))),
        "- Gain target: greater than {0} dBi".format(_display_value(values.get("target_gain"))),
    ]
    bandwidth_reference = values.get("target_bandwidth_reference_db")
    if bandwidth_reference is not None:
        lines.append("- Bandwidth reference: {0} dB".format(_display_value(bandwidth_reference)))
    bandwidth = values.get("target_bandwidth")
    if bandwidth is not None:
        lines.append("- Bandwidth target: greater than {0} GHz".format(_display_value(bandwidth)))
    lines.extend(["", "```json", json.dumps(payload, indent=2, ensure_ascii=False), "```", ""])
    return "\n".join(lines)


def _number_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _display_value(value: Any) -> str:
    return "unspecified" if value is None else str(value)


def _value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def _center_patch_on_stackup(patch: Any, stackup_json_path: Path) -> None:
    if not stackup_json_path.exists():
        return
    stackup = load_json_object(stackup_json_path)
    substrate = stackup.get("substrate", {}) if isinstance(stackup.get("substrate"), dict) else {}
    width = _number_or_none(substrate.get("width"))
    length = _number_or_none(substrate.get("length"))
    if width is None or length is None:
        return
    feed_direction = str(getattr(patch.feed, "direction", "bottom"))
    if feed_direction in {"top", "bottom"}:
        feed_offset = float(patch.feed.x) - float(patch.center_x)
    else:
        feed_offset = float(patch.feed.y) - float(patch.center_y)
    dx = width / 2.0 - float(patch.center_x)
    dy = length / 2.0 - float(patch.center_y)
    if not (math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9)):
        patch.translate(dx=dx, dy=dy)
    _reattach_feed_to_stackup_edge(
        patch=patch,
        substrate_width=width,
        substrate_length=length,
        feed_direction=feed_direction,
        feed_offset=feed_offset,
    )


def _reattach_feed_to_stackup_edge(
    *,
    patch: Any,
    substrate_width: float,
    substrate_length: float,
    feed_direction: str,
    feed_offset: float,
) -> None:
    if feed_direction in {"top", "bottom"}:
        patch.feed.x = float(patch.center_x) + feed_offset
    else:
        patch.feed.y = float(patch.center_y) + feed_offset
    patch.attach_feed_to_edge(feed_direction)
    if feed_direction == "bottom":
        patch.feed.length = max(0.0, float(patch.bottom))
    elif feed_direction == "top":
        patch.feed.length = max(0.0, substrate_length - float(patch.top))
    elif feed_direction == "left":
        patch.feed.length = max(0.0, float(patch.left))
    elif feed_direction == "right":
        patch.feed.length = max(0.0, substrate_width - float(patch.right))
    patch.rebuild_model()


def _ensure_stackup_clearance(
    patch: Any,
    stackup_json_path: Path,
    side_margin_mm: float = 8.0,
    top_bottom_margin_mm: float = 8.0,
) -> None:
    """Ensure substrate/ground are larger than the initial radiating patch."""

    if not stackup_json_path.exists():
        return
    stackup = load_json_object(stackup_json_path)
    substrate = stackup.get("substrate", {}) if isinstance(stackup.get("substrate"), dict) else {}
    ground = stackup.get("ground", {}) if isinstance(stackup.get("ground"), dict) else {}
    current_width = _number_or_none(substrate.get("width")) or 0.0
    current_length = _number_or_none(substrate.get("length")) or 0.0
    feed_margin = max(float(top_bottom_margin_mm), float(getattr(patch.feed, "length", 0.0)))
    required_width = float(patch.width) + 2.0 * float(side_margin_mm)
    required_length = float(patch.length) + 2.0 * feed_margin
    new_width = max(current_width, required_width)
    new_length = max(current_length, required_length)
    if math.isclose(new_width, current_width, abs_tol=1e-9) and math.isclose(
        new_length,
        current_length,
        abs_tol=1e-9,
    ):
        return

    substrate["width"] = new_width
    substrate["length"] = new_length
    stackup["substrate"] = substrate
    if isinstance(ground, dict):
        ground["width"] = max(_number_or_none(ground.get("width")) or 0.0, new_width)
        ground["length"] = max(_number_or_none(ground.get("length")) or 0.0, new_length)
        stackup["ground"] = ground
    stackup_json_path.write_text(json.dumps(stackup, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "DEFAULT_AGENT_CONFIG_PATH",
    "DesignAgentPipelineConfig",
    "DesignAgentPipelineResult",
    "DesignAgentPipelineRunner",
    "load_pipeline_config",
]
