"""Public entry point for the LLM antenna design agent."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from design_agent.llm.client import LLMClient, OpenAICompatibleLLMClient
from design_agent.llm.parser import LLMResponseParser
from design_agent.memory import DesignMemory
from design_agent.models import DesignSpecification
from design_agent.planner import DesignPlanner
from design_agent.skills.initialize_geometry import INITIAL_DESIGN_PROMPT, InitializeGeometrySkill
from design_agent.skills.lightweight_design import LightweightDesignSkill
from design_agent.skills.redesign import OPTIMIZATION_PROMPT, RedesignSkill
from design_agent.skills.reflect_design import REFLECTION_PROMPT, ReflectDesignSkill
from design_agent.skills.select_topology import TOPOLOGY_SELECTION_PROMPT, SelectTopologySkill
from design_agent.state import DesignState
from design_agent.tools.geometry_summary import default_geometry_summary_path
from design_agent.tools.simulation_summary import default_simulation_summary_path
from design_agent.tools.bo_parameterization_summary import default_bo_parameterization_summary_path
from design_agent.tools.bayesian_optimization_runner import (
    BayesianOptimizationAgentRunner,
    BayesianOptimizationPreparation,
)
from design_agent.workflow import DesignWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_INPUTS_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"

PROMPT_TEMPLATES = {
    "initial_design": INITIAL_DESIGN_PROMPT,
    "initialize_geometry": INITIAL_DESIGN_PROMPT,
    "topology_selection": TOPOLOGY_SELECTION_PROMPT,
    "select_topology": TOPOLOGY_SELECTION_PROMPT,
    "reflection": REFLECTION_PROMPT,
    "reflect_design": REFLECTION_PROMPT,
    "optimization": OPTIMIZATION_PROMPT,
    "redesign": OPTIMIZATION_PROMPT,
}

PROMPT_ALIASES = {
    "initialize_geometry": "initial_design",
    "select_topology": "topology_selection",
    "reflect_design": "reflection",
    "redesign": "optimization",
}


class DesignAgent:
    """Single public entry point for LLM-guided antenna design.

    The agent owns the workflow, shared memory, and top-level lifecycle. Concrete
    topology selection, geometry generation, simulation, evaluation, and
    optimization logic live behind skills and tools.
    """

    def __init__(
        self,
        workflow: Optional[DesignWorkflow] = None,
        memory: Optional[DesignMemory] = None,
        planner: Optional[DesignPlanner] = None,
        llm_client: Optional[LLMClient] = None,
        config_path: Optional[Path | str] = None,
    ) -> None:
        """Initialize the design agent with injectable dependencies."""

        self.memory = memory or DesignMemory()
        self.planner = planner or DesignPlanner()
        self.llm_client = llm_client
        self.config_path = Path(config_path) if config_path is not None else PROJECT_ROOT / "config.json"
        self.workflow = workflow or self._build_workflow()

    @classmethod
    def from_config_file(cls, config_path: Path | str = PROJECT_ROOT / "config.json") -> "DesignAgent":
        """Create a design agent with the OpenAI-compatible client from config."""

        client = OpenAICompatibleLLMClient.from_config_file(str(config_path))
        return cls(llm_client=client, config_path=config_path)

    def design(
        self,
        specification: DesignSpecification,
        max_iterations: int = 1,
    ) -> DesignState:
        """Run an antenna design workflow from a target specification.

        Args:
            specification: Target antenna design requirements.
            max_iterations: Maximum number of design iterations to execute.

        Returns:
            Final shared design state.
        """

        state = DesignState(specification=specification, max_iterations=max_iterations)
        state.add_event("agent_started", {"max_iterations": max_iterations})
        return self.workflow.run(state)

    def list_prompt_skills(self) -> List[str]:
        """Return the prompt-backed skills currently available through the agent."""

        return ["initial_design", "topology_selection", "reflection", "optimization", "lightweight_design"]

    def generate_operations_from_files(
        self,
        target_path: Path | str = AGENT_INPUTS_DIR / "target.md",
        geometry_summary_path: Path | str = default_geometry_summary_path(),
        simulation_summary_path: Path | str = default_simulation_summary_path(),
        history_path: Path | str = AGENT_INPUTS_DIR / "history.json",
        bo_parameterization_summary_path: Optional[Path | str] = default_bo_parameterization_summary_path(),
    ) -> Dict[str, Any]:
        """Generate geometry modification operations from target Markdown and summaries."""

        skill = LightweightDesignSkill(llm_client=self._get_llm_client())
        return skill.run_from_files(
            target_path=target_path,
            geometry_summary_path=geometry_summary_path,
            simulation_summary_path=simulation_summary_path,
            history_path=history_path,
            bo_parameterization_summary_path=bo_parameterization_summary_path,
        )

    def generate_operations(self, inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate geometry modification operations from four summarized JSON objects."""

        skill = LightweightDesignSkill(llm_client=self._get_llm_client())
        return skill.run(inputs)

    def run_closed_loop(
        self,
        iterations: int = 1,
        input_dir: Path | str = PROJECT_ROOT / "design_agent_runs" / "agents_inputs",
        output_dir: Path | str = PROJECT_ROOT / "design_agent_runs" / "schem",
        source_run_dir: Path | str = PROJECT_ROOT / "design_agent_runs" / "initial_design_test",
        geometry_only: bool = False,
        build_only: bool = False,
        close_project: bool = False,
        f0_ghz: Optional[float] = None,
        f1_ghz: Optional[float] = None,
        target_frequency_ghz: float = 2.45,
        target_s11: float = -15.0,
        target_gain: float = 6.0,
        target_bandwidth: Optional[float] = None,
        s11_threshold: Optional[float] = None,
        gain: Optional[float] = None,
        initial_geometry_json: Optional[Path | str] = None,
        max_geometry_repair_attempts: int = 5,
    ) -> None:
        """Run the project-level lightweight design closed loop."""

        from design_agent.closed_loop import ClosedLoopDesignRunner

        runner = ClosedLoopDesignRunner(
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            source_run_dir=Path(source_run_dir),
            config_path=self.config_path,
            iterations=iterations,
            build_only=build_only,
            geometry_only=geometry_only,
            close_project=close_project,
            f0_ghz=f0_ghz,
            f1_ghz=f1_ghz,
            target_frequency_ghz=target_frequency_ghz,
            target_s11=target_s11,
            target_gain=target_gain,
            target_bandwidth=target_bandwidth,
            s11_threshold=s11_threshold,
            gain=gain,
            initial_geometry_json=Path(initial_geometry_json) if initial_geometry_json else None,
            llm_client=self._get_llm_client(),
            max_geometry_repair_attempts=max_geometry_repair_attempts,
        )
        runner.run()

    def run_pipeline(self, **kwargs: Any) -> Any:
        """Run the top-level numbered-folder design pipeline."""

        from design_agent.pipeline import DesignAgentPipelineConfig, DesignAgentPipelineRunner

        config = DesignAgentPipelineConfig(**kwargs)
        return DesignAgentPipelineRunner(config=config, llm_client=self._get_llm_client()).run()

    def run_pipeline_from_config(
        self,
        agent_config_path: Path | str = PROJECT_ROOT / "design_agent" / "agent_config.json",
    ) -> Any:
        """Run the top-level design pipeline from ``agent_config.json``."""

        from design_agent.pipeline import DesignAgentPipelineRunner, load_pipeline_config

        config = load_pipeline_config(agent_config_path)
        return DesignAgentPipelineRunner(config=config, llm_client=self._get_llm_client()).run()

    def prepare_bayesian_optimization_from_files(
        self,
        input_dir: Path | str = AGENT_INPUTS_DIR,
        **kwargs: Any,
    ) -> BayesianOptimizationPreparation:
        """Prepare a BO handoff manifest from ``design_agent_runs/agents_inputs``.

        This does not start CST or the optimizer.  It validates the shared JSON
        files, infers target frequency/S11 where available, and records open
        placeholders for the still-undecided variable function and variable
        selection strategy.
        """

        return BayesianOptimizationAgentRunner(input_dir=input_dir).prepare(execute=False, **kwargs)

    def run_bayesian_optimization_from_files(
        self,
        input_dir: Path | str = AGENT_INPUTS_DIR,
        **kwargs: Any,
    ) -> BayesianOptimizationPreparation:
        """Run the existing Bayesian optimization pipeline from agent inputs.

        Runtime parameters that are not safe to infer, such as evaluation count,
        build-vs-solver mode, and solver frequency range, should be supplied by
        the caller.  The returned preparation object includes the BO run
        directory when execution succeeds.
        """

        return BayesianOptimizationAgentRunner(input_dir=input_dir).prepare(execute=True, **kwargs)

    def run_prompt(
        self,
        prompt_name: str = "initial_design",
        specification: Optional[Dict[str, Any] | DesignSpecification] = None,
        output_dir: Optional[Path | str] = None,
        extra_instruction: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        parse_json: bool = True,
    ) -> List[Path]:
        """Run one prompt-backed skill and save the raw and parsed artifacts.

        This is the package-level version of the old script flow:
        select prompt, append optional specification/instructions, call the LLM,
        parse JSON objects, and save the files under ``design_agent_runs``.
        """

        canonical_name = self._canonical_prompt_name(prompt_name)
        prompt = self.build_prompt(canonical_name, specification, extra_instruction)
        target_dir = Path(output_dir).resolve() if output_dir else self.default_output_dir(canonical_name)
        response = self._get_llm_client().generate(
            prompt,
            context={
                "system_prompt": "You are a deterministic RF antenna design agent. Return machine-readable output.",
                "temperature": temperature,
                "timeout": timeout,
            },
        )

        objects: Optional[List[Dict[str, Any]]] = None
        if parse_json:
            objects = LLMResponseParser().parse_json_objects(response)
        return self.save_prompt_response(target_dir, canonical_name, response, objects)

    def run_initial_design_test(
        self,
        output_dir: Optional[Path | str] = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> List[Path]:
        """Run the fixed initial-design smoke test and save three JSON artifacts."""

        test_specification = {
            "center_frequency_hz": 2.45e9,
            "bandwidth_hz": 100e6,
            "max_size_mm": [80.0, 80.0, 3.0],
            "substrate": "Rogers RT5880",
            "performance_targets": {
                "s11_db_max_at_center": -10.0,
                "target_gain_dbi_min": 5.0,
            },
        }
        return self.run_prompt(
            prompt_name="initial_design",
            specification=test_specification,
            output_dir=output_dir or PROJECT_ROOT / "design_agent_runs" / "initial_design_test",
            temperature=temperature,
            timeout=timeout,
            parse_json=True,
        )

    def build_prompt(
        self,
        prompt_name: str,
        specification: Optional[Dict[str, Any] | DesignSpecification] = None,
        extra_instruction: Optional[str] = None,
    ) -> str:
        """Build the final LLM prompt from an embedded skill prompt template."""

        canonical_name = self._canonical_prompt_name(prompt_name)
        prompt = PROMPT_TEMPLATES[canonical_name]
        if specification is not None:
            prompt += (
                "\n\n==================================================\n"
                "USER SPECIFICATION\n"
                "==================================================\n"
                + json.dumps(self._specification_to_dict(specification), indent=2, ensure_ascii=False)
            )
        if extra_instruction:
            prompt += (
                "\n\n==================================================\n"
                "EXTRA USER INSTRUCTION\n"
                "==================================================\n"
                + extra_instruction
            )
        return prompt

    def default_output_dir(self, prompt_name: str) -> Path:
        """Return the default artifact directory for a prompt-backed skill."""

        canonical_name = self._canonical_prompt_name(prompt_name)
        if canonical_name == "initial_design":
            return PROJECT_ROOT / "design_agent_runs" / "initial_design_test"
        return PROJECT_ROOT / "design_agent_runs" / "{0}_test".format(canonical_name)

    def save_prompt_response(
        self,
        output_dir: Path | str,
        prompt_name: str,
        raw_response: str,
        objects: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Path]:
        """Save raw LLM response and optional parsed JSON objects."""

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []

        raw_path = target_dir / "raw_llm_response.txt"
        raw_path.write_text(raw_response, encoding="utf-8")
        saved_paths.append(raw_path)

        if objects:
            for filename, content in zip(self._artifact_names_for_prompt(prompt_name, objects), objects):
                path = target_dir / filename
                path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
                saved_paths.append(path)
        return saved_paths

    def _build_workflow(self) -> DesignWorkflow:
        if self.llm_client is None:
            return DesignWorkflow(memory=self.memory, planner=self.planner)
        return DesignWorkflow(
            memory=self.memory,
            planner=self.planner,
            topology_skill=SelectTopologySkill(llm_client=self.llm_client),
            initialization_skill=InitializeGeometrySkill(llm_client=self.llm_client),
            reflection_skill=ReflectDesignSkill(llm_client=self.llm_client),
            redesign_skill=RedesignSkill(llm_client=self.llm_client),
        )

    def _get_llm_client(self) -> LLMClient:
        if self.llm_client is None:
            self.llm_client = OpenAICompatibleLLMClient.from_config_file(str(self.config_path))
        return self.llm_client

    @staticmethod
    def _canonical_prompt_name(prompt_name: str) -> str:
        normalized = Path(prompt_name).stem
        normalized = PROMPT_ALIASES.get(normalized, normalized)
        if normalized not in PROMPT_TEMPLATES:
            raise FileNotFoundError("Prompt-backed skill not found: {0}".format(prompt_name))
        return normalized

    @staticmethod
    def _specification_to_dict(specification: Dict[str, Any] | DesignSpecification) -> Dict[str, Any]:
        if isinstance(specification, DesignSpecification):
            return asdict(specification)
        return dict(specification)

    def _artifact_names_for_prompt(self, prompt_name: str, objects: List[Dict[str, Any]]) -> List[str]:
        canonical_name = self._canonical_prompt_name(prompt_name)
        if canonical_name == "initial_design" and len(objects) >= 3:
            return ["design_trace.json", "stackup.json", "patch.json"]
        return ["response_{0:02d}.json".format(index + 1) for index in range(len(objects))]
