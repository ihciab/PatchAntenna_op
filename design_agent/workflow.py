"""Workflow orchestration for the LLM design agent."""

from __future__ import annotations

from typing import Dict, Optional

from design_agent.memory import DesignMemory
from design_agent.planner import DesignPlanner
from design_agent.skills.initialize_geometry import InitializeGeometrySkill
from design_agent.skills.redesign import RedesignSkill
from design_agent.skills.reflect_design import ReflectDesignSkill
from design_agent.skills.select_topology import SelectTopologySkill
from design_agent.state import DesignState
from design_agent.tools.cst_tool import CSTTool
from design_agent.tools.evaluator import AntennaEvaluator
from design_agent.tools.geometry_builder import GeometryBuilder
from design_agent.tools.json_exporter import JSONExporter


class DesignWorkflow:
    """Coordinate design skills and tools without embedding business logic."""

    def __init__(
        self,
        memory: Optional[DesignMemory] = None,
        planner: Optional[DesignPlanner] = None,
        topology_skill: Optional[SelectTopologySkill] = None,
        initialization_skill: Optional[InitializeGeometrySkill] = None,
        reflection_skill: Optional[ReflectDesignSkill] = None,
        redesign_skill: Optional[RedesignSkill] = None,
        geometry_builder: Optional[GeometryBuilder] = None,
        cst_tool: Optional[CSTTool] = None,
        evaluator: Optional[AntennaEvaluator] = None,
        json_exporter: Optional[JSONExporter] = None,
    ) -> None:
        """Initialize workflow dependencies.

        Dependencies are injectable to keep the workflow decoupled from concrete
        LLM providers, simulator integrations, and optimization backends.
        """

        self.memory = memory or DesignMemory()
        self.planner = planner or DesignPlanner()
        self.topology_skill = topology_skill or SelectTopologySkill()
        self.initialization_skill = initialization_skill or InitializeGeometrySkill()
        self.reflection_skill = reflection_skill or ReflectDesignSkill()
        self.redesign_skill = redesign_skill or RedesignSkill()
        self.geometry_builder = geometry_builder or GeometryBuilder()
        self.cst_tool = cst_tool or CSTTool()
        self.evaluator = evaluator or AntennaEvaluator()
        self.json_exporter = json_exporter or JSONExporter()

        self._steps = self._build_step_registry()

    def run(self, state: DesignState) -> DesignState:
        """Execute the planned workflow for the supplied state."""

        plan = self.planner.create_plan(state)
        state.add_event("workflow_plan_created", {"steps": plan})

        while state.iteration < state.max_iterations:
            state.add_event("iteration_started", {"iteration": state.iteration})
            for step_name in plan:
                self.run_step(step_name, state)
            self.memory.add_record(state, note="Completed workflow iteration.")
            state.add_event("iteration_completed", {"iteration": state.iteration})
            state.iteration += 1

        state.add_event("workflow_completed", {"iterations": state.iteration})
        return state

    def run_step(self, step_name: str, state: DesignState) -> DesignState:
        """Run a named workflow step against the current state."""

        if step_name not in self._steps:
            raise ValueError("Unknown workflow step: {0}".format(step_name))
        state.add_event("step_started", {"step": step_name})
        self._steps[step_name](state)
        state.add_event("step_completed", {"step": step_name})
        return state

    def _build_step_registry(self) -> Dict[str, object]:
        """Create the internal mapping from step names to bound methods."""

        return {
            "select_topology": self._select_topology,
            "initialize_geometry": self._initialize_geometry,
            "build_geometry": self._build_geometry,
            "simulate": self._simulate,
            "evaluate": self._evaluate,
            "reflect": self._reflect,
            "redesign": self._redesign,
            "export_json": self._export_json,
        }

    def _select_topology(self, state: DesignState) -> None:
        """Delegate topology selection to the topology skill."""

        state.topology = self.topology_skill.run(state)

    def _initialize_geometry(self, state: DesignState) -> None:
        """Delegate initial geometry creation to the initialization skill."""

        state.current_geometry = self.initialization_skill.run(state)

    def _build_geometry(self, state: DesignState) -> None:
        """Delegate simulator-ready geometry construction to the geometry builder."""

        if state.current_geometry is None:
            raise ValueError("Cannot build geometry before initialization.")
        state.current_geometry = self.geometry_builder.build(state.current_geometry, state)

    def _simulate(self, state: DesignState) -> None:
        """Delegate simulation execution to the CST tool."""

        if state.current_geometry is None:
            raise ValueError("Cannot simulate before geometry is available.")
        state.simulation_result = self.cst_tool.simulate(state.current_geometry, state)

    def _evaluate(self, state: DesignState) -> None:
        """Delegate performance evaluation to the evaluator tool."""

        if state.simulation_result is None:
            raise ValueError("Cannot evaluate before simulation results are available.")
        state.evaluation_result = self.evaluator.evaluate(state.simulation_result, state)

    def _reflect(self, state: DesignState) -> None:
        """Delegate design reflection to the reflection skill."""

        suggestion = self.reflection_skill.run(state)
        state.suggestions.append(suggestion)

    def _redesign(self, state: DesignState) -> None:
        """Delegate design update to the redesign skill."""

        state.current_geometry = self.redesign_skill.run(state)

    def _export_json(self, state: DesignState) -> None:
        """Delegate JSON export to the JSON exporter tool."""

        artifact_path = self.json_exporter.export(state)
        state.artifacts["json_export"] = artifact_path
