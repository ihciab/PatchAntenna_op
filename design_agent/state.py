"""Shared design state for the LLM design agent workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from design_agent.models import (
    AntennaTopology,
    DesignSpecification,
    EvaluationResult,
    GeometryCandidate,
    OptimizationSuggestion,
    SimulationResult,
)


@dataclass
class DesignState:
    """Mutable state passed through all workflow stages.

    Attributes:
        specification: Target antenna requirements for the current task.
        topology: Selected high-level antenna topology.
        current_geometry: Most recent geometry candidate.
        simulation_result: Most recent simulator output.
        evaluation_result: Most recent performance evaluation.
        suggestions: Pending redesign or optimization suggestions.
        iteration: Current design iteration index.
        max_iterations: Maximum number of workflow iterations.
        history: Lightweight event log for debugging and future persistence.
        artifacts: Paths or handles to generated files such as JSON exports,
            CST projects, plots, and reports.
        metadata: Extension point for future RAG, MCP, LangGraph, or optimizer
            integrations.
    """

    specification: DesignSpecification
    topology: Optional[AntennaTopology] = None
    current_geometry: Optional[GeometryCandidate] = None
    simulation_result: Optional[SimulationResult] = None
    evaluation_result: Optional[EvaluationResult] = None
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Append a structured event to the in-memory state history."""

        self.history.append(
            {
                "iteration": self.iteration,
                "event_type": event_type,
                "payload": payload or {},
            }
        )
