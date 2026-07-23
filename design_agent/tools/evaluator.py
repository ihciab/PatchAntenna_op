"""Antenna performance evaluator interface."""

from __future__ import annotations

from design_agent.models import EvaluationResult, SimulationResult
from design_agent.state import DesignState


class AntennaEvaluator:
    """Evaluate simulated antenna performance against the target specification."""

    def evaluate(self, result: SimulationResult, state: DesignState) -> EvaluationResult:
        """Compute metrics and objective scores from simulation results."""

        raise NotImplementedError("Antenna evaluation is not implemented yet.")
