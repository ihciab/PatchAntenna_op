"""CST simulation tool interface."""

from __future__ import annotations

from design_agent.models import GeometryCandidate, SimulationResult
from design_agent.state import DesignState


class CSTTool:
    """Adapter for CST project generation, execution, and result extraction."""

    def simulate(self, geometry: GeometryCandidate, state: DesignState) -> SimulationResult:
        """Run CST or a compatible simulator for a geometry candidate."""

        raise NotImplementedError("CST simulation is not implemented yet.")
