"""Geometry builder tool interface."""

from __future__ import annotations

from design_agent.models import GeometryCandidate
from design_agent.state import DesignState


class GeometryBuilder:
    """Convert parameterized design data into simulator-ready geometry."""

    def build(self, geometry: GeometryCandidate, state: DesignState) -> GeometryCandidate:
        """Build or validate geometry for the active simulator backend."""

        raise NotImplementedError("Geometry building is not implemented yet.")
