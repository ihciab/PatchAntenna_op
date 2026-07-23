"""Planning interfaces for design workflow orchestration."""

from __future__ import annotations

from typing import List

from design_agent.state import DesignState


class DesignPlanner:
    """Create high-level workflow step plans from the current design state."""

    def create_plan(self, state: DesignState) -> List[str]:
        """Return an ordered list of workflow step names.

        Args:
            state: Shared design state.

        Returns:
            Names of workflow stages to execute.
        """

        return [
            "select_topology",
            "initialize_geometry",
            "build_geometry",
            "simulate",
            "evaluate",
            "reflect",
            "redesign",
        ]
