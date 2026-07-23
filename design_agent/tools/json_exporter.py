"""JSON export tool for design states and candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from design_agent.state import DesignState


class JSONExporter:
    """Export design state snapshots to JSON-compatible artifacts."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialize the exporter with an optional output directory."""

        self.output_dir = Path(output_dir) if output_dir is not None else Path("design_agent_runs")

    def export(self, state: DesignState) -> str:
        """Export the current design state and return the artifact path."""

        raise NotImplementedError("JSON export is not implemented yet.")
