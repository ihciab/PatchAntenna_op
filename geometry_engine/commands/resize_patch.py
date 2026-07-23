"""ResizePatch DSL command."""

from __future__ import annotations

from typing import Optional

from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class ResizePatchCommand(GeometryCommand):
    """Resize the active patch while preserving its center."""

    dsl_name = "ResizePatch"

    length: Optional[float]
    width: Optional[float]

    def __init__(self, length: Optional[float] = None, width: Optional[float] = None) -> None:
        """Create a resize command."""

        self.length = None if length is None else float(length)
        self.width = None if width is None else float(width)

    def execute(self, context: GeometryContext) -> None:
        """Apply patch size changes."""

        context.patch.resize(length=self.length, width=self.width)
        context.patch.sync_feed_direction()
