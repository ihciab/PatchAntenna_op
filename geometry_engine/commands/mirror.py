"""Mirror DSL commands."""

from __future__ import annotations

from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class MirrorXCommand(GeometryCommand):
    """Mirror feed and slots across the patch center X-axis."""

    dsl_name = "MirrorX"

    def execute(self, context: GeometryContext) -> None:
        """Apply the mirror operation."""

        # TODO: For non-rectangular outlines, mirror the full boundary and run
        # polygon self-intersection checks before accepting the result.
        context.patch.mirror_x()
        context.patch.sync_feed_direction()


class MirrorYCommand(GeometryCommand):
    """Mirror feed and slots across the patch center Y-axis."""

    dsl_name = "MirrorY"

    def execute(self, context: GeometryContext) -> None:
        """Apply the mirror operation."""

        # TODO: For non-rectangular outlines, mirror the full boundary and run
        # polygon self-intersection checks before accepting the result.
        context.patch.mirror_y()
        context.patch.sync_feed_direction()
