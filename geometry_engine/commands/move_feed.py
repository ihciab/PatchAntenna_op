"""MoveFeed DSL command."""

from __future__ import annotations

from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class MoveFeedCommand(GeometryCommand):
    """Move the feed point by dx and dy in millimeters."""

    dsl_name = "MoveFeed"

    dx: float
    dy: float

    def __init__(self, dx: float = 0.0, dy: float = 0.0) -> None:
        """Create a feed movement command."""

        self.dx = float(dx)
        self.dy = float(dy)

    def execute(self, context: GeometryContext) -> None:
        """Move the feed and update its edge direction when it remains valid."""

        context.patch.feed.move(dx=self.dx, dy=self.dy)
        context.patch.sync_feed_direction()
        context.patch.rebuild_model()
