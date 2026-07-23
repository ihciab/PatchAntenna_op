"""CadQuery geometry construction DSL commands."""

from __future__ import annotations

from typing import Sequence, Tuple

from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class RectangleCommand(GeometryCommand):
    """Replace the current conductor with a CadQuery rectangle."""

    dsl_name = "Rectangle"

    def __init__(self, width: float, height: float, x: float = 0.0, y: float = 0.0) -> None:
        """Create a rectangle construction command."""

        self.width = float(width)
        self.height = float(height)
        self.x = float(x)
        self.y = float(y)

    def execute(self, context: GeometryContext) -> None:
        """Construct the rectangle through CadQuery."""

        context.patch.set_rectangle(width=self.width, height=self.height, x=self.x, y=self.y)


class CircleCommand(GeometryCommand):
    """Replace the current conductor with a CadQuery circle."""

    dsl_name = "Circle"

    def __init__(self, radius: float, x: float = 0.0, y: float = 0.0) -> None:
        """Create a circle construction command."""

        self.radius = float(radius)
        self.x = float(x)
        self.y = float(y)

    def execute(self, context: GeometryContext) -> None:
        """Construct the circle through CadQuery."""

        context.patch.set_circle(radius=self.radius, x=self.x, y=self.y)


class PolygonCommand(GeometryCommand):
    """Replace the current conductor with a CadQuery polygon."""

    dsl_name = "Polygon"

    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        """Create a polygon construction command."""

        self.points = [(float(point[0]), float(point[1])) for point in points]

    def execute(self, context: GeometryContext) -> None:
        """Construct the polygon through CadQuery."""

        context.patch.set_polygon(self.points)
