"""CadQuery-backed transform DSL commands."""

from __future__ import annotations

from typing import Optional

from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class TranslateCommand(GeometryCommand):
    """Translate the current CadQuery geometry."""

    dsl_name = "Translate"

    def __init__(self, dx: float = 0.0, dy: float = 0.0) -> None:
        """Create a translate command."""

        self.dx = float(dx)
        self.dy = float(dy)

    def execute(self, context: GeometryContext) -> None:
        """Apply translation to the current patch geometry."""

        context.patch.translate(dx=self.dx, dy=self.dy)


class RotateCommand(GeometryCommand):
    """Rotate the current CadQuery geometry in the XY plane."""

    dsl_name = "Rotate"

    def __init__(
        self,
        angle: float,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
    ) -> None:
        """Create a rotate command."""

        self.angle = float(angle)
        self.center_x = None if center_x is None else float(center_x)
        self.center_y = None if center_y is None else float(center_y)

    def execute(self, context: GeometryContext) -> None:
        """Apply rotation to the current patch geometry."""

        context.patch.rotate(self.angle, center_x=self.center_x, center_y=self.center_y)


class ScaleCommand(GeometryCommand):
    """Scale the current CadQuery geometry."""

    dsl_name = "Scale"

    def __init__(
        self,
        factor: float,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
    ) -> None:
        """Create a scale command."""

        self.factor = float(factor)
        self.center_x = None if center_x is None else float(center_x)
        self.center_y = None if center_y is None else float(center_y)

    def execute(self, context: GeometryContext) -> None:
        """Apply scaling to the current patch geometry."""

        context.patch.scale(self.factor, center_x=self.center_x, center_y=self.center_y)
