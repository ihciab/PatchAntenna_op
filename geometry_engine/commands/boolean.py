"""CadQuery boolean operation DSL commands."""

from __future__ import annotations

from typing import Sequence

from geometry_engine.cadquery_backend import CadQueryPlanarModel
from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class _BooleanBaseCommand(GeometryCommand):
    """Base class for CadQuery boolean commands."""

    shape: str
    kwargs: dict[str, object]

    def __init__(self, shape: str, **kwargs: object) -> None:
        """Create a boolean command."""

        self.shape = str(shape)
        self.kwargs = kwargs

    def _tool_model(self, context: GeometryContext, thickness: float) -> CadQueryPlanarModel:
        """Build the CadQuery boolean tool body."""

        shape = self.shape.lower()
        if shape == "rectangle":
            return CadQueryPlanarModel.rectangle(
                width=float(self.kwargs["width"]),
                height=float(self.kwargs["height"]),
                center_x=float(self.kwargs.get("x", context.patch.center_x)),
                center_y=float(self.kwargs.get("y", context.patch.center_y)),
                z=context.patch.z,
                thickness=thickness,
            )
        if shape == "circle":
            return CadQueryPlanarModel.circle(
                radius=float(self.kwargs["radius"]),
                center_x=float(self.kwargs.get("x", context.patch.center_x)),
                center_y=float(self.kwargs.get("y", context.patch.center_y)),
                z=context.patch.z,
                thickness=thickness,
            )
        if shape == "polygon":
            raw_points = self.kwargs["points"]
            if not isinstance(raw_points, Sequence):
                raise ValueError("Polygon boolean operation requires points=[(x, y), ...].")
            points = [(float(point[0]), float(point[1])) for point in raw_points]
            return CadQueryPlanarModel.polygon(points=points, z=context.patch.z, thickness=thickness)
        raise ValueError(f"Unsupported boolean shape: {self.shape!r}")


class BooleanUnionCommand(_BooleanBaseCommand):
    """Apply CadQuery boolean union."""

    dsl_name = "BooleanUnion"

    def execute(self, context: GeometryContext) -> None:
        """Apply union with the tool body."""

        context.patch.boolean_union(self._tool_model(context, thickness=context.patch.thickness))


class BooleanDifferenceCommand(_BooleanBaseCommand):
    """Apply CadQuery boolean difference."""

    dsl_name = "BooleanDifference"

    def execute(self, context: GeometryContext) -> None:
        """Apply difference with the tool body."""

        context.patch.boolean_difference(self._tool_model(context, thickness=context.patch.thickness * 3.0))
