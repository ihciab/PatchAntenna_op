"""AddSlot DSL command."""

from __future__ import annotations

from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand
from geometry_engine.geometry.slot import Slot


class AddSlotCommand(GeometryCommand):
    """Add a slot cutout to the patch."""

    dsl_name = "AddSlot"

    shape: str
    x: float
    y: float
    width: float
    height: float
    id: str | None

    def __init__(
        self,
        shape: str = "rectangle",
        x: float = 0.0,
        y: float = 0.0,
        width: float = 1.0,
        height: float = 1.0,
        id: str | None = None,
    ) -> None:
        """Create an add-slot command."""

        self.shape = str(shape).lower()
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(width) if self.shape == "circle" else float(height)
        self.id = None if id is None else str(id)

    def execute(self, context: GeometryContext) -> str:
        """Add the slot and return its id."""

        slot_id = self.id or context.next_slot_id()
        slot = Slot(
            id=slot_id,
            shape=self.shape,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
        )
        context.patch.add_slot(slot)
        return slot_id
