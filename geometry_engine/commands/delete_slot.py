"""DeleteSlot DSL command."""

from __future__ import annotations

from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class DeleteSlotCommand(GeometryCommand):
    """Delete a slot by id."""

    dsl_name = "DeleteSlot"

    id: str

    def __init__(self, id: str) -> None:
        """Create a delete-slot command."""

        self.id = str(id)

    def execute(self, context: GeometryContext) -> bool:
        """Delete the requested slot and return whether it existed."""

        return context.patch.delete_slot(self.id)
