"""Slot geometry model.

Only rectangular slots are fully validated in the first version. The class
keeps a shape field so future commands can add richer slot families without
changing the engine core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Slot:
    """A slot cutout inside the patch, measured in millimeters."""

    id: str
    shape: str
    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        """Return the minimum x coordinate."""

        return self.x - self.width / 2.0

    @property
    def right(self) -> float:
        """Return the maximum x coordinate."""

        return self.x + self.width / 2.0

    @property
    def bottom(self) -> float:
        """Return the minimum y coordinate."""

        return self.y - self.height / 2.0

    @property
    def top(self) -> float:
        """Return the maximum y coordinate."""

        return self.y + self.height / 2.0

    def mirror_x(self, axis_y: float) -> None:
        """Mirror the slot across a horizontal axis."""

        self.y = 2.0 * float(axis_y) - self.y

    def mirror_y(self, axis_x: float) -> None:
        """Mirror the slot across a vertical axis."""

        self.x = 2.0 * float(axis_x) - self.x

    def vertices(self, z: float = 0.0) -> List[Dict[str, float]]:
        """Return closed polygon vertices for rectangular cutout export."""

        return [
            {"x": self.left, "y": self.bottom, "z": z},
            {"x": self.right, "y": self.bottom, "z": z},
            {"x": self.right, "y": self.top, "z": z},
            {"x": self.left, "y": self.top, "z": z},
            {"x": self.left, "y": self.bottom, "z": z},
        ]
