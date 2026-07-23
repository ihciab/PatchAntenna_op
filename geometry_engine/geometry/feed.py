"""Feed geometry model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Feed:
    """Patch-edge feed definition in millimeters."""

    x: float
    y: float
    width: float = 3.0
    length: float = 0.0
    direction: str = "bottom"

    def move(self, dx: float = 0.0, dy: float = 0.0) -> None:
        """Move the feed point by a delta in millimeters."""

        self.x += float(dx)
        self.y += float(dy)

    def mirror_x(self, axis_y: float) -> None:
        """Mirror the feed across a horizontal axis."""

        self.y = 2.0 * float(axis_y) - self.y
        if self.direction == "bottom":
            self.direction = "top"
        elif self.direction == "top":
            self.direction = "bottom"

    def mirror_y(self, axis_x: float) -> None:
        """Mirror the feed across a vertical axis."""

        self.x = 2.0 * float(axis_x) - self.x
        if self.direction == "left":
            self.direction = "right"
        elif self.direction == "right":
            self.direction = "left"

    def edge_points(self) -> List[Dict[str, float]]:
        """Return the two points that define the effective port edge."""

        half_width = self.width / 2.0
        if self.direction in {"top", "bottom"}:
            return [
                {"x": self.x - half_width, "y": self.y, "z": 0.0},
                {"x": self.x + half_width, "y": self.y, "z": 0.0},
            ]
        return [
            {"x": self.x, "y": self.y - half_width, "z": 0.0},
            {"x": self.x, "y": self.y + half_width, "z": 0.0},
        ]

    def terminal_point(self) -> Dict[str, float]:
        """Return the feed-line entrance point away from the patch edge."""

        length = max(0.0, float(self.length))
        if self.direction == "bottom":
            return {"x": self.x, "y": self.y - length, "z": 0.0}
        if self.direction == "top":
            return {"x": self.x, "y": self.y + length, "z": 0.0}
        if self.direction == "left":
            return {"x": self.x - length, "y": self.y, "z": 0.0}
        return {"x": self.x + length, "y": self.y, "z": 0.0}

    def terminal_edge_points(self) -> List[Dict[str, float]]:
        """Return the two points that define the feed-line entrance edge."""

        half_width = self.width / 2.0
        point = self.terminal_point()
        x = point["x"]
        y = point["y"]
        if self.direction in {"top", "bottom"}:
            return [
                {"x": x - half_width, "y": y, "z": 0.0},
                {"x": x + half_width, "y": y, "z": 0.0},
            ]
        return [
            {"x": x, "y": y - half_width, "z": 0.0},
            {"x": x, "y": y + half_width, "z": 0.0},
        ]
