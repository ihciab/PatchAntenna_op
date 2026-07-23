"""CadQuery-backed parametric patch model.

The semantic layer keeps patch/feed/slot parameters, while the actual conductor
geometry is constructed and mutated as a CadQuery model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from typing import Any, Dict, List, Optional

from geometry_engine.boundary import GeometryBoundary
from geometry_engine.cadquery_backend import CadQueryBoundaryExtractor, CadQueryPlanarModel
from geometry_engine.geometry.feed import Feed
from geometry_engine.geometry.slot import Slot


@dataclass
class Patch:
    """A CadQuery-backed parameterized patch antenna model in millimeters."""

    length: float = 24.0
    width: float = 28.0
    center_x: float = 25.0
    center_y: float = 25.0
    z: float = 0.0
    material: str = "PEC"
    layer: str = "top"
    feed: Feed = field(default_factory=lambda: Feed(x=25.0, y=13.0, width=3.0, length=0.0, direction="bottom"))
    slots: List[Slot] = field(default_factory=list)
    thickness: float = 0.001
    orientation_degrees: float = 0.0
    shape_kind: str = "rectangle"
    model: CadQueryPlanarModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the initial CadQuery geometry."""

        self.rebuild_model()

    def __deepcopy__(self, memo: Dict[int, Any]) -> "Patch":
        """Deep-copy semantic state and rebuild the CadQuery model."""

        copied = Patch(
            length=self.length,
            width=self.width,
            center_x=self.center_x,
            center_y=self.center_y,
            z=self.z,
            material=self.material,
            layer=self.layer,
            feed=copy.deepcopy(self.feed, memo),
            slots=copy.deepcopy(self.slots, memo),
            thickness=self.thickness,
            orientation_degrees=self.orientation_degrees,
            shape_kind=self.shape_kind,
        )
        memo[id(self)] = copied
        return copied

    @property
    def left(self) -> float:
        """Return the minimum x coordinate of the patch."""

        return self.center_x - self.width / 2.0

    @property
    def right(self) -> float:
        """Return the maximum x coordinate of the patch."""

        return self.center_x + self.width / 2.0

    @property
    def bottom(self) -> float:
        """Return the minimum y coordinate of the patch."""

        return self.center_y - self.length / 2.0

    @property
    def top(self) -> float:
        """Return the maximum y coordinate of the patch."""

        return self.center_y + self.length / 2.0

    def resize(self, length: Optional[float] = None, width: Optional[float] = None) -> None:
        """Resize the patch through CadQuery reconstruction while keeping its center fixed."""

        feed_direction = self.feed.direction
        feed_terminal = self.feed.terminal_point()
        if length is not None:
            self.length = float(length)
        if width is not None:
            self.width = float(width)
        self.attach_feed_to_edge(feed_direction)
        self._restore_feed_length_to_terminal(feed_direction, feed_terminal)
        self.rebuild_model()

    def add_slot(self, slot: Slot) -> None:
        """Add a slot cutout using CadQuery boolean difference."""

        self.slots.append(slot)
        self.model.boolean_difference(self._slot_model(slot))

    def delete_slot(self, slot_id: str) -> bool:
        """Delete a slot by id and rebuild CadQuery geometry."""

        original_count = len(self.slots)
        self.slots = [slot for slot in self.slots if slot.id != slot_id]
        deleted = len(self.slots) != original_count
        if deleted:
            self.rebuild_model()
        return deleted

    def slot_by_id(self, slot_id: str) -> Optional[Slot]:
        """Return a slot by id if present."""

        for slot in self.slots:
            if slot.id == slot_id:
                return slot
        return None

    def mirror_x(self) -> None:
        """Mirror geometry across the patch center X-axis using CadQuery."""

        self.model.mirror_x(self.center_y)
        for slot in self.slots:
            slot.mirror_x(self.center_y)
        self.feed.mirror_x(self.center_y)

    def mirror_y(self) -> None:
        """Mirror geometry across the patch center Y-axis using CadQuery."""

        self.model.mirror_y(self.center_x)
        for slot in self.slots:
            slot.mirror_y(self.center_x)
        self.feed.mirror_y(self.center_x)

    def translate(self, dx: float = 0.0, dy: float = 0.0) -> None:
        """Translate the CadQuery geometry and semantic handles."""

        self.model.translate(dx=dx, dy=dy)
        self.center_x += float(dx)
        self.center_y += float(dy)
        self.feed.move(dx=dx, dy=dy)
        for slot in self.slots:
            slot.x += float(dx)
            slot.y += float(dy)

    def rotate(self, angle_degrees: float, center_x: Optional[float] = None, center_y: Optional[float] = None) -> None:
        """Rotate the CadQuery geometry around a point."""

        origin_x = self.center_x if center_x is None else float(center_x)
        origin_y = self.center_y if center_y is None else float(center_y)
        self.model.rotate_z(angle_degrees, origin_x, origin_y)
        self.center_x, self.center_y = self._rotate_point(self.center_x, self.center_y, angle_degrees, origin_x, origin_y)
        self.feed.x, self.feed.y = self._rotate_point(self.feed.x, self.feed.y, angle_degrees, origin_x, origin_y)
        for slot in self.slots:
            slot.x, slot.y = self._rotate_point(slot.x, slot.y, angle_degrees, origin_x, origin_y)
        self.orientation_degrees = (self.orientation_degrees + float(angle_degrees)) % 360.0

    def scale(self, factor: float, center_x: Optional[float] = None, center_y: Optional[float] = None) -> None:
        """Scale the CadQuery geometry around a point."""

        scale_factor = float(factor)
        origin_x = self.center_x if center_x is None else float(center_x)
        origin_y = self.center_y if center_y is None else float(center_y)
        self.model.translate(dx=-origin_x, dy=-origin_y)
        self.model.scale(scale_factor)
        self.model.translate(dx=origin_x, dy=origin_y)
        self.width *= scale_factor
        self.length *= scale_factor
        self.center_x = origin_x + (self.center_x - origin_x) * scale_factor
        self.center_y = origin_y + (self.center_y - origin_y) * scale_factor
        self.feed.x = origin_x + (self.feed.x - origin_x) * scale_factor
        self.feed.y = origin_y + (self.feed.y - origin_y) * scale_factor
        self.feed.width *= abs(scale_factor)
        self.feed.length *= abs(scale_factor)
        for slot in self.slots:
            slot.x = origin_x + (slot.x - origin_x) * scale_factor
            slot.y = origin_y + (slot.y - origin_y) * scale_factor
            slot.width *= abs(scale_factor)
            slot.height *= abs(scale_factor)

    def set_rectangle(self, width: float, height: float, x: float, y: float) -> None:
        """Replace the current conductor with a CadQuery rectangle."""

        self.width = float(width)
        self.length = float(height)
        self.center_x = float(x)
        self.center_y = float(y)
        self.slots = []
        self.orientation_degrees = 0.0
        self.shape_kind = "rectangle"
        self.attach_feed_to_edge("bottom")
        self.rebuild_model()

    def set_circle(self, radius: float, x: float, y: float) -> None:
        """Replace the current conductor with a CadQuery circle."""

        diameter = 2.0 * float(radius)
        self.width = diameter
        self.length = diameter
        self.center_x = float(x)
        self.center_y = float(y)
        self.slots = []
        self.orientation_degrees = 0.0
        self.shape_kind = "circle"
        self.model = CadQueryPlanarModel.circle(radius=radius, center_x=x, center_y=y, z=self.z, thickness=self.thickness)
        self.attach_feed_to_edge("bottom")

    def set_polygon(self, points: List[tuple[float, float]]) -> None:
        """Replace the current conductor with a CadQuery polygon."""

        self.slots = []
        self.orientation_degrees = 0.0
        self.shape_kind = "polygon"
        self.model = CadQueryPlanarModel.polygon(points=points, z=self.z, thickness=self.thickness)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        self.width = max(xs) - min(xs)
        self.length = max(ys) - min(ys)
        self.center_x = (max(xs) + min(xs)) / 2.0
        self.center_y = (max(ys) + min(ys)) / 2.0
        self.attach_feed_to_edge("bottom")

    def boolean_union(self, tool: CadQueryPlanarModel) -> None:
        """Apply CadQuery boolean union to the conductor."""

        self.model.boolean_union(tool)

    def boolean_difference(self, tool: CadQueryPlanarModel) -> None:
        """Apply CadQuery boolean difference to the conductor."""

        self.model.boolean_difference(tool)

    def vertices(self) -> List[Dict[str, float]]:
        """Return the extracted outer boundary vertices from CadQuery geometry."""

        return [{"x": x, "y": y, "z": self.z} for x, y in self.boundary().outer.vertices]

    def boundary(self) -> GeometryBoundary:
        """Extract the final 2D conductor boundary from the CadQuery model."""

        return CadQueryBoundaryExtractor().extract(self.model, geometry_id="patch_conductor")

    def rebuild_model(self) -> None:
        """Rebuild the CadQuery model from semantic patch and slot parameters."""

        self.model = CadQueryPlanarModel.rectangle(
            width=self.width,
            height=self.length,
            center_x=self.center_x,
            center_y=self.center_y,
            z=self.z,
            thickness=self.thickness,
        )
        feed_model = self._feed_model()
        if feed_model is not None:
            self.model.boolean_union(feed_model)
        for slot in self.slots:
            self.model.boolean_difference(self._slot_model(slot))
        if abs(self.orientation_degrees) > 1e-9:
            self.model.rotate_z(self.orientation_degrees, self.center_x, self.center_y)

    def edge_direction_for_point(self, x: float, y: float, tolerance: float = 1e-7) -> Optional[str]:
        """Return the patch edge containing a point, or None if not on an edge."""

        if not self.is_axis_aligned(tolerance):
            return None
        on_horizontal_span = self.left - tolerance <= x <= self.right + tolerance
        on_vertical_span = self.bottom - tolerance <= y <= self.top + tolerance
        if abs(y - self.bottom) <= tolerance and on_horizontal_span:
            return "bottom"
        if abs(y - self.top) <= tolerance and on_horizontal_span:
            return "top"
        if abs(x - self.left) <= tolerance and on_vertical_span:
            return "left"
        if abs(x - self.right) <= tolerance and on_vertical_span:
            return "right"
        return None

    def sync_feed_direction(self) -> None:
        """Update feed direction from its current edge position when possible."""

        direction = self.edge_direction_for_point(self.feed.x, self.feed.y)
        if direction is not None:
            self.feed.direction = direction

    def attach_feed_to_edge(self, direction: str) -> None:
        """Attach the feed to a named patch edge and keep its span in bounds."""

        self.feed.direction = direction
        half_width = self.feed.width / 2.0
        if direction == "bottom":
            self.feed.y = self.bottom
            self.feed.x = min(max(self.feed.x, self.left + half_width), self.right - half_width)
        elif direction == "top":
            self.feed.y = self.top
            self.feed.x = min(max(self.feed.x, self.left + half_width), self.right - half_width)
        elif direction == "left":
            self.feed.x = self.left
            self.feed.y = min(max(self.feed.y, self.bottom + half_width), self.top - half_width)
        elif direction == "right":
            self.feed.x = self.right
            self.feed.y = min(max(self.feed.y, self.bottom + half_width), self.top - half_width)

    def _restore_feed_length_to_terminal(self, direction: str, terminal: Dict[str, float]) -> None:
        """Keep the feed entrance fixed when patch resizing moves the patch edge."""

        if direction == "bottom":
            self.feed.length = max(0.0, self.feed.y - float(terminal["y"]))
        elif direction == "top":
            self.feed.length = max(0.0, float(terminal["y"]) - self.feed.y)
        elif direction == "left":
            self.feed.length = max(0.0, self.feed.x - float(terminal["x"]))
        elif direction == "right":
            self.feed.length = max(0.0, float(terminal["x"]) - self.feed.x)

    def is_axis_aligned(self, tolerance: float = 1e-7) -> bool:
        """Return whether the semantic patch is still axis-aligned."""

        remainder = abs(self.orientation_degrees % 90.0)
        return remainder <= tolerance or abs(remainder - 90.0) <= tolerance

    def _slot_model(self, slot: Slot) -> CadQueryPlanarModel:
        """Create a CadQuery tool body for a slot cutout."""

        shape = slot.shape.lower()
        if shape == "rectangle":
            return CadQueryPlanarModel.rectangle(
                width=slot.width,
                height=slot.height,
                center_x=slot.x,
                center_y=slot.y,
                z=self.z,
                thickness=self.thickness * 3.0,
            )
        if shape == "circle":
            return CadQueryPlanarModel.circle(
                radius=slot.width / 2.0,
                center_x=slot.x,
                center_y=slot.y,
                z=self.z,
                thickness=self.thickness * 3.0,
            )
        raise ValueError(f"Unsupported slot shape for CadQuery construction: {slot.shape!r}")

    def _feed_model(self) -> Optional[CadQueryPlanarModel]:
        """Create the metal feed-line rectangle when a feed length is known."""

        feed_length = float(getattr(self.feed, "length", 0.0))
        if feed_length <= 0.0:
            return None

        if self.feed.direction == "bottom":
            width = self.feed.width
            height = feed_length
            center_x = self.feed.x
            center_y = self.feed.y - feed_length / 2.0
        elif self.feed.direction == "top":
            width = self.feed.width
            height = feed_length
            center_x = self.feed.x
            center_y = self.feed.y + feed_length / 2.0
        elif self.feed.direction == "left":
            width = feed_length
            height = self.feed.width
            center_x = self.feed.x - feed_length / 2.0
            center_y = self.feed.y
        elif self.feed.direction == "right":
            width = feed_length
            height = self.feed.width
            center_x = self.feed.x + feed_length / 2.0
            center_y = self.feed.y
        else:
            return None

        return CadQueryPlanarModel.rectangle(
            width=width,
            height=height,
            center_x=center_x,
            center_y=center_y,
            z=self.z,
            thickness=self.thickness,
        )

    @staticmethod
    def _rotate_point(
        x: float,
        y: float,
        angle_degrees: float,
        center_x: float,
        center_y: float,
    ) -> tuple[float, float]:
        """Rotate a semantic point to keep handles aligned with CadQuery geometry."""

        angle = math.radians(float(angle_degrees))
        dx = float(x) - float(center_x)
        dy = float(y) - float(center_y)
        return (
            float(center_x) + dx * math.cos(angle) - dy * math.sin(angle),
            float(center_y) + dx * math.sin(angle) + dy * math.cos(angle),
        )
