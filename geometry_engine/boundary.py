"""Kernel-independent 2D boundary data structures and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Point2D = Tuple[float, float]


@dataclass
class BoundaryLoop:
    """One ordered closed 2D boundary loop without a duplicated closing vertex."""

    id: str
    role: str
    vertices: List[Point2D]

    def signed_area(self) -> float:
        """Return the signed area of the loop."""

        area = 0.0
        count = len(self.vertices)
        for index in range(count):
            x1, y1 = self.vertices[index]
            x2, y2 = self.vertices[(index + 1) % count]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def ensure_counter_clockwise(self) -> None:
        """Reverse vertices in-place when the loop orientation is clockwise."""

        if self.signed_area() < 0.0:
            self.vertices.reverse()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the loop to a JSON-compatible dictionary."""

        return {
            "id": self.id,
            "role": self.role,
            "closed": True,
            "orientation": "CCW",
            "vertices": [{"x": x, "y": y} for x, y in self.vertices],
        }


@dataclass
class GeometryBoundary:
    """Final simulator-independent planar conductor boundary."""

    id: str
    unit: str
    plane: str
    outer: BoundaryLoop
    holes: List[BoundaryLoop] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def loops(self) -> List[BoundaryLoop]:
        """Return all loops in export order."""

        return [self.outer] + list(self.holes)

    def normalize(self) -> None:
        """Normalize all boundary loops for export."""

        for loop in self.loops():
            loop.vertices = remove_duplicate_vertices(loop.vertices)
            loop.ensure_counter_clockwise()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the boundary to a JSON-compatible dictionary."""

        return {
            "id": self.id,
            "type": "planar_conductor",
            "unit": self.unit,
            "plane": self.plane,
            "outer_boundary": self.outer.to_dict(),
            "holes": [hole.to_dict() for hole in self.holes],
            "metadata": self.metadata,
        }


@dataclass
class BoundaryValidationResult:
    """Validation result for exported planar geometry."""

    valid: bool
    errors: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise a ValueError if the boundary is invalid."""

        if not self.valid:
            raise ValueError("Boundary validation failed: " + "; ".join(self.errors))


class BoundaryValidator:
    """Validate kernel-independent 2D geometry boundaries."""

    tolerance: float

    def __init__(self, tolerance: float = 1e-7) -> None:
        """Initialize the validator."""

        self.tolerance = float(tolerance)

    def validate(self, boundary: GeometryBoundary) -> BoundaryValidationResult:
        """Validate all loops in a boundary document."""

        errors: List[str] = []
        if boundary.unit != "mm":
            errors.append(f"Geometry unit must be mm, got {boundary.unit!r}.")
        for loop in boundary.loops():
            self._validate_loop(loop, errors)
        return BoundaryValidationResult(valid=not errors, errors=errors)

    def _validate_loop(self, loop: BoundaryLoop, errors: List[str]) -> None:
        """Validate one closed loop."""

        if len(loop.vertices) < 3:
            errors.append(f"Loop {loop.id} must contain at least 3 vertices.")
            return
        duplicates = duplicated_vertices(loop.vertices, self.tolerance)
        if duplicates:
            errors.append(f"Loop {loop.id} contains duplicated vertices: {duplicates[:5]}.")
        if abs(loop.signed_area()) <= self.tolerance:
            errors.append(f"Loop {loop.id} area is zero or too small.")
        if loop.signed_area() < -self.tolerance:
            errors.append(f"Loop {loop.id} must be counter-clockwise.")
        intersections = self_intersections(loop.vertices, self.tolerance)
        if intersections:
            errors.append(f"Loop {loop.id} self-intersects: {intersections[:5]}.")


def remove_duplicate_vertices(vertices: Sequence[Point2D], tolerance: float = 1e-9) -> List[Point2D]:
    """Remove consecutive and closing duplicate vertices."""

    cleaned: List[Point2D] = []
    for vertex in vertices:
        point = (float(vertex[0]), float(vertex[1]))
        if not cleaned or distance(cleaned[-1], point) > tolerance:
            cleaned.append(point)
    if len(cleaned) > 1 and distance(cleaned[0], cleaned[-1]) <= tolerance:
        cleaned.pop()
    return cleaned


def duplicated_vertices(vertices: Sequence[Point2D], tolerance: float) -> List[Dict[str, Any]]:
    """Return non-closing duplicated vertices."""

    seen: Dict[Tuple[int, int], int] = {}
    duplicates: List[Dict[str, Any]] = []
    scale = 1.0 / tolerance
    for index, point in enumerate(vertices):
        key = (round(point[0] * scale), round(point[1] * scale))
        if key in seen:
            duplicates.append({"first_index": seen[key], "duplicate_index": index, "point": point})
        else:
            seen[key] = index
    return duplicates


def self_intersections(vertices: Sequence[Point2D], tolerance: float) -> List[Dict[str, int]]:
    """Return self-intersections between non-adjacent loop segments."""

    intersections: List[Dict[str, int]] = []
    edge_count = len(vertices)
    for i in range(edge_count):
        a1 = vertices[i]
        a2 = vertices[(i + 1) % edge_count]
        for j in range(i + 1, edge_count):
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == edge_count - 1:
                continue
            b1 = vertices[j]
            b2 = vertices[(j + 1) % edge_count]
            if segments_intersect(a1, a2, b1, b2, tolerance):
                intersections.append({"edge_a": i, "edge_b": j})
    return intersections


def segments_intersect(a1: Point2D, a2: Point2D, b1: Point2D, b2: Point2D, tolerance: float) -> bool:
    """Return whether two line segments intersect."""

    def orient(p: Point2D, q: Point2D, r: Point2D) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: Point2D, q: Point2D, r: Point2D) -> bool:
        return (
            min(p[0], r[0]) - tolerance <= q[0] <= max(p[0], r[0]) + tolerance
            and min(p[1], r[1]) - tolerance <= q[1] <= max(p[1], r[1]) + tolerance
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    if o1 * o2 < -tolerance and o3 * o4 < -tolerance:
        return True
    if abs(o1) <= tolerance and on_segment(a1, b1, a2):
        return True
    if abs(o2) <= tolerance and on_segment(a1, b2, a2):
        return True
    if abs(o3) <= tolerance and on_segment(b1, a1, b2):
        return True
    if abs(o4) <= tolerance and on_segment(b1, a2, b2):
        return True
    return False


def distance(p1: Point2D, p2: Point2D) -> float:
    """Return the Euclidean distance between two points."""

    return ((p1[0] - p2[0]) ** 2.0 + (p1[1] - p2[1]) ** 2.0) ** 0.5


def bbox(vertices: Iterable[Point2D]) -> Tuple[float, float, float, float]:
    """Return the bounding box of a vertex sequence."""

    points = list(vertices)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)
