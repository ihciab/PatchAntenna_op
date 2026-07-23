"""CadQuery backend for Geometry Engine construction and boundary extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

from geometry_engine.boundary import BoundaryLoop, GeometryBoundary, Point2D, distance, remove_duplicate_vertices


def require_cadquery() -> Any:
    """Import CadQuery or raise a helpful runtime error."""

    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError(
            "CadQuery is required for geometry_engine. Activate the paper environment "
            "or install cadquery before running Geometry Engine."
        ) from exc
    return cq


@dataclass
class CadQueryPlanarModel:
    """Thin-solid CadQuery model used for 2D conductor operations."""

    workplane: Any
    thickness: float = 0.001

    @classmethod
    def rectangle(
        cls,
        width: float,
        height: float,
        center_x: float,
        center_y: float,
        z: float = 0.0,
        thickness: float = 0.001,
    ) -> "CadQueryPlanarModel":
        """Create a rectangular planar conductor model."""

        cq = require_cadquery()
        workplane = cq.Workplane("XY").box(float(width), float(height), float(thickness)).translate(
            (float(center_x), float(center_y), float(z))
        )
        return cls(workplane=workplane, thickness=float(thickness))

    @classmethod
    def circle(
        cls,
        radius: float,
        center_x: float,
        center_y: float,
        z: float = 0.0,
        thickness: float = 0.001,
    ) -> "CadQueryPlanarModel":
        """Create a circular planar conductor model."""

        cq = require_cadquery()
        workplane = cq.Workplane("XY").circle(float(radius)).extrude(float(thickness)).translate(
            (float(center_x), float(center_y), float(z) - float(thickness) / 2.0)
        )
        return cls(workplane=workplane, thickness=float(thickness))

    @classmethod
    def polygon(
        cls,
        points: Sequence[Point2D],
        z: float = 0.0,
        thickness: float = 0.001,
    ) -> "CadQueryPlanarModel":
        """Create a polygonal planar conductor model."""

        if len(points) < 3:
            raise ValueError("Polygon requires at least 3 points.")
        cq = require_cadquery()
        workplane = cq.Workplane("XY").polyline([(float(x), float(y)) for x, y in points]).close().extrude(
            float(thickness)
        )
        workplane = workplane.translate((0.0, 0.0, float(z) - float(thickness) / 2.0))
        return cls(workplane=workplane, thickness=float(thickness))

    def copy(self) -> "CadQueryPlanarModel":
        """Return a shallow wrapper copy around the current CadQuery workplane."""

        return CadQueryPlanarModel(workplane=self.workplane, thickness=self.thickness)

    def boolean_union(self, tool: "CadQueryPlanarModel") -> None:
        """Union this model with another CadQuery planar model."""

        self.workplane = self.workplane.union(tool.workplane).clean()

    def boolean_difference(self, tool: "CadQueryPlanarModel") -> None:
        """Subtract another CadQuery planar model from this model."""

        self.workplane = self.workplane.cut(tool.workplane).clean()

    def translate(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """Translate the CadQuery model."""

        self.workplane = self.workplane.translate((float(dx), float(dy), float(dz)))

    def rotate_z(self, angle_degrees: float, center_x: float = 0.0, center_y: float = 0.0) -> None:
        """Rotate the CadQuery model around the Z axis."""

        self.workplane = self.workplane.rotate(
            (float(center_x), float(center_y), 0.0),
            (float(center_x), float(center_y), 1.0),
            float(angle_degrees),
        )

    def scale(self, factor: float) -> None:
        """Scale the CadQuery model around the global origin."""

        cq = require_cadquery()
        scaled = self.workplane.val().scale(float(factor))
        self.workplane = cq.Workplane("XY").newObject([scaled])

    def mirror_x(self, axis_y: float = 0.0) -> None:
        """Mirror the CadQuery model across a horizontal line y=axis_y."""

        self.translate(dy=-float(axis_y))
        self.workplane = self.workplane.mirror("XZ")
        self.translate(dy=float(axis_y))

    def mirror_y(self, axis_x: float = 0.0) -> None:
        """Mirror the CadQuery model across a vertical line x=axis_x."""

        self.translate(dx=-float(axis_x))
        self.workplane = self.workplane.mirror("YZ")
        self.translate(dx=float(axis_x))


class CadQueryBoundaryExtractor:
    """Extract kernel-independent 2D loops from a CadQuery planar model."""

    tolerance: float
    curve_samples: int

    def __init__(self, tolerance: float = 1e-7, curve_samples: int = 96) -> None:
        """Initialize the extractor."""

        self.tolerance = float(tolerance)
        self.curve_samples = int(curve_samples)

    def extract(self, model: CadQueryPlanarModel, geometry_id: str = "conductor") -> GeometryBoundary:
        """Extract a GeometryBoundary from the top face of a CadQuery model."""

        solid = model.workplane.val()
        faces = list(solid.Faces())
        if not faces:
            raise ValueError("CadQuery model does not contain any faces.")
        top_face = max(faces, key=lambda face: (face.Center().z, face.Area()))
        loops = self._extract_face_loops(top_face)
        if not loops:
            raise ValueError("No boundary loops could be extracted from CadQuery face.")
        loops.sort(key=lambda loop: abs(loop.signed_area()), reverse=True)
        outer = loops[0]
        outer.id = "outer"
        outer.role = "outer"
        holes = loops[1:]
        for index, hole in enumerate(holes, start=1):
            hole.id = f"hole_{index:03d}"
            hole.role = "hole"
        boundary = GeometryBoundary(
            id=geometry_id,
            unit="mm",
            plane="XY",
            outer=outer,
            holes=holes,
            metadata={"source_kernel": "CadQuery"},
        )
        boundary.normalize()
        return boundary

    def _extract_face_loops(self, face: Any) -> List[BoundaryLoop]:
        """Extract all loops from a CadQuery face."""

        loops: List[BoundaryLoop] = []
        for index, wire in enumerate(face.Wires()):
            vertices = self._ordered_wire_vertices(wire)
            vertices = remove_duplicate_vertices(vertices, self.tolerance)
            if len(vertices) >= 3:
                loops.append(BoundaryLoop(id=f"loop_{index:03d}", role="unknown", vertices=vertices))
        return loops

    def _ordered_wire_vertices(self, wire: Any) -> List[Point2D]:
        """Return ordered vertices sampled from a CadQuery wire."""

        segments = [self._edge_points(edge) for edge in wire.Edges()]
        if not segments:
            return []
        if len(segments) == 1 and distance(segments[0][0], segments[0][-1]) <= self.tolerance:
            return segments[0]

        ordered = segments.pop(0)
        while segments:
            last = ordered[-1]
            match_index = -1
            reverse = False
            for index, segment in enumerate(segments):
                if distance(last, segment[0]) <= self.tolerance:
                    match_index = index
                    reverse = False
                    break
                if distance(last, segment[-1]) <= self.tolerance:
                    match_index = index
                    reverse = True
                    break
            if match_index < 0:
                ordered.extend(segments.pop(0))
                continue
            segment = segments.pop(match_index)
            if reverse:
                segment = list(reversed(segment))
            ordered.extend(segment[1:])
        return ordered

    def _edge_points(self, edge: Any) -> List[Point2D]:
        """Sample ordered 2D points from one CadQuery edge."""

        geom_type = str(edge.geomType()).upper()
        if geom_type == "LINE":
            return [self._vector_to_point(edge.startPoint()), self._vector_to_point(edge.endPoint())]

        sample_count = self.curve_samples
        try:
            sampled_vectors, _params = edge.sample(sample_count)
        except Exception:
            sampled_vectors = [edge.startPoint(), edge.endPoint()]
        points = [self._vector_to_point(vector) for vector in sampled_vectors]
        if len(points) >= 2 and distance(points[0], points[-1]) <= self.tolerance:
            points.pop()
        return points

    @staticmethod
    def _vector_to_point(vector: Any) -> Point2D:
        """Convert a CadQuery vector to an XY point."""

        return (float(vector.x), float(vector.y))
