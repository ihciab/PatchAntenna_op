"""Import helpers for initializing Geometry Engine objects from existing JSON.

The Geometry Engine owns Patch/Feed/Slot objects internally. This module is a
small bridge for existing design-agent patch JSON and CST adapter JSON so DSL
commands can be tested against previous runs without making the engine operate
on JSON directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geometry_engine.geometry.feed import Feed
from geometry_engine.geometry.patch import Patch


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class ParameterizationImportError(ValueError):
    """Raised when existing JSON cannot be mapped to a v1 Patch object."""


class ParameterizationImporter:
    """Create a rectangular Patch object from design-agent JSON artifacts."""

    tolerance: float

    def __init__(self, tolerance: float = 1e-7) -> None:
        """Initialize the importer."""

        self.tolerance = float(tolerance)

    def from_file(self, path: str | Path) -> Patch:
        """Load a patch or parameterization JSON file and return a Patch object."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ParameterizationImportError(f"Expected JSON object: {path}")
        return self.from_dict(payload)

    def from_run_dir(self, run_dir: str | Path) -> Patch:
        """Load ``patch.json`` from a design-agent run directory."""

        return self.from_file(Path(run_dir) / "patch.json")

    def from_dict(self, payload: Dict[str, Any]) -> Patch:
        """Create a Patch object from a patch or parameterization JSON dictionary."""

        if isinstance(payload.get("conductor"), dict):
            return self.from_patch_dict(payload)

        return self.from_parameterization_dict(payload)

    def from_parameterization_dict(self, payload: Dict[str, Any]) -> Patch:
        """Create a Patch object from a parameterization JSON dictionary."""

        components = payload.get("components")
        if not isinstance(components, list) or not components:
            raise ParameterizationImportError("parameterization JSON must contain non-empty components.")
        return self._patch_from_components(components)

    def from_patch_dict(self, payload: Dict[str, Any]) -> Patch:
        """Create a Patch object from a design-agent patch.json dictionary."""

        conductor = payload.get("conductor")
        if not isinstance(conductor, dict):
            raise ParameterizationImportError("patch JSON must contain a conductor object.")
        components = conductor.get("components")
        if not isinstance(components, list) or not components:
            raise ParameterizationImportError("patch JSON conductor must contain non-empty components.")
        patch = self._patch_from_components(components)
        material = conductor.get("material")
        if material:
            patch.material = str(material)
        return patch

    def _patch_from_components(self, components: List[Any]) -> Patch:
        """Create a Patch object from patch/feed component dictionaries."""

        patch_component = self._find_component(components, "patch")
        if patch_component is None:
            patch_component = components[0] if isinstance(components[0], dict) else None
        if patch_component is None:
            raise ParameterizationImportError("Could not find patch component.")

        patch_bbox = self._component_bbox(patch_component)
        patch_z = self._component_z(patch_component)
        patch = Patch(
            width=patch_bbox[2] - patch_bbox[0],
            length=patch_bbox[3] - patch_bbox[1],
            center_x=(patch_bbox[0] + patch_bbox[2]) / 2.0,
            center_y=(patch_bbox[1] + patch_bbox[3]) / 2.0,
            z=patch_z,
            material=self._component_material(patch_component),
        )

        feed_component = self._find_component(components, "feed")
        if feed_component is not None:
            patch.feed = self._feed_from_component(patch_bbox, self._component_bbox(feed_component))
        else:
            patch.attach_feed_to_edge(patch.feed.direction)

        patch.sync_feed_direction()
        return patch

    def _feed_from_component(self, patch_bbox: BBox, feed_bbox: BBox) -> Feed:
        """Infer an edge feed from a rectangular feed component bbox."""

        px1, py1, px2, py2 = patch_bbox
        fx1, fy1, fx2, fy2 = feed_bbox

        if abs(fy2 - py1) <= self.tolerance:
            overlap = self._overlap(fx1, fx2, px1, px2)
            return Feed(
                x=(overlap[0] + overlap[1]) / 2.0,
                y=py1,
                width=overlap[1] - overlap[0],
                length=max(0.0, py1 - fy1),
                direction="bottom",
            )
        if abs(fy1 - py1) <= self.tolerance and fx2 > px1 and fx1 < px2:
            overlap = self._overlap(fx1, fx2, px1, px2)
            return Feed(
                x=(overlap[0] + overlap[1]) / 2.0,
                y=py1,
                width=overlap[1] - overlap[0],
                length=max(0.0, py1 - fy1),
                direction="bottom",
            )
        if abs(fy1 - py2) <= self.tolerance:
            overlap = self._overlap(fx1, fx2, px1, px2)
            return Feed(
                x=(overlap[0] + overlap[1]) / 2.0,
                y=py2,
                width=overlap[1] - overlap[0],
                length=max(0.0, fy2 - py2),
                direction="top",
            )
        if abs(fy2 - py2) <= self.tolerance and fx2 > px1 and fx1 < px2:
            overlap = self._overlap(fx1, fx2, px1, px2)
            return Feed(
                x=(overlap[0] + overlap[1]) / 2.0,
                y=py2,
                width=overlap[1] - overlap[0],
                length=max(0.0, fy2 - py2),
                direction="top",
            )
        if abs(fx2 - px1) <= self.tolerance:
            overlap = self._overlap(fy1, fy2, py1, py2)
            return Feed(
                x=px1,
                y=(overlap[0] + overlap[1]) / 2.0,
                width=overlap[1] - overlap[0],
                length=max(0.0, px1 - fx1),
                direction="left",
            )
        if abs(fx1 - px1) <= self.tolerance and fy2 > py1 and fy1 < py2:
            overlap = self._overlap(fy1, fy2, py1, py2)
            return Feed(
                x=px1,
                y=(overlap[0] + overlap[1]) / 2.0,
                width=overlap[1] - overlap[0],
                length=max(0.0, px1 - fx1),
                direction="left",
            )
        if abs(fx1 - px2) <= self.tolerance:
            overlap = self._overlap(fy1, fy2, py1, py2)
            return Feed(
                x=px2,
                y=(overlap[0] + overlap[1]) / 2.0,
                width=overlap[1] - overlap[0],
                length=max(0.0, fx2 - px2),
                direction="right",
            )
        if abs(fx2 - px2) <= self.tolerance and fy2 > py1 and fy1 < py2:
            overlap = self._overlap(fy1, fy2, py1, py2)
            return Feed(
                x=px2,
                y=(overlap[0] + overlap[1]) / 2.0,
                width=overlap[1] - overlap[0],
                length=max(0.0, fx2 - px2),
                direction="right",
            )

        raise ParameterizationImportError(
            "Feed component must share a non-zero edge segment with the patch bbox. "
            f"patch_bbox={patch_bbox}, feed_bbox={feed_bbox}"
        )

    def _find_component(self, components: List[Any], role_keyword: str) -> Optional[Dict[str, Any]]:
        """Find a component by primitive role or metadata role keyword."""

        keyword = role_keyword.lower()
        for component in components:
            if not isinstance(component, dict):
                continue
            if keyword in str(component.get("name", "")).lower():
                return component
            if keyword in str(component.get("role", "")).lower():
                return component
            metadata = component.get("metadata", {})
            if isinstance(metadata, dict) and keyword in str(metadata.get("role", "")).lower():
                return component
            for primitive in component.get("primitives", []):
                if isinstance(primitive, dict) and keyword in str(primitive.get("role", "")).lower():
                    return component
        return None

    def _component_bbox(self, component: Dict[str, Any]) -> BBox:
        """Return a component bbox, deriving it from points if necessary."""

        bbox = component.get("bbox")
        if isinstance(bbox, Sequence) and len(bbox) >= 4:
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

        raw_points = (
            component.get("points")
            or component.get("resampled_points")
            or component.get("fallback_points")
            or component.get("polygon_vertices")
        )
        if not isinstance(raw_points, list) or not raw_points:
            raise ParameterizationImportError(f"Component has no bbox or points: {component}")
        points = [self._point(point) for point in raw_points]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def _component_z(self, component: Dict[str, Any]) -> float:
        """Infer component z coordinate from polygon vertices when available."""

        raw_points = component.get("polygon_vertices") or component.get("points") or component.get("resampled_points")
        if isinstance(raw_points, list):
            for point in raw_points:
                if isinstance(point, dict) and point.get("z") is not None:
                    return float(point["z"])
        return 0.0

    @staticmethod
    def _component_material(component: Dict[str, Any]) -> str:
        """Infer component material from primitives or metadata."""

        if component.get("material"):
            return str(component["material"])
        metadata = component.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("material"):
            return str(metadata["material"])
        for primitive in component.get("primitives", []):
            if isinstance(primitive, dict) and primitive.get("material"):
                return str(primitive["material"])
        return "PEC"

    @staticmethod
    def _point(value: Any) -> Point:
        """Normalize a JSON point."""

        if isinstance(value, dict):
            return float(value["x"]), float(value["y"])
        if isinstance(value, Sequence) and len(value) >= 2:
            return float(value[0]), float(value[1])
        raise ParameterizationImportError(f"Invalid point: {value!r}")

    @staticmethod
    def _overlap(a1: float, a2: float, b1: float, b2: float) -> Tuple[float, float]:
        """Return positive interval overlap or raise."""

        left = max(a1, b1)
        right = min(a2, b2)
        if right <= left:
            raise ParameterizationImportError(f"Expected positive overlap, got ({left}, {right}).")
        return left, right
