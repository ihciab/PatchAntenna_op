"""Simulator-independent Geometry JSON exporter.

The exporter depends only on kernel-independent boundary data structures. It
does not import CadQuery and does not emit CST commands or simulation settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from geometry_engine.boundary import BoundaryValidator, GeometryBoundary
from geometry_engine.geometry.patch import Patch


class GeometryJSONExporter:
    """Export extracted 2D geometry boundaries to standardized JSON."""

    validator: BoundaryValidator

    def __init__(self, validator: BoundaryValidator | None = None) -> None:
        """Initialize the exporter."""

        self.validator = validator or BoundaryValidator()

    def boundary_to_dict(self, boundary: GeometryBoundary) -> Dict[str, Any]:
        """Convert a boundary object to the standardized Geometry JSON dictionary."""

        boundary.normalize()
        self.validator.validate(boundary).raise_if_invalid()
        return {
            "schema_version": "geometry_engine_geometry_v1",
            "generator": "geometry_engine_cadquery",
            "unit": "mm",
            "coordinate_system": {
                "plane": "XY",
                "x_axis": "right",
                "y_axis": "up",
                "orientation": "right_handed",
            },
            "geometries": [boundary.to_dict()],
            "export_rules": {
                "closed_boundary": True,
                "continuous_boundary": True,
                "ordered_vertices": True,
                "orientation": "counter_clockwise",
                "duplicated_vertices": False,
                "self_intersection": False,
            },
        }

    def to_dict(self, patch: Patch) -> Dict[str, Any]:
        """Extract boundary from a Patch and convert it to Geometry JSON."""

        boundary = patch.boundary()
        payload = self.boundary_to_dict(boundary)
        feed_terminal = patch.feed.terminal_point()
        feed_terminal_edge = patch.feed.terminal_edge_points()
        payload["geometries"][0]["metadata"].update(
            {
                "semantic_type": "patch_with_feed" if patch.feed.length > 0.0 else "patch_conductor",
                "layer": patch.layer,
                "feed": {
                    "type": "edge_feed",
                    "x": patch.feed.x,
                    "y": patch.feed.y,
                    "width": patch.feed.width,
                    "length": patch.feed.length,
                    "direction": patch.feed.direction,
                    "patch_edge": patch.feed.edge_points(),
                    "terminal_point": feed_terminal,
                    "terminal_edge": feed_terminal_edge,
                },
            }
        )
        return payload

    def export(self, patch: Patch, path: str | Path) -> Path:
        """Write Geometry JSON to disk."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict(patch)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path


PatchJSONExporter = GeometryJSONExporter
