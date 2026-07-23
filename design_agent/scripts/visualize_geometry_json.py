"""Visualize Geometry Engine JSON as a parameterized vector drawing.

The script reads the standardized ``geometry_engine_geometry_v1`` JSON and
draws planar conductor outer boundaries plus holes. It does not depend on
CadQuery or CST.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


Point2D = Tuple[float, float]


def load_geometry_json(path: Path) -> Dict[str, Any]:
    """Load and validate the top-level geometry JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    if payload.get("schema_version") != "geometry_engine_geometry_v1":
        raise ValueError(f"Unsupported schema_version: {payload.get('schema_version')!r}")
    return payload


def loop_vertices(loop: Dict[str, Any]) -> List[Point2D]:
    """Extract XY vertices from one boundary loop."""

    raw_vertices = loop.get("vertices")
    if not isinstance(raw_vertices, list) or len(raw_vertices) < 3:
        raise ValueError(f"Boundary loop {loop.get('id')!r} must contain at least 3 vertices.")
    vertices: List[Point2D] = []
    for vertex in raw_vertices:
        if not isinstance(vertex, dict):
            raise ValueError(f"Invalid vertex: {vertex!r}")
        vertices.append((float(vertex["x"]), float(vertex["y"])))
    return vertices


def closed_polygon(vertices: Sequence[Point2D]) -> List[Point2D]:
    """Return vertices with an explicit closing vertex for plotting."""

    points = list(vertices)
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def collect_bounds(geometries: Sequence[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    """Compute plotting bounds for all geometries."""

    all_points: List[Point2D] = []
    for geometry in geometries:
        all_points.extend(loop_vertices(geometry["outer_boundary"]))
        for hole in geometry.get("holes", []):
            all_points.extend(loop_vertices(hole))
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    return min(xs), min(ys), max(xs), max(ys)


def visualize_geometry_json(
    geometry_json_path: Path,
    output_path: Path,
    show_vertices: bool = True,
    show_labels: bool = True,
    max_vertex_markers: int = 64,
    max_vertex_labels: int = 16,
    dpi: int = 200,
) -> Path:
    """Render a Geometry JSON file to SVG/PNG/PDF using Matplotlib."""

    import matplotlib

    if output_path.suffix.lower() not in {".svg", ".png", ".pdf"}:
        raise ValueError("Output path must end with .svg, .png, or .pdf.")
    if output_path.suffix.lower() != ".png":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    payload = load_geometry_json(geometry_json_path)
    geometries = payload.get("geometries")
    if not isinstance(geometries, list) or not geometries:
        raise ValueError("Geometry JSON must contain non-empty geometries.")

    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    conductor_color = "#d4a017"
    conductor_edge = "#1f2937"
    hole_color = "#ffffff"
    hole_edge = "#b91c1c"
    vertex_color = "#2563eb"

    for geometry_index, geometry in enumerate(geometries, start=1):
        outer = closed_polygon(loop_vertices(geometry["outer_boundary"]))
        outer_path = MplPath(outer)
        ax.add_patch(
            PathPatch(
                outer_path,
                facecolor=conductor_color,
                edgecolor=conductor_edge,
                linewidth=1.4,
                alpha=0.72,
                label="conductor" if geometry_index == 1 else None,
            )
        )
        for hole_index, hole in enumerate(geometry.get("holes", []), start=1):
            hole_points = closed_polygon(loop_vertices(hole))
            ax.add_patch(
                PathPatch(
                    MplPath(hole_points),
                    facecolor=hole_color,
                    edgecolor=hole_edge,
                    linewidth=1.2,
                    hatch="///",
                    label="slot / hole" if geometry_index == 1 and hole_index == 1 else None,
                )
            )

        if show_vertices:
            plot_vertices(
                ax,
                outer[:-1],
                vertex_color,
                prefix=f"G{geometry_index}:O",
                show_labels=show_labels,
                max_markers=max_vertex_markers,
                max_labels=max_vertex_labels,
            )
            for hole_index, hole in enumerate(geometry.get("holes", []), start=1):
                plot_vertices(
                    ax,
                    loop_vertices(hole),
                    hole_edge,
                    prefix=f"G{geometry_index}:H{hole_index}",
                    show_labels=show_labels,
                    max_markers=max_vertex_markers,
                    max_labels=max_vertex_labels,
                )

        feed = geometry.get("metadata", {}).get("feed")
        if isinstance(feed, dict):
            try:
                ax.scatter([float(feed["x"])], [float(feed["y"])], marker="x", s=70, c="#111827", label="feed")
                if show_labels:
                    ax.annotate(
                        "feed",
                        (float(feed["x"]), float(feed["y"])),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )
            except (KeyError, TypeError, ValueError):
                pass

    min_x, min_y, max_x, max_y = collect_bounds(geometries)
    margin = max(max_x - min_x, max_y - min_y, 1.0) * 0.12
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.set_title(f"Geometry JSON: {geometry_json_path.name}")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_vertices(
    ax: Any,
    vertices: Sequence[Point2D],
    color: str,
    prefix: str,
    show_labels: bool,
    max_markers: int,
    max_labels: int,
) -> None:
    """Plot vertex markers and optional index labels."""

    if not vertices:
        return
    marker_indices = sampled_indices(len(vertices), max_markers)
    marker_vertices = [vertices[index] for index in marker_indices]
    xs = [point[0] for point in marker_vertices]
    ys = [point[1] for point in marker_vertices]
    ax.scatter(xs, ys, s=14, c=color, zorder=4)
    if show_labels:
        for index in sampled_indices(len(vertices), max_labels):
            x, y = vertices[index]
            ax.annotate(
                f"{prefix}{index}",
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                color=color,
            )


def sampled_indices(count: int, maximum: int) -> List[int]:
    """Return evenly sampled vertex indices for drawing dense loops."""

    if count <= 0 or maximum <= 0:
        return []
    if count <= maximum:
        return list(range(count))
    step = max(1, count // maximum)
    indices = list(range(0, count, step))
    return indices[:maximum]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Visualize Geometry Engine JSON as SVG/PNG/PDF.")
    parser.add_argument("--geometry-json", required=True, help="Path to geometry_engine_geometry_v1 JSON.")
    parser.add_argument("--output", required=True, help="Output path ending with .svg, .png, or .pdf.")
    parser.add_argument("--hide-vertices", action="store_true", help="Do not draw vertex markers.")
    parser.add_argument("--hide-labels", action="store_true", help="Do not draw vertex labels.")
    parser.add_argument("--max-vertex-markers", type=int, default=64, help="Maximum vertex markers per loop.")
    parser.add_argument("--max-vertex-labels", type=int, default=16, help="Maximum vertex labels per loop.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI used for PNG/PDF output.")
    return parser


def main() -> None:
    """CLI entry point."""

    args = build_arg_parser().parse_args()
    output = visualize_geometry_json(
        geometry_json_path=Path(args.geometry_json),
        output_path=Path(args.output),
        show_vertices=not args.hide_vertices,
        show_labels=not args.hide_labels,
        max_vertex_markers=args.max_vertex_markers,
        max_vertex_labels=args.max_vertex_labels,
        dpi=args.dpi,
    )
    print(f"[GeometryVisualizer] exported: {output}")


if __name__ == "__main__":
    main()
