from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry_graph_parameterizer import GraphBasedLocalSplineParameterizer


def _parameterizer(tmp_path: Path) -> GraphBasedLocalSplineParameterizer:
    return GraphBasedLocalSplineParameterizer(
        image_path=tmp_path / "dummy.png",
        save_dir=tmp_path,
        line_tolerance_px=1.2,
        arc_tolerance_px=1.2,
        min_segment_points=6,
        min_component_length_px=2.0,
    )


def test_graph_local_preserves_right_angle_corner(tmp_path: Path) -> None:
    p = _parameterizer(tmp_path)
    points = np.array(
        [[x, 0.0] for x in range(0, 31)]
        + [[30.0, y] for y in range(1, 31)],
        dtype=np.float64,
    )
    records = [{"source_contour_id": "corner", "points": points, "closed": False, "path_length_px": 60.0}]

    graph = p.extract_topology_graph(records)
    split = p.split_graph_edges_by_geometry(graph)
    components = p._fit_graph_components(split)
    validation = p.validate_graph_topology(split, components)

    assert validation["valid"]
    assert len(split["edges"]) >= 2
    assert any(node["type"] == "corner" for node in split["nodes"])
    assert all(component["primitives"][0]["type"] == "line" for component in components)


def test_graph_local_detects_branch_junction(tmp_path: Path) -> None:
    p = _parameterizer(tmp_path)
    horizontal = np.array([[x, 10.0] for x in range(0, 31)], dtype=np.float64)
    vertical = np.array([[15.0, y] for y in range(10, 31)], dtype=np.float64)
    records = [
        {"source_contour_id": "h", "points": horizontal, "closed": False, "path_length_px": 30.0},
        {"source_contour_id": "v", "points": vertical, "closed": False, "path_length_px": 20.0},
    ]

    graph = p.extract_topology_graph(records)
    split = p.split_graph_edges_by_geometry(graph)
    validation = p.validate_graph_topology(split, p._fit_graph_components(split))

    junctions = [node for node in split["nodes"] if node["type"] == "junction"]
    assert validation["valid"]
    assert junctions
    assert max(node["degree"] for node in junctions) >= 3


def test_graph_local_keeps_close_parallel_lines_separate(tmp_path: Path) -> None:
    p = _parameterizer(tmp_path)
    records = [
        {
            "source_contour_id": "a",
            "points": np.array([[x, 5.0] for x in range(0, 41)], dtype=np.float64),
            "closed": False,
            "path_length_px": 40.0,
        },
        {
            "source_contour_id": "b",
            "points": np.array([[x, 9.0] for x in range(0, 41)], dtype=np.float64),
            "closed": False,
            "path_length_px": 40.0,
        },
    ]

    graph = p.extract_topology_graph(records)
    split = p.split_graph_edges_by_geometry(graph)
    components = p._fit_graph_components(split)

    assert len(graph["edges"]) == 2
    assert len(components) == 2
    assert all(component["primitives"][0]["type"] == "line" for component in components)


def test_graph_local_loop_anchor_and_arc_fit(tmp_path: Path) -> None:
    p = _parameterizer(tmp_path)
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=True)
    points = np.column_stack([25.0 + 10.0 * np.cos(theta), 25.0 + 10.0 * np.sin(theta)])
    records = [{"source_contour_id": "loop", "points": points, "closed": True, "path_length_px": 2.0 * np.pi * 10.0}]

    graph = p.extract_topology_graph(records)
    split = p.split_graph_edges_by_geometry(graph)
    components = p._fit_graph_components(split)
    validation = p.validate_graph_topology(split, components)

    assert validation["valid"]
    assert any(node["type"] == "loop_anchor" for node in split["nodes"])
    assert components
    assert all(component["primitives"][0]["source_edge_id"] for component in components)
