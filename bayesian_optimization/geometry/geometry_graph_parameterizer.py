from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bayesian_optimization.geometry.geometry_driven_parameterizer import GeometryDrivenParameterizer


Point = Tuple[float, float]


class GraphBasedLocalSplineParameterizer(GeometryDrivenParameterizer):
    """Topology-aware centerline graph parameterization.

    This mode deliberately avoids the old component-wide B-spline
    intermediate.  VTracer centerlines are converted into a graph, graph edges
    are split only at local geometric events, then each local edge is fitted in
    line > arc > spline order.
    """

    def __init__(
        self,
        image_path: Path | str,
        save_dir: Path | str,
        graph_node_merge_tolerance_px: float = 3.0,
        graph_endpoint_snap_tolerance_px: float = 4.0,
        graph_corner_angle_threshold_deg: float = 38.0,
        graph_curvature_split_percentile: float = 92.0,
        max_local_spline_rms_error_px: float = 2.0,
        max_local_spline_length_shrink_ratio: float = 0.04,
        **kwargs: Any,
    ):
        super().__init__(image_path=image_path, save_dir=save_dir, **kwargs)
        self.stage_dir = self.save_dir / "graph_local_primitives"
        self.edge_dir = self.stage_dir / "00_edges"
        self.vtracer_dir = self.stage_dir / "00_vtracer_centerline"
        self.contour_dir = self.stage_dir / "00_ordered_centerlines"
        self.bspline_dir = self.stage_dir / "00_no_global_bspline"
        self.graph_dir = self.stage_dir / "01_graph"
        self.edge_split_dir = self.stage_dir / "02_edge_split"
        self.local_fit_dir = self.stage_dir / "03_local_fit"
        self.primitive_dir = self.local_fit_dir
        self.validation_dir = self.stage_dir / "04_topology_validation"
        self.export_dir = self.stage_dir / "05_export"
        self.preview_dir = self.export_dir

        self.graph_node_merge_tolerance_px = float(graph_node_merge_tolerance_px)
        self.graph_endpoint_snap_tolerance_px = float(graph_endpoint_snap_tolerance_px)
        self.graph_corner_angle_threshold_deg = float(graph_corner_angle_threshold_deg)
        self.graph_curvature_split_percentile = float(graph_curvature_split_percentile)
        self.max_local_spline_rms_error_px = float(max_local_spline_rms_error_px)
        self.max_local_spline_length_shrink_ratio = float(max_local_spline_length_shrink_ratio)
        self.last_status = {
            "fallback": False,
            "fallback_reason": "",
            "actual_backend": "graph_local_primitives",
        }

    def run(self) -> Path:
        import cv2
        from Rebuild.NewParams import NewParams

        for path in (
            self.stage_dir,
            self.edge_dir,
            self.vtracer_dir,
            self.contour_dir,
            self.graph_dir,
            self.edge_split_dir,
            self.local_fit_dir,
            self.validation_dir,
            self.export_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        print("[GraphBasedLocalSplineParameterizer] stage 1/7: adaptive edge preprocessing")
        params = self._select_preprocessing_params(NewParams)
        raw_edges = params.edges()
        if self._should_use_solid_mask_topology(params=params, mask=raw_edges):
            raw_edges_path = self.edge_dir / "newparams_edge_or_stroke_mask.png"
            self._write_image(raw_edges_path, raw_edges)
            return self._write_solid_mask_topology_result(
                params=params,
                mask=raw_edges,
                raw_edges_path=raw_edges_path,
            )

        patch_topology = self.build_patch_topology_vtracer_input(
            params.original_img(),
            debug_dir=self.edge_dir / "patch_topology_debug",
        )
        self.edge_selection_diagnostics["patch_topology"] = patch_topology["metrics"]
        self._write_json(self.edge_dir / "edge_selection_diagnostics.json", self.edge_selection_diagnostics)
        if patch_topology["accepted"] and self._should_apply_patch_topology_override():
            from bayesian_optimization.geometry.geometry_driven_parameterizer import EDGE_REPRESENTATION_MODE

            params = self._EdgeCandidate(
                image_path=self.image_path,
                save_dir=self.edge_dir / "_candidate_patch_topology",
                edges=patch_topology["skeleton"],
                original_img=params.original_img(),
                edge_representation=EDGE_REPRESENTATION_MODE,
                edge_contour_tracing=False,
                verbose=True,
            )
            raw_edges = patch_topology["skeleton"]
        raw_edges_path = self.edge_dir / "newparams_edge_or_stroke_mask.png"
        self._write_image(raw_edges_path, raw_edges)

        print("[GraphBasedLocalSplineParameterizer] stage 2/7: VTracer centerline extraction")
        parameterizer = params.parameterize(save_dir=self.vtracer_dir)
        vtracer_results = parameterizer.results()
        if len(vtracer_results) > max(80, self.max_centerline_components_for_geometry):
            return self._write_standard_topology_fallback(
                params=params,
                parameterizer=parameterizer,
                reason=f"too_many_centerline_components_for_graph: {len(vtracer_results)}",
            )

        records = self._extract_vtracer_centerline_records(vtracer_results)
        if not records:
            raise ValueError("VTracer centerline extraction produced no ordered centerlines.")

        trace_image_path = Path(parameterizer.trace_image_path() or self.image_path)
        trace_img = cv2.imread(str(trace_image_path), cv2.IMREAD_COLOR)
        if trace_img is None:
            trace_img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if trace_img is None:
            raise FileNotFoundError(f"Cannot read trace image: {trace_image_path}")
        trace_copy_path = self.edge_dir / "vtracer_trace_input.png"
        self._write_image(trace_copy_path, trace_img)

        self._write_json(
            self.contour_dir / "ordered_centerlines_summary.json",
            {
                "source": "vtracer_python.centerline",
                "component_count": len(records),
                "records": [
                    {
                        "component_id": i + 1,
                        "source_component_id": item["source_contour_id"],
                        "point_count": int(len(item["points"])),
                        "closed": bool(item["closed"]),
                        "path_length_px": float(item["path_length_px"]),
                    }
                    for i, item in enumerate(records)
                ],
            },
        )

        print("[GraphBasedLocalSplineParameterizer] stage 3/7: topology graph extraction")
        graph = self.extract_topology_graph(records)
        self._write_json(self.graph_dir / "topology_graph.json", graph)
        self._write_graph_preview(self.graph_dir / "topology_graph.png", trace_img, graph, show_primitives=False)

        print("[GraphBasedLocalSplineParameterizer] stage 4/7: graph edge decomposition")
        split_graph = self.split_graph_edges_by_geometry(graph)
        self._write_json(self.edge_split_dir / "split_edges.json", split_graph)
        self._write_graph_preview(self.edge_split_dir / "split_edges_preview.png", trace_img, split_graph, show_primitives=False)

        print("[GraphBasedLocalSplineParameterizer] stage 5/7: local primitive fitting")
        components = self._fit_graph_components(split_graph)
        if not components:
            raise ValueError("Graph local primitive fitting produced no components.")

        validation = self.validate_graph_topology(split_graph, components)
        self._write_json(self.validation_dir / "topology_validation.json", validation)
        if not validation.get("valid", False):
            raise ValueError(f"graph topology validation failed: {validation.get('issues', [])}")

        print("[GraphBasedLocalSplineParameterizer] stage 6/7: graph-aware JSON export")
        aggregate_metrics = self._aggregate_metrics(components)
        aggregate_metrics.update(
            {
                "backend": "graph_local_primitives",
                "node_count": len(split_graph.get("nodes", [])),
                "edge_count": len(split_graph.get("edges", [])),
                "topology_preservation_score": validation.get("topology_preservation_score", 0.0),
            }
        )
        preview_svg_path = self.export_dir / "graph_primitives_preview.svg"
        preview_png_path = self.export_dir / "graph_primitives_preview.png"
        labels_json_path = self.export_dir / "primitive_labels.json"
        labels_csv_path = self.export_dir / "primitive_labels.csv"
        self._write_svg_preview(preview_svg_path, trace_img.shape[1], trace_img.shape[0], components)
        self._write_png_preview(preview_png_path, trace_img, components)
        self._write_graph_preview(self.local_fit_dir / "local_fit_preview.png", trace_img, split_graph, show_primitives=True)
        self._write_primitive_label_files(labels_json_path, labels_csv_path, components)

        payload = {
            "schema_version": "3.0",
            "backend": "graph_local_primitives",
            "trace_image_path": str(trace_copy_path),
            "svg_path": str(preview_svg_path),
            "metrics_path": str(self.stage_dir / "graph_primitives_metrics.json"),
            "canvas": {
                "width": int(trace_img.shape[1]),
                "height": int(trace_img.shape[0]),
                "unit": "px",
            },
            "nodes": split_graph.get("nodes", []),
            "edges": split_graph.get("edges", []),
            "components": components,
            "constraints": self._graph_constraints(split_graph),
            "metadata": {
                "source_image": str(self.image_path),
                "vtracer_svg": str(parameterizer.svg_path() or ""),
                "vtracer_metrics": str(parameterizer.metrics_path() or ""),
                "vtracer_intermediates": str(parameterizer.intermediate_dir() or ""),
                "edge_selection": self.edge_selection_diagnostics,
                "line_priority": "line > arc > local_spline",
                "global_spline_used": False,
            },
            "stages": {
                "input_image": str(self.image_path),
                "edge_or_stroke_mask": str(raw_edges_path),
                "topology_graph": str(self.graph_dir / "topology_graph.json"),
                "topology_graph_png": str(self.graph_dir / "topology_graph.png"),
                "split_edges": str(self.edge_split_dir / "split_edges.json"),
                "split_edges_preview": str(self.edge_split_dir / "split_edges_preview.png"),
                "local_fit_dir": str(self.local_fit_dir),
                "topology_validation": str(self.validation_dir / "topology_validation.json"),
                "graph_primitives": str(self.export_dir / "graph_primitives.json"),
                "preview_png": str(preview_png_path),
            },
            "metrics": aggregate_metrics,
        }
        json_path = self.save_dir / "curve_parameterization.json"
        self._write_json(self.export_dir / "graph_primitives.json", payload)
        self._write_json(json_path, payload)
        self._write_json(self.stage_dir / "graph_primitives_metrics.json", aggregate_metrics)
        self.last_status = {
            "fallback": False,
            "fallback_reason": "",
            "actual_backend": "graph_local_primitives",
            "topology_validation": validation,
        }
        print(
            "[GraphBasedLocalSplineParameterizer] done: "
            f"nodes={len(split_graph.get('nodes', []))}, "
            f"edges={len(split_graph.get('edges', []))}, "
            f"primitives={aggregate_metrics['primitive_count']}, json={json_path}"
        )
        return json_path

    def extract_topology_graph(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import numpy as np

        node_entries: List[Dict[str, Any]] = []
        endpoint_entries: List[Dict[str, Any]] = []
        for comp_idx, record in enumerate(records):
            pts = self._clean_points(record["points"])
            closed = bool(record.get("closed", False))
            if len(pts) < 2:
                continue
            if closed:
                anchor_idx = 0
                node_entries.append(self._node_entry(comp_idx, anchor_idx, pts[anchor_idx], "loop_anchor"))
            else:
                for idx in (0, len(pts) - 1):
                    entry = self._node_entry(comp_idx, idx, pts[idx], "endpoint")
                    node_entries.append(entry)
                    endpoint_entries.append(entry)

        for endpoint in endpoint_entries:
            point = np.asarray(endpoint["point"], dtype=np.float64)
            for comp_idx, record in enumerate(records):
                if comp_idx == endpoint["component_index"]:
                    continue
                pts = self._clean_points(record["points"])
                if len(pts) < 3:
                    continue
                distances = np.linalg.norm(pts - point, axis=1)
                idx = int(np.argmin(distances))
                if float(distances[idx]) <= self.graph_endpoint_snap_tolerance_px:
                    node_entries.append(self._node_entry(comp_idx, idx, pts[idx], "junction"))

        node_entries.extend(self._segment_intersection_entries(records))
        nodes, node_lookup = self._cluster_node_entries(node_entries)

        edges: List[Dict[str, Any]] = []
        edge_id = 1
        for comp_idx, record in enumerate(records):
            pts = self._clean_points(record["points"])
            if len(pts) < 2:
                continue
            closed = bool(record.get("closed", False))
            indices = sorted(set(int(entry["point_index"]) for entry in node_lookup.get(comp_idx, [])))
            if closed:
                if not indices:
                    indices = [0]
                cycle_indices = indices + [indices[0] + len(pts)]
                for start_i, end_i in zip(cycle_indices[:-1], cycle_indices[1:]):
                    edge_points = self._slice_record_points(pts, start_i, end_i, closed=True)
                    if len(edge_points) < 2:
                        continue
                    start_node = self._node_id_for(comp_idx, start_i % len(pts), node_lookup)
                    end_node = self._node_id_for(comp_idx, end_i % len(pts), node_lookup)
                    edge_points = self._lock_edge_points_to_nodes(edge_points, start_node, end_node, nodes)
                    edges.append(self._edge_record(edge_id, start_node, end_node, edge_points, comp_idx, closed_loop=start_node == end_node))
                    edge_id += 1
            else:
                if 0 not in indices:
                    indices.insert(0, 0)
                if len(pts) - 1 not in indices:
                    indices.append(len(pts) - 1)
                indices = sorted(set(indices))
                for start_i, end_i in zip(indices[:-1], indices[1:]):
                    if end_i <= start_i:
                        continue
                    edge_points = pts[start_i : end_i + 1]
                    start_node = self._node_id_for(comp_idx, start_i, node_lookup)
                    end_node = self._node_id_for(comp_idx, end_i, node_lookup)
                    edge_points = self._lock_edge_points_to_nodes(edge_points, start_node, end_node, nodes)
                    edges.append(self._edge_record(edge_id, start_node, end_node, edge_points, comp_idx, closed_loop=False))
                    edge_id += 1

        self._update_node_degrees(nodes, edges)
        return {
            "backend": "graph_local_primitives",
            "stage": "topology_graph",
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_merge_tolerance_px": self.graph_node_merge_tolerance_px,
                "endpoint_snap_tolerance_px": self.graph_endpoint_snap_tolerance_px,
                "source_component_count": len(records),
            },
        }

    def split_graph_edges_by_geometry(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        nodes = [dict(node) for node in graph.get("nodes", [])]
        next_node_id = max([int(node["id"]) for node in nodes] or [0]) + 1
        split_edges: List[Dict[str, Any]] = []
        next_edge_id = 1
        for edge in graph.get("edges", []):
            pts = np.asarray(edge.get("ordered_points", []), dtype=np.float64).reshape(-1, 2)
            if len(pts) < 2:
                continue
            split_indices = self._geometry_split_indices(pts, bool(edge.get("is_closed_loop", False)))
            local_nodes = [int(edge["start_node"])]
            for idx in split_indices[1:-1]:
                point = pts[int(idx)]
                node = {
                    "id": next_node_id,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "degree": 2,
                    "type": "corner",
                    "source": {
                        "split_from_edge_id": int(edge["id"]),
                        "source_point_index": int(idx),
                    },
                }
                nodes.append(node)
                local_nodes.append(next_node_id)
                next_node_id += 1
            local_nodes.append(int(edge["end_node"]))

            for seq_idx, (start_i, end_i) in enumerate(zip(split_indices[:-1], split_indices[1:])):
                if end_i <= start_i:
                    continue
                sub_points = pts[int(start_i) : int(end_i) + 1]
                if len(sub_points) < 2:
                    continue
                sub_points = self._lock_edge_points_to_nodes(sub_points, local_nodes[seq_idx], local_nodes[seq_idx + 1], nodes)
                split_edges.append(
                    self._edge_record(
                        next_edge_id,
                        local_nodes[seq_idx],
                        local_nodes[seq_idx + 1],
                        sub_points,
                        int(edge.get("source_component_index", 0)),
                        closed_loop=bool(edge.get("is_closed_loop", False)) and local_nodes[seq_idx] == local_nodes[seq_idx + 1],
                        parent_edge_id=int(edge["id"]),
                    )
                )
                next_edge_id += 1

        self._update_node_degrees(nodes, split_edges)
        return {
            "backend": "graph_local_primitives",
            "stage": "split_edges",
            "nodes": nodes,
            "edges": split_edges,
            "metadata": {
                **(graph.get("metadata", {}) or {}),
                "corner_angle_threshold_deg": self.graph_corner_angle_threshold_deg,
                "curvature_split_percentile": self.graph_curvature_split_percentile,
                "pre_split_edge_count": len(graph.get("edges", [])),
            },
        }

    def compute_geometry_features(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np

        pts = np.asarray(edge.get("ordered_points", []), dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            return {}
        centered = pts - np.mean(pts, axis=0)
        cov = np.cov(centered.T) if len(pts) > 2 else np.zeros((2, 2))
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1] if cov.shape == (2, 2) else np.array([0.0, 0.0])
        lambda1 = float(max(eigvals[0], 1e-12))
        lambda2 = float(max(eigvals[1], 1e-12))
        length = self._polyline_length(pts)
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        curvature = self._curvature(pts, closed=False)
        tangents = np.diff(pts, axis=0)
        angles = np.arctan2(tangents[:, 1], tangents[:, 0]) if len(tangents) else np.zeros(0)
        angle_jumps = np.abs(np.diff(np.unwrap(angles))) if len(angles) > 1 else np.zeros(0)
        direction_hist, _ = np.histogram((angles + math.pi) % math.pi, bins=8, range=(0.0, math.pi)) if len(angles) else (np.zeros(8), None)
        arc_metrics = self._arc_candidate_metrics(pts)
        return {
            "linearity_score": float(lambda1 / lambda2),
            "pca_lambda1": lambda1,
            "pca_lambda2": lambda2,
            "curvature_variance": float(np.var(curvature)) if len(curvature) else 0.0,
            "arc_consistency": arc_metrics.get("arc_consistency", 0.0),
            "tangent_stability": float(1.0 / (1.0 + np.mean(angle_jumps))) if len(angle_jumps) else 1.0,
            "point_density": float(len(pts) / max(length, 1e-9)),
            "direction_histogram": [int(v) for v in direction_hist.tolist()],
            "length_chord_ratio": float(length / max(chord, 1e-9)),
            "stable_radius": arc_metrics.get("stable_radius", False),
            "stable_curvature_sign": arc_metrics.get("stable_curvature_sign", False),
            "sweep_deg": arc_metrics.get("sweep_deg", 0.0),
            "radial_cv": arc_metrics.get("radial_cv", 0.0),
        }

    def fit_local_primitive(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        pts = self._clean_points(edge.get("ordered_points", []))
        if len(pts) < 2:
            raise ValueError(f"edge {edge.get('id')} has fewer than 2 points")
        features = self.compute_geometry_features(edge)

        line = self._fit_line_primitive_pca(pts)
        if (
            line["max_error"] <= self.line_tolerance_px
            and features.get("tangent_stability", 1.0) >= 0.72
        ) or self._should_accept_relaxed_fss_line(line, features):
            primitive = line
        else:
            arc = self._fit_arc_primitive(pts)
            if (
                arc is not None
                and arc["max_error"] <= self.arc_tolerance_px
                and self._should_accept_arc(arc, line)
                and bool(features.get("stable_radius", False))
                and bool(features.get("stable_curvature_sign", False))
            ):
                primitive = arc
            else:
                primitive = self._fit_local_spline_primitive(pts)

        primitive["primitive_type"] = primitive.get("type", "spline")
        primitive["source_edge_id"] = int(edge["id"])
        primitive["start_node"] = int(edge["start_node"])
        primitive["end_node"] = int(edge["end_node"])
        primitive["features"] = features
        primitive["parameters"] = self._primitive_parameters(primitive)
        primitive["fallback_points"] = pts.tolist()
        return primitive

    def _should_accept_relaxed_fss_line(self, line: Dict[str, Any], features: Dict[str, Any]) -> bool:
        """Accept slightly noisy long FSS straight traces as lines.

        This gate is deliberately stricter on topology/shape than on pixel RMS:
        it only fires for very high PCA linearity, stable tangent direction, and
        path length close to chord length.  Corner-like edges still fall through
        to arc/spline or further graph splitting.
        """
        max_error = float(line.get("max_error", 0.0) or 0.0)
        mean_error = float(line.get("mean_error", 0.0) or 0.0)
        source_points = int(line.get("source_point_count", 0) or 0)
        linearity = float(features.get("linearity_score", 0.0) or 0.0)
        length_chord = float(features.get("length_chord_ratio", 999.0) or 999.0)
        tangent = float(features.get("tangent_stability", 0.0) or 0.0)

        if source_points < max(12, self.min_segment_points):
            return False
        if linearity < 2500.0:
            return False
        if length_chord > 1.012:
            return False
        if tangent < 0.965:
            return False

        relaxed_max = max(self.line_tolerance_px * 1.75, 2.8)
        relaxed_mean = max(self.line_tolerance_px * 0.9, 1.45)
        return max_error <= relaxed_max and mean_error <= relaxed_mean

    def validate_graph_topology(self, graph: Dict[str, Any], components: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import numpy as np

        nodes = {int(node["id"]): node for node in graph.get("nodes", [])}
        graph_edges = {int(edge["id"]): edge for edge in graph.get("edges", [])}
        primitive_edges = set()
        issues: List[str] = []
        gap_errors: List[float] = []
        for component in components:
            primitive = (component.get("primitives") or [{}])[0]
            edge_id = int(primitive.get("source_edge_id", component.get("source_edge_id", -1)))
            primitive_edges.add(edge_id)
            start_node = nodes.get(int(component.get("start_node", -1)))
            end_node = nodes.get(int(component.get("end_node", -1)))
            points = np.asarray(component.get("fallback_points") or component.get("resampled_points") or [], dtype=np.float64).reshape(-1, 2)
            if start_node is None or end_node is None:
                issues.append(f"component {component.get('component_id')} references missing node")
                continue
            if len(points) >= 2:
                start_gap = float(np.linalg.norm(points[0] - np.array([start_node["x"], start_node["y"]])))
                end_gap = float(np.linalg.norm(points[-1] - np.array([end_node["x"], end_node["y"]])))
                gap_errors.extend([start_gap, end_gap])
                if max(start_gap, end_gap) > max(1.0, self.graph_node_merge_tolerance_px):
                    issues.append(f"component {component.get('component_id')} has endpoint gap {max(start_gap, end_gap):.3f}px")
            if self._polyline_self_intersects(points):
                issues.append(f"component {component.get('component_id')} has self intersection")

        missing_edges = sorted(edge_id for edge_id in graph_edges.keys() if edge_id not in primitive_edges)
        if missing_edges:
            issues.append(f"missing primitive edges: {missing_edges[:12]}")
        tiny_edges = [
            int(edge["id"])
            for edge in graph.get("edges", [])
            if float(edge.get("length", 0.0)) < max(1.0, self.min_component_length_px * 0.2)
        ]
        if tiny_edges:
            issues.append(f"tiny graph edges: {tiny_edges[:12]}")

        max_gap = max(gap_errors) if gap_errors else 0.0
        missing_ratio = len(missing_edges) / max(1, len(graph_edges))
        score = max(0.0, 1.0 - missing_ratio - min(0.5, max_gap / 20.0) - 0.05 * len(tiny_edges))
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "node_count": len(nodes),
            "edge_count": len(graph_edges),
            "component_count": len(components),
            "edge_connectivity_consistency": float(1.0 - missing_ratio),
            "junction_position_error": float(max_gap),
            "max_tiny_gap_px": float(max_gap),
            "topology_preservation_score": float(score),
        }

    def _fit_graph_components(self, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        components: List[Dict[str, Any]] = []
        for component_id, edge in enumerate(graph.get("edges", []), start=1):
            primitive = self.fit_local_primitive(edge)
            primitive["segment_id"] = 1
            primitive["kind"] = primitive.get("type", "spline")
            primitive["visual_label"] = (
                f"E{int(edge['id']):04d} {str(primitive.get('type', 'spline')).upper()} "
                f"nodes={edge['start_node']}->{edge['end_node']}"
            )
            points = primitive.get("fallback_points") or edge.get("ordered_points") or []
            component = {
                "component_id": component_id,
                "source_edge_id": int(edge["id"]),
                "source_component_index": int(edge.get("source_component_index", 0)),
                "closed": bool(edge.get("is_closed_loop", False)),
                "start_node": int(edge["start_node"]),
                "end_node": int(edge["end_node"]),
                "bbox": self._bbox(points),
                "sampled_point_count": int(len(points)),
                "fallback_points": points,
                "resampled_points": points,
                "primitives": [primitive],
                "segments": [primitive],
                "metrics": self._component_metrics(points, [primitive]),
            }
            components.append(component)
            self._write_json(self.local_fit_dir / f"edge_{int(edge['id']):04d}_fit.json", component)
            self._write_edge_fit_preview(self.local_fit_dir / f"edge_{int(edge['id']):04d}_fit.png", edge, primitive)
        return components

    def _fit_line_primitive_pca(self, points: Any) -> Dict[str, Any]:
        import numpy as np

        pts = self._clean_points(points)
        start = pts[0]
        end = pts[-1]
        errors = self._line_errors(pts, start, end)
        if len(pts) > 2:
            centered = pts - np.mean(pts, axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = vh[0]
        else:
            direction = end - start
        return {
            "type": "line",
            "kind": "line",
            "fit_method": "pca_total_least_squares_with_endpoint_lock",
            "start": start.tolist(),
            "end": end.tolist(),
            "points": [start.tolist(), end.tolist()],
            "direction": direction.tolist(),
            "max_error": float(np.max(errors)) if len(errors) else 0.0,
            "mean_error": float(np.mean(errors)) if len(errors) else 0.0,
            "effective_params": 4,
            "parameter_count": 4,
            "source_point_count": int(len(pts)),
        }

    def _fit_local_spline_primitive(self, points: Any) -> Dict[str, Any]:
        import numpy as np

        pts = self._clean_points(points)
        sampled = pts.copy()
        method = "raw_polyline_local_spline_fallback"
        quality = {
            "accepted": False,
            "reason": "scipy_unavailable_or_not_enough_points",
            "rms_error_px": 0.0,
            "length_shrink_ratio": 0.0,
        }
        if len(pts) >= 4:
            try:
                from scipy import interpolate

                distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                cumulative = np.concatenate([[0.0], np.cumsum(distances)])
                total = float(cumulative[-1])
                if total > 1e-9:
                    u = cumulative / total
                    k = min(3, len(pts) - 1)
                    tck, _ = interpolate.splprep([pts[:, 0], pts[:, 1]], u=u, s=0.0, k=k, per=False)
                    sample_count = max(len(pts), int(math.ceil(total / max(0.5, self.resample_step_px))) + 1)
                    u_new = np.linspace(0.0, 1.0, sample_count)
                    x_new, y_new = interpolate.splev(u_new, tck)
                    candidate = np.column_stack([x_new, y_new]).astype(np.float64)
                    candidate[0] = pts[0]
                    candidate[-1] = pts[-1]
                    errors = self._nearest_polyline_errors(pts, candidate)
                    raw_length = self._polyline_length(pts)
                    candidate_length = self._polyline_length(candidate)
                    shrink = max(0.0, (raw_length - candidate_length) / max(raw_length, 1e-9))
                    rms = float(math.sqrt(float(np.mean(errors * errors)))) if len(errors) else 0.0
                    if rms <= self.max_local_spline_rms_error_px and shrink <= self.max_local_spline_length_shrink_ratio:
                        sampled = candidate
                        method = "scipy_splprep_local_endpoint_locked"
                        quality = {
                            "accepted": True,
                            "rms_error_px": rms,
                            "length_shrink_ratio": float(shrink),
                        }
                    else:
                        quality = {
                            "accepted": False,
                            "reason": "local_spline_quality_gate_rejected",
                            "rms_error_px": rms,
                            "length_shrink_ratio": float(shrink),
                        }
            except Exception as exc:
                quality["reason"] = f"scipy_splprep_failed:{exc}"

        errors = self._nearest_polyline_errors(pts, sampled)
        return {
            "type": "spline",
            "kind": "spline",
            "degree": 3,
            "fit_method": method,
            "spline_quality": quality,
            "control_points": sampled.tolist(),
            "points": sampled.tolist(),
            "max_error": float(np.max(errors)) if len(errors) else 0.0,
            "mean_error": float(np.mean(errors)) if len(errors) else 0.0,
            "effective_params": int(len(sampled) * 2),
            "parameter_count": int(len(sampled) * 2),
            "source_point_count": int(len(pts)),
        }

    def _geometry_split_indices(self, points: Any, closed: bool) -> List[int]:
        import numpy as np

        pts = self._clean_points(points)
        n = len(pts)
        if n <= max(4, self.min_segment_points):
            return [0, n - 1]
        split = {0, n - 1}
        vectors = np.diff(pts, axis=0)
        if len(vectors) >= 2:
            angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
            jumps = np.abs(np.diff(angles))
            for rel_idx, jump in enumerate(jumps, start=1):
                if math.degrees(float(jump)) >= self.graph_corner_angle_threshold_deg:
                    split.add(rel_idx)

        curvature = np.abs(self._curvature(pts, closed=False))
        curvature_abs_max = float(np.max(curvature)) if len(curvature) else 0.0
        if len(curvature) >= 5 and curvature_abs_max > 1e-4:
            threshold = float(np.percentile(curvature, self.graph_curvature_split_percentile))
            threshold = max(threshold, curvature_abs_max * 0.35, 1e-4)
            for idx in np.where(curvature >= threshold)[0].tolist():
                if 1 < idx < n - 2:
                    split.add(int(idx))

        rdp = self._rdp_indices(pts, epsilon=max(2.5, self.line_tolerance_px * 2.0))
        split.update(idx for idx in rdp if 0 < idx < n - 1)
        return self._thin_split_indices(sorted(split), n)

    def _thin_split_indices(self, indices: Sequence[int], n: int) -> List[int]:
        min_gap = max(8, int(self.min_segment_points * 2))
        result = [0]
        for idx in sorted(set(int(i) for i in indices if 0 <= int(i) < n)):
            if idx in (0, n - 1):
                continue
            if idx - result[-1] >= min_gap and (n - 1) - idx >= min_gap:
                result.append(idx)
        result.append(n - 1)
        return sorted(set(result))

    def _rdp_indices(self, points: Any, epsilon: float) -> List[int]:
        pts = self._clean_points(points)
        if len(pts) < 3:
            return [0, len(pts) - 1]
        return self._rdp_indices_recursive(pts, 0, len(pts) - 1, float(epsilon))

    def _rdp_indices_recursive(self, pts: Any, start: int, end: int, epsilon: float) -> List[int]:
        import numpy as np

        if end - start < 2:
            return [start, end]
        segment = pts[start : end + 1]
        errors = self._line_errors(segment, segment[0], segment[-1])
        rel_idx = int(np.argmax(errors))
        max_error = float(errors[rel_idx])
        if max_error > epsilon:
            mid = start + rel_idx
            left = self._rdp_indices_recursive(pts, start, mid, epsilon)
            right = self._rdp_indices_recursive(pts, mid, end, epsilon)
            return left[:-1] + right
        return [start, end]

    def _arc_candidate_metrics(self, points: Any) -> Dict[str, Any]:
        import numpy as np

        arc = self._fit_arc_primitive(points)
        if arc is None:
            return {
                "arc_consistency": 0.0,
                "stable_radius": False,
                "stable_curvature_sign": False,
                "sweep_deg": 0.0,
                "radial_cv": 0.0,
            }
        pts = self._clean_points(points)
        center = np.asarray(arc["center"], dtype=np.float64)
        radius = float(arc["radius"])
        radial = np.linalg.norm(pts - center, axis=1)
        radial_cv = float(np.std(radial) / max(radius, 1e-9))
        curvature = self._curvature(pts, closed=False)
        nonzero = curvature[np.abs(curvature) > 1e-9]
        sign_stability = True
        if len(nonzero) >= 3:
            sign_stability = bool(np.mean(np.sign(nonzero) == np.sign(np.median(nonzero))) >= 0.8)
        return {
            "arc_consistency": float(1.0 / (1.0 + radial_cv)),
            "stable_radius": bool(radial_cv <= 0.08),
            "stable_curvature_sign": sign_stability,
            "sweep_deg": float(abs(arc.get("sweep_deg", 0.0))),
            "radial_cv": radial_cv,
        }

    def _node_entry(self, component_index: int, point_index: int, point: Any, kind: str) -> Dict[str, Any]:
        return {
            "component_index": int(component_index),
            "point_index": int(point_index),
            "point": [float(point[0]), float(point[1])],
            "kind": str(kind),
        }

    def _cluster_node_entries(self, entries: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
        import numpy as np

        clusters: List[List[Dict[str, Any]]] = []
        for entry in entries:
            point = np.asarray(entry["point"], dtype=np.float64)
            best_idx = -1
            best_dist = float("inf")
            for idx, cluster in enumerate(clusters):
                center = np.mean(np.asarray([item["point"] for item in cluster], dtype=np.float64), axis=0)
                dist = float(np.linalg.norm(point - center))
                if dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            if best_idx >= 0 and best_dist <= self.graph_node_merge_tolerance_px:
                clusters[best_idx].append(entry)
            else:
                clusters.append([entry])

        nodes: List[Dict[str, Any]] = []
        lookup: Dict[int, List[Dict[str, Any]]] = {}
        for node_id, cluster in enumerate(clusters, start=1):
            pts = np.asarray([item["point"] for item in cluster], dtype=np.float64)
            center = np.mean(pts, axis=0)
            kinds = {str(item.get("kind", "")) for item in cluster}
            node_type = "junction" if "junction" in kinds or len(cluster) >= 3 else ("loop_anchor" if "loop_anchor" in kinds else "endpoint")
            node = {
                "id": int(node_id),
                "x": float(center[0]),
                "y": float(center[1]),
                "degree": 0,
                "type": node_type,
                "source_refs": [
                    {
                        "component_index": int(item["component_index"]),
                        "point_index": int(item["point_index"]),
                        "kind": item.get("kind", ""),
                    }
                    for item in cluster
                ],
            }
            nodes.append(node)
            for item in cluster:
                lookup.setdefault(int(item["component_index"]), []).append(
                    {
                        "point_index": int(item["point_index"]),
                        "node_id": int(node_id),
                    }
                )
        return nodes, lookup

    def _node_id_for(self, component_index: int, point_index: int, lookup: Dict[int, List[Dict[str, Any]]]) -> int:
        candidates = lookup.get(int(component_index), [])
        point_index = int(point_index)
        for item in candidates:
            if int(item["point_index"]) == point_index:
                return int(item["node_id"])
        if candidates:
            nearest = min(candidates, key=lambda item: abs(int(item["point_index"]) - point_index))
            return int(nearest["node_id"])
        raise KeyError(f"missing node for component={component_index}, point={point_index}")

    def _edge_record(
        self,
        edge_id: int,
        start_node: int,
        end_node: int,
        points: Any,
        source_component_index: int,
        closed_loop: bool,
        parent_edge_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        pts = self._clean_points(points)
        features = {
            "local_curvature": self._curvature(pts, closed=False).tolist() if len(pts) >= 3 else [],
            "local_width_estimate": None,
        }
        edge = {
            "id": int(edge_id),
            "start_node": int(start_node),
            "end_node": int(end_node),
            "ordered_points": pts.tolist(),
            "length": float(self._polyline_length(pts)),
            "local_curvature": features["local_curvature"],
            "local_width_estimate": features["local_width_estimate"],
            "is_closed_loop": bool(closed_loop),
            "source_component_index": int(source_component_index),
        }
        if parent_edge_id is not None:
            edge["parent_edge_id"] = int(parent_edge_id)
        return edge

    @staticmethod
    def _lock_edge_points_to_nodes(points: Any, start_node: int, end_node: int, nodes: Sequence[Dict[str, Any]]) -> Any:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        node_map = {int(node["id"]): node for node in nodes}
        start = node_map.get(int(start_node))
        end = node_map.get(int(end_node))
        if start is not None and len(pts) > 0:
            pts[0] = [float(start["x"]), float(start["y"])]
        if end is not None and len(pts) > 1:
            pts[-1] = [float(end["x"]), float(end["y"])]
        return pts

    def _update_node_degrees(self, nodes: List[Dict[str, Any]], edges: Sequence[Dict[str, Any]]) -> None:
        degree = {int(node["id"]): 0 for node in nodes}
        for edge in edges:
            s = int(edge["start_node"])
            e = int(edge["end_node"])
            degree[s] = degree.get(s, 0) + 1
            degree[e] = degree.get(e, 0) + (0 if e == s else 1)
        for node in nodes:
            deg = int(degree.get(int(node["id"]), 0))
            node["degree"] = deg
            if node.get("type") == "corner":
                continue
            if node.get("type") == "loop_anchor":
                continue
            node["type"] = "endpoint" if deg <= 1 else ("junction" if deg >= 3 else "corner")

    def _segment_intersection_entries(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        cleaned = [self._clean_points(record["points"]) for record in records]
        max_segments = sum(max(0, len(pts) - 1) for pts in cleaned)
        if max_segments > 2500:
            return entries
        for ci, pts_a in enumerate(cleaned):
            for cj, pts_b in enumerate(cleaned):
                if cj <= ci:
                    continue
                for ia in range(len(pts_a) - 1):
                    a1, a2 = pts_a[ia], pts_a[ia + 1]
                    for ib in range(len(pts_b) - 1):
                        b1, b2 = pts_b[ib], pts_b[ib + 1]
                        if self._segments_intersect(a1, a2, b1, b2):
                            point = self._line_intersection_point(a1, a2, b1, b2)
                            if point is None:
                                continue
                            ia_near = ia if self._distance_tuple(tuple(a1), tuple(point)) <= self._distance_tuple(tuple(a2), tuple(point)) else ia + 1
                            ib_near = ib if self._distance_tuple(tuple(b1), tuple(point)) <= self._distance_tuple(tuple(b2), tuple(point)) else ib + 1
                            entries.append(self._node_entry(ci, ia_near, point, "junction"))
                            entries.append(self._node_entry(cj, ib_near, point, "junction"))
        return entries

    @staticmethod
    def _segments_intersect(a1: Any, a2: Any, b1: Any, b2: Any) -> bool:
        def orient(p: Any, q: Any, r: Any) -> float:
            return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

        o1 = orient(a1, a2, b1)
        o2 = orient(a1, a2, b2)
        o3 = orient(b1, b2, a1)
        o4 = orient(b1, b2, a2)
        return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)

    @staticmethod
    def _line_intersection_point(a1: Any, a2: Any, b1: Any, b2: Any) -> Optional[List[float]]:
        x1, y1 = float(a1[0]), float(a1[1])
        x2, y2 = float(a2[0]), float(a2[1])
        x3, y3 = float(b1[0]), float(b1[1])
        x4, y4 = float(b2[0]), float(b2[1])
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) <= 1e-12:
            return None
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return [float(px), float(py)]

    @staticmethod
    def _distance_tuple(a: Point, b: Point) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    @staticmethod
    def _slice_record_points(points: Any, start_i: int, end_i: int, closed: bool) -> Any:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        n = len(pts)
        if not closed:
            return pts[start_i : end_i + 1]
        indices = [idx % n for idx in range(int(start_i), int(end_i) + 1)]
        return pts[indices]

    @staticmethod
    def _clean_points(points: Any) -> Any:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        return GeometryDrivenParameterizer._remove_consecutive_duplicates(pts)

    @staticmethod
    def _primitive_parameters(primitive: Dict[str, Any]) -> Dict[str, Any]:
        kind = str(primitive.get("type", "spline"))
        keys = {
            "line": ("start", "end", "direction"),
            "arc": ("start", "end", "center", "radius", "clockwise", "sweep_deg"),
            "spline": ("degree", "control_points", "fit_method", "spline_quality"),
        }.get(kind, ("points",))
        return {key: primitive.get(key) for key in keys if key in primitive}

    @staticmethod
    def _graph_constraints(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        constraints = []
        for edge in graph.get("edges", []):
            constraints.append(
                {
                    "type": "edge_endpoint_lock",
                    "edge_id": int(edge["id"]),
                    "start_node": int(edge["start_node"]),
                    "end_node": int(edge["end_node"]),
                }
            )
        return constraints

    def _polyline_self_intersects(self, points: Any) -> bool:
        pts = self._clean_points(points)
        if len(pts) < 4:
            return False
        for i in range(len(pts) - 1):
            for j in range(i + 2, len(pts) - 1):
                if i == 0 and j == len(pts) - 2:
                    continue
                if self._segments_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                    return True
        return False

    def _write_graph_preview(self, path: Path, image: Any, graph: Dict[str, Any], show_primitives: bool) -> None:
        import cv2
        import numpy as np

        base = image.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        overlay = np.full_like(base, 255)
        h, w = overlay.shape[:2]
        colors = {
            "endpoint": (220, 60, 60),
            "junction": (40, 40, 230),
            "corner": (60, 170, 60),
            "loop_anchor": (170, 60, 180),
        }
        for edge in graph.get("edges", []):
            pts = np.asarray(edge.get("ordered_points", []), dtype=np.float64).reshape(-1, 2)
            if len(pts) < 2:
                continue
            poly = np.round(pts).astype(np.int32)
            poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
            cv2.polylines(overlay, [poly.reshape(-1, 1, 2)], False, (80, 80, 80), 1, lineType=cv2.LINE_AA)
            mid = tuple(poly[len(poly) // 2].tolist())
            cv2.putText(overlay, str(edge.get("id", "")), mid, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)
        for node in graph.get("nodes", []):
            x = int(round(float(node["x"])))
            y = int(round(float(node["y"])))
            if not (0 <= x < w and 0 <= y < h):
                continue
            color = colors.get(str(node.get("type")), (0, 0, 0))
            radius = 5 if node.get("type") == "junction" else 4
            cv2.circle(overlay, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(overlay, str(node["id"]), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        if show_primitives:
            cv2.putText(overlay, "local primitive fit", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
        self._write_image(path, np.hstack([base, overlay]))

    def _write_edge_fit_preview(self, path: Path, edge: Dict[str, Any], primitive: Dict[str, Any]) -> None:
        import cv2
        import numpy as np

        pts = self._clean_points(edge.get("ordered_points", []))
        if len(pts) < 2:
            return
        min_x, min_y, max_x, max_y = self._bbox(pts)
        pad = 16
        width = max(64, int(math.ceil(max_x - min_x + pad * 2)))
        height = max(64, int(math.ceil(max_y - min_y + pad * 2)))
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        def localize(raw: Sequence[Point]) -> Any:
            arr = np.asarray(raw, dtype=np.float64).reshape(-1, 2).copy()
            arr[:, 0] = arr[:, 0] - min_x + pad
            arr[:, 1] = arr[:, 1] - min_y + pad
            return np.round(arr).astype(np.int32)

        raw_poly = localize(pts)
        cv2.polylines(canvas, [raw_poly.reshape(-1, 1, 2)], False, (190, 190, 190), 1, cv2.LINE_AA)
        fit_pts = self._primitive_preview_points(primitive)
        if len(fit_pts) >= 2:
            fit_poly = localize(fit_pts)
            color = {"line": (255, 144, 30), "arc": (40, 140, 242), "spline": (82, 168, 50)}.get(str(primitive.get("type")), (20, 20, 20))
            cv2.polylines(canvas, [fit_poly.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"E{edge['id']} {primitive.get('type')}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        self._write_image(path, canvas)

    @classmethod
    def _write_json(cls, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cls._to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
