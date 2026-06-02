from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]

EDGE_REPRESENTATION_MODE = "patch_topology"


class GeometryDrivenParameterizer:
    """Geometry-driven parameterization with B-spline as an intermediate layer.

    中文说明：
    这个模块实现新的几何驱动流程：
    repair image -> edge repair -> ordered contours -> unified B-spline
    -> curvature/geometric analysis -> compact line/arc/spline primitives.

    English notes:
    The B-spline is intentionally not the final representation. It is used as a
    continuous denoised curve for robust primitive decomposition. The exported
    JSON keeps compact primitives plus sampled fallback points for CST safety.
    """

    def __init__(
        self,
        image_path: Path | str,
        save_dir: Path | str,
        line_tolerance_px: float = 1.2,
        arc_tolerance_px: float = 1.5,
        residual_spline_tolerance_px: float = 2.2,
        resample_step_px: float = 2.5,
        bspline_smoothing: float = 8.0,
        arc_min_sweep_deg: float = 28.0,
        arc_min_error_improvement_ratio: float = 0.55,
        arc_max_radius_to_chord_ratio: float = 4.0,
        arc_min_source_points: int = 24,
        max_centerline_components_for_geometry: int = 20,
        max_bspline_length_shrink_ratio: float = 0.08,
        max_bspline_rms_error_px: float = 2.0,
        close_parallel_edge_distance_px: int = 3,
        close_parallel_edge_ratio_threshold: float = 0.35,
        close_parallel_merge_distance_px: int = 2,
        max_stroke_mask_foreground_ratio: float = 0.08,
        max_stroke_mask_to_canny_ratio: float = 3.0,
        max_sparse_auto_to_canny_ratio: float = 0.45,
        min_merged_canny_retention_ratio: float = 0.50,
        max_decompose_depth: int = 12,
        min_segment_points: int = 8,
        min_component_length_px: float = 10.0,
    ):
        self.image_path = Path(image_path)
        self.save_dir = Path(save_dir)
        self.line_tolerance_px = float(line_tolerance_px)
        self.arc_tolerance_px = float(arc_tolerance_px)
        self.residual_spline_tolerance_px = float(residual_spline_tolerance_px)
        self.resample_step_px = float(resample_step_px)
        self.bspline_smoothing = float(bspline_smoothing)
        self.arc_min_sweep_deg = float(arc_min_sweep_deg)
        self.arc_min_error_improvement_ratio = float(arc_min_error_improvement_ratio)
        self.arc_max_radius_to_chord_ratio = float(arc_max_radius_to_chord_ratio)
        self.arc_min_source_points = int(arc_min_source_points)
        self.max_centerline_components_for_geometry = int(max_centerline_components_for_geometry)
        self.max_bspline_length_shrink_ratio = float(max_bspline_length_shrink_ratio)
        self.max_bspline_rms_error_px = float(max_bspline_rms_error_px)
        self.close_parallel_edge_distance_px = int(close_parallel_edge_distance_px)
        self.close_parallel_edge_ratio_threshold = float(close_parallel_edge_ratio_threshold)
        self.close_parallel_merge_distance_px = int(close_parallel_merge_distance_px)
        self.max_stroke_mask_foreground_ratio = float(max_stroke_mask_foreground_ratio)
        self.max_stroke_mask_to_canny_ratio = float(max_stroke_mask_to_canny_ratio)
        self.max_sparse_auto_to_canny_ratio = float(max_sparse_auto_to_canny_ratio)
        self.min_merged_canny_retention_ratio = float(min_merged_canny_retention_ratio)
        self.max_decompose_depth = int(max_decompose_depth)
        self.min_segment_points = int(min_segment_points)
        self.min_component_length_px = float(min_component_length_px)

        self.stage_dir = self.save_dir / "geometry_primitives"
        self.edge_dir = self.stage_dir / "01_edges"
        self.vtracer_dir = self.stage_dir / "02_vtracer_centerline"
        self.contour_dir = self.stage_dir / "03_ordered_centerlines"
        self.bspline_dir = self.stage_dir / "04_bspline"
        self.primitive_dir = self.stage_dir / "05_primitives"
        self.preview_dir = self.stage_dir / "06_preview"
        self.last_status: Dict[str, Any] = {
            "fallback": False,
            "fallback_reason": "",
            "actual_backend": "geometry_primitives",
        }
        self.edge_selection_diagnostics: Dict[str, Any] = {}

    class _EdgeCandidate:
        """Small NewParams-compatible wrapper for derived edge images."""

        def __init__(
            self,
            image_path: Path,
            save_dir: Path,
            edges: Any,
            original_img: Any,
            edge_representation: str,
            edge_contour_tracing: Optional[bool] = None,
            verbose: bool = True,
        ):
            self.image_path = Path(image_path)
            self.save_dir = Path(save_dir)
            self._edges = edges
            self._original_img = original_img
            self._edge_representation = edge_representation
            self._edge_contour_tracing = edge_contour_tracing
            self.verbose = bool(verbose)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            GeometryDrivenParameterizer._write_image(
                self.save_dir / "repair_fig_edges.png",
                (self._edges > 0).astype("uint8") * 255,
            )

        def edges(self) -> Any:
            return self._edges

        def original_img(self) -> Any:
            return self._original_img

        def edge_representation(self) -> str:
            return self._edge_representation

        def parameterize(self, save_dir: Path, **kwargs: Any) -> Any:
            from Rebuild.NewParams import CurveParameterizer

            if "edge_contour_tracing" not in kwargs:
                if self._edge_contour_tracing is None:
                    kwargs["edge_contour_tracing"] = self._edge_representation == "edge"
                else:
                    kwargs["edge_contour_tracing"] = bool(self._edge_contour_tracing)
            return CurveParameterizer(
                image_path=self.image_path,
                edges=(self._edges > 0).astype("uint8") * 255,
                original_img=self._original_img,
                save_dir=save_dir,
                trace_source="edges",
                verbose=self.verbose,
                **kwargs,
            )

    def run(self) -> Path:
        import cv2
        import numpy as np

        from Rebuild.NewParams import NewParams

        for path in (
            self.stage_dir,
            self.edge_dir,
            self.vtracer_dir,
            self.contour_dir,
            self.bspline_dir,
            self.primitive_dir,
            self.preview_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        print("[GeometryDrivenParameterizer] stage 1/8: adaptive edge preprocessing")
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

        print("[GeometryDrivenParameterizer] stage 2/8: VTracer topology extraction through standard defaults")
        # 中文说明：
        # 先复用 standard pipeline 的 NewParams.parameterize 默认路径。只有旧版拓扑足够简单时，
        # 才继续进入 B-spline / primitive 压缩；复杂粗线或填充结构直接回退到 standard JSON。
        #
        # English notes:
        # Reuse the same NewParams/VTracer entry point as the standard pipeline.
        # Geometry compression is allowed only after the standard topology looks safe.
        # 关键点：这里必须走 vtracer_python.py 的 centerline skeleton 流程。
        # 不能再直接对粗线做 findContours，否则会得到内外双层边界。
        # Key point: use vtracer_python.py centerline skeleton flow. Direct
        # findContours on thick strokes creates inner/outer double contours.
        parameterizer = params.parameterize(
            save_dir=self.vtracer_dir,
        )
        vtracer_results = parameterizer.results()
        if len(vtracer_results) > self.max_centerline_components_for_geometry:
            return self._write_standard_topology_fallback(
                params=params,
                parameterizer=parameterizer,
                reason=(
                    "too_many_centerline_components: "
                    f"{len(vtracer_results)} > {self.max_centerline_components_for_geometry}"
                ),
            )

        contour_records = self._extract_vtracer_centerline_records(vtracer_results)
        if not contour_records:
            raise ValueError("VTracer centerline extraction produced no valid ordered centerlines.")

        trace_image_path = Path(parameterizer.trace_image_path() or self.image_path)
        trace_img = cv2.imread(str(trace_image_path), cv2.IMREAD_COLOR)
        if trace_img is None:
            trace_img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if trace_img is None:
            raise FileNotFoundError(f"Cannot read trace image for preview: {trace_image_path}")
        trace_copy_path = self.edge_dir / "vtracer_trace_input.png"
        self._write_image(trace_copy_path, trace_img)

        print("[GeometryDrivenParameterizer] stage 3/8: ordered centerline export")
        self._write_json(
            self.contour_dir / "ordered_centerlines_summary.json",
            {
                "source": "vtracer_python.centerline",
                "vtracer_svg_path": str(parameterizer.svg_path() or ""),
                "vtracer_metrics_path": str(parameterizer.metrics_path() or ""),
                "vtracer_intermediate_dir": str(parameterizer.intermediate_dir() or ""),
                "component_count": len(contour_records),
                "records": [
                    {
                        "component_id": i + 1,
                        "source_component_id": item["source_contour_id"],
                        "point_count": int(len(item["points"])),
                        "closed": bool(item["closed"]),
                        "path_length_px": float(item["path_length_px"]),
                    }
                    for i, item in enumerate(contour_records)
                ],
            },
        )

        print("[GeometryDrivenParameterizer] stage 4/8: unified B-spline reconstruction")
        components: List[Dict[str, Any]] = []
        for component_index, record in enumerate(contour_records, start=1):
            raw_points = record["points"]
            bspline = self._fit_unified_bspline(raw_points, closed=record["closed"])
            bspline = self._guard_bspline_quality(raw_points, bspline, closed=record["closed"])
            sampled = bspline["sampled_points"]
            if len(sampled) < 4:
                continue

            bspline_debug_path = self.bspline_dir / f"component_{component_index:03d}_bspline.json"
            self._write_json(
                bspline_debug_path,
                {
                    "component_id": component_index,
                    "source_contour_id": record["source_contour_id"],
                    "closed": bool(record["closed"]),
                    "raw_point_count": int(len(raw_points)),
                    "sampled_point_count": int(len(sampled)),
                    "path_length_px": float(self._polyline_length(sampled)),
                    "curvature": bspline["curvature"].tolist(),
                    "sampled_points": sampled.tolist(),
                    "control_points": bspline["control_points"].tolist(),
                    "method": bspline["method"],
                    "source": bspline.get("source", ""),
                    "loss": bspline.get("loss", None),
                    "step": bspline.get("step", None),
                },
            )

            print(
                "[GeometryDrivenParameterizer] stage 5/8: primitive decomposition "
                f"component={component_index}, samples={len(sampled)}"
            )
            primitives = self._decompose_to_primitives(sampled, closed=record["closed"])
            primitives = self._merge_adjacent_lines(primitives)
            primitives = self._annotate_primitives(primitives, sampled_point_count=int(len(sampled)))
            metrics = self._component_metrics(sampled, primitives)
            component = {
                "component_id": component_index,
                "source_contour_id": record["source_contour_id"],
                "closed": bool(record["closed"]),
                "bbox": self._bbox(sampled),
                "sampled_point_count": int(len(sampled)),
                # fallback_points / resampled_points are intentionally kept for CST fallback.
                # fallback_points / resampled_points 专门保留给 CST 回退路径使用。
                "fallback_points": sampled.tolist(),
                "resampled_points": sampled.tolist(),
                "bspline_debug_path": str(bspline_debug_path),
                "primitives": primitives,
                "segments": primitives,
                "metrics": metrics,
            }
            components.append(component)

            self._write_json(
                self.primitive_dir / f"component_{component_index:03d}_primitives.json",
                component,
            )

        if not components:
            raise ValueError("Geometry-driven decomposition produced no valid components.")

        print("[GeometryDrivenParameterizer] stage 6/8: compact JSON export")
        aggregate_metrics = self._aggregate_metrics(components)
        preview_svg_path = self.preview_dir / "geometry_primitives_preview.svg"
        preview_png_path = self.preview_dir / "geometry_primitives_preview.png"
        labels_json_path = self.preview_dir / "primitive_labels.json"
        labels_csv_path = self.preview_dir / "primitive_labels.csv"
        self._write_svg_preview(preview_svg_path, trace_img.shape[1], trace_img.shape[0], components)
        self._write_png_preview(preview_png_path, trace_img, components)
        self._write_primitive_label_files(labels_json_path, labels_csv_path, components)

        payload = {
            "schema_version": "2.0",
            "backend": "geometry_driven_vtracer_centerline_bspline_intermediate",
            "trace_image_path": str(trace_copy_path),
            "svg_path": str(preview_svg_path),
            "metrics_path": str(self.stage_dir / "geometry_primitives_metrics.json"),
            "canvas": {
                "width": int(trace_img.shape[1]),
                "height": int(trace_img.shape[0]),
                "unit": "px",
            },
            "stages": {
                "input_image": str(self.image_path),
                "edge_or_stroke_mask": str(raw_edges_path),
                "vtracer_trace_input": str(trace_copy_path),
                "vtracer_svg": str(parameterizer.svg_path() or ""),
                "vtracer_metrics": str(parameterizer.metrics_path() or ""),
                "vtracer_intermediates": str(parameterizer.intermediate_dir() or ""),
                "ordered_centerlines": str(self.contour_dir / "ordered_centerlines_summary.json"),
                "bspline_dir": str(self.bspline_dir),
                "primitive_dir": str(self.primitive_dir),
                "preview_svg": str(preview_svg_path),
                "preview_png": str(preview_png_path),
                "primitive_labels_json": str(labels_json_path),
                "primitive_labels_csv": str(labels_csv_path),
            },
            "metrics": aggregate_metrics,
            "components": components,
        }
        json_path = self.save_dir / "curve_parameterization.json"
        self._write_json(json_path, payload)
        self._write_json(self.stage_dir / "geometry_primitives_metrics.json", aggregate_metrics)

        print(
            "[GeometryDrivenParameterizer] done: "
            f"components={aggregate_metrics['component_count']}, "
            f"primitives={aggregate_metrics['primitive_count']}, "
            f"params={aggregate_metrics['total_parameters']}, "
            f"json={json_path}"
        )
        return json_path

    def _repair_edges(self, edges: Any) -> Any:
        import cv2
        import numpy as np

        binary = (edges > 0).astype(np.uint8) * 255
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        repaired = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        repaired = cv2.dilate(repaired, close_kernel, iterations=1)
        repaired = cv2.erode(repaired, close_kernel, iterations=1)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((repaired > 0).astype(np.uint8), 8)
        cleaned = np.zeros_like(repaired)
        h, w = repaired.shape[:2]
        min_area = max(8, int(0.00002 * h * w))
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label] = 255
        return cleaned

    def build_patch_topology_vtracer_input(
        self,
        image: Any,
        *,
        debug_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        import numpy as np

        pec_mask, extract_metrics = self.extract_patch_topology_mask(image)
        no_frame, frame_metrics = self.remove_outer_substrate_component(pec_mask)
        denoised = self.light_denoise_topology_mask(no_frame, min_noise_area=6)
        skeleton = self.safe_topology_skeletonize(denoised)
        skeleton = self.suppress_canvas_border_skeleton(skeleton)
        pruned = self.prune_short_spurs(skeleton, max_spur_length_px=4)
        validation = self.validate_topology_preservation(denoised, pruned)

        metrics: Dict[str, Any] = {
            "mode": EDGE_REPRESENTATION_MODE,
            "accepted": bool(validation["accepted"]),
            "fallback_reason": validation["fallback_reason"],
            **extract_metrics,
            **frame_metrics,
            "foreground_pixels": int(np.count_nonzero(denoised)),
            "component_count": self._mask_component_count(denoised),
            "skeleton_length": int(np.count_nonzero(pruned)),
            "junction_count": self._skeleton_junction_count(pruned),
            "endpoint_count": self._skeleton_endpoint_count(pruned),
            "retention_ratio": validation["retention_ratio"],
        }
        metrics["accepted"] = bool(validation["accepted"] and self._patch_topology_metrics_sane(metrics))
        if validation["accepted"] and not metrics["accepted"]:
            metrics["fallback_reason"] = "patch_topology_metrics_not_sane"

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            self._write_image(debug_dir / "01_pec_mask.png", pec_mask)
            self._write_image(debug_dir / "02_outer_frame_removed.png", no_frame)
            self._write_image(debug_dir / "03_light_denoise.png", denoised)
            self._write_image(debug_dir / "04_skeleton.png", skeleton)
            self._write_image(debug_dir / "05_pruned_skeleton.png", pruned)
            self._write_json(debug_dir / "patch_topology_metrics.json", metrics)

        return {
            "accepted": bool(metrics["accepted"]),
            "skeleton": pruned,
            "metrics": metrics,
        }

    def extract_patch_topology_mask(self, image: Any) -> Tuple[Any, Dict[str, Any]]:
        """Extract a filled conductive PEC mask without Canny.

        中文说明：
        贴片天线参数化需要导体拓扑，而不是轮廓边界。这里从原图直接提取
        填充导体区域，避免先生成 Canny 双边再试图修补。
        """
        import cv2
        import numpy as np

        array = np.asarray(image)
        if array.ndim == 2:
            bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif array.ndim == 3 and array.shape[2] == 3:
            bgr = array.astype(np.uint8)
        else:
            raise ValueError("Patch topology mask input must be grayscale or BGR image.")

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1]

        low_saturation = (saturation <= 95) & (gray >= 80) & (gray <= 252)
        bright_low_saturation = (saturation <= 95) & (gray >= 180) & (gray <= 252)
        mask = np.where(low_saturation, 255, 0).astype(np.uint8)

        # 只保留明显连通导体候选；不做 closing/dilation，避免把槽线和馈线错误粘合。
        cleaned = self.light_denoise_topology_mask(mask, min_noise_area=6)
        preferred = self.light_denoise_topology_mask(np.where(bright_low_saturation, 255, 0).astype(np.uint8), min_noise_area=6)
        cleaned, candidate_metrics = self._keep_patch_topology_components(cleaned, preferred_mask=preferred)
        metrics = {
            "pec_mask_pixels": int(np.count_nonzero(cleaned)),
            "pec_mask_component_count": self._mask_component_count(cleaned),
            "pec_mask_strategy": "low_saturation_filled_conductor_no_canny",
            "low_saturation_threshold": 95,
            "min_conductor_gray": 80,
            "preferred_min_conductor_gray": 180,
            "max_conductor_gray": 252,
            **candidate_metrics,
        }
        return cleaned, metrics

    def _keep_patch_topology_components(self, mask: Any, *, preferred_mask: Optional[Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """Keep non-frame conductive components from a filled mask candidate."""
        import cv2
        import numpy as np

        binary = (mask > 0).astype(np.uint8) * 255
        height, width = binary.shape[:2]
        image_area = float(max(1, height * width))
        candidate_binary = binary
        if preferred_mask is not None and int(np.count_nonzero(preferred_mask)) >= 100:
            candidate_binary = (preferred_mask > 0).astype(np.uint8) * 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((candidate_binary > 0).astype(np.uint8), 8)
        kept = np.zeros_like(binary)
        component_records: List[Dict[str, Any]] = []
        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 6:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = max(1, w * h)
            fill_ratio = float(area / bbox_area)
            bbox_area_ratio = float(bbox_area / image_area)
            touches_border = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
            looks_like_frame_or_page = touches_border and bbox_area_ratio > 0.35 and fill_ratio > 0.88
            record = {
                "label": int(label),
                "area": area,
                "bbox": [x, y, w, h],
                "fill_ratio": fill_ratio,
                "bbox_area_ratio": bbox_area_ratio,
                "touches_border": bool(touches_border),
            }
            if looks_like_frame_or_page:
                record["kept"] = False
                record["reject_reason"] = "touching_large_filled_outer_component"
                component_records.append(record)
                continue
            score = float(area)
            if touches_border:
                score *= 0.15
            if fill_ratio < 0.08:
                score *= 0.1
            record["score"] = score
            scored.append((score, label, record))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score = scored[0][0]
            for score, label, record in scored:
                keep = score >= max(50.0, best_score * 0.20)
                if keep:
                    kept[labels == label] = 255
                record["kept"] = bool(keep)
                component_records.append(record)

        return kept, {
            "kept_component_count": int(self._mask_component_count(kept)),
            "used_preferred_bright_mask": bool(preferred_mask is not None and int(np.count_nonzero(preferred_mask)) >= 100),
            "component_filter_records": component_records[:20],
        }

    def remove_outer_substrate_component(
        self,
        mask: Any,
        *,
        outer_frame_min_area_ratio: float = 0.35,
        outer_frame_rectangularity: float = 0.88,
    ) -> Tuple[Any, Dict[str, Any]]:
        import cv2
        import numpy as np

        binary = (mask > 0).astype(np.uint8) * 255
        height, width = binary.shape[:2]
        image_area = float(max(1, height * width))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
        cleaned = binary.copy()
        metrics: Dict[str, Any] = {
            "removed_outer_frame": False,
            "outer_frame_removed_area": 0,
            "outer_frame_min_area_ratio": float(outer_frame_min_area_ratio),
            "outer_frame_rectangularity_threshold": float(outer_frame_rectangularity),
            "outer_frame_bbox": [],
        }

        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = max(1, w * h)
            bbox_area_ratio = float(bbox_area / image_area)
            rectangularity = float(area / bbox_area)
            touches_border = x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
            if touches_border and bbox_area_ratio > outer_frame_min_area_ratio and rectangularity > outer_frame_rectangularity:
                cleaned[labels == label] = 0
                metrics["removed_outer_frame"] = True
                metrics["outer_frame_removed_area"] = area
                metrics["outer_frame_bbox"] = [x, y, w, h]
                break

        return cleaned, metrics

    def light_denoise_topology_mask(self, mask: Any, *, min_noise_area: int = 6) -> Any:
        import cv2
        import numpy as np

        binary = (mask > 0).astype(np.uint8) * 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
        cleaned = np.zeros_like(binary)
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= int(min_noise_area):
                cleaned[labels == label] = 255
        return cleaned

    def safe_topology_skeletonize(self, mask: Any) -> Any:
        return self._skeletonize_binary(mask > 0).astype("uint8") * 255

    def suppress_canvas_border_skeleton(self, skeleton: Any, *, border_margin_px: Optional[int] = None) -> Any:
        """Remove only skeleton pixels hugging the image canvas border."""
        import numpy as np

        cleaned = (skeleton > 0).astype(np.uint8) * 255
        height, width = cleaned.shape[:2]
        if border_margin_px is None:
            border_margin_px = max(6, int(round(min(height, width) * 0.05)))
        margin = max(0, int(border_margin_px))
        if margin <= 0 or height <= margin * 2 or width <= margin * 2:
            return cleaned
        cleaned[:margin, :] = 0
        cleaned[height - margin :, :] = 0
        cleaned[:, :margin] = 0
        cleaned[:, width - margin :] = 0
        return cleaned

    def prune_short_spurs(self, skeleton: Any, *, max_spur_length_px: int = 4) -> Any:
        import cv2
        import numpy as np

        pruned = (skeleton > 0).astype(np.uint8) * 255
        # 只删除极短叶子毛刺。若 endpoint 总数很少，说明结构可能依赖端点，
        # 此时不剪枝，避免伤到馈线入口。
        for _ in range(max(0, int(max_spur_length_px))):
            binary = (pruned > 0).astype(np.uint8)
            endpoints, endpoint_mask = self._skeleton_endpoint_data(binary)
            if len(endpoints) <= 2:
                break
            neighbor_count = cv2.filter2D(binary, cv2.CV_16S, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT) - binary
            junction_mask = (binary == 1) & (neighbor_count >= 3)
            removable = endpoint_mask.copy()
            # junction 邻域内的端点通常是真实短 stub，不轻易删。
            if np.count_nonzero(junction_mask) > 0:
                kernel = np.ones((3, 3), dtype=np.uint8)
                junction_near = cv2.dilate(junction_mask.astype(np.uint8), kernel, iterations=1) > 0
                removable[junction_near] = False
            if int(np.count_nonzero(removable)) == 0:
                break
            pruned[removable] = 0
        return pruned

    def validate_topology_preservation(self, mask: Any, skeleton: Any) -> Dict[str, Any]:
        import numpy as np

        mask_pixels = int(np.count_nonzero(mask))
        skeleton_pixels = int(np.count_nonzero(skeleton))
        if mask_pixels <= 0 or skeleton_pixels <= 0:
            return {"accepted": False, "fallback_reason": "empty_topology_mask_or_skeleton", "retention_ratio": 0.0}
        if mask_pixels < 100 or skeleton_pixels < 20:
            return {
                "accepted": False,
                "fallback_reason": "topology_candidate_too_small",
                "retention_ratio": float(skeleton_pixels / max(1, mask_pixels)),
            }

        before_components = self._mask_component_count(mask)
        after_components = self._mask_component_count(skeleton)
        retention_ratio = float(skeleton_pixels / max(1, mask_pixels))
        if after_components > max(8, before_components * 3):
            return {
                "accepted": False,
                "fallback_reason": "skeleton_component_count_exploded",
                "retention_ratio": retention_ratio,
            }
        if retention_ratio < 0.003:
            return {
                "accepted": False,
                "fallback_reason": "skeleton_retention_too_low",
                "retention_ratio": retention_ratio,
            }
        return {"accepted": True, "fallback_reason": "", "retention_ratio": retention_ratio}

    def _patch_topology_metrics_sane(self, metrics: Dict[str, Any]) -> bool:
        component_count = int(metrics.get("component_count", 0))
        endpoint_count = int(metrics.get("endpoint_count", 0))
        skeleton_length = int(metrics.get("skeleton_length", 0))
        pec_pixels = int(metrics.get("pec_mask_pixels", 0))
        if skeleton_length < 20 or pec_pixels < 100:
            return False
        if component_count > 6:
            return False
        if endpoint_count > 24:
            return False
        return True

    def _mask_component_count(self, mask: Any) -> int:
        import cv2
        import numpy as np

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        count = 0
        for label in range(1, n_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= 2:
                count += 1
        return int(count)

    def _skeleton_endpoint_data(self, skeleton_binary: Any) -> Tuple[List[Point], Any]:
        import cv2
        import numpy as np

        binary = (skeleton_binary > 0).astype(np.uint8)
        neighbor_count = cv2.filter2D(binary, cv2.CV_16S, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT) - binary
        endpoint_mask = (binary == 1) & (neighbor_count == 1)
        ys, xs = np.where(endpoint_mask)
        return [(float(x), float(y)) for y, x in zip(ys, xs)], endpoint_mask

    def _skeleton_endpoint_count(self, skeleton: Any) -> int:
        endpoints, _ = self._skeleton_endpoint_data(skeleton > 0)
        return int(len(endpoints))

    def _skeleton_junction_count(self, skeleton: Any) -> int:
        import cv2
        import numpy as np

        binary = (skeleton > 0).astype(np.uint8)
        neighbor_count = cv2.filter2D(binary, cv2.CV_16S, np.ones((3, 3), dtype=np.uint8), borderType=cv2.BORDER_CONSTANT) - binary
        return int(np.count_nonzero((binary == 1) & (neighbor_count >= 3)))

    def _merge_close_parallel_edges(self, edges: Any) -> Any:
        """Merge close Canny double edges into a single centerline-like mask.

        中文说明：
        当 Canny 产生相距很近的内外双边，而 auto/stroke_mask 又因为过密不可用时，
        按贴片天线先验把近距离平行边视作同一条线。实现上先膨胀闭合这些近邻边，
        再 skeletonize 回单像素中心线。
        """
        import cv2
        import numpy as np

        binary = (edges > 0).astype(np.uint8)
        distance_px = max(1, int(self.close_parallel_merge_distance_px))
        kernel_size = distance_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Keep this conservative: dilation joins only very close double edges.
        # A large close/open operation can bridge real patch edges and erase major structure.
        merged = cv2.dilate(binary * 255, kernel, iterations=1)

        skeleton = self._skeletonize_binary(merged > 0)
        return skeleton.astype(np.uint8) * 255

    @staticmethod
    def _skeletonize_binary(mask: Any) -> Any:
        import cv2
        import numpy as np

        binary = (mask > 0).astype(np.uint8)
        try:
            from skimage.morphology import skeletonize

            return skeletonize(binary > 0).astype(np.uint8)
        except Exception:
            img = binary * 255
            skel = np.zeros_like(img)
            element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            while True:
                eroded = cv2.erode(img, element)
                temp = cv2.dilate(eroded, element)
                temp = cv2.subtract(img, temp)
                skel = cv2.bitwise_or(skel, temp)
                img = eroded.copy()
                if cv2.countNonZero(img) == 0:
                    break
            return (skel > 0).astype(np.uint8)

    def _select_preprocessing_params(self, newparams_cls: Any) -> Any:
        """Choose Canny edges or stroke-mask input using a close-parallel-edge prior.

        中文说明：
        Canny 对有线宽的结构容易提取出内外两层边缘。如果这些额外边缘和 stroke-mask
        在 1-3px 内长期贴得很近，更像双边伪影，而不是真实天线结构；此时切换到
        auto/stroke_mask。否则保留 Canny，以避免粗填充区域被当作整块前景。
        """
        import cv2
        import numpy as np

        canny = newparams_cls(
            self.image_path,
            save_dir=self.edge_dir / "_candidate_canny",
            edge_mode="canny",
            verbose=True,
        )
        auto = newparams_cls(
            self.image_path,
            save_dir=self.edge_dir / "_candidate_auto",
            edge_mode="auto",
            verbose=True,
        )
        foreground = newparams_cls(
            self.image_path,
            save_dir=self.edge_dir / "_candidate_foreground_contour",
            edge_mode="foreground_contour",
            verbose=True,
        )
        stroke = newparams_cls(
            self.image_path,
            save_dir=self.edge_dir / "_candidate_stroke_mask",
            edge_mode="stroke_mask",
            verbose=True,
        )

        canny_edges = (canny.edges() > 0).astype(np.uint8)
        auto_edges = (auto.edges() > 0).astype(np.uint8)
        foreground_edges = (foreground.edges() > 0).astype(np.uint8)
        stroke_edges = (stroke.edges() > 0).astype(np.uint8)
        frame_cleanup = {
            "canny": self._remove_outer_square_frame_edges(canny_edges),
            "auto": self._remove_outer_square_frame_edges(auto_edges),
            "foreground_contour": self._remove_outer_square_frame_edges(foreground_edges),
            "stroke_mask": self._remove_outer_square_frame_edges(stroke_edges),
        }
        canny_edges = frame_cleanup["canny"]["edges"]
        auto_edges = frame_cleanup["auto"]["edges"]
        foreground_edges = frame_cleanup["foreground_contour"]["edges"]
        stroke_edges = frame_cleanup["stroke_mask"]["edges"]
        if frame_cleanup["canny"]["removed_pixels"] > 0:
            canny = self._EdgeCandidate(
                image_path=self.image_path,
                save_dir=self.edge_dir / "_candidate_canny_frame_cleaned",
                edges=canny_edges,
                original_img=canny.original_img(),
                edge_representation="edge",
                edge_contour_tracing=True,
                verbose=True,
            )
        if frame_cleanup["auto"]["removed_pixels"] > 0:
            auto = self._EdgeCandidate(
                image_path=self.image_path,
                save_dir=self.edge_dir / "_candidate_auto_frame_cleaned",
                edges=auto_edges,
                original_img=auto.original_img(),
                edge_representation=auto.edge_representation(),
                edge_contour_tracing=auto.edge_representation() == "edge",
                verbose=True,
            )
        if frame_cleanup["foreground_contour"]["removed_pixels"] > 0:
            foreground = self._EdgeCandidate(
                image_path=self.image_path,
                save_dir=self.edge_dir / "_candidate_foreground_contour_frame_cleaned",
                edges=foreground_edges,
                original_img=foreground.original_img(),
                edge_representation=foreground.edge_representation(),
                edge_contour_tracing=True,
                verbose=True,
            )
        if frame_cleanup["stroke_mask"]["removed_pixels"] > 0:
            stroke = self._EdgeCandidate(
                image_path=self.image_path,
                save_dir=self.edge_dir / "_candidate_stroke_mask_frame_cleaned",
                edges=stroke_edges,
                original_img=stroke.original_img(),
                edge_representation=stroke.edge_representation(),
                edge_contour_tracing=stroke.edge_representation() == "edge",
                verbose=True,
            )
        subject_mask_boundary_edges = self._extract_subject_mask_boundary_edges(canny.original_img())
        color_subject_boundary_edges = self._extract_color_subject_boundary_edges(canny.original_img())
        canny_count = int(np.count_nonzero(canny_edges))
        auto_count = int(np.count_nonzero(auto_edges))
        image_pixel_count = int(auto_edges.shape[0] * auto_edges.shape[1])
        auto_foreground_ratio = float(auto_count / max(1, image_pixel_count))

        distance_px = max(1, int(self.close_parallel_edge_distance_px))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (distance_px * 2 + 1, distance_px * 2 + 1),
        )
        auto_near = cv2.dilate(auto_edges, kernel, iterations=1) > 0
        extra_canny = (canny_edges > 0) & (~(auto_edges > 0))
        extra_count = int(np.count_nonzero(extra_canny))
        close_extra_count = int(np.count_nonzero(extra_canny & auto_near))
        close_extra_ratio = float(close_extra_count / max(1, extra_count))
        canny_to_auto_ratio = float(canny_count / max(1, auto_count))
        auto_to_canny_ratio = float(auto_count / max(1, canny_count))
        auto_is_stroke = auto.edge_representation() == "stroke_mask"
        auto_is_edge = auto.edge_representation() == "edge"
        auto_stroke_mask_valid = (
            auto_is_stroke
            and auto_count > 0
            and auto_foreground_ratio <= self.max_stroke_mask_foreground_ratio
            and auto_to_canny_ratio <= self.max_stroke_mask_to_canny_ratio
        )
        auto_dense_filled_mask = (
            auto_is_stroke
            and auto_count > 0
            and auto_foreground_ratio >= max(self.max_stroke_mask_foreground_ratio * 2.0, 0.12)
            and auto_to_canny_ratio >= 2.0
        )
        auto_edge_sparse_valid = (
            auto_is_edge
            and auto_count > 0
            and canny_count > 0
            and auto_to_canny_ratio <= self.max_sparse_auto_to_canny_ratio
            and close_extra_ratio >= self.close_parallel_edge_ratio_threshold
        )
        canny_double_edge_risk = (
            auto_is_stroke
            and extra_count > 0
            and close_extra_ratio >= self.close_parallel_edge_ratio_threshold
        )

        use_auto = (
            (auto_stroke_mask_valid or auto_edge_sparse_valid)
            and extra_count > 0
            and close_extra_ratio >= self.close_parallel_edge_ratio_threshold
        )

        use_merged_canny = (
            not use_auto
            and auto_is_stroke
            and not auto_stroke_mask_valid
            and extra_count > 0
            and close_extra_ratio >= self.close_parallel_edge_ratio_threshold
        )
        gap_closed_edges = self._close_small_edge_gaps(canny_edges)
        gap_closed_count = int(np.count_nonzero(gap_closed_edges))
        gap_closed_added_pixels = int(max(0, gap_closed_count - canny_count))
        gap_closed_growth_ratio = float(gap_closed_added_pixels / max(1, canny_count))
        gap_closed_valid = gap_closed_count > 0 and gap_closed_growth_ratio <= 0.25
        merged_canny_edges = None
        merged_canny_count = 0
        merged_canny_retention_ratio = 0.0
        merged_canny_valid = False
        merged_canny_reject_reason = ""
        if use_merged_canny:
            merged_canny_edges = self._merge_close_parallel_edges(canny_edges)
            merged_canny_count = int(np.count_nonzero(merged_canny_edges))
            merged_canny_retention_ratio = float(merged_canny_count / max(1, canny_count))
            merged_canny_valid = (
                merged_canny_count > 0
                and merged_canny_retention_ratio >= self.min_merged_canny_retention_ratio
            )
            if not merged_canny_valid:
                merged_canny_reject_reason = "merged_canny_rejected_erases_major_structure"
                use_merged_canny = False

        candidate_scores = []
        candidate_scores.append(
            self._edge_candidate_quality(
                name="canny",
                edges=canny_edges,
                representation=canny.edge_representation(),
                baseline_edge_pixels=canny_count,
                edge_contour_tracing=True,
            )
        )
        candidate_scores.append(
            self._edge_candidate_quality(
                name="auto",
                edges=auto_edges,
                representation=auto.edge_representation(),
                baseline_edge_pixels=canny_count,
                edge_contour_tracing=auto.edge_representation() == "edge",
            )
        )
        candidate_scores.append(
            self._edge_candidate_quality(
                name="foreground_contour",
                edges=foreground_edges,
                representation=foreground.edge_representation(),
                baseline_edge_pixels=canny_count,
                edge_contour_tracing=True,
            )
        )
        candidate_scores.append(
            self._edge_candidate_quality(
                name="stroke_mask",
                edges=stroke_edges,
                representation=stroke.edge_representation(),
                baseline_edge_pixels=canny_count,
                edge_contour_tracing=False,
            )
        )
        if subject_mask_boundary_edges is not None:
            self._write_image(
                self.edge_dir / "_candidate_subject_mask_boundary" / "repair_fig_edges.png",
                subject_mask_boundary_edges.astype(np.uint8) * 255,
            )
            candidate_scores.append(
                self._edge_candidate_quality(
                    name="subject_mask_boundary",
                    edges=subject_mask_boundary_edges,
                    representation="subject_mask_boundary",
                    baseline_edge_pixels=canny_count,
                    edge_contour_tracing=True,
                )
            )
        if color_subject_boundary_edges is not None:
            self._write_image(
                self.edge_dir / "_candidate_color_subject_boundary" / "repair_fig_edges.png",
                color_subject_boundary_edges.astype(np.uint8) * 255,
            )
            candidate_scores.append(
                self._edge_candidate_quality(
                    name="color_subject_boundary",
                    edges=color_subject_boundary_edges,
                    representation="color_subject_boundary",
                    baseline_edge_pixels=canny_count,
                    edge_contour_tracing=True,
                )
            )
        if gap_closed_valid:
            candidate_scores.append(
                self._edge_candidate_quality(
                    name="canny_gap_closed",
                    edges=gap_closed_edges,
                    representation="gap_closed_edge",
                    baseline_edge_pixels=canny_count,
                    edge_contour_tracing=True,
                )
            )
        if merged_canny_valid and merged_canny_edges is not None:
            candidate_scores.append(
                self._edge_candidate_quality(
                    name="canny_merged",
                    edges=merged_canny_edges,
                    representation="merged_canny_centerline",
                    baseline_edge_pixels=canny_count,
                    edge_contour_tracing=False,
                )
            )
        self._apply_subject_boundary_retention_guard(candidate_scores)

        self._write_json(
            self.edge_dir / "edge_candidate_scores.json",
            {
                "selection_goal": "topology_preserving_centerline_input",
                "candidates": candidate_scores,
            },
        )

        quality_selected = self._select_quality_candidate(candidate_scores)
        use_quality_selected = (
            quality_selected is not None
            and not auto_dense_filled_mask
            and quality_selected["name"] not in ("canny", "auto")
            and float(quality_selected.get("score", 0.0)) >= float(candidate_scores[0].get("score", 0.0)) + 0.08
        )

        selected = auto if (use_auto or canny_double_edge_risk or auto_dense_filled_mask) else canny
        selected_edges = auto_edges if (use_auto or canny_double_edge_risk or auto_dense_filled_mask) else canny_edges
        if use_merged_canny and merged_canny_edges is not None:
            selected_edges = merged_canny_edges
            selected = self._EdgeCandidate(
                image_path=self.image_path,
                save_dir=self.edge_dir / "_candidate_canny_merged",
                edges=selected_edges,
                original_img=canny.original_img(),
                edge_representation="merged_canny_centerline",
                edge_contour_tracing=False,
                verbose=True,
            )
        elif use_quality_selected and quality_selected is not None:
            quality_name = str(quality_selected["name"])
            if quality_name == "foreground_contour":
                selected = foreground
                selected_edges = foreground_edges
            elif quality_name == "stroke_mask":
                selected = stroke
                selected_edges = stroke_edges
            elif quality_name == "canny_gap_closed":
                selected_edges = gap_closed_edges
                selected = self._EdgeCandidate(
                    image_path=self.image_path,
                    save_dir=self.edge_dir / "_candidate_canny_gap_closed",
                    edges=selected_edges,
                    original_img=canny.original_img(),
                    edge_representation="gap_closed_edge",
                    edge_contour_tracing=True,
                    verbose=True,
                )
            elif quality_name == "subject_mask_boundary" and subject_mask_boundary_edges is not None:
                selected_edges = subject_mask_boundary_edges
                selected = self._EdgeCandidate(
                    image_path=self.image_path,
                    save_dir=self.edge_dir / "_candidate_subject_mask_boundary",
                    edges=selected_edges,
                    original_img=canny.original_img(),
                    edge_representation="subject_mask_boundary",
                    edge_contour_tracing=True,
                    verbose=True,
                )
            elif quality_name == "color_subject_boundary" and color_subject_boundary_edges is not None:
                selected_edges = color_subject_boundary_edges
                selected = self._EdgeCandidate(
                    image_path=self.image_path,
                    save_dir=self.edge_dir / "_candidate_color_subject_boundary",
                    edges=selected_edges,
                    original_img=canny.original_img(),
                    edge_representation="color_subject_boundary",
                    edge_contour_tracing=True,
                    verbose=True,
                )
            elif quality_name == "canny_merged" and merged_canny_edges is not None:
                selected_edges = merged_canny_edges
                selected = self._EdgeCandidate(
                    image_path=self.image_path,
                    save_dir=self.edge_dir / "_candidate_canny_merged",
                    edges=selected_edges,
                    original_img=canny.original_img(),
                    edge_representation="merged_canny_centerline",
                    edge_contour_tracing=False,
                    verbose=True,
                )

        selected_mode = "auto" if (use_auto or canny_double_edge_risk or auto_dense_filled_mask) else ("canny_merged" if use_merged_canny else "canny")
        if use_quality_selected and quality_selected is not None:
            selected_mode = str(quality_selected["name"])
            selection_reason = "candidate_quality_score_preferred"
        elif auto_edge_sparse_valid and use_auto:
            selection_reason = "sparse_auto_edge_preferred_over_thick_canny"
        elif use_auto:
            selection_reason = "close_parallel_double_edges_detected"
        elif auto_dense_filled_mask:
            selection_reason = "dense_filled_stroke_mask_prefers_solid_topology"
        elif canny_double_edge_risk:
            selection_reason = "force_stroke_mask_to_avoid_canny_double_edges"
        elif use_merged_canny:
            selection_reason = "dense_auto_invalid_close_parallel_canny_merged"
        elif merged_canny_reject_reason:
            selection_reason = merged_canny_reject_reason
        else:
            selection_reason = "canny_kept_no_close_parallel_double_edge"
        diagnostics = {
            "selected_edge_mode": selected_mode,
            "selected_edge_representation": selected.edge_representation(),
            "reason": selection_reason,
            "canny_edge_pixels": canny_count,
            "auto_edge_pixels": auto_count,
            "auto_edge_representation": auto.edge_representation(),
            "auto_foreground_ratio": auto_foreground_ratio,
            "auto_to_canny_ratio": auto_to_canny_ratio,
            "auto_stroke_mask_valid": auto_stroke_mask_valid,
            "auto_dense_filled_mask": auto_dense_filled_mask,
            "auto_edge_sparse_valid": auto_edge_sparse_valid,
            "max_sparse_auto_to_canny_ratio": self.max_sparse_auto_to_canny_ratio,
            "canny_double_edge_risk": canny_double_edge_risk,
            "max_stroke_mask_foreground_ratio": self.max_stroke_mask_foreground_ratio,
            "max_stroke_mask_to_canny_ratio": self.max_stroke_mask_to_canny_ratio,
            "extra_canny_pixels": extra_count,
            "close_extra_canny_pixels": close_extra_count,
            "close_extra_ratio": close_extra_ratio,
            "canny_to_auto_ratio": canny_to_auto_ratio,
            "close_parallel_edge_distance_px": distance_px,
            "close_parallel_edge_ratio_threshold": self.close_parallel_edge_ratio_threshold,
            "close_parallel_merge_distance_px": self.close_parallel_merge_distance_px,
            "merged_canny_edge_pixels": merged_canny_count,
            "merged_canny_retention_ratio": merged_canny_retention_ratio,
            "min_merged_canny_retention_ratio": self.min_merged_canny_retention_ratio,
            "merged_canny_valid": merged_canny_valid,
            "merged_canny_reject_reason": merged_canny_reject_reason,
            "gap_closed_edge_pixels": gap_closed_count,
            "gap_closed_added_pixels": gap_closed_added_pixels,
            "gap_closed_growth_ratio": gap_closed_growth_ratio,
            "gap_closed_valid": gap_closed_valid,
            "outer_square_frame_cleanup": {
                name: {
                    key: value
                    for key, value in item.items()
                    if key != "edges"
                }
                for name, item in frame_cleanup.items()
            },
            "candidate_score_path": str(self.edge_dir / "edge_candidate_scores.json"),
            "candidate_scores": candidate_scores,
            "quality_selected_candidate": quality_selected,
        }
        self._write_json(self.edge_dir / "edge_selection_diagnostics.json", diagnostics)
        self.edge_selection_diagnostics = diagnostics
        print(
            "[GeometryDrivenParameterizer] edge selection: "
            f"{diagnostics['selected_edge_mode']} "
            f"({diagnostics['reason']}), "
            f"canny={canny_count}, auto={auto_count}, "
            f"close_extra_ratio={close_extra_ratio:.3f}"
        )
        return selected

    def _close_small_edge_gaps(self, edges: Any) -> Any:
        import cv2
        import numpy as np

        binary = (edges > 0).astype(np.uint8)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        closed_h = cv2.morphologyEx(binary * 255, cv2.MORPH_CLOSE, h_kernel, iterations=1)
        closed_v = cv2.morphologyEx(binary * 255, cv2.MORPH_CLOSE, v_kernel, iterations=1)
        closed = cv2.bitwise_or(closed_h, closed_v)
        return (closed > 0).astype(np.uint8)

    def _remove_outer_square_frame_edges(self, edges: Any) -> Dict[str, Any]:
        """Remove screenshot-like outer square frames while preserving inner FSS edges."""
        import cv2
        import numpy as np

        binary = (edges > 0).astype(np.uint8)
        cleaned = binary.copy()
        height, width = binary.shape[:2]
        edge_pixels = int(np.count_nonzero(binary))
        result: Dict[str, Any] = {
            "edges": cleaned,
            "removed_pixels": 0,
            "frame_detected": False,
            "reason": "no_candidate_contours",
            "frame_bbox": [],
            "frame_aspect_ratio": 0.0,
            "frame_span_x": 0.0,
            "frame_span_y": 0.0,
            "frame_vertices": 0,
            "frame_area_ratio": 0.0,
            "removed_edge_ratio": 0.0,
        }
        if edge_pixels <= 0 or height <= 0 or width <= 0:
            result["reason"] = "empty_edges"
            return result

        contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return result

        image_area = float(max(1, height * width))
        best = None
        best_score = -1.0
        for contour in contours:
            if len(contour) < 4:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            span_x = float(bw / max(1, width))
            span_y = float(bh / max(1, height))
            aspect = float(bw / max(1, bh))
            area = float(abs(cv2.contourArea(contour)))
            area_ratio = float(area / image_area)
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            vertices = int(len(approx))

            near_border = (
                x <= int(0.08 * width)
                and y <= int(0.08 * height)
                and x + bw >= int(0.92 * width)
                and y + bh >= int(0.92 * height)
            )
            square_like = 0.85 <= aspect <= 1.18
            large_outer = span_x >= 0.82 and span_y >= 0.82 and area_ratio >= 0.40
            simple_frame = vertices <= 8
            if not (near_border and square_like and large_outer and simple_frame):
                continue

            rectangularity = float(area / max(1.0, bw * bh))
            score = span_x + span_y + rectangularity - 0.03 * abs(vertices - 4)
            if score > best_score:
                best_score = score
                best = {
                    "contour": contour,
                    "x": int(x),
                    "y": int(y),
                    "bw": int(bw),
                    "bh": int(bh),
                    "span_x": span_x,
                    "span_y": span_y,
                    "aspect": aspect,
                    "area_ratio": area_ratio,
                    "vertices": vertices,
                }

        if best is None:
            result["reason"] = "no_large_near_square_outer_frame"
            return result

        x = int(best["x"])
        y = int(best["y"])
        bw = int(best["bw"])
        bh = int(best["bh"])
        band = max(3, int(round(min(bw, bh) * 0.012)))
        contour_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(contour_mask, [best["contour"]], -1, 1, thickness=max(2, band // 2))

        frame_band = np.zeros_like(binary, dtype=np.uint8)
        x0 = max(0, x - band)
        y0 = max(0, y - band)
        x1 = min(width, x + bw + band)
        y1 = min(height, y + bh + band)
        frame_band[y0 : min(height, y + band), x0:x1] = 1
        frame_band[max(0, y + bh - band) : y1, x0:x1] = 1
        frame_band[y0:y1, x0 : min(width, x + band)] = 1
        frame_band[y0:y1, max(0, x + bw - band) : x1] = 1

        remove_mask = (contour_mask > 0) & (frame_band > 0) & (binary > 0)
        removed_pixels = int(np.count_nonzero(remove_mask))
        removed_ratio = float(removed_pixels / max(1, edge_pixels))
        if removed_pixels <= 0:
            result.update(
                {
                    "reason": "candidate_found_but_no_frame_pixels_removed",
                    "frame_bbox": [x, y, bw, bh],
                    "frame_aspect_ratio": float(best["aspect"]),
                    "frame_span_x": float(best["span_x"]),
                    "frame_span_y": float(best["span_y"]),
                    "frame_vertices": int(best["vertices"]),
                    "frame_area_ratio": float(best["area_ratio"]),
                }
            )
            return result
        if removed_ratio > 0.35:
            result.update(
                {
                    "reason": "candidate_rejected_would_remove_too_many_edges",
                    "frame_bbox": [x, y, bw, bh],
                    "frame_aspect_ratio": float(best["aspect"]),
                    "frame_span_x": float(best["span_x"]),
                    "frame_span_y": float(best["span_y"]),
                    "frame_vertices": int(best["vertices"]),
                    "frame_area_ratio": float(best["area_ratio"]),
                    "removed_pixels": removed_pixels,
                    "removed_edge_ratio": removed_ratio,
                }
            )
            return result

        cleaned[remove_mask] = 0
        result.update(
            {
                "edges": cleaned,
                "removed_pixels": removed_pixels,
                "frame_detected": True,
                "reason": "removed_screenshot_like_outer_square_frame",
                "frame_bbox": [x, y, bw, bh],
                "frame_aspect_ratio": float(best["aspect"]),
                "frame_span_x": float(best["span_x"]),
                "frame_span_y": float(best["span_y"]),
                "frame_vertices": int(best["vertices"]),
                "frame_area_ratio": float(best["area_ratio"]),
                "removed_edge_ratio": removed_ratio,
            }
        )
        return result

    def _extract_subject_mask_boundary_edges(self, image: Any) -> Optional[Any]:
        import cv2
        import numpy as np

        if image is None:
            return None
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] < 3:
            return None

        border = np.concatenate([img[0, :, :3], img[-1, :, :3], img[:, 0, :3], img[:, -1, :3]], axis=0)
        bg = np.median(border.astype(np.float32), axis=0)
        dist = np.linalg.norm(img[:, :, :3].astype(np.float32) - bg, axis=2)
        threshold = max(18.0, float(np.percentile(dist, 80.0)) * 0.35)
        mask = (dist > threshold).astype(np.uint8)

        image_pixels = int(mask.size)
        foreground_ratio = float(np.count_nonzero(mask) / max(1, image_pixels))
        if foreground_ratio < 0.002 or foreground_ratio > 0.55:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark = (otsu == 0).astype(np.uint8)
            light = (otsu > 0).astype(np.uint8)
            mask = dark if np.count_nonzero(dark) <= np.count_nonzero(light) else light
            foreground_ratio = float(np.count_nonzero(mask) / max(1, image_pixels))
            if foreground_ratio < 0.002 or foreground_ratio > 0.55:
                return None

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        if n_labels <= 1:
            return None
        min_area = max(16, int(0.00005 * image_pixels))
        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for label in range(1, n_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                cleaned[labels == label] = 255
        if int(np.count_nonzero(cleaned)) == 0:
            return None

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary = np.zeros_like(cleaned, dtype=np.uint8)
        if not contours:
            return None
        cv2.drawContours(boundary, contours, -1, 255, thickness=1)
        return (boundary > 0).astype(np.uint8)

    def _extract_color_subject_boundary_edges(self, image: Any) -> Optional[Any]:
        """Extract saturated colored FSS conductor boundaries while ignoring gray screenshot frames."""
        import cv2
        import numpy as np

        if image is None:
            return None
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] < 3:
            return None

        bgr = img[:, :, :3]
        height, width = bgr.shape[:2]
        image_pixels = int(height * width)
        if image_pixels <= 0:
            return None

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)
        border_sat = np.concatenate(
            [saturation[0, :], saturation[-1, :], saturation[:, 0], saturation[:, -1]],
            axis=0,
        )
        sat_threshold = max(35.0, float(np.median(border_sat)) + 25.0)
        mask = ((saturation >= sat_threshold) & (value >= 35.0)).astype(np.uint8)

        foreground_ratio = float(np.count_nonzero(mask) / max(1, image_pixels))
        if foreground_ratio < 0.002 or foreground_ratio > 0.55:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        if n_labels <= 1:
            return None

        min_area = max(32, int(0.0001 * image_pixels))
        cleaned = np.zeros_like(mask, dtype=np.uint8)
        kept_area = 0
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label] = 255
                kept_area += area
        kept_ratio = float(kept_area / max(1, image_pixels))
        if kept_ratio < 0.002 or kept_ratio > 0.55:
            return None

        contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        boundary = np.zeros_like(cleaned, dtype=np.uint8)
        min_contour_area = max(8.0, 0.00001 * float(image_pixels))
        for contour in contours:
            if abs(float(cv2.contourArea(contour))) >= min_contour_area:
                cv2.drawContours(boundary, [contour], -1, 255, thickness=1)

        if int(np.count_nonzero(boundary)) == 0:
            return None
        cleanup = self._remove_outer_square_frame_edges((boundary > 0).astype(np.uint8))
        return (cleanup["edges"] > 0).astype(np.uint8)

    def _edge_candidate_quality(
        self,
        name: str,
        edges: Any,
        representation: str,
        baseline_edge_pixels: int,
        edge_contour_tracing: bool,
    ) -> Dict[str, Any]:
        import cv2
        import numpy as np

        binary = (edges > 0).astype(np.uint8)
        pixel_count = int(np.count_nonzero(binary))
        image_pixels = int(binary.shape[0] * binary.shape[1])
        foreground_ratio = float(pixel_count / max(1, image_pixels))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, n_labels)]
        component_count = len(areas)
        small_limit = max(3, int(0.00002 * image_pixels))
        small_components = int(sum(1 for area in areas if area <= small_limit))
        small_component_ratio = float(small_components / max(1, component_count))
        largest_component_ratio = float((max(areas) if areas else 0) / max(1, pixel_count))
        endpoints, junctions = self._skeleton_endpoint_junction_counts(binary)
        endpoint_density = float(endpoints / max(1, pixel_count))
        junction_density = float(junctions / max(1, pixel_count))
        edge_to_baseline_ratio = float(pixel_count / max(1, baseline_edge_pixels))

        density_penalty = 0.0
        if representation == "stroke_mask":
            density_penalty = max(0.0, foreground_ratio - self.max_stroke_mask_foreground_ratio) * 5.0
        else:
            density_penalty = max(0.0, foreground_ratio - 0.12) * 4.0
        sparse_penalty = max(0.0, 0.0005 - foreground_ratio) * 20.0
        fragmentation_penalty = min(0.45, component_count / 350.0) + 0.35 * small_component_ratio
        endpoint_penalty = min(0.25, endpoint_density * 25.0)
        junction_penalty = min(0.20, junction_density * 50.0)
        ratio_penalty = max(0.0, edge_to_baseline_ratio - 1.35) * 0.35
        continuity_reward = min(0.18, largest_component_ratio * 0.18)

        score = 1.0 - density_penalty - sparse_penalty - fragmentation_penalty - endpoint_penalty - junction_penalty - ratio_penalty + continuity_reward
        valid = pixel_count > 0 and foreground_ratio < 0.45 and edge_to_baseline_ratio < 8.0
        if not valid:
            score -= 1.0

        return {
            "name": str(name),
            "representation": str(representation),
            "edge_contour_tracing": bool(edge_contour_tracing),
            "valid": bool(valid),
            "score": float(score),
            "edge_pixels": pixel_count,
            "foreground_ratio": foreground_ratio,
            "edge_to_baseline_ratio": edge_to_baseline_ratio,
            "component_count": component_count,
            "small_component_count": small_components,
            "small_component_ratio": small_component_ratio,
            "largest_component_ratio": largest_component_ratio,
            "endpoint_count": int(endpoints),
            "junction_count": int(junctions),
            "endpoint_density": endpoint_density,
            "junction_density": junction_density,
            "penalties": {
                "density": float(density_penalty),
                "sparse": float(sparse_penalty),
                "fragmentation": float(fragmentation_penalty),
                "endpoint": float(endpoint_penalty),
                "junction": float(junction_penalty),
                "ratio": float(ratio_penalty),
            },
            "continuity_reward": float(continuity_reward),
        }

    @staticmethod
    def _select_quality_candidate(candidate_scores: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        valid = [item for item in candidate_scores if item.get("valid")]
        if not valid:
            return None
        return max(valid, key=lambda item: float(item.get("score", -999.0)))

    @staticmethod
    def _apply_subject_boundary_retention_guard(candidate_scores: List[Dict[str, Any]]) -> None:
        """Prevent subject-mask contour from erasing valid preprocessed FSS edges."""
        canny = next((item for item in candidate_scores if item.get("name") == "canny"), None)
        subject = next((item for item in candidate_scores if item.get("name") == "subject_mask_boundary"), None)
        if canny is None or subject is None or not subject.get("valid", False):
            return

        retention = float(subject.get("edge_to_baseline_ratio", 0.0) or 0.0)
        canny_junction_density = float(canny.get("junction_density", 0.0) or 0.0)
        canny_component_count = int(canny.get("component_count", 0) or 0)
        canny_foreground_ratio = float(canny.get("foreground_ratio", 0.0) or 0.0)
        canny_largest_component_ratio = float(canny.get("largest_component_ratio", 0.0) or 0.0)

        # In already-preprocessed FSS images, a very sparse subject-mask outer
        # boundary can delete slots/internal traces.  Permit it only if it keeps
        # enough source edge information, or the original edge graph clearly
        # looks pathological.
        baseline_pathological = (
            canny_foreground_ratio >= 0.16
            or (
                canny_component_count >= 20
                and canny_largest_component_ratio <= 0.35
                and canny_junction_density >= 0.05
            )
        )
        min_retention = 0.20
        if retention < min_retention and not baseline_pathological:
            subject["valid"] = False
            subject["score_before_retention_guard"] = subject.get("score", 0.0)
            subject["score"] = float(subject.get("score", 0.0) or 0.0) - 2.0
            subject["reject_reason"] = "subject_mask_boundary_erases_preprocessed_fss_edges"
            subject["min_subject_boundary_to_canny_ratio"] = min_retention
            subject["baseline_pathological"] = bool(baseline_pathological)

    @staticmethod
    def _skeleton_endpoint_junction_counts(binary: Any) -> Tuple[int, int]:
        import cv2
        import numpy as np

        mask = (binary > 0).astype(np.uint8)
        try:
            from skimage.morphology import skeletonize

            skeleton = skeletonize(mask > 0).astype(np.uint8)
        except Exception:
            skeleton = GeometryDrivenParameterizer._skeletonize_binary(mask > 0).astype(np.uint8)

        if int(np.count_nonzero(skeleton)) == 0:
            return 0, 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
        active = skeleton > 0
        degree = neighbor_count - 1
        endpoints = int(np.count_nonzero(active & (degree == 1)))
        junctions = int(np.count_nonzero(active & (degree >= 3)))
        return endpoints, junctions

    def _write_standard_topology_fallback(self, params: Any, parameterizer: Any, reason: str) -> Path:
        """Use standard NewParams/VTracer JSON when geometry compression is unsafe.

        中文说明：
        如果 standard 的中心线拓扑已经碎成很多 component，继续 B-spline/primitive
        压缩会放大错误。此时直接输出 standard JSON，保持旧版本行为。
        """
        print(
            "[GeometryDrivenParameterizer] standard topology fallback: "
            f"{reason}. B-spline primitive compression skipped."
        )
        self.last_status = {
            "fallback": True,
            "fallback_reason": reason,
            "actual_backend": "standard_NewParams_CurveParameterizer",
        }
        import shutil

        for stale_dir in (self.contour_dir, self.bspline_dir, self.primitive_dir, self.preview_dir):
            if stale_dir.exists():
                shutil.rmtree(stale_dir, ignore_errors=True)
        for path in (self.stage_dir, self.contour_dir, self.bspline_dir, self.primitive_dir, self.preview_dir):
            path.mkdir(parents=True, exist_ok=True)

        json_path = parameterizer.save_json(self.save_dir / "curve_parameterization.json")
        self._mark_standard_json_as_fallback(json_path, reason)
        try:
            visual_path = parameterizer.visualize(self.preview_dir / "geometry_primitives_preview.png")
            parameterizer.visualize(self.preview_dir / "standard_topology_fallback_visualization.png")
        except Exception as exc:
            visual_path = ""
            print(f"[GeometryDrivenParameterizer] fallback visualization skipped: {exc}")

        metrics = parameterizer.metrics()
        fallback_payload = {
            "fallback_reason": reason,
            "fallback_backend": "standard_NewParams_CurveParameterizer",
            "edge_representation": params.edge_representation() if hasattr(params, "edge_representation") else "",
            "standard_json": str(json_path),
            "standard_visualization": str(visual_path),
            "trace_image_path": str(parameterizer.trace_image_path() or ""),
            "svg_path": str(parameterizer.svg_path() or ""),
            "metrics_path": str(parameterizer.metrics_path() or ""),
            "component_count": len(parameterizer.results()),
            "metrics": metrics,
        }
        self._write_json(self.stage_dir / "standard_topology_fallback.json", fallback_payload)
        self._write_json(
            self.stage_dir / "geometry_primitives_metrics.json",
            {
                "fallback": True,
                "fallback_reason": reason,
                "component_count": len(parameterizer.results()),
                "primitive_count": metrics.get("total_semantic_segments", 0),
                "primitive_by_type": {
                    "line": metrics.get("total_semantic_line", 0),
                    "arc": metrics.get("total_semantic_arc", 0),
                    "spline": metrics.get("total_semantic_spline", 0),
                },
                "source": "standard_NewParams_CurveParameterizer",
            },
        )
        return json_path

    def _mark_standard_json_as_fallback(self, json_path: Path, reason: str) -> None:
        """Annotate fallback status without changing the standard component schema.

        中文说明：
        components / segments 仍然完全来自旧版 standard pipeline。这里只增加顶层诊断字段，
        方便从输出文件判断 geometry_primitives 是否实际回退。
        """
        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["pipeline_backend"] = "standard_NewParams_CurveParameterizer"
            payload["pipeline_fallback"] = True
            payload["pipeline_fallback_reason"] = reason
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[GeometryDrivenParameterizer] fallback JSON annotation skipped: {exc}")

    def _should_use_solid_mask_topology(self, params: Any, mask: Any) -> bool:
        """Detect filled PEC/slot-like masks that must not be skeletonized.

        Dense stroke masks in this project often mean the repaired image already
        contains the metal area as a filled foreground.  Centerline extraction
        would turn that area into a medial-axis graph and destroy topology, so
        those cases are routed to an outer-contour + holes representation.
        """
        import numpy as np

        representation = params.edge_representation() if hasattr(params, "edge_representation") else ""
        if representation != "stroke_mask":
            return False

        binary = np.asarray(mask) > 0
        foreground_ratio = float(np.count_nonzero(binary) / max(1, binary.size))
        dense_limit = max(self.max_stroke_mask_foreground_ratio * 2.0, 0.12)
        dense_candidate = foreground_ratio >= dense_limit
        if not dense_candidate:
            return False

        print(
            "[GeometryDrivenParameterizer] solid-mask topology selected: "
            f"foreground_ratio={foreground_ratio:.4f}, representation={representation}. "
            "Skip centerline skeletonization."
        )
        return True

    def _should_apply_patch_topology_override(self) -> bool:
        """Only let mask-first skeletonization replace weak edge candidates.

        已经很干净的 auto/foreground edge 往往保留了槽线和外形细节；
        如果此时再从填充导体 mask 取骨架，会把贴片细节压缩成少量中轴线。
        因此 patch_topology 只作为不可靠边缘路径的保守回退。
        """
        mode = str(self.edge_selection_diagnostics.get("selected_edge_mode", ""))
        representation = str(self.edge_selection_diagnostics.get("selected_edge_representation", ""))
        reason = str(self.edge_selection_diagnostics.get("reason", ""))

        if representation == "stroke_mask":
            return False
        if mode in {"auto", "foreground_contour", "subject_mask_boundary"} and reason == "sparse_auto_edge_preferred_over_thick_canny":
            return False
        if mode in {"auto", "foreground_contour"} and representation == "edge":
            return False
        return True

    def _write_solid_mask_topology_result(self, params: Any, mask: Any, raw_edges_path: Path) -> Path:
        import cv2
        import numpy as np

        print("[GeometryDrivenParameterizer] stage 2/8: solid mask contour topology extraction")
        stale_fallback_path = self.stage_dir / "standard_topology_fallback.json"
        if stale_fallback_path.exists():
            stale_fallback_path.unlink()

        original_img = params.original_img() if hasattr(params, "original_img") else cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if original_img is None:
            raise FileNotFoundError(f"Cannot read image for solid-mask topology: {self.image_path}")

        solid_mask = self._extract_solid_topology_mask(original_img, mask)
        solid_mask_path = self.edge_dir / "solid_topology_mask.png"
        self._write_image(solid_mask_path, solid_mask)

        components = self._components_from_solid_mask(solid_mask)
        if not components:
            raise ValueError("Solid-mask topology extraction produced no valid components.")

        for component in components:
            self._write_json(
                self.primitive_dir / f"component_{int(component['component_id']):03d}_solid_topology.json",
                component,
            )

        preview_svg_path = self.preview_dir / "geometry_primitives_preview.svg"
        preview_png_path = self.preview_dir / "geometry_primitives_preview.png"
        labels_json_path = self.preview_dir / "primitive_labels.json"
        labels_csv_path = self.preview_dir / "primitive_labels.csv"
        self._write_svg_preview(preview_svg_path, original_img.shape[1], original_img.shape[0], components)
        self._write_png_preview(preview_png_path, original_img, components)
        self._write_primitive_label_files(labels_json_path, labels_csv_path, components)

        aggregate_metrics = self._aggregate_metrics(components)
        aggregate_metrics.update(
            {
                "backend": "solid_mask_topology",
                "hole_count": int(sum(len(component.get("holes", [])) for component in components)),
                "edge_selection": self.edge_selection_diagnostics,
            }
        )

        solid_summary_path = self.stage_dir / "solid_mask_topology.json"
        solid_summary = {
            "backend": "solid_mask_topology",
            "reason": "dense stroke/foreground mask represents filled metal area; centerline skeletonization is skipped",
            "solid_topology_mask": str(solid_mask_path),
            "component_count": len(components),
            "hole_count": aggregate_metrics["hole_count"],
            "edge_selection": self.edge_selection_diagnostics,
        }
        self._write_json(solid_summary_path, solid_summary)

        payload = {
            "schema_version": "2.1",
            "backend": "solid_mask_topology",
            "trace_image_path": str(self.image_path),
            "svg_path": str(preview_svg_path),
            "metrics_path": str(self.stage_dir / "geometry_primitives_metrics.json"),
            "canvas": {
                "width": int(original_img.shape[1]),
                "height": int(original_img.shape[0]),
                "unit": "px",
            },
            "stages": {
                "input_image": str(self.image_path),
                "edge_or_stroke_mask": str(raw_edges_path),
                "solid_topology_mask": str(solid_mask_path),
                "solid_topology_summary": str(solid_summary_path),
                "primitive_dir": str(self.primitive_dir),
                "preview_svg": str(preview_svg_path),
                "preview_png": str(preview_png_path),
                "primitive_labels_json": str(labels_json_path),
                "primitive_labels_csv": str(labels_csv_path),
            },
            "metrics": aggregate_metrics,
            "components": components,
        }
        json_path = self.save_dir / "curve_parameterization.json"
        self._write_json(json_path, payload)
        self._write_json(self.stage_dir / "geometry_primitives_metrics.json", aggregate_metrics)

        self.last_status = {
            "fallback": False,
            "fallback_reason": "",
            "actual_backend": "solid_mask_topology",
        }
        print(
            "[GeometryDrivenParameterizer] done: solid mask topology, "
            f"components={aggregate_metrics['component_count']}, "
            f"holes={aggregate_metrics['hole_count']}, json={json_path}"
        )
        return json_path

    def _extract_solid_topology_mask(self, original_img: Any, fallback_mask: Any) -> Any:
        import cv2
        import numpy as np

        if original_img.ndim == 3:
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_img.copy()

        fallback_binary = (np.asarray(fallback_mask) > 0).astype(np.uint8)
        fallback_ratio = float(np.count_nonzero(fallback_binary) / max(1, fallback_binary.size))
        if fallback_ratio >= max(self.max_stroke_mask_foreground_ratio * 2.0, 0.12):
            # 密集 stroke_mask 已经表达了“填充导体/主体区域”。
            # 对跳过 FSS 的截图，不再从原图重新 Otsu 选黑色轮廓，避免外框和双边缘主导。
            mask = fallback_binary * 255
            mask = self._remove_large_border_frame_pixels(mask)
            mask = self._remove_border_touching_components(mask)
            if np.count_nonzero(mask) == 0:
                mask = fallback_binary * 255
            mask = self._remove_small_mask_components(mask)
            return mask

        unique = np.unique(gray)
        if len(unique) <= 8 and int(unique[-1]) >= 240:
            threshold = max(1, int(unique[-1]) - 8)
            mask = (gray < threshold).astype(np.uint8) * 255
        else:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark = (otsu == 0).astype(np.uint8) * 255
            light = (otsu > 0).astype(np.uint8) * 255
            dark_ratio = float(np.count_nonzero(dark) / max(1, dark.size))
            light_ratio = float(np.count_nonzero(light) / max(1, light.size))
            mask = dark if 0.001 <= dark_ratio <= max(0.70, light_ratio) else (fallback_mask > 0).astype(np.uint8) * 255

        mask = self._remove_large_border_frame_pixels(mask)
        mask = self._remove_border_touching_components(mask)
        if np.count_nonzero(mask) == 0:
            mask = (fallback_mask > 0).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = self._remove_small_mask_components(mask)
        return mask

    def _remove_large_border_frame_pixels(self, mask: Any) -> Any:
        """Erase thin drawing frames without dropping real border-fed metal."""
        import cv2
        import numpy as np

        binary = (mask > 0).astype(np.uint8)
        h, w = binary.shape[:2]
        cleaned = binary.copy()
        scan_y = max(4, int(0.04 * h))
        scan_x = max(4, int(0.04 * w))
        min_h_run = int(0.75 * w)
        min_v_run = int(0.75 * h)

        def longest_run(indices: Any) -> Tuple[int, int, int]:
            if len(indices) == 0:
                return 0, -1, -1
            best_len, best_start, best_end = 1, int(indices[0]), int(indices[0])
            cur_start = int(indices[0])
            prev = int(indices[0])
            for raw in indices[1:]:
                item = int(raw)
                if item == prev + 1:
                    prev = item
                    continue
                cur_len = prev - cur_start + 1
                if cur_len > best_len:
                    best_len, best_start, best_end = cur_len, cur_start, prev
                cur_start = prev = item
            cur_len = prev - cur_start + 1
            if cur_len > best_len:
                best_len, best_start, best_end = cur_len, cur_start, prev
            return best_len, best_start, best_end

        def erase_horizontal(rows: range) -> None:
            for y in rows:
                xs = np.flatnonzero(cleaned[y] > 0)
                run_len, start, end = longest_run(xs)
                if run_len >= min_h_run:
                    y1 = max(0, y - 2)
                    y2 = min(h, y + 3)
                    x1 = max(0, start - 2)
                    x2 = min(w, end + 3)
                    cleaned[y1:y2, x1:x2] = 0

        def erase_vertical(cols: range) -> None:
            for x in cols:
                ys = np.flatnonzero(cleaned[:, x] > 0)
                run_len, start, end = longest_run(ys)
                if run_len >= min_v_run:
                    y1 = max(0, start - 2)
                    y2 = min(h, end + 3)
                    x1 = max(0, x - 2)
                    x2 = min(w, x + 3)
                    cleaned[y1:y2, x1:x2] = 0

        erase_horizontal(range(0, scan_y))
        erase_horizontal(range(max(0, h - scan_y), h))
        erase_vertical(range(0, scan_x))
        erase_vertical(range(max(0, w - scan_x), w))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(cleaned * 255, cv2.MORPH_OPEN, kernel, iterations=1)
        return (cleaned > 0).astype(np.uint8) * 255

    def _remove_border_touching_components(self, mask: Any) -> Any:
        import cv2
        import numpy as np

        binary = (mask > 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        h, w = binary.shape[:2]
        cleaned = np.zeros_like(binary)
        image_area = h * w
        for label in range(1, n_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            is_frame = touches_border and (bw >= int(0.85 * w) or bh >= int(0.85 * h)) and area < int(0.20 * image_area)
            if is_frame:
                continue
            cleaned[labels == label] = 1
        return cleaned.astype(np.uint8) * 255

    def _remove_small_mask_components(self, mask: Any) -> Any:
        import cv2
        import numpy as np

        binary = (mask > 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        h, w = binary.shape[:2]
        min_area = max(16, int(0.00002 * h * w))
        cleaned = np.zeros_like(binary)
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                cleaned[labels == label] = 1
        return cleaned.astype(np.uint8) * 255

    def _components_from_solid_mask(self, mask: Any) -> List[Dict[str, Any]]:
        import cv2
        import numpy as np

        binary = (mask > 0).astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            return []

        h, w = binary.shape[:2]
        min_outer_area = max(25.0, 0.00002 * h * w)
        min_hole_area = max(16.0, 0.00001 * h * w)
        components: List[Dict[str, Any]] = []

        for contour_index, contour in enumerate(contours):
            parent = int(hierarchy[0][contour_index][3])
            if parent != -1:
                continue
            area = float(abs(cv2.contourArea(contour)))
            if area < min_outer_area:
                continue

            outer_points = self._contour_to_sampled_points(contour, closed=True)
            if len(outer_points) < 4:
                continue

            holes: List[Dict[str, Any]] = []
            child = int(hierarchy[0][contour_index][2])
            while child != -1:
                hole_contour = contours[child]
                hole_area = float(abs(cv2.contourArea(hole_contour)))
                if hole_area >= min_hole_area:
                    hole_points = self._contour_to_sampled_points(hole_contour, closed=True)
                    if len(hole_points) >= 4:
                        hole_primitives = self._polygon_line_primitives(
                            hole_points,
                            label_prefix=f"hole-{len(holes) + 1}",
                        )
                        holes.append(
                            {
                                "hole_id": len(holes) + 1,
                                "source_contour_id": int(child),
                                "closed": True,
                                "area_px": hole_area,
                                "bbox": self._bbox(hole_points),
                                "point_count": int(len(hole_points)),
                                "points": hole_points.tolist(),
                                "resampled_points": hole_points.tolist(),
                                "fallback_points": hole_points.tolist(),
                                "primitives": hole_primitives,
                                "segments": hole_primitives,
                            }
                        )
                child = int(hierarchy[0][child][0])

            primitives = self._polygon_line_primitives(outer_points, label_prefix="outer")
            hole_primitive_count = int(sum(len(hole.get("primitives", [])) for hole in holes))
            total_primitive_count = int(len(primitives) + hole_primitive_count)
            total_parameter_count = int(total_primitive_count * 4)
            component = {
                "component_id": len(components) + 1,
                "source_contour_id": int(contour_index),
                "topology": "solid_with_holes" if holes else "solid",
                "closed": True,
                "area_px": area,
                "bbox": self._bbox(outer_points),
                "sampled_point_count": int(len(outer_points)),
                "fallback_points": outer_points.tolist(),
                "resampled_points": outer_points.tolist(),
                "holes": holes,
                "primitives": primitives,
                "segments": primitives,
                "metrics": {
                    "sampled_point_count": int(len(outer_points)),
                    "primitive_count": total_primitive_count,
                    "primitive_by_type": {"line": total_primitive_count},
                    "parameter_count": total_parameter_count,
                    "mean_error_px": 0.0,
                    "max_error_px": 0.0,
                    "compression_ratio": float(total_parameter_count / max(1, len(outer_points) * 2)),
                    "hole_count": int(len(holes)),
                    "hole_primitive_count": hole_primitive_count,
                },
            }
            components.append(component)

        components.sort(key=lambda item: float(item.get("area_px", 0.0)), reverse=True)
        for index, component in enumerate(components, start=1):
            component["component_id"] = index
        return components

    def _contour_to_sampled_points(self, contour: Any, closed: bool) -> Any:
        import numpy as np

        points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        points = self._remove_consecutive_duplicates(points)
        if len(points) < 3:
            return points
        points = self._remove_trailing_closure(points)
        epsilon = max(0.75, self.resample_step_px * 0.45)
        points = self._rdp_closed_array(points, epsilon=epsilon)
        points = self._remove_consecutive_duplicates(points)
        if closed and len(points) >= 3:
            points = np.vstack([points, points[0]])
        return points

    @staticmethod
    def _rdp_closed_array(points: Any, epsilon: float) -> Any:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 4:
            return pts
        center = np.mean(pts, axis=0)
        split_index = int(np.argmax(np.linalg.norm(pts - center, axis=1)))
        rolled = np.vstack([pts[split_index:], pts[: split_index + 1]])
        simplified = GeometryDrivenParameterizer._rdp_array(rolled, epsilon=epsilon)
        simplified = GeometryDrivenParameterizer._remove_trailing_closure(simplified)
        return simplified

    def _polygon_line_primitives(self, points: Any, label_prefix: str = "solid-boundary") -> List[Dict[str, Any]]:
        import numpy as np

        pts = self._remove_trailing_closure(points)
        primitives: List[Dict[str, Any]] = []
        if len(pts) < 3:
            return primitives
        closed_pts = np.vstack([pts, pts[0]])
        for index in range(len(closed_pts) - 1):
            start = closed_pts[index]
            end = closed_pts[index + 1]
            primitives.append(
                {
                    "type": "line",
                    "kind": "line",
                    "start": [float(start[0]), float(start[1])],
                    "end": [float(end[0]), float(end[1])],
                    "points": [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]],
                    "max_error": 0.0,
                    "mean_error": 0.0,
                    "effective_params": 4,
                    "parameter_count": 4,
                    "source_point_count": 2,
                    "segment_id": index + 1,
                    "source_start_index": index,
                    "source_end_index": index + 1,
                    "start_point_count": 2,
                    "visual_label": f"#{index + 1} LINE {label_prefix}",
                }
            )
        return primitives

    def _extract_contour_records(self, contours_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        import numpy as np

        records: List[Dict[str, Any]] = []
        for contour_id, contour_data in contours_dict.items():
            if not isinstance(contour_data, dict):
                continue
            fitting = contour_data.get("fitting", {})
            raw_points = fitting.get("points")
            if raw_points is None:
                continue
            points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
            points = points[np.isfinite(points).all(axis=1)]
            points = self._remove_consecutive_duplicates(points)
            if len(points) < 4:
                continue
            path_length = self._polyline_length(points)
            if path_length < self.min_component_length_px:
                continue
            endpoint_gap = float(np.linalg.norm(points[0] - points[-1]))
            closed = endpoint_gap < max(2.5, self.resample_step_px * 2.0)
            # findContours-derived boundaries are normally closed; snap small gaps.
            # findContours 得到的边界通常应闭合；这里把小缝隙显式吸附闭合。
            if not closed:
                closed = True
            if closed and endpoint_gap > 1e-9:
                points = np.vstack([points, points[0]])
            records.append(
                {
                    "source_contour_id": str(contour_id),
                    "points": points,
                    "closed": bool(closed),
                    "path_length_px": self._polyline_length(points),
                }
            )
        records.sort(key=lambda item: float(item["path_length_px"]), reverse=True)
        return records

    def _extract_vtracer_centerline_records(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract ordered centerline paths from VTracer semantic results.

        中文说明：
        VTracer 的 centerline 模式已经处理了粗线骨架化/中心线提取，所以这里不能再
        从边缘图 findContours。我们直接读取 components[*].resampled_points 作为
        后续 B-spline 中间层的输入。
        """
        import numpy as np

        records: List[Dict[str, Any]] = []
        for component in components:
            raw_points = component.get("resampled_points") or component.get("points") or []
            points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
            points = points[np.isfinite(points).all(axis=1)]
            points = self._remove_consecutive_duplicates(points)
            if len(points) < 4:
                continue

            closed = bool(component.get("closed", False))
            if closed and float(np.linalg.norm(points[0] - points[-1])) > 1e-9:
                points = np.vstack([points, points[0]])

            path_length = self._polyline_length(points)
            if path_length < self.min_component_length_px:
                continue

            records.append(
                {
                    "source_contour_id": str(component.get("component_id", len(records) + 1)),
                    "points": points,
                    "closed": closed,
                    "path_length_px": path_length,
                    "source_debug_path": component.get("debug_path", ""),
                }
            )

        records.sort(key=lambda item: float(item["path_length_px"]), reverse=True)
        return records

    def _fit_unified_bspline(self, points: Any, closed: bool) -> Dict[str, Any]:
        import numpy as np

        pts = self._remove_consecutive_duplicates(points)
        if closed and len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            pts = np.vstack([pts, pts[0]])

        bspline_contour_result = self._fit_bspline_contour_style(pts, closed=closed)
        if bspline_contour_result is not None:
            sampled = bspline_contour_result["sampled_points"]
            curvature = self._curvature(sampled, closed=closed)
            return {
                "sampled_points": sampled,
                "control_points": bspline_contour_result["control_points"],
                "curvature": curvature,
                "method": bspline_contour_result["method"],
                "source": "Rebuild.BSplineContour_style_make_interp_spline",
                "loss": bspline_contour_result["loss"],
                "step": bspline_contour_result["step"],
            }

        try:
            from scipy import interpolate

            fit_pts = pts[:-1] if closed and len(pts) > 3 else pts
            k = min(3, len(fit_pts) - 1)
            if k < 1:
                raise ValueError("not enough points for scipy B-spline")
            distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            length = float(np.sum(distances))
            sample_count = max(16, int(math.ceil(length / max(0.5, self.resample_step_px))))
            tck, _u = interpolate.splprep(
                [fit_pts[:, 0], fit_pts[:, 1]],
                s=max(0.0, self.bspline_smoothing),
                k=k,
                per=bool(closed),
            )
            u_new = np.linspace(0.0, 1.0, sample_count, endpoint=not closed)
            x_new, y_new = interpolate.splev(u_new, tck)
            sampled = np.column_stack([x_new, y_new]).astype(np.float64)
            if closed:
                sampled = np.vstack([sampled, sampled[0]])
            control_points = np.column_stack([tck[1][0], tck[1][1]]).astype(np.float64)
            method = "scipy_splprep"
        except Exception as exc:
            sampled = self._resample_polyline(pts, closed=closed, step=self.resample_step_px)
            control_points = pts
            method = f"linear_resample_fallback:{exc}"

        curvature = self._curvature(sampled, closed=closed)
        return {
            "sampled_points": sampled,
            "control_points": control_points,
            "curvature": curvature,
            "method": method,
        }

    def _guard_bspline_quality(self, raw_points: Any, bspline: Dict[str, Any], closed: bool) -> Dict[str, Any]:
        """Reject global B-splines that distort long straight or right-angle geometry.

        中文说明：
        对 FSS / 天线这类大量直线和直角的结构，全局闭合 B-spline 很容易把直角圆滑掉，
        甚至把长直线拉成弧线。这里用路径长度缩水和到原始中心线的距离做质量门槛；
        不合格时回退为保角的线性重采样，让后续 line fitting 优先贴住原始直线。
        """
        import math as _math
        import numpy as np

        raw = self._remove_consecutive_duplicates(raw_points)
        sampled = self._remove_consecutive_duplicates(bspline.get("sampled_points", raw))
        if len(raw) < 4 or len(sampled) < 4:
            return bspline

        raw_length = self._polyline_length(raw)
        sampled_length = self._polyline_length(sampled)
        if raw_length <= 1e-9:
            return bspline

        shrink_ratio = max(0.0, (raw_length - sampled_length) / raw_length)
        compare_count = min(len(raw), len(sampled))
        rms_error = 0.0
        max_error = 0.0
        if compare_count > 0:
            errors = np.linalg.norm(raw[:compare_count] - sampled[:compare_count], axis=1)
            rms_error = float(_math.sqrt(float(np.mean(errors * errors))))
            max_error = float(np.max(errors))

        diagnostics = {
            "raw_length_px": float(raw_length),
            "bspline_length_px": float(sampled_length),
            "length_shrink_ratio": float(shrink_ratio),
            "rms_error_to_raw_px": float(rms_error),
            "max_error_to_raw_px": float(max_error),
        }

        if (
            shrink_ratio <= self.max_bspline_length_shrink_ratio
            and rms_error <= self.max_bspline_rms_error_px
        ):
            bspline["quality"] = {**diagnostics, "accepted": True}
            return bspline

        fallback = self._resample_polyline(raw, closed=closed, step=1.0)
        if closed and len(fallback) > 1 and np.linalg.norm(fallback[0] - fallback[-1]) > 1e-9:
            fallback = np.vstack([fallback, fallback[0]])

        print(
            "[GeometryDrivenParameterizer] B-spline rejected; using corner-preserving centerline. "
            f"length_shrink={shrink_ratio:.3f}, rms_error={rms_error:.3f}, max_error={max_error:.3f}"
        )
        return {
            "sampled_points": fallback,
            "control_points": raw,
            "curvature": self._curvature(fallback, closed=closed),
            "method": "corner_preserving_linear_resample_after_bspline_reject",
            "source": "vtracer_centerline_quality_guard",
            "loss": float(rms_error * rms_error),
            "step": 1.0,
            "quality": {
                **diagnostics,
                "accepted": False,
                "reason": "bspline_distorts_corner_or_straight_geometry",
            },
        }

    def _fit_bspline_contour_style(self, points: Any, closed: bool) -> Optional[Dict[str, Any]]:
        """Fit B-spline using the strategy from Rebuild/BSplineContour.py.

        中文说明：
        BSplineContour.py 的核心做法是按 step 抽取控制点，然后用
        scipy.interpolate.make_interp_spline 生成拟合点。这里复用这个方法，
        但输入使用 VTracer centerline，而不是重新 findContours 粗线边界。
        """
        import numpy as np

        try:
            from scipy.interpolate import make_interp_spline
        except Exception:
            return None

        pts = self._remove_consecutive_duplicates(points)
        if len(pts) < 4:
            return None

        fit_pts = self._remove_trailing_closure(pts) if closed else pts
        if len(fit_pts) < 4:
            return None

        len_contour = int(len(fit_pts))
        k = min(3, len_contour - 1)
        if k < 1:
            return None

        step = max(3, min(30, int(round(len_contour / 18))))
        control_points = fit_pts.copy()[::step]
        if len(control_points) < k + 1:
            control_points = fit_pts.copy()

        if closed:
            if np.linalg.norm(control_points[0] - control_points[-1]) > 1e-9:
                control_points = np.vstack([control_points, control_points[0]])
            phi = np.linspace(0.0, 2.0 * np.pi, len(control_points))
            phi_new = np.linspace(0.0, 2.0 * np.pi, len_contour + 1)
        else:
            if np.linalg.norm(control_points[-1] - fit_pts[-1]) > 1e-9:
                control_points = np.vstack([control_points, fit_pts[-1]])
            phi = np.linspace(0.0, 1.0, len(control_points))
            phi_new = np.linspace(0.0, 1.0, len_contour)

        try:
            spline = make_interp_spline(phi, control_points, k=min(k, len(control_points) - 1))
            sampled = np.asarray(spline(phi_new), dtype=np.float64).reshape(-1, 2)
        except Exception:
            return None

        if closed and np.linalg.norm(sampled[0] - sampled[-1]) > 1e-9:
            sampled = np.vstack([sampled, sampled[0]])

        compare_count = min(len(sampled), len(pts))
        loss = float(np.mean((sampled[:compare_count] - pts[:compare_count]) ** 2)) if compare_count else 0.0
        return {
            "sampled_points": sampled,
            "control_points": np.asarray(control_points, dtype=np.float64),
            "method": "BSplineContour_make_interp_spline_control_step",
            "loss": loss,
            "step": int(step),
        }

    def _decompose_to_primitives(self, sampled_points: Any, closed: bool) -> List[Dict[str, Any]]:
        import numpy as np

        points = np.asarray(sampled_points, dtype=np.float64).reshape(-1, 2)
        if closed and len(points) > 1:
            points = self._remove_trailing_closure(points)
            points = np.vstack([points, points[0]])
        return self._decompose_open(points, depth=0)

    def _decompose_open(self, points: Any, depth: int) -> List[Dict[str, Any]]:
        import numpy as np

        pts = self._remove_consecutive_duplicates(points)
        if len(pts) < 2:
            return []

        line = self._fit_line_primitive(pts)
        if line["max_error"] <= self.line_tolerance_px:
            return [line]

        arc = self._fit_arc_primitive(pts)
        if (
            arc is not None
            and arc["max_error"] <= self.arc_tolerance_px
            and self._should_accept_arc(arc, line)
        ):
            return [arc]

        if depth >= self.max_decompose_depth or len(pts) <= max(4, self.min_segment_points):
            return [self._fit_residual_spline_primitive(pts)]

        split_idx = self._best_split_index(pts)
        if split_idx <= 1 or split_idx >= len(pts) - 2:
            split_idx = len(pts) // 2

        left = pts[: split_idx + 1]
        right = pts[split_idx:]
        return self._decompose_open(left, depth + 1) + self._decompose_open(right, depth + 1)

    def _fit_line_primitive(self, points: Any) -> Dict[str, Any]:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        start = pts[0]
        end = pts[-1]
        errors = self._line_errors(pts, start, end)
        return {
            "type": "line",
            "kind": "line",
            "start": start.tolist(),
            "end": end.tolist(),
            "points": [start.tolist(), end.tolist()],
            "max_error": float(np.max(errors)) if len(errors) else 0.0,
            "mean_error": float(np.mean(errors)) if len(errors) else 0.0,
            "effective_params": 4,
            "parameter_count": 4,
            "source_point_count": int(len(pts)),
        }

    def _fit_arc_primitive(self, points: Any) -> Optional[Dict[str, Any]]:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 5:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        a = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(pts))])
        b = x * x + y * y
        try:
            cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
        except Exception:
            return None
        radius_sq = float(cx * cx + cy * cy + c)
        if radius_sq <= 1e-9:
            return None
        center = np.array([cx, cy], dtype=np.float64)
        radius = math.sqrt(radius_sq)
        radial_errors = np.abs(np.linalg.norm(pts - center, axis=1) - radius)
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        if chord < 1e-6 or radius > 1e6:
            return None

        angles = np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
        sweep = float(angles[-1] - angles[0])
        if abs(math.degrees(sweep)) < 5.0:
            return None
        clockwise = sweep < 0
        return {
            "type": "arc",
            "kind": "arc",
            "start": pts[0].tolist(),
            "end": pts[-1].tolist(),
            "center": center.tolist(),
            "radius": float(radius),
            "chord_length": float(chord),
            "radius_to_chord_ratio": float(radius / max(chord, 1e-9)),
            "clockwise": bool(clockwise),
            "sweep_deg": float(math.degrees(sweep)),
            "points": pts[[0, len(pts) // 2, -1]].tolist(),
            "max_error": float(np.max(radial_errors)) if len(radial_errors) else 0.0,
            "mean_error": float(np.mean(radial_errors)) if len(radial_errors) else 0.0,
            "effective_params": 7,
            "parameter_count": 7,
            "source_point_count": int(len(pts)),
        }

    def _should_accept_arc(self, arc: Dict[str, Any], line: Dict[str, Any]) -> bool:
        """Decide whether an arc is worth keeping instead of a line.

        中文说明：
        这里有意偏向直线。只有当 arc 的扫角足够明显、半径不是“近似无穷大”，
        并且相比 line 有足够误差收益时，才保留 arc。
        """
        sweep = abs(float(arc.get("sweep_deg", 0.0) or 0.0))
        if sweep < self.arc_min_sweep_deg:
            return False

        source_points = int(arc.get("source_point_count", 0) or 0)
        if source_points < self.arc_min_source_points:
            return False

        radius_to_chord = abs(float(arc.get("radius_to_chord_ratio", 0.0) or 0.0))
        if radius_to_chord > self.arc_max_radius_to_chord_ratio:
            return False

        line_error = float(line.get("max_error", 0.0) or 0.0)
        arc_error = float(arc.get("max_error", 0.0) or 0.0)
        if line_error <= self.line_tolerance_px * 1.75:
            return False
        if line_error <= 1e-9:
            return False

        improvement = (line_error - arc_error) / line_error
        return improvement >= self.arc_min_error_improvement_ratio

    def _fit_residual_spline_primitive(self, points: Any) -> Dict[str, Any]:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        ctrl = self._rdp_array(pts, epsilon=max(0.8, self.residual_spline_tolerance_px))
        if len(ctrl) < 2:
            ctrl = pts[[0, -1]]
        errors = self._nearest_polyline_errors(pts, ctrl)
        return {
            "type": "spline",
            "kind": "spline",
            "degree": 3,
            "control_points": ctrl.tolist(),
            "points": ctrl.tolist(),
            "max_error": float(np.max(errors)) if len(errors) else 0.0,
            "mean_error": float(np.mean(errors)) if len(errors) else 0.0,
            "effective_params": int(len(ctrl) * 2),
            "parameter_count": int(len(ctrl) * 2),
            "source_point_count": int(len(pts)),
        }

    def _best_split_index(self, points: Any) -> int:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        curvature = np.abs(self._curvature(pts, closed=False))
        if len(curvature) >= 5 and float(np.max(curvature)) > 1e-9:
            lo = max(2, len(pts) // 8)
            hi = min(len(pts) - 3, len(pts) - len(pts) // 8)
            if hi > lo:
                local = curvature[lo:hi + 1]
                return int(lo + int(np.argmax(local)))

        line_errors = self._line_errors(pts, pts[0], pts[-1])
        if len(line_errors) >= 5:
            lo = max(2, len(pts) // 8)
            hi = min(len(pts) - 3, len(pts) - len(pts) // 8)
            if hi > lo:
                return int(lo + int(np.argmax(line_errors[lo:hi + 1])))
        return len(pts) // 2

    def _merge_adjacent_lines(self, primitives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        import numpy as np

        merged: List[Dict[str, Any]] = []
        for primitive in primitives:
            if (
                merged
                and primitive.get("type") == "line"
                and merged[-1].get("type") == "line"
            ):
                prev = merged[-1]
                pts = np.asarray([prev["start"], prev["end"], primitive["end"]], dtype=np.float64)
                candidate = self._fit_line_primitive(pts)
                if candidate["max_error"] <= self.line_tolerance_px:
                    candidate["source_point_count"] = int(
                        prev.get("source_point_count", 0) + primitive.get("source_point_count", 0)
                    )
                    merged[-1] = candidate
                    continue
            merged.append(primitive)
        return merged

    @staticmethod
    def _annotate_primitives(primitives: List[Dict[str, Any]], sampled_point_count: int) -> List[Dict[str, Any]]:
        annotated: List[Dict[str, Any]] = []
        running_start = 0
        max_index = max(0, int(sampled_point_count) - 1)
        for segment_id, primitive in enumerate(primitives, start=1):
            item = dict(primitive)
            source_count = int(item.get("source_point_count", 0) or 0)
            start_index = min(max_index, max(0, running_start))
            end_index = min(max_index, start_index + max(0, source_count - 1))
            item["segment_id"] = int(segment_id)
            item["source_start_index"] = int(start_index)
            item["source_end_index"] = int(end_index)
            item["start_point_count"] = source_count
            item["visual_label"] = (
                f"#{segment_id} {str(item.get('type', 'spline')).upper()} "
                f"start={start_index} n={source_count} "
                f"params={int(item.get('parameter_count', item.get('effective_params', 0)) or 0)}"
            )
            running_start += max(1, source_count - 1)
            annotated.append(item)
        return annotated

    def _component_metrics(self, sampled: Any, primitives: List[Dict[str, Any]]) -> Dict[str, Any]:
        primitive_count = len(primitives)
        by_type: Dict[str, int] = {}
        total_params = 0
        max_error = 0.0
        mean_error_sum = 0.0
        for primitive in primitives:
            kind = str(primitive.get("type", "spline"))
            by_type[kind] = by_type.get(kind, 0) + 1
            total_params += int(primitive.get("parameter_count", primitive.get("effective_params", 0)) or 0)
            max_error = max(max_error, float(primitive.get("max_error", 0.0) or 0.0))
            mean_error_sum += float(primitive.get("mean_error", 0.0) or 0.0)
        return {
            "sampled_point_count": int(len(sampled)),
            "primitive_count": primitive_count,
            "primitive_by_type": by_type,
            "parameter_count": int(total_params),
            "mean_error_px": mean_error_sum / max(1, primitive_count),
            "max_error_px": max_error,
            "compression_ratio": float(total_params / max(1, int(len(sampled) * 2))),
        }

    def _aggregate_metrics(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        primitive_count = 0
        total_params = 0
        sampled_points = 0
        max_error = 0.0
        mean_errors: List[float] = []
        by_type: Dict[str, int] = {}
        for component in components:
            metrics = component.get("metrics", {})
            primitive_count += int(metrics.get("primitive_count", 0) or 0)
            total_params += int(metrics.get("parameter_count", 0) or 0)
            sampled_points += int(metrics.get("sampled_point_count", 0) or 0)
            max_error = max(max_error, float(metrics.get("max_error_px", 0.0) or 0.0))
            mean_errors.append(float(metrics.get("mean_error_px", 0.0) or 0.0))
            for kind, count in (metrics.get("primitive_by_type", {}) or {}).items():
                by_type[str(kind)] = by_type.get(str(kind), 0) + int(count)
        return {
            "component_count": len(components),
            "primitive_count": int(primitive_count),
            "primitive_by_type": by_type,
            "total_parameters": int(total_params),
            "sampled_point_count": int(sampled_points),
            "mean_error_px": sum(mean_errors) / max(1, len(mean_errors)),
            "max_error_px": float(max_error),
            "compression_ratio": float(total_params / max(1, sampled_points * 2)),
        }

    def _write_svg_preview(self, path: Path, width: int, height: int, components: List[Dict[str, Any]]) -> None:
        colors = {
            "line": "#1e90ff",
            "arc": "#f28c28",
            "spline": "#32a852",
        }
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        for component in components:
            if component.get("topology") in ("solid", "solid_with_holes"):
                for primitive in component.get("primitives", []):
                    pts = self._primitive_preview_points(primitive)
                    if len(pts) < 2:
                        continue
                    text = " ".join(f"{float(x):.3f},{float(y):.3f}" for x, y in pts)
                    parts.append(
                        f'<polyline points="{text}" fill="none" stroke="#1e90ff" '
                        'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                        f'<title>{self._svg_escape(str(primitive.get("visual_label", "solid outer line")))}</title>'
                        '</polyline>'
                    )
                    start_x, start_y = pts[0]
                    parts.append(f'<circle cx="{start_x:.3f}" cy="{start_y:.3f}" r="2.5" fill="#d62728"/>')
                for hole in component.get("holes", []):
                    for primitive in hole.get("primitives", []):
                        pts = self._primitive_preview_points(primitive)
                        if len(pts) < 2:
                            continue
                        text = " ".join(f"{float(x):.3f},{float(y):.3f}" for x, y in pts)
                        parts.append(
                            f'<polyline points="{text}" fill="none" stroke="#1e90ff" '
                            'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                            f'<title>{self._svg_escape(str(primitive.get("visual_label", "solid hole line")))}</title>'
                            '</polyline>'
                        )
                        start_x, start_y = pts[0]
                        parts.append(f'<circle cx="{start_x:.3f}" cy="{start_y:.3f}" r="2.5" fill="#d62728"/>')
                continue
            for primitive in component.get("primitives", []):
                pts = self._primitive_preview_points(primitive)
                if len(pts) < 2:
                    continue
                text = " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
                color = colors.get(str(primitive.get("type", "spline")), "#333333")
                parts.append(
                    f'<polyline points="{text}" fill="none" stroke="{color}" '
                    'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
                    f'<title>{self._svg_escape(str(primitive.get("visual_label", "")))}</title>'
                    '</polyline>'
                )
                start_x, start_y = pts[0]
                parts.append(f'<circle cx="{start_x:.3f}" cy="{start_y:.3f}" r="3" fill="#d62728"/>')
        parts.append(
            '<g font-family="Consolas, monospace" font-size="12">'
            '<text x="12" y="22" fill="#1e90ff">line</text>'
            '<text x="70" y="22" fill="#f28c28">arc</text>'
            '<text x="120" y="22" fill="#32a852">spline</text>'
            '<text x="190" y="22" fill="#d62728">red=start point</text>'
            '</g>'
        )
        parts.append("</svg>")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(parts), encoding="utf-8")

    def _write_png_preview(self, path: Path, image: Any, components: List[Dict[str, Any]]) -> None:
        import cv2
        import numpy as np

        base = image.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        overlay = np.full_like(base, 255)
        colors = {
            "line": (255, 144, 30),
            "arc": (40, 140, 242),
            "spline": (82, 168, 50),
        }
        for component in components:
            if component.get("topology") in ("solid", "solid_with_holes"):
                for primitive in component.get("primitives", []):
                    pts = np.asarray(self._primitive_preview_points(primitive), dtype=np.float64).reshape(-1, 2)
                    if len(pts) < 2:
                        continue
                    poly = np.round(pts).astype(np.int32)
                    poly[:, 0] = np.clip(poly[:, 0], 0, overlay.shape[1] - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, overlay.shape[0] - 1)
                    cv2.polylines(overlay, [poly.reshape(-1, 1, 2)], False, colors["line"], 2, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, tuple(poly[0].tolist()), 3, (40, 40, 220), -1, lineType=cv2.LINE_AA)
                for hole in component.get("holes", []):
                    for primitive in hole.get("primitives", []):
                        pts = np.asarray(self._primitive_preview_points(primitive), dtype=np.float64).reshape(-1, 2)
                        if len(pts) < 2:
                            continue
                        poly = np.round(pts).astype(np.int32)
                        poly[:, 0] = np.clip(poly[:, 0], 0, overlay.shape[1] - 1)
                        poly[:, 1] = np.clip(poly[:, 1], 0, overlay.shape[0] - 1)
                        cv2.polylines(overlay, [poly.reshape(-1, 1, 2)], False, colors["line"], 2, lineType=cv2.LINE_AA)
                        cv2.circle(overlay, tuple(poly[0].tolist()), 3, (40, 40, 220), -1, lineType=cv2.LINE_AA)
                continue
            for primitive in component.get("primitives", []):
                pts = np.asarray(self._primitive_preview_points(primitive), dtype=np.float64).reshape(-1, 2)
                if len(pts) < 2:
                    continue
                poly = np.round(pts).astype(np.int32)
                poly[:, 0] = np.clip(poly[:, 0], 0, overlay.shape[1] - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, overlay.shape[0] - 1)
                color = colors.get(str(primitive.get("type", "spline")), (40, 40, 40))
                cv2.polylines(overlay, [poly.reshape(-1, 1, 2)], False, color, 2, lineType=cv2.LINE_AA)
                start = tuple(poly[0].tolist())
                cv2.circle(overlay, start, 3, (40, 40, 220), -1, lineType=cv2.LINE_AA)
        cv2.putText(overlay, "line", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["line"], 2, cv2.LINE_AA)
        cv2.putText(overlay, "arc", (72, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["arc"], 2, cv2.LINE_AA)
        cv2.putText(overlay, "spline", (122, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["spline"], 2, cv2.LINE_AA)
        cv2.putText(overlay, "red=start point", (205, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 220), 2, cv2.LINE_AA)
        self._write_image(path, np.hstack([base, overlay]))

    def _write_primitive_label_files(self, json_path: Path, csv_path: Path, components: List[Dict[str, Any]]) -> None:
        """Write detailed primitive labels outside preview images.

        中文说明：
        图上只保留颜色和起点，详细字段单独保存，避免可视化被文字盖住。
        """
        rows: List[Dict[str, Any]] = []
        for component in components:
            component_id = int(component.get("component_id", 0) or 0)
            for primitive in component.get("primitives", []):
                rows.append(
                    {
                        "component_id": component_id,
                        "segment_id": int(primitive.get("segment_id", 0) or 0),
                        "type": str(primitive.get("type", primitive.get("kind", ""))),
                        "source_start_index": int(primitive.get("source_start_index", 0) or 0),
                        "source_end_index": int(primitive.get("source_end_index", 0) or 0),
                        "source_point_count": int(primitive.get("source_point_count", 0) or 0),
                        "parameter_count": int(primitive.get("parameter_count", primitive.get("effective_params", 0)) or 0),
                        "max_error": float(primitive.get("max_error", 0.0) or 0.0),
                        "mean_error": float(primitive.get("mean_error", 0.0) or 0.0),
                        "radius": primitive.get("radius", ""),
                        "sweep_deg": primitive.get("sweep_deg", ""),
                        "boundary_role": "outer" if component.get("topology") in ("solid", "solid_with_holes") else "",
                        "hole_id": "",
                        "visual_label": str(primitive.get("visual_label", "")),
                    }
                )
            for hole in component.get("holes", []):
                hole_id = int(hole.get("hole_id", 0) or 0)
                for primitive in hole.get("primitives", []):
                    rows.append(
                        {
                            "component_id": component_id,
                            "segment_id": int(primitive.get("segment_id", 0) or 0),
                            "type": str(primitive.get("type", primitive.get("kind", ""))),
                            "source_start_index": int(primitive.get("source_start_index", 0) or 0),
                            "source_end_index": int(primitive.get("source_end_index", 0) or 0),
                            "source_point_count": int(primitive.get("source_point_count", 0) or 0),
                            "parameter_count": int(primitive.get("parameter_count", primitive.get("effective_params", 0)) or 0),
                            "max_error": float(primitive.get("max_error", 0.0) or 0.0),
                            "mean_error": float(primitive.get("mean_error", 0.0) or 0.0),
                            "radius": primitive.get("radius", ""),
                            "sweep_deg": primitive.get("sweep_deg", ""),
                            "boundary_role": "hole",
                            "hole_id": hole_id,
                            "visual_label": str(primitive.get("visual_label", "")),
                        }
                    )

        self._write_json(json_path, {"labels": rows})
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "component_id",
            "segment_id",
            "type",
            "source_start_index",
            "source_end_index",
            "source_point_count",
            "parameter_count",
            "max_error",
            "mean_error",
            "radius",
            "sweep_deg",
            "boundary_role",
            "hole_id",
            "visual_label",
        ]
        lines = [",".join(columns)]
        for row in rows:
            values = []
            for column in columns:
                value = str(row.get(column, ""))
                if any(ch in value for ch in [",", '"', "\n"]):
                    value = '"' + value.replace('"', '""') + '"'
                values.append(value)
            lines.append(",".join(values))
        csv_path.write_text("\n".join(lines), encoding="utf-8")

    def _primitive_preview_points(self, primitive: Dict[str, Any]) -> List[Point]:
        primitive_type = str(primitive.get("type", primitive.get("kind", "spline")))
        if primitive_type == "line":
            return [tuple(primitive["start"]), tuple(primitive["end"])]
        if primitive_type == "arc":
            return self._sample_arc_primitive(primitive, max_step_deg=8.0)
        return [tuple(point) for point in primitive.get("control_points", primitive.get("points", []))]

    @staticmethod
    def _label_position(points: Sequence[Point]) -> Point:
        if len(points) == 0:
            return 0.0, 0.0
        idx = min(max(0, len(points) // 2), len(points) - 1)
        x, y = points[idx]
        return float(x) + 5.0, float(y) - 5.0

    @staticmethod
    def _svg_escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _sample_arc_primitive(primitive: Dict[str, Any], max_step_deg: float = 8.0) -> List[Point]:
        import numpy as np

        center = np.asarray(primitive["center"], dtype=np.float64)
        start = np.asarray(primitive["start"], dtype=np.float64)
        end = np.asarray(primitive["end"], dtype=np.float64)
        radius = float(primitive["radius"])
        a0 = math.atan2(start[1] - center[1], start[0] - center[0])
        a1 = math.atan2(end[1] - center[1], end[0] - center[0])
        clockwise = bool(primitive.get("clockwise", False))
        if clockwise and a1 > a0:
            a1 -= 2.0 * math.pi
        if not clockwise and a1 < a0:
            a1 += 2.0 * math.pi
        sweep = a1 - a0
        count = max(3, int(math.ceil(abs(math.degrees(sweep)) / max(1.0, max_step_deg))) + 1)
        angles = np.linspace(a0, a1, count)
        pts = np.column_stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)])
        return [(float(x), float(y)) for x, y in pts]

    @staticmethod
    def _line_errors(points: Any, start: Any, end: Any):
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        a = np.asarray(start, dtype=np.float64)
        b = np.asarray(end, dtype=np.float64)
        ab = b - a
        denom = float(np.linalg.norm(ab))
        if denom <= 1e-12:
            return np.linalg.norm(pts - a, axis=1)
        return np.abs(np.cross(ab, pts - a) / denom)

    @staticmethod
    def _nearest_polyline_errors(points: Any, polyline: Any):
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        line = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
        if len(line) < 2:
            return np.linalg.norm(pts - line[0], axis=1) if len(line) else np.zeros(len(pts))
        errors = []
        for point in pts:
            best = float("inf")
            for a, b in zip(line[:-1], line[1:]):
                ab = b - a
                denom = float(np.dot(ab, ab))
                if denom <= 1e-12:
                    dist = float(np.linalg.norm(point - a))
                else:
                    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
                    proj = a + t * ab
                    dist = float(np.linalg.norm(point - proj))
                best = min(best, dist)
            errors.append(best)
        return np.asarray(errors, dtype=np.float64)

    @staticmethod
    def _curvature(points: Any, closed: bool):
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            return np.zeros(len(pts), dtype=np.float64)
        if closed:
            prev_pts = np.roll(pts, 1, axis=0)
            next_pts = np.roll(pts, -1, axis=0)
        else:
            prev_pts = np.vstack([pts[0], pts[:-1]])
            next_pts = np.vstack([pts[1:], pts[-1]])
        a = pts - prev_pts
        b = next_pts - pts
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(next_pts - prev_pts, axis=1))
        denom = np.maximum(denom, 1e-9)
        return 2.0 * cross / denom

    @staticmethod
    def _resample_polyline(points: Any, closed: bool, step: float):
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            return pts
        if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            pts = np.vstack([pts, pts[0]])
        distances = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(distances)])
        total = float(cumulative[-1])
        if total <= 1e-9:
            return pts[:1]
        count = max(2, int(math.ceil(total / max(0.5, step))))
        targets = np.linspace(0.0, total, count + 1)
        x = np.interp(targets, cumulative, pts[:, 0])
        y = np.interp(targets, cumulative, pts[:, 1])
        return np.column_stack([x, y])

    @staticmethod
    def _rdp_array(points: Any, epsilon: float):
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            return pts
        start = pts[0]
        end = pts[-1]
        errors = GeometryDrivenParameterizer._line_errors(pts, start, end)
        idx = int(np.argmax(errors))
        max_error = float(errors[idx])
        if max_error > epsilon:
            left = GeometryDrivenParameterizer._rdp_array(pts[: idx + 1], epsilon)
            right = GeometryDrivenParameterizer._rdp_array(pts[idx:], epsilon)
            return np.vstack([left[:-1], right])
        return np.vstack([start, end])

    @staticmethod
    def _remove_consecutive_duplicates(points: Any, tolerance: float = 1e-9):
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) <= 1:
            return pts
        keep = [pts[0]]
        for point in pts[1:]:
            if float(np.linalg.norm(point - keep[-1])) > tolerance:
                keep.append(point)
        return np.asarray(keep, dtype=np.float64)

    @staticmethod
    def _remove_trailing_closure(points: Any, tolerance: float = 1e-9):
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        while len(pts) > 1 and float(np.linalg.norm(pts[0] - pts[-1])) <= tolerance:
            pts = pts[:-1]
        return pts

    @staticmethod
    def _polyline_length(points: Any) -> float:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @staticmethod
    def _bbox(points: Any) -> List[float]:
        import numpy as np

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return [
            float(np.min(pts[:, 0])),
            float(np.min(pts[:, 1])),
            float(np.max(pts[:, 0])),
            float(np.max(pts[:, 1])),
        ]

    @staticmethod
    def _write_image(path: Path, image: Any) -> None:
        import cv2
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix or ".png"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            raise ValueError(f"Failed to encode image as {suffix}: {path}")
        encoded.tofile(str(path))

    @classmethod
    def _write_json(cls, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cls._to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        try:
            import numpy as np
        except Exception:
            np = None
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        if np is not None:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        return value
