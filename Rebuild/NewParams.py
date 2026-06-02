from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np


PathLike = Union[str, Path]
REBUILD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REBUILD_DIR.parent
AUTOCAD_ROOT = PROJECT_ROOT / "AutoCAD_v8.5.4"

for _path in (PROJECT_ROOT, AUTOCAD_ROOT):
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)


class NewParams:
    """Image preprocessing entry for structure parameterization.

    Input:
        image_path: path to a structure image.

    Output:
        - original image
        - gray image
        - blurred image
        - edge image saved as ``*_edges.png``

    This class intentionally keeps the old public methods:
    original_img(), gray(), blurred(), edges(), edge_path().
    """

    def __init__(
            self,
            image_path: PathLike,
            save_dir: Optional[PathLike] = None,
            edge_filename: Optional[str] = None,
            canny_threshold1: int = 60,
            canny_threshold2: int = 180,
            blur_kernel_size: int = 5,
            aperture_size: int = 3,
            morph_kernel_size: int = 3,
            dilate_iterations: int = 1,
            erode_iterations: int = 0,
            edge_mode: str = "auto",
            remove_border_edges: bool = True,
            border_margin: int = 2,
            remove_large_frame_edges: bool = True,
            large_frame_span_ratio: float = 0.82,
            auto_process: bool = True,
            verbose: bool = True,
    ):
        self.image_path = Path(image_path)
        self.save_dir = Path(save_dir) if save_dir is not None else self.image_path.parent
        self.edge_filename = edge_filename or f"{self.image_path.stem}_edges.png"
        self.verbose = bool(verbose)

        self.canny_threshold1 = int(canny_threshold1)
        self.canny_threshold2 = int(canny_threshold2)
        self.blur_kernel_size = self._as_odd_kernel_size(blur_kernel_size, "blur_kernel_size")
        self.aperture_size = self._as_canny_aperture(aperture_size)
        self.morph_kernel_size = self._as_odd_kernel_size(morph_kernel_size, "morph_kernel_size")
        self.dilate_iterations = int(dilate_iterations)
        self.erode_iterations = int(erode_iterations)
        self.edge_mode = str(edge_mode).lower().strip()
        if self.edge_mode not in ("auto", "canny", "foreground_contour", "stroke_mask"):
            raise ValueError("edge_mode must be one of: 'auto', 'canny', 'foreground_contour', 'stroke_mask'.")
        self.remove_border_edges = bool(remove_border_edges)
        self.border_margin = int(border_margin)
        self.remove_large_frame_edges = bool(remove_large_frame_edges)
        self.large_frame_span_ratio = float(large_frame_span_ratio)

        self.__original_img = None
        self.__gray_img = None
        self.__blurred_img = None
        self.__edges_img = None
        self.__edge_path = None
        self.__edge_representation = "edge"

        if auto_process:
            self.process(save=True)

    def process(self, save: bool = True) -> np.ndarray:
        self._log("=" * 72)
        self._log("[NewParams] Start image preprocessing")
        self._log(f"[NewParams] image_path: {self.image_path}")
        self._log(f"[NewParams] save_dir:   {self.save_dir}")
        self._log(f"[NewParams] edge_mode:  {self.edge_mode}")

        self.__original_img = self._read_image(self.image_path)
        self._log(f"[NewParams] original image shape: {self.__original_img.shape}")

        self.__gray_img = self._to_gray(self.__original_img)
        self.__blurred_img = cv2.GaussianBlur(
            self.__gray_img,
            (self.blur_kernel_size, self.blur_kernel_size),
            0,
        )
        self.__edges_img = self._extract_edges(self.__original_img, self.__blurred_img)
        self._log(
            "[NewParams] Canny params: "
            f"thresholds=({self.canny_threshold1}, {self.canny_threshold2}), "
            f"blur={self.blur_kernel_size}, aperture={self.aperture_size}"
        )
        self._log(f"[NewParams] edge pixels: {int(np.count_nonzero(self.__edges_img))}")
        self._log(f"[NewParams] edge representation: {self.__edge_representation}")

        if save:
            self.save_edges()

        self._log("[NewParams] Preprocessing finished")
        return self.__edges_img

    def parameterize(
            self,
            save_dir: Optional[PathLike] = None,
            trace_source: str = "edges",
            **kwargs,
    ) -> "CurveParameterizer":
        """Run the VTracer-backed parameterization pipeline.

        ``NewParams`` is a preprocessing wrapper, so its default is to pass the
        extracted edge image into the parameterizer.  Use
        ``trace_source="original"`` when the filled foreground mask is the
        intended target.
        """
        if "edge_contour_tracing" not in kwargs:
            kwargs["edge_contour_tracing"] = self.__edge_representation != "stroke_mask"

        return CurveParameterizer(
            image_path=self.image_path,
            edges=self.edges(),
            original_img=self.original_img(),
            save_dir=save_dir if save_dir is not None else self.save_dir / "parameterization",
            trace_source=trace_source,
            verbose=self.verbose,
            **kwargs,
        )

    def save_edges(self, output_path: Optional[PathLike] = None) -> Path:
        if self.__edges_img is None:
            raise RuntimeError("No edge image found. Call process() before save_edges().")

        target_path = Path(output_path) if output_path is not None else self.save_dir / self.edge_filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_image(target_path, self.__edges_img)
        self.__edge_path = target_path
        self._log(f"[NewParams] edge image saved: {target_path}")
        return target_path

    def original_img(self) -> np.ndarray:
        return self._require_image(self.__original_img, "original image")

    def gray(self) -> np.ndarray:
        return self._require_image(self.__gray_img, "gray image")

    def blurred(self) -> np.ndarray:
        return self._require_image(self.__blurred_img, "blurred image")

    def edges(self) -> np.ndarray:
        return self._require_image(self.__edges_img, "edge image")

    def edge_path(self) -> Optional[Path]:
        return self.__edge_path

    def edge_representation(self) -> str:
        return self.__edge_representation

    def _extract_edges(self, original_img: np.ndarray, blurred_img: np.ndarray) -> np.ndarray:
        self.__edge_representation = "edge"
        stroke_mask = None
        if self.edge_mode in ("auto", "stroke_mask"):
            stroke_mask = self._extract_foreground_stroke_mask(original_img)
            if self.edge_mode == "stroke_mask" and stroke_mask is not None:
                self._log("[NewParams] edge_mode=stroke_mask selected stroke_mask centerline input")
                self.__edge_representation = "stroke_mask"
                return stroke_mask
            if self.edge_mode == "stroke_mask":
                self._log("[NewParams] edge_mode=stroke_mask failed; falling back to edge extraction")
            if self.edge_mode == "auto" and stroke_mask is not None and self._image_looks_normalized_grayscale(original_img):
                self._log("[NewParams] edge_mode=auto selected stroke_mask centerline input")
                self.__edge_representation = "stroke_mask"
                return stroke_mask

        contour_edges = None
        if self.edge_mode in ("auto", "foreground_contour"):
            contour_edges = self._extract_foreground_contour_edges(original_img)
            if self.edge_mode == "foreground_contour" and contour_edges is not None:
                return contour_edges

        edges = cv2.Canny(
            blurred_img,
            self.canny_threshold1,
            self.canny_threshold2,
            apertureSize=self.aperture_size,
        )

        if self.morph_kernel_size > 1 and (self.dilate_iterations > 0 or self.erode_iterations > 0):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.morph_kernel_size, self.morph_kernel_size),
            )
            if self.dilate_iterations > 0:
                edges = cv2.dilate(edges, kernel, iterations=self.dilate_iterations)
            if self.erode_iterations > 0:
                edges = cv2.erode(edges, kernel, iterations=self.erode_iterations)

        if self.remove_border_edges or self.remove_large_frame_edges:
            edges = self._remove_border_or_frame_components(
                edges,
                margin=self.border_margin if self.remove_border_edges else 0,
                frame_span_ratio=self.large_frame_span_ratio if self.remove_large_frame_edges else 1.1,
            )

        if contour_edges is not None and self._foreground_contour_edges_look_valid(contour_edges, edges):
            self._log("[NewParams] edge_mode=auto selected foreground_contour edges")
            return contour_edges

        return edges

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        return image

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        suffix = path.suffix or ".png"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            raise ValueError(f"Failed to encode image as {suffix}: {path}")
        encoded.tofile(str(path))

    @staticmethod
    def _remove_border_or_frame_components(edges: np.ndarray, margin: int, frame_span_ratio: float) -> np.ndarray:
        margin = max(0, int(margin))
        frame_span_ratio = float(frame_span_ratio)
        binary = (edges > 0).astype(np.uint8)
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n_labels <= 1:
            return edges

        h, w = edges.shape[:2]
        border_mask = np.zeros_like(binary, dtype=bool)
        if margin > 0:
            border_mask[:margin, :] = True
            border_mask[max(0, h - margin):, :] = True
            border_mask[:, :margin] = True
            border_mask[:, max(0, w - margin):] = True

        remove_labels = set(int(v) for v in np.unique(labels[border_mask]) if int(v) != 0)
        frame_labels: List[int] = []
        if frame_span_ratio < 1.0:
            for label in range(1, n_labels):
                bw = int(stats[label, cv2.CC_STAT_WIDTH])
                bh = int(stats[label, cv2.CC_STAT_HEIGHT])
                if bw >= frame_span_ratio * w and bh >= frame_span_ratio * h:
                    frame_labels.append(label)
        if not remove_labels and not frame_labels:
            return edges

        out = edges.copy()
        for label in remove_labels:
            out[labels == label] = 0
        for label in frame_labels:
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            pad = max(6, margin * 3)
            label_mask = labels == label
            frame_mask = np.zeros_like(label_mask, dtype=bool)
            frame_mask[y:min(h, y + pad), x:x2] = True
            frame_mask[max(0, y2 - pad):y2, x:x2] = True
            frame_mask[y:y2, x:min(w, x + pad)] = True
            frame_mask[y:y2, max(0, x2 - pad):x2] = True
            out[label_mask & frame_mask] = 0
        return out

    @staticmethod
    def _extract_foreground_contour_edges(image: np.ndarray) -> Optional[np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(image)

        warm_hue = (hsv[:, :, 0] <= 45) | (hsv[:, :, 0] >= 170)
        bright = hsv[:, :, 2] >= 120
        not_too_saturated = hsv[:, :, 1] <= 170
        red_dominant = r.astype(np.int16) >= b.astype(np.int16) + 18
        mask = (warm_hue & bright & not_too_saturated & red_dominant).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        h, w = mask.shape[:2]
        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        cleaned = np.zeros_like(mask)
        min_area = max(80, int(0.002 * h * w))
        for label in range(1, n_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
            cleaned[labels == label] = 255

        fg_ratio = float(np.count_nonzero(cleaned)) / float(max(1, h * w))
        if fg_ratio < 0.004 or fg_ratio > 0.55:
            return None

        contours, _hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if len(c) >= 8 and abs(cv2.contourArea(c)) >= min_area * 0.2]
        if not contours:
            return None
        edges = np.zeros_like(mask)
        cv2.drawContours(edges, contours, -1, 255, 1)
        return edges

    @staticmethod
    def _foreground_contour_edges_look_valid(contour_edges: np.ndarray, canny_edges: np.ndarray) -> bool:
        contour_pixels = int(np.count_nonzero(contour_edges))
        canny_pixels = int(np.count_nonzero(canny_edges))
        if contour_pixels < 80:
            return False
        if canny_pixels <= 0:
            return True
        ratio = contour_pixels / float(max(1, canny_pixels))
        return 0.04 <= ratio <= 1.2

    @staticmethod
    def _image_looks_normalized_grayscale(image: np.ndarray) -> bool:
        b, g, r = cv2.split(image)
        max_channel_delta = np.maximum.reduce(
            [
                np.abs(b.astype(np.int16) - g.astype(np.int16)),
                np.abs(g.astype(np.int16) - r.astype(np.int16)),
                np.abs(b.astype(np.int16) - r.astype(np.int16)),
            ]
        )
        return float(np.percentile(max_channel_delta, 98.0)) <= 6.0

    @staticmethod
    def _extract_foreground_stroke_mask(image: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray < 245).astype(np.uint8) * 255
        h, w = mask.shape[:2]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        if n_labels <= 1:
            return None

        cleaned = mask.copy()
        for label in range(1, n_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            if bw >= 0.75 * w and bh >= 0.75 * h:
                pad = max(6, int(round(min(w, h) * 0.01)))
                label_mask = labels == label
                frame_mask = np.zeros_like(label_mask, dtype=bool)
                x2 = min(w, x + bw)
                y2 = min(h, y + bh)
                frame_mask[y:min(h, y + pad), x:x2] = True
                frame_mask[max(0, y2 - pad):y2, x:x2] = True
                frame_mask[y:y2, x:min(w, x + pad)] = True
                frame_mask[y:y2, max(0, x2 - pad):x2] = True
                cleaned[label_mask & frame_mask] = 0

        n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), 8)
        out = np.zeros_like(cleaned)
        min_area = max(20, int(0.00008 * h * w))
        for label in range(1, n_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                continue
            out[labels == label] = 255

        fg_ratio = float(np.count_nonzero(out)) / float(max(1, h * w))
        if fg_ratio < 0.0005 or fg_ratio > 0.35:
            return None
        return out

    @staticmethod
    def _require_image(image: Optional[np.ndarray], name: str) -> np.ndarray:
        if image is None:
            raise RuntimeError(f"No {name} found. Call process() first.")
        return image

    @staticmethod
    def _as_odd_kernel_size(value: int, name: str) -> int:
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be >= 1.")
        if value % 2 == 0:
            raise ValueError(f"{name} must be an odd number.")
        return value

    @staticmethod
    def _as_canny_aperture(value: int) -> int:
        value = int(value)
        if value not in (3, 5, 7):
            raise ValueError("aperture_size must be one of 3, 5, or 7.")
        return value


class CurveParameterizer:
    """VTracer-backed curve parameterizer.

    This class replaces the previous local contour fitting code with the newer
    centerline semantic segmentation pipeline from ``vtracer_python.py``.

    Compatible input styles:
        1. CurveParameterizer(image_path="xxx.png")
        2. CurveParameterizer(edges=edge_img, original_img=bgr_img)
        3. NewParams(...).parameterize()

    Main outputs:
        - results(): structured Python data
        - save_json(): JSON file
        - visualize(): PNG overlay
        - svg_path()/metrics_path(): VTracer native outputs
    """

    def __init__(
            self,
            edges: Optional[np.ndarray] = None,
            original_img: Optional[np.ndarray] = None,
            image_path: Optional[PathLike] = None,
            save_dir: Optional[PathLike] = None,
            output_prefix: Optional[str] = None,
            auto_process: bool = True,
            verbose: bool = True,
            keep_intermediates: bool = True,
            fit_tolerance: float = 1.5,
            resample_step: float = 3.0,
            filter_speckle: int = 4,
            gaussian_sigma: float = 1.05,
            semantic_window_size: int = 12,
            keypoint_angle_threshold_deg: float = 32.0,
            keypoint_refine_radius: int = 5,
            dp_complexity_weight: float = 1.02,
            dp_max_segment_points: int = 78,
            line_merge_angle_deg: float = 13.5,
            arc_radius_rel_tol: float = 0.22,
            arc_center_tol: float = 2.2,
            arc_min_sweep_deg: float = 17.0,
            spline_ctrl_penalty: float = 0.11,
            trace_source: str = "original",
            median_ksize: Optional[int] = None,
            edge_contour_tracing: bool = True,
            **legacy_kwargs,
    ):
        self.edges = self._normalize_edges(edges) if edges is not None else None
        self.original_img = original_img
        self.image_path = Path(image_path) if image_path is not None else None
        self.save_dir = Path(save_dir) if save_dir is not None else self._default_save_dir()
        self.output_prefix = output_prefix or self._default_output_prefix()
        self.verbose = bool(verbose)
        self.keep_intermediates = bool(keep_intermediates)

        self.fit_tolerance = float(fit_tolerance)
        self.resample_step = float(legacy_kwargs.pop("sample_interval", resample_step))
        self.legacy_kwargs = dict(legacy_kwargs)
        self.filter_speckle = int(filter_speckle)
        self.gaussian_sigma = float(gaussian_sigma)
        self.semantic_window_size = int(semantic_window_size)
        self.keypoint_angle_threshold_deg = float(keypoint_angle_threshold_deg)
        self.keypoint_refine_radius = int(keypoint_refine_radius)
        self.dp_complexity_weight = float(dp_complexity_weight)
        self.dp_max_segment_points = int(dp_max_segment_points)
        self.line_merge_angle_deg = float(line_merge_angle_deg)
        self.arc_radius_rel_tol = float(arc_radius_rel_tol)
        self.arc_center_tol = float(arc_center_tol)
        self.arc_min_sweep_deg = float(arc_min_sweep_deg)
        self.spline_ctrl_penalty = float(spline_ctrl_penalty)
        self.trace_source = str(trace_source).lower().strip()
        if self.trace_source not in ("original", "edges", "auto"):
            raise ValueError("trace_source must be one of: 'original', 'edges', 'auto'.")
        self.median_ksize = median_ksize
        self.edge_contour_tracing = bool(edge_contour_tracing)

        self.__results: List[Dict[str, Any]] = []
        self.__metrics: Dict[str, Any] = {}
        self.__trace_image_path: Optional[Path] = None
        self.__svg_path: Optional[Path] = None
        self.__metrics_path: Optional[Path] = None
        self.__intermediate_dir: Optional[Path] = None
        self.__last_json_path: Optional[Path] = None
        self.__last_visual_path: Optional[Path] = None

        if auto_process:
            self.parameterize()

    def parameterize(self) -> List[Dict[str, Any]]:
        from bayesian_optimization.tools.vtracer_python import TraceConfig, VTracerPython

        t0 = time.perf_counter()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.__trace_image_path = self._prepare_trace_image()
        self.__svg_path = self.save_dir / f"{self.output_prefix}.svg"
        self.__metrics_path = self.save_dir / f"{self.output_prefix}_metrics.json"
        self.__intermediate_dir = self.save_dir / "intermediates"

        if self.__intermediate_dir.exists():
            shutil.rmtree(self.__intermediate_dir, ignore_errors=True)
        self.__intermediate_dir.mkdir(parents=True, exist_ok=True)

        self._log("=" * 72)
        self._log("[CurveParameterizer] Start VTracer-backed parameterization")
        self._log(f"[CurveParameterizer] trace image:      {self.__trace_image_path}")
        self._log(f"[CurveParameterizer] output dir:       {self.save_dir}")
        self._log(f"[CurveParameterizer] svg path:         {self.__svg_path}")
        self._log(f"[CurveParameterizer] metrics path:     {self.__metrics_path}")
        self._log(f"[CurveParameterizer] intermediates:    {self.__intermediate_dir}")
        self._log("[CurveParameterizer] config:")
        self._log(f"  fit_tolerance={self.fit_tolerance}, resample_step={self.resample_step}, filter_speckle={self.filter_speckle}")
        self._log(f"  gaussian_sigma={self.gaussian_sigma}, semantic_window_size={self.semantic_window_size}")
        self._log(f"  keypoint_angle_threshold_deg={self.keypoint_angle_threshold_deg}, keypoint_refine_radius={self.keypoint_refine_radius}")
        self._log(f"  dp_complexity_weight={self.dp_complexity_weight}, dp_max_segment_points={self.dp_max_segment_points}")
        self._log(f"  line_merge_angle_deg={self.line_merge_angle_deg}")
        self._log(f"  arc_radius_rel_tol={self.arc_radius_rel_tol}, arc_center_tol={self.arc_center_tol}, arc_min_sweep_deg={self.arc_min_sweep_deg}")
        self._log(f"  spline_ctrl_penalty={self.spline_ctrl_penalty}")
        self._log(f"  trace_source={self.trace_source}, median_ksize={self._effective_median_ksize()}")
        self._log(f"  edge_contour_tracing={self.edge_contour_tracing}")
        if self.legacy_kwargs:
            self._log(f"[CurveParameterizer] ignored legacy kwargs: {sorted(self.legacy_kwargs.keys())}")

        cfg = TraceConfig(
            image_path=str(self.__trace_image_path),
            color_mode="bw",
            trace_style="centerline",
            mode="spline",
            metrics_path=str(self.__metrics_path),
            save_intermediates=str(self.__intermediate_dir),
            fit_tolerance=self.fit_tolerance,
            resample_step=self.resample_step,
            filter_speckle=self.filter_speckle,
            median_ksize=self._effective_median_ksize(),
        )
        cfg.gaussian_sigma = self.gaussian_sigma
        cfg.semantic_window_size = self.semantic_window_size
        cfg.keypoint_angle_threshold_deg = self.keypoint_angle_threshold_deg
        cfg.keypoint_refine_radius = self.keypoint_refine_radius
        cfg.keypoint_use_model_guided = True
        cfg.keypoint_model_multiscale_votes = False
        cfg.dp_complexity_weight = self.dp_complexity_weight
        cfg.dp_max_segment_points = self.dp_max_segment_points
        cfg.line_merge_angle_deg = self.line_merge_angle_deg
        cfg.arc_radius_rel_tol = self.arc_radius_rel_tol
        cfg.arc_center_tol = self.arc_center_tol
        cfg.arc_min_sweep_deg = self.arc_min_sweep_deg
        cfg.spline_ctrl_penalty = self.spline_ctrl_penalty
        cfg.centerline_trace_edge_contours = bool(self.edge_contour_tracing and self._effective_trace_source() == "edges")

        tracer = VTracerPython(cfg)
        tracer.to_svg(str(self.__svg_path))

        self.__metrics = self._load_json_file(self.__metrics_path, default={})
        self.__results = self._load_semantic_results()
        self._print_summary(elapsed=time.perf_counter() - t0)

        if not self.keep_intermediates and self.__intermediate_dir.exists():
            shutil.rmtree(self.__intermediate_dir, ignore_errors=True)
            self._log("[CurveParameterizer] intermediates removed")

        return self.__results

    def results(self) -> List[Dict[str, Any]]:
        return self.__results

    def metrics(self) -> Dict[str, Any]:
        return self.__metrics

    def svg_path(self) -> Optional[Path]:
        return self.__svg_path

    def metrics_path(self) -> Optional[Path]:
        return self.__metrics_path

    def intermediate_dir(self) -> Optional[Path]:
        return self.__intermediate_dir

    def trace_image_path(self) -> Optional[Path]:
        return self.__trace_image_path

    def serializable_results(self, include_points: bool = True) -> List[Dict[str, Any]]:
        results = self.__results if include_points else self._results_without_points()
        return self._to_jsonable(results)

    def save_json(self, output_path: Optional[PathLike] = None, include_points: bool = True, indent: int = 2) -> Path:
        output_path = Path(output_path) if output_path is not None else self.save_dir / f"{self.output_prefix}_parameterization.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": "vtracer_python",
            "trace_image_path": str(self.__trace_image_path) if self.__trace_image_path else "",
            "svg_path": str(self.__svg_path) if self.__svg_path else "",
            "metrics_path": str(self.__metrics_path) if self.__metrics_path else "",
            "metrics": self.__metrics,
            "components": self.serializable_results(include_points=include_points),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(payload), f, ensure_ascii=False, indent=indent)
        self.__last_json_path = output_path
        self._log(f"[CurveParameterizer] JSON saved: {output_path}")
        return output_path

    def visualize(
            self,
            output_path: Optional[PathLike] = None,
            show_points: bool = False,
            show_segment_starts: bool = False,
            line_width: int = 2,
    ) -> Path:
        output_path = Path(output_path) if output_path is not None else self.save_dir / f"{self.output_prefix}_visualization.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        base = self._build_visual_base()
        overlay = np.full_like(base, 255)
        colors = {
            "line": (30, 144, 255),
            "arc": (0, 165, 255),
            "spline": (50, 205, 50),
        }

        h, w = overlay.shape[:2]
        for component in self.__results:
            points = np.asarray(component.get("resampled_points", []), dtype=np.float64)
            closed = bool(component.get("closed", False))
            if len(points) >= 2:
                pi = self._clip_points(points, w, h)
                cv2.polylines(overlay, [pi], closed, (220, 220, 220), 1, lineType=cv2.LINE_AA)

            for segment in component.get("segments", []):
                kind = str(segment.get("kind", segment.get("type", "spline")))
                color = colors.get(kind, (80, 80, 80))
                segment_points = self._slice_segment_points(
                    points,
                    int(segment.get("start_idx", 0)),
                    int(segment.get("end_idx", 0)),
                    closed=closed,
                )
                if len(segment_points) >= 2:
                    cv2.polylines(
                        overlay,
                        [self._clip_points(segment_points, w, h)],
                        False,
                        color,
                        line_width,
                        lineType=cv2.LINE_AA,
                    )

                if show_segment_starts and len(points) > 0:
                    start_idx = int(segment.get("start_idx", 0)) % len(points)
                    start_point = self._clip_points(points[[start_idx]], w, h)[0]
                    cv2.circle(overlay, tuple(start_point), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            if len(points) > 0:
                component_start = self._clip_points(points[[0]], w, h)[0]
                cv2.circle(overlay, tuple(component_start), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            if show_points and len(points) > 0:
                stride = max(1, len(points) // 120)
                for p in self._clip_points(points[::stride], w, h):
                    cv2.circle(overlay, tuple(p), 1, (70, 70, 70), -1, lineType=cv2.LINE_AA)

        self._draw_visual_legend(overlay)
        panel = np.hstack([base, overlay])
        self._write_image(output_path, panel)
        self.__last_visual_path = output_path
        self._log(f"[CurveParameterizer] visualization saved: {output_path}")
        return output_path

    def _prepare_trace_image(self) -> Path:
        source = self._effective_trace_source()

        if source == "original" and self.image_path is not None and self.image_path.exists():
            return self.image_path

        target = self.save_dir / f"{self.output_prefix}_trace_input.png"
        if source == "edges":
            if self.edges is None:
                raise ValueError("trace_source='edges' requires an edge image.")
            image = cv2.cvtColor(self.edges, cv2.COLOR_GRAY2BGR)
        elif self.original_img is not None:
            image = self.original_img
        elif self.image_path is not None and self.image_path.exists():
            return self.image_path
        else:
            raise ValueError("CurveParameterizer needs image_path or original_img/edges.")

        self._write_image(target, image)
        self._log(f"[CurveParameterizer] temporary trace image written: {target}")
        return target

    def _effective_median_ksize(self) -> int:
        if self.median_ksize is not None:
            return int(self.median_ksize)
        source = self._effective_trace_source()
        return 1 if source == "edges" else 3

    def _effective_trace_source(self) -> str:
        if self.trace_source == "auto":
            return "edges" if self.edges is not None else "original"
        return self.trace_source

    def _load_semantic_results(self) -> List[Dict[str, Any]]:
        if self.__intermediate_dir is None or not self.__intermediate_dir.exists():
            return []

        results = []
        component_dirs = sorted(path for path in self.__intermediate_dir.iterdir() if path.is_dir() and path.name.startswith("component_"))
        self._log(f"[CurveParameterizer] semantic component dirs: {len(component_dirs)}")

        for component_dir in component_dirs:
            debug_path = component_dir / "semantic_debug.json"
            if not debug_path.exists():
                self._log(f"[CurveParameterizer] missing semantic_debug.json: {component_dir}")
                continue

            debug = self._load_json_file(debug_path, default={})
            primitives = debug.get("primitives") or debug.get("final_segments") or []
            resampled_points = debug.get("resampled_points") or []

            component = {
                "component_id": self._component_id_from_name(component_dir.name),
                "component_dir": str(component_dir),
                "debug_path": str(debug_path),
                "closed": bool(debug.get("closed", False)),
                "sampled_point_count": int(len(resampled_points)),
                "resampled_points": resampled_points,
                "smoothed_points": debug.get("smoothed_points", []),
                "raw_keypoints": debug.get("raw_keypoints", []),
                "refined_keypoints": debug.get("refined_keypoints", []),
                "keypoints": debug.get("keypoints", []),
                "global_lines": debug.get("global_lines", []),
                "collapsed_full_loop_arc": bool(debug.get("collapsed_full_loop_arc", False)),
                "segments": [self._normalize_segment(seg, idx) for idx, seg in enumerate(primitives)],
                "debug": {
                    "initial_segments": debug.get("initial_segments", []),
                    "dp_segments": debug.get("dp_segments", []),
                    "merged_segments": debug.get("merged_segments", []),
                    "final_segments": debug.get("final_segments", []),
                    "boundary_field_weight": debug.get("boundary_field_weight", 0.0),
                    "boundary_field_mean": debug.get("boundary_field_mean", 0.0),
                    "boundary_field_max": debug.get("boundary_field_max", 0.0),
                },
            }
            results.append(component)
            self._log(
                f"[CurveParameterizer] component {component['component_id']}: "
                f"closed={component['closed']}, points={component['sampled_point_count']}, "
                f"segments={len(component['segments'])}"
            )

        return results

    @staticmethod
    def _normalize_segment(segment: Dict[str, Any], segment_id: int) -> Dict[str, Any]:
        kind = str(segment.get("kind", segment.get("type", "spline"))).lower()
        if kind == "circle":
            kind = "arc"
        if kind in ("bspline", "curve"):
            kind = "spline"

        out = dict(segment)
        out["segment_id"] = int(segment_id)
        out["kind"] = kind
        out["type"] = kind
        out["start_idx"] = int(out.get("start_idx", 0) or 0)
        out["end_idx"] = int(out.get("end_idx", 0) or 0)
        out["effective_params"] = int(out.get("effective_params", 0) or 0)
        out["max_error"] = float(out.get("max_error", 0.0) or 0.0)
        out["mean_error"] = float(out.get("mean_error", 0.0) or 0.0)
        return out

    def _print_summary(self, elapsed: float) -> None:
        total_segments = sum(len(c.get("segments", [])) for c in self.__results)
        total_points = sum(int(c.get("sampled_point_count", 0)) for c in self.__results)
        total_effective = sum(
            int(seg.get("effective_params", 0) or 0)
            for c in self.__results
            for seg in c.get("segments", [])
        )
        by_kind: Dict[str, int] = {}
        for c in self.__results:
            for seg in c.get("segments", []):
                kind = str(seg.get("kind", "unknown"))
                by_kind[kind] = by_kind.get(kind, 0) + 1

        self._log("[CurveParameterizer] Parameterization summary")
        self._log(f"  components:       {len(self.__results)}")
        self._log(f"  sampled points:   {total_points}")
        self._log(f"  semantic segments:{total_segments}")
        self._log(f"  by kind:          {by_kind}")
        self._log(f"  effective params: {total_effective}")
        if self.__metrics:
            self._log(f"  metrics total_semantic_segments: {self.__metrics.get('total_semantic_segments')}")
            self._log(f"  metrics mean_error_px:          {self.__metrics.get('mean_error_px', self.__metrics.get('mean_component_error_px'))}")
            self._log(f"  metrics max_component_error_px:  {self.__metrics.get('max_component_error_px')}")
        self._log(f"  elapsed_sec:      {elapsed:.3f}")
        self._log("[CurveParameterizer] Parameterization finished")

    def _build_visual_base(self) -> np.ndarray:
        if self.original_img is not None:
            if self.original_img.ndim == 2:
                return cv2.cvtColor(self.original_img, cv2.COLOR_GRAY2BGR)
            return self.original_img.copy()

        if self.__trace_image_path is not None and self.__trace_image_path.exists():
            img = cv2.imread(str(self.__trace_image_path), cv2.IMREAD_COLOR)
            if img is not None:
                return img

        if self.edges is not None:
            canvas = np.full((*self.edges.shape[:2], 3), 255, dtype=np.uint8)
            canvas[self.edges > 0] = (40, 40, 40)
            return canvas

        raise RuntimeError("No image available for visualization.")

    @staticmethod
    def _draw_visual_legend(canvas: np.ndarray) -> None:
        cv2.putText(canvas, "line", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 144, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "arc", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "spline", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 205, 50), 2, cv2.LINE_AA)
        cv2.putText(canvas, "red=start", (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _slice_segment_points(points: np.ndarray, start_idx: int, end_idx: int, closed: bool) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        n = len(points)
        s = int(start_idx) % n
        e = int(end_idx) % n
        if closed:
            if s <= e:
                return points[s:e + 1]
            return np.vstack([points[s:], points[:e + 1]])
        s = max(0, min(n - 1, s))
        e = max(0, min(n - 1, e))
        if s <= e:
            return points[s:e + 1]
        return points[e:s + 1]

    @staticmethod
    def _clip_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
        arr = np.round(points).astype(np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)
        return arr

    def _default_save_dir(self) -> Path:
        if self.image_path is not None:
            return self.image_path.parent / "parameterization_output"
        return Path.cwd() / "parameterization_output"

    def _default_output_prefix(self) -> str:
        if self.image_path is not None:
            return self.image_path.stem
        return "curve_parameterization"

    def _results_without_points(self) -> List[Dict[str, Any]]:
        compact = []
        for component in self.__results:
            item = dict(component)
            item.pop("resampled_points", None)
            item.pop("smoothed_points", None)
            item.pop("debug", None)
            compact.append(item)
        return compact

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _component_id_from_name(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except Exception:
            return -1

    @staticmethod
    def _normalize_edges(edges: np.ndarray) -> np.ndarray:
        if edges.ndim == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        edges = edges.astype(np.uint8)
        _, binary = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def _load_json_file(path: Optional[Path], default):
        if path is None or not Path(path).exists():
            return default
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        suffix = path.suffix or ".png"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            raise ValueError(f"Failed to encode image as {suffix}: {path}")
        encoded.tofile(str(path))

    @classmethod
    def _to_jsonable(cls, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._to_jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [cls._to_jsonable(v) for v in value]
        return value


if __name__ == "__main__":
    image_path = r"D:\cst2py_box\Auto_py2cst_v0.71\clean_parametric_dataset_50\images\clean_00000.png"
    params = NewParams(image_path=image_path)
    parameterizer = params.parameterize()
    parameterizer.save_json()
    parameterizer.visualize()
