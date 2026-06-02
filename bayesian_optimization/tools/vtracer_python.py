"""
VTracer Python implementation.

这个文件是当前项目里“参数化算法本体”的主要位置。

它同时支持两类矢量化/参数化任务：

1. 普通填充轮廓矢量化
   - 输入彩色或黑白栅格图；
   - 通过颜色量化、轮廓提取、polygon/spline 方式生成 SVG。

2. 中心线语义参数化，也就是本项目当前最关心的路径
   - 输入黑白结构图；
   - 二值化得到前景 mask；
   - 细化成 skeleton / 或读取外部 centerline seed polyline；
   - 把骨架转成有序折线；
   - 按弧长重采样；
   - 检测角点/语义边界；
   - 用动态规划在 line / arc / spline 三类 primitive 之间选择；
   - 合并相邻同类或几何一致的片段；
   - 输出 SVG、metrics JSON、semantic_debug.json。

最重要的调用链：

    VTracerPython.to_svg()
        -> _to_centerline_svg()                         # 中心线路径入口
            -> _binarize_foreground_mask()
            -> _zhang_suen_thinning()
            -> _skeleton_component_to_path()
            -> _resample_polyline()
            -> _fit_centerline_path()                   # 语义参数化核心
                -> _detect_multiscale_keypoints()
                -> _oversegment_semantic_path()
                -> _dynamic_programming_segments()
                -> _merge_semantic_segments()
                -> _build_semantic_path_fit()
                    -> _fit_segment_model()
                        -> _fit_line_segment()
                        -> _fit_arc_segment()
                        -> _fit_spline_segment()

调试时优先看：

- TraceConfig：所有关键参数；
- _fit_centerline_path：总流程；
- _dynamic_programming_segments_on_intervals：DP 如何选分段；
- _fit_segment_model：单段如何在 line / arc / spline 之间选择；
- semantic_debug.json：每个样本的中间结果。
"""

import argparse
import html
import json
import math
import os
import warnings

# 在导入 sklearn/joblib 之前：Windows 下 loky 枚举物理核常失败；设 LOKY 并抑制该条已知 UserWarning
if os.environ.get("LOKY_MAX_CPU_COUNT") is None:
    try:
        import multiprocessing as _mp

        os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, int(_mp.cpu_count() or 4)))
    except Exception:
        os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    import svgwrite
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    class _SimpleSVGDrawing:
        """Small subset of svgwrite used by this module."""

        def __init__(self, filename: str, size: Tuple[str, str], viewBox: str):
            self.filename = filename
            self.size = size
            self.viewBox = viewBox
            self._elements: List[str] = []

        @staticmethod
        def _attr_name(name: str) -> str:
            return "viewBox" if name == "viewBox" else name.replace("_", "-")

        @classmethod
        def _format_attrs(cls, attrs: Dict[str, Any]) -> str:
            parts = []
            for key, value in attrs.items():
                if value is None:
                    continue
                name = cls._attr_name(str(key))
                text = html.escape(str(value), quote=True)
                parts.append(f'{name}="{text}"')
            return " ".join(parts)

        def path(self, d: str, **attrs: Any) -> str:
            return f'<path {self._format_attrs({"d": d, **attrs})} />'

        def add(self, element: str) -> None:
            self._elements.append(element)

        def save(self) -> None:
            os.makedirs(os.path.dirname(self.filename) or ".", exist_ok=True)
            width, height = self.size
            attrs = self._format_attrs(
                {
                    "xmlns": "http://www.w3.org/2000/svg",
                    "width": width,
                    "height": height,
                    "viewBox": self.viewBox,
                }
            )
            body = "\n  ".join(self._elements)
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write(f"<svg {attrs}>\n")
                if body:
                    f.write(f"  {body}\n")
                f.write("</svg>\n")

    class _SVGWriteFallback:
        Drawing = _SimpleSVGDrawing

        @staticmethod
        def rgb(r: int, g: int, b: int) -> str:
            return f"rgb({int(r)},{int(g)},{int(b)})"

    svgwrite = _SVGWriteFallback()

try:
    from sklearn.cluster import DBSCAN, MiniBatchKMeans
except Exception:  # pragma: no cover
    MiniBatchKMeans = None
    DBSCAN = None


ColorMode = Literal["color", "bw"]
TraceMode = Literal["pixel", "polygon", "spline"]
HierarchicalMode = Literal["stacked", "cutout"]
TraceStyle = Literal["fill", "centerline"]
SegmentKind = Literal["line", "arc", "spline"]


# ---------------------------------------------------------------------------
# 数据结构区
# ---------------------------------------------------------------------------
# 这些 dataclass 是算法在各阶段之间传递信息的“数据包”。
# 读懂这些结构，比直接钻进函数细节更容易建立全局认识。

@dataclass
class LayerPath:
    """SVG 中的一条 path。

    对 fill 模式来说，一种颜色通常对应多个 LayerPath；
    对 centerline 模式来说，一个 skeleton component 通常对应一条 path。
    """

    layer_id: int
    color: Tuple[int, int, int]
    d: str
    fill: str = "none"
    stroke: str = "none"
    stroke_width: float = 0.0


@dataclass
class ComponentMetric:
    """单个连通中心线组件的评估指标。

    这些字段最终会写入 *_metrics.json。
    其中 semantic_* 字段描述最终语义分段结果；
    error / reduction_ratio 字段用于判断拟合精度和参数压缩程度。
    """

    component_id: int
    closed: bool
    samples: int
    control_points: int
    effective_params: int
    line_segments: int
    arc_segments: int
    curve_segments: int
    bspline_control_points: int
    keypoints: int
    """DP+合并后的语义段数量（与 primitive_summary 条目一致）。"""
    semantic_segment_count: int
    semantic_segments_line: int
    semantic_segments_arc: int
    semantic_segments_spline: int
    rmse_px: float
    mean_error_px: float
    max_error_px: float
    reduction_ratio: float


@dataclass
class BSplineFit:
    """B 样条拟合结果。

    control_points 是样条控制点；
    fitted_points 是按原采样参数回代得到的拟合点；
    errors 是原点到拟合点的距离，用于 DP 代价和 metrics。
    """

    control_points: np.ndarray
    fitted_points: np.ndarray
    errors: np.ndarray
    closed: bool
    degree: int
    knots: Optional[np.ndarray] = None


@dataclass
class CenterlinePathFit:
    """一条中心线组件完成参数化后的整体结果。

    d 是 SVG path 字符串；
    primitive_summary 是 line / arc / spline 的简要参数列表；
    semantic_debug 会写到 semantic_debug.json，用于调试分段过程。
    """

    d: str
    errors: np.ndarray
    effective_params: int
    line_segments: int
    arc_segments: int
    curve_segments: int
    bspline_control_points: int
    keypoints: int = 0
    primitive_summary: Optional[List[Dict[str, Any]]] = None
    semantic_debug: Optional[Dict[str, Any]] = None


@dataclass
class GlobalLine:
    """全局直线簇。

    一些结构里会有多段短线其实处在同一条直线上。
    先识别 GlobalLine，可以让后续 DP 或合并阶段把这些短线重新归并。
    """

    line_id: int
    theta: float
    distance: float
    normal: np.ndarray
    direction: np.ndarray
    anchor: np.ndarray
    support_segments: int


@dataclass
class ArcFit:
    """圆弧拟合结果。"""

    center: np.ndarray
    radius: float
    start_angle: float
    sweep_angle: float
    errors: np.ndarray


@dataclass
class SemanticSegment:
    """语义分段的基本单位。

    kind 只能是 line / arc / spline。
    start_idx / end_idx 是该段在重采样折线中的索引范围。
    errors 用于衡量该 primitive 对该段的拟合误差。
    effective_params 是该段的有效参数复杂度：
    - line 通常为 1；
    - arc 通常为 2；
    - spline 与控制点数量有关。
    """

    kind: SegmentKind
    start_idx: int
    end_idx: int
    points: np.ndarray
    errors: np.ndarray
    effective_params: int
    line_segments: int
    arc_segments: int
    curve_segments: int
    bspline_control_points: int
    line_start: Optional[np.ndarray] = None
    line_end: Optional[np.ndarray] = None
    arc_fit: Optional[ArcFit] = None
    bspline_fit: Optional[BSplineFit] = None
    global_line_id: Optional[int] = None


@dataclass
class TraceConfig:
    """VTracer 参数配置。

    调参基本都从这里开始。对于当前项目最重要的是 centerline 相关参数：

    - fit_tolerance：允许的拟合误差尺度；
    - resample_step：中心线重采样间距；
    - gaussian_sigma：重采样折线平滑强度；
    - keypoint_*：角点/语义边界候选检测；
    - dp_*：动态规划分段代价；
    - line_*：直线合并与直线簇判断；
    - arc_*：圆弧拟合和圆弧合并；
    - spline_ctrl_penalty：样条控制点惩罚，越大越不喜欢 spline。
    """

    image_path: str
    color_mode: ColorMode = "color"
    color_count: int = 12
    color_precision: int = 6
    hierarchical: HierarchicalMode = "stacked"
    mode: TraceMode = "spline"
    trace_style: TraceStyle = "fill"
    corner_threshold: float = 60.0
    segment_length: float = 6.0
    splice_threshold: float = 45.0
    simplify_tolerance: float = 2.0
    path_precision: int = 2
    filter_speckle: int = 10
    median_ksize: int = 3
    fit_tolerance: float = 1.5
    min_control_points: int = 6
    max_control_points: int = 60
    resample_step: float = 4.0
    gaussian_sigma: float = 1.0
    semantic_window_size: int = 11
    keypoint_angle_threshold_deg: float = 28.0
    keypoint_refine_radius: int = 4
    # Model-guided multiscale keypoints: fuse turn-angle signal with line-vs-arc residual in local windows.
    keypoint_use_model_guided: bool = True
    keypoint_model_multiscale_votes: bool = False
    keypoint_angle_guided_weight: float = 0.55
    keypoint_model_guided_weight: float = 0.45
    keypoint_scale_half_fractions: Tuple[float, ...] = (0.014, 0.028, 0.048)
    keypoint_multiscale_min_votes: int = 3
    keypoint_nms_half_fraction: float = 0.35
    keypoint_model_peak_bridge: int = 9
    # 沿折线索引间距 ≤ 该值且较钝的关键点并入更尖邻居，抑制高斯平滑后在真角旁产生的伪关键点（矩形/正多边形）
    keypoint_merge_max_polyline_gap: int = 4
    # DP 候选边界（keypoint + semantic break）近邻压缩半径；0 关闭。用于去掉真角旁的重复伪边界。
    boundary_candidate_merge_max_polyline_gap: int = 5
    # DP 候选贴近 keypoint 时，只有自身转角弱于该值才作为直线伪边界删除，避免复杂样本真角被吞。
    boundary_candidate_min_sharpness_to_keep: float = 18.0
    # protected 边界是否处在直线段上的多尺度判定半径；用于允许跨过直线上的伪角点合并。
    protected_straight_window_radius: int = 4
    dp_max_segment_points: int = 72
    dp_complexity_weight: float = 0.8
    # 边界概率场 B(i)∈[0,1] 进入 DP：对段内未切断的局部峰加权惩罚（软约束，替代仅靠硬关键点）。
    # 0 关闭；建议与 dp_complexity_weight 同量级小正数，由评测微调。
    dp_boundary_field_weight: float = 0.08
    line_merge_angle_deg: float = 12.0
    # 后验 line-run 压缩：连续 line 段整体仍满足直线误差时合并，减少直线上的多余点。
    line_run_merge_error_factor: float = 1.18
    # 后验 line-short-line 桥接：中间短非 line 段若整体仍是直线，则压成单线。
    line_bridge_max_points: int = 8
    line_cluster_eps: float = 0.16
    line_cluster_min_samples: int = 2
    arc_radius_rel_tol: float = 0.2
    arc_center_tol: float = 2.0
    arc_min_sweep_deg: float = 20.0
    spline_ctrl_penalty: float = 0.08
    smoothness_weight: float = 1e-8
    skeleton_prune_length: int = 8
    centerline_trace_edge_contours: bool = False
    save_intermediates: Optional[str] = None
    metrics_path: Optional[str] = None
    # centerline：若指向存在的 .npy（形状 N×2 float64），则跳过骨架细化，直接对该折线重采样后走语义分段/DP
    centerline_seed_npy_path: Optional[str] = None


@dataclass
class ResolvedPaths:
    """命令行入口解析出的输入/输出路径。"""

    input_path: str
    output_path: str
    sample_name: str
    metrics_path: Optional[str]
    save_intermediates: Optional[str]


class VTracerPython:
    """
    A local Python raster-to-vector tracer that supports:
    1) classic filled contour extraction
    2) centerline skeleton extraction for thick raster strokes
    3) adaptive low-parameter cubic B-spline fitting with metrics
    """

    def __init__(self, config: TraceConfig):
        """读取输入图并初始化运行状态。

        这里还没有开始参数化，只是读图、可选中值滤波、
        保存 RGB/BGR 版本，并初始化颜色量化和 path 输出容器。
        """

        self.cfg = config
        bgr = cv2.imread(config.image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {config.image_path}")

        if config.median_ksize > 1:
            k = config.median_ksize if config.median_ksize % 2 == 1 else config.median_ksize + 1
            bgr = cv2.medianBlur(bgr, k)

        self.bgr = bgr
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.rgb.shape[:2]
        self.palette: Optional[np.ndarray] = None
        self.labels_2d: Optional[np.ndarray] = None
        self.layer_paths: List[LayerPath] = []
        self._active_spline_cache: Dict[Tuple[int, int, bool], SemanticSegment] = {}

    def quantize(self) -> None:
        """颜色量化入口，主要用于 fill/polygon/spline 普通矢量化。

        当前中心线参数化通常设置 color_mode="bw"，
        因此这里只会得到黑白二值标签。
        """

        if self.cfg.color_mode == "bw":
            gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.labels_2d = mask.astype(np.int32)
            self.palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
            return

        bits = int(np.clip(self.cfg.color_precision, 1, 8))
        shift = 8 - bits
        rgb_reduced = ((self.rgb >> shift) << shift).astype(np.uint8)
        pixels = rgb_reduced.reshape(-1, 3).astype(np.float32)

        n_colors = int(np.clip(self.cfg.color_count, 2, 256))
        labels, centers = self._cluster_colors(pixels, n_colors)
        self.labels_2d = labels.reshape(self.height, self.width).astype(np.int32)
        self.palette = centers

    def _cluster_colors(self, pixels: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray]:
        """把 RGB 像素聚类到有限颜色数。

        优先用 sklearn MiniBatchKMeans；不可用或失败时回退到 cv2.kmeans。
        这个函数服务于 fill 模式，不是中心线语义分段的核心。
        """

        if MiniBatchKMeans is not None:
            try:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_colors,
                    batch_size=4096,
                    n_init=5,
                    random_state=42,
                )
                labels = kmeans.fit_predict(pixels)
                centers = np.clip(np.round(kmeans.cluster_centers_), 0, 255).astype(np.uint8)
                return labels.astype(np.int32), centers
            except Exception as ex:
                print(f"[warn] sklearn kmeans failed, fallback to cv2.kmeans: {ex}")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.8)
        compactness, labels, centers = cv2.kmeans(
            pixels.astype(np.float32),
            n_colors,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        _ = compactness
        labels = labels.reshape(-1).astype(np.int32)
        centers = np.clip(np.round(centers), 0, 255).astype(np.uint8)
        return labels, centers

    # ------------------------------------------------------------------
    # 顶层导出流程：普通填充矢量化 / 中心线参数化
    # ------------------------------------------------------------------

    def to_svg(self, output_path: str) -> None:
        """总入口：根据 trace_style 分派到不同矢量化流程。

        - trace_style="centerline"：走中心线语义参数化；
        - 其他情况：走普通填充区域矢量化。
        """

        if self.cfg.trace_style == "centerline":
            self._to_centerline_svg(output_path)
            return

        if self.palette is None or self.labels_2d is None:
            self.quantize()

        assert self.palette is not None
        assert self.labels_2d is not None

        layer_ids = np.arange(len(self.palette), dtype=np.int32)
        layer_areas = np.bincount(self.labels_2d.reshape(-1), minlength=len(self.palette))
        order = layer_ids[np.argsort(-layer_areas)] if self.cfg.hierarchical == "stacked" else layer_ids
        self.layer_paths = []

        def process_layer(layer_id: int) -> List[LayerPath]:
            mask = (self.labels_2d == layer_id).astype(np.uint8) * 255
            mask = self._remove_speckles(mask)
            d_list = self._trace_mask(mask)
            color = tuple(int(v) for v in self.palette[layer_id].tolist())
            return [LayerPath(layer_id=layer_id, color=color, d=d, fill="fill") for d in d_list]

        max_workers = min(8, max(1, len(order)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(process_layer, int(layer_id)) for layer_id in order]
            for i, future in enumerate(as_completed(futures), start=1):
                self.layer_paths.extend(future.result())
                print(f"[trace] layer {i}/{len(order)} done")

        if self.cfg.hierarchical == "stacked":
            rank = {int(layer_id): i for i, layer_id in enumerate(order)}
            self.layer_paths.sort(key=lambda x: rank.get(x.layer_id, 10**9))

        dwg = svgwrite.Drawing(
            filename=output_path,
            size=(f"{self.width}px", f"{self.height}px"),
            viewBox=f"0 0 {self.width} {self.height}",
        )
        for layer in self.layer_paths:
            fill = svgwrite.rgb(layer.color[0], layer.color[1], layer.color[2])
            dwg.add(dwg.path(d=layer.d, fill=fill, stroke="none", fill_rule="evenodd"))
        dwg.save()
        print(f"[done] SVG saved: {output_path}")

    def _to_centerline_svg(self, output_path: str) -> None:
        """中心线参数化主入口。

        如果 cfg.centerline_seed_npy_path 指向已有折线，则跳过骨架化，
        直接调用 _to_centerline_svg_from_seed_polyline。

        否则流程是：
        1. 二值化前景；
        2. Zhang-Suen 细化成 skeleton；
        3. 去掉短毛刺；
        4. 找 skeleton 连通域；
        5. 每个连通域转为有序路径；
        6. 重采样；
        7. 调用 _fit_centerline_path 做语义分段和拟合；
        8. 写 SVG 和 metrics。
        """

        seed_path = getattr(self.cfg, "centerline_seed_npy_path", None)
        if isinstance(seed_path, str) and seed_path and os.path.isfile(seed_path):
            self._to_centerline_svg_from_seed_polyline(output_path, seed_path)
            return

        mask = self._binarize_foreground_mask()
        if self.cfg.save_intermediates:
            self._save_intermediate("mask_binary.png", mask)

        if bool(getattr(self.cfg, "centerline_trace_edge_contours", False)):
            self._to_centerline_svg_from_edge_contours(output_path, mask)
            return

        skeleton = self._zhang_suen_thinning(mask)
        if self.cfg.skeleton_prune_length > 0:
            skeleton = self._prune_skeleton_spurs(skeleton, int(self.cfg.skeleton_prune_length))
        if self.cfg.save_intermediates:
            self._save_intermediate("mask_skeleton.png", skeleton)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        components = [(label, int(stats[label, cv2.CC_STAT_AREA])) for label in range(1, n_labels)]
        components.sort(key=lambda item: -item[1])

        self.layer_paths = []
        metrics: List[ComponentMetric] = []
        color = (0, 0, 0)

        output_component_id = 1
        for label_id, area in components:
            if area < max(3, self.cfg.filter_speckle):
                continue
            component_mask = (labels == label_id).astype(np.uint8) * 255
            for points, closed in self._skeleton_component_to_paths(component_mask):
                if len(points) < 4:
                    continue
                path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
                if closed and len(points) > 2:
                    path_length += float(np.linalg.norm(points[0] - points[-1]))
                if path_length < max(8.0, float(self.cfg.resample_step) * 3.0):
                    continue

                sampled = self._resample_polyline(points, closed=closed, step=max(0.5, self.cfg.resample_step))
                fit = self._fit_centerline_path(sampled, closed=closed)
                d = fit.d
                if not d:
                    continue
                component_id = output_component_id
                output_component_id += 1
                if self.cfg.save_intermediates and fit.semantic_debug is not None:
                    self._write_semantic_debug(component_id, fit.semantic_debug)

                self.layer_paths.append(
                    LayerPath(
                        layer_id=component_id,
                        color=color,
                        d=d,
                        fill="none",
                        stroke="stroke",
                        stroke_width=1.0,
                    )
                )

                errors = fit.errors
                prim = fit.primitive_summary or []
                n_sem = len(prim)
                n_line_k = sum(1 for p in prim if p.get("kind") == "line")
                n_arc_k = sum(1 for p in prim if p.get("kind") == "arc")
                n_spline_k = sum(1 for p in prim if p.get("kind") == "spline")
                metrics.append(
                    ComponentMetric(
                        component_id=component_id,
                        closed=closed,
                        samples=int(len(sampled)),
                        control_points=int(fit.effective_params),
                        effective_params=int(fit.effective_params),
                        line_segments=int(fit.line_segments),
                        arc_segments=int(fit.arc_segments),
                        curve_segments=int(fit.curve_segments),
                        bspline_control_points=int(fit.bspline_control_points),
                        keypoints=int(fit.keypoints),
                        semantic_segment_count=int(n_sem),
                        semantic_segments_line=int(n_line_k),
                        semantic_segments_arc=int(n_arc_k),
                        semantic_segments_spline=int(n_spline_k),
                        rmse_px=float(np.sqrt(np.mean(errors**2))),
                        mean_error_px=float(np.mean(errors)),
                        max_error_px=float(np.max(errors)),
                        reduction_ratio=float(fit.effective_params / max(1, len(sampled))),
                    )
                )

        dwg = svgwrite.Drawing(
            filename=output_path,
            size=(f"{self.width}px", f"{self.height}px"),
            viewBox=f"0 0 {self.width} {self.height}",
        )
        for layer in self.layer_paths:
            stroke = svgwrite.rgb(layer.color[0], layer.color[1], layer.color[2])
            dwg.add(
                dwg.path(
                    d=layer.d,
                    fill="none",
                    stroke=stroke,
                    stroke_width=layer.stroke_width,
                    stroke_linecap="round",
                    stroke_linejoin="miter",
                    stroke_miterlimit=8,
                )
            )
        dwg.save()
        print(f"[done] SVG saved: {output_path}")
        self._write_metrics(output_path, metrics)

    def _to_centerline_svg_from_edge_contours(self, output_path: str, mask: np.ndarray) -> None:
        """Trace already-extracted edge images by contour order.

        For one-pixel or thin boundary maps, skeleton graph splitting can create
        visible breaks at junction-like pixels.  Contour tracing preserves the
        ordered boundary directly, which is a better fit for pre-extracted
        closed structure outlines.
        """

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_len = max(8, int(self.cfg.filter_speckle))
        contours = [c for c in contours if len(c) >= min_len]
        contours.sort(key=lambda c: -float(cv2.arcLength(c, closed=True)))

        self.layer_paths = []
        metrics: List[ComponentMetric] = []
        color = (0, 0, 0)
        component_id = 1

        for contour in contours:
            pts = contour.reshape(-1, 2).astype(np.float64)
            pts = self._dedupe_consecutive(pts)
            if len(pts) < 4:
                continue
            closed = True
            sampled = self._resample_polyline(pts, closed=closed, step=max(0.5, self.cfg.resample_step))
            fit = self._fit_centerline_path(sampled, closed=closed)
            d = fit.d
            if not d:
                continue
            if self.cfg.save_intermediates and fit.semantic_debug is not None:
                if isinstance(fit.semantic_debug, dict):
                    fit.semantic_debug["pipeline"] = "edge_contour_then_semantic_dp"
                self._write_semantic_debug(component_id, fit.semantic_debug)

            self.layer_paths.append(
                LayerPath(
                    layer_id=component_id,
                    color=color,
                    d=d,
                    fill="none",
                    stroke="stroke",
                    stroke_width=1.0,
                )
            )

            errors = fit.errors
            prim = fit.primitive_summary or []
            n_sem = len(prim)
            n_line_k = sum(1 for p in prim if p.get("kind") == "line")
            n_arc_k = sum(1 for p in prim if p.get("kind") == "arc")
            n_spline_k = sum(1 for p in prim if p.get("kind") == "spline")
            metrics.append(
                ComponentMetric(
                    component_id=component_id,
                    closed=closed,
                    samples=int(len(sampled)),
                    control_points=int(fit.effective_params),
                    effective_params=int(fit.effective_params),
                    line_segments=int(fit.line_segments),
                    arc_segments=int(fit.arc_segments),
                    curve_segments=int(fit.curve_segments),
                    bspline_control_points=int(fit.bspline_control_points),
                    keypoints=int(fit.keypoints),
                    semantic_segment_count=int(n_sem),
                    semantic_segments_line=int(n_line_k),
                    semantic_segments_arc=int(n_arc_k),
                    semantic_segments_spline=int(n_spline_k),
                    rmse_px=float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0,
                    mean_error_px=float(np.mean(errors)) if len(errors) else 0.0,
                    max_error_px=float(np.max(errors)) if len(errors) else 0.0,
                    reduction_ratio=float(fit.effective_params / max(1, len(sampled))),
                )
            )
            component_id += 1

        dwg = svgwrite.Drawing(
            filename=output_path,
            size=(f"{self.width}px", f"{self.height}px"),
            viewBox=f"0 0 {self.width} {self.height}",
        )
        for layer in self.layer_paths:
            stroke = svgwrite.rgb(layer.color[0], layer.color[1], layer.color[2])
            dwg.add(
                dwg.path(
                    d=layer.d,
                    fill="none",
                    stroke=stroke,
                    stroke_width=layer.stroke_width,
                    stroke_linecap="round",
                    stroke_linejoin="miter",
                    stroke_miterlimit=8,
                )
            )
        dwg.save()
        print(f"[done] SVG saved: {output_path}")
        self._write_metrics(output_path, metrics)

    def _to_centerline_svg_from_seed_polyline(self, output_path: str, seed_path: str) -> None:
        # 外部 seed polyline 模式：用于已经有一条拟合中心线/轮廓线的情况。
        # 它可以绕过 skeleton 提取，直接进入重采样 + 语义分段。
        """从外部折线种子（如 OptimizedBSplineFitter 的 fitting['points']）生成中心线语义 SVG，跳过骨架。"""
        raw = np.load(seed_path)
        if raw.ndim != 2 or raw.shape[1] != 2:
            raise ValueError(f"centerline seed must be (N,2), got {getattr(raw, 'shape', None)} from {seed_path}")
        pts = np.asarray(raw, dtype=np.float64)
        if len(pts) < 4:
            raise ValueError(f"centerline seed too short: {len(pts)}")
        gap = float(np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1]))
        closed = gap < max(2.5, float(self.cfg.resample_step) * 1.8)

        sampled = self._resample_polyline(pts, closed=closed, step=max(0.5, self.cfg.resample_step))
        fit = self._fit_centerline_path(sampled, closed=closed)
        d = fit.d
        if not d:
            raise RuntimeError("centerline seed produced empty path d-string")

        component_id = 1
        if self.cfg.save_intermediates and fit.semantic_debug is not None:
            if isinstance(fit.semantic_debug, dict):
                fit.semantic_debug["centerline_seed_npy_path"] = seed_path
                fit.semantic_debug["pipeline"] = "seed_polyline_then_semantic_dp"
            self._write_semantic_debug(component_id, fit.semantic_debug)

        color = (0, 0, 0)
        self.layer_paths = []
        errors = fit.errors
        prim = fit.primitive_summary or []
        n_sem = len(prim)
        n_line_k = sum(1 for p in prim if p.get("kind") == "line")
        n_arc_k = sum(1 for p in prim if p.get("kind") == "arc")
        n_spline_k = sum(1 for p in prim if p.get("kind") == "spline")
        metrics = [
            ComponentMetric(
                component_id=component_id,
                closed=closed,
                samples=int(len(sampled)),
                control_points=int(fit.effective_params),
                effective_params=int(fit.effective_params),
                line_segments=int(fit.line_segments),
                arc_segments=int(fit.arc_segments),
                curve_segments=int(fit.curve_segments),
                bspline_control_points=int(fit.bspline_control_points),
                keypoints=int(fit.keypoints),
                semantic_segment_count=int(n_sem),
                semantic_segments_line=int(n_line_k),
                semantic_segments_arc=int(n_arc_k),
                semantic_segments_spline=int(n_spline_k),
                rmse_px=float(np.sqrt(np.mean(errors**2))) if len(errors) else 0.0,
                mean_error_px=float(np.mean(errors)) if len(errors) else 0.0,
                max_error_px=float(np.max(errors)) if len(errors) else 0.0,
                reduction_ratio=float(fit.effective_params / max(1, len(sampled))),
            )
        ]
        self.layer_paths.append(
            LayerPath(
                layer_id=component_id,
                color=color,
                d=d,
                fill="none",
                stroke="stroke",
                stroke_width=1.0,
            )
        )

        dwg = svgwrite.Drawing(
            filename=output_path,
            size=(f"{self.width}px", f"{self.height}px"),
            viewBox=f"0 0 {self.width} {self.height}",
        )
        for layer in self.layer_paths:
            stroke = svgwrite.rgb(layer.color[0], layer.color[1], layer.color[2])
            dwg.add(
                dwg.path(
                    d=layer.d,
                    fill="none",
                    stroke=stroke,
                    stroke_width=layer.stroke_width,
                    stroke_linecap="round",
                    stroke_linejoin="miter",
                    stroke_miterlimit=8,
                )
            )
        dwg.save()
        print(f"[done] SVG saved: {output_path}")
        self._write_metrics(output_path, metrics)

    def _write_metrics(self, output_path: str, metrics: List[ComponentMetric]) -> None:
        """把 ComponentMetric 汇总后写入 *_metrics.json。"""

        if not metrics:
            return
        target_path = self.cfg.metrics_path
        if not target_path:
            root, _ = os.path.splitext(output_path)
            target_path = f"{root}_metrics.json"
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        aggregate = {
            "component_count": len(metrics),
            "mean_rmse_px": float(np.mean([m.rmse_px for m in metrics])),
            "max_component_error_px": float(np.max([m.max_error_px for m in metrics])),
            "mean_reduction_ratio": float(np.mean([m.reduction_ratio for m in metrics])),
            "total_line_segments": int(sum(m.line_segments for m in metrics)),
            "total_arc_segments": int(sum(m.arc_segments for m in metrics)),
            "total_curve_segments": int(sum(m.curve_segments for m in metrics)),
            "total_semantic_segments": int(sum(m.semantic_segment_count for m in metrics)),
            "total_semantic_line": int(sum(m.semantic_segments_line for m in metrics)),
            "total_semantic_arc": int(sum(m.semantic_segments_arc for m in metrics)),
            "total_semantic_spline": int(sum(m.semantic_segments_spline for m in metrics)),
            "components": [asdict(m) for m in metrics],
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, ensure_ascii=True, indent=2)
        print(f"[done] metrics saved: {target_path}")

    def _save_intermediate(self, filename: str, image: np.ndarray) -> None:
        """保存中间图像，比如 binary mask 和 skeleton。"""

        assert self.cfg.save_intermediates is not None
        os.makedirs(self.cfg.save_intermediates, exist_ok=True)
        cv2.imwrite(os.path.join(self.cfg.save_intermediates, filename), image)

    def _write_semantic_debug(self, component_id: int, debug_payload: Dict[str, Any]) -> None:
        """保存某个 component 的语义分段调试信息。

        输出目录结构：
            save_intermediates/component_XXXX/semantic_debug.json

        这是调试算法最重要的文件之一。
        """

        assert self.cfg.save_intermediates is not None
        component_dir = os.path.join(self.cfg.save_intermediates, f"component_{component_id:03d}")
        os.makedirs(component_dir, exist_ok=True)

        with open(os.path.join(component_dir, "semantic_debug.json"), "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=True, indent=2)

        resampled = np.asarray(debug_payload.get("resampled_points", []), dtype=np.float64)
        smoothed = np.asarray(debug_payload.get("smoothed_points", []), dtype=np.float64)
        raw_keypoints = debug_payload.get("raw_keypoints", [])
        refined_keypoints = debug_payload.get("refined_keypoints", [])
        keypoints = debug_payload.get("keypoints", [])
        initial_segments = debug_payload.get("initial_segments", [])
        merged_segments = debug_payload.get("merged_segments", [])

        cv2.imwrite(
            os.path.join(component_dir, "resampled_points.png"),
            self._draw_debug_points(resampled, closed=bool(debug_payload.get("closed", False))),
        )
        cv2.imwrite(
            os.path.join(component_dir, "initial_segments.png"),
            self._draw_debug_segments(smoothed, initial_segments, closed=bool(debug_payload.get("closed", False))),
        )
        cv2.imwrite(
            os.path.join(component_dir, "merged_segments.png"),
            self._draw_debug_segments(smoothed, merged_segments, closed=bool(debug_payload.get("closed", False))),
        )
        cv2.imwrite(
            os.path.join(component_dir, "raw_keypoints.png"),
            self._draw_debug_keypoints(smoothed, raw_keypoints, closed=bool(debug_payload.get("closed", False))),
        )
        cv2.imwrite(
            os.path.join(component_dir, "refined_keypoints.png"),
            self._draw_debug_keypoints(smoothed, refined_keypoints, closed=bool(debug_payload.get("closed", False))),
        )
        cv2.imwrite(
            os.path.join(component_dir, "final_keypoints.png"),
            self._draw_debug_keypoints(smoothed, keypoints, closed=bool(debug_payload.get("closed", False))),
        )

    # ------------------------------------------------------------------
    # 中心线准备：二值化、骨架化、骨架路径化、重采样
    # ------------------------------------------------------------------

    def _draw_debug_points(self, points: np.ndarray, closed: bool) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        if len(points) >= 2:
            poly = np.round(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [poly], closed, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        for x, y in points:
            cv2.circle(canvas, (int(round(x)), int(round(y))), 1, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        return canvas

    def _draw_debug_segments(self, points: np.ndarray, segments: List[Dict[str, Any]], closed: bool) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        colors = {
            "line": (60, 160, 60),
            "arc": (220, 140, 30),
            "spline": (180, 60, 180),
        }
        if len(points) >= 2:
            background = np.round(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [background], closed, (220, 220, 220), 1, lineType=cv2.LINE_AA)
        for seg in segments:
            start_idx = int(seg["start_idx"])
            end_idx = int(seg["end_idx"])
            seg_points = self._slice_polyline(points, start_idx, end_idx, closed=closed)
            if len(seg_points) < 2:
                continue
            poly = np.round(seg_points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                canvas,
                [poly],
                False,
                colors.get(str(seg.get("kind", "spline")), (0, 0, 0)),
                2,
                lineType=cv2.LINE_AA,
            )
        return canvas

    def _draw_debug_keypoints(self, points: np.ndarray, keypoints: List[int], closed: bool) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        if len(points) >= 2:
            poly = np.round(points).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [poly], closed, (210, 210, 210), 1, lineType=cv2.LINE_AA)
        for idx in keypoints:
            pt = points[int(np.clip(idx, 0, len(points) - 1))]
            cv2.circle(canvas, (int(round(pt[0])), int(round(pt[1]))), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        return canvas

    def _binarize_foreground_mask(self) -> np.ndarray:
        """把输入图转为中心线提取所需的前景 mask。"""

        gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if float(np.mean(mask > 0)) > 0.5:
            mask = 255 - mask
        return self._remove_speckles(mask)

    def _remove_speckles(self, mask: np.ndarray) -> np.ndarray:
        """删除面积过小的噪点连通域。"""

        if self.cfg.filter_speckle <= 0:
            return mask
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self.cfg.filter_speckle:
                out[labels == label] = 255
        return out

    def _zhang_suen_thinning(self, mask: np.ndarray) -> np.ndarray:
        """Zhang-Suen 细化，把粗线条压成单像素 skeleton。"""

        img = (mask > 0).astype(np.uint8)
        while True:
            marker1 = self._thinning_iteration(img, step=0)
            img[marker1] = 0
            marker2 = self._thinning_iteration(img, step=1)
            img[marker2] = 0
            if not np.any(marker1) and not np.any(marker2):
                break
        return (img * 255).astype(np.uint8)

    def _thinning_iteration(self, img: np.ndarray, step: int) -> np.ndarray:
        padded = np.pad(img, 1, mode="constant")
        p2 = padded[:-2, 1:-1]
        p3 = padded[:-2, 2:]
        p4 = padded[1:-1, 2:]
        p5 = padded[2:, 2:]
        p6 = padded[2:, 1:-1]
        p7 = padded[2:, :-2]
        p8 = padded[1:-1, :-2]
        p9 = padded[:-2, :-2]

        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
        transitions = sum(((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1)) for i in range(8))
        neighbor_count = sum(neighbors)

        base = (img == 1) & (transitions == 1) & (neighbor_count >= 2) & (neighbor_count <= 6)
        if step == 0:
            base &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
        else:
            base &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

        base[0, :] = False
        base[-1, :] = False
        base[:, 0] = False
        base[:, -1] = False
        return base

    def _prune_skeleton_spurs(self, skeleton: np.ndarray, max_length: int) -> np.ndarray:
        """裁掉 skeleton 上较短的毛刺分支。"""

        if max_length <= 0:
            return skeleton
        current = skeleton.copy()
        while True:
            changed = False
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(current, connectivity=8)
            for label in range(1, n_labels):
                if int(stats[label, cv2.CC_STAT_AREA]) < 3:
                    continue
                component_mask = (labels == label).astype(np.uint8) * 255
                pixels, adj, degrees = self._component_graph(component_mask)
                endpoints = [i for i, deg in enumerate(degrees) if deg == 1]
                for endpoint in endpoints:
                    trail = [endpoint]
                    prev = -1
                    curr = endpoint
                    while True:
                        nxt = [nb for nb in adj[curr] if nb != prev]
                        if not nxt:
                            break
                        nxt_idx = nxt[0]
                        trail.append(nxt_idx)
                        prev, curr = curr, nxt_idx
                        if degrees[curr] != 2 or len(trail) > max_length:
                            break
                    if degrees[curr] >= 3 and len(trail) <= max_length:
                        for idx in trail[:-1]:
                            y, x = pixels[idx]
                            current[y, x] = 0
                            changed = True
            if not changed:
                return current

    def _component_graph(
        self, component_mask: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[List[int]], List[int]]:
        ys, xs = np.where(component_mask > 0)
        pixels = sorted(zip(ys.tolist(), xs.tolist()))
        index = {coord: i for i, coord in enumerate(pixels)}
        adj: List[List[int]] = [[] for _ in pixels]
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        for i, (y, x) in enumerate(pixels):
            for dy, dx in directions:
                j = index.get((y + dy, x + dx))
                if j is not None:
                    adj[i].append(j)
        degrees = [len(neighbors) for neighbors in adj]
        return pixels, adj, degrees

    def _skeleton_component_to_path(self, component_mask: np.ndarray) -> Tuple[np.ndarray, bool]:
        """把一个 skeleton 连通域转成有序折线，并判断它是否闭合。"""

        pixels, adj, degrees = self._component_graph(component_mask)
        if not pixels:
            return np.empty((0, 2), dtype=np.float64), False

        endpoints = [i for i, deg in enumerate(degrees) if deg == 1]
        if len(endpoints) == 0:
            path_indices = self._walk_closed_component(pixels, adj)
            closed = True
        else:
            path_indices = self._longest_geodesic_path(adj, degrees)
            closed = False

        coords = np.array([[pixels[idx][1], pixels[idx][0]] for idx in path_indices], dtype=np.float64)
        return self._dedupe_consecutive(coords), closed

    def _skeleton_component_to_paths(self, component_mask: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
        """Split a skeleton component into all graph trails.

        A connected edge map can contain many branches.  Keeping only the
        longest path drops shorter slots and internal line features, so each
        endpoint/junction trail is treated as a separate open centerline path.
        Degree-2 loops are still emitted as one closed path.
        """

        pixels, adj, degrees = self._component_graph(component_mask)
        if not pixels:
            return []

        key_nodes = [i for i, deg in enumerate(degrees) if deg != 2]
        if not key_nodes:
            points, closed = self._skeleton_component_to_path(component_mask)
            return [(points, closed)] if len(points) else []

        visited_edges: set[Tuple[int, int]] = set()
        paths: List[Tuple[np.ndarray, bool]] = []

        def edge_key(a: int, b: int) -> Tuple[int, int]:
            return (a, b) if a <= b else (b, a)

        def edge_seen(a: int, b: int) -> bool:
            return edge_key(a, b) in visited_edges

        def mark_edge(a: int, b: int) -> None:
            visited_edges.add(edge_key(a, b))

        def to_points(indices: List[int]) -> np.ndarray:
            coords = np.array([[pixels[idx][1], pixels[idx][0]] for idx in indices], dtype=np.float64)
            return self._dedupe_consecutive(coords)

        for start in key_nodes:
            for neighbor in adj[start]:
                if edge_seen(start, neighbor):
                    continue
                trail = [start]
                prev = start
                curr = neighbor
                mark_edge(prev, curr)

                while True:
                    trail.append(curr)
                    if degrees[curr] != 2:
                        break
                    candidates = [nb for nb in adj[curr] if nb != prev]
                    if not candidates:
                        break
                    nxt = candidates[0]
                    if edge_seen(curr, nxt):
                        break
                    mark_edge(curr, nxt)
                    prev, curr = curr, nxt

                pts = to_points(trail)
                if len(pts) >= 2:
                    paths.append((pts, False))

        for start in range(len(pixels)):
            for neighbor in adj[start]:
                if edge_seen(start, neighbor):
                    continue
                trail = [start]
                prev = start
                curr = neighbor
                mark_edge(prev, curr)
                while True:
                    trail.append(curr)
                    candidates = [nb for nb in adj[curr] if nb != prev and not edge_seen(curr, nb)]
                    if not candidates:
                        break
                    nxt = candidates[0]
                    mark_edge(curr, nxt)
                    prev, curr = curr, nxt
                    if curr == start:
                        break
                pts = to_points(trail)
                if len(pts) >= 2:
                    paths.append((pts, True))

        return paths

    def _walk_closed_component(self, pixels: List[Tuple[int, int]], adj: List[List[int]]) -> List[int]:
        start = min(range(len(pixels)), key=lambda i: pixels[i])
        path = [start]
        prev = -1
        curr = start
        visited_edges = set()
        while True:
            candidates = [nb for nb in adj[curr] if (curr, nb) not in visited_edges]
            if not candidates:
                break
            if prev == -1:
                nxt = min(candidates, key=lambda i: pixels[i])
            else:
                curr_pt = np.array(pixels[curr], dtype=np.float64)
                prev_pt = np.array(pixels[prev], dtype=np.float64)
                tangent = curr_pt - prev_pt

                def score(idx: int) -> Tuple[float, float]:
                    vec = np.array(pixels[idx], dtype=np.float64) - curr_pt
                    norm = float(np.linalg.norm(vec))
                    if norm < 1e-9:
                        return (-1e9, 0.0)
                    align = float(np.dot(tangent, vec) / (np.linalg.norm(tangent) * norm + 1e-9))
                    return (align, -norm)

                nxt = max(candidates, key=score)
            visited_edges.add((curr, nxt))
            visited_edges.add((nxt, curr))
            if nxt == start:
                break
            path.append(nxt)
            prev, curr = curr, nxt
            if len(path) > len(pixels) * 2:
                break
        return path

    def _longest_geodesic_path(self, adj: List[List[int]], degrees: List[int]) -> List[int]:
        endpoints = [i for i, deg in enumerate(degrees) if deg == 1]
        candidate_starts = endpoints if len(endpoints) >= 2 else [0]

        best_dist = -1
        best_parent: List[int] = []
        best_end = 0
        best_start = 0
        for start in candidate_starts:
            dist, parent = self._bfs_distances(adj, start)
            if endpoints:
                farthest = max(endpoints, key=lambda i: dist[i])
            else:
                farthest = int(np.argmax(dist))
            if dist[farthest] > best_dist:
                best_dist = dist[farthest]
                best_parent = parent
                best_end = farthest
                best_start = start

        path = []
        curr = best_end
        while curr != -1:
            path.append(curr)
            if curr == best_start:
                break
            curr = best_parent[curr]
        path.reverse()
        return path

    def _bfs_distances(self, adj: List[List[int]], start: int) -> Tuple[np.ndarray, List[int]]:
        dist = np.full(len(adj), -1, dtype=np.int32)
        parent = [-1 for _ in adj]
        queue: deque[int] = deque([start])
        dist[start] = 0
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] != -1:
                    continue
                dist[nb] = dist[node] + 1
                parent[nb] = node
                queue.append(nb)
        return dist, parent

    def _resample_polyline(self, points: np.ndarray, closed: bool, step: float) -> np.ndarray:
        """按弧长均匀重采样折线。

        后续几乎所有语义分段计算都在这个“采样间距稳定”的折线上进行。
        """

        pts = self._dedupe_consecutive(points)
        if len(pts) < 2:
            return pts
        work = np.vstack([pts, pts[0]]) if closed else pts
        seg = np.linalg.norm(np.diff(work, axis=0), axis=1)
        total = float(np.sum(seg))
        if total < 1e-9:
            return pts
        cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        min_count = 3 if closed else 2
        count = max(min_count, int(math.ceil(total / max(step, 1e-6))))
        targets = np.linspace(0.0, total, count, endpoint=not closed)
        out = []
        idx = 0
        for t in targets:
            while idx + 1 < len(cumulative) and cumulative[idx + 1] < t:
                idx += 1
            if idx >= len(work) - 1:
                out.append(work[-1])
                continue
            span = cumulative[idx + 1] - cumulative[idx]
            ratio = 0.0 if span < 1e-9 else (t - cumulative[idx]) / span
            out.append(work[idx] * (1.0 - ratio) + work[idx + 1] * ratio)
        return np.array(out, dtype=np.float64)

    def _gaussian_smooth_polyline(self, points: np.ndarray, closed: bool) -> np.ndarray:
        """对重采样折线做轻度平滑，减少像素锯齿对曲率/角点判断的影响。"""

        sigma = float(self.cfg.gaussian_sigma)
        if sigma <= 1e-6 or len(points) < 3:
            return points.copy()
        mode = "wrap" if closed else "nearest"
        smoothed = np.column_stack(
            [
                gaussian_filter1d(points[:, 0], sigma=sigma, mode=mode),
                gaussian_filter1d(points[:, 1], sigma=sigma, mode=mode),
            ]
        )
        if not closed:
            smoothed[0] = points[0]
            smoothed[-1] = points[-1]
        return smoothed

    # ------------------------------------------------------------------
    # 语义分段主流程：关键点、边界场、DP、合并
    # ------------------------------------------------------------------

    def _fit_centerline_path(self, points: np.ndarray, closed: bool) -> CenterlinePathFit:
        """中心线语义参数化的核心流程。

        输入是一条已经重采样的折线。该函数依次完成：
        1. 平滑折线；
        2. 检测多尺度关键点；
        3. 计算“哪里像分段边界”的概率场；
        4. 先做较保守的过分段；
        5. 检测全局直线；
        6. 用动态规划选出更优的 line / arc / spline 组合；
        7. 把可合并的相邻段继续压缩；
        8. 对闭合恒曲率环尝试压成整圆弧；
        9. 构造 SVG path 与 semantic_debug。
        """

        if len(points) < 2:
            return CenterlinePathFit(
                d="",
                errors=np.zeros(0, dtype=np.float64),
                effective_params=0,
                line_segments=0,
                arc_segments=0,
                curve_segments=0,
                bspline_control_points=0,
            )

        smoothed = self._gaussian_smooth_polyline(points, closed=closed)
        self._active_spline_cache = {}
        raw_keypoints = self._detect_multiscale_keypoints(smoothed, closed=closed)
        refined_keypoints = self._refine_keypoints(smoothed, raw_keypoints, closed=closed)
        boundary_field = self._compute_boundary_probability_field(smoothed, closed=closed)
        initial_segments = self._snap_segments_to_reference_points(
            points, self._oversegment_semantic_path(smoothed, refined_keypoints, closed=closed), closed=closed
        )
        global_lines = self._detect_global_lines(initial_segments)
        dp_segments = self._snap_segments_to_reference_points(
            points,
            self._dynamic_programming_segments(
                smoothed,
                refined_keypoints,
                initial_segments,
                global_lines,
                closed=closed,
                boundary_field=boundary_field,
            ),
            closed=closed,
        )
        reassigned_segments = self._snap_segments_to_reference_points(
            points, self._reassign_segments_to_global_lines(dp_segments, global_lines), closed=closed
        )
        merged_segments = self._snap_segments_to_reference_points(
            points,
            self._merge_semantic_segments(
                smoothed,
                reassigned_segments,
                closed=closed,
                global_lines=global_lines,
                protected_boundaries=set(int(v) for v in refined_keypoints),
            ),
            closed=closed,
        )
        collapsed_segments = merged_segments
        loop_collapsed = False
        if closed:
            collapsed_try = self._maybe_collapse_constant_curvature_closed_loop(
                smoothed, merged_segments, closed=closed, global_lines=global_lines
            )
            if len(collapsed_try) < len(merged_segments):
                collapsed_segments = self._snap_segments_to_reference_points(
                    points, collapsed_try, closed=closed
                )
                loop_collapsed = True
        fit = self._build_semantic_path_fit(smoothed, collapsed_segments, closed=closed)
        fit.semantic_debug = {
            "closed": bool(closed),
            "resampled_points": np.round(points, 4).tolist(),
            "smoothed_points": np.round(smoothed, 4).tolist(),
            "raw_keypoints": raw_keypoints,
            "refined_keypoints": refined_keypoints,
            "initial_segments": [self._segment_debug_record(seg) for seg in initial_segments],
            "global_lines": [self._global_line_summary(line) for line in global_lines],
            "dp_segments": [self._segment_debug_record(seg) for seg in dp_segments],
            "merged_segments": [self._segment_debug_record(seg) for seg in merged_segments],
            "collapsed_full_loop_arc": bool(loop_collapsed),
            "final_segments": [self._segment_debug_record(seg) for seg in collapsed_segments],
            "keypoints": refined_keypoints,
            "primitives": fit.primitive_summary or [],
            "boundary_field_weight": float(getattr(self.cfg, "dp_boundary_field_weight", 0.0)),
            "boundary_field_mean": float(np.mean(boundary_field)) if len(boundary_field) else 0.0,
            "boundary_field_max": float(np.max(boundary_field)) if len(boundary_field) else 0.0,
        }
        return fit

    def _snap_segments_to_reference_points(
        self, reference_points: np.ndarray, segments: List[SemanticSegment], closed: bool
    ) -> List[SemanticSegment]:
        snapped: List[SemanticSegment] = []
        n = len(reference_points)
        for seg in segments:
            if len(seg.points) >= 1:
                pts = np.array(seg.points, copy=True)
                pts[0] = reference_points[int(seg.start_idx) % n]
                pts[-1] = reference_points[int(seg.end_idx) % n]
            else:
                pts = seg.points
            snapped_seg = SemanticSegment(
                kind=seg.kind,
                start_idx=seg.start_idx,
                end_idx=seg.end_idx,
                points=pts,
                errors=seg.errors,
                effective_params=seg.effective_params,
                line_segments=seg.line_segments,
                arc_segments=seg.arc_segments,
                curve_segments=seg.curve_segments,
                bspline_control_points=seg.bspline_control_points,
                line_start=reference_points[int(seg.start_idx) % n] if seg.kind == "line" else seg.line_start,
                line_end=reference_points[int(seg.end_idx) % n] if seg.kind == "line" else seg.line_end,
                arc_fit=seg.arc_fit,
                bspline_fit=seg.bspline_fit,
                global_line_id=seg.global_line_id,
            )
            snapped.append(snapped_seg)
        return snapped

    def _line_arc_mean_residual_from_window(self, wpts: np.ndarray) -> float:
        """line vs arc 平均误差差（窗口已裁好）；供关键点 nudge 与批量窗口复用。"""
        m = len(wpts)
        if m < 5:
            return 0.0
        line_seg = self._fit_line_segment(wpts, 0, m - 1)
        el = float(np.mean(line_seg.errors))
        arc_seg = self._fit_arc_segment(wpts, 0, m - 1)
        if arc_seg is None:
            return max(0.0, el * 0.15)
        ea = float(np.mean(arc_seg.errors))
        return max(0.0, el - ea)

    def _keypoint_line_arc_mean_residual(self, points: np.ndarray, center: int, half: int, closed: bool) -> float:
        wpts = self._local_window_points(points, center, half, closed)
        return self._line_arc_mean_residual_from_window(wpts)

    @staticmethod
    def _robust_unit_scores(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        lo = float(np.percentile(values, 8.0))
        hi = float(np.percentile(values, 92.0))
        if hi <= lo + 1e-9:
            return np.zeros_like(values, dtype=np.float64)
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    def _local_maxima_mask_cyclic(self, score: np.ndarray, radius: int, closed: bool) -> np.ndarray:
        n = int(score.shape[0])
        if n == 0:
            return np.zeros(0, dtype=bool)
        out = np.zeros(n, dtype=bool)
        r = max(1, int(radius))
        for i in range(n):
            if not closed and (i < r or i >= n - r):
                continue
            v = float(score[i])
            if v <= 1e-12:
                continue
            is_max = True
            for d in range(-r, r + 1):
                if d == 0:
                    continue
                j = (i + d) % n if closed else i + d
                if not closed and (j < 0 or j >= n):
                    continue
                if float(score[j]) > v + 1e-9:
                    is_max = False
                    break
            out[i] = is_max
        return out

    def _compute_boundary_probability_field(self, points: np.ndarray, closed: bool) -> np.ndarray:
        """为每个折线点估计“这里应该是边界”的软概率。

        DP 不只依赖硬关键点，也会参考这个场：
        如果一个候选段跨过了明显的内部边界峰，代价会升高。
        """

        """每顶点边界强度 B(i)∈[0,1]：多尺度转角 + 曲率 Lap + 轻量 line/arc 歧义（与关键点信号同源，供 DP 软惩罚）。"""
        n = len(points)
        if n < 3:
            return np.zeros(n, dtype=np.float64)
        base = max(2, int(self.cfg.semantic_window_size // 4))
        scales = sorted(
            {base, max(base + 1, int(self.cfg.semantic_window_size // 2)), max(base + 2, int(self.cfg.semantic_window_size))}
        )
        deviation_scores = np.zeros(n, dtype=np.float64)
        curvature_scores = np.zeros(n, dtype=np.float64)
        for scale in scales:
            if closed:
                i = np.arange(n, dtype=np.int64)
                prev_idx = (i - scale) % n
                next_idx = (i + scale) % n
            else:
                if n <= 2 * scale:
                    continue
                i = np.arange(scale, n - scale, dtype=np.int64)
                prev_idx = i - scale
                next_idx = i + scale
            pa = points[prev_idx]
            pb = points[i]
            pc = points[next_idx]
            angle = self._batch_unsigned_turn_angle_deg(pa, pb, pc)
            deviation = 180.0 - angle
            deviation_scores[i] = np.maximum(deviation_scores[i], deviation)
            chord = np.linalg.norm(pc - pa, axis=1).clip(min=1e-6)
            lap = np.linalg.norm(pa - 2.0 * pb + pc, axis=1) / chord
            curvature_scores[i] = np.maximum(curvature_scores[i], lap)
        dev_u = self._robust_unit_scores(deviation_scores)
        cur_u = self._robust_unit_scores(curvature_scores)
        half = base
        res = np.zeros(n, dtype=np.float64)
        if n >= 2 * half + 1:
            if closed:
                ca = np.arange(n, dtype=np.int64)
                offsets = np.arange(-half, half + 1, dtype=np.int64)
                idx = (ca[:, None] + offsets[None, :]) % n
                wstack = points[idx]
                for k in range(n):
                    res[k] = self._line_arc_mean_residual_from_window(wstack[k])
            else:
                ca = np.arange(half, n - half, dtype=np.int64)
                offsets = np.arange(-half, half + 1, dtype=np.int64)
                idx = ca[:, None] + offsets[None, :]
                wstack = points[idx]
                for k in range(len(ca)):
                    res[int(ca[k])] = self._line_arc_mean_residual_from_window(wstack[k])
        res_u = self._robust_unit_scores(res)
        b = np.clip(0.52 * dev_u + 0.28 * cur_u + 0.20 * res_u, 0.0, 1.0)
        mode = "wrap" if closed else "nearest"
        if n >= 5:
            b = gaussian_filter1d(b, sigma=0.75, mode=mode)
        return b

    def _dp_interior_boundary_peak_penalty(self, boundary_field: np.ndarray, start_idx: int, end_idx: int) -> float:
        """对 (start_idx, end_idx) 段内、B 的局部峰求和并加权：鼓励在真角处切段，而非仅靠硬边界列表。"""
        w = float(getattr(self.cfg, "dp_boundary_field_weight", 0.0))
        if w <= 0.0 or boundary_field is None or len(boundary_field) < 3:
            return 0.0
        b = boundary_field
        n = len(b)
        if end_idx <= start_idx + 1:
            return 0.0
        thr = max(0.12, 0.25 * float(np.max(b)))
        s = 0.0
        for k in range(start_idx + 1, end_idx):
            bk = float(b[k])
            if bk < thr:
                continue
            b_prev = float(b[k - 1])
            b_next = float(b[k + 1]) if k + 1 < n else b_prev
            if bk + 1e-9 < b_prev or bk + 1e-9 < b_next:
                continue
            s += bk
        return w * s

    def _detect_multiscale_keypoints(self, points: np.ndarray, closed: bool) -> List[int]:
        """从多个尺度检测关键点。

        关键点既看几何转角，也可融合局部 line-vs-arc 模型残差，
        用于找出角点、曲率突变点和 primitive 交界。
        """

        n = len(points)
        if n < 3:
            if closed:
                return [0]
            return [0, max(0, n - 1)]

        base = max(2, int(self.cfg.semantic_window_size // 4))
        scales = sorted(
            {base, max(base + 1, int(self.cfg.semantic_window_size // 2)), max(base + 2, int(self.cfg.semantic_window_size))}
        )
        deviation_scores = np.zeros(n, dtype=np.float64)
        curvature_scores = np.zeros(n, dtype=np.float64)
        raw_mask = np.zeros(n, dtype=bool)
        thr_ang = float(self.cfg.keypoint_angle_threshold_deg)

        for scale in scales:
            if closed:
                i = np.arange(n, dtype=np.int64)
                prev_idx = (i - scale) % n
                next_idx = (i + scale) % n
            else:
                if n <= 2 * scale:
                    continue
                i = np.arange(scale, n - scale, dtype=np.int64)
                prev_idx = i - scale
                next_idx = i + scale
            pa = points[prev_idx]
            pb = points[i]
            pc = points[next_idx]
            angle = self._batch_unsigned_turn_angle_deg(pa, pb, pc)
            deviation = 180.0 - angle
            deviation_scores[i] = np.maximum(deviation_scores[i], deviation)
            chord = np.linalg.norm(pc - pa, axis=1).clip(min=1e-6)
            lap = np.linalg.norm(pa - 2.0 * pb + pc, axis=1) / chord
            curvature_scores[i] = np.maximum(curvature_scores[i], lap)
            raw_mask[i] |= deviation >= thr_ang

        if np.any(curvature_scores > 0):
            curvature_limit = float(np.percentile(curvature_scores[curvature_scores > 0], 80))
            raw_mask |= curvature_scores >= curvature_limit

        if not closed:
            raw_mask[0] = True
            raw_mask[-1] = True

        geo_candidates = self._cluster_keypoint_mask(raw_mask, deviation_scores, closed=closed)
        if not bool(getattr(self.cfg, "keypoint_use_model_guided", True)):
            return geo_candidates if geo_candidates else ([0] if closed else [0, n - 1])

        fracs = getattr(self.cfg, "keypoint_scale_half_fractions", (0.014, 0.028, 0.048))
        if isinstance(fracs, (list, tuple)):
            half_widths = sorted(
                {max(2, min(max(3, n // 4), int(float(f) * n))) for f in fracs if float(f) > 0.0}
            )
        else:
            half_widths = [max(2, min(max(3, n // 4), int(0.025 * n)))]

        w_ang = float(getattr(self.cfg, "keypoint_angle_guided_weight", 0.55))
        w_mod = float(getattr(self.cfg, "keypoint_model_guided_weight", 0.45))
        s = w_ang + w_mod
        if s > 1e-9:
            w_ang, w_mod = w_ang / s, w_mod / s

        def _fused_for_half(half: int) -> np.ndarray:
            theta_s = np.zeros(n, dtype=np.float64)
            res_s = np.zeros(n, dtype=np.float64)
            if closed:
                idx_range = range(n)
            else:
                idx_range = range(half, n - half)
            for i in idx_range:
                prev_idx = (i - half) % n
                next_idx = (i + half) % n
                ang = self._unsigned_turn_angle_deg(points[prev_idx], points[i], points[next_idx])
                theta_s[i] = max(0.0, 180.0 - ang)
                res_s[i] = self._keypoint_line_arc_mean_residual(points, i, half, closed=closed)
            t_u = self._robust_unit_scores(theta_s)
            r_u = self._robust_unit_scores(res_s)
            return w_ang * t_u + w_mod * r_u

        if not bool(getattr(self.cfg, "keypoint_model_multiscale_votes", False)):
            half_nudge = half_widths[len(half_widths) // 2]
            rad = max(1, int(self.cfg.keypoint_refine_radius))
            half = int(half_nudge)
            cand_idx: set[int] = set()
            for kp in geo_candidates:
                kk = int(kp) % n
                for d in range(-rad, rad + 1):
                    j = (kk + d) % n if closed else int(np.clip(kk + d, 0, n - 1))
                    if not closed and (j < half or j >= n - half):
                        continue
                    cand_idx.add(int(j))
            cand_list = sorted(cand_idx)
            if cand_list:
                ca = np.asarray(cand_list, dtype=np.int64)
                if closed:
                    pa = points[(ca - half) % n]
                    pb = points[ca]
                    pc = points[(ca + half) % n]
                    ang = self._batch_unsigned_turn_angle_deg(pa, pb, pc)
                    theta_list = np.maximum(0.0, 180.0 - ang).tolist()
                    offsets = np.arange(-half, half + 1, dtype=np.int64)
                    idx = (ca[:, None] + offsets[None, :]) % n
                    wstack = points[idx]
                    res_list = [self._line_arc_mean_residual_from_window(wstack[k]) for k in range(wstack.shape[0])]
                else:
                    pa = points[ca - half]
                    pb = points[ca]
                    pc = points[ca + half]
                    ang = self._batch_unsigned_turn_angle_deg(pa, pb, pc)
                    theta_list = np.maximum(0.0, 180.0 - ang).tolist()
                    res_list = [
                        self._keypoint_line_arc_mean_residual(points, int(j), half, closed=closed) for j in cand_list
                    ]
            else:
                theta_list = []
                res_list = []
            fused_map: Dict[int, float] = {}
            if cand_list:
                t_arr = np.asarray(theta_list, dtype=np.float64)
                r_arr = np.asarray(res_list, dtype=np.float64)
                t_u = self._robust_unit_scores(t_arr)
                r_u = self._robust_unit_scores(r_arr)
                for idx, j in enumerate(cand_list):
                    fused_map[int(j)] = float(w_ang * t_u[idx] + w_mod * r_u[idx])
            nudged: List[int] = []
            for kp in geo_candidates:
                kk = int(kp) % n
                best_i = kk
                best_v = float(fused_map.get(kk, 0.0))
                for d in range(-rad, rad + 1):
                    j = (kk + d) % n if closed else int(np.clip(kk + d, 0, n - 1))
                    v = float(fused_map.get(int(j), 0.0))
                    if v > best_v:
                        best_v = v
                        best_i = j
                nudged.append(best_i)
            combined = sorted(set(int(v) % n for v in nudged))
            if not closed:
                combined = sorted(set([0, n - 1] + combined))
            out = self._dedupe_sorted_indices(combined, n, closed=closed)
            return out if out else ([0] if closed else [0, n - 1])

        min_votes = max(1, int(getattr(self.cfg, "keypoint_multiscale_min_votes", 3)))
        nms_frac = float(getattr(self.cfg, "keypoint_nms_half_fraction", 0.35))
        votes = np.zeros(n, dtype=np.int32)
        for half in half_widths:
            fused = _fused_for_half(half)
            thr = float(np.percentile(fused, 90.0)) if n >= 16 else 0.0
            thr = max(thr, float(np.max(fused)) * 0.28)
            nms_r = max(2, min(half, int(max(2.0, nms_frac * float(half)))))
            peaks = self._local_maxima_mask_cyclic(fused, nms_r, closed=closed) & (fused >= thr)
            votes += peaks.astype(np.int32)

        n_scales = max(1, len(half_widths))
        model_peaks = [int(i) for i in np.flatnonzero(votes >= min_votes)]
        bridge = max(3, int(getattr(self.cfg, "keypoint_model_peak_bridge", 9)))

        def _near_geo(pk: int) -> bool:
            for g in geo_candidates:
                if closed:
                    d = min((pk - g) % n, (g - pk) % n)
                else:
                    d = abs(pk - g)
                if d <= bridge:
                    return True
            return False

        strong_vote = min(n_scales, max(min_votes + 1, n_scales))
        model_peaks = [p for p in model_peaks if _near_geo(p) or int(votes[p]) >= strong_vote]
        combined = sorted(set(int(v) % n for v in (geo_candidates + model_peaks)))
        if not closed:
            combined = sorted(set([0, n - 1] + combined))
        out = self._dedupe_sorted_indices(combined, n, closed=closed)
        return out if out else ([0] if closed else [0, n - 1])

    def _refine_keypoints(self, points: np.ndarray, keypoints: List[int], closed: bool) -> List[int]:
        """把粗关键点在局部窗口内精修到更合适的位置。"""

        n = len(points)
        if not keypoints:
            return [0] if closed else [0, max(0, n - 1)]

        refined = sorted(set(int(k) % n for k in keypoints))
        if not closed:
            refined = sorted(set([0, n - 1] + refined))

        iterations = range(len(refined)) if closed else range(1, max(1, len(refined) - 1))
        out = refined[:]
        search_radius = max(1, int(self.cfg.keypoint_refine_radius))
        for pos in iterations:
            prev_idx = out[pos - 1] if pos > 0 else (out[-1] if closed else None)
            next_idx = out[(pos + 1) % len(out)] if pos + 1 < len(out) else (out[0] if closed else None)
            if prev_idx is None or next_idx is None:
                continue
            current = out[pos]
            best_idx = current
            best_cost = float("inf")
            low = current - search_radius
            high = current + search_radius
            for candidate in range(low, high + 1):
                cand = candidate % n if closed else int(np.clip(candidate, 0, n - 1))
                if not self._refined_keypoint_candidate_valid(prev_idx, cand, next_idx, n, closed):
                    continue
                cost = self._keypoint_local_cost(points, prev_idx, cand, next_idx, closed=closed)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = cand
            out[pos] = best_idx
        out = self._dedupe_sorted_indices(out, n, closed=closed)
        out = self._prune_almost_collinear_keypoints(points, out, closed=closed)
        mg = int(getattr(self.cfg, "keypoint_merge_max_polyline_gap", 4))
        if mg >= 2 and len(out) >= 3:
            out = self._merge_proximate_keypoints_by_polyline_sharpness(points, out, closed=closed, max_gap=mg)
        out = self._dedupe_sorted_indices(out, n, closed=closed)
        return out if out else ([0] if closed else [0, n - 1])

    def _prune_almost_collinear_keypoints(
        self, points: np.ndarray, keypoints: List[int], closed: bool, min_keep_closed: int = 3, min_keep_open: int = 2
    ) -> List[int]:
        """去掉相邻关键点中近似共线的中间点，减轻受保护边界过密 → DP 过切与边界漂移。"""
        n = len(points)
        if n < 4 or len(keypoints) < 2:
            return keypoints
        kp = [int(k) % n for k in keypoints]
        mk = min_keep_closed if closed else min_keep_open
        if len(kp) <= mk:
            return kp
        thresh_deg = min(173.5, max(166.0, 180.0 - 0.42 * float(self.cfg.line_merge_angle_deg)))
        while len(kp) > mk:
            removed = False
            L = len(kp)
            for i in range(L):
                if not closed and (i == 0 or i == L - 1):
                    continue
                pi = (i - 1) % L
                ni = (i + 1) % L
                ang = self._unsigned_turn_angle_deg(points[kp[pi]], points[kp[i]], points[kp[ni]])
                if ang >= thresh_deg:
                    del kp[i]
                    removed = True
                    break
            if not removed:
                break
        return kp

    def _vertex_turn_sharpness(self, points: np.ndarray, idx: int, closed: bool) -> float:
        """折线顶点处转角偏离 180° 的程度，越大表示越像真角点。"""
        n = len(points)
        i = int(idx) % n
        prev_i = (i - 1) % n if closed else i - 1
        next_i = (i + 1) % n if closed else i + 1
        if not closed and (prev_i < 0 or next_i >= n):
            return 0.0
        ang = self._unsigned_turn_angle_deg(points[prev_i], points[i], points[next_i])
        return max(0.0, 180.0 - ang)

    def _merge_proximate_keypoints_by_polyline_sharpness(
        self, points: np.ndarray, keypoints: List[int], closed: bool, max_gap: int
    ) -> List[int]:
        """合并沿折线过近的一对关键点，去掉较钝者（缓解平滑后在角点旁多检出一个关键点）。"""
        n = len(points)
        if n < 4 or max_gap < 2 or len(keypoints) < 2:
            return keypoints
        kp = sorted({int(k) % n for k in keypoints})
        if not closed:
            kp = sorted(set([0, n - 1] + kp))
        min_keep = 3 if closed else 2

        def sharp(i: int) -> float:
            return self._vertex_turn_sharpness(points, i, closed=closed)

        if closed:
            while len(kp) > min_keep:
                L = len(kp)
                candidates: List[Tuple[int, float, int, int]] = []
                for i in range(L):
                    a, b = kp[i], kp[(i + 1) % L]
                    gap = (b - a) % n
                    if gap == 0:
                        gap = n
                    if not (1 <= gap <= max_gap):
                        continue
                    sa, sb = sharp(a), sharp(b)
                    drop = int(a if sa <= sb else b)
                    dull = float(min(sa, sb))
                    candidates.append((gap, dull, i, drop))
                if not candidates:
                    break
                candidates.sort(key=lambda t: (t[0], t[1], t[2]))
                drop_idx = int(candidates[0][3])
                kp = sorted({int(x) for x in kp if int(x) != drop_idx})
            return kp

        changed = True
        while changed and len(kp) > min_keep:
            changed = False
            for i in range(len(kp) - 1):
                a, b = kp[i], kp[i + 1]
                gap = int(b) - int(a)
                if not (1 <= gap <= max_gap):
                    continue
                ra = a in (0, n - 1)
                rb = b in (0, n - 1)
                if ra and rb:
                    continue
                if ra:
                    dead = int(b)
                elif rb:
                    dead = int(a)
                else:
                    dead = int(a if sharp(a) <= sharp(b) else b)
                kp = [x for x in kp if int(x) != dead]
                changed = True
                break
        return kp

    def _oversegment_semantic_path(self, points: np.ndarray, keypoints: List[int], closed: bool) -> List[SemanticSegment]:
        """先生成一个偏保守的初始分段。

        这一步宁可稍微切碎一些，后续 DP 与合并阶段再负责删掉多余边界。
        """

        n = len(points)
        if n < 3:
            return [self._fit_segment_model(points, 0, max(0, n - 1), global_lines=None)]

        window = int(max(5, self.cfg.semantic_window_size))
        if window % 2 == 0:
            window += 1
        window = min(window, n if n % 2 == 1 else max(3, n - 1))
        half = max(1, window // 2)

        labels: List[str] = []
        signatures: List[Tuple[float, ...]] = []
        for center in range(n):
            window_points = self._local_window_points(points, center, half=half, closed=closed)
            line_seg = self._fit_line_segment(window_points, 0, len(window_points) - 1)
            arc_seg = self._fit_arc_segment(window_points, 0, len(window_points) - 1)
            line_ok = self._line_fit_is_acceptable(window_points, line_seg.errors)
            arc_ok = arc_seg is not None and self._arc_fit_is_acceptable(window_points, arc_seg.arc_fit)
            if line_ok and (not arc_ok or float(np.max(line_seg.errors)) <= float(np.max(arc_seg.errors)) * 1.1):
                labels.append("line")
                signatures.append((self._segment_direction_deg(window_points),))
            elif arc_ok and arc_seg is not None and arc_seg.arc_fit is not None:
                labels.append("arc")
                signatures.append(
                    (
                        float(arc_seg.arc_fit.radius),
                        float(arc_seg.arc_fit.center[0]),
                        float(arc_seg.arc_fit.center[1]),
                        float(np.sign(arc_seg.arc_fit.sweep_angle)),
                    )
                )
            else:
                labels.append("spline")
                signatures.append(tuple())

        # 沿轮廓对 line/arc/spline 标签做短环上多数票，抑制大语义窗下 line↔arc 抖动导致的伪断点
        # （否则 DP 块内会塞入过多 initial 边界，例如五边形直边被切成多段）。
        k_smooth = max(1, min(6, max(1, half * 2 // 3)))
        if n > 2 * k_smooth + 1:
            smooth_lbl: List[str] = []
            for i in range(n):
                cnt = {"line": 0, "arc": 0, "spline": 0}
                for d in range(-k_smooth, k_smooth + 1):
                    j = (i + d) % n if closed else max(0, min(n - 1, i + d))
                    cnt[labels[j]] += 1
                smooth_lbl.append(max(cnt, key=lambda kk: cnt[kk]))
        else:
            smooth_lbl = labels

        boundaries = set(int(k) % n for k in keypoints)
        boundaries.add(0)
        if not closed:
            boundaries.add(n - 1)
        for i in range(1, n):
            if self._semantic_break(smooth_lbl[i - 1], smooth_lbl[i], signatures[i - 1], signatures[i]):
                boundaries.add(i)

        ordered = sorted(boundaries)
        if not closed:
            ranges = [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1) if ordered[i + 1] > ordered[i]]
        else:
            if len(ordered) == 1:
                ranges = [(0, n - 1)]
            else:
                ranges = [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]

        segments = [self._fit_segment_model(self._slice_polyline(points, s, e, closed=closed), s, e, global_lines=None) for s, e in ranges]
        return self._coalesce_tiny_segments(points, segments, closed=closed)

    def _merge_boundary_candidates_near_keypoints(
        self,
        points: np.ndarray,
        boundaries: List[int],
        keypoints: set,
        closed: bool,
    ) -> List[int]:
        """只删除贴近强关键点的语义伪边界；不压缩 keypoint-keypoint 或 semantic-semantic。

        直接聚类所有候选会让初始 oversegment 变长，导致线段被样条吞掉；这里保持初始语义细节，
        只去掉真角旁由 line/arc 标签抖动产生的非关键点边界。
        """
        n = len(points)
        gap = int(getattr(self.cfg, "boundary_candidate_merge_max_polyline_gap", 0))
        if n < 4 or gap <= 0 or len(boundaries) <= 2:
            return sorted(set(int(v) % n for v in boundaries))

        vals = sorted(set(int(v) % n for v in boundaries))
        if not closed:
            vals = sorted(set([0, n - 1] + vals))

        kps = sorted({int(k) % n for k in keypoints})
        if not kps:
            return vals

        def near_keypoint(idx: int) -> bool:
            for kp in kps:
                if closed:
                    d = min((idx - kp) % n, (kp - idx) % n)
                else:
                    d = abs(idx - kp)
                if 0 < d <= gap:
                    return True
            return False

        out: List[int] = []
        for idx in vals:
            if int(idx) in keypoints:
                out.append(int(idx))
                continue
            if not closed and idx in (0, n - 1):
                out.append(int(idx))
                continue
            if near_keypoint(int(idx)) and self._vertex_turn_sharpness(points, int(idx), closed=closed) < float(
                getattr(self.cfg, "boundary_candidate_min_sharpness_to_keep", 18.0)
            ):
                continue
            out.append(int(idx))
        min_keep = 3 if closed else 2
        if len(out) < min_keep:
            return vals
        return sorted(set(out))

    def _detect_global_lines(self, initial_segments: List[SemanticSegment]) -> List[GlobalLine]:
        """从初始线段中检测重复出现的全局直线方向/位置。"""

        line_segments = [seg for seg in initial_segments if seg.kind == "line" and len(seg.points) >= 3]
        if len(line_segments) < 2 or DBSCAN is None:
            return []

        diag = math.hypot(self.width, self.height)
        features = []
        descriptors = []
        for seg in line_segments:
            descriptor = self._fit_infinite_line(seg.points)
            if descriptor is None:
                continue
            theta, distance, normal, direction, anchor, errors = descriptor
            features.append([math.cos(2.0 * theta), math.sin(2.0 * theta), distance / max(1.0, diag)])
            descriptors.append((seg, theta, distance, normal, direction, anchor, errors))

        if len(features) < 2:
            return []

        labels = DBSCAN(eps=float(self.cfg.line_cluster_eps), min_samples=int(self.cfg.line_cluster_min_samples)).fit_predict(np.asarray(features, dtype=np.float64))
        global_lines: List[GlobalLine] = []
        next_id = 0
        for cluster_id in sorted(set(int(v) for v in labels.tolist()) - {-1}):
            cluster_points = [descriptors[i][0].points for i, lbl in enumerate(labels) if int(lbl) == cluster_id]
            if not cluster_points:
                continue
            joined = np.vstack(cluster_points)
            descriptor = self._fit_infinite_line(joined)
            if descriptor is None:
                continue
            theta, distance, normal, direction, anchor, _errors = descriptor
            global_lines.append(
                GlobalLine(
                    line_id=next_id,
                    theta=theta,
                    distance=distance,
                    normal=normal,
                    direction=direction,
                    anchor=anchor,
                    support_segments=int(sum(1 for lbl in labels if int(lbl) == cluster_id)),
                )
            )
            next_id += 1
        return global_lines

    def _dynamic_programming_segments(
        self,
        points: np.ndarray,
        refined_keypoints: List[int],
        initial_segments: List[SemanticSegment],
        global_lines: List[GlobalLine],
        closed: bool,
        boundary_field: Optional[np.ndarray] = None,
    ) -> List[SemanticSegment]:
        """语义分段 DP 总入口。

        闭合轮廓和开放折线的边界处理方式不同，因此这里先分流，
        最终都落到 _dynamic_programming_segments_on_intervals。
        """

        if closed:
            return self._dynamic_programming_segments_closed(
                points, refined_keypoints, initial_segments, global_lines, boundary_field=boundary_field
            )
        return self._dynamic_programming_segments_open(
            points, refined_keypoints, initial_segments, global_lines, boundary_field=boundary_field
        )

    def _dynamic_programming_segments_open(
        self,
        points: np.ndarray,
        refined_keypoints: List[int],
        initial_segments: List[SemanticSegment],
        global_lines: List[GlobalLine],
        boundary_field: Optional[np.ndarray] = None,
    ) -> List[SemanticSegment]:
        n = len(points)
        mandatory = sorted(set([0, n - 1] + [int(v) for v in refined_keypoints]))
        if len(mandatory) < 2:
            return [self._fit_segment_model(points, 0, n - 1, global_lines=global_lines)]

        segment_list: List[SemanticSegment] = []
        for block_start, block_end in zip(mandatory[:-1], mandatory[1:]):
            boundary_set = {block_start, block_end}
            for seg in initial_segments:
                if block_start < int(seg.start_idx) < block_end:
                    boundary_set.add(int(seg.start_idx))
                if block_start < int(seg.end_idx) < block_end:
                    boundary_set.add(int(seg.end_idx))
            boundaries = self._merge_boundary_candidates_near_keypoints(
                points,
                sorted(v for v in boundary_set if block_start <= v <= block_end),
                set(int(v) for v in refined_keypoints),
                closed=False,
            )
            boundaries = self._compress_boundary_indices(boundaries)
            local_points = points[block_start : block_end + 1]
            local_bounds = [idx - block_start for idx in boundaries]
            local_bf = None
            if boundary_field is not None and len(boundary_field) == n:
                local_bf = boundary_field[block_start : block_end + 1]
            local_segments = self._dynamic_programming_segments_on_intervals(
                local_points, local_bounds, global_lines, boundary_field=local_bf
            )
            for seg in local_segments:
                orig_start = seg.start_idx + block_start
                orig_end = seg.end_idx + block_start
                orig_points = points[orig_start : orig_end + 1]
                segment_list.append(self._fit_segment_model(orig_points, orig_start, orig_end, global_lines=global_lines))
        return segment_list

    def _dynamic_programming_segments_closed(
        self,
        points: np.ndarray,
        refined_keypoints: List[int],
        initial_segments: List[SemanticSegment],
        global_lines: List[GlobalLine],
        boundary_field: Optional[np.ndarray] = None,
    ) -> List[SemanticSegment]:
        n = len(points)
        mandatory = sorted(set(int(v) % n for v in refined_keypoints))
        if len(mandatory) < 2:
            start = mandatory[0] if mandatory else 0
            work_points = np.vstack([points[start:], points[: start + 1]])
            boundary_set = {0, n}
            for seg in initial_segments:
                boundary_set.add((int(seg.start_idx) - start) % n)
                rel_end = (int(seg.end_idx) - start) % n
                if rel_end == 0 and seg.end_idx != start:
                    rel_end = n
                boundary_set.add(rel_end)
            rel_keypoints = {(int(v) - start) % n for v in refined_keypoints}
            rel_keypoints.add(0)
            rel_keypoints.add(n)
            boundaries = self._merge_boundary_candidates_near_keypoints(
                work_points,
                sorted(v for v in boundary_set if 0 <= v <= n),
                rel_keypoints,
                closed=False,
            )
            bf_work = None
            if boundary_field is not None and len(boundary_field) == n:
                bf_work = np.concatenate([boundary_field[start:], boundary_field[: start + 1]])
            open_segments = self._dynamic_programming_segments_on_intervals(
                work_points, boundaries, global_lines, boundary_field=bf_work
            )
            mapped: List[SemanticSegment] = []
            for seg in open_segments:
                work_end = int(seg.end_idx)
                orig_start = (int(seg.start_idx) + start) % n
                orig_end = start if work_end == n else (work_end + start) % n
                orig_points = self._slice_polyline(points, orig_start, orig_end, closed=True)
                mapped.append(self._fit_segment_model(orig_points, orig_start, orig_end, global_lines=global_lines))
            return self._rotate_segments(mapped)

        segment_list: List[SemanticSegment] = []
        for idx, block_start in enumerate(mandatory):
            block_end = mandatory[(idx + 1) % len(mandatory)]
            boundary_set = {block_start, block_end}
            for seg in initial_segments:
                start_rel = (int(seg.start_idx) - block_start) % n
                end_rel = (int(seg.end_idx) - block_start) % n
                span = (block_end - block_start) % n
                if 0 < start_rel < span:
                    boundary_set.add(int(seg.start_idx))
                if 0 < end_rel < span:
                    boundary_set.add(int(seg.end_idx))
            work_points = self._slice_polyline(points, block_start, block_end, closed=True)
            ordered = [0]
            span = len(work_points) - 1
            for boundary in sorted(boundary_set):
                rel = (int(boundary) - block_start) % n
                if 0 < rel < span:
                    ordered.append(rel)
            ordered.append(span)
            ordered = self._merge_boundary_candidates_near_keypoints(
                work_points,
                sorted(set(ordered)),
                {0, span},
                closed=False,
            )
            local_bf = None
            if boundary_field is not None and len(boundary_field) == n:
                local_bf = self._slice_ring_1d(boundary_field, block_start, block_end, closed=True)
            local_segments = self._dynamic_programming_segments_on_intervals(
                work_points, self._compress_boundary_indices(ordered), global_lines, boundary_field=local_bf
            )
            for seg in local_segments:
                orig_start = (seg.start_idx + block_start) % n
                orig_end = (seg.end_idx + block_start) % n
                orig_points = self._slice_polyline(points, orig_start, orig_end, closed=True)
                segment_list.append(self._fit_segment_model(orig_points, orig_start, orig_end, global_lines=global_lines))
        return self._rotate_segments(segment_list)

    def _dynamic_programming_segments_on_intervals(
        self,
        points: np.ndarray,
        boundaries: List[int],
        global_lines: List[GlobalLine],
        boundary_field: Optional[np.ndarray] = None,
    ) -> List[SemanticSegment]:
        """在一组候选边界上执行动态规划。

        每个状态表示“到第 i 个候选边界为止的最优解释”；
        转移时会枚举前一个边界 j，并对区间 [j, i] 拟合
        line / arc / spline，取总代价最低的组合。

        代价由：
        - 拟合误差；
        - primitive 复杂度 effective_params；
        - 可能被跨过的内部边界峰惩罚
        共同组成。
        """

        last_idx = max(0, len(points) - 1)
        boundaries = sorted({max(0, min(last_idx, int(v))) for v in boundaries})
        if boundaries and boundaries[0] != 0:
            boundaries.insert(0, 0)
        if boundaries and boundaries[-1] != last_idx:
            boundaries.append(last_idx)
        m = len(boundaries)
        if m < 2:
            return [self._fit_segment_model(points, 0, len(points) - 1, global_lines=global_lines)]

        dp = [float("inf")] * m
        prev = [-1] * m
        best_seg: List[Optional[SemanticSegment]] = [None] * m
        dp[0] = 0.0
        max_span = max(4, int(self.cfg.dp_max_segment_points))

        for i in range(1, m):
            for j in range(max(0, i - max_span), i):
                start_idx = boundaries[j]
                end_idx = boundaries[i]
                if end_idx <= start_idx:
                    continue
                seg_points = points[start_idx : end_idx + 1]
                if len(seg_points) < 2:
                    continue
                segment = self._fit_segment_model(seg_points, start_idx, end_idx, global_lines=global_lines)
                cost = self._segment_dp_cost(segment)
                if boundary_field is not None and len(boundary_field) == len(points):
                    cost += self._dp_interior_boundary_peak_penalty(boundary_field, start_idx, end_idx)
                total = dp[j] + cost
                if total < dp[i]:
                    dp[i] = total
                    prev[i] = j
                    best_seg[i] = segment

        if best_seg[-1] is None:
            return [self._fit_segment_model(points, boundaries[0], boundaries[-1], global_lines=global_lines)]

        segments: List[SemanticSegment] = []
        cursor = m - 1
        while cursor > 0 and best_seg[cursor] is not None:
            segments.append(best_seg[cursor])
            cursor = prev[cursor]
        segments.reverse()
        return segments

    def _compress_boundary_indices(self, boundaries: List[int]) -> List[int]:
        if len(boundaries) <= 2:
            return boundaries
        min_gap = 2
        max_count = max(12, min(28, int(self.cfg.dp_max_segment_points // 2)))
        compressed = [int(boundaries[0])]
        for idx in boundaries[1:-1]:
            if int(idx) - compressed[-1] >= min_gap:
                compressed.append(int(idx))
        if compressed[-1] != int(boundaries[-1]):
            compressed.append(int(boundaries[-1]))
        if len(compressed) <= max_count:
            return compressed
        interior = compressed[1:-1]
        keep = max(0, max_count - 2)
        if keep <= 0:
            return [compressed[0], compressed[-1]]
        sample_positions = np.linspace(0, len(interior) - 1, keep, dtype=int)
        sampled = [interior[pos] for pos in sample_positions.tolist()]
        return [compressed[0]] + sorted(set(sampled)) + [compressed[-1]]

    def _reassign_segments_to_global_lines(
        self, segments: List[SemanticSegment], global_lines: List[GlobalLine]
    ) -> List[SemanticSegment]:
        if not global_lines:
            return segments
        out: List[SemanticSegment] = []
        for seg in segments:
            if len(seg.points) < 2:
                out.append(seg)
                continue
            global_line_seg = self._best_global_line_segment(seg.points, seg.start_idx, seg.end_idx, global_lines)
            if global_line_seg is not None:
                # Global lines are only a snapping aid.  They must not be allowed
                # to overwrite a valid arc/spline merely because they exist.
                line_is_geometrically_valid = self._line_fit_is_acceptable(seg.points, global_line_seg.errors)
                line_cost_is_competitive = self._segment_dp_cost(global_line_seg) <= self._segment_dp_cost(seg) + 0.05
                if line_is_geometrically_valid and line_cost_is_competitive:
                    out.append(global_line_seg)
                    continue
            out.append(seg)
        return out

    def _merge_semantic_segments(
        self,
        points: np.ndarray,
        segments: List[SemanticSegment],
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
        protected_boundaries: Optional[set],
    ) -> List[SemanticSegment]:
        """对 DP 输出做后处理合并。

        DP 负责全局选择，但为了稳定保留真实角点，常常仍会留下可合并短段。
        这里继续尝试：
        - 相邻同几何段合并；
        - 连续直线 run 压缩；
        - line-short-line 桥接；
        - arc-line-arc 归并。
        """

        if len(segments) <= 1:
            return segments

        merged = segments[:]
        changed = True
        while changed and len(merged) > 1:
            changed = False
            pair_count = len(merged) if closed else len(merged) - 1
            i = 0
            while i < pair_count and len(merged) > 1:
                j = (i + 1) % len(merged)
                if not closed and j >= len(merged):
                    break
                candidate = self._try_merge_semantic_pair(
                    points,
                    merged[i],
                    merged[j],
                    closed=closed,
                    global_lines=global_lines,
                    protected_boundaries=protected_boundaries,
                )
                if candidate is None:
                    i += 1
                    continue
                merged[i] = candidate
                del merged[j]
                changed = True
                pair_count = len(merged) if closed else len(merged) - 1
            if closed and len(merged) > 1:
                merged = self._rotate_segments(merged)
        triple_merged = self._merge_arc_line_arc_triplets(
            points, merged, closed=closed, global_lines=global_lines, protected_boundaries=protected_boundaries
        )
        finalized = self._finalize_adjacent_same_geometry_merges(
            points,
            triple_merged,
            closed=closed,
            global_lines=global_lines,
            protected_boundaries=protected_boundaries,
        )
        line_run_merged = self._compress_line_runs(
            points,
            finalized,
            closed=closed,
            global_lines=global_lines,
            protected_boundaries=protected_boundaries,
        )
        return self._compress_line_bridge_runs(
            points,
            line_run_merged,
            closed=closed,
            global_lines=global_lines,
        )

    def _arcs_nearly_same_circle(self, first: SemanticSegment, second: SemanticSegment) -> bool:
        """相邻弧是否近似同一圆（略宽于 DP 合并时的弧一致性）。"""
        if first.arc_fit is None or second.arc_fit is None:
            return False
        if np.sign(first.arc_fit.sweep_angle) != np.sign(second.arc_fit.sweep_angle):
            return False
        rs = max(1.0, first.arc_fit.radius, second.arc_fit.radius)
        tr = float(self.cfg.arc_radius_rel_tol) * 1.55
        if abs(first.arc_fit.radius - second.arc_fit.radius) / rs > tr:
            return False
        ct = float(self.cfg.arc_center_tol) * 1.5
        return float(np.linalg.norm(first.arc_fit.center - second.arc_fit.center)) <= ct

    def _adaptive_line_arc_bridge_max_pts(self, n_poly: int) -> int:
        """收尾 line–arc 桥接：直线段允许多少个采样点（随轮廓长度缩放，无单独开关）。"""
        return max(5, min(14, max(1, int(n_poly)) // 22))

    def _try_finalize_merge_short_line_arc_to_arc(
        self,
        line_seg: SemanticSegment,
        arc_seg: SemanticSegment,
        union_points: np.ndarray,
        poly_start: int,
        poly_end: int,
        gl: List[GlobalLine],
        max_bridge: int,
    ) -> Optional[SemanticSegment]:
        """相邻短直线 + 弧：若并段可拟为单弧且与已知弧一致，则合并（用于 finalize）。"""
        if arc_seg.arc_fit is None:
            return None
        if len(line_seg.points) > max_bridge:
            return None
        if len(union_points) < 4:
            return None
        cand = self._fit_segment_model(union_points, poly_start, poly_end, gl)
        if cand.kind != "arc" or cand.arc_fit is None:
            return None
        rs = max(1.0, arc_seg.arc_fit.radius, cand.arc_fit.radius)
        tr = float(self.cfg.arc_radius_rel_tol) * 1.62
        if abs(arc_seg.arc_fit.radius - cand.arc_fit.radius) / rs > tr:
            return None
        ct = float(self.cfg.arc_center_tol) * 1.48
        if float(np.linalg.norm(arc_seg.arc_fit.center - cand.arc_fit.center)) > ct:
            return None
        if np.sign(arc_seg.arc_fit.sweep_angle) != np.sign(cand.arc_fit.sweep_angle):
            return None
        ft = float(self.cfg.fit_tolerance)
        lim = max(ft * 1.26, max(self._segment_max_error(line_seg), self._segment_max_error(arc_seg)) * 1.14)
        if self._segment_max_error(cand) > lim:
            return None
        return cand

    def _try_finalize_merge_pair(
        self,
        points: np.ndarray,
        first: SemanticSegment,
        second: SemanticSegment,
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
        protected_boundaries: Optional[set],
    ) -> Optional[SemanticSegment]:
        """收尾：同圆多弧并一段、共线多线并一段（略放宽误差门限）。"""
        if protected_boundaries is not None and int(second.start_idx) in protected_boundaries:
            if not self._polyline_vertex_nearly_straight(points, int(second.start_idx), closed):
                return None
        union_points = self._slice_polyline(points, first.start_idx, second.end_idx, closed=closed)
        if len(union_points) < 3:
            return None
        gl = global_lines or []

        if first.kind == "arc" and second.kind == "arc" and self._arcs_nearly_same_circle(first, second):
            cand = self._fit_segment_model(union_points, first.start_idx, second.end_idx, gl)
            if cand.kind != "arc" or cand.arc_fit is None:
                return None
            lim = max(
                float(self.cfg.fit_tolerance) * 1.32,
                max(self._segment_max_error(first), self._segment_max_error(second)) * 1.18,
            )
            if self._segment_max_error(cand) > lim:
                return None
            return cand

        if first.kind == "line" and second.kind == "line":
            a1 = self._segment_direction_deg(first.points)
            a2 = self._segment_direction_deg(second.points)
            span_pts = max(len(first.points), len(second.points))
            # 短边对栅格噪声更敏感：略放宽共线角阈，利于收尾并段、对齐 GT 直线段
            ang_cap = float(self.cfg.line_merge_angle_deg) * (1.22 if span_pts >= 16 else 1.36)
            if self._angle_difference_deg(a1, a2) > ang_cap:
                return None
            cand = self._fit_segment_model(union_points, first.start_idx, second.end_idx, gl)
            if cand.kind != "line":
                return None
            lim = max(
                float(self.cfg.fit_tolerance) * 1.12,
                max(self._segment_max_error(first), self._segment_max_error(second)) * 1.12,
            )
            if self._segment_max_error(cand) > lim:
                return None
            return cand

        max_bridge = self._adaptive_line_arc_bridge_max_pts(len(points))

        if first.kind == "line" and second.kind == "arc":
            return self._try_finalize_merge_short_line_arc_to_arc(
                first, second, union_points, first.start_idx, second.end_idx, gl, max_bridge
            )
        if first.kind == "arc" and second.kind == "line":
            return self._try_finalize_merge_short_line_arc_to_arc(
                second, first, union_points, first.start_idx, second.end_idx, gl, max_bridge
            )

        return None

    def _finalize_adjacent_same_geometry_merges(
        self,
        points: np.ndarray,
        segments: List[SemanticSegment],
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
        protected_boundaries: Optional[set],
    ) -> List[SemanticSegment]:
        """在弧–线–弧三元合并之后，再扫几轮：同圆相邻弧、近似共线相邻线。"""
        if len(segments) < 2:
            return segments
        merged = segments[:]
        n0 = len(merged)
        max_passes = min(10, max(6, 4 + n0 // 6))
        for _ in range(max_passes):
            changed = False
            pair_count = len(merged) if closed else len(merged) - 1
            i = 0
            while i < pair_count and len(merged) > 1:
                j = (i + 1) % len(merged)
                if not closed and j >= len(merged):
                    break
                cand = self._try_finalize_merge_pair(
                    points, merged[i], merged[j], closed=closed, global_lines=global_lines, protected_boundaries=protected_boundaries
                )
                if cand is None:
                    i += 1
                    continue
                merged[i] = cand
                del merged[j]
                changed = True
                pair_count = len(merged) if closed else len(merged) - 1
            if closed and len(merged) > 1:
                merged = self._rotate_segments(merged)
            if not changed:
                break
        return merged

    def _try_compress_line_run(
        self,
        points: np.ndarray,
        run: List[SemanticSegment],
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
        protected_boundaries: Optional[set],
    ) -> Optional[SemanticSegment]:
        if len(run) < 2 or any(seg.kind != "line" for seg in run):
            return None
        if len(run) == 2:
            boundary = int(run[1].start_idx)
            if protected_boundaries is not None and boundary in protected_boundaries:
                return None
            if self._vertex_turn_sharpness(points, boundary, closed) > 5.0:
                return None
        start_idx = int(run[0].start_idx)
        end_idx = int(run[-1].end_idx)
        union_points = self._slice_polyline(points, start_idx, end_idx, closed=closed)
        if len(union_points) < 3:
            return None

        cand = self._fit_line_segment(union_points, start_idx, end_idx)
        factor = max(1.0, float(getattr(self.cfg, "line_run_merge_error_factor", 1.18)))
        ft = float(self.cfg.fit_tolerance)
        per_max = max(self._segment_max_error(seg) for seg in run)
        per_mean = max(float(np.mean(seg.errors)) if len(seg.errors) else 0.0 for seg in run)
        max_lim = max(ft * 0.92, per_max * factor, 0.72)
        mean_lim = max(ft * 0.38, per_mean * factor, 0.32)
        cand_max = self._segment_max_error(cand)
        cand_mean = float(np.mean(cand.errors)) if len(cand.errors) else 0.0
        if cand_max > max_lim or cand_mean > mean_lim:
            return None

        # 防止跨过真实拐角：所有子线方向必须与整段方向一致。
        run_dir = self._segment_direction_deg(cand.points)
        span_pts = max(len(seg.points) for seg in run)
        ang_cap = float(self.cfg.line_merge_angle_deg) * (1.05 if span_pts >= 16 else 1.25)
        for seg in run:
            if self._angle_difference_deg(self._segment_direction_deg(seg.points), run_dir) > ang_cap:
                return None

        # 若存在全局线，优先复用全局线段，但仍要求误差不高于本地直线。
        gl = self._best_global_line_segment(union_points, start_idx, end_idx, global_lines or [])
        if gl is not None and self._segment_max_error(gl) <= cand_max * 1.03 + 1e-9:
            return gl
        return cand

    def _compress_line_runs(
        self,
        points: np.ndarray,
        segments: List[SemanticSegment],
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
        protected_boundaries: Optional[set],
    ) -> List[SemanticSegment]:
        """迭代合并相邻共线 line 段，专门减少直线上的冗余分段点。"""
        if len(segments) < 2:
            return segments
        merged = self._rotate_segments(segments) if closed else segments[:]
        max_passes = max(6, len(merged) * 2)
        for _ in range(max_passes):
            if len(merged) < 2:
                break
            changed = False
            pair_count = len(merged) if closed else len(merged) - 1
            for i in range(pair_count):
                j = (i + 1) % len(merged)
                if not closed and j >= len(merged):
                    break
                cand = self._try_compress_line_run(
                    points,
                    [merged[i], merged[j]],
                    closed=closed,
                    global_lines=global_lines,
                    protected_boundaries=protected_boundaries,
                )
                if cand is None:
                    continue
                merged[i] = cand
                del merged[j]
                changed = True
                break
            if not changed:
                break
            if closed and len(merged) > 1:
                merged = self._rotate_segments(merged)
        return self._rotate_segments(merged) if closed else merged

    def _try_compress_line_bridge_triplet(
        self,
        points: np.ndarray,
        left: SemanticSegment,
        mid: SemanticSegment,
        right: SemanticSegment,
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
    ) -> Optional[SemanticSegment]:
        if left.kind != "line" or right.kind != "line" or mid.kind == "line":
            return None
        if len(mid.points) > max(2, int(getattr(self.cfg, "line_bridge_max_points", 8))):
            return None
        union_points = self._slice_polyline(points, left.start_idx, right.end_idx, closed=closed)
        if len(union_points) < 4:
            return None
        cand = self._fit_line_segment(union_points, left.start_idx, right.end_idx)
        ft = float(self.cfg.fit_tolerance)
        factor = max(1.0, float(getattr(self.cfg, "line_run_merge_error_factor", 1.18)))
        per_max = max(self._segment_max_error(left), self._segment_max_error(mid), self._segment_max_error(right))
        max_lim = max(ft * 0.86, per_max * factor, 0.68)
        mean_lim = max(ft * 0.34, 0.30)
        cand_max = self._segment_max_error(cand)
        cand_mean = float(np.mean(cand.errors)) if len(cand.errors) else 0.0
        if cand_max > max_lim or cand_mean > mean_lim:
            return None
        run_dir = self._segment_direction_deg(cand.points)
        ang_cap = float(self.cfg.line_merge_angle_deg) * 1.12
        if self._angle_difference_deg(self._segment_direction_deg(left.points), run_dir) > ang_cap:
            return None
        if self._angle_difference_deg(self._segment_direction_deg(right.points), run_dir) > ang_cap:
            return None
        gl = self._best_global_line_segment(union_points, left.start_idx, right.end_idx, global_lines or [])
        if gl is not None and self._segment_max_error(gl) <= cand_max * 1.03 + 1e-9:
            return gl
        return cand

    def _compress_line_bridge_runs(
        self,
        points: np.ndarray,
        segments: List[SemanticSegment],
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
    ) -> List[SemanticSegment]:
        """将 line-短非line-line 直线桥接压缩，处理直线中夹一个短 arc/spline 的多余点。"""
        if len(segments) < 3:
            return segments
        merged = self._rotate_segments(segments) if closed else segments[:]
        max_passes = max(6, len(merged))
        for _ in range(max_passes):
            if len(merged) < 3:
                break
            changed = False
            count = len(merged) if closed else len(merged) - 2
            for i in range(count):
                a = i
                b = (i + 1) % len(merged)
                c = (i + 2) % len(merged)
                if not closed and c >= len(merged):
                    break
                cand = self._try_compress_line_bridge_triplet(
                    points, merged[a], merged[b], merged[c], closed=closed, global_lines=global_lines
                )
                if cand is None:
                    continue
                merged[a] = cand
                # 删除顺序从大到小，兼容闭合取模后 a/b/c 正常递增的旋转列表。
                for idx in sorted({b, c}, reverse=True):
                    del merged[idx]
                changed = True
                break
            if not changed:
                break
            if closed and len(merged) > 1:
                merged = self._rotate_segments(merged)
        return self._rotate_segments(merged) if closed else merged

    def _merge_arc_line_arc_triplets(
        self,
        points: np.ndarray,
        segments: List[SemanticSegment],
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
        protected_boundaries: Optional[set],
    ) -> List[SemanticSegment]:
        """Collapse arc–line–arc (or arc–arc–arc) runs that belong to one circular arc."""
        if len(segments) < 3:
            return segments
        gl = global_lines or []
        out = segments[:]
        max_passes = max(len(out) * 6, 24)
        for _ in range(max_passes):
            if len(out) < 3:
                break
            m = len(out)
            progressed = False
            if closed:
                for i in range(m):
                    A = out[i]
                    B = out[(i + 1) % m]
                    C = out[(i + 2) % m]
                    cand = self._try_merge_three_to_single_arc(
                        points, A, B, C, closed=closed, global_lines=gl, protected_boundaries=protected_boundaries
                    )
                    if cand is None:
                        continue
                    rot = out[i:] + out[:i]
                    out = [cand] + rot[3:]
                    progressed = True
                    break
            else:
                for i in range(m - 2):
                    A, B, C = out[i], out[i + 1], out[i + 2]
                    cand = self._try_merge_three_to_single_arc(
                        points, A, B, C, closed=closed, global_lines=gl, protected_boundaries=protected_boundaries
                    )
                    if cand is None:
                        continue
                    out = out[:i] + [cand] + out[i + 3 :]
                    progressed = True
                    break
            if not progressed:
                break
        return out

    def _try_merge_three_to_single_arc(
        self,
        points: np.ndarray,
        first: SemanticSegment,
        mid: SemanticSegment,
        last: SemanticSegment,
        closed: bool,
        global_lines: List[GlobalLine],
        protected_boundaries: Optional[set],
    ) -> Optional[SemanticSegment]:
        if not (first.kind == "arc" and last.kind == "arc" and mid.kind in ("line", "arc")):
            return None
        if protected_boundaries is not None and int(mid.start_idx) in protected_boundaries:
            if not self._polyline_vertex_nearly_straight(points, int(mid.start_idx), closed):
                return None
        if protected_boundaries is not None and int(last.start_idx) in protected_boundaries:
            if not self._polyline_vertex_nearly_straight(points, int(last.start_idx), closed):
                return None
        union_points = self._slice_polyline(points, first.start_idx, last.end_idx, closed=closed)
        if len(union_points) < 6:
            return None
        cand = self._fit_segment_model(union_points, first.start_idx, last.end_idx, global_lines)
        if cand.kind != "arc" or cand.arc_fit is None:
            return None
        lim = max(
            float(self.cfg.fit_tolerance) * 1.4,
            max(
                self._segment_max_error(first),
                self._segment_max_error(mid),
                self._segment_max_error(last),
            )
            * 1.22,
        )
        if self._segment_max_error(cand) > lim:
            return None
        if first.arc_fit is None or last.arc_fit is None:
            return None
        rs = max(1.0, cand.arc_fit.radius, first.arc_fit.radius, last.arc_fit.radius)
        tol_r = float(self.cfg.arc_radius_rel_tol) * 1.35
        if abs(first.arc_fit.radius - cand.arc_fit.radius) / rs > tol_r or abs(last.arc_fit.radius - cand.arc_fit.radius) / rs > tol_r:
            return None
        ct = float(self.cfg.arc_center_tol) * 1.45
        if np.linalg.norm(first.arc_fit.center - cand.arc_fit.center) > ct or np.linalg.norm(last.arc_fit.center - cand.arc_fit.center) > ct:
            return None
        if (
            np.sign(first.arc_fit.sweep_angle) != np.sign(last.arc_fit.sweep_angle)
            or np.sign(cand.arc_fit.sweep_angle) != np.sign(first.arc_fit.sweep_angle)
        ):
            return None
        return cand

    def _maybe_collapse_constant_curvature_closed_loop(
        self,
        points: np.ndarray,
        segments: List[SemanticSegment],
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
    ) -> List[SemanticSegment]:
        """若闭合轮廓整体近似恒曲率，则尝试压成一个整圆弧。"""

        """若闭合轮廓曲率近似常数且整环可用单段圆弧拟合，则合并为一段弧（常见：整圆）。"""
        if not closed or len(segments) < 2:
            return segments
        n = len(points)
        if n < 20:
            return segments
        half = max(2, min(8, n // 40))
        stride = max(1, n // 64)
        turns: List[float] = []
        for i in range(0, n, stride):
            p0 = points[(i - half) % n]
            p1 = points[i % n]
            p2 = points[(i + half) % n]
            ang = self._unsigned_turn_angle_deg(p0, p1, p2)
            turns.append(float(180.0 - ang))
        turn_arr = np.asarray(turns, dtype=np.float64)
        if turn_arr.size < 5:
            return segments
        t_mean = float(np.mean(turn_arr))
        t_std = float(np.std(turn_arr))
        if t_std > max(7.5, 0.48 * (t_mean + 3.0)):
            return segments
        if t_mean > 58.0 and t_std > 7.0:
            return segments

        full = points.astype(np.float64, copy=False)
        arc_cand = self._fit_arc_segment(full, 0, n - 1)
        if arc_cand is None or arc_cand.arc_fit is None:
            return segments
        af = arc_cand.arc_fit
        sweep_deg = abs(math.degrees(af.sweep_angle))
        if sweep_deg < 255.0:
            return segments

        max_e_arc = float(np.max(arc_cand.errors))
        mean_e_arc = float(np.mean(arc_cand.errors))
        ft = float(self.cfg.fit_tolerance)
        # 较长闭合折线采样更密时，对弧误差上界略放宽（有 cap），减少假阴性
        n_relax = min(1.1, 1.0 + max(0, n - 120) / 3200.0)
        lim_max = max(1.05, ft * 1.38 * n_relax)
        lim_mean = max(0.52, ft * 0.62 * min(1.06, n_relax))
        if not self._arc_fit_is_acceptable(full, af):
            if max_e_arc > lim_max or mean_e_arc > lim_mean:
                return segments
        rel_spread = float(np.std(arc_cand.errors) / max(af.radius, 1.0))
        if rel_spread > 0.042:
            return segments

        unified = self._fit_segment_model(full, 0, n - 1, global_lines or [])
        chosen = unified if unified.kind == "arc" and unified.arc_fit is not None else arc_cand
        if chosen.kind != "arc" or chosen.arc_fit is None:
            return segments

        per_max = max(self._segment_max_error(s) for s in segments)
        per_lim = max(per_max * 1.18, ft * 1.42 * n_relax)
        if self._segment_max_error(chosen) > per_lim:
            return segments
        total_eff = sum(int(s.effective_params) for s in segments)
        if chosen.effective_params >= total_eff and self._segment_max_error(chosen) > per_max * 1.04:
            return segments
        return [chosen]

    def _try_merge_semantic_pair(
        self,
        points: np.ndarray,
        first: SemanticSegment,
        second: SemanticSegment,
        closed: bool,
        global_lines: Optional[List[GlobalLine]],
        protected_boundaries: Optional[set],
    ) -> Optional[SemanticSegment]:
        if protected_boundaries is not None and int(second.start_idx) in protected_boundaries:
            span = max(len(first.points), len(second.points))
            # 仅「双 line 且跨距很短」时用宽松弱角（与 finalize 一致）；其余（异类型或较长边）仍严格，保持 v2 级边界/类型收益
            short_line_pair = first.kind == "line" and second.kind == "line" and span < 14
            use_strict = not short_line_pair
            if not self._polyline_vertex_nearly_straight(points, int(second.start_idx), closed, strict=use_strict):
                return None
        union_points = self._slice_polyline(points, first.start_idx, second.end_idx, closed=closed)
        if len(union_points) < 2:
            return None
        candidate = self._fit_segment_model(union_points, first.start_idx, second.end_idx, global_lines=global_lines)
        pair_error_limit = max(float(self.cfg.fit_tolerance) * 1.2, max(self._segment_max_error(first), self._segment_max_error(second)) * 1.15)
        current_params = first.effective_params + second.effective_params
        if candidate.effective_params < current_params:
            pair_error_limit *= 1.05
        if self._segment_max_error(candidate) > pair_error_limit:
            return None

        if first.kind == second.kind == candidate.kind and self._same_type_merge_is_consistent(first, second, candidate):
            return candidate

        cost_slack = 0.09 if candidate.effective_params < current_params else 0.05
        if candidate.effective_params <= current_params and self._segment_dp_cost(candidate) <= self._segment_dp_cost(first) + self._segment_dp_cost(second) + cost_slack:
            return candidate
        return None

    def _same_type_merge_is_consistent(
        self, first: SemanticSegment, second: SemanticSegment, merged: SemanticSegment
    ) -> bool:
        if merged.kind == "line":
            first_angle = self._segment_direction_deg(first.points)
            second_angle = self._segment_direction_deg(second.points)
            span_pts = max(len(first.points), len(second.points))
            # 与 finalize 一致：短直线受栅格影响更大，主合并阶段略放宽共线角判据
            ang_cap = float(self.cfg.line_merge_angle_deg) * (1.0 if span_pts >= 16 else 1.14)
            return self._angle_difference_deg(first_angle, second_angle) <= ang_cap
        if merged.kind == "arc":
            if first.arc_fit is None or second.arc_fit is None or merged.arc_fit is None:
                return False
            radius_scale = max(1.0, merged.arc_fit.radius)
            radius_ok = (
                abs(first.arc_fit.radius - second.arc_fit.radius) / radius_scale <= float(self.cfg.arc_radius_rel_tol)
                and abs(first.arc_fit.radius - merged.arc_fit.radius) / radius_scale <= float(self.cfg.arc_radius_rel_tol)
                and abs(second.arc_fit.radius - merged.arc_fit.radius) / radius_scale <= float(self.cfg.arc_radius_rel_tol)
            )
            center_ok = (
                np.linalg.norm(first.arc_fit.center - second.arc_fit.center) <= float(self.cfg.arc_center_tol)
                and np.linalg.norm(first.arc_fit.center - merged.arc_fit.center) <= float(self.cfg.arc_center_tol) * 1.2
                and np.linalg.norm(second.arc_fit.center - merged.arc_fit.center) <= float(self.cfg.arc_center_tol) * 1.2
            )
            direction_ok = (
                np.sign(first.arc_fit.sweep_angle) == np.sign(second.arc_fit.sweep_angle)
                == np.sign(merged.arc_fit.sweep_angle)
            )
            return bool(radius_ok and center_ok and direction_ok)
        return merged.effective_params + 1 <= first.effective_params + second.effective_params

    # ------------------------------------------------------------------
    # primitive 拟合与最终 path 构造
    # ------------------------------------------------------------------

    def _build_semantic_path_fit(
        self, points: np.ndarray, segments: List[SemanticSegment], closed: bool
    ) -> CenterlinePathFit:
        """把最终语义段列表转换为完整的 SVG path 与汇总结果。"""

        if not segments:
            return CenterlinePathFit(
                d="",
                errors=np.zeros(0, dtype=np.float64),
                effective_params=0,
                line_segments=0,
                arc_segments=0,
                curve_segments=0,
                bspline_control_points=0,
            )

        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        start_point = segments[0].points[0]
        cmds = [f"M {fmt(start_point[0])} {fmt(start_point[1])}"]
        all_errors: List[np.ndarray] = []
        primitive_summary: List[Dict[str, Any]] = []
        total_effective_params = 0
        total_line_segments = 0
        total_arc_segments = 0
        total_curve_segments = 0
        total_bspline_control_points = 0

        for segment in segments:
            primitive_summary.append(self._segment_to_summary(segment))
            all_errors.append(segment.errors)
            total_effective_params += segment.effective_params
            total_line_segments += segment.line_segments
            total_arc_segments += segment.arc_segments
            total_curve_segments += segment.curve_segments
            total_bspline_control_points += segment.bspline_control_points

            if segment.kind == "line":
                endpoint = segment.line_end if segment.line_end is not None else segment.points[-1]
                cmds.append(f"L {fmt(endpoint[0])} {fmt(endpoint[1])}")
                continue

            if segment.kind == "arc" and segment.arc_fit is not None:
                cmds.append(self._arc_fit_to_svg_command(segment.arc_fit, segment.points[-1]))
                continue

            if segment.bspline_fit is not None:
                seg_cmds = self._open_fit_to_svg_commands(segment.bspline_fit)
                seg_cmds = self._replace_last_segment_endpoint(seg_cmds, segment.points[-1])
                cmds.extend(seg_cmds)
                continue

            endpoint = segment.points[-1]
            cmds.append(f"L {fmt(endpoint[0])} {fmt(endpoint[1])}")
            total_line_segments += 1

        if closed:
            cmds.append("Z")

        errors = np.concatenate(all_errors) if all_errors else np.zeros(0, dtype=np.float64)
        return CenterlinePathFit(
            d=" ".join(cmds),
            errors=errors,
            effective_params=max(1, total_effective_params),
            line_segments=total_line_segments,
            arc_segments=total_arc_segments,
            curve_segments=total_curve_segments,
            bspline_control_points=total_bspline_control_points,
            keypoints=len(self._semantic_keypoints(segments, closed=closed)),
            primitive_summary=primitive_summary,
        )

    def _fit_segment_model(
        self,
        points: np.ndarray,
        start_idx: int,
        end_idx: int,
        global_lines: Optional[List[GlobalLine]],
    ) -> SemanticSegment:
        """给一段点选择最合适的 primitive 类型。

        候选模型包括：
        - line：最低复杂度；
        - arc：中等复杂度；
        - spline：表达力最强，也最容易过拟合。

        选择并不是单纯取最小误差，而是结合：
        - 可接受误差阈值；
        - 模型复杂度；
        - line / arc 的几何合理性；
        - 与近似最优模型的比较。
        """

        if len(points) <= 2:
            return self._fit_line_segment(points, start_idx, end_idx)

        local_line_seg = self._fit_line_segment(points, start_idx, end_idx)
        global_line_seg = self._best_global_line_segment(points, start_idx, end_idx, global_lines or [])
        arc_seg = self._fit_arc_segment(points, start_idx, end_idx)
        candidates: List[SemanticSegment] = []
        acceptable_line_candidates: List[SemanticSegment] = []

        if self._line_fit_is_acceptable(points, local_line_seg.errors):
            candidates.append(local_line_seg)
            acceptable_line_candidates.append(local_line_seg)
        if global_line_seg is not None and self._line_fit_is_acceptable(points, global_line_seg.errors):
            candidates.append(global_line_seg)
            acceptable_line_candidates.append(global_line_seg)
        if arc_seg is not None and self._arc_fit_is_acceptable(points, arc_seg.arc_fit):
            candidates.append(arc_seg)

        if acceptable_line_candidates:
            best_line = min(acceptable_line_candidates, key=self._segment_dp_cost)
            line_error = self._segment_max_error(best_line)
            if len(points) <= 24 or line_error <= max(0.45, float(self.cfg.fit_tolerance) * 0.55):
                return best_line

        if arc_seg is not None and self._arc_fit_is_acceptable(points, arc_seg.arc_fit):
            arc_error = self._segment_max_error(arc_seg)
            if len(points) <= 28 or arc_error <= max(0.35, float(self.cfg.fit_tolerance) * 0.45):
                return arc_seg

        if candidates:
            cheap_best = min(candidates, key=self._segment_dp_cost)
            cheap_error = self._segment_max_error(cheap_best)
            if cheap_best.kind != "spline" and cheap_error <= max(0.3, float(self.cfg.fit_tolerance) * 0.35):
                return cheap_best

        spline_needed = True
        if acceptable_line_candidates:
            best_line = min(acceptable_line_candidates, key=self._segment_dp_cost)
            if len(points) <= 36 and self._segment_max_error(best_line) <= max(0.8, float(self.cfg.fit_tolerance) * 0.9):
                spline_needed = False
        if not spline_needed and candidates:
            scored = [(self._segment_dp_cost(seg), self._model_rank(seg.kind), seg) for seg in candidates]
            best_cost = min(cost for cost, _rank, _seg in scored)
            near_best = [item for item in scored if item[0] <= best_cost + 0.12]
            near_best.sort(key=lambda item: (item[1], item[0], item[2].effective_params))
            return near_best[0][2]

        spline_seg = self._fit_spline_segment(points, start_idx, end_idx)
        candidates.append(spline_seg)

        scored = [(self._segment_dp_cost(seg), self._model_rank(seg.kind), seg) for seg in candidates]
        best_cost = min(cost for cost, _rank, _seg in scored)
        near_best = [item for item in scored if item[0] <= best_cost + 0.12]
        near_best.sort(key=lambda item: (item[1], item[0], item[2].effective_params))
        return near_best[0][2]

    def _fit_line_segment(self, points: np.ndarray, start_idx: int, end_idx: int) -> SemanticSegment:
        """把一段点拟合为直线 primitive。"""

        descriptor = self._fit_infinite_line(points)
        if descriptor is None:
            errors = self._segment_line_errors(points)
        else:
            _theta, _distance, _normal, _direction, _anchor, errors = descriptor
        return SemanticSegment(
            kind="line",
            start_idx=start_idx,
            end_idx=end_idx,
            points=points,
            errors=errors,
            effective_params=1,
            line_segments=1,
            arc_segments=0,
            curve_segments=0,
            bspline_control_points=0,
            line_start=points[0],
            line_end=points[-1],
        )

    def _fit_arc_segment(self, points: np.ndarray, start_idx: int, end_idx: int) -> Optional[SemanticSegment]:
        """把一段点拟合为圆弧 primitive；无法形成有效圆弧时返回 None。"""

        arc_fit = self._fit_circle_arc(points)
        if arc_fit is None:
            return None
        return SemanticSegment(
            kind="arc",
            start_idx=start_idx,
            end_idx=end_idx,
            points=points,
            errors=arc_fit.errors,
            effective_params=2,
            line_segments=0,
            arc_segments=1,
            curve_segments=0,
            bspline_control_points=0,
            arc_fit=arc_fit,
        )

    def _fit_spline_segment(self, points: np.ndarray, start_idx: int, end_idx: int) -> SemanticSegment:
        """把一段点拟合为自适应 B 样条 primitive。"""

        cache_key = (int(start_idx), int(end_idx), False)
        cached = self._active_spline_cache.get(cache_key)
        if cached is not None:
            return cached
        fit = self._fit_bspline_adaptive(points, closed=False)
        segment = SemanticSegment(
            kind="spline",
            start_idx=start_idx,
            end_idx=end_idx,
            points=points,
            errors=fit.errors,
            effective_params=self._fit_effective_params(fit),
            line_segments=0,
            arc_segments=0,
            curve_segments=self._fit_curve_segment_count(fit),
            bspline_control_points=int(len(fit.control_points)),
            bspline_fit=fit,
        )
        self._active_spline_cache[cache_key] = segment
        return segment

    def _best_global_line_segment(
        self, points: np.ndarray, start_idx: int, end_idx: int, global_lines: List[GlobalLine]
    ) -> Optional[SemanticSegment]:
        best: Optional[SemanticSegment] = None
        best_cost = float("inf")
        for line in global_lines:
            errors = np.abs((points - line.anchor[None, :]) @ line.normal)
            candidate = SemanticSegment(
                kind="line",
                start_idx=start_idx,
                end_idx=end_idx,
                points=points,
                errors=errors,
                effective_params=1,
                line_segments=1,
                arc_segments=0,
                curve_segments=0,
                bspline_control_points=0,
                line_start=points[0],
                line_end=points[-1],
                global_line_id=line.line_id,
            )
            cost = self._segment_dp_cost(candidate)
            if cost < best_cost:
                best_cost = cost
                best = candidate
        return best

    def _fit_infinite_line(
        self, points: np.ndarray
    ) -> Optional[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        if len(points) < 2:
            return None
        anchor = np.mean(points, axis=0)
        centered = points - anchor
        try:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        direction = vh[0]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return None
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        if normal[1] < 0 or (abs(normal[1]) < 1e-9 and normal[0] < 0):
            normal = -normal
            direction = -direction
        theta = math.atan2(normal[1], normal[0]) % math.pi
        distance = float(np.dot(normal, anchor))
        errors = np.abs(centered @ normal)
        return theta, distance, normal, direction, anchor, errors

    def _cluster_keypoint_mask(self, mask: np.ndarray, scores: np.ndarray, closed: bool) -> List[int]:
        runs = self._cluster_true_runs(mask, closed=closed)
        candidates: List[int] = []
        for run in runs:
            best = max(run, key=lambda idx: (float(scores[idx]), -abs(idx - run[len(run) // 2])))
            candidates.append(int(best))
        return candidates

    def _keypoint_local_cost(self, points: np.ndarray, left_idx: int, center_idx: int, right_idx: int, closed: bool) -> float:
        left_points = self._slice_polyline(points, left_idx, center_idx, closed=closed)
        right_points = self._slice_polyline(points, center_idx, right_idx, closed=closed)
        if len(left_points) < 2 or len(right_points) < 2:
            return float("inf")
        return self._quick_segment_cost(left_points, left_idx, center_idx) + self._quick_segment_cost(
            right_points, center_idx, right_idx
        )

    def _quick_segment_cost(self, points: np.ndarray, start_idx: int, end_idx: int) -> float:
        line_seg = self._fit_line_segment(points, start_idx, end_idx)
        candidates = [line_seg]
        arc_seg = self._fit_arc_segment(points, start_idx, end_idx)
        if self._line_fit_is_acceptable(points, line_seg.errors):
            candidates = [line_seg]
        elif arc_seg is not None and self._arc_fit_is_acceptable(points, arc_seg.arc_fit):
            candidates.append(arc_seg)
        else:
            if arc_seg is not None:
                candidates.append(arc_seg)
        return min(self._segment_dp_cost(seg) for seg in candidates)

    def _refined_keypoint_candidate_valid(
        self, prev_idx: int, candidate_idx: int, next_idx: int, n: int, closed: bool
    ) -> bool:
        if closed:
            left = (candidate_idx - prev_idx) % n
            right = (next_idx - candidate_idx) % n
            return left >= 2 and right >= 2
        return prev_idx + 1 < candidate_idx < next_idx - 1

    def _dedupe_sorted_indices(self, indices: List[int], n: int, closed: bool) -> List[int]:
        if not indices:
            return []
        ordered = sorted(set(int(v) % n for v in indices))
        if not closed:
            return ordered
        cleaned: List[int] = []
        for idx in ordered:
            if cleaned and (idx - cleaned[-1]) % n < 2:
                continue
            cleaned.append(idx)
        if len(cleaned) > 1 and (cleaned[0] - cleaned[-1]) % n < 2:
            cleaned.pop()
        return cleaned or [0]

    def _polyline_vertex_nearly_straight(self, points: np.ndarray, idx: int, closed: bool, strict: bool = False) -> bool:
        """接缝顶点在折线上是否近似共线（用于受保护关键点：仅弱角允许跨点合并）。

        strict=True 用于主语义合并 `_try_merge_semantic_pair`：略收紧，减轻 GT 真拐角处
        因平滑折线「看似钝」而误并导致的 breakpoints_mae 回退；finalize / 弧三元仍默认 strict=False。
        """
        n = len(points)
        if n < 3:
            return True
        i = int(idx) % n
        if not closed and (i <= 0 or i >= n - 1):
            return True
        lm = float(self.cfg.line_merge_angle_deg)
        if strict:
            delta = min(15.0, max(8.0, lm * 0.72))
        else:
            delta = min(22.0, max(12.0, lm * 0.95))
        threshold = 180.0 - delta
        radii = [1]
        max_r = max(1, int(getattr(self.cfg, "protected_straight_window_radius", 4)))
        for r in (2, max_r):
            if r not in radii:
                radii.append(r)
        # 多尺度看是否“在一条直线趋势上”：相邻 1 像素可能受骨架/采样锯齿影响，较大窗口更能识别直边伪角。
        for r in radii:
            if not closed and (i - r < 0 or i + r >= n):
                continue
            prev_i = (i - r) % n if closed else i - r
            next_i = (i + r) % n if closed else i + r
            ang = self._unsigned_turn_angle_deg(points[prev_i], points[i], points[next_i])
            if ang >= threshold:
                return True
        return False

    def _segment_dp_cost(self, segment: SemanticSegment) -> float:
        """计算一个 primitive 在 DP 中的代价。"""

        mean_error = float(np.mean(segment.errors)) if len(segment.errors) else 0.0
        max_error = float(np.max(segment.errors)) if len(segment.errors) else 0.0
        complexity = float(self.cfg.dp_complexity_weight) * float(segment.effective_params)
        return mean_error + 0.35 * max_error + complexity

    @staticmethod
    def _model_rank(kind: str) -> int:
        if kind == "line":
            return 0
        if kind == "arc":
            return 1
        return 2

    @staticmethod
    def _project_point_to_line(point: np.ndarray, anchor: np.ndarray, direction: np.ndarray) -> np.ndarray:
        t = float(np.dot(point - anchor, direction))
        return anchor + t * direction

    def _fit_circle_arc(self, points: np.ndarray) -> Optional[ArcFit]:
        """最小二乘圆拟合，并生成圆弧参数。"""

        if len(points) < 5:
            return None
        x = points[:, 0]
        y = points[:, 1]
        a = np.column_stack([x, y, np.ones(len(points), dtype=np.float64)])
        b = -(x**2 + y**2)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        center = np.array([-0.5 * coeffs[0], -0.5 * coeffs[1]], dtype=np.float64)
        radius_sq = float(np.dot(center, center) - coeffs[2])
        if not np.isfinite(radius_sq) or radius_sq <= 1e-6:
            return None
        radius = math.sqrt(radius_sq)
        distances = np.linalg.norm(points - center, axis=1)
        errors = np.abs(distances - radius)
        angles = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
        sweep = float(angles[-1] - angles[0])
        if abs(math.degrees(sweep)) < float(self.cfg.arc_min_sweep_deg):
            return None
        return ArcFit(center=center, radius=radius, start_angle=float(angles[0]), sweep_angle=sweep, errors=errors)

    def _local_window_points(self, points: np.ndarray, center: int, half: int, closed: bool) -> np.ndarray:
        n = len(points)
        if closed:
            idx = [int((center + offset) % n) for offset in range(-half, half + 1)]
            return points[idx]
        start = max(0, center - half)
        end = min(n - 1, center + half)
        return points[start : end + 1]

    def _semantic_break(
        self, prev_label: str, curr_label: str, prev_signature: Tuple[float, ...], curr_signature: Tuple[float, ...]
    ) -> bool:
        if prev_label != curr_label:
            return True
        if prev_label == "line" and prev_signature and curr_signature:
            return self._angle_difference_deg(prev_signature[0], curr_signature[0]) > float(self.cfg.line_merge_angle_deg)
        if prev_label == "arc" and len(prev_signature) == 4 and len(curr_signature) == 4:
            radius_scale = max(1.0, prev_signature[0], curr_signature[0])
            radius_delta = abs(prev_signature[0] - curr_signature[0]) / radius_scale
            center_delta = math.hypot(prev_signature[1] - curr_signature[1], prev_signature[2] - curr_signature[2])
            return radius_delta > float(self.cfg.arc_radius_rel_tol) or center_delta > float(self.cfg.arc_center_tol)
        return False

    def _coalesce_tiny_segments(
        self, points: np.ndarray, segments: List[SemanticSegment], closed: bool
    ) -> List[SemanticSegment]:
        if len(segments) <= 1:
            return segments
        min_samples = 4
        out = segments[:]
        i = 0
        while i < len(out):
            if len(out[i].points) >= min_samples:
                i += 1
                continue
            if len(out) == 1:
                break
            if i == 0 and not closed:
                neighbor = 1
            elif i == len(out) - 1 and not closed:
                neighbor = i - 1
            else:
                prev_len = len(out[(i - 1) % len(out)].points)
                next_len = len(out[(i + 1) % len(out)].points)
                neighbor = (i - 1) % len(out) if prev_len >= next_len else (i + 1) % len(out)

            if neighbor < i:
                merged = self._fit_segment_model(
                    self._slice_polyline(points, out[neighbor].start_idx, out[i].end_idx, closed=closed),
                    out[neighbor].start_idx,
                    out[i].end_idx,
                    global_lines=None,
                )
                out[neighbor] = merged
                del out[i]
                i = max(0, neighbor - 1)
            else:
                merged = self._fit_segment_model(
                    self._slice_polyline(points, out[i].start_idx, out[neighbor].end_idx, closed=closed),
                    out[i].start_idx,
                    out[neighbor].end_idx,
                    global_lines=None,
                )
                out[i] = merged
                del out[neighbor]
            if len(out) <= 1:
                break
        return out

    @staticmethod
    def _rotate_segments(segments: List[SemanticSegment]) -> List[SemanticSegment]:
        if not segments:
            return segments
        start = min(range(len(segments)), key=lambda idx: (segments[idx].start_idx, segments[idx].end_idx))
        return segments[start:] + segments[:start]

    @staticmethod
    def _segment_max_error(segment: SemanticSegment) -> float:
        if len(segment.errors) == 0:
            return 0.0
        return float(np.max(segment.errors))

    def _line_fit_is_acceptable(self, points: np.ndarray, errors: np.ndarray) -> bool:
        if len(points) <= 2:
            return True
        chord = float(np.linalg.norm(points[-1] - points[0]))
        if chord < 1e-6:
            return False
        path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        max_error = float(np.max(errors))
        mean_error = float(np.mean(errors))
        line_tol = max(0.6, float(self.cfg.fit_tolerance) * 0.75)
        return path_length <= chord * 1.12 and max_error <= line_tol and mean_error <= line_tol * 0.6

    def _arc_fit_is_acceptable(self, points: np.ndarray, arc_fit: Optional[ArcFit]) -> bool:
        if arc_fit is None:
            return False
        if len(points) < 5:
            return False
        max_error = float(np.max(arc_fit.errors))
        mean_error = float(np.mean(arc_fit.errors))
        if max_error > max(0.8, float(self.cfg.fit_tolerance)):
            return False
        if mean_error > max(0.5, float(self.cfg.fit_tolerance) * 0.55):
            return False
        if self._line_fit_is_acceptable(points, self._segment_line_errors(points)):
            return False
        return abs(math.degrees(arc_fit.sweep_angle)) >= float(self.cfg.arc_min_sweep_deg)

    @staticmethod
    def _segment_direction_deg(points: np.ndarray) -> float:
        delta = points[-1] - points[0]
        return math.degrees(math.atan2(delta[1], delta[0]))

    @staticmethod
    def _angle_difference_deg(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _arc_fit_to_svg_command(self, arc_fit: ArcFit, endpoint: np.ndarray) -> str:
        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        large_arc = 1 if abs(arc_fit.sweep_angle) > math.pi else 0
        sweep_flag = 1 if arc_fit.sweep_angle >= 0 else 0
        radius = max(1e-6, arc_fit.radius)
        return f"A {fmt(radius)} {fmt(radius)} 0 {large_arc} {sweep_flag} {fmt(endpoint[0])} {fmt(endpoint[1])}"

    def _segment_to_summary(self, segment: SemanticSegment) -> Dict[str, Any]:
        """把内部 SemanticSegment 压成可写入 JSON 的简要参数。"""

        summary: Dict[str, Any] = {
            "kind": segment.kind,
            "start_idx": int(segment.start_idx),
            "end_idx": int(segment.end_idx),
            "effective_params": int(segment.effective_params),
            "max_error": round(self._segment_max_error(segment), 4),
        }
        if segment.global_line_id is not None:
            summary["global_line_id"] = int(segment.global_line_id)
        if segment.kind == "line":
            summary["start"] = np.round(segment.points[0], 4).tolist()
            summary["end"] = np.round(segment.points[-1], 4).tolist()
        elif segment.kind == "arc" and segment.arc_fit is not None:
            summary["center"] = np.round(segment.arc_fit.center, 4).tolist()
            summary["radius"] = round(float(segment.arc_fit.radius), 4)
            summary["sweep_deg"] = round(math.degrees(segment.arc_fit.sweep_angle), 4)
        elif segment.kind == "spline" and segment.bspline_fit is not None:
            summary["control_points"] = np.round(segment.bspline_fit.control_points, 4).tolist()
        return summary

    def _semantic_keypoints(self, segments: List[SemanticSegment], closed: bool) -> List[int]:
        if not segments:
            return []
        keypoints = [int(seg.start_idx) for seg in segments]
        if not closed:
            keypoints.append(int(segments[-1].end_idx))
        keypoints = sorted(set(keypoints))
        return keypoints

    def _segment_debug_record(self, segment: SemanticSegment) -> Dict[str, Any]:
        record = self._segment_to_summary(segment)
        record["sample_count"] = int(len(segment.points))
        record["mean_error"] = round(float(np.mean(segment.errors)) if len(segment.errors) else 0.0, 4)
        return record

    def _global_line_summary(self, line: GlobalLine) -> Dict[str, Any]:
        return {
            "line_id": int(line.line_id),
            "theta_deg": round(math.degrees(line.theta), 4),
            "distance": round(float(line.distance), 4),
            "support_segments": int(line.support_segments),
            "anchor": np.round(line.anchor, 4).tolist(),
        }

    def _fit_bspline_adaptive(self, points: np.ndarray, closed: bool) -> BSplineFit:
        """在不同控制点数量之间搜索合适的 B 样条拟合。

        目标不是控制点越多越好，而是在误差可接受前提下尽量少用控制点。
        """

        degree = 3
        min_ctrl = max(degree + 1, int(self.cfg.min_control_points))
        max_ctrl = max(min_ctrl, int(self.cfg.max_control_points))
        max_ctrl = min(max_ctrl, max(min_ctrl, len(points)))

        params = self._chord_parameters(points, closed=closed)
        best_fit: Optional[BSplineFit] = None
        best_score = float("inf")

        for num_ctrl in range(min_ctrl, max_ctrl + 1):
            fit = self._fit_periodic_bspline(points, params, num_ctrl) if closed else self._fit_open_bspline(points, params, num_ctrl)
            rmse = float(np.sqrt(np.mean(fit.errors**2)))
            max_error = float(np.max(fit.errors))
            score = rmse + float(self.cfg.spline_ctrl_penalty) * num_ctrl
            if max_error <= float(self.cfg.fit_tolerance) * 1.1 and score < best_score:
                best_score = score
                best_fit = fit

        if best_fit is not None:
            return best_fit

        fallback_fit: Optional[BSplineFit] = None
        fallback_score = float("inf")
        for num_ctrl in range(min_ctrl, max_ctrl + 1):
            fit = self._fit_periodic_bspline(points, params, num_ctrl) if closed else self._fit_open_bspline(points, params, num_ctrl)
            rmse = float(np.sqrt(np.mean(fit.errors**2)))
            max_error = float(np.max(fit.errors))
            score = max_error + 0.2 * rmse + float(self.cfg.spline_ctrl_penalty) * num_ctrl
            if score < fallback_score:
                fallback_score = score
                fallback_fit = fit

        assert fallback_fit is not None
        return fallback_fit

    def _fit_polygonal_centerline_path(self, points: np.ndarray) -> Optional[CenterlinePathFit]:
        if len(points) < 6:
            return None

        contour = points.astype(np.float32).reshape(-1, 1, 2)
        min_eps = max(1.0, float(self.cfg.fit_tolerance) * 0.8)
        max_eps = max(min_eps + 0.5, float(self.cfg.fit_tolerance) * 3.5)
        eps_candidates = np.linspace(min_eps, max_eps, num=8)
        best_candidate: Optional[Tuple[np.ndarray, np.ndarray]] = None
        best_score: Tuple[int, float, float] = (10**9, float("inf"), float("inf"))

        for epsilon in eps_candidates:
            approx = cv2.approxPolyDP(contour, epsilon=float(epsilon), closed=True).reshape(-1, 2)
            approx = self._merge_collinear_polygon_vertices(approx)
            if len(approx) < 3 or len(approx) > 6:
                continue

            indices = self._map_vertices_to_sample_indices(points, approx, closed=True)
            if indices is None:
                continue

            segment_errors = []
            for i in range(len(indices)):
                seg = self._slice_polyline(points, indices[i], indices[(i + 1) % len(indices)], closed=True)
                if len(seg) < 2:
                    segment_errors.append(float("inf"))
                    continue
                a = approx[i].astype(np.float64)
                b = approx[(i + 1) % len(approx)].astype(np.float64)
                segment_errors.append(float(np.max(self._point_to_line_errors(seg, a, b))))

            max_error = max(segment_errors) if segment_errors else float("inf")
            mean_error = float(np.mean(segment_errors)) if segment_errors else float("inf")
            max_error_limit, mean_error_limit = self._polygon_error_limits(len(approx))
            if max_error > max_error_limit:
                continue
            if mean_error > mean_error_limit:
                continue

            score = (len(approx), max_error, mean_error)
            if score < best_score:
                best_score = score
                best_candidate = (approx.astype(np.float64), np.array(segment_errors, dtype=np.float64))

        if best_candidate is None:
            return None
        polygon_points, segment_errors = best_candidate
        return self._build_polyline_fit(polygon_points, closed=True, errors=segment_errors)

    def _polygon_error_limits(self, vertex_count: int) -> Tuple[float, float]:
        base = float(self.cfg.fit_tolerance)
        if vertex_count <= 3:
            return (max(5.5, base * 3.8), max(3.2, base * 2.2))
        if vertex_count == 4:
            return (max(3.0, base * 2.0), max(1.9, base * 1.3))
        return (max(4.1, base * 2.1), max(2.8, base * 1.2))  # 3.1, 1.8

    def _fit_piecewise_corner_preserving_path(
        self, points: np.ndarray, corner_indices: List[int], closed: bool
    ) -> Optional[CenterlinePathFit]:
        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        if closed:
            anchors = sorted(set(int(i) for i in corner_indices))
            if len(anchors) < 3:
                return None
            start_index = anchors[0]
            cmds = [f"M {fmt(points[start_index, 0])} {fmt(points[start_index, 1])}"]
            segments = [(anchors[i], anchors[(i + 1) % len(anchors)]) for i in range(len(anchors))]
        else:
            anchors = [0] + sorted(set(int(i) for i in corner_indices)) + [len(points) - 1]
            anchors = [anchors[i] for i in range(len(anchors)) if i == 0 or anchors[i] != anchors[i - 1]]
            if len(anchors) < 2:
                return None
            cmds = [f"M {fmt(points[anchors[0], 0])} {fmt(points[anchors[0], 1])}"]
            segments = [(anchors[i], anchors[i + 1]) for i in range(len(anchors) - 1)]

        all_errors: List[np.ndarray] = []
        total_effective_params = 0
        total_line_segments = 0
        total_curve_segments = 0
        total_bspline_control_points = 0
        for seg_idx, (start_idx, end_idx) in enumerate(segments):
            seg_points = self._slice_polyline(points, start_idx, end_idx, closed=closed)
            if len(seg_points) < 2:
                continue

            if self._segment_is_nearly_linear(seg_points):
                cmds.append(f"L {fmt(seg_points[-1, 0])} {fmt(seg_points[-1, 1])}")
                all_errors.append(self._segment_line_errors(seg_points))
                total_effective_params += 1
                total_line_segments += 1
                continue

            fit = self._fit_bspline_adaptive(seg_points, closed=False)
            seg_cmds = self._open_fit_to_svg_commands(fit)
            if not seg_cmds:
                cmds.append(f"L {fmt(seg_points[-1, 0])} {fmt(seg_points[-1, 1])}")
                all_errors.append(self._segment_line_errors(seg_points))
                total_effective_params += 1
                total_line_segments += 1
                continue

            if closed and seg_idx == len(segments) - 1:
                seg_cmds = self._replace_last_segment_endpoint(seg_cmds, points[start_idx])
            cmds.extend(seg_cmds)
            all_errors.append(fit.errors)
            total_effective_params += 3 * len(seg_cmds)
            total_curve_segments += len(seg_cmds)
            total_bspline_control_points += int(len(fit.control_points))

        if closed:
            cmds.append("Z")

        errors = np.concatenate(all_errors) if all_errors else np.zeros(0, dtype=np.float64)
        return CenterlinePathFit(
            d=" ".join(cmds),
            errors=errors,
            effective_params=max(1, total_effective_params),
            line_segments=total_line_segments,
            arc_segments=0,
            curve_segments=total_curve_segments,
            bspline_control_points=total_bspline_control_points,
        )

    def _build_polyline_fit(
        self, points: np.ndarray, closed: bool, errors: Optional[np.ndarray] = None
    ) -> CenterlinePathFit:
        d = self._to_polygon_path(points) if closed else self._to_open_polyline_path(points)
        if errors is None:
            errors = self._segment_line_errors(points)
        line_segments = len(points) if closed else max(1, len(points) - 1)
        return CenterlinePathFit(
            d=d,
            errors=np.asarray(errors, dtype=np.float64),
            effective_params=max(1, line_segments),
            line_segments=line_segments,
            arc_segments=0,
            curve_segments=0,
            bspline_control_points=0,
        )

    def _chord_parameters(self, points: np.ndarray, closed: bool) -> np.ndarray:
        if len(points) < 2:
            return np.zeros(len(points), dtype=np.float64)
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cumulative[-1])
        if total < 1e-9:
            return np.zeros(len(points), dtype=np.float64)
        params = cumulative / total
        if closed:
            params = np.clip(params, 0.0, 1.0 - 1e-9)
        return params

    def _fit_periodic_bspline(self, points: np.ndarray, params: np.ndarray, num_ctrl: int) -> BSplineFit:
        u = params * num_ctrl
        basis = np.zeros((len(points), num_ctrl), dtype=np.float64)
        for row, ui in enumerate(u):
            span = int(math.floor(ui)) % num_ctrl
            t = ui - math.floor(ui)
            coeffs = self._periodic_basis(t)
            for j, coeff in enumerate(coeffs):
                basis[row, (span + j) % num_ctrl] += coeff
        control = self._solve_control_points(basis, points, closed=True)
        fitted = basis @ control
        errors = np.linalg.norm(fitted - points, axis=1)
        return BSplineFit(control_points=control, fitted_points=fitted, errors=errors, closed=True, degree=3)

    def _fit_open_bspline(self, points: np.ndarray, params: np.ndarray, num_ctrl: int) -> BSplineFit:
        degree = 3
        knots = self._open_uniform_knots(num_ctrl, degree)
        domain = float(knots[-1])
        u = params * domain
        basis = np.vstack([self._open_basis_row(ui, num_ctrl, degree, knots)[0] for ui in u])
        control = self._solve_control_points(basis, points, closed=False)
        fitted = basis @ control
        errors = np.linalg.norm(fitted - points, axis=1)
        return BSplineFit(
            control_points=control,
            fitted_points=fitted,
            errors=errors,
            closed=False,
            degree=degree,
            knots=knots,
        )

    def _solve_control_points(self, basis: np.ndarray, points: np.ndarray, closed: bool) -> np.ndarray:
        regularization = float(self.cfg.smoothness_weight)
        if regularization <= 0:
            return np.linalg.lstsq(basis, points, rcond=None)[0]

        penalty = self._second_difference_matrix(basis.shape[1], closed=closed)
        aug_a = np.vstack([basis, math.sqrt(regularization) * penalty])
        aug_b = np.vstack([points, np.zeros((penalty.shape[0], 2), dtype=np.float64)])
        return np.linalg.lstsq(aug_a, aug_b, rcond=None)[0]

    def _second_difference_matrix(self, num_ctrl: int, closed: bool) -> np.ndarray:
        if closed:
            mat = np.zeros((num_ctrl, num_ctrl), dtype=np.float64)
            for i in range(num_ctrl):
                mat[i, i] = 1.0
                mat[i, (i + 1) % num_ctrl] = -2.0
                mat[i, (i + 2) % num_ctrl] = 1.0
            return mat

        rows = max(1, num_ctrl - 2)
        mat = np.zeros((rows, num_ctrl), dtype=np.float64)
        for i in range(rows):
            mat[i, i] = 1.0
            mat[i, i + 1] = -2.0
            mat[i, i + 2] = 1.0
        return mat

    def _fit_to_svg_path(self, fit: BSplineFit) -> str:
        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        if fit.closed:
            start = self._evaluate_periodic_fit(fit.control_points, 0.0)[0]
            cmds = [f"M {fmt(start[0])} {fmt(start[1])}"]
            spans = len(fit.control_points)
            for span in range(spans):
                p0, d0 = self._evaluate_periodic_fit(fit.control_points, float(span))
                p1, d1 = self._evaluate_periodic_fit(fit.control_points, float(span + 1))
                c1 = p0 + d0 / 3.0
                c2 = p1 - d1 / 3.0
                cmds.append(
                    "C "
                    f"{fmt(c1[0])} {fmt(c1[1])} "
                    f"{fmt(c2[0])} {fmt(c2[1])} "
                    f"{fmt(p1[0])} {fmt(p1[1])}"
                )
            cmds.append("Z")
            return " ".join(cmds)

        assert fit.knots is not None
        spans = []
        degree = fit.degree
        for i in range(degree, len(fit.control_points)):
            u0 = float(fit.knots[i])
            u1 = float(fit.knots[i + 1])
            if u1 > u0 + 1e-9:
                spans.append((u0, u1))
        if not spans:
            return ""
        start = self._evaluate_open_fit(fit.control_points, fit.knots, spans[0][0])[0]
        cmds = [f"M {fmt(start[0])} {fmt(start[1])}"]
        for u0, u1 in spans:
            dt = u1 - u0
            p0, d0 = self._evaluate_open_fit(fit.control_points, fit.knots, u0)
            p1, d1 = self._evaluate_open_fit(fit.control_points, fit.knots, u1)
            c1 = p0 + d0 * dt / 3.0
            c2 = p1 - d1 * dt / 3.0
            cmds.append(
                "C "
                f"{fmt(c1[0])} {fmt(c1[1])} "
                f"{fmt(c2[0])} {fmt(c2[1])} "
                f"{fmt(p1[0])} {fmt(p1[1])}"
            )
        return " ".join(cmds)

    def _open_fit_to_svg_commands(self, fit: BSplineFit) -> List[str]:
        if fit.closed or fit.knots is None:
            return []

        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        spans = []
        degree = fit.degree
        for i in range(degree, len(fit.control_points)):
            u0 = float(fit.knots[i])
            u1 = float(fit.knots[i + 1])
            if u1 > u0 + 1e-9:
                spans.append((u0, u1))
        cmds: List[str] = []
        for u0, u1 in spans:
            dt = u1 - u0
            p0, d0 = self._evaluate_open_fit(fit.control_points, fit.knots, u0)
            p1, d1 = self._evaluate_open_fit(fit.control_points, fit.knots, u1)
            c1 = p0 + d0 * dt / 3.0
            c2 = p1 - d1 * dt / 3.0
            cmds.append(
                "C "
                f"{fmt(c1[0])} {fmt(c1[1])} "
                f"{fmt(c2[0])} {fmt(c2[1])} "
                f"{fmt(p1[0])} {fmt(p1[1])}"
            )
        return cmds

    def _fit_curve_segment_count(self, fit: BSplineFit) -> int:
        if fit.closed:
            return int(len(fit.control_points))
        if fit.knots is None:
            return 0
        degree = fit.degree
        return int(
            sum(
                1
                for i in range(degree, len(fit.control_points))
                if float(fit.knots[i + 1]) > float(fit.knots[i]) + 1e-9
            )
        )

    def _fit_effective_params(self, fit: BSplineFit) -> int:
        return 3 * self._fit_curve_segment_count(fit)

    # ------------------------------------------------------------------
    # fill/polygon 旧矢量化流程与几何工具函数
    # ------------------------------------------------------------------

    def _evaluate_periodic_fit(self, control: np.ndarray, u: float) -> Tuple[np.ndarray, np.ndarray]:
        num_ctrl = len(control)
        v = u % num_ctrl
        span = int(math.floor(v)) % num_ctrl
        t = v - math.floor(v)
        coeffs = self._periodic_basis(t)
        derivs = self._periodic_basis_derivative(t)
        point = np.zeros(2, dtype=np.float64)
        deriv = np.zeros(2, dtype=np.float64)
        for j in range(4):
            point += coeffs[j] * control[(span + j) % num_ctrl]
            deriv += derivs[j] * control[(span + j) % num_ctrl]
        return point, deriv

    def _evaluate_open_fit(
        self, control: np.ndarray, knots: np.ndarray, u: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        basis, deriv = self._open_basis_row(u, len(control), 3, knots)
        point = basis @ control
        tangent = deriv @ control
        return point, tangent

    @staticmethod
    def _periodic_basis(t: float) -> np.ndarray:
        return np.array(
            [
                ((1.0 - t) ** 3) / 6.0,
                (3.0 * t**3 - 6.0 * t**2 + 4.0) / 6.0,
                (-3.0 * t**3 + 3.0 * t**2 + 3.0 * t + 1.0) / 6.0,
                t**3 / 6.0,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _periodic_basis_derivative(t: float) -> np.ndarray:
        return np.array(
            [
                (-3.0 * t**2 + 6.0 * t - 3.0) / 6.0,
                (9.0 * t**2 - 12.0 * t) / 6.0,
                (-9.0 * t**2 + 6.0 * t + 3.0) / 6.0,
                (3.0 * t**2) / 6.0,
            ],
            dtype=np.float64,
        )

    def _open_uniform_knots(self, num_ctrl: int, degree: int) -> np.ndarray:
        interior = list(range(1, max(1, num_ctrl - degree)))
        end = float(max(1, num_ctrl - degree))
        knots = [0.0] * (degree + 1) + [float(v) for v in interior] + [end] * (degree + 1)
        return np.array(knots, dtype=np.float64)

    def _open_basis_row(
        self, u: float, num_ctrl: int, degree: int, knots: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        uu = float(np.clip(u, knots[0], knots[-1]))
        rows = []
        basis0 = np.zeros(num_ctrl, dtype=np.float64)
        for i in range(num_ctrl):
            left = knots[i]
            right = knots[i + 1]
            if (left <= uu < right) or (
                abs(uu - knots[-1]) < 1e-9 and abs(right - knots[-1]) < 1e-9 and left <= uu <= right
            ):
                basis0[i] = 1.0
        rows.append(basis0)

        for k in range(1, degree + 1):
            prev = rows[-1]
            curr = np.zeros(num_ctrl, dtype=np.float64)
            for i in range(num_ctrl):
                left_denom = knots[i + k] - knots[i]
                right_denom = knots[i + k + 1] - knots[i + 1]
                left_term = 0.0 if left_denom < 1e-12 else ((uu - knots[i]) / left_denom) * prev[i]
                right_term = 0.0
                if i + 1 < num_ctrl and right_denom >= 1e-12:
                    right_term = ((knots[i + k + 1] - uu) / right_denom) * prev[i + 1]
                curr[i] = left_term + right_term
            rows.append(curr)

        deriv = np.zeros(num_ctrl, dtype=np.float64)
        lower = rows[-2] if degree > 0 else rows[-1]
        for i in range(num_ctrl):
            left_denom = knots[i + degree] - knots[i]
            right_denom = knots[i + degree + 1] - knots[i + 1]
            left = 0.0 if left_denom < 1e-12 else degree * lower[i] / left_denom
            right = 0.0
            if i + 1 < num_ctrl and right_denom >= 1e-12:
                right = degree * lower[i + 1] / right_denom
            deriv[i] = left - right
        return rows[-1], deriv

    def _trace_mask(self, mask: np.ndarray) -> List[str]:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None or len(contours) == 0:
            return []

        hierarchy = hierarchy[0]
        all_paths: List[str] = []
        for idx, contour in enumerate(contours):
            if hierarchy[idx][3] != -1:
                continue
            if cv2.contourArea(contour) < self.cfg.filter_speckle:
                continue

            parts: List[str] = []
            outer = self._contour_to_subpath(contour)
            if outer:
                parts.append(outer)

            child = hierarchy[idx][2]
            while child != -1:
                hole = contours[child]
                if cv2.contourArea(hole) >= self.cfg.filter_speckle:
                    hole_d = self._contour_to_subpath(hole)
                    if hole_d:
                        parts.append(hole_d)
                child = hierarchy[child][0]

            if parts:
                all_paths.append(" ".join(parts))

        return all_paths

    def _contour_to_subpath(self, contour: np.ndarray) -> str:
        pts = contour[:, 0, :].astype(np.float64)
        if len(pts) < 3:
            return ""

        if self.cfg.mode == "pixel":
            proc = self._dedupe_consecutive(pts)
            corners = self._detect_corners(proc)
        else:
            proc = self._remove_staircases(pts)
            proc = self._greedy_simplify(proc, tolerance=max(0.1, float(self.cfg.simplify_tolerance)))
            corners = self._detect_corners(proc)
            if self.cfg.mode == "spline":
                proc, corners = self._corner_preserving_subdivide(proc, corners)

        if len(proc) < 3:
            return ""
        if self.cfg.mode == "spline":
            return self._to_spline_path(proc, corners)
        return self._to_polygon_path(proc)

    def _remove_staircases(self, points: np.ndarray) -> np.ndarray:
        points = self._dedupe_consecutive(points)
        if len(points) < 5:
            return points

        orient = 1.0 if self._polygon_area(points) >= 0 else -1.0
        keep = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            p_prev = points[(i - 1) % len(points)]
            p = points[i]
            p_next = points[(i + 1) % len(points)]
            v1 = p - p_prev
            v2 = p_next - p
            if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
                keep[i] = False
                continue
            manhattan = abs(v1[0]) < 1e-9 or abs(v1[1]) < 1e-9
            manhattan = manhattan and (abs(v2[0]) < 1e-9 or abs(v2[1]) < 1e-9)
            short = np.linalg.norm(v1) <= 1.5 and np.linalg.norm(v2) <= 1.5
            tri = self._cross2(v1, v2)
            if manhattan and short and tri * orient < 0:
                keep[i] = False
            if abs(tri) < 1e-9:
                keep[i] = False

        out = points[keep]
        return points if len(out) < 3 else out

    def _greedy_simplify(self, points: np.ndarray, tolerance: float) -> np.ndarray:
        points = self._dedupe_consecutive(points)
        if len(points) < 4:
            return points

        closed = np.vstack([points, points[0]])
        out = [closed[0]]
        i = 0
        last = len(closed) - 1
        while i < last:
            j = i + 2
            last_ok = i + 1
            while j <= last:
                penalty = self._subpath_penalty(closed, i, j)
                if penalty <= tolerance:
                    last_ok = j
                    j += 1
                else:
                    break
            out.append(closed[last_ok])
            i = last_ok

        out_arr = np.array(out[:-1], dtype=np.float64)
        return points if len(out_arr) < 3 else out_arr

    def _subpath_penalty(self, pts: np.ndarray, i: int, j: int) -> float:
        a = pts[i]
        c = pts[j]
        base = c - a
        norm = np.linalg.norm(base)
        if norm < 1e-9 or j <= i + 1:
            return 0.0
        mids = pts[i + 1 : j]
        if len(mids) == 0:
            return 0.0
        penalties = np.abs(base[0] * (mids[:, 1] - a[1]) - base[1] * (mids[:, 0] - a[0])) / norm
        return float(np.max(penalties))

    def _detect_corners(self, points: np.ndarray) -> np.ndarray:
        return self._detect_polyline_corners(points, closed=True)

    def _detect_polyline_corners(self, points: np.ndarray, closed: bool) -> np.ndarray:
        n = len(points)
        corners = np.zeros(n, dtype=bool)
        if n < 3:
            return corners

        max_window = min(4, max(1, n // 8))
        scores = np.zeros(n, dtype=np.float64)
        start = 0 if closed else 1
        stop = n if closed else n - 1
        for i in range(start, stop):
            best_angle = 0.0
            for offset in range(1, max_window + 1):
                prev_idx = (i - offset) % n if closed else i - offset
                next_idx = (i + offset) % n if closed else i + offset
                if not closed and (prev_idx < 0 or next_idx >= n):
                    break
                angle = abs(self._turn_angle_deg(points[prev_idx], points[i], points[next_idx]))
                best_angle = max(best_angle, angle)
            scores[i] = best_angle

        if not closed:
            scores[0] = 0.0
            scores[-1] = 0.0

        candidate_mask = scores >= float(self.cfg.corner_threshold)
        for cluster in self._cluster_true_runs(candidate_mask, closed=closed):
            if not cluster:
                continue
            best_idx = max(cluster, key=lambda idx: scores[idx])
            corners[best_idx] = True
        return corners

    def _detect_centerline_anchor_indices(self, points: np.ndarray, closed: bool) -> List[int]:
        corners = self._detect_polyline_corners(points, closed=closed)
        anchors = np.flatnonzero(corners).tolist()
        if not anchors:
            return anchors

        min_gap = max(4, int(round(max(2.0, self.cfg.resample_step) * 3.0)))
        filtered: List[int] = []
        for idx in anchors:
            if not filtered:
                filtered.append(int(idx))
                continue
            prev = filtered[-1]
            if closed:
                if min(abs(int(idx) - prev), len(points) - abs(int(idx) - prev)) < min_gap:
                    if self._corner_score(points, int(idx), closed) > self._corner_score(points, filtered[-1], closed):
                        filtered[-1] = int(idx)
                    continue
            else:
                if abs(int(idx) - filtered[-1]) < min_gap:
                    if self._corner_score(points, int(idx), closed) > self._corner_score(points, filtered[-1], closed):
                        filtered[-1] = int(idx)
                    continue
            filtered.append(int(idx))

        if closed and len(filtered) > 1:
            first = filtered[0]
            last = filtered[-1]
            cyclic_gap = min(abs(first - last), len(points) - abs(first - last))
            if cyclic_gap < min_gap:
                first_score = self._corner_score(points, first, closed)
                last_score = self._corner_score(points, last, closed)
                if last_score > first_score:
                    filtered[0] = last
                filtered.pop()
        return filtered

    def _corner_score(self, points: np.ndarray, idx: int, closed: bool) -> float:
        n = len(points)
        if n < 3:
            return 0.0
        max_window = min(4, max(1, n // 8))
        best_angle = 0.0
        for offset in range(1, max_window + 1):
            prev_idx = (idx - offset) % n if closed else idx - offset
            next_idx = (idx + offset) % n if closed else idx + offset
            if not closed and (prev_idx < 0 or next_idx >= n):
                break
            angle = abs(self._turn_angle_deg(points[prev_idx], points[idx], points[next_idx]))
            best_angle = max(best_angle, angle)
        return best_angle

    def _corner_preserving_subdivide(
        self, points: np.ndarray, corners: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pts = points.copy()
        crn = corners.copy()
        max_iter = 5
        target = max(2.0, float(self.cfg.segment_length))
        for _ in range(max_iter):
            seg_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
            if float(seg_lengths.max()) <= target:
                break

            n = len(pts)
            new_pts: List[np.ndarray] = []
            new_crn: List[bool] = []
            for i in range(n):
                im1 = (i - 1) % n
                ip1 = (i + 1) % n
                ip2 = (i + 2) % n

                p_im1 = pts[im1]
                p_i = pts[i]
                p_ip1 = pts[ip1]
                p_ip2 = pts[ip2]

                new_pts.append(p_i)
                new_crn.append(bool(crn[i]))
                q = 0.5 * (p_i + p_ip1) if crn[i] or crn[ip1] else (-p_im1 + 9.0 * p_i + 9.0 * p_ip1 - p_ip2) / 16.0
                new_pts.append(q)
                new_crn.append(False)

            pts = np.array(new_pts, dtype=np.float64)
            crn = np.array(new_crn, dtype=bool)
        return pts, crn

    def _to_polygon_path(self, points: np.ndarray) -> str:
        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        cmds = [f"M {fmt(points[0, 0])} {fmt(points[0, 1])}"]
        for x, y in points[1:]:
            cmds.append(f"L {fmt(x)} {fmt(y)}")
        cmds.append("Z")
        return " ".join(cmds)

    def _to_open_polyline_path(self, points: np.ndarray) -> str:
        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        cmds = [f"M {fmt(points[0, 0])} {fmt(points[0, 1])}"]
        for x, y in points[1:]:
            cmds.append(f"L {fmt(x)} {fmt(y)}")
        return " ".join(cmds)

    def _to_spline_path(self, points: np.ndarray, corners: np.ndarray) -> str:
        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        n = len(points)
        cmds = [f"M {fmt(points[0, 0])} {fmt(points[0, 1])}"]
        for i in range(n):
            p0 = points[(i - 1) % n]
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]

            splice_angle = abs(self._turn_angle_deg(p0, p1, p2))
            force_line = corners[i] or corners[(i + 1) % n] or splice_angle >= self.cfg.splice_threshold
            if force_line:
                cmds.append(f"L {fmt(p2[0])} {fmt(p2[1])}")
                continue

            c1 = p1 + (p2 - p0) / 6.0
            c2 = p2 - (p3 - p1) / 6.0
            cmds.append(
                "C "
                f"{fmt(c1[0])} {fmt(c1[1])} "
                f"{fmt(c2[0])} {fmt(c2[1])} "
                f"{fmt(p2[0])} {fmt(p2[1])}"
            )
        cmds.append("Z")
        return " ".join(cmds)

    @staticmethod
    def _dedupe_consecutive(points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        out = [points[0]]
        for p in points[1:]:
            if np.linalg.norm(p - out[-1]) > 1e-9:
                out.append(p)
        if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < 1e-9:
            out.pop()
        return np.array(out, dtype=np.float64)

    @staticmethod
    def _polygon_area(points: np.ndarray) -> float:
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    @staticmethod
    def _cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def _turn_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1 = a - b
        v2 = c - b
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle = math.degrees(math.acos(cosang))
        sign = np.sign(v1[0] * v2[1] - v1[1] * v2[0])
        return angle * float(sign)

    @staticmethod
    def _unsigned_turn_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1 = a - b
        v2 = c - b
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            return 180.0
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return math.degrees(math.acos(cosang))

    @staticmethod
    def _batch_unsigned_turn_angle_deg(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
        """Unsigned turn angle in degrees; pa,pb,pc are (m,2)."""
        v1 = pa - pb
        v2 = pc - pb
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        out = np.full(pa.shape[0], 180.0, dtype=np.float64)
        ok = (n1 > 1e-9) & (n2 > 1e-9)
        if not np.any(ok):
            return out
        denom = (n1[ok] * n2[ok]).clip(min=1e-12)
        cosang = np.sum(v1[ok] * v2[ok], axis=1) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        out[ok] = np.degrees(np.arccos(cosang))
        return out

    @staticmethod
    def _slice_ring_1d(values: np.ndarray, start_idx: int, end_idx: int, closed: bool) -> np.ndarray:
        """与 `_slice_polyline` 顶点顺序一致的一维环切片（用于 boundary_field 与 work_points 对齐）。"""
        if not closed:
            if end_idx < start_idx:
                return np.zeros(0, dtype=np.float64)
            return np.asarray(values[start_idx : end_idx + 1], dtype=np.float64)
        if end_idx >= start_idx:
            return np.asarray(values[start_idx : end_idx + 1], dtype=np.float64)
        return np.concatenate([values[start_idx:], values[: end_idx + 1]])

    @staticmethod
    def _slice_polyline(points: np.ndarray, start_idx: int, end_idx: int, closed: bool) -> np.ndarray:
        if not closed:
            if end_idx < start_idx:
                return np.empty((0, 2), dtype=np.float64)
            return points[start_idx : end_idx + 1]
        if end_idx >= start_idx:
            return points[start_idx : end_idx + 1]
        return np.vstack([points[start_idx:], points[: end_idx + 1]])

    def _segment_is_nearly_linear(self, points: np.ndarray) -> bool:
        if len(points) <= 2:
            return True
        chord = float(np.linalg.norm(points[-1] - points[0]))
        if chord < 2.0:
            return False
        path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if path_length > chord * 1.08:
            return False
        errors = self._segment_line_errors(points)
        max_error = float(np.max(errors))
        mean_error = float(np.mean(errors))
        line_tolerance = min(max(0.2, float(self.cfg.fit_tolerance) * 0.35), 0.75)
        return max_error <= line_tolerance and mean_error <= line_tolerance * 0.5

    @staticmethod
    def _segment_line_errors(points: np.ndarray) -> np.ndarray:
        start = points[0]
        end = points[-1]
        return VTracerPython._point_to_line_errors(points, start, end)

    @staticmethod
    def _point_to_line_errors(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        delta = end - start
        denom = float(np.dot(delta, delta))
        if denom < 1e-12:
            return np.linalg.norm(points - start, axis=1)
        t = np.clip(((points - start) @ delta) / denom, 0.0, 1.0)
        proj = start + t[:, None] * delta
        return np.linalg.norm(points - proj, axis=1)

    def _replace_last_segment_endpoint(self, commands: List[str], endpoint: np.ndarray) -> List[str]:
        if not commands:
            return commands
        p = int(max(0, self.cfg.path_precision))

        def fmt(v: float) -> str:
            return f"{v:.{p}f}" if p > 0 else str(int(round(v)))

        tail = commands[-1]
        pieces = tail.split()
        if len(pieces) < 7 or pieces[0] != "C":
            return commands[:-1] + [f"L {fmt(endpoint[0])} {fmt(endpoint[1])}"]
        pieces[-2] = fmt(endpoint[0])
        pieces[-1] = fmt(endpoint[1])
        return commands[:-1] + [" ".join(pieces)]

    def _merge_collinear_polygon_vertices(self, points: np.ndarray) -> np.ndarray:
        pts = np.array(points, dtype=np.float64)
        if len(pts) < 4:
            return pts

        changed = True
        while changed and len(pts) > 3:
            changed = False
            keep = np.ones(len(pts), dtype=bool)
            edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
            median_edge = float(np.median(edges)) if len(edges) > 0 else 0.0
            short_edge_threshold = max(4.0, median_edge * 0.18)
            for i in range(len(pts)):
                a = pts[(i - 1) % len(pts)]
                b = pts[i]
                c = pts[(i + 1) % len(pts)]
                angle = abs(self._unsigned_turn_angle_deg(a, b, c))
                prev_len = float(np.linalg.norm(b - a))
                next_len = float(np.linalg.norm(c - b))
                if angle >= 168.0:
                    keep[i] = False
                    changed = True
                    continue
                if min(prev_len, next_len) <= short_edge_threshold and angle >= 135.0:
                    keep[i] = False
                    changed = True
            if changed:
                pts = pts[keep]
        return pts

    def _map_vertices_to_sample_indices(
        self, samples: np.ndarray, vertices: np.ndarray, closed: bool
    ) -> Optional[List[int]]:
        indices: List[int] = []
        used = set()
        for vertex in vertices:
            dist = np.linalg.norm(samples - vertex, axis=1)
            idx = int(np.argmin(dist))
            if idx in used:
                return None
            used.add(idx)
            indices.append(idx)

        if not indices:
            return None

        if closed:
            order_breaks = sum(1 for i in range(1, len(indices)) if indices[i] <= indices[i - 1])
            if order_breaks > 1:
                return None
        else:
            if any(indices[i] <= indices[i - 1] for i in range(1, len(indices))):
                return None
        return indices

    @staticmethod
    def _cluster_true_runs(mask: np.ndarray, closed: bool) -> List[List[int]]:
        indices = np.flatnonzero(mask).tolist()
        if not indices:
            return []
        clusters: List[List[int]] = [[indices[0]]]
        for idx in indices[1:]:
            if idx == clusters[-1][-1] + 1:
                clusters[-1].append(idx)
            else:
                clusters.append([idx])
        if closed and len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][-1] == len(mask) - 1:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()
        return clusters


def apply_preset(args: argparse.Namespace) -> None:
    if args.trace_style == "centerline":
        args.color_mode = "bw"
        args.mode = "spline"
        args.color_count = 2
        args.filter_speckle = max(args.filter_speckle, 4)
        return

    if args.preset == "bw":
        args.color_mode = "bw"
        args.mode = "polygon"
        args.color_count = 2
        args.filter_speckle = max(args.filter_speckle, 4)
    elif args.preset == "poster":
        args.color_mode = "color"
        args.mode = "spline"
        args.color_count = min(args.color_count, 12)
        args.color_precision = min(args.color_precision, 6)
    elif args.preset == "photo":
        args.color_mode = "color"
        args.mode = "spline"
        args.color_count = max(args.color_count, 24)
        args.color_precision = max(args.color_precision, 7)
        args.filter_speckle = max(args.filter_speckle, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VTracer-style raster to vector converter in Python (local CLI project)."
    )
    parser.add_argument("-i", "--input", required=True, help="Input raster image (png/jpg/...)")
    parser.add_argument("-o", "--output", default=None, help="Output SVG path; omit to use organized outputs/")
    parser.add_argument("--sample_name", default=None, help="Sample name used under outputs/<sample_name>/")
    parser.add_argument(
        "--inputs_dir",
        default=None,
        help="Optional root directory for resolving relative input names (default: <project>/inputs)",
    )
    parser.add_argument(
        "--outputs_dir",
        default=None,
        help="Optional root directory for organized outputs (default: <project>/outputs)",
    )
    parser.add_argument("--preset", choices=["bw", "poster", "photo"], default="poster")
    parser.add_argument("--trace_style", choices=["fill", "centerline"], default="fill")
    parser.add_argument("--colormode", choices=["color", "bw"], default="color", dest="color_mode")
    parser.add_argument("-n", "--color_count", type=int, default=12, help="Target cluster count")
    parser.add_argument(
        "-p",
        "--color_precision",
        type=int,
        default=6,
        help="Significant bits in each RGB channel (1-8)",
    )
    parser.add_argument(
        "--hierarchical",
        choices=["stacked", "cutout"],
        default="stacked",
        help="Layer compositing strategy",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["pixel", "polygon", "spline"],
        default="spline",
        help="Path fitting mode",
    )
    parser.add_argument(
        "-f",
        "--filter_speckle",
        type=int,
        default=10,
        help="Discard patches smaller than X pixels",
    )
    parser.add_argument(
        "-c",
        "--corner_threshold",
        type=float,
        default=60.0,
        help="Minimum angle (degree) to detect corner",
    )
    parser.add_argument(
        "-l",
        "--segment_length",
        type=float,
        default=6.0,
        help="Subdivision target max segment length",
    )
    parser.add_argument(
        "-s",
        "--splice_threshold",
        type=float,
        default=45.0,
        help="Angle threshold (degree) to force line splice in spline mode",
    )
    parser.add_argument(
        "--simplify_tolerance",
        type=float,
        default=2.0,
        help="Penalty tolerance for greedy path simplification",
    )
    parser.add_argument("--fit_tolerance", type=float, default=1.5, help="Max fitting error in pixels")
    parser.add_argument("--min_control_points", type=int, default=6, help="Min control points for B-spline")
    parser.add_argument("--max_control_points", type=int, default=60, help="Max control points for B-spline")
    parser.add_argument("--resample_step", type=float, default=4.0, help="Centerline resample step in pixels")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0, help="Gaussian smoothing sigma for resampled centerlines")
    parser.add_argument("--semantic_window_size", type=int, default=11, help="Sliding-window size used for semantic over-segmentation")
    parser.add_argument("--keypoint_angle_threshold_deg", type=float, default=28.0, help="Minimum multi-scale turn angle deviation to seed a raw keypoint")
    parser.add_argument("--keypoint_refine_radius", type=int, default=4, help="Local search radius used to refine candidate keypoints")
    parser.add_argument("--dp_max_segment_points", type=int, default=72, help="Maximum number of candidate boundary hops considered by DP")
    parser.add_argument("--dp_complexity_weight", type=float, default=0.8, help="Complexity penalty weight used by dynamic-programming segmentation")
    parser.add_argument("--line_merge_angle_deg", type=float, default=12.0, help="Maximum angle difference for merging adjacent line segments")
    parser.add_argument("--line_cluster_eps", type=float, default=0.16, help="DBSCAN epsilon for global line clustering")
    parser.add_argument("--line_cluster_min_samples", type=int, default=2, help="Minimum supporting line segments for a global line cluster")
    parser.add_argument("--arc_radius_rel_tol", type=float, default=0.2, help="Relative radius tolerance for merging adjacent arc segments")
    parser.add_argument("--arc_center_tol", type=float, default=2.0, help="Center distance tolerance for merging adjacent arc segments")
    parser.add_argument("--arc_min_sweep_deg", type=float, default=20.0, help="Minimum sweep angle for accepting an arc segment")
    parser.add_argument("--spline_ctrl_penalty", type=float, default=0.08, help="Penalty weight on B-spline control-point count")
    parser.add_argument(
        "--smoothness_weight",
        type=float,
        default=1e-4,
        help="Regularization weight for low-parameter B-spline fitting",
    )
    parser.add_argument(
        "--skeleton_prune_length",
        type=int,
        default=8,
        help="Prune endpoint branches shorter than this many skeleton pixels",
    )
    parser.add_argument(
        "--path_precision",
        type=int,
        default=2,
        help="Decimal places in output path coordinates",
    )
    parser.add_argument(
        "--median_ksize",
        type=int,
        default=3,
        help="Median blur kernel size before tracing (odd, >=1)",
    )
    parser.add_argument("--save_intermediates", default=None, help="Directory to save binary/skeleton images")
    parser.add_argument("--metrics_path", default=None, help="Optional JSON path for fitting metrics")
    args = parser.parse_args()
    apply_preset(args)
    return args


def resolve_paths(args: argparse.Namespace) -> ResolvedPaths:
    project_root = Path(__file__).resolve().parent
    inputs_root = Path(args.inputs_dir).expanduser().resolve() if args.inputs_dir else project_root / "inputs"
    outputs_root = Path(args.outputs_dir).expanduser().resolve() if args.outputs_dir else project_root / "outputs"

    raw_input = Path(args.input).expanduser()
    input_candidates = [raw_input]
    if not raw_input.is_absolute():
        input_candidates = [Path.cwd() / raw_input, inputs_root / raw_input]
    input_path = next((candidate.resolve() for candidate in input_candidates if candidate.exists()), input_candidates[0].resolve())

    sample_name = args.sample_name or input_path.stem
    sample_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sample_name).strip("_") or "sample"

    default_stem = f"{sample_name}_centerline" if args.trace_style == "centerline" else sample_name
    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.suffix:
            output_path = output_path.with_suffix(".svg")
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        else:
            output_path = output_path.resolve()
        output_stem = output_path.stem
    else:
        output_path = (outputs_root / sample_name / "final" / f"{default_stem}.svg").resolve()
        output_stem = default_stem

    metrics_path = args.metrics_path
    if metrics_path:
        metrics_path = str(Path(metrics_path).expanduser().resolve())
    elif args.trace_style == "centerline":
        metrics_path = str((outputs_root / sample_name / "metrics" / f"{output_stem}_metrics.json").resolve())

    save_intermediates = args.save_intermediates
    if save_intermediates:
        save_intermediates = str(Path(save_intermediates).expanduser().resolve())
    elif args.trace_style == "centerline":
        save_intermediates = str((outputs_root / sample_name / "intermediates" / output_stem).resolve())

    return ResolvedPaths(
        input_path=str(input_path),
        output_path=str(output_path),
        sample_name=sample_name,
        metrics_path=metrics_path,
        save_intermediates=save_intermediates,
    )


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    os.makedirs(os.path.dirname(paths.output_path) or ".", exist_ok=True)
    cfg = TraceConfig(
        image_path=paths.input_path,
        color_mode=args.color_mode,
        color_count=args.color_count,
        color_precision=args.color_precision,
        hierarchical=args.hierarchical,
        mode=args.mode,
        trace_style=args.trace_style,
        corner_threshold=args.corner_threshold,
        segment_length=args.segment_length,
        splice_threshold=args.splice_threshold,
        simplify_tolerance=args.simplify_tolerance,
        path_precision=args.path_precision,
        filter_speckle=args.filter_speckle,
        median_ksize=args.median_ksize,
        fit_tolerance=args.fit_tolerance,
        min_control_points=args.min_control_points,
        max_control_points=args.max_control_points,
        resample_step=args.resample_step,
        gaussian_sigma=args.gaussian_sigma,
        semantic_window_size=args.semantic_window_size,
        keypoint_angle_threshold_deg=args.keypoint_angle_threshold_deg,
        keypoint_refine_radius=args.keypoint_refine_radius,
        dp_max_segment_points=args.dp_max_segment_points,
        dp_complexity_weight=args.dp_complexity_weight,
        line_merge_angle_deg=args.line_merge_angle_deg,
        line_cluster_eps=args.line_cluster_eps,
        line_cluster_min_samples=args.line_cluster_min_samples,
        arc_radius_rel_tol=args.arc_radius_rel_tol,
        arc_center_tol=args.arc_center_tol,
        arc_min_sweep_deg=args.arc_min_sweep_deg,
        spline_ctrl_penalty=args.spline_ctrl_penalty,
        smoothness_weight=args.smoothness_weight,
        skeleton_prune_length=args.skeleton_prune_length,
        save_intermediates=paths.save_intermediates,
        metrics_path=paths.metrics_path,
    )
    tracer = VTracerPython(cfg)
    tracer.to_svg(paths.output_path)


if __name__ == "__main__":
    main()
