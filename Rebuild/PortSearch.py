from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from .port_topology_detector import (
        PatchPortCandidate,
        PatchPortDetectionResult,
        PatchPortTopologyDetector,
    )
except ImportError:  # pragma: no cover - keeps legacy direct PortSearch imports working.
    from port_topology_detector import (
        PatchPortCandidate,
        PatchPortDetectionResult,
        PatchPortTopologyDetector,
    )


# 颜色字典：键为颜色名，值为 HSV 区间列表。
# 之所以用“列表”而不是单个区间，是因为有些颜色在 HSV 中需要多个区间表示，例如 red。
HSV_COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    'black': [((0, 0, 0), (180, 255, 30))],
    'gray':[((0, 0, 46), (180, 43, 220))],
    'white':[((0, 0, 221), (180, 30, 255))],
    'red': [
        ((0, 43, 46), (10, 255, 255)),  # 红色范围1（低H值）
        ((156, 43, 46), (180, 255, 255))  # 红色范围2（高H值）
    ],
    'orange': [((11, 43, 46), (25, 255, 255))],
    'yellow': [((26, 43, 46), (34, 255, 255))],
    'green': [((35, 43, 46), (77, 255, 255))],
    'cyan':[((78, 43, 46), (99, 255, 255))],
    'blue': [((100, 43, 46), (124, 255, 255))],
    'purple': [((125, 43, 46), (155, 255, 255))],
}


@dataclass(frozen=True)
class BorderRelation:
    """描述一条主体边与图片边界之间的几何关系。"""

    # distance：该边到某个图片边界的最短距离
    distance: float
    # mean_distance：该边整体到某个图片边界的平均距离
    mean_distance: float
    # closest_sides：与该边最近的图片边界方向，如 left/right/top/bottom
    closest_sides: tuple[str, ...]
    # overlap_length：该边与图片边界重合的长度，若大于 0 说明边直接贴在图像边缘上
    overlap_length: float
    # intersects_border：该边是否与图片边界有接触
    intersects_border: bool
    # lies_on_border：该边是否与图片边界存在正长度重合
    lies_on_border: bool


@dataclass(frozen=True)
class SubjectComponent:
    """描述最终被选为主体的连通域。"""

    # label：连通域标签编号，由 connectedComponentsWithStats 返回
    label: int
    # area：连通域面积，即白色像素数
    area: int
    # bbox：连通域外接矩形，格式为 (x, y, w, h)
    # x, y：左上角坐标；w, h：宽和高
    bbox: tuple[int, int, int, int]
    # centroid：连通域质心坐标，格式为 (cx, cy)
    centroid: tuple[float, float]


@dataclass(frozen=True)
class SubjectEdgeResult:
    """封装一次完整检测的结果和中间产物。"""

    # subject_color：本次检测指定的主体颜色名
    subject_color: str
    # original_bgr / original_rgb：原始图像，分别用于 OpenCV 和 Matplotlib
    original_bgr: np.ndarray
    original_rgb: np.ndarray
    # border_removed_bgr / border_removed_rgb：去除外框后的图像
    border_removed_bgr: np.ndarray
    border_removed_rgb: np.ndarray
    # border_mask：被识别为“边框/贴边细线”的区域掩膜，白色部分表示被删除区域
    border_mask: np.ndarray
    # grayscale：去边框后的灰度图，主要用于调试观察
    grayscale: np.ndarray
    # foreground_mask：按颜色提取后的前景掩膜，白色部分就是候选主体
    foreground_mask: np.ndarray
    # subject_mask：从前景掩膜里最终筛选出来的主体连通域
    subject_mask: np.ndarray
    # subject_component：主体连通域的面积、包围框等统计信息
    subject_component: SubjectComponent
    # subject_contour：主体外轮廓的原始点集
    subject_contour: np.ndarray
    # subject_polygon：经过多边形近似后的主体轮廓点集
    subject_polygon: np.ndarray
    # closest_edge_index：最近边在 subject_polygon 边集中的编号
    closest_edge_index: int
    # closest_edge：最近边两个端点的坐标，格式为 [[x1, y1], [x2, y2]]
    closest_edge: np.ndarray
    # closest_edge_length：最近边长度
    closest_edge_length: float
    # distance_to_image_border：最近边到图片边界的最短距离
    distance_to_image_border: float
    # mean_distance_to_image_border：最近边整体到图片边界的平均距离
    mean_distance_to_image_border: float
    # closest_border_sides：最近边对应的图片边界方向
    closest_border_sides: tuple[str, ...]
    # border_overlap_length：最近边与图片边界重合的长度
    border_overlap_length: float
    # border_contact_mode：边界接触模式，取值为 overlap / touch / separate
    border_contact_mode: str


class SubjectEdgeAnalyzer:
    """主体最近边分析器。

    用法：
    1. 创建实例，配置主体筛选、轮廓近似、去边框等参数
    2. 调用 analyze(image, subject_color=...)
    3. 如需可视化，调用 visualize(result)
    """

    def __init__(
        self,
        *,
        threshold_value: Optional[int] = None,
        blur_ksize: int = 5,
        min_component_area: int = 200,
        approx_epsilon_ratio: float = 0.0025,
        border_dark_threshold: int = 160,
        border_margin_ratio: float = 0.04,
    ) -> None:
        # threshold_value：保留的固定阈值参数，当前主体检测主流程以颜色掩膜为主
        self.threshold_value = threshold_value
        # blur_ksize：高斯模糊核大小，用于平滑颜色/灰度波动
        self.blur_ksize = blur_ksize
        # min_component_area：主体最小面积，小于该面积的连通域直接忽略
        self.min_component_area = min_component_area
        # approx_epsilon_ratio：轮廓近似比例，值越大，多边形边越少
        self.approx_epsilon_ratio = approx_epsilon_ratio
        # border_dark_threshold：去边框时，“多暗算深色边框”的阈值
        self.border_dark_threshold = border_dark_threshold
        # border_margin_ratio：判定一个大矩形是否贴近整张图四周的边距比例
        self.border_margin_ratio = border_margin_ratio

    def analyze(
        self,
        image: str | Path | np.ndarray,
        *,
        subject_color: str = "gray",
    ) -> SubjectEdgeResult:
        """执行一次完整检测。

        参数说明：
        - image：输入图像，可以是路径，也可以是 ndarray
        - subject_color：主体颜色名，决定 Foreground Mask 中哪些像素会被置为 255
        """

        # original_bgr：OpenCV 读取后的原图
        original_bgr = self._load_image(image)
        # original_rgb：为了 Matplotlib 显示而转换出的 RGB 图
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        # border_removed_bgr：去掉图片外框/贴边细线后的图
        # border_mask：被识别为边框的区域掩膜
        border_removed_bgr, border_mask = self._remove_image_border(original_bgr)
        border_removed_rgb = cv2.cvtColor(border_removed_bgr, cv2.COLOR_BGR2RGB)
        # grayscale：去边框后的灰度图，主要用于调试
        grayscale = cv2.cvtColor(border_removed_bgr, cv2.COLOR_BGR2GRAY)
        # foreground_mask：按 subject_color 提取出的前景掩膜
        foreground_mask = self._build_foreground_mask(border_removed_bgr, subject_color=subject_color)
        # subject_mask：最终选中的主体连通域掩膜
        # component：主体连通域统计结果
        try:
            subject_mask, component = self._extract_subject_mask(foreground_mask)
            resolved_subject_color = subject_color
        except ValueError:
            if subject_color.lower() == "auto":
                raise
            foreground_mask = self._build_auto_subject_mask(border_removed_bgr)
            subject_mask, component = self._extract_subject_mask(foreground_mask)
            resolved_subject_color = f"{subject_color}->auto"
        # subject_contour：主体外轮廓原始点集
        subject_contour = self._extract_subject_contour(subject_mask)
        # subject_polygon：主体轮廓多边形近似结果，后续按“边”来计算距离
        subject_polygon = self._approximate_contour(subject_contour)
        # edge_index：最近边编号
        # edge：最近边两个端点
        # relation：最近边与图片边界的几何关系
        edge_index, edge, relation = self._find_nearest_edge(subject_polygon, subject_mask.shape)

        return SubjectEdgeResult(
            subject_color=resolved_subject_color,
            original_bgr=original_bgr,
            original_rgb=original_rgb,
            border_removed_bgr=border_removed_bgr,
            border_removed_rgb=border_removed_rgb,
            border_mask=border_mask,
            grayscale=grayscale,
            foreground_mask=foreground_mask,
            subject_mask=subject_mask,
            subject_component=component,
            subject_contour=subject_contour,
            subject_polygon=subject_polygon,
            closest_edge_index=edge_index,
            closest_edge=edge,
            closest_edge_length=float(np.linalg.norm(edge[1] - edge[0])),
            distance_to_image_border=relation.distance,
            mean_distance_to_image_border=relation.mean_distance,
            closest_border_sides=relation.closest_sides,
            border_overlap_length=relation.overlap_length,
            border_contact_mode=self._border_contact_mode(relation),
        )

    def visualize(
        self,
        result: SubjectEdgeResult,
        *,
        figsize: tuple[float, float] = (14.0, 10.0),
        save_path: Optional[str | Path] = None,
        show: bool = False,
    ):
        """把中间结果和最终结果画成 2x2 图。

        4 张子图分别是：
        - 原图
        - 去边框后的图
        - 前景掩膜
        - 最终最近边高亮结果
        """

        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

        axes[0, 0].imshow(result.original_rgb)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(result.border_removed_rgb)
        axes[0, 1].set_title("Border Removed")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(result.foreground_mask, cmap="gray")
        axes[1, 0].set_title("Foreground Mask")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(result.original_rgb)
        axes[1, 1].set_title("Nearest Edge To Image Border")
        self._plot_analysis_overlay(axes[1, 1], result)
        axes[1, 1].axis("off")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, axes

    def detect_patch_ports(
        self,
        result: SubjectEdgeResult | np.ndarray,
        *,
        use_foreground_mask: bool = False,
        border_distance_px: int = 8,
        debug_dir: Optional[str | Path] = None,
    ) -> PatchPortDetectionResult:
        """Optional patch-antenna port detector built on conductor topology.

        旧的 analyze()/visualize() 行为保持不变；这个方法只作为新增阶段，
        供后续 CST 自动端口、馈电方向推断等流程按需调用。
        """

        detector = PatchPortTopologyDetector(
            border_distance_px=border_distance_px,
            min_component_area=self.min_component_area,
        )

        if isinstance(result, SubjectEdgeResult):
            valid_region_mask = self._build_valid_port_region_mask(result)
            # 默认使用 subject_mask，避免 foreground 中残留标注/边框把骨架端点污染。
            # 调试时可以显式切换到 foreground_mask，观察清洗前后的端点差异。
            port_result = detector.detect_ports(
                subject_mask=None if use_foreground_mask else result.subject_mask,
                foreground_mask=result.foreground_mask if use_foreground_mask else None,
                original_image=result.original_bgr,
                valid_region_mask=valid_region_mask,
                debug_dir=debug_dir,
            )
            if port_result.ports or use_foreground_mask:
                return port_result

            conductor_mask = self._build_patch_conductor_candidate_mask(result.border_removed_bgr)
            if conductor_mask is None:
                return port_result

            fallback_debug_dir = Path(debug_dir) if debug_dir is not None else None
            fallback_result = detector.detect_ports(
                subject_mask=conductor_mask,
                foreground_mask=None,
                original_image=result.original_bgr,
                valid_region_mask=valid_region_mask,
                debug_dir=fallback_debug_dir,
            )
            if fallback_debug_dir is not None:
                cv2.imwrite(str(fallback_debug_dir / "conductor_candidate_mask.png"), conductor_mask)
            fallback_result.debug_metadata["fallback_from_subject_mask"] = True
            fallback_result.debug_metadata["fallback_reason"] = "subject_mask produced no patch port candidates"
            fallback_result.debug_metadata["primary_attempt"] = port_result.debug_metadata
            return fallback_result

        return detector.detect_ports(
            subject_mask=result,
            foreground_mask=None,
            original_image=None,
            debug_dir=debug_dir,
        )

    def _build_valid_port_region_mask(self, result: SubjectEdgeResult) -> Optional[np.ndarray]:
        """Return the first-layer design contour mask when it is clearly substrate-like.

        端口检测的金属 mask 有时来自 fallback conductor mask，但截图边缘并不等于介质板边缘。
        因此使用 analyze() 已经提取到的第一层大轮廓作为有效端口区域，
        让后续 border 判断和 port plane 外偏移都限制在该轮廓内部。
        """

        height, width = result.subject_mask.shape
        x, y, w, h = result.subject_component.bbox
        bbox_area_ratio = float((w * h) / max(1, height * width))
        margin_x = max(6, int(round(width * 0.12)))
        margin_y = max(6, int(round(height * 0.12)))
        near_outer_canvas = (
            x <= margin_x
            and y <= margin_y
            and width - (x + w) <= margin_x
            and height - (y + h) <= margin_y
        )
        if bbox_area_ratio < 0.35 or not near_outer_canvas:
            return None

        contour = result.subject_contour
        if contour is None or len(contour) < 3:
            return None
        region = self._fill_subject_region_holes(result.subject_mask)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            region = np.zeros_like(result.subject_mask)
            cv2.drawContours(region, [hull], -1, 255, thickness=-1)
        return region

    @staticmethod
    def _fill_subject_region_holes(mask: np.ndarray) -> np.ndarray:
        region = np.where(mask > 0, 255, 0).astype(np.uint8)
        flood = region.copy()
        height, width = flood.shape
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        return cv2.bitwise_or(region, holes)

    def _build_patch_conductor_candidate_mask(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Build a low-saturation conductor mask for patch screenshots.

        有些天线截图中 PEC 被画成米黄色/浅灰色，而背景基底是高饱和青色。
        当 col_mats 的颜色信息与截图不一致时，旧 subject_mask 可能选中基底。
        这里仅作为端口检测 fallback，避免影响原有主体边界/参数化流程。
        """

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        saturation = hsv[:, :, 1]

        # 低饱和且非白的区域通常对应米黄色/灰色金属；高饱和的青色基底被排除。
        mask = np.where((saturation <= 90) & (grayscale >= 20) & (grayscale < 252), 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        height, width = mask.shape
        candidates: list[tuple[int, int]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            fill_ratio = area / max(1, w * h)
            looks_like_page = w >= 0.9 * width and h >= 0.9 * height and fill_ratio < 0.4
            if not looks_like_page:
                candidates.append((area, label))

        if not candidates:
            return None

        _, selected_label = max(candidates, key=lambda item: item[0])
        return np.where(labels == selected_label, 255, 0).astype(np.uint8)

    def _remove_image_border(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """去除图片边框和贴边细线。

        返回：
        - cleaned：去边框后的图像
        - border_mask：被识别为边框的掩膜
        """

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # dark_mask：深色像素掩膜，主要用来定位黑色或深灰色边框
        dark_mask = np.where(grayscale <= self.border_dark_threshold, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

        # border_mask：最终需要擦除的区域
        border_mask = np.zeros_like(dark_mask)
        self._mark_outer_frame_mask(dark_mask, border_mask)
        self._mark_thin_border_components(dark_mask, border_mask)

        # cleaned：把边框区域直接涂白后的图像
        cleaned = image.copy()
        cleaned[border_mask > 0] = 255
        return cleaned, border_mask

    def _load_image(self, image: str | Path | np.ndarray) -> np.ndarray:
        """统一处理输入图像类型，返回 BGR 三通道图。"""

        if isinstance(image, np.ndarray):
            array = image.copy()
            if array.ndim == 2:
                return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
            if array.ndim == 3 and array.shape[2] == 3:
                return array
            raise ValueError("Input ndarray must be grayscale or 3-channel BGR.")

        image_path = Path(image)
        loaded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return loaded

    def _blur(self, grayscale: np.ndarray) -> np.ndarray:
        """对灰度图做高斯模糊，减小噪声。"""

        if self.blur_ksize <= 1:
            return grayscale
        ksize = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
        return cv2.GaussianBlur(grayscale, (ksize, ksize), 0)

    def _build_foreground_mask(
        self,
        image_bgr: np.ndarray,
        *,
        subject_color: str,
    ) -> np.ndarray:
        """按指定颜色构建前景掩膜。

        掩膜中：
        - 255 表示“属于主体颜色”
        - 0 表示“非主体颜色”
        """

        mask = self._build_color_mask(image_bgr, subject_color=subject_color)
        kernel = np.ones((3, 3), dtype=np.uint8)
        # 开运算：去掉小白点噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # 闭运算：补齐小裂缝，让主体更连贯
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _build_auto_subject_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Build a foreground mask when the configured PEC color is stale.

        This is mainly for direct/raw-image mode.  FSS repair normalizes PEC to
        the color named in col_mats, but raw clean images can use cyan/yellow
        or another display color while col_mats still says gray=PEC.
        """

        best_mask: Optional[np.ndarray] = None
        best_score = -1
        for color_name in HSV_COLOR_RANGES:
            if color_name == "white":
                continue
            mask = self._build_color_mask(image_bgr, subject_color=color_name)
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            score = self._largest_valid_component_area(mask)
            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is not None and best_score >= self.min_component_area:
            return best_mask

        return self._build_non_background_mask(image_bgr)

    def _build_non_background_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import numpy as np

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        background = np.median(pixels, axis=0)
        distance = np.linalg.norm(lab.astype(np.float32) - background.reshape(1, 1, 3), axis=2)
        threshold = max(12.0, float(np.percentile(distance, 65)))
        mask = np.where(distance > threshold, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _largest_valid_component_area(self, mask: np.ndarray) -> int:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self.min_component_area:
                best = max(best, area)
        return best

    def _build_color_mask(
        self,
        image_bgr: np.ndarray,
        *,
        subject_color: str,
    ) -> np.ndarray:
        """根据 HSV 颜色字典生成颜色掩膜。"""

        color_key = subject_color.lower()
        if color_key not in HSV_COLOR_RANGES:
            available = ", ".join(sorted(HSV_COLOR_RANGES))
            raise ValueError(f"Unsupported subject_color '{subject_color}'. Available colors: {available}")

        # hsv：把 BGR 图转到 HSV 色彩空间，便于按颜色范围做筛选
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        blurred_hsv = self._blur_hsv(hsv)
        # mask：最终输出的颜色掩膜
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        # lower / upper：颜色区间上下界
        for lower, upper in HSV_COLOR_RANGES[color_key]:
            lower_bound = np.array(lower, dtype=np.uint8)
            upper_bound = np.array(upper, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(blurred_hsv, lower_bound, upper_bound))
        return mask

    def _blur_hsv(self, hsv: np.ndarray) -> np.ndarray:
        """对 HSV 图做高斯模糊，减小颜色抖动。"""

        if self.blur_ksize <= 1:
            return hsv
        ksize = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
        return cv2.GaussianBlur(hsv, (ksize, ksize), 0)

    def _extract_subject_mask(self, foreground_mask: np.ndarray) -> tuple[np.ndarray, SubjectComponent]:
        """从 Foreground Mask 中选出最终主体连通域。"""

        # num_labels：连通域总数（包含背景）
        # labels：每个像素所属的连通域标签图
        # stats：每个连通域的统计信息，如面积、外接框
        # centroids：每个连通域的质心
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_mask, connectivity=8)
        height, width = foreground_mask.shape

        # candidates：候选主体列表，元素为 (score, component)
        # score 越大，越可能是真正主体
        candidates: list[tuple[float, SubjectComponent]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue

            # x, y, w, h：连通域外接矩形
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            # fill_ratio：连通域在外接矩形中的填充率，越低越像细框或边框
            fill_ratio = area / max(1, w * h)
            # looks_like_frame：该连通域是否像“铺满整图边缘的大外框”
            looks_like_frame = w >= 0.9 * width and h >= 0.9 * height and fill_ratio < 0.15
            # score：主体评分，若像外框则降低其分数
            score = area * (0.2 if looks_like_frame else 1.0)
            component = SubjectComponent(
                label=label,
                area=area,
                bbox=(x, y, w, h),
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
            )
            candidates.append((score, component))

        if not candidates:
            raise ValueError("No valid subject component was found in the image.")

        _, selected = max(candidates, key=lambda item: item[0])
        subject_mask = np.where(labels == selected.label, 255, 0).astype(np.uint8)
        return subject_mask, selected

    def _mark_outer_frame_mask(self, dark_mask: np.ndarray, border_mask: np.ndarray) -> None:
        """在深色掩膜中寻找“贴近整张图四周的大矩形外框”。"""

        height, width = dark_mask.shape
        image_area = float(height * width)
        # margin_x / margin_y：允许外框离图像边缘的最大偏移量
        margin_x = max(3, int(round(width * self.border_margin_ratio)))
        margin_y = max(3, int(round(height * self.border_margin_ratio)))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # best_contour：最像外层矩形边框的轮廓
        best_contour: Optional[np.ndarray] = None
        best_area = -1.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 0.75 * image_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            near_edges = x <= margin_x and y <= margin_y and (width - (x + w)) <= margin_x and (height - (y + h)) <= margin_y
            if not near_edges:
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue

            if area > best_area:
                best_area = area
                best_contour = contour

        if best_contour is None:
            return

        thickness = max(5, int(round(min(height, width) * 0.012)))
        cv2.drawContours(border_mask, [best_contour], -1, 255, thickness=thickness)

    def _mark_thin_border_components(self, dark_mask: np.ndarray, border_mask: np.ndarray) -> None:
        """删除贴边的细线、细条等伪主体。"""

        height, width = dark_mask.shape
        image_area = float(height * width)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

        # max_thin_width / max_thin_height：多细才算“贴边细线”
        max_thin_width = max(4, int(round(width * 0.03)))
        max_thin_height = max(4, int(round(height * 0.03)))
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
            is_thin = w <= max_thin_width or h <= max_thin_height
            is_small = area <= 0.08 * image_area
            if touches_border and is_thin and is_small:
                border_mask[labels == label] = 255

    def _extract_subject_contour(self, subject_mask: np.ndarray) -> np.ndarray:
        """提取主体最外层轮廓。"""

        contours, _ = cv2.findContours(subject_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("No external contour was found for the selected subject.")
        return max(contours, key=cv2.contourArea)

    def _approximate_contour(self, contour: np.ndarray) -> np.ndarray:
        """将主体轮廓近似成多边形，便于按“边”进行几何计算。"""

        if self.approx_epsilon_ratio <= 0:
            return contour.reshape(-1, 2).astype(float)

        epsilon = self.approx_epsilon_ratio * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True)
        if len(polygon) < 2:
            polygon = contour
        return polygon.reshape(-1, 2).astype(float)

    def _find_nearest_edge(
        self,
        polygon: np.ndarray,
        image_shape: tuple[int, int],
    ) -> tuple[int, np.ndarray, BorderRelation]:
        """在主体多边形的所有边中，找到离图片边缘最近的一条边。"""

        if len(polygon) < 2:
            raise ValueError("Subject polygon does not contain enough points to build edges.")

        best_index = -1
        # best_edge：当前最优边的两个端点
        best_edge: Optional[np.ndarray] = None
        # best_relation：当前最优边与图片边界的关系
        best_relation: Optional[BorderRelation] = None
        # best_score：当前最优边的排序分数
        best_score: Optional[tuple[float, ...]] = None

        for edge_index in range(len(polygon)):
            # start / end：当前边的两个端点
            start = polygon[edge_index]
            end = polygon[(edge_index + 1) % len(polygon)]
            relation = self._edge_border_relation(start, end, image_shape)
            edge_length = float(np.linalg.norm(end - start))
            # score 的排序含义：
            # 1. 优先选与边界重合的边
            # 2. 再看平均距离
            # 3. 再看最短距离
            # 4. 再看边长，越长越优先
            score = (
                0.0 if relation.lies_on_border else 1.0,
                relation.mean_distance,
                relation.distance,
                -edge_length,
            )

            if best_score is None or score < best_score:
                best_index = edge_index
                best_edge = np.vstack((start, end))
                best_relation = relation
                best_score = score

        if best_edge is None or best_relation is None:
            raise RuntimeError("Failed to locate the nearest subject edge.")

        return best_index, best_edge, best_relation

    def _edge_border_relation(
        self,
        start: np.ndarray,
        end: np.ndarray,
        image_shape: tuple[int, int],
    ) -> BorderRelation:
        """计算一条主体边与四个图片边界的关系。"""

        height, width = image_shape
        tolerance = 1e-6
        # border_segments：四条图片边界，每条边界都表示成一个线段
        border_segments = {
            "left": (np.array([0.0, 0.0]), np.array([0.0, float(height - 1)])),
            "right": (
                np.array([float(width - 1), 0.0]),
                np.array([float(width - 1), float(height - 1)]),
            ),
            "top": (np.array([0.0, 0.0]), np.array([float(width - 1), 0.0])),
            "bottom": (
                np.array([0.0, float(height - 1)]),
                np.array([float(width - 1), float(height - 1)]),
            ),
        }

        min_distance = float("inf")
        min_mean_distance = float("inf")
        closest_sides: list[str] = []
        # max_overlap：该边与某个图片边界的最大重合长度
        max_overlap = 0.0
        overlap_sides: list[str] = []

        for side, (border_start, border_end) in border_segments.items():
            distance = self._segment_to_segment_distance(start, end, border_start, border_end)
            mean_distance = self._segment_mean_distance_to_border_side(start, end, side, image_shape)
            if mean_distance < min_mean_distance - tolerance:
                min_mean_distance = mean_distance
                min_distance = distance
                closest_sides = [side]
            elif abs(mean_distance - min_mean_distance) <= tolerance:
                if distance < min_distance - tolerance:
                    min_distance = distance
                    closest_sides = [side]
                elif abs(distance - min_distance) <= tolerance and side not in closest_sides:
                    closest_sides.append(side)

            overlap_length = self._segment_overlap_with_border(start, end, side, image_shape)
            if overlap_length > max_overlap + tolerance:
                max_overlap = overlap_length
                overlap_sides = [side]
            elif abs(overlap_length - max_overlap) <= tolerance and overlap_length > tolerance and side not in overlap_sides:
                overlap_sides.append(side)

        if max_overlap > tolerance:
            closest_sides = overlap_sides
            min_distance = 0.0
            min_mean_distance = 0.0

        return BorderRelation(
            distance=min_distance,
            mean_distance=min_mean_distance,
            closest_sides=tuple(closest_sides),
            overlap_length=max_overlap,
            intersects_border=min_distance <= tolerance,
            lies_on_border=max_overlap > tolerance,
        )

    def _segment_mean_distance_to_border_side(
        self,
        start: np.ndarray,
        end: np.ndarray,
        side: str,
        image_shape: tuple[int, int],
    ) -> float:
        """计算一条边到指定图片边界的平均距离。"""

        height, width = image_shape
        if side == "left":
            return float((start[0] + end[0]) / 2.0)
        if side == "right":
            return float(((width - 1 - start[0]) + (width - 1 - end[0])) / 2.0)
        if side == "top":
            return float((start[1] + end[1]) / 2.0)
        if side == "bottom":
            return float(((height - 1 - start[1]) + (height - 1 - end[1])) / 2.0)
        raise ValueError(f"Unsupported border side: {side}")

    def _segment_overlap_with_border(
        self,
        start: np.ndarray,
        end: np.ndarray,
        side: str,
        image_shape: tuple[int, int],
    ) -> float:
        """计算一条边与指定图片边界的重合长度。"""

        height, width = image_shape
        tolerance = 1e-6

        if side == "left" and abs(start[0]) <= tolerance and abs(end[0]) <= tolerance:
            return abs(float(end[1] - start[1]))
        if side == "right" and abs(start[0] - (width - 1)) <= tolerance and abs(end[0] - (width - 1)) <= tolerance:
            return abs(float(end[1] - start[1]))
        if side == "top" and abs(start[1]) <= tolerance and abs(end[1]) <= tolerance:
            return abs(float(end[0] - start[0]))
        if side == "bottom" and abs(start[1] - (height - 1)) <= tolerance and abs(end[1] - (height - 1)) <= tolerance:
            return abs(float(end[0] - start[0]))
        return 0.0

    def _segment_to_segment_distance(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
    ) -> float:
        """计算两条线段之间的最短距离。"""

        if self._segments_intersect(a1, a2, b1, b2):
            return 0.0

        return min(
            self._point_to_segment_distance(a1, b1, b2),
            self._point_to_segment_distance(a2, b1, b2),
            self._point_to_segment_distance(b1, a1, a2),
            self._point_to_segment_distance(b2, a1, a2),
        )

    def _point_to_segment_distance(
        self,
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        """计算点到线段的最短距离。"""

        segment = end - start
        segment_norm = float(np.dot(segment, segment))
        if segment_norm <= 1e-12:
            return float(np.linalg.norm(point - start))

        ratio = float(np.dot(point - start, segment) / segment_norm)
        ratio = max(0.0, min(1.0, ratio))
        projection = start + ratio * segment
        return float(np.linalg.norm(point - projection))

    def _segments_intersect(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        b1: np.ndarray,
        b2: np.ndarray,
    ) -> bool:
        """判断两条线段是否相交。"""

        tolerance = 1e-6

        def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
            return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

        def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
            return (
                min(p[0], r[0]) - tolerance <= q[0] <= max(p[0], r[0]) + tolerance
                and min(p[1], r[1]) - tolerance <= q[1] <= max(p[1], r[1]) + tolerance
            )

        o1 = orientation(a1, a2, b1)
        o2 = orientation(a1, a2, b2)
        o3 = orientation(b1, b2, a1)
        o4 = orientation(b1, b2, a2)

        opposite_a = (o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance)
        opposite_b = (o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance)
        if opposite_a and opposite_b:
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

    def _border_contact_mode(self, relation: BorderRelation) -> str:
        """根据几何关系给最近边的接触模式命名。"""

        if relation.lies_on_border:
            return "overlap"
        if relation.intersects_border:
            return "touch"
        return "separate"

    def _plot_analysis_overlay(self, ax, result: SubjectEdgeResult) -> None:
        """在原图上叠加主体轮廓、最近边和最近边界。"""

        contour = result.subject_polygon
        closed_contour = np.vstack((contour, contour[0]))
        edge = result.closest_edge

        ax.contour(result.subject_mask, levels=[127], colors=["#00bcd4"], linewidths=2.0)
        ax.plot(closed_contour[:, 0], closed_contour[:, 1], color="#00bcd4", linewidth=2.0, label="subject contour")
        ax.plot(edge[:, 0], edge[:, 1], color="#ff1744", linewidth=4.0, label="nearest edge")
        ax.scatter(edge[:, 0], edge[:, 1], color="#ffeb3b", edgecolors="black", linewidths=0.8, s=70, zorder=5)

        for side in result.closest_border_sides:
            self._plot_border_side(ax, side, result.subject_mask.shape)

        x, y, w, h = result.subject_component.bbox
        ax.text(
            x,
            max(0, y - 12),
            "dist={:.2f} mode={} side={}".format(
                result.distance_to_image_border,
                result.border_contact_mode,
                ",".join(result.closest_border_sides) or "none",
            ),
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.0},
        )
        ax.legend(loc="lower right", fontsize=8)

    def _plot_border_side(self, ax, side: str, image_shape: tuple[int, int]) -> None:
        """把最近的图片边界画成虚线，用于可视化。"""

        height, width = image_shape
        lines = {
            "left": ((0, 0), (0, height - 1)),
            "right": ((width - 1, 0), (width - 1, height - 1)),
            "top": ((0, 0), (width - 1, 0)),
            "bottom": ((0, height - 1), (width - 1, height - 1)),
        }
        start, end = lines[side]
        ax.plot(
            (start[0], end[0]),
            (start[1], end[1]),
            linestyle="--",
            linewidth=1.5,
            color="#ff9800",
            alpha=0.9,
        )


__all__ = [
    "BorderRelation",
    "HSV_COLOR_RANGES",
    "PatchPortCandidate",
    "PatchPortDetectionResult",
    "PatchPortTopologyDetector",
    "SubjectComponent",
    "SubjectEdgeAnalyzer",
    "SubjectEdgeResult",
]
