from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CSTPortGeometry:
    center: tuple[float, float]
    feed_direction: tuple[float, float]
    port_normal: tuple[float, float]
    feed_width: float
    port_width: float
    port_height: float
    rectangle_points: np.ndarray
    confidence: float
    recommended_air_margin: float = 0.0
    recommended_port_height: float = 0.0
    recommended_port_width: float = 0.0
    valid_region_limited: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rectangle_points"] = np.round(self.rectangle_points, 3).tolist()
        payload["center"] = [round(float(v), 3) for v in self.center]
        payload["feed_direction"] = [round(float(v), 4) for v in self.feed_direction]
        payload["port_normal"] = [round(float(v), 4) for v in self.port_normal]
        payload["feed_width"] = round(float(self.feed_width), 3)
        payload["port_width"] = round(float(self.port_width), 3)
        payload["port_height"] = round(float(self.port_height), 3)
        payload["confidence"] = round(float(self.confidence), 3)
        payload["recommended_air_margin"] = round(float(self.recommended_air_margin), 3)
        payload["recommended_port_height"] = round(float(self.recommended_port_height), 3)
        payload["recommended_port_width"] = round(float(self.recommended_port_width), 3)
        payload["valid_region_limited"] = bool(self.valid_region_limited)
        return payload


class CSTPortGeometryBuilder:
    """Build CST-oriented excitation geometry from a detected patch feed endpoint."""

    def __init__(
        self,
        *,
        trace_distance_px: float = 14.0,
        width_sample_count: int = 8,
        outward_offset_width_factor: float = 0.9,
        min_outward_offset_px: float = 3.0,
        port_width_padding_factor: float = 1.35,
        port_height_factor: float = 0.25,
    ) -> None:
        self.trace_distance_px = float(trace_distance_px)
        self.width_sample_count = int(width_sample_count)
        self.outward_offset_width_factor = float(outward_offset_width_factor)
        self.min_outward_offset_px = float(min_outward_offset_px)
        self.port_width_padding_factor = float(port_width_padding_factor)
        self.port_height_factor = float(port_height_factor)

    def build_port_geometry(
        self,
        *,
        endpoint: tuple[int, int],
        border_side: str,
        subject_mask: np.ndarray,
        skeleton_mask: np.ndarray,
        valid_region_mask: Optional[np.ndarray] = None,
    ) -> Optional[CSTPortGeometry]:
        """Convert a skeleton endpoint into a CST-compatible port plane.

        endpoint 本身只能表示馈线终点，不能直接作为 CST port 的激励面。
        必须先估计馈线方向，再构造与馈线正交、并位于导体外侧的 excitation plane。
        """

        binary_mask = self._normalize_mask(subject_mask)
        skeleton = self._normalize_mask(skeleton_mask)
        if binary_mask.shape != skeleton.shape:
            return None
        valid_region = self._normalize_valid_region(valid_region_mask, binary_mask.shape)

        path_points = self._trace_skeleton_path(skeleton, endpoint)
        feed_direction, direction_confidence = self._estimate_feed_direction(
            endpoint=endpoint,
            path_points=path_points,
            border_side=border_side,
        )
        if feed_direction is None:
            return None

        feed_width, width_confidence = self._estimate_feed_width(
            endpoint=endpoint,
            path_points=path_points,
            binary_mask=binary_mask,
            feed_direction=feed_direction,
        )
        if feed_width <= 0:
            return None

        port_width = max(2.0, feed_width * self.port_width_padding_factor)
        port_height = max(2.0, feed_width * self.port_height_factor)
        recommended_air_margin = max(2.0, feed_width * 0.5)
        recommended_port_height = max(3.0, feed_width * 1.5)
        offset = self._compute_outward_offset(feed_width)
        raw_center = (
            float(endpoint[0]) + feed_direction[0] * offset,
            float(endpoint[1]) + feed_direction[1] * offset,
        )
        center = self._push_center_outside_mask(raw_center, feed_direction, binary_mask)
        center, valid_region_limited = self._limit_center_to_valid_region(
            center=center,
            endpoint=endpoint,
            feed_direction=feed_direction,
            valid_region_mask=valid_region,
        )
        rectangle_points = self._build_port_rectangle(
            center=center,
            feed_direction=feed_direction,
            port_width=port_width,
            port_height=port_height,
        )
        port_normal = (-feed_direction[1], feed_direction[0])
        confidence = max(0.0, min(1.0, 0.55 * direction_confidence + 0.45 * width_confidence))

        return CSTPortGeometry(
            center=(float(center[0]), float(center[1])),
            feed_direction=(float(feed_direction[0]), float(feed_direction[1])),
            port_normal=(float(port_normal[0]), float(port_normal[1])),
            feed_width=float(feed_width),
            port_width=float(port_width),
            port_height=float(port_height),
            rectangle_points=rectangle_points,
            confidence=float(confidence),
            recommended_air_margin=float(recommended_air_margin),
            recommended_port_height=float(recommended_port_height),
            recommended_port_width=float(port_width),
            valid_region_limited=bool(valid_region_limited),
        )

    @staticmethod
    def _normalize_mask(mask: np.ndarray) -> np.ndarray:
        array = np.asarray(mask)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        return np.where(array > 0, 255, 0).astype(np.uint8)

    @staticmethod
    def _normalize_valid_region(mask: Optional[np.ndarray], shape: tuple[int, int]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        region = CSTPortGeometryBuilder._normalize_mask(mask)
        if region.shape != shape or int(np.count_nonzero(region)) == 0:
            return None

        # 有效设计区域来自第一层外轮廓，用它限制 port plane，
        # 防止截图底部/四周多余像素把激励面推到介质板外面。
        region = CSTPortGeometryBuilder._fill_binary_holes(region)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        filled = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(filled, [hull], -1, 255, thickness=-1)
        return filled

    @staticmethod
    def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
        flood = mask.copy()
        height, width = flood.shape
        flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        return cv2.bitwise_or(mask, holes)

    def _trace_skeleton_path(self, skeleton_mask: np.ndarray, endpoint: tuple[int, int]) -> list[tuple[int, int]]:
        """Trace several pixels from the endpoint back into the feed branch."""

        x0, y0 = endpoint
        height, width = skeleton_mask.shape
        if not (0 <= x0 < width and 0 <= y0 < height) or skeleton_mask[y0, x0] == 0:
            return [endpoint]

        path = [endpoint]
        previous: Optional[tuple[int, int]] = None
        current = endpoint
        walked = 0.0

        # 只沿 endpoint 唯一路径向内追踪一小段；遇到 junction 就停止，
        # 避免把贴片主体里的复杂骨架方向误认为馈线方向。
        while walked < self.trace_distance_px:
            neighbors = self._skeleton_neighbors(skeleton_mask, current)
            if previous is not None:
                neighbors = [point for point in neighbors if point != previous]
            if not neighbors:
                break
            if len(neighbors) > 1:
                break

            next_point = neighbors[0]
            walked += float(np.hypot(next_point[0] - current[0], next_point[1] - current[1]))
            path.append(next_point)
            previous, current = current, next_point

        return path

    @staticmethod
    def _skeleton_neighbors(skeleton_mask: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = point
        height, width = skeleton_mask.shape
        neighbors: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and skeleton_mask[ny, nx] > 0:
                    neighbors.append((int(nx), int(ny)))
        return neighbors

    def _estimate_feed_direction(
        self,
        *,
        endpoint: tuple[int, int],
        path_points: list[tuple[int, int]],
        border_side: str,
    ) -> tuple[Optional[tuple[float, float]], float]:
        if len(path_points) < 2:
            border_direction = self._border_outward_direction(border_side)
            return border_direction, 0.45 if border_direction is not None else 0.0

        endpoint_array = np.array(endpoint, dtype=np.float64)
        samples = np.array(path_points[1:], dtype=np.float64)
        distances = np.linalg.norm(samples - endpoint_array.reshape(1, 2), axis=1)
        far_samples = samples[distances >= min(3.0, self.trace_distance_px * 0.35)]
        if len(far_samples) == 0:
            far_samples = samples[-1:].copy()

        inward_point = np.mean(far_samples, axis=0)
        measured = endpoint_array - inward_point
        measured_norm = float(np.linalg.norm(measured))
        if measured_norm <= 1e-6:
            border_direction = self._border_outward_direction(border_side)
            return border_direction, 0.4 if border_direction is not None else 0.0

        measured = measured / measured_norm
        border_direction = self._border_outward_direction(border_side)
        if border_direction is not None:
            border = np.array(border_direction, dtype=np.float64)
            alignment = float(np.dot(measured, border))
            if alignment < 0:
                measured = -measured
                alignment = -alignment
            # 贴片馈线通常从图像/有效导体边界进入；用边界方向稳定锯齿 skeleton。
            # 当 skeleton 与边界方向高度一致时，优先相信边界法向，避免锯齿让端口斜插导体。
            if alignment >= 0.8:
                blended = measured * 0.2 + border * 0.8
                blended /= max(1e-6, float(np.linalg.norm(blended)))
                return (float(blended[0]), float(blended[1])), max(0.7, min(1.0, alignment))
            if alignment >= 0.45:
                blended = measured * 0.35 + border * 0.65
                blended /= max(1e-6, float(np.linalg.norm(blended)))
                return (float(blended[0]), float(blended[1])), max(0.55, min(1.0, alignment))
            return border_direction, 0.5

        return (float(measured[0]), float(measured[1])), 0.75

    @staticmethod
    def _border_outward_direction(border_side: str) -> Optional[tuple[float, float]]:
        mapping = {
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
            "top": (0.0, -1.0),
            "bottom": (0.0, 1.0),
        }
        return mapping.get(str(border_side).lower())

    def _estimate_feed_width(
        self,
        *,
        endpoint: tuple[int, int],
        path_points: list[tuple[int, int]],
        binary_mask: np.ndarray,
        feed_direction: tuple[float, float],
    ) -> tuple[float, float]:
        distance_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        width_values: list[float] = []
        usable_points = path_points[: max(2, self.width_sample_count)]
        for x, y in usable_points:
            if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]:
                width = float(distance_map[y, x] * 2.0)
                if width > 0:
                    width_values.append(width)

        cross_widths: list[float] = []
        for point in usable_points[: max(2, min(5, len(usable_points)))]:
            cross_width = self._scan_cross_section_width(
                point=point,
                feed_direction=feed_direction,
                binary_mask=binary_mask,
            )
            if cross_width > 0:
                cross_widths.append(cross_width)

        if not width_values and not cross_widths:
            return 0.0, 0.0

        # endpoint 处的 distanceTransform 常被端点边界压小；
        # 用中位数抑制单点异常，再结合正交截面扫描，避免窄馈线被低估。
        distance_width = float(np.median(width_values)) if width_values else 0.0
        cross_width = float(np.median(cross_widths)) if cross_widths else 0.0
        feed_width = max(distance_width, cross_width)
        confidence = 0.65
        combined = width_values + cross_widths
        if len(combined) >= 3:
            spread = float(np.std(combined) / max(feed_width, 1.0))
            confidence = max(0.45, min(0.95, 0.9 - spread))
        return max(1.0, feed_width), confidence

    def _scan_cross_section_width(
        self,
        *,
        point: tuple[int, int],
        feed_direction: tuple[float, float],
        binary_mask: np.ndarray,
    ) -> float:
        """Scan conductor width along the direction orthogonal to feed propagation."""

        x0, y0 = point
        normal = np.array([-feed_direction[1], feed_direction[0]], dtype=np.float64)
        normal /= max(1e-6, float(np.linalg.norm(normal)))
        height, width = binary_mask.shape

        def is_inside(distance: float) -> bool:
            x = int(round(float(x0) + normal[0] * distance))
            y = int(round(float(y0) + normal[1] * distance))
            return 0 <= x < width and 0 <= y < height and binary_mask[y, x] > 0

        if not is_inside(0.0):
            return 0.0

        negative = 0
        while negative < 80 and is_inside(float(-(negative + 1))):
            negative += 1
        positive = 0
        while positive < 80 and is_inside(float(positive + 1)):
            positive += 1
        return float(negative + positive + 1)

    def _compute_outward_offset(self, feed_width: float) -> float:
        return max(self.min_outward_offset_px, feed_width * self.outward_offset_width_factor)

    def _push_center_outside_mask(
        self,
        center: tuple[float, float],
        feed_direction: tuple[float, float],
        binary_mask: np.ndarray,
    ) -> tuple[float, float]:
        x, y = center
        dx, dy = feed_direction
        height, width = binary_mask.shape
        for _ in range(16):
            ix, iy = int(round(x)), int(round(y))
            if ix < 0 or ix >= width or iy < 0 or iy >= height or binary_mask[iy, ix] == 0:
                return float(x), float(y)
            x += dx
            y += dy
        return float(x), float(y)

    def _limit_center_to_valid_region(
        self,
        *,
        center: tuple[float, float],
        endpoint: tuple[int, int],
        feed_direction: tuple[float, float],
        valid_region_mask: Optional[np.ndarray],
    ) -> tuple[tuple[float, float], bool]:
        if valid_region_mask is None:
            return center, False

        height, width = valid_region_mask.shape

        def inside(point: tuple[float, float]) -> bool:
            x = int(round(point[0]))
            y = int(round(point[1]))
            return 0 <= x < width and 0 <= y < height and valid_region_mask[y, x] > 0

        if inside(center):
            return center, False

        endpoint_float = (float(endpoint[0]), float(endpoint[1]))
        if not inside(endpoint_float):
            nearest = self._nearest_valid_region_point(endpoint_float, valid_region_mask)
            return nearest if nearest is not None else endpoint_float, True

        direction = np.array(feed_direction, dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return endpoint_float, True
        direction /= direction_norm

        # 当外偏移会离开第一层轮廓时，不能继续使用截图外部空间。
        # 保守做法是把调试/未来任意端口的 plane 拉回馈线端面，
        # 保证它贴着金属 feed terminal，而不是悬浮在介质板外。
        last_inside = endpoint_float
        for step in range(1, 64):
            candidate = (
                float(endpoint[0]) + float(direction[0]) * step,
                float(endpoint[1]) + float(direction[1]) * step,
            )
            if not inside(candidate):
                break
            last_inside = candidate
        return last_inside, True

    @staticmethod
    def _nearest_valid_region_point(
        point: tuple[float, float],
        valid_region_mask: np.ndarray,
    ) -> Optional[tuple[float, float]]:
        ys, xs = np.where(valid_region_mask > 0)
        if len(xs) == 0:
            return None
        dx = xs.astype(np.float64) - float(point[0])
        dy = ys.astype(np.float64) - float(point[1])
        index = int(np.argmin(dx * dx + dy * dy))
        return float(xs[index]), float(ys[index])

    @staticmethod
    def _build_port_rectangle(
        *,
        center: tuple[float, float],
        feed_direction: tuple[float, float],
        port_width: float,
        port_height: float,
    ) -> np.ndarray:
        cx, cy = center
        tx, ty = feed_direction
        normal = np.array([-ty, tx], dtype=np.float64)
        normal /= max(1e-6, float(np.linalg.norm(normal)))
        tangent = np.array([tx, ty], dtype=np.float64)
        tangent /= max(1e-6, float(np.linalg.norm(tangent)))

        c = np.array([cx, cy], dtype=np.float64)
        half_width = port_width * 0.5
        half_height = port_height * 0.5
        return np.array(
            [
                c - normal * half_width - tangent * half_height,
                c + normal * half_width - tangent * half_height,
                c + normal * half_width + tangent * half_height,
                c - normal * half_width + tangent * half_height,
            ],
            dtype=np.float64,
        )

    def write_debug_outputs(
        self,
        *,
        debug_dir: str | Path,
        original_image: Optional[np.ndarray],
        subject_mask: np.ndarray,
        valid_region_mask: Optional[np.ndarray] = None,
        endpoint: tuple[int, int],
        geometry: CSTPortGeometry,
        metadata: dict[str, Any],
    ) -> None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

        base = self._debug_base(original_image, subject_mask)
        endpoint_xy = (int(endpoint[0]), int(endpoint[1]))
        center_xy = (int(round(geometry.center[0])), int(round(geometry.center[1])))

        direction_vis = base.copy()
        cv2.circle(direction_vis, endpoint_xy, 4, (0, 255, 255), -1)
        arrow_end = (
            int(round(endpoint[0] + geometry.feed_direction[0] * 25.0)),
            int(round(endpoint[1] + geometry.feed_direction[1] * 25.0)),
        )
        cv2.arrowedLine(direction_vis, endpoint_xy, arrow_end, (255, 255, 0), 2, tipLength=0.25)
        cv2.imwrite(str(debug_path / "feed_direction.png"), direction_vis)

        width_vis = base.copy()
        cv2.circle(width_vis, endpoint_xy, max(2, int(round(geometry.feed_width / 2.0))), (0, 255, 255), 1)
        cv2.imwrite(str(debug_path / "feed_width.png"), width_vis)

        plane_vis = base.copy()
        if valid_region_mask is not None:
            region = self._normalize_valid_region(valid_region_mask, subject_mask.shape[:2])
            if region is not None:
                contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(plane_vis, contours, -1, (255, 255, 0), 1)
        rect = np.round(geometry.rectangle_points).astype(np.int32)
        cv2.polylines(plane_vis, [rect], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.circle(plane_vis, endpoint_xy, 4, (0, 255, 255), -1)
        cv2.circle(plane_vis, center_xy, 4, (0, 0, 255), -1)
        cv2.imwrite(str(debug_path / "port_plane_overlay.png"), plane_vis)

        (debug_path / "port_geometry_debug.json").write_text(
            json_dumps(metadata),
            encoding="utf-8",
        )

    @staticmethod
    def _debug_base(original_image: Optional[np.ndarray], subject_mask: np.ndarray) -> np.ndarray:
        if original_image is not None:
            base = original_image.copy()
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        else:
            base = cv2.cvtColor(CSTPortGeometryBuilder._normalize_mask(subject_mask), cv2.COLOR_GRAY2BGR)
        mask = CSTPortGeometryBuilder._normalize_mask(subject_mask)
        gray_overlay = np.zeros_like(base)
        gray_overlay[mask > 0] = (120, 120, 120)
        return cv2.addWeighted(base, 0.65, gray_overlay, 0.35, 0)


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2)
