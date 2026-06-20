from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from skimage.morphology import skeletonize

try:
    from .port_geometry_builder import CSTPortGeometryBuilder
    from .port_em_validator import PortEMValidator
except ImportError:  # pragma: no cover - allows direct script-style imports.
    from port_geometry_builder import CSTPortGeometryBuilder
    from port_em_validator import PortEMValidator


@dataclass(frozen=True)
class PatchPortCandidate:
    id: int
    point: tuple[int, int]
    direction: str
    touches_border: bool
    local_width: float
    path_length_to_patch: float
    connected_to_main_patch: bool
    score: float
    confidence: float


@dataclass(frozen=True)
class PatchPortDetectionResult:
    ports: list[PatchPortCandidate]
    skeleton_mask: np.ndarray
    endpoint_mask: np.ndarray
    debug_metadata: dict


@dataclass(frozen=True)
class _MainPatchRegion:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    looks_like_frame: bool


@dataclass(frozen=True)
class _ParameterizedLineSegment:
    id: int
    start: tuple[float, float]
    end: tuple[float, float]
    length: float
    component_id: Optional[int]
    primitive_index: Optional[int]
    source: str


@dataclass(frozen=True)
class _ParameterizedPortHint:
    point: tuple[int, int]
    direction: str
    local_width: float
    score: float
    confidence: float
    segment_id: int
    distance_to_image_border: float
    extremity_distance: float
    length_ratio: float


class PatchPortTopologyDetector:
    """Topology-first detector for patch antenna feed ports.

    Stage 1 intentionally keeps the reasoning conservative:
    skeleton -> endpoints -> near-border endpoints -> connected-to-main-patch score.
    Later graph/geodesic reasoning can be added behind the same public result.
    """

    def __init__(
        self,
        *,
        border_distance_px: int = 8,
        min_component_area: int = 200,
        min_skeleton_component_size: int = 3,
        min_port_score: float = 10.0,
        max_ports: int = 4,
    ) -> None:
        self.border_distance_px = int(border_distance_px)
        self.min_component_area = int(min_component_area)
        self.min_skeleton_component_size = int(min_skeleton_component_size)
        self.min_port_score = float(min_port_score)
        self.max_ports = int(max_ports)

    def detect_ports(
        self,
        *,
        subject_mask: Optional[np.ndarray] = None,
        foreground_mask: Optional[np.ndarray] = None,
        original_image: Optional[np.ndarray] = None,
        valid_region_mask: Optional[np.ndarray] = None,
        parameterization_payload: Optional[dict[str, Any]] = None,
        parameterization_path: Optional[str | Path] = None,
        debug_dir: Optional[str | Path] = None,
    ) -> PatchPortDetectionResult:
        """Detect patch feed-port candidates from a conductive mask."""

        if subject_mask is None and foreground_mask is None:
            raise ValueError("Either subject_mask or foreground_mask must be provided.")

        source_name = "subject_mask" if subject_mask is not None else "foreground_mask"
        source_mask = subject_mask if subject_mask is not None else foreground_mask
        binary_mask = self._normalize_mask(source_mask)
        valid_region = self._normalize_valid_region(valid_region_mask, binary_mask.shape)
        valid_region_bbox = self._mask_bbox(valid_region) if valid_region is not None else None
        labels, main_patch = self._find_main_patch_region(binary_mask)
        parameterization_payload = self._load_parameterization_payload(
            parameterization_payload=parameterization_payload,
            parameterization_path=parameterization_path,
        )
        parameterized_segments = self._extract_parameterized_line_segments(parameterization_payload)
        parameterized_hints = self._build_parameterized_port_hints(
            segments=parameterized_segments,
            payload=parameterization_payload,
            image_shape=binary_mask.shape,
        )

        skeleton_mask = self._build_skeleton(binary_mask)
        endpoints, endpoint_mask = self._find_skeleton_endpoints(skeleton_mask)

        distance_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        raw_candidates = self._score_port_candidates(
            endpoints=endpoints,
            binary_mask=binary_mask,
            labels=labels,
            main_patch=main_patch,
            distance_map=distance_map,
            image_shape=binary_mask.shape,
            valid_region_mask=valid_region,
            valid_region_bbox=valid_region_bbox,
        )
        terminal_face_candidates = self._find_terminal_face_candidates(
            binary_mask=binary_mask,
            labels=labels,
            main_patch=main_patch,
            image_shape=binary_mask.shape,
            valid_region_mask=valid_region,
            valid_region_bbox=valid_region_bbox,
        )
        terminal_keys = {(candidate.point, candidate.direction) for candidate in terminal_face_candidates}
        raw_candidates, feed_branch_metadata = self._apply_em_aware_scoring(
            candidates=raw_candidates + terminal_face_candidates,
            binary_mask=binary_mask,
            skeleton_mask=skeleton_mask,
            distance_map=distance_map,
            terminal_keys=terminal_keys,
        )
        if parameterized_segments:
            raw_candidates, parameterized_evidence = self._apply_parameterized_line_guidance(
                candidates=raw_candidates,
                segments=parameterized_segments,
            )
            self._attach_parameterized_evidence(feed_branch_metadata, parameterized_evidence)
        parameterized_candidates = self._port_hint_candidates(
            hints=parameterized_hints,
            segments=parameterized_segments,
        )
        parameterized_candidate_keys = {
            (candidate.point, candidate.direction) for candidate in parameterized_candidates
        }
        raw_candidates = self._rank_port_candidates(raw_candidates + parameterized_candidates)
        feed_branch_metadata.extend(self._port_hint_metadata(parameterized_hints, parameterized_segments))

        if (main_patch is None or main_patch.looks_like_frame) and not parameterized_hints:
            selected_ports: list[PatchPortCandidate] = []
        else:
            selected_ports = [
                candidate
                for candidate in raw_candidates
                if candidate.connected_to_main_patch and candidate.score >= self.min_port_score
            ][: self.max_ports]

        port_geometry_metadata: list[dict[str, Any]] = []
        if selected_ports:
            selected_ports, port_geometry_metadata = self._refine_selected_port_geometries(
                selected_ports=selected_ports,
                binary_mask=binary_mask,
                skeleton_mask=skeleton_mask,
                valid_region_mask=valid_region,
                original_image=original_image,
                skip_refine_keys=parameterized_candidate_keys,
                debug_dir=Path(debug_dir) if debug_dir is not None else None,
            )

        metadata = {
            "detector": "PatchPortTopologyDetector",
            "stage": "stage1_skeleton_endpoint_border_scoring_with_port_geometry_refinement",
            "source_mask": source_name,
            "image_shape": list(binary_mask.shape),
            "valid_region": self._valid_region_to_dict(valid_region, valid_region_bbox),
            "thresholds": {
                "border_distance_px": self.border_distance_px,
                "min_component_area": self.min_component_area,
                "min_skeleton_component_size": self.min_skeleton_component_size,
                "min_port_score": self.min_port_score,
                "max_ports": self.max_ports,
            },
            "main_patch": self._main_patch_to_dict(main_patch),
            "endpoint_count": int(len(endpoints)),
            "border_endpoint_count": int(len(raw_candidates)),
            "terminal_face_candidate_count": int(len(terminal_face_candidates)),
            "parameterization_guidance": self._parameterization_guidance_to_dict(
                payload=parameterization_payload,
                path=parameterization_path,
                segments=parameterized_segments,
                hints=parameterized_hints,
            ),
            "selected_port_count": int(len(selected_ports)),
            "candidate_ports": [asdict(candidate) for candidate in raw_candidates],
            "feed_branch_analysis": feed_branch_metadata,
            "selected_ports": [asdict(candidate) for candidate in selected_ports],
            "port_geometries": port_geometry_metadata,
            "notes": [
                "path_length_to_patch is a Stage-1 centroid-distance proxy; graph geodesic length can replace it later.",
                "Frame-like largest components are rejected to avoid selecting screenshot/page borders as ports.",
                "Selected port point remains the metal feed contact; outward excitation-plane center is reported only in port_geometries.",
            ],
        }

        if debug_dir is not None:
            self._write_debug_outputs(
                debug_dir=Path(debug_dir),
                original_image=original_image,
                binary_mask=binary_mask,
                skeleton_mask=skeleton_mask,
                endpoint_mask=endpoint_mask,
                valid_region_mask=valid_region,
                endpoints=endpoints,
                ports=selected_ports,
                main_patch=main_patch,
                metadata=metadata,
            )

        return PatchPortDetectionResult(
            ports=selected_ports,
            skeleton_mask=skeleton_mask,
            endpoint_mask=endpoint_mask,
            debug_metadata=metadata,
        )

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert any grayscale/BGR mask-like input to a clean 0/255 uint8 mask."""

        if mask is None:
            raise ValueError("Mask must not be None.")
        array = np.asarray(mask)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if array.ndim != 2:
            raise ValueError("Mask must be a 2-D array or a 3-channel image.")

        binary = np.where(array > 0, 255, 0).astype(np.uint8)

        # 不在这里做 3x3 opening：很多馈线在截图里只有 1-2 px 宽，
        # 一次开运算就可能把真正的端口分支直接抹掉。
        return self._remove_tiny_mask_components(binary)

    def _normalize_valid_region(
        self,
        valid_region_mask: Optional[np.ndarray],
        image_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        if valid_region_mask is None:
            return None
        region = self._normalize_mask(valid_region_mask)
        if region.shape != image_shape or int(np.count_nonzero(region)) == 0:
            return None

        # 第一层轮廓代表可用于端口判断的有效设计区域。
        # 只保留最大外轮廓并填充，避免截图四周多余像素影响 border/port plane。
        region = self._fill_binary_holes(region)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        filled = np.zeros(image_shape, dtype=np.uint8)
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

    @staticmethod
    def _mask_bbox(mask: Optional[np.ndarray]) -> Optional[tuple[int, int, int, int]]:
        if mask is None:
            return None
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())
        return x1, y1, int(x2 - x1 + 1), int(y2 - y1 + 1)

    @staticmethod
    def _point_inside_mask(point: tuple[int, int], mask: Optional[np.ndarray]) -> bool:
        if mask is None:
            return True
        x, y = point
        height, width = mask.shape
        if 0 <= x < width and 0 <= y < height and mask[y, x] > 0:
            return True

        # 轮廓填充和导体 mask 在 raster 边界处常有 1-2 px 量化差。
        # 这里允许贴着第一层轮廓的端面候选进入评分，但几何外偏仍会被 valid_region 拉回轮廓内部。
        tolerance_px = 2
        x1 = max(0, x - tolerance_px)
        x2 = min(width, x + tolerance_px + 1)
        y1 = max(0, y - tolerance_px)
        y2 = min(height, y + tolerance_px + 1)
        return bool(np.any(mask[y1:y2, x1:x2] > 0))

    def _remove_tiny_mask_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """Remove isolated conductive dust while preserving thin connected feeds."""

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels <= 1:
            return binary_mask

        min_noise_area = max(1, min(self.min_component_area // 10, 25))
        cleaned = np.zeros_like(binary_mask)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_noise_area:
                cleaned[labels == label] = 255
        return cleaned

    def _build_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        """Skeletonize the conductive region while keeping topology."""

        # skeletonize 后的结果必须保持拓扑连通性，
        # 否则后续 endpoint 检测会出现伪端点，端口会被错误地吸到断裂处。
        skeleton = skeletonize(binary_mask > 0)
        skeleton_mask = np.where(skeleton, 255, 0).astype(np.uint8)
        return self._remove_tiny_skeleton_components(skeleton_mask)

    def _remove_tiny_skeleton_components(self, skeleton_mask: np.ndarray) -> np.ndarray:
        """Remove isolated skeleton dust without pruning real feed branches."""

        if self.min_skeleton_component_size <= 1:
            return skeleton_mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton_mask, connectivity=8)
        cleaned = np.zeros_like(skeleton_mask)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self.min_skeleton_component_size:
                cleaned[labels == label] = 255
        return cleaned

    def _find_skeleton_endpoints(self, skeleton_mask: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Find skeleton pixels with exactly one 8-neighborhood skeleton neighbor."""

        skeleton_binary = (skeleton_mask > 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(skeleton_binary, cv2.CV_16S, kernel, borderType=cv2.BORDER_CONSTANT)
        neighbor_count = neighbor_count - skeleton_binary

        # endpoint 的定义保持非常朴素：8 邻域里恰好只有一个骨架邻居。
        # 这样可以稳定捕捉馈线入口，同时避免把 junction/corner 当成端口。
        endpoint_mask = np.where((skeleton_binary == 1) & (neighbor_count == 1), 255, 0).astype(np.uint8)
        ys, xs = np.where(endpoint_mask > 0)
        endpoints = [(int(x), int(y)) for y, x in zip(ys, xs)]
        return endpoints, endpoint_mask

    @staticmethod
    def _skeleton_neighbors(skeleton_mask: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
        """Return 8-neighborhood skeleton pixels around a point."""

        x, y = point
        height, width = skeleton_mask.shape
        neighbors: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height and skeleton_mask[ny, nx] > 0:
                    neighbors.append((int(nx), int(ny)))
        return neighbors

    def _find_main_patch_region(self, binary_mask: np.ndarray) -> tuple[np.ndarray, Optional[_MainPatchRegion]]:
        """Select the main conductive component using existing connected-component philosophy."""

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        height, width = binary_mask.shape

        candidates: list[tuple[float, _MainPatchRegion]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            fill_ratio = area / max(1, w * h)
            touches_all_sides = x <= 1 and y <= 1 and x + w >= width - 1 and y + h >= height - 1
            looks_like_frame = touches_all_sides and fill_ratio < 0.35

            # 外框类连通域常常来自截图边框，不是真正的 PEC 主体。
            # 保留它的统计信息用于 debug，但大幅降权，后续也不会输出端口。
            score = area * (0.02 if looks_like_frame else 1.0)
            region = _MainPatchRegion(
                label=int(label),
                area=area,
                bbox=(x, y, w, h),
                centroid=(float(centroids[label][0]), float(centroids[label][1])),
                looks_like_frame=bool(looks_like_frame),
            )
            candidates.append((score, region))

        if not candidates:
            return labels, None

        _, selected = max(candidates, key=lambda item: item[0])
        return labels, selected

    def _load_parameterization_payload(
        self,
        *,
        parameterization_payload: Optional[dict[str, Any]],
        parameterization_path: Optional[str | Path],
    ) -> Optional[dict[str, Any]]:
        if isinstance(parameterization_payload, dict):
            return parameterization_payload
        if parameterization_path is None:
            return None
        path = Path(parameterization_path)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_parameterized_line_segments(
        self,
        payload: Optional[dict[str, Any]],
    ) -> list[_ParameterizedLineSegment]:
        if not isinstance(payload, dict):
            return []

        segments: list[_ParameterizedLineSegment] = []
        seen: set[tuple[float, float, float, float]] = set()

        def add_segment(
            start: Optional[tuple[float, float]],
            end: Optional[tuple[float, float]],
            *,
            source: str,
            component_id: Optional[int] = None,
            primitive_index: Optional[int] = None,
        ) -> None:
            if start is None or end is None:
                return
            length = float(np.hypot(end[0] - start[0], end[1] - start[1]))
            if length <= 1e-6:
                return
            key = (
                round(float(start[0]), 4),
                round(float(start[1]), 4),
                round(float(end[0]), 4),
                round(float(end[1]), 4),
            )
            reverse_key = (key[2], key[3], key[0], key[1])
            if key in seen or reverse_key in seen:
                return
            seen.add(key)
            segments.append(
                _ParameterizedLineSegment(
                    id=len(segments),
                    start=(float(start[0]), float(start[1])),
                    end=(float(end[0]), float(end[1])),
                    length=length,
                    component_id=component_id,
                    primitive_index=primitive_index,
                    source=source,
                )
            )

        for component_index, component in enumerate(payload.get("components", []) or []):
            if not isinstance(component, dict):
                continue
            component_id = self._optional_int(component.get("component_id"), default=component_index)
            primitives = component.get("primitives")
            if not isinstance(primitives, list):
                continue
            for primitive_index, primitive in enumerate(primitives):
                if not isinstance(primitive, dict) or not self._looks_like_line_primitive(primitive):
                    continue
                start, end = self._primitive_line_endpoints(primitive)
                add_segment(
                    start,
                    end,
                    source="component_primitives",
                    component_id=component_id,
                    primitive_index=primitive_index,
                )

        if segments:
            return segments

        primitives = payload.get("primitives")
        if isinstance(primitives, list):
            for primitive_index, primitive in enumerate(primitives):
                if not isinstance(primitive, dict) or not self._looks_like_line_primitive(primitive):
                    continue
                start, end = self._primitive_line_endpoints(primitive)
                add_segment(
                    start,
                    end,
                    source="top_level_primitives",
                    primitive_index=primitive_index,
                )

        if segments:
            return segments

        for edge_index, edge in enumerate(payload.get("edges", []) or []):
            if not isinstance(edge, dict):
                continue
            points = edge.get("ordered_points")
            if not isinstance(points, list) or len(points) < 2:
                continue
            start = self._parse_xy_point(points[0])
            end = self._parse_xy_point(points[-1])
            add_segment(start, end, source="edge_endpoints", primitive_index=edge_index)

        return segments

    @staticmethod
    def _looks_like_line_primitive(primitive: dict[str, Any]) -> bool:
        labels = [
            str(primitive.get("type", "")).lower(),
            str(primitive.get("kind", "")).lower(),
            str(primitive.get("primitive_type", "")).lower(),
        ]
        return any(label == "line" or label.endswith("_line") for label in labels)

    def _primitive_line_endpoints(
        self,
        primitive: dict[str, Any],
    ) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
        start = self._parse_xy_point(primitive.get("start"))
        end = self._parse_xy_point(primitive.get("end"))
        if start is not None and end is not None:
            return start, end

        parameters = primitive.get("parameters")
        if isinstance(parameters, dict):
            start = self._parse_xy_point(parameters.get("start"))
            end = self._parse_xy_point(parameters.get("end"))
            if start is not None and end is not None:
                return start, end

        points = primitive.get("points")
        if isinstance(points, list) and len(points) >= 2:
            return self._parse_xy_point(points[0]), self._parse_xy_point(points[-1])
        return None, None

    @staticmethod
    def _parse_xy_point(value: Any) -> Optional[tuple[float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _build_parameterized_port_hints(
        self,
        *,
        segments: list[_ParameterizedLineSegment],
        payload: Optional[dict[str, Any]],
        image_shape: tuple[int, int],
    ) -> list[_ParameterizedPortHint]:
        if not segments:
            return []

        bbox = self._segments_bbox(segments)
        if bbox is None:
            return []
        min_x, min_y, max_x, max_y = bbox
        bbox_w = max(1.0, max_x - min_x)
        bbox_h = max(1.0, max_y - min_y)
        extremity_tolerance = max(5.0, min(36.0, min(bbox_w, bbox_h) * 0.08))
        max_terminal_ratio = 0.45
        min_terminal_length = max(4.0, min(10.0, min(bbox_w, bbox_h) * 0.008))
        canvas_w, canvas_h = self._payload_canvas_size(payload, image_shape)

        hints: list[_ParameterizedPortHint] = []
        for segment in segments:
            if segment.length < min_terminal_length:
                continue
            sx, sy = segment.start
            ex, ey = segment.end
            dx = ex - sx
            dy = ey - sy
            cx = (sx + ex) / 2.0
            cy = (sy + ey) / 2.0
            is_horizontal = abs(dy) <= max(2.0, abs(dx) * 0.2)
            is_vertical = abs(dx) <= max(2.0, abs(dy) * 0.2)

            side_records: list[tuple[str, float]] = []
            if is_horizontal:
                length_ratio = segment.length / bbox_w
                if length_ratio <= max_terminal_ratio:
                    side_records.append(("top", max(0.0, cy - min_y)))
                    side_records.append(("bottom", max(0.0, max_y - cy)))
            if is_vertical:
                length_ratio = segment.length / bbox_h
                if length_ratio <= max_terminal_ratio:
                    side_records.append(("left", max(0.0, cx - min_x)))
                    side_records.append(("right", max(0.0, max_x - cx)))

            for side, extremity_distance in side_records:
                if extremity_distance > extremity_tolerance:
                    continue
                reference_span = bbox_w if side in {"top", "bottom"} else bbox_h
                length_ratio = segment.length / max(1.0, reference_span)
                if side == "top":
                    image_border_distance = max(0.0, cy)
                elif side == "bottom":
                    image_border_distance = max(0.0, canvas_h - 1.0 - cy)
                elif side == "left":
                    image_border_distance = max(0.0, cx)
                else:
                    image_border_distance = max(0.0, canvas_w - 1.0 - cx)
                border_bonus = 1.5 if image_border_distance <= max(self.border_distance_px, 24) else 0.0
                score = 24.0 - 9.0 * length_ratio - 0.08 * extremity_distance + border_bonus
                confidence = max(0.5, min(1.0, 1.0 - 0.8 * length_ratio - 0.01 * extremity_distance))
                hints.append(
                    _ParameterizedPortHint(
                        point=(int(round(cx)), int(round(cy))),
                        direction=side,
                        local_width=round(float(segment.length), 3),
                        score=round(float(score), 3),
                        confidence=round(float(confidence), 3),
                        segment_id=segment.id,
                        distance_to_image_border=round(float(image_border_distance), 3),
                        extremity_distance=round(float(extremity_distance), 3),
                        length_ratio=round(float(length_ratio), 4),
                    )
                )

        hints.sort(
            key=lambda hint: (
                -hint.score,
                hint.distance_to_image_border,
                hint.extremity_distance,
                hint.segment_id,
            )
        )
        deduped: list[_ParameterizedPortHint] = []
        seen: set[tuple[tuple[int, int], str]] = set()
        for hint in hints:
            key = (hint.point, hint.direction)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hint)
        return deduped[: max(1, self.max_ports)]

    @staticmethod
    def _segments_bbox(
        segments: list[_ParameterizedLineSegment],
    ) -> Optional[tuple[float, float, float, float]]:
        if not segments:
            return None
        xs: list[float] = []
        ys: list[float] = []
        for segment in segments:
            xs.extend([segment.start[0], segment.end[0]])
            ys.extend([segment.start[1], segment.end[1]])
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _payload_canvas_size(
        payload: Optional[dict[str, Any]],
        image_shape: tuple[int, int],
    ) -> tuple[float, float]:
        height, width = image_shape
        if isinstance(payload, dict):
            canvas = payload.get("canvas")
            if isinstance(canvas, dict):
                try:
                    canvas_w = float(canvas.get("width"))
                    canvas_h = float(canvas.get("height"))
                    if canvas_w > 0 and canvas_h > 0:
                        return canvas_w, canvas_h
                except (TypeError, ValueError):
                    pass
        return float(width), float(height)

    def _port_hint_candidates(
        self,
        *,
        hints: list[_ParameterizedPortHint],
        segments: list[_ParameterizedLineSegment],
    ) -> list[PatchPortCandidate]:
        if not hints:
            return []
        bbox = self._segments_bbox(segments)
        candidates: list[PatchPortCandidate] = []
        for hint in hints:
            path_length = 0.0
            if bbox is not None:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                path_length = float(np.hypot(hint.point[0] - cx, hint.point[1] - cy))
            candidates.append(
                PatchPortCandidate(
                    id=len(candidates),
                    point=hint.point,
                    direction=hint.direction,
                    touches_border=bool(hint.distance_to_image_border <= self.border_distance_px),
                    local_width=hint.local_width,
                    path_length_to_patch=round(path_length, 3),
                    connected_to_main_patch=True,
                    score=hint.score,
                    confidence=hint.confidence,
                )
            )
        return candidates

    def _apply_parameterized_line_guidance(
        self,
        *,
        candidates: list[PatchPortCandidate],
        segments: list[_ParameterizedLineSegment],
    ) -> tuple[list[PatchPortCandidate], dict[tuple[tuple[int, int], str], dict[str, Any]]]:
        if not candidates or not segments:
            return candidates, {}

        guided: list[PatchPortCandidate] = []
        evidence: dict[tuple[tuple[int, int], str], dict[str, Any]] = {}
        for candidate in candidates:
            best_segment, distance = self._nearest_parameterized_segment(candidate.point, segments)
            compatible = best_segment is not None and self._segment_direction_matches_port(best_segment, candidate.direction)
            near_threshold = max(8.0, min(40.0, float(candidate.local_width) * 1.25))
            if best_segment is not None and distance <= near_threshold:
                delta = 4.0 if compatible else 1.0
            else:
                delta = -14.0
            score = round(float(candidate.score + delta), 3)
            confidence = candidate.confidence
            if delta < 0:
                confidence = round(float(max(0.02, candidate.confidence * 0.35)), 3)
            guided_candidate = PatchPortCandidate(
                id=candidate.id,
                point=candidate.point,
                direction=candidate.direction,
                touches_border=candidate.touches_border,
                local_width=candidate.local_width,
                path_length_to_patch=candidate.path_length_to_patch,
                connected_to_main_patch=candidate.connected_to_main_patch,
                score=score,
                confidence=confidence,
            )
            guided.append(guided_candidate)
            evidence[(candidate.point, candidate.direction)] = {
                "enabled": True,
                "nearest_segment_id": best_segment.id if best_segment is not None else None,
                "nearest_line_distance_px": round(float(distance), 3),
                "near_threshold_px": round(float(near_threshold), 3),
                "direction_compatible": bool(compatible),
                "score_delta": round(float(delta), 3),
            }
        return self._rank_port_candidates(guided), evidence

    @staticmethod
    def _nearest_parameterized_segment(
        point: tuple[int, int],
        segments: list[_ParameterizedLineSegment],
    ) -> tuple[Optional[_ParameterizedLineSegment], float]:
        if not segments:
            return None, float("inf")
        px = float(point[0])
        py = float(point[1])
        best_segment: Optional[_ParameterizedLineSegment] = None
        best_distance = float("inf")
        for segment in segments:
            distance = PatchPortTopologyDetector._point_to_line_segment_distance(
                (px, py),
                segment.start,
                segment.end,
            )
            if distance < best_distance:
                best_segment = segment
                best_distance = distance
        return best_segment, best_distance

    @staticmethod
    def _point_to_line_segment_distance(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        px, py = point
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return float(np.hypot(px - sx, py - sy))
        ratio = ((px - sx) * dx + (py - sy) * dy) / denom
        ratio = max(0.0, min(1.0, float(ratio)))
        qx = sx + ratio * dx
        qy = sy + ratio * dy
        return float(np.hypot(px - qx, py - qy))

    @staticmethod
    def _segment_direction_matches_port(segment: _ParameterizedLineSegment, direction: str) -> bool:
        dx = segment.end[0] - segment.start[0]
        dy = segment.end[1] - segment.start[1]
        if direction in {"top", "bottom"}:
            return abs(dy) <= max(2.0, abs(dx) * 0.25)
        if direction in {"left", "right"}:
            return abs(dx) <= max(2.0, abs(dy) * 0.25)
        return False

    @staticmethod
    def _attach_parameterized_evidence(
        metadata: list[dict[str, Any]],
        evidence: dict[tuple[tuple[int, int], str], dict[str, Any]],
    ) -> None:
        if not metadata or not evidence:
            return
        for item in metadata:
            point = item.get("candidate_point")
            direction = item.get("candidate_direction")
            if isinstance(point, list) and len(point) >= 2 and isinstance(direction, str):
                key = ((int(point[0]), int(point[1])), direction)
                item["parameterized_line_evidence"] = evidence.get(key)

    @staticmethod
    def _port_hint_metadata(
        hints: list[_ParameterizedPortHint],
        segments: list[_ParameterizedLineSegment],
    ) -> list[dict[str, Any]]:
        by_id = {segment.id: segment for segment in segments}
        metadata: list[dict[str, Any]] = []
        for hint in hints:
            segment = by_id.get(hint.segment_id)
            metadata.append(
                {
                    "candidate_point": list(hint.point),
                    "candidate_direction": hint.direction,
                    "candidate_source": "parameterized_line_terminal",
                    "terminal_face_candidate": True,
                    "base_score": hint.score,
                    "adjusted_score": hint.score,
                    "parameterized_line_evidence": {
                        "enabled": True,
                        "generated_candidate": True,
                        "segment_id": hint.segment_id,
                        "segment_start": list(segment.start) if segment is not None else None,
                        "segment_end": list(segment.end) if segment is not None else None,
                        "line_length_px": round(float(segment.length), 3) if segment is not None else None,
                        "distance_to_image_border_px": hint.distance_to_image_border,
                        "extremity_distance_px": hint.extremity_distance,
                        "length_ratio": hint.length_ratio,
                    },
                }
            )
        return metadata

    @staticmethod
    def _parameterization_guidance_to_dict(
        *,
        payload: Optional[dict[str, Any]],
        path: Optional[str | Path],
        segments: list[_ParameterizedLineSegment],
        hints: list[_ParameterizedPortHint],
    ) -> dict[str, Any]:
        return {
            "enabled": bool(isinstance(payload, dict)),
            "path": str(path) if path is not None else None,
            "line_segment_count": int(len(segments)),
            "terminal_hint_count": int(len(hints)),
            "terminal_hints": [asdict(hint) for hint in hints],
        }

    def _score_port_candidates(
        self,
        *,
        endpoints: list[tuple[int, int]],
        binary_mask: np.ndarray,
        labels: np.ndarray,
        main_patch: Optional[_MainPatchRegion],
        distance_map: np.ndarray,
        image_shape: tuple[int, int],
        valid_region_mask: Optional[np.ndarray] = None,
        valid_region_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> list[PatchPortCandidate]:
        """Score only endpoints near the image border as Stage-1 port candidates."""

        candidates: list[PatchPortCandidate] = []
        for endpoint in endpoints:
            if not self._point_inside_mask(endpoint, valid_region_mask):
                continue
            active_bbox = valid_region_bbox or (main_patch.bbox if main_patch is not None else None)
            side, border_distance, border_source = self._is_border_endpoint(endpoint, image_shape, active_bbox=active_bbox)
            if side == "interior":
                continue

            x, y = endpoint
            connected_to_main = bool(main_patch is not None and labels[y, x] == main_patch.label)
            path_length = self._centroid_distance(endpoint, main_patch)
            local_width = self._estimate_local_width(
                point=endpoint,
                direction=side,
                binary_mask=binary_mask,
                distance_width=float(distance_map[y, x] * 2.0),
            )

            score = 0.0
            if border_source == "image" and border_distance <= self.border_distance_px:
                score += 2.0
            elif border_source == "bbox":
                # bbox 边界代表有效导体/版图边界，不一定等于整张 PNG 边界。
                # 这类端口比真正贴图像边界略弱，但应优先于 inset 槽顶伪端点。
                score += 1.5
            score += 5.0 if connected_to_main else 0.0
            score += 2.0
            score += min(1.0, path_length / 80.0)
            if main_patch is not None:
                _, _, bbox_w, bbox_h = main_patch.bbox
                reference_width = float(min(bbox_w, bbox_h))
                if reference_width > 0:
                    width_ratio = local_width / reference_width
                    # 贴片端口通常是窄馈线；实体贴片中轴的伪端点往往横跨大矩形宽度。
                    # 这里仅做温和降权，避免把宽矩形 skeleton 端点误当端口。
                    if width_ratio <= 0.35:
                        score += 2.0
                    elif width_ratio >= 0.55:
                        score -= 4.0
            confidence = max(0.0, min(1.0, score / 13.0))

            candidates.append(
                PatchPortCandidate(
                    id=len(candidates),
                    point=(int(x), int(y)),
                    direction=side,
                    touches_border=bool(border_source == "image" and border_distance <= self.border_distance_px),
                    local_width=round(local_width, 3),
                    path_length_to_patch=round(path_length, 3),
                    connected_to_main_patch=connected_to_main,
                    score=round(score, 3),
                    confidence=round(confidence, 3),
                )
            )

        return self._rank_port_candidates(candidates)

    def _find_terminal_face_candidates(
        self,
        *,
        binary_mask: np.ndarray,
        labels: np.ndarray,
        main_patch: Optional[_MainPatchRegion],
        image_shape: tuple[int, int],
        valid_region_mask: Optional[np.ndarray] = None,
        valid_region_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> list[PatchPortCandidate]:
        """Find feed-pad terminal faces on the active conductor bbox.

        对 test47 这类右下角馈电焊盘，真正的 CST 端口应落在焊盘靠近外部边界的端面，
        而不是 skeleton 在焊盘下边缘产生的伪端点。这里只检测 bbox 边缘上的窄连续端面，
        作为 endpoint 候选的保守补充。
        """

        if main_patch is None or main_patch.looks_like_frame:
            return []

        bx, by, bw, bh = main_patch.bbox
        label = main_patch.label
        height, width = image_shape
        region_bbox = valid_region_bbox
        sides = {
            "left": {
                "x": bx,
                "axis_start": by,
                "axis_end": by + bh - 1,
                "distance_to_image": abs(bx - region_bbox[0]) if region_bbox is not None else bx,
            },
            "right": {
                "x": bx + bw - 1,
                "axis_start": by,
                "axis_end": by + bh - 1,
                "distance_to_image": abs((region_bbox[0] + region_bbox[2] - 1) - (bx + bw - 1))
                if region_bbox is not None
                else width - 1 - (bx + bw - 1),
            },
            "top": {
                "y": by,
                "axis_start": bx,
                "axis_end": bx + bw - 1,
                "distance_to_image": abs(by - region_bbox[1]) if region_bbox is not None else by,
            },
            "bottom": {
                "y": by + bh - 1,
                "axis_start": bx,
                "axis_end": bx + bw - 1,
                "distance_to_image": abs((region_bbox[1] + region_bbox[3] - 1) - (by + bh - 1))
                if region_bbox is not None
                else height - 1 - (by + bh - 1),
            },
        }

        max_image_gap = max(self.border_distance_px, 36)
        max_face_ratio = 0.24
        min_face_length = 6
        candidates: list[PatchPortCandidate] = []

        for side, info in sides.items():
            if int(info["distance_to_image"]) > max_image_gap:
                continue

            runs = self._side_label_runs(labels, label, side, info)
            side_length = int(info["axis_end"] - info["axis_start"] + 1)
            for start, end in runs:
                run_length = end - start + 1
                if run_length < min_face_length:
                    continue
                if run_length > max(min_face_length, int(round(side_length * max_face_ratio))):
                    # 大面积贴片外边缘不是馈电端面，避免把整块贴片边界当端口。
                    continue

                point = self._side_run_center_point(side, info, start, end)
                x, y = point
                if not (0 <= x < width and 0 <= y < height):
                    continue
                if not self._point_inside_mask(point, valid_region_mask):
                    continue
                if labels[y, x] != label:
                    continue

                width_estimate = self._estimate_local_width(
                    point=point,
                    direction=side,
                    binary_mask=binary_mask,
                    distance_width=float(run_length),
                )
                path_length = self._centroid_distance(point, main_patch)
                score = 17.0 + min(1.0, path_length / 120.0)
                candidates.append(
                    PatchPortCandidate(
                        id=len(candidates),
                        point=point,
                        direction=side,
                        touches_border=bool(int(info["distance_to_image"]) <= self.border_distance_px),
                        local_width=round(float(width_estimate), 3),
                        path_length_to_patch=round(path_length, 3),
                        connected_to_main_patch=True,
                        score=round(score, 3),
                        confidence=1.0,
                    )
                )

        return candidates

    @staticmethod
    def _side_label_runs(
        labels: np.ndarray,
        label: int,
        side: str,
        info: dict[str, int],
    ) -> list[tuple[int, int]]:
        values: list[int] = []
        if side in {"left", "right"}:
            x = int(info["x"])
            for y in range(int(info["axis_start"]), int(info["axis_end"]) + 1):
                values.append(1 if labels[y, x] == label else 0)
        else:
            y = int(info["y"])
            for x in range(int(info["axis_start"]), int(info["axis_end"]) + 1):
                values.append(1 if labels[y, x] == label else 0)

        runs: list[tuple[int, int]] = []
        run_start: Optional[int] = None
        for offset, value in enumerate(values):
            if value and run_start is None:
                run_start = offset
            elif not value and run_start is not None:
                runs.append((int(info["axis_start"]) + run_start, int(info["axis_start"]) + offset - 1))
                run_start = None
        if run_start is not None:
            runs.append((int(info["axis_start"]) + run_start, int(info["axis_start"]) + len(values) - 1))
        return runs

    @staticmethod
    def _side_run_center_point(side: str, info: dict[str, int], start: int, end: int) -> tuple[int, int]:
        mid = int(round((start + end) / 2.0))
        if side in {"left", "right"}:
            return int(info["x"]), mid
        return mid, int(info["y"])

    def _apply_em_aware_scoring(
        self,
        *,
        candidates: list[PatchPortCandidate],
        binary_mask: np.ndarray,
        skeleton_mask: np.ndarray,
        distance_map: np.ndarray,
        terminal_keys: set[tuple[tuple[int, int], str]],
    ) -> tuple[list[PatchPortCandidate], list[dict[str, Any]]]:
        validator = PortEMValidator()
        rescored: list[PatchPortCandidate] = []
        metadata: list[dict[str, Any]] = []

        for candidate in candidates:
            is_terminal = (candidate.point, candidate.direction) in terminal_keys
            analysis = self._analyze_feed_branch(
                candidate=candidate,
                binary_mask=binary_mask,
                skeleton_mask=skeleton_mask,
                distance_map=distance_map,
                terminal_face_candidate=is_terminal,
            )
            validation = validator.validate(analysis)

            em_score = float(validation.get("total_score", 0.0))
            terminal_bonus = 2.0 if is_terminal else 0.0
            adjusted_score = candidate.score + em_score + terminal_bonus
            confidence_scale = float(validation.get("confidence_scale", 1.0))
            adjusted_confidence = max(0.05, min(1.0, candidate.confidence * confidence_scale))

            rescored.append(
                PatchPortCandidate(
                    id=candidate.id,
                    point=candidate.point,
                    direction=candidate.direction,
                    touches_border=candidate.touches_border,
                    local_width=candidate.local_width,
                    path_length_to_patch=candidate.path_length_to_patch,
                    connected_to_main_patch=candidate.connected_to_main_patch,
                    score=round(float(adjusted_score), 3),
                    confidence=round(float(adjusted_confidence), 3),
                )
            )
            item = dict(analysis)
            item["candidate_point"] = list(candidate.point)
            item["candidate_direction"] = candidate.direction
            item["base_score"] = candidate.score
            item["em_validation"] = validation
            item["terminal_face_candidate"] = bool(is_terminal)
            item["adjusted_score"] = round(float(adjusted_score), 3)
            metadata.append(item)

        return self._rank_port_candidates(rescored), metadata

    def _analyze_feed_branch(
        self,
        *,
        candidate: PatchPortCandidate,
        binary_mask: np.ndarray,
        skeleton_mask: np.ndarray,
        distance_map: np.ndarray,
        terminal_face_candidate: bool,
    ) -> dict[str, Any]:
        start = self._nearest_skeleton_point(candidate.point, skeleton_mask, radius=max(12, int(candidate.local_width // 2) + 8))
        if start is None:
            start = candidate.point

        path = self._trace_feed_centerline(
            start=start,
            direction=candidate.direction,
            skeleton_mask=skeleton_mask,
            distance_map=distance_map,
        )
        widths = self._sample_width_profile(path, distance_map)
        branch_length = self._polyline_length(path)
        euclidean = float(np.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])) if len(path) > 1 else 0.0
        tortuosity = branch_length / max(euclidean, 1e-6) if branch_length > 0 else 99.0
        mean_width = float(np.mean(widths)) if widths else max(float(candidate.local_width), 1.0)
        width_std = float(np.std(widths)) if widths else 0.0
        min_width = float(np.min(widths)) if widths else mean_width
        max_width = float(np.max(widths)) if widths else mean_width
        aspect_ratio = branch_length / max(mean_width, 1e-6)
        junction_point, junction_confidence, transition = self._detect_feed_patch_transition(path, widths)

        return {
            "feed_centerline": [[int(x), int(y)] for x, y in path],
            "branch_length": round(float(branch_length), 3),
            "branch_direction": self._branch_direction(path, candidate.direction),
            "branch_curvature": round(float(self._branch_curvature(path)), 4),
            "branch_tortuosity": round(float(tortuosity), 3),
            "width_profile": [round(float(width), 3) for width in widths],
            "mean_width": round(float(mean_width), 3),
            "width_std": round(float(width_std), 3),
            "max_width": round(float(max_width), 3),
            "min_width": round(float(min_width), 3),
            "feed_aspect_ratio": round(float(aspect_ratio), 3),
            "junction_point": list(junction_point) if junction_point is not None else None,
            "junction_confidence": round(float(junction_confidence), 3),
            "patch_transition_detected": bool(transition),
            "terminal_face_candidate": bool(terminal_face_candidate),
        }

    def _nearest_skeleton_point(
        self,
        point: tuple[int, int],
        skeleton_mask: np.ndarray,
        *,
        radius: int,
    ) -> Optional[tuple[int, int]]:
        x, y = point
        height, width = skeleton_mask.shape
        x1, x2 = max(0, x - radius), min(width - 1, x + radius)
        y1, y2 = max(0, y - radius), min(height - 1, y + radius)
        ys, xs = np.where(skeleton_mask[y1 : y2 + 1, x1 : x2 + 1] > 0)
        if len(xs) == 0:
            return None
        xs = xs + x1
        ys = ys + y1
        distances = (xs - x) ** 2 + (ys - y) ** 2
        index = int(np.argmin(distances))
        return int(xs[index]), int(ys[index])

    def _trace_feed_centerline(
        self,
        *,
        start: tuple[int, int],
        direction: str,
        skeleton_mask: np.ndarray,
        distance_map: np.ndarray,
        max_steps: int = 140,
    ) -> list[tuple[int, int]]:
        path = [start]
        previous: Optional[tuple[int, int]] = None
        current = start
        inward = self._inward_vector(direction)
        previous_vector: Optional[np.ndarray] = None
        unstable_turns = 0

        for _ in range(max_steps):
            neighbors = self._skeleton_neighbors(skeleton_mask, current)
            if previous is not None:
                neighbors = [point for point in neighbors if point != previous]
            if not neighbors:
                break
            if previous is not None and len(neighbors) > 1:
                break

            next_point = self._choose_next_feed_point(current, neighbors, inward)
            if next_point is None:
                break

            current_width = float(distance_map[current[1], current[0]] * 2.0)
            next_width = float(distance_map[next_point[1], next_point[0]] * 2.0)
            if len(path) >= 4 and current_width > 0 and next_width > 2.5 * current_width:
                path.append(next_point)
                break

            step_vector = np.array([next_point[0] - current[0], next_point[1] - current[1]], dtype=np.float64)
            step_norm = float(np.linalg.norm(step_vector))
            if step_norm > 0:
                step_vector /= step_norm
                if previous_vector is not None:
                    dot = float(np.clip(np.dot(previous_vector, step_vector), -1.0, 1.0))
                    angle = float(np.degrees(np.arccos(dot)))
                    if angle > 70.0:
                        unstable_turns += 1
                    if unstable_turns >= 2:
                        break
                previous_vector = step_vector

            path.append(next_point)
            previous, current = current, next_point

        return path

    @staticmethod
    def _inward_vector(direction: str) -> np.ndarray:
        mapping = {
            "left": np.array([1.0, 0.0]),
            "right": np.array([-1.0, 0.0]),
            "top": np.array([0.0, 1.0]),
            "bottom": np.array([0.0, -1.0]),
        }
        return mapping.get(direction, np.array([0.0, 0.0]))

    @staticmethod
    def _choose_next_feed_point(
        current: tuple[int, int],
        neighbors: list[tuple[int, int]],
        inward: np.ndarray,
    ) -> Optional[tuple[int, int]]:
        if not neighbors:
            return None
        if float(np.linalg.norm(inward)) <= 1e-6:
            return neighbors[0]
        best = None
        best_score = -1e9
        for point in neighbors:
            vector = np.array([point[0] - current[0], point[1] - current[1]], dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            score = float(np.dot(vector / norm, inward))
            if score > best_score:
                best = point
                best_score = score
        return best

    @staticmethod
    def _sample_width_profile(path: list[tuple[int, int]], distance_map: np.ndarray) -> list[float]:
        widths: list[float] = []
        for x, y in path:
            if 0 <= y < distance_map.shape[0] and 0 <= x < distance_map.shape[1]:
                width = float(distance_map[y, x] * 2.0)
                if width > 0:
                    widths.append(width)
        return widths

    @staticmethod
    def _polyline_length(path: list[tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0.0
        return float(
            sum(
                np.hypot(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1])
                for index in range(1, len(path))
            )
        )

    @staticmethod
    def _branch_direction(path: list[tuple[int, int]], fallback_direction: str) -> list[float]:
        if len(path) < 2:
            vector = -PatchPortTopologyDetector._inward_vector(fallback_direction)
        else:
            vector = np.array([path[0][0] - path[-1][0], path[0][1] - path[-1][1]], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return [0.0, 0.0]
        vector = vector / norm
        return [round(float(vector[0]), 4), round(float(vector[1]), 4)]

    @staticmethod
    def _branch_curvature(path: list[tuple[int, int]]) -> float:
        if len(path) < 3:
            return 0.0
        angles: list[float] = []
        for index in range(1, len(path) - 1):
            a = np.array([path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1]], dtype=np.float64)
            b = np.array([path[index + 1][0] - path[index][0], path[index + 1][1] - path[index][1]], dtype=np.float64)
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 0 or nb <= 0:
                continue
            dot = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
            angles.append(float(np.arccos(dot)))
        return float(np.mean(angles)) if angles else 0.0

    @staticmethod
    def _detect_feed_patch_transition(
        path: list[tuple[int, int]],
        widths: list[float],
    ) -> tuple[Optional[tuple[int, int]], float, bool]:
        for index in range(1, min(len(widths), len(path))):
            current = max(widths[index - 1], 1e-6)
            next_width = widths[index]
            if next_width > 2.5 * current:
                confidence = min(1.0, (next_width / current) / 5.0)
                return path[index], float(confidence), True
        return None, 0.0, False

    @staticmethod
    def _rank_port_candidates(candidates: list[PatchPortCandidate]) -> list[PatchPortCandidate]:
        candidates.sort(key=lambda candidate: (-candidate.score, -candidate.path_length_to_patch, candidate.id))
        return [
            PatchPortCandidate(
                id=index,
                point=candidate.point,
                direction=candidate.direction,
                touches_border=candidate.touches_border,
                local_width=candidate.local_width,
                path_length_to_patch=candidate.path_length_to_patch,
                connected_to_main_patch=candidate.connected_to_main_patch,
                score=candidate.score,
                confidence=candidate.confidence,
            )
            for index, candidate in enumerate(candidates)
        ]

    def _is_border_endpoint(
        self,
        point: tuple[int, int],
        image_shape: tuple[int, int],
        *,
        active_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> tuple[str, int, str]:
        """Return the closest border side and distance for a skeleton endpoint."""

        x, y = point
        height, width = image_shape
        distances = {
            "left": int(x),
            "right": int(width - 1 - x),
            "top": int(y),
            "bottom": int(height - 1 - y),
        }
        image_side, image_distance = min(distances.items(), key=lambda item: item[1])
        if image_distance <= self.border_distance_px:
            return image_side, int(image_distance), "image"

        if active_bbox is not None:
            bx, by, bw, bh = active_bbox
            bbox_distances = {
                "left": abs(int(x - bx)),
                "right": abs(int((bx + bw - 1) - x)),
                "top": abs(int(y - by)),
                "bottom": abs(int((by + bh - 1) - y)),
            }
            bbox_side, bbox_distance = min(bbox_distances.items(), key=lambda item: item[1])
            bbox_threshold = max(self.border_distance_px, min(20, int(round(min(bw, bh) * 0.15))))
            if bbox_distance <= bbox_threshold:
                return bbox_side, int(bbox_distance), "bbox"

        return "interior", int(image_distance), "none"

    def _estimate_local_width(
        self,
        *,
        point: tuple[int, int],
        direction: str,
        binary_mask: np.ndarray,
        distance_width: float,
    ) -> float:
        """Estimate feed width near the endpoint along the axis perpendicular to propagation."""

        x, y = point
        height, width = binary_mask.shape
        if not (0 <= x < width and 0 <= y < height):
            return float(distance_width)

        # endpoint 处的 distanceTransform 常被端点本身的边界压小。
        # 因此沿端口截面方向扫描连续导体像素，给 CST 端口一个更真实的宽度。
        if direction in ("top", "bottom"):
            left = x
            while left - 1 >= 0 and binary_mask[y, left - 1] > 0:
                left -= 1
            right = x
            while right + 1 < width and binary_mask[y, right + 1] > 0:
                right += 1
            scan_width = float(right - left + 1)
        elif direction in ("left", "right"):
            top = y
            while top - 1 >= 0 and binary_mask[top - 1, x] > 0:
                top -= 1
            bottom = y
            while bottom + 1 < height and binary_mask[bottom + 1, x] > 0:
                bottom += 1
            scan_width = float(bottom - top + 1)
        else:
            scan_width = 0.0

        return float(max(distance_width, scan_width))

    def _centroid_distance(self, point: tuple[int, int], main_patch: Optional[_MainPatchRegion]) -> float:
        if main_patch is None:
            return 0.0
        x, y = point
        cx, cy = main_patch.centroid
        return float(np.hypot(float(x) - cx, float(y) - cy))

    def _main_patch_to_dict(self, main_patch: Optional[_MainPatchRegion]) -> Optional[dict[str, Any]]:
        if main_patch is None:
            return None
        return {
            "label": main_patch.label,
            "area": main_patch.area,
            "bbox": list(main_patch.bbox),
            "centroid": [round(main_patch.centroid[0], 3), round(main_patch.centroid[1], 3)],
            "looks_like_frame": main_patch.looks_like_frame,
        }

    @staticmethod
    def _valid_region_to_dict(
        valid_region_mask: Optional[np.ndarray],
        valid_region_bbox: Optional[tuple[int, int, int, int]],
    ) -> Optional[dict[str, Any]]:
        if valid_region_mask is None or valid_region_bbox is None:
            return None
        height, width = valid_region_mask.shape
        area = int(np.count_nonzero(valid_region_mask))
        return {
            "bbox": list(valid_region_bbox),
            "area": area,
            "area_ratio": round(float(area / max(1, height * width)), 4),
            "source": "first_layer_contour",
        }

    def _refine_selected_port_geometries(
        self,
        *,
        selected_ports: list[PatchPortCandidate],
        binary_mask: np.ndarray,
        skeleton_mask: np.ndarray,
        valid_region_mask: Optional[np.ndarray],
        original_image: Optional[np.ndarray],
        skip_refine_keys: Optional[set[tuple[tuple[int, int], str]]] = None,
        debug_dir: Optional[Path],
    ) -> tuple[list[PatchPortCandidate], list[dict[str, Any]]]:
        """Attach CST-oriented geometry metadata without moving the contact point.

        现有 CST 入口仍然消费 point/direction/local_width 这组三元组。
        因此 point 必须保持在馈线端面接触中心，不能写成外偏移后的 rectangle center；
        否则 waveguide port 会悬空，和导体断开。
        """

        builder = CSTPortGeometryBuilder()
        refined_ports: list[PatchPortCandidate] = []
        geometry_metadata: list[dict[str, Any]] = []
        geometry_debug_dir: Optional[Path] = None
        if debug_dir is not None:
            geometry_debug_dir = debug_dir.parent / "03_port_geometry" if debug_dir.name == "03_port_detection" else debug_dir / "03_port_geometry"

        for port in selected_ports:
            endpoint = port.point
            if skip_refine_keys is not None and (port.point, port.direction) in skip_refine_keys:
                refined_ports.append(port)
                geometry_metadata.append(
                    {
                        "port_id": port.id,
                        "endpoint": list(endpoint),
                        "raw_endpoint": list(endpoint),
                        "cst_contact_point": list(endpoint),
                        "feed_width": port.local_width,
                        "refined": False,
                        "reason": "parameterized_line_terminal_kept",
                    }
                )
                continue

            geometry = builder.build_port_geometry(
                endpoint=endpoint,
                border_side=port.direction,
                subject_mask=binary_mask,
                skeleton_mask=skeleton_mask,
                valid_region_mask=valid_region_mask,
            )
            if geometry is None:
                refined_ports.append(port)
                geometry_metadata.append(
                    {
                        "port_id": port.id,
                        "endpoint": list(endpoint),
                        "refined": False,
                        "reason": "geometry_builder_failed",
                    }
                )
                continue

            contact_point = endpoint
            if geometry.valid_region_limited:
                clipped_point = (int(round(float(geometry.center[0]))), int(round(float(geometry.center[1]))))
                if self._point_inside_mask(clipped_point, binary_mask) and self._point_inside_mask(clipped_point, valid_region_mask):
                    # 当截图外部空间被第一层轮廓裁掉时，CST 消费的 contact point
                    # 也要落回介质板边界内，并继续贴住馈线金属端面。
                    contact_point = clipped_point

            refined_score = min(20.0, port.score + geometry.confidence)
            refined_ports.append(
                PatchPortCandidate(
                    id=port.id,
                    point=contact_point,
                    direction=port.direction,
                    touches_border=port.touches_border,
                    local_width=round(float(geometry.feed_width), 3),
                    path_length_to_patch=port.path_length_to_patch,
                    connected_to_main_patch=port.connected_to_main_patch,
                    score=round(refined_score, 3),
                    confidence=round(max(port.confidence, geometry.confidence), 3),
                )
            )

            item = geometry.to_json_dict()
            item.update(
                {
                    "port_id": port.id,
                    "endpoint": list(contact_point),
                    "raw_endpoint": list(endpoint),
                    "cst_contact_point": list(contact_point),
                    "refined": True,
                    "original_local_width": port.local_width,
                }
            )
            geometry_metadata.append(item)

            if geometry_debug_dir is not None and port.id == selected_ports[0].id:
                debug_payload = {
                    "builder": "CSTPortGeometryBuilder",
                    "port_id": port.id,
                    "endpoint": list(contact_point),
                    "raw_endpoint": list(endpoint),
                    "cst_contact_point": list(contact_point),
                    "external_plane_center": [round(float(v), 3) for v in geometry.center],
                    "valid_region_limited": bool(geometry.valid_region_limited),
                    "geometry": item,
                    "thresholds": {
                        "trace_distance_px": builder.trace_distance_px,
                        "outward_offset_width_factor": builder.outward_offset_width_factor,
                        "min_outward_offset_px": builder.min_outward_offset_px,
                        "port_width_padding_factor": builder.port_width_padding_factor,
                        "port_height_factor": builder.port_height_factor,
                    },
                }
                builder.write_debug_outputs(
                    debug_dir=geometry_debug_dir,
                    original_image=original_image,
                    subject_mask=binary_mask,
                    valid_region_mask=valid_region_mask,
                    endpoint=contact_point,
                    geometry=geometry,
                    metadata=debug_payload,
                )

        return refined_ports, geometry_metadata

    def _write_debug_outputs(
        self,
        *,
        debug_dir: Path,
        original_image: Optional[np.ndarray],
        binary_mask: np.ndarray,
        skeleton_mask: np.ndarray,
        endpoint_mask: np.ndarray,
        valid_region_mask: Optional[np.ndarray],
        endpoints: list[tuple[int, int]],
        ports: list[PatchPortCandidate],
        main_patch: Optional[_MainPatchRegion],
        metadata: dict[str, Any],
    ) -> None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "skeleton.png"), skeleton_mask)
        if valid_region_mask is not None:
            cv2.imwrite(str(debug_dir / "valid_port_region.png"), valid_region_mask)

        endpoint_vis = self._base_overlay(original_image, binary_mask)
        if valid_region_mask is not None:
            contours, _ = cv2.findContours(valid_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(endpoint_vis, contours, -1, (255, 255, 0), 1)
        endpoint_vis[skeleton_mask > 0] = (255, 255, 255)
        for x, y in endpoints:
            cv2.circle(endpoint_vis, (int(x), int(y)), 3, (0, 255, 255), -1)
        cv2.imwrite(str(debug_dir / "endpoints.png"), endpoint_vis)

        selected_vis = self._base_overlay(original_image, binary_mask)
        if valid_region_mask is not None:
            contours, _ = cv2.findContours(valid_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(selected_vis, contours, -1, (255, 255, 0), 1)
        selected_vis[skeleton_mask > 0] = (255, 255, 255)
        for port in ports:
            cv2.circle(selected_vis, port.point, 5, (0, 0, 255), -1)
            cv2.putText(
                selected_vis,
                f"{port.id}:{port.direction}",
                (port.point[0] + 6, max(10, port.point[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        if main_patch is not None:
            cx, cy = main_patch.centroid
            cv2.circle(selected_vis, (int(round(cx)), int(round(cy))), 5, (255, 255, 0), -1)
        cv2.imwrite(str(debug_dir / "selected_ports.png"), selected_vis)

        feed_branch_analysis = metadata.get("feed_branch_analysis", [])
        selected_points = {tuple(port.point) for port in ports}
        if isinstance(feed_branch_analysis, list):
            branch_vis = self._base_overlay(original_image, binary_mask)
            if valid_region_mask is not None:
                contours, _ = cv2.findContours(valid_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(branch_vis, contours, -1, (255, 255, 0), 1)
            junction_vis = branch_vis.copy()
            for item in feed_branch_analysis:
                path = item.get("feed_centerline", []) if isinstance(item, dict) else []
                if len(path) >= 2:
                    points = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(branch_vis, [points], isClosed=False, color=(255, 255, 0), thickness=2)
                point = item.get("candidate_point") if isinstance(item, dict) else None
                if isinstance(point, list) and len(point) == 2:
                    color = (0, 0, 255) if tuple(point) in selected_points else (0, 165, 255)
                    cv2.circle(branch_vis, (int(point[0]), int(point[1])), 4, color, -1)
                junction = item.get("junction_point") if isinstance(item, dict) else None
                if isinstance(junction, list) and len(junction) == 2:
                    cv2.circle(junction_vis, (int(junction[0]), int(junction[1])), 5, (255, 0, 255), -1)
            cv2.imwrite(str(debug_dir / "feed_branch_paths.png"), branch_vis)
            cv2.imwrite(str(debug_dir / "junction_detection.png"), junction_vis)

            width_profiles = [
                {
                    "candidate_point": item.get("candidate_point"),
                    "candidate_direction": item.get("candidate_direction"),
                    "width_profile": item.get("width_profile", []),
                    "mean_width": item.get("mean_width"),
                    "width_std": item.get("width_std"),
                    "min_width": item.get("min_width"),
                    "max_width": item.get("max_width"),
                }
                for item in feed_branch_analysis
                if isinstance(item, dict)
            ]
            (debug_dir / "feed_width_profiles.json").write_text(
                json.dumps(width_profiles, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            em_scores = [
                {
                    "candidate_point": item.get("candidate_point"),
                    "candidate_direction": item.get("candidate_direction"),
                    "terminal_face_candidate": item.get("terminal_face_candidate"),
                    "base_score": item.get("base_score"),
                    "adjusted_score": item.get("adjusted_score"),
                    "em_validation": item.get("em_validation"),
                    "feed_aspect_ratio": item.get("feed_aspect_ratio"),
                    "branch_tortuosity": item.get("branch_tortuosity"),
                    "patch_transition_detected": item.get("patch_transition_detected"),
                }
                for item in feed_branch_analysis
                if isinstance(item, dict)
            ]
            (debug_dir / "em_port_scores.json").write_text(
                json.dumps(em_scores, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            rejected = [
                item
                for item in em_scores
                if isinstance(item.get("candidate_point"), list)
                and tuple(item["candidate_point"]) not in selected_points
            ]
            (debug_dir / "rejected_candidates.json").write_text(
                json.dumps(rejected, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        metadata["debug_outputs"] = {
            "skeleton": str(debug_dir / "skeleton.png"),
            "endpoints": str(debug_dir / "endpoints.png"),
            "selected_ports": str(debug_dir / "selected_ports.png"),
            "feed_branch_paths": str(debug_dir / "feed_branch_paths.png"),
            "feed_width_profiles": str(debug_dir / "feed_width_profiles.json"),
            "em_port_scores": str(debug_dir / "em_port_scores.json"),
            "junction_detection": str(debug_dir / "junction_detection.png"),
            "rejected_candidates": str(debug_dir / "rejected_candidates.json"),
            "port_debug": str(debug_dir / "port_debug.json"),
        }
        if valid_region_mask is not None:
            metadata["debug_outputs"]["valid_port_region"] = str(debug_dir / "valid_port_region.png")
        geometry_dir = debug_dir.parent / "03_port_geometry" if debug_dir.name == "03_port_detection" else debug_dir / "03_port_geometry"
        if geometry_dir.exists():
            metadata["debug_outputs"]["port_geometry"] = str(geometry_dir)
        (debug_dir / "port_debug.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _base_overlay(self, original_image: Optional[np.ndarray], binary_mask: np.ndarray) -> np.ndarray:
        if original_image is None:
            base = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        else:
            base = np.asarray(original_image).copy()
            if base.ndim == 2:
                base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            if base.dtype != np.uint8:
                base = np.clip(base, 0, 255).astype(np.uint8)
        return base


__all__ = [
    "PatchPortCandidate",
    "PatchPortDetectionResult",
    "PatchPortTopologyDetector",
]
