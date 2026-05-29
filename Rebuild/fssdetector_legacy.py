import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class FSSLegacyMixin:
    def auto_crop_color_from_mask(self, original_color_img, binary_mask, padding=None, padding_ratio=0.1):
        """
        根据二值掩码自动裁剪彩色图像，并保留一定留白。

        参数:
            original_color_img: 彩色图像，形状为 (H, W, 3)。
            binary_mask: 二值掩码，黑色视为前景，形状为 (H, W)。
            padding: 固定留白像素；若为 None，则使用 padding_ratio。
            padding_ratio: 自动留白比例，默认 10%。

        返回:
            裁剪后的彩色图像和对应掩码。
        """
        if len(binary_mask.shape) != 2:
            raise ValueError("binary_mask 必须是单通道图像")
        if original_color_img.shape[:2] != binary_mask.shape:
            raise ValueError("original_color_img and binary_mask must have the same shape")

        # 提取前景像素外接矩形。
        coords = cv2.findNonZero(255 - binary_mask)
        if coords is None:
            return original_color_img, binary_mask  # 全白图像时直接返回

        x, y, w, h = cv2.boundingRect(coords)

        if padding is None:
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)
        else:
            pad_x = pad_y = padding

        # 计算裁剪区域坐标。
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, original_color_img.shape[1])
        y2 = min(y + h + pad_y, original_color_img.shape[0])

        # 裁剪彩色图像和掩码。
        cropped_color = original_color_img[y1:y2, x1:x2]
        cropped_mask = binary_mask[y1:y2, x1:x2]
        return cropped_color, cropped_mask

    def visualize_color_clusters(self, original_img, labels, centers, output_dir="testcluster"):
        """
        在白色背景上展示各个聚类颜色及其空间分布。

        参数:
            original_img: 原始图像，RGB 格式。
            labels: 聚类标签数组。
            centers: 聚类中心数组。
            output_dir: 输出目录。
        """
        # 创建输出目录。
        os.makedirs(output_dir, exist_ok=True)

        # 获取图像尺寸。
        height, width, _ = original_img.shape

        # 创建白底画布。
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 为每个聚类生成独立可视化图像。
        cluster_visualizations = []
        k = len(centers)

        for i in range(k):
            # 当前聚类的掩码。
            mask = (labels == i).reshape(height, width)

            # 在白底上只保留当前聚类。
            cluster_img = white_bg.copy()
            cluster_img[mask] = centers[i]

            # 保留原变量，兼容后续可能的调试扩展。
            output_path = os.path.join(output_dir, f"cluster_{i + 1}.jpg")
            _ = output_path

            cluster_visualizations.append(cluster_img)

        # 生成总览图。
        fig, axes = plt.subplots(1, k + 1, figsize=(15, 10))
        axes = axes.flatten()

        # 显示原图。
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # 显示各聚类结果。
        for i in range(k):
            axes[i + 1].imshow(cluster_visualizations[i])
            axes[i + 1].set_title(f"Cluster {i + 1}")
            axes[i + 1].axis("off")

        # 隐藏多余子图。
        for j in range(k + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_clusters.jpg"))
        plt.close()

        return cluster_visualizations

    def classify_colors(self, color_list, bg_color, clusters_images, output_dir="testcluster"):
        """
        对聚类颜色做角色划分，确定轮廓、填充、背景和主体索引。
        """
        # 将 color_list 转为 numpy 数组便于计算。
        colors = np.array(color_list, dtype=np.float32)

        # 目标参考颜色。
        black = np.array([0, 0, 0], dtype=np.float32)
        white = np.array([255, 255, 255], dtype=np.float32)
        bg = np.array(bg_color, dtype=np.float32)

        # 计算各聚类中心到参考颜色的距离。
        dist_to_black = np.linalg.norm(colors - black, axis=1)
        dist_to_white = np.linalg.norm(colors - white, axis=1)
        dist_to_bg = np.linalg.norm(colors - bg, axis=1)

        # 距离黑色最近的通常视为轮廓颜色。
        outline_index = int(np.argmin(dist_to_black))
        padding_index = int(np.argmin(dist_to_white))
        bg_index = int(np.argmin(dist_to_bg))

        # 其余索引视为主体候选。
        all_indices = set(range(len(color_list)))
        used_indices = set([outline_index, padding_index, bg_index])
        fig_index = list(all_indices - used_indices)

        # 防止主体索引为空。
        if not fig_index:
            fig_index.append(bg_index)

        # 查看轮廓簇在整图中的面积占比。
        outline_img = clusters_images[outline_index]
        total_pixels = outline_img.shape[0] * outline_img.shape[1]

        # 统计轮廓簇的非白像素比例。
        mask_outline = ~(np.all(outline_img == [255, 255, 255], axis=-1))
        non_white_count_outline = np.count_nonzero(mask_outline)
        ratio_outline = non_white_count_outline / total_pixels

        # 统计主体簇的非白像素比例。
        mask_fig = ~(np.all(clusters_images[fig_index[0]] == [255, 255, 255], axis=-1))
        non_white_count_fig = np.count_nonzero(mask_fig)
        ratio_fig = non_white_count_fig / total_pixels

        mask_bg = ~(np.all(clusters_images[bg_index] == [255, 255, 255], axis=-1))
        non_white_count_outline = np.count_nonzero(mask_bg)
        ratio_bg = non_white_count_outline / total_pixels

        # 如果轮廓面积很大而主体很小，则交换两者。
        if ratio_outline >= 0.7 and ratio_fig < 0.1:
            print("交换 outline 与 fig")
            _ = fig_index
            fig_index = [outline_index]
            outline_index = _[0]

        # 如果轮廓和主体都很大，则将轮廓并入主体。
        if ratio_outline >= 0.25 and ratio_fig >= 0.25:
            print("合并 outline 与 fig")
            fig_index.append(outline_index)
        diff_bg_white = np.linalg.norm(bg_color - white, axis=0)
        if diff_bg_white > 20 and ratio_bg >= 0.1:
            print("背景颜色不是纯白且占比较大，加入主体候选")
            fig_index.append(bg_index)

        # 保存各类别对应的聚类图，便于调试。
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(output_dir, f"outline_index_{color_list[outline_index]}.png"),
            cv2.cvtColor(clusters_images[outline_index], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, f"bg_index_{color_list[bg_index]}.png"),
            cv2.cvtColor(clusters_images[bg_index], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, f"padding_index_{color_list[padding_index]}.png"),
            cv2.cvtColor(clusters_images[padding_index], cv2.COLOR_RGB2BGR),
        )

        for i, idx in enumerate(fig_index):
            cv2.imwrite(
                os.path.join(output_dir, f"fig_index_{i + 1:03d}_{color_list[idx]}.png"),
                cv2.cvtColor(clusters_images[idx], cv2.COLOR_RGB2BGR),
            )

        return outline_index, padding_index, bg_index, fig_index

    def get_max_contour_rect(self, image):
        """
        查找图像中最大轮廓的外接矩形。
        若没有轮廓，则返回图像向内收缩 10 像素后的矩形。

        参数:
            image: RGB 图像，形状为 (H, W, 3)，支持 uint8/float32/float64。

        返回:
            tuple: 矩形坐标 `(x, y, width, height)`。
        """
        # 检查输入图像格式。
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("image must be an RGB image with shape (H, W, 3)")

        H, W = image.shape[0], image.shape[1]  # 图像高和宽

        # 1. 预处理图像，统一转为 uint8。
        if image.dtype in (np.float32, np.float64):
            # float 图像映射到 0-255。
            img_uint8 = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint8:
            img_uint8 = image.copy()
        else:
            raise TypeError("image dtype must be uint8, float32, or float64")

        # 纯白视为背景，其余像素视为前景。
        is_white = (
            (img_uint8[:, :, 0] == 255)
            & (img_uint8[:, :, 1] == 255)
            & (img_uint8[:, :, 2] == 255)
        )
        binary = np.where(is_white, 0, 255).astype(np.uint8)

        # 2. 查找外轮廓。
        # OpenCV 4.x 中 findContours 返回 (contours, hierarchy)。
        contours, hierarchy = cv2.findContours(
            binary,
            mode=cv2.RETR_EXTERNAL,  # 只保留外轮廓
            method=cv2.CHAIN_APPROX_SIMPLE,  # 简化轮廓点
        )
        _ = hierarchy

        # 3. 根据是否有轮廓返回结果。
        if len(contours) == 0:
            # 无轮廓时返回向内收缩 10 像素后的矩形。
            shrink = 10
            x = max(0, shrink)
            y = max(0, shrink)
            width = max(0, W - 2 * shrink)
            height = max(0, H - 2 * shrink)
            return (x, y, width, height)
        else:
            # 选择面积最大的轮廓并返回外接矩形。
            max_area = -1
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour
            x, y, width, height = cv2.boundingRect(max_contour)
            return (x, y, width, height)

    def _parse_box_to_xyxy(self, box, img_w, img_h, box_format="xyxy"):
        """
        将单个框转换为裁剪后的整数 `xyxy` 坐标。

        支持输入:
        - `[x1, y1, x2, y2]`，`box_format='xyxy'`
        - `[x, y, w, h]`，`box_format='xywh'`
        - `{'x1','y1','x2','y2'}`
        - `{'xyxy': [...]}`
        - `{'bbox': [...], 'box_format': 'xyxy'|'xywh'}`
        """
        coords = None
        local_format = box_format

        if isinstance(box, dict):
            if "xyxy" in box and len(box["xyxy"]) == 4:
                coords = box["xyxy"]
                local_format = "xyxy"
            elif all(k in box for k in ("x1", "y1", "x2", "y2")):
                coords = [box["x1"], box["y1"], box["x2"], box["y2"]]
                local_format = "xyxy"
            elif "bbox" in box and len(box["bbox"]) == 4:
                coords = box["bbox"]
                local_format = box.get("box_format", box_format)
            else:
                raise ValueError(f"Unsupported box dict format: {box}")
        else:
            if not hasattr(box, "__len__") or len(box) != 4:
                raise ValueError(f"Box must contain 4 numbers, got: {box}")
            coords = box

        x1, y1, x2, y2 = [float(v) for v in coords]
        if local_format.lower() == "xywh":
            x2 = x1 + x2
            y2 = y1 + y2

        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        x_min = int(np.clip(np.floor(x_min), 0, img_w - 1))
        y_min = int(np.clip(np.floor(y_min), 0, img_h - 1))
        x_max = int(np.clip(np.ceil(x_max), 1, img_w))
        y_max = int(np.clip(np.ceil(y_max), 1, img_h))

        if x_max <= x_min + 1 or y_max <= y_min + 1:
            return None
        return x_min, y_min, x_max, y_max

    def _normalize_detection_label(self, label):
        """
        统一检测类别名格式。
        """
        return str(label).strip().lower()

    def _skeletonize_binary_mask(self, mask):
        binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
        if np.count_nonzero(binary) == 0:
            return binary

        ximgproc = getattr(cv2, "ximgproc", None)
        thinning = getattr(ximgproc, "thinning", None) if ximgproc is not None else None
        if thinning is not None:
            return thinning(binary)

        skeleton = np.zeros_like(binary)
        work = binary.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while np.count_nonzero(work) > 0:
            eroded = cv2.erode(work, kernel)
            opened = cv2.dilate(eroded, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
            work = eroded
        return skeleton

    def compute_component_geometry(self, component_mask, x, y, w, h, patch_shape, contrast_value=0.0):
        """
        Compute geometry features used to distinguish annotation pixels from FSS structure.
        """
        area = float(np.count_nonzero(component_mask))
        ph, pw = patch_shape[:2]
        patch_area = float(max(1, ph * pw))
        bbox_area = float(max(1, w * h))
        fill_ratio = area / bbox_area
        aspect_ratio = w / float(max(1, h))
        elongation = max(w, h) / float(max(1, min(w, h)))
        touches_border = (x <= 1) or (y <= 1) or (x + w >= pw - 1) or (y + h >= ph - 1)

        contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = float(max(1.0, cv2.contourArea(hull)))
            contour_area = float(max(area, cv2.contourArea(contour)))
            solidity = contour_area / hull_area

        skeleton = self._skeletonize_binary_mask(component_mask.astype(np.uint8) * 255)
        skeleton_length = float(np.count_nonzero(skeleton))
        stroke_width = area / max(1.0, skeleton_length)

        return {
            "area": area,
            "area_ratio": area / patch_area,
            "bbox_area": bbox_area,
            "fill_ratio": fill_ratio,
            "aspect_ratio": aspect_ratio,
            "elongation": elongation,
            "solidity": solidity,
            "skeleton_length": skeleton_length,
            "stroke_width": stroke_width,
            "touches_border": touches_border,
            "min_side": min(w, h),
            "max_side": max(w, h),
            "contrast": float(contrast_value),
        }

    def classify_annotation_component(self, geometry, target_kind="arrow"):
        """
        Conservative component classifier. False negatives are preferred over destructive masks.
        Returns: "keep", "fallback", or "reject".
        """
        area_ratio = geometry["area_ratio"]
        fill_ratio = geometry["fill_ratio"]
        elongation = geometry["elongation"]
        solidity = geometry["solidity"]
        stroke_width = geometry["stroke_width"]
        skeleton_length = geometry["skeleton_length"]
        touches_border = geometry["touches_border"]
        contrast = geometry["contrast"]
        min_side = geometry["min_side"]

        # Large, solid components are much more likely to be FSS structure.
        if area_ratio >= 0.55 and not touches_border:
            return "reject"
        if area_ratio >= 0.22 and solidity >= 0.88 and fill_ratio >= 0.72 and not touches_border:
            return "reject"
        if contrast < 10.0 and area_ratio >= 0.04:
            return "reject"

        if target_kind == "line":
            score = 0
            if elongation >= 5.0:
                score += 3
            elif elongation >= 3.0:
                score += 2
            if stroke_width <= 3.2:
                score += 2
            elif stroke_width <= 5.0:
                score += 1
            if fill_ratio <= 0.45:
                score += 1
            if area_ratio <= 0.18:
                score += 1
            if skeleton_length >= 8:
                score += 1
            if touches_border:
                score += 1
            if score >= 5:
                return "keep"
            if score >= 4 and area_ratio <= 0.30:
                return "fallback"
            return "reject"

        if target_kind == "text":
            score = 0
            if area_ratio <= 0.16:
                score += 2
            elif area_ratio <= 0.28:
                score += 1
            if stroke_width <= 5.2:
                score += 1
            if fill_ratio <= 0.78:
                score += 1
            if min_side <= 18:
                score += 1
            if contrast >= 16.0:
                score += 1
            if solidity <= 0.90:
                score += 1
            if score >= 4:
                return "keep"
            if score >= 3 and area_ratio <= 0.22:
                return "fallback"
            return "reject"

        # Arrows can contain a thin shaft plus a small solid head.
        score = 0
        if area_ratio <= 0.28:
            score += 2
        elif area_ratio <= 0.45:
            score += 1
        if elongation >= 2.0:
            score += 1
        if stroke_width <= 6.5:
            score += 2
        elif stroke_width <= 9.0:
            score += 1
        if fill_ratio <= 0.78:
            score += 1
        if contrast >= 16.0:
            score += 1
        if touches_border:
            score += 1
        if score >= 4:
            return "keep"
        if score >= 3 and area_ratio <= 0.35:
            return "fallback"
        return "reject"

    def _build_candidate_mask_in_patch(self, patch_bgr, border_width=2, target_kind="arrow"):
        """
        在检测框内部生成候选掩码。
        候选掩码仅表示“可能属于目标图元的像素”。
        """
        ph, pw = patch_bgr.shape[:2]
        if ph < 4 or pw < 4:
            return np.zeros((ph, pw), dtype=np.uint8)

        ring = max(1, min(border_width, min(ph, pw) // 3))
        border_mask = np.zeros((ph, pw), dtype=np.uint8)
        border_mask[:ring, :] = 1
        border_mask[-ring:, :] = 1
        border_mask[:, :ring] = 1
        border_mask[:, -ring:] = 1

        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        border_gray = gray[border_mask == 1]
        local_bg_gray = float(np.median(border_gray)) if border_gray.size > 0 else float(np.median(gray))
        gray_f = gray.astype(np.float32)
        dark_delta = local_bg_gray - gray_f
        bright_delta = gray_f - local_bg_gray
        dark_threshold = max(16.0, 0.12 * max(1.0, local_bg_gray))
        bright_threshold = max(28.0, 0.18 * max(1.0, 255.0 - local_bg_gray))
        dark_mask = np.where((dark_delta >= dark_threshold) | (gray <= 125), 255, 0).astype(np.uint8)
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        low_saturation = hsv[:, :, 1] <= 55
        bright_mask = np.where(
            (bright_delta >= bright_threshold) & (gray >= 190) & low_saturation,
            255,
            0,
        ).astype(np.uint8)

        edges = cv2.Canny(gray, 45, 135)
        edge_near_dark = cv2.bitwise_and(
            cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1),
            dark_mask,
        )
        candidate = cv2.bitwise_or(dark_mask, edge_near_dark)

        if target_kind == "text":
            # 文字更接近“黑色笔画”，而不是单纯的细长几何结构。
            # 因此这里专门补一套面向黑字的候选像素提取逻辑：
            # 1. 用边缘像素估计局部背景亮度；
            # 2. 找出显著比局部背景更暗的像素；
            # 3. 再叠加自适应阈值，把抗锯齿边缘和细笔画补回来。
            if border_gray.size > 0:
                local_bg_gray = float(np.median(border_gray))
            else:
                local_bg_gray = float(np.median(gray))

            dark_delta = local_bg_gray - gray.astype(np.float32)
            dark_threshold = max(12.0, 0.10 * local_bg_gray)
            text_dark_mask = np.where(dark_delta >= dark_threshold, 255, 0).astype(np.uint8)

            adaptive_text_mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                15,
                4,
            )

            candidate = cv2.bitwise_or(candidate, text_dark_mask)
            candidate = cv2.bitwise_or(candidate, bright_mask)
            candidate = cv2.bitwise_or(candidate, adaptive_text_mask)

            # 文字笔画相互距离较近，但又不能闭得太狠，否则多个字符会糊成整块。
            text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, text_kernel, iterations=1)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, text_kernel, iterations=1)

        elif target_kind == "line":
            # 对线段类别额外引入 Hough 直线结果，提升细长结构的召回率。
            line_mask = np.zeros((ph, pw), dtype=np.uint8)
            line_threshold = max(12, int(min(ph, pw) * 0.18))
            min_line_length = max(10, int(max(ph, pw) * 0.35))
            max_line_gap = max(2, int(min(ph, pw) * 0.08))
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180.0,
                threshold=line_threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap,
            )
            if lines is not None:
                for line in lines[:, 0, :]:
                    x1, y1, x2, y2 = [int(v) for v in line]
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
                candidate = cv2.bitwise_and(
                    candidate,
                    cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1),
                )
                candidate = cv2.bitwise_or(candidate, line_mask)
            candidate = self._skeletonize_binary_mask(candidate)

        elif target_kind == "arrow":
            candidate = cv2.bitwise_or(
                candidate,
                cv2.bitwise_and(edges, cv2.dilate(dark_mask, np.ones((3, 3), np.uint8), iterations=1)),
            )

        kernel = np.ones((3, 3), np.uint8)
        if target_kind == "line":
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
        return candidate

    def _refine_target_mask_in_patch(
        self,
        patch_bgr,
        target_kind="arrow",
        border_width=2,
        min_component_area=8,
    ):
        """
        在单个候选区域内细化目标图元掩码。
        返回 patch 坐标系下的二值掩码，255 表示需要去除。
        """
        ph, pw = patch_bgr.shape[:2]
        refined_mask = np.zeros((ph, pw), dtype=np.uint8)
        if ph < 4 or pw < 4:
            return refined_mask

        candidate = self._build_candidate_mask_in_patch(
            patch_bgr=patch_bgr,
            border_width=border_width,
            target_kind=target_kind,
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        fallback_labels = []
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        ring = max(1, min(border_width, min(ph, pw) // 3))
        border_mask = np.zeros((ph, pw), dtype=np.uint8)
        border_mask[:ring, :] = 1
        border_mask[-ring:, :] = 1
        border_mask[:, :ring] = 1
        border_mask[:, -ring:] = 1
        border_gray = gray[border_mask == 1]
        local_bg_gray = float(np.median(border_gray)) if border_gray.size > 0 else float(np.median(gray))

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = float(stats[i, cv2.CC_STAT_AREA])

            if area < min_component_area:
                continue

            component = labels == i
            component_gray = gray[component]
            contrast_value = abs(float(np.median(component_gray)) - local_bg_gray) if component_gray.size else 0.0
            geometry = self.compute_component_geometry(
                component_mask=component,
                x=x,
                y=y,
                w=w,
                h=h,
                patch_shape=patch_bgr.shape,
                contrast_value=contrast_value,
            )
            decision = self.classify_annotation_component(geometry, target_kind=target_kind)
            if decision == "keep":
                refined_mask[component] = 255
            elif decision == "fallback":
                fallback_labels.append(i)

        # 严格规则没命中时，回退到较弱条件。
        if np.count_nonzero(refined_mask) == 0 and fallback_labels:
            for i in fallback_labels:
                refined_mask[labels == i] = 255

        if target_kind == "line":
            refined_mask = self._skeletonize_binary_mask(refined_mask)
            dilate_size = 2
            dilate_iter = 1
        elif target_kind == "text":
            dilate_size = 2
            dilate_iter = 1
        else:
            dilate_size = 2
            dilate_iter = 1
        refined_mask = cv2.dilate(
            refined_mask,
            np.ones((dilate_size, dilate_size), np.uint8),
            iterations=dilate_iter,
        )
        return refined_mask

    def _refine_arrow_mask_in_patch(self, patch_bgr, border_width=2, min_component_area=8):
        """
        在单个候选区域内细化箭头掩码。
        """
        return self._refine_target_mask_in_patch(
            patch_bgr=patch_bgr,
            target_kind="arrow",
            border_width=border_width,
            min_component_area=min_component_area,
        )

    def _refine_line_mask_in_patch(self, patch_bgr, border_width=2, min_component_area=8):
        """
        在单个候选区域内细化线段掩码。
        """
        return self._refine_target_mask_in_patch(
            patch_bgr=patch_bgr,
            target_kind="line",
            border_width=border_width,
            min_component_area=min_component_area,
        )

    def _refine_text_mask_in_patch(self, patch_bgr, border_width=2, min_component_area=3):
        """
        在单个候选区域内细化文字掩码。

        单独拆出这个函数，是为了让“文字像素提取”与“箭头/线段像素提取”
        在结构上完全分开。你后续如果只想调黑字阈值、连通域面积或者
        字符膨胀强度，直接改这个入口就行，不会影响另外两类规则。
        """
        return self._refine_target_mask_in_patch(
            patch_bgr=patch_bgr,
            target_kind="text",
            border_width=border_width,
            min_component_area=min_component_area,
        )

    def build_arrow_mask_from_boxes(
        self,
        img_bgr,
        arrow_boxes,
        box_format="xyxy",
        expand_ratio=0.08,
        expand_pixels=2,
        border_width=2,
        min_component_area=8,
    ):
        """
        根据箭头检测框构建整图掩码。
        输出掩码中 255 表示需要移除的箭头像素。
        """
        detections = []
        if arrow_boxes is not None:
            for box in arrow_boxes:
                detections.append({
                    "bbox": box,
                    "box_format": box_format,
                    "class_name": "arrow",
                })
        return self.build_detection_mask_from_detections(
            img_bgr=img_bgr,
            detections=detections,
            expand_ratio=expand_ratio,
            expand_pixels=expand_pixels,
            border_width=border_width,
            min_component_area=min_component_area,
        )

    def build_detection_mask_from_detections(
        self,
        img_bgr,
        detections,
        box_format="xyxy",
        expand_ratio=0.08,
        expand_pixels=2,
        border_width=2,
        min_component_area=8,
    ):
        """
        根据检测结果构建整图掩码。
        支持按检测类别分别细化 `arrow` 与 `line` 像素。
        """
        if img_bgr is None:
            raise ValueError("img_bgr cannot be None")
        if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
            raise ValueError("img_bgr must be a 3-channel image")

        img_h, img_w = img_bgr.shape[:2]
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if detections is None:
            return full_mask

        for detection in detections:
            local_box_format = box_format
            if isinstance(detection, dict):
                local_box_format = detection.get("box_format", box_format)

            parsed = self._parse_box_to_xyxy(
                detection,
                img_w=img_w,
                img_h=img_h,
                box_format=local_box_format,
            )
            if parsed is None:
                continue
            x1, y1, x2, y2 = parsed

            bw = x2 - x1
            bh = y2 - y1
            pad = max(expand_pixels, int(max(bw, bh) * expand_ratio))

            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(img_w, x2 + pad)
            ry2 = min(img_h, y2 + pad)
            if rx2 <= rx1 + 1 or ry2 <= ry1 + 1:
                continue

            patch = img_bgr[ry1:ry2, rx1:rx2]
            class_name = "arrow"
            if isinstance(detection, dict):
                class_name = self._normalize_detection_label(detection.get("class_name", "arrow"))

            if class_name == "text":
                patch_mask = self._refine_text_mask_in_patch(
                    patch_bgr=patch,
                    border_width=border_width,
                    min_component_area=min_component_area,
                )
            elif class_name == "line":
                patch_mask = self._refine_line_mask_in_patch(
                    patch_bgr=patch,
                    border_width=border_width,
                    min_component_area=min_component_area,
                )
            else:
                patch_mask = self._refine_arrow_mask_in_patch(
                    patch_bgr=patch,
                    border_width=border_width,
                    min_component_area=min_component_area,
                )

            roi = full_mask[ry1:ry2, rx1:rx2]
            full_mask[ry1:ry2, rx1:rx2] = np.maximum(roi, patch_mask)

        return full_mask

    def _estimate_patch_background_color(self, patch_bgr, patch_mask=None, border_width=2):
        """
        使用检测框边缘估计局部背景颜色和背景稳定性。
        """
        ph, pw = patch_bgr.shape[:2]
        ring = max(1, min(border_width, min(ph, pw) // 3))

        border_mask = np.zeros((ph, pw), dtype=np.uint8)
        border_mask[:ring, :] = 1
        border_mask[-ring:, :] = 1
        border_mask[:, :ring] = 1
        border_mask[:, -ring:] = 1

        valid_border = border_mask == 1
        if patch_mask is not None:
            valid_border = np.logical_and(valid_border, patch_mask == 0)

        border_pixels = patch_bgr[valid_border]
        if border_pixels.size == 0:
            border_pixels = patch_bgr[border_mask == 1]
        if border_pixels.size == 0:
            bg_color = np.array([255, 255, 255], dtype=np.uint8)
            bg_std = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return bg_color, bg_std

        bg_color = np.median(border_pixels, axis=0).astype(np.uint8)
        bg_std = np.std(border_pixels.astype(np.float32), axis=0)
        return bg_color, bg_std

    def _remove_target_from_patch(
        self,
        patch_bgr,
        patch_mask,
        global_background_color=None,
        border_width=2,
        target_kind="arrow",
    ):
        """
        根据局部背景与整体背景关系处理待移除区域。

        规则:
            1. 若局部背景稳定，且与传入整体背景色接近，则直接填充整体背景色。
            2. 若局部背景与整体背景不接近，则改为白色，并标记后续修补 mask。
            3. 若局部背景不稳定，也默认改为白色，并标记后续修补 mask。
        """
        if np.count_nonzero(patch_mask) == 0:
            return patch_bgr.copy(), np.zeros_like(patch_mask), np.zeros_like(patch_mask)

        cleaned_patch = patch_bgr.copy()
        bg_color, bg_std = self._estimate_patch_background_color(
            patch_bgr=patch_bgr,
            patch_mask=patch_mask,
            border_width=border_width,
        )
        bg_is_stable = float(np.max(bg_std)) <= 18.0
        if target_kind == "line":
            soft_mask = cv2.dilate(patch_mask, np.ones((2, 2), np.uint8), iterations=1)
        else:
            soft_mask = cv2.dilate(patch_mask, np.ones((2, 2), np.uint8), iterations=1)

        global_bg = None
        if global_background_color is not None:
            global_bg = np.asarray(global_background_color, dtype=np.float32).reshape(-1)
            if global_bg.size == 3:
                global_bg = global_bg.astype(np.uint8)
            else:
                global_bg = None

        context_kernel = np.ones((5, 5), np.uint8)
        context_ring = cv2.dilate(soft_mask, context_kernel, iterations=2)
        context_ring = cv2.subtract(context_ring, soft_mask)
        context_pixels = patch_bgr[context_ring > 0]
        if context_pixels.size > 0:
            context_color = np.median(context_pixels, axis=0).astype(np.uint8)
            context_std = np.std(context_pixels.astype(np.float32), axis=0)
            if float(np.max(context_std)) <= 42.0 or target_kind in ("line", "text"):
                cleaned_patch[soft_mask > 0] = context_color
                return cleaned_patch, soft_mask, soft_mask

        if bg_is_stable and global_bg is not None:
            color_distance = float(np.linalg.norm(bg_color.astype(np.float32) - global_bg.astype(np.float32)))
            if color_distance <= 28.0:
                cleaned_patch[soft_mask > 0] = global_bg
                return cleaned_patch, soft_mask, soft_mask

        cleaned_patch[soft_mask > 0] = 255
        return cleaned_patch, soft_mask, soft_mask

    def draw_yolo_detections(self, img_bgr, detections, mask=None):
        """
        绘制 YOLO 检测框与类别标签，便于输出调试图。
        """
        vis_img = img_bgr.copy()

        if mask is not None and np.count_nonzero(mask) > 0:
            overlay = vis_img.copy()
            overlay[mask > 0] = (0, 0, 255)
            vis_img = cv2.addWeighted(overlay, 0.28, vis_img, 0.72, 0)

        color_map = {
            "arrow": (0, 140, 255),
            "line": (255, 170, 0),
            "text": (0, 220, 120),
        }

        if detections is None:
            return vis_img

        img_h, img_w = vis_img.shape[:2]
        for detection in detections:
            parsed = self._parse_box_to_xyxy(detection, img_w=img_w, img_h=img_h, box_format="xyxy")
            if parsed is None:
                continue
            x1, y1, x2, y2 = parsed
            class_name = self._normalize_detection_label(detection.get("class_name", "target"))
            score = float(detection.get("score", 0.0))
            color = color_map.get(class_name, (0, 255, 0))
            label = f"{class_name}:{score:.2f}"

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            text_origin_y = y1 - 8 if y1 >= 24 else y1 + 20
            cv2.putText(
                vis_img,
                label,
                (x1, text_origin_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return vis_img

    def remove_arrows_by_boxes(
        self,
        img_bgr,
        arrow_boxes,
        box_format="xyxy",
        global_background_color=None,
        expand_ratio=0.08,
        expand_pixels=2,
        border_width=2,
        min_component_area=8,
    ):
        """
        根据检测框移除箭头像素。
        返回清理后的图像以及箭头掩码。
        """
        detections = []
        if arrow_boxes is not None:
            for box in arrow_boxes:
                detections.append({
                    "bbox": box,
                    "box_format": box_format,
                    "class_name": "arrow",
                })
        return self.remove_yolo_targets_by_detections(
            img_bgr=img_bgr,
            detections=detections,
            global_background_color=global_background_color,
            expand_ratio=expand_ratio,
            expand_pixels=expand_pixels,
            border_width=border_width,
            min_component_area=min_component_area,
        )

    def remove_yolo_targets_by_detections(
        self,
        img_bgr,
        detections,
        box_format="xyxy",
        global_background_color=None,
        expand_ratio=0.08,
        expand_pixels=2,
        border_width=2,
        min_component_area=8,
    ):
        """
        根据 YOLO 检测结果移除指定目标像素。
        返回:
            cleaned_img: 预清理后的图像
            repair_mask: 需要交给后续修补流程的区域
            removed_mask: 本轮被移除的全部区域
        """
        if img_bgr is None:
            raise ValueError("img_bgr cannot be None")
        if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
            raise ValueError("img_bgr must be a 3-channel image")

        img_h, img_w = img_bgr.shape[:2]
        cleaned = img_bgr.copy()
        repair_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        removed_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if detections is None:
            return cleaned, repair_mask, removed_mask

        sorted_detections = list(detections)
        sorted_detections.sort(
            key=lambda item: (
                -((item.get("xyxy", item.get("bbox", [0, 0, 0, 0]))[2] - item.get("xyxy", item.get("bbox", [0, 0, 0, 0]))[0])
                * (item.get("xyxy", item.get("bbox", [0, 0, 0, 0]))[3] - item.get("xyxy", item.get("bbox", [0, 0, 0, 0]))[1]))
            )
        )

        for detection in sorted_detections:
            local_box_format = detection.get("box_format", box_format) if isinstance(detection, dict) else box_format
            parsed = self._parse_box_to_xyxy(
                detection,
                img_w=img_w,
                img_h=img_h,
                box_format=local_box_format,
            )
            if parsed is None:
                continue
            x1, y1, x2, y2 = parsed

            bw = x2 - x1
            bh = y2 - y1
            pad = max(expand_pixels, int(max(bw, bh) * expand_ratio))

            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(img_w, x2 + pad)
            ry2 = min(img_h, y2 + pad)
            if rx2 <= rx1 + 1 or ry2 <= ry1 + 1:
                continue

            patch = cleaned[ry1:ry2, rx1:rx2]
            class_name = self._normalize_detection_label(detection.get("class_name", "arrow")) if isinstance(detection, dict) else "arrow"

            if class_name == "text":
                patch_mask = self._refine_text_mask_in_patch(
                    patch_bgr=patch,
                    border_width=border_width,
                    min_component_area=max(1, min_component_area),
                )
            elif class_name == "line":
                patch_mask = self._refine_line_mask_in_patch(
                    patch_bgr=patch,
                    border_width=border_width,
                    min_component_area=min_component_area,
                )
            else:
                patch_mask = self._refine_arrow_mask_in_patch(
                    patch_bgr=patch,
                    border_width=border_width,
                    min_component_area=min_component_area,
                )

            if np.count_nonzero(patch_mask) == 0:
                continue

            cleaned_patch, patch_repair_mask, patch_removed_mask = self._remove_target_from_patch(
                patch_bgr=patch,
                patch_mask=patch_mask,
                global_background_color=global_background_color,
                border_width=border_width,
                target_kind=class_name,
            )
            repair_roi = repair_mask[ry1:ry2, rx1:rx2]
            repair_mask[ry1:ry2, rx1:rx2] = np.maximum(repair_roi, patch_repair_mask)
            removed_roi = removed_mask[ry1:ry2, rx1:rx2]
            removed_mask[ry1:ry2, rx1:rx2] = np.maximum(removed_roi, patch_removed_mask)
            cleaned[ry1:ry2, rx1:rx2] = cleaned_patch

        return cleaned, repair_mask, removed_mask

    def repair_img(self, color, binary_img):
        """
        根据二值线条图恢复彩色结果。
        """
        h, w = binary_img.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)  # 输出 RGB 图像
        result = 255 - result
        kernel_erode = np.ones((1, 1), np.uint8)
        red_mask = cv2.erode(binary_img, kernel_erode, iterations=3)

        # 先用闭运算连接断裂线条，同时尽量避免线条明显变粗。
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close, iterations=6)

        # 生成需要修补的缝隙掩码。
        mask = cv2.subtract(closed, red_mask)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        # 转为三通道后使用 inpaint 进行修补。
        red_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        repaired = cv2.inpaint(red_rgb, mask, 3, cv2.INPAINT_TELEA)

        # 提取修补后的线条区域，并恢复目标颜色。
        repaired_gray = cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY)
        _, final_mask = cv2.threshold(repaired_gray, 127, 255, cv2.THRESH_BINARY)

        result[final_mask == 255] = color

        return result
