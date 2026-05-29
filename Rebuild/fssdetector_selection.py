import os
import re
import sys

import cv2
import numpy as np

try:
    from . import Preprocess_tool
except ImportError:
    import Preprocess_tool


class FSSSelectionMixin:
    def is_regular_shape(self, contour):
        """判断轮廓是否为规则形状，如矩形、正方形或圆形。"""
        area = cv2.contourArea(contour)
        if area < 100:  # 排除很小的噪声轮廓
            return False

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter < 30:  # 排除周长过短的轮廓
            return False

        # 1. 判断是否接近矩形或正方形：四条边并且角度接近直角
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        vertices = len(approx)

        if vertices == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            right_angle_count = 0
            for i in range(4):
                vec1 = pts[(i + 1) % 4] - pts[i]
                vec2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
                if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
                    continue
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)
                dot_product = np.dot(vec1_norm, vec2_norm)
                if abs(dot_product) < 0.3:
                    right_angle_count += 1
            if right_angle_count == 4:
                return True

        # 2. 用圆度判断是否接近圆形
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.6:
            return True

        return False

    def is_axis_like_layout(self, text_centers, sub_text_infos, x_min, x_max, y_min, y_max):
        """
        判断一组文本是否更像坐标轴标注，而不是目标子图内容。

        text_centers: [(x_center, y_center), ...]，图像绝对坐标
        sub_text_infos: 子图 OCR 结果，用于判断是否包含数字
        返回值: True / False
        """
        if len(text_centers) < 5:
            return False

        # 只保留包含数字的文本中心点，因为坐标轴刻度通常以数字为主
        numeric_texts = [
            (x, y)
            for (x, y), info in zip(text_centers, sub_text_infos)
            if re.search(r"\d", info["text"])
        ]

        if len(numeric_texts) < 5:
            return False

        xs = np.array([x for x, y in numeric_texts])
        ys = np.array([y for x, y in numeric_texts])

        width = max(1.0, x_max - x_min)
        height = max(1.0, y_max - y_min)

        # 1. 如果几乎都落在同一行或同一列，通常是坐标轴
        x_std = np.std(xs)
        y_std = np.std(ys)
        if y_std < max(3.0, 0.01 * height):
            return True
        if x_std < max(3.0, 0.01 * width):
            return True

        total = len(xs)

        # 2. 做归一化直方图，判断是否存在明显的窄带聚集
        rel_xs = (xs - x_min) / width
        rel_ys = (ys - y_min) / height

        bins = 10
        hist_x, edges_x = np.histogram(rel_xs, bins=bins, range=(0.0, 1.0))
        max_bin_count_x = hist_x.max()
        max_bin_idx_x = hist_x.argmax()
        bin_width_x = 1.0 / bins
        if (max_bin_count_x / total) >= 0.5 and bin_width_x <= 0.25:
            bin_left = edges_x[max_bin_idx_x]
            bin_right = edges_x[max_bin_idx_x + 1]
            if bin_left <= 0.15 or bin_right >= 0.85:
                return True
            return True

        hist_y, edges_y = np.histogram(rel_ys, bins=bins, range=(0.0, 1.0))
        max_bin_count_y = hist_y.max()
        max_bin_idx_y = hist_y.argmax()
        bin_width_y = 1.0 / bins
        if (max_bin_count_y / total) >= 0.5 and bin_width_y <= 0.25:
            bin_top = edges_y[max_bin_idx_y]
            bin_bottom = edges_y[max_bin_idx_y + 1]
            if bin_top <= 0.15 or bin_bottom >= 0.85:
                return True
            return True

        # 3. 进一步检查是否集中分布在图像边缘的窄带区域
        threshold_band = 0.20
        right_mask = rel_xs >= (1.0 - threshold_band)
        left_mask = rel_xs <= threshold_band
        if right_mask.sum() / total >= 0.5 or left_mask.sum() / total >= 0.5:
            return True

        top_mask = rel_ys <= threshold_band
        bottom_mask = rel_ys >= (1.0 - threshold_band)
        if top_mask.sum() / total >= 0.5 or bottom_mask.sum() / total >= 0.5:
            return True

        return False

    def select_target_subimg(self, subimg_with_coords, text_info_list, return_with_coords=False):
        """
        基于 OCR 文本内容筛选子图。

        过滤规则:
        1. 含公式符号，如 `+` 或 `X`
        2. 含 `Frequency` 或 `Hz`
        3. 文本排布明显像坐标轴标注

        返回值:
            通过筛选的子图列表
        """

        def is_short_alnum(text):
            return bool(re.fullmatch(r"[a-zA-Z0-9]{1,3}", text.strip()))

        def is_long_letter_combination(text):
            t = text.strip()
            return bool(re.fullmatch(r"[a-zA-Z]+", t) and len(t) >= 4)

        def has_formula(text):
            t = text.strip()
            return ("+" in t) or ("X" in t)

        kept_subimgs = []

        for idx, (subimg, (x_min, x_max, y_min, y_max)) in enumerate(subimg_with_coords):
            if subimg is None or getattr(subimg, "size", 0) == 0:
                continue

            sub_texts = []
            text_centers = []
            sub_text_infos = []

            # 收集当前子图范围内的 OCR 文本
            for info in text_info_list:
                box = info["box"]
                x_center = sum(p[0] for p in box) / 4.0
                y_center = sum(p[1] for p in box) / 4.0

                if (x_min - 1 <= x_center <= x_max + 1) and (y_min - 1 <= y_center <= y_max + 1):
                    sub_texts.append(info["text"])
                    text_centers.append((x_center, y_center))
                    sub_text_infos.append(info)

            if any(has_formula(t) for t in sub_texts):
                print(f"Subimage {idx}: formula detected, skip")
                continue

            if any(("frequency" in t.lower() or "hz" in t.lower()) for t in sub_texts):
                print(f"Subimage {idx}: frequency figure detected, skip")
                continue

            if self.is_axis_like_layout(text_centers, sub_text_infos, x_min, x_max, y_min, y_max):
                print(f"Subimage {idx}: axis-like layout detected, skip")
                continue

            # 这里保留原有辅助判断函数，暂时不额外启用
            _ = is_short_alnum
            _ = is_long_letter_combination

            if return_with_coords:
                kept_subimgs.append((subimg, (x_min, x_max, y_min, y_max)))
            else:
                kept_subimgs.append(subimg)
            print(f"Subimage {idx}: kept after filtering")

        if not kept_subimgs:
            print("All subimages were filtered out")
            return []

        return kept_subimgs

    def evaluate_square_like(self, contour, img_shape):
        # 1. 外接矩形面积占比
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        rect_area = w * h if w * h > 0 else 1
        area_ratio = contour_area / rect_area

        # 2. 形状正交性评分，越接近直角越像方形结构
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        angle_score = 0
        if len(approx) >= 4:
            corners = approx.reshape(-1, 2)
            total_diff = 0
            valid_angles = 0

            for i in range(len(corners)):
                p1 = corners[i]
                p2 = corners[(i + 1) % len(corners)]
                p3 = corners[(i + 2) % len(corners)]

                v1 = p1 - p2
                v2 = p3 - p2

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                diff = abs(angle - 90)
                total_diff += diff
                valid_angles += 1

            if valid_angles > 0:
                avg_angle_error = total_diff / valid_angles
                angle_score = max(0, 1 - avg_angle_error / 45)

        # 3. 综合评分
        score = 0.5 * area_ratio + 0.5 * angle_score
        return score, area_ratio, angle_score

    def find_square_like_index(self, regular_contours, gray):
        best_idx = -1
        best_score = -999

        for i, cnt in enumerate(regular_contours):
            score, _, _ = self.evaluate_square_like(cnt, gray.shape)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def filter_heatmap_contours(self, contours, img, var_threshold=1000):
        keep = []
        scores = []

        for cnt in contours:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            pixels = img[mask == 255]
            hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
            hsv = hsv.reshape(-1, 3)

            # 用 HSV 的饱和度和亮度方差衡量颜色是否过于复杂
            var_s, var_v = np.var(hsv[:, 1]), np.var(hsv[:, 2])
            var_score = max(var_s, var_v)
            scores.append(var_score)

            if var_score < var_threshold:
                keep.append(cnt)

        # 防止全部被过滤，至少保留一个最稳定的轮廓
        if len(keep) == 0 and len(contours) > 0:
            min_idx = int(np.argmin(scores))
            keep.append(contours[min_idx])

        return keep

    def _get_yolo_model(self):
        if self._yolo_load_failed:
            return None
        if self._yolo_model is not None:
            return self._yolo_model
        if not self.yolo_model_path or not os.path.exists(self.yolo_model_path):
            self._yolo_load_failed = True
            print(f"YOLO 子图模型不存在，跳过 YOLO 分图: {self.yolo_model_path}")
            return None

        try:
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                dll_dirs = [
                    os.path.join(sys.prefix, "Library", "bin"),
                    os.path.join(sys.prefix, "DLLs"),
                    os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
                ]
                for dll_dir in dll_dirs:
                    if os.path.isdir(dll_dir):
                        try:
                            os.add_dll_directory(dll_dir)
                        except OSError:
                            pass
            from ultralytics import YOLO

            self._yolo_model = YOLO(self.yolo_model_path)
            return self._yolo_model
        except Exception as exc:
            self._yolo_load_failed = True
            print(f"YOLO 子图模型加载失败，回退原逻辑: {exc}")
            return None

    def _resolve_subfigure_class_id(self, model):
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            for cls_id, cls_name in names.items():
                if str(cls_name).strip().lower() == "subfigure":
                    return int(cls_id)
        elif isinstance(names, list):
            for cls_id, cls_name in enumerate(names):
                if str(cls_name).strip().lower() == "subfigure":
                    return int(cls_id)
        return self.yolo_subfigure_class_id

    def _normalize_yolo_label(self, label):
        """
        统一 YOLO 类别名格式，便于做大小写无关匹配。
        """
        return re.sub(r"[^a-z0-9_]+", "", str(label).strip().lower())

    def _resolve_yolo_class_ids(self, model, target_class_names):
        """
        根据类别名解析目标类别 ID。
        返回格式:
            {class_id: normalized_class_name, ...}
        """
        normalized_targets = {
            self._normalize_yolo_label(name)
            for name in target_class_names
            if str(name).strip()
        }
        if not normalized_targets:
            return {}

        names = getattr(model, "names", None)
        if isinstance(names, dict):
            iterable = names.items()
        elif isinstance(names, list):
            iterable = enumerate(names)
        else:
            iterable = []

        resolved = {}
        for cls_id, cls_name in iterable:
            normalized_name = self._normalize_yolo_label(cls_name)
            if normalized_name in normalized_targets:
                resolved[int(cls_id)] = normalized_name

        # 某些训练导出的模型名字表可能不完整；对 text 额外保留一个
        # “按类别 ID 回退”的入口，避免因为名字缺失而完全无法处理文字。
        if ("text" in normalized_targets) and (len(resolved) == 0):
            fallback_text_class_id = getattr(self, "yolo_text_class_id", None)
            if fallback_text_class_id is not None:
                resolved[int(fallback_text_class_id)] = "text"
        return resolved

    def _box_to_xyxy_from_ocr_polygon(self, box):
        """
        把 OCR 四边形框转换成常规 xyxy 矩形框。

        为什么这里要单独做一个函数：
        1. OCR 返回的是四边形点集，后面 YOLO 返回的是矩形框；
        2. 我们要判断 YOLO 的 text 框和 OCR 文本是否重合，因此需要统一坐标格式；
        3. 这个函数只做坐标转换，不做任何业务判断，方便后续你单独调试。
        """
        box_arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        x_min = float(np.min(box_arr[:, 0]))
        y_min = float(np.min(box_arr[:, 1]))
        x_max = float(np.max(box_arr[:, 0]))
        y_max = float(np.max(box_arr[:, 1]))
        return [x_min, y_min, x_max, y_max]

    def _compute_box_iou_xyxy(self, box_a, box_b):
        """
        计算两个 xyxy 矩形框的 IoU。

        这里保持实现尽量直白，方便你后续手动调整阈值和匹配策略。
        """
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def _is_valid_ocr_text_candidate(self, text, score):
        """
        判断 OCR 结果是否足够像“我们要删除的字母/标注文字”。

        这里用比较保守的规则：
        1. OCR 分数太低的结果先丢掉；
        2. 只接受包含英文字母或数字的短文本；
        3. 不把长句子、纯符号块、整段标题直接当成局部尺寸标注。

        这个判断本质上是“先缩小误杀范围”，
        如果你后续想扩大召回，可以优先改这里。
        """
        text_str = str(text).strip()
        if len(text_str) == 0:
            return False

        if float(score) < getattr(self, "ocr_text_score_threshold", 0.30):
            return False

        normalized_text = text_str.replace(" ", "")
        if len(normalized_text) > 12:
            return False

        if not re.search(r"[A-Za-z0-9]", normalized_text):
            return False

        # 只保留常见尺寸标注形态：短字母、字母数字混合、单个数字等。
        if re.fullmatch(r"[A-Za-z0-9_\-\.]+", normalized_text):
            return True

        return False

    def _collect_local_text_infos(self, region_coords, text_info_list):
        """
        把当前子图范围内的 OCR 结果裁到局部坐标系。

        输入:
            region_coords: (x_min, x_max, y_min, y_max)，原图绝对坐标。
            text_info_list: 原图级别 OCR 结果，来自 self.text_info_list。

        返回:
            local_infos: 每个元素都附带局部 box、局部 xyxy、原始 text/score。

        为什么需要“局部坐标”：
        YOLO 的文字检测是在当前子图上做的，
        OCR 则是之前在原图上跑的，
        所以两边如果要比重合度，必须统一到同一个坐标系里。
        """
        if text_info_list is None:
            return []

        x_min, x_max, y_min, y_max = region_coords
        local_infos = []

        for info in text_info_list:
            box = np.asarray(info["box"], dtype=np.float32).reshape(-1, 2)
            cx = float(np.mean(box[:, 0]))
            cy = float(np.mean(box[:, 1]))

            # 使用中心点判断 OCR 文本是否属于当前子图。
            if not ((x_min - 1) <= cx <= (x_max + 1) and (y_min - 1) <= cy <= (y_max + 1)):
                continue

            local_box = box.copy()
            local_box[:, 0] -= float(x_min)
            local_box[:, 1] -= float(y_min)

            local_infos.append({
                "text": info["text"],
                "score": float(info["score"]),
                "box": local_box,
                "xyxy": self._box_to_xyxy_from_ocr_polygon(local_box),
            })

        return local_infos

    def _confirm_text_detections_with_ocr(self, yolo_text_detections, local_text_infos):
        """
        同时使用 YOLO 和 OCR 确认“某个检测框是否真的是文字”。

        确认逻辑：
        1. 先用 YOLO 给出 text 候选框；
        2. 再在当前子图对应的 OCR 结果里寻找重叠文本；
        3. 只有当 YOLO 框和 OCR 文本满足“位置重叠 + OCR 文本看起来像短标注”
           时，才把它视为真正要删除的文字。

        返回的 confirmed_detections 会保留 matched_ocr_texts，
        方便你后续直接打印或可视化对齐效果。
        """
        if yolo_text_detections is None or len(yolo_text_detections) == 0:
            return []
        if local_text_infos is None or len(local_text_infos) == 0:
            return []

        confirmed_detections = []
        iou_threshold = getattr(self, "yolo_ocr_match_iou_threshold", 0.10)

        for detection in yolo_text_detections:
            yolo_box = detection["xyxy"]
            yx1, yy1, yx2, yy2 = yolo_box
            matched_texts = []

            for info in local_text_infos:
                if not self._is_valid_ocr_text_candidate(info["text"], info["score"]):
                    continue

                ocr_box = info["xyxy"]
                iou = self._compute_box_iou_xyxy(yolo_box, ocr_box)

                ocr_cx = 0.5 * (ocr_box[0] + ocr_box[2])
                ocr_cy = 0.5 * (ocr_box[1] + ocr_box[3])
                center_inside = (yx1 <= ocr_cx <= yx2) and (yy1 <= ocr_cy <= yy2)

                # 对于字母这种小目标，IoU 很容易偏低，因此：
                # 1. 允许中心点落入；
                # 2. 或者 IoU 超过较低阈值；
                # 两者满足其一，就认为 YOLO 和 OCR 指向的是同一段文字。
                if center_inside or (iou >= iou_threshold):
                    matched_texts.append({
                        "text": info["text"],
                        "score": info["score"],
                        "box": info["box"],
                        "xyxy": info["xyxy"],
                        "iou": iou,
                    })

            if len(matched_texts) == 0:
                continue

            confirmed_detection = dict(detection)
            confirmed_detection["class_name"] = "text"
            confirmed_detection["matched_ocr_texts"] = matched_texts
            confirmed_detections.append(confirmed_detection)

        return confirmed_detections

    def remove_confirmed_text_with_yolo_and_ocr(
        self,
        img,
        region_coords,
        text_info_list,
        global_background_color=None,
        expand_ratio=0.06,
        expand_pixels=2,
        border_width=2,
        min_component_area=3,
    ):
        """
        在当前子图内部执行“文字确认 + 文字去除”。

        这一段是文字处理的核心入口，流程分为三步：

        第一步，YOLO 给出 text 候选框：
            这里只负责“哪里像文字”，不直接删除。

        第二步，用 OCR 再确认：
            OCR 已经在外部全图跑过，这里把 OCR 结果裁到当前子图坐标系，
            然后只保留和 YOLO 检测框真正重叠的短文字。

        第三步，像素级去除：
            真正删除时并不是把整块矩形抹掉，而是只删除框内最像黑色文字的像素。
            如果文字压在接近整体背景色的区域上，就直接填充背景色；
            否则填白并保留 repair mask，交给后续修补阶段处理。

        返回:
            cleaned_img: 去除文字后的子图
            repair_mask: 需要后续修补的区域
            confirmed_detections: 被 YOLO+OCR 双重确认的文字框
            removed_mask: 本轮实际被去掉的像素
            detection_vis: 方便调试的可视化图
        """
        local_text_infos = self._collect_local_text_infos(
            region_coords=region_coords,
            text_info_list=text_info_list,
        )

        yolo_text_detections = self.detect_elements_with_yolo(
            img=img,
            target_class_names=getattr(self, "yolo_text_labels", ("text",)),
        )

        confirmed_detections = self._confirm_text_detections_with_ocr(
            yolo_text_detections=yolo_text_detections,
            local_text_infos=local_text_infos,
        )

        if len(confirmed_detections) == 0:
            empty_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            detection_vis = self.draw_yolo_detections(img, [], mask=empty_mask)
            return img.copy(), empty_mask, [], empty_mask, detection_vis

        cleaned_img, repair_mask, removed_mask = self.remove_yolo_targets_by_detections(
            img_bgr=img,
            detections=confirmed_detections,
            global_background_color=global_background_color,
            expand_ratio=expand_ratio,
            expand_pixels=expand_pixels,
            border_width=border_width,
            min_component_area=min_component_area,
        )
        detection_vis = self.draw_yolo_detections(img, confirmed_detections, mask=removed_mask)
        return cleaned_img, repair_mask, confirmed_detections, removed_mask, detection_vis

    def _package_processed_subimg_item(
        self,
        subimg,
        coords,
        text_info_list,
        global_background_color_bgr,
    ):
        """
        把单个子图包装成统一结构，方便后续 pipeline 继续处理。

        这里统一返回 dict，而不是只返回 numpy 图像，原因是：
        1. 文字去除后可能已经产生 repair mask；
        2. 后续 pipeline 还要把这些 mask 并入总修补流程；
        3. 你后面如果要手动调参，也更容易在同一个结构里看到所有中间结果。
        """
        cleaned_img, text_repair_mask, text_detections, text_removed_mask, text_detection_vis = (
            self.remove_confirmed_text_with_yolo_and_ocr(
                img=subimg,
                region_coords=coords,
                text_info_list=text_info_list,
                global_background_color=global_background_color_bgr,
            )
        )

        return {
            "image": cleaned_img,
            "coords": coords,
            "pre_repair_mask": text_repair_mask,
            "pre_removed_mask": text_removed_mask,
            "pre_detections": text_detections,
            "pre_detection_vis": text_detection_vis,
        }

    def detect_subfigures_with_yolo(self, img, expand_pixels=5):
        """
        使用 YOLO 检测子图并裁剪。

        返回格式:
            [(subimg, (x_min, x_max, y_min, y_max)), ...]
        """
        model = self._get_yolo_model()
        if model is None:
            return []

        try:
            result_list = model.predict(
                source=img,
                conf=self.yolo_conf_threshold,
                iou=self.yolo_iou_threshold,
                device=getattr(self, "yolo_device", "cpu"),
                verbose=False,
            )
        except Exception as exc:
            print(f"YOLO 子图检测失败，回退原逻辑: {exc}")
            return []

        if not result_list:
            return []

        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return []

        subfigure_class_id = self._resolve_subfigure_class_id(model)
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.array([])

        h, w = img.shape[:2]
        subimgs = []
        for idx, box in enumerate(xyxy):
            if len(cls) > idx and int(cls[idx]) != int(subfigure_class_id):
                continue

            x1, y1, x2, y2 = box
            x1 = max(0, int(np.floor(x1)) - expand_pixels)
            y1 = max(0, int(np.floor(y1)) - expand_pixels)
            x2 = min(w, int(np.ceil(x2)) + expand_pixels)
            y2 = min(h, int(np.ceil(y2)) + expand_pixels)

            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            subimg = img[y1:y2, x1:x2]
            subimgs.append((subimg, (x1, x2, y1, y2)))

        subimgs.sort(key=lambda item: (item[1][2], item[1][0]))
        return subimgs

    def detect_elements_with_yolo(self, img, target_class_names=("arrow", "line"), conf_threshold=None):
        """
        使用 YOLO 检测指定类别元素，返回原始检测框列表。

        返回格式:
            [
                {
                    "xyxy": [x1, y1, x2, y2],
                    "class_id": int,
                    "class_name": "arrow" | "line",
                    "score": float,
                },
                ...
            ]
        """
        model = self._get_yolo_model()
        if model is None:
            return []

        target_class_map = self._resolve_yolo_class_ids(model, target_class_names)
        if not target_class_map:
            print(f"YOLO 模型中未找到目标类别: {target_class_names}")
            return []

        if conf_threshold is None:
            conf_threshold = self.yolo_conf_threshold

        try:
            result_list = model.predict(
                source=img,
                conf=float(conf_threshold),
                iou=self.yolo_iou_threshold,
                device=getattr(self, "yolo_device", "cpu"),
                verbose=False,
            )
        except Exception as exc:
            print(f"YOLO 元素检测失败，跳过 arrow/line 预处理: {exc}")
            return []

        if not result_list:
            return []

        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.array([])
        conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy), dtype=np.float32)

        detections = []
        for idx, box in enumerate(xyxy):
            if len(cls) <= idx:
                continue

            class_id = int(cls[idx])
            if class_id not in target_class_map:
                continue

            detections.append({
                "xyxy": [float(v) for v in box.tolist()],
                "class_id": class_id,
                "class_name": target_class_map[class_id],
                "score": float(conf[idx]) if len(conf) > idx else 0.0,
            })

        detections.sort(key=lambda item: (item["xyxy"][1], item["xyxy"][0]))
        return detections

    def remove_detected_elements_with_yolo(
        self,
        img,
        target_class_names=("arrow", "line"),
        global_background_color=None,
        expand_ratio=0.08,
        expand_pixels=2,
        border_width=2,
        min_component_area=8,
    ):
        """
        先用 YOLO 检出目标元素，再在框内做像素级细化去除。

        返回:
            cleaned_img, repair_mask, detections, removed_mask
        """
        detections = self.detect_elements_with_yolo(
            img=img,
            target_class_names=target_class_names,
            conf_threshold=(
                getattr(self, "yolo_element_conf_threshold", self.yolo_conf_threshold)
                if any(str(name).strip().lower() in ("arrow", "line") for name in target_class_names)
                else self.yolo_conf_threshold
            ),
        )

        if len(detections) == 0:
            empty_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            return img.copy(), empty_mask, [], empty_mask

        cleaned_img, repair_mask, removed_mask = self.remove_yolo_targets_by_detections(
            img_bgr=img,
            detections=detections,
            global_background_color=global_background_color,
            expand_ratio=expand_ratio,
            expand_pixels=expand_pixels,
            border_width=border_width,
            min_component_area=min_component_area,
        )
        return cleaned_img, repair_mask, detections, removed_mask

    def process_image(self, img, expand_pixels=5):
        """
        处理单张图像。

        优先尝试使用 YOLO 划分子图；如果 YOLO 未稳定得到结果，
        再回退到轮廓分析和 OCR 辅助切分逻辑。
        """
        original = img.copy()
        border_width = 4
        original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        global_background_color_rgb = Preprocess_tool.detect_background_color(original_rgb)
        global_background_color_bgr = np.asarray(global_background_color_rgb[::-1], dtype=np.uint8)

        h, w = original.shape[:2]

        # 先把边框轻微抹白，减少边界干扰
        if len(original.shape) == 2:
            original[:border_width, :] = 255
            original[-border_width:, :] = 255
            original[:, :border_width] = 255
            original[:, -border_width:] = 255
        else:
            original[:border_width, :, :] = 255
            original[-border_width:, :, :] = 255
            original[:, :border_width, :] = 255
            original[:, -border_width:, :] = 255

        # 预处理：灰度、模糊、边缘检测
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 60, 180)
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=1)
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_imgs = self.detect_subfigures_with_yolo(img, expand_pixels=expand_pixels)
        if len(yolo_imgs) >= 2:
            print(f"YOLO 检测到 {len(yolo_imgs)} 个子图，优先使用 YOLO 分图")
            target_imgs = self.select_target_subimg(
                yolo_imgs,
                self.text_info_list,
                return_with_coords=True,
            )
            if len(target_imgs) > 0:
                return [
                    self._package_processed_subimg_item(
                        subimg=subimg,
                        coords=coords,
                        text_info_list=self.text_info_list,
                        global_background_color_bgr=global_background_color_bgr,
                    )
                    for subimg, coords in target_imgs
                ]
            print("YOLO 子图经过 OCR 规则过滤后为空，直接返回 YOLO 原始分图")
            return [
                self._package_processed_subimg_item(
                    subimg=subimg,
                    coords=coords,
                    text_info_list=self.text_info_list,
                    global_background_color_bgr=global_background_color_bgr,
                )
                for subimg, coords in yolo_imgs
            ]

        if not contours:
            print("No contours detected, return original image")
            return [
                self._package_processed_subimg_item(
                    subimg=original,
                    coords=(0, w, 0, h),
                    text_info_list=self.text_info_list,
                    global_background_color_bgr=global_background_color_bgr,
                )
            ]

        # 筛选较大的轮廓，排除明显噪声
        min_area_ratio = 0.01
        min_area = min_area_ratio * h * w
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        n_contours = len(large_contours)
        print("number_of_large_contours", n_contours)

        if n_contours > 1:
            print("轮廓分析显示疑似多子图，YOLO 未稳定检出多子图，回退原逻辑")

            imgs = Preprocess_tool.process_special_text_boxes(
                text_info_list=self.text_info_list,
                img=img,
            )

            if len(imgs) == 0:
                print("当前规则未找到合适子图，继续使用上一版轮廓逻辑")
                flite_contours = self.filter_heatmap_contours(large_contours, img)
                regular_contours = [c for c in flite_contours if self.is_regular_shape(c)]
                square_like_idx = self.find_square_like_index(regular_contours, gray)
                target_contour = regular_contours[square_like_idx]
                x, y, width, height = cv2.boundingRect(target_contour)

                expanded_min_x = max(0, x - expand_pixels)
                expanded_min_y = max(0, y - expand_pixels)
                expanded_max_x = min(w, x + width + expand_pixels)
                expanded_max_y = min(h, y + height + expand_pixels)

                cropped_img = img[expanded_min_y:expanded_max_y, expanded_min_x:expanded_max_x]
                return [
                    self._package_processed_subimg_item(
                        subimg=cropped_img,
                        coords=(expanded_min_x, expanded_max_x, expanded_min_y, expanded_max_y),
                        text_info_list=self.text_info_list,
                        global_background_color_bgr=global_background_color_bgr,
                    )
                ]

            traget_imgs = self.select_target_subimg(
                imgs,
                self.text_info_list,
                return_with_coords=True,
            )
            return [
                self._package_processed_subimg_item(
                    subimg=subimg,
                    coords=coords,
                    text_info_list=self.text_info_list,
                    global_background_color_bgr=global_background_color_bgr,
                )
                for subimg, coords in traget_imgs
            ]

        if n_contours == 1:
            print("当前为单图场景")
            return [
                self._package_processed_subimg_item(
                    subimg=img,
                    coords=(0, w, 0, h),
                    text_info_list=self.text_info_list,
                    global_background_color_bgr=global_background_color_bgr,
                )
            ]

        return [
            self._package_processed_subimg_item(
                subimg=img,
                coords=(0, w, 0, h),
                text_info_list=self.text_info_list,
                global_background_color_bgr=global_background_color_bgr,
            )
        ]
