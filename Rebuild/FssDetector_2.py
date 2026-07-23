import os
import re
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import Preprocess_tool

try:
    from .fssdetector_ocr import TextSystemOCRAdapter
except ImportError:
    from fssdetector_ocr import TextSystemOCRAdapter


class FSSfigDetector:
    """
    FSSfigDetector类用于检测和处理图像中的图形元素
    包含颜色量化、图形识别、边缘处理等功能
    """

    def __init__(
        self,
        max_k=6,
        min_color_diff=30,
        text_info_list=None,
        ocr_engine=None,
        show_debug_windows=False,
        yolo_model_path=r"D:\Line_build\runs\detect\train_continue_from_best\weights\best.pt",
        yolo_subfigure_class_id=2,
        yolo_conf_threshold=0.20,
        yolo_iou_threshold=0.50,
    ):
        """
        初始化检测器
        参数:
            max_k: 颜色量化的最大k值
            min_color_diff: 最小颜色差异阈值
        """

        self.max_k = max_k
        self.min_color_diff = min_color_diff
        self.text_info_list = text_info_list or []
        if ocr_engine is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ocr_engine = TextSystemOCRAdapter(project_dir=project_root)
        self.ocr_engine = ocr_engine
        self.show_debug_windows = show_debug_windows
        self.yolo_model_path = yolo_model_path
        self.yolo_subfigure_class_id = yolo_subfigure_class_id
        self.yolo_conf_threshold = yolo_conf_threshold
        self.yolo_iou_threshold = yolo_iou_threshold
        self._yolo_model = None
        self._yolo_load_failed = False
        self.supported_image_extensions = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
        }


    def is_regular_shape(self, contour):
        """判断轮廓是否为规则形状（矩形、正方形、圆形）"""
        area = cv2.contourArea(contour)
        if area < 100:  # 排除极小轮廓（避免误判）
            return False

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter < 30:  # 排除周长过短的轮廓
            return False

        # 1. 判断是否为矩形/正方形（4条边 + 直角）
        epsilon = 0.04 * perimeter  # 多边形近似精度（经验值）
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        vertices = len(approx)

        if vertices == 4:
            # 验证4个角是否为直角（向量点积判断）
            pts = approx.reshape(4, 2).astype(np.float32)
            right_angle_count = 0
            for i in range(4):
                vec1 = pts[(i + 1) % 4] - pts[i]
                vec2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
                if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
                    continue  # 避免除以0
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)
                dot_product = np.dot(vec1_norm, vec2_norm)
                if abs(dot_product) < 0.3:  # 允许±0.3误差（约72°~108°）
                    right_angle_count += 1
            if right_angle_count == 4:
                return True  # 矩形/正方形

        # 2. 判断是否为圆形（圆度指标）
        # 圆度公式：4π×面积 / 周长²（圆形≈1，不规则形状<0.6）
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.6:  # 圆度阈值（可调整，0.6~0.8较合适）
            return True  # 圆形

        # 非规则形状
        return False

    def is_axis_like_layout(self, text_centers, sub_text_infos, x_min, x_max, y_min, y_max):
        """
        text_centers: [(x_center, y_center), ...] (图像绝对坐标)
        sub_text_infos: 子图OCR结果，包含文字内容，用来检查是否包含数字
        通过三种方式判断是否为坐标/轴标签图：
          1) 横向或纵向共线（std 很小）
          2) 纵向或横向密集窄带（例如右侧一列文字）
          3) 文本个数太少则不判为坐标图
          4) 包含数字的文本就参与坐标列检测
        返回 True/False
        """
        if len(text_centers) < 5:
            return False

        # 过滤出包含数字的文本的坐标点（这些通常是坐标轴上的数字）
        numeric_texts = [(x, y) for (x, y), info in zip(text_centers, sub_text_infos) if
                         re.search(r'\d', info['text'])]

        if len(numeric_texts) < 5:  # 如果包含数字的文本太少，不能作为坐标图判定
            return False

        xs = np.array([x for x, y in numeric_texts])
        ys = np.array([y for x, y in numeric_texts])

        width = max(1.0, x_max - x_min)
        height = max(1.0, y_max - y_min)

        # 1) 共线判断（同一行或同一列）
        x_std = np.std(xs)
        y_std = np.std(ys)
        if y_std < max(3.0, 0.01 * height):  # 同一行
            return True
        if x_std < max(3.0, 0.01 * width):  # 同一列
            return True

        total = len(xs)

        # 2) 直方图密度判断：把坐标映射到子图相对坐标 [0,1] 再做bins
        rel_xs = (xs - x_min) / width
        rel_ys = (ys - y_min) / height

        # 检查竖直窄带（列）
        bins = 10
        hist_x, edges_x = np.histogram(rel_xs, bins=bins, range=(0.0, 1.0))
        max_bin_count_x = hist_x.max()
        max_bin_idx_x = hist_x.argmax()
        bin_width_x = 1.0 / bins
        # 判断：如果某个bin内文字占比高且该bin宽度相对小（说明为窄带/列）
        if (max_bin_count_x / total) >= 0.5 and bin_width_x <= 0.25:
            # 进一步判断该窄带是否靠近子图一侧（常见坐标轴在边缘）
            bin_left = edges_x[max_bin_idx_x]
            bin_right = edges_x[max_bin_idx_x + 1]
            # 若窄带位于子图左右 15% 内也认为是坐标列
            if bin_left <= 0.15 or bin_right >= 0.85:
                return True
            # 否则也可能是中间竖列（也判为坐标图）
            return True

        # 同理检查横向窄带（行）
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

        # 3) 另外，检测是否存在明显的竖直条（窄带检测的另一种方式）
        # 判断在子图右侧/左侧某个窄带（宽度占比 threshold_band）内的文字占比
        threshold_band = 0.20  # 窄带宽度占比
        # 右侧窄带
        right_mask = rel_xs >= (1.0 - threshold_band)
        left_mask = rel_xs <= threshold_band
        if right_mask.sum() / total >= 0.5 or left_mask.sum() / total >= 0.5:
            return True
        # 顶部/底部窄带
        top_mask = rel_ys <= threshold_band
        bottom_mask = rel_ys >= (1.0 - threshold_band)
        if top_mask.sum() / total >= 0.5 or bottom_mask.sum() / total >= 0.5:
            return True

        return False

    def select_target_subimg(self, subimg_with_coords, text_info_list):
        """
        按 OCR 文本过滤子图，新增排除逻辑（含更强的坐标图/竖列检测）：
        - 含公式(包含+ 或 X) -> 排除
        - 含 Frequency 或 Hz -> 打印 "该图为频率图 已做后续处理" 并排除
        - 文本呈明显一行/一列或密集窄带（竖列/横列） -> 打印 "该图为坐标图" 并排除
        - 只要包含数字的文本就参与坐标列检测

        返回：
            保留的子图列表（已排除不符合规则的）
        """


        # ===== 原有规则 =====
        def is_short_alnum(text):
            return bool(re.fullmatch(r'[a-zA-Z0-9]{1,3}', text.strip()))

        def is_long_letter_combination(text):
            t = text.strip()
            return bool(re.fullmatch(r'[a-zA-Z]+', t) and len(t) >= 4)

        def has_formula(text):
            t = text.strip()
            return ('+' in t) or ('X' in t)

        # ====== 主循环 ======
        kept_subimgs = []  # 不排除的子图

        for idx, (subimg, (x_min, x_max, y_min, y_max)) in enumerate(subimg_with_coords):

            if subimg is None or getattr(subimg, "size", 0) == 0:
                continue

            sub_texts = []
            text_centers = []
            sub_text_infos = []  # 新增：收集子图内的OCR info对象

            # 取当前子图的文字
            for info in text_info_list:
                box = info['box']
                # box 预期为四个点 [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
                x_center = sum(p[0] for p in box) / 4.0
                y_center = sum(p[1] for p in box) / 4.0

                if (x_min - 1 <= x_center <= x_max + 1) and (y_min - 1 <= y_center <= y_max + 1):
                    sub_texts.append(info['text'])
                    text_centers.append((x_center, y_center))
                    sub_text_infos.append(info)

            # ========== 排除规则 1：公式 ==========
            if any(has_formula(t) for t in sub_texts):
                print(f"子图{idx}：包含公式 → 已排除")
                continue

            # ========== 排除规则 2：Frequency / Hz ==========
            if any(('frequency' in t.lower() or 'hz' in t.lower()) for t in sub_texts):
                print(f"子图{idx}：该图为频率图 已做后续处理")
                continue

            # ========== 排除规则 3：坐标图（增强版）==========
            # 改为调用 self.is_axis_like_layout，并传入筛选好的 sub_text_infos
            if self.is_axis_like_layout(text_centers, sub_text_infos, x_min, x_max, y_min, y_max):
                print(f"子图{idx}：该图为坐标图")
                continue

            # ========== 原有规则：长字母组合 ==========
            invalid = False
            # for t in sub_texts:
            #     if is_long_letter_combination(t):
            #         print(f"子图{idx}：包含长字母组合 {t} → 排除")
            #         invalid = True
            #         break
            # if invalid:
            #     continue

            # 如果都通过过滤，则保留
            kept_subimgs.append(subimg)
            print(f"子图{idx}：通过筛选，已保留")

        if not kept_subimgs:
            print("所有图像均被排除，无保留子图")
            return []

        return kept_subimgs
    def evaluate_square_like(self,contour, img_shape):
        # 1. bounding rectangle area ratio
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        rect_area = w * h if w * h > 0 else 1
        area_ratio = contour_area / rect_area

        # 2. shape orthogonality score
        # 多边形逼近判断直角
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

                # 角度计算
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                diff = abs(angle - 90)
                total_diff += diff
                valid_angles += 1

            if valid_angles > 0:
                avg_angle_error = total_diff / valid_angles
                angle_score = max(0, 1 - avg_angle_error / 45)  # 误差越大分越低

        # 综合评分
        score = 0.5 * area_ratio + 0.5 * angle_score
        return score, area_ratio, angle_score
    def find_square_like_index(self,regular_contours, gray):
        best_idx = -1
        best_score = -999

        for i, cnt in enumerate(regular_contours):
            score, _, _ = self.evaluate_square_like(cnt, gray.shape)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx
   
    def filter_heatmap_contours(self,contours, img, var_threshold=1000):
        keep = []
        scores = []  # variance score record

        for cnt in contours:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            pixels = img[mask == 255]
            # cv2.imshow('mask', mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # 转HSV
            hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
            hsv = hsv.reshape(-1, 3)

            # 使用 S 和 V 的方差（颜色变化指标）
            var_s, var_v = np.var(hsv[:, 1]), np.var(hsv[:, 2])
            var_score = max(var_s, var_v)
            scores.append(var_score)

            # 颜色变化小 → 保留
            if var_score < var_threshold:
                keep.append(cnt)

        # ✅ 防止全部被删除，至少保留一个最稳定的轮廓
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
            print(f"YOLO子图模型不存在，跳过YOLO分图: {self.yolo_model_path}")
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
            print(f"YOLO子图模型加载失败，回退原逻辑: {exc}")
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

    def detect_subfigures_with_yolo(self, img, expand_pixels=5):
        """
        使用YOLO检测子图并裁剪，返回:
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
                verbose=False,
            )
        except Exception as exc:
            print(f"YOLO子图检测失败，回退原逻辑: {exc}")
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

    def process_image(self, img, expand_pixels=5):
        """
        处理单张图像：仅保留规则形状（矩形、正方形、圆形）的轮廓，若存在左右分布的两个规则轮廓，
        则保留接近正方形的部分，否则返回原图

        ###result是ocr检测的结果 这里先直接传进来
        """
        original = img.copy()
        border_width=4

        # 获取图像尺寸
        h, w = original.shape[:2]

        # 判断是否为灰度图
        if len(original.shape) == 2:  # 灰度图
            original[:border_width, :] = 255
            original[-border_width:, :] = 255
            original[:, :border_width] = 255
            original[:, -border_width:] = 255
        else:  # 彩色图
            original[:border_width, :, :] = 255
            original[-border_width:, :, :] = 255
            original[:, :border_width, :] = 255
            original[:, -border_width:, :] = 255


        # 预处理：灰度化 + 模糊 + 边缘检测
        gray = cv2.cvtColor( original, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 60, 180)  # 可根据图像调整阈值
        kernel = np.ones((3, 3), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=1)
        # 查找轮廓（只保留外轮廓，减少内部细节干扰）
        contours, _ = cv2.findContours( closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yolo_imgs = self.detect_subfigures_with_yolo(img, expand_pixels=expand_pixels)
        if len(yolo_imgs) >= 2:
            print(f"YOLO检测到{len(yolo_imgs)}个子图，优先使用YOLO分图")
            target_imgs = self.select_target_subimg(yolo_imgs, self.text_info_list)
            if len(target_imgs) > 0:
                return target_imgs
            print("YOLO子图经过OCR规则过滤后为空，直接返回YOLO原始分图")
            return [subimg for subimg, _ in yolo_imgs]

        if not contours:
            print("图像未检测到轮廓，返回原图")
            return original

        # 1. 筛选面积较大的轮廓（排除小噪声）
        min_area_ratio = 0.01  # 最小轮廓面积为图像总面积的5%
        min_area = min_area_ratio * h * w
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        n_contours = len(large_contours)
        print("number_of_large_contours", n_contours)
        if(n_contours >1):
            print("轮廓分析显示疑似多子图，YOLO未稳定检出多子图，回退原逻辑")


            imgs=Preprocess_tool.process_special_text_boxes(text_info_list=self.text_info_list,img=img)

            ####优化这个函数即可
            #traget_imgs=self.select_target_subimg(imgs, self.text_info_list)
            if(len(imgs)==0):
                print("目前算法暂未找到合适子图 改用上一版本逻辑")
                flite_contours = self.filter_heatmap_contours(large_contours, img)
                regular_contours = [c for c in flite_contours if self.is_regular_shape(c)]
                # square_like_idx = self.score_contours(regular_contours,closed_edges)
                square_like_idx = self.find_square_like_index(regular_contours, gray)
                # 7. 裁剪并扩展目标区域
                target_contour = regular_contours[square_like_idx]
                x, y, width, height = cv2.boundingRect(target_contour)

                # 扩展边界（不超出图像范围）
                expanded_min_x = max(0, x - expand_pixels)
                expanded_min_y = max(0, y - expand_pixels)
                expanded_max_x = min(w, x + width + expand_pixels)
                expanded_max_y = min(h, y + height + expand_pixels)

                cropped_img = img[expanded_min_y:expanded_max_y, expanded_min_x:expanded_max_x]
                return cropped_img

            else:
                traget_imgs = self.select_target_subimg(imgs, self.text_info_list)
                return traget_imgs
        if(n_contours==1):
            print("为单一图像")
            return img
        return img

    def adjust_image_by_mask(self, original_color_img, binary_mask,bg_color):
        """
        根据二值掩码调整彩色图像：保留黑色像素区域的原图颜色，白色像素区域设为白色

        参数:
            original_color_img: 彩色图像 (H, W, 3)
            binary_mask: 二值图像（黑色(0)为保留区域，白色(255)为需设为白色的区域）(H, W)

        返回:
            调整后的彩色图像
        """
        # 输入验证
        if original_color_img is None:
            raise ValueError("原始彩色图像不能为空")
        if binary_mask is None:
            raise ValueError("二值掩码不能为空")
        if len(binary_mask.shape) != 2:
            raise ValueError("binary_mask 应该是单通道图像")
        if original_color_img.shape[:2] != binary_mask.shape:
            raise ValueError("原始图像和掩码尺寸不一致")

        # 创建输出图像的副本以避免修改原图
        result_img = original_color_img.copy()

        # 创建白色区域的掩码（阈值设为127，确保任何非黑色区域都被处理）
        white_mask = binary_mask > 127

        # 将白色掩码对应区域的像素设置为白色
        # 注意：OpenCV图像通常是BGR格式，但设置白色(255,255,255)在任何颜色空间中都是有效的
        result_img[white_mask] = [255, 255, 255]  # 设置为白色

        return result_img

    def auto_crop_color_from_mask(self, original_color_img, binary_mask, padding=None, padding_ratio=0.1):
        """
        根据黑色为前景的二值图像裁剪彩色原图，保留一定留白

        参数:
            original_color_img: 彩色图像 (H, W, 3)
            binary_mask: 二值图像（黑色为前景，白色为背景）(H, W)
            padding: 固定留白像素（如果为None则使用padding_ratio）
            padding_ratio: 自动留白比例（默认10%）

        返回:
            裁剪后的彩色图像和掩码
        """
        if len(binary_mask.shape) != 2:
            raise ValueError("binary_mask 应该是单通道图像")
        if original_color_img.shape[:2] != binary_mask.shape:
            raise ValueError("原图和彩色图像尺寸不一致")

        # 反转二值图：黑色为前景
        coords = cv2.findNonZero(255 - binary_mask)
        if coords is None:
            return original_color_img, binary_mask  # 全白图像，直接返回

        x, y, w, h = cv2.boundingRect(coords)

        if padding is None:
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)
        else:
            pad_x = pad_y = padding

        # 裁剪区域坐标
        x1 = max(x - pad_x, 0)
        y1 = max(y - pad_y, 0)
        x2 = min(x + w + pad_x, original_color_img.shape[1])
        y2 = min(y + h + pad_y, original_color_img.shape[0])

        # 裁剪彩色图像
        cropped_color = original_color_img[y1:y2, x1:x2]
        cropped_mask = binary_mask[y1:y2, x1:x2]
        return cropped_color, cropped_mask

    def visualize_color_clusters(self, original_img, labels, centers, output_dir="testcluster"):
        """
        在白色背景上展示各个聚类颜色及其位置分布

        参数:
            original_img: 原始图像 (RGB格式)
            labels: 聚类标签数组
            centers: 聚类中心数组
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取图像尺寸
        height, width, _ = original_img.shape

        # 创建白色背景画布
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 为每个聚类创建可视化图像
        cluster_visualizations = []
        k = len(centers)

        for i in range(k):
            # 创建当前聚类的掩码
            mask = (labels == i).reshape(height, width)

            # 创建可视化图像：在白色背景上只显示当前聚类颜色
            cluster_img = white_bg.copy()
            cluster_img[mask] = centers[i]

            # 保存单个聚类图像
            output_path = os.path.join(output_dir, f"cluster_{i + 1}.jpg")

            cluster_visualizations.append(cluster_img)

        # 创建所有聚类的组合图
        fig, axes = plt.subplots(1, k + 1, figsize=(15, 10))
        axes = axes.flatten()

        # 显示原始图像
        axes[0].imshow(original_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # 显示各个聚类
        for i in range(k):
            axes[i + 1].imshow(cluster_visualizations[i])
            axes[i + 1].set_title(f"Cluster {i + 1}")
            axes[i + 1].axis('off')

        # 隐藏多余的子图
        for j in range(k + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "all_clusters.jpg"))
        plt.close()

        return cluster_visualizations

    def color_difference(self, color1, color2):
        """
        计算两个颜色在LAB空间中的差异
        """
        # 转换为LAB空间计算差异
        lab1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2LAB)[0][0]
        lab2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2LAB)[0][0]

        # CIE76色差公式
        delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2))
        return delta_e

    def auto_kmeans_color_quantization(self,img, output_path, max_k=6, min_color_diff=30):
        """
        自动确定最佳k值并使用K-Means进行颜色量化（改进版）

        参数:
            img: 输入图像（BGR格式）
            output_path: 输出图片路径
            max_k: 尝试的最大k值(默认为6)
            min_color_diff: 最小可接受的颜色差异（CIE76标准，默认30）
        """
        if img is None:
            raise FileNotFoundError("无法读取图片")

        # 转换到LAB颜色空间，更适合颜色聚类
        #img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 用于显示和输出

        # 获取图像尺寸并重塑为2D像素数组
        height, width, channels = img_rgb.shape
        pixels = img_rgb.reshape((-1, 3))

        # 为了加速计算，对像素进行随机采样
        sample_size = min(15000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[indices].astype(np.float32)

        # 计算不同k值下的评估指标
        wcss = []
        silhouette_scores = []
        calinski_scores = []
        k_values = range(1, max_k + 1)

        for k in k_values:
            print(f"正在计算 k={k}...")
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(sample_pixels)

            # 计算WCSS(簇内平方和)
            wcss.append(kmeans.inertia_)

            # 计算轮廓系数 (仅当k>1时有效)
            if k > 1:
                silhouette_avg = silhouette_score(sample_pixels, labels)
                silhouette_scores.append(silhouette_avg)
                calinski_scores.append(calinski_harabasz_score(sample_pixels, labels))
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)

        # 自动选择最佳k值 - 综合多种指标
        # 1. 计算WCSS的变化率
        wcss_diff = np.diff(wcss)
        wcss_change_rate = np.abs(wcss_diff[1:] / wcss_diff[:-1])

        # 肘点通常是变化率开始趋于稳定的点
        if len(wcss_change_rate) > 0:
            elbow_idx = np.argmin(wcss_change_rate) + 1  # 转换为k值索引
        else:
            elbow_idx = 1  # 默认值

        # 2. 轮廓系数最高的点（k>1）
        if len(silhouette_scores[1:]) > 0:
            max_silhouette_idx = np.argmax(silhouette_scores[1:]) + 1  # +1因为跳过了k=1
        else:
            max_silhouette_idx = 1

        # 3. Calinski-Harabasz指数最高的点（k>1）
        if len(calinski_scores[1:]) > 0:
            max_calinski_idx = np.argmax(calinski_scores[1:]) + 1
        else:
            max_calinski_idx = 1

        # 综合考虑多个指标，取最可能的k值范围
        candidate_ks = [elbow_idx + 1, max_silhouette_idx + 1, max_calinski_idx + 1]
        candidate_ks = [k for k in candidate_ks if 2 <= k <= max_k]

        if not candidate_ks:
            best_k = 2  # 保底值
        else:
            # 计算候选k值的频率，取最常出现的
            best_k = min(set(candidate_ks), key=candidate_ks.count)

        # 检查聚类中心之间的颜色差异，如果太小则增加k值
        # 使用最佳k值进行初步聚类以检查颜色差异
        temp_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        temp_kmeans.fit(sample_pixels)
        temp_centers = temp_kmeans.cluster_centers_

        # 转换回RGB以便计算颜色差异
        temp_centers_rgb = [cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_LAB2RGB)[0][0]
                            for center in temp_centers]

        # 检查是否有颜色差异过小的簇
        has_small_diff = False
        for i in range(len(temp_centers_rgb)):
            for j in range(i + 1, len(temp_centers_rgb)):
                diff = self.color_difference(temp_centers_rgb[i], temp_centers_rgb[j])
                if diff < min_color_diff:
                    has_small_diff = True
                    break
            if has_small_diff:
                break

        # 如果存在差异过小的颜色且未达到最大k值，则增加k值
        if has_small_diff and best_k < max_k:
            best_k += 1
            print(f"检测到相似颜色，将k值调整为: {best_k}")

        print(f"自动选择的最佳k值: {best_k}")
        print(f"轮廓系数: {silhouette_scores[best_k - 1]:.4f} (k={best_k})")

        # 使用最佳k值对整个图像进行聚类
        print(f"使用k={best_k}进行最终聚类...")
        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(pixels.astype(np.float32))

        # 获取聚类中心和标签
        centers_rgb = kmeans.cluster_centers_.astype(np.uint8)
        # 转换回RGB颜色空间以便显示
        # centers_rgb = np.array([cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_LAB2RGB)[0][0]
        #                         for center in centers_lab])

        # 重建量化后的图像（先在LAB空间重建，再转换回RGB）
        # quantized_img_lab = quantized_pixels_lab.reshape((height, width, channels))
        # quantized_img_rgb = cv2.cvtColor( centers_rgb, cv2.COLOR_LAB2RGB)
        quantized_img_rgb =  centers_rgb[labels].reshape((height, width, channels))
        # 创建颜色分布图
        print("创建颜色分布可视化...")
        print(f"使用的颜色: {centers_rgb.tolist()}")

        # 创建颜色条图例
        plt.figure(figsize=(10, 2))
        for i, color in enumerate(centers_rgb):
            plt.fill_between([i, i + 1], 0, 1, color=color / 255)
            plt.text(i + 0.5, 0.5, f"C{i + 1}",
                     ha='center', va='center',
                     color='white' if np.mean(color) < 128 else 'black',
                     fontsize=12, fontweight='bold')
        plt.xlim(0, len(centers_rgb))
        plt.axis('off')
        plt.title(f"颜色 palette (k={best_k})")
        plt.close()

        # 转换回BGR格式用于OpenCV保存
        quantized_bgr = cv2.cvtColor(quantized_img_rgb, cv2.COLOR_RGB2BGR)
        return quantized_bgr,  centers_rgb


    def _normalize_cluster_images(self, clusters_images, color_list=None, center_match_threshold=10.0):
        """
        Convert cluster data to a list of RGB cluster-visualization images.

        auto_kmeans_color_quantization returns a single quantized BGR image plus
        RGB cluster centers. Most downstream code expects one image per cluster,
        so this method rebuilds that list when needed.
        """
        if isinstance(clusters_images, np.ndarray):
            if clusters_images.ndim == 4 and clusters_images.shape[-1] == 3:
                return [clusters_images[i] for i in range(clusters_images.shape[0])]

            if clusters_images.ndim == 3 and clusters_images.shape[-1] == 3:
                if color_list is None:
                    raise ValueError(
                        "color_list is required when clusters_images is a single quantized image"
                    )

                colors = np.array(color_list, dtype=np.float32)
                quantized_rgb = cv2.cvtColor(clusters_images, cv2.COLOR_BGR2RGB)
                h, w = quantized_rgb.shape[:2]
                normalized_cluster_images = []
                for color in colors:
                    mask = np.linalg.norm(
                        quantized_rgb.astype(np.float32) - color.reshape(1, 1, 3),
                        axis=2
                    ) <= center_match_threshold
                    cluster_vis = np.ones((h, w, 3), dtype=np.uint8) * 255
                    cluster_vis[mask] = color.astype(np.uint8)
                    normalized_cluster_images.append(cluster_vis)
                return normalized_cluster_images

            raise ValueError(f"Unsupported clusters_images ndarray shape: {clusters_images.shape}")

        if isinstance(clusters_images, (list, tuple)):
            normalized_cluster_images = []
            for i, cluster_img in enumerate(clusters_images):
                cluster_img = np.asarray(cluster_img)
                if cluster_img.ndim == 2:
                    cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_GRAY2RGB)
                elif cluster_img.ndim == 3 and cluster_img.shape[2] == 4:
                    cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_RGBA2RGB)
                elif cluster_img.ndim != 3 or cluster_img.shape[2] != 3:
                    raise ValueError(
                        f"Cluster image at index {i} must have shape (H, W, 3), got {cluster_img.shape}"
                    )
                normalized_cluster_images.append(cluster_img)
            return normalized_cluster_images

        raise TypeError("clusters_images must be a list/tuple or numpy array")


    def filter_scattered_figures(self, fig_index, clusters_images, area_ratio_threshold=0.033, scattered_threshold=3):
        """
        过滤fig_index中像素分布分散且占比很少的图像索引

        参数:
            fig_index: 要过滤的fig_index数组
            clusters_images: 聚类图像数组
            area_ratio_threshold: 最小像素占比阈值（低于此值的图像将被剔除）
            scattered_threshold: 分散度阈值（大于此值表示像素分布较分散）

        返回:
            过滤后的fig_index数组
        """
        # 创建一个列表来存储过滤后的索引
        filtered_indices = []

        for idx in fig_index:
            # 获取当前fig图像
            fig_img = clusters_images[idx].copy()
            if fig_img.ndim == 2:
                fig_img = cv2.cvtColor(fig_img, cv2.COLOR_GRAY2RGB)
            elif fig_img.ndim == 3 and fig_img.shape[2] == 4:
                fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2RGB)
            elif fig_img.ndim != 3 or fig_img.shape[2] != 3:
                raise ValueError(f"Cluster image at index {idx} must have shape (H, W, 3), got {fig_img.shape}")
            h, w = fig_img.shape[:2]
            # 将图像转换为灰度图以便处理
            gray = cv2.cvtColor(fig_img, cv2.COLOR_RGB2GRAY)

            # 二值化图像，将非白色像素设为前景
            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

            # 计算总像素数
            total_pixels = gray.shape[0] * gray.shape[1]

            # 计算非白色像素数及其占比
            non_white_pixels = cv2.countNonZero(binary)
            area_ratio = non_white_pixels / total_pixels

            # 连通域分析，判断像素分布是否分散
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # 计算有效连通域数量（排除过小的连通域）
            valid_components = 0
            for i in range(1, num_labels):  # 跳过背景
                area = stats[i, cv2.CC_STAT_AREA]
                # 只考虑有一定面积的连通域
                if area > 100:  # 最小连通域面积阈值
                    valid_components += 1

            # 判断是否保留此索引：占比足够大 或 连通域数量不多(不分散)
            if area_ratio >= area_ratio_threshold :
                filtered_indices.append(idx)
                print(f"保留索引 {idx}: 像素占比={area_ratio:.4f}, 有效连通域数={valid_components}")
            else:
                print(f"剔除索引 {idx}: 像素占比={area_ratio:.4f}, 有效连通域数={valid_components}")
        return filtered_indices

    def classify_colors(self, color_list, bg_color, clusters_images, output_dir="testcluster"):
        """
        分类颜色，确定轮廓、填充、背景和图形索引
        """
        # 将 color_list 转换为 numpy 数组便于处理
        colors = np.array(color_list, dtype=np.float32)

        # 目标颜色
        black = np.array([0, 0, 0], dtype=np.float32)
        white = np.array([255, 255, 255], dtype=np.float32)
        bg = np.array(bg_color, dtype=np.float32)

        # 计算欧氏距离
        dist_to_black = np.linalg.norm(colors - black, axis=1)
        dist_to_white = np.linalg.norm(colors - white, axis=1)
        dist_to_bg = np.linalg.norm(colors - bg, axis=1)

        # 找到最小距离对应的下标  
        ##outline 是黑色的颜色 有可能是结构部分 
        outline_index = int(np.argmin(dist_to_black))
        padding_index = int(np.argmin(dist_to_white))
        bg_index = int(np.argmin(dist_to_bg))

        # 剩余下标（排除上述三个）
        all_indices = set(range(len(color_list)))
        used_indices = set([outline_index, padding_index, bg_index])
        fig_index = list(all_indices - used_indices)

        # 检查fig_index是否为空
        if not fig_index:
            fig_index.append(bg_index)

        # 查看outline所含的元素
        outline_img = clusters_images[outline_index]
        total_pixels = outline_img.shape[0] * outline_img.shape[1]

        # 非白色像素掩膜
        mask_outline = ~(np.all(outline_img == [255, 255, 255], axis=-1))
        non_white_count_outline = np.count_nonzero(mask_outline)
        ratio_outline = non_white_count_outline / total_pixels

        # 非白色像素掩膜
        mask_fig = ~(np.all(clusters_images[fig_index[0]] == [255, 255, 255], axis=-1))
        non_white_count_fig = np.count_nonzero(mask_fig)
        ratio_fig = non_white_count_fig / total_pixels

        mask_bg = ~(np.all(clusters_images[bg_index] == [255, 255, 255], axis=-1))
        non_white_count_outline = np.count_nonzero(mask_bg)
        ratio_bg = non_white_count_outline / total_pixels

        # 如果outline占比较多而fig占比较少，交换两者
        if ratio_outline >= 0.7 and ratio_fig < 0.1:
            print("交换outline与fig")
            _ = fig_index
            fig_index = [outline_index]
            outline_index = _[0]

        # 如果outline和fig占比都比较多 则合并两者
        if ratio_outline >= 0.25 and ratio_fig >= 0.25:
            print("合并outline和fig")
            fig_index.append(outline_index)
        diff_bg_white=np.linalg.norm(bg_color - white, axis=0)
        if  diff_bg_white>20 and ratio_bg >= 0.1:
            print("背景颜色不是白色且 占比很大")
            fig_index.append(bg_index)



        # 保存各个分类的图像
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"outline_index_{color_list[outline_index]}.png"),
                    cv2.cvtColor(clusters_images[outline_index], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"bg_index_{color_list[bg_index]}.png"),
                    cv2.cvtColor(clusters_images[bg_index], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"padding_index_{color_list[padding_index]}.png"),
                    cv2.cvtColor(clusters_images[padding_index], cv2.COLOR_RGB2BGR))

        for i, idx in enumerate(fig_index):
            cv2.imwrite(os.path.join(output_dir, f"fig_index_{i+1:03d}_{color_list[idx]}.png"),
                        cv2.cvtColor(clusters_images[idx], cv2.COLOR_RGB2BGR))

        return outline_index, padding_index, bg_index, fig_index

    def classify_colors_with_priors(
        self,
        color_list,
        bg_color,
        clusters_images,
        output_dir="testcluster_v2",
        white_color_threshold=18.0,
        center_match_threshold=10.0,
        min_fig_area_ratio=0.01,
    ):
        """
        Improved color classification with priors:
        1) Most samples have white padding around borders.
        2) If figure is black-like, background tends to white.
        3) If figure is not black, background tends to bg_color.

        Returns:
            outline_index, padding_index, bg_index, fig_index
        """
        if len(color_list) == 0 or len(clusters_images) == 0:
            raise ValueError("color_list/clusters_images cannot be empty")

        colors = np.array(color_list, dtype=np.float32)
        n_clusters = len(colors)
        white = np.array([255, 255, 255], dtype=np.float32)
        black = np.array([0, 0, 0], dtype=np.float32)
        bg = np.array(bg_color, dtype=np.float32)

        normalized_cluster_images = self._normalize_cluster_images(
            clusters_images,
            color_list=colors,
            center_match_threshold=center_match_threshold,
        )

        if len(normalized_cluster_images) == 0:
            raise ValueError("No cluster images available after normalization")

        h, w = normalized_cluster_images[0].shape[:2]
        total_pixels = float(max(1, h * w))

        border_w = max(2, min(h, w) // 80)
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:border_w, :] = True
        border_mask[-border_w:, :] = True
        border_mask[:, :border_w] = True
        border_mask[:, -border_w:] = True
        border_pixels = float(np.count_nonzero(border_mask))

        cluster_masks = []
        non_white_union = np.zeros((h, w), dtype=bool)
        near_white_indices = []

        # First pass: derive each cluster mask from the visualization image.
        for i, cluster_img in enumerate(normalized_cluster_images):
            if cluster_img.ndim != 3 or cluster_img.shape[2] != 3:
                raise ValueError(f"Cluster image at index {i} must have shape (H, W, 3), got {cluster_img.shape}")

            cluster_img_f = cluster_img.astype(np.float32)
            center = colors[i].reshape(1, 1, 3)
            dist_to_center_map = np.linalg.norm(cluster_img_f - center, axis=2)
            dist_center_to_white = np.linalg.norm(colors[i] - white)

            if dist_center_to_white < white_color_threshold:
                near_white_indices.append(i)
                match_thr = max(4.0, white_color_threshold * 0.45)
            else:
                match_thr = center_match_threshold

            mask = dist_to_center_map <= match_thr
            if dist_center_to_white >= white_color_threshold:
                non_white_union |= mask
            cluster_masks.append(mask)

        # Fix near-pure-white center ambiguity (placeholder white collision).
        for i in near_white_indices:
            if np.linalg.norm(colors[i] - white) <= 2.5:
                cluster_masks[i] = ~non_white_union

        # Collect stats for scoring.
        stats = []
        for i, mask in enumerate(cluster_masks):
            area = int(np.count_nonzero(mask))
            area_ratio = area / total_pixels
            border_hits = int(np.count_nonzero(mask & border_mask))
            border_ratio = border_hits / max(1.0, border_pixels)

            if area > 0:
                ys, xs = np.where(mask)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                bbox_area = float(max(1, (x2 - x1 + 1) * (y2 - y1 + 1)))
                fill_ratio = area / bbox_area
                comp_num, _, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                comp_count = max(0, int(comp_num - 1))
            else:
                fill_ratio = 0.0
                comp_count = 0

            dist_black = float(np.linalg.norm(colors[i] - black))
            dist_white = float(np.linalg.norm(colors[i] - white))
            dist_bg = float(np.linalg.norm(colors[i] - bg))

            stats.append({
                "index": i,
                "area_ratio": float(area_ratio),
                "border_ratio": float(border_ratio),
                "fill_ratio": float(fill_ratio),
                "comp_count": int(comp_count),
                "dist_black": dist_black,
                "dist_white": dist_white,
                "dist_bg": dist_bg,
            })

        # -------- Step 1: background index --------
        max_color_dist = np.sqrt(3.0 * (255.0 ** 2))

        def bg_score(s):
            bg_ref_dist = min(s["dist_bg"], s["dist_white"])
            bg_similarity = 1.0 - (bg_ref_dist / max_color_dist)
            return 0.48 * bg_similarity + 0.34 * s["border_ratio"] + 0.18 * min(1.0, s["area_ratio"] / 0.35)

        bg_index = max(range(n_clusters), key=lambda i: bg_score(stats[i]))

        # -------- Step 2: padding index --------
        remaining = [i for i in range(n_clusters) if i != bg_index]
        if remaining:
            # padding is usually the closest-to-white cluster among non-bg
            padding_index = min(remaining, key=lambda i: stats[i]["dist_white"])
        else:
            padding_index = bg_index

        # -------- Step 3: figure cluster candidates --------
        fig_candidates = []
        for i in range(n_clusters):
            if i == bg_index:
                continue

            s = stats[i]
            # Remove obvious white-border padding from fig candidates.
            if i == padding_index and s["dist_white"] < white_color_threshold and s["border_ratio"] > 0.30:
                continue

            fig_score = (
                0.44 * min(1.0, s["area_ratio"] / 0.14) +
                0.24 * min(1.0, s["fill_ratio"]) +
                0.18 * (1.0 - s["border_ratio"]) +
                0.14 * (1.0 if s["comp_count"] <= 20 else 0.0)
            )
            if s["area_ratio"] >= min_fig_area_ratio or fig_score >= 0.45:
                fig_candidates.append((i, fig_score))

        if not fig_candidates:
            # Fallback: take largest non-bg cluster.
            fallback = max((i for i in range(n_clusters) if i != bg_index), key=lambda i: stats[i]["area_ratio"])
            fig_index = [fallback]
        else:
            fig_candidates.sort(key=lambda x: x[1], reverse=True)
            best_score = fig_candidates[0][1]
            fig_index = [
                i for i, s in fig_candidates
                if (s >= best_score * 0.62 and stats[i]["area_ratio"] >= 0.004)
            ]
            if not fig_index:
                fig_index = [fig_candidates[0][0]]

        fig_index = sorted(set(fig_index))

        # -------- Step 4: outline index --------
        outline_candidates = [i for i in range(n_clusters) if i not in fig_index and i != bg_index]
        if not outline_candidates:
            outline_candidates = [i for i in range(n_clusters) if i != bg_index]

        def outline_score(s):
            black_similarity = 1.0 - min(1.0, s["dist_black"] / 180.0)
            thinness = 1.0 - min(1.0, s["fill_ratio"])
            sparsity = 1.0 - min(1.0, s["area_ratio"] / 0.12)
            comp_bonus = min(1.0, s["comp_count"] / 25.0)
            return 0.36 * black_similarity + 0.26 * thinness + 0.20 * sparsity + 0.10 * comp_bonus + 0.08 * s["border_ratio"]

        outline_index = max(outline_candidates, key=lambda i: outline_score(stats[i]))

        # Avoid choosing near-white cluster as outline when alternatives exist.
        if stats[outline_index]["dist_white"] < white_color_threshold and len(outline_candidates) > 1:
            non_white_outline_cands = [i for i in outline_candidates if stats[i]["dist_white"] >= white_color_threshold]
            if non_white_outline_cands:
                outline_index = max(non_white_outline_cands, key=lambda i: outline_score(stats[i]))

        # Keep outputs mutually consistent.
        fig_index = [i for i in fig_index if i != outline_index and i != bg_index]
        if not fig_index:
            fallback = [i for i in range(n_clusters) if i not in (outline_index, bg_index)]
            fig_index = [max(fallback, key=lambda j: stats[j]["area_ratio"])] if fallback else [outline_index]

        # Save debug outputs.
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(
            os.path.join(output_dir, f"outline_index_{color_list[outline_index]}_v2.png"),
            cv2.cvtColor(normalized_cluster_images[outline_index], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, f"bg_index_{color_list[bg_index]}_v2.png"),
            cv2.cvtColor(normalized_cluster_images[bg_index], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(output_dir, f"padding_index_{color_list[padding_index]}_v2.png"),
            cv2.cvtColor(normalized_cluster_images[padding_index], cv2.COLOR_RGB2BGR),
        )
        for i, idx in enumerate(fig_index):
            cv2.imwrite(
                os.path.join(output_dir, f"fig_index_{i+1:03d}_{color_list[idx]}_v2.png"),
                cv2.cvtColor(normalized_cluster_images[idx], cv2.COLOR_RGB2BGR),
            )

        stats_path = os.path.join(output_dir, "classify_v2_stats.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write("index area_ratio border_ratio fill_ratio comp_count dist_black dist_white dist_bg\n")
            for s in stats:
                f.write(
                    f"{s['index']} {s['area_ratio']:.6f} {s['border_ratio']:.6f} "
                    f"{s['fill_ratio']:.6f} {s['comp_count']} "
                    f"{s['dist_black']:.3f} {s['dist_white']:.3f} {s['dist_bg']:.3f}\n"
                )

        return outline_index, padding_index, bg_index, fig_index

    def compose_figures(self, clusters_images, fig_index):
        """
        组合多个图形图像
        """
        # 取第一幅图作为初始图（拷贝避免修改原图）
        composed = clusters_images[fig_index[0]].copy()

        for idx in fig_index[1:]:
            img = clusters_images[idx]
            # 创建掩膜：判断哪些像素不是白色（即需要"叠加"到 composed 上）
            mask = ~(np.all(img == [255, 255, 255], axis=-1))

            # 只替换白色区域，颜色不透明覆盖
            composed[mask] = img[mask]

        return composed

    def get_max_contour_rect(self,image):
        """
        查找图像中最大轮廓的外接矩形坐标；若无轮廓，返回图像向内收缩10像素的矩形坐标

        参数:
            image: numpy数组，形状为(H, W, 3)，输入的RGB图像（支持uint8或float32/float64类型，
                  float需在0-1范围，全白定义为(255,255,255)或(1.0,1.0,1.0)）

        返回:
            tuple: 矩形坐标(x, y, width, height)，其中：
                  - x, y：矩形左上角坐标
                  - width, height：矩形宽和高
        """
        # 检查输入图像格式
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("输入必须是RGB图像（形状为(H, W, 3)）")

        H, W = image.shape[0], image.shape[1]  # 图像高、宽（H对应y方向，W对应x方向）

        # 1. 图像预处理：转换为uint8并生成二值图（非全白为前景，全白为背景）
        if image.dtype in (np.float32, np.float64):
            # float图像转换为uint8（0-255）
            img_uint8 = (image * 255).astype(np.uint8)
        elif image.dtype == np.uint8:
            img_uint8 = image.copy()
        else:
            raise TypeError("输入图像数据类型必须是uint8或float32/float64（float需在0-1范围）")

        # 二值化：全白像素（255,255,255）为背景（0），其余为前景（255）
        is_white = (img_uint8[:, :, 0] == 255) & \
                   (img_uint8[:, :, 1] == 255) & \
                   (img_uint8[:, :, 2] == 255)
        binary = np.where(is_white, 0, 255).astype(np.uint8)

        # 2. 查找轮廓（只检测外轮廓，简化轮廓点）
        # 注意：OpenCV 4.x中findContours返回(contours, hierarchy)
        contours, hierarchy = cv2.findContours(
            binary,
            mode=cv2.RETR_EXTERNAL,  # 只保留外轮廓
            method=cv2.CHAIN_APPROX_SIMPLE  # 简化轮廓（减少点数量）
        )

        # 3. 判断是否存在轮廓
        if len(contours) == 0:
            # 无轮廓：返回向内收缩10像素的矩形（处理边缘情况）
            shrink = 10
            x = max(0, shrink)
            y = max(0, shrink)
            width = max(0, W - 2 * shrink)
            height = max(0, H - 2 * shrink)
            return (x, y, width, height)
        else:
            # 有轮廓：找到面积最大的轮廓，返回其外接矩形
            # 计算每个轮廓的面积，筛选最大面积的轮廓
            max_area = -1
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour
            # 获取最大轮廓的外接矩形（x:左上角x，y:左上角y，w:宽，h:高）
            x, y, width, height = cv2.boundingRect(max_contour)
            return (x, y, width, height)
    def _parse_box_to_xyxy(self, box, img_w, img_h, box_format="xyxy"):
        """
        Convert one box to clipped integer xyxy coordinates.
        Supported input:
        - [x1, y1, x2, y2] with box_format='xyxy'
        - [x, y, w, h] with box_format='xywh'
        - {'x1','y1','x2','y2'} dict
        - {'xyxy': [...]} dict
        - {'bbox': [...], 'box_format': 'xyxy'|'xywh'} dict
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

    def _refine_arrow_mask_in_patch(self, patch_bgr, border_width=2, min_component_area=8):
        """
        Refine arrow pixels in one patch.
        Returns a binary mask in patch coordinates (255 means remove).
        """
        ph, pw = patch_bgr.shape[:2]
        refined_mask = np.zeros((ph, pw), dtype=np.uint8)
        if ph < 4 or pw < 4:
            return refined_mask

        ring = max(1, min(border_width, min(ph, pw) // 3))
        border_mask = np.zeros((ph, pw), dtype=np.uint8)
        border_mask[:ring, :] = 1
        border_mask[-ring:, :] = 1
        border_mask[:, :ring] = 1
        border_mask[:, -ring:] = 1

        patch_lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        border_lab = patch_lab[border_mask == 1]
        if border_lab.size == 0:
            return refined_mask

        bg_lab = np.median(border_lab, axis=0)
        dist = np.linalg.norm(patch_lab - bg_lab.reshape(1, 1, 3), axis=2)
        border_dist = dist[border_mask == 1]
        adaptive_thr = float(np.percentile(border_dist, 90) + 6.0)
        color_thr = max(18.0, adaptive_thr)
        candidate = np.where(dist >= color_thr, 255, 0).astype(np.uint8)

        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        dark_thr = max(25, int(np.percentile(gray, 35)))
        _, dark_mask = cv2.threshold(gray, dark_thr, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(gray, 40, 120)
        dark_edges = cv2.bitwise_and(dark_mask, edges)
        candidate = cv2.bitwise_or(candidate, dark_edges)

        kernel = np.ones((3, 3), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=1)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        patch_area = float(ph * pw)
        fallback_labels = []

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = float(stats[i, cv2.CC_STAT_AREA])

            if area < min_component_area:
                continue

            box_area = float(max(1, w * h))
            fill_ratio = area / box_area
            elongation = max(w, h) / float(max(1, min(w, h)))
            stroke_proxy = area / float(max(1.0, 2.0 * (w + h)))
            touches_border = (x <= 1) or (y <= 1) or (x + w >= pw - 1) or (y + h >= ph - 1)

            # Skip likely structure blobs.
            if area >= patch_area * 0.60 and not touches_border:
                continue
            if fill_ratio >= 0.80 and not touches_border:
                continue

            score = 0
            if elongation >= 2.0:
                score += 1
            if fill_ratio <= 0.60:
                score += 1
            if touches_border:
                score += 1
            if min(w, h) <= 6:
                score += 1
            if stroke_proxy <= 4.0:
                score += 1
            if area <= patch_area * 0.35:
                score += 1

            if score >= 3:
                refined_mask[labels == i] = 255
            elif touches_border and area <= patch_area * 0.45:
                fallback_labels.append(i)

        # Fallback when strict rules miss the arrow entirely.
        if np.count_nonzero(refined_mask) == 0 and fallback_labels:
            for i in fallback_labels:
                refined_mask[labels == i] = 255

        refined_mask = cv2.dilate(refined_mask, np.ones((2, 2), np.uint8), iterations=1)
        return refined_mask

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
        Build full-image mask from arrow boxes.
        Output mask uses 255 for pixels to be removed.
        """
        if img_bgr is None:
            raise ValueError("img_bgr cannot be None")
        if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
            raise ValueError("img_bgr must be a 3-channel image")

        img_h, img_w = img_bgr.shape[:2]
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if arrow_boxes is None:
            return full_mask

        for box in arrow_boxes:
            parsed = self._parse_box_to_xyxy(box, img_w=img_w, img_h=img_h, box_format=box_format)
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
            patch_mask = self._refine_arrow_mask_in_patch(
                patch_bgr=patch,
                border_width=border_width,
                min_component_area=min_component_area,
            )

            roi = full_mask[ry1:ry2, rx1:rx2]
            full_mask[ry1:ry2, rx1:rx2] = np.maximum(roi, patch_mask)

        return full_mask

    def remove_arrows_by_boxes(
        self,
        img_bgr,
        arrow_boxes,
        box_format="xyxy",
        inpaint_radius=2,
        inpaint_method=cv2.INPAINT_TELEA,
        expand_ratio=0.08,
        expand_pixels=2,
        border_width=2,
        min_component_area=8,
    ):
        """
        Remove arrow pixels based on detection boxes.
        Returns cleaned image and mask.
        """
        arrow_mask = self.build_arrow_mask_from_boxes(
            img_bgr=img_bgr,
            arrow_boxes=arrow_boxes,
            box_format=box_format,
            expand_ratio=expand_ratio,
            expand_pixels=expand_pixels,
            border_width=border_width,
            min_component_area=min_component_area,
        )
        if np.count_nonzero(arrow_mask) == 0:
            return img_bgr.copy(), arrow_mask

        cleaned = cv2.inpaint(img_bgr, arrow_mask, inpaint_radius, inpaint_method)
        return cleaned, arrow_mask

    def image_inpainting(self,img, mask, method=cv2.INPAINT_TELEA):
        """
        修复图像中掩码标记的区域（白色部分修复，黑色部分保留）

        参数:
            img_path: 原始图像路径
            mask_path: 掩码图像路径
            method: 修复算法（cv2.INPAINT_TELEA 或 cv2.INPAINT_NS）
            save_path: 修复结果保存路径
        返回:
            restored: 修复后的图像
        """
        # 读取原始图像（彩色）
        # 预处理掩码：确保二值化（白色255，黑色0），去除可能的噪声
        # 阈值化：将非纯黑的像素都视为待修复区域（白色）
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # 可选：对掩码进行轻微形态学操作，优化边界（避免修复区域边缘过于锐利）
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # 闭合小漏洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # 去除小噪点

        # 确保掩码尺寸与原图一致
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            # 重新阈值化（ resize 可能引入中间值）
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # 执行图像修复
        restored = cv2.inpaint(img, mask, 3, method)  # 第三个参数为修复邻域半径（通常3-5）

        return restored
    def repair_img(self, color, binary_img):
        """
        修复图像中的断裂线条
        """
        h, w = binary_img.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)  # 形状 (h, w, 3)
        result = 255 - result
        kernel_erode = np.ones((1, 1), np.uint8)
        red_mask = cv2.erode(binary_img, kernel_erode, iterations=3)

        # 3. 修复断裂线条（针对细线条优化）
        # 用细长结构元素进行闭运算，优先连接纵向/横向断裂（适应合花格对称结构）
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 小核避免线条变粗
        closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close, iterations=6)

        # 4. 修复较大缺口（保留线条边缘特征）
        # 生成修补掩码（标记断裂区域）
        mask = cv2.subtract(closed, red_mask)
        mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        # 转换为3通道用于inpaint
        red_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        repaired = cv2.inpaint(red_rgb, mask, 3, cv2.INPAINT_TELEA)  # 小半径保留细节

        # 5. 还原红色线条到原图（避免颜色丢失）
        # 提取修复后的线条
        repaired_gray = cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY)
        _, final_mask = cv2.threshold(repaired_gray, 127, 255, cv2.THRESH_BINARY)

        # 将修复后的红色线条叠加回原图
        result[final_mask == 255] = color  # 用指定颜色修复

        return result

    def filter_small_connected_components(self,rgb_image, connectivity=8, background_value=255):
        """
        对RGB图像进行连通域检测，二值化逻辑为：仅全白像素为背景，其余为前景，
        并过滤掉像素数量小于总像素数2%的连通域

        参数:
            rgb_image: numpy数组，形状为(H, W, 3)，输入的RGB图像（数据类型可为uint8或float32/float64，
                      float需在0-1范围，全白定义为(1.0,1.0,1.0)，uint8全白定义为(255,255,255)）
            connectivity: 连通域连接方式，4或8（默认8连通）
            background_value: 过滤后小连通域区域的像素值（默认0，即黑色；若为float图像建议设为0.0）

        返回:
            filtered_image: numpy数组，形状与输入一致，过滤掉小连通域后的RGB图像（小连通域区域设为background_value）
        """
        # 检查输入图像格式
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("输入必须是RGB图像（形状为(H, W, 3)）")

        # 确定全白的阈值（根据数据类型）
        if rgb_image.dtype in (np.float32, np.float64):
            # float类型图像：全白为(1.0, 1.0, 1.0)，允许微小精度误差（如1.0000001视为全白）
            white_threshold = 1.0 - 1e-6  # 宽松判断，避免浮点精度问题
            # 判断是否为全白像素：三个通道均≥white_threshold
            is_white = (rgb_image[:, :, 0] >= white_threshold) & \
                       (rgb_image[:, :, 1] >= white_threshold) & \
                       (rgb_image[:, :, 2] >= white_threshold)
        elif rgb_image.dtype == np.uint8:
            # uint8类型图像：全白为(255, 255, 255)
            is_white = (rgb_image[:, :, 0] == 255) & \
                       (rgb_image[:, :, 1] == 255) & \
                       (rgb_image[:, :, 2] == 255)
        else:
            raise TypeError("输入图像数据类型必须是uint8或float32/float64（float需在0-1范围）")

        # 生成二值图：全白为背景（0），非全白为前景（255）
        binary = np.where(is_white, 0, 255).astype(np.uint8)

        # 连通域检测（返回标签、统计信息等）
        # stats格式：[x, y, width, height, area]（area为连通域像素数量）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)

        # 计算总像素数和最小阈值（总像素的2%）
        total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
        min_pixel_threshold = 0.02 * total_pixels

        # 创建过滤后的图像（复制原图）
        filtered_image = rgb_image.copy()

        # 遍历所有连通域（跳过标签0，即背景：全白区域）
        for label in range(1, num_labels):
            # 若连通域像素数小于阈值，则标记为背景值
            if stats[label, 4] < min_pixel_threshold:
                filtered_image[labels == label] = background_value

        return filtered_image

    def fill_hole(self,rgb_image, connectivity=8, threshold_method='otsu', background_value=[0, 0, 0]):
        """
        对RGB图像进行连通域检测，并过滤掉像素数量小于总像素数2%的连通域

        参数:
            rgb_image: numpy数组，形状为(H, W, 3)，输入的RGB图像（数据类型可为uint8或float32/float64，float需在0-1范围）
            connectivity: 连通域连接方式，4或8（默认8连通）
            threshold_method: 二值化方法，目前支持'otsu'（自动阈值，默认）
            background_value: 过滤后背景区域的像素值（默认0，即黑色）

        返回:
            filtered_image: numpy数组，形状与输入一致，过滤掉小连通域后的RGB图像（小连通域区域设为background_value）
        """
        # 检查输入图像格式
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("输入必须是RGB图像（形状为(H, W, 3)）")

        # 转换为灰度图像
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

        # 统一数据类型为uint8（0-255范围）
        if gray.dtype in (np.float32, np.float64):
            # 假设float图像像素值在0-1范围内
            gray = (gray * 255).astype(np.uint8)
        elif gray.dtype != np.uint8:
            raise TypeError("输入图像数据类型必须是uint8或float32/float64（float需在0-1范围）")

        # 二值化处理（区分前景和背景）
        if threshold_method == 'otsu':
            # Otsu算法自动确定阈值
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError("目前仅支持'otsu'阈值方法")

        # 连通域检测（返回标签、统计信息等）
        # stats格式：[x, y, width, height, area]（area为连通域像素数量）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)

        # 计算总像素数和最小阈值（总像素的2%）
        total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
        min_pixel_threshold = 0.02 * total_pixels

        # 创建过滤后的图像（复制原图）
        filtered_image = rgb_image.copy()

        # 遍历所有连通域（跳过标签0，即背景）
        for label in range(1, num_labels):
            # 若连通域像素数小于阈值，则标记为背景
            if stats[label, 4] < min_pixel_threshold:
                filtered_image[labels == label, 0] = background_value[0]  # R通道
                filtered_image[labels == label, 1] = background_value[1]  # G通道
                filtered_image[labels == label, 2] = background_value[2]  # B通道

        return filtered_image
    def process_edges(self, image_path, output_path):
        """
        边缘处理主函数：连接断点 + 连通域分析 + 小区域过滤

        参数:
            image_path: 输入图像路径
            output_path: 输出路径
            min_area: 最小连通域面积阈值

        返回:
            处理结果字典
        """
        # 使用默认参数或传入的参数

        # 读取图像
        img = cv2.imread(image_path)
        results = []
        if img is None:
            raise FileNotFoundError(f"图像文件未找到: {image_path}")
        # cv2.imshow("original_color", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ### 通过OCR和轮廓的方式分离此图 需要优化
        imgs=self.process_image(img)


        # 统一转为列表格式
        if isinstance(imgs, list):
            img_list = imgs
        elif isinstance(imgs, np.ndarray):
            img_list = [imgs] if imgs.ndim == 3 else [imgs[i] for i in range(imgs.shape[0])]
        else:
            img_list = [imgs]  # 兼容PIL Image等单图对象
        for img in img_list:

            img = cv2.resize(img, (845, 845))

            original_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ##检测背景颜色  效果还可以
            bg_color = Preprocess_tool.detect_background_color(original_color)
            print(f"检测到的背景颜色: {bg_color}")

            ##用keans检测颜色并分离 

            clusters_images, clusters_colors = self.auto_kmeans_color_quantization(img, output_path)
            clusters_images = self._normalize_cluster_images(
                clusters_images,
                color_list=clusters_colors,
            )
            ##检测各个分离颜色 猜测可能的成分 最接近(000)的颜色为轮廓图  最接近255 255 255的是灰度信息  剩下的是主体
            outline_index, padding_index, bg_index, fig_index = self.classify_colors_with_priors(
                color_list=clusters_colors,
                bg_color=bg_color,
                clusters_images=clusters_images,
                output_dir=output_path
            )

            # all_index = copy.copy(fig_index)
            # all_index.append(outline_index)
            # if (padding_index != bg_index):
            #     all_index.append(padding_index)

            print(outline_index, padding_index, bg_index, fig_index)
            ##根据联通域分析  过滤掉过于散落的子图   需要优化 
            filt_fig_index = self.filter_scattered_figures(fig_index, clusters_images)
            all_index = copy.copy(filt_fig_index)
            all_index.append(outline_index)
            if (padding_index != bg_index):
                all_index.append(padding_index)
            try:
                main_fig = self.compose_figures(clusters_images, filt_fig_index)
            except Exception as e:
                print(f"该图像无法处理（可能是电路图）: {image_path}, 错误信息: {e}")
                continue  # 继续处理下一张图像

            origin_img = self.compose_figures(clusters_images, all_index)

            diff = cv2.absdiff(origin_img, main_fig)
            # 处理通道差异（彩色图取三通道最大差异，灰度图直接使用）
            if len(img.shape) == 3:  # 彩色图（3通道）
                # 每个像素取三通道中的最大差异值，转为单通道（shape: (h, w)）
                max_diff = np.max(diff, axis=2)
            else:  # 灰度图（单通道）
                max_diff = diff

            # 生成mask：差异>阈值设为255（白色），否则设为0（黑色）
            _, mask = cv2.threshold(max_diff, 50
                                    , 255, cv2.THRESH_BINARY)

            # 转换为灰度图



            gray = cv2.cvtColor(main_fig, cv2.COLOR_BGR2GRAY)

            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            binary_inv = cv2.bitwise_not(binary)  # 将黑色线条变为白色前景（前景=255）

            process_binary = cv2.bitwise_not(binary_inv)
            # 统计黑色像素点 检测连通域 如果小于0.008*总黑色像素 则认为是噪声区域并擦除
            black_pixels = np.count_nonzero(process_binary == 0)
            min_area = 0.008 * black_pixels
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_inv, connectivity=8, ltype=cv2.CV_32S
            )
            # 创建过滤后的边缘图（仅保留大于阈值的连通域）
            filtered_binary = np.zeros_like(process_binary)
            # 处理每个连通域
            valid_labels = 0

            # 转换为灰度图
            for i in range(1, num_labels):
                # 获取当前连通域的统计信息 有bug呀
                area = stats[i, cv2.CC_STAT_AREA]

                # 过滤小面积区域
                if area < min_area:
                    continue

                # 标记有效连通域计数
                valid_labels += 1

                # 在过滤图中保留此区域
                filtered_binary[labels == i] = 255

            filtered_binary = cv2.bitwise_not(filtered_binary)
            if self.show_debug_windows:
                cv2.imshow('11', filtered_binary)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            #### 调整一下图片 将图片中过小的部分调整为背景颜色  以便后续修复
            adjust_fig = self.adjust_image_by_mask(original_color_img=main_fig, binary_mask=filtered_binary,
                                                   bg_color=bg_color)
            # cv2.imshow('11', adjust_fig )
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            adjust_fig = cv2.cvtColor(adjust_fig, cv2.COLOR_RGB2BGR)
            repair_fig = cv2.cvtColor(self.image_inpainting(img=adjust_fig, mask=mask), cv2.COLOR_RGB2BGR)
            # repair_fig = self.repair_img(color=center_colors[fig_index[0]], binary_img=mask)

            result = self.filter_small_connected_components(repair_fig, connectivity=8, background_value=255)

            ####后续要改
            if (len(fig_index)) > 1:
                result2 = self.fill_hole(result, connectivity=8, background_value=clusters_colors[fig_index[1]])
            else:
                result2 = self.fill_hole(result, connectivity=8, background_value=clusters_colors[fig_index[0]])
            print(f"原始连通域数量: {num_labels - 1}, 过滤后保留: {valid_labels}")

            results.append({
                "original": img,
                "binary": binary,
                "process_binary": filtered_binary,
                "bg_fig": clusters_images[bg_index],
                "adjust_fig": adjust_fig,
                "repair_fig": result2,
                "main_fig": main_fig,
                "mask": mask
            })
        return results
    def visualize_results(self, results):
        """
        可视化处理结果
        """
        plt.figure(figsize=(15, 10))

        # 原始图像
        plt.subplot(231)
        plt.imshow(results['original'])
        plt.title('Original Image')
        plt.axis('off')

        # 二值图像
        plt.subplot(232)
        plt.imshow(results['binary'], cmap='gray')
        plt.title('Binary Image')
        plt.axis('off')

        # 处理后的二值图（灰度）
        plt.subplot(233)
        plt.imshow(results['process_binary'], cmap='gray')
        plt.title('Processed Binary (Grayscale)')
        plt.axis('off')

        # 带原图颜色的结果
        plt.subplot(234)
        plt.imshow(results['adjust_fig'])
        plt.title('adjust_fig')
        plt.axis('off')

        # Canny边缘
        plt.subplot(235)
        plt.imshow(results['bg_fig'])
        plt.title('bg_fig')
        plt.axis('off')

        # 彩色连通域
        plt.subplot(236)
        plt.imshow(results['repair_fig'])
        plt.title(f'repair_fig')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def _collect_image_files(self, input_path):
        """
        收集输入文件夹中的图片文件。
        """
        if not os.path.isdir(input_path):
            raise FileNotFoundError(f"输入文件夹不存在: {input_path}")

        image_files = []
        for name in sorted(os.listdir(input_path)):
            file_path = os.path.join(input_path, name)
            if not os.path.isfile(file_path):
                continue
            if os.path.splitext(name)[1].lower() in self.supported_image_extensions:
                image_files.append(file_path)

        return image_files

    def _extract_text_info(self, image_path):
        """
        如果配置了 OCR 引擎，则按当前图片更新文本信息。
        """
        if self.ocr_engine is None:
            return self.text_info_list

        text_info_list = []
        result = self.ocr_engine.predict(image_path)
        for res in result:
            boxes = res['rec_polys']
            texts = res['rec_texts']
            scores = res['rec_scores']

            for box, text, score in zip(boxes, texts, scores):
                text_info_list.append({
                    "box": box,
                    "text": text,
                    "score": score
                })

        self.text_info_list = text_info_list
        return text_info_list

    def _detect_single_image(self, image_path, output_folder=None, visualize=False, refresh_text_info=True):
        """
        处理单张图片并保存结果。
        """
        if refresh_text_info:
            self._extract_text_info(image_path)

        if output_folder is None:
            output_folder = os.path.splitext(os.path.basename(image_path))[0]

        os.makedirs(output_folder, exist_ok=True)

        results = self.process_edges(image_path=image_path, output_path=output_folder)

        image_name = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name)[0]

        for index, result in enumerate(results):
            self._save_results(result, output_folder, image_name_no_ext, index)

        if visualize:
            for result in results:
                self.visualize_results(result)

        return results

    def detect(self, image_path, output_folder=None, visualize=False):
        """
        主要的检测函数，整合所有处理步骤

        参数:
            image_path: 输入图像路径
            output_folder: 输出文件夹路径（默认为图像文件名对应的文件夹）
            visualize: 是否可视化结果

        返回:
            处理结果字典
        """
        # 设置输出文件夹
        if output_folder is None:
            output_folder = os.path.splitext(os.path.basename(image_path))[0]

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 处理图像
        results = self.process_edges(image_path=image_path, output_path=output_folder)

        # 分离文件名和扩展名，然后重新组合
        image_name = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name)[0]

        # 保存结果
        for index, result in enumerate(results):
            self._save_results(result, output_folder, image_name_no_ext, index)

        # 可视化结果
        if visualize:
            self.visualize_results(results)

        return results

    def detect(self, image_path, output_folder=None, visualize=False):
        """
        支持单张图片或图片文件夹的统一检测入口。
        """
        if os.path.isdir(image_path):
            image_files = self._collect_image_files(image_path)
            if not image_files:
                raise ValueError(f"文件夹中未找到可处理的图片: {image_path}")

            if output_folder is None:
                folder_name = os.path.basename(os.path.normpath(image_path))
                output_folder = f"{folder_name}_results"

            os.makedirs(output_folder, exist_ok=True)

            batch_results = []
            for single_image_path in image_files:
                image_name_no_ext = os.path.splitext(os.path.basename(single_image_path))[0]
                single_output_folder = os.path.join(output_folder, image_name_no_ext)
                print(f"开始处理图片: {single_image_path}")
                results = self._detect_single_image(
                    image_path=single_image_path,
                    output_folder=single_output_folder,
                    visualize=visualize,
                    refresh_text_info=True
                )
                batch_results.append({
                    "image_path": single_image_path,
                    "output_folder": single_output_folder,
                    "results": results
                })

            return batch_results

        return self._detect_single_image(
            image_path=image_path,
            output_folder=output_folder,
            visualize=visualize,
            refresh_text_info=True
        )

    def _save_results(self, results, output_folder, image_name_no_ext, index):
        """
        保存处理结果到指定文件夹
        根据索引将不同的结果保存到不同的子文件夹中

        参数:
            results: 处理结果字典
            output_folder: 主输出文件夹
            image_name_no_ext: 图像文件名（不含扩展名）
            index: 当前处理的图像索引，用于创建子文件夹名称
        """
        # 创建子文件夹路径
        sub_folder = os.path.join(output_folder, f"{index:03d}")  # 子文件夹名为 000, 001, ...

        # 如果子文件夹不存在，则创建
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        # 保存关键结果到子文件夹
        cv2.imwrite(os.path.join(sub_folder, "adjust_fig.png"), results['adjust_fig'])
        cv2.imwrite(os.path.join(sub_folder, "bg_fig.png"), cv2.cvtColor(results['bg_fig'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(sub_folder, "process.png"), results['original'])
        cv2.imwrite(os.path.join(sub_folder, "repair_fig.png"), cv2.cvtColor(results['repair_fig'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(sub_folder, f'{image_name_no_ext}_mask.png'), results['mask'])
        cv2.imwrite(os.path.join(sub_folder, f'{image_name_no_ext}.png'), results['adjust_fig'])


# 示例用法
if __name__ == "__main__":
    # 创建检测器实例
    detector = FSSfigDetector(max_k=6)

    # 指定图像路径
    image_path = "./fss_out/024.png"  # 图像路径

    # 执行检测
    results = detector.detect(image_path)

    print("检测完成！")
