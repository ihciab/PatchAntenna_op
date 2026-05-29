"""
图像预处理器模块
原代码位置：Rebuild/FSSfigDetector.py

功能：
- 图像裁剪和预处理
- 颜色量化和分类
- 图像修复和边缘处理
- FSS结构图检测和处理

重构说明：
- 将原 FSSfigDetector 类重构为 ImagePreprocessor 类
- 保持原有算法和逻辑完全不变
- 保持原有接口方法不变
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class ImagePreprocessor:
    """
    图像预处理器类
    原类：Rebuild/FSSfigDetector.py 中的 FSSfigDetector
    
    用于检测和处理图像中的图形元素，包含颜色量化、图形识别、边缘处理等功能
    """
    """
    FSSfigDetector类用于检测和处理图像中的图形元素
    包含颜色量化、图形识别、边缘处理等功能
    """

    def __init__(self, max_k=6, min_color_diff=30):
        """
        初始化检测器
        参数:
            max_k: 颜色量化的最大k值
            min_color_diff: 最小颜色差异阈值
        """

        self.max_k = max_k
        self.min_color_diff = min_color_diff

    def process_image(self,img,expand_pixels=15):
        """
        处理单张图像：判断是否存在左右两张图片，若存在则保留接近正方形的部分，否则返回原图

        参数:
            image_path: 输入图像路径
            save_dir: 结果保存目录
        返回:
            processed_img: 处理后的图像
        """
        # 创建保存目录

        h, w = img.shape[:2]
        original = img.copy()

        # 预处理：灰度化 + 模糊 + 边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)  # 可根据图像调整阈值

        # 形态学操作：连接边缘，减少断裂
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 查找轮廓（只保留外轮廓，减少内部细节干扰）
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"图像未检测到轮廓，返回原图")
            return original

        # 筛选面积较大的轮廓（排除小噪声）
        min_area_ratio = 0.05  # 最小轮廓面积为图像总面积的5%
        min_area = min_area_ratio * h * w
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        if len(large_contours) < 2:
            # 不足2个大轮廓，判断为单张图像
            print(f"图像未检测到左右两张图片，返回原图")
            return original

        # 对大轮廓按面积排序（取前2个最大的）
        large_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)[:2]

        # 提取轮廓的边界框信息（x:左上角x，y:左上角y，w:宽度，h:高度）
        bboxes = [cv2.boundingRect(c) for c in large_contours]

        # 判断两个轮廓是否分布在左右两侧（有明显间隔）
        def is_horizontal_distribution(bbox1, bbox2):
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2

            # 1. 计算垂直方向（y轴）重叠比例（判断是否水平对齐）
            # 轮廓1的y范围：[y1, y1+h1]，轮廓2的y范围：[y2, y2+h2]
            y_overlap_start = max(y1, y2)
            y_overlap_end = min(y1 + h1, y2 + h2)
            y_overlap = max(0, y_overlap_end - y_overlap_start)  # 重叠高度

            # 重叠比例 = 重叠高度 / 两个轮廓中较小的高度（确保至少50%重叠）
            min_height = min(h1, h2)
            y_overlap_ratio = y_overlap / min_height if min_height > 0 else 0

            # 2. 判断水平方向（x轴）是否左右分开（无重叠或少量重叠）
            # 轮廓1的x范围：[x1, x1+w1]，轮廓2的x范围：[x2, x2+w2]
            x_overlap_start = max(x1, x2)
            x_overlap_end = min(x1 + w1, x2 + w2)
            x_overlap = max(0, x_overlap_end - x_overlap_start)  # 水平重叠宽度

            # 水平重叠比例需≤20%（确保左右分开）
            min_width = min(w1, w2)
            x_overlap_ratio = x_overlap / min_width if min_width > 0 else 1

            # 水平分布条件：垂直重叠≥50% 且 水平重叠≤20%
            return y_overlap_ratio >= 0.5 and x_overlap_ratio <= 0.2

            # 判断是否为水平分布的左右轮廓

        if not is_horizontal_distribution(bboxes[0], bboxes[1]):
            print(f"图像非水平分布，返回原图")
            return original

        # 计算两个轮廓的宽高比（接近1表示接近正方形）
        def get_aspect_ratio(bbox):
            x, y, w_b, h_b = bbox
            return w_b / h_b if h_b != 0 else 0  # 宽/高

        ratios = [get_aspect_ratio(bbox) for bbox in bboxes]

        # 筛选接近正方形的轮廓（宽高比在0.8~1.2之间）
        square_like_idx = -1
        for i, ratio in enumerate(ratios):
            if 0.8 <= ratio <= 1.2:  # 可根据实际需求调整阈值
                square_like_idx = i
                break

        if square_like_idx == -1:
            # 两个轮廓都不接近正方形，保留面积较大的那个
            square_like_idx = 0  # 取面积大的（已排序）
            print(f"图像未找到接近正方形的轮廓，保留面积较大的部分")

        # 裁剪并提取目标区域
        target_contour = large_contours[square_like_idx]
        x, y, width, height = cv2.boundingRect(target_contour)

        # 计算原始边界坐标
        min_x, min_y = x, y
        max_x, max_y = x + width, y + height

        # 扩展边界，但确保不超出图像范围
        expanded_min_x = max(0, min_x - expand_pixels)
        expanded_min_y = max(0, min_y - expand_pixels)
        expanded_max_x = min(w, max_x + expand_pixels)
        expanded_max_y = min(h, max_y + expand_pixels)


        # mask = np.zeros_like(gray)
        # cv2.drawContours(mask, [target_contour], -1, 255, -1)  # 填充轮廓为掩码
        # processed_img = cv2.bitwise_and(original, original, mask=mask)
        cropped_img = img[expanded_min_y:expanded_max_y, expanded_min_x:expanded_max_x]
        return cropped_img
    def detect_background_color(self, image, white_threshold=240, white_ratio_threshold=0.3,
                                edge_crop_ratio=0.01, dark_threshold=30,
                                edge_band_width=5, dominant_ratio_threshold=0.2):
        """
        优化白色背景识别的检测函数

        :param image: 输入图像 (BGR格式)
        :param white_threshold: 白色判断阈值（默认240，降低阈值更易识别白色）
        :param white_ratio_threshold: 白色占比阈值（超过此比例则判定为白色背景）
        :param edge_crop_ratio: 最外层边缘裁剪比例
        :param dark_threshold: 深色背景判断阈值
        :param edge_band_width: 边缘带宽度（像素数）
        :param dominant_ratio_threshold: 主要颜色占比阈值
        :return: 背景颜色 (B, G, R)
        """
        # 输入验证
        if image is None or len(image.shape) != 3:
            return (255, 255, 255)

        height, width = image.shape[:2]

        # 将图像分为四个区域
        regions = [
            image[0:int(height / 2), 0:int(width / 2)],  # 左上
            image[0:int(height / 2), int(width / 2):width],  # 右上
            image[int(height / 2):height, 0:int(width / 2)],  # 左下
            image[int(height / 2):height, int(width / 2):width]  # 右下
        ]

        region_backgrounds = []

        for region in regions:
            r_height, r_width = region.shape[:2]
            if r_height < edge_band_width * 4 or r_width < edge_band_width * 4:
                continue

            # 轻微裁剪最外层可能的干扰
            margin_h = max(1, int(r_height * edge_crop_ratio))
            margin_w = max(1, int(r_width * edge_crop_ratio))
            cropped = region[margin_h:r_height - margin_h, margin_w:r_width - margin_w]
            c_height, c_width = cropped.shape[:2]

            # 提取区域的边缘带像素
            top_edge = cropped[0:edge_band_width, :]
            bottom_edge = cropped[c_height - edge_band_width:c_height, :]
            left_edge = cropped[edge_band_width:c_height - edge_band_width, 0:edge_band_width]
            right_edge = cropped[edge_band_width:c_height - edge_band_width, c_width - edge_band_width:c_width]

            # 合并所有边缘像素
            edge_pixels = np.concatenate([
                top_edge.reshape(-1, 3),
                bottom_edge.reshape(-1, 3),
                left_edge.reshape(-1, 3),
                right_edge.reshape(-1, 3)
            ])

            # 分析整个区域的像素（不预先过滤白色，这是识别白色背景的关键）
            all_pixels = cropped.reshape(-1, 3)

            # 专门检测白色背景：统计区域中接近白色的像素比例
            white_pixel_mask = np.all(all_pixels >= white_threshold, axis=1)
            white_ratio = np.sum(white_pixel_mask) / len(all_pixels)

            if white_ratio >= white_ratio_threshold:
                # 白色像素占比足够高，直接判定为白色背景
                region_color = (255, 255, 255)
            else:
                # 非白色背景，进行常规处理（过滤白边）
                valid_all_pixels = all_pixels[~white_pixel_mask]
                valid_edge_pixels = edge_pixels[~np.all(edge_pixels >= white_threshold, axis=1)]

                if len(valid_all_pixels) == 0:
                    region_color = (255, 255, 255)
                else:
                    # 统计区域内主要颜色
                    unique_colors, counts = np.unique(valid_all_pixels, axis=0, return_counts=True)
                    max_count_idx = np.argmax(counts)
                    dominant_color = unique_colors[max_count_idx]
                    dominant_ratio = counts[max_count_idx] / len(valid_all_pixels)

                    # 主要颜色占比低时，使用边缘颜色
                    if dominant_ratio < dominant_ratio_threshold and len(valid_edge_pixels) > 0:
                        edge_colors, edge_counts = np.unique(valid_edge_pixels, axis=0, return_counts=True)
                        if len(edge_colors) > 0:
                            dominant_color = edge_colors[np.argmax(edge_counts)]

                    # 处理深色背景
                    color_avg = np.mean(dominant_color)
                    if color_avg <= dark_threshold:
                        region_color = (255, 255, 255)
                    else:
                        region_color = tuple(int(c) for c in dominant_color)

            region_backgrounds.append(region_color)

        if not region_backgrounds:
            return (255, 255, 255)

        # 取四个区域中出现次数最多的颜色
        color_counts = Counter(region_backgrounds)
        most_common_color = color_counts.most_common(1)[0][0]

        return most_common_color

    def adjust_image_by_mask(self, original_color_img, binary_mask):
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
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
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
        quantized_pixels_lab = centers_rgb[labels]
        # quantized_img_lab = quantized_pixels_lab.reshape((height, width, channels))
        # quantized_img_rgb = cv2.cvtColor( centers_rgb, cv2.COLOR_LAB2RGB)
        quantized_img_rgb = quantized_pixels_lab.reshape((height, width, channels))
        # 创建颜色分布图
        print("创建颜色分布可视化...")
        cluster_visualizations = self.visualize_color_clusters(
            img_rgb,
            labels.reshape(height, width),
            centers_rgb,
            output_dir=output_path,
        )

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
        return quantized_bgr, cluster_visualizations, centers_rgb

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
            h, w, _ = fig_img.shape
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

        # 保存各个分类的图像
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"outline_index_{color_list[outline_index]}.png"),
                    cv2.cvtColor(clusters_images[outline_index], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"bg_index_{color_list[bg_index]}.png"),
                    cv2.cvtColor(clusters_images[bg_index], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"padding_index_{color_list[padding_index]}.png"),
                    cv2.cvtColor(clusters_images[padding_index], cv2.COLOR_RGB2BGR))

        if len(fig_index) > 0:
            cv2.imwrite(os.path.join(output_dir, f"fig_index001_{color_list[fig_index[0]]}.png"),
                        cv2.cvtColor(clusters_images[fig_index[0]], cv2.COLOR_RGB2BGR))

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

    def image_inpainting(self,img, mask, method=cv2.INPAINT_NS):
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

    def repair_image2(self,img, colors):
        """
        修复图像中的小孔洞缺陷

        参数:
            img: 输入图像，应为RGB格式的numpy数组
            colors: 用于修复的颜色数组，每个颜色应为(r, g, b)元组

        返回:
            修复后的RGB图像
        """
        # 确保输入是RGB格式
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError("输入图像必须是RGB格式")

        # 复制原始图像用于修复
        result = img.copy()

        # 1. 将图像转换为灰度图以便二值化
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. 二值化处理 - 假设白色(高亮度)区域包含缺陷和背景
        # 使用Otsu自适应阈值寻找最佳阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. 寻找连通区域并筛选出面积较小的区域作为待修复区域
        # 查找所有连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # 计算所有区域的面积分布，确定"小面积"的阈值
        areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过第一个区域(背景)
        if len(areas) == 0:
            return result  # 没有找到需要修复的区域

        # 使用K-means聚类将区域分为"小"和"大"两类
        kmeans = KMeans(n_clusters=2, random_state=42).fit(areas.reshape(-1, 1))
        cluster_centers = kmeans.cluster_centers_.flatten()
        small_area_threshold = np.min(cluster_centers)  # 较小的聚类中心作为阈值

        # 4. 对每个小区域进行修复
        for i in range(1, num_labels):  # 从1开始，跳过背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= small_area_threshold:  # 只处理小面积区域
                # 获取该区域的掩码
                mask = (labels == i).astype(np.uint8) * 255

                # 找到区域的边界框
                x, y, w, h, _ = stats[i]

                # 扩展边界框以获取周围像素
                expand = 5
                x1 = max(0, x - expand)
                y1 = max(0, y - expand)
                x2 = min(img.shape[1], x + w + expand)
                y2 = min(img.shape[0], y + h + expand)

                # 获取周围区域的像素
                surrounding_region = img[y1:y2, x1:x2]
                surrounding_mask = cv2.bitwise_not(mask[y1:y2, x1:x2]) / 255

                # 收集周围非缺陷区域的像素
                surrounding_pixels = []
                for c in range(3):  # RGB三个通道
                    channel = surrounding_region[:, :, c]
                    masked = channel * surrounding_mask
                    surrounding_pixels.extend(masked[masked > 0].tolist())

                if not surrounding_pixels:
                    # 如果没有周围像素，使用颜色列表中的第一个颜色
                    best_color = colors[0]
                else:
                    # 计算周围像素的平均颜色
                    avg_color = np.mean(np.array(surrounding_pixels).reshape(-1, 3), axis=0)

                    # 从颜色列表中找到与平均颜色最接近的颜色
                    min_distance = float('inf')
                    best_color = colors[0]
                    for color in colors:
                        distance = np.sqrt(np.sum((np.array(color) - avg_color) ** 2))
                        if distance < min_distance:
                            min_distance = distance
                            best_color = color

                # 5. 用选定的颜色填充缺陷区域
                result[labels == i] = best_color

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
        if img is None:
            raise FileNotFoundError(f"图像文件未找到: {image_path}")
        img=self.process_image(img)

        img = cv2.resize(img, (800, 800))
        # img = cv2.resize(img, (768, 768))

        original_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ##检测背景颜色
        bg_color = self.detect_background_color(original_color)
        print(f"检测到的背景颜色: {bg_color}")

        ##用keans检测颜色并分离
        quantized_bgr, clusters_images, center_colors = self.auto_kmeans_color_quantization(img, output_path)
        ##检测各个分离颜色 猜测可能的成分 最接近(000)的颜色为轮廓图  最接近255 255 255的是灰度信息  剩下的是主体
        outline_index, padding_index, bg_index, fig_index = self.classify_colors(
            color_list=center_colors,
            bg_color=bg_color,
            clusters_images=clusters_images,
            output_dir=output_path
        )

        all_index = copy.copy(fig_index)
        all_index.append(outline_index)
        if (padding_index != bg_index):
            all_index.append(padding_index)

        print(outline_index, padding_index, bg_index, fig_index)
        filt_fig_index = self.filter_scattered_figures(fig_index, clusters_images)

        main_fig = self.compose_figures(clusters_images, filt_fig_index)


        origin_img = self.compose_figures(clusters_images, all_index)


        diff = cv2.absdiff(origin_img, main_fig)
        # 处理通道差异（彩色图取三通道最大差异，灰度图直接使用）
        if len(img.shape) == 3:  # 彩色图（3通道）
            # 每个像素取三通道中的最大差异值，转为单通道（shape: (h, w)）
            max_diff = np.max(diff, axis=2)
        else:  # 灰度图（单通道）
            max_diff = diff

        # 生成mask：差异>阈值设为255（白色），否则设为0（黑色）
        _, mask = cv2.threshold(max_diff, 10, 255, cv2.THRESH_BINARY)

        # 转换为灰度图
        gray = cv2.cvtColor(main_fig, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary_inv = cv2.bitwise_not(binary)  # 将黑色线条变为白色前景（前景=255）


        process_binary = cv2.bitwise_not(binary_inv)
        process_binary = cv2.medianBlur(process_binary, 3)
        # 统计黑色像素点 检测连通域 如果小于0.1*总黑色像素 则认为是噪声区域并擦除
        black_pixels = np.count_nonzero(process_binary == 0)
        min_area = 0.02 * black_pixels
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
            x, y, w, h, area = stats[i]

            # 过滤小面积区域
            if area < min_area:
                #print("11")
                continue

            # 标记有效连通域计数
            valid_labels += 1

            # 在过滤图中保留此区域
            filtered_binary[labels == i] = 255



        filtered_binary=cv2.medianBlur(filtered_binary, 3)
        filtered_binary=cv2.bitwise_not( filtered_binary)
        # cv2.imshow(f"<UNK>", filtered_binary)
        # cv2.waitKey(0)
        adjust_fig, cropped_mask = self.auto_crop_color_from_mask(original_color_img=main_fig,
                                                                  binary_mask=filtered_binary)


        #repair_fig = self.repair_img(color=center_colors[fig_index[0]], binary_img=cropped_mask)


        # 改用lama模型后 repair_fig不再需要
        adjust_fig = self.adjust_image_by_mask(original_color_img=main_fig, binary_mask=filtered_binary)
        adjust_fig = cv2.cvtColor(adjust_fig, cv2.COLOR_RGB2BGR)
        repair_fig=cv2.cvtColor(self.image_inpainting(img=adjust_fig,mask=mask), cv2.COLOR_RGB2BGR)
        #repair_fig = self.repair_img(color=center_colors[fig_index[0]], binary_img=mask)
        white_threshold = 20  # 可以根据实际需求调整这个值

        # 计算每个像素与白色[255, 255, 255]之间的欧氏距离
        # 首先将repair_fig转换为float类型以避免整数运算问题
        repair_fig_float = repair_fig.astype(np.float32)
        white = np.array([255, 255, 255], dtype=np.float32)

        # 计算每个像素到白色的欧氏距离
        # axis=-1表示沿着最后一个维度（颜色通道）计算
        distances = np.linalg.norm(repair_fig_float - white, axis=-1)

        # 创建掩码：距离小于阈值的像素被视为"接近白色"
        white_mask = distances < white_threshold
        # 将白色像素替换为背景颜色
        repair_fig[white_mask] = bg_color


        #result=self.repair_image2(img=repair_fig, colors=center_colors)
        result=self.filter_small_connected_components(repair_fig,connectivity=8,  background_value=255)

        ####后续要改
        if(len(fig_index))>1:
            result2 = self.fill_hole(result, connectivity=8, background_value=center_colors[fig_index[1]])
        else:
            result2 = self.fill_hole(result, connectivity=8, background_value=center_colors[fig_index[0]])
        print(f"原始连通域数量: {num_labels - 1}, 过滤后保留: {valid_labels}")

        return {
            "original": img,
            "binary": binary,
            "process_binary": filtered_binary,
            "bg_fig": clusters_images[bg_index],
            "adjust_fig": adjust_fig,
            "repair_fig":result2,
            "main_fig": main_fig,
            "mask": mask
        }

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
        self._save_results(results, output_folder, image_name_no_ext)

        # 可视化结果
        if visualize:
            self.visualize_results(results)

        return results

    def _save_results(self, results, output_folder, image_name_no_ext):
        """
        保存处理结果到指定文件夹
        """
        # 保存关键结果
        cv2.imwrite(os.path.join(output_folder, "adjust_fig.png"), results['adjust_fig'])
        cv2.imwrite(os.path.join(output_folder, "bg_fig.png"), cv2.cvtColor(results['bg_fig'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, "process_binary.png"), results['process_binary'])
        cv2.imwrite(os.path.join(output_folder, "repair_fig.png"),
                    cv2.cvtColor(results['repair_fig'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, f'{image_name_no_ext}_mask.png'),
                    cv2.cvtColor(results['mask'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_folder, f'{image_name_no_ext}.png'),
                    results['adjust_fig'])


# 向后兼容的类名（保持原有接口）
FSSfigDetector = ImagePreprocessor


# 示例用法
if __name__ == "__main__":
    # 创建检测器实例
    detector = ImagePreprocessor(max_k=6)

    # 指定图像路径
    image_path = "7.png"  # 图像路径
  ####图片7有问题 是二值化的问题 然后有两种颜色 没法判别  先不处理
    # 执行检测
    results = detector.detect(image_path)

    print("检测完成！")
