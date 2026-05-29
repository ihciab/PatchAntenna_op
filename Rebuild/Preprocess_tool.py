import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    
    
def process_special_text_boxes(text_info_list, img, coord_threshold=10):



        """
        处理文本框：筛选格式为(x)的文本，计算中心坐标，根据中心点数量分割图像
        支持2个或3个中心点的分割逻辑，输出统一为图像列表

        参数：
        text_info_list: 包含文本框坐标、内容、置信度的列表
        img: 原始图像
        coord_threshold: 坐标差阈值（判断是否在同一线的阈值，单位：像素）

        返回：
        list: 分割后的图像列表（空列表表示未分割或无法分割）
        """
        # 1. 筛选格式为“(x)”的文本（x为单个数字或字母）
        pattern = re.compile(r'\([a-zA-Z0-9]\)')  # 匹配子串，如 "abc(1)def" 中的 (1)
        matched_items = []
        expand_pixel=2
        for idx, info in enumerate(text_info_list):
            text = info['text']
            if pattern.match(text):
                # 计算文本框中心坐标（四个顶点的x、y平均值）
                box = info['box']  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_center = sum(p[0] for p in box) / 4.0
                y_center = sum(p[1] for p in box) / 4.0
                matched_items.append({
                    'index': idx,
                    'center': (x_center, y_center),  # (x, y)
                    'box': box
                })

        # 2. 处理匹配结果（统一返回图像列表）
        imgs = []
        img_h, img_w = img.shape[:2]  # 图像高度（y范围）和宽度（x范围）
        if img is None:
            print("无法读取图像")
            return imgs  # 返回空列表

        # 情况1：未找到匹配项
        if not matched_items:
            print("无")
            return imgs  # 返回空列表

        # 情况2：找到2个中心点（原有逻辑，输出2张图）
        elif len(matched_items) == 2:
            (x1, y1), (x2, y2) = [item['center'] for item in matched_items]

            # 子情况2.1：同一竖直线（竖向排列，每个点对应一个子图）
            if abs(x1 - x2) < coord_threshold and abs(y1 - y2) > coord_threshold:
                print("2个点：同一竖直线，生成2个子图（上向截取）")
                for (x_center, y_center) in [(x1, y1), (x2, y2)]:
                    # 子图宽度：0.5倍图像宽（左右各0.25倍）
                    x_half = 0.5 * img_w
                    x_min = max(0, int(x_center - x_half- expand_pixel))  # 左边界（不超出图像）
                    x_max = min(img_w, int(x_center + x_half+ expand_pixel))  # 右边界

                    # 子图高度：0.5倍图像高（向上截取，以中心点为下限）
                    y_up = 0.5 * img_h
                    y_min = max(0, int(y_center - y_up- expand_pixel))  # 上边界（向上延伸0.5倍高）
                    y_max = min(img_h, int(y_center+ expand_pixel))  # 下边界（不超过中心点）

                    # 截取子图
                    subimg = img[y_min:y_max, x_min:x_max]
                    imgs.append((subimg, (x_min, x_max, y_min, y_max)))
            elif abs(y1 - y2) < coord_threshold and abs(x1 - x2) > coord_threshold:
                # 同一水平线 → 竖直分割（取x中点）
                print("2个点：同一水平线，生成2个子图（上向截取）")
                for (x_center, y_center) in [(x1, y1), (x2, y2)]:
                    # 子图宽度：0.5倍图像宽（左右各0.25倍）
                    x_half = 0.25 * img_w
                    x_min = max(0, int(x_center - x_half- expand_pixel))
                    x_max = min(img_w, int(x_center + x_half+ expand_pixel))

                    # 子图高度：1倍图像高（向上截取，以中心点为下限）
                    y_up = 1.0 * img_h
                    y_min = max(0, int(y_center - y_up- expand_pixel))  # 上边界（向上延伸1倍高）
                    y_max = min(img_h, int(y_center+ expand_pixel))  # 下边界（不超过中心点）

                    # 截取子图
                    subimg = img[y_min:y_max, x_min:x_max]
                    imgs.append((subimg, (x_min, x_max, y_min, y_max)))
            else:
                print("2个点：无法按要求分割")

            # 情况3：找到3个中心点
        elif len(matched_items) == 3:
            centers = [item['center'] for item in matched_items]  # [(x0,y0), (x1,y1), (x2,y2)]
            y_coords = [y for (x, y) in centers]

            # 子情况3.1：三个点在同一水平线
            if (abs(y_coords[0] - y_coords[1]) < coord_threshold and
                    abs(y_coords[1] - y_coords[2]) < coord_threshold and
                    abs(y_coords[0] - y_coords[2]) < coord_threshold):
                print("3个点：同一水平线，生成3个子图（上向截取）")
                for (x_center, y_center) in centers:
                    x_half = 0.16 * img_w
                    x_min = max(0, int(x_center - x_half))
                    x_max = min(img_w, int(x_center + x_half))
                    y_up = 1.0 * img_h
                    y_min = max(0, int(y_center - y_up))
                    y_max = min(img_h, int(y_center))
                    subimg = img[y_min:y_max, x_min:x_max]
                    imgs.append((subimg, (x_min, x_max, y_min, y_max)))

            # 子情况3.2/3.3：两点水平线+单独点（细分横坐标位置）
            else:
                # 寻找同一水平线的两点对
                pair = None
                single = None
                pair_indices = None
                for i in range(3):
                    for j in range(i + 1, 3):
                        y_i, y_j = centers[i][1], centers[j][1]
                        if abs(y_i - y_j) < coord_threshold:
                            pair = (centers[i], centers[j])  # 同一水平线的两点
                            pair_indices = (i, j)
                            single_idx = [0, 1, 2]
                            single_idx.remove(i)
                            single_idx.remove(j)
                            single = centers[single_idx[0]]  # 单独点
                            break
                    if pair:
                        break

                if not pair:
                    print("3个点：无符合的水平线两点对，不生成子图")
                    return imgs

                # 提取两点对的坐标并排序（x从小到大）
                (x_p1, y_p), (x_p2, y_p) = pair  # 两点y坐标相同（同一水平线）
                x_p_min = min(x_p1, x_p2)
                x_p_max = max(x_p1, x_p2)
                x_single, y_single = single  # 单独点坐标

                # 判断单独点的横坐标是否在两点对之间（丁字分布=3.2，外部=3.3）
                if x_p_min < x_single < x_p_max:
                    # 情况3.2：单独点在两点中间（丁字分布）
                    print("3个点：两点水平线+单独点在中间（丁字分布），生成3个子图（上向截取）")
                    all_centers = [pair[0], pair[1], single]
                    for (x_center, y_center) in all_centers:
                        x_half = 0.25 * img_w
                        x_min = max(0, int(x_center - x_half))
                        x_max = min(img_w, int(x_center + x_half))
                        y_up = 0.5 * img_h
                        y_min = max(0, int(y_center - y_up))
                        y_max = min(img_h, int(y_center))
                        subimg = img[y_min:y_max, x_min:x_max]
                        imgs.append((subimg, (x_min, x_max, y_min, y_max)))

                else:
                    # 情况3.3：单独点在两点外（不规范三子图）→ 计算等间距第三点
                    print("3个点：两点水平线+单独点在外部（不规范三子图），生成3个子图（上向截取）")
                    # 计算两点对的间距
                    pair_spacing = abs(x_p2 - x_p1)
                    # 确定单独点位置，计算等间距第三点坐标（纵坐标与两点对一致）
                    if x_single < x_p_min:
                        # 单独点在左侧 → 第三点在两点对右侧（x_p_max + pair_spacing, y_p）
                        third_x = x_p_min - pair_spacing-7
                        third_center = (third_x, y_p)

                    else:
                        # 单独点在右侧 → 第三点在两点对左侧（x_p_min - pair_spacing, y_p）
                        third_x = x_p_max + pair_spacing
                        third_center = (third_x, y_p)
                    # 生成三个点：单独点 + 两点对 + 计算出的第三点 → 按同一水平线逻辑截取
                    final_centers = [third_center, (x_p_min, y_p), (x_p_max, y_p)][:3]  # 取前3个（确保3个点）
                    # 按x坐标排序，确保从左到右
                    final_centers_sorted = sorted(final_centers, key=lambda c: c[0])
                    for (x_center, y_center) in final_centers_sorted:
                        x_half = 0.16 * img_w
                        x_min = max(0, int(x_center - x_half))
                        x_max = min(img_w, int(x_center + x_half))
                        y_up = 1.0 * img_h
                        y_min = max(0, int(y_center - y_up))
                        y_max = min(img_h, int(y_center))
                        subimg = img[y_min:y_max, x_min:x_max]
                        imgs.append((subimg, (x_min, x_max, y_min, y_max)))

        # 情况4：找到4个中心点（新增逻辑）
        elif len(matched_items) == 4:
            centers = [item['center'] for item in matched_items]  # [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
            y_coords = [y for (x, y) in centers]

            # 判断截取方向：点在图像上半部分 -> 向下截取；点在图像下半部分 -> 向上截取
            avg_y = sum(y_coords) / 4.0
            is_upper_half = avg_y < (img_h / 2.0)

            # 子情况4.1：矩形排列（两两一行，共两行）
            # 判断逻辑：y坐标可分成两组，每组2个点（同一行），两组y差>阈值
            y_sorted = sorted(y_coords)
            # 计算两组的y差（前两个为一组，后两个为一组）
            group1_y_diff = y_sorted[1] - y_sorted[0]
            group2_y_diff = y_sorted[3] - y_sorted[2]
            inter_group_diff = y_sorted[2] - y_sorted[1]
            if (group1_y_diff < coord_threshold and
                    group2_y_diff < coord_threshold and
                    inter_group_diff > coord_threshold):
                print(f"4个点：矩形排列（两两一行），生成4个子图（{'下' if is_upper_half else '上'}向截取）")
                for (x_center, y_center) in centers:
                    # 宽0.5倍图像宽（左右0.25）
                    x_half = 0.25 * img_w
                    x_min = max(0, int(x_center - x_half))
                    x_max = min(img_w, int(x_center + x_half))

                    y_range = 0.5 * img_h
                    if is_upper_half:
                        # 向下截取
                        y_min = max(0, int(y_center))
                        y_max = min(img_h, int(y_center + y_range))
                    else:
                        # 向上截取
                        y_min = max(0, int(y_center - y_range))
                        y_max = min(img_h, int(y_center))

                    subimg = img[y_min:y_max, x_min:x_max]
                    imgs.append((subimg, (x_min, x_max, y_min, y_max)))

            # 子情况4.2：横向排列（四个点在同一水平线）
            elif (abs(y_coords[0] - y_coords[1]) < coord_threshold and
                  abs(y_coords[1] - y_coords[2]) < coord_threshold and
                  abs(y_coords[2] - y_coords[3]) < coord_threshold and
                  abs(y_coords[0] - y_coords[3]) < coord_threshold):
                print(f"4个点：横向排列（同一水平线），生成4个子图（{'下' if is_upper_half else '上'}向截取）")
                for (x_center, y_center) in centers:
                    # 宽0.25倍图像宽（左右0.125）
                    x_half = 0.125 * img_w
                    x_min = max(0, int(x_center - x_half))
                    x_max = min(img_w, int(x_center + x_half))

                    y_range = 0.5 * img_h
                    if is_upper_half:
                        # 向下截取
                        y_min = max(0, int(y_center))
                        y_max = min(img_h, int(y_center + y_range))
                    else:
                        # 向上截取
                        y_min = max(0, int(y_center - y_range))
                        y_max = min(img_h, int(y_center))

                    subimg = img[y_min:y_max, x_min:x_max]
                    imgs.append((subimg, (x_min, x_max, y_min, y_max)))

            # 其他排列（不处理）
            else:
                print("4个点：非矩形/横向排列，不生成子图")
                return imgs
        # 其他数量的点（不处理）


    # 其他数量的点（不处理）
        else:
            print(f"找到{len(matched_items)}个点，仅支持2或3个点的分割")
            return imgs

        return imgs  # 统一返回图像列表

def detect_background_color(image,
                                white_threshold=240, white_ratio_threshold=0.3,
                                edge_crop_ratio=0.01, dark_threshold=30,
                                edge_band_width=5, dominant_ratio_threshold=0.2,
                                black_threshold=15, black_ratio_threshold=0.25):
        """
        优化后的背景颜色检测，修复了无法识别黑色背景的问题。

        改动要点：
        - 增加了对“接近黑色”的专门检测（black_threshold, black_ratio_threshold）。
        - 修复了原来把低亮度颜色直接替换为白色的错误逻辑。
        - 增强了当区域像素全部被过滤掉时的回退策略（优先使用边缘颜色判断，最后使用区域主色）。
        - 保持对白色背景的检测逻辑（white_threshold / white_ratio_threshold）。
        返回 (B, G, R) 三元组，值为 int。
        """

        # 输入验证
        if image is None or len(image.shape) != 3:
            return (255, 255, 255)

        height, width = image.shape[:2]

        # 将图像分为四个区域（左上、右上、左下、右下）
        regions = [
            image[0:int(height / 2), 0:int(width / 2)],
            image[0:int(height / 2), int(width / 2):width],
            image[int(height / 2):height, 0:int(width / 2)],
            image[int(height / 2):height, int(width / 2):width]
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

            # 合并所有边缘像素（用于回退）
            edge_pixels = np.concatenate([
                top_edge.reshape(-1, 3),
                bottom_edge.reshape(-1, 3),
                left_edge.reshape(-1, 3),
                right_edge.reshape(-1, 3)
            ]) if (top_edge.size + bottom_edge.size + left_edge.size + right_edge.size) > 0 else np.empty((0, 3),
                                                                                                          dtype=np.uint8)

            # 区域所有像素
            all_pixels = cropped.reshape(-1, 3)
            if len(all_pixels) == 0:
                continue

            # 检测白色背景（占比）
            white_pixel_mask = np.all(all_pixels >= white_threshold, axis=1)
            white_ratio = np.sum(white_pixel_mask) / len(all_pixels)
            if white_ratio >= white_ratio_threshold:
                region_color = (255, 255, 255)
                region_backgrounds.append(region_color)
                continue

            # 检测接近黑色背景（占比）
            black_pixel_mask = np.all(all_pixels <= black_threshold, axis=1)
            black_ratio = np.sum(black_pixel_mask) / len(all_pixels)
            if black_ratio >= black_ratio_threshold:
                region_color = (0, 0, 0)
                region_backgrounds.append(region_color)
                continue

            # 非白非黑：过滤白色像素后统计主色
            valid_all_pixels = all_pixels[~white_pixel_mask]
            # 过滤边缘处的白色像素用于边缘统计
            valid_edge_pixels = edge_pixels[
                ~np.all(edge_pixels >= white_threshold, axis=1)] if edge_pixels.size else np.empty((0, 3),
                                                                                                   dtype=np.uint8)

            # 回退：如果过滤后没有有效像素，优先使用边缘信息判断深色/黑色/白色，再回退到整体
            if len(valid_all_pixels) == 0:
                if valid_edge_pixels.size > 0:
                    edge_avg = np.mean(valid_edge_pixels)
                    if edge_avg <= dark_threshold:
                        region_color = (0, 0, 0)
                    elif edge_avg >= white_threshold:
                        region_color = (255, 255, 255)
                    else:
                        # 使用边缘主色
                        edge_colors, edge_counts = np.unique(valid_edge_pixels, axis=0, return_counts=True)
                        region_color = tuple(int(c) for c in edge_colors[np.argmax(edge_counts)])
                else:
                    # 最后回退为整区的主色（即便包含了白色）
                    all_colors, all_counts = np.unique(all_pixels, axis=0, return_counts=True)
                    region_color = tuple(int(c) for c in all_colors[np.argmax(all_counts)])
                region_backgrounds.append(region_color)
                continue

            # 统计区域内主要颜色（排除白色）
            unique_colors, counts = np.unique(valid_all_pixels, axis=0, return_counts=True)
            max_count_idx = np.argmax(counts)
            dominant_color = unique_colors[max_count_idx]
            dominant_ratio = counts[max_count_idx] / len(valid_all_pixels)

            # 当主色占比较低时，尝试使用边缘主色作为背景（如果可用）
            if dominant_ratio < dominant_ratio_threshold and len(valid_edge_pixels) > 0:
                edge_colors, edge_counts = np.unique(valid_edge_pixels, axis=0, return_counts=True)
                if len(edge_colors) > 0:
                    dominant_color = edge_colors[np.argmax(edge_counts)]

            # 如果 dominant_color 非常暗（平均亮度小于等于 dark_threshold），将其作为黑色背景
            color_avg = np.mean(dominant_color)
            if color_avg <= dark_threshold:
                region_color = (0, 0, 0)
            else:
                region_color = tuple(int(c) for c in dominant_color)

            region_backgrounds.append(region_color)

        if not region_backgrounds:
            return (255, 255, 255)

        # 取四个区域中出现次数最多的颜色（投票）
        color_counts = Counter(region_backgrounds)
        most_common_color = color_counts.most_common(1)[0][0]

        return most_common_color
