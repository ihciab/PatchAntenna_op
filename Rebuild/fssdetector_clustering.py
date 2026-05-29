import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


class FSSClusteringMixin:
    def adjust_image_by_mask(self, original_color_img, binary_mask, bg_color):
        """
        根据二值掩码调整彩色图像：
        保留黑色区域对应的原图内容，把白色区域设置为白底。
        """
        if original_color_img is None:
            raise ValueError("original_color_img cannot be None")
        if binary_mask is None:
            raise ValueError("binary_mask cannot be None")
        if len(binary_mask.shape) != 2:
            raise ValueError("binary_mask 应该是单通道图像")
        if original_color_img.shape[:2] != binary_mask.shape:
            raise ValueError("original_color_img and binary_mask must have the same shape")

        result_img = original_color_img.copy()
        white_mask = binary_mask > 127
        result_img[white_mask] = [255, 255, 255]
        return result_img

    def color_difference(self, color1, color2):
        """计算两个颜色在 LAB 空间中的距离。"""
        lab1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2LAB)[0][0]
        lab2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2LAB)[0][0]
        delta_e = np.sqrt(np.sum((lab1 - lab2) ** 2))
        return delta_e

    def auto_kmeans_color_quantization(self, img, output_path, max_k=6, min_color_diff=30):
        """
        自动选择较合适的 K 值，并对图像执行颜色聚类量化。

        返回值:
            quantized_bgr, cluster_centers_rgb
        """
        if img is None:
            raise FileNotFoundError("无法读取图片")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img_rgb.shape
        pixels = img_rgb.reshape((-1, 3))

        # 为了加速，只对部分像素做采样来估计最佳 K
        sample_size = min(15000, len(pixels))
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[indices].astype(np.float32)

        wcss = []
        silhouette_scores = []
        calinski_scores = []
        k_values = range(1, max_k + 1)

        for k in k_values:
            print(f"正在计算 k={k}...")
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(sample_pixels)

            wcss.append(kmeans.inertia_)
            if k > 1:
                silhouette_scores.append(silhouette_score(sample_pixels, labels))
                calinski_scores.append(calinski_harabasz_score(sample_pixels, labels))
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)

        # 综合多个指标自动挑选 K
        wcss_diff = np.diff(wcss)
        wcss_change_rate = np.abs(wcss_diff[1:] / wcss_diff[:-1]) if len(wcss_diff) >= 2 else []

        if len(wcss_change_rate) > 0:
            elbow_idx = np.argmin(wcss_change_rate) + 1
        else:
            elbow_idx = 1

        if len(silhouette_scores[1:]) > 0:
            max_silhouette_idx = np.argmax(silhouette_scores[1:]) + 1
        else:
            max_silhouette_idx = 1

        if len(calinski_scores[1:]) > 0:
            max_calinski_idx = np.argmax(calinski_scores[1:]) + 1
        else:
            max_calinski_idx = 1

        candidate_ks = [elbow_idx + 1, max_silhouette_idx + 1, max_calinski_idx + 1]
        candidate_ks = [k for k in candidate_ks if 2 <= k <= max_k]

        if not candidate_ks:
            best_k = 2
        else:
            best_k = min(set(candidate_ks), key=candidate_ks.count)

        # 如果中心颜色之间太近，再尝试把 K 增大一点
        temp_kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        temp_kmeans.fit(sample_pixels)
        temp_centers = temp_kmeans.cluster_centers_
        temp_centers_rgb = temp_centers.astype(np.uint8)

        has_small_diff = False
        for i in range(len(temp_centers_rgb)):
            for j in range(i + 1, len(temp_centers_rgb)):
                diff = self.color_difference(temp_centers_rgb[i], temp_centers_rgb[j])
                if diff < min_color_diff:
                    has_small_diff = True
                    break
            if has_small_diff:
                break

        if has_small_diff and best_k < max_k:
            best_k += 1
            print(f"检测到相似颜色，将 k 调整为: {best_k}")

        white = np.array([255, 255, 255], dtype=np.float32)
        border_w = max(2, min(height, width) // 80)
        border_pixels = np.concatenate([
            img_rgb[:border_w, :, :].reshape(-1, 3),
            img_rgb[-border_w:, :, :].reshape(-1, 3),
            img_rgb[:, :border_w, :].reshape(-1, 3),
            img_rgb[:, -border_w:, :].reshape(-1, 3),
        ], axis=0).astype(np.float32)
        border_white_ratio = float(
            np.mean(np.linalg.norm(border_pixels - white.reshape(1, 3), axis=1) <= 24.0)
        )
        center_white_dist = np.linalg.norm(
            temp_centers.astype(np.float32) - white.reshape(1, 3),
            axis=1,
        )
        temp_cluster_count = len(temp_centers)
        cluster_counts = np.bincount(temp_kmeans.labels_, minlength=temp_cluster_count).astype(np.float32)
        cluster_area_ratios = cluster_counts / max(1.0, float(len(temp_kmeans.labels_)))
        has_pure_white_center = bool(np.any(center_white_dist <= 18.0))
        has_merged_near_white_shape = bool(np.any(
            (center_white_dist > 18.0)
            & (center_white_dist <= 85.0)
            & (cluster_area_ratios >= 0.08)
        ))

        if (
            best_k < max_k
            and border_white_ratio >= 0.06
            and has_merged_near_white_shape
            and not has_pure_white_center
        ):
            best_k += 1
            print(f"检测到白边与浅色结构合并，将 k 调整为: {best_k}")

        print(f"自动选择的最佳 k 值: {best_k}")
        print(f"轮廓系数: {silhouette_scores[best_k - 1]:.4f} (k={best_k})")

        print(f"使用 k={best_k} 进行最终聚类...")
        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(pixels.astype(np.float32))
        centers_rgb = kmeans.cluster_centers_.astype(np.uint8)

        quantized_img_rgb = centers_rgb[labels].reshape((height, width, channels))
        print("创建颜色分布可视化...")
        print(f"使用的颜色: {centers_rgb.tolist()}")

        plt.figure(figsize=(10, 2))
        for i, color in enumerate(centers_rgb):
            plt.fill_between([i, i + 1], 0, 1, color=color / 255)
            plt.text(
                i + 0.5,
                0.5,
                f"C{i + 1}",
                ha="center",
                va="center",
                color="white" if np.mean(color) < 128 else "black",
                fontsize=12,
                fontweight="bold",
            )
        plt.xlim(0, len(centers_rgb))
        plt.axis("off")
        plt.title(f"颜色 palette (k={best_k})")
        plt.close()

        quantized_bgr = cv2.cvtColor(quantized_img_rgb, cv2.COLOR_RGB2BGR)
        return quantized_bgr, centers_rgb

    def _normalize_cluster_images(self, clusters_images, color_list, center_match_threshold=10.0):
        """
        把聚类结果统一转换成“每个聚类一张 RGB 图”的列表。
        这样后续分类和组合都只处理同一种输入格式。
        """
        colors = np.array(color_list, dtype=np.float32)

        if isinstance(clusters_images, np.ndarray):
            if clusters_images.ndim == 4 and clusters_images.shape[-1] == 3:
                return [clusters_images[i] for i in range(clusters_images.shape[0])]

            if clusters_images.ndim == 3 and clusters_images.shape[-1] == 3:
                quantized_rgb = cv2.cvtColor(clusters_images, cv2.COLOR_BGR2RGB)
                h, w = quantized_rgb.shape[:2]
                normalized_cluster_images = []
                for color in colors:
                    mask = np.linalg.norm(
                        quantized_rgb.astype(np.float32) - color.reshape(1, 1, 3),
                        axis=2,
                    ) <= center_match_threshold
                    cluster_vis = np.ones((h, w, 3), dtype=np.uint8) * 255
                    cluster_vis[mask] = color.astype(np.uint8)
                    normalized_cluster_images.append(cluster_vis)
                return normalized_cluster_images

            raise ValueError(f"Unsupported clusters_images ndarray shape: {clusters_images.shape}")

        if isinstance(clusters_images, (list, tuple)):
            return list(clusters_images)

        raise TypeError("clusters_images must be a list/tuple or numpy array")

    def filter_scattered_figures(self, fig_index, clusters_images, area_ratio_threshold=0.033, scattered_threshold=3):
        """
        过滤掉像素占比很低、看起来较分散的图形候选。
        """
        filtered_indices = []

        for idx in fig_index:
            fig_img = clusters_images[idx].copy()
            gray = fig_img if fig_img.ndim == 2 else cv2.cvtColor(fig_img, cv2.COLOR_RGB2GRAY)

            _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
            total_pixels = gray.shape[0] * gray.shape[1]
            non_white_pixels = cv2.countNonZero(binary)
            area_ratio = non_white_pixels / total_pixels

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            valid_components = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 100:
                    valid_components += 1

            if area_ratio >= area_ratio_threshold:
                filtered_indices.append(idx)
                print(f"保留索引 {idx}: 像素占比={area_ratio:.4f}, 有效连通域数={valid_components}")
            else:
                print(f"剔除索引 {idx}: 像素占比={area_ratio:.4f}, 有效连通域数={valid_components}")

        return filtered_indices

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
        结合颜色先验对聚类结果做角色划分：
        1. padding
        2. background
        3. figure
        4. outline
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

        # 第一遍：根据每张聚类图反推出对应的像素掩码
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

        # 纯白中心容易和占位白底冲突，这里做一次修正
        for i in near_white_indices:
            if np.linalg.norm(colors[i] - white) <= 2.5:
                cluster_masks[i] = ~non_white_union

        # 统计每个聚类区域的结构特征，供后续打分使用
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

        # 第一步：选 padding
        def padding_score(s):
            white_similarity = 1.0 - min(1.0, s["dist_white"] / 160.0)
            return 0.62 * white_similarity + 0.26 * s["border_ratio"] + 0.12 * (
                1.0 - min(1.0, s["area_ratio"] / 0.9)
            )

        padding_index = max(range(n_clusters), key=lambda i: padding_score(stats[i]))

        # 第二步：选背景
        bg_candidates = [i for i in range(n_clusters) if i != padding_index]
        if not bg_candidates:
            bg_candidates = [padding_index]

        def bg_score(s):
            bg_similarity = 1.0 - min(1.0, s["dist_bg"] / 180.0)
            black_penalty = min(1.0, s["dist_black"] / 150.0)
            white_penalty = min(1.0, s["dist_white"] / 150.0)
            return 0.54 * bg_similarity + 0.16 * s["border_ratio"] + 0.16 * black_penalty + 0.14 * white_penalty

        if np.linalg.norm(bg - white) < 18.0:
            bg_index = padding_index
        else:
            bg_index = max(bg_candidates, key=lambda i: bg_score(stats[i]))

        # 第三步：选 figure
        candidate_pool = [i for i in range(n_clusters) if i not in {padding_index, bg_index}]
        if not candidate_pool:
            candidate_pool = [i for i in range(n_clusters) if i != bg_index]
        if not candidate_pool:
            candidate_pool = [0]

        def fig_score(s):
            black_similarity = 1.0 - min(1.0, s["dist_black"] / 180.0)
            white_distance = min(1.0, s["dist_white"] / 180.0)
            bg_distance = min(1.0, s["dist_bg"] / 180.0)
            area_term = min(1.0, s["area_ratio"] / 0.12)
            fill_term = min(1.0, s["fill_ratio"])
            return (
                0.18 * black_similarity
                + 0.18 * white_distance
                + 0.22 * bg_distance
                + 0.24 * area_term
                + 0.18 * fill_term
            )

        fig_candidates = []
        for i in candidate_pool:
            s = stats[i]
            score_value = fig_score(s)
            if s["area_ratio"] >= min_fig_area_ratio or score_value >= 0.45:
                fig_candidates.append((i, score_value))

        if not fig_candidates:
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

        # 第四步：选 outline
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

        # 避免把近白色的区域选成 outline
        if stats[outline_index]["dist_white"] < white_color_threshold and len(outline_candidates) > 1:
            non_white_outline_cands = [i for i in outline_candidates if stats[i]["dist_white"] >= white_color_threshold]
            if non_white_outline_cands:
                outline_index = max(non_white_outline_cands, key=lambda i: outline_score(stats[i]))

        fig_index = [i for i in fig_index if i != outline_index and i != bg_index]
        if not fig_index:
            fallback = [i for i in range(n_clusters) if i not in (outline_index, bg_index)]
            fig_index = [max(fallback, key=lambda j: stats[j]["area_ratio"])] if fallback else [outline_index]

        # 保存调试图和统计信息
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
        """把多个聚类结果叠加成一张图。"""
        composed = clusters_images[fig_index[0]].copy()

        for idx in fig_index[1:]:
            img = clusters_images[idx]
            mask = ~(np.all(img == [255, 255, 255], axis=-1))
            composed[mask] = img[mask]

        return composed

    def image_inpainting(self, img, mask, method=cv2.INPAINT_TELEA):
        """对掩码指定区域执行图像修复。"""
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        restored = cv2.inpaint(img, mask, 3, method)
        return restored

    def filter_small_connected_components(
        self,
        rgb_image,
        connectivity=8,
        background_value=255,
        min_area_ratio=0.0015,
        keep_near_largest=True,
    ):
        """
        对 RGB 图像做连通域过滤，移除很小的前景区域。
        """
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("rgb_image must be an RGB image with shape (H, W, 3)")

        if rgb_image.dtype in (np.float32, np.float64):
            white_threshold = 1.0 - 1e-6
            is_white = (
                (rgb_image[:, :, 0] >= white_threshold)
                & (rgb_image[:, :, 1] >= white_threshold)
                & (rgb_image[:, :, 2] >= white_threshold)
            )
        elif rgb_image.dtype == np.uint8:
            is_white = (
                (rgb_image[:, :, 0] == 255)
                & (rgb_image[:, :, 1] == 255)
                & (rgb_image[:, :, 2] == 255)
            )
        else:
            raise TypeError("rgb_image dtype must be uint8, float32, or float64")

        binary = np.where(is_white, 0, 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)

        total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
        min_pixel_threshold = max(12, int(float(min_area_ratio) * total_pixels))

        filtered_image = rgb_image.copy()
        if keep_near_largest and num_labels > 1:
            foreground_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = int(np.argmax(foreground_areas) + 1)
            largest_mask = (labels == largest_label).astype(np.uint8) * 255
            near_largest = cv2.dilate(largest_mask, np.ones((9, 9), np.uint8), iterations=2) > 0
        else:
            near_largest = None

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if near_largest is not None and np.any((labels == label) & near_largest):
                continue
            if area < min_pixel_threshold:
                filtered_image[labels == label] = background_value

        return filtered_image

    def fill_hole(
        self,
        rgb_image,
        connectivity=8,
        threshold_method="otsu",
        background_value=[0, 0, 0],
        min_area_ratio=0.0015,
        max_hole_area_ratio=0.012,
    ):
        """
        对 RGB 图像做简单孔洞填充与小连通域过滤。
        """
        if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
            raise ValueError("rgb_image must be an RGB image with shape (H, W, 3)")

        if rgb_image.dtype in (np.float32, np.float64):
            image_u8 = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
        elif rgb_image.dtype == np.uint8:
            image_u8 = rgb_image
        else:
            raise TypeError("rgb_image dtype must be uint8, float32, or float64")

        if threshold_method != "otsu":
            raise ValueError("Only otsu threshold_method is supported")

        # The previous Otsu-on-gray path dropped light-colored metal regions.
        # At this stage the canvas is already normalized to white background,
        # so foreground is any pixel that is meaningfully different from white.
        is_white = (
            (image_u8[:, :, 0] >= 245)
            & (image_u8[:, :, 1] >= 245)
            & (image_u8[:, :, 2] >= 245)
        )
        foreground = np.where(is_white, 0, 255).astype(np.uint8)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )

        flood = foreground.copy()
        flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_and(cv2.bitwise_not(flood), cv2.bitwise_not(foreground))

        total_pixels = rgb_image.shape[0] * rgb_image.shape[1]
        max_hole_area = max(8, int(float(max_hole_area_ratio) * total_pixels))
        hole_labels_count, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=connectivity)
        small_holes = np.zeros_like(holes)
        for label in range(1, hole_labels_count):
            area = int(hole_stats[label, cv2.CC_STAT_AREA])
            if area <= max_hole_area:
                small_holes[hole_labels == label] = 255

        if np.count_nonzero(small_holes) > 0:
            foreground = cv2.bitwise_or(foreground, small_holes)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=connectivity)

        min_pixel_threshold = max(12, int(float(min_area_ratio) * total_pixels))

        filtered_image = np.ones_like(rgb_image, dtype=np.uint8) * 255
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) < min_pixel_threshold:
                continue
            filtered_image[labels == label, 0] = background_value[0]
            filtered_image[labels == label, 1] = background_value[1]
            filtered_image[labels == label, 2] = background_value[2]

        return filtered_image
