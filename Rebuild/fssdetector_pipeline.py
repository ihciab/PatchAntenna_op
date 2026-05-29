import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import Preprocess_tool
except ImportError:
    import Preprocess_tool


class FSSPipelineMixin:
    def process_edges(self, image_path, output_path):
        """
        杈圭紭澶勭悊涓绘祦绋嬶細璇诲彇杈撳叆鍥惧儚锛屽厛鎵ц瀛愬浘鍒囧垎鍜岄娓呯悊锛?        鍐嶈繘鍏ラ鑹茶仛绫汇€佷富浣撶瓫閫夈€佽繛閫氬煙杩囨护涓庝慨琛ユ祦绋嬨€?        杩斿洖鍊兼槸涓€涓垪琛紝鍒楄〃涓殑姣忎竴椤瑰搴斾竴寮犲瓙鍥剧殑瀹屾暣涓棿缁撴灉銆?        """
        img = cv2.imread(image_path)
        results = []
        if img is None:
            raise FileNotFoundError(f"鍥惧儚鏂囦欢鏈壘鍒? {image_path}")

        imgs = self.process_image(img)

        if isinstance(imgs, list):
            img_list = imgs
        elif isinstance(imgs, np.ndarray):
            img_list = [imgs] if imgs.ndim == 3 else [imgs[i] for i in range(imgs.shape[0])]
        else:
            img_list = [imgs]

        for item in img_list:
            # 新版 process_image() 可能返回 dict，携带子图坐标、预清理 mask 和调试信息；
            # 这里仍兼容旧版直接返回 ndarray 的路径，避免旧调用方式失效。
            if isinstance(item, dict):
                input_img = item.get("image")
                subimg_coords = item.get("coords")
                pre_repair_mask = item.get("pre_repair_mask")
                pre_removed_mask = item.get("pre_removed_mask")
                pre_detections = item.get("pre_detections", [])
                pre_detection_vis = item.get("pre_detection_vis")
            else:
                input_img = item
                subimg_coords = None
                pre_repair_mask = None
                pre_removed_mask = None
                pre_detections = []
                pre_detection_vis = None

            if input_img is None:
                continue

            # 统一到固定分辨率；图像用线性插值，mask 用最近邻避免二值边界被插花。
            if len(input_img.shape) == 2:
                resized_input = cv2.resize(input_img, (845, 845), interpolation=cv2.INTER_LINEAR)
                resized_input = cv2.cvtColor(resized_input, cv2.COLOR_GRAY2BGR)
            else:
                resized_input = cv2.resize(input_img, (845, 845), interpolation=cv2.INTER_LINEAR)

            if pre_repair_mask is None:
                pre_repair_mask = np.zeros(resized_input.shape[:2], dtype=np.uint8)
            else:
                pre_repair_mask = cv2.resize(
                    pre_repair_mask,
                    (845, 845),
                    interpolation=cv2.INTER_NEAREST,
                )

            if pre_removed_mask is None:
                pre_removed_mask = np.zeros(resized_input.shape[:2], dtype=np.uint8)
            else:
                pre_removed_mask = cv2.resize(
                    pre_removed_mask,
                    (845, 845),
                    interpolation=cv2.INTER_NEAREST,
                )

            original_input_rgb = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)
            global_bg_color_rgb = Preprocess_tool.detect_background_color(original_input_rgb)
            global_bg_color_bgr = np.asarray(global_bg_color_rgb[::-1], dtype=np.uint8)

            # 文字已在 process_image() 中通过 YOLO + OCR 预处理；这里继续处理 arrow / line。
            yolo_clean_img, yolo_mask, yolo_detections, yolo_removed_mask = self.remove_detected_elements_with_yolo(
                img=resized_input,
                target_class_names=getattr(self, "yolo_element_labels", ("arrow", "line")),
                global_background_color=global_bg_color_bgr,
                expand_pixels=getattr(self, "yolo_element_expand_pixels", 3),
                border_width=getattr(self, "yolo_element_border_width", 2),
                min_component_area=getattr(self, "yolo_element_min_component_area", 8),
            )

            if len(yolo_detections) > 0:
                yolo_count_by_class = {}
                for detection in yolo_detections:
                    class_name = detection["class_name"]
                    yolo_count_by_class[class_name] = yolo_count_by_class.get(class_name, 0) + 1
                print(f"YOLO 棰勬竻鐞嗘娴嬬粨鏋? {yolo_count_by_class}")

            yolo_detection_vis = self.draw_yolo_detections(
                resized_input,
                yolo_detections,
                mask=yolo_removed_mask,
            )

            working_img = yolo_clean_img
            bg_color = global_bg_color_rgb
            print(f"妫€娴嬪埌鐨勮儗鏅鑹? {bg_color}")

            quantized_clusters_image, clusters_colors = self.auto_kmeans_color_quantization(working_img, output_path)
            clusters_images = self._normalize_cluster_images(
                quantized_clusters_image,
                color_list=clusters_colors,
            )

            outline_index, padding_index, bg_index, fig_index = self.classify_colors_with_priors(
                color_list=clusters_colors,
                bg_color=bg_color,
                clusters_images=clusters_images,
                output_dir=output_path,
            )

            print(outline_index, padding_index, bg_index, fig_index)

            filt_fig_index = self.filter_scattered_figures(fig_index, clusters_images)
            all_index = copy.copy(filt_fig_index)
            all_index.append(outline_index)
            if padding_index != bg_index:
                all_index.append(padding_index)

            try:
                main_fig = self.compose_figures(clusters_images, filt_fig_index)
            except Exception as exc:
                print(f"璇ュ浘鍍忔棤娉曞鐞嗭紝宸茶烦杩?{image_path}, 閿欒淇℃伅: {exc}")
                continue

            origin_img = self.compose_figures(clusters_images, all_index)

            diff = cv2.absdiff(origin_img, main_fig)
            if len(resized_input.shape) == 3:
                max_diff = np.max(diff, axis=2)
            else:
                max_diff = diff

            # 鏈€缁堜慨琛?mask 鐢变笁閮ㄥ垎骞惰捣鏉ワ細
            # 1. 鑱氱被绛涢€夊墠鍚庡樊寮傚緱鍒扮殑 diff_mask
            # 2. arrow / line 鍘婚櫎闃舵淇濈暀涓嬫潵鐨勪慨琛ュ尯鍩?yolo_mask
            # 3. 鏂囧瓧闃舵淇濈暀涓嬫潵鐨勪慨琛ュ尯鍩?pre_repair_mask
            _, diff_mask = cv2.threshold(max_diff, 50, 255, cv2.THRESH_BINARY)
            estimated_structure_region = self._estimate_structure_region(
                resized_input,
                bg_color_rgb=bg_color,
            )
            structure_mask = self.build_structure_mask(main_fig)
            if np.count_nonzero(structure_mask) == 0:
                structure_mask = estimated_structure_region
            structure_region = cv2.bitwise_or(structure_mask, estimated_structure_region)
            dilated_structure_mask = cv2.dilate(
                structure_region,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                iterations=1,
            )
            diff_annotation_mask = self._filter_diff_annotation_mask(diff_mask, structure_region)
            annotation_mask = cv2.bitwise_or(yolo_mask, pre_repair_mask)
            repair_candidate_mask = annotation_mask
            repair_mask = self.gate_repair_mask_by_structure(
                repair_candidate_mask,
                structure_region,
                dilation_pixels=getattr(self, "repair_structure_dilation_pixels", 10),
            )
            background_only_regions = cv2.subtract(
                cv2.threshold(repair_candidate_mask, 1, 255, cv2.THRESH_BINARY)[1],
                repair_mask,
            )
            combined_removed_mask = cv2.bitwise_or(yolo_removed_mask, pre_removed_mask)

            gray = cv2.cvtColor(main_fig, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_inv = cv2.bitwise_not(binary)
            process_binary = cv2.bitwise_not(binary_inv)

            black_pixels = np.count_nonzero(process_binary == 0)
            min_area = 0.008 * black_pixels
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_inv,
                connectivity=8,
                ltype=cv2.CV_32S,
            )
            filtered_binary = np.zeros_like(process_binary)
            valid_labels = 0

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
                valid_labels += 1
                filtered_binary[labels == i] = 255

            filtered_binary = cv2.bitwise_not(filtered_binary)

            if self.show_debug_windows:
                cv2.imshow("11", filtered_binary)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            adjust_fig = self.adjust_image_by_mask(
                original_color_img=main_fig,
                binary_mask=filtered_binary,
                bg_color=bg_color,
            )
            adjust_fig = cv2.cvtColor(adjust_fig, cv2.COLOR_RGB2BGR)
            pre_inpaint_fig = self.fill_background_only_regions(
                adjust_fig,
                background_only_regions=background_only_regions,
                bg_color_rgb=None,
            )
            repair_fig = cv2.cvtColor(self.image_inpainting(img=pre_inpaint_fig, mask=repair_mask), cv2.COLOR_RGB2BGR)
            repair_fig = self._fill_annotation_regions_with_structure_color(
                repair_fig,
                repair_mask=repair_mask,
                structure_color=clusters_colors[filt_fig_index[0] if len(filt_fig_index) > 0 else fig_index[0]],
            )

            result = self.filter_small_connected_components(
                repair_fig,
                connectivity=8,
                background_value=255,
            )

            if len(fig_index) > 1:
                result2 = self.fill_hole(
                    result,
                    connectivity=8,
                    background_value=clusters_colors[fig_index[1]],
                )
            else:
                result2 = self.fill_hole(
                    result,
                    connectivity=8,
                    background_value=clusters_colors[fig_index[0]],
                )

            print(f"鍘熷杩為€氬煙鏁伴噺: {num_labels - 1}, 杩囨护鍚庝繚鐣? {valid_labels}")

            results.append({
                "original": resized_input,
                "subimg_coords": subimg_coords,
                "text_detection_vis": pre_detection_vis,
                "text_mask": pre_repair_mask,
                "text_removed_mask": pre_removed_mask,
                "text_detections": pre_detections,
                "yolo_clean_img": working_img,
                "yolo_detection_vis": yolo_detection_vis,
                "binary": binary,
                "process_binary": filtered_binary,
                "bg_fig": clusters_images[bg_index],
                "adjust_fig": adjust_fig,
                "repair_fig": result2,
                "main_fig": main_fig,
                "mask": repair_mask,
                "repair_needed_mask": repair_mask,
                "gated_repair_mask": repair_mask,
                "diff_mask": diff_mask,
                "diff_annotation_mask": diff_annotation_mask,
                "annotation_mask": annotation_mask,
                "refined_component_mask": annotation_mask,
                "structure_mask": structure_mask,
                "structure_region": structure_region,
                "dilated_structure_mask": dilated_structure_mask,
                "background_only_regions": background_only_regions,
                "yolo_mask": yolo_mask,
                "yolo_removed_mask": yolo_removed_mask,
                "removed_mask": combined_removed_mask,
                "yolo_detections": yolo_detections,
            })

        return results

    def _estimate_structure_region(self, img_bgr, bg_color_rgb=None):
        if img_bgr is None:
            raise ValueError("img_bgr cannot be None")
        if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
            raise ValueError("img_bgr must be a 3-channel image")

        h, w = img_bgr.shape[:2]
        if bg_color_rgb is None:
            border = np.concatenate([
                img_bgr[0, :, :],
                img_bgr[-1, :, :],
                img_bgr[:, 0, :],
                img_bgr[:, -1, :],
            ], axis=0)
            bg_bgr = np.median(border, axis=0).astype(np.uint8)
        else:
            bg_rgb = np.asarray(bg_color_rgb, dtype=np.uint8).reshape(3)
            bg_bgr = bg_rgb[::-1]

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(np.uint8([[bg_bgr]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        dist_bg = np.linalg.norm(lab - bg_lab.reshape(1, 1, 3), axis=2)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        primary_region = np.where(
            ((saturation >= 55) & (value >= 80))
            | ((dist_bg >= 45.0) & (value >= 110)),
            255,
            0,
        ).astype(np.uint8)
        fallback_region = np.where(dist_bg >= 32.0, 255, 0).astype(np.uint8)

        if np.count_nonzero(primary_region) >= int(0.003 * h * w):
            raw_region = primary_region
        else:
            raw_region = fallback_region

        raw_region = cv2.morphologyEx(raw_region, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        raw_region = cv2.morphologyEx(raw_region, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_region, connectivity=8)
        total_area = float(max(1, h * w))
        min_area = max(80, int(total_area * 0.0008))
        structure_region = np.zeros((h, w), dtype=np.uint8)

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = float(max(1, bw * bh))
            fill_ratio = area / bbox_area
            area_ratio = area / total_area
            elongation = max(bw, bh) / float(max(1, min(bw, bh)))

            if elongation >= 8.0 and fill_ratio <= 0.08 and area_ratio < 0.02:
                continue
            if fill_ratio < 0.035 and area_ratio < 0.01:
                continue

            structure_region[labels == label] = 255

        if np.count_nonzero(structure_region) == 0:
            structure_region = raw_region

        structure_region = cv2.dilate(structure_region, np.ones((3, 3), np.uint8), iterations=1)
        return cv2.threshold(structure_region, 1, 255, cv2.THRESH_BINARY)[1]

    def _filter_diff_annotation_mask(self, diff_mask, structure_region):
        if diff_mask is None or structure_region is None:
            raise ValueError("diff_mask and structure_region cannot be None")

        constrained = cv2.bitwise_and(diff_mask, structure_region)
        if np.count_nonzero(constrained) == 0:
            return constrained

        eroded_structure = cv2.erode(structure_region, np.ones((5, 5), np.uint8), iterations=1)
        structure_boundary = cv2.subtract(structure_region, eroded_structure)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(constrained, connectivity=8)

        h, w = constrained.shape[:2]
        total_area = float(max(1, h * w))
        filtered = np.zeros_like(constrained)

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 3:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            component = labels == label

            bbox_area = float(max(1, bw * bh))
            fill_ratio = area / bbox_area
            area_ratio = area / total_area
            elongation = max(bw, bh) / float(max(1, min(bw, bh)))
            boundary_overlap = float(np.count_nonzero(component & (structure_boundary > 0))) / float(max(1, area))

            if boundary_overlap >= 0.52 and area >= 12:
                continue
            if area_ratio >= 0.025:
                continue
            if fill_ratio >= 0.80 and area >= 60:
                continue

            if area <= 180 or elongation >= 1.8 or fill_ratio <= 0.55:
                filtered[component] = 255
                continue

            _ = x, y

        filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(filtered)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox_area = bw * bh
            if bbox_area <= int(0.018 * h * w):
                cv2.drawContours(solid, [contour], -1, 255, thickness=-1)
            else:
                cv2.drawContours(solid, [contour], -1, 255, thickness=1)
            _ = x, y

        solid = cv2.dilate(solid, np.ones((2, 2), np.uint8), iterations=1)
        return cv2.bitwise_and(solid, structure_region)

    def build_structure_mask(self, main_fig):
        """
        Build a subject/structure mask from the clustered main figure.
        White canvas is background; non-white connected regions are candidate structure.
        """
        if main_fig is None:
            raise ValueError("main_fig cannot be None")
        if len(main_fig.shape) != 3 or main_fig.shape[2] != 3:
            raise ValueError("main_fig must be a 3-channel image")

        image = main_fig.astype(np.uint8)
        non_white = np.where(
            (image[:, :, 0] < 245) | (image[:, :, 1] < 245) | (image[:, :, 2] < 245),
            255,
            0,
        ).astype(np.uint8)
        non_white = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        non_white = cv2.morphologyEx(non_white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        h, w = non_white.shape[:2]
        total_area = float(max(1, h * w))
        min_area = max(24, int(total_area * 0.0006))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(non_white, connectivity=8)
        structure_mask = np.zeros_like(non_white)

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            bbox_area = float(max(1, bw * bh))
            fill_ratio = area / bbox_area
            elongation = max(bw, bh) / float(max(1, min(bw, bh)))

            # Reject isolated hairline artifacts; keep larger closed or filled regions.
            if elongation >= 12.0 and fill_ratio <= 0.08 and area < int(total_area * 0.02):
                continue
            structure_mask[labels == label] = 255

        return cv2.threshold(structure_mask, 1, 255, cv2.THRESH_BINARY)[1]

    def gate_repair_mask_by_structure(self, candidate_mask, structure_mask, dilation_pixels=6):
        """
        Keep only repair candidates that touch or are near the physical structure.
        """
        if candidate_mask is None or structure_mask is None:
            raise ValueError("candidate_mask and structure_mask cannot be None")

        candidate = cv2.threshold(candidate_mask, 1, 255, cv2.THRESH_BINARY)[1]
        structure = cv2.threshold(structure_mask, 1, 255, cv2.THRESH_BINARY)[1]
        if np.count_nonzero(candidate) == 0 or np.count_nonzero(structure) == 0:
            return np.zeros_like(candidate)

        k = max(1, int(dilation_pixels) * 2 + 1)
        dilated_structure = cv2.dilate(
            structure,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
        gated = cv2.bitwise_and(candidate, dilated_structure)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        filtered = np.zeros_like(candidate)
        for label in range(1, num_labels):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 2:
                continue
            near_pixels = int(np.count_nonzero(component & (dilated_structure > 0)))
            overlap_ratio = near_pixels / float(max(1, area))
            if overlap_ratio >= 0.25 or near_pixels >= min(area, 8):
                filtered[component] = 255

        return cv2.bitwise_or(gated, filtered)

    def is_background_only_region(self, region_mask, structure_mask):
        """
        True when a mask has no meaningful contact with the structure mask.
        """
        if region_mask is None or structure_mask is None:
            return True
        region = cv2.threshold(region_mask, 1, 255, cv2.THRESH_BINARY)[1]
        structure = cv2.threshold(structure_mask, 1, 255, cv2.THRESH_BINARY)[1]
        if np.count_nonzero(region) == 0:
            return True
        return np.count_nonzero(cv2.bitwise_and(region, structure)) == 0

    def fill_background_only_regions(self, image_bgr, background_only_regions, bg_color_rgb=None):
        """
        Fill removed background-only annotation pixels with the background color instead of inpainting.
        """
        if image_bgr is None:
            raise ValueError("image_bgr cannot be None")
        if background_only_regions is None:
            return image_bgr

        mask = cv2.threshold(background_only_regions, 1, 255, cv2.THRESH_BINARY)[1]
        if np.count_nonzero(mask) == 0:
            return image_bgr

        filled = image_bgr.copy()
        if bg_color_rgb is None:
            bg_bgr = np.array([255, 255, 255], dtype=np.uint8)
        else:
            bg_rgb = np.asarray(bg_color_rgb, dtype=np.uint8).reshape(3)
            bg_bgr = bg_rgb[::-1]
        filled[mask > 0] = bg_bgr
        return filled

    def _fill_annotation_regions_with_structure_color(self, image, repair_mask, structure_color):
        if image is None:
            raise ValueError("image cannot be None")
        if repair_mask is None:
            return image

        filled = image.copy()
        mask = cv2.threshold(repair_mask, 1, 255, cv2.THRESH_BINARY)[1]
        if np.count_nonzero(mask) == 0:
            return filled

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        color = np.asarray(structure_color, dtype=np.uint8).reshape(3)
        filled[mask > 0] = color
        return filled

    def visualize_results(self, results):
        """
        鍙鍖栧鐞嗙粨鏋溿€?        """
        plt.figure(figsize=(15, 10))

        plt.subplot(231)
        plt.imshow(cv2.cvtColor(results["original"], cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(232)
        plt.imshow(results["binary"], cmap="gray")
        plt.title("Binary Image")
        plt.axis("off")

        plt.subplot(233)
        plt.imshow(results["process_binary"], cmap="gray")
        plt.title("Processed Binary")
        plt.axis("off")

        plt.subplot(234)
        plt.imshow(cv2.cvtColor(results["adjust_fig"], cv2.COLOR_BGR2RGB))
        plt.title("Adjust Fig")
        plt.axis("off")

        plt.subplot(235)
        plt.imshow(results["bg_fig"])
        plt.title("BG Fig")
        plt.axis("off")

        plt.subplot(236)
        plt.imshow(cv2.cvtColor(results["repair_fig"], cv2.COLOR_BGR2RGB))
        plt.title("Repair Fig")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def _collect_image_files(self, input_path):
        """
        鏀堕泦杈撳叆鏂囦欢澶逛腑鐨勫彲澶勭悊鍥惧儚鏂囦欢銆?        """
        if not os.path.isdir(input_path):
            raise FileNotFoundError(f"杈撳叆鏂囦欢澶逛笉瀛樺湪: {input_path}")

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
        濡傛灉閰嶇疆浜?OCR 寮曟搸锛屽垯鏍规嵁褰撳墠鍥惧儚鎻愬彇鏂囨湰淇℃伅銆?        """
        if self.ocr_engine is None:
            return self.text_info_list

        text_info_list = []
        result = self.ocr_engine.predict(image_path)
        for res in result:
            boxes = res["rec_polys"]
            texts = res["rec_texts"]
            scores = res["rec_scores"]

            for box, text, score in zip(boxes, texts, scores):
                text_info_list.append({
                    "box": box,
                    "text": text,
                    "score": score,
                })

        self.text_info_list = text_info_list
        return text_info_list

    def _get_valid_ocr_text_infos(self):
        """
        Return OCR text entries that are strong enough to count as real text.
        """
        score_threshold = float(getattr(self, "ocr_text_score_threshold", 0.30))
        valid_infos = []
        for info in self.text_info_list or []:
            text = str(info.get("text", "")).strip()
            score = float(info.get("score", 0.0))
            if text and score >= score_threshold:
                valid_infos.append(info)
        return valid_infos

    def _make_passthrough_result(self, img_bgr, yolo_text_detections=None):
        """
        Build a normal result package without running FSS cleanup.
        Used when YOLO-text reports that the image has no text.
        """
        if img_bgr is None:
            raise ValueError("img_bgr cannot be None")

        h, w = img_bgr.shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        detection_vis = self.draw_yolo_detections(
            img_bgr,
            yolo_text_detections or [],
            mask=empty_mask,
        )

        return {
            "original": img_bgr.copy(),
            "subimg_coords": (0, w, 0, h),
            "text_detection_vis": detection_vis,
            "text_mask": empty_mask.copy(),
            "text_removed_mask": empty_mask.copy(),
            "text_detections": [],
            "yolo_clean_img": img_bgr.copy(),
            "yolo_detection_vis": detection_vis,
            "binary": binary,
            "process_binary": binary,
            "bg_fig": white_bg,
            "adjust_fig": img_bgr.copy(),
            "repair_fig": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            "main_fig": img_bgr.copy(),
            "mask": empty_mask.copy(),
            "repair_needed_mask": empty_mask.copy(),
            "gated_repair_mask": empty_mask.copy(),
            "diff_mask": empty_mask.copy(),
            "diff_annotation_mask": empty_mask.copy(),
            "annotation_mask": empty_mask.copy(),
            "refined_component_mask": empty_mask.copy(),
            "structure_mask": empty_mask.copy(),
            "structure_region": empty_mask.copy(),
            "dilated_structure_mask": empty_mask.copy(),
            "background_only_regions": empty_mask.copy(),
            "yolo_mask": empty_mask.copy(),
            "yolo_removed_mask": empty_mask.copy(),
            "removed_mask": empty_mask.copy(),
            "yolo_detections": [],
            "skip_fss_processing": True,
            "skip_reason": "no_text_detected_by_yolo",
        }

    def _detect_single_image(self, image_path, output_folder=None, visualize=False, refresh_text_info=True):
        """
        澶勭悊鍗曞紶鍥剧墖骞朵繚瀛樼粨鏋溿€?        """
        if refresh_text_info:
            self._extract_text_info(image_path)

        if output_folder is None:
            output_folder = os.path.splitext(os.path.basename(image_path))[0]

        os.makedirs(output_folder, exist_ok=True)

        input_img = cv2.imread(image_path)
        if input_img is None:
            raise FileNotFoundError(f"鍥惧儚鏂囦欢鏈壘鍒? {image_path}")

        valid_ocr_text_infos = self._get_valid_ocr_text_infos()
        yolo_text_detections = self.detect_elements_with_yolo(
            img=input_img,
            target_class_names=getattr(self, "yolo_text_labels", ("text",)),
        )

        if len(yolo_text_detections) == 0:
            print(
                f"YOLO 鏈娴嬪埌鏂囧瓧锛岃烦杩?FSS 娓呯悊锛岀洿鎺ヨ緭鍑哄師鍥剧粰鍚庣画娴佺▼ "
                f"(OCR={len(valid_ocr_text_infos)}锛屼粎浣滃弬鑰?"
            )
            results = [self._make_passthrough_result(input_img, yolo_text_detections)]
        else:
            print(
                f"鏂囨湰棰勬鏌? OCR={len(valid_ocr_text_infos)}, "
                f"YOLO-text={len(yolo_text_detections)}锛岀户缁墽琛?FSS 娓呯悊"
            )
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
        缁熶竴妫€娴嬪叆鍙ｏ紝鏀寔鍗曞紶鍥剧墖鎴栧浘鐗囨枃浠跺す銆?        """
        if os.path.isdir(image_path):
            image_files = self._collect_image_files(image_path)
            if not image_files:
                raise ValueError(f"鏂囦欢澶逛腑鏈壘鍒板彲澶勭悊鐨勫浘鐗? {image_path}")

            if output_folder is None:
                folder_name = os.path.basename(os.path.normpath(image_path))
                output_folder = f"{folder_name}_results"

            os.makedirs(output_folder, exist_ok=True)

            batch_results = []
            for single_image_path in image_files:
                image_name_no_ext = os.path.splitext(os.path.basename(single_image_path))[0]
                single_output_folder = os.path.join(output_folder, image_name_no_ext)
                print(f"寮€濮嬪鐞嗗浘鐗? {single_image_path}")
                results = self._detect_single_image(
                    image_path=single_image_path,
                    output_folder=single_output_folder,
                    visualize=visualize,
                    refresh_text_info=True,
                )
                batch_results.append({
                    "image_path": single_image_path,
                    "output_folder": single_output_folder,
                    "results": results,
                })

            return batch_results

        return self._detect_single_image(
            image_path=image_path,
            output_folder=output_folder,
            visualize=visualize,
            refresh_text_info=True,
        )

    def _save_results(self, results, output_folder, image_name_no_ext, index):
        """
        淇濆瓨澶勭悊缁撴灉鍒版寚瀹氭枃浠跺す銆?        涓嶅悓涓棿缁撴灉浼氭寜鐓х储寮曞啓鍏ヤ笉鍚岀殑瀛愮洰褰曚腑銆?        """
        sub_folder = os.path.join(output_folder, f"{index:03d}")
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)

        cv2.imwrite(os.path.join(sub_folder, "adjust_fig.png"), results["adjust_fig"])
        cv2.imwrite(os.path.join(sub_folder, "bg_fig.png"), cv2.cvtColor(results["bg_fig"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(sub_folder, "process.png"), results["original"])

        if results.get("text_detection_vis") is not None:
            cv2.imwrite(os.path.join(sub_folder, "text_detections.png"), results["text_detection_vis"])

        if "yolo_clean_img" in results:
            cv2.imwrite(os.path.join(sub_folder, "yolo_clean.png"), results["yolo_clean_img"])
        if "yolo_detection_vis" in results:
            cv2.imwrite(os.path.join(sub_folder, "yolo_detections.png"), results["yolo_detection_vis"])

        cv2.imwrite(os.path.join(sub_folder, "repair_fig.png"), cv2.cvtColor(results["repair_fig"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_mask.png"), results["mask"])

        if "diff_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_diff_mask.png"), results["diff_mask"])
        if "diff_annotation_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_diff_annotation_mask.png"), results["diff_annotation_mask"])
        if "annotation_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_annotation_mask.png"), results["annotation_mask"])
        if "refined_component_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, "refined_component_mask.png"), results["refined_component_mask"])
        if "repair_needed_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_repair_needed_mask.png"), results["repair_needed_mask"])
        if "gated_repair_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, "gated_repair_mask.png"), results["gated_repair_mask"])
        if "structure_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, "structure_mask.png"), results["structure_mask"])
        if "structure_region" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_structure_region.png"), results["structure_region"])
        if "dilated_structure_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_dilated_structure_mask.png"), results["dilated_structure_mask"])
        if "background_only_regions" in results:
            cv2.imwrite(os.path.join(sub_folder, "background_only_regions.png"), results["background_only_regions"])
        if "text_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_text_mask.png"), results["text_mask"])
        if "text_removed_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_text_removed_mask.png"), results["text_removed_mask"])
        if "yolo_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_yolo_mask.png"), results["yolo_mask"])
        if "yolo_removed_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_yolo_removed_mask.png"), results["yolo_removed_mask"])
        if "removed_mask" in results:
            cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}_removed_mask.png"), results["removed_mask"])

        cv2.imwrite(os.path.join(sub_folder, f"{image_name_no_ext}.png"), results["adjust_fig"])

