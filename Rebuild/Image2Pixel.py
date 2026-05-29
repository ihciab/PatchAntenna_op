import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

target_w = target_h = 256

class Image2Pixel:
    def __init__(self, num_pixel = 100):
        self.__Num_pixel = num_pixel

    def __image2pixel(self, img,
                      folder_path,
                      __curves_tree,
                      __contours_dict,
                      __solids_mask,
                      __col_mats,
                      show=True):
        """
        将图像像素化为 (Num_pixel, Num_pixel) 的网格
        返回像素开关矩阵，PEC区域为1，其余为0
        使用 self.__Curves_tree, self.__Solids_mask 和 self.__Contours_dict
        """

        # 重新获取图像用于像素化处理（获取图像尺寸）
        img_height, img_width = img.shape[:2]

        num_pixel = self.__Num_pixel

        # 初始化像素矩阵（全0）
        mat_pixel = np.zeros((num_pixel, num_pixel), dtype=np.int32)

        # 创建PEC区域的掩码
        # 参考 __solids2solids 的做法，遍历 Curves_tree 来构建solid区域
        pec_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # 遍历 Curves_tree，类似 __solids2solids 的逻辑
        for i, super_curve in enumerate(__curves_tree):
            # 构建solid名称，从顶层轮廓开始（参考 __solids2solids）
            nn = super_curve
            # 构建轮廓列表，用于创建掩码（使用拟合后的样条点）
            curve_list = []

            # 添加顶层轮廓的拟合点
            if super_curve in __contours_dict:
                outer_contour_points = __contours_dict[super_curve]['fitting']['points']
                if outer_contour_points is not None and len(outer_contour_points) > 0:
                    # 转换为OpenCV格式：确保是 (N, 1, 2) 格式，整数类型
                    if outer_contour_points.ndim == 2 and outer_contour_points.shape[1] == 2:
                        outer_contour = outer_contour_points.reshape(-1, 1, 2).astype(np.int32)
                    else:
                        outer_contour = outer_contour_points.astype(np.int32)
                    curve_list.append(outer_contour)

            # 处理子轮廓（内层轮廓）
            for curve in __curves_tree[super_curve]:
                nn += ('-' + curve)
                # 添加子轮廓的拟合点
                if curve in __contours_dict:
                    inner_contour_points = __contours_dict[curve]['fitting']['points']
                    if inner_contour_points is not None and len(inner_contour_points) > 0:
                        # 转换为OpenCV格式：确保是 (N, 1, 2) 格式，整数类型
                        if inner_contour_points.ndim == 2 and inner_contour_points.shape[1] == 2:
                            inner_contour = inner_contour_points.reshape(-1, 1, 2).astype(np.int32)
                        else:
                            inner_contour = inner_contour_points.astype(np.int32)
                        curve_list.append(inner_contour)

            # 检查该solid的材料是否为PEC（参考 __solids2solids 的逻辑）
            if nn in __solids_mask:
                color_name = __solids_mask[nn]
                if __col_mats.get(color_name) == 'PEC':
                    # 创建该solid区域的掩码
                    temp_mask = np.zeros((img_height, img_width), dtype=np.uint8)

                    # 首先绘制最外层轮廓（填充）
                    if len(curve_list) > 0:
                        cv2.drawContours(temp_mask, [curve_list[0]], -1, [255], -1)

                    # 然后减去内层轮廓（如果有），模拟 __solids2solids 中的subtract操作
                    if len(curve_list) > 1:
                        for inner_contour in curve_list[1:]:
                            cv2.drawContours(temp_mask, [inner_contour], -1, [0], -1)

                    # 将PEC区域添加到总掩码中
                    pec_mask = cv2.bitwise_or(pec_mask, temp_mask)
                    # 调试输出每个PEC solid的掩码
                    if show:
                        try:
                            dbg_dir = os.path.join(folder_path, 'debug_pixel')
                            os.makedirs(dbg_dir, exist_ok=True)
                            t_m = cv2.resize(temp_mask, (target_w, target_h), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(os.path.join(dbg_dir, f'solid_{nn}.png'), t_m)
                        except Exception as e:
                            print(f"[DEBUG] failed to save solid mask {nn}: {e}")
            else:
                if show:
                    print(f"[DEBUG] solid name not found in Solids_mask: {nn}")

        # 将图像划分为 Num_pixel x Num_pixel 的网格
        # 计算每个网格的尺寸
        grid_h = img_height / num_pixel
        grid_w = img_width / num_pixel

        # 对每个像素网格进行采样，判断是否为PEC
        for i in range(num_pixel):
            for j in range(num_pixel):
                # 计算网格在原图中的位置（使用中心点）
                center_y = int((i + 0.5) * grid_h)
                center_x = int((j + 0.5) * grid_w)

                # 确保坐标在图像范围内
                center_y = min(max(0, center_y), img_height - 1)
                center_x = min(max(0, center_x), img_width - 1)

                # 检查中心点是否在PEC区域内
                if pec_mask[center_y, center_x] > 0:
                    mat_pixel[i, j] = 1

        # 调试：保存PEC总掩码
        if show:
            try:
                dbg_dir = os.path.join(folder_path, 'debug_pixel')
                os.makedirs(dbg_dir, exist_ok=True)
                p_m = cv2.resize(pec_mask, (target_w, target_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(dbg_dir, 'pec_mask.png'), p_m)
                overlay = img.copy()
                if overlay.ndim == 2:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                red = overlay.copy()
                red[:, :, 2] = np.maximum(red[:, :, 2], pec_mask)
                r = cv2.resize(red, (target_w, target_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(dbg_dir, 'overlay_pec.png'), r)
                print(f"[DEBUG] saved pec_mask and overlay to {dbg_dir}")
            except Exception as e:
                print(f"[DEBUG] failed to save pec debug images: {e}")

        # 绘制像素矩阵图像
        if show:
            self.__visualize_pixel_matrix(mat_pixel, folder_path)

        return mat_pixel

    def __visualize_pixel_matrix(self, mat_pixel, folder_path):
        """
        可视化像素矩阵，生成并保存图像
        """
        num_pixel = self.__Num_pixel

        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # 左侧：像素矩阵可视化（二值图）
        axes[0].imshow(mat_pixel, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        axes[0].set_title(
            f'PEC Pixel Matrix ({num_pixel}x{num_pixel})\nPEC pixels: {np.sum(mat_pixel)}/{num_pixel * num_pixel}')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].grid(True, alpha=0.3)

        # 添加网格线以便更清楚地看到每个像素
        for i in range(num_pixel + 1):
            axes[0].axhline(i - 0.5, color='blue', linewidth=0.5, alpha=0.3)
            axes[0].axvline(i - 0.5, color='blue', linewidth=0.5, alpha=0.3)

        # 右侧：放大显示（使用颜色映射）
        im = axes[1].imshow(mat_pixel, cmap='RdYlBu_r', interpolation='nearest', vmin=0, vmax=1)
        axes[1].set_title(f'Color Map View')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im, ax=axes[1], label='PEC (1) / Non-PEC (0)')

        # 添加网格线
        for i in range(num_pixel + 1):
            axes[1].axhline(i - 0.5, color='black', linewidth=0.3, alpha=0.5)
            axes[1].axvline(i - 0.5, color='black', linewidth=0.3, alpha=0.5)

        plt.tight_layout()

        # 保存图像到文件夹
        output_dir = folder_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f'pixel_matrix_{num_pixel}x{num_pixel}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pixel matrix image saved to: {output_path}")
        plt.show(block=False)
        plt.pause(3)  # 显示5秒
        plt.close()

        # 也可以保存原始像素数据（可选）
        data_path = os.path.join(output_dir, f'pixel_matrix_{num_pixel}x{num_pixel}.txt')
        np.savetxt(data_path, mat_pixel, fmt='%d', delimiter=' ')
        print(f"Pixel matrix data saved to: {data_path}")

    def image2pixel(self, img,
                      folder_path,
                      __curves_tree,
                      __contours_dict,
                      __solids_mask,
                      __col_mats,
                      show=True):
        self.__image2pixel(img, 
                      folder_path,
                      __curves_tree,
                      __contours_dict,
                      __solids_mask,
                      __col_mats,
                      show)

    def set_num_pixel(self, n_p:int)->None:
        self.__Num_pixel = n_p
