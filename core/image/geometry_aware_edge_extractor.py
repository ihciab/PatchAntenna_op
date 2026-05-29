"""
几何感知边缘提取器（重构版：二次极端抑制器）
基于固定阈值Canny，通过几何一致性筛选提升F1分数

核心思想（重构）：
- 几何感知必须是"二次极端抑制器"，而不是"增强器"
- 轮廓数绝不能超过固定阈值原始数量（铁律）
- 使用评分系统，而不是OR逻辑
- 融入GT先验（闭合、外围、最大面积）

设计原则：
1. 不修改固定阈值Canny（保持baseline）
2. 只做筛选，不新增边缘
3. 轮廓数量是第一约束，几何质量是第二约束
4. Top-K（K≈原始轮廓数）是最强武器
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares


class GeometryAwareEdgeExtractor:
    """
    几何感知边缘提取器
    
    在固定阈值Canny的基础上，通过几何一致性筛选提升边缘质量
    """
    
    def __init__(self,
                 min_contour_length_ratio: float = 0.005,
                 max_curvature_variance: float = 0.5,
                 line_fit_threshold: float = 2.0,
                 arc_fit_threshold: float = 2.0,
                 min_arc_radius: float = 5.0,
                 max_arc_radius: float = 500.0,
                 prefer_closed: bool = True,
                 min_points_for_fit: int = 5,
                 # 新增：Top-K和评分系统参数
                 max_contour_count: Optional[int] = None,  # 最大轮廓数（None表示不限制，但会使用原始数量）
                 geometry_score_threshold: float = 0.6,     # 几何评分阈值（0-1）
                 use_gt_prior: bool = True):                # 是否使用GT先验规则
        """
        初始化几何感知边缘提取器
        
        参数:
            min_contour_length_ratio: 最小轮廓长度比例（相对于图像周长）
            max_curvature_variance: 最大曲率方差（弧度²）
            line_fit_threshold: 直线拟合误差阈值（像素）
            arc_fit_threshold: 圆弧拟合误差阈值（像素）
            min_arc_radius: 最小圆弧半径（像素）
            max_arc_radius: 最大圆弧半径（像素）
            prefer_closed: 是否优先保留闭合轮廓
            min_points_for_fit: 拟合所需的最小点数
        """
        self.min_contour_length_ratio = min_contour_length_ratio
        self.max_curvature_variance = max_curvature_variance
        self.line_fit_threshold = line_fit_threshold
        self.arc_fit_threshold = arc_fit_threshold
        self.min_arc_radius = min_arc_radius
        self.max_arc_radius = max_arc_radius
        self.prefer_closed = prefer_closed
        self.min_points_for_fit = min_points_for_fit
        self.max_contour_count = max_contour_count
        self.geometry_score_threshold = geometry_score_threshold
        self.use_gt_prior = use_gt_prior
    
    def extract_edges(self, image: np.ndarray, 
                     canny_low: int = 100, 
                     canny_high: int = 200) -> np.ndarray:
        """
        从图像中提取几何感知边缘
        
        参数:
            image: 输入图像（BGR或灰度）
            canny_low: Canny低阈值
            canny_high: Canny高阈值
        
        返回:
            几何感知边缘图像（二值图）
        """
        # Step 1: 固定阈值Canny（baseline，不修改）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 归一化（与原始方法保持一致）
        gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 固定阈值Canny
        edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=5)
        
        # 形态学操作（保持原逻辑）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        # Step 2: 统计原始轮廓数（用于Top-K约束）
        contours_original, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        original_contour_count = len(contours_original)
        
        # Step 3: 几何感知筛选（二次极端抑制）
        # 铁律：轮廓数绝不能超过原始数量
        max_count = self.max_contour_count if self.max_contour_count is not None else original_contour_count
        # 确保不超过原始数量
        max_count = min(max_count, original_contour_count)
        
        filtered_edges = self._geometry_aware_filter(
            edges, 
            image.shape[:2],
            max_contour_count=max_count
        )
        
        return filtered_edges
    
    def _geometry_aware_filter(self, edges: np.ndarray, 
                               image_shape: Tuple[int, int],
                               max_contour_count: Optional[int] = None) -> np.ndarray:
        """
        几何感知筛选（重构版：评分系统+Top-K）
        
        核心改变：
        1. 使用评分系统，而不是OR逻辑
        2. Top-K机制：只保留评分最高的K个轮廓（K≤原始数量）
        3. GT先验规则：闭合、外围、最大面积优先
        
        参数:
            edges: 初始边缘图像
            image_shape: 图像尺寸 (height, width)
            max_contour_count: 最大轮廓数（铁律：不超过原始数量）
        
        返回:
            筛选后的边缘图像
        """
        # 查找轮廓
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return edges
        
        # 计算图像周长和面积（用于归一化）
        image_perimeter = 2 * (image_shape[0] + image_shape[1])
        image_area = image_shape[0] * image_shape[1]
        min_contour_length = image_perimeter * self.min_contour_length_ratio
        
        # 计算每个轮廓的几何评分
        scored_contours = []
        
        for i, contour in enumerate(contours):
            # 基础筛选：长度和点数
            contour_length = cv2.arcLength(contour, False)
            if contour_length < min_contour_length:
                continue
            
            points = contour.reshape(-1, 2).astype(float)
            if len(points) < self.min_points_for_fit:
                continue
            
            # 计算几何评分（0-1范围）
            score = self._calculate_geometry_score(
                contour, points, contour_length, 
                image_perimeter, image_area, image_shape, hierarchy, i
            )
            
            scored_contours.append((score, contour, points))
        
        # 如果没有候选轮廓，返回原始边缘
        if len(scored_contours) == 0:
            return edges
        
        # 按评分排序（降序）
        scored_contours.sort(reverse=True, key=lambda x: x[0])
        
        # Top-K机制：只保留评分最高的K个轮廓
        # K = min(max_contour_count, 原始轮廓数)
        if max_contour_count is not None:
            k = min(max_contour_count, len(scored_contours))
        else:
            k = len(scored_contours)
        
        # 额外筛选：只保留评分超过阈值的轮廓
        # 但确保不超过K个
        good_contours = []
        for score, contour, _ in scored_contours[:k]:
            if score >= self.geometry_score_threshold:
                good_contours.append(contour)
        
        # 如果筛选后没有轮廓，至少保留评分最高的1个（回退机制）
        if len(good_contours) == 0 and len(scored_contours) > 0:
            good_contours = [scored_contours[0][1]]
        
        # 铁律：确保轮廓数不超过max_contour_count
        if max_contour_count is not None and len(good_contours) > max_contour_count:
            # 只保留评分最高的max_contour_count个
            good_contours = [c[1] for c in scored_contours[:max_contour_count]]
        
        # 绘制筛选后的轮廓
        filtered_edges = np.zeros_like(edges)
        if len(good_contours) > 0:
            cv2.drawContours(filtered_edges, good_contours, -1, 255, 1)
        
        return filtered_edges
    
    def _calculate_geometry_score(self, 
                                  contour: np.ndarray,
                                  points: np.ndarray,
                                  contour_length: float,
                                  image_perimeter: float,
                                  image_area: float,
                                  image_shape: Tuple[int, int],
                                  hierarchy: Optional[np.ndarray],
                                  contour_idx: int) -> float:
        """
        计算轮廓的几何评分（0-1范围）
        
        评分组成：
        1. 几何拟合质量（Line/Arc拟合误差）
        2. 曲率稳定性
        3. 轮廓长度（归一化）
        4. GT先验规则（闭合、外围、面积）
        
        参数:
            contour: 轮廓点
            points: 轮廓点数组（已转换）
            contour_length: 轮廓长度
            image_perimeter: 图像周长
            image_area: 图像面积
            image_shape: 图像尺寸
            hierarchy: 轮廓层级
            contour_idx: 轮廓索引
        
        返回:
            几何评分（0-1）
        """
        score = 0.0
        
        # 1. 几何拟合质量（权重：0.4）
        fits_geometry, fit_quality = self._check_geometric_fit_with_quality(points)
        if fits_geometry:
            score += 0.4 * fit_quality
        else:
            # 即使不能拟合，也给予少量分数（避免完全排除）
            score += 0.1 * (1.0 - fit_quality)
        
        # 2. 曲率稳定性（权重：0.15）
        curvature_ok, curvature_score = self._check_curvature_stability_with_score(points)
        score += 0.15 * curvature_score
        
        # 3. 轮廓长度（权重：0.2）
        # 归一化到[0, 1]：长度越长，分数越高
        length_ratio = contour_length / image_perimeter
        length_score = min(1.0, length_ratio * 10.0)  # 假设理想长度约为周长的10%
        score += 0.2 * length_score
        
        # 4. GT先验规则（权重：0.25）
        if self.use_gt_prior:
            prior_score = self._calculate_gt_prior_score(
                contour, points, contour_length, image_area, image_shape, hierarchy, contour_idx
            )
            score += 0.25 * prior_score
        else:
            # 如果不使用GT先验，给予基础分数
            score += 0.25 * 0.5
        
        return min(1.0, score)  # 确保不超过1.0
    
    def _check_curvature_stability_with_score(self, points: np.ndarray) -> Tuple[bool, float]:
        """
        检查曲率稳定性（带评分版本）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            (是否稳定, 稳定性评分0-1)
        """
        if len(points) < 3:
            return False, 0.0
        
        # 计算方向角
        diffs = np.diff(points, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        # 计算角度变化（曲率）
        angle_diffs = np.diff(angles)
        # 处理角度跳跃（-π到π或π到-π）
        angle_diffs = np.angle(np.exp(1j * angle_diffs))
        
        # 计算曲率方差
        curvature_variance = np.var(angle_diffs)
        
        # 转换为评分（0-1）：方差越小，评分越高
        # 使用sigmoid函数平滑过渡
        score = 1.0 / (1.0 + curvature_variance / (self.max_curvature_variance + 1e-9))
        
        is_stable = curvature_variance < self.max_curvature_variance
        
        return is_stable, score
    
    def _check_curvature_stability(self, points: np.ndarray) -> bool:
        """
        模块2: 检查曲率稳定性
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            是否通过曲率稳定性检查
        """
        if len(points) < 3:
            return False
        
        # 计算方向角
        diffs = np.diff(points, axis=0)
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        # 计算角度变化（曲率）
        angle_diffs = np.diff(angles)
        # 处理角度跳跃（-π到π或π到-π）
        angle_diffs = np.angle(np.exp(1j * angle_diffs))
        
        # 计算曲率方差
        curvature_variance = np.var(angle_diffs)
        
        return curvature_variance < self.max_curvature_variance
    
    def _check_geometric_fit_with_quality(self, points: np.ndarray) -> Tuple[bool, float]:
        """
        检查几何拟合一致性（带评分版本）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            (是否拟合, 拟合质量0-1)
        """
        line_quality = self._fits_line_with_quality(points)
        arc_quality = self._fits_arc_with_quality(points)
        
        # 取最佳拟合质量
        best_quality = max(line_quality, arc_quality)
        fits = best_quality > 0.3  # 阈值：质量>0.3认为可以拟合
        
        return fits, best_quality
    
    def _check_geometric_fit(self, points: np.ndarray) -> bool:
        """
        模块3: 检查几何拟合一致性（核心）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            是否可以被简单几何（Line/Arc）解释
        """
        # 3.1 尝试直线拟合
        if self._fits_line(points):
            return True
        
        # 3.2 尝试圆弧拟合
        if self._fits_arc(points):
            return True
        
        # 如果既不是直线也不是圆弧，删除
        return False
    
    def _fits_line_with_quality(self, points: np.ndarray) -> float:
        """
        直线拟合（带质量评分版本）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            拟合质量（0-1），0表示不能拟合，1表示完美拟合
        """
        if len(points) < 2:
            return 0.0
        
        # 计算轮廓长度
        contour_length = cv2.arcLength(points.astype(np.int32), False)
        if contour_length < 1e-6:
            return 0.0
        
        # 使用PCA拟合直线
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        # SVD分解
        try:
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            if len(s) == 0 or s[0] < 1e-6:
                return 0.0
            direction = Vt[0]  # 主方向
        except:
            return 0.0
        
        # 计算所有点到直线的距离
        projections = np.dot(centered, direction)
        projected_points = np.outer(projections, direction)
        residuals = centered - projected_points
        distances = np.linalg.norm(residuals, axis=1)
        
        max_error = np.max(distances)
        mean_error = np.mean(distances)
        relative_error = max_error / (contour_length + 1e-9)
        
        # 计算拟合质量（0-1）
        # 质量 = 1 - 归一化误差
        error_score = max_error / (self.line_fit_threshold + 1e-9)
        relative_score = relative_error / 0.1  # 假设相对误差0.1为阈值
        
        # 综合评分：误差越小，质量越高
        quality = 1.0 / (1.0 + error_score + relative_score)
        
        return quality
    
    def _fits_line(self, points: np.ndarray) -> bool:
        """
        3.1 直线一致性检查（改进版：更宽松）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            是否拟合为直线
        """
        if len(points) < 2:
            return False
        
        # 计算轮廓长度
        contour_length = cv2.arcLength(points.astype(np.int32), False)
        if contour_length < 1e-6:
            return False
        
        # 使用PCA拟合直线
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        # SVD分解
        try:
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            if len(s) == 0 or s[0] < 1e-6:
                return False
            direction = Vt[0]  # 主方向
        except:
            return False
        
        # 计算所有点到直线的距离
        projections = np.dot(centered, direction)
        projected_points = np.outer(projections, direction)
        residuals = centered - projected_points
        distances = np.linalg.norm(residuals, axis=1)
        
        max_error = np.max(distances)
        mean_error = np.mean(distances)
        
        # 改进的判据：同时考虑最大误差和平均误差，以及相对误差
        relative_error = max_error / (contour_length + 1e-9)
        
        # 更宽松的条件：最大误差 < 阈值 或 (平均误差小 且 相对误差小)
        return (max_error < self.line_fit_threshold or 
                (mean_error < self.line_fit_threshold * 0.7 and relative_error < 0.1))
    
    def _fits_arc_with_quality(self, points: np.ndarray) -> float:
        """
        圆弧拟合（带质量评分版本）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            拟合质量（0-1），0表示不能拟合，1表示完美拟合
        """
        if len(points) < 3:
            return 0.0
        
        # 使用最小二乘法拟合圆
        try:
            center, radius = self._fit_circle_taubin(points)
            
            # 检查半径范围
            if radius < self.min_arc_radius or radius > self.max_arc_radius:
                return 0.0
            
            # 计算拟合误差
            distances = np.linalg.norm(points - center, axis=1)
            errors = np.abs(distances - radius)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            
            # 检查相对误差
            relative_error = np.std(distances) / (radius + 1e-9)
            
            # 计算拟合质量（0-1）
            error_score = max_error / (self.arc_fit_threshold + 1e-9)
            relative_score = relative_error / 0.1  # 假设相对误差0.1为阈值
            
            # 综合评分：误差越小，质量越高
            quality = 1.0 / (1.0 + error_score + relative_score)
            
            return quality
        except:
            return 0.0
    
    def _fits_arc(self, points: np.ndarray) -> bool:
        """
        3.2 圆弧一致性检查（改进版：更宽松）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            是否拟合为圆弧
        """
        if len(points) < 3:
            return False
        
        # 使用最小二乘法拟合圆
        try:
            center, radius = self._fit_circle_taubin(points)
            
            # 检查半径范围（更宽松）
            if radius < self.min_arc_radius or radius > self.max_arc_radius:
                return False
            
            # 计算拟合误差
            distances = np.linalg.norm(points - center, axis=1)
            errors = np.abs(distances - radius)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            
            # 检查相对误差
            relative_error = np.std(distances) / (radius + 1e-9)
            
            # 改进的判据：更宽松的条件
            # 1. 最大误差 < 阈值 且 相对误差 < 10%
            # 2. 或者 平均误差小 且 相对误差 < 8%
            return ((max_error < self.arc_fit_threshold and relative_error < 0.10) or
                    (mean_error < self.arc_fit_threshold * 0.7 and relative_error < 0.08))
        except:
            return False
    
    def _fit_circle_taubin(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用Taubin方法拟合圆（更稳定）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            (center, radius)
        """
        x = points[:, 0]
        y = points[:, 1]
        n = len(points)
        
        # Taubin方法
        # 计算质心
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # 中心化
        u = x - x_mean
        v = y - y_mean
        
        # 计算矩阵元素
        Suu = np.sum(u * u)
        Suv = np.sum(u * v)
        Svv = np.sum(v * v)
        Suuu = np.sum(u * u * u)
        Suvv = np.sum(u * v * v)
        Svvv = np.sum(v * v * v)
        Suuv = np.sum(u * u * v)
        
        # 构建线性系统
        A = np.array([
            [Suu, Suv],
            [Suv, Svv]
        ])
        
        b = np.array([
            0.5 * (Suuu + Suvv),
            0.5 * (Svvv + Suuv)
        ])
        
        # 求解
        try:
            uc, vc = np.linalg.solve(A, b)
            center = np.array([uc + x_mean, vc + y_mean])
            
            # 计算半径
            distances = np.linalg.norm(points - center, axis=1)
            radius = np.mean(distances)
            
            return center, radius
        except:
            # 回退到简单方法
            center = np.array([x_mean, y_mean])
            distances = np.linalg.norm(points - center, axis=1)
            radius = np.mean(distances)
            return center, radius
    
    def _calculate_gt_prior_score(self,
                                  contour: np.ndarray,
                                  points: np.ndarray,
                                  contour_length: float,
                                  image_area: float,
                                  image_shape: Tuple[int, int],
                                  hierarchy: Optional[np.ndarray],
                                  contour_idx: int) -> float:
        """
        计算GT先验评分（0-1）
        
        GT先验规则：
        1. 闭合轮廓优先（权重：0.4）
        2. 外围轮廓优先（权重：0.3）
        3. 最大面积优先（权重：0.3）
        
        参数:
            contour: 轮廓点
            points: 轮廓点数组
            contour_length: 轮廓长度
            image_area: 图像面积
            image_shape: 图像尺寸
            hierarchy: 轮廓层级
            contour_idx: 轮廓索引
        
        返回:
            GT先验评分（0-1）
        """
        score = 0.0
        
        # 1. 闭合轮廓优先（权重：0.4）
        is_closed = self._is_contour_closed(points)
        if is_closed:
            score += 0.4
        else:
            # 开放轮廓给予少量分数
            score += 0.1
        
        # 2. 外围轮廓优先（权重：0.3）
        # 判断是否为外围轮廓：没有父轮廓，或者层级为0
        is_outer = True
        if hierarchy is not None and len(hierarchy) > 0 and len(hierarchy[0]) > contour_idx:
            h = hierarchy[0][contour_idx]
            parent_idx = h[3]  # parent索引
            if parent_idx >= 0:
                is_outer = False
        
        if is_outer:
            score += 0.3
        else:
            # 内轮廓（孔洞）给予少量分数
            score += 0.1
        
        # 3. 最大面积优先（权重：0.3）
        # 计算轮廓面积
        contour_area = cv2.contourArea(contour)
        if contour_area > 0:
            # 归一化面积（相对于图像面积）
            area_ratio = contour_area / image_area
            # 面积越大，分数越高（使用对数缩放，避免过大面积占主导）
            area_score = min(1.0, np.log1p(area_ratio * 100) / np.log1p(10))
            score += 0.3 * area_score
        else:
            # 面积为0（可能是开放轮廓），给予少量分数
            score += 0.05
        
        return min(1.0, score)
    
    def _is_contour_closed(self, points: np.ndarray, 
                          tolerance: float = 2.0) -> bool:
        """
        模块4: 检查轮廓是否闭合
        
        参数:
            points: 轮廓点数组 (N, 2)
            tolerance: 闭合容差（像素）
        
        返回:
            是否闭合
        """
        if len(points) < 3:
            return False
        
        distance = np.linalg.norm(points[0] - points[-1])
        return distance < tolerance
    
    def _has_strong_geometric_evidence(self, points: np.ndarray) -> bool:
        """
        检查开放轮廓是否有强几何证据
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            是否有强几何证据
        """
        # 更严格的直线拟合
        if self._fits_line(points):
            # 检查是否接近直线（更严格）
            mean = np.mean(points, axis=0)
            centered = points - mean
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            direction = Vt[0]
            projections = np.dot(centered, direction)
            projected_points = np.outer(projections, direction)
            residuals = centered - projected_points
            distances = np.linalg.norm(residuals, axis=1)
            max_error = np.max(distances)
            
            # 开放轮廓需要更严格的拟合
            return max_error < (self.line_fit_threshold * 0.7)
        
        # 更严格的圆弧拟合
        if self._fits_arc(points):
            try:
                center, radius = self._fit_circle_taubin(points)
                distances = np.linalg.norm(points - center, axis=1)
                errors = np.abs(distances - radius)
                max_error = np.max(errors)
                
                # 开放轮廓需要更严格的拟合
                return max_error < (self.arc_fit_threshold * 0.7)
            except:
                return False
        
        return False


def create_geometry_aware_extractor(**kwargs) -> GeometryAwareEdgeExtractor:
    """
    创建几何感知边缘提取器（便捷函数）
    
    参数:
        **kwargs: 传递给GeometryAwareEdgeExtractor的参数
    
    返回:
        GeometryAwareEdgeExtractor实例
    """
    return GeometryAwareEdgeExtractor(**kwargs)
