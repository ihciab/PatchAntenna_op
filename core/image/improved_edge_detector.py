"""
改进的边缘检测模块
针对重叠图形和颜色差异较小区域的边缘检测优化

改进点：
1. 自适应Canny阈值（基于图像统计）
2. 多通道边缘检测（利用颜色信息）
3. 颜色对比度增强
4. 多种边缘检测策略的组合
5. 形态学后处理优化
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from enum import Enum


class EdgeDetectionStrategy(Enum):
    """边缘检测策略枚举"""
    ADAPTIVE_CANNY = "adaptive_canny"  # 自适应Canny
    MULTI_CHANNEL = "multi_channel"  # 多通道边缘检测
    COLOR_ENHANCED = "color_enhanced"  # 颜色增强边缘检测
    COMBINED = "combined"  # 组合策略
    AUTO = "auto"  # 自动选择最佳策略


class ImprovedEdgeDetector:
    """
    改进的边缘检测器
    针对重叠图形和颜色差异较小区域的优化
    """
    
    def __init__(self, 
                 strategy: EdgeDetectionStrategy = EdgeDetectionStrategy.AUTO,
                 enhance_contrast: bool = True,
                 use_color_info: bool = True):
        """
        初始化改进的边缘检测器
        
        参数:
            strategy: 边缘检测策略
            enhance_contrast: 是否增强对比度
            use_color_info: 是否使用颜色信息
        """
        self.strategy = strategy
        self.enhance_contrast = enhance_contrast
        self.use_color_info = use_color_info
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        检测边缘（主入口）
        
        参数:
            image: 输入图像（BGR格式）
        
        返回:
            边缘图像（二值图）
        """
        if self.strategy == EdgeDetectionStrategy.AUTO:
            return self._auto_detect_edges(image)
        elif self.strategy == EdgeDetectionStrategy.ADAPTIVE_CANNY:
            return self._adaptive_canny(image)
        elif self.strategy == EdgeDetectionStrategy.MULTI_CHANNEL:
            return self._multi_channel_edges(image)
        elif self.strategy == EdgeDetectionStrategy.COLOR_ENHANCED:
            return self._color_enhanced_edges(image)
        elif self.strategy == EdgeDetectionStrategy.COMBINED:
            return self._combined_edges(image)
        else:
            return self._adaptive_canny(image)
    
    def _auto_detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        自动选择最佳边缘检测策略
        
        参数:
            image: 输入图像
        
        返回:
            边缘图像
        """
        # 分析图像特征
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 计算图像对比度（标准差）
        contrast = np.std(gray)
        
        # 计算颜色多样性（如果是彩色图）
        color_diversity = 0.0
        if len(image.shape) == 3:
            # 计算各通道的方差
            b_var = np.var(image[:, :, 0])
            g_var = np.var(image[:, :, 1])
            r_var = np.var(image[:, :, 2])
            color_diversity = (b_var + g_var + r_var) / 3.0
        
        # 根据图像特征选择策略
        if contrast < 30:
            # 低对比度图像：使用颜色增强策略
            return self._color_enhanced_edges(image)
        elif color_diversity > 500:
            # 高颜色多样性：使用多通道策略
            return self._multi_channel_edges(image)
        else:
            # 默认：使用组合策略
            return self._combined_edges(image)
    
    def _calculate_adaptive_canny_thresholds(self, blurred: np.ndarray) -> Tuple[int, int]:
        """
        计算自适应Canny阈值（改进版：多级回退策略）
        
        参数:
            blurred: 模糊后的灰度图像
        
        返回:
            (low_threshold, high_threshold)
        """
        # 方法1：基于中位数和标准差（最常用）
        median_val = np.median(blurred)
        std_val = np.std(blurred)
        mean_val = np.mean(blurred)
        
        # 计算自适应阈值
        low_threshold = max(10, int(median_val - std_val * 0.5))
        high_threshold = min(255, int(median_val + std_val * 2.0))
        
        # 如果阈值范围太小，使用百分位数方法（回退策略1）
        if high_threshold - low_threshold < 50:
            p10 = np.percentile(blurred, 10)
            p90 = np.percentile(blurred, 90)
            low_threshold = max(10, int(p10))
            high_threshold = min(255, int(p90))
        
        # 如果仍然太小，使用固定比例（回退策略2）
        if high_threshold - low_threshold < 50:
            low_threshold = max(10, int(mean_val * 0.5))
            high_threshold = min(255, int(mean_val * 2.0))
        
        # 极端情况：如果图像几乎全黑或全白（回退策略3）
        if median_val < 10 or median_val > 245:
            # 使用固定阈值
            low_threshold = 50
            high_threshold = 150
        
        # 确保高阈值至少是低阈值的2倍（Canny推荐）
        if high_threshold < low_threshold * 2:
            high_threshold = min(255, low_threshold * 2)
        
        # 确保阈值在合理范围内
        low_threshold = max(10, min(200, low_threshold))
        high_threshold = max(50, min(255, high_threshold))
        
        return int(low_threshold), int(high_threshold)
    
    def _adaptive_canny(self, image: np.ndarray) -> np.ndarray:
        """
        自适应Canny边缘检测（改进版：使用多级回退策略）
        
        参数:
            image: 输入图像
        
        返回:
            边缘图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 归一化
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 【改进1】自适应阈值计算（使用多级回退策略）
        low_threshold, high_threshold = self._calculate_adaptive_canny_thresholds(blurred)
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=5)
        
        # 形态学后处理
        edges = self._morphological_postprocess(edges)
        
        return edges
    
    def _multi_channel_edges(self, image: np.ndarray) -> np.ndarray:
        """
        多通道边缘检测（利用颜色信息，改进版：优化噪声控制）
        
        参数:
            image: 输入图像（BGR格式）
        
        返回:
            边缘图像
        """
        if len(image.shape) != 3:
            # 如果是灰度图，回退到自适应Canny
            return self._adaptive_canny(image)
        
        # 【改进2】分别对每个BGR通道进行边缘检测
        b_edges = self._adaptive_canny(image[:, :, 0])
        g_edges = self._adaptive_canny(image[:, :, 1])
        r_edges = self._adaptive_canny(image[:, :, 2])
        
        # 合并BGR通道边缘（取并集）
        bgr_combined = cv2.bitwise_or(b_edges, cv2.bitwise_or(g_edges, r_edges))
        
        # 【改进3】使用Lab颜色空间增强边缘检测
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # 对L通道（亮度）进行边缘检测
        l_edges = self._adaptive_canny(l_channel)
        
        # 对a和b通道（色度）进行边缘检测（色度通道对颜色边界更敏感）
        # 【优化】对色度通道使用更严格的阈值，减少噪声
        a_edges = self._adaptive_canny_with_strict_threshold(a_channel)
        b_edges_lab = self._adaptive_canny_with_strict_threshold(b_channel)
        
        # 合并Lab通道边缘
        lab_combined = cv2.bitwise_or(l_edges, cv2.bitwise_or(a_edges, b_edges_lab))
        
        # 【改进】使用加权合并（BGR和Lab），而不是简单并集
        # BGR边缘权重0.7，Lab边缘权重0.3（降低Lab权重以减少噪声）
        final_edges = cv2.addWeighted(bgr_combined, 0.7, lab_combined, 0.3, 0)
        final_edges = (final_edges > 127).astype(np.uint8) * 255
        
        # 形态学后处理（使用自适应策略）
        final_edges = self._morphological_postprocess(final_edges)
        
        return final_edges
    
    def _adaptive_canny_with_strict_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        自适应Canny边缘检测（严格阈值版本，用于减少噪声）
        
        参数:
            image: 输入图像
        
        返回:
            边缘图像
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 归一化
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用更严格的阈值（提高阈值以减少噪声）
        low_threshold, high_threshold = self._calculate_adaptive_canny_thresholds(blurred)
        # 提高阈值（乘以1.5倍）
        low_threshold = int(low_threshold * 1.5)
        high_threshold = int(high_threshold * 1.5)
        low_threshold = max(10, min(200, low_threshold))
        high_threshold = max(50, min(255, high_threshold))
        
        # Canny边缘检测
        edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=5)
        
        return edges
    
    def _color_enhanced_edges(self, image: np.ndarray) -> np.ndarray:
        """
        颜色增强边缘检测（针对颜色差异小的区域，改进版）
        
        参数:
            image: 输入图像
        
        返回:
            边缘图像
        """
        if len(image.shape) != 3:
            # 如果是灰度图，先增强对比度
            enhanced = self._enhance_contrast_gray(image)
            return self._adaptive_canny(enhanced)
        
        # 【改进4】颜色对比度增强（使用更强的增强策略）
        enhanced = self._enhance_color_contrast_strong(image)
        
        # 使用多通道边缘检测
        edges = self._multi_channel_edges(enhanced)
        
        return edges
    
    def _enhance_color_contrast_strong(self, image: np.ndarray) -> np.ndarray:
        """
        强颜色对比度增强（针对颜色差异小的区域）
        
        参数:
            image: BGR图像
        
        返回:
            增强后的图像
        """
        # 转换到Lab颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道（亮度）应用更强的CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 更强的clipLimit
        l_enhanced = clahe.apply(l)
        
        # 对a和b通道进行更强的对比度拉伸
        # 使用更激进的归一化
        a_min, a_max = np.percentile(a, [5, 95])
        b_min, b_max = np.percentile(b, [5, 95])
        
        if a_max > a_min:
            a_enhanced = np.clip((a - a_min) / (a_max - a_min) * 255, 0, 255).astype(np.uint8)
        else:
            a_enhanced = a
        
        if b_max > b_min:
            b_enhanced = np.clip((b - b_min) / (b_max - b_min) * 255, 0, 255).astype(np.uint8)
        else:
            b_enhanced = b
        
        # 合并通道
        lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        
        # 转换回BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _combined_edges(self, image: np.ndarray) -> np.ndarray:
        """
        组合策略边缘检测（多种方法结合，改进版：优化融合策略）
        
        参数:
            image: 输入图像
        
        返回:
            边缘图像
        """
        # 方法1：自适应Canny（基础方法，权重0.6）
        edges1 = self._adaptive_canny(image)
        
        # 方法2：多通道边缘检测（利用颜色信息，权重0.3）
        if len(image.shape) == 3:
            edges2 = self._multi_channel_edges(image)
        else:
            edges2 = edges1.copy()
        
        # 方法3：颜色增强边缘检测（针对低对比度，权重0.1）
        if len(image.shape) == 3:
            edges3 = self._color_enhanced_edges(image)
        else:
            edges3 = edges1.copy()
        
        # 【改进】使用加权融合而不是简单并集
        # 方法1（基础）权重最高，方法2和3作为补充
        combined = cv2.addWeighted(edges1, 0.6, edges2, 0.3, 0)
        combined = cv2.addWeighted(combined, 1.0, edges3, 0.1, 0)
        # 使用更高的阈值来减少噪声
        combined = (combined > 100).astype(np.uint8) * 255
        
        # 形态学后处理（使用自适应策略）
        combined = self._morphological_postprocess(combined)
        
        return combined
    
    def _enhance_contrast_gray(self, gray: np.ndarray) -> np.ndarray:
        """
        增强灰度图对比度
        
        参数:
            gray: 灰度图
        
        返回:
            增强后的灰度图
        """
        # CLAHE（对比度受限的自适应直方图均衡化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 归一化
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return enhanced
    
    def _enhance_color_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        增强颜色对比度
        
        参数:
            image: BGR图像
        
        返回:
            增强后的图像
        """
        # 转换到Lab颜色空间（更适合颜色处理）
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道（亮度）应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # 增强a和b通道的对比度（色度）
        # 使用线性拉伸
        a_enhanced = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        b_enhanced = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 合并通道
        lab_enhanced = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        
        # 转换回BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _adaptive_morphological_postprocess(self, edges: np.ndarray) -> np.ndarray:
        """
        自适应形态学后处理（改进版：根据边缘特征调整，优化噪声控制）
        
        参数:
            edges: 边缘图像
        
        返回:
            处理后的边缘图像
        """
        # 分析边缘特征
        edge_density = np.count_nonzero(edges) / edges.size if edges.size > 0 else 0.0
        
        # 根据边缘密度自适应调整核大小和处理策略
        h, w = edges.shape[:2]
        base_kernel_size = max(3, min(5, int(min(h, w) / 200)))
        
        if edge_density > 0.3:  # 边缘非常密集（可能噪声过多）
            # 使用较大的核和更强的腐蚀来去除噪声
            kernel_size = min(7, base_kernel_size + 2)
            dilate_iterations = 1
            erode_iterations = 2  # 更强的腐蚀去除噪声
        elif edge_density > 0.1:  # 边缘密集
            kernel_size = base_kernel_size
            dilate_iterations = 1
            erode_iterations = 1
        elif edge_density < 0.01:  # 边缘稀疏
            kernel_size = min(7, base_kernel_size + 2)
            dilate_iterations = 2  # 更多膨胀连接边缘
            erode_iterations = 1
        else:  # 中等密度
            kernel_size = base_kernel_size
            dilate_iterations = 2
            erode_iterations = 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # 先膨胀（连接断开的边缘）
        edges = cv2.dilate(edges, kernel, iterations=dilate_iterations)
        
        # 再腐蚀（去除细小噪声）
        edges = cv2.erode(edges, kernel, iterations=erode_iterations)
        
        # 【改进】如果边缘密度仍然很高，进行额外的噪声过滤
        if edge_density > 0.3:
            # 使用面积过滤：去除小连通区域
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            min_area = (h * w) * 0.0001  # 最小面积阈值（图像面积的0.01%）
            filtered_edges = np.zeros_like(edges)
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(filtered_edges, [contour], -1, 255, 1)
            edges = filtered_edges
        
        return edges
    
    def _morphological_postprocess(self, edges: np.ndarray) -> np.ndarray:
        """
        形态学后处理（改进版：使用自适应策略）
        
        参数:
            edges: 边缘图像
        
        返回:
            处理后的边缘图像
        """
        return self._adaptive_morphological_postprocess(edges)
    
    def detect_edges_with_info(self, image: np.ndarray) -> Dict:
        """
        检测边缘并返回详细信息
        
        参数:
            image: 输入图像
        
        返回:
            包含边缘图像和检测信息的字典
        """
        edges = self.detect_edges(image)
        
        # 计算边缘统计信息
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # 检测轮廓数量
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_count = len(contours)
        
        return {
            'edges': edges,
            'edge_pixel_count': int(edge_pixels),
            'edge_ratio': float(edge_ratio),
            'contour_count': int(contour_count),
            'strategy': self.strategy.value if isinstance(self.strategy, EdgeDetectionStrategy) else str(self.strategy)
        }


def create_improved_edge_detector(strategy: str = "auto", 
                                  enhance_contrast: bool = True,
                                  use_color_info: bool = True) -> ImprovedEdgeDetector:
    """
    创建改进的边缘检测器（便捷函数）
    
    参数:
        strategy: 策略名称（"auto", "adaptive_canny", "multi_channel", "color_enhanced", "combined"）
        enhance_contrast: 是否增强对比度
        use_color_info: 是否使用颜色信息
    
    返回:
        ImprovedEdgeDetector实例
    """
    strategy_map = {
        "auto": EdgeDetectionStrategy.AUTO,
        "adaptive_canny": EdgeDetectionStrategy.ADAPTIVE_CANNY,
        "multi_channel": EdgeDetectionStrategy.MULTI_CHANNEL,
        "color_enhanced": EdgeDetectionStrategy.COLOR_ENHANCED,
        "combined": EdgeDetectionStrategy.COMBINED,
    }
    
    strategy_enum = strategy_map.get(strategy.lower(), EdgeDetectionStrategy.AUTO)
    
    return ImprovedEdgeDetector(
        strategy=strategy_enum,
        enhance_contrast=enhance_contrast,
        use_color_info=use_color_info
    )
