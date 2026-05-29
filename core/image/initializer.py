"""
图像初始化模块
原代码位置：Rebuild/ImageInit.py

功能：
- 读取和预处理图像
- 灰度化、归一化
- 高斯模糊去噪
- Canny边缘检测（支持改进的边缘检测）
- 自动将图像主体居中

重构说明：
- 将原 ImageInit 类重构为 ImageInitializer 类
- 保持原有算法和逻辑不变
- 保持原有接口方法不变
- 新增改进的边缘检测支持
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei"]

# 可选导入改进的边缘检测器
try:
    from core.image.improved_edge_detector import ImprovedEdgeDetector, EdgeDetectionStrategy
    IMPROVED_EDGE_DETECTOR_AVAILABLE = True
except ImportError:
    IMPROVED_EDGE_DETECTOR_AVAILABLE = False


class ImageInitializer:
    """
    图像初始化类
    原类：Rebuild/ImageInit.py 中的 ImageInit
    
    用于图像预处理、边缘检测和图像居中
    """
    
    def __init__(self, path, show=True, save='', 
                 use_improved_edges: bool = False,
                 edge_strategy: str = "auto",
                 use_adaptive_threshold: bool = False):
        """
        初始化图像处理器
        
        原方法：Rebuild/ImageInit.py 中的 __init__
        
        参数:
            path: 图像路径
            show: 是否显示处理结果
            save: 保存目录（如果为空则不保存）
            use_improved_edges: 是否使用改进的边缘检测（针对重叠图形和颜色差异小的区域）
            edge_strategy: 边缘检测策略（"auto", "adaptive_canny", "multi_channel", "color_enhanced", "combined"）
            use_adaptive_threshold: 是否使用自适应阈值（默认False，如需启用可设置为True）
        """
        self.__path = path
        self.__Save = save
        self.__original_img = None
        self.__main_contour = None
        self.__centered_img = None
        self.__gray_img = None
        self.__blurred_img = None
        self.__edges_img = None
        self.__use_improved_edges = use_improved_edges and IMPROVED_EDGE_DETECTOR_AVAILABLE
        self.__edge_strategy = edge_strategy
        self.__use_adaptive_threshold = use_adaptive_threshold
        self.__img_process(show=show)
    
    def trans2center(self, thresh_image):
        """
        检测图像主体中心并通过仿射变换将其平移至图像中心
        
        原方法：Rebuild/ImageInit.py 中的 trans2center
        
        参数:
            thresh_image: 预处理后的二值化图像（用于轮廓检测）
        
        返回:
            (centered_edge, centered_image) 元组
            - centered_edge: 主体居中后的边缘图
            - centered_image: 主体居中后的原图
        """
        # 原代码逻辑：1. 检测图像主体（最大轮廓）并计算其中心坐标
        # 查找所有外部轮廓（减少内部轮廓干扰）
        contours, _ = cv2.findContours(
            thresh_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("未检测到图像主体轮廓，请检查图像质量或调整预处理参数")
        
        # 原代码逻辑：选择面积最大的轮廓作为主体
        main_contour = max(contours, key=cv2.contourArea)
        self.__main_contour = main_contour
        # 原代码逻辑：计算主体最小外接矩形的中心（更贴合主体形状）
        rect = cv2.minAreaRect(main_contour)
        (x_center, y_center), _, _ = rect  # 提取主体中心坐标
        main_center = (int(x_center), int(y_center))
        
        # 原代码逻辑：2. 通过仿射变换将主体中心平移至图像中心
        h, w = thresh_image.shape[:2]
        img_center = (w // 2, h // 2)  # 图像中心坐标
        
        # 原代码逻辑：计算平移向量（图像中心 - 主体中心 = 需要移动的距离）
        dx = img_center[0] - main_center[0]
        dy = img_center[1] - main_center[1]
        
        # 原代码逻辑：生成仅包含平移的仿射变换矩阵
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        # 原代码逻辑：执行仿射变换，保持图像尺寸不变，边缘用边缘像素填充（避免黑边）
        centered_edge = cv2.warpAffine(
            thresh_image, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        centered_image = cv2.warpAffine(
            self.__original_img, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return centered_edge, centered_image
    
    def __result_vision(self):
        """
        可视化处理结果
        
        原方法：Rebuild/ImageInit.py 中的 __result_vision
        """
        plt.figure()
        plt.subplot(2, 2, 1)
        img_rgb = cv2.cvtColor(self.__original_img.copy(), cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("original_img")
        plt.axis("off")
        plt.subplot(2, 2, 2)
        cen_rgb = cv2.cvtColor(self.__centered_img.copy(), cv2.COLOR_BGR2RGB)
        plt.imshow(cen_rgb)
        plt.title("cen_rgb")
        plt.axis("off")
        plt.subplot(2, 2, 3)
        plt.imshow(self.__blurred_img, cmap='gray')
        plt.title("blurred_img")
        plt.axis("off")
        plt.subplot(2, 2, 4)
        plt.imshow(self.__edges_img, cmap='gray')
        plt.title("edges")
        plt.axis("off")
        if self.__Save:
            plt.savefig(self.__Save + '\ImageInit.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(3)  # 显示3秒
        plt.close()
    
    def _calculate_adaptive_canny_thresholds(self, blurred: np.ndarray) -> Tuple[int, int]:
        """
        计算自适应Canny阈值
        
        根据图像统计信息动态计算阈值，适应不同对比度和亮度的图像
        
        参数:
            blurred: 高斯模糊后的灰度图像
        
        返回:
            (low_threshold, high_threshold) 元组
        """
        # 计算图像统计信息
        median_val = float(np.median(blurred))
        std_val = float(np.std(blurred))
        mean_val = float(np.mean(blurred))
        
        # 方法1：基于中位数和标准差（主要方法）
        # 低阈值：中位数 - 标准差 * 0.5
        # 高阈值：中位数 + 标准差 * 2.0
        low_threshold = max(10, int(median_val - std_val * 0.5))
        high_threshold = min(255, int(median_val + std_val * 2.0))
        
        # 如果阈值范围太小，使用固定比例方法
        if high_threshold - low_threshold < 50:
            # 方法2：基于中位数的固定比例
            low_threshold = max(10, int(median_val * 0.5))
            high_threshold = min(255, int(median_val * 2.0))
        
        # 如果仍然范围太小，尝试基于百分位数的方法
        if high_threshold - low_threshold < 30:
            # 方法3：基于百分位数（更稳健）
            try:
                low_threshold = max(10, int(np.percentile(blurred, 10)))
                high_threshold = min(255, int(np.percentile(blurred, 90)))
            except:
                # 如果百分位数计算失败，使用均值方法
                low_threshold = max(10, int(mean_val * 0.4))
                high_threshold = min(255, int(mean_val * 1.6))
        
        # 确保高阈值至少是低阈值的2倍（Canny算法推荐）
        if high_threshold < low_threshold * 2:
            high_threshold = min(255, low_threshold * 2)
        
        # 极端情况处理：如果图像几乎全黑或全白，回退到固定阈值
        if std_val < 5.0:  # 标准差很小，说明图像对比度极低
            # 使用更保守的固定阈值
            low_threshold = 50
            high_threshold = 150
        elif median_val < 10:  # 图像几乎全黑
            low_threshold = 10
            high_threshold = 50
        elif median_val > 245:  # 图像几乎全白
            low_threshold = 200
            high_threshold = 255
        
        return int(low_threshold), int(high_threshold)
    
    def __img_process(self, show=False):
        """
        图像处理主流程
        
        原方法：Rebuild/ImageInit.py 中的 __img_process
        
        参数:
            show: 是否显示处理结果
        """
        # 原代码逻辑：读取图片
        image = cv2.imread(self.__path)
        if image is None:
            raise ValueError(f"无法读取图片: {self.__path}，请检查路径是否正确")
        self.__original_img = image
        
        # 原代码逻辑：转为灰度图并将图片归一化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.__gray_img = gray
        
        # 原代码逻辑：高斯滤波，滤除噪点
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 原代码注释掉的逻辑：
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        # blurred = cv2.dilate(blurred, kernel, iterations=1)
        # blurred = cv2.erode(blurred, kernel, iterations=4)
        self.__blurred_img = blurred
        
        # 【改进】使用改进的边缘检测（如果启用）
        if self.__use_improved_edges:
            from core.image.improved_edge_detector import create_improved_edge_detector
            edge_detector = create_improved_edge_detector(
                strategy=self.__edge_strategy,
                enhance_contrast=True,
                use_color_info=True
            )
            # 使用原始彩色图像进行改进的边缘检测
            edges = edge_detector.detect_edges(image)
        else:
            # 【改进】自适应阈值Canny边缘检测
            if self.__use_adaptive_threshold:
                # 计算自适应阈值
                low_threshold, high_threshold = self._calculate_adaptive_canny_thresholds(blurred)
                
                # 使用自适应阈值进行Canny边缘检测
                edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=5)
            else:
                # 原代码逻辑：利用Canny方法边缘提取（固定阈值）
                edges = cv2.Canny(blurred, 2500, 5000, apertureSize=5)
                # 原代码注释掉的逻辑：
                # edges = cv2.Canny(blurred, 100, 2000, apertureSize=5)
            
            # 原代码逻辑：设置开闭核对边缘轮廓进行开闭运算，合并细小边框
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.erode(edges, kernel, iterations=1)
        # 原代码逻辑：将图片移到中心
        self.__edges_img, self.__centered_img = self.trans2center(edges)
        
        if show:
            self.__result_vision()
    
    def original_img(self):
        """
        获取原始图像
        
        原方法：Rebuild/ImageInit.py 中的 original_img
        
        返回:
            原始图像（BGR格式）
        """
        return self.__original_img
    
    def centered_img(self):
        """
        获取居中后的图像
        
        原方法：Rebuild/ImageInit.py 中的 centered_img
        
        返回:
            居中后的图像（BGR格式）
        """
        return self.__centered_img
    
    def gray(self):
        """
        获取灰度图
        
        原方法：Rebuild/ImageInit.py 中的 gray
        
        返回:
            灰度图像
        """
        return self.__gray_img
    
    def blurred(self):
        """
        获取模糊图
        
        原方法：Rebuild/ImageInit.py 中的 blurred
        
        返回:
            高斯模糊后的图像
        """
        return self.__blurred_img
    
    def edges(self):
        """
        获取边缘图
        
        原方法：Rebuild/ImageInit.py 中的 edges
        
        返回:
            Canny边缘检测后的图像
        """
        return self.__edges_img


# 向后兼容的类名（保持原有接口）
ImageInit = ImageInitializer
