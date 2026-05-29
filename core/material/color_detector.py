"""
颜色检测器模块
原代码位置：Rebuild/ColorDetector.py

功能：
- 基于HSV颜色空间检测图像中的颜色分布
- 支持10种预设颜色（黑、灰、白、红、橙、黄、绿、青、蓝、紫）
- 支持自定义颜色范围
- 计算颜色占比和主要颜色

重构说明：
- 将原 ColorDetector 类重构为 ColorDetector 类（保持类名不变）
- 保持原有算法和逻辑完全不变
"""
import cv2
import numpy as np
from typing import Tuple


class ColorDetector:
    """
    颜色检测器类
    原类：Rebuild/ColorDetector.py 中的 ColorDetector
    
    用于检测图像中的颜色分布和主要颜色
    """
    def __init__(self):
        # 定义常见颜色的HSV范围
        self.Color_results = None
        """
        Color_match:dict = {
            color{
                area（色块面积）: 0.0
                mask（色块掩码）: ndarray
                percentage（色块在整体的百分比）: 0.0
            }
        }
        """
        # 预设的HSV色彩空间，但是opencv中HSV色彩空间的范围和通常的不一样需要做映射,并通过self.__set_dcr()对其初始化
        self.__Default_color_ranges = None
        self.__set_dcr()
        # 预设的可视化三通道RGB色彩空间，需要将HSV映射到对应的RGB中（但是cv中是有现成的转换函数的），并通过self.__set_dcv()对其初始化
        self.__Default_color_visuals = None
        self.__set_dcv()
        # 可修改的HSV色彩空间，在没有输入的情况下使用预设的HSV色彩空间，即Default_color_ranges
        self.Color_ranges = {}
        # 可修改的RGB色彩空间，在没有输入的情况下使用预设的RGB色彩空间，即Default_color_visuals
        self.Color_visuals = {}

    def __set_dcr(self):
        self.__Default_color_ranges = {
            'black': [((0, 0, 0), (180, 255, 30))],
            'gray':[((0, 0, 46), (180, 43, 220))],
            'white':[((0, 0, 221), (180, 30, 255))],
            'red': [
                ((0, 43, 46), (10, 255, 255)),  # 红色范围1（低H值）
                ((156, 43, 46), (180, 255, 255))  # 红色范围2（高H值）
            ],
            'orange': [((11, 43, 46), (25, 255, 255))],
            'yellow': [((26, 43, 46), (34, 255, 255))],
            'green': [((35, 43, 46), (77, 255, 255))],
            'cyan':[((78, 43, 46), (99, 255, 255))],
            'blue': [((100, 43, 46), (124, 255, 255))],
            'purple': [((125, 43, 46), (155, 255, 255))],
        }

    def __set_dcv(self):
        self.__Default_color_visuals = {
                'black': (0, 0, 0),
                'gray': (128, 128, 128),
                'white': (255, 255, 255),
                'red': (0, 0, 255),
                'orange': (0, 165, 255),
                'yellow': (0, 255, 255),
                'green': (0, 255, 0),
                'cyan': (255, 255, 0),
                'blue': (255, 0, 0),
                'purple': (128, 0, 128),
            }

    def add_color(self, color_name:str, lower_hsv: Tuple[int, int, int], upper_hsv: Tuple[int, int, int]):
        """添加自定义颜色范围"""
        if color_name not in self.Color_ranges:
            self.Color_ranges[color_name] = []
            self.Color_visuals[color_name] = []
        self.Color_ranges[color_name].append((lower_hsv, upper_hsv))
        temp_hsv = np.array([[[
            (upper_hsv[0]-lower_hsv[0])*0.5,
            (upper_hsv[1]-lower_hsv[1])*0.5,
            (upper_hsv[2]-lower_hsv[2])*0.5
        ]]], dtype=np.uint8)
        r, g ,b = cv2.cvtColor(temp_hsv, cv2.COLOR_HSV2RGB)[0][0]
        self.Color_visuals[color_name].append((r, g, b))

    def remove_color(self, color_name):
        """移除指定颜色范围"""
        if color_name in self.Color_ranges:
            del self.Color_ranges[color_name]
            del self.Color_visuals[color_name]

    def detect_colors(self, image=None, image_path=None):
        """
        检测图片中的颜色分布

        参数:
            color_ranges: 如果不使用自定义色域，则用默认色域
            image_path: 图片路径，如果为None，则使用image参数
            image: 已加载的图片对象，如果为None，则使用image_path加载图片

        返回:
            包含各种颜色检测结果的字典
        """
        # 色域选择
        if len(self.Color_ranges) > 0:
            color_ranges = self.Color_ranges
        else:
            color_ranges = self.__Default_color_ranges

        # 加载图片
        if image is None:
            if image_path is None:
                raise ValueError("必须提供image_path或image参数")
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"无法加载图片: {image_path}")
        else:
            img = image.copy()

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 存储检测结果
        result = {}

        for color_name, ranges in color_ranges.items():
            # 初始化掩码
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            # 处理多范围颜色（如红色）
            for (lower, upper) in ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                partial_mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.bitwise_or(mask, partial_mask)

            # 计算颜色区域面积
            area = cv2.countNonZero(mask)
            result[color_name] = {
                'mask': mask,
                'area': area,
                'percentage': area / (img.shape[0] * img.shape[1]) * 100
            }
        self.Color_results = result

        return result

    def get_dominant_colors(self, top_n=3, threshold=1):
        """
        获取图片中占比最大的几种颜色
        参数:
            top_n: 返回的主要颜色数量
            threshold: 最小百分比阈值

        返回:
            包含主要颜色及其百分比的列表
        """
        if len(self.Color_results) > 0:
            detection_result = self.Color_results
        else:
            raise ValueError("应当使用detect_colors先对图片进行颜色检测")
        # 过滤并排序颜色
        filtered_colors = [
            (color, data['percentage'])
            for color, data in detection_result.items()
            if data['percentage'] >= threshold
        ]
        filtered_colors.sort(key=lambda x: x[1], reverse=True)

        # 返回前top_n种颜色
        return filtered_colors[:top_n]

    def visualize_results(self, image, detection_result, threshold=1.0, color_visuals=None):
        """
        可视化颜色检测结果

        参数:
            image: 原始图片
            detection_result: detect_colors方法返回的结果
            threshold: 显示颜色的最小百分比阈值
            color_visuals: hsv->rgb的映射，如果是None的话则使用默认映射组

        返回:
            可视化结果图片
        """
        if color_visuals is None:
            color_visuals = self.__Default_color_visuals

        result_img = image.copy()

        # 按面积降序排序
        sorted_colors = sorted(
            detection_result.items(),
            key=lambda x: x[1]['area'],
            reverse=True
        )

        # 在原图上绘制颜色区域
        for color_name, data in sorted_colors:
            if data['percentage'] >= threshold:
                # 创建彩色掩码
                color_mask = np.zeros_like(result_img)
                color_mask[:, :] = color_visuals[color_name]
                color_mask = cv2.bitwise_and(color_mask, color_mask, mask=data['mask'])

                # 叠加到原图
                result_img = cv2.addWeighted(result_img, 0.7, color_mask, 0.3, 0)

        # 添加文本标签
        y_offset = 30
        for color_name, data in sorted_colors:
            if data['percentage'] >= threshold:
                text = f"{color_name}: {data['percentage']:.1f}%"
                cv2.putText(
                    result_img,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color_visuals[color_name],
                    2
                )
                y_offset += 30

        return result_img

    def default_color_ranges(self):
        return self.__Default_color_ranges

    def default_color_visuals(self):
        return self.__Default_color_visuals


# 使用示例
if __name__ == "__main__":
    # 创建颜色检测器实例
    detector = ColorDetector()
    path = r"D:\pyproject\Auto_py2cst_test\test\test21.png"
    # detector.add_color('c', (0,0,0), (5,5,5))
    # detector.add_color('c', (0, 0, 0), (55, 65, 56))
    # print(detector.Color_visuals)
    # print(detector.Color_ranges)

    # 检测图片中的颜色
    Results = detector.detect_colors(image_path=path)
    for result in Results:
        if Results[result]['percentage'] > 1e-2:
            cv2.imshow(f'{result}', Results[result]['mask'])
    cv2.waitKey(0)

    # 获取主要颜色
    dominant_colors = detector.get_dominant_colors(top_n=3)
    print("主要颜色:")
    for Color, percentage in dominant_colors:
        print(f"- {Color}: {percentage:.1f}%")

    # 可视化结果
    Img = cv2.imread(path)
    if Img is not None:
        visualized = detector.visualize_results(image=Img, detection_result=Results)
        cv2.imshow("Color Detection", visualized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
