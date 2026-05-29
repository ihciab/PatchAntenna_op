"""
材料映射模块
原代码位置：Rebuild/Curves2Component.py

功能：
- 将轮廓曲线转换为solid组件
- 根据颜色检测确定solid的材料类型
- 基于轮廓树和颜色检测结果建立solid-颜色映射

重构说明：
- 将原 Curves2Components 类重构为 MaterialMapper 类
- 保持原有算法和逻辑完全不变
- 更新导入路径以适配新结构
"""
import numpy as np
import cv2
from core.material.color_detector import ColorDetector


class MaterialMapper(ColorDetector):
    """
    材料映射类
    原类：Rebuild/Curves2Component.py 中的 Curves2Components
    
    用于将轮廓曲线转换为solid组件，并根据颜色确定材料类型
    """
    def __init__(self, original_img, img_shape, con_dict, cur_tree, color_ranges):
        # 原代码：super(Curves2Components, self).__init__()
        super(MaterialMapper, self).__init__()
        self.Original_img = original_img
        self.__Img_shape = img_shape # 当前图片的尺寸
        self.__Con_Dict = con_dict # 轮廓字典，详见BSplineContour类
        self.__Cur_Tree = cur_tree # 曲线树，描述轮廓的嵌套关系，详见BSplineContour类
        self.__Solid_mask = {} #  保存solid的命名和对应颜色
        self.__Color_ranges = color_ranges
        self.__basic_process()
        self.__curve2solids(show=False)

    def __basic_process(self):
        if self.__Color_ranges is not None:
            for color_name, ranges in self.__Color_ranges.items():
                # 处理多范围颜色（如红色）
                for (lower, upper) in ranges:
                    self.add_color(color_name, lower, upper)
        else:
            pass

    def __curve2solids(self, show=False):
        for super_curve in self.__Cur_Tree:
            # 获取具有父子关系的轮廓list
            temp = super_curve
            mask_img = np.zeros(self.__Img_shape, dtype=np.uint8)
            curve_list = [self.__Con_Dict[super_curve]['contour']]
            for curve in self.__Cur_Tree[super_curve]:
                temp += ('-' + curve)
                curve_list.append(self.__Con_Dict[curve]['contour'])

            # 根据轮廓list创建轮廓掩码
            cv2.drawContours(mask_img, curve_list, -1, [1, 1, 1], -1)

            #对掩码下的原图进行颜色检测，并获取占比最高的两种颜色，通常是目标颜色和掩码（黑色）
            self.detect_colors(self.Original_img * mask_img)
            dominant_colors = self.get_dominant_colors(top_n=2)
            for Color, percentage in dominant_colors:
                if Color != 'black':
                    self.__Solid_mask[temp] = Color
                    break
        # print(self.__Solid_mask)
        #
            if show:
                cv2.imshow('1', self.Original_img * mask_img)
                cv2.waitKey(0)

    def solids_col(self):
        return self.__Solid_mask



# 向后兼容的类名（保持原有接口）
Curves2Components = MaterialMapper


if __name__ == '__main__':
    from core.geometry.bspline_fitter import BSplineFitter
    from core.material.color_detector import ColorDetector
    from core.image.initializer import ImageInitializer

    image_path = r"D:\pyproject\Auto_py2cst_test\test\test25.png"

    # 初始化图片
    Init_img = ImageInitializer(image_path)
    img = Init_img.centered_img()
    edge = Init_img.edges()

    # 对轮廓进行B样条拟合
    BC = BSplineFitter(img, edge, show=False)
    contours_dict = BC.get_contours_dict()
    curves_tree = BC.get_curves_tree()

    # 对原始图片进行颜色检测
    CD = ColorDetector()
    color_results = CD.detect_colors(image=img)

    # 根据轮廓拟合曲线的父子类关系确定solid的材料
    C2C = MaterialMapper(img, img.shape, contours_dict, curves_tree, None)


