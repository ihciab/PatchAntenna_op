import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei"]

class ImageInit:
    def __init__(self, path, show=True, save=''):
        self.__path = path
        self.__Save = save
        self.__original_img = None
        self.__main_contour = None
        self.__centered_img = None
        self.__gray_img = None
        self.__blurred_img = None
        self.__edges_img =None
        self.__Img_center = None
        self.__img_process(show=show)

    def trans2center(self, thresh_image):
        """
        检测图像主体中心并通过仿射变换将其平移至图像中心
        :param thresh_image: 预处理后的二值化图像（用于轮廓检测）
        :return: 主体居中后的图像
        """
        # 1. 检测图像主体（最大轮廓）并计算其中心坐标
        # 查找所有外部轮廓（减少内部轮廓干扰）
        contours, _ = cv2.findContours(
            thresh_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise ValueError("未检测到图像主体轮廓，请检查图像质量或调整预处理参数")

        # 选择面积最大的轮廓作为主体
        main_contour = max(contours, key=cv2.contourArea)
        self.__main_contour = main_contour
        # 计算主体最小外接矩形的中心（更贴合主体形状）
        rect = cv2.minAreaRect(main_contour)
        (x_center, y_center), _, _ = rect  # 提取主体中心坐标
        main_center = (int(x_center), int(y_center))

        # 2. 通过仿射变换将主体中心平移至图像中心
        h, w = thresh_image.shape[:2]
        img_center = (w // 2, h // 2)  # 图像中心坐标
        self.__Img_center = img_center

        # 计算平移向量（图像中心 - 主体中心 = 需要移动的距离）
        dx = img_center[0] - main_center[0]
        dy = img_center[1] - main_center[1]

        # 生成仅包含平移的仿射变换矩阵
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        # 执行仿射变换，保持图像尺寸不变，边缘用边缘像素填充（避免黑边）
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
        plt.pause(3)  # 显示5秒
        plt.close()

    def __img_process(self, show=False):
        # 读取图片
        image = cv2.imread(self.__path)
        if image is None:
            raise ValueError(f"无法读取图片: {self.__path}，请检查路径是否正确")
        self.__original_img = image

        # 转为灰度图并将图片归一化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.__gray_img = gray

        # 高斯滤波，滤除噪点
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        # blurred = cv2.dilate(blurred, kernel, iterations=1)
        # blurred = cv2.erode(blurred, kernel, iterations=4)
        self.__blurred_img = blurred

        # 利用Canny方法边缘提取
        edges = cv2.Canny(blurred, 2500, 5000, apertureSize=5)
        # edges = cv2.Canny(blurred, 100, 2000, apertureSize=5)

        # cv2.imshow('1', edges)
        # cv2.waitKey(0)

        # 设置开闭核对边缘轮廓进行开闭运算，合并细小边框
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        # 将图片移到中心
        self.__edges_img, self.__centered_img = self.trans2center(edges)

        if show:
            self.__result_vision()

    def original_img(self):
        return self.__original_img

    def centered_img(self):
        return self.__centered_img

    def gray(self):
        return self.__gray_img

    def blurred(self):
        return self.__blurred_img

    def edges(self):
        return self.__edges_img

    def center_point(self):
        return self.__Img_center




if __name__ == '__main__':
    II = ImageInit(r"D:\CST2023proj\autocst\testFss0\0\fss_clear\repair_fig.png")
