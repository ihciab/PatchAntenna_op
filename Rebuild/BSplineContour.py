"""
功能类
对轮廓进行B样条拟合，类要保存拟合轮廓的点和轮廓之前的嵌套关系
"""

from Rebuild.ImageInit import *
import math
from scipy.interpolate import make_interp_spline

class BSplineContour:
    def __init__(self, img, edges, lr=0.1, delta = 2, step = 30, threshold:float=1e-1, show=True, save=''):
        self.__default_lr = None
        self.__default_delta = None
        self.__default_step = None
        self.__default_threshold = None
        self.__init_paras(img, edges, lr, delta, step, threshold, save)
        self.__init_process(show)

    def __init_paras(self, img, edges, lr, delta, step, threshold:float, save):
        self.__img = img
        self.__edges = edges
        self.__default_lr = lr
        self.__default_delta = delta
        self.__default_step = step
        self.__default_threshold = threshold
        self.__Save = save
        self.__contours_dict = {}
        """
        {
            1（编号）：
                {
                轮廓：
                控制点：
                    {点数：}，
                    {点：},
                拟合点：
                    {点数：}，
                    {点：},
                loss: {
                    {mse_loss: },
                    {step: },
                    }    
                } 
            ...
        }
        """
        self.__contours, self.__hierarchy = cv2.findContours(self.__edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    def derived_paras(self, img, edges, lr, delta, step, threshold, save):
        self.__init_paras(img, edges, lr, delta, step, threshold, save)

    def __init_process(self, show):
        self.__bspline_process(show)
        self.__curves_tree = self.__node_tree()


    def __node_tree(self):
        tree = {}
        if self.__hierarchy is None:
            raise ValueError("未检测到轮廓，请检查图片或调整阈值参数")
        h = np.copy(self.__hierarchy[0])
        for i in range(len(h)):
            if i < 2:
                if i not in tree:
                    tree[f'{i}'] = []
                j = i
                while h[j][3] > -1:
                    if h[j][3] % 2 != 0:
                        tree[f'{h[j][3]}'].append(f'{i}')
                        break
                    else:
                        j = h[j][3]
        return tree

    def use_node_tree(self):
        self.__node_tree()

    def __write_dict(self, label,
                     contour,
                     len_control,
                     control_points,
                     len_contour,
                     new_contour,
                     mse_loss,
                     step):
        if label not in self.__contours_dict:
            self.__contours_dict[f'{label}'] = \
                {
                    'contour': None,
                    'control': {
                        'size': None,
                        'points': None,
                    },
                    'fitting': {
                        'size': None,
                        'points': None,
                    },
                    'loss': {
                        'mse_loss': None,
                        'step': None,
                    }
                }
            self.__contours_dict[f'{label}'] = \
                {
                    'contour': contour,
                    'control': {
                        'size': len_control,
                        'points': control_points,
                    },
                    'fitting': {
                        'size': len_contour,
                        'points': new_contour,
                    },
                    'loss': {
                        'mse_loss': mse_loss,
                        'step': step,
                    }
                }
        else:
            self.__contours_dict[f'{label}'] = \
                {
                    'contour': contour,
                    'control': {
                        'size': len_control,
                        'points': control_points,
                    },
                    'fitting': {
                        'size': len_contour,
                        'points': new_contour,
                    },
                    'loss': {
                        'mse_loss': mse_loss,
                        'step': step,
                    }
                }

    def use_write_dict(self, label,
                     contour,
                     len_control,
                     control_points,
                     len_contour,
                     new_contour,
                     mse_loss,
                     step):
        self.__write_dict(label,
                          contour.copy(),
                          len_control,
                          control_points,
                          len_contour,
                          new_contour,
                          mse_loss,
                          step)

    @staticmethod
    def isqrt(x):
        # 输入验证：必须为非负整数
        if not isinstance(x, int):
            raise TypeError(f"'{type(x).__name__}' object cannot be interpreted as an integer")
        if x < 0:
            raise ValueError("isqrt() argument must be a non-negative integer")

        # 边界情况：x=0时直接返回0
        if x == 0:
            return 0

        # 牛顿迭代法计算整数平方根
        n = x  # 初始猜测值
        while True:
            # 迭代公式：m = (当前猜测值 + x//当前猜测值) // 2
            m = (n + x // n) // 2
            if m >= n:  # 收敛条件：迭代值不再减小，此时n即为结果
                return n
            n = m


    def __find_closest_factors(self, n):
        """
        找到正整数 n 的一组最相近因数（a ≤ b，a×b = n）
        :param n: 待分解的正整数（n ≥ 1）
        :return: 最相近的因数对 (a, b)，若 n=1 则返回 (1, 1)
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError("输入必须是正整数")

        # 从平方根向下遍历，找到第一个能整除 n 的数
        sqrt_n = int(self.isqrt(n))
        for a in range(sqrt_n, 0, -1):
            if n % a == 0:  # a 是因数
                b = n // a  # 对应另一个因数
                c = 0 #是否有余
                if b == n:
                    a = sqrt_n
                    b = n // a
                    c = 1
                return a, b, c

        # 只有 n=1 会走到这里（1的因数只有1）
        return 1, 1

    def __result_visual(self, k):
        # 分配画布
        size = len(self.__contours) / 2 + 1
        a, b, c = self.__find_closest_factors(int(size))
        # plt.figure(figsize=(min(a, b) + c, max(a, b)))
        plt.figure()

        # 子图1：原始图片
        img_with_contour = self.__img.copy()
        img_rgb = cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB)
        plt.subplot(min(a, b) + c, max(a, b), 1)
        plt.imshow(img_rgb)
        plt.title("原始图片")
        plt.axis("off")

        # 子图2~n：B样条拟合结果
        for i, contour in enumerate(self.__contours_dict):
            plt.subplot(min(a, b) + c, max(a, b), i + 2)
            # 控制点
            plt.scatter(*self.__contours_dict[contour]['control']['points'].T, marker='D', color="red", s=10, label="轮廓控制点")

            # 拟合点
            plt.plot(*self.__contours_dict[contour]['fitting']['points'].T, ls='--', color="blue", linewidth=2,
                     label=f"B样条拟合 (k={k})")


            # 原始边框
            contour_points = self.__contours[int(contour)].copy().reshape(-1, 2)  # 转换形状为(N, 2)

            # 提取x和y坐标
            x = contour_points[:, 0]
            y = contour_points[:, 1]
            plt.scatter(x, y, marker='o', color="green", s=1, label="原始轮廓点")

            plt.legend()
            plt.axis("equal")  # 保持坐标比例一致
            mse_loss = self.__contours_dict[contour]['loss']['mse_loss']
            step = self.__contours_dict[contour]['loss']['step']
            plt.title(f'mse_loss:{mse_loss:.2f};step:{step}')
        plt.tight_layout()
        if self.__Save:
            plt.savefig(self.__Save + '\BSpline.png', dpi=300, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(3)  # 显示5秒
        plt.close()

    def use_result_visual(self, k):
        self.__result_visual(k)


    def __bspline_process(self, show=True):
        len_control = None
        control_points = None
        new_contour = None
        mse_loss = None
        k = None
        if not self.__contours:
            raise ValueError("未检测到轮廓，请检查图片或调整阈值参数")
        for i, contour in enumerate(self.__contours):
            # 获取一个轮廓
            if i < 2:
                # 转换形状为(N, 2)
                contour_points = contour.reshape(-1, 2).copy()

                # 提取边框点的个数
                len_contour = len(contour_points)

                # 确定样条阶数k（确保k < 数据点数量）
                k = 3 if len_contour > 3 else len_contour - 1

                # 设置控制点
                step = self.__default_step
                while step>1:
                    # print('##########################')
                    # print(f'step:{step}')
                    if len_contour % step == 0:
                        control_points = contour_points.copy()[::int(step)]
                        control_points = np.concatenate([control_points, contour_points[0].reshape(-1, 2)])
                        len_control = len(control_points)
                    else:
                        control_points = contour_points.copy()[::int(step)]
                        control_points[-1] = contour_points[0]
                        len_control = len(control_points)
                    # print(f'len_control:{len_control}')
                    # print(f'len_contour:{len_contour}')

                    # 生成B样条插值器（闭合轮廓可考虑使用周期性边界条件）
                    phi = np.linspace(0, 2. * np.pi, len_control)
                    # spl = make_interp_spline(phi, control_points, k=k, bc_type="periodic")
                    spl = make_interp_spline(phi, control_points, k=k)

                    # 生成拟合曲线的密集点
                    phi1 = np.linspace(0, 2. * np.pi, len_contour)
                    # x_smooth, y_smooth = spl(phi1).T
                    # new_contour = np.array([x_smooth, y_smooth]).T
                    new_contour = spl(phi1)
                    mse_loss = np.mean((new_contour - contour_points) ** 2)
                    # print(f'mse_loss:{mse_loss}')
                    if mse_loss < self.__default_threshold or step == 1:
                        phi2 = np.linspace(0, 2. * np.pi, len_control)
                        # print(len_contour)
                        # x_smooth, y_smooth = spl(phi1).T
                        # new_contour = np.array([x_smooth, y_smooth]).T
                        new_contour = spl(phi2)
                        break
                    else:
                        d_step = self.__default_lr * mse_loss / self.__default_delta
                        # print(f'd_step:{d_step}')
                        step = 1 if step < 1 else int(step - d_step)



                    # 填写轮廓信息
                self.__write_dict(f'{i}',
                                  contour.copy(),
                                  len_control,
                                  control_points,
                                  len_contour,
                                  new_contour,
                                  mse_loss,
                                  step)

        if show:
            self.__result_visual(k)

    def img(self):
        return self.__img

    def edges(self):
        return self.__edges

    def hyperparameter(self, lr=None, delta=None, step=None, threshold=None, auto_update=False):
        if lr is not None:
            self.__default_lr = lr
        if delta is not None:
            self.__default_delta = delta
        if step is not None:
            self.__default_step = step
        if threshold is not None:
            self.__default_threshold = threshold
        if auto_update:
            if any(x is not None for x in (lr, delta, step, threshold)):
                print("正在根据新的超参自动修正")
                self.__bspline_process(show=False)
        return self.__default_lr, self.__default_delta, self.__default_step, self.__default_threshold

    def print_hyperparameter(self):
        print(
            f'自适应B样条拟合超参-学习率:{self.__default_lr}, 步长补偿值：{self.__default_delta}, 初始步长：{self.__default_step}, 误差阈值：{self.__default_threshold}')

    def get_contours(self):
        return self.__contours

    def get_hierarchy(self):
        return self.__hierarchy

    def get_contours_dict(self):
        return self.__contours_dict

    def get_curves_tree(self):
        return self.__curves_tree

    def get_default_step(self):
        return self.__default_step

    def get_default_threshold(self):
        return self.__default_threshold

    def get_default_lr(self):
        return self.__default_lr

    def get_default_delta(self):
        return self.__default_delta

    def get_edges(self):
        return self.__edges




if __name__ == "__main__":
    II = ImageInit(r"D:\LineFormer-main\019\000\repair_fig.png")
    Img = II.centered_img()
    Edges = II.edges()
    BC = BSplineContour(Img, Edges)
    BC.print_hyperparameter()



