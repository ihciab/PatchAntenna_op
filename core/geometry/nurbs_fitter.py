"""
NURBS轮廓拟合模块
原代码位置：Rebuild/NURBSpline.py

功能：
- 使用NURBS（非均匀有理B样条）进行轮廓拟合
- 支持自定义阶数和权重
- 继承自BSplineFitter，复用轮廓检测和树结构功能

重构说明：
- 将原 NURBSpineContour 类重构为 NURBSFitter 类
- 保持原有算法和逻辑完全不变
- 更新导入路径以适配新结构
"""
import geomdl.knotvector
import numpy as np
from geomdl import NURBS
from geomdl import utilities
from core.geometry.bspline_fitter import BSplineFitter


class NURBSFitter(BSplineFitter):
    """
    NURBS轮廓拟合类
    原类：Rebuild/NURBSpline.py 中的 NURBSpineContour
    
    用于使用NURBS（非均匀有理B样条）进行轮廓拟合
    """
    def __init__(self, img, edges, lr=0.1, delta=2, step=30, threshold: float = 10.0, degree:int=3, show=True, save=''):
        self.derived_paras(img, edges, lr, delta, step, threshold, save)
        self.__degree = degree
        self.__bspline_process(show)
        # 设置轮廓树（使用父类的私有方法）
        # 原代码：self._BSplineContour__curves_tree = self._BSplineContour__node_tree()
        self._BSplineFitter__curves_tree = self._BSplineFitter__node_tree()

    def __generate_knots(self, len_ctrl, start=0, stop=2 * np.pi)->list:
        knots = [
            *[start for _ in range(self.__degree)],
            *np.linspace(start, stop, len_ctrl - self.__degree + 1),
            *[stop for _ in range(self.__degree)]
        ]
        return knots

    def __generate_weight(self, size, generate_type:str='DEFAULT'):
        if generate_type == 'DEFAULT':
            temp_weight = np.ones(size) + 1e-9
            return temp_weight

        if generate_type == 'SOFTMAX':
            temp_weight = np.zeros(size) + 1e-9
            return np.exp(temp_weight) / np.sum(np.exp(temp_weight))

        if generate_type == 'backward':
            pass

    def __bspline_process(self, show=True):
        len_control = None
        control_points = None
        new_contour = None
        mse_loss = None
        if not self.get_contours():
            raise ValueError("未检测到轮廓，请检查图片或调整阈值参数")
        for i, contour in enumerate(self.get_contours()):
            # 获取一个轮廓
            if i % 2 == 1:
                # 转换形状为(N, 2)
                contour_points = contour.reshape(-1, 2).copy()

                # 提取边框点的个数
                len_contour = len(contour_points)

                # 设置控制点
                step = self.get_default_step()
                while step > 1:
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

                    # 检查控制点数量是否足够（至少需要degree+1个控制点）
                    min_control_points = self.__degree + 1
                    if len_control < min_control_points:
                        # 如果控制点不足，尝试减小step或跳过该轮廓
                        if step > 1:
                            step = max(1, step - 1)
                            continue
                        else:
                            # step已经是1，控制点仍然不足，跳过该轮廓
                            print(f"警告: 轮廓 {i} 的控制点数量不足（{len_control} < {min_control_points}），跳过NURBS拟合")
                            break

                    # 一下类参数的设置顺序不要乱
                    # 生成NURBS类(开启采样点归一化)
                    curve = NURBS.Curve(normalize_kv=True)
                    # 设置样条函数的阶数（确保不超过控制点数量限制）
                    curve.degree = min(self.__degree, len_control - 1)
                    # 设置拟合控制点
                    curve.ctrlpts = control_points
                    # 生成并设置节点向量，检测自定义是否符合节点规则（len(knots) = len(ctrl) + 1 + degree），否则生成默认节点
                    knots = self.__generate_knots(len_control)
                    # curve.knotvector = utilities.generate_knot_vector(curve_degree, ctrl_pt_count)
                    if geomdl.knotvector.check(self.__degree, knots, len_control):
                        # print('1')
                        curve.knotvector = knots
                    else:
                        # print('default')
                        curve.knotvector = utilities.generate_knot_vector(self.__degree, len_control)

                    # 设置控制点拟合权重
                    curve.weights = self.__generate_weight(len_control)
                    # 设置采样点用于计算和原始曲线的误差
                    curve.sample_size = len_contour
                    # 去除求解所得的新轮廓
                    new_contour = np.asarray(curve.evalpts)
                    # 计算MSE误差
                    mse_loss = np.mean((new_contour - contour_points) ** 2)
                    d_dist = (new_contour - contour_points) ** 2 @ np.ones((2, 1)).reshape(-1) * 100
                    # 达到误差要求或者步长为1则返回新的轮廓
                    if mse_loss < self.get_default_threshold() or step == 1:
                        curve.sample_size = len_contour
                        new_contour = np.asarray(curve.evalpts)
                        self.use_write_dict(f'{i}',
                                            contour.copy(),
                                            len_control,
                                            control_points,
                                            len_contour,
                                            new_contour,
                                            mse_loss,
                                            step)
                        break
                    else:
                        d_step = self.get_default_lr() * mse_loss / self.get_default_delta()
                        # print(f'd_step:{d_step}')
                        step = 1 if step < 1 else int(step - d_step)

                    # 填写轮廓信息


        if show:
            self.use_result_visual(self.__degree)


# 向后兼容的类名（保持原有接口）
NURBSpineContour = NURBSFitter


if __name__ == '__main__':
    from core.image.initializer import ImageInitializer
    II = ImageInitializer(r"D:\datasheet\test\test21.png")
    Img = II.centered_img()
    Edges = II.edges()
    NBS = NURBSFitter(img=Img, edges=Edges)
