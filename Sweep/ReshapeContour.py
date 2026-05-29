import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimSun", "Microsoft YaHei"]

class ReshapeContour:
    def __init__(self, contour:np.ndarray, step:int, scale_num:float=0.8, scale_d=0.05):
        self.__Mode: bool = (len(contour) % step == 0)
        self.__Contour = contour.copy()[::step]
        # print(len(self.__Contour))
        self.__Scale_num = scale_num
        self.__Scale_d = scale_d
        self.__Magnitude_x = max(contour[:, 0]) - min(contour[:, 0])
        self.__Magnitude_y = max(contour[:, 1]) - min(contour[:, 1])
        self.__D_x = int(self.__Magnitude_x * self.__Scale_d)
        self.__D_y = int(self.__Magnitude_y * self.__Scale_d)
        self.__New_contour = None

    def __visual_result(self, show_coordinates=False):
        x = self.__New_contour[:, 0]
        y = self.__New_contour[:, 1]
        # 创建图形和坐标轴
        plt.figure(figsize=(8, 6))

        num_points = len(x)
        # 按顺序用箭头连接点
        for i in range(num_points - 1):
            # 从点i到点i+1绘制箭头
            plt.arrow(float(x[i]), float(y[i]), float(x[i + 1] - x[i]), float(y[i + 1] - y[i]),
                      head_width=5, head_length=3,
                      fc='blue', ec='blue', linewidth=2,
                      length_includes_head=True, zorder=2)

        # 绘制点（放在箭头上层）
        plt.scatter(x, y, color='red', s=100, zorder=3)

        # 如果需要，显示每个点的坐标和序号
        if show_coordinates:
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.text(xi + 0.1, yi + 0.1, f'P{i}: ({xi:.1f}, {yi:.1f})', fontsize=9)

        # 添加标题和标签
        plt.title(f'随机生成的{num_points}个点及顺序连接（箭头指示方向）', fontsize=14)
        plt.xlabel('X坐标', fontsize=12)
        plt.ylabel('Y坐标', fontsize=12)

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7, zorder=1)

        # 调整布局并显示图形
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)  # 显示5秒
        plt.close()

    def reshape_contour(self, show=False, show_coordinates=False):
        # 生成随机点 (x, y)，范围在0到10之间
        target_num = int(len(self.__Contour) * self.__Scale_num)
        # print(len(self.__Contour))
        # print(target_num)
        target_index = np.random.choice(len(self.__Contour), size=target_num, replace=False, p=None)
        move_x = np.random.randint(-int(self.__D_x / 2), int(self.__D_x / 2), size=(target_num, 1))
        move_y = np.random.randint(-int(self.__D_y / 2), int(self.__D_y / 2), size=(target_num, 1))
        move = np.concatenate((move_x, move_y), axis=1)
        self.__Contour[target_index] = self.__Contour[target_index] + move
        new_contour = self.__Contour
        if self.__Mode:
            new_contour = np.concatenate([new_contour, new_contour[0].reshape(-1, 2)])
        else:
            new_contour[-1] = new_contour[0]
        self.__New_contour =  new_contour
        if show:
            self.__visual_result(show_coordinates)
        return new_contour


if __name__ == "__main__":
    sp = 20
    d = 20
    data = pd.read_csv(r'5.csv').values

    # 生成并显示10个点
    # np.random.seed(42)
    RC = ReshapeContour(data, sp)

    RC.reshape_contour(show=True)
