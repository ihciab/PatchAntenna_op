import pandas as pd
import numpy as np

class IntersectionDetection:
    # 暴力求解
    @staticmethod
    def __cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    @staticmethod
    def __on_segment(a, b, p):
        return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


    @staticmethod
    def __quick_reject(a, b, c, d):
        x1_min, x1_max = min(a[0], b[0]), max(a[0], b[0])
        y1_min, y1_max = min(a[1], b[1]), max(a[1], b[1])
        x2_min, x2_max = min(c[0], d[0]), max(c[0], d[0])
        y2_min, y2_max = min(c[1], d[1]), max(c[1], d[1])
        return x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min


    def __segment_intersect(self, a, b, c, d):
        if self.__quick_reject(a, b, c, d):
            return False
        d1 = self.__cross(a, b, c)
        d2 = self.__cross(a, b, d)
        d3 = self.__cross(c, d, a)
        d4 = self.__cross(c, d, b)
        if (d1 * d2 < 0) and (d3 * d4 < 0):
            return True
        if d1 == 0 and self.__on_segment(a, b, c):
            return True
        if d2 == 0 and self.__on_segment(a, b, d):
            return True
        if d3 == 0 and self.__on_segment(c, d, a):
            return True
        if d4 == 0 and self.__on_segment(c, d, b):
            return True
        return False


    def find_intersecting_pairs(self, contour)->bool:
        """找出n个线段中所有相交的线段对（返回索引对）"""
        segments = []
        for i in range(len(contour) - 1):
            segments.append([contour[i], contour[i + 1]])
        n = len(segments)
        intersecting_pairs = []
        # 遍历所有i < j的线段对，避免重复检测
        for i in range(n):
            a, b = segments[i]  # 线段i的两个端点
            for j in range(i + 1, n):
                c, d = segments[j]  # 线段j的两个端点
                if self.__segment_intersect(a, b, c, d):
                    intersecting_pairs.append((i, j))
        # print(len(intersecting_pairs))
        # print(len(segments))
        return len(intersecting_pairs) == len(segments)


# 测试示例
if __name__ == "__main__":
    size = 20
    d = 20
    data = pd.read_csv(r'5.csv').values
    data_test = data[::size]

    d_points = int(len(data_test) * 0.8)
    x1 = np.random.choice(len(data_test), size=d_points, replace=False, p=None)
    y1 = np.random.randint(d, size=(d_points, 2))

    data_test[x1] = data_test[x1] + y1
    data_test = np.concatenate([data_test, data_test[0].reshape(-1, 2)])

    # 找出所有相交的线段对
    ID = IntersectionDetection()
    pairs = ID.find_intersecting_pairs(data_test)
    # print(len(data_test))
    # print(len(pairs))
    # print("相交的线段对（索引）：", pairs)
    # 预期输出：[(0,1), (0,3), (0,4), (2,3)]