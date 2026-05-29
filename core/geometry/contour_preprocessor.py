"""
轮廓预处理器
统一处理所有轮廓，确保轮廓质量
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List


class ContourPreprocessor:
    """
    轮廓预处理器
    统一处理所有轮廓，确保轮廓质量
    """
    
    @staticmethod
    def preprocess_contour(contour: np.ndarray, 
                          min_points: int = 2,
                          remove_duplicates: bool = True,
                          interpolate: bool = True) -> Optional[np.ndarray]:
        """
        预处理轮廓
        - 去重
        - 插值
        - 验证
        
        参数:
            contour: 轮廓点数组
            min_points: 最小点数
            remove_duplicates: 是否去重
            interpolate: 是否插值
        
        返回:
            处理后的轮廓点数组，如果无效则返回None
        """
        if contour is None or len(contour) == 0:
            return None
        
        # 转换为2D数组
        try:
            points = np.asarray(contour, dtype=float)
            # OpenCV轮廓格式可能是(N, 1, 2)或(N, 2)
            if len(points.shape) == 3:
                # (N, 1, 2) -> (N, 2)
                if points.shape[1] == 1 and points.shape[2] == 2:
                    points = points.reshape(-1, 2)
                else:
                    return None
            elif len(points.shape) == 2:
                if points.shape[1] == 2:
                    # (N, 2) - 正确格式
                    pass
                elif points.shape[0] == 2:
                    # (2, N) -> (N, 2)
                    points = points.T
                else:
                    return None
            elif len(points.shape) == 1:
                if points.size == 2:
                    points = points.reshape(1, 2)
                else:
                    return None
            else:
                return None
        except Exception as e:
            return None
        
        if len(points) < 1:
            return None
        
        # 1. 去重
        if remove_duplicates and len(points) > 1:
            points = ContourPreprocessor._remove_duplicates(points)
        
        # 2. 插值（如果点数不足）
        if interpolate and len(points) < min_points:
            points = ContourPreprocessor._interpolate_points(points, min_points)
        
        # 3. 验证
        if len(points) < min_points:
            return None
        
        return points
    
    @staticmethod
    def _remove_duplicates(points: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
        """
        移除重复点
        
        参数:
            points: 轮廓点数组
            threshold: 距离阈值
        
        返回:
            去重后的点数组
        """
        if len(points) < 2:
            return points
        
        keep_mask = np.ones(len(points), dtype=bool)
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            if dist < threshold:
                keep_mask[i] = False
        
        result = points[keep_mask]
        
        # 如果去重后为空，至少保留第一个点
        if len(result) == 0:
            result = points[:1]
        
        return result
    
    @staticmethod
    def _interpolate_points(points: np.ndarray, min_points: int = 3) -> np.ndarray:
        """
        在点之间插值，确保至少有min_points个点
        
        参数:
            points: 轮廓点数组
            min_points: 最小点数
        
        返回:
            插值后的点数组
        """
        if len(points) >= min_points:
            return points
        
        if len(points) == 0:
            return points
        
        if len(points) == 1:
            # 单个点，无法插值
            return points
        
        # 两点之间插值
        if len(points) == 2:
            p0, p1 = points[0], points[1]
            if min_points == 3:
                mid = (p0 + p1) / 2.0
                return np.array([p0, mid, p1])
            else:
                # 插值更多点
                t_values = np.linspace(0, 1, min_points)
                interpolated = np.array([p0 + t * (p1 - p0) for t in t_values])
                return interpolated
        
        # 多个点，在相邻点之间插值
        result = [points[0]]
        for i in range(len(points) - 1):
            p0, p1 = points[i], points[i + 1]
            # 在两点之间插入一个中点
            mid = (p0 + p1) / 2.0
            result.append(mid)
            result.append(p1)
        
        result = np.array(result)
        
        # 如果还不够，继续插值
        while len(result) < min_points:
            new_result = [result[0]]
            for i in range(len(result) - 1):
                p0, p1 = result[i], result[i + 1]
                mid = (p0 + p1) / 2.0
                new_result.append(mid)
                new_result.append(p1)
            result = np.array(new_result)
            if len(result) >= min_points:
                break
        
        return result[:min_points] if len(result) > min_points else result
    
    @staticmethod
    def filter_contours_by_features(contours: List[np.ndarray], 
                                    hierarchy: Optional[np.ndarray] = None,
                                    min_area: float = 10.0,
                                    min_perimeter: float = 10.0,
                                    min_points: int = 3) -> Tuple[List[np.ndarray], List[int]]:
        """
        基于轮廓特征过滤轮廓（不依赖索引奇偶性）
        
        参数:
            contours: 轮廓列表
            hierarchy: 层级信息
            min_area: 最小面积
            min_perimeter: 最小周长
            min_points: 最小点数
        
        返回:
            (有效轮廓列表, 有效索引列表)
        """
        valid_contours = []
        valid_indices = []
        
        for i, contour in enumerate(contours):
            # 基本验证
            if contour is None or len(contour) == 0:
                continue
            
            try:
                points = np.asarray(contour, dtype=float).reshape(-1, 2)
            except Exception:
                continue
            
            if len(points) < min_points:
                continue
            
            # 计算轮廓特征
            try:
                area = cv2.contourArea(points.astype(np.int32))
                perimeter = cv2.arcLength(points.astype(np.int32), True)
            except Exception:
                # 如果计算失败，使用简单方法
                if len(points) >= 2:
                    diffs = np.diff(points, axis=0)
                    distances = np.linalg.norm(diffs, axis=1)
                    perimeter = float(np.sum(distances))
                    # 简单面积估计（使用边界框）
                    if len(points) >= 3:
                        x_min, y_min = np.min(points, axis=0)
                        x_max, y_max = np.max(points, axis=0)
                        area = float((x_max - x_min) * (y_max - y_min))
                    else:
                        area = 0.0
                else:
                    continue
            
            # 过滤条件
            if area < min_area:
                continue
            
            if perimeter < min_perimeter:
                continue
            
            # 层级信息（可选）
            # 可以基于层级信息进行进一步过滤
            
            valid_contours.append(contour)
            valid_indices.append(i)
        
        return valid_contours, valid_indices
