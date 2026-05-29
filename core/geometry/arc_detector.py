"""
工程级Arc判别模块
使用稳健的圆拟合和多重判据来准确识别圆弧段

原问题：
1. 使用"角度曲率"而非"几何曲率"，在非均匀采样下判别方向是反的
2. 没有做"弧段连续性"判别（曲率符号一致、中心一致）
3. 最大误差判据对噪声极度敏感
4. Arc和NURBS的判据空间高度重叠

新方案：
1. 使用Pratt圆拟合（稳健最小二乘）
2. 半径稳定性（相对RMS）
3. 中心稳定性（局部三点圆）
4. 覆盖角度（不是点数）
5. 曲率符号一致性
"""

import numpy as np
from typing import Tuple, Optional, Dict


def fit_circle_pratt(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Pratt circle fitting (stable least squares)
    比标准最小二乘更稳健，对噪声不敏感
    
    参数:
        points: 点数组 (N, 2)
    
    返回:
        (圆心, 半径)
    """
    if len(points) < 3:
        raise ValueError("至少需要3个点才能拟合圆")
    
    x = points[:, 0]
    y = points[:, 1]
    
    # 中心化（提高数值稳定性）
    x_m = np.mean(x)
    y_m = np.mean(y)
    
    u = x - x_m
    v = y - y_m
    
    # 计算Pratt方法的系数
    Suu = np.sum(u * u)
    Suv = np.sum(u * v)
    Svv = np.sum(v * v)
    Suuu = np.sum(u * u * u)
    Suvv = np.sum(u * v * v)
    Svvv = np.sum(v * v * v)
    Svuu = np.sum(v * u * u)
    
    # 构建线性方程组
    A = np.array([[Suu, Suv],
                  [Suv, Svv]])
    B = np.array([(Suuu + Suvv) / 2.0,
                  (Svvv + Svuu) / 2.0])
    
    try:
        uc, vc = np.linalg.solve(A, B)
        center = np.array([uc + x_m, vc + y_m])
        
        # 计算半径（使用所有点到圆心的平均距离）
        dists = np.linalg.norm(points - center, axis=1)
        radius = np.mean(dists)
        
        return center, radius
    except np.linalg.LinAlgError:
        # 如果求解失败，使用简单方法
        center = np.array([x_m, y_m])
        dists = np.linalg.norm(points - center, axis=1)
        radius = np.mean(dists)
        return center, radius


def arc_quality(points: np.ndarray,
                sigma_r_max: float = 0.02,
                sigma_c_max: float = 1.0,
                theta_min_deg: float = 15.0) -> Tuple[bool, Optional[Dict]]:
    """
    稳健Arc判别（工程级）
    
    判据：
    1. 半径稳定性（相对RMS）< sigma_r_max
    2. 中心稳定性（局部三点圆）< sigma_c_max
    3. 覆盖角度 > theta_min_deg
    4. 曲率符号一致性
    
    参数:
        points: 点数组 (N, 2)
        sigma_r_max: 最大相对半径误差（默认2%）
        sigma_c_max: 最大中心漂移（像素，默认1.0）
        theta_min_deg: 最小覆盖角度（度，默认15°）
    
    返回:
        (是否成功, Arc信息字典)
    """
    if len(points) < 6:
        return False, None
    
    try:
        # --- 1. 全局圆拟合（Pratt方法） ---
        center, radius = fit_circle_pratt(points)
        
        # --- 2. 半径稳定性（相对RMS） ---
        dists = np.linalg.norm(points - center, axis=1)
        sigma_r = np.std(dists) / (radius + 1e-9)
        
        if sigma_r > sigma_r_max:
            return False, None
        
        # --- 3. 局部中心稳定性（滑动窗口三点圆） ---
        centers = []
        window_size = min(5, len(points) // 2)
        
        for i in range(len(points) - window_size + 1):
            try:
                window_points = points[i:i+window_size]
                if len(window_points) >= 3:
                    c, _ = fit_circle_pratt(window_points)
                    centers.append(c)
            except:
                continue
        
        if len(centers) < 3:
            # 如果局部中心太少，使用全局中心
            centers = [center]
        
        centers = np.array(centers)
        if len(centers) > 0:
            # 计算中心漂移的标准差
            center_std = np.std(centers, axis=0)
            sigma_c = np.max(center_std)
            
            if sigma_c > sigma_c_max:
                return False, None
        else:
            sigma_c = 0.0
        
        # --- 4. 覆盖角度（不是点数！） ---
        # 计算每个点相对于圆心的角度
        angles = np.arctan2(points[:, 1] - center[1],
                           points[:, 0] - center[0])
        
        # 处理角度跳跃（unwrap）
        angles_unwrapped = np.unwrap(angles)
        delta_theta = np.degrees(np.ptp(angles_unwrapped))
        
        if delta_theta < theta_min_deg:
            return False, None
        
        # --- 5. 曲率符号一致性（可选，但很重要） ---
        # 计算每个点的曲率符号（左弯/右弯）
        if len(points) >= 3:
            # 计算相邻向量的叉积（判断左弯/右弯）
            vectors = np.diff(points, axis=0)
            cross_products = []
            
            for i in range(len(vectors) - 1):
                v1 = vectors[i]
                v2 = vectors[i+1]
                # 2D叉积（z分量）
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                cross_products.append(cross)
            
            if len(cross_products) > 0:
                cross_products = np.array(cross_products)
                # 检查符号一致性（应该全正或全负）
                positive_ratio = np.sum(cross_products > 0) / len(cross_products)
                negative_ratio = np.sum(cross_products < 0) / len(cross_products)
                
                # 如果符号不一致（混合），可能不是圆弧
                if positive_ratio > 0.3 and negative_ratio > 0.3:
                    # 符号不一致，但允许一定噪声
                    if min(positive_ratio, negative_ratio) > 0.2:
                        return False, None
        
        # 所有判据通过，返回Arc信息
        return True, {
            "center": center,
            "radius": radius,
            "angle_span": delta_theta,
            "sigma_r": sigma_r,
            "sigma_c": sigma_c,
            "num_points": len(points)
        }
    
    except Exception as e:
        # 如果任何步骤失败，返回False
        return False, None


def is_arc_segment(points: np.ndarray,
                  sigma_r_max: float = 0.02,
                  sigma_c_max: float = 1.0,
                  theta_min_deg: float = 15.0) -> bool:
    """
    便捷函数：判断是否为Arc段
    
    参数:
        points: 点数组 (N, 2)
        sigma_r_max: 最大相对半径误差
        sigma_c_max: 最大中心漂移
        theta_min_deg: 最小覆盖角度
    
    返回:
        是否为Arc段
    """
    is_arc, _ = arc_quality(points, sigma_r_max, sigma_c_max, theta_min_deg)
    return is_arc
