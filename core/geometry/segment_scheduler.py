"""
全局优先级调度器（决策树）
统一调度Arc/Line/NURBS，解决Arc抢段、Line被弧污染、NURBS优势发挥不充分的问题

核心原则：
1. 表达能力优先级（硬约束）：Line > Arc > NURBS
2. 判别方式必须是"对比式"，而非"独立成立"
3. 几何一致性 + 评分驱动 + CAD表达优先级

决策树结构：
Step 1: Line判别（最高优先级）
Step 2: Arc硬判据（必要非充分）
Step 3: Arc vs NURBS对比（核心）
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from geomdl import NURBS
from core.geometry.arc_detector import arc_quality
from core.geometry.arc_nurbs_scheduler import rms_point_curve, build_simple_nurbs


def compute_turning_angles(points: np.ndarray) -> np.ndarray:
    """
    计算轮廓点的转角（累计角度变化）
    
    参数:
        points: 点数组 (N, 2)
    
    返回:
        转角数组 (N-2,)，单位：弧度
    """
    if len(points) < 3:
        return np.array([])
    
    p_prev = points[:-2]
    p = points[1:-1]
    p_next = points[2:]
    
    v1 = p - p_prev
    v2 = p_next - p
    
    # 计算角度（使用点积）
    dot_products = np.sum(v1 * v2, axis=1)
    norms1 = np.linalg.norm(v1, axis=1)
    norms2 = np.linalg.norm(v2, axis=1)
    
    cos_angles = np.clip(
        dot_products / (norms1 * norms2 + 1e-8),
        -1.0, 1.0
    )
    
    angles = np.arccos(cos_angles)
    return angles


def is_line_segment(points: np.ndarray,
                   linearity_threshold: float = 0.02,
                   max_residual_threshold: float = 1.0,
                   total_angle_threshold_deg: float = 10.0) -> Tuple[bool, Dict]:
    """
    Step 1: Line判别（最高优先级）
    
    判据：
    1. PCA线性度：λ₂/λ₁ < 0.02
    2. 最大正交残差：到拟合直线的最大距离 < 1 px
    3. 总转角：累计角度变化 < 10°
    
    参数:
        points: 点数组 (N, 2)
        linearity_threshold: PCA线性度阈值
        max_residual_threshold: 最大正交残差阈值（像素）
        total_angle_threshold_deg: 总转角阈值（度）
    
    返回:
        (是否为直线, Line信息字典)
    """
    # 确保points是numpy数组
    points = np.asarray(points)
    
    if len(points) < 2:
        return False, {}
    
    # 确保points是2D数组，形状为(N, 2)
    if len(points.shape) == 1:
        # 一维数组，无法处理
        return False, {}
    if len(points.shape) == 2:
        if points.shape[1] != 2:
            # 如果不是2列，尝试修复
            if points.shape[0] == 2:
                points = points.T
            else:
                # 无法修复，只取前两列
                if points.shape[1] > 2:
                    points = points[:, :2]
                else:
                    return False, {}
    else:
        # 超过2维，无法处理
        return False, {}
    
    if len(points) == 2:
        # 只有两个点，肯定是直线
        return True, {
            'p0': points[0],
            'p1': points[1],
            'direction': points[1] - points[0],
            'max_residual': 0.0,
            'linearity': 0.0,
            'total_angle_deg': 0.0
        }
    
    # 1. PCA线性度
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # SVD分解
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    if len(S) < 2 or S[0] < 1e-9:
        # 如果主方向不明显，不是直线
        return False, {}
    
    linearity = S[1] / (S[0] + 1e-9)
    
    # 2. 最大正交残差
    # 使用右奇异向量（Vt的第一行）作为主方向，形状为(2,)
    direction = Vt[0, :]  # 主方向，形状为(2,)
    proj = centered @ direction  # (N, 2) @ (2,) -> (N,)
    recon = np.outer(proj, direction) + mean  # (N,) @ (2,) -> (N, 2)
    residuals = np.linalg.norm(points - recon, axis=1)
    max_residual = np.max(residuals)
    
    # 3. 总转角
    turning_angles = compute_turning_angles(points)
    total_angle = np.sum(np.abs(turning_angles))
    total_angle_deg = np.degrees(total_angle)
    
    # 判断是否为直线
    is_line = (
        linearity < linearity_threshold and
        max_residual < max_residual_threshold and
        total_angle_deg < total_angle_threshold_deg
    )
    
    if is_line:
        direction_vec = direction / (np.linalg.norm(direction) + 1e-9)
        proj = centered @ direction_vec
        i0 = int(np.argmin(proj))
        i1 = int(np.argmax(proj))
        p0 = points[i0]
        p1 = points[i1]
        
        return True, {
            'p0': p0,
            'p1': p1,
            'direction': direction_vec,
            'linearity': linearity,
            'max_residual': max_residual,
            'total_angle_deg': total_angle_deg
        }
    
    return False, {}


def arc_hard_valid(arc_info: Dict,
                  sigma_r_max: float = 0.02,
                  angle_span_min_deg: float = 20.0,
                  num_points_min: int = 8) -> bool:
    """
    Step 2: Arc硬判据（必要非充分条件）
    
    参数:
        arc_info: Arc信息字典（来自arc_quality）
        sigma_r_max: 最大相对半径误差
        angle_span_min_deg: 最小覆盖角度（度）
        num_points_min: 最小点数
    
    返回:
        是否满足硬判据
    """
    if arc_info is None:
        return False
    
    sigma_r = arc_info.get("sigma_r", float('inf'))
    angle_span = arc_info.get("angle_span", 0.0)
    num_points = arc_info.get("num_points", 0)
    
    return (
        sigma_r < sigma_r_max and
        angle_span > angle_span_min_deg and
        num_points >= num_points_min
    )


def dispatch_segment(points: np.ndarray,
                    arc_info: Optional[Dict] = None,
                    nurbs_curve: Optional[NURBS.Curve] = None,
                    nurbs_pts: Optional[np.ndarray] = None,
                    arc_pts: Optional[np.ndarray] = None,
                    # Line判别参数
                    linearity_threshold: float = 0.02,
                    max_residual_threshold: float = 1.0,
                    total_angle_threshold_deg: float = 10.0,
                    # Arc硬判据参数
                    sigma_r_max: float = 0.02,
                    angle_span_min_deg: float = 20.0,
                    num_points_min: int = 8,
                    # Arc vs NURBS对比参数
                    arc_gain_th: float = 1.5,
                    arc_ctrl: int = 3,
                    nurbs_ctrl: int = 8) -> Tuple[str, Dict]:
    """
    全局优先级调度器（决策树）
    
    决策流程：
    1. Step 1: Line判别（最高优先级）
    2. Step 2: Arc硬判据（必要非充分）
    3. Step 3: Arc vs NURBS对比（核心）
    
    参数:
        points: 原始点数组 (N, 2)
        arc_info: Arc信息字典（可选，如果提供则跳过判别）
        nurbs_curve: NURBS曲线对象（可选，如果提供则使用）
        nurbs_pts: NURBS采样点数组（可选）
        arc_pts: Arc采样点数组（可选，默认使用原始点）
        linearity_threshold: PCA线性度阈值
        max_residual_threshold: 最大正交残差阈值（像素）
        total_angle_threshold_deg: 总转角阈值（度）
        sigma_r_max: 最大相对半径误差
        angle_span_min_deg: 最小覆盖角度（度）
        num_points_min: 最小点数
        arc_gain_th: Arc优势阈值
        arc_ctrl: Arc参数数量
        nurbs_ctrl: NURBS控制点数量
    
    返回:
        (决策结果: "line"/"arc"/"nurbs", 详细信息字典)
    """
    result = {
        'decision': None,
        'reason': None,
        'info': {}
    }
    
    # ========== Step 1: Line判别（最高优先级） ==========
    is_line, line_info = is_line_segment(
        points,
        linearity_threshold=linearity_threshold,
        max_residual_threshold=max_residual_threshold,
        total_angle_threshold_deg=total_angle_threshold_deg
    )
    
    if is_line:
        result['decision'] = 'line'
        result['reason'] = 'line_priority'
        result['info'] = line_info
        return 'line', result
    
    # ========== Step 2: Arc硬判据（必要非充分） ==========
    if arc_info is None:
        is_arc, arc_info = arc_quality(
            points,
            sigma_r_max=sigma_r_max,
            sigma_c_max=1.0,
            theta_min_deg=angle_span_min_deg
        )
    else:
        is_arc = True
    
    if not is_arc:
        result['decision'] = 'nurbs'
        result['reason'] = 'arc_quality_failed'
        result['info'] = {}
        return 'nurbs', result
    
    # 检查Arc硬判据
    if not arc_hard_valid(arc_info, sigma_r_max, angle_span_min_deg, num_points_min):
        result['decision'] = 'nurbs'
        result['reason'] = 'arc_hard_check_failed'
        result['info'] = arc_info
        return 'nurbs', result
    
    # ========== Step 3: Arc vs NURBS对比（核心） ==========
    # 如果没有提供NURBS曲线，自动构建一个简单的用于对比
    if nurbs_curve is None or nurbs_pts is None:
        try:
            # 使用稀疏控制点构建NURBS（降低复杂度）
            control_points = points[::max(1, len(points) // nurbs_ctrl)]
            nurbs_curve, nurbs_pts = build_simple_nurbs(control_points, degree=3)
        except Exception as e:
            # 如果NURBS构建失败，使用Arc（保守策略）
            result['decision'] = 'arc'
            result['reason'] = f'nurbs_build_failed_{str(e)}'
            result['info'] = arc_info
            return 'arc', result
    
    # 如果没有提供Arc点，使用原始点
    if arc_pts is None:
        arc_pts = points
    
    # 计算Arc拟合误差
    E_arc = rms_point_curve(points, arc_pts)
    
    # 计算NURBS拟合误差
    E_nurbs = rms_point_curve(points, nurbs_pts)
    
    # 计算复杂度（参数数量）
    C_arc = arc_ctrl  # Arc: center(2) + radius(1) = 3
    C_nurbs = nurbs_ctrl  # NURBS: 控制点数量
    
    # 调度分数：Arc优势 = (NURBS误差/Arc误差) × (NURBS复杂度/Arc复杂度)
    if E_arc < 1e-9:
        score_arc = float('inf')  # Arc完美拟合
    else:
        score_arc = (E_nurbs / E_arc) * (C_nurbs / C_arc)
    
    # 决策：如果Arc优势明显，使用Arc；否则使用NURBS
    if score_arc > arc_gain_th:
        result['decision'] = 'arc'
        result['reason'] = f'arc_dominant_score_{score_arc:.2f}'
        result['info'] = arc_info
        result['info']['score'] = score_arc
        result['info']['E_arc'] = E_arc
        result['info']['E_nurbs'] = E_nurbs
        return 'arc', result
    else:
        result['decision'] = 'nurbs'
        result['reason'] = f'nurbs_better_score_{score_arc:.2f}'
        result['info'] = {
            'score': score_arc,
            'E_arc': E_arc,
            'E_nurbs': E_nurbs
        }
        return 'nurbs', result


def dispatch_segment_simple(points: np.ndarray,
                           **kwargs) -> Tuple[str, Dict]:
    """
    便捷函数：完整的分段调度流程
    
    参数:
        points: 点数组 (N, 2)
        **kwargs: 其他参数（传递给dispatch_segment）
    
    返回:
        (决策结果, 详细信息字典)
    """
    return dispatch_segment(points, **kwargs)


def arc_angles_from_points(center: np.ndarray, points: np.ndarray) -> Dict:
    center = np.asarray(center, dtype=float).reshape(2,)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return {'start_angle': 0.0, 'end_angle': 0.0, 'ccw': True, 'angle_span': 0.0}
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ang_u = np.unwrap(ang)
    start = float(np.degrees(ang_u[0]))
    end = float(np.degrees(ang_u[-1]))
    span = float(abs(end - start))
    return {'start_angle': start, 'end_angle': end, 'ccw': end >= start, 'angle_span': span}
