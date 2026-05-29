"""
Arc→NURBS调度器
决定是否采用Arc，还是回退到NURBS

核心思想：
- Arc是高门槛特例，NURBS是默认归宿
- 只有当Arc在"表达能力+稳定性+CAD友好性"上明显优于NURBS时，才用Arc
- 与评分体系强耦合，确保Arc Radius Error和Spline Residual协同最优
"""

import numpy as np
from typing import Dict, Tuple, Optional
from geomdl import NURBS, utilities


def rms_point_curve(points: np.ndarray, curve_pts: np.ndarray) -> float:
    """
    计算点到曲线的RMS误差
    
    参数:
        points: 原始点数组 (N, 2)
        curve_pts: 曲线采样点数组 (M, 2)
    
    返回:
        RMS误差
    """
    if len(curve_pts) == 0:
        return float('inf')
    
    errs = []
    for p in points:
        dists = np.linalg.norm(curve_pts - p, axis=1)
        errs.append(np.min(dists))
    
    return np.sqrt(np.mean(np.square(errs)))


def build_simple_nurbs(points: np.ndarray, degree: int = 3) -> Tuple[NURBS.Curve, np.ndarray]:
    """
    构建简单的NURBS曲线（用于对比）
    
    参数:
        points: 控制点数组 (N, 2)
        degree: NURBS阶数
    
    返回:
        (NURBS曲线对象, 采样点数组)
    """
    if len(points) < 2:
        raise ValueError("控制点数量不足")
    
    # 确保阶数不超过控制点数量
    degree = min(degree, len(points) - 1)
    if degree < 1:
        degree = 1
    
    # 创建NURBS曲线
    curve = NURBS.Curve(normalize_kv=True)
    curve.degree = degree
    curve.ctrlpts = points.tolist()
    curve.knotvector = utilities.generate_knot_vector(degree, len(points))
    curve.weights = [1.0] * len(points)  # 均匀权重
    
    # 采样点（用于误差计算）
    curve.sample_size = len(points) * 2
    sample_pts = np.array(curve.evalpts)
    
    return curve, sample_pts


def arc_nurbs_dispatch(points: np.ndarray,
                      arc_info: Dict,
                      nurbs_curve: Optional[NURBS.Curve] = None,
                      nurbs_pts: Optional[np.ndarray] = None,
                      arc_pts: Optional[np.ndarray] = None,
                      arc_ctrl: int = 3,
                      nurbs_ctrl: int = 8,
                      arc_gain_th: float = 1.5,
                      sigma_r_max: float = 0.02,
                      theta_min_arc: float = 20.0) -> Tuple[str, str]:
    """
    Arc→NURBS调度决策
    
    三层判据：
    1. 硬排他（Hard Exclusion）：直接否决Arc
    2. Arc vs NURBS对比判据：比较"性价比"
    3. 评分体系一致性约束：与metrics强耦合
    
    参数:
        points: 原始点数组 (N, 2)
        arc_info: Arc信息字典（来自arc_quality）
        nurbs_curve: NURBS曲线对象（可选，如果提供则使用，否则自动构建）
        nurbs_pts: NURBS采样点数组（可选）
        arc_pts: Arc采样点数组（可选，默认使用原始点）
        arc_ctrl: Arc参数数量（用于复杂度计算）
        nurbs_ctrl: NURBS控制点数量（用于复杂度计算）
        arc_gain_th: Arc优势阈值（>1.5表示Arc明显优于NURBS）
        sigma_r_max: 最大相对半径误差（硬约束）
        theta_min_arc: 最小覆盖角度（硬约束，度）
    
    返回:
        (决策结果: "arc"或"nurbs", 原因字符串)
    """
    # --- 第一层：硬排他（Hard Exclusion） ---
    sigma_r = arc_info.get("sigma_r", float('inf'))
    angle_span = arc_info.get("angle_span", 0.0)
    
    # 硬约束1：覆盖角度太小 → 交给Line/NURBS
    if angle_span < theta_min_arc:
        return "nurbs", f"small_angle_{angle_span:.1f}deg"
    
    # 硬约束2：半径不稳定 → 交给NURBS
    if sigma_r > sigma_r_max:
        return "nurbs", f"unstable_radius_{sigma_r:.4f}"
    
    # 硬约束3：点数太少 → 交给NURBS
    if len(points) < 6:
        return "nurbs", "insufficient_points"
    
    # --- 第二层：Arc vs NURBS 对比判据 ---
    # 如果没有提供NURBS曲线，自动构建一个简单的用于对比
    if nurbs_curve is None or nurbs_pts is None:
        try:
            # 使用稀疏控制点构建NURBS（降低复杂度）
            control_points = points[::max(1, len(points) // nurbs_ctrl)]
            nurbs_curve, nurbs_pts = build_simple_nurbs(control_points, degree=3)
        except Exception as e:
            # 如果NURBS构建失败，使用Arc
            return "arc", f"nurbs_build_failed_{str(e)}"
    
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
    
    # 调度分数：Arc优势 = (NURBS误差/Arc误差) * (NURBS复杂度/Arc复杂度)
    # 分数 > arc_gain_th 表示Arc明显优于NURBS
    if E_arc < 1e-9:
        score_arc = float('inf')  # Arc完美拟合
    else:
        score_arc = (E_nurbs / E_arc) * (C_nurbs / C_arc)
    
    # --- 第三层：评分体系一致性约束 ---
    # 约束1：Arc Radius Error要求（sigma_r < 0.02已在硬约束中）
    # 约束2：Arc拟合误差应该明显小于NURBS（至少20%优势）
    if E_arc > E_nurbs * 0.8:
        return "nurbs", f"arc_error_too_large_{E_arc:.3f}_vs_{E_nurbs:.3f}"
    
    # 决策：如果Arc优势明显，使用Arc；否则使用NURBS
    if score_arc > arc_gain_th:
        return "arc", f"arc_dominant_score_{score_arc:.2f}"
    else:
        return "nurbs", f"nurbs_better_score_{score_arc:.2f}"


def dispatch_segment(points: np.ndarray,
                    arc_info: Optional[Dict] = None,
                    sigma_r_max: float = 0.02,
                    sigma_c_max: float = 1.0,
                    theta_min_deg: float = 15.0,
                    arc_gain_th: float = 1.5) -> Tuple[str, Dict]:
    """
    完整的分段调度流程：Arc判别 → Arc/NURBS调度
    
    参数:
        points: 点数组 (N, 2)
        arc_info: Arc信息字典（可选，如果提供则跳过判别）
        sigma_r_max: 最大相对半径误差
        sigma_c_max: 最大中心漂移
        theta_min_deg: 最小覆盖角度
        arc_gain_th: Arc优势阈值
    
    返回:
        (决策结果: "arc"或"nurbs", 详细信息字典)
    """
    # 如果没有提供Arc信息，先进行Arc判别
    if arc_info is None:
        is_arc, arc_info = arc_quality(points, sigma_r_max, sigma_c_max, theta_min_deg)
        if not is_arc:
            return "nurbs", {"reason": "arc_quality_failed"}
    
    # 进行Arc→NURBS调度
    decision, reason = arc_nurbs_dispatch(
        points,
        arc_info,
        arc_gain_th=arc_gain_th,
        sigma_r_max=sigma_r_max,
        theta_min_arc=theta_min_deg
    )
    
    result = {
        "decision": decision,
        "reason": reason,
        "arc_info": arc_info if decision == "arc" else None
    }
    
    return decision, result
