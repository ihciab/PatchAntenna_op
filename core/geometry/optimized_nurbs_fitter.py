"""
优化的NURBS轮廓拟合模块
基于几何语义的分段NURBS拟合（Line/Arc/NURBS混合表示）

优化点：
1. 曲率驱动的权重初始化（真正利用NURBS的有理性）
2. 弧长参数化（替代均匀角度参数化）
3. 分段拟合（Line/Arc/NURBS混合表示）
4. 层级感知的拟合强度
5. CAD友好的结构化输出

原代码参考：core/geometry/nurbs_fitter.py
"""

import numpy as np
import cv2
from geomdl import NURBS, utilities
import geomdl.knotvector
from typing import Dict, List, Tuple, Optional
from core.geometry.bspline_fitter import BSplineFitter
from core.geometry.arc_detector import arc_quality
from core.geometry.segment_scheduler import dispatch_segment, arc_angles_from_points, is_line_segment
from core.geometry.improved_segment_merger import ImprovedSegmentMerger
from core.geometry.contour_preprocessor import ContourPreprocessor


class OptimizedNURBSFitter(BSplineFitter):
    """
    优化的NURBS轮廓拟合类
    实现分段NURBS拟合，真正利用NURBS的有理性
    """
    
    def __init__(self, img, edges, 
                 lr=0.1, delta=2, step=30, threshold: float = 10.0, 
                 degree: int = 3,
                 line_threshold: float = 2.0,
                 arc_threshold: float = 2.0,
                 curvature_threshold: float = 0.15,
                 curvature_weight_alpha: float = 5.0,
                 use_segmentation: bool = True,
                 show: bool = True, 
                 save: str = '',
                 # 改进的段合并参数
                 use_improved_merger: bool = True,
                 angle_threshold_deg: float = 15.0,
                 distance_threshold: float = 5.0,
                 min_segment_length: float = 20.0,
                 max_segments: Optional[int] = None,
                 error_tolerance_factor: float = 1.5,
                 # 拓扑一致性改进参数
                 use_simple_knots_by_default: bool = True,
                 max_weight_multiplier: float = 2.0):
        """
        初始化优化NURBS拟合器
        
        参数:
            img: 输入图像
            edges: 边缘图像
            lr: 学习率
            delta: 步长补偿值
            step: 初始步长
            threshold: 误差阈值
            degree: NURBS阶数
            line_threshold: 直线拟合误差阈值（像素）
            arc_threshold: 圆弧拟合误差阈值（像素）
            curvature_threshold: 曲率变化阈值（用于分段）
            curvature_weight_alpha: 曲率权重系数（越大，高曲率区域权重越大）
            use_segmentation: 是否使用分段拟合（Line/Arc/NURBS）
            show: 是否显示结果
            save: 保存路径
            use_improved_merger: 是否使用改进的段合并器
            angle_threshold_deg: 合并角度阈值（度）
            distance_threshold: 合并距离阈值（像素）
            min_segment_length: 最小段长度（像素）
            max_segments: 最大段数（None表示无限制）
            error_tolerance_factor: 误差容忍因子
            use_simple_knots_by_default: 默认使用简单节点向量（提高拓扑一致性）
            max_weight_multiplier: 权重上限倍数（限制权重过大导致不稳定）
        """
        self.derived_paras(img, edges, lr, delta, step, threshold, save)
        self.__degree = degree
        self.line_threshold = line_threshold
        self.arc_threshold = arc_threshold
        self.curvature_threshold = curvature_threshold
        self.curvature_weight_alpha = curvature_weight_alpha
        self.use_segmentation = use_segmentation
        self.use_simple_knots_by_default = use_simple_knots_by_default
        self.max_weight_multiplier = max_weight_multiplier
        
        # 初始化改进的段合并器
        self.use_improved_merger = use_improved_merger
        if use_improved_merger:
            self.segment_merger = ImprovedSegmentMerger(
                angle_threshold_deg=angle_threshold_deg,
                distance_threshold=distance_threshold,
                min_segment_length=min_segment_length,
                max_segments=max_segments,
                error_tolerance_factor=error_tolerance_factor
            )
        else:
            self.segment_merger = None
        
        # 存储分段信息（用于CAD导出）
        self._segments_info = {}
        
        # 执行拟合
        self.__optimized_nurbs_process(show)
        # 设置轮廓树
        self._BSplineFitter__curves_tree = self._BSplineFitter__node_tree()
    
    def arc_length_param(self, points: np.ndarray) -> np.ndarray:
        """
        计算弧长参数化（关键优化）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            归一化的弧长参数 (N,)
        """
        if len(points) < 2:
            return np.array([0.0, 1.0])
        
        # 计算相邻点之间的距离
        diffs = np.diff(points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        
        # 处理零距离（避免除零）
        distances = np.maximum(distances, 1e-6)
        
        # 累积弧长
        cumulative = np.insert(np.cumsum(distances), 0, 0.0)
        
        # 归一化到[0, 1]
        if cumulative[-1] > 1e-6:
            return cumulative / cumulative[-1]
        else:
            return np.linspace(0, 1, len(points))
    
    def discrete_curvature(self, points: np.ndarray) -> np.ndarray:
        """
        计算离散曲率（基于角度变化）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            曲率估计值 (N,)，已填充边界
        """
        if len(points) < 3:
            return np.zeros(len(points))
        
        p_prev = points[:-2]
        p = points[1:-1]
        p_next = points[2:]
        
        v1 = p - p_prev
        v2 = p_next - p
        
        # 计算角度（使用点积）
        dot_products = np.sum(v1 * v2, axis=1)
        norms1 = np.linalg.norm(v1, axis=1)
        norms2 = np.linalg.norm(v2, axis=1)
        
        # 避免除零
        cos_angles = np.clip(
            dot_products / (norms1 * norms2 + 1e-8),
            -1.0, 1.0
        )
        
        # 角度（与曲率成正相关）
        angles = np.arccos(cos_angles)
        
        # 填充边界（使用边界值）
        return np.pad(angles, (1, 1), mode='edge')
    
    def curvature_driven_weights(self, points: np.ndarray) -> np.ndarray:
        """
        曲率驱动的权重初始化（改进版：限制权重上限，提高稳定性）
        
        参数:
            points: 控制点数组 (N, 2)
        
        返回:
            权重数组 (N,)
        """
        if len(points) < 3:
            return np.ones(len(points))
        
        # 计算曲率
        curvature = self.discrete_curvature(points)
        
        # 归一化曲率
        max_curvature = np.max(curvature)
        if max_curvature > 1e-6:
            normalized_curvature = curvature / max_curvature
        else:
            normalized_curvature = np.zeros_like(curvature)
        
        # 【改进】限制权重上限，避免权重过大导致不稳定
        # 原公式：weights = 1.0 + alpha * normalized_curvature
        # 改进：限制最大权重为 max_weight_multiplier
        raw_weights = 1.0 + self.curvature_weight_alpha * normalized_curvature
        max_weight = 1.0 + self.max_weight_multiplier  # 例如：1.0 + 2.0 = 3.0
        weights = np.clip(raw_weights, 1.0, max_weight)
        
        # 【改进】对直线段强制权重=1.0（提高拓扑一致性）
        # 检测直线段：如果曲率很小，认为是直线
        is_straight = normalized_curvature < 0.1  # 曲率很小的点
        if np.any(is_straight):
            weights[is_straight] = 1.0
        
        return weights
    
    def _generate_arc_length_knot_vector(self, arc_length_params: np.ndarray,
                                        degree: int, num_ctrlpts: int) -> List[float]:
        """
        【优化3】根据弧长参数生成节点向量
        
        参数:
            arc_length_params: 弧长参数数组 (N,)，已归一化到[0, 1]
            degree: NURBS阶数
            num_ctrlpts: 控制点数量
        
        返回:
            节点向量列表
        """
        # 标准NURBS节点向量结构：
        # 前degree+1个节点 = 0（clamped start）
        # 中间节点：根据弧长参数分布
        # 后degree+1个节点 = 1（clamped end）
        
        num_knots = degree + num_ctrlpts + 1
        
        # 前degree+1个节点 = 0
        knots = [0.0] * (degree + 1)
        
        # 中间节点：根据弧长参数分布
        # 使用累积弧长来分配中间节点
        if len(arc_length_params) > 1:
            # 中间节点数量
            num_internal = num_knots - 2 * (degree + 1)
            
            if num_internal > 0:
                # 在弧长参数中均匀采样（但基于弧长分布）
                # 使用累积分布函数（CDF）来分配节点
                internal_knots = []
                
                # 方法：在[0, 1]区间内，根据弧长参数的分布来分配节点
                # 使用线性插值，但考虑弧长参数的密度
                for i in range(1, num_internal + 1):
                    # 均匀分布在(0, 1)区间
                    t = i / (num_internal + 1)
                    
                    # 找到对应的弧长参数值（使用插值）
                    # 简化：直接使用均匀分布，但可以改进为基于弧长密度
                    internal_knots.append(t)
                
                knots.extend(internal_knots)
        
        # 后degree+1个节点 = 1
        knots.extend([1.0] * (degree + 1))
        
        # 确保节点向量单调递增
        for i in range(1, len(knots)):
            if knots[i] < knots[i-1]:
                knots[i] = knots[i-1] + 1e-6
        
        return knots
    
    def fit_line(self, points: np.ndarray) -> Tuple[bool, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        拟合直线段
        
        参数:
            points: 点数组 (N, 2)
        
        返回:
            (是否成功, (起点, 终点))
        """
        if len(points) < 2:
            return False, None
        
        p0 = points[0]
        p1 = points[-1]
        
        # 方向向量
        v = p1 - p0
        v_norm = float(np.linalg.norm(v))
        
        if v_norm < 1e-6:
            return False, None

        if v_norm < 8.0:
            return False, None
        
        v = v / v_norm
        
        # 计算所有点到直线的距离
        proj = np.dot(points - p0, v)
        closest_points = p0 + np.outer(proj, v)
        errors = np.linalg.norm(points - closest_points, axis=1)
        
        max_error = float(np.max(errors))
        rel_error = max_error / (v_norm + 1e-9)

        if max_error < self.line_threshold and rel_error < 0.12:
            return True, (p0, p1)
        return False, None
    
    def fit_arc(self, points: np.ndarray) -> Tuple[bool, Optional[Tuple[np.ndarray, float]]]:
        """
        拟合圆弧段（使用法向一致性判据，工业CAD标准方法）
        
        参数:
            points: 点数组 (N, 2)
        
        返回:
            (是否成功, (圆心, 半径))
        """
        if len(points) < 3:
            return False, None
        
        x = points[:, 0]
        y = points[:, 1]
        
        # 使用最小二乘法拟合圆
        A = np.column_stack([2*x, 2*y, np.ones(len(points))])
        b = x**2 + y**2
        
        try:
            c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = c[0], c[1]
            r = np.sqrt(c[2] + cx**2 + cy**2)
            
            # 计算拟合误差
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            errors = np.abs(dist - r)
            max_error = np.max(errors)
            
            # 【优化1】使用"法向一致性"判据，而不是curvature_std
            # 这是工业CAD中判圆弧的标准做法
            center = np.array([cx, cy])
            
            # 1. 点到圆心距离的方差（半径误差）
            radius_error = np.std(dist)
            relative_radius_error = radius_error / (r + 1e-9)
            
            # 2. 法向角变化（法向一致性）
            # 计算每个点的法向向量（从圆心指向点）
            vectors = points - center
            vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vector_norms = np.maximum(vector_norms, 1e-9)  # 避免除零
            normals = vectors / vector_norms  # 归一化法向向量
            
            # 计算相邻法向向量的角度变化
            if len(normals) >= 2:
                # 计算相邻法向向量的点积（角度）
                dot_products = np.sum(normals[:-1] * normals[1:], axis=1)
                dot_products = np.clip(dot_products, -1.0, 1.0)
                angles = np.arccos(dot_products)  # 角度变化
                angle_variance = np.std(angles)  # 法向角变化的标准差
            else:
                angle_variance = 0.0
            
            # 判据：半径误差和法向角变化都要小
            # 阈值：相对半径误差 < 5%，法向角变化 < 0.1弧度（约5.7度）
            tau_r = 0.05  # 5%相对误差
            tau_theta = 0.1  # 0.1弧度
            
            if relative_radius_error > tau_r or angle_variance > tau_theta:
                # 不满足圆弧判据
                return False, None
            
            # 同时检查最大误差（像素级）
            if max_error < self.arc_threshold:
                return True, (np.array([cx, cy]), r)
        except:
            pass
        
        return False, None
    
    def initial_segmentation(self, points: np.ndarray) -> List[np.ndarray]:
        """
        基于曲率变化进行初步分段（改进版：自适应曲率阈值）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            分段列表
        """
        if len(points) < 3:
            return [points]
        
        # 计算曲率
        curvature = self.discrete_curvature(points)
        
        if len(curvature) == 0:
            return [points]
        
        # 【改进】自适应曲率阈值：使用百分位数而不是固定阈值
        if len(curvature) > 0:
            # 使用75%分位数作为阈值（更保守，减少过度分段）
            adaptive_threshold = np.percentile(curvature, 75)
            # 但不要低于原始阈值太多
            adaptive_threshold = max(adaptive_threshold, self.curvature_threshold * 0.8)
        else:
            adaptive_threshold = self.curvature_threshold
        
        # 计算曲率变化率
        dk = np.abs(np.diff(curvature))
        
        # 【改进】使用自适应阈值
        break_indices = np.where(dk > adaptive_threshold)[0] + 1

        corner_set: set[int] = set()
        span = int(np.clip(len(points) // 200, 4, 12))
        if len(points) >= 2 * span + 3:
            p_prev = points[:-2 * span]
            p = points[span:-span]
            p_next = points[2 * span:]
            v1 = p - p_prev
            v2 = p_next - p
            norms1 = np.linalg.norm(v1, axis=1)
            norms2 = np.linalg.norm(v2, axis=1)
            dot_products = np.sum(v1 * v2, axis=1)
            cos_angles = np.clip(dot_products / (norms1 * norms2 + 1e-8), -1.0, 1.0)
            turning_angles = np.arccos(cos_angles)
            corner_tol = np.deg2rad(18.0)
            min_vec = max(6.0, float(span) * 1.2)
            corner_mask = (np.abs(turning_angles - (np.pi / 2)) <= corner_tol) & (norms1 >= min_vec) & (norms2 >= min_vec)
            corner_breaks = (np.where(corner_mask)[0] + span).astype(int)
            corner_set = set(int(x) for x in corner_breaks.tolist())

        break_indices = np.array(sorted(set(int(x) for x in break_indices.tolist()) | corner_set), dtype=int)
        
        # 构建分段
        segments = []
        start = 0

        # 【改进】根据轮廓复杂度动态调整最小点数
        if len(points) < 200:
            min_points = int(np.clip(len(points) // 80, 8, 30))
        elif len(points) > 500:
            min_points = int(np.clip(len(points) // 50, 20, 50))
        else:
            min_points = int(np.clip(len(points) // 60, 12, 40))
        
        for b in break_indices.tolist():
            if b - start + 1 < 3:
                continue
            if b in corner_set:
                if b - start + 1 < max(6, min_points // 2):
                    continue
            elif b - start + 1 < min_points:
                continue
            segments.append(points[start:b + 1])
            start = b

        if start < len(points):
            tail = points[start:]
            if segments and len(tail) < min_points and start not in corner_set:
                segments[-1] = np.vstack([segments[-1], tail[1:]]) if len(tail) > 1 else segments[-1]
            else:
                segments.append(tail)

        if not segments:
            segments = [points]
        
        return segments

    def _regularize_segments_for_cad(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return segments

        def _to_pt(v) -> Optional[np.ndarray]:
            """安全地将值转换为2D点"""
            if v is None:
                return None
            try:
                val = np.asarray(v, dtype=float)
                # 如果是1D数组，检查大小
                if len(val.shape) == 1:
                    if val.size == 2:
                        return val
                    else:
                        return None
                # 如果是2D数组，取第一个点
                elif len(val.shape) == 2:
                    if val.shape[1] == 2 and val.shape[0] > 0:
                        return val[0]
                    elif val.shape[0] == 2 and val.shape[1] > 0:
                        return val[:, 0]
                    else:
                        return None
                else:
                    return None
            except Exception:
                return None

        join_tol = 2.5
        angle_tol_deg = 7.5
        tan_tol = float(np.tan(np.deg2rad(angle_tol_deg)))

        segs: List[Dict] = []
        for seg in segments:
            t = seg.get('type')
            if t == 'line':
                # 【修复】确保start和end是有效的2D点
                def _get_point_safe(seg: Dict, key1: str, key2: str, default: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
                    val = seg.get(key1, seg.get(key2, None))
                    if val is None:
                        return default
                    try:
                        val = np.asarray(val, dtype=float)
                        # 如果是1D数组，检查大小
                        if len(val.shape) == 1:
                            if val.size == 2:
                                return val
                            else:
                                return default
                        # 如果是2D数组，取第一个点
                        elif len(val.shape) == 2:
                            if val.shape[1] == 2 and val.shape[0] > 0:
                                return val[0]
                            elif val.shape[0] == 2 and val.shape[1] > 0:
                                return val[:, 0]
                            else:
                                return default
                        else:
                            return default
                    except Exception:
                        return default
                
                # 尝试获取start和end
                p0 = _get_point_safe(seg, 'start', 'p0')
                p1 = _get_point_safe(seg, 'end', 'p1')
                
                # 如果无法获取，尝试从points中获取
                if p0 is None or p1 is None:
                    pts = seg.get('points', None)
                    if pts is not None:
                        try:
                            pts_arr = np.asarray(pts, dtype=float)
                            if len(pts_arr.shape) == 2 and pts_arr.shape[1] == 2 and len(pts_arr) >= 2:
                                if p0 is None:
                                    p0 = pts_arr[0]
                                if p1 is None:
                                    p1 = pts_arr[-1]
                        except Exception:
                            pass
                
                # 如果仍然无法获取，跳过这个段
                if p0 is None or p1 is None:
                    # 使用默认值或跳过
                    new_seg = dict(seg)
                    segs.append(new_seg)
                    continue
                
                # 确保是1D数组（形状为(2,)）
                p0 = np.asarray(p0, dtype=float).flatten()
                p1 = np.asarray(p1, dtype=float).flatten()
                
                if p0.size != 2 or p1.size != 2:
                    # 如果形状不对，跳过
                    new_seg = dict(seg)
                    segs.append(new_seg)
                    continue
                
                v = p1 - p0
                if float(np.linalg.norm(v)) > 1e-8:
                    ax = abs(float(v[0]))
                    ay = abs(float(v[1]))
                    if ay <= ax * tan_tol:
                        y = float((p0[1] + p1[1]) / 2.0)
                        p0 = np.array([p0[0], y], dtype=float)
                        p1 = np.array([p1[0], y], dtype=float)
                        orientation = 'h'
                    elif ax <= ay * tan_tol:
                        x = float((p0[0] + p1[0]) / 2.0)
                        p0 = np.array([x, p0[1]], dtype=float)
                        p1 = np.array([x, p1[1]], dtype=float)
                        orientation = 'v'
                    else:
                        orientation = None
                else:
                    orientation = None

                new_seg = dict(seg)
                new_seg['p0'] = p0.tolist()
                new_seg['p1'] = p1.tolist()
                new_seg['start'] = p0.tolist()
                new_seg['end'] = p1.tolist()
                new_seg['_ori'] = orientation
                pts = seg.get('points', None)
                if isinstance(pts, list) and len(pts) >= 2:
                    pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
                    pts_arr[0] = p0
                    pts_arr[-1] = p1
                    new_seg['points'] = pts_arr.tolist()
                segs.append(new_seg)
            elif t == 'arc':
                new_seg = dict(seg)
                try:
                    c = _to_pt(new_seg.get('center', None))
                    if c is not None:
                        new_seg['center'] = c.tolist()
                except Exception:
                    pass
                try:
                    if 'radius' in new_seg:
                        new_seg['radius'] = float(new_seg['radius'])
                except Exception:
                    pass
                pts = seg.get('points', None)
                if isinstance(pts, list) and len(pts) >= 3 and 'center' in seg and 'radius' in seg:
                    c = _to_pt(seg['center'])
                    pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
                    ang = arc_angles_from_points(c, pts_arr)
                    new_seg['start_angle'] = float(ang.get('start_angle', 0.0))
                    new_seg['end_angle'] = float(ang.get('end_angle', 0.0))
                    new_seg['ccw'] = bool(ang.get('ccw', True))
                segs.append(new_seg)
            else:
                segs.append(dict(seg))

        def _get_endpoints(s: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            st = s.get('type')
            if st == 'line':
                p0 = s.get('p0', s.get('start', None))
                p1 = s.get('p1', s.get('end', None))
                if p0 is None or p1 is None:
                    # 尝试从points中获取
                    pts = s.get('points', None)
                    if pts is not None:
                        try:
                            pts_arr = np.asarray(pts, dtype=float)
                            if len(pts_arr.shape) == 2 and pts_arr.shape[1] == 2 and len(pts_arr) >= 2:
                                return pts_arr[0].copy(), pts_arr[-1].copy()
                        except Exception:
                            pass
                    return None, None
                p0_pt = _to_pt(p0)
                p1_pt = _to_pt(p1)
                if p0_pt is None or p1_pt is None:
                    return None, None
                return p0_pt, p1_pt
            pts = s.get('points', None)
            if pts is not None:
                try:
                    pts_arr = np.asarray(pts, dtype=float)
                    if len(pts_arr.shape) == 2 and pts_arr.shape[1] == 2 and len(pts_arr) >= 2:
                        return pts_arr[0].copy(), pts_arr[-1].copy()
                except Exception:
                    pass
            return None, None

        def _set_start(s: Dict, p: np.ndarray):
            st = s.get('type')
            if st == 'line':
                if 'p0' in s:
                    s['p0'] = p.tolist()
                s['start'] = p.tolist()
            pts = s.get('points', None)
            if isinstance(pts, list) and len(pts) >= 2:
                pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
                pts_arr[0] = p
                s['points'] = pts_arr.tolist()

        def _set_end(s: Dict, p: np.ndarray):
            st = s.get('type')
            if st == 'line':
                if 'p1' in s:
                    s['p1'] = p.tolist()
                s['end'] = p.tolist()
            pts = s.get('points', None)
            if isinstance(pts, list) and len(pts) >= 2:
                pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
                pts_arr[-1] = p
                s['points'] = pts_arr.tolist()

        for i in range(len(segs) - 1):
            a = segs[i]
            b = segs[i + 1]
            a0, a1 = _get_endpoints(a)
            b0, b1 = _get_endpoints(b)
            if a1 is None or b0 is None:
                continue
            if float(np.linalg.norm(a1 - b0)) > join_tol:
                continue
            ori_a = a.get('_ori', None)
            ori_b = b.get('_ori', None)
            if ori_a == 'h' and ori_b == 'v':
                join = np.array([float(b0[0]), float(a1[1])], dtype=float)
                _set_end(a, join)
                _set_start(b, join)
            elif ori_a == 'v' and ori_b == 'h':
                join = np.array([float(a1[0]), float(b0[1])], dtype=float)
                _set_end(a, join)
                _set_start(b, join)
            else:
                join = (a1 + b0) / 2.0
                _set_end(a, join)
                _set_start(b, join)

        if len(segs) >= 2:
            first = segs[0]
            last = segs[-1]
            f0, f1 = _get_endpoints(first)
            l0, l1 = _get_endpoints(last)
            if f0 is not None and l1 is not None and float(np.linalg.norm(l1 - f0)) <= join_tol:
                ori_f = first.get('_ori', None)
                ori_l = last.get('_ori', None)
                if ori_l == 'h' and ori_f == 'v':
                    join = np.array([float(f0[0]), float(l1[1])], dtype=float)
                    _set_end(last, join)
                    _set_start(first, join)
                elif ori_l == 'v' and ori_f == 'h':
                    join = np.array([float(l1[0]), float(f0[1])], dtype=float)
                    _set_end(last, join)
                    _set_start(first, join)
                else:
                    join = (l1 + f0) / 2.0
                    _set_end(last, join)
                    _set_start(first, join)

        merged: List[Dict] = []
        i = 0
        while i < len(segs):
            cur = segs[i]
            if i + 1 < len(segs):
                nxt = segs[i + 1]
                if cur.get('type') == 'line' and nxt.get('type') == 'line':
                    cur_end = _to_pt(cur.get('p1', cur.get('end')))
                    nxt_start = _to_pt(nxt.get('p0', nxt.get('start')))
                    if float(np.linalg.norm(cur_end - nxt_start)) <= join_tol and cur.get('_ori', None) == nxt.get('_ori', None) and cur.get('_ori', None) in ('h', 'v'):
                        p0 = _to_pt(cur.get('p0', cur.get('start')))
                        p1 = _to_pt(nxt.get('p1', nxt.get('end')))
                        pts_a = cur.get('points', None)
                        pts_b = nxt.get('points', None)
                        if isinstance(pts_a, list) and isinstance(pts_b, list) and len(pts_a) >= 2 and len(pts_b) >= 2:
                            pts = np.vstack([np.asarray(pts_a, dtype=float).reshape(-1, 2), np.asarray(pts_b, dtype=float).reshape(-1, 2)[1:]])
                            pts[0] = p0
                            pts[-1] = p1
                            pts_list = pts.tolist()
                        else:
                            pts_list = [p0.tolist(), p1.tolist()]

                        new_seg = dict(cur)
                        new_seg['p0'] = p0.tolist()
                        new_seg['p1'] = p1.tolist()
                        new_seg['start'] = p0.tolist()
                        new_seg['end'] = p1.tolist()
                        new_seg['points'] = pts_list
                        merged.append(new_seg)
                        i += 2
                        continue

                if cur.get('type') == 'arc' and nxt.get('type') == 'arc':
                    cur_c = cur.get('center', None)
                    nxt_c = nxt.get('center', None)
                    cur_r = cur.get('radius', None)
                    nxt_r = nxt.get('radius', None)
                    if cur_c is not None and nxt_c is not None and cur_r is not None and nxt_r is not None:
                        c1 = _to_pt(cur_c)
                        c2 = _to_pt(nxt_c)
                        if float(np.linalg.norm(c1 - c2)) <= 3.0 and abs(float(cur_r) - float(nxt_r)) <= 3.0:
                            pts_a = cur.get('points', None)
                            pts_b = nxt.get('points', None)
                            if isinstance(pts_a, list) and isinstance(pts_b, list) and len(pts_a) >= 3 and len(pts_b) >= 3:
                                pts = np.vstack([np.asarray(pts_a, dtype=float).reshape(-1, 2), np.asarray(pts_b, dtype=float).reshape(-1, 2)[1:]])
                                new_seg = dict(cur)
                                new_seg['points'] = pts.tolist()
                                ang = arc_angles_from_points(_to_pt(new_seg['center']), pts)
                                new_seg['start_angle'] = float(ang.get('start_angle', 0.0))
                                new_seg['end_angle'] = float(ang.get('end_angle', 0.0))
                                new_seg['ccw'] = bool(ang.get('ccw', True))
                                merged.append(new_seg)
                                i += 2
                                continue

            merged.append(cur)
            i += 1

        for s in merged:
            if '_ori' in s:
                s.pop('_ori', None)
        return merged
    
    def build_nurbs_segment(self, points: np.ndarray, 
                           hierarchy_level: int = 0,
                           use_simple_knots: bool = False) -> NURBS.Curve:
        """
        构建NURBS曲线段（使用弧长参数化和曲率驱动权重）
        
        参数:
            points: 控制点数组 (N, 2)
            hierarchy_level: 层级（0=外轮廓，1=内孔，2=装饰）
            use_simple_knots: 是否使用简单的均匀节点向量（参考NURBS拟合，提高拓扑一致性）
        
        返回:
            NURBS曲线对象
        """
        if len(points) < 2:
            raise ValueError("控制点数量不足")
        
        # 根据层级调整阶数
        if hierarchy_level == 0:
            # 外轮廓：高精度
            degree = self.__degree
        elif hierarchy_level == 1:
            # 内孔：中等精度
            degree = max(2, self.__degree - 1)
        else:
            # 装饰：低精度
            degree = 2
        
        # 确保阶数不超过控制点数量
        degree = min(degree, len(points) - 1)
        if degree < 1:
            degree = 1
        
        # 创建NURBS曲线
        curve = NURBS.Curve(normalize_kv=True)
        curve.degree = degree
        
        # 设置控制点
        curve.ctrlpts = points.tolist()
        
        # 【改进拓扑一致性】优先使用简单节点向量（提高稳定性）
        if use_simple_knots:
            # 参考NURBS拟合的简单稳定方法
            import geomdl.knotvector
            from geomdl import utilities
            # 生成均匀节点向量（类似NURBS拟合的__generate_knots）
            knots = self._generate_uniform_knots(len(points), degree)
            if geomdl.knotvector.check(degree, knots, len(points)):
                curve.knotvector = knots
            else:
                # 回退到默认节点向量
                curve.knotvector = utilities.generate_knot_vector(degree, len(points))
            # 使用全1权重（参考NURBS拟合，提高稳定性）
            curve.weights = (np.ones(len(points)) + 1e-9).tolist()
        else:
            # 【改进】改进的弧长参数化：更稳定的实现
            # 计算弧长参数
            arc_length_params = self.arc_length_param(points)
            
            # 【改进】使用更稳定的节点向量生成方法
            # 如果弧长参数分布不均匀，使用简单方法
            if len(arc_length_params) > 2:
                param_diff = np.diff(arc_length_params)
                if np.std(param_diff) / (np.mean(param_diff) + 1e-8) > 0.5:
                    # 参数分布不均匀，使用简单方法
                    knots = self._generate_uniform_knots(len(points), degree)
                    curve.knotvector = knots
                    curve.weights = (np.ones(len(points)) + 1e-9).tolist()
                else:
                    # 参数分布均匀，使用弧长参数化
                    knot_vector = self._generate_arc_length_knot_vector(
                        arc_length_params, degree, len(points)
                    )
                    curve.knotvector = knot_vector
                    # 使用改进的曲率驱动权重（限制上限）
                    curve.weights = self.curvature_driven_weights(points).tolist()
            else:
                # 点数太少，使用简单方法
                knots = self._generate_uniform_knots(len(points), degree)
                curve.knotvector = knots
                curve.weights = (np.ones(len(points)) + 1e-9).tolist()
        
        return curve
    
    def _generate_uniform_knots(self, len_ctrl: int, degree: int, start: float = 0.0, stop: float = 1.0) -> List[float]:
        """
        生成均匀节点向量（参考NURBS拟合的__generate_knots方法）
        
        参数:
            len_ctrl: 控制点数量
            degree: 阶数
            start: 起始值（默认0.0）
            stop: 结束值（默认1.0）
        
        返回:
            节点向量列表
        """
        knots = [
            *[start for _ in range(degree)],
            *np.linspace(start, stop, len_ctrl - degree + 1).tolist(),
            *[stop for _ in range(degree)]
        ]
        return knots
    
    def __optimized_nurbs_process(self, show=True):
        """
        优化的NURBS拟合流程
        """
        if not self.get_contours():
            raise ValueError("未检测到轮廓，请检查图片或调整阈值参数")
        
        # 【改进】基于轮廓特征过滤，不依赖索引奇偶性
        all_contours = self.get_contours()
        valid_contours, valid_indices = ContourPreprocessor.filter_contours_by_features(
            all_contours,
            self.get_hierarchy(),
            min_area=5.0,  # 最小面积（像素²）
            min_perimeter=10.0,  # 最小周长（像素）
            min_points=2  # 最小点数（更宽松）
        )
        
        for i, contour in zip(valid_indices, valid_contours):
            # 【改进】使用预处理器预处理轮廓
            contour_points = ContourPreprocessor.preprocess_contour(contour, min_points=2)
            if contour_points is None:
                print(f"   [WARN] 轮廓 {i} 预处理失败，跳过")
                continue
            
            len_contour = len(contour_points)
            
            # 获取层级信息（用于调整拟合强度）
            hierarchy_level = 0  # 默认外轮廓
            if self.get_hierarchy() is not None and len(self.get_hierarchy()[0]) > i:
                h = self.get_hierarchy()[0][i]
                if h[3] > -1:
                    # 有父轮廓，是内孔或装饰
                    hierarchy_level = 1
            
            if self.use_segmentation:
                # 分段拟合模式（Line/Arc/NURBS混合）
                result = self._fit_segmented_contour_with_fallback(
                    contour_points, i, hierarchy_level
                )
            else:
                # 传统模式（整条轮廓一个NURBS，但使用优化权重）
                result = self._fit_whole_contour_with_fallback(
                    contour_points, i, hierarchy_level
                )
            
            # 存储结果（兼容原格式）
            if result:
                self.use_write_dict(
                    f'{i}',
                    contour.copy(),
                    result['len_control'],
                    result['control_points'],
                    len_contour,
                    result['fitted_points'],
                    result['mse_loss'],
                    result['step']
                )
                
                # 存储分段信息（如果使用分段模式）
                if self.use_segmentation and 'segments' in result:
                    self._segments_info[f'{i}'] = result['segments']
        
        if show:
            self.use_result_visual(self.__degree)
    
    def _validate_contour(self, contour_points: np.ndarray) -> bool:
        """
        验证轮廓是否有效
        
        参数:
            contour_points: 轮廓点数组
        
        返回:
            是否有效
        """
        if contour_points is None:
            return False
        contour_points = np.asarray(contour_points, dtype=float)
        if len(contour_points.shape) != 2 or contour_points.shape[1] != 2:
            return False
        if len(contour_points) < 3:
            return False
        # 检查是否有重复点
        if len(np.unique(contour_points, axis=0)) < 3:
            return False
        return True
    
    def _fit_segmented_contour_with_fallback(self, contour_points: np.ndarray,
                             contour_id: int, hierarchy_level: int) -> Optional[Dict]:
        """
        分段拟合轮廓（带回退机制）
        
        参数:
            contour_points: 轮廓点数组
            contour_id: 轮廓ID
            hierarchy_level: 层级
        
        返回:
            拟合结果字典
        """
        # 尝试主要拟合方法
        try:
            result = self._fit_segmented_contour(contour_points, contour_id, hierarchy_level)
            if result and len(result.get('segments', [])) > 0:
                return result
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 分段拟合失败，尝试回退: {e}")
        
        # 回退方法：使用简单分段
        try:
            if len(contour_points) >= 2:
                return {
                    'len_control': len(contour_points),
                    'control_points': contour_points,
                    'fitted_points': contour_points,
                    'mse_loss': 0.0,
                    'step': 1,
                    'segments': [{
                        'type': 'spline',
                        'points': contour_points.tolist(),
                        'error': 0.0
                    }]
                }
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 回退方法失败: {e}")
        
        return None
    
    def _fit_whole_contour_with_fallback(self, contour_points: np.ndarray,
                          contour_id: int, hierarchy_level: int) -> Optional[Dict]:
        """
        整条轮廓拟合（带回退机制）
        
        参数:
            contour_points: 轮廓点数组
            contour_id: 轮廓ID
            hierarchy_level: 层级
        
        返回:
            拟合结果字典
        """
        # 尝试主要拟合方法
        try:
            result = self._fit_whole_contour(contour_points, contour_id, hierarchy_level)
            if result:
                return result
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 整条拟合失败，尝试回退: {e}")
        
        # 回退方法：使用原始轮廓
        try:
            if len(contour_points) >= 2:
                return {
                    'len_control': len(contour_points),
                    'control_points': contour_points,
                    'fitted_points': contour_points,
                    'mse_loss': 0.0,
                    'step': 1
                }
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 回退方法失败: {e}")
        
        return None
    
    def _fit_segmented_contour(self, contour_points: np.ndarray, 
                             contour_id: int, hierarchy_level: int) -> Optional[Dict]:
        """
        分段拟合轮廓（Line/Arc/NURBS混合表示，改进版：添加错误处理和回退）
        
        参数:
            contour_points: 轮廓点数组
            contour_id: 轮廓ID
            hierarchy_level: 层级
        
        返回:
            拟合结果字典
        """
        # 【改进】验证轮廓有效性（更宽松）
        if not self._validate_contour(contour_points):
            # 即使验证失败，也尝试使用预处理后的点
            print(f"   [WARN] 轮廓 {contour_id} 验证失败，使用简单分段")
            if len(contour_points) >= 2:
                return {
                    'len_control': len(contour_points),
                    'control_points': contour_points,
                    'fitted_points': contour_points,
                    'mse_loss': 0.0,
                    'step': 1,
                    'segments': [{
                        'type': 'spline',
                        'points': contour_points.tolist(),
                        'error': 0.0
                    }]
                }
            return None

        try:
            contour_points = np.asarray(contour_points, dtype=float).reshape(-1, 2)
            if len(contour_points) >= 20:
                is_closed = float(np.linalg.norm(contour_points[0] - contour_points[-1])) <= 2.0
                if is_closed:
                    curvature = self.discrete_curvature(contour_points)
                    if isinstance(curvature, np.ndarray) and len(curvature) == len(contour_points):
                        k = int(np.argmax(curvature))
                        if 0 < k < len(contour_points) - 1:
                            contour_points = np.vstack([contour_points[k:], contour_points[:k]])
        except Exception:
            pass
        
        # Step 1: 初步分段（基于曲率，添加错误处理）
        try:
            initial_segments = self.initial_segmentation(contour_points)
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 分段失败，尝试整条拟合: {e}")
            # 回退到整条轮廓拟合
            return self._fit_whole_contour(contour_points, contour_id, hierarchy_level)
        
        # Step 2: 对每段进行分类和拟合
        all_fitted_points = []
        all_control_points = []
        segments_info = []
        
        for seg_idx, seg_points in enumerate(initial_segments):
            if len(seg_points) < 2:
                continue
            
            # 【改进】验证分段有效性
            if not self._validate_contour(seg_points):
                continue
            
            # 确保seg_points是2D数组，形状为(N, 2)
            try:
                seg_points = np.asarray(seg_points, dtype=float)
                if len(seg_points.shape) == 1:
                    # 一维数组，跳过
                    continue
                if seg_points.shape[1] != 2:
                    # 如果不是2列，尝试修复
                    if seg_points.shape[0] == 2 and len(seg_points.shape) == 2:
                        seg_points = seg_points.T
                    else:
                        # 无法修复，跳过
                        continue
            except Exception as e:
                print(f"   [WARN] 轮廓 {contour_id} 分段 {seg_idx} 处理失败，跳过: {e}")
                continue
            
            seg_points_for_line = seg_points
            if len(seg_points_for_line) > 300:
                step = int(np.ceil(len(seg_points_for_line) / 300))
                seg_points_for_line = seg_points_for_line[::max(1, step)]

            max_res_th = float(np.clip(float(self.line_threshold) * 0.85, 0.8, 1.8))
            is_line, line_info = is_line_segment(
                seg_points_for_line,
                linearity_threshold=0.02,
                max_residual_threshold=max_res_th,
                total_angle_threshold_deg=12.0
            )

            if is_line:
                try:
                    seg_pts_raw = np.asarray(seg_points, dtype=float).reshape(-1, 2)
                    if len(seg_pts_raw) >= 2:
                        chord_len = float(np.linalg.norm(seg_pts_raw[-1] - seg_pts_raw[0]))
                        poly_len = float(np.sum(np.linalg.norm(np.diff(seg_pts_raw, axis=0), axis=1)))
                    else:
                        chord_len = 0.0
                        poly_len = 0.0

                    max_residual = float(line_info.get('max_residual', 0.0))
                    total_angle_deg = float(line_info.get('total_angle_deg', 0.0))

                    if chord_len < 18.0 or poly_len < 22.0:
                        is_line = bool(max_residual <= 0.35 and total_angle_deg <= 3.0)
                    elif chord_len < 30.0 or poly_len < 35.0:
                        is_line = bool(max_residual <= 0.55 and total_angle_deg <= 6.0)
                except Exception:
                    pass

            if is_line:
                try:
                    is_line_ok, _ = self.fit_line(seg_points)
                    if is_line_ok:
                        dispatch_result = {'info': line_info, 'reason': 'line_priority'}
                        decision = "line"
                    else:
                        decision = "nurbs"
                        dispatch_result = {'info': {}, 'reason': 'line_fit_rejected'}
                except Exception:
                    decision = "nurbs"
                    dispatch_result = {'info': {}, 'reason': 'line_fit_exception'}
            else:
                simplified_points = seg_points
                if len(simplified_points) > 250:
                    step = int(np.ceil(len(simplified_points) / 250))
                    simplified_points = simplified_points[::max(1, step)]
                try:
                    approx = cv2.approxPolyDP(
                        np.asarray(simplified_points, dtype=np.float32).reshape(-1, 1, 2),
                        1.0,
                        False
                    ).reshape(-1, 2).astype(float)
                    if len(approx) >= 8:
                        simplified_points = approx
                except Exception:
                    pass

                decision, dispatch_result = dispatch_segment(
                    simplified_points,
                    linearity_threshold=0.0,
                    max_residual_threshold=0.0,
                    total_angle_threshold_deg=0.0,
                    sigma_r_max=0.02,
                    angle_span_min_deg=20.0,
                    num_points_min=8,
                    arc_gain_th=1.5,
                    arc_ctrl=3,
                    nurbs_ctrl=8
                )
            
            if decision == "line":
                # Line段
                line_info = dispatch_result['info']
                try:
                    seg_start = np.asarray(seg_points[0], dtype=float).reshape(2)
                    seg_end = np.asarray(seg_points[-1], dtype=float).reshape(2)
                    p0 = seg_start
                    p1 = seg_end
                except Exception:
                    p0 = np.asarray(line_info['p0'], dtype=float).reshape(2)
                    p1 = np.asarray(line_info['p1'], dtype=float).reshape(2)
                
                try:
                    v = p1 - p0
                    vn = float(np.linalg.norm(v))
                    if vn > 1e-9:
                        vu = v / vn
                        proj = (seg_points - p0) @ vu
                        closest = p0 + np.outer(proj, vu)
                        line_err = float(np.max(np.linalg.norm(seg_points - closest, axis=1)))
                    else:
                        line_err = 0.0
                except Exception:
                    line_err = float(line_info.get('max_residual', 0.0))

                # 生成直线点
                num_points = len(seg_points)
                t_line = np.linspace(0, 1, num_points)
                line_points = p0 + np.outer(t_line, p1 - p0)
                all_fitted_points.append(line_points)
                all_control_points.append(np.array([p0, p1]))
                segments_info.append({
                    'type': 'line',
                    'p0': p0.tolist(),
                    'p1': p1.tolist(),
                    'points': seg_points.tolist(),  # 添加points用于评估
                    'start': p0.tolist(),
                    'end': p1.tolist(),
                    'error': line_err,
                    'dispatch_reason': dispatch_result['reason']
                })
                continue
            
            elif decision == "arc":
                # Arc段
                arc_info = dispatch_result['info']
                center = arc_info["center"]
                radius = arc_info["radius"]
                angle_info = arc_angles_from_points(center, simplified_points)
                
                # 使用原始点作为Arc点
                arc_pts = seg_points
                all_fitted_points.append(arc_pts)
                all_control_points.append(seg_points[::max(1, len(seg_points)//10)])
                try:
                    dists = np.linalg.norm(seg_points - center.reshape(1, 2), axis=1)
                    arc_err = float(np.max(np.abs(dists - float(radius))))
                except Exception:
                    arc_err = float(arc_info.get('sigma_r', 0.0))
                segments_info.append({
                    'type': 'arc',
                    'center': center.tolist(),
                    'radius': float(radius),
                    'points': seg_points.tolist(),  # 添加points用于评估
                    'error': arc_err,
                    'angle_span': arc_info.get('angle_span', 0.0),
                    'sigma_r': arc_info.get('sigma_r', 0.0),
                    'start_angle': angle_info.get('start_angle', 0.0),
                    'end_angle': angle_info.get('end_angle', 0.0),
                    'ccw': bool(angle_info.get('ccw', True)),
                    'dispatch_reason': dispatch_result['reason']
                })
                continue
            
            # 否则使用NURBS（decision == "nurbs"）
            
            # 【优化】NURBS拟合前强制类型约束
            # 不要让NURBS算法去拟合直线段（即使它能拟合）
            # 这样导出的DXF才是POLYLINE而不是SPLINE
            
            def _is_forced_line(points_arr: np.ndarray) -> bool:
                pts = np.asarray(points_arr, dtype=float).reshape(-1, 2)
                if len(pts) < 2:
                    return False
                p0 = pts[0]
                p1 = pts[-1]
                v = p1 - p0
                v_norm = float(np.linalg.norm(v))
                if v_norm < 25.0:
                    return False
                if v_norm < 1e-6:
                    return False
                v = v / v_norm
                proj = np.dot(pts - p0, v)
                closest_points = p0 + np.outer(proj, v)
                errors = np.linalg.norm(pts - closest_points, axis=1)
                max_error = float(np.max(errors))
                rel_error = max_error / (v_norm + 1e-9)

                if len(pts) >= 3:
                    p_prev = pts[:-2]
                    p_mid = pts[1:-1]
                    p_next = pts[2:]
                    v1 = p_mid - p_prev
                    v2 = p_next - p_mid
                    dot = np.sum(v1 * v2, axis=1)
                    n1 = np.linalg.norm(v1, axis=1)
                    n2 = np.linalg.norm(v2, axis=1)
                    cosv = np.clip(dot / (n1 * n2 + 1e-8), -1.0, 1.0)
                    angles = np.arccos(cosv)
                    total_angle_deg = float(np.degrees(np.sum(np.abs(angles))))
                else:
                    total_angle_deg = 0.0

                err_th = min(float(self.line_threshold) * 0.75, 1.0)
                return bool(max_error < err_th and rel_error < 0.06 and total_angle_deg < 8.0)

            if _is_forced_line(seg_points):
                # 如果确实是直线，强制使用Line（不进入NURBS）
                try:
                    seg_start = np.asarray(seg_points[0], dtype=float).reshape(2)
                    seg_end = np.asarray(seg_points[-1], dtype=float).reshape(2)
                    p0 = seg_start
                    p1 = seg_end
                except Exception:
                    p0 = np.asarray(seg_points[0], dtype=float).reshape(2)
                    p1 = np.asarray(seg_points[-1], dtype=float).reshape(2)
                try:
                    v = p1 - p0
                    vn = float(np.linalg.norm(v))
                    if vn > 1e-9:
                        vu = v / vn
                        proj = (seg_points - p0) @ vu
                        closest = p0 + np.outer(proj, vu)
                        line_err = float(np.max(np.linalg.norm(seg_points - closest, axis=1)))
                    else:
                        line_err = 0.0
                except Exception:
                    line_err = 0.0
                num_points = len(seg_points)
                t_line = np.linspace(0, 1, num_points)
                line_points = p0 + np.outer(t_line, p1 - p0)
                all_fitted_points.append(line_points)
                all_control_points.append(np.array([p0, p1]))
                segments_info.append({
                    'type': 'line',
                    'p0': p0.tolist(),
                    'p1': p1.tolist(),
                    'points': seg_points.tolist(),  # 添加points用于评估
                    'start': p0.tolist(),
                    'end': p1.tolist(),
                    'error': line_err,
                    'dispatch_reason': 'forced_line_before_nurbs'
                })
                continue
            
            # 否则使用NURBS拟合（真正的自由曲线）
            # 【优化】减少控制点（Control Point Reduction）
            # 工业软件不喜欢成百上千个控制点的样条
            # 使用自适应step策略，但更激进地减少控制点
            step = self.get_default_step()
            best_result = None
            best_mse = float('inf')
            
            # 【优化】增加最大控制点数量限制
            max_control_points = min(20, len(seg_points) // 3)  # 最多20个控制点
            
            # 【性能优化】确保seg_points是numpy数组（提前处理，避免在循环中重复处理）
            seg_points_arr = np.asarray(seg_points, dtype=float)
            if len(seg_points_arr.shape) == 1:
                # 如果是1D数组，跳过（应该是2D数组）
                seg_points_arr = None
            elif seg_points_arr.shape[1] != 2:
                # 如果不是2列，尝试转置
                if seg_points_arr.shape[0] == 2:
                    seg_points_arr = seg_points_arr.T
                else:
                    seg_points_arr = None
            
            if seg_points_arr is None or len(seg_points_arr) < 3:
                # 如果seg_points无效，跳过NURBS拟合
                continue
            
            # 【性能优化】限制迭代次数，避免过慢
            max_iterations = 10
            iteration_count = 0
            
            while step > 1 and iteration_count < max_iterations:
                iteration_count += 1
                
                if len(seg_points_arr) % step == 0:
                    control_points = seg_points_arr[::int(step)]
                    if len(control_points) < 2:
                        step = max(1, step - 1)
                        continue
                else:
                    control_points = seg_points_arr[::int(step)]
                    if len(control_points) < 2:
                        step = max(1, step - 1)
                        continue

                try:
                    if len(control_points) >= 1:
                        if not np.allclose(control_points[0], seg_points_arr[0], atol=1e-12):
                            control_points = np.vstack([seg_points_arr[0].reshape(1, 2), control_points])
                        if not np.allclose(control_points[-1], seg_points_arr[-1], atol=1e-12):
                            control_points = np.vstack([control_points, seg_points_arr[-1].reshape(1, 2)])
                        if len(control_points) >= 2:
                            diffs = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
                            keep_mask = np.ones(len(control_points), dtype=bool)
                            keep_mask[1:] = diffs > 1e-12
                            control_points = control_points[keep_mask]
                except Exception:
                    pass
                
                # 确保控制点数量足够
                min_control_points = self.__degree + 1
                if len(control_points) < min_control_points:
                    step = max(1, step - 1)
                    continue
                
                # 【优化】限制最大控制点数量
                if len(control_points) > max_control_points:
                    # 进一步降采样
                    step = int(len(seg_points_arr) / max_control_points)
                    control_points = seg_points_arr[::max(1, step)]
                    if len(control_points) < min_control_points:
                        step = max(1, step - 1)
                        continue

                    try:
                        if len(control_points) >= 1:
                            if not np.allclose(control_points[0], seg_points_arr[0], atol=1e-12):
                                control_points = np.vstack([seg_points_arr[0].reshape(1, 2), control_points])
                            if not np.allclose(control_points[-1], seg_points_arr[-1], atol=1e-12):
                                control_points = np.vstack([control_points, seg_points_arr[-1].reshape(1, 2)])
                            if len(control_points) >= 2:
                                diffs = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
                                keep_mask = np.ones(len(control_points), dtype=bool)
                                keep_mask[1:] = diffs > 1e-12
                                control_points = control_points[keep_mask]
                    except Exception:
                        pass
                
                try:
                    # 【改进拓扑一致性】默认使用简单节点向量（提高稳定性）
                    # 分段拟合时，使用简单方法可以保证连接处的连续性
                    if self.use_simple_knots_by_default:
                        use_simple = True  # 默认使用简单方法
                    else:
                        use_simple = (len(control_points) >= 8)  # 控制点较多时使用简单方法
                    
                    # 构建NURBS曲线
                    curve = self.build_nurbs_segment(control_points, hierarchy_level, use_simple_knots=use_simple)
                    
                    # 生成拟合点
                    curve.sample_size = len(seg_points_arr)
                    fitted_points = np.asarray(curve.evalpts)
                    try:
                        if isinstance(fitted_points, np.ndarray) and fitted_points.ndim == 2 and fitted_points.shape[1] == 2:
                            fitted_points[0] = seg_points_arr[0]
                            fitted_points[-1] = seg_points_arr[-1]
                    except Exception:
                        pass
                    
                    # 【性能优化】使用向量化计算误差，而不是循环
                    try:
                        fitted_points = np.asarray(fitted_points, dtype=float).reshape(-1, 2)
                        if len(fitted_points) == len(seg_points_arr):
                            mse = float(np.mean(np.linalg.norm(fitted_points - seg_points_arr, axis=1)))
                        else:
                            from scipy.spatial.distance import cdist
                            dists = cdist(seg_points_arr, fitted_points)
                            errors = np.min(dists, axis=1)
                            mse = float(np.mean(errors))
                    except Exception:
                        mse = float('inf')
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_result = {
                            'control_points': control_points,
                            'fitted_points': fitted_points,
                            'curve': curve,
                            'step': step
                        }
                    
                    # 如果误差足够小，停止
                    if mse < self.get_default_threshold():
                        break
                except Exception as e:
                    pass
                
                # 减小step
                d_step = self.get_default_lr() * best_mse / self.get_default_delta()
                step = 1 if step < 1 else int(step - d_step)

            if best_result is None:
                try:
                    sp_raw = np.asarray(seg_points, dtype=float).reshape(-1, 2)
                    if len(sp_raw) >= 2:
                        all_fitted_points.append(sp_raw)
                        all_control_points.append(sp_raw[::max(1, len(sp_raw) // 10)])
                        segments_info.append({
                            'type': 'spline',
                            'points': seg_points.tolist(),
                            'start': sp_raw[0].tolist(),
                            'end': sp_raw[-1].tolist(),
                            'error': float('inf'),
                            'dispatch_reason': 'nurbs_fit_failed_fallback'
                        })
                except Exception:
                    pass
                continue

            fp_raw = np.asarray(best_result['fitted_points'], dtype=float)
            if fp_raw.ndim == 2 and fp_raw.shape[1] >= 2:
                fp = fp_raw[:, :2].copy()
            else:
                fp = fp_raw.reshape(-1, 2)

            sp_raw = np.asarray(seg_points, dtype=float)
            if sp_raw.ndim == 2 and sp_raw.shape[1] >= 2:
                sp = sp_raw[:, :2].copy()
            else:
                sp = sp_raw.reshape(-1, 2)
            if len(fp) >= 2 and len(sp) >= 2:
                try:
                    d0 = float(np.linalg.norm(fp[0] - sp[0]))
                    d1 = float(np.linalg.norm(fp[-1] - sp[0]))
                    if d1 + 0.5 < d0:
                        fp = fp[::-1].copy()
                except Exception:
                    pass
                try:
                    fp[0] = sp[0]
                    fp[-1] = sp[-1]
                except Exception:
                    pass

            all_fitted_points.append(fp)
            all_control_points.append(best_result['control_points'])
            segments_info.append({
                'type': 'nurbs',
                'degree': best_result['curve'].degree,
                'ctrlpts_count': len(best_result['control_points']),
                'points': seg_points.tolist(),
                'start': sp[0].tolist() if isinstance(sp, np.ndarray) and sp.ndim == 2 and len(sp) >= 1 else seg_points[0].tolist(),
                'end': sp[-1].tolist() if isinstance(sp, np.ndarray) and sp.ndim == 2 and len(sp) >= 1 else seg_points[-1].tolist(),
                'control_points': best_result['control_points'].tolist(),
                'weights': best_result['curve'].weights,
                'error': float(best_mse) if np.isfinite(best_mse) else float('inf')
            })
        
        # 组合所有拟合点（确保连接处连续）
        if len(all_fitted_points) > 0:
            # 【改进拓扑一致性】确保分段连接处连续
            fitted_points_list = []
            for i, fp in enumerate(all_fitted_points):
                fp = np.asarray(fp, dtype=float)
                # 【修复】确保fp是2D数组
                if len(fp.shape) == 1:
                    # 如果是1D数组，尝试reshape
                    if fp.size == 2:
                        fp = fp.reshape(1, 2)
                    else:
                        continue
                elif fp.shape[1] != 2:
                    # 如果不是2列，尝试转置
                    if fp.shape[0] == 2:
                        fp = fp.T
                    else:
                        continue
                
                if i > 0 and len(fitted_points_list) > 0:
                    prev_fp = fitted_points_list[-1]
                    prev_fp = np.asarray(prev_fp, dtype=float)
                    if prev_fp.ndim == 1:
                        if prev_fp.size == 2:
                            prev_fp = prev_fp.reshape(1, 2)
                        else:
                            fitted_points_list.append(fp)
                            continue
                    elif prev_fp.ndim == 2 and prev_fp.shape[1] != 2:
                        if prev_fp.shape[0] == 2:
                            prev_fp = prev_fp.T
                        else:
                            fitted_points_list.append(fp)
                            continue

                    prev_end = np.asarray(prev_fp[-1], dtype=float).reshape(2)

                    fp2 = np.asarray(fp, dtype=float).copy()
                    if len(fp2) >= 2:
                        try:
                            d_start = float(np.linalg.norm(fp2[0] - prev_end))
                            d_end = float(np.linalg.norm(fp2[-1] - prev_end))
                            if d_end + 0.5 < d_start:
                                fp2 = fp2[::-1].copy()
                        except Exception:
                            pass

                    curr_start = np.asarray(fp2[0], dtype=float).reshape(2)
                    gap = float(np.linalg.norm(curr_start - prev_end))

                    if gap <= 1.0:
                        try:
                            fp2[0] = prev_end
                        except Exception:
                            pass
                        if len(fp2) >= 2:
                            try:
                                if float(np.linalg.norm(fp2[1] - fp2[0])) < 1e-9:
                                    fp2 = fp2[1:]
                            except Exception:
                                pass
                        fitted_points_list.append(fp2)
                    else:
                        try:
                            steps = int(np.ceil(gap / 0.75))
                            steps = max(2, min(steps, 2000))
                            bridge = np.linspace(prev_end, curr_start, steps + 1, dtype=float)
                            if bridge.shape[0] > 1:
                                fitted_points_list.append(bridge[1:])
                        except Exception:
                            pass
                        if len(fp2) >= 1:
                            try:
                                fp2[0] = curr_start
                            except Exception:
                                pass
                        if len(fp2) >= 2:
                            try:
                                if float(np.linalg.norm(fp2[1] - fp2[0])) < 1e-9:
                                    fp2 = fp2[1:]
                            except Exception:
                                pass
                        fitted_points_list.append(fp2)
                else:
                    fitted_points_list.append(fp)
            
            if len(fitted_points_list) > 0:
                fitted_points = np.vstack(fitted_points_list)
                # 【修复】确保all_control_points中的元素都是numpy数组
                valid_control_points = []
                for cp in all_control_points:
                    cp_arr = np.asarray(cp, dtype=float)
                    if len(cp_arr.shape) == 2 and cp_arr.shape[1] == 2:
                        valid_control_points.append(cp_arr)
                if len(valid_control_points) > 0:
                    control_points = np.vstack(valid_control_points)
                else:
                    control_points = contour_points[::max(1, len(contour_points)//10)]
            else:
                fitted_points = contour_points
                control_points = contour_points[::max(1, len(contour_points)//10)]
        else:
            fitted_points = contour_points
            control_points = contour_points[::max(1, len(contour_points)//10)]

        try:
            pts = np.asarray(fitted_points, dtype=float).reshape(-1, 2)
            if len(pts) >= 3:
                diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                keep = np.ones(len(pts), dtype=bool)
                keep[1:] = diffs > 1e-9
                pts = pts[keep]
                if len(pts) >= 3:
                    prev = np.roll(pts, 1, axis=0)
                    nxt = np.roll(pts, -1, axis=0)
                    d13 = np.linalg.norm(nxt - prev, axis=1)
                    d12 = np.linalg.norm(pts - prev, axis=1)
                    d23 = np.linalg.norm(nxt - pts, axis=1)
                    spike = (d13 <= 1e-6) & (d12 > 1e-6) & (d23 > 1e-6)
                    if bool(np.any(spike)):
                        pts = pts[~spike]
                fitted_points = pts
        except Exception:
            pass

        try:
            pts = np.asarray(fitted_points, dtype=float).reshape(-1, 2)
            if len(pts) >= 4:
                try:
                    contour_pts = np.asarray(contour_points, dtype=float).reshape(-1, 2)
                    expected_closed = len(contour_pts) >= 2 and float(np.linalg.norm(contour_pts[0] - contour_pts[-1])) <= 2.0
                except Exception:
                    expected_closed = bool(np.allclose(pts[0], pts[-1], atol=1.0))

                is_closed = bool(expected_closed)
                if is_closed and bool(np.allclose(pts[0], pts[-1], atol=1.0)):
                    pts = pts[:-1]

                def _orient(a, b, c) -> float:
                    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

                def _seg_intersect(p1, p2, p3, p4) -> bool:
                    o1 = _orient(p1, p2, p3)
                    o2 = _orient(p1, p2, p4)
                    o3 = _orient(p3, p4, p1)
                    o4 = _orient(p3, p4, p2)
                    eps = 1e-12
                    if abs(o1) <= eps or abs(o2) <= eps or abs(o3) <= eps or abs(o4) <= eps:
                        return False
                    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

                def _first_hit(poly: np.ndarray):
                    n = len(poly)
                    if n < 4:
                        return None
                    segs = [(i, (i + 1) % n) for i in range(n)] if is_closed else [(i, i + 1) for i in range(n - 1)]
                    for a in range(len(segs)):
                        i1, i2 = segs[a]
                        p1, p2 = poly[i1], poly[i2]
                        for b in range(a + 1, len(segs)):
                            j1, j2 = segs[b]
                            if i1 in (j1, j2) or i2 in (j1, j2):
                                continue
                            if abs(i1 - j1) <= 1 or abs(i1 - j2) <= 1 or abs(i2 - j1) <= 1 or abs(i2 - j2) <= 1:
                                continue
                            if _seg_intersect(p1, p2, poly[j1], poly[j2]):
                                return i1, i2, j1, j2
                    return None

                if is_closed and len(pts) <= 2000:
                    chk = pts
                    hit0 = None
                    if len(chk) > 800:
                        step = int(np.ceil(len(chk) / 800))
                        step = max(1, step)
                        chk0 = chk[::step]
                        chk1 = chk[(step // 2)::step] if step > 1 else chk0
                        hit0 = _first_hit(chk0)
                        if hit0 is None and len(chk1) >= 4:
                            hit0 = _first_hit(chk1)
                    else:
                        hit0 = _first_hit(chk)

                    if hit0 is not None:
                        def _downsample(arr: np.ndarray, max_n: int) -> np.ndarray:
                            arr = np.asarray(arr, dtype=float).reshape(-1, 2)
                            if len(arr) <= max_n:
                                return arr
                            s = int(np.ceil(len(arr) / max_n))
                            s = max(1, s)
                            return arr[::s]

                        def _sym_mad(a: np.ndarray, b: np.ndarray) -> float:
                            a = _downsample(a, 300)
                            b = _downsample(b, 300)
                            if len(a) == 0 or len(b) == 0:
                                return float('inf')
                            d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
                            da = float(np.mean(np.sqrt(np.min(d2, axis=1))))
                            d2b = np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=2)
                            db = float(np.mean(np.sqrt(np.min(d2b, axis=1))))
                            return 0.5 * (da + db)

                        contour_ds = None
                        try:
                            contour_ds = np.asarray(contour_points, dtype=float).reshape(-1, 2)
                            if len(contour_ds) >= 2 and float(np.linalg.norm(contour_ds[0] - contour_ds[-1])) <= 1.0:
                                contour_ds = contour_ds[:-1]
                        except Exception:
                            contour_ds = None

                        base_cost = float('inf')
                        if contour_ds is not None and len(contour_ds) >= 4:
                            base_cost = _sym_mad(pts, contour_ds)

                        poly = pts.copy()
                        for _ in range(4):
                            hit = _first_hit(poly)
                            if hit is None:
                                break
                            i1, _, j1, _ = hit
                            i = int(i1)
                            j = int(j1)
                            if i > j:
                                i, j = j, i
                            a = i + 1
                            b = j
                            if a < b:
                                poly[a:b + 1] = poly[a:b + 1][::-1]

                        accept = True
                        if contour_ds is not None and len(contour_ds) >= 4 and np.isfinite(base_cost):
                            new_cost = _sym_mad(poly, contour_ds)
                        accept = bool(new_cost <= base_cost * 1.2 or new_cost <= base_cost + 1.0)

                        if accept:
                            poly = np.vstack([poly, poly[0].reshape(1, 2)])
                            fitted_points = poly
        except Exception:
            pass

        try:
            contour_points_arr = np.asarray(contour_points, dtype=float).reshape(-1, 2)
            is_closed = len(contour_points_arr) >= 2 and float(np.linalg.norm(contour_points_arr[0] - contour_points_arr[-1])) <= 2.0
            fp = np.asarray(fitted_points, dtype=float).reshape(-1, 2)
            if is_closed and len(fp) >= 2:
                d = float(np.linalg.norm(fp[0] - fp[-1]))
                if d <= 2.0:
                    fp[-1] = fp[0]
                    fitted_points = fp
                else:
                    fitted_points = np.vstack([fp, fp[0].reshape(1, 2)])
        except Exception:
            pass
        
        # 计算总体误差
        errors = []
        # 【修复】确保contour_points是numpy数组
        contour_points_arr = np.asarray(contour_points, dtype=float)
        if len(contour_points_arr.shape) == 2 and contour_points_arr.shape[1] == 2:
            for cp in contour_points_arr:
                # 【修复】确保cp是numpy数组
                cp = np.asarray(cp, dtype=float).reshape(1, -1)
                if cp.shape[1] != 2:
                    continue
                dists = np.linalg.norm(fitted_points - cp, axis=1)
                errors.append(np.min(dists))
        mse_loss = np.mean(errors) if errors else float('inf')
        
        result = {
            'len_control': len(control_points),
            'control_points': control_points,
            'fitted_points': fitted_points,
            'mse_loss': mse_loss,
            'step': 1,  # 分段模式下step意义不大
            'segments': segments_info  # 分段信息（用于CAD导出）
        }
        
        # 存储分段信息到实例变量（用于后续获取）
        # 【改进】添加错误处理和回退
        try:
            # 【性能优化】只合并一次，避免重复调用
            if self.use_improved_merger and self.segment_merger is not None:
                contour_points_arr = np.asarray(contour_points, dtype=float).reshape(-1, 2)
                contour_length = float(np.sum(np.linalg.norm(np.diff(contour_points_arr, axis=0), axis=1))) if len(contour_points_arr) >= 2 else 0.0
                num_points = int(len(contour_points_arr))

                base_min_length = float(getattr(self.segment_merger, 'min_segment_length', 20.0))
                base_angle = float(getattr(self.segment_merger, 'angle_threshold_deg', 15.0))
                base_distance = float(getattr(self.segment_merger, 'distance_threshold', 5.0))
                base_max_segments = getattr(self.segment_merger, 'max_segments', None)
                base_err_factor = float(getattr(self.segment_merger, 'error_tolerance_factor', 1.5))

                if contour_length < 100.0:
                    adaptive_min_length = max(10.0, base_min_length * 0.5)
                    adaptive_angle = min(22.0, base_angle * 1.3)
                    adaptive_distance = min(6.5, base_distance * 1.2)
                elif contour_length > 1000.0:
                    adaptive_min_length = min(40.0, base_min_length * 1.6)
                    adaptive_angle = max(12.0, base_angle * 0.9)
                    adaptive_distance = min(8.0, base_distance * 1.4)
                else:
                    adaptive_min_length = base_min_length
                    adaptive_angle = base_angle
                    adaptive_distance = base_distance

                if num_points < 50:
                    adaptive_max_segments = min(5, int(base_max_segments or 15))
                elif num_points > 500:
                    adaptive_max_segments = min(12, int(base_max_segments or 24))
                else:
                    adaptive_max_segments = base_max_segments

                adaptive_merger = ImprovedSegmentMerger(
                    angle_threshold_deg=float(adaptive_angle),
                    distance_threshold=float(adaptive_distance),
                    min_segment_length=float(adaptive_min_length),
                    max_segments=adaptive_max_segments,
                    error_tolerance_factor=float(base_err_factor)
                )
                merged_segments = adaptive_merger.merge_segments(segments_info)
            else:
                merged_segments = self._merge_segments_info(segments_info)
            # 【改进】如果合并后没有段，使用原始分段
            if len(merged_segments) == 0:
                print(f"   [WARN] 轮廓 {contour_id} 合并后没有段，使用原始分段")
                merged_segments = segments_info
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 段合并失败，使用原始分段: {e}")
            merged_segments = segments_info
        
        try:
            merged_segments = self._regularize_segments_for_cad(merged_segments)
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} CAD正则化失败: {e}")
            # 继续使用未正则化的段
        
        # 【性能优化】如果段数仍然很多，再合并一次
        if len(merged_segments) > 20:
            try:
                merged_segments = self._merge_segments_info(merged_segments)
                merged_segments = self._regularize_segments_for_cad(merged_segments)
            except Exception as e:
                print(f"   [WARN] 轮廓 {contour_id} 二次合并失败: {e}")
        
        if len(merged_segments) > 20 and self.use_improved_merger and self.segment_merger is not None:
            try:
                max_keep = int(np.clip(10 + len(contour_points_arr) // 250, 14, 22))
                merged_segments = self.segment_merger._force_merge_to_limit(merged_segments, max_keep)
                merged_segments = self._regularize_segments_for_cad(merged_segments)
            except Exception as e:
                print(f"   [WARN] 轮廓 {contour_id} 段数上限合并失败: {e}")
        
        self._segments_info[str(contour_id)] = merged_segments
        result['segments'] = merged_segments
        
        return result

    def _merge_segments_info(self, segments: List[Dict]) -> List[Dict]:
        """
        合并分段信息（改进版：使用改进的段合并器）
        """
        if not segments or len(segments) < 2:
            return segments
        
        # 如果使用改进的段合并器，使用它
        if self.use_improved_merger and self.segment_merger is not None:
            return self.segment_merger.merge_segments(segments)
        
        # 否则使用原始合并逻辑（向后兼容）
        def _as_points(seg: Dict) -> np.ndarray:
            """安全地获取段的点数组"""
            pts = seg.get('points', None)
            if pts is None:
                # 尝试从start和end构建
                start = seg.get('start', seg.get('p0', None))
                end = seg.get('end', seg.get('p1', None))
                if start is not None and end is not None:
                    try:
                        start_arr = np.asarray(start, dtype=float).flatten()
                        end_arr = np.asarray(end, dtype=float).flatten()
                        if start_arr.size == 2 and end_arr.size == 2:
                            return np.array([start_arr, end_arr])
                    except Exception:
                        pass
                return np.zeros((0, 2), dtype=float)
            try:
                pts_arr = np.asarray(pts, dtype=float)
                # 确保是2D数组
                if len(pts_arr.shape) == 1:
                    if pts_arr.size == 2:
                        return pts_arr.reshape(1, 2)
                    else:
                        return np.zeros((0, 2), dtype=float)
                elif len(pts_arr.shape) == 2:
                    if pts_arr.shape[1] == 2:
                        return pts_arr
                    elif pts_arr.shape[0] == 2:
                        return pts_arr.T
                    else:
                        return np.zeros((0, 2), dtype=float)
                else:
                    return np.zeros((0, 2), dtype=float)
            except Exception:
                return np.zeros((0, 2), dtype=float)

        join_tol = 3.0
        changed = True
        current = segments
        while changed and len(current) >= 2:
            changed = False
            merged: List[Dict] = []
            i = 0
            while i < len(current):
                cur = current[i]
                if i + 1 >= len(current):
                    merged.append(cur)
                    break

                nxt = current[i + 1]
                cur_pts = _as_points(cur)
                nxt_pts = _as_points(nxt)
                if len(cur_pts) < 2 or len(nxt_pts) < 2:
                    merged.append(cur)
                    i += 1
                    continue

                if float(np.linalg.norm(cur_pts[-1] - nxt_pts[0])) > join_tol:
                    merged.append(cur)
                    i += 1
                    continue

                cur_type = cur.get('type')
                nxt_type = nxt.get('type')

                if cur_type == 'line' and nxt_type == 'line':
                    try:
                        pts = np.vstack([cur_pts, nxt_pts[1:]])
                        ok, line_res = self.fit_line(pts)
                        if ok and line_res is not None:
                            p0, p1 = line_res
                            # 【修复】确保p0和p1是有效的2D点
                            p0 = np.asarray(p0, dtype=float).flatten()
                            p1 = np.asarray(p1, dtype=float).flatten()
                            if p0.size != 2 or p1.size != 2:
                                merged.append(cur)
                                i += 1
                                continue
                            
                            merged.append({
                                'type': 'line',
                                'p0': p0.tolist(),
                                'p1': p1.tolist(),
                                'points': pts.tolist(),
                                'start': p0.tolist(),
                                'end': p1.tolist(),
                                'dispatch_reason': 'merged_line'
                            })
                            i += 2
                            changed = True
                            continue
                    except Exception as e:
                        # 如果合并失败，保留当前段
                        merged.append(cur)
                        i += 1
                        continue

                if cur_type == 'arc' and nxt_type == 'arc':
                    pts = np.vstack([cur_pts, nxt_pts[1:]])
                    ok, arc_res = self.fit_arc(pts)
                    if ok and arc_res is not None:
                        center, radius = arc_res
                        merged.append({
                            'type': 'arc',
                            'center': np.asarray(center, dtype=float).reshape(2,).tolist(),
                            'radius': float(radius),
                            'points': pts.tolist(),
                            'dispatch_reason': 'merged_arc'
                        })
                        i += 2
                        changed = True
                        continue

                merged.append(cur)
                i += 1

            current = merged

        return current
    
    def _fit_whole_contour(self, contour_points: np.ndarray,
                          contour_id: int, hierarchy_level: int) -> Optional[Dict]:
        """
        整条轮廓拟合（传统模式，但使用优化权重）
        
        参数:
            contour_points: 轮廓点数组
            contour_id: 轮廓ID
            hierarchy_level: 层级
        
        返回:
            拟合结果字典
        """
        len_contour = len(contour_points)
        step = self.get_default_step()
        
        while step > 1:
            # 设置控制点
            if len_contour % step == 0:
                control_points = contour_points.copy()[::int(step)]
                control_points = np.concatenate([control_points, contour_points[0].reshape(-1, 2)])
                len_control = len(control_points)
            else:
                control_points = contour_points.copy()[::int(step)]
                control_points[-1] = contour_points[0]
                len_control = len(control_points)
            
            # 检查控制点数量
            min_control_points = self.__degree + 1
            if len_control < min_control_points:
                if step > 1:
                    step = max(1, step - 1)
                    continue
                else:
                    print(f"警告: 轮廓 {contour_id} 的控制点数量不足，跳过NURBS拟合")
                    return None
            
            try:
                # 【改进拓扑一致性】默认使用简单节点向量（提高稳定性）
                if self.use_simple_knots_by_default:
                    use_simple = True  # 默认使用简单方法
                else:
                    use_simple = (len_control >= 10)  # 控制点较多时使用简单方法
                
                # 构建NURBS曲线
                curve = self.build_nurbs_segment(control_points, hierarchy_level, use_simple_knots=use_simple)
                
                # 生成拟合点
                curve.sample_size = len_contour
                new_contour = np.asarray(curve.evalpts)
                
                # 计算MSE误差
                mse_loss = np.mean((new_contour - contour_points) ** 2)
                
                # 达到误差要求或step为1则返回
                if mse_loss < self.get_default_threshold() or step == 1:
                    return {
                        'len_control': len_control,
                        'control_points': control_points,
                        'fitted_points': new_contour,
                        'mse_loss': mse_loss,
                        'step': step
                    }
                else:
                    d_step = self.get_default_lr() * mse_loss / self.get_default_delta()
                    step = 1 if step < 1 else int(step - d_step)
            except Exception as e:
                print(f"警告: 轮廓 {contour_id} NURBS拟合失败: {e}")
                step = max(1, step - 1)
        
        return None
    
    def get_segments_info(self) -> Dict:
        """
        获取分段信息（用于CAD导出和几何感知评分）
        
        返回:
            分段信息字典，格式：{contour_id: [segment1, segment2, ...]}
        """
        # 从实例变量中获取分段信息
        if hasattr(self, '_segments_info') and self._segments_info:
            return self._segments_info.copy()
        
        # 如果没有分段信息，返回空字典
        return {}


# 向后兼容的类名
OptimizedNURBSContour = OptimizedNURBSFitter


if __name__ == "__main__":
    from core.image.initializer import ImageInitializer
    
    # 测试
    II = ImageInitializer(r"D:\datasheet\test\test21.png")
    Img = II.centered_img()
    Edges = II.edges()
    
    ONF = OptimizedNURBSFitter(
        img=Img, 
        edges=Edges,
        use_segmentation=True,
        curvature_weight_alpha=5.0
    )
    print("优化NURBS拟合完成")
    print(f"检测到 {len(ONF.get_contours_dict())} 个轮廓")
