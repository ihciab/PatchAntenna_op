"""
优化的B样条轮廓拟合模块
基于几何语义的Line/Arc/Spline混合表示

优化点：
1. 弧长参数化（替代均匀角度参数化）
2. 曲率感知的分段拟合
3. Line/Arc/Spline自动分类
4. 层级感知的参数约束

原代码参考：core/geometry/bspline_fitter.py
"""

import cv2
import numpy as np
from scipy.interpolate import make_interp_spline
from typing import Dict, List, Tuple, Optional
from core.geometry.arc_detector import arc_quality
from core.geometry.arc_nurbs_scheduler import arc_nurbs_dispatch, build_simple_nurbs
from core.geometry.segment_scheduler import dispatch_segment, arc_angles_from_points
from core.geometry.improved_segment_merger import ImprovedSegmentMerger
from core.geometry.contour_preprocessor import ContourPreprocessor
from core.image.initializer import ImageInitializer


class OptimizedBSplineFitter:
    """
    优化的B样条轮廓拟合类
    实现几何语义感知的混合表示（Line/Arc/Spline）
    """
    
    def __init__(self, img, edges, 
                 line_threshold: float = 2.0,
                 arc_threshold: float = 2.0,
                 curvature_threshold: float = 0.15,
                 spline_degree: int = 3,
                 show: bool = True,
                 save: str = '',
                 # 改进的段合并参数
                 use_improved_merger: bool = True,
                 angle_threshold_deg: float = 15.0,
                 distance_threshold: float = 5.0,
                 min_segment_length: float = 20.0,
                 max_segments: Optional[int] = None,
                 error_tolerance_factor: float = 1.5):
        """
        初始化优化拟合器
        
        参数:
            img: 输入图像
            edges: 边缘图像
            line_threshold: 直线拟合误差阈值（像素）
            arc_threshold: 圆弧拟合误差阈值（像素）
            curvature_threshold: 曲率变化阈值（用于分段）
            spline_degree: B样条阶数
            show: 是否显示结果
            save: 保存路径
            use_improved_merger: 是否使用改进的段合并器
            angle_threshold_deg: 合并角度阈值（度）
            distance_threshold: 合并距离阈值（像素）
            min_segment_length: 最小段长度（像素）
            max_segments: 最大段数（None表示无限制）
            error_tolerance_factor: 误差容忍因子
        """
        self.img = img
        self.edges = edges
        self.line_threshold = line_threshold
        self.arc_threshold = arc_threshold
        self.curvature_threshold = curvature_threshold
        self.spline_degree = spline_degree
        self.show = show
        self.save = save
        
        # 初始化改进的段合并器
        self.use_improved_merger = use_improved_merger
        self._angle_threshold_deg = angle_threshold_deg
        self._distance_threshold = distance_threshold
        self._min_segment_length = min_segment_length
        self._max_segments = max_segments
        self._error_tolerance_factor = error_tolerance_factor
        
        if use_improved_merger:
            # 初始创建，但参数会根据轮廓自适应调整
            self.segment_merger = ImprovedSegmentMerger(
                angle_threshold_deg=angle_threshold_deg,
                distance_threshold=distance_threshold,
                min_segment_length=min_segment_length,
                max_segments=max_segments,
                error_tolerance_factor=error_tolerance_factor
            )
        else:
            self.segment_merger = None
        
        # 检测轮廓
        self.contours, self.hierarchy = cv2.findContours(
            self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        
        # 存储结果
        self.contours_dict = {}
        self.curves_tree = {}
        
        # 执行拟合
        self._fit_all_contours()
        self._build_curves_tree()
    
    def arc_length_param(self, points: np.ndarray) -> np.ndarray:
        """
        计算弧长参数化
        
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
            曲率估计值 (N-2,)
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
        
        # 避免除零
        cos_angles = np.clip(
            dot_products / (norms1 * norms2 + 1e-8),
            -1.0, 1.0
        )
        
        # 角度（与曲率成正相关）
        angles = np.arccos(cos_angles)
        
        return angles
    
    def initial_segmentation(self, points: np.ndarray) -> Tuple[List[np.ndarray], set]:
        """
        基于曲率变化进行初步分段（改进版：自适应曲率阈值）
        
        参数:
            points: 轮廓点数组 (N, 2)
        
        返回:
            (分段列表, 角点索引集合) 元组
        """
        if len(points) < 3:
            return [points], set()
        
        # 计算曲率
        curvature = self.discrete_curvature(points)
        
        if len(curvature) == 0:
            return [points], set()

        padded_curvature = np.pad(curvature, (1, 1), mode='edge')
        
        # 【改进】自适应曲率阈值：使用百分位数而不是固定阈值
        # 这样可以避免对噪声敏感，同时适应不同复杂度的轮廓
        if len(padded_curvature) > 0:
            # 使用75%分位数作为阈值（更保守，减少过度分段）
            adaptive_threshold = np.percentile(padded_curvature, 75)
            # 但不要低于原始阈值太多
            adaptive_threshold = max(adaptive_threshold, self.curvature_threshold * 0.8)
        else:
            adaptive_threshold = self.curvature_threshold
        
        # 计算曲率变化率
        dk = np.abs(np.diff(padded_curvature))
        
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
        # 简单轮廓（<200点）：较小最小点数
        # 复杂轮廓（>500点）：较大最小点数
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
        
        # 确保至少有一段
        if len(segments) == 0:
            segments = [points]
        
        return segments, corner_set
    
    def fit_line(self, points: np.ndarray) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
        """
        拟合直线段
        
        参数:
            points: 点数组 (N, 2)
        
        返回:
            (最大误差, (起点, 终点))
        """
        if len(points) < 2:
            return float('inf'), (points[0], points[-1])
        
        p0 = points[0]
        p1 = points[-1]
        
        # 方向向量
        v = p1 - p0
        v_norm = np.linalg.norm(v)
        
        if v_norm < 1e-6:
            # 点重合，返回高误差
            return float('inf'), (p0, p1)
        
        v = v / v_norm
        
        # 计算所有点到直线的距离
        proj = np.dot(points - p0, v)
        closest_points = p0 + np.outer(proj, v)
        errors = np.linalg.norm(points - closest_points, axis=1)
        
        max_error = np.max(errors)
        
        return max_error, (p0, p1)
    
    def fit_circle(self, points: np.ndarray) -> Tuple[float, Tuple[np.ndarray, float]]:
        """
        拟合圆弧段
        
        参数:
            points: 点数组 (N, 2)
        
        返回:
            (最大误差, (圆心, 半径))
        """
        if len(points) < 3:
            return float('inf'), (points[0], 0.0)
        
        x = points[:, 0]
        y = points[:, 1]
        
        # 使用最小二乘法拟合圆
        # 方程: (x-cx)^2 + (y-cy)^2 = r^2
        # 展开: x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)
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
                max_error = float('inf')
            
            return max_error, (np.array([cx, cy]), r)
        except:
            return float('inf'), (points[0], 0.0)
    
    def fit_spline_segment(self, points: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        使用B样条拟合自由曲线段（弧长参数化）
        
        参数:
            points: 点数组 (N, 2)
            t: 弧长参数 (N,)
        
        返回:
            拟合后的点数组
        """
        if len(points) < 3:
            return points
        
        # 确定样条阶数
        k = min(self.spline_degree, len(points) - 1)
        if k < 1:
            k = 1
        
        try:
            # 使用弧长参数化（关键优化）
            spl = make_interp_spline(t, points, k=k)
            
            # 生成密集点
            t_dense = np.linspace(t[0], t[-1], len(points) * 2)
            fitted_points = spl(t_dense)
            
            return fitted_points
        except Exception as e:
            # 如果spline失败，返回原始点
            return points
    
    def classify_segment(self, points: np.ndarray) -> Dict:
        """
        对一段轮廓进行分类（Line/Arc/Spline）
        【优化】使用全局优先级调度器（dispatch_segment）进行统一决策
        
        参数:
            points: 点数组 (N, 2)
        
        返回:
            分类结果字典
        """
        if len(points) < 2:
            return {
                'type': 'spline',
                'points': points,
                'error': float('inf')
            }
        
        # 确保points是2D数组，形状为(N, 2)
        points = np.asarray(points)
        if len(points.shape) == 1:
            return {
                'type': 'spline',
                'points': points,
                'error': float('inf')
            }
        if len(points.shape) == 2:
            if points.shape[1] != 2:
                if points.shape[0] == 2:
                    points = points.T
                elif points.shape[1] > 2:
                    points = points[:, :2]
                else:
                    return {
                        'type': 'spline',
                        'points': points,
                        'error': float('inf')
                    }
        else:
            return {
                'type': 'spline',
                'points': points,
                'error': float('inf')
            }
        
        simplified_points = points
        if len(simplified_points) > 250:
            step = int(np.ceil(len(simplified_points) / 250))
            simplified_points = simplified_points[::max(1, step)]
        try:
            approx = cv2.approxPolyDP(
                np.asarray(simplified_points, dtype=np.int32).reshape(-1, 1, 2),
                1.0,
                False
            ).reshape(-1, 2).astype(float)
            if len(approx) >= 2:
                simplified_points = approx
        except Exception:
            pass

        try:
            decision, dispatch_result = dispatch_segment(
                simplified_points,
                # Line判别参数
                linearity_threshold=0.03,
                max_residual_threshold=max(float(self.line_threshold), 2.5),
                total_angle_threshold_deg=30.0,
                # Arc硬判据参数
                sigma_r_max=0.02,
                angle_span_min_deg=20.0,
                num_points_min=8,
                # Arc vs NURBS对比参数
                arc_gain_th=1.5,
                arc_ctrl=3,
                nurbs_ctrl=8
            )
            
            if decision == "line":
                # Line段
                line_info = dispatch_result['info']
                p0, p1 = line_info['p0'], line_info['p1']
                return {
                    'type': 'line',
                    'start': p0,
                    'end': p1,
                    'points': points,
                    'error': line_info.get('max_residual', line_info.get('max_error', 0.0)),
                    'dispatch_reason': dispatch_result['reason']
                }
            
            elif decision == "arc":
                # Arc段
                arc_info = dispatch_result['info']
                center, radius = arc_info["center"], arc_info["radius"]
                angle_info = arc_angles_from_points(center, simplified_points)
                return {
                    'type': 'arc',
                    'center': center,
                    'radius': radius,
                    'points': points,
                    'error': arc_info.get('sigma_r', 0.0) * radius,
                    'angle_span': arc_info.get('angle_span', 0.0),
                    'sigma_r': arc_info.get('sigma_r', 0.0),
                    'start_angle': angle_info.get('start_angle', 0.0),
                    'end_angle': angle_info.get('end_angle', 0.0),
                    'ccw': bool(angle_info.get('ccw', True)),
                    'dispatch_reason': dispatch_result['reason']
                }
            
            else:  # decision == "nurbs" or "bspline"
                line_error, (lp0, lp1) = self.fit_line(points)
                if line_error < float(self.line_threshold) and np.linalg.norm(lp1 - lp0) >= 10.0:
                    return {
                        'type': 'line',
                        'start': lp0,
                        'end': lp1,
                        'points': points,
                        'error': float(line_error),
                        'dispatch_reason': 'line_strict_override'
                    }

                arc_error, arc_result = self.fit_circle(points)
                if arc_result is not None and arc_error < float(self.arc_threshold):
                    center, radius = arc_result
                    angle_info = arc_angles_from_points(center, simplified_points)
                    return {
                        'type': 'arc',
                        'center': center,
                        'radius': radius,
                        'points': points,
                        'error': float(arc_error),
                        'start_angle': angle_info.get('start_angle', 0.0),
                        'end_angle': angle_info.get('end_angle', 0.0),
                        'ccw': bool(angle_info.get('ccw', True)),
                        'dispatch_reason': 'arc_strict_override'
                    }

                # B-spline段
                t = self.arc_length_param(points)
                fitted_points = self.fit_spline_segment(points, t)
                
                # 计算误差
                errors = []
                for p in points:
                    dists = np.linalg.norm(fitted_points - p, axis=1)
                    errors.append(np.min(dists))
                max_error = np.max(errors) if errors else float('inf')
                
                return {
                    'type': 'bspline',
                    'points': points,
                    'error': max_error,
                    'dispatch_reason': dispatch_result['reason']
                }
        except Exception as e:
            # 如果调度失败，使用原始逻辑作为回退
            print(f"   [WARN] 调度失败，使用原始分类: {e}")
            # 尝试直线拟合
            line_error, (p0, p1) = self.fit_line(points)
            
            if line_error < self.line_threshold:
                return {
                    'type': 'line',
                    'start': p0,
                    'end': p1,
                    'points': points,
                    'error': line_error
                }
            
            # 否则使用Spline
            t = self.arc_length_param(points)
            fitted_points = self.fit_spline_segment(points, t)
            
            # 计算误差
            errors = []
            for p in points:
                dists = np.linalg.norm(fitted_points - p, axis=1)
                errors.append(np.min(dists))
            max_error = np.max(errors) if errors else float('inf')
            
            return {
                'type': 'spline',
                'points': fitted_points,
                'control_points': points,
                'error': max_error
            }
    
    def merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        合并相邻的同类段（改进版：使用改进的段合并器）
        
        参数:
            segments: 分段列表
        
        返回:
            合并后的分段列表
        """
        if len(segments) < 2:
            return segments
        
        # 如果使用改进的段合并器，使用它
        if self.use_improved_merger and self.segment_merger is not None:
            return self.segment_merger.merge_segments(segments)
        
        # 否则使用原始合并逻辑（向后兼容）
        def _undirected_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
            v1 = np.asarray(v1, dtype=float)
            v2 = np.asarray(v2, dtype=float)
            n1 = np.linalg.norm(v1) + 1e-10
            n2 = np.linalg.norm(v2) + 1e-10
            cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            ang = float(np.degrees(np.arccos(cosv)))
            return min(ang, 180.0 - ang)

        def _fit_line_pca(points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
            pts = np.asarray(points, dtype=float).reshape(-1, 2)
            mean = np.mean(pts, axis=0)
            x = pts - mean
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            direction = vt[0]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            proj = x @ direction
            p0 = mean + direction * float(np.min(proj))
            p1 = mean + direction * float(np.max(proj))
            closest = mean + np.outer(proj, direction)
            max_err = float(np.max(np.linalg.norm(pts - closest, axis=1))) if len(pts) else 0.0
            return max_err, p0, p1
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # 尝试与下一段合并
            if i + 1 < len(segments):
                next_seg = segments[i + 1]
                
                # 合并条件：类型相同且连接
                if (current['type'] == next_seg['type'] == 'line'):
                    cur_pts = np.asarray(current.get('points', []), dtype=float).reshape(-1, 2)
                    nxt_pts = np.asarray(next_seg.get('points', []), dtype=float).reshape(-1, 2)
                    if len(cur_pts) >= 2 and len(nxt_pts) >= 2:
                        if np.linalg.norm(cur_pts[-1] - nxt_pts[0]) <= 3.0:
                            v1 = cur_pts[-1] - cur_pts[0]
                            v2 = nxt_pts[-1] - nxt_pts[0]
                            if np.linalg.norm(v1) > 1e-8 and np.linalg.norm(v2) > 1e-8:
                                if _undirected_angle_deg(v1, v2) <= 5.0:
                                    pts = np.vstack([cur_pts, nxt_pts[1:]])
                                    max_err, p0, p1 = _fit_line_pca(pts)
                                    if max_err <= max(float(self.line_threshold), 2.5) * 1.5:
                                        merged.append({
                                            'type': 'line',
                                            'start': p0,
                                            'end': p1,
                                            'points': pts,
                                            'error': max(float(current.get('error', 0.0)), float(next_seg.get('error', 0.0)), float(max_err))
                                        })
                                        i += 2
                                        continue
                
                elif (current['type'] == next_seg['type'] == 'arc'):
                    # 检查是否同一圆
                    if current['type'] == 'arc':
                        center_diff = np.linalg.norm(current['center'] - next_seg['center'])
                        radius_diff = abs(current['radius'] - next_seg['radius'])
                        
                        if center_diff < 5.0 and radius_diff < 5.0:
                            # 合并圆弧段
                            merged.append({
                                'type': 'arc',
                                'center': current['center'],
                                'radius': current['radius'],
                                'points': np.vstack([current['points'], next_seg['points'][1:]]),
                                'error': max(current['error'], next_seg['error'])
                            })
                            i += 2
                            continue
            
            # 不能合并，保留当前段
            merged.append(current)
            i += 1
        
        return merged

    def _regularize_segments_for_cad(self, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return segments

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
                new_seg['start'] = p0
                new_seg['end'] = p1
                new_seg['_ori'] = orientation
                pts = seg.get('points', None)
                if pts is not None:
                    pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
                    if len(pts_arr) >= 2:
                        pts_arr[0] = p0
                        pts_arr[-1] = p1
                        new_seg['points'] = pts_arr
                segs.append(new_seg)
            elif t == 'arc':
                new_seg = dict(seg)
                if 'center' in new_seg:
                    new_seg['center'] = np.asarray(new_seg['center'], dtype=float).reshape(2,)
                if 'points' in new_seg and new_seg['points'] is not None:
                    new_seg['points'] = np.asarray(new_seg['points'], dtype=float).reshape(-1, 2)
                segs.append(new_seg)
            else:
                new_seg = dict(seg)
                if 'points' in new_seg and new_seg['points'] is not None:
                    new_seg['points'] = np.asarray(new_seg['points'], dtype=float).reshape(-1, 2)
                segs.append(new_seg)

        def _endpoints(s: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            st = s.get('type')
            if st == 'line':
                return np.asarray(s.get('start'), dtype=float).reshape(2,), np.asarray(s.get('end'), dtype=float).reshape(2,)
            pts = s.get('points', None)
            if pts is None:
                return None, None
            pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
            if len(pts_arr) < 2:
                return None, None
            return pts_arr[0].copy(), pts_arr[-1].copy()

        def _set_start(s: Dict, p: np.ndarray):
            if s.get('type') == 'line':
                s['start'] = p
            pts = s.get('points', None)
            if pts is not None:
                pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
                if len(pts_arr) >= 2:
                    pts_arr[0] = p
                    s['points'] = pts_arr

        def _set_end(s: Dict, p: np.ndarray):
            if s.get('type') == 'line':
                s['end'] = p
            pts = s.get('points', None)
            if pts is not None:
                pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
                if len(pts_arr) >= 2:
                    pts_arr[-1] = p
                    s['points'] = pts_arr

        for i in range(len(segs) - 1):
            a = segs[i]
            b = segs[i + 1]
            _, a1 = _endpoints(a)
            b0, _ = _endpoints(b)
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
            f0, _ = _endpoints(first)
            _, l1 = _endpoints(last)
            if f0 is not None and l1 is not None and float(np.linalg.norm(l1 - f0)) <= join_tol:
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
                    cur_end = np.asarray(cur.get('end'), dtype=float).reshape(2,)
                    nxt_start = np.asarray(nxt.get('start'), dtype=float).reshape(2,)
                    if float(np.linalg.norm(cur_end - nxt_start)) <= join_tol and cur.get('_ori', None) == nxt.get('_ori', None) and cur.get('_ori', None) in ('h', 'v'):
                        p0 = np.asarray(cur.get('start'), dtype=float).reshape(2,)
                        p1 = np.asarray(nxt.get('end'), dtype=float).reshape(2,)
                        pts_a = cur.get('points', None)
                        pts_b = nxt.get('points', None)
                        if pts_a is not None and pts_b is not None:
                            pts = np.vstack([np.asarray(pts_a, dtype=float).reshape(-1, 2), np.asarray(pts_b, dtype=float).reshape(-1, 2)[1:]])
                            if len(pts) >= 2:
                                pts[0] = p0
                                pts[-1] = p1
                        else:
                            pts = np.vstack([p0.reshape(1, 2), p1.reshape(1, 2)])

                        new_seg = dict(cur)
                        new_seg['start'] = p0
                        new_seg['end'] = p1
                        new_seg['points'] = pts
                        merged.append(new_seg)
                        i += 2
                        continue

            merged.append(cur)
            i += 1

        for s in merged:
            s.pop('_ori', None)

        merged = self.merge_segments(merged)

        normalized: List[Dict] = []
        for seg in merged:
            t = str(seg.get('type', ''))
            new_seg = dict(seg)
            if t == 'line':
                if 'start' in new_seg:
                    new_seg['start'] = np.asarray(new_seg['start'], dtype=float).reshape(2,).tolist()
                if 'end' in new_seg:
                    new_seg['end'] = np.asarray(new_seg['end'], dtype=float).reshape(2,).tolist()
                pts = new_seg.get('points', None)
                if pts is not None:
                    new_seg['points'] = np.asarray(pts, dtype=float).reshape(-1, 2).tolist()
            elif t == 'arc':
                if 'center' in new_seg:
                    new_seg['center'] = np.asarray(new_seg['center'], dtype=float).reshape(2,).tolist()
                if 'radius' in new_seg and new_seg['radius'] is not None:
                    new_seg['radius'] = float(new_seg['radius'])
                pts = new_seg.get('points', None)
                if pts is not None:
                    pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
                    new_seg['points'] = pts_arr.tolist()
                    if 'center' in new_seg and 'radius' in new_seg and len(pts_arr) >= 3:
                        c = np.asarray(new_seg['center'], dtype=float).reshape(2,)
                        ang = arc_angles_from_points(c, pts_arr)
                        new_seg['start_angle'] = float(ang.get('start_angle', 0.0))
                        new_seg['end_angle'] = float(ang.get('end_angle', 0.0))
                        new_seg['ccw'] = bool(ang.get('ccw', True))
            else:
                pts = new_seg.get('points', None)
                if pts is not None:
                    new_seg['points'] = np.asarray(pts, dtype=float).reshape(-1, 2).tolist()

            normalized.append(new_seg)

        return normalized
    
    def _filter_small_segments(self, segments: List[Dict], min_length: float = 5.0) -> List[Dict]:
        """
        过滤特别小的段
        
        参数:
            segments: 段列表
            min_length: 最小段长度（像素）
        
        返回:
            过滤后的段列表
        """
        if not segments:
            return segments
        
        filtered = []
        for seg in segments:
            seg_type = seg.get('type', '')
            
            # 计算段的长度
            seg_length = 0.0
            
            if seg_type == 'line':
                # 直线：计算端点距离
                start = seg.get('start', seg.get('p0', None))
                end = seg.get('end', seg.get('p1', None))
                if start is not None and end is not None:
                    try:
                        start_pt = np.asarray(start, dtype=float).flatten()
                        end_pt = np.asarray(end, dtype=float).flatten()
                        if start_pt.size == 2 and end_pt.size == 2:
                            seg_length = float(np.linalg.norm(end_pt - start_pt))
                    except:
                        pass
            
            elif seg_type == 'arc':
                # 圆弧：计算弧长（近似）
                radius = seg.get('radius', 0.0)
                angle_span = seg.get('angle_span', 0.0)
                if radius > 0 and angle_span > 0:
                    seg_length = float(radius * abs(angle_span))
            
            else:
                # Spline/B-spline：计算点序列的总长度
                pts = seg.get('points', None)
                if pts is not None:
                    try:
                        pts_arr = np.asarray(pts, dtype=float)
                        if len(pts_arr.shape) == 2 and pts_arr.shape[1] == 2 and len(pts_arr) >= 2:
                            diffs = np.diff(pts_arr, axis=0)
                            distances = np.linalg.norm(diffs, axis=1)
                            seg_length = float(np.sum(distances))
                    except:
                        pass
            
            # 如果段长度大于最小长度，保留
            if seg_length >= min_length:
                filtered.append(seg)
        
        return filtered
    
    def _validate_contour(self, contour_points: np.ndarray) -> bool:
        """
        验证轮廓是否有效（改进版：更宽松的条件）
        
        参数:
            contour_points: 轮廓点数组
        
        返回:
            是否有效
        """
        if contour_points is None:
            return False
        
        # 使用预处理器验证和修复
        processed = ContourPreprocessor.preprocess_contour(
            contour_points, 
            min_points=2,  # 更宽松：至少2个点
            remove_duplicates=True,
            interpolate=True
        )
        
        return processed is not None and len(processed) >= 2
    
    def _fit_contour_with_fallback(self, contour: np.ndarray, contour_id: int, hierarchy_info: Optional[Dict] = None) -> Dict:
        """
        拟合轮廓（带回退机制）
        
        参数:
            contour: 轮廓点数组
            contour_id: 轮廓ID
            hierarchy_info: 层级信息
        
        返回:
            拟合结果字典
        """
        # 尝试主要拟合方法
        try:
            result = self._fit_contour(contour, contour_id, hierarchy_info)
            if result and len(result.get('segments', [])) > 0:
                return result
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 主要拟合方法失败: {e}")
        
        # 回退方法1：使用简单分段
        try:
            result = self._fit_contour_simple(contour, contour_id, hierarchy_info)
            if result and len(result.get('segments', [])) > 0:
                return result
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 简单拟合方法失败: {e}")
        
        # 回退方法2：直接使用原始轮廓（预处理后）
        try:
            points = ContourPreprocessor.preprocess_contour(contour, min_points=2)
            if points is not None and len(points) >= 2:
                return {
                    'contour': contour,
                    'segments': [{
                        'type': 'spline',
                        'points': points.tolist(),
                        'error': float('inf')
                    }],
                    'fitted_points': points,
                    'error': float('inf'),
                    'hierarchy': hierarchy_info
                }
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 回退方法失败: {e}")
        
        # 最终回退：返回空结果
        return {
            'contour': contour,
            'segments': [],
            'fitted_points': np.array([[0, 0]]),
            'error': float('inf'),
            'hierarchy': hierarchy_info
        }
    
    def _fit_contour_simple(self, contour: np.ndarray, contour_id: int, hierarchy_info: Optional[Dict] = None) -> Dict:
        """
        简单拟合方法（回退用）
        
        参数:
            contour: 轮廓点数组
            contour_id: 轮廓ID
            hierarchy_info: 层级信息
        
        返回:
            拟合结果字典
        """
        # 预处理轮廓
        points = ContourPreprocessor.preprocess_contour(contour, min_points=2)
        if points is None or len(points) < 2:
            return {
                'contour': contour,
                'segments': [],
                'fitted_points': np.array([[0, 0]]),
                'error': float('inf'),
                'hierarchy': hierarchy_info
            }
        
        # 简单分段：直接使用原始点作为spline
        segments = [{
            'type': 'spline',
            'points': points.tolist(),
            'error': 0.0
        }]
        
        return {
            'contour': contour,
            'segments': segments,
            'fitted_points': points,
            'error': 0.0,
            'hierarchy': hierarchy_info
        }
    
    def _fit_contour(self, contour: np.ndarray, contour_id: int, hierarchy_info: Optional[Dict] = None) -> Dict:
        """
        拟合单个轮廓（改进版：添加错误处理和回退机制）
        
        参数:
            contour: 轮廓点数组
            contour_id: 轮廓ID
            hierarchy_info: 层级信息
        
        返回:
            拟合结果字典
        """
        # 【改进】使用预处理器预处理轮廓
        points = ContourPreprocessor.preprocess_contour(contour, min_points=2)
        if points is None:
            return {
                'contour': contour,
                'segments': [],
                'fitted_points': np.array([[0, 0]]),
                'error': float('inf'),
                'hierarchy': hierarchy_info
            }
        
        # 【改进】验证轮廓有效性（更宽松）
        if not self._validate_contour(points):
            # 即使验证失败，也尝试使用预处理后的点
            return {
                'contour': contour,
                'segments': [{
                    'type': 'spline',
                    'points': points.tolist(),
                    'error': float('inf')
                }],
                'fitted_points': points,
                'error': float('inf'),
                'hierarchy': hierarchy_info
            }
        
        # Step 1: 初步分段（基于曲率）
        try:
            initial_segments, corner_indices = self.initial_segmentation(points)
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 分段失败，使用整条轮廓: {e}")
            # 回退：使用整条轮廓作为一个段
            initial_segments = [points]
            corner_indices = set()
        
        # Step 2: 对每段进行分类和拟合（添加错误处理）
        classified_segments = []
        all_fitted_points = []
        
        for seg_idx, seg_points in enumerate(initial_segments):
            if len(seg_points) < 2:
                continue
            
            # 【改进】验证分段有效性
            if not self._validate_contour(seg_points):
                continue
            
            # 分类并拟合（添加错误处理）
            try:
                seg_result = self.classify_segment(seg_points)
                if seg_result is None:
                    # 如果分类失败，使用简单spline作为回退
                    t = self.arc_length_param(seg_points)
                    fitted_points = self.fit_spline_segment(seg_points, t)
                    seg_result = {
                        'type': 'spline',
                        'points': fitted_points,
                        'error': float('inf')
                    }
                classified_segments.append(seg_result)
                
                # 收集拟合点
                if seg_result['type'] == 'line':
                    # 直线：生成点
                    num_points = len(seg_points)
                    t_line = np.linspace(0, 1, num_points)
                    line_points = seg_result['start'] + np.outer(
                        t_line, seg_result['end'] - seg_result['start']
                    )
                    all_fitted_points.append(line_points)
                elif seg_result['type'] == 'arc':
                    # 圆弧：生成点
                    num_points = len(seg_points)
                    angles = np.linspace(0, 2*np.pi, num_points)
                    # 简化：使用原始点
                    all_fitted_points.append(seg_points)
                else:  # spline
                    # Spline：使用拟合点
                    all_fitted_points.append(seg_result['points'])
            except Exception as e:
                # 如果分类或收集拟合点失败，使用简单spline作为回退
                print(f"   [WARN] 轮廓 {contour_id} 分段 {seg_idx} 处理失败，使用简单spline: {e}")
                try:
                    t = self.arc_length_param(seg_points)
                    fitted_points = self.fit_spline_segment(seg_points, t)
                    seg_result = {
                        'type': 'spline',
                        'points': fitted_points,
                        'error': float('inf')
                    }
                    classified_segments.append(seg_result)
                    all_fitted_points.append(fitted_points)
                except Exception:
                    # 如果连回退都失败，跳过这个分段
                    continue
        
        # Step 3: 合并相邻同类段（添加错误处理和回退，自适应参数）
        try:
            # 【改进】根据轮廓特征自适应调整合并参数
            if self.use_improved_merger and self.segment_merger is not None:
                # 计算轮廓特征
                contour_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
                num_points = len(points)
                
                # 根据轮廓大小调整参数
                if contour_length < 100:
                    # 小轮廓：更宽松的参数
                    adaptive_min_length = max(10.0, self._min_segment_length * 0.5)
                    adaptive_angle = min(20.0, self._angle_threshold_deg * 1.3)
                    adaptive_distance = min(6.0, self._distance_threshold * 1.2)
                elif contour_length > 1000:
                    # 大轮廓：更严格的参数
                    adaptive_min_length = min(30.0, self._min_segment_length * 1.5)
                    adaptive_angle = max(10.0, self._angle_threshold_deg * 0.8)
                    adaptive_distance = max(4.0, self._distance_threshold * 0.8)
                else:
                    # 中等轮廓：使用默认参数
                    adaptive_min_length = self._min_segment_length
                    adaptive_angle = self._angle_threshold_deg
                    adaptive_distance = self._distance_threshold
                
                # 根据轮廓点数调整最大控制点
                if num_points < 50:
                    adaptive_max_segments = min(5, self._max_segments or 15)
                elif num_points > 500:
                    adaptive_max_segments = min(20, self._max_segments or 30)
                else:
                    adaptive_max_segments = self._max_segments
                
                # 创建自适应合并器
                adaptive_merger = ImprovedSegmentMerger(
                    angle_threshold_deg=adaptive_angle,
                    distance_threshold=adaptive_distance,
                    min_segment_length=adaptive_min_length,
                    max_segments=adaptive_max_segments,
                    error_tolerance_factor=self._error_tolerance_factor
                )
                merged_segments = adaptive_merger.merge_segments(classified_segments)
            else:
                merged_segments = self.merge_segments(classified_segments)
            
            # 【改进】如果合并后没有段，使用原始分段
            if len(merged_segments) == 0:
                print(f"   [WARN] 轮廓 {contour_id} 合并后没有段，使用原始分段")
                merged_segments = classified_segments
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} 段合并失败，使用原始分段: {e}")
            merged_segments = classified_segments
        
        try:
            merged_segments = self._regularize_segments_for_cad(merged_segments)
        except Exception as e:
            print(f"   [WARN] 轮廓 {contour_id} CAD正则化失败: {e}")
            # 继续使用未正则化的段
        
        # 【改进2】过滤特别小的段
        merged_segments = self._filter_small_segments(merged_segments, min_length=5.0)
        
        # Step 4: 角点对齐（将靠近角点的采样点移动到角点位置）
        if len(all_fitted_points) > 0 and len(corner_indices) > 0:
            # 获取角点坐标
            corner_points = points[list(corner_indices)]
            corner_points = np.asarray(corner_points, dtype=float).reshape(-1, 2)
            
            # 对齐所有拟合点到角点
            aligned_fitted_points = []
            for fp in all_fitted_points:
                fp = np.asarray(fp, dtype=float)
                if len(fp.shape) == 1 or fp.shape[1] != 2:
                    aligned_fitted_points.append(fp)
                    continue
                
                # 对每个拟合点，检查是否靠近角点
                aligned_fp = fp.copy()
                corner_tolerance = 3.0  # 角点对齐容差（像素）
                
                for corner_pt in corner_points:
                    # 计算所有点到角点的距离
                    distances = np.linalg.norm(aligned_fp - corner_pt.reshape(1, 2), axis=1)
                    # 将距离小于容差的点移动到角点
                    close_mask = distances < corner_tolerance
                    if np.any(close_mask):
                        aligned_fp[close_mask] = corner_pt
                
                aligned_fitted_points.append(aligned_fp)
            
            all_fitted_points = aligned_fitted_points
        
        # Step 5: 组合所有拟合点（确保连接处连续）
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
                
                if i > 0:
                    # 检查前一段的终点和当前段的起点
                    prev_fp = fitted_points_list[-1]
                    prev_fp = np.asarray(prev_fp, dtype=float)
                    # 【修复】确保prev_fp是2D数组
                    if len(prev_fp.shape) == 1:
                        if prev_fp.size == 2:
                            prev_fp = prev_fp.reshape(1, 2)
                        else:
                            fitted_points_list.append(fp)
                            continue
                    elif prev_fp.shape[1] != 2:
                        if prev_fp.shape[0] == 2:
                            prev_fp = prev_fp.T
                        else:
                            fitted_points_list.append(fp)
                            continue
                    
                    prev_end = prev_fp[-1]
                    curr_start = fp[0]
                    
                    # 【修复】确保prev_end和curr_start是1D数组（形状为(2,)）
                    prev_end = np.asarray(prev_end, dtype=float).flatten()
                    curr_start = np.asarray(curr_start, dtype=float).flatten()
                    
                    if prev_end.size != 2 or curr_start.size != 2:
                        fitted_points_list.append(fp)
                        continue
                    
                    # 如果距离较大，插入中间点
                    if np.linalg.norm(prev_end - curr_start) > 1.0:
                        # 插入中间点，确保连续
                        mid_point = (prev_end + curr_start) / 2.0
                        fitted_points_list.append(np.vstack([
                            prev_end.reshape(1, 2), 
                            mid_point.reshape(1, 2), 
                            fp
                        ]))
                    else:
                        # 直接连接
                        fitted_points_list.append(fp)
                else:
                    fitted_points_list.append(fp)
            
            if len(fitted_points_list) > 0:
                fitted_points = np.vstack(fitted_points_list)
            else:
                fitted_points = points
        else:
            fitted_points = points
        
        # 计算总体误差
        errors = []
        for p in points:
            dists = np.linalg.norm(fitted_points - p, axis=1)
            errors.append(np.min(dists))
        total_error = np.mean(errors) if errors else float('inf')
        
        return {
            'contour': contour,
            'segments': merged_segments,
            'fitted_points': fitted_points,
            'error': total_error,
            'hierarchy': hierarchy_info
        }
    
    def _fit_all_contours(self):
        """拟合所有轮廓（改进版：不依赖索引奇偶性，基于轮廓特征过滤）"""
        if not self.contours:
            return
        
        # 【改进】基于轮廓特征过滤，不依赖索引奇偶性
        valid_contours, valid_indices = ContourPreprocessor.filter_contours_by_features(
            self.contours,
            self.hierarchy,
            min_area=5.0,  # 最小面积（像素²）
            min_perimeter=10.0,  # 最小周长（像素）
            min_points=2  # 最小点数（更宽松）
        )
        
        for valid_idx, (i, contour) in enumerate(zip(valid_indices, valid_contours)):
            # 获取层级信息
            hierarchy_info = None
            if self.hierarchy is not None and len(self.hierarchy[0]) > i:
                h = self.hierarchy[0][i]
                hierarchy_info = {
                    'next': h[0],
                    'prev': h[1],
                    'child': h[2],
                    'parent': h[3]
                }
            
            # 拟合轮廓（带回退机制）
            result = self._fit_contour_with_fallback(contour, i, hierarchy_info)
            
            # 存储结果（兼容原格式）
            self.contours_dict[str(i)] = {
                'contour': result['contour'],
                'segments': result['segments'],
                'fitting': {
                    'size': len(result['fitted_points']),
                    'points': result['fitted_points']
                },
                'error': result['error'],
                'hierarchy': result['hierarchy']
            }
    
    def _build_curves_tree(self):
        """构建轮廓树（复用原代码逻辑）"""
        tree = {}
        if self.hierarchy is None:
            return tree
        
        h = np.copy(self.hierarchy[0])
        for i in range(len(h)):
            if i % 2 == 1:
                if str(i) not in tree:
                    tree[str(i)] = []
                j = i
                while h[j][3] > -1:
                    if h[j][3] % 2 != 0:
                        parent_key = str(h[j][3])
                        if parent_key not in tree:
                            tree[parent_key] = []
                        tree[parent_key].append(str(i))
                        break
                    else:
                        j = h[j][3]
        
        self.curves_tree = tree
        return tree
    
    def get_contours_dict(self) -> Dict:
        """获取轮廓字典"""
        return self.contours_dict
    
    def get_curves_tree(self) -> Dict:
        """获取轮廓树"""
        return self.curves_tree
    
    def get_segments_info(self) -> Dict:
        """
        获取分段信息（用于CAD导出和几何感知评分）
        
        返回:
            分段信息字典，格式：{contour_id: [segment1, segment2, ...]}
        """
        segments_dict = {}
        
        for contour_id, data in self.contours_dict.items():
            # 从contours_dict中提取segments字段
            if 'segments' in data:
                segments = data['segments']
                if isinstance(segments, list):
                    segments_dict[contour_id] = segments
        
        return segments_dict
    
    def get_contours(self):
        """获取原始轮廓"""
        return self.contours
    
    def get_hierarchy(self):
        """获取层级信息"""
        return self.hierarchy


# 向后兼容的类名
OptimizedBSplineContour = OptimizedBSplineFitter


if __name__ == "__main__":
    from core.image.initializer import ImageInitializer
    
    # 测试
    II = ImageInitializer(r"D:\datasheet\test\test21.png")
    Img = II.centered_img()
    Edges = II.edges()
    
    OF = OptimizedBSplineFitter(Img, Edges)
    print("优化B样条拟合完成")
    print(f"检测到 {len(OF.get_contours_dict())} 个轮廓")
