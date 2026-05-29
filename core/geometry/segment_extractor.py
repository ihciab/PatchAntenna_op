"""
分段提取器
从拟合曲线中自动识别Line/Arc/Spline段

用于为传统方法（B样条、NURBS）添加分段信息，使其能够进行几何感知评分
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class SegmentExtractor:
    """
    分段提取器
    从拟合曲线中自动识别Line/Arc/Spline段
    """
    
    def __init__(self,
                 line_threshold: float = 2.0,
                 arc_threshold: float = 2.0,
                 curvature_threshold: float = 0.15):
        """
        初始化分段提取器
        
        参数:
            line_threshold: 直线拟合误差阈值（像素）
            arc_threshold: 圆弧拟合误差阈值（像素）
            curvature_threshold: 曲率变化阈值（用于分段）
        """
        self.line_threshold = line_threshold
        self.arc_threshold = arc_threshold
        self.curvature_threshold = curvature_threshold
    
    def discrete_curvature(self, points: np.ndarray) -> np.ndarray:
        """计算离散曲率"""
        if len(points) < 3:
            return np.zeros(len(points))
        
        p_prev = points[:-2]
        p = points[1:-1]
        p_next = points[2:]
        
        v1 = p - p_prev
        v2 = p_next - p
        
        dot_products = np.sum(v1 * v2, axis=1)
        norms1 = np.linalg.norm(v1, axis=1)
        norms2 = np.linalg.norm(v2, axis=1)
        
        cos_angles = np.clip(
            dot_products / (norms1 * norms2 + 1e-8),
            -1.0, 1.0
        )
        
        angles = np.arccos(cos_angles)
        return np.pad(angles, (1, 1), mode='edge')
    
    def initial_segmentation(self, points: np.ndarray) -> List[np.ndarray]:
        """基于曲率变化进行初步分段"""
        if len(points) < 3:
            return [points]
        
        curvature = self.discrete_curvature(points)
        if len(curvature) == 0:
            return [points]
        
        dk = np.abs(np.diff(curvature))
        break_indices = np.where(dk > self.curvature_threshold)[0] + 1
        
        segments = []
        start = 0
        
        for b in break_indices:
            if b > start:
                segments.append(points[start:b+1])
                start = b
        
        if start < len(points):
            segments.append(points[start:])
        
        if len(segments) == 0:
            segments = [points]
        
        return segments
    
    def fit_line(self, points: np.ndarray) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """拟合直线段，返回最大误差和(起点, 终点)"""
        if len(points) < 2:
            return float('inf'), None
        
        p0 = points[0]
        p1 = points[-1]
        
        v = p1 - p0
        v_norm = np.linalg.norm(v)
        
        if v_norm < 1e-6:
            return 0.0, (p0, p1)
        
        v = v / v_norm
        proj = np.dot(points - p0, v)
        closest = p0 + np.outer(proj, v)
        errors = np.linalg.norm(points - closest, axis=1)
        max_error = np.max(errors)
        
        return max_error, (p0, p1)
    
    def fit_arc(self, points: np.ndarray) -> Tuple[float, Optional[Tuple[np.ndarray, float]]]:
        """拟合圆弧段，返回最大误差和(圆心, 半径)"""
        if len(points) < 3:
            return float('inf'), None
        
        x = points[:, 0]
        y = points[:, 1]
        
        A = np.column_stack([2*x, 2*y, np.ones(len(points))])
        b = x**2 + y**2
        
        try:
            c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            cx, cy = c[0], c[1]
            r = np.sqrt(c[2] + cx**2 + cy**2)
            
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
                return float('inf'), None
            
            return max_error, (np.array([cx, cy]), r)
        except:
            return float('inf'), None
    
    def extract_segments(self, fitted_points: np.ndarray) -> List[Dict]:
        """
        从拟合曲线中提取分段信息
        
        参数:
            fitted_points: 拟合曲线点 (N, 2)
        
        返回:
            分段信息列表
        """
        fitted_points = np.asarray(fitted_points, dtype=float).reshape(-1, 2)
        if len(fitted_points) < 3:
            return []

        fitted_points = self._simplify_polyline(fitted_points)
        
        # Step 1: 初步分段
        initial_segments = self._initial_segmentation_minlen(fitted_points)
        
        # Step 2: 对每段进行分类
        segments = []
        
        for seg_points in initial_segments:
            if len(seg_points) < 2:
                continue

            seg_points = np.asarray(seg_points, dtype=float).reshape(-1, 2)
            
            # 尝试直线拟合
            line_error, line_result = self.fit_line(seg_points)
            if line_result is not None and line_error < self.line_threshold:
                p0, p1 = line_result
                if np.linalg.norm(p1 - p0) < 8.0:
                    continue
                segments.append({
                    'type': 'line',
                    'points': seg_points,
                    'p0': p0.tolist(),
                    'p1': p1.tolist(),
                    'model': {
                        'point': p0,
                        'direction': (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-10)
                    }
                })
                continue
            
            # 尝试圆弧拟合
            if len(seg_points) >= 3:
                arc_error, arc_result = self.fit_arc(seg_points)
                if arc_result is not None and arc_error < self.arc_threshold:
                    center, radius = arc_result
                    if float(radius) < 3.0:
                        continue
                    segments.append({
                        'type': 'arc',
                        'points': seg_points,
                        'center': center.tolist(),
                        'radius': float(radius),
                        'model': {
                            'center': center,
                            'radius': radius
                        }
                    })
                    continue
            
            # 否则是Spline
            segments.append({
                'type': 'spline',
                'points': seg_points,
                'model': {
                    'curve': self._create_curve_func(seg_points)
                }
            })

        segments = self._merge_segments(segments)
        segments = [s for s in segments if self._segment_length(s) >= 8.0]
        return segments

    def _simplify_polyline(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) > 2500:
            step = int(np.ceil(len(pts) / 2500))
            pts = pts[::max(1, step)]

        if len(pts) < 3:
            return pts

        closed = bool(np.linalg.norm(pts[0] - pts[-1]) <= 2.0)
        try:
            approx = cv2.approxPolyDP(
                np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2),
                1.0,
                closed
            ).reshape(-1, 2).astype(float)
            if len(approx) >= 3:
                return approx
        except Exception:
            pass
        return pts

    def _initial_segmentation_minlen(self, points: np.ndarray) -> List[np.ndarray]:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) < 3:
            return [pts]

        curvature = self.discrete_curvature(pts)
        if len(curvature) == 0:
            return [pts]

        dk = np.abs(np.diff(curvature))
        break_indices = (np.where(dk > self.curvature_threshold)[0] + 1).tolist()

        min_points = 10
        segments = []
        start = 0

        for b in break_indices:
            if b - start + 1 < min_points:
                continue
            segments.append(pts[start:b + 1])
            start = b

        if start < len(pts):
            tail = pts[start:]
            if segments and len(tail) < min_points:
                segments[-1] = np.vstack([segments[-1], tail[1:]]) if len(tail) > 1 else segments[-1]
            else:
                segments.append(tail)

        if not segments:
            segments = [pts]

        return segments

    def _segment_length(self, seg: Dict) -> float:
        pts = np.asarray(seg.get('points', []), dtype=float).reshape(-1, 2)
        if len(pts) < 2:
            return 0.0
        d = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))

    def _fit_line_pca(self, points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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
        return max_err, p0, p1, direction

    def _undirected_angle_deg(self, v1: np.ndarray, v2: np.ndarray) -> float:
        a = float(np.degrees(np.arccos(np.clip(
            float(np.dot(v1, v2)) / ((np.linalg.norm(v1) + 1e-10) * (np.linalg.norm(v2) + 1e-10)),
            -1.0, 1.0
        ))))
        return min(a, 180.0 - a)

    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        if len(segments) < 2:
            return segments

        join_tol = 3.0
        merge_angle_deg = 5.0
        center_tol = 4.0
        rel_r_tol = 0.05

        merged: List[Dict] = []

        def _seg_start_end(seg: Dict) -> Tuple[np.ndarray, np.ndarray]:
            pts = np.asarray(seg.get('points', []), dtype=float).reshape(-1, 2)
            if len(pts) < 2:
                p = np.zeros(2, dtype=float)
                return p, p
            return pts[0], pts[-1]

        i = 0
        while i < len(segments):
            cur = segments[i]
            if i + 1 >= len(segments):
                merged.append(cur)
                break

            nxt = segments[i + 1]
            cur_pts = np.asarray(cur.get('points', []), dtype=float).reshape(-1, 2)
            nxt_pts = np.asarray(nxt.get('points', []), dtype=float).reshape(-1, 2)
            if len(cur_pts) < 2 or len(nxt_pts) < 2:
                merged.append(cur)
                i += 1
                continue

            cur_end = cur_pts[-1]
            nxt_start = nxt_pts[0]
            can_join = float(np.linalg.norm(cur_end - nxt_start)) <= join_tol

            if cur.get('type') == 'line' and nxt.get('type') == 'line' and can_join:
                v1 = cur_pts[-1] - cur_pts[0]
                v2 = nxt_pts[-1] - nxt_pts[0]
                if np.linalg.norm(v1) > 1e-8 and np.linalg.norm(v2) > 1e-8:
                    ang = self._undirected_angle_deg(v1, v2)
                    if ang <= merge_angle_deg:
                        pts = np.vstack([cur_pts, nxt_pts[1:]])
                        max_err, p0, p1, _ = self._fit_line_pca(pts)
                        if max_err <= max(self.line_threshold, 2.0) * 1.5:
                            merged.append({
                                'type': 'line',
                                'points': pts,
                                'p0': p0.tolist(),
                                'p1': p1.tolist(),
                                'model': {
                                    'point': p0,
                                    'direction': (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-10)
                                }
                            })
                            i += 2
                            continue

            if cur.get('type') == 'arc' and nxt.get('type') == 'arc' and can_join:
                if 'center' in cur and 'center' in nxt and 'radius' in cur and 'radius' in nxt:
                    c1 = np.asarray(cur['center'], dtype=float).reshape(2,)
                    c2 = np.asarray(nxt['center'], dtype=float).reshape(2,)
                    r1 = float(cur['radius'])
                    r2 = float(nxt['radius'])
                    if np.linalg.norm(c1 - c2) <= center_tol and abs(r1 - r2) / (max(r1, r2, 1e-9)) <= rel_r_tol:
                        pts = np.vstack([cur_pts, nxt_pts[1:]])
                        arc_err, arc_result = self.fit_arc(pts)
                        if arc_result is not None and arc_err <= max(self.arc_threshold, 2.0) * 1.5:
                            center, radius = arc_result
                            merged.append({
                                'type': 'arc',
                                'points': pts,
                                'center': center.tolist(),
                                'radius': float(radius),
                                'model': {
                                    'center': center,
                                    'radius': radius
                                }
                            })
                            i += 2
                            continue

            if cur.get('type') == 'spline' and nxt.get('type') == 'spline' and can_join:
                pts = np.vstack([cur_pts, nxt_pts[1:]])
                merged.append({
                    'type': 'spline',
                    'points': pts,
                    'model': {
                        'curve': self._create_curve_func(pts)
                    }
                })
                i += 2
                continue

            merged.append(cur)
            i += 1

        return merged
    
    def _create_curve_func(self, points: np.ndarray):
        """创建曲线函数（用于Spline段）"""
        def curve_func(t):
            idx = int(t * (len(points) - 1))
            idx = np.clip(idx, 0, len(points) - 1)
            return points[idx]
        return curve_func


def extract_segments_from_fitted_contour(fitted_points: np.ndarray,
                                        line_threshold: float = 2.0,
                                        arc_threshold: float = 2.0,
                                        curvature_threshold: float = 0.15) -> List[Dict]:
    """
    便捷函数：从拟合轮廓中提取分段信息
    
    参数:
        fitted_points: 拟合曲线点 (N, 2)
        line_threshold: 直线拟合误差阈值
        arc_threshold: 圆弧拟合误差阈值
        curvature_threshold: 曲率变化阈值
    
    返回:
        分段信息列表
    """
    extractor = SegmentExtractor(
        line_threshold=line_threshold,
        arc_threshold=arc_threshold,
        curvature_threshold=curvature_threshold
    )
    return extractor.extract_segments(fitted_points)
