"""
改进的段合并模块
实施方案一：更激进的段合并策略

改进点：
1. 放宽合并条件（角度阈值15°，距离阈值5.0像素）
2. 全局合并优化（贪心算法）
3. 最小段长度限制（20像素）
4. 段数上限控制
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ImprovedSegmentMerger:
    """
    改进的段合并器
    实现更激进的段合并策略，大幅减少段数
    """
    
    def __init__(self,
                 angle_threshold_deg: float = 15.0,  # 原5.0
                 distance_threshold: float = 5.0,    # 原3.0
                 min_segment_length: float = 20.0,    # 新增
                 max_segments: Optional[int] = None,  # 新增
                 error_tolerance_factor: float = 1.5):  # 新增
        """
        初始化改进的段合并器
        
        参数:
            angle_threshold_deg: 合并角度阈值（度）
            distance_threshold: 合并距离阈值（像素）
            min_segment_length: 最小段长度（像素）
            max_segments: 最大段数（None表示无限制）
            error_tolerance_factor: 误差容忍因子（允许合并后误差增加）
        """
        self.angle_threshold_deg = angle_threshold_deg
        self.distance_threshold = distance_threshold
        self.min_segment_length = min_segment_length
        self.max_segments = max_segments
        self.error_tolerance_factor = error_tolerance_factor
    
    def merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        改进的段合并算法
        
        参数:
            segments: 分段列表
        
        返回:
            合并后的分段列表
        """
        if len(segments) < 2:
            return segments
        
        # Step 1: 过滤过短的段（强制合并）
        segments = self._filter_short_segments(segments)
        
        # Step 2: 全局贪心合并
        segments = self._greedy_merge(segments)
        
        # Step 3: 如果超过最大段数，强制合并
        if self.max_segments is not None and len(segments) > self.max_segments:
            segments = self._force_merge_to_limit(segments, self.max_segments)
        
        return segments
    
    def _filter_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        过滤过短的段，强制与相邻段合并
        """
        if not segments:
            return segments
        
        filtered = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            seg_length = self._segment_length(current)
            
            if seg_length < self.min_segment_length:
                # 尝试与相邻段合并
                merged = False
                
                # 优先与下一段合并
                if i + 1 < len(segments):
                    merged_seg = self._try_merge_two_segments(current, segments[i + 1])
                    if merged_seg is not None:
                        filtered.append(merged_seg)
                        i += 2
                        merged = True
                
                # 如果下一段合并失败，尝试与上一段合并
                if not merged and len(filtered) > 0:
                    merged_seg = self._try_merge_two_segments(filtered[-1], current)
                    if merged_seg is not None:
                        filtered[-1] = merged_seg
                        i += 1
                        merged = True
                
                # 如果都无法合并，保留（但标记为需要后续处理）
                if not merged:
                    filtered.append(current)
                    i += 1
            else:
                filtered.append(current)
                i += 1
        
        return filtered
    
    def _greedy_merge(self, segments: List[Dict]) -> List[Dict]:
        """
        全局贪心合并算法
        优先合并误差最小的相邻段对
        """
        if len(segments) < 2:
            return segments
        
        merged = list(segments)
        changed = True
        
        # 迭代合并，直到无法继续合并
        while changed and len(merged) >= 2:
            changed = False
            best_merge_idx = None
            best_merge_error = float('inf')
            best_merged_seg = None
            
            # 找到最佳合并候选
            for i in range(len(merged) - 1):
                seg1 = merged[i]
                seg2 = merged[i + 1]
                
                merged_seg = self._try_merge_two_segments(seg1, seg2)
                if merged_seg is not None:
                    # 计算合并后的误差
                    merge_error = merged_seg.get('error', float('inf'))
                    
                    # 如果误差更小或可接受，记录
                    if merge_error < best_merge_error:
                        best_merge_error = merge_error
                        best_merge_idx = i
                        best_merged_seg = merged_seg
            
            # 执行最佳合并
            if best_merged_seg is not None:
                merged[best_merge_idx] = best_merged_seg
                merged.pop(best_merge_idx + 1)
                changed = True
        
        return merged
    
    def _try_merge_two_segments(self, seg1: Dict, seg2: Dict) -> Optional[Dict]:
        """
        尝试合并两个相邻段
        
        返回:
            合并后的段，如果无法合并则返回None
        """
        # 获取点
        pts1 = self._get_segment_points(seg1)
        pts2 = self._get_segment_points(seg2)
        
        if len(pts1) < 2 or len(pts2) < 2:
            return None
        
        # 检查连接性
        dist = np.linalg.norm(pts1[-1] - pts2[0])
        if dist > self.distance_threshold:
            return None
        
        # 合并点
        combined_pts = np.vstack([pts1, pts2[1:]])
        
        # 尝试合并同类段
        if seg1.get('type') == seg2.get('type'):
            if seg1.get('type') == 'line':
                return self._merge_line_segments(seg1, seg2, combined_pts)
            elif seg1.get('type') == 'arc':
                return self._merge_arc_segments(seg1, seg2, combined_pts)
            elif seg1.get('type') in ['spline', 'bspline', 'nurbs']:
                return self._merge_spline_segments(seg1, seg2, combined_pts)
        
        # 尝试合并不同类段（如果误差可接受）
        return self._try_merge_different_types(seg1, seg2, combined_pts)
    
    def _merge_line_segments(self, seg1: Dict, seg2: Dict, combined_pts: np.ndarray) -> Optional[Dict]:
        """合并直线段"""
        # 【修复】确保start和end是numpy数组
        def _get_point(seg: Dict, key: str, default: np.ndarray) -> np.ndarray:
            val = seg.get(key, None)
            if val is None:
                return default
            val = np.asarray(val, dtype=float).flatten()
            if val.size != 2:
                return default
            return val
        
        # 计算方向向量
        start1 = _get_point(seg1, 'start', combined_pts[0])
        end1 = _get_point(seg1, 'end', combined_pts[-1])
        start2 = _get_point(seg2, 'start', combined_pts[0])
        end2 = _get_point(seg2, 'end', combined_pts[-1])
        
        v1 = end1 - start1
        v2 = end2 - start2
        
        # 检查角度
        angle = self._undirected_angle_deg(v1, v2)
        if angle > self.angle_threshold_deg:
            return None
        
        # 拟合合并后的直线
        max_err, p0, p1 = self._fit_line_pca(combined_pts)
        
        # 检查误差（允许误差增加）
        max_allowed_error = max(
            float(seg1.get('error', 0.0)),
            float(seg2.get('error', 0.0))
        ) * self.error_tolerance_factor
        
        if max_err <= max_allowed_error:
            return {
                'type': 'line',
                'start': p0,
                'end': p1,
                'points': combined_pts,
                'error': max_err
            }
        
        return None
    
    def _merge_arc_segments(self, seg1: Dict, seg2: Dict, combined_pts: np.ndarray) -> Optional[Dict]:
        """合并圆弧段"""
        # 检查是否同一圆
        center1 = np.asarray(seg1.get('center', [0, 0]))
        center2 = np.asarray(seg2.get('center', [0, 0]))
        radius1 = float(seg1.get('radius', 0.0))
        radius2 = float(seg2.get('radius', 0.0))
        
        center_diff = np.linalg.norm(center1 - center2)
        radius_diff = abs(radius1 - radius2)
        
        # 放宽合并条件
        if center_diff < self.distance_threshold * 2 and radius_diff < self.distance_threshold:
            # 使用平均圆心和半径
            center = (center1 + center2) / 2.0
            radius = (radius1 + radius2) / 2.0
            
            # 计算合并后的误差
            dists = np.linalg.norm(combined_pts - center, axis=1)
            errors = np.abs(dists - radius)
            max_err = np.max(errors)
            
            # 检查误差
            max_allowed_error = max(
                float(seg1.get('error', 0.0)),
                float(seg2.get('error', 0.0))
            ) * self.error_tolerance_factor
            
            if max_err <= max_allowed_error:
                return {
                    'type': 'arc',
                    'center': center,
                    'radius': radius,
                    'points': combined_pts,
                    'error': max_err
                }
        
        return None
    
    def _merge_spline_segments(self, seg1: Dict, seg2: Dict, combined_pts: np.ndarray) -> Optional[Dict]:
        """合并样条段"""
        # 简单合并：直接连接
        max_err = max(
            float(seg1.get('error', 0.0)),
            float(seg2.get('error', 0.0))
        )
        
        return {
            'type': seg1.get('type', 'spline'),
            'points': combined_pts,
            'error': max_err
        }
    
    def _try_merge_different_types(self, seg1: Dict, seg2: Dict, combined_pts: np.ndarray) -> Optional[Dict]:
        """
        尝试合并不同类段（如果误差可接受）
        优先转换为更简单的类型
        """
        # 尝试拟合为直线
        max_err, p0, p1 = self._fit_line_pca(combined_pts)
        if (seg1.get('type') == 'line' or seg2.get('type') == 'line') and max_err < 1.0:
            return {
                'type': 'line',
                'start': p0,
                'end': p1,
                'points': combined_pts,
                'error': max_err
            }
        
        # 否则合并为样条
        return self._merge_spline_segments(seg1, seg2, combined_pts)
    
    def _force_merge_to_limit(self, segments: List[Dict], max_segments: int) -> List[Dict]:
        """
        强制合并到指定段数上限
        优先合并误差最小的相邻段对
        """
        merged = list(segments)
        
        while len(merged) > max_segments:
            # 找到误差最小的相邻段对
            best_idx = 0
            best_error = float('inf')
            
            for i in range(len(merged) - 1):
                seg1 = merged[i]
                seg2 = merged[i + 1]
                error = max(
                    float(seg1.get('error', 0.0)),
                    float(seg2.get('error', 0.0))
                )
                
                if error < best_error:
                    best_error = error
                    best_idx = i
            
            # 合并最佳段对
            if best_idx < len(merged) - 1:
                merged_seg = self._try_merge_two_segments(merged[best_idx], merged[best_idx + 1])
                if merged_seg is not None:
                    merged[best_idx] = merged_seg
                    merged.pop(best_idx + 1)
                else:
                    # 如果无法合并，强制合并为样条
                    pts1 = self._get_segment_points(merged[best_idx])
                    pts2 = self._get_segment_points(merged[best_idx + 1])
                    combined_pts = np.vstack([pts1, pts2[1:]])
                    merged[best_idx] = {
                        'type': 'spline',
                        'points': combined_pts,
                        'error': best_error
                    }
                    merged.pop(best_idx + 1)
            else:
                break
        
        return merged
    
    # ========== 辅助方法 ==========
    
    def _get_segment_points(self, seg: Dict) -> np.ndarray:
        """获取段的点数组"""
        if 'points' in seg:
            return np.asarray(seg['points'], dtype=float).reshape(-1, 2)
        elif 'start' in seg and 'end' in seg:
            return np.array([seg['start'], seg['end']], dtype=float)
        else:
            return np.zeros((0, 2), dtype=float)
    
    def _segment_length(self, seg: Dict) -> float:
        """计算段的长度"""
        pts = self._get_segment_points(seg)
        if len(pts) < 2:
            return 0.0
        diffs = np.diff(pts, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))
    
    def _undirected_angle_deg(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算无向角度（度）"""
        v1 = np.asarray(v1, dtype=float)
        v2 = np.asarray(v2, dtype=float)
        n1 = np.linalg.norm(v1) + 1e-10
        n2 = np.linalg.norm(v2) + 1e-10
        cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosv)))
        return min(ang, 180.0 - ang)
    
    def _fit_line_pca(self, points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """使用PCA拟合直线"""
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) < 2:
            return 0.0, pts[0] if len(pts) > 0 else np.zeros(2), pts[-1] if len(pts) > 0 else np.zeros(2)
        
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
