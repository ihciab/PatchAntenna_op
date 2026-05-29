"""
几何感知评分指标体系
专门针对Segmented Line/Arc/NURBS Spline结构的评估

设计目标：
- 不再用单一MSE评估所有方法
- 回答"你到底在哪一类几何上做得好/不好"
- 工程可用 + 论文可复现

评估指标：
1. Line Accuracy（直线准确率）
2. Arc Radius Error（圆弧半径误差）
3. Spline Residual（自由曲线残差）
4. Structure Recognition Accuracy（结构识别准确率）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import linear_sum_assignment


class GeometryAwareMetrics:
    """
    几何感知评分器
    专门针对Line/Arc/Spline混合结构的评估
    """
    
    def __init__(self, 
                 angle_threshold: float = 2.0,
                 distance_threshold: float = 2.0,
                 arc_tau: float = 0.05,
                 spline_sigma: float = 0.01,
                 w_line: float = 0.35,
                 w_arc: float = 0.40,
                 w_spline: float = 0.25):
        """
        初始化评分器
        
        参数:
            angle_threshold: 直线方向误差阈值（度）
            distance_threshold: 直线偏移误差阈值（像素）
            arc_tau: 圆弧半径误差衰减系数（相对误差）
            spline_sigma: 自由曲线残差衰减系数（归一化）
            w_line: 直线准确率权重
            w_arc: 圆弧半径得分权重
            w_spline: 自由曲线残差得分权重
        """
        self.angle_threshold = angle_threshold
        self.distance_threshold = distance_threshold
        self.arc_tau = arc_tau
        self.spline_sigma = spline_sigma
        self.w_line = w_line
        self.w_arc = w_arc
        self.w_spline = w_spline
    
    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        计算两个向量之间的角度（度）
        
        参数:
            v1, v2: 方向向量
        
        返回:
            角度（度）
        """
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    @staticmethod
    def line_distance(points: np.ndarray, 
                     point_on_line: np.ndarray,
                     direction: np.ndarray) -> float:
        """
        计算点到直线的最大垂直距离
        
        参数:
            points: 点集 (N, 2)
            point_on_line: 直线上一点
            direction: 方向向量（单位向量）
        
        返回:
            最大垂直距离
        """
        if len(points) == 0:
            return 0.0
        
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        vecs = points - point_on_line
        proj = vecs @ direction
        closest = np.outer(proj, direction) + point_on_line
        distances = np.linalg.norm(points - closest, axis=1)
        
        return float(np.max(distances))
    
    def evaluate_line_accuracy(self, 
                               gt_segments: List[Dict],
                               pred_segments: List[Dict]) -> Dict:
        """
        评估直线准确率
        
        参数:
            gt_segments: Ground Truth分段列表
            pred_segments: 预测分段列表
        
        返回:
            评估结果字典
        """
        # 提取直线段
        gt_lines = [s for s in gt_segments if s.get('type') == 'line']
        pred_lines = [s for s in pred_segments if s.get('type') == 'line']
        
        if len(gt_lines) == 0:
            return {
                'accuracy': 1.0,  # 如果没有GT直线，认为完美
                'correct_count': 0,
                'total_count': 0,
                'angle_errors': [],
                'distance_errors': []
            }

        def _endpoints(seg: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            pts = seg.get('points', None)
            if pts is None:
                return None
            pts = np.asarray(pts, dtype=float).reshape(-1, 2)
            if len(pts) < 2:
                return None
            p0 = pts[0]
            p1 = pts[-1]
            if np.linalg.norm(p1 - p0) < 1e-8:
                return None
            return p0, p1, pts

        def _undirected_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
            a = float(self.angle_between(v1, v2))
            return min(a, 180.0 - a)

        def _point_to_line_dist(p: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> float:
            u = line_dir / (np.linalg.norm(line_dir) + 1e-10)
            v = p - line_point
            proj = float(np.dot(v, u))
            closest = line_point + proj * u
            return float(np.linalg.norm(p - closest))

        def _segment_offset_dist(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
            da = a1 - a0
            db = b1 - b0
            d1 = max(_point_to_line_dist(a0, b0, db), _point_to_line_dist(a1, b0, db))
            d2 = max(_point_to_line_dist(b0, a0, da), _point_to_line_dist(b1, a0, da))
            return 0.5 * (d1 + d2)
        
        gt_reps = []
        for seg in gt_lines:
            rep = _endpoints(seg)
            if rep is not None:
                gt_reps.append((seg, rep[0], rep[1], rep[2]))

        pred_reps = []
        for seg in pred_lines:
            rep = _endpoints(seg)
            if rep is not None:
                pred_reps.append((seg, rep[0], rep[1], rep[2]))

        if len(gt_reps) == 0:
            return {
                'accuracy': 1.0,
                'correct_count': 0,
                'total_count': 0,
                'angle_errors': [],
                'distance_errors': [],
                'mean_angle_error': 0.0,
                'mean_distance_error': 0.0,
                'matched_count': 0
            }

        if len(pred_reps) == 0:
            return {
                'accuracy': 0.0,
                'correct_count': 0,
                'total_count': len(gt_reps),
                'angle_errors': [],
                'distance_errors': [],
                'mean_angle_error': 0.0,
                'mean_distance_error': 0.0,
                'matched_count': 0
            }

        m = len(gt_reps)
        n = len(pred_reps)
        cost = np.full((m, n), 1e9, dtype=float)
        angle_mat = np.full((m, n), np.nan, dtype=float)
        dist_mat = np.full((m, n), np.nan, dtype=float)

        for i, (_gseg, g0, g1, _gpts) in enumerate(gt_reps):
            gd = g1 - g0
            for j, (_pseg, p0, p1, _ppts) in enumerate(pred_reps):
                pd = p1 - p0
                angle_err = _undirected_angle_deg(gd, pd)
                dist_err = _segment_offset_dist(g0, g1, p0, p1)
                angle_mat[i, j] = angle_err
                dist_mat[i, j] = dist_err
                cost[i, j] = (angle_err / max(self.angle_threshold, 1e-9)) + (dist_err / max(self.distance_threshold, 1e-9))

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_pairs = []
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            if not np.isfinite(cost[i, j]) or cost[i, j] > 1e8:
                continue
            matched_pairs.append((gt_reps[i][0], pred_reps[j][0], float(angle_mat[i, j]), float(dist_mat[i, j])))
        
        # 计算准确率
        correct_count = 0
        angle_errors = []
        distance_errors = []
        
        for gt_line, pred_line, angle_err, dist_err in matched_pairs:
            angle_errors.append(angle_err)
            distance_errors.append(dist_err)
            
            # 判断是否正确（角度和距离都在阈值内）
            if angle_err < self.angle_threshold and dist_err < self.distance_threshold:
                correct_count += 1
        
        accuracy = correct_count / max(len(gt_reps), 1)
        
        return {
            'accuracy': float(accuracy),
            'correct_count': correct_count,
            'total_count': len(gt_reps),
            'angle_errors': angle_errors,
            'distance_errors': distance_errors,
            'mean_angle_error': float(np.mean(angle_errors)) if angle_errors else 0.0,
            'mean_distance_error': float(np.mean(distance_errors)) if distance_errors else 0.0,
            'matched_count': len(matched_pairs)
        }
    
    def evaluate_arc_radius_score(self,
                                  gt_segments: List[Dict],
                                  pred_segments: List[Dict]) -> Dict:
        """
        评估圆弧半径得分
        
        参数:
            gt_segments: Ground Truth分段列表
            pred_segments: 预测分段列表
        
        返回:
            评估结果字典
        """
        # 提取圆弧段
        gt_arcs = [s for s in gt_segments if s.get('type') == 'arc']
        pred_arcs = [s for s in pred_segments if s.get('type') == 'arc']
        
        if len(gt_arcs) == 0:
            return {
                'score': 1.0,  # 如果没有GT圆弧，认为完美
                'mean_relative_error': 0.0,
                'matched_count': 0,
                'total_count': 0,
                'relative_errors': []
            }
        
        # 匹配GT和预测的圆弧段
        matched_pairs = []
        used_pred = set()
        
        for gt_arc in gt_arcs:
            best_match = None
            best_score = float('inf')
            
            # 获取GT圆弧参数
            if 'radius' in gt_arc:
                gt_radius = float(gt_arc['radius'])
            elif 'model' in gt_arc and 'radius' in gt_arc['model']:
                gt_radius = float(gt_arc['model']['radius'])
            else:
                continue
            
            for i, pred_arc in enumerate(pred_arcs):
                if i in used_pred:
                    continue
                
                # 获取预测圆弧参数
                if 'radius' in pred_arc:
                    pred_radius = float(pred_arc['radius'])
                elif 'model' in pred_arc and 'radius' in pred_arc['model']:
                    pred_radius = float(pred_arc['model']['radius'])
                else:
                    continue
                
                # 计算相对误差
                rel_error = abs(pred_radius - gt_radius) / (gt_radius + 1e-9)
                
                if rel_error < best_score:
                    best_score = rel_error
                    best_match = (i, pred_arc, rel_error)
            
            if best_match is not None:
                matched_pairs.append((gt_arc, best_match[1], best_match[2]))
                used_pred.add(best_match[0])
        
        # 计算得分
        relative_errors = [err for _, _, err in matched_pairs]
        
        if len(relative_errors) == 0:
            return {
                'score': 0.0,
                'mean_relative_error': 0.0,
                'matched_count': 0,
                'total_count': len(gt_arcs),
                'relative_errors': []
            }
        
        # 使用指数衰减计算得分
        scores = [np.exp(-err / self.arc_tau) for err in relative_errors]
        mean_score = float(np.mean(scores))
        mean_rel_error = float(np.mean(relative_errors))
        
        return {
            'score': mean_score,
            'mean_relative_error': mean_rel_error,
            'matched_count': len(matched_pairs),
            'total_count': len(gt_arcs),
            'relative_errors': relative_errors,
            'individual_scores': scores
        }
    
    def evaluate_spline_residual_score(self,
                                      gt_segments: List[Dict],
                                      pred_segments: List[Dict]) -> Dict:
        """
        评估自由曲线残差得分
        
        参数:
            gt_segments: Ground Truth分段列表
            pred_segments: 预测分段列表
        
        返回:
            评估结果字典
        """
        # 【优化2】只提取真正的spline段（不是line或arc）
        # 排除被识别为line或arc的段，只计算真正的自由曲线
        # 避免惩罚"几何语义识别能力"
        gt_splines = [s for s in gt_segments 
                     if s.get('type') in ['spline', 'nurbs', 'bspline', 'curve']
                     and s.get('type') not in ['line', 'arc', 'circle']]
        pred_splines = [s for s in pred_segments 
                       if s.get('type') in ['spline', 'nurbs', 'bspline', 'curve']
                       and s.get('type') not in ['line', 'arc', 'circle']]
        
        if len(gt_splines) == 0:
            return {
                'score': 1.0,  # 如果没有GT spline，认为完美
                'mean_normalized_residual': 0.0,
                'matched_count': 0,
                'total_count': 0,
                'normalized_residuals': []
            }
        
        # 匹配GT和预测的Spline段
        matched_pairs = []
        used_pred = set()
        
        for gt_spline in gt_splines:
            best_match = None
            best_residual = float('inf')
            
            gt_points = np.array(gt_spline.get('points', []))
            if len(gt_points) < 3:
                continue
            
            # 计算GT的bounding box
            bbox_gt = np.ptp(gt_points, axis=0)
            diagonal_gt = np.linalg.norm(bbox_gt) + 1e-9
            
            for i, pred_spline in enumerate(pred_splines):
                if i in used_pred:
                    continue
                
                # 获取预测的拟合点或曲线
                pred_points = None
                
                if 'points' in pred_spline:
                    pred_points = np.array(pred_spline['points'])
                elif 'fitted_points' in pred_spline:
                    pred_points = np.array(pred_spline['fitted_points'])
                elif 'model' in pred_spline and 'curve' in pred_spline['model']:
                    # 如果有曲线函数，采样点
                    curve_func = pred_spline['model']['curve']
                    if callable(curve_func):
                        t_samples = np.linspace(0, 1, 200)
                        pred_points = np.array([curve_func(t) for t in t_samples])
                
                if pred_points is None or len(pred_points) < 3:
                    continue
                
                # 计算点到曲线距离的RMS
                residuals = []
                for p in gt_points:
                    dists = np.linalg.norm(pred_points - p, axis=1)
                    residuals.append(np.min(dists))
                
                rms = np.sqrt(np.mean(np.square(residuals)))
                normalized_residual = rms / diagonal_gt
                
                if normalized_residual < best_residual:
                    best_residual = normalized_residual
                    best_match = (i, pred_spline, normalized_residual)
            
            if best_match is not None:
                matched_pairs.append((gt_spline, best_match[1], best_match[2]))
                used_pred.add(best_match[0])
        
        # 计算得分
        normalized_residuals = [res for _, _, res in matched_pairs]
        
        if len(normalized_residuals) == 0:
            return {
                'score': 0.0,
                'mean_normalized_residual': 0.0,
                'matched_count': 0,
                'total_count': len(gt_splines),
                'normalized_residuals': []
            }
        
        # 使用指数衰减计算得分
        scores = [np.exp(-res / self.spline_sigma) for res in normalized_residuals]
        mean_score = float(np.mean(scores))
        mean_residual = float(np.mean(normalized_residuals))
        
        return {
            'score': mean_score,
            'mean_normalized_residual': mean_residual,
            'matched_count': len(matched_pairs),
            'total_count': len(gt_splines),
            'normalized_residuals': normalized_residuals,
            'individual_scores': scores
        }
    
    def evaluate_structure_recognition_accuracy(self,
                                               gt_segments: List[Dict],
                                               pred_segments: List[Dict]) -> Dict:
        """
        评估结构识别准确率（附加指标）
        正确分类的Line/Arc/Spline段比例
        
        参数:
            gt_segments: Ground Truth分段列表
            pred_segments: 预测分段列表
        
        返回:
            评估结果字典
        """
        if len(gt_segments) == 0:
            return {
                'accuracy': 1.0,
                'correct_count': 0,
                'total_count': 0,
                'line_accuracy': 1.0,
                'arc_accuracy': 1.0,
                'spline_accuracy': 1.0
            }
        
        # 匹配分段（基于空间位置）
        matched_pairs = []
        used_pred = set()
        
        for gt_seg in gt_segments:
            best_match = None
            best_distance = float('inf')
            
            gt_points = np.array(gt_seg.get('points', []))
            if len(gt_points) == 0:
                continue
            
            gt_center = np.mean(gt_points, axis=0)
            gt_type = gt_seg.get('type', 'unknown')
            
            for i, pred_seg in enumerate(pred_segments):
                if i in used_pred:
                    continue
                
                pred_points = np.array(pred_seg.get('points', []))
                if len(pred_points) == 0:
                    continue
                
                pred_center = np.mean(pred_points, axis=0)
                distance = np.linalg.norm(gt_center - pred_center)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = (i, pred_seg)
            
            if best_match is not None and best_distance < 50.0:  # 距离阈值
                matched_pairs.append((gt_seg, best_match[1]))
                used_pred.add(best_match[0])
        
        # 计算类型匹配准确率
        correct_count = 0
        line_correct = 0
        line_total = 0
        arc_correct = 0
        arc_total = 0
        spline_correct = 0
        spline_total = 0
        
        for gt_seg, pred_seg in matched_pairs:
            gt_type = gt_seg.get('type', 'unknown')
            pred_type = pred_seg.get('type', 'unknown')
            
            # 类型映射（处理不同命名）
            type_map = {
                'line': 'line',
                'arc': 'arc',
                'circle': 'arc',
                'spline': 'spline',
                'nurbs': 'spline',
                'bspline': 'spline',
                'curve': 'spline'
            }
            
            gt_type_normalized = type_map.get(gt_type, gt_type)
            pred_type_normalized = type_map.get(pred_type, pred_type)
            
            if gt_type_normalized == pred_type_normalized:
                correct_count += 1
            
            # 按类型统计
            if gt_type_normalized == 'line':
                line_total += 1
                if gt_type_normalized == pred_type_normalized:
                    line_correct += 1
            elif gt_type_normalized == 'arc':
                arc_total += 1
                if gt_type_normalized == pred_type_normalized:
                    arc_correct += 1
            elif gt_type_normalized == 'spline':
                spline_total += 1
                if gt_type_normalized == pred_type_normalized:
                    spline_correct += 1
        
        total_count = len(gt_segments)
        accuracy = correct_count / max(total_count, 1)
        
        return {
            'accuracy': float(accuracy),
            'correct_count': correct_count,
            'total_count': total_count,
            'line_accuracy': line_correct / max(line_total, 1),
            'arc_accuracy': arc_correct / max(arc_total, 1),
            'spline_accuracy': spline_correct / max(spline_total, 1),
            'line_count': line_total,
            'arc_count': arc_total,
            'spline_count': spline_total
        }
    
    def compute_total_score(self,
                           gt_segments: List[Dict],
                           pred_segments: List[Dict]) -> Dict:
        """
        计算总评分
        
        参数:
            gt_segments: Ground Truth分段列表
            pred_segments: 预测分段列表
        
        返回:
            完整评估结果字典
        """
        # 检查是否有GT分段信息
        has_gt = len(gt_segments) > 0
        
        # 1. Line Accuracy
        if has_gt:
            line_metrics = self.evaluate_line_accuracy(gt_segments, pred_segments)
            S_line = line_metrics['accuracy']
        else:
            # 当GT为空时，基于预测分段的质量进行评估
            S_line = self._evaluate_line_quality(pred_segments)
            line_metrics = {'accuracy': S_line, 'self_evaluation': True}
        
        # 2. Arc Radius Score
        if has_gt:
            arc_metrics = self.evaluate_arc_radius_score(gt_segments, pred_segments)
            S_arc = arc_metrics['score']
        else:
            # 当GT为空时，基于预测分段的质量进行评估
            S_arc = self._evaluate_arc_quality(pred_segments)
            arc_metrics = {'score': S_arc, 'self_evaluation': True}
        
        # 3. Spline Residual Score
        if has_gt:
            spline_metrics = self.evaluate_spline_residual_score(gt_segments, pred_segments)
            S_spline = spline_metrics['score']
        else:
            # 当GT为空时，基于预测分段的质量进行评估
            S_spline = self._evaluate_spline_quality(pred_segments)
            spline_metrics = {'score': S_spline, 'self_evaluation': True}
        
        # 4. Structure Recognition Accuracy（附加指标）
        if has_gt:
            structure_metrics = self.evaluate_structure_recognition_accuracy(gt_segments, pred_segments)
        else:
            # 当GT为空时，无法计算识别准确率
            structure_metrics = {'accuracy': 0.0, 'self_evaluation': True}
        
        # 5. 总评分
        total_score = (
            self.w_line * S_line +
            self.w_arc * S_arc +
            self.w_spline * S_spline
        )
        
        return {
            'LineAccuracy': S_line,
            'ArcRadiusScore': S_arc,
            'SplineResidual': S_spline,
            'TotalScore': total_score,
            'StructureRecognitionAccuracy': structure_metrics['accuracy'],
            'details': {
                'line': line_metrics,
                'arc': arc_metrics,
                'spline': spline_metrics,
                'structure': structure_metrics
            },
            'weights': {
                'w_line': self.w_line,
                'w_arc': self.w_arc,
                'w_spline': self.w_spline
            }
        }
    
    @staticmethod
    def convert_segments_to_standard_format(segments: List[Dict]) -> List[Dict]:
        """
        将不同格式的分段转换为标准格式
        
        参数:
            segments: 分段列表（可能来自不同拟合器）
        
        返回:
            标准格式的分段列表
        """
        standard_segments = []
        
        for seg in segments:
            standard_seg = {
                'type': seg.get('type', 'unknown'),
                'points': np.array(seg.get('points', []))
            }
            
            # 根据类型提取模型参数
            seg_type = standard_seg['type']
            
            if seg_type == 'line':
                if 'start' in seg and 'end' in seg:
                    start = np.array(seg['start'])
                    end = np.array(seg['end'])
                    direction = end - start
                    standard_seg['model'] = {
                        'point': start,
                        'direction': direction
                    }
                elif 'p0' in seg and 'p1' in seg:
                    p0 = np.array(seg['p0'])
                    p1 = np.array(seg['p1'])
                    direction = p1 - p0
                    standard_seg['model'] = {
                        'point': p0,
                        'direction': direction
                    }
            
            elif seg_type == 'arc':
                if 'center' in seg and 'radius' in seg:
                    standard_seg['model'] = {
                        'center': np.array(seg['center']),
                        'radius': float(seg['radius'])
                    }
                elif 'model' in seg:
                    standard_seg['model'] = seg['model']
            
            elif seg_type in ['spline', 'nurbs', 'bspline']:
                # Spline段：尝试提取曲线函数或拟合点
                if 'model' in seg and 'curve' in seg['model']:
                    standard_seg['model'] = {'curve': seg['model']['curve']}
                elif 'fitted_points' in seg:
                    # 如果没有曲线函数，使用拟合点
                    fitted_points = np.array(seg['fitted_points'])
                    # 创建简单的线性插值函数
                    def curve_func(t):
                        idx = int(t * (len(fitted_points) - 1))
                        idx = np.clip(idx, 0, len(fitted_points) - 1)
                        return fitted_points[idx]
                    standard_seg['model'] = {'curve': curve_func}
                elif 'points' in seg:
                    # 使用原始点创建插值函数
                    points = np.array(seg['points'])
                    def curve_func(t):
                        idx = int(t * (len(points) - 1))
                        idx = np.clip(idx, 0, len(points) - 1)
                        return points[idx]
                    standard_seg['model'] = {'curve': curve_func}
            
            standard_segments.append(standard_seg)
        
        return standard_segments
    
    def _evaluate_line_quality(self, pred_segments: List[Dict]) -> float:
        """评估直线段质量（自评估模式，当GT为空时使用）"""
        line_segments = [s for s in pred_segments if s.get('type') == 'line']
        if len(line_segments) == 0:
            return 0.0
        
        scores = []
        for seg in line_segments:
            if 'points' in seg:
                points = np.array(seg['points'])
                if len(points) >= 2:
                    # 计算直线拟合误差
                    p0 = points[0]
                    p1 = points[-1]
                    v = p1 - p0
                    v_norm = np.linalg.norm(v)
                    if v_norm > 1e-6:
                        v = v / v_norm
                        proj = np.dot(points - p0, v)
                        closest = p0 + np.outer(proj, v)
                        errors = np.linalg.norm(points - closest, axis=1)
                        max_error = np.max(errors)
                        # 误差越小，得分越高
                        score = np.exp(-max_error / self.distance_threshold)
                        scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _evaluate_arc_quality(self, pred_segments: List[Dict]) -> float:
        """评估圆弧段质量（自评估模式，当GT为空时使用）"""
        arc_segments = [s for s in pred_segments if s.get('type') == 'arc']
        if len(arc_segments) == 0:
            return 0.0
        
        scores = []
        for seg in arc_segments:
            if 'points' in seg and 'radius' in seg:
                points = np.array(seg['points'])
                radius = float(seg['radius'])
                center = np.array(seg.get('center', [0, 0]))
                
                if len(points) >= 3 and radius > 0:
                    # 计算圆弧拟合误差
                    dist = np.linalg.norm(points - center, axis=1)
                    errors = np.abs(dist - radius)
                    max_error = np.max(errors)
                    # 相对误差
                    rel_error = max_error / (radius + 1e-9)
                    score = np.exp(-rel_error / self.arc_tau)
                    scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _evaluate_spline_quality(self, pred_segments: List[Dict]) -> float:
        """评估Spline段质量（自评估模式，当GT为空时使用）"""
        spline_segments = [s for s in pred_segments if s.get('type') in ['spline', 'nurbs', 'bspline']]
        if len(spline_segments) == 0:
            return 0.0
        
        scores = []
        for seg in spline_segments:
            if 'points' in seg:
                points = np.array(seg['points'])
                if len(points) >= 3:
                    # 计算点分布的均匀性（简化：基于点间距的方差）
                    diffs = np.diff(points, axis=0)
                    distances = np.linalg.norm(diffs, axis=1)
                    if len(distances) > 0:
                        # 均匀性得分（方差越小，得分越高）
                        mean_dist = np.mean(distances)
                        if mean_dist > 1e-6:
                            cv = np.std(distances) / mean_dist  # 变异系数
                            score = np.exp(-cv)  # 变异系数越小，得分越高
                            scores.append(score)
                        else:
                            scores.append(0.0)
        
        return float(np.mean(scores)) if scores else 0.0


def compute_geometry_aware_score(gt_segments: List[Dict],
                                 pred_segments: List[Dict],
                                 **kwargs) -> Dict:
    """
    便捷函数：计算几何感知评分
    
    参数:
        gt_segments: Ground Truth分段列表
        pred_segments: 预测分段列表
        **kwargs: 传递给GeometryAwareMetrics的参数
    
    返回:
        评估结果字典
    """
    evaluator = GeometryAwareMetrics(**kwargs)
    
    # 转换为标准格式
    gt_standard = evaluator.convert_segments_to_standard_format(gt_segments)
    pred_standard = evaluator.convert_segments_to_standard_format(pred_segments)
    
    # 计算评分
    return evaluator.compute_total_score(gt_standard, pred_standard)
