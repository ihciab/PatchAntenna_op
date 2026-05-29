"""
曲线拟合评估模块
提供精细的拟合效果评估指标

评估维度：
1. 几何精度（MSE, Hausdorff距离, 面积误差）
2. 拓扑一致性（连通性, 嵌套关系）
3. 语义准确性（直线/圆弧识别准确率）
4. 工程适用性（规整度, 可制造性）
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.optimize import linear_sum_assignment


class CurveEvaluator:
    """
    曲线拟合评估器
    提供多维度、精细的拟合效果评估
    """
    
    def __init__(self, original_contour: np.ndarray, fitted_contour: np.ndarray,
                 original_img: Optional[np.ndarray] = None,
                 structure_type: Optional[str] = None):
        """
        初始化评估器
        
        参数:
            original_contour: 原始轮廓点
            fitted_contour: 拟合后的轮廓点
            original_img: 原始图像（可选，用于面积计算）
            structure_type: 结构类型（patch/solid/wire/slot）
        """
        self.original_contour = original_contour
        self.fitted_contour = fitted_contour
        self.original_img = original_img
        self.structure_type = structure_type
        
        # 缓存计算结果
        self._metrics_cache = {}
    
    def evaluate_all(self) -> Dict:
        """执行所有评估指标"""
        metrics = {}
        
        # 1. 几何精度
        metrics['geometric'] = self.evaluate_geometric_accuracy()
        
        # 2. 拓扑一致性
        metrics['topology'] = self.evaluate_topology_consistency()
        
        # 3. 语义准确性（如果有几何基元信息）
        metrics['semantic'] = self.evaluate_semantic_accuracy()
        
        # 4. 工程适用性
        metrics['engineering'] = self.evaluate_engineering_fitness()
        
        # 5. 综合评分
        metrics['overall'] = self.compute_overall_score(metrics)
        
        return metrics
    
    def evaluate_geometric_accuracy(self) -> Dict:
        """
        评估几何精度
        包括：MSE, Hausdorff距离, 面积误差, 周长误差
        """
        if 'geometric' in self._metrics_cache:
            return self._metrics_cache['geometric']
        
        metrics = {}
        
        # 1. 均方误差（MSE）
        mse = self._compute_mse()
        metrics['mse'] = mse
        metrics['rmse'] = np.sqrt(mse)
        
        # 2. 平均绝对误差（MAE）
        mae = self._compute_mae()
        metrics['mae'] = mae
        
        # 3. Hausdorff距离（最大偏差）
        hausdorff_dist = self._compute_hausdorff_distance()
        metrics['hausdorff_distance'] = hausdorff_dist
        
        # 4. 面积误差
        area_error = self._compute_area_error()
        metrics['area_error'] = area_error
        metrics['area_error_relative'] = area_error['relative']
        
        # 5. 周长误差
        perimeter_error = self._compute_perimeter_error()
        metrics['perimeter_error'] = perimeter_error
        metrics['perimeter_error_relative'] = perimeter_error['relative']
        
        # 6. 形状相似度（基于轮廓匹配）
        shape_similarity = self._compute_shape_similarity()
        metrics['shape_similarity'] = shape_similarity
        
        # 7. 关键点误差（如果有关键点）
        keypoint_error = self._compute_keypoint_error()
        if keypoint_error is not None:
            metrics['keypoint_error'] = keypoint_error
        
        self._metrics_cache['geometric'] = metrics
        return metrics
    
    def evaluate_topology_consistency(self) -> Dict:
        """
        评估拓扑一致性
        包括：连通性, 闭合性, 嵌套关系
        """
        if 'topology' in self._metrics_cache:
            return self._metrics_cache['topology']
        
        metrics = {}
        
        # 1. 闭合性检查
        is_closed_orig = self._is_contour_closed(self.original_contour)
        is_closed_fitted = self._is_contour_closed(self.fitted_contour)
        metrics['closed_consistency'] = is_closed_orig == is_closed_fitted
        metrics['is_closed'] = is_closed_fitted
        
        # 2. 连通性检查
        connectivity = self._check_connectivity()
        metrics['connectivity'] = connectivity
        
        # 3. 自相交检查
        self_intersection = self._check_self_intersection()
        metrics['self_intersection'] = self_intersection
        
        # 4. 方向一致性（顺时针/逆时针）
        orientation_consistency = self._check_orientation_consistency()
        metrics['orientation_consistency'] = orientation_consistency
        
        self._metrics_cache['topology'] = metrics
        return metrics
    
    def evaluate_semantic_accuracy(self, primitives: Optional[List[Dict]] = None) -> Dict:
        """
        评估语义准确性
        评估几何基元（直线/圆弧）的识别准确率
        """
        if 'semantic' in self._metrics_cache and primitives is None:
            return self._metrics_cache['semantic']
        
        metrics = {}
        
        if primitives is None:
            # 如果没有提供基元信息，返回空指标
            metrics['primitives_count'] = 0
            metrics['line_ratio'] = 0.0
            metrics['arc_ratio'] = 0.0
            metrics['bspline_ratio'] = 0.0
            return metrics
        
        # 统计基元类型
        total_length = 0
        line_length = 0
        arc_length = 0
        bspline_length = 0
        
        for prim in primitives:
            if 'points' in prim:
                points = prim['points']
                if len(points) > 1:
                    # 计算长度
                    seg_length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
                    total_length += seg_length
                    
                    if prim['type'] == 'line':
                        line_length += seg_length
                    elif prim['type'] == 'arc':
                        arc_length += seg_length
                    elif prim['type'] == 'bspline':
                        bspline_length += seg_length
        
        metrics['primitives_count'] = len(primitives)
        if total_length > 0:
            metrics['line_ratio'] = line_length / total_length
            metrics['arc_ratio'] = arc_length / total_length
            metrics['bspline_ratio'] = bspline_length / total_length
        else:
            metrics['line_ratio'] = 0.0
            metrics['arc_ratio'] = 0.0
            metrics['bspline_ratio'] = 0.0
        
        # 基元拟合误差
        if primitives:
            primitive_errors = []
            for prim in primitives:
                if 'error' in prim:
                    primitive_errors.append(prim['error'])
            if primitive_errors:
                metrics['mean_primitive_error'] = np.mean(primitive_errors)
                metrics['max_primitive_error'] = np.max(primitive_errors)
        
        if primitives is not None:
            self._metrics_cache['semantic'] = metrics
        return metrics
    
    def evaluate_engineering_fitness(self) -> Dict:
        """
        评估工程适用性
        包括：规整度, 可制造性, 参数合理性
        """
        if 'engineering' in self._metrics_cache:
            return self._metrics_cache['engineering']
        
        metrics = {}
        
        # 1. 规整度（直角、共线等）
        regularity = self._compute_regularity()
        metrics['regularity'] = regularity
        
        # 2. 最小特征尺寸（可制造性）
        min_feature_size = self._compute_min_feature_size()
        metrics['min_feature_size'] = min_feature_size
        
        # 3. 角度规整度（接近90度的角度比例）
        angle_regularity = self._compute_angle_regularity()
        metrics['angle_regularity'] = angle_regularity
        
        # 4. 曲率连续性
        curvature_continuity = self._compute_curvature_continuity()
        metrics['curvature_continuity'] = curvature_continuity
        
        self._metrics_cache['engineering'] = metrics
        return metrics
    
    def compute_overall_score(self, metrics: Dict) -> Dict:
        """计算综合评分"""
        score = {}
        
        # 几何精度得分（0-100）
        geo = metrics.get('geometric', {})
        geo_score = 100.0
        if 'rmse' in geo:
            # RMSE越小越好，归一化到0-100
            rmse = geo['rmse']
            geo_score = max(0, 100 - min(rmse * 10, 100))
        score['geometric'] = geo_score
        
        # 拓扑一致性得分
        topo = metrics.get('topology', {})
        topo_score = 100.0
        if not topo.get('closed_consistency', True):
            topo_score -= 20
        if not topo.get('connectivity', {}).get('is_connected', True):
            topo_score -= 30
        if topo.get('self_intersection', False):
            topo_score -= 25
        if not topo.get('orientation_consistency', True):
            topo_score -= 10
        topo_score = max(0, topo_score)
        score['topology'] = topo_score
        
        # 语义准确性得分
        sem = metrics.get('semantic', {})
        sem_score = 100.0
        if 'mean_primitive_error' in sem:
            error = sem['mean_primitive_error']
            sem_score = max(0, 100 - min(error * 20, 100))
        score['semantic'] = sem_score
        
        # 工程适用性得分
        eng = metrics.get('engineering', {})
        eng_score = eng.get('regularity', {}).get('score', 50.0)
        score['engineering'] = eng_score
        
        # 综合得分（加权平均）
        weights = {
            'geometric': 0.4,
            'topology': 0.3,
            'semantic': 0.2,
            'engineering': 0.1
        }
        
        overall = sum(weights[k] * score[k] for k in weights.keys())
        score['overall'] = overall
        
        return score
    
    # ========== 辅助计算方法 ==========
    
    def _compute_mse(self) -> float:
        """计算均方误差"""
        # 使用最近点匹配
        dist_matrix = cdist(self.original_contour, self.fitted_contour)
        min_dists_orig = np.min(dist_matrix, axis=1)
        min_dists_fitted = np.min(dist_matrix, axis=0)
        
        mse = (np.mean(min_dists_orig**2) + np.mean(min_dists_fitted**2)) / 2
        return mse
    
    def _compute_mae(self) -> float:
        """计算平均绝对误差"""
        dist_matrix = cdist(self.original_contour, self.fitted_contour)
        min_dists_orig = np.min(dist_matrix, axis=1)
        min_dists_fitted = np.min(dist_matrix, axis=0)
        
        mae = (np.mean(min_dists_orig) + np.mean(min_dists_fitted)) / 2
        return mae
    
    def _compute_hausdorff_distance(self) -> float:
        """计算Hausdorff距离"""
        try:
            dist1 = directed_hausdorff(self.original_contour, self.fitted_contour)[0]
            dist2 = directed_hausdorff(self.fitted_contour, self.original_contour)[0]
            return max(dist1, dist2)
        except:
            return float('inf')
    
    def _compute_area_error(self) -> Dict:
        """计算面积误差"""
        try:
            # 确保轮廓格式正确（cv2.contourArea需要特定格式）
            orig_contour = self._ensure_contour_format(self.original_contour)
            fitted_contour = self._ensure_contour_format(self.fitted_contour)
            
            if orig_contour is None or fitted_contour is None:
                return {
                    'absolute': float('inf'),
                    'relative': float('inf'),
                    'original': 0.0,
                    'fitted': 0.0
                }
            
            area_orig = cv2.contourArea(orig_contour)
            area_fitted = cv2.contourArea(fitted_contour)
            
            absolute_error = abs(area_orig - area_fitted)
            relative_error = absolute_error / max(area_orig, 1.0)
            
            return {
                'absolute': absolute_error,
                'relative': relative_error,
                'original': area_orig,
                'fitted': area_fitted
            }
        except Exception as e:
            # 如果计算失败，返回默认值
            return {
                'absolute': float('inf'),
                'relative': float('inf'),
                'original': 0.0,
                'fitted': 0.0
            }
    
    def _ensure_contour_format(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        确保轮廓格式符合cv2.contourArea的要求
        cv2.contourArea需要：(N, 1, 2)或(N, 2)格式，dtype为int32或float32
        """
        if contour is None or len(contour) == 0:
            return None
        
        try:
            # 转换为numpy数组
            if not isinstance(contour, np.ndarray):
                contour = np.array(contour)
            
            # 确保是2D数组
            if len(contour.shape) == 1:
                # 一维数组，尝试reshape
                if len(contour) % 2 == 0:
                    contour = contour.reshape(-1, 2)
                else:
                    return None
            elif len(contour.shape) == 2:
                # 二维数组
                if contour.shape[1] != 2:
                    # 尝试转置
                    if contour.shape[0] == 2:
                        contour = contour.T
                    else:
                        return None
            elif len(contour.shape) == 3:
                # 三维数组 (N, 1, 2)格式
                if contour.shape[2] == 2:
                    contour = contour.reshape(-1, 2)
                else:
                    return None
            else:
                return None
            
            # 确保至少3个点
            if len(contour) < 3:
                return None
            
            # 转换为正确的数据类型（int32或float32）
            if contour.dtype not in [np.int32, np.float32]:
                # 如果是整数类型，转换为int32；否则转换为float32
                if np.issubdtype(contour.dtype, np.integer):
                    contour = contour.astype(np.int32)
                else:
                    contour = contour.astype(np.float32)
            
            return contour
        except Exception:
            return None
    
    def _compute_perimeter_error(self) -> Dict:
        """计算周长误差"""
        try:
            # 确保轮廓格式正确
            orig_contour = self._ensure_contour_format(self.original_contour)
            fitted_contour = self._ensure_contour_format(self.fitted_contour)
            
            if orig_contour is None or fitted_contour is None:
                return {
                    'absolute': float('inf'),
                    'relative': float('inf'),
                    'original': 0.0,
                    'fitted': 0.0
                }
            
            peri_orig = cv2.arcLength(orig_contour, True)
            peri_fitted = cv2.arcLength(fitted_contour, True)
            
            absolute_error = abs(peri_orig - peri_fitted)
            relative_error = absolute_error / max(peri_orig, 1.0)
            
            return {
                'absolute': absolute_error,
                'relative': relative_error,
                'original': peri_orig,
                'fitted': peri_fitted
            }
        except Exception:
            return {
                'absolute': float('inf'),
                'relative': float('inf'),
                'original': 0.0,
                'fitted': 0.0
            }
    
    def _compute_shape_similarity(self) -> float:
        """计算形状相似度（基于Hu矩）"""
        try:
            # 确保轮廓格式正确
            orig_contour = self._ensure_contour_format(self.original_contour)
            fitted_contour = self._ensure_contour_format(self.fitted_contour)
            
            if orig_contour is None or fitted_contour is None:
                return 0.0
            
            # 计算Hu矩
            moments_orig = cv2.moments(orig_contour)
            moments_fitted = cv2.moments(fitted_contour)
            
            hu_orig = cv2.HuMoments(moments_orig).flatten()
            hu_fitted = cv2.HuMoments(moments_fitted).flatten()
            
            # 归一化（对数变换）
            hu_orig = -np.sign(hu_orig) * np.log10(np.abs(hu_orig) + 1e-10)
            hu_fitted = -np.sign(hu_fitted) * np.log10(np.abs(hu_fitted) + 1e-10)
            
            # 计算相似度（1 - 归一化距离）
            dist = np.linalg.norm(hu_orig - hu_fitted)
            similarity = 1.0 / (1.0 + dist)
            
            return similarity
        except:
            return 0.0
    
    def _compute_keypoint_error(self) -> Optional[Dict]:
        """计算关键点误差（如果有关键点）"""
        # 简化实现：使用轮廓的极值点作为关键点
        try:
            # 找到极值点
            extrema_orig = self._find_extrema_points(self.original_contour)
            extrema_fitted = self._find_extrema_points(self.fitted_contour)
            
            if len(extrema_orig) == 0 or len(extrema_fitted) == 0:
                return None
            
            # 匹配最近的关键点
            dist_matrix = cdist(extrema_orig, extrema_fitted)
            min_dists = np.min(dist_matrix, axis=1)
            
            return {
                'mean_error': np.mean(min_dists),
                'max_error': np.max(min_dists),
                'count_original': len(extrema_orig),
                'count_fitted': len(extrema_fitted)
            }
        except:
            return None
    
    def _find_extrema_points(self, contour: np.ndarray) -> np.ndarray:
        """找到轮廓的极值点（最左、最右、最上、最下）"""
        if len(contour) == 0:
            return np.array([])
        
        x = contour[:, 0]
        y = contour[:, 1]
        
        extrema = []
        extrema.append(contour[np.argmin(x)])  # 最左
        extrema.append(contour[np.argmax(x)])  # 最右
        extrema.append(contour[np.argmin(y)])  # 最上
        extrema.append(contour[np.argmax(y)])  # 最下
        
        return np.array(extrema)
    
    def _is_contour_closed(self, contour: np.ndarray) -> bool:
        """检查轮廓是否闭合"""
        try:
            if contour is None or len(contour) < 3:
                return False
            # 确保是2D数组
            if len(contour.shape) == 2 and contour.shape[1] == 2:
                return np.allclose(contour[0], contour[-1], atol=1.0)
            return False
        except:
            return False
    
    def _check_connectivity(self) -> Dict:
        """检查连通性"""
        # 简化实现：检查点之间的距离
        if len(self.fitted_contour) < 2:
            return {'is_connected': False, 'max_gap': float('inf')}
        
        gaps = []
        for i in range(len(self.fitted_contour) - 1):
            dist = np.linalg.norm(self.fitted_contour[i+1] - self.fitted_contour[i])
            gaps.append(dist)
        
        max_gap = np.max(gaps) if gaps else 0.0
        is_connected = max_gap < 10.0  # 阈值可调
        
        return {
            'is_connected': is_connected,
            'max_gap': max_gap,
            'mean_gap': np.mean(gaps) if gaps else 0.0
        }
    
    def _check_self_intersection(self) -> bool:
        """检查自相交（简化实现）"""
        contour = np.asarray(self.fitted_contour, dtype=float)
        if contour.ndim != 2 or contour.shape[1] != 2:
            return False
        if len(contour) < 4:
            return False

        is_closed = bool(np.allclose(contour[0], contour[-1], atol=1.0))
        if is_closed:
            contour = contour[:-1]
        n = len(contour)
        if n < 4:
            return False

        segments = [(i, (i + 1) % n) for i in range(n)] if is_closed else [(i, i + 1) for i in range(n - 1)]
        for a in range(len(segments)):
            i1, i2 = segments[a]
            p1, p2 = contour[i1], contour[i2]
            for b in range(a + 1, len(segments)):
                j1, j2 = segments[b]
                if i1 in (j1, j2) or i2 in (j1, j2):
                    continue
                p3, p4 = contour[j1], contour[j2]
                if self._segments_intersect(p1, p2, p3, p4):
                    return True
        return False
    
    def _segments_intersect(self, p1: np.ndarray, p2: np.ndarray,
                           p3: np.ndarray, p4: np.ndarray) -> bool:
        """检查两条线段是否相交"""
        p1 = np.asarray(p1, dtype=float).reshape(2)
        p2 = np.asarray(p2, dtype=float).reshape(2)
        p3 = np.asarray(p3, dtype=float).reshape(2)
        p4 = np.asarray(p4, dtype=float).reshape(2)

        def orient(a, b, c) -> float:
            return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)

        eps = 1e-12
        if abs(o1) <= eps or abs(o2) <= eps or abs(o3) <= eps or abs(o4) <= eps:
            return False

        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
    
    def _check_orientation_consistency(self) -> bool:
        """检查方向一致性"""
        try:
            # 确保轮廓格式正确
            orig_contour = self._ensure_contour_format(self.original_contour)
            fitted_contour = self._ensure_contour_format(self.fitted_contour)
            
            if orig_contour is None or fitted_contour is None:
                return False
            
            # 计算有向面积
            area_orig = cv2.contourArea(orig_contour, oriented=True)
            area_fitted = cv2.contourArea(fitted_contour, oriented=True)
            
            # 符号相同表示方向一致
            return np.sign(area_orig) == np.sign(area_fitted)
        except:
            return False
    
    def _compute_regularity(self) -> Dict:
        """计算规整度"""
        # 检查直角、共线等
        right_angle_count = 0
        total_angles = 0
        
        if len(self.fitted_contour) >= 3:
            for i in range(len(self.fitted_contour)):
                p1 = self.fitted_contour[i]
                p2 = self.fitted_contour[(i+1) % len(self.fitted_contour)]
                p3 = self.fitted_contour[(i+2) % len(self.fitted_contour)]
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angle_deg = np.degrees(angle)
                    
                    total_angles += 1
                    # 检查是否接近90度
                    if abs(angle_deg - 90) < 5 or abs(angle_deg - 270) < 5:
                        right_angle_count += 1
        
        regularity_score = (right_angle_count / max(total_angles, 1)) * 100.0
        
        return {
            'right_angle_ratio': right_angle_count / max(total_angles, 1),
            'score': regularity_score
        }
    
    def _compute_min_feature_size(self) -> float:
        """计算最小特征尺寸"""
        if len(self.fitted_contour) < 2:
            return 0.0
        
        # 计算所有边的长度
        edge_lengths = []
        for i in range(len(self.fitted_contour)):
            p1 = self.fitted_contour[i]
            p2 = self.fitted_contour[(i+1) % len(self.fitted_contour)]
            length = np.linalg.norm(p2 - p1)
            if length > 1e-10:
                edge_lengths.append(length)
        
        return np.min(edge_lengths) if edge_lengths else 0.0
    
    def _compute_angle_regularity(self) -> float:
        """计算角度规整度（接近90度的角度比例）"""
        return self._compute_regularity()['right_angle_ratio']
    
    def _compute_curvature_continuity(self) -> float:
        """计算曲率连续性"""
        if len(self.fitted_contour) < 3:
            return 0.0
        
        # 计算曲率变化
        curvatures = []
        for i in range(len(self.fitted_contour)):
            p1 = self.fitted_contour[i]
            p2 = self.fitted_contour[(i+1) % len(self.fitted_contour)]
            p3 = self.fitted_contour[(i+2) % len(self.fitted_contour)]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                # 简化的曲率估计
                cross = np.cross(v1, v2)
                curvature = abs(cross) / (np.linalg.norm(v1)**3 + 1e-10)
                curvatures.append(curvature)
        
        if len(curvatures) < 2:
            return 0.0
        
        # 计算曲率变化的平滑度
        curvature_changes = np.abs(np.diff(curvatures))
        continuity = 1.0 / (1.0 + np.mean(curvature_changes))
        
        return continuity
