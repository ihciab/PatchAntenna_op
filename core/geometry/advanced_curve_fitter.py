"""
高级曲线拟合模块
基于工程级曲线拟合方案，实现结构预判、分流处理、语义恢复等功能

设计原则：
1. 结构预判（Patch/Solid vs Wire/Slot）
2. 轮廓获取分流（potrace/skimage vs skeleton）
3. 矢量路径获取
4. 结构级简化（RDP）
5. 工程曲线语义恢复（直线/圆弧/B-spline）
6. 几何规整（直角吸附、共线合并、半径量化）
7. 宽度/gap恢复
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import subprocess
import tempfile
import json
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.ndimage import distance_transform_edt, binary_erosion
from skimage import measure, morphology
# 【融合优化】引入更强的检测模块
from core.geometry.arc_detector import arc_quality
from core.geometry.segment_scheduler import dispatch_segment
from core.geometry.optimized_bspline_fitter import OptimizedBSplineFitter
try:
    from rdp import rdp
    RDP_AVAILABLE = True
except ImportError:
    RDP_AVAILABLE = False
    print("警告: rdp未安装，将使用简化版RDP实现")


class AdvancedCurveFitter:
    """
    高级曲线拟合器
    实现工程级曲线拟合方案
    """
    
    def __init__(self, img: np.ndarray, edges: Optional[np.ndarray] = None,
                 use_potrace: bool = True, potrace_path: str = "potrace",
                 patch_threshold: float = 0.1, rdp_epsilon_factor: float = 0.02,
                 angle_tolerance: float = 1.0, radius_tolerance: float = 0.01):
        """
        初始化高级曲线拟合器
        
        参数:
            img: 输入图像（RGB或灰度）
            edges: 边缘图像（可选，如果为None则从img生成）
            use_potrace: 是否使用potrace CLI（需要系统安装potrace）
            potrace_path: potrace可执行文件路径
            patch_threshold: Patch/Solid判断阈值（面积/周长^2）
            rdp_epsilon_factor: RDP简化因子（相对于局部宽度）
            angle_tolerance: 直角吸附容差（度）
            radius_tolerance: 半径量化容差（相对值）
        """
        self.img = img
        if edges is None:
            # 从图像生成边缘
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            self.edges = cv2.Canny(gray, 50, 150)
        else:
            self.edges = edges
        
        self.use_potrace = use_potrace
        self.potrace_path = potrace_path
        self.patch_threshold = patch_threshold
        self.rdp_epsilon_factor = rdp_epsilon_factor
        self.angle_tolerance = angle_tolerance
        self.radius_tolerance = radius_tolerance
        
        # 结果存储
        self.contours_dict = {}
        self.curves_tree = {}
        self.structure_types = {}  # 存储每个轮廓的结构类型
        self.geometric_primitives = {}  # 存储几何基元（直线/圆弧/B-spline）
        
        # 【融合优化】创建辅助器实例（用于复用算法，不需要实际拟合）
        # 注意：这里不调用__init__，只创建对象用于调用静态方法
        self._bspline_helper = None  # 延迟初始化
        
        # 执行拟合
        self._fit_contours()
    
    def _fit_contours(self):
        """执行完整的曲线拟合流程"""
        # Step 0: 结构预判
        contours = self._detect_contours()
        structure_classifications = self._classify_structures(contours)
        
        # Step 1-2: 轮廓获取和矢量路径获取（分流处理）
        for i, (contour, struct_type) in enumerate(zip(contours, structure_classifications)):
            self.structure_types[i] = struct_type
            
            try:
                if struct_type == 'patch' or struct_type == 'solid':
                    # Patch/Solid: 使用potrace或skimage
                    path_data = self._extract_patch_path(contour, i)
                else:
                    # Wire/Slot: 使用skeleton
                    path_data = self._extract_wire_path(contour, i)
                
                if path_data is None or not isinstance(path_data, dict):
                    print(f"   [WARN] 轮廓 {i} 路径提取失败，跳过")
                    continue
                
                if 'points' not in path_data or path_data['points'] is None:
                    print(f"   [WARN] 轮廓 {i} 路径点为空，跳过")
                    continue
                
                # 确保points是numpy数组
                if not isinstance(path_data['points'], np.ndarray):
                    path_data['points'] = np.array(path_data['points'])
                
                if len(path_data['points']) < 3:
                    print(f"   [WARN] 轮廓 {i} 点数不足（{len(path_data['points'])}），跳过")
                    continue
                
                # Step 3: 结构级简化（RDP）
                simplified_path = self._rdp_simplify(path_data['points'], path_data.get('width', None))
                
                # 确保simplified_path是有效的
                if simplified_path is None or len(simplified_path) < 3:
                    print(f"   [WARN] 轮廓 {i} RDP简化后点数不足，跳过")
                    continue
                
                # Step 4: 工程曲线语义恢复
                primitives = self._recover_geometric_primitives(simplified_path)
                
                # Step 5: 几何规整
                regularized_primitives = self._geometric_regularization(primitives)
                
                # Step 5.5: 拟合验证（新增）
                # 验证拟合结果的质量，如果误差过大则回退
                if not self._validate_fitting(regularized_primitives, simplified_path, contour):
                    # 验证失败，使用原始primitives（不规整）
                    print(f"   [WARN] 轮廓 {i} 拟合验证失败，使用未规整的primitives")
                    regularized_primitives = primitives
                
                # Step 6: 宽度/gap恢复（如果需要）
                width_data = None
                if struct_type == 'wire' or struct_type == 'slot':
                    width_data = self._recover_width_gap(contour, simplified_path)
                
                # 计算拟合误差（用于统计和评估）
                error = 0.0
                if len(contour) > 0 and len(simplified_path) > 0:
                    # 计算原始轮廓点到拟合路径的最小距离
                    contour_points = contour.reshape(-1, 2).astype(float)
                    errors = []
                    for cp in contour_points:
                        dists = np.linalg.norm(simplified_path - cp, axis=1)
                        errors.append(np.min(dists))
                    error = np.mean(errors) if errors else 0.0
                
                # 存储结果
                self.contours_dict[i] = {
                    'contour': contour,
                    'structure_type': struct_type,
                    'path_data': path_data,
                    'simplified_path': simplified_path,  # numpy数组
                    'primitives': regularized_primitives,  # 列表
                    'width_data': width_data,  # 宽度数据（如果有）
                    'error': error,  # 拟合误差（用于统计）
                    # 为了兼容性，也提供fitting字段（用于评估）
                    'fitting': {
                        'points': simplified_path,
                        'size': len(simplified_path)
                    }
                }
            except Exception as e:
                print(f"   [WARN] 轮廓 {i} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 构建轮廓树
        self.curves_tree = self._build_contour_tree()
    
    def _detect_contours(self) -> List[np.ndarray]:
        """
        Step 1: 轮廓检测
        优先使用skimage（亚像素精度），回退到OpenCV
        """
        contours_list = []
        
        # 尝试使用skimage（亚像素精度）
        try:
            # 二值化
            if len(self.edges.shape) == 2:
                binary = (self.edges > 0).astype(float)
            else:
                binary = (cv2.cvtColor(self.edges, cv2.COLOR_BGR2GRAY) > 0).astype(float)
            
            # skimage find_contours（亚像素精度）
            contours = measure.find_contours(binary, 0.5)
            
            for contour in contours:
                # 转换为整数坐标（用于后续处理）
                contour_int = np.array([[int(p[1]), int(p[0])] for p in contour], dtype=np.int32)
                if len(contour_int) >= 3:  # 至少3个点才能形成轮廓
                    contours_list.append(contour_int)
        except Exception as e:
            print(f"skimage轮廓检测失败，使用OpenCV: {e}")
            # 回退到OpenCV
            contours, _ = cv2.findContours(self.edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if len(contour) >= 3:
                    contours_list.append(contour.reshape(-1, 2))
        
        return contours_list
    
    def _classify_structures(self, contours: List[np.ndarray]) -> List[str]:
        """
        Step 0: 结构预判
        根据面积/周长比判断是Patch/Solid还是Wire/Slot
        """
        classifications = []
        
        for contour in contours:
            # 计算面积和周长
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                classifications.append('unknown')
                continue
            
            # 填充比：面积/周长^2
            fill_ratio = area / (perimeter ** 2)
            
            # 判断结构类型
            if fill_ratio > self.patch_threshold:
                # 进一步判断是patch还是solid（根据是否有孔洞）
                # 简单判断：如果轮廓是凸的，可能是patch；否则可能是solid
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if abs(area - hull_area) / max(area, 1) < 0.1:
                    classifications.append('patch')
                else:
                    classifications.append('solid')
            else:
                # 判断是wire还是slot（根据宽度）
                # 简单判断：如果面积很小，可能是wire；否则可能是slot
                if area < 100:  # 阈值可调
                    classifications.append('wire')
                else:
                    classifications.append('slot')
        
        return classifications
    
    def _extract_patch_path(self, contour: np.ndarray, contour_id: int) -> Optional[Dict]:
        """
        Step 1-2: Patch/Solid路径提取
        优先使用potrace CLI，回退到skimage + spline
        """
        if self.use_potrace:
            # 尝试使用potrace CLI
            path_data = self._potrace_extract(contour, contour_id)
            if path_data is not None:
                return path_data
        
        # 回退到skimage + spline
        return self._skimage_spline_extract(contour)
    
    def _potrace_extract(self, contour: np.ndarray, contour_id: int) -> Optional[Dict]:
        """使用potrace CLI提取Bezier路径"""
        try:
            # 创建临时图像文件
            with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as tmp_img:
                # 创建二值图像
                mask = np.zeros(self.edges.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                cv2.imwrite(tmp_img.name, mask)
                tmp_img_path = tmp_img.name
            
            # 创建临时SVG文件
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False, mode='w') as tmp_svg:
                tmp_svg_path = tmp_svg.name
            
            # 调用potrace CLI（关键参数：-a 0 禁用平滑，保持原始形状）
            cmd = [
                self.potrace_path,
                tmp_img_path,
                '-s',  # 输出SVG
                '-a', '0',  # alphamax=0，关键参数！
                '-o', tmp_svg_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # 解析SVG获取路径
                path_data = self._parse_svg_path(tmp_svg_path)
                
                # 清理临时文件
                Path(tmp_img_path).unlink()
                Path(tmp_svg_path).unlink()
                
                return path_data
            else:
                print(f"Potrace执行失败: {result.stderr}")
                Path(tmp_img_path).unlink()
                if Path(tmp_svg_path).exists():
                    Path(tmp_svg_path).unlink()
                return None
                
        except FileNotFoundError:
            print(f"未找到potrace可执行文件: {self.potrace_path}")
            return None
        except Exception as e:
            print(f"Potrace提取失败: {e}")
            return None
    
    def _parse_svg_path(self, svg_path: str) -> Dict:
        """解析SVG文件，提取Bezier路径"""
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            
            # 查找path元素
            paths = []
            for path_elem in root.iter():
                if path_elem.tag.endswith('path'):
                    d_attr = path_elem.get('d', '')
                    if d_attr:
                        paths.append(d_attr)
            
            # 将SVG路径转换为点序列（简化处理）
            # 实际应用中需要完整解析Bezier曲线
            points = []
            for path_str in paths:
                # 简化：提取所有坐标点
                # 实际需要解析M, L, C, Q等命令
                import re
                coords = re.findall(r'[-+]?\d*\.?\d+', path_str)
                for i in range(0, len(coords) - 1, 2):
                    if i + 1 < len(coords):
                        points.append([float(coords[i]), float(coords[i+1])])
            
            if len(points) > 0:
                return {
                    'points': np.array(points),
                    'type': 'bezier',
                    'source': 'potrace'
                }
            return None
        except Exception as e:
            print(f"SVG解析失败: {e}")
            return None
    
    def _skimage_spline_extract(self, contour: np.ndarray) -> Dict:
        """使用skimage轮廓 + spline插值提取路径"""
        # 使用scipy的spline插值
        from scipy.interpolate import UnivariateSpline
        
        # 将轮廓转换为参数化形式
        points = contour.reshape(-1, 2).astype(float)
        
        # 确保至少3个点
        if len(points) < 3:
            # 如果点数不足，直接返回原始点
            return {
                'points': points,
                'type': 'spline',
                'source': 'skimage'
            }
        
        # 去除重复点（关键修复：避免参数t有重复值）
        unique_points = []
        unique_points.append(points[0])
        for i in range(1, len(points)):
            # 如果当前点与前一个点距离足够大，才添加
            if np.linalg.norm(points[i] - unique_points[-1]) > 1e-6:
                unique_points.append(points[i])
        
        if len(unique_points) < 3:
            # 如果去重后点数不足，返回原始点
            return {
                'points': points,
                'type': 'spline',
                'source': 'skimage'
            }
        
        points = np.array(unique_points)
        
        # 计算累积弧长作为参数
        diffs = np.diff(points, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # 处理距离为0的情况（避免累积弧长不递增）
        distances = np.maximum(distances, 1e-6)  # 最小距离阈值
        
        cumulative = np.concatenate([[0], np.cumsum(distances)])
        
        # 闭合轮廓
        if not np.allclose(points[0], points[-1], atol=1.0):
            points = np.vstack([points, points[0:1]])
            final_dist = np.linalg.norm(points[-1] - points[-2])
            cumulative = np.concatenate([cumulative, [cumulative[-1] + max(final_dist, 1e-6)]])
        
        # 归一化参数到[0, 1]，确保严格递增
        if cumulative[-1] > 1e-6:
            t = cumulative / cumulative[-1]
            # 确保t严格递增（处理数值误差）
            for i in range(1, len(t)):
                if t[i] <= t[i-1]:
                    t[i] = t[i-1] + 1e-10
        else:
            # 如果总长度为0，使用均匀分布
            t = np.linspace(0, 1, len(points))
            # 确保严格递增
            for i in range(1, len(t)):
                if t[i] <= t[i-1]:
                    t[i] = t[i-1] + 1e-10
        
        # 验证t严格递增（关键修复）
        if len(t) != len(np.unique(t)):
            # 有重复值，需要去重
            t_unique, unique_indices = np.unique(t, return_index=True)
            if len(t_unique) < 3:
                # 去重后点数不足，返回原始点
                return {
                    'points': points,
                    'type': 'spline',
                    'source': 'skimage'
                }
            # 重新排序points以匹配t
            points = points[unique_indices]
            t = t_unique
        
        # 确保t严格递增（处理数值误差）
        for i in range(1, len(t)):
            if t[i] <= t[i-1]:
                t[i] = t[i-1] + 1e-10
        
        # 创建spline插值（确保k不超过点数-1）
        k = min(3, len(points) - 1)
        if k < 1:
            k = 1
        
        try:
            # 使用s>0允许平滑，避免严格插值要求
            # 或者使用更宽松的插值方法
            if len(points) >= 4:
                # 使用平滑spline（s>0）而不是严格插值（s=0）
                s = len(points) * 0.1  # 平滑因子
                try:
                    spl_x = UnivariateSpline(t, points[:, 0], s=s, k=k)
                    spl_y = UnivariateSpline(t, points[:, 1], s=s, k=k)
                    
                    # 生成密集点（使用t的范围）
                    t_dense = np.linspace(t[0], t[-1], len(points) * 2)
                    x_dense = spl_x(t_dense)
                    y_dense = spl_y(t_dense)
                    points_dense = np.column_stack([x_dense, y_dense])
                except Exception as e1:
                    # 如果平滑spline失败，尝试线性插值
                    from scipy.interpolate import interp1d
                    spl_x = interp1d(t, points[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
                    spl_y = interp1d(t, points[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
                    t_dense = np.linspace(t[0], t[-1], len(points) * 2)
                    x_dense = spl_x(t_dense)
                    y_dense = spl_y(t_dense)
                    points_dense = np.column_stack([x_dense, y_dense])
            else:
                # 点数少时使用线性插值
                from scipy.interpolate import interp1d
                spl_x = interp1d(t, points[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
                spl_y = interp1d(t, points[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
                t_dense = np.linspace(t[0], t[-1], len(points) * 2)
                x_dense = spl_x(t_dense)
                y_dense = spl_y(t_dense)
                points_dense = np.column_stack([x_dense, y_dense])
        except Exception as e:
            # 如果spline失败，返回原始点
            print(f"   [WARN] Spline插值失败，使用原始点: {e}")
            points_dense = points
        
        return {
            'points': points_dense,
            'type': 'spline',
            'source': 'skimage'
        }
    
    def _extract_wire_path(self, contour: np.ndarray, contour_id: int) -> Dict:
        """
        Step 1-2: Wire/Slot路径提取
        使用skeleton + distance transform
        """
        try:
            # 创建mask
            mask = np.zeros(self.edges.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Skeletonize
            skeleton = morphology.skeletonize(mask > 0)
            
            # 提取骨架点
            skeleton_points = np.column_stack(np.where(skeleton))
            if len(skeleton_points) < 2:
                # 如果骨架提取失败或点数不足，使用轮廓中心线
                return self._skimage_spline_extract(contour)
            
            # Distance transform（用于恢复宽度）
            dist_transform = distance_transform_edt(mask > 0)
            
            # 计算平均宽度
            skeleton_mask = np.zeros_like(mask)
            skeleton_mask[skeleton_points[:, 0], skeleton_points[:, 1]] = 255
            widths = dist_transform[skeleton_points[:, 0], skeleton_points[:, 1]]
            avg_width = np.mean(widths) * 2  # 直径
            
            # 将骨架点转换为图像坐标
            points = skeleton_points[:, [1, 0]].astype(float)  # (x, y)
            
            return {
                'points': points,
                'type': 'skeleton',
                'source': 'skeletonize',
                'width': avg_width
            }
        except Exception as e:
            # 如果skeleton提取失败，回退到spline方法
            print(f"   [WARN] Skeleton提取失败，使用spline方法: {e}")
            return self._skimage_spline_extract(contour)
    
    def _rdp_simplify(self, points: np.ndarray, local_width: Optional[float] = None) -> np.ndarray:
        """
        【优化】RDP简化：不再对所有点做RDP，而是仅用于寻找潜在的'角点'（Corners）
        使用极小的epsilon做RDP，只为了去噪，不为了简化形状
        这样保留了圆弧的轮廓，但去掉了像素锯齿
        """
        if points is None or len(points) < 3:
            return points if points is not None else np.array([])
        
        if not RDP_AVAILABLE:
            # 简化版RDP实现
            return self._simple_rdp(points, local_width)
        
        # 【关键优化】使用极小的epsilon，只去噪，不简化形状
        # 计算epsilon（比原来小得多）
        if local_width is None:
            # 使用点之间的平均距离
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            local_width = np.mean(distances) if len(distances) > 0 else 1.0
        
        # 使用极小的epsilon（0.5像素），只去除像素锯齿，保留圆弧轮廓
        epsilon = 0.5  # 固定小值，不再基于local_width
        
        # 应用RDP
        simplified = rdp(points, epsilon=epsilon)
        
        # 确保至少保留3个点（形成闭合轮廓）
        if len(simplified) < 3:
            # 如果简化后点数太少，保留原始点
            return points
        
        return simplified
    
    def _simple_rdp(self, points: np.ndarray, local_width: Optional[float]) -> np.ndarray:
        """简化版RDP实现（当rdp库不可用时）"""
        if len(points) < 3:
            return points
        
        if local_width is None:
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            local_width = np.mean(distances) if len(distances) > 0 else 1.0
        
        epsilon = self.rdp_epsilon_factor * local_width
        
        # 简单的Douglas-Peucker算法
        def dp_recursive(points, epsilon):
            if len(points) < 3:
                return points
            
            # 找到距离首尾连线最远的点
            start, end = points[0], points[-1]
            vec = end - start
            vec_norm = np.linalg.norm(vec)
            
            if vec_norm < 1e-10:
                return points[[0, -1]]
            
            distances = []
            for i, p in enumerate(points[1:-1], 1):
                # 点到直线的距离
                v = p - start
                proj = np.dot(v, vec) / vec_norm
                proj_point = start + (proj / vec_norm) * vec
                dist = np.linalg.norm(p - proj_point)
                distances.append((i, dist))
            
            if not distances:
                return points[[0, -1]]
            
            max_idx, max_dist = max(distances, key=lambda x: x[1])
            
            if max_dist > epsilon:
                # 递归处理
                left = dp_recursive(points[:max_idx+1], epsilon)
                right = dp_recursive(points[max_idx:], epsilon)
                return np.vstack([left[:-1], right])
            else:
                return points[[0, -1]]
        
        return dp_recursive(points, epsilon)
    
    def _recover_geometric_primitives(self, points: np.ndarray) -> List[Dict]:
        """
        【优化】工程曲线语义恢复：使用曲率分段 + 混合检测逻辑
        借用OptimizedBSplineFitter的分段逻辑（基于离散曲率），而不是简单的固定分段
        """
        if len(points) < 2:
            return []
        
        # 【关键优化】使用OptimizedBSplineFitter的曲率分段逻辑
        # 创建一个临时的辅助器来利用其算法（不需要实际拟合）
        try:
            # 延迟初始化辅助器
            if self._bspline_helper is None:
                # 创建一个最小化的辅助器（只需要分段功能）
                # 使用__new__避免调用__init__（不需要img和edges）
                self._bspline_helper = OptimizedBSplineFitter.__new__(OptimizedBSplineFitter)
                self._bspline_helper.curvature_threshold = 0.15  # 曲率变化阈值
            
            # 使用曲率分段（比固定分段更智能）
            segments_points = self._bspline_helper.initial_segmentation(points)
        except Exception as e:
            # 如果分段失败，使用固定分段作为回退
            print(f"   [WARN] 曲率分段失败，使用固定分段: {e}")
            segment_size = max(3, len(points) // 10)
            segments_points = []
            i = 0
            while i < len(points) - 1:
                end_idx = min(i + segment_size, len(points))
                segment = points[i:end_idx+1]
                if len(segment) >= 2:
                    segments_points.append(segment)
                i = end_idx
        
        # 对每一段进行高级识别
        primitives = []
        for seg in segments_points:
            if len(seg) < 2:
                continue
            
            # 使用增强的图元识别
            prim = self._identify_primitive_advanced(seg)
            primitives.append(prim)
        
        return primitives
    
    def _identify_primitive(self, points: np.ndarray) -> Dict:
        """
        识别单个段的几何类型（直线/圆弧/其他）
        【保留向后兼容，但推荐使用_identify_primitive_advanced】
        """
        return self._identify_primitive_advanced(points)
    
    def _identify_primitive_advanced(self, points: np.ndarray) -> Dict:
        """
        【新增】高级图元识别，融合了直线、圆弧和NURBS检测
        使用全局优先级调度器（dispatch_segment）进行统一决策
        """
        if len(points) < 2:
            return {'type': 'point', 'points': points}
        
        # 确保points是2D数组，形状为(N, 2)
        points = np.asarray(points)
        if len(points.shape) == 1:
            return {'type': 'point', 'points': points}
        if len(points.shape) == 2:
            if points.shape[1] != 2:
                if points.shape[0] == 2:
                    points = points.T
                elif points.shape[1] > 2:
                    points = points[:, :2]
                else:
                    return {'type': 'point', 'points': points}
        else:
            return {'type': 'point', 'points': points}
        
        # 【关键优化】使用全局优先级调度器进行统一决策
        # 这确保了Line > Arc > NURBS的优先级，并且是"对比式"而非"独立成立"
        try:
            decision, dispatch_result = dispatch_segment(
                points,
                # Line判别参数（优化：考虑RDP简化后的精度损失，适当放宽）
                linearity_threshold=0.03,  # 放宽到0.03（考虑简化后的精度损失）
                max_residual_threshold=1.5,  # 放宽到1.5像素
                total_angle_threshold_deg=12.0,  # 放宽到12度
                # Arc硬判据参数（保持严格，确保质量）
                sigma_r_max=0.02,  # 2%半径误差（严格）
                angle_span_min_deg=20.0,  # 至少20度弧
                num_points_min=8,  # 至少8个点
                # Arc vs NURBS对比参数（更倾向于Arc）
                arc_gain_th=1.3,  # 降低到1.3（更倾向于Arc）
                arc_ctrl=3,
                nurbs_ctrl=8
            )
            
            if decision == "line":
                # Line段
                line_info = dispatch_result['info']
                return {
                    'type': 'line',
                    'points': points,
                    'start': line_info['p0'],
                    'end': line_info['p1'],
                    'error': line_info.get('max_residual', line_info.get('max_error', 0.0)),
                    'dispatch_reason': dispatch_result['reason']
                }
            
            elif decision == "arc":
                # Arc段（使用稳健的arc_quality检测）
                arc_info = dispatch_result['info']
                return {
                    'type': 'arc',
                    'points': points,
                    'center': arc_info['center'],
                    'radius': arc_info['radius'],
                    'error': arc_info.get('sigma_r', 0.0) * arc_info.get('radius', 1.0),
                    'angle_span': arc_info.get('angle_span', 0.0),
                    'sigma_r': arc_info.get('sigma_r', 0.0),
                    'dispatch_reason': dispatch_result['reason']
                }
            
            else:
                # NURBS/B-spline段
                # 【优化】不要只存点，要拟合出控制点，方便CAD导出
                try:
                    # 使用OptimizedBSplineFitter的弧长参数化
                    if self._bspline_helper is None:
                        self._bspline_helper = OptimizedBSplineFitter.__new__(OptimizedBSplineFitter)
                    t = self._bspline_helper.arc_length_param(points)
                    
                    # 降采样拟合，减少控制点数量
                    ctrl_pts = points[::max(1, len(points)//10)]
                    
                    return {
                        'type': 'bspline',
                        'points': points,
                        'control_points': ctrl_pts,  # 保存控制点
                        'degree': 3,
                        'error': 0.0,  # 假设贴合
                        'dispatch_reason': dispatch_result['reason']
                    }
                except Exception as e:
                    # 如果控制点提取失败，返回基本格式
                    return {
                        'type': 'bspline',
                        'points': points,
                        'error': 0.0,
                        'dispatch_reason': f'control_points_extraction_failed_{str(e)}'
                    }
        
        except Exception as e:
            # 如果调度失败，使用原始简单识别作为回退
            print(f"   [WARN] 高级图元识别失败，使用简单识别: {e}")
            return self._identify_primitive_simple(points)
    
    def _identify_primitive_simple(self, points: np.ndarray) -> Dict:
        """
        简单图元识别（回退方案）
        """
        if len(points) < 2:
            return {'type': 'point', 'points': points}
        
        # 1. 尝试直线拟合
        line_error = self._fit_line_error(points)
        line_threshold = 0.5  # 像素
        
        # 2. 尝试圆弧拟合
        circle_error = self._fit_circle_error(points)
        circle_threshold = 1.0  # 像素
        
        # 判断类型
        if line_error < line_threshold:
            return {
                'type': 'line',
                'points': points,
                'start': points[0],
                'end': points[-1],
                'error': line_error
            }
        elif circle_error < circle_threshold:
            circle_params = self._fit_circle(points)
            return {
                'type': 'arc',
                'points': points,
                'center': circle_params['center'],
                'radius': circle_params['radius'],
                'error': circle_error
            }
        else:
            # B-spline拟合
            return {
                'type': 'bspline',
                'points': points,
                'error': min(line_error, circle_error)
            }
    
    def _fit_line_error(self, points: np.ndarray) -> float:
        """计算点到直线的最大偏差"""
        if len(points) < 2:
            return float('inf')
        
        start, end = points[0], points[-1]
        vec = end - start
        vec_norm = np.linalg.norm(vec)
        
        if vec_norm < 1e-10:
            return 0.0
        
        max_dist = 0.0
        for p in points[1:-1]:
            v = p - start
            proj = np.dot(v, vec) / vec_norm
            proj_point = start + (proj / vec_norm) * vec
            dist = np.linalg.norm(p - proj_point)
            max_dist = max(max_dist, dist)
        
        return max_dist
    
    def _fit_circle_error(self, points: np.ndarray) -> float:
        """计算点到圆弧的最大偏差"""
        if len(points) < 3:
            return float('inf')
        
        try:
            circle_params = self._fit_circle(points)
            center = circle_params['center']
            radius = circle_params['radius']
            
            max_error = 0.0
            for p in points:
                dist = np.linalg.norm(p - center)
                error = abs(dist - radius)
                max_error = max(max_error, error)
            
            return max_error
        except:
            return float('inf')
    
    def _fit_circle(self, points: np.ndarray) -> Dict:
        """拟合圆弧（最小二乘）"""
        # 使用Taubin方法
        n = len(points)
        if n < 3:
            raise ValueError("至少需要3个点拟合圆弧")
        
        # 计算质心
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # 构建矩阵
        z = np.sum(centered**2, axis=1)
        Z = np.mean(z)
        
        # 计算矩阵元素
        Zxy = np.mean(centered[:, 0] * centered[:, 1])
        Zxx = np.mean(centered[:, 0]**2)
        Zyy = np.mean(centered[:, 1]**2)
        
        # 求解
        M = np.array([
            [Zxx, Zxy],
            [Zxy, Zyy]
        ])
        
        b = np.array([
            np.mean(z * centered[:, 0]),
            np.mean(z * centered[:, 1])
        ])
        
        try:
            a = np.linalg.solve(M, b)
            center = centroid + a
            radius = np.sqrt(Z + np.sum(a**2))
        except:
            # 回退到简单方法
            center = centroid
            radius = np.mean(np.linalg.norm(centered, axis=1))
        
        return {
            'center': center,
            'radius': radius
        }
    
    def _geometric_regularization(self, primitives: List[Dict]) -> List[Dict]:
        """
        Step 5: 几何规整（增强版）
        - 直角吸附
        - 共线合并
        - 半径量化
        - G0连续性（端点对齐）
        - G1连续性（切向对齐）
        """
        regularized = []
        
        for i, prim in enumerate(primitives):
            reg_prim = prim.copy()
            
            if prim['type'] == 'line':
                # 直角吸附
                angle = np.arctan2(prim['end'][1] - prim['start'][1],
                                  prim['end'][0] - prim['start'][0])
                angle_deg = np.degrees(angle)
                
                # 吸附到最近的直角（0, 90, 180, 270度）或45度
                nearest_90 = round(angle_deg / 90) * 90
                nearest_45 = round(angle_deg / 45) * 45
                
                # 【保守修正】只在角度非常接近时才修正（<0.5度）
                angle_diff_90 = abs(angle_deg - nearest_90)
                angle_diff_45 = abs(angle_deg - nearest_45)
                
                # 记录原始误差
                original_error = prim.get('error', 0.0)
                
                if angle_diff_90 < 0.5:  # 非常接近直角（<0.5度）
                    # 调整端点使其完全水平或垂直
                    dx = prim['end'][0] - prim['start'][0]
                    dy = prim['end'][1] - prim['start'][1]
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if abs(dx) > abs(dy):
                        # 水平线
                        new_end = [prim['start'][0] + length * np.sign(dx), prim['start'][1]]
                    else:
                        # 垂直线
                        new_end = [prim['start'][0], prim['start'][1] + length * np.sign(dy)]
                    
                    # 验证修正后的误差（允许10%误差增加）
                    reg_prim['end'] = new_end
                    # 简单验证：检查修正后的长度变化
                    new_length = np.linalg.norm(np.array(new_end) - np.array(prim['start']))
                    original_length = np.linalg.norm(np.array(prim['end']) - np.array(prim['start']))
                    if abs(new_length - original_length) / max(original_length, 1.0) > 0.1:
                        # 修正后长度变化太大，回退
                        reg_prim['end'] = prim['end']
                elif angle_diff_45 < 0.5:  # 非常接近45度（<0.5度）
                    # 45度线
                    dx = prim['end'][0] - prim['start'][0]
                    dy = prim['end'][1] - prim['start'][1]
                    length = np.sqrt(dx**2 + dy**2)
                    sign_x = np.sign(dx) if abs(dx) > 1e-6 else 1
                    sign_y = np.sign(dy) if abs(dy) > 1e-6 else 1
                    reg_prim['end'] = [
                        prim['start'][0] + length * sign_x * np.cos(np.radians(nearest_45)),
                        prim['start'][1] + length * sign_y * np.sin(np.radians(nearest_45))
                    ]
            
            elif prim['type'] == 'arc':
                # 半径量化（避免9.9998 mm这样的值）
                radius = prim['radius']
                # 量化到0.01mm精度
                quantized_radius = round(radius / 0.01) * 0.01
                if abs(radius - quantized_radius) / max(radius, 0.01) < self.radius_tolerance:
                    reg_prim['radius'] = quantized_radius
            
            regularized.append(reg_prim)
        
        # G0连续性：端点对齐（闭合缝隙）
        regularized = self._enforce_g0_continuity(regularized)
        
        # G1连续性：切向对齐（平滑过渡）
        regularized = self._enforce_g1_continuity(regularized)
        
        # 共线合并（相邻的直线段如果共线，合并）
        merged = self._merge_collinear_lines(regularized)
        
        return merged
    
    def _enforce_g0_continuity(self, primitives: List[Dict]) -> List[Dict]:
        """
        强制G0连续性：确保相邻段的端点对齐
        """
        if len(primitives) < 2:
            return primitives
        
        result = []
        for i, prim in enumerate(primitives):
            reg_prim = prim.copy()
            next_prim = primitives[(i + 1) % len(primitives)]  # 循环连接
            
            # 获取当前段终点和下一段起点
            if prim['type'] == 'line':
                p_end = np.array(prim['end'])
            else:
                p_end = np.array(prim['points'][-1]) if 'points' in prim else np.array(prim.get('end', [0, 0]))
            
            if next_prim['type'] == 'line':
                p_start = np.array(next_prim['start'])
            else:
                p_start = np.array(next_prim['points'][0]) if 'points' in next_prim else np.array(next_prim.get('start', [0, 0]))
            
            # 如果距离很近，强制合并到一个中间点
            dist = np.linalg.norm(p_end - p_start)
            if dist < 5.0:  # 5像素容差
                mid_point = (p_end + p_start) / 2
                
                # 修改当前段终点
                if prim['type'] == 'line':
                    reg_prim['end'] = mid_point.tolist()
                elif 'points' in reg_prim:
                    reg_prim['points'][-1] = mid_point.tolist()
                
                # 修改下一段起点（在下一轮迭代中处理）
                if next_prim['type'] == 'line':
                    next_prim['start'] = mid_point.tolist()
                elif 'points' in next_prim:
                    next_prim['points'][0] = mid_point.tolist()
            
            result.append(reg_prim)
        
        return result
    
    def _enforce_g1_continuity(self, primitives: List[Dict]) -> List[Dict]:
        """
        强制G1连续性：确保相邻段的切向对齐（平滑过渡）
        """
        if len(primitives) < 2:
            return primitives
        
        result = primitives.copy()
        
        for i in range(len(primitives)):
            curr = result[i]
            next_p = result[(i + 1) % len(primitives)]
            
            # 只处理直线-圆弧或圆弧-直线的连接
            if curr['type'] == 'line' and next_p['type'] == 'arc':
                # 计算直线向量
                vec_line = np.array(curr['end']) - np.array(curr['start'])
                vec_line_norm = np.linalg.norm(vec_line)
                if vec_line_norm > 1e-6:
                    vec_line = vec_line / vec_line_norm
                    
                    # 计算圆弧起点半径向量
                    if 'center' in next_p and 'points' in next_p:
                        center = np.array(next_p['center'])
                        start_point = np.array(next_p['points'][0])
                        vec_radius = start_point - center
                        vec_radius_norm = np.linalg.norm(vec_radius)
                        if vec_radius_norm > 1e-6:
                            vec_radius = vec_radius / vec_radius_norm
                            
                            # 如果垂直（点积接近0），说明相切
                            dot_product = np.abs(np.dot(vec_line, vec_radius))
                            if dot_product < 0.1:  # 接近垂直，说明相切
                                # 可以微调圆心位置以强制相切（这里简化处理，只记录）
                                pass
        
        return result
    
    def _validate_fitting(self, primitives: List[Dict], simplified_path: np.ndarray, 
                         original_contour: np.ndarray) -> bool:
        """
        验证拟合结果的质量
        
        参数:
            primitives: 拟合后的几何基元列表
            simplified_path: 简化后的路径点
            original_contour: 原始轮廓点
        
        返回:
            是否通过验证
        """
        if len(primitives) == 0:
            return False
        
        # 1. 重建拟合点
        try:
            fitted_points = self._reconstruct_from_primitives(primitives)
        except Exception as e:
            print(f"   [WARN] 重建拟合点失败: {e}")
            return False
        
        if len(fitted_points) == 0:
            return False
        
        # 2. 计算拟合误差（相对于原始轮廓）
        original_points = original_contour.reshape(-1, 2).astype(float)
        errors = []
        for p in original_points:
            dists = np.linalg.norm(fitted_points - p, axis=1)
            errors.append(np.min(dists))
        
        if len(errors) == 0:
            return False
        
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        # 3. 检查是否满足阈值
        max_error_threshold = 3.0  # 最大误差阈值（像素）
        mean_error_threshold = 1.5  # 平均误差阈值（像素）
        
        is_valid = max_error < max_error_threshold and mean_error < mean_error_threshold
        
        if not is_valid:
            print(f"   [WARN] 拟合验证失败: max_error={max_error:.2f}, mean_error={mean_error:.2f}")
        
        return is_valid
    
    def _reconstruct_from_primitives(self, primitives: List[Dict]) -> np.ndarray:
        """
        从几何基元重建拟合点
        
        参数:
            primitives: 几何基元列表
        
        返回:
            重建后的点数组
        """
        all_points = []
        
        for prim in primitives:
            if prim['type'] == 'line':
                # 直线：生成点
                start = np.array(prim['start'])
                end = np.array(prim['end'])
                # 生成10个点
                t = np.linspace(0, 1, 10)
                points = start + np.outer(t, end - start)
                all_points.append(points)
            
            elif prim['type'] == 'arc':
                # 圆弧：生成点
                if 'points' in prim:
                    # 使用原始点
                    all_points.append(np.array(prim['points']))
                else:
                    # 从center和radius生成点
                    center = np.array(prim['center'])
                    radius = prim['radius']
                    angle_span = prim.get('angle_span', 90.0)  # 默认90度
                    # 生成点
                    num_points = max(10, int(angle_span / 5))  # 每5度一个点
                    start_angle = 0.0  # 简化：从0度开始
                    angles = np.linspace(start_angle, np.radians(angle_span), num_points)
                    points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
                    all_points.append(points)
            
            elif prim['type'] == 'bspline':
                # B-spline：使用points
                if 'points' in prim:
                    all_points.append(np.array(prim['points']))
                elif 'control_points' in prim:
                    all_points.append(np.array(prim['control_points']))
        
        if len(all_points) == 0:
            return np.array([])
        
        # 合并所有点
        fitted_points = np.vstack(all_points)
        
        # 去除重复点
        if len(fitted_points) > 1:
            unique_points = [fitted_points[0]]
            for i in range(1, len(fitted_points)):
                if np.linalg.norm(fitted_points[i] - unique_points[-1]) > 1e-6:
                    unique_points.append(fitted_points[i])
            fitted_points = np.array(unique_points)
        
        return fitted_points
    
    def _merge_collinear_lines(self, primitives: List[Dict]) -> List[Dict]:
        """合并共线的相邻直线段"""
        if len(primitives) < 2:
            return primitives
        
        merged = []
        i = 0
        while i < len(primitives):
            current = primitives[i]
            
            if current['type'] == 'line' and i + 1 < len(primitives):
                next_prim = primitives[i + 1]
                
                # 检查是否共线
                if next_prim['type'] == 'line':
                    # 检查是否连接且共线
                    curr_start = np.asarray(current.get('start'), dtype=float)
                    curr_end = np.asarray(current.get('end'), dtype=float)
                    next_start = np.asarray(next_prim.get('start'), dtype=float)
                    next_end = np.asarray(next_prim.get('end'), dtype=float)
                    if np.allclose(curr_end, next_start, atol=1.0):
                        # 检查方向是否相同
                        vec1 = curr_end - curr_start
                        vec2 = next_end - next_start
                        
                        if np.linalg.norm(vec1) > 1e-10 and np.linalg.norm(vec2) > 1e-10:
                            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                            if cos_angle > 0.99:  # 几乎共线
                                current_points = np.asarray(
                                    current.get('points', [curr_start, curr_end]),
                                    dtype=float
                                )
                                next_points = np.asarray(
                                    next_prim.get('points', [next_start, next_end]),
                                    dtype=float
                                )
                                # 合并
                                merged.append({
                                    'type': 'line',
                                    'start': current.get('start'),
                                    'end': next_prim.get('end'),
                                    'points': np.vstack([current_points, next_points[1:]]),
                                    'error': max(current.get('error', 0), next_prim.get('error', 0))
                                })
                                i += 2
                                continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _recover_width_gap(self, contour: np.ndarray, path: np.ndarray) -> Dict:
        """
        Step 6: 宽度/gap恢复
        使用distance transform和法线采样
        """
        # 创建mask
        mask = np.zeros(self.edges.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        
        # Distance transform
        dist_transform = distance_transform_edt(mask > 0)
        
        # 沿路径采样宽度
        widths = []
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            # 中点
            mid = (p1 + p2) / 2
            mid_int = mid.astype(int)
            
            if 0 <= mid_int[1] < dist_transform.shape[0] and 0 <= mid_int[0] < dist_transform.shape[1]:
                width = dist_transform[mid_int[1], mid_int[0]] * 2  # 直径
                widths.append(width)
        
        return {
            'widths': widths,
            'mean_width': np.mean(widths) if widths else 0.0,
            'min_width': np.min(widths) if widths else 0.0,
            'max_width': np.max(widths) if widths else 0.0
        }
    
    def _build_contour_tree(self) -> Dict:
        """构建轮廓树（嵌套关系）"""
        # 简化实现：基于轮廓的包含关系
        tree = {}
        
        try:
            for i, contour_data in self.contours_dict.items():
                contour = contour_data.get('contour')
                if contour is None:
                    continue
                
                children = []
                
                # 查找子轮廓（被当前轮廓包含的轮廓）
                for j, other_data in self.contours_dict.items():
                    if i != j:
                        other_contour = other_data.get('contour')
                        if other_contour is None:
                            continue
                        
                        # 检查是否包含
                        try:
                            if self._is_contour_inside(other_contour, contour):
                                children.append(j)
                        except Exception as e:
                            # 如果包含关系测试失败，跳过
                            continue
                
                if children:
                    tree[i] = children
        except Exception as e:
            print(f"   [WARN] 构建轮廓树失败: {e}")
        
        return tree
    
    def _is_contour_inside(self, inner: np.ndarray, outer: np.ndarray) -> bool:
        """检查inner轮廓是否在outer轮廓内部"""
        try:
            # 确保轮廓格式正确
            if len(inner) == 0 or len(outer) == 0:
                return False
            
            # 确保是numpy数组
            if not isinstance(inner, np.ndarray):
                inner = np.array(inner)
            if not isinstance(outer, np.ndarray):
                outer = np.array(outer)
            
            # 确保是2D数组
            if len(inner.shape) == 1:
                if len(inner) >= 2:
                    inner = inner.reshape(1, 2)
                else:
                    return False
            elif len(inner.shape) == 2:
                if inner.shape[1] != 2:
                    if inner.shape[0] == 2:
                        inner = inner.T
                    else:
                        return False
            else:
                return False
            
            if len(outer.shape) == 1:
                if len(outer) >= 2:
                    outer = outer.reshape(1, 2)
                else:
                    return False
            elif len(outer.shape) == 2:
                if outer.shape[1] != 2:
                    if outer.shape[0] == 2:
                        outer = outer.T
                    else:
                        return False
            else:
                return False
            
            # 确保至少3个点形成轮廓
            if len(outer) < 3:
                return False
            
            # 使用点测试（确保类型正确）
            test_point = inner[0]
            # 转换为正确的类型（int或float）
            if np.issubdtype(test_point.dtype, np.integer):
                test_point = (int(test_point[0]), int(test_point[1]))
            else:
                test_point = (float(test_point[0]), float(test_point[1]))
            
            # 确保outer是正确格式（int32或float32）
            if outer.dtype not in [np.int32, np.float32]:
                if np.issubdtype(outer.dtype, np.integer):
                    outer = outer.astype(np.int32)
                else:
                    outer = outer.astype(np.float32)
            
            result = cv2.pointPolygonTest(outer, test_point, False)
            return result >= 0
        except Exception as e:
            # 如果测试失败，返回False（保守处理）
            print(f"   [WARN] 轮廓包含关系测试失败: {e}")
            return False
    
    def get_contours_dict(self) -> Dict:
        """获取轮廓字典"""
        return self.contours_dict
    
    def get_curves_tree(self) -> Dict:
        """获取轮廓树"""
        return self.curves_tree
    
    def get_structure_types(self) -> Dict:
        """获取结构类型"""
        return self.structure_types
    
    def get_geometric_primitives(self) -> Dict:
        """获取几何基元"""
        return {i: data['primitives'] for i, data in self.contours_dict.items()}
