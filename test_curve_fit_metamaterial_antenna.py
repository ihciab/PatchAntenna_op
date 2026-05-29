"""
超材料/天线图像处理测试框架
专门测试图形处理、曲线拟合与提取的效果
针对实际应用场景：超材料单元、天线结构等

原代码位置：old_code/Rebuild/test_metamaterial_antenna.py
迁移位置：tests/test_metamaterial_antenna.py

迁移说明：
- 本文件从 old_code/Rebuild/test_metamaterial_antenna.py 迁移而来
- 更新了导入路径以适配新的代码结构
- 功能保持不变，用于测试图像处理、曲线拟合与提取的效果
- 简化了目录结构，所有测试文件直接放在tests/目录下
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cdist

# Windows控制台UTF-8支持
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# 添加项目路径（从tests到项目根目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 更新导入路径以适配新的代码结构
# 原路径：Rebuild.ImageInit -> 新路径：core.image.initializer
from core.image.initializer import ImageInitializer as ImageInit
# 原路径：Rebuild.BSplineContour -> 新路径：core.geometry.bspline_fitter
from core.geometry.bspline_fitter import BSplineContour
# 原路径：Rebuild.NURBSpline -> 新路径：core.geometry.nurbs_fitter
from core.geometry.nurbs_fitter import NURBSpineContour
# 原路径：Rebuild.ColorDetector -> 新路径：core.material.color_detector
from core.material.color_detector import ColorDetector
# 原路径：Rebuild.Curves2Component -> 新路径：core.material.material_mapper
from core.material.material_mapper import Curves2Components
# 原路径：Rebuild.validation_dataset_generator -> 新路径：tools.dataset_generator.validation_generator
from tools.dataset_generator.validation_generator import ValidationDatasetGenerator, generate_validation_dataset
# 新增：评估器
from core.geometry.curve_evaluator import CurveEvaluator
# 新增：优化的B样条拟合器
from core.geometry.optimized_bspline_fitter import OptimizedBSplineFitter
# 新增：优化的NURBS拟合器
from core.geometry.optimized_nurbs_fitter import OptimizedNURBSFitter
# 新增：几何感知评分器
from core.geometry.geometry_aware_metrics import GeometryAwareMetrics, compute_geometry_aware_score
# 新增：分段提取器（为传统方法添加分段信息）
from core.geometry.segment_extractor import SegmentExtractor, extract_segments_from_fitted_contour


class MetamaterialAntennaTestGenerator:
    """超材料/天线测试图像生成器"""
    
    def __init__(self, width=1000, height=1000, background_color=(255, 255, 255)):
        """
        初始化生成器
        
        参数:
            width: 图像宽度
            height: 图像高度
            background_color: 背景颜色 (B, G, R)
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.gt_segments = []
        self.reset()
    
    def reset(self):
        """重置画布"""
        self.img = np.ones((self.height, self.width, 3), dtype=np.uint8) * np.array(
            self.background_color, dtype=np.uint8
        )
        self.gt_segments = []
    
    def get_gt_segments(self) -> List[Dict]:
        return list(self.gt_segments)
    
    def _add_gt_line(self, start: Tuple[float, float], end: Tuple[float, float], num_points: int = 50):
        p0 = np.array(start, dtype=float)
        p1 = np.array(end, dtype=float)
        t = np.linspace(0.0, 1.0, max(2, int(num_points)))
        points = (p0[None, :] + (p1 - p0)[None, :] * t[:, None]).astype(float)
        self.gt_segments.append({
            'type': 'line',
            'p0': p0.tolist(),
            'p1': p1.tolist(),
            'start': p0.tolist(),
            'end': p1.tolist(),
            'points': points.tolist()
        })
    
    def _add_gt_arc(self,
                    center: Tuple[float, float],
                    radius: float,
                    start_angle_deg: float,
                    end_angle_deg: float,
                    num_points: int = 120):
        c = np.array(center, dtype=float)
        r = float(radius)
        a0 = float(start_angle_deg)
        a1 = float(end_angle_deg)
        
        if a1 < a0:
            a1 += 360.0
        
        angles = np.deg2rad(np.linspace(a0, a1, max(3, int(num_points))))
        points = np.column_stack([c[0] + r * np.cos(angles), c[1] + r * np.sin(angles)]).astype(float)
        self.gt_segments.append({
            'type': 'arc',
            'center': c.tolist(),
            'radius': r,
            'start_angle': a0,
            'end_angle': a1,
            'points': points.tolist()
        })
    
    def _add_gt_circle(self, center: Tuple[float, float], radius: float, num_points: int = 240):
        self._add_gt_arc(center, radius, 0.0, 360.0, num_points=num_points)
    
    def _add_gt_spline(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        if len(pts) < 3:
            return
        self.gt_segments.append({
            'type': 'spline',
            'points': pts.tolist()
        })
    
    def _polyline_to_gt_lines(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        if len(pts) < 2:
            return
        for i in range(len(pts) - 1):
            self._add_gt_line(tuple(pts[i]), tuple(pts[i + 1]), num_points=10)
    
    def save(self, path: str):
        """保存图像"""
        cv2.imwrite(path, self.img)
    
    # ========== 基础几何结构 ==========
    
    def draw_rectangular_patch(self, center: Tuple[int, int], width: int, height: int, 
                               color: Tuple[int, int, int] = (0, 0, 0), filled: bool = True):
        """绘制矩形贴片（常见天线结构）"""
        x, y = center
        pt1 = (x - width // 2, y - height // 2)
        pt2 = (x + width // 2, y + height // 2)
        if filled:
            cv2.rectangle(self.img, pt1, pt2, color, -1)
        else:
            cv2.rectangle(self.img, pt1, pt2, color, 2)
        
        x1, y1 = pt1
        x2, y2 = pt2
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]], dtype=float)
        self._polyline_to_gt_lines(corners)
        return pt1, pt2
    
    def draw_circular_patch(self, center: Tuple[int, int], radius: int,
                           color: Tuple[int, int, int] = (0, 0, 0), filled: bool = True):
        """绘制圆形贴片"""
        if filled:
            cv2.circle(self.img, center, radius, color, -1)
        else:
            cv2.circle(self.img, center, radius, color, 2)
        self._add_gt_circle(center, radius)
        return center, radius
    
    def draw_ring(self, center: Tuple[int, int], outer_radius: int, inner_radius: int,
                  color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制环形结构（嵌套圆形）"""
        # 绘制外圆（填充）
        cv2.circle(self.img, center, outer_radius, color, -1)
        # 绘制内圆（用背景色填充，形成孔）
        cv2.circle(self.img, center, inner_radius, self.background_color, -1)
        self._add_gt_circle(center, outer_radius)
        self._add_gt_circle(center, inner_radius)
        return center, outer_radius, inner_radius
    
    def draw_slot(self, center: Tuple[int, int], length: int, width: int, angle: float = 0,
                  color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制槽形结构（直线槽）"""
        # 计算旋转后的端点
        angle_rad = np.radians(angle)
        dx = length / 2 * np.cos(angle_rad)
        dy = length / 2 * np.sin(angle_rad)
        pt1 = (int(center[0] - dx), int(center[1] - dy))
        pt2 = (int(center[0] + dx), int(center[1] + dy))
        cv2.rectangle(self.img, pt1, pt2, color, -1)
        x1, y1 = pt1
        x2, y2 = pt2
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]], dtype=float)
        self._polyline_to_gt_lines(corners)
        return pt1, pt2
    
    def draw_arc_slot(self, center: Tuple[int, int], radius: int, start_angle: float, 
                      end_angle: float, width: int, color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制弧形槽 - 针对金属结构，使用填充"""
        # 使用填充绘制弧形槽
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.ellipse(mask, center, (radius + width // 2, radius + width // 2), 
                   0, start_angle, end_angle, 255, width)
        self.img[mask > 0] = color
        self._add_gt_arc(center, radius + width // 2, start_angle, end_angle)
        return center, radius, start_angle, end_angle
    
    def draw_curve_line_combination(self, start: Tuple[int, int], end: Tuple[int, int],
                                   curve_type: str = 'arc', curve_params: Dict = None,
                                   color: Tuple[int, int, int] = (0, 0, 0), 
                                   line_width: int = 15):
        """绘制曲线与直线的组合结构"""
        if curve_params is None:
            curve_params = {}
        
        if curve_type == 'arc':
            # 弧线+直线组合
            center = curve_params.get('center', 
                                     ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2))
            radius = curve_params.get('radius', 
                                     int(np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2) / 2))
            start_angle = curve_params.get('start_angle', 
                                          np.degrees(np.arctan2(start[1]-center[1], start[0]-center[0])))
            end_angle = curve_params.get('end_angle',
                                        np.degrees(np.arctan2(end[1]-center[1], end[0]-center[0])))
            
            # 绘制弧线
            self.draw_arc_slot(center, radius, start_angle, end_angle, line_width, color)
            # 绘制连接直线
            cv2.line(self.img, start, end, color, line_width)
            self._add_gt_line(start, end)
        
        elif curve_type == 'bezier':
            # 贝塞尔曲线+直线组合
            control = curve_params.get('control', 
                                 ((start[0] + end[0]) // 2, start[1] - 100))
            
            # 生成贝塞尔曲线点
            points = []
            for t in np.linspace(0, 1, 100):
                x = (1-t)**2 * start[0] + 2*(1-t)*t * control[0] + t**2 * end[0]
                y = (1-t)**2 * start[1] + 2*(1-t)*t * control[1] + t**2 * end[1]
                points.append([int(x), int(y)])
            
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(self.img, [pts], False, color, line_width)
            self._add_gt_spline(np.array(points, dtype=float))
            # 绘制连接直线
            cv2.line(self.img, start, end, color, line_width)
            self._add_gt_line(start, end)
        
        return start, end
    
    def draw_complex_shape(self, shape_type: str, params: Dict,
                          color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制复杂形状：曲线、弧形与直线的组合"""
        center = params.get('center', (500, 500))
        size = params.get('size', 200)
        
        if shape_type == 'curved_patch':
            # 曲线贴片：矩形+弧形边缘
            # 主体矩形
            self.draw_rectangular_patch(center, size, size, color, filled=True)
            # 弧形边缘装饰
            for i in range(4):
                angle = i * 90
                arc_center = (
                    int(center[0] + size//2 * np.cos(np.radians(angle))),
                    int(center[1] + size//2 * np.sin(np.radians(angle)))
                )
                self.draw_arc_slot(arc_center, size//4, angle-45, angle+45, 20, color)
        
        elif shape_type == 'hybrid_antenna':
            # 混合天线：直线+曲线+弧线组合
            # 中心矩形
            self.draw_rectangular_patch(center, size//2, size//2, color, filled=True)
            # 四个方向的曲线臂
            for angle in [0, 90, 180, 270]:
                start = (
                    int(center[0] + size//4 * np.cos(np.radians(angle))),
                    int(center[1] + size//4 * np.sin(np.radians(angle)))
                )
                end = (
                    int(center[0] + size * np.cos(np.radians(angle))),
                    int(center[1] + size * np.sin(np.radians(angle)))
                )
                # 直线部分
                cv2.line(self.img, start, end, color, 15)
                self._add_gt_line(start, end)
                # 末端弧形
                arc_center = end
                self.draw_arc_slot(arc_center, size//6, angle-30, angle+30, 20, color)
        
        elif shape_type == 'waveguide':
            # 波导结构：直线+曲线过渡
            start = (center[0] - size, center[1])
            mid = center
            end = (center[0] + size, center[1])
            
            # 直线段
            cv2.line(self.img, start, (mid[0] - size//4, mid[1]), color, 20)
            cv2.line(self.img, (mid[0] + size//4, mid[1]), end, color, 20)
            self._add_gt_line(start, (mid[0] - size//4, mid[1]))
            self._add_gt_line((mid[0] + size//4, mid[1]), end)
            
            # 中间曲线过渡
            curve_center = mid
            self.draw_arc_slot(curve_center, size//4, 180, 0, 20, color)
        
        return center
    
    def draw_meander_line(self, start: Tuple[int, int], width: int, height: int, 
                         segments: int, line_width: int = 15,
                         color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制蜿蜒线（折线结构）- 针对金属结构，使用粗线条"""
        x, y = start
        points = [start]
        direction = 1
        
        for i in range(segments):
            if i % 2 == 0:
                x += width * direction
            else:
                y += height
                direction *= -1
            points.append((x, y))
        
        pts = np.array(points, dtype=np.int32)
        # 使用粗线条，适合金属结构
        cv2.polylines(self.img, [pts], False, color, line_width)
        self._polyline_to_gt_lines(pts)
        return pts
    
    def draw_spiral(self, center: Tuple[int, int], start_radius: int, end_radius: int,
                   turns: int, line_width: int = 15, color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制螺旋结构 - 针对金属结构，使用粗线条"""
        points = []
        step = max(1, turns * 360 // 1000)  # 减少点数，提高性能
        for i in range(0, turns * 360, step):
            angle = np.radians(i)
            radius = start_radius + (end_radius - start_radius) * (i / (turns * 360))
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            points.append([x, y])
        
        pts = np.array(points, dtype=np.int32)
        # 使用粗线条，适合金属结构
        cv2.polylines(self.img, [pts], False, color, line_width)
        self._add_gt_spline(pts)
        return pts
    
    def draw_fractal_tree(self, start: Tuple[int, int], length: int, angle: float,
                         depth: int, color: Tuple[int, int, int] = (0, 0, 0), 
                         line_width: int = 12):
        """绘制分形树结构 - 针对金属结构，使用粗线条"""
        def draw_branch(pt, len, ang, dep, w):
            if dep == 0:
                return []
            
            end_x = int(pt[0] + len * np.cos(ang))
            end_y = int(pt[1] + len * np.sin(ang))
            end_pt = (end_x, end_y)
            
            # 线条宽度随深度递减
            cv2.line(self.img, pt, end_pt, color, max(3, int(w)))
            self._add_gt_line(pt, end_pt)
            
            branches = [end_pt]
            # 左分支
            branches.extend(draw_branch(end_pt, len * 0.7, ang - np.pi/6, dep - 1, w * 0.8))
            # 右分支
            branches.extend(draw_branch(end_pt, len * 0.7, ang + np.pi/6, dep - 1, w * 0.8))
            
            return branches
        
        draw_branch(start, length, angle, depth, line_width)
        return start
    
    def draw_koch_snowflake(self, center: Tuple[int, int], radius: int, iterations: int = 2,
                            color: Tuple[int, int, int] = (0, 0, 0), thickness: int = 12):
        """绘制科赫雪花分形 - 针对金属结构，使用粗线条"""
        def koch_curve(p1, p2, depth):
            if depth == 0:
                return [p1, p2]
            
            dx = (p2[0] - p1[0]) / 3
            dy = (p2[1] - p1[1]) / 3
            
            p1_new = (p1[0] + dx, p1[1] + dy)
            p3_new = (p2[0] - dx, p2[1] - dy)
            
            angle = np.arctan2(dy, dx)
            length = np.sqrt(dx*dx + dy*dy)
            p2_new = (
                p1_new[0] + length * np.cos(angle + np.pi/3),
                p1_new[1] + length * np.sin(angle + np.pi/3)
            )
            
            points = []
            points.extend(koch_curve(p1, p1_new, depth - 1)[:-1])
            points.extend(koch_curve(p1_new, p2_new, depth - 1)[:-1])
            points.extend(koch_curve(p2_new, p3_new, depth - 1)[:-1])
            points.extend(koch_curve(p3_new, p2, depth - 1))
            
            return points
        
        cx, cy = center
        points = []
        for i in range(3):
            angle = i * 2 * np.pi / 3 - np.pi / 2
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            points.append((x, y))
        
        all_points = []
        for i in range(3):
            p1 = points[i]
            p2 = points[(i + 1) % 3]
            curve_points = koch_curve(p1, p2, iterations)
            all_points.extend(curve_points[:-1])
        
        pts = np.array(all_points, dtype=np.int32)
        # 使用粗线条，适合金属结构
        cv2.polylines(self.img, [pts], True, color, thickness)
        self._polyline_to_gt_lines(np.vstack([pts, pts[:1]]))
        return pts
    
    def draw_sierpinski_triangle(self, p1: Tuple[int, int], p2: Tuple[int, int], 
                                p3: Tuple[int, int], depth: int = 3,
                                color: Tuple[int, int, int] = (0, 0, 0), thickness: int = 8):
        """绘制谢尔宾斯基三角形分形 - 针对金属结构，使用粗线条或填充"""
        def draw_triangle(pt1, pt2, pt3, d, w):
            if d == 0:
                pts = np.array([pt1, pt2, pt3], dtype=np.int32)
                # 使用填充，更适合金属结构
                cv2.fillPoly(self.img, [pts], color)
                return
            
            mid1 = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            mid2 = ((pt2[0] + pt3[0]) // 2, (pt2[1] + pt3[1]) // 2)
            mid3 = ((pt3[0] + pt1[0]) // 2, (pt3[1] + pt1[1]) // 2)
            
            draw_triangle(pt1, mid1, mid3, d - 1, w)
            draw_triangle(mid1, pt2, mid2, d - 1, w)
            draw_triangle(mid3, mid2, pt3, d - 1, w)
        
        draw_triangle(p1, p2, p3, depth, thickness)
        self._polyline_to_gt_lines(np.array([p1, p2, p3, p1], dtype=float))
        return p1, p2, p3
    
    # ========== 复杂组合结构 ==========
    
    def draw_fss_unit(self, center: Tuple[int, int], size: int, pattern_type: str = 'cross',
                     color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制FSS（频率选择表面）单元"""
        x, y = center
        
        if pattern_type == 'cross':
            # 十字形
            cv2.rectangle(self.img, (x - size//2, y - size//20), 
                         (x + size//2, y + size//20), color, -1)
            cv2.rectangle(self.img, (x - size//20, y - size//2), 
                         (x + size//20, y + size//2), color, -1)
            
            a = size // 2
            b = size // 20
            outline = np.array([
                [x - a, y - b],
                [x - b, y - b],
                [x - b, y - a],
                [x + b, y - a],
                [x + b, y - b],
                [x + a, y - b],
                [x + a, y + b],
                [x + b, y + b],
                [x + b, y + a],
                [x - b, y + a],
                [x - b, y + b],
                [x - a, y + b],
                [x - a, y - b]
            ], dtype=float)
            self._polyline_to_gt_lines(outline)
        
        elif pattern_type == 'square_loop':
            # 方形环
            outer = size // 2
            inner = size // 3
            self.draw_ring(center, outer, inner, color)
        
        elif pattern_type == 'circular_ring':
            # 圆形环
            outer = size // 2
            inner = size // 3
            self.draw_ring(center, outer, inner, color)
        
        elif pattern_type == 'spiral':
            # 螺旋
            self.draw_spiral(center, size // 6, size // 2, 2, 2, color)
        
        return center, pattern_type
    
    def draw_antenna_structure(self, center: Tuple[int, int], width: int, height: int,
                              antenna_type: str = 'patch', 
                              color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制天线结构"""
        x, y = center
        
        if antenna_type == 'patch':
            # 矩形贴片天线
            self.draw_rectangular_patch(center, width, height, color, filled=True)
        
        elif antenna_type == 'slot':
            # 槽天线
            self.draw_slot(center, width, height, 0, color)
        
        elif antenna_type == 'meander':
            # 蜿蜒天线
            self.draw_meander_line((x - width//2, y), width//5, height//10, 10, 2, color)
        
        elif antenna_type == 'fractal':
            # 分形天线
            self.draw_fractal_tree(center, min(width, height) // 2, -np.pi/2, 4, color, 2)
        
        return center, antenna_type
    
    def draw_layered_structure(self, layers: List[Dict], base_color: Tuple[int, int, int] = (0, 0, 0)):
        """绘制多层叠加结构（不同材料用不同颜色）"""
        """
        layers格式:
        [
            {
                'type': 'rect' | 'circle' | 'ring' | 'complex',
                'center': (x, y),
                'params': {...},
                'color': (B, G, R)  # 不同颜色代表不同材料
            },
            ...
        ]
        """
        for layer in layers:
            layer_type = layer['type']
            center = layer['center']
            color = layer.get('color', base_color)
            params = layer.get('params', {})
            
            if layer_type == 'rect':
                self.draw_rectangular_patch(center, params.get('width', 100), 
                                          params.get('height', 100), color, 
                                          params.get('filled', True))
            elif layer_type == 'circle':
                self.draw_circular_patch(center, params.get('radius', 50), color,
                                       params.get('filled', True))
            elif layer_type == 'ring':
                self.draw_ring(center, params.get('outer_radius', 80),
                             params.get('inner_radius', 50), color)
            elif layer_type == 'complex':
                # 复杂形状
                shape_type = params.get('shape_type', 'curved_patch')
                self.draw_complex_shape(shape_type, params, color)
        
        return layers
    
    def draw_multi_color_hierarchy(self, hierarchy: List[Dict]):
        """绘制多颜色层级结构（支持5+种颜色）"""
        """
        hierarchy格式:
        [
            {
                'level': 1,  # 层级
                'shapes': [
                    {'type': 'rect', 'center': (x, y), 'params': {...}},
                    ...
                ],
                'color': (B, G, R)  # 该层级的颜色
            },
            ...
        ]
        """
        # 按层级从外到内绘制
        sorted_hierarchy = sorted(hierarchy, key=lambda x: x.get('level', 0), reverse=True)
        
        for level_data in sorted_hierarchy:
            color = level_data.get('color', (0, 0, 0))
            shapes = level_data.get('shapes', [])
            
            for shape in shapes:
                shape_type = shape['type']
                center = shape['center']
                params = shape.get('params', {})
                
                if shape_type == 'rect':
                    self.draw_rectangular_patch(center, params.get('width', 100),
                                              params.get('height', 100), color, True)
                elif shape_type == 'circle':
                    self.draw_circular_patch(center, params.get('radius', 50), color, True)
                elif shape_type == 'ring':
                    self.draw_ring(center, params.get('outer_radius', 80),
                                 params.get('inner_radius', 50), color)
                elif shape_type == 'complex':
                    self.draw_complex_shape(params.get('shape_type', 'curved_patch'),
                                          params, color)
        
        return hierarchy
    
    # ========== 测试用例生成 ==========
    
    def generate_test_case(self, case_type: str, **kwargs) -> np.ndarray:
        """
        生成测试用例图像
        
        参数:
            case_type: 测试用例类型
                - 'simple_patch': 简单贴片
                - 'nested_rings': 嵌套环形
                - 'slot_antenna': 槽天线
                - 'meander_antenna': 蜿蜒天线
                - 'fss_unit': FSS单元
                - 'layered_materials': 多层材料
                - 'complex_combination': 复杂组合
        """
        self.reset()
        
        if case_type == 'simple_patch':
            # 简单矩形贴片
            center = kwargs.get('center', (500, 500))
            width = kwargs.get('width', 200)
            height = kwargs.get('height', 150)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_rectangular_patch(center, width, height, color, filled=True)
        
        elif case_type == 'nested_rings':
            # 嵌套环形结构
            center = kwargs.get('center', (500, 500))
            outer_radius = kwargs.get('outer_radius', 200)
            inner_radius = kwargs.get('inner_radius', 100)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_ring(center, outer_radius, inner_radius, color)
        
        elif case_type == 'slot_antenna':
            # 槽天线
            center = kwargs.get('center', (500, 500))
            length = kwargs.get('length', 300)
            width = kwargs.get('width', 20)
            angle = kwargs.get('angle', 45)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_slot(center, length, width, angle, color)
        
        elif case_type == 'meander_antenna':
            # 蜿蜒天线
            start = kwargs.get('start', (200, 400))
            width = kwargs.get('width', 50)
            height = kwargs.get('height', 30)
            segments = kwargs.get('segments', 8)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_meander_line(start, width, height, segments, 15, color)
        
        elif case_type == 'fss_unit':
            # FSS单元
            center = kwargs.get('center', (500, 500))
            size = kwargs.get('size', 300)
            pattern = kwargs.get('pattern', 'cross')
            color = kwargs.get('color', (0, 0, 0))
            self.draw_fss_unit(center, size, pattern, color)
        
        elif case_type == 'layered_materials':
            # 多层材料结构（不同颜色）
            layers = kwargs.get('layers', [
                {'type': 'rect', 'center': (500, 500), 'params': {'width': 300, 'height': 300}, 
                 'color': (0, 0, 255)},  # 红色外层
                {'type': 'circle', 'center': (500, 500), 'params': {'radius': 100}, 
                 'color': (255, 0, 0)},  # 蓝色内层
            ])
            self.draw_layered_structure(layers)
        
        elif case_type == 'complex_combination':
            # 复杂组合：贴片 + 槽 + 环形
            center = kwargs.get('center', (500, 500))
            # 外层矩形
            self.draw_rectangular_patch(center, 400, 400, (0, 0, 0), filled=True)
            # 内层环形
            self.draw_ring(center, 150, 100, (0, 0, 0))
            # 槽
            self.draw_slot(center, 200, 15, 45, (0, 0, 0))
        
        elif case_type == 'koch_snowflake':
            # 科赫雪花分形
            center = kwargs.get('center', (500, 500))
            radius = kwargs.get('radius', 200)
            iterations = kwargs.get('iterations', 2)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_koch_snowflake(center, radius, iterations, color, 12)
        
        elif case_type == 'sierpinski_triangle':
            # 谢尔宾斯基三角形分形
            size = kwargs.get('size', 400)
            center = kwargs.get('center', (500, 500))
            depth = kwargs.get('depth', 3)
            color = kwargs.get('color', (0, 0, 0))
            cx, cy = center
            p1 = (cx, cy - size // 2)
            p2 = (cx - size // 2, cy + size // 2)
            p3 = (cx + size // 2, cy + size // 2)
            self.draw_sierpinski_triangle(p1, p2, p3, depth, color, 8)
        
        elif case_type == 'fractal_tree':
            # 分形树
            center = kwargs.get('center', (500, 700))
            length = kwargs.get('length', 200)
            depth = kwargs.get('depth', 4)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_fractal_tree(center, length, -np.pi/2, depth, color, 12)
        
        elif case_type == 'multi_color_layers':
            # 多颜色多层结构
            layers = kwargs.get('layers', [
                {'type': 'rect', 'center': (500, 500), 'params': {'width': 400, 'height': 400}, 
                 'color': (0, 0, 255)},  # 红色外层
                {'type': 'circle', 'center': (500, 500), 'params': {'radius': 150}, 
                 'color': (255, 0, 0)},  # 蓝色中层
                {'type': 'ring', 'center': (500, 500), 'params': {'outer_radius': 100, 'inner_radius': 50}, 
                 'color': (0, 255, 0)},  # 绿色内层
            ])
            self.draw_layered_structure(layers)
        
        elif case_type == 'complex_fss':
            # 复杂FSS结构：多个单元组合
            center = kwargs.get('center', (500, 500))
            # 中心十字
            self.draw_fss_unit(center, 300, 'cross', (0, 0, 0))
            # 四个角的方形环
            offset = 200
            for dx, dy in [(-offset, -offset), (offset, -offset), (-offset, offset), (offset, offset)]:
                self.draw_fss_unit((center[0] + dx, center[1] + dy), 150, 'square_loop', (0, 0, 0))
        
        elif case_type == 'spiral_antenna':
            # 螺旋天线
            center = kwargs.get('center', (500, 500))
            start_radius = kwargs.get('start_radius', 50)
            end_radius = kwargs.get('end_radius', 200)
            turns = kwargs.get('turns', 3)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_spiral(center, start_radius, end_radius, turns, 15, color)
        
        elif case_type == 'curved_patch':
            # 曲线贴片：矩形+弧形边缘
            center = kwargs.get('center', (500, 500))
            size = kwargs.get('size', 300)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_complex_shape('curved_patch', {'center': center, 'size': size}, color)
        
        elif case_type == 'hybrid_antenna':
            # 混合天线：直线+曲线+弧线组合
            center = kwargs.get('center', (500, 500))
            size = kwargs.get('size', 400)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_complex_shape('hybrid_antenna', {'center': center, 'size': size}, color)
        
        elif case_type == 'waveguide':
            # 波导结构
            center = kwargs.get('center', (500, 500))
            size = kwargs.get('size', 500)
            color = kwargs.get('color', (0, 0, 0))
            self.draw_complex_shape('waveguide', {'center': center, 'size': size}, color)
        
        elif case_type == 'curve_line_combo':
            # 曲线+直线组合
            start = kwargs.get('start', (200, 500))
            end = kwargs.get('end', (800, 500))
            curve_type = kwargs.get('curve_type', 'arc')
            curve_params = kwargs.get('curve_params', {})
            color = kwargs.get('color', (0, 0, 0))
            self.draw_curve_line_combination(start, end, curve_type, curve_params, color, 15)
        
        elif case_type == 'multi_color_hierarchy':
            # 多颜色层级结构
            hierarchy = kwargs.get('hierarchy', [
                {'level': 1, 'shapes': [
                    {'type': 'rect', 'center': (500, 500), 'params': {'width': 500, 'height': 500}}
                ], 'color': (0, 0, 255)},  # 红色外层
                {'level': 2, 'shapes': [
                    {'type': 'circle', 'center': (500, 500), 'params': {'radius': 200}}
                ], 'color': (255, 0, 0)},  # 蓝色中层
                {'level': 3, 'shapes': [
                    {'type': 'ring', 'center': (500, 500), 'params': {'outer_radius': 120, 'inner_radius': 80}}
                ], 'color': (0, 255, 0)},  # 绿色内层
            ])
            self.draw_multi_color_hierarchy(hierarchy)
        
        return self.img.copy()


class MetamaterialAntennaTester:
    """超材料/天线图像处理测试类"""
    
    def __init__(self, output_dir: str = "test_metamaterial_output", 
                 generator: Optional[MetamaterialAntennaTestGenerator] = None,
                 save_json_report: bool = False):
        """
        初始化测试器
        
        参数:
            output_dir: 输出目录
            generator: 测试图像生成器（如果为None则创建默认的）
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.save_json_report = bool(save_json_report)
        
        if generator is None:
            self.generator = MetamaterialAntennaTestGenerator()
        else:
            self.generator = generator
        
        self.test_results = []

    def _compute_centering_translation(self, bgr_img: np.ndarray) -> Tuple[int, int]:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blurred, 2500, 5000, apertureSize=5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0

        main_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(main_contour)
        (x_center, y_center), _, _ = rect
        main_center = (int(x_center), int(y_center))

        h, w = edges.shape[:2]
        img_center = (w // 2, h // 2)
        dx = img_center[0] - main_center[0]
        dy = img_center[1] - main_center[1]
        return int(dx), int(dy)

    def _translate_gt_segments(self, gt_segments: List[Dict], dx: int, dy: int) -> List[Dict]:
        if not gt_segments or (dx == 0 and dy == 0):
            return list(gt_segments) if gt_segments else []

        translated = []
        shift = np.array([dx, dy], dtype=float)

        for seg in gt_segments:
            seg2 = dict(seg)

            if 'points' in seg2 and seg2['points'] is not None:
                pts = np.asarray(seg2['points'], dtype=float)
                if pts.size:
                    seg2['points'] = (pts + shift[None, :]).tolist()

            for k in ('p0', 'p1', 'start', 'end', 'center'):
                if k in seg2 and seg2[k] is not None and isinstance(seg2[k], (list, tuple)) and len(seg2[k]) == 2:
                    v = np.asarray(seg2[k], dtype=float) + shift
                    seg2[k] = v.tolist()

            translated.append(seg2)

        return translated

    def _extract_boundary_gt_segments(self, centered_bgr_img: np.ndarray) -> List[Dict]:
        if centered_bgr_img is None or centered_bgr_img.size == 0:
            return []

        if centered_bgr_img.ndim == 2:
            mask = (centered_bgr_img > 0).astype(np.uint8) * 255
        elif centered_bgr_img.ndim == 3 and centered_bgr_img.shape[2] == 3:
            mask = np.any(centered_bgr_img < 250, axis=2).astype(np.uint8) * 255
        else:
            return []

        if not np.any(mask):
            return []

        contours, _hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        gt_segments = []
        for c in contours:
            pts = c.reshape(-1, 2).astype(float)
            if len(pts) < 3:
                continue
            if len(pts) > 3000:
                step = int(np.ceil(len(pts) / 3000))
                pts = pts[::max(1, step)]
            try:
                gt_segments.extend(extract_segments_from_fitted_contour(pts))
            except Exception:
                continue

        return gt_segments
    
    def _convert_to_json_serializable(self, obj, _depth: int = 0, _visited: Optional[set] = None):
        try:
            if _visited is None:
                _visited = set()

            if _depth > 12:
                return str(obj)

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                oid = id(obj)
                if oid in _visited:
                    return str(obj)
                _visited.add(oid)
                return {str(key): self._convert_to_json_serializable(value, _depth=_depth + 1, _visited=_visited) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                oid = id(obj)
                if oid in _visited:
                    return str(obj)
                _visited.add(oid)
                return [self._convert_to_json_serializable(item, _depth=_depth + 1, _visited=_visited) for item in obj]
            elif isinstance(obj, set):
                oid = id(obj)
                if oid in _visited:
                    return str(obj)
                _visited.add(oid)
                return [self._convert_to_json_serializable(item, _depth=_depth + 1, _visited=_visited) for item in obj]
            elif hasattr(obj, '__dict__'):
                oid = id(obj)
                if oid in _visited:
                    return str(obj)
                _visited.add(oid)
                return self._convert_to_json_serializable(obj.__dict__, _depth=_depth + 1, _visited=_visited)
            else:
                return obj
        except Exception as e:
            # 如果转换失败，返回字符串表示
            return str(obj)

    def _summarize_segments_for_report(self,
                                      segments: List[Dict],
                                      max_segments: int = 200,
                                      max_points: int = 60) -> Dict:
        if not segments:
            return {
                'segment_count': 0,
                'line_count': 0,
                'arc_count': 0,
                'spline_count': 0,
                'segments': [],
                'truncated': False
            }

        def _as_points(x) -> np.ndarray:
            if x is None:
                return np.zeros((0, 2), dtype=float)
            pts = np.asarray(x, dtype=float).reshape(-1, 2)
            return pts

        def _downsample_points(pts: np.ndarray) -> List[List[float]]:
            if len(pts) <= max_points:
                return pts.astype(float).tolist()
            step = int(np.ceil(len(pts) / max_points))
            return pts[::max(1, step)].astype(float).tolist()

        def _polyline_length(pts: np.ndarray) -> float:
            if len(pts) < 2:
                return 0.0
            d = np.diff(pts, axis=0)
            return float(np.sum(np.linalg.norm(d, axis=1)))

        def _line_endpoints(seg: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            for a, b in (('start', 'end'), ('p0', 'p1')):
                if a in seg and b in seg:
                    p0 = np.asarray(seg[a], dtype=float).reshape(2,)
                    p1 = np.asarray(seg[b], dtype=float).reshape(2,)
                    if np.linalg.norm(p1 - p0) > 1e-8:
                        return p0, p1
            pts = _as_points(seg.get('points', None))
            if len(pts) >= 2 and np.linalg.norm(pts[-1] - pts[0]) > 1e-8:
                return pts[0], pts[-1]
            return None

        def _arc_angles(center: np.ndarray, pts: np.ndarray) -> Optional[Dict]:
            if len(pts) < 2:
                return None
            c = np.asarray(center, dtype=float).reshape(2,)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            ang_u = np.unwrap(ang)
            a0 = float(np.degrees(ang_u[0]))
            a1 = float(np.degrees(ang_u[-1]))
            span = float(abs(a1 - a0))
            return {'start_angle': a0, 'end_angle': a1, 'angle_span': span, 'ccw': bool(a1 >= a0)}

        summaries = []
        truncated = False

        line_count = 0
        arc_count = 0
        spline_count = 0

        for idx, seg in enumerate(segments):
            if idx >= max_segments:
                truncated = True
                break

            seg_type = str(seg.get('type', 'unknown'))
            pts = _as_points(seg.get('points', None))

            if seg_type == 'line':
                line_count += 1
                ep = _line_endpoints(seg)
                if ep is None:
                    continue
                p0, p1 = ep
                v = p1 - p0
                ang = float(np.degrees(np.arctan2(v[1], v[0])))
                summaries.append({
                    'type': 'line',
                    'start': p0.tolist(),
                    'end': p1.tolist(),
                    'angle_deg': ang,
                    'length': float(np.linalg.norm(v)),
                    'polyline_length': _polyline_length(pts),
                    'points_sampled': _downsample_points(pts) if len(pts) else []
                })
            elif seg_type == 'arc':
                arc_count += 1
                center = seg.get('center', None)
                radius = seg.get('radius', None)
                if center is None or radius is None:
                    continue
                c = np.asarray(center, dtype=float).reshape(2,)
                r = float(radius)
                angle_info = _arc_angles(c, pts) if len(pts) else None
                item = {
                    'type': 'arc',
                    'center': c.tolist(),
                    'radius': r,
                    'polyline_length': _polyline_length(pts),
                    'points_sampled': _downsample_points(pts) if len(pts) else []
                }
                if angle_info is not None:
                    item.update(angle_info)
                summaries.append(item)
            elif seg_type in ['spline', 'nurbs', 'bspline', 'curve']:
                spline_count += 1
                summaries.append({
                    'type': seg_type,
                    'point_count': int(len(pts)),
                    'polyline_length': _polyline_length(pts),
                    'points_sampled': _downsample_points(pts) if len(pts) else []
                })

        return {
            'segment_count': int(len(segments)),
            'line_count': int(line_count),
            'arc_count': int(arc_count),
            'spline_count': int(spline_count),
            'segments': summaries,
            'truncated': bool(truncated)
        }
    
    def test_image_processing(self, test_image: Optional[np.ndarray], test_name: str, 
                              image_path: Optional[str] = None,
                              gt_path: Optional[str] = None,
                              gt_segments: Optional[List[Dict]] = None) -> Dict:
        """
        测试图像处理流程
        
        参数:
            test_image: 测试图像数组（如果为None，则从image_path加载）
            test_name: 测试名称
            image_path: 图像路径（如果提供，优先使用此路径）
        """
        print(f"\n测试: {test_name}")
        print("=" * 60)
        
        # 确定图像路径
        if image_path and os.path.exists(image_path):
            test_img_path = image_path
            test_image = cv2.imread(image_path)
            if test_image is None:
                raise ValueError(f"无法读取图像: {image_path}")
        else:
            # 保存测试图像
            test_img_path = os.path.join(self.output_dir, f"{test_name}_input.png")
            if test_image is not None:
                cv2.imwrite(test_img_path, test_image)
            else:
                raise ValueError("必须提供test_image或有效的image_path")
        
        try:
            gt_segments_centered = None
            if gt_segments:
                dx, dy = self._compute_centering_translation(test_image)
                gt_segments_centered = self._translate_gt_segments(gt_segments, dx, dy)

            # 1. 图像初始化
            print("1. 图像初始化...")
            ii = ImageInit(test_img_path, show=False, save="")
            img = ii.centered_img()
            edges = ii.edges()
            print(f"   [OK] 图像尺寸: {img.shape}")
            print(f"   [OK] 边缘图尺寸: {edges.shape}")

            gt_boundary_segments = self._extract_boundary_gt_segments(img)
            gt_segments_for_scoring = gt_boundary_segments if len(gt_boundary_segments) > 0 else gt_segments_centered
            
            # 2. 轮廓检测（B样条）
            print("\n2. B样条轮廓拟合...")
            bc = BSplineContour(
                img=img,
                edges=edges,
                lr=0.1,
                delta=2,
                step=30,
                threshold=10.0,
                show=False,
                save=""
            )
            contours_dict_bs = bc.get_contours_dict()
            curves_tree_bs = bc.get_curves_tree()
            print(f"   [OK] 检测到 {len(contours_dict_bs)} 个轮廓")
            print(f"   [OK] 轮廓树节点数: {len(curves_tree_bs)}")
            
            # 3. NURBS轮廓拟合
            print("\n3. NURBS轮廓拟合...")
            nbs = NURBSpineContour(
                img=img,
                edges=edges,
                lr=0.1,
                delta=2,
                step=30,
                threshold=10.0,
                degree=3,
                show=False,
                save=""
            )
            contours_dict_nurbs = nbs.get_contours_dict()
            curves_tree_nurbs = nbs.get_curves_tree()
            print(f"   [OK] 检测到 {len(contours_dict_nurbs)} 个轮廓")
            
            # 3.2. 优化NURBS拟合（新增）
            print("\n3.2. 优化NURBS拟合（曲率驱动权重+分段）...")
            try:
                optimized_nurbs = OptimizedNURBSFitter(
                    img=img,
                    edges=edges,
                    lr=0.1,
                    delta=2,
                    step=30,
                    threshold=10.0,
                    degree=3,
                    line_threshold=2.0,
                    arc_threshold=2.0,
                    curvature_threshold=0.15,
                    curvature_weight_alpha=5.0,
                    use_segmentation=True,
                    show=False,
                    save=""
                )
                contours_dict_optimized_nurbs = optimized_nurbs.get_contours_dict()
                curves_tree_optimized_nurbs = optimized_nurbs.get_curves_tree()
                segments_info_optimized_nurbs = optimized_nurbs.get_segments_info()
                # 存储分段信息供评估使用
                self._current_segments_info_optimized_nurbs = segments_info_optimized_nurbs
                print(f"   [OK] 检测到 {len(contours_dict_optimized_nurbs)} 个轮廓")
            except Exception as e:
                print(f"   [WARN] 优化NURBS拟合失败: {e}")
                contours_dict_optimized_nurbs = {}
                curves_tree_optimized_nurbs = {}
                segments_info_optimized_nurbs = {}
            
            # 3.3. 优化B样条拟合（新增）
            print("\n3.3. 优化B样条拟合（几何语义感知）...")
            try:
                optimized_bs_fitter = OptimizedBSplineFitter(
                    img=img,
                    edges=edges,
                    line_threshold=2.0,
                    arc_threshold=2.0,
                    curvature_threshold=0.15,
                    spline_degree=3,
                    show=False,
                    save=""
                )
                contours_dict_optimized_bs = optimized_bs_fitter.get_contours_dict()
                curves_tree_optimized_bs = optimized_bs_fitter.get_curves_tree()
                print(f"   [OK] 检测到 {len(contours_dict_optimized_bs)} 个轮廓")
            except Exception as e:
                print(f"   [WARN] 优化B样条拟合失败: {e}")
                contours_dict_optimized_bs = {}
                curves_tree_optimized_bs = {}
            
            # 4. 颜色检测（如果有颜色）
            print("\n4. 颜色检测...")
            detector = ColorDetector()
            color_results = detector.detect_colors(image_path=test_img_path)
            print(f"   [OK] 检测到 {len(color_results)} 种颜色")
            
            # 5. 曲线转组件
            print("\n5. 曲线转组件...")
            c2c = Curves2Components(
                original_img=img,
                img_shape=img.shape,
                con_dict=contours_dict_bs,
                cur_tree=curves_tree_bs,
                color_ranges=None
            )
            solids_mask = c2c.solids_col()
            print(f"   [OK] 生成 {len(solids_mask)} 个solid映射")
            
            # 6. 统计信息
            bspline_stats = self._calculate_statistics(contours_dict_bs)
            nurbs_stats = self._calculate_statistics(contours_dict_nurbs)
            optimized_nurbs_stats = self._calculate_statistics(contours_dict_optimized_nurbs) if contours_dict_optimized_nurbs else {}
            optimized_bs_stats = self._calculate_statistics(contours_dict_optimized_bs) if contours_dict_optimized_bs else {}
            
            # 6.5. 精细评估对比（新增）
            print("\n6.5. 精细评估对比...")
            evaluation_comparison = self._compare_fitting_methods(
                img, edges,
                contours_dict_bs, contours_dict_nurbs, 
                contours_dict_optimized_nurbs, contours_dict_optimized_bs,
                gt_segments=gt_segments_for_scoring
            )
            print(f"   [OK] B样条综合得分: {evaluation_comparison.get('bspline', {}).get('overall', {}).get('overall', 0):.2f}")
            print(f"   [OK] NURBS综合得分: {evaluation_comparison.get('nurbs', {}).get('overall', {}).get('overall', 0):.2f}")
            if 'optimized_nurbs' in evaluation_comparison:
                print(f"   [OK] 优化NURBS综合得分: {evaluation_comparison['optimized_nurbs'].get('overall', {}).get('overall', 0):.2f}")
            if 'optimized_bs' in evaluation_comparison:
                print(f"   [OK] 优化B样条综合得分: {evaluation_comparison['optimized_bs'].get('overall', {}).get('overall', 0):.2f}")
            
            # 7. Ground Truth评估（如果提供）
            gt_evaluation = None
            if gt_path and os.path.exists(gt_path):
                print("\n7. Ground Truth评估...")
                try:
                    with open(gt_path, 'r', encoding='utf-8') as f:
                        gt = json.load(f)
                    gt_evaluation_bs = self._evaluate_against_gt(contours_dict_bs, gt)
                    gt_evaluation_nurbs = self._evaluate_against_gt(contours_dict_nurbs, gt)
                    gt_evaluation = {
                        'bspline': gt_evaluation_bs,
                        'nurbs': gt_evaluation_nurbs,
                        'ground_truth': gt
                    }
                    print(f"   [OK] 结构类型: {gt.get('structure_type', 'unknown')}")
                    print(f"   [OK] 轮廓数量匹配 (B样条): {gt_evaluation_bs['contour_count_match']}")
                    print(f"   [OK] 轮廓数量匹配 (NURBS): {gt_evaluation_nurbs['contour_count_match']}")
                except Exception as e:
                    print(f"   ⚠ GT评估失败: {e}")
            
            # 生成可视化对比图
            print("\n8. 生成可视化对比图...")
            vis_path = self._visualize_comparison(
                test_image, img, edges, contours_dict_bs, contours_dict_nurbs, test_name
            )
            vis_methods_path = self._visualize_method_overview(
                processed_img=img,
                edges=edges,
                contours_dict_by_method={
                    'bspline': contours_dict_bs,
                    'nurbs': contours_dict_nurbs,
                    'optimized_nurbs': contours_dict_optimized_nurbs,
                    'optimized_bs': contours_dict_optimized_bs,
                },
                gt_segments=gt_segments_for_scoring,
                evaluation_comparison=evaluation_comparison,
                test_name=test_name
            )
            
            result = {
                'test_name': test_name,
                'success': True,
                'image_path': test_img_path,
                'visualization_path': vis_path,
                'visualization_methods_path': vis_methods_path,
                'gt_boundary_segment_count': len(gt_boundary_segments),
                'bspline': {
                    'contour_count': len(contours_dict_bs),
                    'tree_nodes': len(curves_tree_bs),
                    'statistics': bspline_stats
                },
                'nurbs': {
                    'contour_count': len(contours_dict_nurbs),
                    'tree_nodes': len(curves_tree_nurbs),
                    'statistics': nurbs_stats
                },
                'optimized_nurbs': {
                    'contour_count': len(contours_dict_optimized_nurbs) if contours_dict_optimized_nurbs else 0,
                    'tree_nodes': len(curves_tree_optimized_nurbs) if curves_tree_optimized_nurbs else 0,
                    'statistics': optimized_nurbs_stats
                },
                'optimized_bs': {
                    'contour_count': len(contours_dict_optimized_bs) if contours_dict_optimized_bs else 0,
                    'tree_nodes': len(curves_tree_optimized_bs) if curves_tree_optimized_bs else 0,
                    'statistics': optimized_bs_stats
                },
                'evaluation_comparison': evaluation_comparison,
                'colors': {
                    'detected_count': len(color_results),
                    'results': color_results  # 将在保存时转换
                },
                'solids': {
                    'count': len(solids_mask)
                }
            }
            
            if gt_evaluation:
                result['gt_evaluation'] = gt_evaluation
            
            print("\n[OK] 测试完成")
            return result
            
        except Exception as e:
            print(f"\n[FAIL] 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'test_name': test_name,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_statistics(self, contours_dict: Dict) -> Dict:
        """计算轮廓统计信息（兼容多种格式）"""
        if not contours_dict:
            return {}
        
        # 处理不同格式的轮廓字典
        mse_values = []
        step_values = []
        control_counts = []
        fitting_counts = []
        
        for data in contours_dict.values():
            # 标准格式（bspline/nurbs）：有 'loss' 键
            if 'loss' in data and isinstance(data['loss'], dict):
                if 'mse_loss' in data['loss']:
                    mse_values.append(data['loss']['mse_loss'])
                if 'step' in data['loss']:
                    step_values.append(data['loss']['step'])
            # 优化格式（optimized）：有 'error' 键
            elif 'error' in data:
                mse_values.append(data['error'])
                step_values.append(0)  # 优化格式没有step概念
            
            # 控制点
            if 'control' in data and isinstance(data['control'], dict):
                if 'size' in data['control']:
                    control_counts.append(data['control']['size'])
                elif 'points' in data['control']:
                    control_counts.append(len(data['control']['points']))
            elif 'segments' in data:
                # 优化格式：统计所有段的控制点
                total_control = 0
                for seg in data['segments']:
                    if 'points' in seg:
                        total_control += len(seg['points'])
                    elif 'control_points' in seg:
                        total_control += len(seg['control_points'])
                if total_control > 0:
                    control_counts.append(total_control)
            elif 'path_data' in data and isinstance(data['path_data'], dict):
                # 高级格式：使用path_data中的points
                if 'points' in data['path_data']:
                    control_counts.append(len(data['path_data']['points']))
            
            # 拟合点
            if 'fitting' in data:
                if isinstance(data['fitting'], dict):
                    if 'size' in data['fitting']:
                        fitting_counts.append(data['fitting']['size'])
                    elif 'points' in data['fitting']:
                        fitting_counts.append(len(data['fitting']['points']))
                elif isinstance(data['fitting'], np.ndarray):
                    fitting_counts.append(len(data['fitting']))
            elif 'fitted_points' in data:
                # 优化格式
                if isinstance(data['fitted_points'], np.ndarray):
                    fitting_counts.append(len(data['fitted_points']))
            # 其他格式不统计拟合点数
        
        # 计算统计值（如果列表为空，返回默认值）
        result = {}
        
        if mse_values:
            result['avg_mse'] = float(np.mean(mse_values))
            result['min_mse'] = float(np.min(mse_values))
            result['max_mse'] = float(np.max(mse_values))
        else:
            result['avg_mse'] = 0.0
            result['min_mse'] = 0.0
            result['max_mse'] = 0.0
        
        if step_values:
            result['avg_step'] = float(np.mean(step_values))
        else:
            result['avg_step'] = 0.0
        
        if control_counts:
            result['avg_control_points'] = float(np.mean(control_counts))
        else:
            result['avg_control_points'] = 0.0
        
        if fitting_counts:
            result['avg_fitting_points'] = float(np.mean(fitting_counts))
        else:
            result['avg_fitting_points'] = 0.0
        
        return result
    
    def _compare_fitting_methods(self, img: np.ndarray, edges: np.ndarray,
                                 contours_dict_bs: Dict, contours_dict_nurbs: Dict,
                                 contours_dict_optimized_nurbs: Dict,
                                 contours_dict_optimized_bs: Dict,
                                 gt_segments: Optional[List[Dict]] = None) -> Dict:
        """
        对比不同拟合方法的效果
        
        参数:
            img: 原始图像
            edges: 边缘图像
            contours_dict_bs: B样条轮廓字典
            contours_dict_nurbs: NURBS轮廓字典
            contours_dict_optimized_nurbs: 优化NURBS轮廓字典
            contours_dict_optimized_bs: 优化B样条轮廓字典
        返回:
            对比评估结果
        """
        comparison = {}
        
        # 获取原始轮廓（从边缘检测）
        original_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # 评估B样条拟合
        if contours_dict_bs:
            comparison['bspline'] = self._evaluate_method(
                original_contours, contours_dict_bs, 'bspline', gt_segments=gt_segments
            )
        
        # 评估NURBS拟合
        if contours_dict_nurbs:
            comparison['nurbs'] = self._evaluate_method(
                original_contours, contours_dict_nurbs, 'nurbs', gt_segments=gt_segments
            )
        
        # 评估优化NURBS拟合
        if contours_dict_optimized_nurbs:
            # 获取分段信息（如果可用）
            segments_info_optimized_nurbs = getattr(self, '_current_segments_info_optimized_nurbs', {})
            comparison['optimized_nurbs'] = self._evaluate_method(
                original_contours, contours_dict_optimized_nurbs, 'optimized_nurbs',
                segments_info=segments_info_optimized_nurbs,
                gt_segments=gt_segments
            )
        
        # 评估优化B样条拟合
        if contours_dict_optimized_bs:
            comparison['optimized_bs'] = self._evaluate_method(
                original_contours, contours_dict_optimized_bs, 'optimized_bs', gt_segments=gt_segments
            )
        
        return comparison
    
    def _self_evaluate_segments(self, pred_segments: List[Dict]) -> Dict:
        """
        自评估分段质量（当GT为空时使用）
        
        参数:
            pred_segments: 预测分段列表
        
        返回:
            自评估结果字典
        """
        if len(pred_segments) == 0:
            return {
                'LineAccuracy': 0.0,
                'ArcRadiusScore': 0.0,
                'SplineResidual': 0.0,
                'TotalScore': 0.0,
                'StructureRecognitionAccuracy': 0.0,
                'self_evaluation': True,
                'details': {
                    'line_count': 0,
                    'arc_count': 0,
                    'spline_count': 0,
                    'total_count': 0
                }
            }
        
        # 统计分段类型
        line_count = sum(1 for s in pred_segments if s.get('type') == 'line')
        arc_count = sum(1 for s in pred_segments if s.get('type') == 'arc')
        spline_count = sum(1 for s in pred_segments if s.get('type') in ['spline', 'nurbs', 'bspline'])
        total_count = len(pred_segments)
        
        # 自评估得分（基于分段质量）
        # Line准确率：基于直线拟合误差
        line_scores = []
        for seg in pred_segments:
            if seg.get('type') == 'line' and 'points' in seg:
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
                        score = np.exp(-max_error / 2.0)  # 2像素阈值
                        line_scores.append(score)
        
        line_accuracy = np.mean(line_scores) if line_scores else 0.0
        
        # Arc得分：基于圆弧拟合误差
        arc_scores = []
        for seg in pred_segments:
            if seg.get('type') == 'arc' and 'points' in seg and 'radius' in seg:
                points = np.array(seg['points'])
                radius = seg['radius']
                center = np.array(seg.get('center', [0, 0]))
                
                if len(points) >= 3:
                    # 计算圆弧拟合误差
                    dist = np.linalg.norm(points - center, axis=1)
                    errors = np.abs(dist - radius)
                    max_error = np.max(errors)
                    # 相对误差
                    rel_error = max_error / (radius + 1e-9)
                    score = np.exp(-rel_error / 0.05)  # 5%阈值
                    arc_scores.append(score)
        
        arc_score = np.mean(arc_scores) if arc_scores else 0.0
        
        # Spline得分：基于点分布的均匀性（简化评估）
        spline_scores = []
        for seg in pred_segments:
            if seg.get('type') in ['spline', 'nurbs', 'bspline'] and 'points' in seg:
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
                            spline_scores.append(score)
                        else:
                            spline_scores.append(0.0)
        
        spline_residual = np.mean(spline_scores) if spline_scores else 0.0
        
        # 总评分
        w_line, w_arc, w_spline = 0.35, 0.40, 0.25
        total_score = w_line * line_accuracy + w_arc * arc_score + w_spline * spline_residual
        
        return {
            'LineAccuracy': float(line_accuracy),
            'ArcRadiusScore': float(arc_score),
            'SplineResidual': float(spline_residual),
            'TotalScore': float(total_score),
            'StructureRecognitionAccuracy': 1.0,  # 自评估无法计算识别准确率
            'self_evaluation': True,
            'details': {
                'line_count': line_count,
                'arc_count': arc_count,
                'spline_count': spline_count,
                'total_count': total_count,
                'line_scores': line_scores,
                'arc_scores': arc_scores,
                'spline_scores': spline_scores
            }
        }
    
    def _evaluate_method(self, original_contours: List, fitted_contours_dict: Dict,
                        method_name: str, geometric_primitives: Optional[Dict] = None,
                        segments_info: Optional[Dict] = None,
                        gt_segments: Optional[List[Dict]] = None) -> Dict:
        """
        评估单个拟合方法
        
        参数:
            original_contours: 原始轮廓列表
            fitted_contours_dict: 拟合后的轮廓字典
            method_name: 方法名称
            geometric_primitives: 几何基元（仅用于高级拟合）
        
        返回:
            评估结果
        """
        evaluations = []
        gt_segments_all = []
        used_original_indices = set()
        
        original_contours_2d = []
        for oc in original_contours:
            try:
                oc2 = oc.reshape(-1, 2)
                if len(oc2) >= 2:
                    original_contours_2d.append(oc2)
            except Exception:
                continue

        def _downsample(arr: np.ndarray, max_points: int = 400) -> np.ndarray:
            if arr is None:
                return arr
            if len(arr) <= max_points:
                return arr
            idx = np.linspace(0, len(arr) - 1, max_points, dtype=int)
            return arr[idx]

        def _symmetric_mse(a: np.ndarray, b: np.ndarray) -> float:
            if a is None or b is None or len(a) < 2 or len(b) < 2:
                return float('inf')
            try:
                dist_matrix = cdist(a, b)
                min_a = np.min(dist_matrix, axis=1)
                min_b = np.min(dist_matrix, axis=0)
                return float((np.mean(min_a ** 2) + np.mean(min_b ** 2)) / 2.0)
            except Exception:
                return float('inf')

        def _contour_area(arr: np.ndarray) -> float:
            try:
                c = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
                if len(c) < 3:
                    return 0.0
                return float(abs(cv2.contourArea(c)))
            except Exception:
                return 0.0

        fitted_entries: List[Tuple[str, Dict, np.ndarray]] = []
        for contour_id, contour_data in fitted_contours_dict.items():
            fitted_contour = None
            if 'fitting' in contour_data:
                fitted_contour = contour_data['fitting']
                if isinstance(fitted_contour, dict):
                    if 'points' in fitted_contour:
                        fitted_contour = fitted_contour['points']
                    else:
                        continue
            elif 'simplified_path' in contour_data:
                simplified = contour_data['simplified_path']
                if isinstance(simplified, dict) and 'points' in simplified:
                    fitted_contour = simplified['points']
                elif isinstance(simplified, np.ndarray):
                    fitted_contour = simplified
                else:
                    continue
            elif 'contour' in contour_data:
                fitted_contour = contour_data['contour']
            else:
                continue

            if not isinstance(fitted_contour, np.ndarray):
                try:
                    fitted_contour = np.asarray(fitted_contour)
                except Exception:
                    continue

            try:
                fitted_entries.append((contour_id, contour_data, fitted_contour))
            except Exception:
                continue

        assignment_map: Dict[int, int] = {}
        if len(original_contours_2d) > 0 and len(fitted_entries) > 0:
            orig_ds_list = []
            orig_area_list = []
            for oc2 in original_contours_2d:
                oc2f = np.asarray(oc2, dtype=float).reshape(-1, 2)
                orig_ds_list.append(_downsample(oc2f))
                orig_area_list.append(_contour_area(oc2f))

            fit_ds_list = []
            fit_area_list = []
            for _, _, fc in fitted_entries:
                fc2 = np.asarray(fc, dtype=float).reshape(-1, 2)
                fit_ds_list.append(_downsample(fc2))
                fit_area_list.append(_contour_area(fc2))

            cost_matrix = np.full((len(fitted_entries), len(original_contours_2d)), float('inf'), dtype=float)
            for r in range(len(fitted_entries)):
                for c in range(len(original_contours_2d)):
                    mse_cost = _symmetric_mse(orig_ds_list[c], fit_ds_list[r])
                    ao = float(orig_area_list[c])
                    af = float(fit_area_list[r])
                    rel_area = abs(ao - af) / max(ao, af, 1.0)
                    cost_matrix[r, c] = float(mse_cost + 2000.0 * rel_area)

            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                    assignment_map[int(r)] = int(c)
            except Exception:
                pass

            for r in range(len(fitted_entries)):
                if r not in assignment_map:
                    try:
                        assignment_map[r] = int(np.argmin(cost_matrix[r]))
                    except Exception:
                        pass
        
        # 对每个拟合轮廓进行评估
        for entry_idx, (contour_id, contour_data, fitted_contour) in enumerate(fitted_entries):
            # 获取拟合轮廓
            # 找到对应的原始轮廓（最近匹配）
            if len(original_contours_2d) == 0:
                continue

            best_idx = assignment_map.get(int(entry_idx), None)

            if best_idx is None:
                continue

            original_contour = original_contours_2d[best_idx].astype(float)
            if best_idx not in used_original_indices:
                try:
                    gt_segments_all.extend(extract_segments_from_fitted_contour(original_contour))
                except Exception:
                    pass
                used_original_indices.add(best_idx)
            
            # 确保fitted_contour是正确的形状
            if len(fitted_contour.shape) == 1:
                # 一维数组，需要reshape
                if len(fitted_contour) % 2 == 0:
                    fitted_contour = fitted_contour.reshape(-1, 2)
                else:
                    continue
            elif len(fitted_contour.shape) == 2:
                # 已经是二维数组
                if fitted_contour.shape[1] != 2:
                    # 尝试转置
                    if fitted_contour.shape[0] == 2:
                        fitted_contour = fitted_contour.T
                    else:
                        continue
            else:
                continue
            
            # 创建评估器
            try:
                evaluator = CurveEvaluator(
                    original_contour=original_contour,
                    fitted_contour=fitted_contour,
                    structure_type=contour_data.get('structure_type')
                )
            except Exception as e:
                print(f"   [WARN] 轮廓 {contour_id} 评估失败: {e}")
                continue
            
            # 获取几何基元（如果可用）
            primitives = None
            if geometric_primitives and contour_id in geometric_primitives:
                primitives = geometric_primitives[contour_id]
            
            # 执行评估
            metrics = evaluator.evaluate_all()
            
            # 添加语义评估（如果有基元）
            if primitives:
                semantic_metrics = evaluator.evaluate_semantic_accuracy(primitives)
                metrics['semantic'] = semantic_metrics
            
            evaluations.append({
                'contour_id': contour_id,
                'metrics': metrics
            })
        
        # 几何感知评分（如果有分段信息）
        geometry_aware_score = None
        pred_segments = []
        
        # 对于所有方法，尝试提取分段信息
        if method_name in ['optimized_nurbs', 'optimized_bs']:
            # 对于支持分段的方法，从拟合结果中提取分段信息
            if method_name == 'optimized_nurbs' and segments_info:
                # 优化NURBS：从segments_info中提取分段
                for contour_id, seg_list in segments_info.items():
                    if isinstance(seg_list, list):
                        pred_segments.extend(seg_list)
            else:
                # 对于其他方法，尝试从contours_dict中提取
                for contour_id, contour_data in fitted_contours_dict.items():
                    # 检查是否有分段信息
                    if 'segments' in contour_data:
                        segments = contour_data['segments']
                        if isinstance(segments, list):
                            pred_segments.extend(segments)
        else:
            # 对于传统方法（B样条、NURBS），从拟合点中提取分段信息
            for contour_id, contour_data in fitted_contours_dict.items():
                # 获取拟合点
                fitted_points = None
                if 'fitting' in contour_data:
                    fitted = contour_data['fitting']
                    if isinstance(fitted, dict) and 'points' in fitted:
                        fitted_points = np.array(fitted['points'])
                    elif isinstance(fitted, np.ndarray):
                        fitted_points = fitted
                elif 'fitted_points' in contour_data:
                    fitted_points = np.array(contour_data['fitted_points'])
                
                if fitted_points is not None and len(fitted_points) > 0:
                    # 确保是2D数组
                    if len(fitted_points.shape) == 1:
                        if len(fitted_points) % 2 == 0:
                            fitted_points = fitted_points.reshape(-1, 2)
                        else:
                            continue
                    elif len(fitted_points.shape) == 2:
                        if fitted_points.shape[1] != 2:
                            if fitted_points.shape[0] == 2:
                                fitted_points = fitted_points.T
                            else:
                                continue
                    else:
                        continue
                    
                    # 从拟合点中提取分段信息
                    try:
                        segments = extract_segments_from_fitted_contour(fitted_points)
                        pred_segments.extend(segments)
                    except Exception as e:
                        print(f"   [WARN] 轮廓 {contour_id} 分段提取失败: {e}")
        
        # 如果有分段信息，计算几何感知评分
        if len(pred_segments) > 0:
            try:
                if gt_segments is not None and len(gt_segments) > 0:
                    geometry_aware_score = compute_geometry_aware_score(
                        gt_segments, pred_segments
                    )
                elif len(gt_segments_all) > 0:
                    geometry_aware_score = compute_geometry_aware_score(
                        gt_segments_all, pred_segments
                    )
                else:
                    geometry_aware_score = self._self_evaluate_segments(pred_segments)
            except Exception as e:
                print(f"   [WARN] 几何感知评分失败: {e}")

        parameterization = self._summarize_segments_for_report(pred_segments)
        
        # 汇总评估结果
        if evaluations:
            # 计算平均得分
            overall_scores = [e['metrics']['overall']['overall'] for e in evaluations]
            avg_overall = np.mean(overall_scores) if overall_scores else 0.0
            
            # 计算各维度平均得分
            geo_scores = [e['metrics']['overall']['geometric'] for e in evaluations]
            topo_scores = [e['metrics']['overall']['topology'] for e in evaluations]
            sem_scores = [e['metrics']['overall']['semantic'] for e in evaluations]
            eng_scores = [e['metrics']['overall']['engineering'] for e in evaluations]
            
            result = {
                'method': method_name,
                'contour_count': len(evaluations),
                'overall': {
                    'overall': avg_overall,
                    'geometric': np.mean(geo_scores) if geo_scores else 0.0,
                    'topology': np.mean(topo_scores) if topo_scores else 0.0,
                    'semantic': np.mean(sem_scores) if sem_scores else 0.0,
                    'engineering': np.mean(eng_scores) if eng_scores else 0.0
                },
                'detailed': evaluations
            }
            
            # 添加几何感知评分
            if geometry_aware_score:
                result['geometry_aware'] = geometry_aware_score

            result['parameterization'] = parameterization
            
            return result
        else:
            result = {
                'method': method_name,
                'contour_count': 0,
                'overall': {
                    'overall': 0.0,
                    'geometric': 0.0,
                    'topology': 0.0,
                    'semantic': 0.0,
                    'engineering': 0.0
                },
                'detailed': []
            }
            
            # 添加几何感知评分
            if geometry_aware_score:
                result['geometry_aware'] = geometry_aware_score

            result['parameterization'] = parameterization
            
            return result
    
    def _evaluate_against_gt(self, contours_dict: Dict, gt: Dict) -> Dict:
        """
        与Ground Truth对比评估
        
        参数:
            contours_dict: 检测到的轮廓字典
            gt: Ground Truth字典
        
        返回:
            评估结果字典
        """
        evaluation = {
            'contour_count_match': False,
            'structure_type': gt.get('structure_type', 'unknown'),
            'gt_layer': gt.get('layer', 0),
            'detected_contours': len(contours_dict),
        }
        
        # 根据结构类型进行特定评估
        stype = gt.get('structure_type', '')
        
        if stype in ['rect_patch', 'ring_patch', 'polygon_patch']:
            # 简单结构：应该检测到1-2个轮廓（外轮廓+可能的孔）
            expected_min = 1
            expected_max = 2
            evaluation['contour_count_match'] = expected_min <= len(contours_dict) <= expected_max
        
        elif stype in ['SRR', 'CSRR', 'multi_ring_SRR']:
            # 环结构：应该检测到多个轮廓
            evaluation['contour_count_match'] = len(contours_dict) >= 1
        
        elif stype == 'sierpinski_carpet':
            # 分形结构：轮廓数量取决于深度
            depth = gt.get('depth', 1)
            expected_min = 1
            evaluation['contour_count_match'] = len(contours_dict) >= expected_min
        
        elif stype == 'unit_array':
            # 阵列结构：应该检测到多个单元
            num_units = gt.get('num_units', 1)
            evaluation['contour_count_match'] = len(contours_dict) >= num_units
        
        else:
            evaluation['contour_count_match'] = len(contours_dict) > 0
        
        return evaluation
    
    def _visualize_comparison(self, original_img: np.ndarray, processed_img: np.ndarray,
                             edges: np.ndarray, contours_dict_bs: Dict, contours_dict_nurbs: Dict,
                             test_name: str) -> str:
        """
        生成原图与拟合结果的对比可视化
        
        参数:
            original_img: 原始测试图像
            processed_img: 处理后的图像
            edges: 边缘图像
            contours_dict_bs: B样条轮廓字典
            contours_dict_nurbs: NURBS轮廓字典
            test_name: 测试名称
        
        返回:
            可视化图像保存路径
        """
        vis_path = os.path.join(self.output_dir, f"{test_name}_comparison.png")
        
        # 计算子图数量
        num_contours = max(len(contours_dict_bs), len(contours_dict_nurbs), 1)
        cols = 3  # 原图、边缘、拟合结果
        rows = max(2, 1 + (num_contours + 2) // 3)  # 至少2行
        
        fig = plt.figure(figsize=(cols * 6, rows * 5))
        
        # 子图1: 原始测试图像
        ax1 = plt.subplot(rows, cols, 1)
        if len(original_img.shape) == 3:
            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = original_img
        ax1.imshow(img_rgb)
        ax1.set_title("原始测试图像", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 子图2: 边缘检测结果
        ax2 = plt.subplot(rows, cols, 2)
        ax2.imshow(edges, cmap='gray')
        ax2.set_title("边缘检测结果", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 子图3: B样条拟合结果（第一个轮廓）
        ax3 = plt.subplot(rows, cols, 3)
        if contours_dict_bs:
            first_id = list(contours_dict_bs.keys())[0]
            first_data = contours_dict_bs[first_id]
            
            # 原始轮廓点
            if 'contour' in first_data and first_data['contour'] is not None:
                contour_points = first_data['contour'].reshape(-1, 2)
                ax3.scatter(contour_points[:, 0], contour_points[:, 1], 
                          c='green', s=3, alpha=0.6, label='原始轮廓点', zorder=1)
            
            # 控制点
            if first_data['control']['points'] is not None:
                control_pts = first_data['control']['points']
                ax3.scatter(control_pts[:, 0], control_pts[:, 1], 
                          c='red', s=50, marker='D', label='控制点', zorder=3, edgecolors='black', linewidths=1)
            
            # 拟合曲线
            if first_data['fitting']['points'] is not None:
                fitting_pts = first_data['fitting']['points']
                ax3.plot(fitting_pts[:, 0], fitting_pts[:, 1], 
                       'b-', linewidth=2.5, label='B样条拟合', zorder=2)
            
            ax3.set_title(f'B样条拟合结果\n轮廓 {first_id}\nMSE: {first_data["loss"]["mse_loss"]:.2f}, 控制点: {first_data["control"]["size"]}', 
                        fontsize=12, fontweight='bold')
            ax3.legend(fontsize=10, loc='best')
            ax3.axis('equal')
            ax3.grid(True, alpha=0.3)
            ax3.set_aspect('equal', adjustable='box')
        else:
            ax3.text(0.5, 0.5, '未检测到轮廓', ha='center', va='center', fontsize=14)
            ax3.set_title('B样条拟合结果\n(无轮廓)', fontsize=12)
            ax3.axis('off')
        
        # 如果有多个轮廓，继续显示（最多显示6个额外轮廓）
        contour_idx = 0
        for contour_id, data in list(contours_dict_bs.items())[1:7]:
            contour_idx += 1
            row = 2 + contour_idx // 3
            col = 1 + (contour_idx % 3)
            
            if row > rows:
                break
            
            pos = (row - 1) * cols + col
            if pos > rows * cols:
                break
            
            ax = plt.subplot(rows, cols, pos)
            
            # 原始轮廓点
            if 'contour' in data and data['contour'] is not None:
                contour_points = data['contour'].reshape(-1, 2)
                ax.scatter(contour_points[:, 0], contour_points[:, 1], 
                          c='green', s=2, alpha=0.5, label='原始点', zorder=1)
            
            # 控制点
            if data['control']['points'] is not None:
                control_pts = data['control']['points']
                ax.scatter(control_pts[:, 0], control_pts[:, 1], 
                          c='red', s=30, marker='D', label='控制点', zorder=3, edgecolors='black', linewidths=0.5)
            
            # 拟合曲线
            if data['fitting']['points'] is not None:
                fitting_pts = data['fitting']['points']
                ax.plot(fitting_pts[:, 0], fitting_pts[:, 1], 
                       'b-', linewidth=2, label='拟合曲线', zorder=2)
            
            ax.set_title(f'轮廓 {contour_id}\nMSE: {data["loss"]["mse_loss"]:.2f}', 
                        fontsize=10)
            ax.legend(fontsize=8, loc='best')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"   [OK] 可视化对比图已保存: {vis_path}")
        plt.close()
        
        return vis_path

    @staticmethod
    def _extract_fitted_points(contour_data: Dict) -> Optional[np.ndarray]:
        fitted_points = None
        if 'fitting' in contour_data:
            fitted = contour_data['fitting']
            if isinstance(fitted, dict) and 'points' in fitted:
                fitted_points = np.array(fitted['points'])
            elif isinstance(fitted, np.ndarray):
                fitted_points = fitted
        elif 'simplified_path' in contour_data:
            simplified = contour_data['simplified_path']
            if isinstance(simplified, dict) and 'points' in simplified:
                fitted_points = np.array(simplified['points'])
            elif isinstance(simplified, np.ndarray):
                fitted_points = simplified
        elif 'fitted_points' in contour_data:
            fitted_points = np.array(contour_data['fitted_points'])

        if fitted_points is None:
            return None

        fitted_points = np.asarray(fitted_points)
        if fitted_points.ndim == 1:
            if fitted_points.size % 2 != 0:
                return None
            fitted_points = fitted_points.reshape(-1, 2)
        elif fitted_points.ndim == 2:
            if fitted_points.shape[1] != 2:
                if fitted_points.shape[0] == 2:
                    fitted_points = fitted_points.T
                else:
                    return None
        else:
            return None

        if len(fitted_points) < 2:
            return None

        return fitted_points

    @staticmethod
    def _draw_polyline_bgr(canvas: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int = 2):
        pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
        if len(pts) < 2:
            return
        closed = bool(np.linalg.norm(pts[0] - pts[-1]) <= 2.5)
        cv2.polylines(canvas, [pts], closed, color, thickness, lineType=cv2.LINE_AA)

    def _draw_segments_bgr(self, canvas: np.ndarray, segments: List[Dict], color: Tuple[int, int, int], thickness: int = 2):
        for seg in segments or []:
            pts = seg.get('points', None)
            if pts is None:
                continue
            pts = np.asarray(pts, dtype=float).reshape(-1, 2)
            if len(pts) < 2:
                continue
            self._draw_polyline_bgr(canvas, pts, color, thickness)

    def _visualize_method_overview(self,
                                   processed_img: np.ndarray,
                                   edges: np.ndarray,
                                   contours_dict_by_method: Dict[str, Dict],
                                   gt_segments: Optional[List[Dict]],
                                   evaluation_comparison: Optional[Dict],
                                   test_name: str) -> str:
        vis_path = os.path.join(self.output_dir, f"{test_name}_methods_overview.png")

        overlays: List[Tuple[str, np.ndarray]] = []

        base = processed_img.copy()
        gt_overlay = base.copy()
        if gt_segments:
            self._draw_segments_bgr(gt_overlay, gt_segments, (0, 0, 255), thickness=2)
        overlays.append(("GT(boundary segments)", gt_overlay))

        method_colors = {
            'bspline': (0, 180, 0),
            'nurbs': (180, 0, 0),
            'optimized_nurbs': (180, 0, 180),
            'optimized_bs': (0, 180, 180),
        }

        for method_name, contours_dict in contours_dict_by_method.items():
            overlay = base.copy()
            for contour_data in (contours_dict or {}).values():
                pts = self._extract_fitted_points(contour_data)
                if pts is None:
                    continue
                self._draw_polyline_bgr(overlay, pts, method_colors.get(method_name, (0, 0, 0)), thickness=2)

            title = method_name
            if evaluation_comparison and method_name in evaluation_comparison:
                ga = evaluation_comparison[method_name].get('geometry_aware', None)
                if isinstance(ga, dict) and 'TotalScore' in ga:
                    title = f"{method_name}  GA={ga['TotalScore']:.3f}"
            overlays.append((title, overlay))

        cols = 3
        rows = int(np.ceil((len(overlays) + 2) / cols))
        fig = plt.figure(figsize=(cols * 6, rows * 5))

        ax1 = plt.subplot(rows, cols, 1)
        ax1.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        ax1.set_title("centered_img", fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot(rows, cols, 2)
        ax2.imshow(edges, cmap='gray')
        ax2.set_title("edges", fontsize=12, fontweight='bold')
        ax2.axis('off')

        for i, (title, bgr) in enumerate(overlays, start=3):
            ax = plt.subplot(rows, cols, i)
            ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"   [OK] 多方法可视化已保存: {vis_path}")
        plt.close()

        return vis_path
    
    def run_test_suite(self, test_cases: List[Dict], use_pre_generated: bool = False, 
                       images_dir: str = "test_images",
                       validation_dataset_dir: Optional[str] = None) -> Dict:
        """
        运行测试套件
        
        参数:
            test_cases: 测试用例列表，每个用例格式:
                {
                    'name': 'test_name',
                    'type': 'case_type',
                    'params': {...},
                    'image_path': 'path/to/image.png'  # 可选，如果提供则使用预生成图像
                }
            use_pre_generated: 是否优先使用预生成的图像
            images_dir: 预生成图像目录
            validation_dataset_dir: 验证集目录（如果提供，将从该目录加载图像和GT）
        """
        print("\n" + "=" * 70)
        print("开始运行超材料/天线图像处理测试套件")
        print("=" * 70)
        
        results = {
            'total': len(test_cases),
            'passed': 0,
            'failed': 0,
            'tests': []
        }
        
        # 如果提供了验证集目录，加载验证集数据
        validation_images = {}
        validation_gts = {}
        if validation_dataset_dir and os.path.exists(validation_dataset_dir):
            print(f"\n加载验证集数据: {validation_dataset_dir}")
            images_path = os.path.join(validation_dataset_dir, "images")
            gt_path = os.path.join(validation_dataset_dir, "gt")
            
            if os.path.exists(images_path) and os.path.exists(gt_path):
                # 加载所有验证集图像和GT
                for img_file in os.listdir(images_path):
                    if img_file.endswith('.png'):
                        name = os.path.splitext(img_file)[0]
                        validation_images[name] = os.path.join(images_path, img_file)
                        gt_file = os.path.join(gt_path, f"{name}.json")
                        if os.path.exists(gt_file):
                            validation_gts[name] = gt_file
                print(f"  找到 {len(validation_images)} 个验证集样本")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] ", end="")
            
            test_image = None
            image_path = None
            gt_path = None
            gt_segments = None
            
            # 优先使用验证集数据
            if validation_dataset_dir and validation_images:
                # 尝试从验证集加载
                val_name = test_case.get('name', f"val_{i-1:05d}")
                if val_name in validation_images:
                    image_path = validation_images[val_name]
                    gt_path = validation_gts.get(val_name)
                    print(f"使用验证集样本: {val_name}")
            
            # 其次使用预生成图像
            if image_path is None:
                if use_pre_generated or 'image_path' in test_case:
                    if 'image_path' in test_case:
                        image_path = test_case['image_path']
                    else:
                        # 尝试从images_dir加载
                        potential_path = os.path.join(images_dir, f"{test_case['name']}.png")
                        if os.path.exists(potential_path):
                            image_path = potential_path
                
                # 如果没有预生成图像，则生成
                if image_path is None or not os.path.exists(image_path):
                    test_image = self.generator.generate_test_case(
                        test_case['type'],
                        **test_case.get('params', {})
                    )
                    gt_segments = self.generator.get_gt_segments()
            
            # 获取GT路径（如果存在）
            if gt_path is None:
                if 'gt_path' in test_case:
                    gt_path = test_case['gt_path']
                elif use_pre_generated:
                    # 尝试从数据集目录加载GT
                    potential_gt_path = os.path.join(images_dir.replace('images', 'gt'), 
                                                    f"{test_case['name']}.json")
                    if os.path.exists(potential_gt_path):
                        gt_path = potential_gt_path
            
            # 运行测试
            result = self.test_image_processing(
                test_image, test_case['name'], image_path, gt_path, gt_segments=gt_segments
            )
            results['tests'].append(result)
            
            if result['success']:
                results['passed'] += 1
            else:
                results['failed'] += 1
        
        # 生成报告
        self._generate_report(results)
        
        return results
    
    def _generate_report(self, results: Dict):
        """生成测试报告"""
        print("\n" + "=" * 70)
        print("测试总结")
        print("=" * 70)
        print(f"总测试数: {results['total']}")
        print(f"通过: {results['passed']}")
        print(f"失败: {results['failed']}")
        print(f"通过率: {results['passed']/results['total']*100:.1f}%")
        
        # 生成对比表格
        comparison_table = self._generate_comparison_table(results)
        if comparison_table:
            print("\n" + "=" * 70)
            print("拟合方法对比表格")
            print("=" * 70)
            if len(comparison_table) > 5000:
                print(comparison_table[:5000] + "\n...(console truncated, full table saved to file)...\n")
            else:
                print(comparison_table)
            
            # 保存对比表格到文件
            table_path = os.path.join(self.output_dir, "comparison_table.txt")
            try:
                with open(table_path, 'w', encoding='utf-8') as f:
                    f.write("拟合方法对比表格\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(comparison_table)
                print(f"\n对比表格已保存: {table_path}")
            except Exception as e:
                print(f"\n[WARN] 对比表格保存失败: {e}")
        
        if self.save_json_report:
            report_path = os.path.join(self.output_dir, "test_report.json")
            try:
                serializable_results = self._convert_to_json_serializable(results)
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, ensure_ascii=False)
                print(f"\n报告已保存: {report_path}")
            except Exception as e:
                print(f"\n[WARN] JSON报告保存失败: {e}")
    
    def _generate_comparison_table(self, results: Dict) -> Optional[str]:
        """
        生成拟合方法对比表格
        
        参数:
            results: 测试结果字典
        
        返回:
            对比表格字符串
        """
        # 收集所有测试的评估数据
        all_evaluations = {
            'bspline': [],
            'nurbs': [],
            'optimized_nurbs': [],
            'optimized_bs': [],
        }
        all_parameterizations = {
            'bspline': [],
            'nurbs': [],
            'optimized_nurbs': [],
            'optimized_bs': [],
        }
        
        for test_result in results.get('tests', []):
            if not test_result.get('success', False):
                continue
            
            eval_comparison = test_result.get('evaluation_comparison', {})
            
            for method in ['bspline', 'nurbs', 'optimized_nurbs', 'optimized_bs']:
                if method in eval_comparison:
                    method_data = eval_comparison[method]
                    if 'overall' in method_data:
                        all_evaluations[method].append(method_data['overall'])
                    if 'parameterization' in method_data:
                        all_parameterizations[method].append(method_data['parameterization'])
        
        # 如果没有评估数据，返回None
        if not any(all_evaluations.values()):
            return None
        
        # 计算统计信息
        stats = {}
        for method, evaluations in all_evaluations.items():
            if not evaluations:
                continue
            
            overall_scores = [e.get('overall', 0.0) for e in evaluations]
            geo_scores = [e.get('geometric', 0.0) for e in evaluations]
            topo_scores = [e.get('topology', 0.0) for e in evaluations]
            sem_scores = [e.get('semantic', 0.0) for e in evaluations]
            eng_scores = [e.get('engineering', 0.0) for e in evaluations]

            param_list = all_parameterizations.get(method, [])
            seg_counts = [p.get('segment_count', 0) for p in param_list if isinstance(p, dict)]
            line_counts = [p.get('line_count', 0) for p in param_list if isinstance(p, dict)]
            arc_counts = [p.get('arc_count', 0) for p in param_list if isinstance(p, dict)]
            spline_counts = [p.get('spline_count', 0) for p in param_list if isinstance(p, dict)]
            
            stats[method] = {
                'count': len(evaluations),
                'overall': {
                    'mean': np.mean(overall_scores) if overall_scores else 0.0,
                    'std': np.std(overall_scores) if overall_scores else 0.0,
                    'min': np.min(overall_scores) if overall_scores else 0.0,
                    'max': np.max(overall_scores) if overall_scores else 0.0
                },
                'geometric': {
                    'mean': np.mean(geo_scores) if geo_scores else 0.0,
                    'std': np.std(geo_scores) if geo_scores else 0.0
                },
                'topology': {
                    'mean': np.mean(topo_scores) if topo_scores else 0.0,
                    'std': np.std(topo_scores) if topo_scores else 0.0
                },
                'semantic': {
                    'mean': np.mean(sem_scores) if sem_scores else 0.0,
                    'std': np.std(sem_scores) if sem_scores else 0.0
                },
                'engineering': {
                    'mean': np.mean(eng_scores) if eng_scores else 0.0,
                    'std': np.std(eng_scores) if eng_scores else 0.0
                },
                'segments': {
                    'total': {
                        'mean': float(np.mean(seg_counts)) if seg_counts else 0.0,
                        'std': float(np.std(seg_counts)) if seg_counts else 0.0
                    },
                    'line': {
                        'mean': float(np.mean(line_counts)) if line_counts else 0.0,
                        'std': float(np.std(line_counts)) if line_counts else 0.0
                    },
                    'arc': {
                        'mean': float(np.mean(arc_counts)) if arc_counts else 0.0,
                        'std': float(np.std(arc_counts)) if arc_counts else 0.0
                    },
                    'spline': {
                        'mean': float(np.mean(spline_counts)) if spline_counts else 0.0,
                        'std': float(np.std(spline_counts)) if spline_counts else 0.0
                    }
                }
            }
        
        # 生成表格
        table_lines = []
        
        # 表头
        methods = ['bspline', 'nurbs', 'optimized_nurbs', 'optimized_bs']
        method_names = {
            'bspline': 'B样条拟合', 
            'nurbs': 'NURBS拟合',
            'optimized_nurbs': '优化NURBS拟合',
            'optimized_bs': '优化B样条拟合',
        }
        table_width = 90
        
        # 总体对比
        table_lines.append("【总体对比】")
        table_lines.append("-" * table_width)
        table_lines.append(f"{'方法':<20} {'样本数':<8} {'综合得分':<14} {'几何精度':<14} {'拓扑一致性':<14} {'总段':<8}")
        table_lines.append("-" * table_width)
        
        # 收集几何感知评分数据
        geometry_aware_data = {}
        for test_result in results.get('tests', []):
            if not test_result.get('success', False):
                continue
            eval_comparison = test_result.get('evaluation_comparison', {})
            for method in methods:
                if method in eval_comparison:
                    method_data = eval_comparison[method]
                    if 'geometry_aware' in method_data:
                        if method not in geometry_aware_data:
                            geometry_aware_data[method] = []
                        geometry_aware_data[method].append(method_data['geometry_aware'])
        
        for method in methods:
            if method not in stats:
                continue
            s = stats[method]
            seg_total = s.get('segments', {}).get('total', {})
            seg_total_mean = float(seg_total.get('mean', 0.0))
            seg_total_std = float(seg_total.get('std', 0.0))
            table_lines.append(
                f"{method_names[method]:<20} "
                f"{s['count']:<8} "
                f"{s['overall']['mean']:.2f}±{s['overall']['std']:.2f}  "
                f"{s['geometric']['mean']:.2f}±{s['geometric']['std']:.2f}  "
                f"{s['topology']['mean']:.2f}±{s['topology']['std']:.2f}  "
                f"{seg_total_mean:.1f}±{seg_total_std:.1f}"
            )
        
        table_lines.append("-" * table_width)
        table_lines.append("")

        if any(all_parameterizations.values()):
            table_lines.append("【分段统计对比】")
            table_lines.append("-" * table_width)
            table_lines.append(f"{'方法':<20} {'总段':<14} {'直线':<14} {'圆弧':<14} {'曲线':<14}")
            table_lines.append("-" * table_width)
            for method in methods:
                if method not in stats:
                    continue
                s = stats[method]
                seg_total = s.get('segments', {}).get('total', {})
                seg_line = s.get('segments', {}).get('line', {})
                seg_arc = s.get('segments', {}).get('arc', {})
                seg_spline = s.get('segments', {}).get('spline', {})
                table_lines.append(
                    f"{method_names.get(method, method):<20} "
                    f"{float(seg_total.get('mean', 0.0)):.1f}±{float(seg_total.get('std', 0.0)):.1f}  "
                    f"{float(seg_line.get('mean', 0.0)):.1f}±{float(seg_line.get('std', 0.0)):.1f}  "
                    f"{float(seg_arc.get('mean', 0.0)):.1f}±{float(seg_arc.get('std', 0.0)):.1f}  "
                    f"{float(seg_spline.get('mean', 0.0)):.1f}±{float(seg_spline.get('std', 0.0)):.1f}"
                )
            table_lines.append("-" * table_width)
            table_lines.append("")
        
        # 几何感知评分对比（如果有）
        if geometry_aware_data:
            table_lines.append("【几何感知评分对比】")
            table_lines.append("-" * table_width)
            table_lines.append(f"{'方法':<20} {'直线准确率':<15} {'圆弧半径得分':<15} {'自由曲线残差':<15} {'总评分':<15}")
            table_lines.append("-" * table_width)
            
            for method in methods:
                if method in geometry_aware_data:
                    scores = geometry_aware_data[method]
                    line_acc = [s.get('LineAccuracy', 0.0) for s in scores]
                    arc_score = [s.get('ArcRadiusScore', 0.0) for s in scores]
                    spline_res = [s.get('SplineResidual', 0.0) for s in scores]
                    total_score = [s.get('TotalScore', 0.0) for s in scores]
                    
                    table_lines.append(
                        f"{method_names.get(method, method):<20} "
                        f"{np.mean(line_acc):.3f}±{np.std(line_acc):.3f}  "
                        f"{np.mean(arc_score):.3f}±{np.std(arc_score):.3f}  "
                        f"{np.mean(spline_res):.3f}±{np.std(spline_res):.3f}  "
                        f"{np.mean(total_score):.3f}±{np.std(total_score):.3f}"
                    )
                else:
                    table_lines.append(f"{method_names.get(method, method):<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
            table_lines.append("-" * table_width)
            table_lines.append("")
        
        # 详细对比
        table_lines.append("【详细对比 - 各维度得分】")
        table_lines.append("-" * table_width)
        table_lines.append(f"{'方法':<20} {'语义准确性':<15} {'工程适用性':<15} {'得分范围':<15}")
        table_lines.append("-" * table_width)
        
        for method in methods:
            if method not in stats:
                continue
            s = stats[method]
            score_range = f"{s['overall']['min']:.1f}-{s['overall']['max']:.1f}"
            table_lines.append(
                f"{method_names[method]:<20} "
                f"{s['semantic']['mean']:.2f}±{s['semantic']['std']:.2f}    "
                f"{s['engineering']['mean']:.2f}±{s['engineering']['std']:.2f}    "
                f"{score_range:<15}"
            )
        
        table_lines.append("-" * table_width)
        table_lines.append("")
        
        # 排名
        if len(stats) > 1:
            table_lines.append("【排名（按综合得分）】")
            table_lines.append("-" * table_width)
            sorted_methods = sorted(stats.items(), key=lambda x: x[1]['overall']['mean'], reverse=True)
            for rank, (method, s) in enumerate(sorted_methods, 1):
                table_lines.append(f"{rank}. {method_names[method]}: {s['overall']['mean']:.2f}分")
            table_lines.append("-" * table_width)
        
        return "\n".join(table_lines)


def create_default_test_cases() -> List[Dict]:
    """创建默认测试用例"""
    return [
        # 基础结构
        {
            'name': 'simple_rect_patch',
            'type': 'simple_patch',
            'params': {
                'center': (500, 500),
                'width': 200,
                'height': 150,
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'nested_rings',
            'type': 'nested_rings',
            'params': {
                'center': (500, 500),
                'outer_radius': 200,
                'inner_radius': 100,
                'color': (0, 0, 0)
            }
        },
        # 天线结构
        {
            'name': 'slot_antenna_45deg',
            'type': 'slot_antenna',
            'params': {
                'center': (500, 500),
                'length': 300,
                'width': 20,
                'angle': 45,
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'meander_antenna',
            'type': 'meander_antenna',
            'params': {
                'start': (200, 400),
                'width': 50,
                'height': 30,
                'segments': 8,
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'spiral_antenna',
            'type': 'spiral_antenna',
            'params': {
                'center': (500, 500),
                'start_radius': 50,
                'end_radius': 200,
                'turns': 3,
                'color': (0, 0, 0)
            }
        },
        # FSS单元
        {
            'name': 'fss_cross',
            'type': 'fss_unit',
            'params': {
                'center': (500, 500),
                'size': 300,
                'pattern': 'cross',
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'fss_square_loop',
            'type': 'fss_unit',
            'params': {
                'center': (500, 500),
                'size': 300,
                'pattern': 'square_loop',
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'complex_fss',
            'type': 'complex_fss',
            'params': {
                'center': (500, 500)
            }
        },
        # 分形结构
        {
            'name': 'koch_snowflake',
            'type': 'koch_snowflake',
            'params': {
                'center': (500, 500),
                'radius': 200,
                'iterations': 2,
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'sierpinski_triangle',
            'type': 'sierpinski_triangle',
            'params': {
                'center': (500, 500),
                'size': 400,
                'depth': 3,
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'fractal_tree',
            'type': 'fractal_tree',
            'params': {
                'center': (500, 700),
                'length': 200,
                'depth': 4,
                'color': (0, 0, 0)
            }
        },
        # 多层材料（多颜色）
        {
            'name': 'layered_materials_2color',
            'type': 'layered_materials',
            'params': {
                'layers': [
                    {'type': 'rect', 'center': (500, 500), 
                     'params': {'width': 300, 'height': 300}, 
                     'color': (0, 0, 255)},  # 红色
                    {'type': 'circle', 'center': (500, 500), 
                     'params': {'radius': 100}, 
                     'color': (255, 0, 0)},  # 蓝色
                ]
            }
        },
        {
            'name': 'multi_color_layers',
            'type': 'multi_color_layers',
            'params': {
                'layers': [
                    {'type': 'rect', 'center': (500, 500), 
                     'params': {'width': 400, 'height': 400}, 
                     'color': (0, 0, 255)},  # 红色外层
                    {'type': 'circle', 'center': (500, 500), 
                     'params': {'radius': 150}, 
                     'color': (255, 0, 0)},  # 蓝色中层
                    {'type': 'ring', 'center': (500, 500), 
                     'params': {'outer_radius': 100, 'inner_radius': 50}, 
                     'color': (0, 255, 0)},  # 绿色内层
                ]
            }
        },
        {
            'name': 'multi_color_complex',
            'type': 'multi_color_layers',
            'params': {
                'layers': [
                    {'type': 'rect', 'center': (500, 500), 
                     'params': {'width': 500, 'height': 500}, 
                     'color': (0, 0, 255)},  # 红色
                    {'type': 'circle', 'center': (400, 400), 
                     'params': {'radius': 100}, 
                     'color': (255, 0, 0)},  # 蓝色
                    {'type': 'circle', 'center': (600, 400), 
                     'params': {'radius': 100}, 
                     'color': (0, 255, 0)},  # 绿色
                    {'type': 'circle', 'center': (500, 600), 
                     'params': {'radius': 100}, 
                     'color': (255, 255, 0)},  # 青色
                ]
            }
        },
        # 复杂组合
        {
            'name': 'complex_combination',
            'type': 'complex_combination',
            'params': {
                'center': (500, 500)
            }
        },
        # 复杂曲线组合结构
        {
            'name': 'curved_patch',
            'type': 'curved_patch',
            'params': {
                'center': (500, 500),
                'size': 300,
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'hybrid_antenna',
            'type': 'hybrid_antenna',
            'params': {
                'center': (500, 500),
                'size': 400,
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'waveguide',
            'type': 'waveguide',
            'params': {
                'center': (500, 500),
                'size': 500,
                'color': (0, 0, 0)
            }
        },
        {
            'name': 'curve_line_combo',
            'type': 'curve_line_combo',
            'params': {
                'start': (200, 500),
                'end': (800, 500),
                'curve_type': 'arc',
                'curve_params': {
                    'center': (500, 400),
                    'radius': 150
                },
                'color': (0, 0, 0)
            }
        },
        # 多颜色层级结构（5色）
        {
            'name': 'multi_color_5level',
            'type': 'multi_color_hierarchy',
            'params': {
                'hierarchy': [
                    {'level': 1, 'shapes': [
                        {'type': 'rect', 'center': (500, 500), 'params': {'width': 600, 'height': 600}}
                    ], 'color': (0, 0, 255)},  # 红色
                    {'level': 2, 'shapes': [
                        {'type': 'circle', 'center': (500, 500), 'params': {'radius': 250}}
                    ], 'color': (255, 0, 0)},  # 蓝色
                    {'level': 3, 'shapes': [
                        {'type': 'rect', 'center': (500, 500), 'params': {'width': 300, 'height': 300}}
                    ], 'color': (0, 255, 0)},  # 绿色
                    {'level': 4, 'shapes': [
                        {'type': 'circle', 'center': (500, 500), 'params': {'radius': 120}}
                    ], 'color': (255, 255, 0)},  # 青色
                    {'level': 5, 'shapes': [
                        {'type': 'ring', 'center': (500, 500), 'params': {'outer_radius': 60, 'inner_radius': 30}}
                    ], 'color': (255, 0, 255)},  # 洋红
                ]
            }
        },
    ]


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='超材料/天线图像处理测试')
    parser.add_argument('--output', type=str, default='test_metamaterial_output',
                       help='输出目录')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'simple', 'complex', 'layered', 'fractal', 'curved'],
                       help='测试类型')
    parser.add_argument('--image', type=str, default=None,
                       help='使用自定义图像路径（跳过生成）')
    parser.add_argument('--use-pregen', action='store_true',
                       help='使用预生成的图像（从test_images目录）')
    parser.add_argument('--images-dir', type=str, default='test_images',
                      help='预生成图像目录')
    parser.add_argument('--validation-dataset', type=str, default=None,
                      help='验证集目录（包含images和gt子目录）')
    parser.add_argument('--generate-validation', type=int, default=None,
                      help='生成验证集并测试（指定样本数量）')
    parser.add_argument('--save-json', action='store_true',
                      help='保存test_report.json')
    
    args = parser.parse_args()
    
    # 如果指定生成验证集
    if args.generate_validation:
        validation_dir = os.path.join(args.output, "validation_dataset")
        print(f"\n生成验证集（{args.generate_validation}个样本）...")
        generate_validation_dataset(
            dataset_size=args.generate_validation,
            output_dir=validation_dir,
            img_size=512
        )
        args.validation_dataset = validation_dir
        print(f"\n验证集已生成，开始测试...")
    
    # 创建测试器
    tester = MetamaterialAntennaTester(output_dir=args.output, save_json_report=args.save_json)
    
    if args.image:
        # 使用自定义图像
        test_image = cv2.imread(args.image)
        if test_image is None:
            print(f"错误: 无法读取图像 {args.image}")
            return
        
        result = tester.test_image_processing(test_image, "custom_image")
        print("\n测试完成!")
    elif args.validation_dataset:
        # 使用验证集进行测试
        print(f"\n使用验证集进行测试: {args.validation_dataset}")
        
        # 加载验证集摘要
        summary_path = os.path.join(args.validation_dataset, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # 创建测试用例（从验证集摘要）
            test_cases = []
            for item in summary:
                test_cases.append({
                    'name': item['name'],
                    'type': 'validation_sample',
                    'image_path': item.get('image_path', 
                                          os.path.join(args.validation_dataset, 'images', f"{item['name']}.png")),
                    'gt_path': item.get('gt_path',
                                       os.path.join(args.validation_dataset, 'gt', f"{item['name']}.json")),
                    'complexity_level': item.get('complexity_level', -1),
                    'structure_tags': item.get('structure_tags', []),
                })
            
            print(f"  加载了 {len(test_cases)} 个验证集样本")
            
            # 根据复杂度等级筛选（可选）
            if args.test != 'all':
                if args.test == 'simple':
                    test_cases = [tc for tc in test_cases if tc.get('complexity_level', 5) <= 1]
                elif args.test == 'complex':
                    test_cases = [tc for tc in test_cases if tc.get('complexity_level', 0) >= 3]
                elif args.test == 'layered':
                    test_cases = [tc for tc in test_cases if 'multi_color' in tc.get('structure_tags', [])]
                elif args.test == 'fractal':
                    test_cases = [tc for tc in test_cases if 'fractal' in str(tc.get('structure_tags', []))]
                elif args.test == 'curved':
                    test_cases = [tc for tc in test_cases if 'freeform' in str(tc.get('structure_tags', []))]
            
            # 运行测试
            results = tester.run_test_suite(
                test_cases,
                use_pre_generated=True,
                images_dir=os.path.join(args.validation_dataset, "images"),
                validation_dataset_dir=args.validation_dataset
            )
        else:
            print(f"错误: 未找到验证集摘要文件 {summary_path}")
            print("请先运行验证集生成器或提供正确的验证集目录")
            return
        
        print("\n所有测试完成!")
    else:
        # 使用默认测试用例
        test_cases = create_default_test_cases()
        
        # 根据测试类型筛选
        if args.test == 'simple':
            test_cases = [tc for tc in test_cases if tc['name'] in 
                         ['simple_rect_patch', 'nested_rings']]
        elif args.test == 'complex':
            test_cases = [tc for tc in test_cases if 'complex' in tc['name'] or 
                         'fss' in tc['name'] or 'hybrid' in tc['name']]
        elif args.test == 'layered':
            test_cases = [tc for tc in test_cases if 'layered' in tc['name'] or 
                         'multi_color' in tc['name']]
        elif args.test == 'fractal':
            test_cases = [tc for tc in test_cases if 'fractal' in tc['name'] or 
                         'koch' in tc['name'] or 'sierpinski' in tc['name']]
        elif args.test == 'curved':
            test_cases = [tc for tc in test_cases if 'curved' in tc['name'] or 
                         'waveguide' in tc['name'] or 'curve_line' in tc['name']]
        
        # 运行测试
        results = tester.run_test_suite(test_cases, 
                                       use_pre_generated=args.use_pregen,
                                       images_dir=args.images_dir)
        print("\n所有测试完成!")


if __name__ == '__main__':
    main()
