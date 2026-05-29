"""
从GT JSON参数生成真值边缘
根据validation_generator.py中的几何参数重新生成边缘轮廓
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from math import cos, sin, pi


class GTEdgeGenerator:
    """从GT JSON参数生成真值边缘"""
    
    def __init__(self, img_size: int = 512):
        """
        初始化GT边缘生成器
        
        参数:
            img_size: 图像尺寸
        """
        self.img_size = img_size
    
    def generate_gt_edges_from_json(self, gt_json: Dict) -> np.ndarray:
        """
        从GT JSON生成真值边缘图像
        
        参数:
            gt_json: GT JSON字典
        
        返回:
            真值边缘图像（二值图，边缘为255，背景为0）
        """
        # 创建空白图像
        gt_edges = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 获取几何信息
        geometry = gt_json.get('geometry', {})
        geometry_type = geometry.get('type', '')
        
        # 根据类型生成边缘
        if geometry_type == 'rect':
            self._generate_rect_edges(gt_edges, geometry)
        elif geometry_type == 'circle':
            self._generate_circle_edges(gt_edges, geometry)
        elif geometry_type == 'polygon':
            self._generate_polygon_edges(gt_edges, geometry)
        elif geometry_type == 'ring_hole':
            self._generate_ring_hole_edges(gt_edges, geometry)
        elif geometry_type == 'nested_rings':
            self._generate_nested_rings_edges(gt_edges, geometry)
        elif geometry_type == 'srr':
            self._generate_srr_edges(gt_edges, geometry)
        elif geometry_type == 'csrr':
            self._generate_csrr_edges(gt_edges, geometry)
        elif geometry_type == 'patch_slot':
            self._generate_patch_slot_edges(gt_edges, geometry)
        elif geometry_type == 'rect_patch':
            self._generate_rect_patch_edges(gt_edges, geometry)
        elif geometry_type == 'rect_patch_with_slot':
            # rect_patch_with_slot类型
            self._generate_rect_patch_edges(gt_edges, geometry)
        elif geometry_type == 'fractal_slot_patch':
            # 分形槽贴片，使用rect_patch方法
            self._generate_rect_patch_edges(gt_edges, geometry)
        elif geometry_type == 'sierpinski_carpet':
            # 谢尔宾斯基地毯，无法精确重建，跳过
            pass
        elif geometry_type == 'unit_array':
            # 单元阵列，无法精确重建，跳过
            pass
        elif geometry_type == 'freeform':
            # 自由曲线无法从参数精确重建，返回空
            pass
        elif geometry_type == 'freeform_with_hole':
            # 自由曲线无法从参数精确重建，但可以生成孔洞边缘
            # 注意：外边界无法精确重建，只能生成孔洞
            self._generate_hole_edges(gt_edges, geometry)
        elif geometry_type == 'multi_freeform':
            # 多个自由曲线，无法精确重建
            pass
        elif geometry_type == 'multi_ring_srr':
            self._generate_multi_ring_srr_edges(gt_edges, geometry)
        elif geometry_type == 'level5_composite':
            # 复杂组合结构，递归处理各组件
            self._generate_composite_edges(gt_edges, geometry)
        else:
            # 未知类型，尝试通用方法
            pass
        
        return gt_edges
    
    def _generate_rect_edges(self, img: np.ndarray, geometry: Dict):
        """生成矩形边缘"""
        w = geometry.get('w', 200)
        h = geometry.get('h', 150)
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        pts = np.array([
            [cx - w//2, cy - h//2],
            [cx + w//2, cy - h//2],
            [cx + w//2, cy + h//2],
            [cx - w//2, cy + h//2]
        ], dtype=np.int32)
        
        cv2.polylines(img, [pts], True, 255, 1)
    
    def _generate_circle_edges(self, img: np.ndarray, geometry: Dict):
        """生成圆形边缘"""
        r = geometry.get('r', 100)
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        cv2.circle(img, (cx, cy), int(r), 255, 1)
    
    def _generate_polygon_edges(self, img: np.ndarray, geometry: Dict):
        """生成多边形边缘"""
        num_sides = geometry.get('num_sides', geometry.get('n_sides', 6))
        radius = geometry.get('radius', geometry.get('r', 100))
        rotation = geometry.get('rotation', 0)
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        points = []
        for i in range(num_sides):
            angle = np.radians(rotation + i * 360 / num_sides)
            x = cx + radius * cos(angle)
            y = cy + radius * sin(angle)
            points.append([int(x), int(y)])
        
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(img, [pts], True, 255, 1)
    
    def _generate_ring_hole_edges(self, img: np.ndarray, geometry: Dict):
        """生成环形（带孔）边缘"""
        outer_r = geometry.get('outer_r', 150)
        inner_r = geometry.get('inner_r', 75)
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        # 外圆边缘
        cv2.circle(img, (cx, cy), int(outer_r), 255, 1)
        # 内圆边缘（孔洞）
        cv2.circle(img, (cx, cy), int(inner_r), 255, 1)
    
    def _generate_nested_rings_edges(self, img: np.ndarray, geometry: Dict):
        """生成嵌套环边缘"""
        radii = geometry.get('radii', [150, 100, 50])
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        # 交替绘制外圆和内圆
        for i, r in enumerate(radii):
            cv2.circle(img, (cx, cy), int(r), 255, 1)
    
    def _generate_nested_rings_edges(self, img: np.ndarray, geometry: Dict):
        """生成嵌套环边缘"""
        radii = geometry.get('radii', [150, 100, 50])
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        # 交替绘制外圆和内圆
        for i, r in enumerate(radii):
            cv2.circle(img, (cx, cy), int(r), 255, 1)
    
    def _generate_srr_edges(self, img: np.ndarray, geometry: Dict):
        """生成SRR边缘"""
        outer_r = geometry.get('outer_r', 150)
        width = geometry.get('width', 25)
        gap_angle = geometry.get('gap_angle', 30)
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        inner_r = outer_r - width
        angles = np.linspace(gap_angle/2, 360 - gap_angle/2, 200)
        
        # 外圆边缘
        outer_points = []
        for a in angles:
            angle_rad = np.radians(a)
            x = cx + outer_r * cos(angle_rad)
            y = cy + outer_r * sin(angle_rad)
            outer_points.append([int(x), int(y)])
        
        # 内圆边缘
        inner_points = []
        for a in reversed(angles):
            angle_rad = np.radians(a)
            x = cx + inner_r * cos(angle_rad)
            y = cy + inner_r * sin(angle_rad)
            inner_points.append([int(x), int(y)])
        
        # 连接内外边缘形成闭合轮廓
        ring_points = outer_points + inner_points
        pts = np.array(ring_points, dtype=np.int32)
        cv2.polylines(img, [pts], True, 255, 1)
    
    def _generate_csrr_edges(self, img: np.ndarray, geometry: Dict):
        """生成CSRR边缘"""
        outer_r = geometry.get('outer_r', 150)
        width = geometry.get('width', 25)
        gap_angle = geometry.get('gap_angle', 30)
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        inner_r = outer_r - width
        
        # 外圆边缘（带缺口）
        angles = np.linspace(gap_angle/2, 360 - gap_angle/2, 200)
        outer_points = []
        for a in angles:
            angle_rad = np.radians(a)
            x = cx + outer_r * cos(angle_rad)
            y = cy + outer_r * sin(angle_rad)
            outer_points.append([int(x), int(y)])
        pts_outer = np.array(outer_points, dtype=np.int32)
        cv2.polylines(img, [pts_outer], False, 255, 1)
        
        # 内圆边缘（带缺口）
        inner_points = []
        for a in reversed(angles):
            angle_rad = np.radians(a)
            x = cx + inner_r * cos(angle_rad)
            y = cy + inner_r * sin(angle_rad)
            inner_points.append([int(x), int(y)])
        pts_inner = np.array(inner_points, dtype=np.int32)
        cv2.polylines(img, [pts_inner], False, 255, 1)
    
    def _generate_patch_slot_edges(self, img: np.ndarray, geometry: Dict):
        """生成贴片+槽边缘"""
        w = geometry.get('w', 250)
        h = geometry.get('h', 180)
        slot_angle = geometry.get('slot_angle', 0)
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        # 矩形贴片边缘
        rect_pts = np.array([
            [cx - w//2, cy - h//2],
            [cx + w//2, cy - h//2],
            [cx + w//2, cy + h//2],
            [cx - w//2, cy + h//2]
        ], dtype=np.int32)
        cv2.polylines(img, [rect_pts], True, 255, 1)
        
        # 槽边缘（如果有槽信息）
        if 'slot' in geometry:
            slot = geometry['slot']
            sw = slot.get('w', w * 0.3)
            sh = slot.get('h', h * 0.2)
            angle = slot.get('angle', slot_angle)
            
            angle_rad = np.radians(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            slot_pts = []
            for dx, dy in [(-sw//2, -sh//2), (sw//2, -sh//2), 
                          (sw//2, sh//2), (-sw//2, sh//2)]:
                x = cx + dx * cos_a - dy * sin_a
                y = cy + dy * cos_a + dx * sin_a
                slot_pts.append([int(x), int(y)])
            
            slot_array = np.array(slot_pts, dtype=np.int32)
            cv2.polylines(img, [slot_array], True, 255, 1)
    
    def _generate_rect_patch_edges(self, img: np.ndarray, geometry: Dict):
        """生成矩形贴片边缘"""
        w = geometry.get('w', 250)
        h = geometry.get('h', 180)
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        pts = np.array([
            [cx - w//2, cy - h//2],
            [cx + w//2, cy - h//2],
            [cx + w//2, cy + h//2],
            [cx - w//2, cy + h//2]
        ], dtype=np.int32)
        
        cv2.polylines(img, [pts], True, 255, 1)
        
        # 如果有槽，添加槽边缘
        if 'slot' in geometry:
            self._generate_slot_edges(img, geometry['slot'], cx, cy)
    
    def _generate_slot_edges(self, img: np.ndarray, slot: Dict, cx: int, cy: int):
        """生成槽边缘"""
        sw = slot.get('w', 50)
        sh = slot.get('h', 20)
        angle = slot.get('angle', 0)
        
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        slot_pts = []
        for dx, dy in [(-sw//2, -sh//2), (sw//2, -sh//2), 
                      (sw//2, sh//2), (-sw//2, sh//2)]:
            x = cx + dx * cos_a - dy * sin_a
            y = cy + dy * cos_a + dx * sin_a
            slot_pts.append([int(x), int(y)])
        
        slot_array = np.array(slot_pts, dtype=np.int32)
        cv2.polylines(img, [slot_array], True, 255, 1)
    
    def _generate_hole_edges(self, img: np.ndarray, geometry: Dict):
        """生成孔洞边缘"""
        hole_r = geometry.get('hole_r', 50)
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        cv2.circle(img, (cx, cy), int(hole_r), 255, 1)
    
    def _generate_multi_ring_srr_edges(self, img: np.ndarray, geometry: Dict):
        """生成多环SRR边缘"""
        rings = geometry.get('rings', [])
        gap_angle = geometry.get('gap_angle', 30)
        # 默认中心为图像中心
        center = geometry.get('center', [self.img_size // 2, self.img_size // 2])
        cx, cy = int(center[0]), int(center[1])
        
        for ring_info in rings:
            r = ring_info.get('r', 100)
            width = ring_info.get('width', 20)
            outer_r = r
            inner_r = r - width
            
            angles = np.linspace(gap_angle/2, 360 - gap_angle/2, 200)
            
            # 外圆边缘
            outer_points = []
            for a in angles:
                angle_rad = np.radians(a)
                x = cx + outer_r * cos(angle_rad)
                y = cy + outer_r * sin(angle_rad)
                outer_points.append([int(x), int(y)])
            
            # 内圆边缘
            inner_points = []
            for a in reversed(angles):
                angle_rad = np.radians(a)
                x = cx + inner_r * cos(angle_rad)
                y = cy + inner_r * sin(angle_rad)
                inner_points.append([int(x), int(y)])
            
            ring_points = outer_points + inner_points
            pts = np.array(ring_points, dtype=np.int32)
            cv2.polylines(img, [pts], True, 255, 1)
    
    def _generate_composite_edges(self, img: np.ndarray, geometry: Dict):
        """生成组合结构边缘"""
        components = geometry.get('components', [])
        
        for comp in components:
            comp_type = comp.get('type', '')
            center = comp.get('center', [self.img_size // 2, self.img_size // 2])
            cx, cy = int(center[0]), int(center[1])
            
            if comp_type == 'rect':
                w = comp.get('w', 200)
                h = comp.get('h', 150)
                pts = np.array([
                    [cx - w//2, cy - h//2],
                    [cx + w//2, cy - h//2],
                    [cx + w//2, cy + h//2],
                    [cx - w//2, cy + h//2]
                ], dtype=np.int32)
                cv2.polylines(img, [pts], True, 255, 1)
            
            elif comp_type == 'ring':
                outer_r = comp.get('outer_r', 100)
                inner_r = comp.get('inner_r', 50)
                cv2.circle(img, (cx, cy), int(outer_r), 255, 1)
                cv2.circle(img, (cx, cy), int(inner_r), 255, 1)
            
            elif comp_type == 'srr':
                outer_r = comp.get('outer_r', 100)
                width = comp.get('width', 20)
                gap_angle = comp.get('gap_angle', 30)
                inner_r = outer_r - width
                
                angles = np.linspace(gap_angle/2, 360 - gap_angle/2, 200)
                outer_points = []
                for a in angles:
                    angle_rad = np.radians(a)
                    x = cx + outer_r * cos(angle_rad)
                    y = cy + outer_r * sin(angle_rad)
                    outer_points.append([int(x), int(y)])
                
                inner_points = []
                for a in reversed(angles):
                    angle_rad = np.radians(a)
                    x = cx + inner_r * cos(angle_rad)
                    y = cy + inner_r * sin(angle_rad)
                    inner_points.append([int(x), int(y)])
                
                ring_points = outer_points + inner_points
                pts = np.array(ring_points, dtype=np.int32)
                cv2.polylines(img, [pts], True, 255, 1)
            
            elif comp_type == 'polygon':
                num_sides = comp.get('num_sides', 6)
                radius = comp.get('radius', 100)
                rotation = comp.get('rotation', 0)
                
                points = []
                for i in range(num_sides):
                    angle = np.radians(rotation + i * 360 / num_sides)
                    x = cx + radius * cos(angle)
                    y = cy + radius * sin(angle)
                    points.append([int(x), int(y)])
                
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(img, [pts], True, 255, 1)
            
            elif comp_type == 'connector':
                # 连接线
                from_pt = comp.get('from', [cx, cy])
                to_pt = comp.get('to', [cx, cy])
                cv2.line(img, tuple(from_pt), tuple(to_pt), 255, 1)
            
            # 其他类型（freeform等）无法精确重建，跳过


def generate_gt_edges_from_json_file(gt_json_path: str, img_size: int = 512) -> np.ndarray:
    """
    从GT JSON文件生成真值边缘（便捷函数）
    
    参数:
        gt_json_path: GT JSON文件路径
        img_size: 图像尺寸
    
    返回:
        真值边缘图像
    """
    import json
    
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_json = json.load(f)
    
    generator = GTEdgeGenerator(img_size=img_size)
    return generator.generate_gt_edges_from_json(gt_json)
