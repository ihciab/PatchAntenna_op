"""
验证集专用超材料/天线数据集生成器 V3
专门用于验证PNG → 几何反演/结构识别/CAD重建pipeline

原代码位置：old_code/Rebuild/validation_dataset_generator.py
迁移位置：tools/dataset_generator/validation_generator.py

设计原则：
1. 复杂度可控、可解释
2. 覆盖pipeline的每一个关键假设
3. 能回答"什么时候开始失败"
4. 不是训练集，是验证集

迁移说明：
- 本文件从 old_code/Rebuild/validation_dataset_generator.py 迁移而来
- 功能保持不变，仅更新了文件位置
- 用于生成验证集数据，支持复杂度分级（Level 0-5）
- 支持多种结构类型：矩形、圆形、SRR、CSRR、分形、自由曲线等
- 支持多种颜色模型和背景类型，模拟真实PCB/天线/超材料场景
"""

import os
import json
import random
import numpy as np
import cv2
from math import cos, sin, pi, sqrt
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

try:
    from scipy.interpolate import splprep, splev
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy未安装，自由曲线功能将受限")


class ValidationDatasetGenerator:
    """验证集专用生成器"""
    
    def __init__(self, img_size: int = 512):
        """
        初始化生成器
        
        参数:
            img_size: 图像尺寸（正方形）
        """
        self.img_size = img_size
        self.margin = img_size // 10
        
        # 复杂度分布（验证集专用，增加高复杂度样本）
        self.complexity_distribution = {
            0: 0.08,  # Level 0: 极简单（Sanity check）
            1: 0.17,  # Level 1: 简单（Base）
            2: 0.20,  # Level 2: 中等（Topology）
            3: 0.20,  # Level 3: 复杂（Geometry）
            4: 0.20,  # Level 4: 极端（Failure cases）
            5: 0.15,  # Level 5: 超复杂（组合+极端，新增）
        }
    
    def random_metal_color(self) -> Tuple[int, int, int]:
        """
        生成随机金属颜色（考虑真实PCB/天线/超材料/微波电路场景）
        
        返回:
            BGR颜色元组
        """
        # 真实场景金属颜色类型（扩展）
        color_type = random.choice([
            'copper_fresh',      # 新鲜铜（红棕色）
            'copper_oxidized',   # 氧化铜（深棕色）
            'gold_plating',      # 镀金（金黄色）
            'gold_old',          # 旧金（暗金色）
            'silver',            # 银（银灰色）
            'aluminum',          # 铝（银白色）
            'brass',             # 黄铜（黄棕色）
            'tin_lead',          # 锡铅合金（灰白色）
            'nickel',            # 镍（银灰色带蓝）
            'solder_mask_green', # 阻焊层绿色（PCB常见）
            'solder_mask_blue',  # 阻焊层蓝色
            'solder_mask_red',   # 阻焊层红色
            'solder_mask_black', # 阻焊层黑色
            'solder_mask_white', # 阻焊层白色
        ])
        
        if color_type == 'copper_fresh':
            # 新鲜铜：红棕色，高饱和
            h = random.uniform(8, 25) / 180.0
            s = random.uniform(0.5, 0.8)
            v = random.uniform(0.6, 0.85)
        elif color_type == 'copper_oxidized':
            # 氧化铜：深棕色，低亮度
            h = random.uniform(10, 20) / 180.0
            s = random.uniform(0.6, 0.9)
            v = random.uniform(0.3, 0.5)
        elif color_type == 'gold_plating':
            # 镀金：亮金黄色
            h = random.uniform(18, 35) / 180.0
            s = random.uniform(0.6, 0.9)
            v = random.uniform(0.75, 0.95)
        elif color_type == 'gold_old':
            # 旧金：暗金色
            h = random.uniform(20, 35) / 180.0
            s = random.uniform(0.5, 0.7)
            v = random.uniform(0.5, 0.7)
        elif color_type == 'silver':
            # 银：低饱和，高亮度
            h = random.uniform(0, 180) / 180.0
            s = random.uniform(0.05, 0.25)
            v = random.uniform(0.8, 0.95)
        elif color_type == 'aluminum':
            # 铝：银白色，略带蓝
            h = random.uniform(100, 130) / 180.0  # 蓝色区域
            s = random.uniform(0.1, 0.3)
            v = random.uniform(0.75, 0.9)
        elif color_type == 'brass':
            # 黄铜：黄棕色
            h = random.uniform(25, 45) / 180.0
            s = random.uniform(0.4, 0.7)
            v = random.uniform(0.65, 0.85)
        elif color_type == 'tin_lead':
            # 锡铅合金：灰白色
            h = random.uniform(0, 180) / 180.0
            s = random.uniform(0.05, 0.2)
            v = random.uniform(0.7, 0.85)
        elif color_type == 'nickel':
            # 镍：银灰色带蓝
            h = random.uniform(150, 180) / 180.0  # 蓝紫色区域
            s = random.uniform(0.15, 0.35)
            v = random.uniform(0.6, 0.8)
        elif color_type == 'solder_mask_green':
            # PCB阻焊层绿色（最常见）
            h = random.uniform(50, 80) / 180.0  # 绿色
            s = random.uniform(0.4, 0.7)
            v = random.uniform(0.3, 0.6)
        elif color_type == 'solder_mask_blue':
            # PCB阻焊层蓝色
            h = random.uniform(100, 130) / 180.0  # 蓝色
            s = random.uniform(0.5, 0.8)
            v = random.uniform(0.4, 0.7)
        elif color_type == 'solder_mask_red':
            # PCB阻焊层红色
            h = random.uniform(0, 10) / 180.0  # 红色
            s = random.uniform(0.6, 0.9)
            v = random.uniform(0.4, 0.7)
        elif color_type == 'solder_mask_black':
            # PCB阻焊层黑色
            h = random.uniform(0, 180) / 180.0
            s = random.uniform(0.0, 0.2)
            v = random.uniform(0.1, 0.3)
        else:  # solder_mask_white
            # PCB阻焊层白色
            h = random.uniform(0, 180) / 180.0
            s = random.uniform(0.0, 0.1)
            v = random.uniform(0.85, 0.95)
        
        hsv = np.uint8([[[int(h*180), int(s*255), int(v*255)]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(bgr.tolist())
    
    def random_multi_metal_colors(self, num_colors: int = 3) -> List[Tuple[int, int, int]]:
        """
        生成多个不同的金属颜色（用于多组件结构）
        
        参数:
            num_colors: 需要的颜色数量
        
        返回:
            颜色列表
        """
        colors = []
        for _ in range(num_colors):
            colors.append(self.random_metal_color())
        return colors
    
    def add_color_perturbation(self, img: np.ndarray, mask: np.ndarray, 
                              base_color: Tuple[int, int, int],
                              perturbation_strength: float = 0.1) -> np.ndarray:
        """
        在同一layer内添加低频颜色扰动（模拟电镀不均/光照）
        
        参数:
            img: 图像
            mask: 需要添加扰动的区域
            base_color: 基础颜色
            perturbation_strength: 扰动强度 (0-1)
        
        返回:
            修改后的图像
        """
        if not np.any(mask):
            return img
        
        # 生成低频噪声（Perlin-like）
        noise = np.random.normal(0, perturbation_strength * 30, 
                                (self.img_size, self.img_size, 3))
        noise = cv2.GaussianBlur(noise, (51, 51), 0)
        
        # 应用到mask区域（先转float32避免溢出）
        img_float = img.astype(np.float32)
        img_float[mask] = img_float[mask] + noise[mask]
        
        # Clip并转换回uint8
        img = np.clip(img_float, 0, 255).astype(np.uint8)
        
        return img
    
    def substrate_background(self) -> np.ndarray:
        """
        生成真实介质背景（扩展模式，考虑PCB/微波电路场景）
        
        返回:
            背景图像
        """
        bg_type = random.choice([
            'uniform',      # 均匀介质板
            'texture',      # 轻纹理PCB
            'scan',         # 扫描灰度背景
            'low_contrast', # 低对比噪声
            'fr4_green',    # FR4绿色基板（PCB最常见）
            'fr4_brown',    # FR4棕色基板
            'ceramic_white', # 陶瓷白色基板（微波电路）
            'fiber_texture', # 纤维纹理（PCB基板）
            'aged_pcb',     # 老化PCB（偏黄）
        ])
        
        if bg_type == 'uniform':
            # 均匀介质板（FR4）
            base = random.randint(200, 245)
            bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * base
        
        elif bg_type == 'texture':
            # 轻纹理PCB（纤维/噪声）
            base = random.randint(220, 240)
            bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * base
            # 低频纹理（先转float避免溢出）
            noise = np.random.normal(0, random.uniform(2, 6), 
                                   (self.img_size, self.img_size))
            noise = cv2.GaussianBlur(noise, (51, 51), 0)
            bg_float = bg.astype(np.float32)
            for c in range(3):
                bg_float[:, :, c] = bg_float[:, :, c] + noise
            bg = np.clip(bg_float, 0, 255).astype(np.uint8)
        
        elif bg_type == 'scan':
            # 扫描灰度背景
            base = random.randint(180, 220)
            bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * base
            # 扫描线效果（先转int避免溢出）
            bg_int = bg.astype(np.int16)
            for i in range(0, self.img_size, 5):
                offset = random.randint(-3, 3)
                bg_int[i:i+2, :, :] = bg_int[i:i+2, :, :] + offset
            bg = np.clip(bg_int, 0, 255).astype(np.uint8)
        
        elif bg_type == 'low_contrast':
            # 低对比噪声背景（挑战二值化）
            base = random.randint(200, 230)
            bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * base
            noise = np.random.normal(0, random.uniform(5, 10), 
                                   (self.img_size, self.img_size, 3))
            # 先转float避免溢出
            bg_float = bg.astype(np.float32) + noise
            bg = np.clip(bg_float, 0, 255).astype(np.uint8)
        
        elif bg_type == 'fr4_green':
            # FR4绿色基板（PCB最常见）
            # 绿色阻焊层下的基板颜色
            base_h = random.uniform(50, 80) / 180.0  # 绿色
            base_s = random.uniform(0.2, 0.4)
            base_v = random.uniform(0.3, 0.5)
            hsv_bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8)
            hsv_bg[:, :, 0] = int(base_h * 180)
            hsv_bg[:, :, 1] = int(base_s * 255)
            hsv_bg[:, :, 2] = int(base_v * 255)
            bg = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
            # 添加轻微纹理
            noise = np.random.normal(0, random.uniform(3, 8), (self.img_size, self.img_size, 3))
            noise = cv2.GaussianBlur(noise, (31, 31), 0)
            bg_float = bg.astype(np.float32) + noise
            bg = np.clip(bg_float, 0, 255).astype(np.uint8)
        
        elif bg_type == 'fr4_brown':
            # FR4棕色基板
            base_h = random.uniform(15, 35) / 180.0  # 棕色
            base_s = random.uniform(0.3, 0.5)
            base_v = random.uniform(0.4, 0.6)
            hsv_bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8)
            hsv_bg[:, :, 0] = int(base_h * 180)
            hsv_bg[:, :, 1] = int(base_s * 255)
            hsv_bg[:, :, 2] = int(base_v * 255)
            bg = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
            # 添加轻微纹理
            noise = np.random.normal(0, random.uniform(3, 8), (self.img_size, self.img_size, 3))
            noise = cv2.GaussianBlur(noise, (31, 31), 0)
            bg_float = bg.astype(np.float32) + noise
            bg = np.clip(bg_float, 0, 255).astype(np.uint8)
        
        elif bg_type == 'ceramic_white':
            # 陶瓷白色基板（微波电路常用）
            base = random.randint(230, 250)
            bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * base
            # 添加轻微纹理（模拟陶瓷表面）
            noise = np.random.normal(0, random.uniform(2, 5), (self.img_size, self.img_size, 3))
            noise = cv2.GaussianBlur(noise, (21, 21), 0)
            bg_float = bg.astype(np.float32) + noise
            bg = np.clip(bg_float, 0, 255).astype(np.uint8)
        
        elif bg_type == 'fiber_texture':
            # 纤维纹理（PCB基板纤维结构）
            base = random.randint(220, 240)
            bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * base
            # 生成纤维纹理（方向性噪声）
            for _ in range(random.randint(3, 6)):
                angle = random.uniform(0, 180)
                length = random.randint(50, 150)
                thickness = random.randint(1, 3)
                x1 = random.randint(0, self.img_size)
                y1 = random.randint(0, self.img_size)
                x2 = int(x1 + length * np.cos(np.radians(angle)))
                y2 = int(y1 + length * np.sin(np.radians(angle)))
                # 绘制纤维线
                cv2.line(bg, (x1, y1), (x2, y2), 
                        (base + random.randint(-10, 10),) * 3, thickness)
            # 添加整体噪声
            noise = np.random.normal(0, random.uniform(2, 5), (self.img_size, self.img_size, 3))
            noise = cv2.GaussianBlur(noise, (21, 21), 0)
            bg_float = bg.astype(np.float32) + noise
            bg = np.clip(bg_float, 0, 255).astype(np.uint8)
        
        else:  # aged_pcb
            # 老化PCB（偏黄，氧化效果）
            base_h = random.uniform(20, 40) / 180.0  # 黄色
            base_s = random.uniform(0.15, 0.3)
            base_v = random.uniform(0.5, 0.7)
            hsv_bg = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8)
            hsv_bg[:, :, 0] = int(base_h * 180)
            hsv_bg[:, :, 1] = int(base_s * 255)
            hsv_bg[:, :, 2] = int(base_v * 255)
            bg = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
            # 添加老化纹理（不均匀）
            noise = np.random.normal(0, random.uniform(5, 12), (self.img_size, self.img_size, 3))
            noise = cv2.GaussianBlur(noise, (41, 41), 0)
            bg_float = bg.astype(np.float32) + noise
            bg = np.clip(bg_float, 0, 255).astype(np.uint8)
        
        return bg
    
    def fill_polygon(self, img: np.ndarray, pts: List[Tuple[int, int]], 
                    color: Tuple[int, int, int]):
        """填充多边形"""
        pts_array = np.array(pts, dtype=np.int32)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts_array], 255)
        img[mask == 255] = color
    
    def random_position(self, size: float) -> Tuple[int, int]:
        """
        生成随机位置（考虑边距和结构尺寸）
        
        参数:
            size: 结构的大致尺寸
        
        返回:
            (cx, cy) 随机中心位置
        """
        margin = max(self.margin, int(size * 0.6))
        cx = random.randint(margin, self.img_size - margin)
        cy = random.randint(margin, self.img_size - margin)
        return (cx, cy)
    
    def apply_manufacturing_noise(self, img: np.ndarray, noise_level: float = 0.3) -> np.ndarray:
        """
        应用制造级噪声（融合V2功能）
        
        参数:
            img: 输入图像（单通道mask）
            noise_level: 噪声强度 (0-1)
        
        返回:
            添加噪声后的图像
        """
        result = img.copy()
        
        # 1. 边界扰动（亚像素jitter）
        if noise_level > 0.1:
            kernel_size = random.choice([3, 5])
            if random.random() < 0.5:
                # 轻微腐蚀（模拟蚀刻误差）
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                result = cv2.erode(result, kernel, iterations=1)
            else:
                # 轻微膨胀（模拟沉积误差）
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                result = cv2.dilate(result, kernel, iterations=1)
        
        # 2. 局部缺陷（随机小孔/断裂）
        if noise_level > 0.2 and random.random() < 0.3:
            num_defects = random.randint(1, 3)
            for _ in range(num_defects):
                defect_x = random.randint(0, self.img_size - 1)
                defect_y = random.randint(0, self.img_size - 1)
                defect_r = random.randint(2, 5)
                cv2.circle(result, (defect_x, defect_y), defect_r, 0, -1)
        
        return result
    
    def freeform_patch(self, img: np.ndarray, center: Tuple[int, int], 
                      scale: float, color: Tuple[int, int, int]) -> List[Tuple[int, int]]:
        """
        生成自由曲线贴片（Bezier/Spline）
        
        参数:
            img: 图像
            center: 中心点
            scale: 缩放因子
            color: 颜色
        
        返回:
            生成的轮廓点列表
        """
        cx, cy = center
        n_ctrl = random.randint(5, 8)
        
        # 生成控制点（围绕中心）
        ctrl = []
        for i in range(n_ctrl):
            angle = 2 * pi * i / n_ctrl + random.uniform(-0.4, 0.4)
            r = scale * random.uniform(0.6, 1.0)
            ctrl.append([cx + r * cos(angle), cy + r * sin(angle)])
        ctrl.append(ctrl[0])  # 闭合
        
        if SCIPY_AVAILABLE:
            # 使用B-spline插值
            ctrl_array = np.array(ctrl).T
            try:
                tck, u = splprep(ctrl_array, s=0, per=True)
                u_new = np.linspace(0, 1, 300)
                x, y = splev(u_new, tck)
                pts = list(zip(x.astype(int), y.astype(int)))
            except:
                # 如果spline失败，使用简单多边形
                pts = [(int(p[0]), int(p[1])) for p in ctrl]
        else:
            # 使用Bezier曲线（简单实现）
            pts = self._bezier_curve(ctrl, num_points=300)
        
        self.fill_polygon(img, pts, color)
        return pts
    
    def _bezier_curve(self, control_points: List[List[float]], 
                     num_points: int = 100) -> List[Tuple[int, int]]:
        """
        生成Bezier曲线（简单实现）
        
        参数:
            control_points: 控制点列表
            num_points: 生成的点数
        
        返回:
            曲线点列表
        """
        n = len(control_points) - 1
        if n < 1:
            return [(int(p[0]), int(p[1])) for p in control_points]
        
        points = []
        for t in np.linspace(0, 1, num_points):
            x, y = 0, 0
            for i, p in enumerate(control_points):
                # Bernstein基函数
                bern = self._bernstein(n, i, t)
                x += bern * p[0]
                y += bern * p[1]
            points.append((int(x), int(y)))
        
        return points
    
    def _bernstein(self, n: int, i: int, t: float) -> float:
        """Bernstein基函数"""
        from math import comb
        try:
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        except:
            # Python < 3.8 兼容
            from math import factorial
            return (factorial(n) / (factorial(i) * factorial(n - i))) * \
                   (t ** i) * ((1 - t) ** (n - i))
    
    # ========== 基础结构生成函数（Primitives，融合V2）==========
    
    def gen_rect_patch(self, img: np.ndarray, cx: int, cy: int, 
                      w: Optional[int] = None, h: Optional[int] = None) -> Dict:
        """生成矩形贴片（基础primitive）"""
        if w is None:
            w = int(np.clip(np.random.normal(200, 50), 100, 400))
        if h is None:
            h = int(np.clip(np.random.normal(150, 40), 80, 300))
        
        pts = [
            (cx - w//2, cy - h//2),
            (cx + w//2, cy - h//2),
            (cx + w//2, cy + h//2),
            (cx - w//2, cy + h//2),
        ]
        # 单通道mask使用fillPoly
        if len(img.shape) == 2:
            pts_array = np.array(pts, dtype=np.int32)
            cv2.fillPoly(img, [pts_array], 255)
        else:
            self.fill_polygon(img, pts, 255)
        
        return {"type": "rect", "w": w, "h": h, "center": [cx, cy]}
    
    def gen_slot(self, img: np.ndarray, cx: int, cy: int, 
                length: Optional[float] = None, width: Optional[float] = None,
                angle: Optional[float] = None) -> Dict:
        """生成槽（孔洞primitive）"""
        if length is None:
            length = np.random.uniform(50, 200)
        if width is None:
            width = np.random.uniform(8, 25)
        if angle is None:
            angle = np.random.uniform(0, 360)
        
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        slot = []
        for dx, dy in [(-length/2, -width/2), (length/2, -width/2), 
                      (length/2, width/2), (-length/2, width/2)]:
            x = cx + dx * cos_a - dy * sin_a
            y = cy + dy * cos_a + dx * sin_a
            slot.append((int(x), int(y)))
        
        # 单通道mask使用fillPoly（孔洞为0）
        if len(img.shape) == 2:
            pts_array = np.array(slot, dtype=np.int32)
            cv2.fillPoly(img, [pts_array], 0)
        else:
            self.fill_polygon(img, slot, 0)
        
        return {"type": "slot", "length": length, "width": width, "angle": angle, "center": [cx, cy]}
    
    def gen_ring(self, img: np.ndarray, cx: int, cy: int,
                outer_r: Optional[float] = None, inner_r: Optional[float] = None) -> Dict:
        """生成环形（基础primitive）"""
        if outer_r is None:
            outer_r = np.random.uniform(80, 200)
        if inner_r is None:
            inner_r = outer_r * np.random.uniform(0.3, 0.7)
        
        # 单通道mask
        if len(img.shape) == 2:
            cv2.circle(img, (cx, cy), int(outer_r), 255, -1)
            cv2.circle(img, (cx, cy), int(inner_r), 0, -1)
        else:
            cv2.circle(img, (cx, cy), int(outer_r), (255, 255, 255), -1)
            cv2.circle(img, (cx, cy), int(inner_r), (0, 0, 0), -1)
        
        return {"type": "ring", "outer_r": outer_r, "inner_r": inner_r, "center": [cx, cy]}
    
    def gen_srr(self, img: np.ndarray, cx: int, cy: int,
               outer_r: Optional[float] = None, width: Optional[float] = None,
               gap_angle: Optional[float] = None) -> Dict:
        """生成SRR（基础primitive）"""
        if outer_r is None:
            outer_r = np.random.uniform(100, 180)
        if width is None:
            width = np.random.uniform(15, 35)
        if gap_angle is None:
            gap_angle = np.random.uniform(20, 50)
        
        angles = np.linspace(gap_angle/2, 360 - gap_angle/2, 200)
        outer_points = []
        for a in angles:
            angle_rad = np.radians(a)
            x = cx + outer_r * cos(angle_rad)
            y = cy + outer_r * sin(angle_rad)
            outer_points.append((int(x), int(y)))
        
        inner_r = outer_r - width
        inner_points = []
        for a in reversed(angles):
            angle_rad = np.radians(a)
            x = cx + inner_r * cos(angle_rad)
            y = cy + inner_r * sin(angle_rad)
            inner_points.append((int(x), int(y)))
        
        ring_points = outer_points + inner_points
        # 单通道mask使用fillPoly
        if len(img.shape) == 2:
            pts_array = np.array(ring_points, dtype=np.int32)
            cv2.fillPoly(img, [pts_array], 255)
        else:
            self.fill_polygon(img, ring_points, 255)
        
        return {"type": "srr", "outer_r": outer_r, "width": width, "gap_angle": gap_angle, "center": [cx, cy]}
    
    def gen_polygon(self, img: np.ndarray, cx: int, cy: int,
                   num_sides: Optional[int] = None, radius: Optional[float] = None,
                   rotation: Optional[float] = None) -> Dict:
        """生成多边形（基础primitive）"""
        if num_sides is None:
            # 连续分布，非离散
            num_sides = int(np.clip(np.random.normal(6, 1.5), 3, 10))
        if radius is None:
            radius = np.random.uniform(80, 180)
        if rotation is None:
            rotation = np.random.uniform(0, 360)
        
        points = []
        for i in range(num_sides):
            angle = np.radians(rotation + i * 360 / num_sides)
            x = cx + radius * cos(angle)
            y = cy + radius * sin(angle)
            points.append((int(x), int(y)))
        
        # 单通道mask使用fillPoly
        if len(img.shape) == 2:
            pts_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(img, [pts_array], 255)
        else:
            self.fill_polygon(img, points, 255)
        
        return {"type": "polygon", "num_sides": num_sides, "radius": radius, 
                "rotation": rotation, "center": [cx, cy]}
    
    # ========== 扩展结构生成函数（融合realistic_dataset_generator）==========
    
    def gen_rect_patch_with_slot(self, img: np.ndarray, cx: int, cy: int) -> Dict:
        """
        生成矩形贴片+槽（真实天线结构，融合realistic_dataset_generator）
        
        返回:
            Ground Truth字典
        """
        w = random.randint(200, 360)
        h = random.randint(120, 260)
        has_slot = random.choice([True, False])
        
        # 绘制矩形贴片（填充）
        rect = [
            (cx - w//2, cy - h//2),
            (cx + w//2, cy - h//2),
            (cx + w//2, cy + h//2),
            (cx - w//2, cy + h//2),
        ]
        # 单通道mask使用fillPoly
        if len(img.shape) == 2:
            pts_array = np.array(rect, dtype=np.int32)
            cv2.fillPoly(img, [pts_array], 255)
        else:
            self.fill_polygon(img, rect, 255)
        
        gt = {
            "type": "rect_patch",
            "w": w,
            "h": h,
            "center": [cx, cy]
        }
        
        if has_slot:
            # 添加槽（孔洞）
            sw = int(w * random.uniform(0.15, 0.4))
            sh = int(h * random.uniform(0.1, 0.25))
            slot_angle = random.choice([0, 45, 90, -45])
            
            # 旋转槽
            if slot_angle == 0:
                slot = [
                    (cx - sw//2, cy - sh//2),
                    (cx + sw//2, cy - sh//2),
                    (cx + sw//2, cy + sh//2),
                    (cx - sw//2, cy + sh//2),
                ]
            else:
                # 旋转后的槽
                angle_rad = np.radians(slot_angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                slot = []
                for dx, dy in [(-sw//2, -sh//2), (sw//2, -sh//2), (sw//2, sh//2), (-sw//2, sh//2)]:
                    x = cx + dx * cos_a - dy * sin_a
                    y = cy + dy * cos_a + dx * sin_a
                    slot.append((int(x), int(y)))
            
            # 单通道mask使用fillPoly（孔洞为0）
            if len(img.shape) == 2:
                slot_array = np.array(slot, dtype=np.int32)
                cv2.fillPoly(img, [slot_array], 0)
            else:
                self.fill_polygon(img, slot, (0, 0, 0))
            
            gt["slot"] = {"w": sw, "h": sh, "angle": slot_angle}
        
        return gt
    
    def gen_csrr(self, img: np.ndarray, cx: int, cy: int) -> Dict:
        """
        生成CSRR（Complementary SRR，互补分裂环谐振器）- 真实超材料结构
        融合自realistic_dataset_generator
        
        返回:
            Ground Truth字典
        """
        outer_r = random.randint(130, 190)
        width = random.randint(18, 35)
        gap_angle = random.randint(20, 40)
        
        # 先绘制完整圆环
        if len(img.shape) == 2:
            cv2.circle(img, (cx, cy), int(outer_r), 255, -1)
            cv2.circle(img, (cx, cy), int(outer_r - width), 0, -1)
        else:
            cv2.circle(img, (cx, cy), int(outer_r), (255, 255, 255), -1)
            cv2.circle(img, (cx, cy), int(outer_r - width), (0, 0, 0), -1)
        
        # 然后添加缺口（用背景填充）
        gap_start = gap_angle / 2
        gap_end = 360 - gap_angle / 2
        
        # 在缺口处绘制扇形，用0填充形成缺口
        gap_points = [(cx, cy)]
        for a in np.linspace(gap_start, gap_end, 30):
            angle_rad = np.radians(a)
            x = cx + outer_r * cos(angle_rad)
            y = cy + outer_r * sin(angle_rad)
            gap_points.append((int(x), int(y)))
        
        if len(img.shape) == 2:
            gap_array = np.array(gap_points, dtype=np.int32)
            cv2.fillPoly(img, [gap_array], 0)
        else:
            self.fill_polygon(img, gap_points, (0, 0, 0))
        
        return {
            "type": "csrr",
            "outer_r": outer_r,
            "width": width,
            "gap_angle": gap_angle,
            "center": [cx, cy]
        }
    
    def gen_sierpinski_carpet(self, img: np.ndarray, cx: int, cy: int) -> Dict:
        """
        生成Sierpinski Carpet（谢尔宾斯基地毯）- 真实分形超材料结构
        融合自realistic_dataset_generator
        
        返回:
            Ground Truth字典
        """
        size = random.randint(240, 360)
        depth = random.choice([1, 2, 3])
        
        def recurse(x: int, y: int, s: int, d: int):
            """递归生成分形"""
            if d == 0:
                # 基础单元：填充矩形
                if len(img.shape) == 2:
                    pts_array = np.array([
                        (x, y), (x+s, y), (x+s, y+s), (x, y+s)
                    ], dtype=np.int32)
                    cv2.fillPoly(img, [pts_array], 255)
                else:
                    self.fill_polygon(img, [
                        (x, y), (x+s, y), (x+s, y+s), (x, y+s)
                    ], (255, 255, 255))
            else:
                ns = s // 3
                for i in range(3):
                    for j in range(3):
                        if i == 1 and j == 1:
                            # 中心不填充
                            continue
                        recurse(x + i*ns, y + j*ns, ns, d - 1)
        
        recurse(cx - size//2, cy - size//2, size, depth)
        
        return {
            "type": "sierpinski_carpet",
            "size": size,
            "depth": depth,
            "center": [cx, cy]
        }
    
    def gen_fractal_slot_patch(self, img: np.ndarray, cx: int, cy: int) -> Dict:
        """
        生成分形槽贴片（真实天线结构）
        融合自realistic_dataset_generator
        
        返回:
            Ground Truth字典
        """
        base_size = random.randint(200, 300)
        depth = random.choice([2, 3])
        
        def fractal_slot(x: int, y: int, w: int, h: int, d: int):
            """递归生成分形槽"""
            if d == 0:
                # 基础槽
                slot_w = int(w * 0.3)
                slot_h = int(h * 0.3)
                slot = [
                    (x + w//2 - slot_w//2, y + h//2 - slot_h//2),
                    (x + w//2 + slot_w//2, y + h//2 - slot_h//2),
                    (x + w//2 + slot_w//2, y + h//2 + slot_h//2),
                    (x + w//2 - slot_w//2, y + h//2 + slot_h//2),
                ]
                # 单通道mask使用fillPoly（孔洞为0）
                if len(img.shape) == 2:
                    slot_array = np.array(slot, dtype=np.int32)
                    cv2.fillPoly(img, [slot_array], 0)
                else:
                    self.fill_polygon(img, slot, (0, 0, 0))
            else:
                # 递归：在四个角生成更小的槽
                nw, nh = w // 2, h // 2
                for dx, dy in [(0, 0), (nw, 0), (0, nh), (nw, nh)]:
                    fractal_slot(x + dx, y + dy, nw, nh, d - 1)
        
        # 先绘制基础矩形
        base_rect = [
            (cx - base_size//2, cy - base_size//2),
            (cx + base_size//2, cy - base_size//2),
            (cx + base_size//2, cy + base_size//2),
            (cx - base_size//2, cy + base_size//2),
        ]
        if len(img.shape) == 2:
            base_array = np.array(base_rect, dtype=np.int32)
            cv2.fillPoly(img, [base_array], 255)
        else:
            self.fill_polygon(img, base_rect, (255, 255, 255))
        
        # 添加分形槽
        fractal_slot(cx - base_size//2, cy - base_size//2, base_size, base_size, depth)
        
        return {
            "type": "fractal_slot_patch",
            "size": base_size,
            "depth": depth,
            "center": [cx, cy]
        }
    
    def gen_multi_ring_srr(self, img: np.ndarray, cx: int, cy: int) -> Dict:
        """
        生成多环SRR（真实超材料结构）
        融合自realistic_dataset_generator
        
        返回:
            Ground Truth字典
        """
        num_rings = random.choice([2, 3])
        base_r = random.randint(100, 150)
        ring_spacing = random.randint(15, 25)
        gap_angle = random.randint(20, 40)
        
        rings_info = []
        for i in range(num_rings):
            r = base_r + i * ring_spacing
            width = random.randint(15, 25)
            
            # 生成环（带缺口）
            angles = np.linspace(gap_angle/2, 360 - gap_angle/2, 200)
            outer_points = []
            for a in angles:
                angle_rad = np.radians(a)
                x = cx + r * cos(angle_rad)
                y = cy + r * sin(angle_rad)
                outer_points.append((int(x), int(y)))
            
            inner_r = r - width
            inner_points = []
            for a in reversed(angles):
                angle_rad = np.radians(a)
                x = cx + inner_r * cos(angle_rad)
                y = cy + inner_r * sin(angle_rad)
                inner_points.append((int(x), int(y)))
            
            ring_points = outer_points + inner_points
            # 单通道mask使用fillPoly
            if len(img.shape) == 2:
                ring_array = np.array(ring_points, dtype=np.int32)
                cv2.fillPoly(img, [ring_array], 255)
            else:
                self.fill_polygon(img, ring_points, (255, 255, 255))
            
            rings_info.append({"r": r, "width": width})
        
        return {
            "type": "multi_ring_srr",
            "num_rings": num_rings,
            "rings": rings_info,
            "gap_angle": gap_angle,
            "center": [cx, cy]
        }
    
    def gen_unit_array(self, img: np.ndarray, cx: int, cy: int) -> Dict:
        """
        生成单元阵列（真实FSS/超材料阵列）
        融合自realistic_dataset_generator
        
        返回:
            Ground Truth字典
        """
        unit_type = random.choice(['cross', 'square_loop', 'circular_loop'])
        num_units_x = random.choice([2, 3, 4])
        num_units_y = num_units_x
        unit_size = random.randint(80, 120)
        spacing = unit_size + random.randint(20, 40)
        
        start_x = cx - (num_units_x - 1) * spacing // 2
        start_y = cy - (num_units_y - 1) * spacing // 2
        
        units = []
        for i in range(num_units_x):
            for j in range(num_units_y):
                ux = start_x + i * spacing
                uy = start_y + j * spacing
                
                if unit_type == 'cross':
                    # 十字形
                    w = unit_size // 20
                    cross_pts1 = [
                        (ux - unit_size//2, uy - w),
                        (ux + unit_size//2, uy - w),
                        (ux + unit_size//2, uy + w),
                        (ux - unit_size//2, uy + w),
                    ]
                    cross_pts2 = [
                        (ux - w, uy - unit_size//2),
                        (ux + w, uy - unit_size//2),
                        (ux + w, uy + unit_size//2),
                        (ux - w, uy + unit_size//2),
                    ]
                    if len(img.shape) == 2:
                        cv2.fillPoly(img, [np.array(cross_pts1, dtype=np.int32)], 255)
                        cv2.fillPoly(img, [np.array(cross_pts2, dtype=np.int32)], 255)
                    else:
                        self.fill_polygon(img, cross_pts1, (255, 255, 255))
                        self.fill_polygon(img, cross_pts2, (255, 255, 255))
                
                elif unit_type == 'square_loop':
                    # 方形环
                    outer = unit_size // 2
                    inner = unit_size // 3
                    if len(img.shape) == 2:
                        cv2.circle(img, (ux, uy), outer, 255, -1)
                        cv2.circle(img, (ux, uy), inner, 0, -1)
                    else:
                        cv2.circle(img, (ux, uy), outer, (255, 255, 255), -1)
                        cv2.circle(img, (ux, uy), inner, (0, 0, 0), -1)
                
                elif unit_type == 'circular_loop':
                    # 圆形环
                    outer = unit_size // 2
                    inner = unit_size // 3
                    if len(img.shape) == 2:
                        cv2.circle(img, (ux, uy), outer, 255, -1)
                        cv2.circle(img, (ux, uy), inner, 0, -1)
                    else:
                        cv2.circle(img, (ux, uy), outer, (255, 255, 255), -1)
                        cv2.circle(img, (ux, uy), inner, (0, 0, 0), -1)
                
                units.append({"x": ux, "y": uy, "type": unit_type})
        
        return {
            "type": "unit_array",
            "unit_type": unit_type,
            "num_units": num_units_x * num_units_y,
            "unit_size": unit_size,
            "spacing": spacing,
            "units": units,
            "center": [cx, cy]
        }
    
    # ========== 复杂度分级生成函数 ==========
    
    def gen_level0(self, img: np.ndarray, gt: Dict) -> Dict:
        """
        Level 0: 极简单（Sanity check）
        - 单矩形/单圆/单多边形
        - 零噪声
        """
        color = self.random_metal_color()
        shape_type = random.choice(['rect', 'circle', 'polygon'])
        cx, cy = self.img_size // 2, self.img_size // 2
        
        if shape_type == 'rect':
            w, h = random.randint(200, 300), random.randint(150, 250)
            pts = [
                (cx - w//2, cy - h//2),
                (cx + w//2, cy - h//2),
                (cx + w//2, cy + h//2),
                (cx - w//2, cy + h//2),
            ]
            self.fill_polygon(img, pts, color)
            gt["structure_tags"].append("simple_patch")
            gt["geometry"] = {"type": "rect", "w": w, "h": h}
        
        elif shape_type == 'circle':
            r = random.randint(100, 180)
            cv2.circle(img, (cx, cy), r, color, -1)
            gt["structure_tags"].append("simple_patch")
            gt["geometry"] = {"type": "circle", "r": r}
        
        else:  # polygon
            n_sides = random.randint(3, 8)
            r = random.randint(100, 180)
            rotation = random.uniform(0, 360)
            pts = []
            for i in range(n_sides):
                angle = np.radians(rotation + i * 360 / n_sides)
                x = cx + r * cos(angle)
                y = cy + r * sin(angle)
                pts.append((int(x), int(y)))
            self.fill_polygon(img, pts, color)
            gt["structure_tags"].append("simple_patch")
            gt["geometry"] = {"type": "polygon", "n_sides": n_sides, "r": r}
        
        return gt
    
    def gen_level1(self, img: np.ndarray, gt: Dict) -> Dict:
        """
        Level 1: 简单（Base）
        - patch + 1 slot
        - 单SRR
        """
        color = self.random_metal_color()
        struct_type = random.choice(['patch_slot', 'srr', 'rect_patch_with_slot'])
        cx, cy = self.img_size // 2, self.img_size // 2
        
        if struct_type == 'rect_patch_with_slot':
            # 矩形贴片+槽（融合realistic_dataset_generator）
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            patch_gt = self.gen_rect_patch_with_slot(mask, cx, cy)
            # 应用到RGB图像
            mask_region = mask > 0
            if np.any(mask_region):
                img[mask_region] = color
            gt["structure_tags"].extend(["patch"])
            if "slot" in patch_gt:
                gt["structure_tags"].append("slot")
            gt["geometry"] = patch_gt
        
        elif struct_type == 'patch_slot':
            # 矩形贴片 + 槽
            w, h = random.randint(200, 300), random.randint(150, 250)
            pts = [
                (cx - w//2, cy - h//2),
                (cx + w//2, cy - h//2),
                (cx + w//2, cy + h//2),
                (cx - w//2, cy + h//2),
            ]
            self.fill_polygon(img, pts, color)
            
            # 添加槽
            sw = int(w * random.uniform(0.2, 0.4))
            sh = int(h * random.uniform(0.15, 0.3))
            slot_angle = random.uniform(0, 360)
            angle_rad = np.radians(slot_angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            slot = []
            for dx, dy in [(-sw//2, -sh//2), (sw//2, -sh//2), 
                          (sw//2, sh//2), (-sw//2, sh//2)]:
                x = cx + dx * cos_a - dy * sin_a
                y = cy + dy * cos_a + dx * sin_a
                slot.append((int(x), int(y)))
            self.fill_polygon(img, slot, (0, 0, 0))  # 孔洞
            
            gt["structure_tags"].extend(["patch", "slot"])
            gt["geometry"] = {"type": "patch_slot", "w": w, "h": h, "slot_angle": slot_angle}
        
        else:  # srr
            outer_r = random.uniform(130, 180)
            width = random.uniform(18, 35)
            gap_angle = random.uniform(20, 50)
            
            angles = np.linspace(gap_angle/2, 360 - gap_angle/2, 200)
            outer_points = []
            for a in angles:
                angle_rad = np.radians(a)
                x = cx + outer_r * cos(angle_rad)
                y = cy + outer_r * sin(angle_rad)
                outer_points.append((int(x), int(y)))
            
            inner_r = outer_r - width
            inner_points = []
            for a in reversed(angles):
                angle_rad = np.radians(a)
                x = cx + inner_r * cos(angle_rad)
                y = cy + inner_r * sin(angle_rad)
                inner_points.append((int(x), int(y)))
            
            ring_points = outer_points + inner_points
            self.fill_polygon(img, ring_points, color)
            
            gt["structure_tags"].append("srr")
            gt["geometry"] = {"type": "srr", "outer_r": outer_r, "width": width, 
                            "gap_angle": gap_angle}
        
        return gt
    
    def gen_level2(self, img: np.ndarray, gt: Dict) -> Dict:
        """
        Level 2: 中等（Topology）
        - hole / nested / SRR / CSRR
        """
        color = self.random_metal_color()
        struct_type = random.choice(['ring_hole', 'nested_rings', 'csrr', 'fractal_slot', 'multi_ring_srr'])
        cx, cy = self.img_size // 2, self.img_size // 2
        
        if struct_type == 'ring_hole':
            outer_r = random.uniform(120, 200)
            inner_r = outer_r * random.uniform(0.3, 0.6)
            cv2.circle(img, (cx, cy), int(outer_r), color, -1)
            cv2.circle(img, (cx, cy), int(inner_r), (0, 0, 0), -1)
            gt["structure_tags"].extend(["ring", "hole"])
            gt["geometry"] = {"type": "ring_hole", "outer_r": outer_r, "inner_r": inner_r}
        
        elif struct_type == 'nested_rings':
            # 嵌套环
            r1 = random.uniform(150, 200)
            r2 = r1 * random.uniform(0.5, 0.7)
            r3 = r2 * random.uniform(0.5, 0.7)
            cv2.circle(img, (cx, cy), int(r1), color, -1)
            cv2.circle(img, (cx, cy), int(r2), (0, 0, 0), -1)
            cv2.circle(img, (cx, cy), int(r3), color, -1)
            gt["structure_tags"].extend(["nested", "ring"])
            gt["geometry"] = {"type": "nested_rings", "radii": [r1, r2, r3]}
        
        elif struct_type == 'csrr':
            # 使用融合的函数
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            csrr_gt = self.gen_csrr(mask, cx, cy)
            # 应用到RGB图像
            mask_region = mask > 0
            if np.any(mask_region):
                img[mask_region] = color
            gt["structure_tags"].append("csrr")
            gt["geometry"] = csrr_gt
        
        elif struct_type == 'fractal_slot':
            # 分形槽贴片（融合realistic_dataset_generator）
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            fractal_gt = self.gen_fractal_slot_patch(mask, cx, cy)
            # 应用到RGB图像
            mask_region = mask > 0
            if np.any(mask_region):
                img[mask_region] = color
            gt["structure_tags"].extend(["fractal", "slot", "patch"])
            gt["geometry"] = fractal_gt
        
        else:  # multi_ring_srr
            # 多环SRR（融合realistic_dataset_generator）
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            multi_srr_gt = self.gen_multi_ring_srr(mask, cx, cy)
            # 应用到RGB图像
            mask_region = mask > 0
            if np.any(mask_region):
                img[mask_region] = color
            gt["structure_tags"].extend(["multi_ring_srr", "srr"])
            gt["geometry"] = multi_srr_gt
        
        return gt
    
    def gen_level3(self, img: np.ndarray, gt: Dict) -> Dict:
        """
        Level 3: 复杂（Geometry）
        - 自由曲线 / spline / 非规则边界
        """
        color = self.random_metal_color()
        struct_type = random.choice(['freeform', 'freeform_with_hole', 'multi_freeform', 
                                    'sierpinski_carpet', 'unit_array'])
        cx, cy = self.img_size // 2, self.img_size // 2
        
        if struct_type == 'freeform':
            scale = random.uniform(160, 200)
            self.freeform_patch(img, (cx, cy), scale, color)
            gt["structure_tags"].append("freeform_boundary")
            gt["geometry"] = {"type": "freeform", "scale": scale}
        
        elif struct_type == 'freeform_with_hole':
            scale = random.uniform(180, 220)
            self.freeform_patch(img, (cx, cy), scale, color)
            # 添加孔洞
            hole_r = scale * random.uniform(0.2, 0.4)
            cv2.circle(img, (cx, cy), int(hole_r), (0, 0, 0), -1)
            gt["structure_tags"].extend(["freeform_boundary", "hole"])
            gt["geometry"] = {"type": "freeform_with_hole", "scale": scale, "hole_r": hole_r}
        
        elif struct_type == 'multi_freeform':
            # 多个自由曲线组合
            for i in range(2):
                offset_x = random.randint(-80, 80)
                offset_y = random.randint(-80, 80)
                scale = random.uniform(100, 150)
                self.freeform_patch(img, (cx + offset_x, cy + offset_y), scale, color)
            gt["structure_tags"].extend(["freeform_boundary", "multi_component"])
            gt["geometry"] = {"type": "multi_freeform", "num_components": 2}
        
        elif struct_type == 'sierpinski_carpet':
            # 谢尔宾斯基地毯（融合realistic_dataset_generator）
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            sierpinski_gt = self.gen_sierpinski_carpet(mask, cx, cy)
            # 应用到RGB图像
            mask_region = mask > 0
            if np.any(mask_region):
                img[mask_region] = color
            gt["structure_tags"].extend(["sierpinski", "fractal"])
            gt["geometry"] = sierpinski_gt
        
        else:  # unit_array
            # 单元阵列（融合realistic_dataset_generator）
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            array_gt = self.gen_unit_array(mask, cx, cy)
            # 应用到RGB图像
            mask_region = mask > 0
            if np.any(mask_region):
                img[mask_region] = color
            gt["structure_tags"].extend(["array", "fss"])
            gt["geometry"] = array_gt
        
        return gt
    
    def gen_level4(self, img: np.ndarray, gt: Dict) -> Dict:
        """
        Level 4: 极端（Failure cases）
        - 极小gap + 多组件 + 噪声 + 低对比
        """
        color = self.random_metal_color()
        struct_type = random.choice(['small_gap', 'high_curvature', 'self_intersection', 
                                    'noise_overlay'])
        cx, cy = self.img_size // 2, self.img_size // 2
        
        if struct_type == 'small_gap':
            # 极近的两个组件
            gap = random.randint(3, 8)  # 极小gap
            r1, r2 = random.uniform(100, 150), random.uniform(100, 150)
            cv2.circle(img, (cx - gap//2 - int(r1), cy), int(r1), color, -1)
            cv2.circle(img, (cx + gap//2 + int(r2), cy), int(r2), color, -1)
            gt["structure_tags"].extend(["multi_component"])
            gt["risk_factors"].extend(["small_gap"])
            gt["geometry"] = {"type": "small_gap", "gap": gap, "r1": r1, "r2": r2}
        
        elif struct_type == 'high_curvature':
            # 高曲率自由曲线
            scale = random.uniform(150, 200)
            # 生成高曲率控制点
            n_ctrl = 12  # 更多控制点
            ctrl = []
            for i in range(n_ctrl):
                angle = 2 * pi * i / n_ctrl
                r = scale * random.uniform(0.4, 0.9)
                ctrl.append([cx + r * cos(angle), cy + r * sin(angle)])
            ctrl.append(ctrl[0])
            
            if SCIPY_AVAILABLE:
                ctrl_array = np.array(ctrl).T
                try:
                    tck, u = splprep(ctrl_array, s=0, per=True)
                    u_new = np.linspace(0, 1, 500)
                    x, y = splev(u_new, tck)
                    pts = list(zip(x.astype(int), y.astype(int)))
                    self.fill_polygon(img, pts, color)
                except:
                    self.fill_polygon(img, [(int(p[0]), int(p[1])) for p in ctrl], color)
            else:
                pts = self._bezier_curve(ctrl, 500)
                self.fill_polygon(img, pts, color)
            
            gt["structure_tags"].extend(["freeform_boundary"])
            gt["risk_factors"].extend(["high_curvature"])
            gt["geometry"] = {"type": "high_curvature", "scale": scale}
        
        elif struct_type == 'self_intersection':
            # 自相交结构（挑战拓扑识别）
            # 两个重叠的自由曲线
            scale1, scale2 = random.uniform(120, 160), random.uniform(120, 160)
            offset = random.randint(30, 60)
            self.freeform_patch(img, (cx - offset, cy), scale1, color)
            self.freeform_patch(img, (cx + offset, cy), scale2, color)
            gt["structure_tags"].extend(["multi_component", "freeform_boundary"])
            gt["risk_factors"].extend(["self_intersection"])
            gt["geometry"] = {"type": "self_intersection", "scale1": scale1, "scale2": scale2}
        
        else:  # noise_overlay
            # 噪声叠加
            w, h = random.randint(200, 300), random.randint(150, 250)
            pts = [
                (cx - w//2, cy - h//2),
                (cx + w//2, cy - h//2),
                (cx + w//2, cy + h//2),
                (cx - w//2, cy + h//2),
            ]
            self.fill_polygon(img, pts, color)
            
            # 添加大量局部缺陷
            for _ in range(random.randint(10, 20)):
                defect_x = random.randint(0, self.img_size - 1)
                defect_y = random.randint(0, self.img_size - 1)
                defect_r = random.randint(2, 5)
                cv2.circle(img, (defect_x, defect_y), defect_r, 
                          random.choice([(0, 0, 0), color]), -1)
            
            gt["structure_tags"].append("patch")
            gt["risk_factors"].extend(["noise_overlay", "local_defects"])
            gt["geometry"] = {"type": "noise_overlay", "w": w, "h": h}
        
        gt["intended_failure"] = True
        return gt
    
    def gen_level5(self, img: np.ndarray, gt: Dict) -> Dict:
        """
        Level 5: 超复杂（融合V2组合生成 + 极端参数）
        - 多个组件组合（3-6个）
        - 多颜色（每个组件不同颜色）
        - 自由曲线 + 传统结构混合
        - 阵列 + 连接结构
        - 极端参数组合
        """
        # 创建单通道mask用于组合
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        components = []
        
        # 生成多个主要组件（3-6个，不同颜色）
        num_main_components = random.randint(3, 6)
        colors = self.random_multi_metal_colors(num_main_components)
        
        # 存储每个组件的mask和颜色
        component_masks = []
        
        for i, color in enumerate(colors):
            comp_type = random.choice(['freeform', 'srr', 'ring', 'rect', 'multi_freeform', 
                                     'complex_nested', 'polygon', 'csrr', 'sierpinski_carpet',
                                     'fractal_slot', 'multi_ring_srr', 'unit_array'])
            # 随机位置（融合V2）
            comp_size = random.uniform(100, 200)
            cx, cy = self.random_position(comp_size)
            
            # 创建临时mask
            temp_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
            if comp_type == 'freeform':
                scale = random.uniform(100, 180)
                pts = self.freeform_patch(temp_mask, (cx, cy), scale, 255)
                components.append({"type": "freeform", "scale": scale, "center": [cx, cy], "color": color})
            
            elif comp_type == 'srr':
                srr_gt = self.gen_srr(temp_mask, cx, cy)
                components.append({**srr_gt, "color": color})
            
            elif comp_type == 'ring':
                ring_gt = self.gen_ring(temp_mask, cx, cy)
                components.append({**ring_gt, "color": color})
            
            elif comp_type == 'rect':
                rect_gt = self.gen_rect_patch(temp_mask, cx, cy)
                components.append({**rect_gt, "color": color})
            
            elif comp_type == 'polygon':
                poly_gt = self.gen_polygon(temp_mask, cx, cy)
                components.append({**poly_gt, "color": color})
            
            elif comp_type == 'multi_freeform':
                # 多个重叠的自由曲线
                for j in range(2):
                    offset_x = random.randint(-40, 40)
                    offset_y = random.randint(-40, 40)
                    scale = random.uniform(70, 120)
                    self.freeform_patch(temp_mask, (cx + offset_x, cy + offset_y), scale, 255)
                components.append({"type": "multi_freeform", "num": 2, "center": [cx, cy], "color": color})
            
            else:  # complex_nested
                # 嵌套复杂结构：外层ring + 内层自由曲线 + 孔洞
                outer_r = random.uniform(100, 150)
                cv2.circle(temp_mask, (cx, cy), int(outer_r), 255, -1)
                # 内层自由曲线
                inner_scale = outer_r * random.uniform(0.4, 0.6)
                self.freeform_patch(temp_mask, (cx, cy), inner_scale, 255)
                # 添加孔洞
                hole_r = inner_scale * random.uniform(0.3, 0.5)
                cv2.circle(temp_mask, (cx, cy), int(hole_r), 0, -1)
                components.append({"type": "complex_nested", "outer_r": outer_r, "center": [cx, cy], "color": color})
            
            # 应用制造噪声
            noise_level = random.uniform(0.2, 0.4)
            temp_mask = self.apply_manufacturing_noise(temp_mask, noise_level)
            component_masks.append((temp_mask, color))
        
            # 合并所有组件到图像（使用不同颜色）
        for comp_mask, comp_color in component_masks:
            mask_region = comp_mask > 0
            if np.any(mask_region):
                img[mask_region] = comp_color
        
        # 添加连接结构（40%概率）
        if random.random() < 0.4 and len(components) >= 2:
            # 在两个组件之间添加连接线
            c1 = components[0]['center']
            c2 = components[1]['center']
            # 绘制粗连接线
            line_width = random.randint(15, 30)
            connector_color = self.random_metal_color()
            cv2.line(img, tuple(c1), tuple(c2), connector_color, line_width)
            components.append({"type": "connector", "from": c1, "to": c2, "color": connector_color})
        
        # 添加槽/孔洞（50%概率）
        if random.random() < 0.5:
            # 在某个组件上添加槽
            target_comp = random.choice(components[:num_main_components])
            slot_cx = target_comp['center'][0] + random.randint(-50, 50)
            slot_cy = target_comp['center'][1] + random.randint(-50, 50)
            slot_gt = self.gen_slot(mask, slot_cx, slot_cy)
            # 在RGB图像上应用槽（黑色孔洞）
            slot_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            self.gen_slot(slot_mask, slot_cx, slot_cy)
            img[slot_mask > 0] = (0, 0, 0)  # 黑色孔洞
            components.append({**slot_gt, "type": "slot_hole"})
        
        # 添加阵列子区域（50%概率）
        if random.random() < 0.5:
            array_cx, array_cy = self.random_position(100)
            unit_size = np.random.uniform(30, 70)
            num_units = random.choice([2, 3, 4])
            spacing = unit_size * np.random.uniform(1.2, 1.8)
            array_color = self.random_metal_color()
            
            for i in range(num_units):
                for j in range(num_units):
                    if random.random() < 0.65:  # 65%概率有单元
                        ux = int(array_cx + (i - num_units/2) * spacing)
                        uy = int(array_cy + (j - num_units/2) * spacing)
                        unit_type = random.choice(['ring', 'srr', 'small_freeform'])
                        if unit_type == 'ring':
                            r = unit_size / 2
                            cv2.circle(img, (ux, uy), int(r), array_color, -1)
                            cv2.circle(img, (ux, uy), int(r * 0.6), (0, 0, 0), -1)
                        elif unit_type == 'srr':
                            unit_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                            self.gen_srr(unit_mask, ux, uy, unit_size/2, unit_size/6)
                            unit_region = unit_mask > 0
                            if np.any(unit_region):
                                img[unit_region] = array_color
                        else:  # small_freeform
                            unit_mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                            self.freeform_patch(unit_mask, (ux, uy), unit_size/2, 255)
                            unit_region = unit_mask > 0
                            if np.any(unit_region):
                                img[unit_region] = array_color
            
            components.append({
                "type": "array",
                "num_units": num_units * num_units,
                "unit_size": unit_size,
                "center": [array_cx, array_cy],
                "color": array_color
            })
        
        gt["structure_tags"].extend(["multi_component", "complex_combination", "multi_color", 
                                    "composite_structure"])
        gt["risk_factors"].extend(["high_complexity", "multiple_overlaps", "diverse_geometry", 
                                  "extreme_parameters"])
        gt["geometry"] = {
            "type": "level5_composite",
            "num_components": len(components),
            "components": components
        }
        gt["intended_failure"] = True
        
        return gt
    
    def generate_sample(self) -> Tuple[np.ndarray, Dict]:
        """
        生成单个验证样本
        
        返回:
            (图像, Ground Truth字典)
        """
        # 根据分布选择复杂度等级
        level = random.choices(
            list(self.complexity_distribution.keys()),
            weights=list(self.complexity_distribution.values())
        )[0]
        
        # 生成背景
        img = self.substrate_background()
        
        # 初始化GT
        gt = {
            "complexity_level": level,
            "structure_tags": [],
            "risk_factors": [],
            "intended_failure": False,
            "geometry": {}
        }
        
        # 根据等级生成结构
        if level == 0:
            self.gen_level0(img, gt)
        elif level == 1:
            self.gen_level1(img, gt)
        elif level == 2:
            self.gen_level2(img, gt)
        elif level == 3:
            self.gen_level3(img, gt)
        elif level == 4:
            self.gen_level4(img, gt)
        else:  # level 5
            self.gen_level5(img, gt)
        
        # 添加颜色扰动和丰富化（Level 2+）
        if level >= 2:
            # 获取实际背景颜色（从图像边缘采样，而不是重新生成）
            edge_color = img[0, 0].copy()
            # 创建mask：不是背景颜色的区域
            mask = np.any(img != edge_color, axis=2)
            if np.any(mask):
                # Level 3+ 使用更强的扰动
                perturbation = random.uniform(0.08, 0.20) if level >= 3 else random.uniform(0.05, 0.15)
                img = self.add_color_perturbation(img, mask, 
                                                 self.random_metal_color(), 
                                                 perturbation)
        
        # Level 4+ 添加额外的颜色丰富化（模拟真实场景）
        if level >= 4:
            # 在部分区域添加轻微的颜色变化（模拟不同材料区域、氧化、老化）
            edge_color = img[0, 0].copy()
            mask = np.any(img != edge_color, axis=2)
            if np.any(mask):
                # 随机选择部分区域进行颜色微调
                region_mask = np.zeros_like(mask, dtype=bool)
                # 随机选择30-50%的区域
                selected_pixels = np.random.choice(np.sum(mask), 
                                                   size=int(np.sum(mask) * random.uniform(0.3, 0.5)),
                                                   replace=False)
                mask_indices = np.where(mask)
                region_mask[mask_indices[0][selected_pixels], mask_indices[1][selected_pixels]] = True
                
                if np.any(region_mask):
                    # 模拟不同效果
                    effect_type = random.choice(['oxidation', 'aging', 'material_mix', 'lighting'])
                    
                    if effect_type == 'oxidation':
                        # 氧化效果：变暗、偏棕
                        color_shift = np.array([-10, -5, -15])  # BGR: 偏棕
                        img_float = img.astype(np.float32)
                        img_float[region_mask] = np.clip(img_float[region_mask] + color_shift, 0, 255)
                        img = img_float.astype(np.uint8)
                    
                    elif effect_type == 'aging':
                        # 老化效果：偏黄
                        color_shift = np.array([-5, 5, 10])  # BGR: 偏黄
                        img_float = img.astype(np.float32)
                        img_float[region_mask] = np.clip(img_float[region_mask] + color_shift, 0, 255)
                        img = img_float.astype(np.uint8)
                    
                    elif effect_type == 'material_mix':
                        # 材料混合：轻微颜色变化
                        color_shift = np.random.uniform(-12, 12, 3)
                        img_float = img.astype(np.float32)
                        img_float[region_mask] = np.clip(img_float[region_mask] + color_shift, 0, 255)
                        img = img_float.astype(np.uint8)
                    
                    else:  # lighting
                        # 光照不均：亮度变化
                        brightness_shift = np.random.uniform(-20, 20)
                        img_float = img.astype(np.float32)
                        img_float[region_mask] = np.clip(img_float[region_mask] + brightness_shift, 0, 255)
                        img = img_float.astype(np.uint8)
        
        # Level 5 添加更多颜色变化（多材料、多工艺）
        if level == 5:
            edge_color = img[0, 0].copy()
            mask = np.any(img != edge_color, axis=2)
            if np.any(mask):
                # 添加局部区域的颜色变化（模拟不同工艺区域）
                num_regions = random.randint(2, 4)
                for _ in range(num_regions):
                    # 随机选择一个区域
                    region_mask = np.zeros_like(mask, dtype=bool)
                    if np.any(mask):
                        selected_pixels = np.random.choice(
                            np.sum(mask),
                            size=int(np.sum(mask) * random.uniform(0.1, 0.25)),
                            replace=False
                        )
                        mask_indices = np.where(mask)
                        region_mask[mask_indices[0][selected_pixels], 
                                   mask_indices[1][selected_pixels]] = True
                        
                        if np.any(region_mask):
                            # 应用不同的材料效果
                            material_effect = random.choice([
                                'gold_plating',  # 局部镀金
                                'nickel_plating', # 局部镀镍
                                'oxidation_patch', # 氧化斑块
                                'solder_residue',  # 焊料残留
                            ])
                            
                            if material_effect == 'gold_plating':
                                # 局部镀金：偏黄
                                color_shift = np.array([-8, 8, 15])
                            elif material_effect == 'nickel_plating':
                                # 局部镀镍：偏蓝灰
                                color_shift = np.array([5, 3, -5])
                            elif material_effect == 'oxidation_patch':
                                # 氧化斑块：变暗偏棕
                                color_shift = np.array([-15, -8, -20])
                            else:  # solder_residue
                                # 焊料残留：偏灰白
                                color_shift = np.array([8, 8, 8])
                            
                            img_float = img.astype(np.float32)
                            img_float[region_mask] = np.clip(img_float[region_mask] + color_shift, 0, 255)
                            img = img_float.astype(np.uint8)
        
        return img, gt


def generate_validation_dataset(dataset_size: int = 400, output_dir: str = "validation_dataset",
                               img_size: int = 512, seed: Optional[int] = None):
    """
    生成验证集
    
    参数:
        dataset_size: 数据集大小（验证集不需要太大）
        output_dir: 输出目录
        img_size: 图像尺寸
        seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 创建目录
    images_dir = os.path.join(output_dir, "images")
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    
    generator = ValidationDatasetGenerator(img_size=img_size)
    summary = []
    
    print("=" * 70)
    print(f"开始生成验证集（验证集专用generator）")
    print(f"数据集大小: {dataset_size}")
    print(f"图像尺寸: {img_size}x{img_size}")
    print(f"输出目录: {output_dir}")
    print("=" * 70)
    print("复杂度分布:")
    for level, prob in generator.complexity_distribution.items():
        print(f"  Level {level}: {prob*100:.0f}%")
    print("=" * 70)
    
    for idx in range(dataset_size):
        if (idx + 1) % 50 == 0:
            print(f"进度: [{idx + 1}/{dataset_size}] ({100*(idx+1)/dataset_size:.1f}%)")
        
        # 生成样本
        img, gt = generator.generate_sample()
        
        # 保存图像
        name = f"val_{idx:05d}"
        img_path = os.path.join(images_dir, f"{name}.png")
        cv2.imwrite(img_path, img)
        
        # 保存Ground Truth
        gt_path = os.path.join(gt_dir, f"{name}.json")
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)
        
        # 添加到摘要
        summary.append({
            "name": name,
            "image_path": img_path,
            "gt_path": gt_path,
            **gt
        })
    
    # 保存摘要
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(f"验证集生成完成！")
    print(f"  - 图像数量: {dataset_size}")
    print(f"  - 图像目录: {images_dir}")
    print(f"  - GT目录: {gt_dir}")
    print(f"  - 摘要文件: {summary_path}")
    print("=" * 70)
    
    # 统计信息
    level_counts = {}
    tag_counts = {}
    risk_counts = {}
    
    for item in summary:
        level = item.get('complexity_level', -1)
        level_counts[level] = level_counts.get(level, 0) + 1
        
        for tag in item.get('structure_tags', []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        for risk in item.get('risk_factors', []):
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    print("\n复杂度等级统计:")
    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} ({100*level_counts[level]/dataset_size:.1f}%)")
    
    print("\n结构标签统计:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {tag}: {count}")
    
    if risk_counts:
        print("\n风险因素统计:")
        for risk, count in sorted(risk_counts.items(), key=lambda x: -x[1]):
            print(f"  {risk}: {count}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='验证集专用生成器')
    parser.add_argument('--size', type=int, default=400,
                       help='数据集大小（默认：400）')
    parser.add_argument('--output', type=str, default='validation_dataset',
                       help='输出目录（默认：validation_dataset）')
    parser.add_argument('--img-size', type=int, default=512,
                       help='图像尺寸（默认：512）')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（用于可重复性）')
    
    args = parser.parse_args()
    
    generate_validation_dataset(
        dataset_size=args.size,
        output_dir=args.output,
        img_size=args.img_size,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
