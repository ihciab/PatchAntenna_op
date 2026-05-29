"""
生成“单闭合、无噪声、参数化几何段组合”的测试数据集。

要求对齐：
1) 每个样本只有一个闭合图形
2) 图形由 line / arc / circle / bspline 段组合
3) 从简单到复杂共 50 个样本
4) 无噪声纯净图（白底黑色闭合线，不填充）
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class Segment:
    type: str  # line | arc | circle | bspline
    params: Dict


def _line_points(p0: Point, p1: Point, n: int = 60) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(2, n))
    p0a = np.array(p0, dtype=np.float64)
    p1a = np.array(p1, dtype=np.float64)
    pts = (1.0 - t)[:, None] * p0a[None, :] + t[:, None] * p1a[None, :]
    return pts


def _arc_points(center: Point, r: float, a0_deg: float, a1_deg: float, n: int = 80) -> np.ndarray:
    a = np.linspace(math.radians(a0_deg), math.radians(a1_deg), max(3, n))
    cx, cy = center
    x = cx + r * np.cos(a)
    y = cy + r * np.sin(a)
    return np.column_stack([x, y])


def _circle_points(center: Point, r: float, n: int = 180) -> np.ndarray:
    return _arc_points(center, r, 0.0, 360.0, n=n)


def _cubic_bezier_points(p0: Point, p1: Point, p2: Point, p3: Point, n: int = 80) -> np.ndarray:
    t = np.linspace(0.0, 1.0, max(3, n))
    p0a = np.array(p0, dtype=np.float64)
    p1a = np.array(p1, dtype=np.float64)
    p2a = np.array(p2, dtype=np.float64)
    p3a = np.array(p3, dtype=np.float64)
    pts = (
        ((1 - t) ** 3)[:, None] * p0a[None, :]
        + (3 * ((1 - t) ** 2) * t)[:, None] * p1a[None, :]
        + (3 * (1 - t) * (t**2))[:, None] * p2a[None, :]
        + (t**3)[:, None] * p3a[None, :]
    )
    return pts


def _segments_to_points(segments: List[Segment]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for seg in segments:
        p = seg.params
        if seg.type == "line":
            pts = _line_points(tuple(p["p0"]), tuple(p["p1"]), n=p.get("samples", 60))
        elif seg.type == "arc":
            pts = _arc_points(tuple(p["center"]), float(p["radius"]), float(p["start_deg"]), float(p["end_deg"]), n=p.get("samples", 90))
        elif seg.type == "circle":
            pts = _circle_points(tuple(p["center"]), float(p["radius"]), n=p.get("samples", 220))
        elif seg.type == "bspline":
            pts = _cubic_bezier_points(tuple(p["p0"]), tuple(p["p1"]), tuple(p["p2"]), tuple(p["p3"]), n=p.get("samples", 90))
        else:
            raise ValueError(f"Unknown segment type: {seg.type}")
        if chunks:
            pts = pts[1:]  # 去重连接点
        chunks.append(pts)

    if not chunks:
        return np.zeros((0, 2), dtype=np.float64)
    contour = np.vstack(chunks)
    if len(contour) >= 2 and np.linalg.norm(contour[0] - contour[-1]) > 1e-6:
        contour = np.vstack([contour, contour[0:1]])
    return contour


def _regular_polygon_segments(cx: float, cy: float, r: float, sides: int, rot_deg: float = 0.0) -> List[Segment]:
    pts: List[Point] = []
    for i in range(sides):
        a = math.radians(rot_deg + i * (360.0 / sides))
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    segs: List[Segment] = []
    for i in range(sides):
        segs.append(Segment("line", {"p0": pts[i], "p1": pts[(i + 1) % sides], "samples": 50}))
    return segs


def _rounded_rect_segments(cx: float, cy: float, w: float, h: float, rr: float) -> List[Segment]:
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    x1, y1 = cx + w / 2.0, cy + h / 2.0
    r = min(rr, w / 2.0 - 1, h / 2.0 - 1)
    return [
        Segment("line", {"p0": (x0 + r, y0), "p1": (x1 - r, y0), "samples": 50}),
        Segment("arc", {"center": (x1 - r, y0 + r), "radius": r, "start_deg": -90, "end_deg": 0, "samples": 60}),
        Segment("line", {"p0": (x1, y0 + r), "p1": (x1, y1 - r), "samples": 50}),
        Segment("arc", {"center": (x1 - r, y1 - r), "radius": r, "start_deg": 0, "end_deg": 90, "samples": 60}),
        Segment("line", {"p0": (x1 - r, y1), "p1": (x0 + r, y1), "samples": 50}),
        Segment("arc", {"center": (x0 + r, y1 - r), "radius": r, "start_deg": 90, "end_deg": 180, "samples": 60}),
        Segment("line", {"p0": (x0, y1 - r), "p1": (x0, y0 + r), "samples": 50}),
        Segment("arc", {"center": (x0 + r, y0 + r), "radius": r, "start_deg": 180, "end_deg": 270, "samples": 60}),
    ]


def _bezier_blob_segments(cx: float, cy: float, scale: float) -> List[Segment]:
    # 四段三次曲线构成闭合轮廓
    p0 = (cx, cy - 1.00 * scale)
    p1 = (cx + 0.95 * scale, cy - 0.50 * scale)
    p2 = (cx + 0.85 * scale, cy + 0.65 * scale)
    p3 = (cx, cy + 1.00 * scale)
    p4 = (cx - 0.95 * scale, cy + 0.45 * scale)
    p5 = (cx - 0.80 * scale, cy - 0.60 * scale)
    return [
        Segment("bspline", {"p0": p0, "p1": (cx + 0.45 * scale, cy - 1.05 * scale), "p2": (cx + 1.0 * scale, cy - 0.9 * scale), "p3": p1}),
        Segment("bspline", {"p0": p1, "p1": (cx + 1.1 * scale, cy - 0.05 * scale), "p2": (cx + 1.1 * scale, cy + 0.75 * scale), "p3": p2}),
        Segment("bspline", {"p0": p2, "p1": (cx + 0.45 * scale, cy + 1.2 * scale), "p2": (cx - 0.45 * scale, cy + 1.1 * scale), "p3": p4}),
        Segment("bspline", {"p0": p4, "p1": (cx - 1.15 * scale, cy + 0.20 * scale), "p2": (cx - 1.0 * scale, cy - 0.95 * scale), "p3": p5}),
        Segment("bspline", {"p0": p5, "p1": (cx - 0.35 * scale, cy - 1.1 * scale), "p2": (cx - 0.08 * scale, cy - 1.05 * scale), "p3": p0}),
    ]


def _gear_like_segments(cx: float, cy: float, r0: float, teeth: int) -> List[Segment]:
    pts: List[Point] = []
    for i in range(teeth * 2):
        a = math.radians(i * 180.0 / teeth)
        rr = r0 * (1.0 if i % 2 == 0 else 0.78)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    segs: List[Segment] = []
    for i in range(len(pts)):
        segs.append(Segment("line", {"p0": pts[i], "p1": pts[(i + 1) % len(pts)], "samples": 30}))
    return segs


def _mixed_line_arc_shape(cx: float, cy: float, s: float) -> List[Segment]:
    # 类“胶囊+折线拐角”闭合轮廓
    return [
        Segment("line", {"p0": (cx - 1.1 * s, cy - 0.5 * s), "p1": (cx + 0.6 * s, cy - 0.5 * s), "samples": 45}),
        Segment("arc", {"center": (cx + 0.6 * s, cy - 0.1 * s), "radius": 0.4 * s, "start_deg": -90, "end_deg": 20, "samples": 65}),
        Segment("line", {"p0": (cx + 0.98 * s, cy + 0.04 * s), "p1": (cx + 0.35 * s, cy + 0.95 * s), "samples": 45}),
        Segment("arc", {"center": (cx - 0.05 * s, cy + 0.68 * s), "radius": 0.46 * s, "start_deg": 30, "end_deg": 175, "samples": 70}),
        Segment("line", {"p0": (cx - 0.51 * s, cy + 0.72 * s), "p1": (cx - 1.1 * s, cy + 0.20 * s), "samples": 45}),
        Segment("arc", {"center": (cx - 1.1 * s, cy - 0.15 * s), "radius": 0.35 * s, "start_deg": 90, "end_deg": 270, "samples": 60}),
    ]


def _polyline_to_line_segments(points: List[Point], samples: int = 24) -> List[Segment]:
    segs: List[Segment] = []
    for i in range(len(points)):
        segs.append(Segment("line", {"p0": points[i], "p1": points[(i + 1) % len(points)], "samples": samples}))
    return segs


def _c_split_ring_segments(cx: float, cy: float, r_outer: float, thickness: float, gap_deg: float) -> List[Segment]:
    """单闭合 C 形分裂环轮廓（超材料常见单元外边界）"""
    r_inner = max(8.0, r_outer - thickness)
    g0 = gap_deg / 2.0
    g1 = 360.0 - gap_deg / 2.0
    p_out0 = (cx + r_outer * math.cos(math.radians(g0)), cy + r_outer * math.sin(math.radians(g0)))
    p_out1 = (cx + r_outer * math.cos(math.radians(g1)), cy + r_outer * math.sin(math.radians(g1)))
    p_in0 = (cx + r_inner * math.cos(math.radians(g0)), cy + r_inner * math.sin(math.radians(g0)))
    p_in1 = (cx + r_inner * math.cos(math.radians(g1)), cy + r_inner * math.sin(math.radians(g1)))
    return [
        Segment("line", {"p0": p_out0, "p1": p_in0, "samples": 24}),
        Segment("arc", {"center": (cx, cy), "radius": r_inner, "start_deg": g0, "end_deg": g1, "samples": 100}),
        Segment("line", {"p0": p_in1, "p1": p_out1, "samples": 24}),
        Segment("arc", {"center": (cx, cy), "radius": r_outer, "start_deg": g1, "end_deg": g0 + 360.0, "samples": 130}),
    ]


def _meander_loop_segments(cx: float, cy: float, w: float, h: float, turns: int) -> List[Segment]:
    """天线 meander 风格闭合轮廓，纯 line 组成。"""
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    x1, y1 = cx + w / 2.0, cy + h / 2.0
    ys = np.linspace(y0, y1, turns * 2 + 1)
    pts: List[Point] = [(x0, y0)]
    side = 1
    for k in range(1, len(ys)):
        yk = float(ys[k])
        xk = x1 if side > 0 else x0
        pts.append((xk, yk))
        side *= -1
    # 回到底部形成闭合环
    pts.extend([(x0 + 0.15 * w, y1), (x0 + 0.15 * w, y0 + 0.12 * h), (x0, y0)])
    # 去掉末尾重复起点，交给闭合逻辑
    if np.linalg.norm(np.array(pts[-1]) - np.array(pts[0])) < 1e-6:
        pts = pts[:-1]
    return _polyline_to_line_segments(pts, samples=20)


def _koch_curve(p0: Point, p1: Point, depth: int) -> List[Point]:
    if depth <= 0:
        return [p0, p1]
    x0, y0 = p0
    x1, y1 = p1
    a = (x0 + (x1 - x0) / 3.0, y0 + (y1 - y0) / 3.0)
    c = (x0 + 2.0 * (x1 - x0) / 3.0, y0 + 2.0 * (y1 - y0) / 3.0)
    vx, vy = (c[0] - a[0], c[1] - a[1])
    b = (
        a[0] + vx * math.cos(-math.pi / 3.0) - vy * math.sin(-math.pi / 3.0),
        a[1] + vx * math.sin(-math.pi / 3.0) + vy * math.cos(-math.pi / 3.0),
    )
    p01 = _koch_curve(p0, a, depth - 1)
    p12 = _koch_curve(a, b, depth - 1)
    p23 = _koch_curve(b, c, depth - 1)
    p34 = _koch_curve(c, p1, depth - 1)
    return p01[:-1] + p12[:-1] + p23[:-1] + p34


def _koch_snowflake_segments(cx: float, cy: float, r: float, depth: int) -> List[Segment]:
    base = []
    for i in range(3):
        a = math.radians(90 + i * 120)
        base.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts = []
    for i in range(3):
        seg = _koch_curve(base[i], base[(i + 1) % 3], depth)
        if pts:
            seg = seg[1:]
        pts.extend(seg)
    return _polyline_to_line_segments(pts, samples=8)


def _tree_fractal_outline_segments(cx: float, cy: float, h: float, levels: int = 4) -> List[Segment]:
    """
    生成树状分形“外轮廓”闭合线（不交叉）。
    通过构造左边界并镜像生成右边界，形成单闭合轮廓。
    """
    base_y = cy + 0.55 * h
    trunk_top = cy + 0.05 * h
    x_l = cx - 0.09 * h
    x_r = cx + 0.09 * h

    # 左边界：从树干底部沿主干上行，并在各层添加分支外凸点
    left: List[Point] = [(x_l, base_y), (x_l, trunk_top)]
    for lv in range(levels):
        y = trunk_top - (lv + 1) * (0.16 * h)
        spread = (0.24 + 0.09 * lv) * h
        left.extend([
            (cx - spread * 0.65, y + 0.05 * h),
            (cx - spread, y - 0.06 * h),
            (cx - spread * 0.48, y - 0.12 * h),
        ])
    # 顶部收尖
    left.append((cx - 0.02 * h, cy - 0.62 * h))

    # 右边界镜像（逆序拼接）
    right = [(2 * cx - x, y) for x, y in reversed(left)]

    pts = left + right
    # 去掉可能的重复末点
    cleaned: List[Point] = []
    for p in pts:
        if not cleaned or np.linalg.norm(np.array(cleaned[-1]) - np.array(p)) > 1e-6:
            cleaned.append(p)
    return _polyline_to_line_segments(cleaned, samples=18)


def _radial_wavy_polygon_segments(
    cx: float, cy: float, base_r: float, n_points: int, a1: float, k1: int, a2: float, k2: int
) -> List[Segment]:
    """
    非自交高复杂轮廓：按角度单调采样极坐标半径，构造闭合折线。
    r(theta)=base + a1*sin(k1*theta)+a2*cos(k2*theta)，并做下限保护。
    """
    pts: List[Point] = []
    for i in range(n_points):
        th = 2.0 * math.pi * i / n_points
        rr = base_r + a1 * math.sin(k1 * th) + a2 * math.cos(k2 * th)
        rr = max(base_r * 0.45, rr)
        pts.append((cx + rr * math.cos(th), cy + rr * math.sin(th)))
    return _polyline_to_line_segments(pts, samples=10)


def _line_bspline_dense_segments(cx: float, cy: float, scale: float) -> List[Segment]:
    """
    line+bspline 高占比模板（非自交）：先走多边折线，再用3段样条柔化闭合。
    """
    p = [
        (cx - 1.05 * scale, cy - 0.50 * scale),
        (cx - 0.35 * scale, cy - 0.95 * scale),
        (cx + 0.40 * scale, cy - 0.88 * scale),
        (cx + 1.00 * scale, cy - 0.22 * scale),
        (cx + 0.86 * scale, cy + 0.62 * scale),
        (cx + 0.12 * scale, cy + 0.98 * scale),
        (cx - 0.76 * scale, cy + 0.72 * scale),
    ]
    segs: List[Segment] = []
    for i in range(4):
        segs.append(Segment("line", {"p0": p[i], "p1": p[i + 1], "samples": 22}))
    segs.extend(
        [
            Segment("bspline", {"p0": p[4], "p1": (cx + 0.70 * scale, cy + 0.95 * scale), "p2": (cx + 0.35 * scale, cy + 1.10 * scale), "p3": p[5], "samples": 70}),
            Segment("bspline", {"p0": p[5], "p1": (cx - 0.20 * scale, cy + 1.08 * scale), "p2": (cx - 0.78 * scale, cy + 0.92 * scale), "p3": p[6], "samples": 70}),
            Segment("bspline", {"p0": p[6], "p1": (cx - 1.10 * scale, cy + 0.40 * scale), "p2": (cx - 1.15 * scale, cy - 0.15 * scale), "p3": p[0], "samples": 70}),
        ]
    )
    return segs


def _segments_intersect(a0: Point, a1: Point, b0: Point, b1: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p: Point, q: Point, r: Point) -> bool:
        return min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9

    o1 = orient(a0, a1, b0)
    o2 = orient(a0, a1, b1)
    o3 = orient(b0, b1, a0)
    o4 = orient(b0, b1, a1)

    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True
    if abs(o1) < 1e-9 and on_seg(a0, b0, a1):
        return True
    if abs(o2) < 1e-9 and on_seg(a0, b1, a1):
        return True
    if abs(o3) < 1e-9 and on_seg(b0, a0, b1):
        return True
    if abs(o4) < 1e-9 and on_seg(b0, a1, b1):
        return True
    return False


def _has_self_intersection(contour: np.ndarray) -> bool:
    if contour is None or len(contour) < 6:
        return False
    pts = contour[:-1] if np.linalg.norm(contour[0] - contour[-1]) < 1e-6 else contour
    n = len(pts)
    for i in range(n):
        a0 = tuple(pts[i])
        a1 = tuple(pts[(i + 1) % n])
        for j in range(i + 1, n):
            # 跳过相邻边和首尾相邻边
            if j in (i, (i + 1) % n, (i - 1) % n):
                continue
            if i == 0 and j == n - 1:
                continue
            b0 = tuple(pts[j])
            b1 = tuple(pts[(j + 1) % n])
            if _segments_intersect(a0, a1, b0, b1):
                return True
    return False


def _build_segments_for_index(i: int, img_size: int) -> Tuple[int, List[str], List[Segment]]:
    cx = img_size / 2.0
    cy = img_size / 2.0
    level = min(5, i // 8)

    if 0 <= i <= 7:
        cases = [
            ([Segment("circle", {"center": (cx, cy), "radius": 112, "samples": 240})], ["basic", "circle"]),
            (_regular_polygon_segments(cx, cy, 126, sides=3, rot_deg=90), ["basic", "triangle"]),
            (_regular_polygon_segments(cx, cy, 126, sides=4, rot_deg=45), ["basic", "diamond"]),
            (_regular_polygon_segments(cx, cy, 122, sides=5, rot_deg=18), ["basic", "pentagon"]),
            (_regular_polygon_segments(cx, cy, 120, sides=6, rot_deg=0), ["basic", "hexagon"]),
            (_rounded_rect_segments(cx, cy, w=290, h=210, rr=24), ["basic", "rounded_rect"]),
            (_rounded_rect_segments(cx, cy, w=250, h=250, rr=54), ["basic", "rounded_square"]),
            (_mixed_line_arc_shape(cx, cy, s=108), ["basic", "mixed_line_arc"]),
        ]
        segs, tags = cases[i]
    elif 8 <= i <= 15:
        j = i - 8
        cases = [
            (_polyline_to_line_segments([(cx - 150, cy - 90), (cx + 150, cy - 90), (cx + 150, cy + 90), (cx + 45, cy + 90), (cx + 45, cy + 50), (cx - 45, cy + 50), (cx - 45, cy + 90), (cx - 150, cy + 90)], 22), ["antenna", "patch_center_notch"]),
            (_polyline_to_line_segments([(cx - 150, cy - 90), (cx + 150, cy - 90), (cx + 150, cy + 90), (cx + 90, cy + 90), (cx + 90, cy + 35), (cx + 18, cy + 35), (cx + 18, cy + 90), (cx - 150, cy + 90)], 22), ["antenna", "patch_offset_notch"]),
            (_meander_loop_segments(cx, cy, w=260, h=210, turns=4), ["antenna", "meander_4"]),
            (_meander_loop_segments(cx, cy, w=260, h=210, turns=5), ["antenna", "meander_5"]),
            ([
                Segment("arc", {"center": (cx, cy), "radius": 120, "start_deg": -160, "end_deg": 60, "samples": 130}),
                Segment("line", {"p0": (cx + 60, cy + 104), "p1": (cx + 6, cy + 35), "samples": 36}),
                Segment("arc", {"center": (cx, cy), "radius": 52, "start_deg": 40, "end_deg": -175, "samples": 100}),
                Segment("line", {"p0": (cx - 52, cy - 4), "p1": (cx - 112, cy - 42), "samples": 26}),
            ], ["antenna", "spiral_like"]),
            ((lambda b: [b[0], b[1], Segment("line", {"p0": b[2].params["p0"], "p1": b[3].params["p3"], "samples": 28}), b[4]])(_bezier_blob_segments(cx - 8, cy + 2, 92)), ["antenna", "spline_patch"]),
            (_rounded_rect_segments(cx, cy, 300, 180, 40) + [Segment("line", {"p0": (cx + 110, cy + 90), "p1": (cx + 25, cy + 30), "samples": 24})], ["antenna", "loaded_patch_outline"]),
            (_polyline_to_line_segments([(cx - 145, cy - 80), (cx + 115, cy - 80), (cx + 145, cy - 20), (cx + 80, cy + 85), (cx - 40, cy + 105), (cx - 145, cy + 48)], 20), ["antenna", "irregular_patch"]),
        ]
        segs, tags = cases[j]
    elif 16 <= i <= 23:
        j = i - 16
        cases = [
            (_c_split_ring_segments(cx, cy, r_outer=145, thickness=30, gap_deg=36), ["metamaterial", "single_split_ring"]),
            (_c_split_ring_segments(cx, cy, r_outer=142, thickness=22, gap_deg=58), ["metamaterial", "wide_gap_split_ring"]),
            (_c_split_ring_segments(cx, cy, r_outer=132, thickness=28, gap_deg=26), ["metamaterial", "narrow_gap_split_ring"]),
            (_gear_like_segments(cx, cy, 136, teeth=10), ["metamaterial", "polygon_resonator_10"]),
            (_gear_like_segments(cx, cy, 136, teeth=12), ["metamaterial", "polygon_resonator_12"]),
            (_gear_like_segments(cx, cy, 136, teeth=14), ["metamaterial", "polygon_resonator_14"]),
            ([
                Segment("line", {"p0": (cx - 130, cy - 70), "p1": (cx + 95, cy - 70), "samples": 30}),
                Segment("arc", {"center": (cx + 95, cy - 20), "radius": 50, "start_deg": -90, "end_deg": 90, "samples": 72}),
                Segment("line", {"p0": (cx + 95, cy + 70), "p1": (cx - 110, cy + 70), "samples": 30}),
                Segment("arc", {"center": (cx - 110, cy + 10), "radius": 60, "start_deg": 90, "end_deg": 270, "samples": 78}),
            ], ["metamaterial", "arc_line_resonator"]),
            ((lambda b: [Segment("line", {"p0": b[0].params["p0"], "p1": b[1].params["p3"], "samples": 22}), b[2], b[3], b[4]])(_bezier_blob_segments(cx, cy, 86)), ["metamaterial", "hybrid_resonator"]),
        ]
        segs, tags = cases[j]
    elif 24 <= i <= 31:
        j = i - 24
        cases = [
            (_koch_snowflake_segments(cx, cy, r=116, depth=1), ["fractal", "koch_d1"]),
            (_koch_snowflake_segments(cx, cy, r=116, depth=2), ["fractal", "koch_d2"]),
            (_koch_snowflake_segments(cx, cy, r=110, depth=3), ["fractal", "koch_d3"]),
            (_tree_fractal_outline_segments(cx, cy + 4, h=210, levels=3), ["fractal", "tree_d3"]),
            (_tree_fractal_outline_segments(cx, cy + 6, h=220, levels=4), ["fractal", "tree_d4"]),
            (_polyline_to_line_segments([(cx - 150, cy - 25), (cx - 70, cy - 120), (cx + 55, cy - 112), (cx + 145, cy - 32), (cx + 120, cy + 92), (cx - 15, cy + 128), (cx - 125, cy + 72)], 16), ["fractal_like", "heptagon_irregular"]),
            (_polyline_to_line_segments([(cx, cy - 130), (cx + 92, cy - 94), (cx + 132, cy), (cx + 92, cy + 94), (cx, cy + 130), (cx - 92, cy + 94), (cx - 132, cy), (cx - 92, cy - 94)], 14), ["fractal_like", "octagon_warped"]),
            (_gear_like_segments(cx, cy, 126, teeth=16), ["fractal_like", "high_tooth_gear"]),
        ]
        segs, tags = cases[j]
    elif 32 <= i <= 39:
        j = i - 32
        b1 = _bezier_blob_segments(cx, cy, 98)
        b2 = _bezier_blob_segments(cx + 8, cy - 6, 100)
        b3 = _bezier_blob_segments(cx, cy, 96)
        b4 = _bezier_blob_segments(cx, cy, 94)
        b5 = _bezier_blob_segments(cx, cy, 92)
        b6 = _bezier_blob_segments(cx, cy, 90)
        b7 = _bezier_blob_segments(cx, cy, 102)
        b8 = _bezier_blob_segments(cx, cy, 96)
        cases = [
            (b1, ["bspline", "blob_a"]),
            (b2, ["bspline", "blob_b"]),
            ([b3[0], Segment("line", {"p0": b3[1].params["p0"], "p1": b3[2].params["p3"], "samples": 30}), b3[3], b3[4]], ["bspline", "line_bridge"]),
            ([b4[0], Segment("arc", {"center": (cx + 60, cy + 15), "radius": 46, "start_deg": -120, "end_deg": 130, "samples": 80}), b4[3], b4[4]], ["bspline", "arc_bridge"]),
            ([b5[0], b5[1], Segment("line", {"p0": b5[2].params["p0"], "p1": (cx - 105, cy + 18), "samples": 26}), Segment("arc", {"center": (cx - 65, cy - 30), "radius": 58, "start_deg": 120, "end_deg": 272, "samples": 80})], ["bspline", "line_arc_mix_1"]),
            ([Segment("line", {"p0": b6[0].params["p0"], "p1": b6[1].params["p3"], "samples": 28}), Segment("arc", {"center": (cx + 22, cy + 28), "radius": 78, "start_deg": -25, "end_deg": 180, "samples": 95}), b6[3], b6[4]], ["bspline", "line_arc_mix_2"]),
            ([b7[0], b7[1], b7[2], Segment("line", {"p0": b7[3].params["p0"], "p1": b7[0].params["p0"], "samples": 24})], ["bspline", "triangle_spline_hybrid"]),
            ([Segment("arc", {"center": (cx, cy), "radius": 112, "start_deg": 210, "end_deg": 350, "samples": 80}), Segment("line", {"p0": (cx + 110, cy - 18), "p1": b8[2].params["p3"], "samples": 28}), b8[3], b8[4]], ["bspline", "outer_arc_hybrid"]),
        ]
        segs, tags = cases[j]
        # 增加 line + bspline 高占比标签
        if j in (2, 5, 6):
            tags = tags + ["line_bspline_dense"]
    else:
        j = i - 40
        # 40..49: 全部为高复杂、非自交、互异模板
        cases = [
            (_radial_wavy_polygon_segments(cx, cy, 122, 56, 22, 5, 14, 9), ["complex", "wavy_harmonic_5_9", "line"]),
            (_radial_wavy_polygon_segments(cx, cy, 118, 60, 28, 7, 10, 11), ["complex", "wavy_harmonic_7_11", "line"]),
            (_line_bspline_dense_segments(cx, cy, 116), ["complex", "line_bspline_dense_a", "line_bspline_dense"]),
            (_line_bspline_dense_segments(cx + 6, cy - 4, 108), ["complex", "line_bspline_dense_b", "line_bspline_dense"]),
            (_line_bspline_dense_segments(cx - 5, cy + 7, 100) + [Segment("arc", {"center": (cx + 14, cy - 8), "radius": 88, "start_deg": 205, "end_deg": 340, "samples": 85})], ["complex", "line_bspline_arc_dense", "line_bspline_dense"]),
            (_koch_snowflake_segments(cx, cy, 104, 3), ["complex", "fractal_koch_deep", "fractal"]),
            (_tree_fractal_outline_segments(cx, cy + 6, h=230, levels=5), ["complex", "fractal_tree_deep", "fractal"]),
            (_radial_wavy_polygon_segments(cx, cy, 120, 64, 16, 13, 12, 17), ["complex", "wavy_harmonic_13_17", "line"]),
            (_bezier_blob_segments(cx, cy, 96) + [Segment("line", {"p0": (cx - 96, cy + 6), "p1": (cx + 98, cy - 2), "samples": 32})], ["complex", "spline_chord_hybrid", "bspline_line"]),
            (_rounded_rect_segments(cx, cy, 286, 206, 34) + _radial_wavy_polygon_segments(cx, cy, 82, 24, 12, 3, 8, 5)[:8], ["complex", "rounded_wavy_hybrid", "arc_line"]),
        ]
        segs, tags = cases[j]

    return level, tags, segs


def generate_clean_parametric_dataset(
    output_dir: str,
    dataset_size: int = 50,
    img_size: int = 512,
) -> None:
    if dataset_size != 50:
        raise ValueError("This generator is designed for exactly 50 samples.")

    images_dir = os.path.join(output_dir, "images")
    gt_dir = os.path.join(output_dir, "gt")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    summary: List[Dict] = []
    print("=" * 72)
    print("Generating clean parametric closed-shape dataset")
    print(f"size={dataset_size}, img={img_size}, out={output_dir}")
    print("=" * 72)

    for i in range(dataset_size):
        level, tags, segs = _build_segments_for_index(i, img_size)
        contour = _segments_to_points(segs)
        # 强制无自交：若有交叉，退化为凸包闭合线（保持单闭合、无填充）
        if _has_self_intersection(contour):
            pts_i = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts_i).reshape(-1, 2).astype(np.float64)
            if len(hull) >= 3:
                segs = _polyline_to_line_segments([(float(x), float(y)) for x, y in hull.tolist()], samples=16)
                contour = _segments_to_points(segs)
                tags = tags + ["non_intersecting_hull_fix"]
        contour_i = np.round(contour).astype(np.int32)
        contour_i[:, 0] = np.clip(contour_i[:, 0], 0, img_size - 1)
        contour_i[:, 1] = np.clip(contour_i[:, 1], 0, img_size - 1)

        # 纯净图像：白底黑色单闭合线（不填充）
        img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
        # 关闭抗锯齿，减少二值化后毛刺/分叉，避免关键点虚增
        cv2.polylines(img, [contour_i], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_8)

        name = f"clean_{i:05d}"
        img_path = os.path.join(images_dir, f"{name}.png")
        gt_path = os.path.join(gt_dir, f"{name}.json")
        cv2.imwrite(img_path, img)

        gt = {
            "name": name,
            "complexity_level": level,
            "single_closed_shape": True,
            "noise_free": True,
            "structure_tags": tags,
            "segments": [{"type": s.type, **s.params} for s in segs],
            "segment_count": len(segs),
            "contour_point_count": int(len(contour_i)),
            "shape_bbox": {
                "xmin": int(np.min(contour_i[:, 0])),
                "ymin": int(np.min(contour_i[:, 1])),
                "xmax": int(np.max(contour_i[:, 0])),
                "ymax": int(np.max(contour_i[:, 1])),
            },
        }
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)

        summary.append(
            {
                "name": name,
                "image_path": img_path,
                "gt_path": gt_path,
                "complexity_level": level,
                "structure_tags": tags,
                "segment_count": len(segs),
            }
        )

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    levels = {}
    for s in summary:
        lv = s["complexity_level"]
        levels[lv] = levels.get(lv, 0) + 1
    print("Done. Complexity distribution:")
    for lv in sorted(levels):
        print(f"  level {lv}: {levels[lv]}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean 50-sample parametric closed-shape dataset")
    parser.add_argument("--output", type=str, default="clean_parametric_dataset_50", help="Output directory")
    parser.add_argument("--size", type=int, default=50, help="Dataset size (must be 50)")
    parser.add_argument("--img-size", type=int, default=512, help="Image size")
    args = parser.parse_args()

    generate_clean_parametric_dataset(
        output_dir=args.output,
        dataset_size=args.size,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()

