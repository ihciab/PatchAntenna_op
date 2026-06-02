"""
在 clean_parametric_dataset_50 上做“参数化分段”专项评测。

优化目标（与矢量导出一致）：在**高精度拟合**前提下，使分段与 **GT 参考分段**
尽量一致，并使**有效控制量（effective_params）尽量少**。

**总 effective_params（本评测中的 pred / gt 标量）**  
与 `vtracer_python.SemanticSegment.effective_params` 一致，按段累加，表示整条轮廓矢量
的「有效自由度」近似：直线段约 1，圆弧约 2，样条段为内部 B 样条控制量相关的加权计数
（越大表示曲线表示越复杂）。`effective_params_pred` 为各预测 primitive 之和；
`effective_params_gt` 为对 GT JSON 各段的启发式估计，用于简洁度对比，不等于 CST 里
的真实控制点数，但与 VTracer 段级复杂度同量级。

核心指标（不是整条轮廓整体误差）：
1) 分段点：GT 与预测分段边界位置是否一致
2) 分段类型：line/arc/spline 序列（LCS）
3) 简洁度：预测各段 effective_params 之和相对 GT 启发式下界的超出程度

默认会为每个样本写出 `*_segmentation_result.png`（原图左 + 预测分段右）；仅大批量扫参时用
`--no-png` 跳过以省磁盘与时间。

定方案与管线说明见仓库文档
`docs/clean_parametric_中心线语义分段方案.md`（推荐 `TraceConfig`、合并阶段与自适应规则）。

命令行对比两份已生成的报告（不写新评测）::

    python tests/test_clean_parametric_segmentation_validation.py \\
        --diff-report path/to/baseline_report.json path/to/tuned_report.json

管线 ``--pipeline optimized_bs_vtracer``：与 ``test_curve_fit_metamaterial_antenna`` 一致，先
``ImageInitializer`` + ``OptimizedBSplineFitter`` 得到拟合折线写入 ``.npy``，再以居中图为输入、
种子折线跳过骨架，走 ``vtracer_python`` 语义分段/DP；评测指标与 PNG 均基于该**整条链路**输出。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 文件用途说明
# ---------------------------------------------------------------------------
# 这个文件是 clean_parametric_dataset_50 的“参数化分段评估器”。
#
# 它不直接实现底层参数化算法，而是负责：
# 1. 读取数据集 summary.json 中的样本列表；
# 2. 读取每个样本的 GT JSON，也就是参考几何分段；
# 3. 调用 vtracer_python.VTracerPython 对图片做中心线提取和语义分段；
# 4. 从 VTracer 写出的 semantic_debug.json 中取出预测 primitive；
# 5. 对比 GT 与预测结果：
#    - 分段边界位置是否接近；
#    - 分段类型序列 line / arc / spline 是否一致；
#    - 预测参数数量 effective_params 是否足够简洁；
# 6. 输出 clean_parametric_segmentation_report.json 和可视化 PNG。
#
# 换句话说：
# - “参数化算法本体”在 vtracer_python.py；
# - “这个文件”是评估算法表现的测试/报告脚本。
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_ROOT = os.path.dirname(ROOT)
for _path in (ROOT, PARENT_ROOT, os.path.join(ROOT, "AutoCAD_v8.5.4"), os.path.join(PARENT_ROOT, "AutoCAD_v8.5.4")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Windows：joblib/loky 枚举物理核失败时反复 UserWarning；须在导入 sklearn 链（vtracer）之前设定
if os.environ.get("LOKY_MAX_CPU_COUNT") is None:
    try:
        import multiprocessing as _mp

        os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, int(_mp.cpu_count() or 4)))
    except Exception:
        os.environ["LOKY_MAX_CPU_COUNT"] = "4"

from bayesian_optimization.tools.vtracer_python import TraceConfig, VTracerPython  # noqa: E402
from tools.dataset_generator.clean_parametric_dataset_generator import Segment, _segments_to_points  # noqa: E402


@dataclass
class SampleEval:
    """单个样本的评估结果。

    每处理一张图片，就会生成一个 SampleEval。最后 run_eval 会把所有
    SampleEval 汇总成 report JSON。

    重要字段：
    - gt_segments / pred_segments：GT 与预测的分段数量；
    - breakpoints_mae_ratio：GT 分段点到最近预测分段点的归一化平均距离；
    - breakpoints_count_error：分段数量差；
    - type_lcs_score：line/arc/spline 类型序列的 LCS 相似度；
    - effective_params_*：有效参数数量，越小越简洁；
    - param_score：综合分数。
    """

    name: str
    success: bool
    gt_segments: int
    pred_segments: int
    breakpoints_mae_ratio: float
    breakpoints_count_error: int
    type_lcs_score: float
    param_score: float
    effective_params_gt: int = 0
    effective_params_pred: int = 0
    parsimony_score: float = 0.0
    elapsed_sec: float = 0.0
    result_vis_path: str = ""
    error: str = ""
    pipeline: str = "vtracer"


def _norm_type(t: str) -> str:
    """把不同来源的类型名规范成 line / arc / spline 三类。"""

    t = t.lower().strip()
    if t in ("line",):
        return "line"
    if t in ("arc", "circle"):
        return "arc"
    if t in ("bspline", "spline", "curve"):
        return "spline"
    return t


def _boundary_ratios_from_gt_segments(gt_segments: List[Dict[str, Any]]) -> Tuple[List[float], List[str]]:
    """从 GT JSON 还原分段边界在整条轮廓上的比例位置。

    GT 里每段是几何 primitive，例如 line / arc / bspline。
    这里用数据集生成器的 _segments_to_points 对每段重新采样，
    再根据每段采样点数量估计该段在整条轮廓中的长度占比。
    """

    seg_objs = [Segment(type=s["type"], params={k: v for k, v in s.items() if k != "type"}) for s in gt_segments]
    # 通过每段采样点数量恢复分段边界在全轮廓中的位置（比例）
    seg_points: List[np.ndarray] = []
    for seg in seg_objs:
        pts = _segments_to_points([seg])
        seg_points.append(pts)

    lengths = [max(1, len(p)) for p in seg_points]
    total = int(sum(lengths))
    acc = 0
    boundaries = []
    types = []
    for idx, s in enumerate(gt_segments):
        boundaries.append(acc / max(1, total))
        types.append(_norm_type(str(s["type"])))
        acc += lengths[idx]
    return boundaries, types


def _boundary_ratios_from_pred_primitives(primitives: List[Dict[str, Any]], n_points: int) -> Tuple[List[float], List[str]]:
    """从 VTracer 预测 primitive 中提取分段边界比例和类型序列。

    semantic_debug.json 会记录每个 primitive 的 start_idx/end_idx。
    这里把 start_idx 除以重采样点总数，得到可与 GT 比较的边界比例。
    """

    if n_points <= 0:
        return [], []
    boundaries = []
    types = []
    for p in primitives:
        s = int(p.get("start_idx", 0))
        boundaries.append(float(s) / float(max(1, n_points)))
        types.append(_norm_type(str(p.get("kind", ""))))
    return boundaries, types


def _breakpoint_mae(gt_b: List[float], pr_b: List[float]) -> float:
    """计算 GT 分段边界到最近预测边界的平均距离。"""

    if not gt_b and not pr_b:
        return 0.0
    if not gt_b or not pr_b:
        return 1.0
    errs = []
    for g in gt_b:
        errs.append(min(abs(g - p) for p in pr_b))
    return float(np.mean(errs))


def _gt_segment_effective_estimate(seg: Dict[str, Any]) -> int:
    # 估算 GT 段的有效参数量。这个值不是 CST 真实参数个数，
    # 而是为了和 VTracer 的 SemanticSegment.effective_params 同量纲比较：
    # line≈1，arc/circle≈2，spline 按控制点规模估算。
    """与 VTracer `effective_params` 同量级：line≈1，arc/circle≈2，bspline≈控制点规模。"""
    t = str(seg.get("type", "")).lower()
    if t == "line":
        return 1
    if t in ("arc", "circle"):
        return 2
    if t in ("bspline", "spline"):
        ctrl_keys = [k for k in seg if isinstance(k, str) and len(k) >= 2 and k[0] == "p" and k[1:].isdigit()]
        ncp = len(ctrl_keys) if ctrl_keys else 4
        return int(max(4, min(ncp, 24)))
    return 3


def _gt_effective_params_heuristic(gt_segments: List[Dict[str, Any]]) -> int:
    """把所有 GT 段的有效参数量累加，作为简洁度比较下界。"""

    if not gt_segments:
        return 1
    return max(1, sum(_gt_segment_effective_estimate(s) for s in gt_segments))


def _pred_effective_params_sum(primitives: List[Dict[str, Any]]) -> int:
    """累加预测 primitive 的 effective_params。"""

    return int(sum(int(p.get("effective_params", 0) or 0) for p in primitives))


def _parsimony_score(pred_eff: int, gt_eff: int) -> float:
    # 简洁度得分：预测参数量不超过 GT 启发式下界时给满分；
    # 超出越多，得分越低。
    """预测总有效参量相对 GT 启发式下界：超出越多分越低；不惩罚 pred<=gt。"""
    if pred_eff <= gt_eff:
        return 1.0
    excess = (pred_eff - gt_eff) / float(max(gt_eff, 1))
    return max(0.0, 1.0 - min(1.0, excess * 0.32))


def _lcs_ratio(a: List[str], b: List[str]) -> float:
    """计算两个分段类型序列的最长公共子序列比例。"""

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return float(dp[n][m]) / float(max(n, m))


def _slice_segment_points(points: np.ndarray, start_idx: int, end_idx: int, closed: bool) -> np.ndarray:
    """按照 start_idx/end_idx 从一条折线或闭合环中切出某一段点。"""

    n = len(points)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    s = int(start_idx) % n
    e = int(end_idx) % n
    if closed:
        if s <= e:
            return points[s : e + 1]
        return np.vstack([points[s:], points[: e + 1]])
    s = max(0, min(n - 1, s))
    e = max(0, min(n - 1, e))
    if s <= e:
        return points[s : e + 1]
    return points[e : s + 1]


def _draw_result_visualization(
    image_path: str,
    out_png_path: str,
    points: np.ndarray,
    primitives: List[Dict[str, Any]],
    closed: bool,
    sample_name: str,
) -> bool:
    # 写出单个样本的预测分段可视化 PNG。
    # 输出图是左右拼接：左侧是输入图，右侧是预测分段结果。
    # 不同 primitive 类型用不同颜色，方便人工排查过分段、欠分段和类型误判。
    """写出左右对比 PNG；LINE_8 略快于抗锯齿。成功返回 True。"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    overlay = np.full((h, w, 3), 255, dtype=np.uint8)

    color_map = {
        "line": (30, 144, 255),    # blue-ish
        "arc": (0, 165, 255),      # orange
        "spline": (50, 205, 50),   # green
    }

    # 画预测分段线（按段类型着色）
    for prim in primitives:
        kind = _norm_type(str(prim.get("kind", "")))
        clr = color_map.get(kind, (100, 100, 100))
        s = int(prim.get("start_idx", 0))
        e = int(prim.get("end_idx", 0))
        seg_pts = _slice_segment_points(points, s, e, closed=closed)
        if len(seg_pts) >= 2:
            pi = np.round(seg_pts).astype(np.int32)
            pi[:, 0] = np.clip(pi[:, 0], 0, w - 1)
            pi[:, 1] = np.clip(pi[:, 1], 0, h - 1)
            cv2.polylines(overlay, [pi], isClosed=False, color=clr, thickness=2, lineType=cv2.LINE_8)

    # 画分段点
    for prim in primitives:
        for key, c in (("start_idx", (0, 0, 255)), ("end_idx", (128, 0, 255))):
            idx = int(prim.get(key, 0)) % max(1, len(points))
            p = points[idx]
            x = int(round(np.clip(p[0], 0, w - 1)))
            y = int(round(np.clip(p[1], 0, h - 1)))
            cv2.circle(overlay, (x, y), 3, c, -1, lineType=cv2.LINE_8)

    # 图例
    lt = cv2.LINE_8
    cv2.putText(overlay, "pred line", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_map["line"], 2, lt)
    cv2.putText(overlay, "pred arc", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_map["arc"], 2, lt)
    cv2.putText(overlay, "pred spline", (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_map["spline"], 2, lt)
    cv2.putText(overlay, "segment points", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 0, 200), 2, lt)

    # 拼图：原图 + 结果
    left = img.copy()
    cv2.putText(left, f"{sample_name} | input", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, lt)
    cv2.putText(overlay, f"{sample_name} | predicted segmentation", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, lt)
    panel = np.hstack([left, overlay])
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    return bool(cv2.imwrite(out_png_path, panel))


def _largest_optimized_bs_fitting_polyline(contours_dict: Dict[str, Any]) -> np.ndarray:
    # 从 OptimizedBSplineFitter 的结果中取最长 fitting 折线。
    # optimized_bs_vtracer 管线会把这条折线作为 VTracer 的 centerline_seed。
    """从 OptimizedBSplineFitter.get_contours_dict() 中取 fitting 点最多的轮廓。"""
    best = np.zeros((0, 2), dtype=np.float64)
    best_n = 0
    for _cid, data in contours_dict.items():
        if not isinstance(data, dict):
            continue
        fit = data.get("fitting") or {}
        fp = fit.get("points")
        if fp is None:
            continue
        arr = np.asarray(fp, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 4:
            continue
        if len(arr) > best_n:
            best_n = len(arr)
            best = arr
    return best


def _ensure_geomdl_available_for_optimized_bspline() -> bool:
    """Make OptimizedBSplineFitter importable even when optional geomdl is absent.

    The optimized_bs path only needs a fitted polyline seed for VTracer.
    Some helper modules imported by OptimizedBSplineFitter compare Arc vs NURBS
    and import geomdl at module import time.  If geomdl is not installed, this
    local fallback provides the tiny API surface those comparisons need.  It is
    intentionally scoped to this experimental validation path and does not
    change the default vtracer pipeline.

    Returns:
        True if a local fallback stub was installed, False if real geomdl exists.
    """

    try:
        import geomdl  # noqa: F401,WPS433
        return False
    except ModuleNotFoundError:
        pass

    class _FallbackCurve:
        def __init__(self, normalize_kv: bool = True):
            self.normalize_kv = normalize_kv
            self.degree = 3
            self.ctrlpts: List[List[float]] = []
            self.knotvector: List[float] = []
            self.weights: List[float] = []
            self.sample_size = 50

        @property
        def evalpts(self) -> List[List[float]]:
            pts = np.asarray(self.ctrlpts, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
                return []
            if len(pts) == 1:
                return pts[:, :2].tolist()
            seg_len = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
            total = float(cumulative[-1])
            if total <= 1e-9:
                return pts[:, :2].tolist()
            query = np.linspace(0.0, total, max(2, int(self.sample_size)))
            x = np.interp(query, cumulative, pts[:, 0])
            y = np.interp(query, cumulative, pts[:, 1])
            return np.column_stack([x, y]).tolist()

    def _generate_knot_vector(degree: int, ctrlpt_count: int) -> List[float]:
        n = max(1, int(ctrlpt_count) + int(degree) + 1)
        return np.linspace(0.0, 1.0, n).tolist()

    def _check_knot_vector(_degree: int, _knots: List[float], _ctrlpt_count: int) -> bool:
        return True

    geomdl_mod = types.ModuleType("geomdl")
    nurbs_mod = types.ModuleType("geomdl.NURBS")
    utilities_mod = types.ModuleType("geomdl.utilities")
    knotvector_mod = types.ModuleType("geomdl.knotvector")

    nurbs_mod.Curve = _FallbackCurve
    utilities_mod.generate_knot_vector = _generate_knot_vector
    knotvector_mod.check = _check_knot_vector

    geomdl_mod.NURBS = nurbs_mod
    geomdl_mod.utilities = utilities_mod
    geomdl_mod.knotvector = knotvector_mod

    sys.modules.setdefault("geomdl", geomdl_mod)
    sys.modules.setdefault("geomdl.NURBS", nurbs_mod)
    sys.modules.setdefault("geomdl.utilities", utilities_mod)
    sys.modules.setdefault("geomdl.knotvector", knotvector_mod)
    return True


def _resolve_dataset_file(sample: Dict[str, Any], dataset_dir: str, key: str, subdir: str, suffix: str) -> str:
    """解析样本图片或 GT 路径。

    summary.json 里可能保存的是生成数据集时的绝对路径。
    如果该路径在当前机器不存在，就退回到当前 dataset_dir 下的本地相对路径。
    """

    path = str(sample.get(key, ""))
    if path and os.path.isfile(path):
        return path
    name = str(sample.get("name", ""))
    local_path = os.path.join(dataset_dir, subdir, f"{name}{suffix}")
    if os.path.isfile(local_path):
        return local_path
    return path


def _prepare_optimized_bspline_seed(image_path: str, sample_out: str) -> Tuple[str, str]:
    # 为 optimized_bs_vtracer 管线准备中心线种子：
    # 1. ImageInitializer 读图、居中、提取边缘；
    # 2. OptimizedBSplineFitter 对边缘轮廓做 B 样条拟合；
    # 3. 取最长 fitting polyline 保存为 .npy；
    # 4. 后续 VTracer 读取该 .npy，跳过骨架化，直接做语义分段。
    """与 test_curve_fit_metamaterial_antenna 一致：ImageInitializer + OptimizedBSplineFitter，写出居中图与种子折线。"""
    geomdl_stubbed = _ensure_geomdl_available_for_optimized_bspline()

    from core.image.initializer import ImageInitializer as ImageInit  # noqa: WPS433
    from core.geometry.optimized_bspline_fitter import OptimizedBSplineFitter  # noqa: WPS433

    os.makedirs(sample_out, exist_ok=True)
    ii = ImageInit(image_path, show=False, save="")
    img = ii.centered_img()
    edges = ii.edges()
    if img is None or edges is None:
        raise RuntimeError("ImageInitializer 未产生 centered_img / edges")
    centered_png = os.path.join(sample_out, "pipeline_centered_input.png")
    if not cv2.imwrite(centered_png, img):
        raise RuntimeError(f"无法写入居中图: {centered_png}")

    fitter = OptimizedBSplineFitter(
        img=img,
        edges=edges,
        line_threshold=2.0,
        arc_threshold=2.0,
        curvature_threshold=0.15,
        spline_degree=3,
        show=False,
        save="",
    )
    contours_dict = fitter.get_contours_dict()
    seed = _largest_optimized_bs_fitting_polyline(contours_dict)
    if len(seed) < 4:
        raise RuntimeError("OptimizedBSplineFitter 未得到足够长的 fitting 折线")
    seed_npy = os.path.join(sample_out, "optimized_bs_seed_polyline.npy")
    np.save(seed_npy, seed.astype(np.float64))
    meta_path = os.path.join(sample_out, "optimized_bs_seed_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_image": image_path,
                "centered_image": centered_png,
                "seed_npy": seed_npy,
                "seed_points": int(len(seed)),
                "contour_count": int(len(contours_dict)),
                "geomdl_stubbed": bool(geomdl_stubbed),
                "note": (
                    "This experimental path first runs OptimizedBSplineFitter, "
                    "then passes its longest fitting polyline to VTracerPython "
                    "via TraceConfig.centerline_seed_npy_path."
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return centered_png, seed_npy


def _eval_single(
    sample: Dict[str, Any],
    output_dir: str,
    keep_debug: bool = False,
    save_png: bool = True,
    tuned: bool = True,
    dp_boundary_field_weight: float | None = None,
    boundary_candidate_merge_gap: int | None = None,
    pipeline: str = "vtracer",
) -> SampleEval:
    """评估单个 clean_parametric_dataset_50 样本。

    单样本流程：
    1. 读取 GT JSON；
    2. 根据 pipeline 准备输入图或中心线 seed；
    3. 构造 TraceConfig；
    4. 调用 VTracerPython.to_svg 执行中心线参数化；
    5. 从 intermediates/component_*/semantic_debug.json 读取预测 primitive；
    6. 计算分段边界、类型序列、参数简洁度；
    7. 可选写出可视化 PNG；
    8. 返回 SampleEval。
    """

    name = sample["name"]
    gt_path = sample["gt_path"]
    image_path = sample["image_path"]
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    gt_segments = gt.get("segments", [])
    gt_boundary_ratio, gt_types = _boundary_ratios_from_gt_segments(gt_segments)

    sample_out = os.path.join(output_dir, name)
    os.makedirs(sample_out, exist_ok=True)
    inter_dir = os.path.join(sample_out, "intermediates")
    svg_path = os.path.join(sample_out, f"{name}.svg")
    metrics_path = os.path.join(sample_out, f"{name}_metrics.json")

    t0 = time.perf_counter()
    vis_image_path = image_path
    trace_image_path = image_path
    centerline_seed: str | None = None

    if pipeline == "optimized_bs_vtracer":
        # 可选管线：先用 OptimizedBSplineFitter 生成中心线种子，
        # 再把种子交给 VTracer 做语义分段。
        # 如果这一步失败，直接返回失败样本，不再进入 VTracer。
        try:
            trace_image_path, centerline_seed = _prepare_optimized_bspline_seed(image_path, sample_out)
            vis_image_path = trace_image_path
        except Exception as prep_e:
            gt_eff = _gt_effective_params_heuristic(gt_segments)
            return SampleEval(
                name,
                False,
                len(gt_types),
                0,
                1.0,
                len(gt_types),
                0.0,
                0.0,
                effective_params_gt=gt_eff,
                effective_params_pred=0,
                parsimony_score=0.0,
                elapsed_sec=round(time.perf_counter() - t0, 4),
                error=f"optimized_bs_prep: {prep_e}",
                pipeline=pipeline,
            )

    cfg = TraceConfig(
        image_path=trace_image_path,
        color_mode="bw",
        trace_style="centerline",
        mode="spline",
        metrics_path=metrics_path,
        save_intermediates=inter_dir,
        fit_tolerance=1.5,
        resample_step=3.0,
        filter_speckle=4,
        centerline_seed_npy_path=centerline_seed,
    )
    if tuned:
        # tuned=True 时使用当前较优的一组语义分段参数。
        # 这些参数主要影响中心线重采样、关键点检测、DP 分段复杂度惩罚、
        # 直线/圆弧合并阈值和样条控制点惩罚。
        cfg.fit_tolerance = 1.5
        cfg.resample_step = 3.0
        cfg.filter_speckle = 4
        cfg.gaussian_sigma = 1.05
        cfg.semantic_window_size = 12
        cfg.keypoint_angle_threshold_deg = 32.0
        cfg.keypoint_refine_radius = 5
        cfg.keypoint_use_model_guided = True
        cfg.keypoint_model_multiscale_votes = False
        cfg.dp_complexity_weight = 1.02
        cfg.dp_max_segment_points = 78
        cfg.line_merge_angle_deg = 13.5
        cfg.arc_radius_rel_tol = 0.22
        cfg.arc_center_tol = 2.2
        cfg.arc_min_sweep_deg = 17.0
        cfg.spline_ctrl_penalty = 0.11
    if dp_boundary_field_weight is not None:
        cfg.dp_boundary_field_weight = float(dp_boundary_field_weight)
    if boundary_candidate_merge_gap is not None:
        cfg.boundary_candidate_merge_max_polyline_gap = int(boundary_candidate_merge_gap)
    try:
        # 这里是真正调用参数化算法的地方。
        # VTracerPython.to_svg 内部完成：
        # 图像二值化/骨架或 seed polyline -> 重采样 -> 语义分段 -> SVG 输出。
        tracer = VTracerPython(cfg)
        tracer.to_svg(svg_path)
    except Exception as e:
        gt_eff = _gt_effective_params_heuristic(gt_segments)
        return SampleEval(
            name,
            False,
            len(gt_types),
            0,
            1.0,
            len(gt_types),
            0.0,
            0.0,
            effective_params_gt=gt_eff,
            effective_params_pred=0,
            parsimony_score=0.0,
            elapsed_sec=round(time.perf_counter() - t0, 4),
            error=str(e),
            pipeline=pipeline,
        )

    # 读取 semantic_debug（每个组件文件夹）
    primitives: List[Dict[str, Any]] = []
    n_points = 0
    ref_points = np.zeros((0, 2), dtype=np.float64)
    is_closed = True
    if os.path.isdir(inter_dir):
        comp_dirs = sorted([d for d in os.listdir(inter_dir) if d.startswith("component_")])
        for cd in comp_dirs:
            dbg_path = os.path.join(inter_dir, cd, "semantic_debug.json")
            if not os.path.isfile(dbg_path):
                continue
            with open(dbg_path, "r", encoding="utf-8") as f:
                dbg = json.load(f)
            prim = dbg.get("primitives") or []
            pts = dbg.get("resampled_points") or []
            closed = bool(dbg.get("closed", True))
            if len(prim) > len(primitives):
                primitives = prim
                n_points = len(pts)
                ref_points = np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2), dtype=np.float64)
                is_closed = closed

    if not keep_debug and os.path.isdir(inter_dir):
        shutil.rmtree(inter_dir, ignore_errors=True)

    pred_boundary_ratio, pred_types = _boundary_ratios_from_pred_primitives(primitives, n_points)
    bp_mae = _breakpoint_mae(gt_boundary_ratio, pred_boundary_ratio)
    cnt_err = abs(len(gt_boundary_ratio) - len(pred_boundary_ratio))
    type_score = _lcs_ratio(gt_types, pred_types)
    gt_eff = _gt_effective_params_heuristic(gt_segments)
    pred_eff = _pred_effective_params_sum(primitives)
    parsimony = _parsimony_score(pred_eff, gt_eff)

    bp_score = max(0.0, 1.0 - min(1.0, bp_mae / 0.08) - min(0.4, 0.06 * cnt_err))
    param_score = 0.52 * bp_score + 0.30 * type_score + 0.18 * parsimony

    vis_path = ""
    if save_png and len(ref_points) > 0 and len(primitives) > 0:
        cand = os.path.join(sample_out, f"{name}_segmentation_result.png")
        if _draw_result_visualization(
            image_path=vis_image_path,
            out_png_path=cand,
            points=ref_points,
            primitives=primitives,
            closed=is_closed,
            sample_name=name,
        ):
            vis_path = cand

    return SampleEval(
        name=name,
        success=True,
        gt_segments=len(gt_types),
        pred_segments=len(pred_types),
        breakpoints_mae_ratio=bp_mae,
        breakpoints_count_error=cnt_err,
        type_lcs_score=type_score,
        param_score=param_score,
        effective_params_gt=gt_eff,
        effective_params_pred=pred_eff,
        parsimony_score=parsimony,
        elapsed_sec=round(time.perf_counter() - t0, 4),
        result_vis_path=vis_path,
        pipeline=pipeline,
    )


def run_eval(
    dataset_dir: str,
    output_dir: str,
    max_samples: int | None = None,
    keep_debug: bool = False,
    save_png: bool = True,
    tuned: bool = True,
    jobs: int = 1,
    dp_boundary_field_weight: float | None = None,
    boundary_candidate_merge_gap: int | None = None,
    pipeline: str = "vtracer",
) -> Dict[str, Any]:
    """在整个 clean_parametric_dataset_50 上运行评估。

    - dataset_dir：数据集目录，包含 summary.json / images / gt；
    - output_dir：报告、SVG、PNG、metrics 的输出目录；
    - max_samples：只跑前 N 个样本，调参时很有用；
    - keep_debug：是否保留 VTracer 中间文件；
    - save_png：是否保存每个样本的可视化；
    - tuned：是否使用调优参数；
    - jobs：并行进程数；
    - pipeline：vtracer 或 optimized_bs_vtracer。
    """

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(dataset_dir, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    for sample in samples:
        sample["image_path"] = _resolve_dataset_file(sample, dataset_dir, "image_path", "images", ".png")
        sample["gt_path"] = _resolve_dataset_file(sample, dataset_dir, "gt_path", "gt", ".json")
    if max_samples is not None:
        samples = samples[:max_samples]

    rows: List[SampleEval]
    if jobs <= 1:
        # 顺序执行，最适合调试；异常和日志更容易看。
        rows = []
        for i, s in enumerate(samples):
            rows.append(
                _eval_single(
                    s,
                    output_dir=output_dir,
                    keep_debug=keep_debug,
                    save_png=save_png,
                    tuned=tuned,
                    dp_boundary_field_weight=dp_boundary_field_weight,
                    boundary_candidate_merge_gap=boundary_candidate_merge_gap,
                    pipeline=pipeline,
                )
            )
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(samples)}] done")
    else:
        # 并行执行，用于批量扫参或全量评估。
        # Windows 下进程池会重新导入模块，因此 worker 逻辑放在
        # seg_eval_parallel_worker.py 中，避免闭包/局部函数无法 pickle。
        from bayesian_optimization.tools import seg_eval_parallel_worker as _pool

        jw = max(1, min(int(jobs), 32, len(samples)))
        payloads: List[Tuple[Dict[str, Any], str, bool, bool, bool, float | None, int | None, str]] = [
            (s, output_dir, keep_debug, save_png, tuned, dp_boundary_field_weight, boundary_candidate_merge_gap, pipeline)
            for s in samples
        ]
        print(f"parallel eval ProcessPoolExecutor max_workers={jw} (set --jobs 1 for sequential)")
        with ProcessPoolExecutor(max_workers=jw) as ex:
            rows = [SampleEval(**d) for d in ex.map(_pool.run_one_sample, payloads)]
        print(f"[{len(rows)}/{len(samples)}] done (parallel)")

    ok = [r for r in rows if r.success]
    fail = [r for r in rows if not r.success]
    report = {
        "dataset_dir": dataset_dir,
        "pipeline": pipeline,
        "boundary_candidate_merge_gap": boundary_candidate_merge_gap,
        "total": len(rows),
        "success": len(ok),
        "failed": len(fail),
        "metric_desc": {
            "breakpoints_mae_ratio": "GT分段点到预测分段点的最小距离平均（归一化比例，越小越好）",
            "breakpoints_count_error": "分段数量偏差 abs(gt - pred)",
            "type_lcs_score": "分段类型序列LCS归一化得分（line/arc/spline，越大越好）",
            "effective_params_pred_mean": "各样本预测 primitive 的 effective_params 之和的平均（段复杂度总和）",
            "effective_params_gt_mean": "GT 段 effective_params 启发式之和的平均（对比用下界）",
            "parsimony_score": "预测 effective_params 相对 GT 启发式下界：超出越少越好",
            "param_score": "综合分：边界52% + 类型30% + 简洁度18%（高精度下少控制量）",
        },
        "aggregate": {
            "breakpoints_mae_ratio_mean": float(np.mean([r.breakpoints_mae_ratio for r in ok])) if ok else None,
            "breakpoints_count_error_mean": float(np.mean([r.breakpoints_count_error for r in ok])) if ok else None,
            "type_lcs_score_mean": float(np.mean([r.type_lcs_score for r in ok])) if ok else None,
            "parsimony_score_mean": float(np.mean([r.parsimony_score for r in ok])) if ok else None,
            "effective_params_pred_mean": float(np.mean([r.effective_params_pred for r in ok])) if ok else None,
            "effective_params_gt_mean": float(np.mean([r.effective_params_gt for r in ok])) if ok else None,
            "param_score_mean": float(np.mean([r.param_score for r in ok])) if ok else None,
            "param_score_p50": float(np.percentile([r.param_score for r in ok], 50)) if ok else None,
            "param_score_p90": float(np.percentile([r.param_score for r in ok], 90)) if ok else None,
            "elapsed_sec_mean": float(np.mean([r.elapsed_sec for r in ok])) if ok else None,
        },
        "samples": [r.__dict__ for r in rows],
    }
    if pipeline == "optimized_bs_vtracer":
        report["coordinate_note"] = (
            "本管线使用 ImageInitializer 居中后的 BGR 图与边缘；VTracer 输入为居中图 + OptimizedBSpline 拟合折线种子。"
            "param_score 仍按沿重采样折线的边界比例与 GT 对齐，与纯 vtracer（原图骨架）数值不宜直接等同为「谁更准」。"
        )
    out = os.path.join(output_dir, "clean_parametric_segmentation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report: {out}")
    return report


def run_eval_optimized_bs_vtracer(
    dataset_dir: str,
    output_dir: str,
    max_samples: int | None = None,
    keep_debug: bool = True,
    save_png: bool = True,
    tuned: bool = True,
    jobs: int = 1,
    dp_boundary_field_weight: float | None = None,
    boundary_candidate_merge_gap: int | None = None,
) -> Dict[str, Any]:
    """Experimental two-stage parameterization evaluation.

    This function is the safe trial entry requested for the new flow:

    1. Read each clean_parametric_dataset_50 sample.
    2. Run ImageInitializer + OptimizedBSplineFitter, matching the optimized_bs
       path used in test_curve_fit_metamaterial_antenna.py.
    3. Save the longest fitted B-spline polyline as optimized_bs_seed_polyline.npy.
    4. Pass that seed into VTracerPython with TraceConfig.centerline_seed_npy_path.
    5. Let the existing VTracer semantic DP classify the seed into line/arc/spline.

    The original run_eval(..., pipeline="vtracer") path is unchanged.  Use this
    wrapper when you want an explicit experiment without touching the existing
    default pipeline.
    """

    return run_eval(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        max_samples=max_samples,
        keep_debug=keep_debug,
        save_png=save_png,
        tuned=tuned,
        jobs=jobs,
        dp_boundary_field_weight=dp_boundary_field_weight,
        boundary_candidate_merge_gap=boundary_candidate_merge_gap,
        pipeline="optimized_bs_vtracer",
    )


def diff_segmentation_reports(path_a: str, path_b: str, label_a: str = "baseline", label_b: str = "tuned") -> None:
    # 比较两份评估报告，只读已有 JSON，不重新跑评估。
    # 常用于比较 baseline / tuned 或不同参数组合，并找出分数变化最大的样本。
    """打印两份评测 JSON 的 aggregate 差与 param_score 差异最大的若干样本。"""
    with open(path_a, "r", encoding="utf-8") as f:
        rep_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        rep_b = json.load(f)
    agg_a = rep_a.get("aggregate") or {}
    agg_b = rep_b.get("aggregate") or {}
    print(f"pipelines: A={rep_a.get('pipeline')!r}  B={rep_b.get('pipeline')!r}")
    if rep_a.get("coordinate_note") or rep_b.get("coordinate_note"):
        if rep_a.get("coordinate_note"):
            print(f"A coordinate_note: {rep_a['coordinate_note']}")
        if rep_b.get("coordinate_note"):
            print(f"B coordinate_note: {rep_b['coordinate_note']}")
    keys = sorted(k for k in agg_a if isinstance(agg_a.get(k), (int, float)) and agg_a.get(k) is not None)
    print(f"{'metric':<42} {label_a:>12} {label_b:>12} {'delta':>10}")
    for k in keys:
        va, vb = float(agg_a[k]), float(agg_b[k])
        print(f"{k:<42} {va:12.6f} {vb:12.6f} {vb - va:10.6f}")
    sa = {s["name"]: s for s in rep_a.get("samples", [])}
    sb = {s["name"]: s for s in rep_b.get("samples", [])}
    deltas: List[Tuple[str, float, float, float]] = []
    for name in sa:
        if name not in sb:
            continue
        pa, pb = float(sa[name].get("param_score", 0)), float(sb[name].get("param_score", 0))
        deltas.append((name, pa, pb, pb - pa))
    deltas.sort(key=lambda t: t[3])
    print("\nLargest param_score drops (first 12, B vs A):")
    for name, pa, pb, d in deltas[:12]:
        print(f"  {name}: {pa:.4f} -> {pb:.4f} ({d:+.4f})")
    print("\nLargest param_score gains (first 8):")
    for name, pa, pb, d in deltas[-8:][::-1]:
        print(f"  {name}: {pa:.4f} -> {pb:.4f} ({d:+.4f})")


def main() -> None:
    """命令行入口。

    常用命令：
    - 跑全部样本：
      python test_clean_parametric_segmentation_validation.py

    - 只跑前 10 个，不保存 PNG：
      python test_clean_parametric_segmentation_validation.py --max-samples 10 --no-png

    - 比较两份报告：
      python test_clean_parametric_segmentation_validation.py --diff-report baseline.json tuned.json
    """

    parser = argparse.ArgumentParser(description="Clean parametric dataset segmentation validation")
    parser.add_argument("--dataset", type=str, default=os.path.join(ROOT, "clean_parametric_dataset_50"))
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "clean_parametric_eval_output"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--keep-debug", action="store_true")
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="不写入 *_segmentation_result.png（默认写入每个样本的左右对比图）",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="使用基线 TraceConfig（较松）；默认 tuned=高精度+少控制量+圆弧合并",
    )
    parser.add_argument(
        "--diff-report",
        nargs=2,
        metavar=("REPORT_A", "REPORT_B"),
        help="仅对比两份 clean_parametric_segmentation_report.json（如 baseline vs tuned），不写新评测",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="并行评测进程数（ProcessPoolExecutor，默认 1 顺序）。建议与 --no-png 同用以降低 IO",
    )
    parser.add_argument(
        "--dp-boundary-field-weight",
        type=float,
        default=None,
        metavar="W",
        help="覆盖 TraceConfig.dp_boundary_field_weight（如 0 关闭 DP 边界场惩罚；默认用 vtracer 配置）",
    )
    parser.add_argument(
        "--boundary-candidate-merge-gap",
        type=int,
        default=None,
        metavar="N",
        help="覆盖 TraceConfig.boundary_candidate_merge_max_polyline_gap（0 关闭；默认用 vtracer 配置）",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["vtracer", "optimized_bs_vtracer"],
        default="vtracer",
        help="vtracer=原图骨架中心线；optimized_bs_vtracer=ImageInit+OptimizedBSpline 种子折线再语义分段",
    )
    args = parser.parse_args()

    if args.diff_report:
        diff_segmentation_reports(args.diff_report[0], args.diff_report[1], label_a="A", label_b="B")
        return

    t_wall0 = time.perf_counter()
    rep = run_eval(
        args.dataset,
        args.output,
        max_samples=args.max_samples,
        keep_debug=args.keep_debug,
        save_png=not args.no_png,
        tuned=not args.baseline,
        jobs=max(1, int(args.jobs)),
        dp_boundary_field_weight=args.dp_boundary_field_weight,
        boundary_candidate_merge_gap=args.boundary_candidate_merge_gap,
        pipeline=str(args.pipeline),
    )
    agg = rep.get("aggregate", {})
    print("=" * 70)
    print(f"pipeline={rep.get('pipeline')}")
    print(f"boundary_candidate_merge_gap={rep.get('boundary_candidate_merge_gap')}")
    print(f"total={rep['total']} success={rep['success']} failed={rep['failed']}")
    print(f"breakpoints_mae_ratio_mean={agg.get('breakpoints_mae_ratio_mean')}")
    print(f"breakpoints_count_error_mean={agg.get('breakpoints_count_error_mean')}")
    print(f"type_lcs_score_mean={agg.get('type_lcs_score_mean')}")
    print(f"parsimony_score_mean={agg.get('parsimony_score_mean')} eff_pred_mean={agg.get('effective_params_pred_mean')} eff_gt_mean={agg.get('effective_params_gt_mean')}")
    print(f"param_score_mean={agg.get('param_score_mean')} p50={agg.get('param_score_p50')} p90={agg.get('param_score_p90')}")
    print(f"elapsed_sec_mean={agg.get('elapsed_sec_mean')}")
    print(f"wall_sec_total={time.perf_counter() - t_wall0:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

