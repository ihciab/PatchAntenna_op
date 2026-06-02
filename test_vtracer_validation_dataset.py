"""
VTracer 中心线管线系统测试（validation_dataset）

流程对齐代码逻辑：骨架折线 → 关键点/DP/合并分段 → 每段 line|arc|spline 拟合
（spline 段内弦长参数化）→ SVG 路径 + metrics。

评价维度（写入 batch_summary 与控制台）：
  A. 分段（语义）：semantic_segment_count 及 line/arc/spline 段数、keypoints、路径基元数
  B. 拟合精度：rmse_px、mean_error_px、max_error_px（分量级再聚合）
  C. 鲁棒性：metrics_skipped（无骨架分量）、运行异常

GT 外轮廓语义段数仅作复杂度参考，不与中心线一一对应。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "AutoCAD_v8.5.4"))

from core.geometry.segment_extractor import extract_segments_from_fitted_contour  # noqa: E402
from bayesian_optimization.tools.vtracer_python import TraceConfig, VTracerPython  # noqa: E402


# ---------------------------------------------------------------------------
# 指标定义（与 vtracer_python.ComponentMetric / aggregate 字段一致）
# ---------------------------------------------------------------------------
METRIC_GROUPS: Dict[str, Dict[str, str]] = {
    "segmentation": {
        "total_semantic_segments": "DP+合并后语义段总数",
        "total_semantic_line": "语义段中直线类数量",
        "total_semantic_arc": "语义段中圆弧类数量",
        "total_semantic_spline": "语义段中样条类数量",
        "total_keypoints": "各连通域 keypoints 之和",
        "total_path_primitives": "SVG 路径基元数 L+A+C 计数之和",
        "component_count": "骨架连通域数量（有输出的分量）",
    },
    "fitting": {
        "mean_rmse_px": "各分量 RMSE 的算术平均",
        "max_component_error_px": "所有分量中最大点到拟合曲线距离",
        "mean_mean_error_px": "各分量 mean_error 的平均",
        "mean_reduction_ratio": "有效参数/采样点数 的平均压缩比",
    },
    "robustness": {
        "metrics_skipped": "无 metrics 文件（通常无有效骨架分量）",
        "run_failed": "抛异常或缺失图像",
    },
}


@dataclass
class BatchThresholds:
    """可选门禁；命令行传入时未满足则进程 exit 1。"""

    max_mean_rmse_px: Optional[float] = None
    max_max_error_px: Optional[float] = None
    max_metrics_skip_rate: Optional[float] = None  # 0~1


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def _safe_save_figure(fig: plt.Figure, save_path: str, dpi: int = 130) -> None:
    """优先紧凑保存，失败则降级，避免极端 bbox 计算异常中断批处理。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        return
    except Exception:
        pass
    try:
        fig.savefig(save_path, dpi=max(96, int(dpi * 0.85)))
        return
    except Exception:
        # 最后再试一次最保守参数
        fig.savefig(save_path, dpi=96, bbox_inches=None)


def _render_parameterized_model_from_svg(svg_path: str, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    将最终参数化模型（SVG path）渲染到栅格图。
    这里按 path 命令端点连线显示模型几何骨架，便于和原图对照。
    """
    if not svg_path or not os.path.isfile(svg_path):
        return None
    h, w = image_shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    try:
        root = ET.parse(svg_path).getroot()
    except Exception:
        return None

    number_re = r"-?\d*\.?\d+(?:[eE][+-]?\d+)?"
    token_re = re.compile(rf"[MmLlCcAaZz]|{number_re}")

    def is_cmd(tok: str) -> bool:
        return len(tok) == 1 and tok.isalpha()

    def draw_line(p0: Tuple[float, float], p1: Tuple[float, float], color: Tuple[int, int, int] = (30, 30, 30)) -> None:
        x0 = int(round(np.clip(p0[0], 0, w - 1)))
        y0 = int(round(np.clip(p0[1], 0, h - 1)))
        x1 = int(round(np.clip(p1[0], 0, w - 1)))
        y1 = int(round(np.clip(p1[1], 0, h - 1)))
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_AA)

    for elem in root.iter():
        if not str(elem.tag).endswith("path"):
            continue
        d = elem.attrib.get("d", "")
        if not d:
            continue
        tokens = token_re.findall(d)
        if not tokens:
            continue

        i = 0
        cmd = "M"
        cur = (0.0, 0.0)
        sub_start = (0.0, 0.0)
        while i < len(tokens):
            if is_cmd(tokens[i]):
                cmd = tokens[i]
                i += 1
            if cmd in ("M", "m"):
                first = True
                while i + 1 < len(tokens) and not is_cmd(tokens[i]):
                    x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
                    nxt = (x + cur[0], y + cur[1]) if cmd == "m" else (x, y)
                    if first:
                        cur = nxt
                        sub_start = nxt
                        first = False
                    else:
                        draw_line(cur, nxt)
                        cur = nxt
            elif cmd in ("L", "l"):
                while i + 1 < len(tokens) and not is_cmd(tokens[i]):
                    x = float(tokens[i]); y = float(tokens[i + 1]); i += 2
                    nxt = (x + cur[0], y + cur[1]) if cmd == "l" else (x, y)
                    draw_line(cur, nxt)
                    cur = nxt
            elif cmd in ("C", "c"):
                while i + 5 < len(tokens) and not is_cmd(tokens[i]):
                    vals = [float(tokens[i + k]) for k in range(6)]
                    i += 6
                    endp = (vals[4], vals[5])
                    if cmd == "c":
                        endp = (cur[0] + endp[0], cur[1] + endp[1])
                    draw_line(cur, endp)
                    cur = endp
            elif cmd in ("A", "a"):
                while i + 6 < len(tokens) and not is_cmd(tokens[i]):
                    vals = [float(tokens[i + k]) for k in range(7)]
                    i += 7
                    endp = (vals[5], vals[6])
                    if cmd == "a":
                        endp = (cur[0] + endp[0], cur[1] + endp[1])
                    draw_line(cur, endp)
                    cur = endp
            elif cmd in ("Z", "z"):
                draw_line(cur, sub_start)
                cur = sub_start
            else:
                # 未覆盖命令时，防止卡死：跳到下一个命令
                while i < len(tokens) and not is_cmd(tokens[i]):
                    i += 1

    return canvas


def _load_summary(validation_dataset_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(validation_dataset_dir, "summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_under_dataset(base: str, rel: str) -> str:
    if os.path.isabs(rel):
        return rel
    return os.path.join(base, rel.replace("validation_dataset\\", "").replace("validation_dataset/", ""))


def _gt_boundary_segment_count(bgr: np.ndarray) -> int:
    if bgr is None or bgr.size == 0:
        return 0
    if bgr.ndim == 2:
        mask = (bgr > 0).astype(np.uint8) * 255
    elif bgr.ndim == 3 and bgr.shape[2] == 3:
        mask = np.any(bgr < 250, axis=2).astype(np.uint8) * 255
    else:
        return 0
    if not np.any(mask):
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    total = 0
    for c in contours:
        pts = c.reshape(-1, 2).astype(float)
        if len(pts) < 3:
            continue
        if len(pts) > 3000:
            step = int(np.ceil(len(pts) / 3000))
            pts = pts[:: max(1, step)]
        try:
            segs = extract_segments_from_fitted_contour(pts)
            total += len(segs)
        except Exception:
            continue
    return total


def _run_vtracer_centerline(image_path: str, metrics_path: str, svg_path: str) -> Dict[str, Any]:
    cfg = TraceConfig(
        image_path=image_path,
        color_mode="bw",
        trace_style="centerline",
        mode="spline",
        hierarchical="stacked",
        metrics_path=metrics_path,
        save_intermediates=None,
        fit_tolerance=1.5,
        resample_step=4.0,
        filter_speckle=10,
    )
    tracer = VTracerPython(cfg)
    tracer.to_svg(svg_path)
    if not os.path.isfile(metrics_path):
        return {
            "component_count": 0,
            "mean_rmse_px": None,
            "max_component_error_px": None,
            "mean_mean_error_px": None,
            "mean_reduction_ratio": None,
            "total_line_segments": 0,
            "total_arc_segments": 0,
            "total_curve_segments": 0,
            "total_semantic_segments": 0,
            "total_semantic_line": 0,
            "total_semantic_arc": 0,
            "total_semantic_spline": 0,
            "components": [],
            "metrics_skipped": True,
        }
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _reduce_components(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从 metrics JSON 的 components 列表聚合（兼容旧版无 semantic_* 字段）。"""
    if not components:
        return {
            "total_semantic_segments": 0,
            "total_semantic_line": 0,
            "total_semantic_arc": 0,
            "total_semantic_spline": 0,
            "total_keypoints": 0,
            "total_path_primitives": 0,
        }
    ts = sum(int(c.get("semantic_segment_count", 0)) for c in components)
    tl = sum(int(c.get("semantic_segments_line", 0)) for c in components)
    ta = sum(int(c.get("semantic_segments_arc", 0)) for c in components)
    tsp = sum(int(c.get("semantic_segments_spline", 0)) for c in components)
    tk = sum(int(c.get("keypoints", 0)) for c in components)
    prim = 0
    for c in components:
        prim += int(c.get("line_segments", 0)) + int(c.get("arc_segments", 0)) + int(c.get("curve_segments", 0))
    return {
        "total_semantic_segments": ts,
        "total_semantic_line": tl,
        "total_semantic_arc": ta,
        "total_semantic_spline": tsp,
        "total_keypoints": tk,
        "total_path_primitives": prim,
    }


def evaluate_sample(
    validation_dataset_dir: str,
    sample: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    name = sample.get("name", "sample")
    image_path = _resolve_under_dataset(validation_dataset_dir, sample.get("image_path", ""))
    if not os.path.isfile(image_path):
        return {
            "sample_name": name,
            "success": False,
            "error": f"missing image: {image_path}",
            "complexity_level": sample.get("complexity_level"),
        }

    t0 = time.perf_counter()
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gt_seg_count = _gt_boundary_segment_count(bgr)

    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"{name}_vtracer_metrics.json")
    svg_path = os.path.join(out_dir, f"{name}_centerline.svg")

    try:
        agg = _run_vtracer_centerline(image_path, metrics_path=metrics_path, svg_path=svg_path)
    except Exception as e:
        return {
            "sample_name": name,
            "success": False,
            "error": str(e),
            "complexity_level": sample.get("complexity_level"),
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        }

    if "error" in agg:
        return {
            "sample_name": name,
            "success": False,
            "error": agg.get("error", "unknown"),
            "complexity_level": sample.get("complexity_level"),
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        }

    components = agg.get("components") or []
    reduced = _reduce_components(components)
    mr = agg.get("mean_rmse_px")
    mxe = agg.get("max_component_error_px")
    mrr = agg.get("mean_reduction_ratio")

    mean_mean_err: Optional[float] = None
    if components:
        errs = [float(c.get("mean_error_px", 0.0)) for c in components]
        mean_mean_err = float(np.mean(errs)) if errs else None

    # 优先使用 aggregate 顶层（与 vtracer 写出一致）
    ts = int(agg.get("total_semantic_segments", reduced["total_semantic_segments"]))
    tlin = int(agg.get("total_semantic_line", reduced["total_semantic_line"]))
    tarc = int(agg.get("total_semantic_arc", reduced["total_semantic_arc"]))
    tspl = int(agg.get("total_semantic_spline", reduced["total_semantic_spline"]))

    return {
        "sample_name": name,
        "success": True,
        "complexity_level": sample.get("complexity_level"),
        "structure_tags": sample.get("structure_tags", []),
        "image_path": image_path,
        "gt_boundary_primitive_segments": gt_seg_count,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "vtracer": {
            "component_count": int(agg.get("component_count", 0)),
            "mean_rmse_px": None if mr is None else float(mr),
            "max_component_error_px": None if mxe is None else float(mxe),
            "mean_mean_error_px": mean_mean_err,
            "mean_reduction_ratio": None if mrr is None else float(mrr),
            "total_line_segments": int(agg.get("total_line_segments", 0)),
            "total_arc_segments": int(agg.get("total_arc_segments", 0)),
            "total_curve_segments": int(agg.get("total_curve_segments", 0)),
            "total_semantic_segments": ts,
            "total_semantic_line": tlin,
            "total_semantic_arc": tarc,
            "total_semantic_spline": tspl,
            "total_path_primitives": int(
                agg.get(
                    "total_line_segments", 0
                )
                + agg.get("total_arc_segments", 0)
                + agg.get("total_curve_segments", 0)
            ),
            "total_keypoints": reduced["total_keypoints"],
            "metrics_skipped": bool(agg.get("metrics_skipped", False)),
            "metrics_path": metrics_path,
            "svg_path": svg_path,
        },
    }


def _save_sample_visualization(result: Dict[str, Any], save_path: str) -> None:
    """单样本可视化：原图 vs 参数化模型 + 分段对比 + 拟合精度。"""
    if not result.get("success"):
        return
    image_path = result.get("image_path")
    if not image_path or not os.path.isfile(image_path):
        return
    v = result.get("vtracer", {})
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rmse = _safe_float(v.get("mean_rmse_px"))
    mx = _safe_float(v.get("max_component_error_px"))
    mean_e = _safe_float(v.get("mean_mean_error_px"))
    gt_seg = int(result.get("gt_boundary_primitive_segments", 0))
    sem_seg = int(v.get("total_semantic_segments", 0))
    prim = int(v.get("total_path_primitives", 0))
    keypoints = int(v.get("total_keypoints", 0))
    c_line = int(v.get("total_semantic_line", 0))
    c_arc = int(v.get("total_semantic_arc", 0))
    c_spl = int(v.get("total_semantic_spline", 0))
    svg_path = str(v.get("svg_path", ""))
    model_bgr = _render_parameterized_model_from_svg(svg_path, (img.shape[0], img.shape[1]))

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.05, 1.0])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(rgb)
    ax_img.set_title(f"Original image | {result.get('sample_name')} (complexity={result.get('complexity_level')})")
    ax_img.axis("off")

    ax_model = fig.add_subplot(gs[0, 1])
    if model_bgr is not None:
        ax_model.imshow(cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB))
        ax_model.set_title("Final parameterized model (from SVG path)")
    else:
        ax_model.text(0.5, 0.5, "Model preview unavailable", ha="center", va="center")
        ax_model.set_title("Final parameterized model")
    ax_model.axis("off")

    ax_seg = fig.add_subplot(gs[1, 0])
    seg_labels = ["GT boundary", "Semantic", "Path prim", "Keypoints"]
    seg_vals = [gt_seg, sem_seg, prim, keypoints]
    ax_seg.bar(seg_labels, seg_vals, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#9467bd"])
    ax_seg.set_title("Segmentation counts")
    ax_seg.tick_params(axis="x", rotation=20)
    for i, val in enumerate(seg_vals):
        ax_seg.text(i, val + max(0.2, 0.02 * max(seg_vals + [1])), str(val), ha="center", va="bottom", fontsize=9)

    ax_fit = fig.add_subplot(gs[1, 1])
    err_labels = ["mean_rmse_px", "mean_error_px", "max_error_px"]
    err_vals = [
        0.0 if rmse is None else rmse,
        0.0 if mean_e is None else mean_e,
        0.0 if mx is None else mx,
    ]
    colors = ["#17becf", "#bcbd22", "#d62728"]
    ax_fit.bar(err_labels, err_vals, color=colors)
    ax_fit.set_title(f"Fitting errors | L/A/S={c_line}/{c_arc}/{c_spl}")
    ax_fit.tick_params(axis="x", rotation=20)
    for i, val in enumerate(err_vals):
        txt = "N/A" if (i == 0 and rmse is None) or (i == 1 and mean_e is None) or (i == 2 and mx is None) else f"{val:.3f}"
        ax_fit.text(i, val + max(0.02, 0.02 * max(err_vals + [1])), txt, ha="center", va="bottom", fontsize=9)

    try:
        fig.tight_layout()
    except Exception:
        pass
    _safe_save_figure(fig, save_path, dpi=130)
    plt.close(fig)


def _save_overall_visualization(samples: List[Dict[str, Any]], summary: Dict[str, Any], save_path: str) -> None:
    """整体统计图：误差分布、分段分布、复杂度-误差、L/A/S占比。"""
    ok = [s for s in samples if s.get("success")]
    rmse = [float((s.get("vtracer") or {}).get("mean_rmse_px")) for s in ok if (s.get("vtracer") or {}).get("mean_rmse_px") is not None]
    sem = [float((s.get("vtracer") or {}).get("total_semantic_segments", 0)) for s in ok]
    cx = [
        (int(s.get("complexity_level")) if s.get("complexity_level") is not None else -1,
         (s.get("vtracer") or {}).get("mean_rmse_px"))
        for s in ok
    ]

    line_sum = int(summary.get("segmentation", {}).get("total_semantic_line_sum", 0))
    arc_sum = int(summary.get("segmentation", {}).get("total_semantic_arc_sum", 0))
    spl_sum = int(summary.get("segmentation", {}).get("total_semantic_spline_sum", 0))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    if rmse:
        ax1.hist(rmse, bins=30, color="#1f77b4", alpha=0.85)
        ax1.set_title("RMSE distribution")
        ax1.set_xlabel("mean_rmse_px")
    else:
        ax1.text(0.5, 0.5, "No RMSE data", ha="center", va="center")
    ax1.grid(alpha=0.25, linestyle="--")

    if sem:
        ax2.hist(sem, bins=30, color="#2ca02c", alpha=0.85)
        ax2.set_title("Semantic segment count distribution")
        ax2.set_xlabel("total_semantic_segments")
    else:
        ax2.text(0.5, 0.5, "No segmentation data", ha="center", va="center")
    ax2.grid(alpha=0.25, linestyle="--")

    cx_valid = [(c, float(r)) for c, r in cx if r is not None and c >= 0]
    if cx_valid:
        xs = [p[0] for p in cx_valid]
        ys = [p[1] for p in cx_valid]
        ax3.scatter(xs, ys, s=18, alpha=0.7, color="#ff7f0e")
        ax3.set_title("Complexity vs RMSE")
        ax3.set_xlabel("complexity_level")
        ax3.set_ylabel("mean_rmse_px")
    else:
        ax3.text(0.5, 0.5, "No complexity-RMSE points", ha="center", va="center")
    ax3.grid(alpha=0.25, linestyle="--")

    vals = [line_sum, arc_sum, spl_sum]
    labels = ["line", "arc", "spline"]
    if sum(vals) > 0:
        ax4.pie(vals, labels=labels, autopct="%1.1f%%", startangle=120, colors=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax4.set_title("Semantic kind ratio (L/A/S)")
    else:
        ax4.text(0.5, 0.5, "No semantic kind data", ha="center", va="center")

    fig.suptitle(
        "VTracer validation summary | "
        f"N={summary.get('total_samples')} success={summary.get('success_count')} "
        f"skip={summary.get('metrics_skipped_count')}",
        fontsize=12,
    )
    try:
        fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    except Exception:
        pass
    _safe_save_figure(fig, save_path, dpi=140)
    plt.close(fig)


def _select_top_error_samples(samples: List[Dict[str, Any]], top_k: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    ok = [s for s in samples if s.get("success")]

    def metric_val(s: Dict[str, Any], key: str) -> float:
        v = (s.get("vtracer") or {}).get(key)
        fv = _safe_float(v)
        return -1.0 if fv is None else float(fv)

    by_rmse = sorted(ok, key=lambda s: metric_val(s, "mean_rmse_px"), reverse=True)
    by_maxerr = sorted(ok, key=lambda s: metric_val(s, "max_component_error_px"), reverse=True)
    return {
        "top_rmse": by_rmse[: max(1, top_k)],
        "top_max_error": by_maxerr[: max(1, top_k)],
    }


def _save_worst_case_dashboard(
    top_samples: List[Dict[str, Any]],
    key: str,
    title: str,
    save_path: str,
    max_show: int = 12,
) -> None:
    shown = [s for s in top_samples[:max_show] if s.get("success")]
    if not shown:
        return

    cols = 4
    rows = int(np.ceil(len(shown) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.6 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, s in enumerate(shown):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        img_path = s.get("image_path")
        vis_path = s.get("sample_visualization_path")
        score = _safe_float((s.get("vtracer") or {}).get(key))
        name = s.get("sample_name", f"sample_{idx}")

        # 优先展示单样本对比图（已含原图+参数化模型+指标）
        if vis_path and os.path.isfile(str(vis_path)):
            im = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        elif img_path and os.path.isfile(str(img_path)):
            im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        else:
            im = None
        if im is not None:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        else:
            ax.text(0.5, 0.5, "image missing", ha="center", va="center")
        ax.set_title(f"{name}\n{key}={score if score is not None else 'N/A'}", fontsize=9)
        ax.axis("off")

    for idx in range(len(shown), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=13)
    try:
        fig.tight_layout(rect=[0, 0.01, 1, 0.95])
    except Exception:
        pass
    _safe_save_figure(fig, save_path, dpi=120)
    plt.close(fig)


def _save_top_error_reports(samples: List[Dict[str, Any]], output_dir: str, top_k: int = 20) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    top = _select_top_error_samples(samples, top_k=top_k)

    def slim(s: Dict[str, Any], key: str) -> Dict[str, Any]:
        v = s.get("vtracer") or {}
        return {
            "sample_name": s.get("sample_name"),
            "complexity_level": s.get("complexity_level"),
            "structure_tags": s.get("structure_tags", []),
            key: v.get(key),
            "mean_rmse_px": v.get("mean_rmse_px"),
            "max_component_error_px": v.get("max_component_error_px"),
            "total_semantic_segments": v.get("total_semantic_segments"),
            "total_keypoints": v.get("total_keypoints"),
            "sample_visualization_path": s.get("sample_visualization_path"),
            "svg_path": v.get("svg_path"),
            "metrics_path": v.get("metrics_path"),
        }

    rmse_json = os.path.join(output_dir, "top_rmse_samples.json")
    max_json = os.path.join(output_dir, "top_max_error_samples.json")
    with open(rmse_json, "w", encoding="utf-8") as f:
        json.dump([slim(s, "mean_rmse_px") for s in top["top_rmse"]], f, ensure_ascii=False, indent=2)
    with open(max_json, "w", encoding="utf-8") as f:
        json.dump([slim(s, "max_component_error_px") for s in top["top_max_error"]], f, ensure_ascii=False, indent=2)

    rmse_png = os.path.join(output_dir, "top_rmse_samples_grid.png")
    max_png = os.path.join(output_dir, "top_max_error_samples_grid.png")
    _save_worst_case_dashboard(
        top["top_rmse"],
        key="mean_rmse_px",
        title=f"Top-{min(top_k, len(top['top_rmse']))} Worst by mean_rmse_px",
        save_path=rmse_png,
    )
    _save_worst_case_dashboard(
        top["top_max_error"],
        key="max_component_error_px",
        title=f"Top-{min(top_k, len(top['top_max_error']))} Worst by max_component_error_px",
        save_path=max_png,
    )
    return {
        "top_rmse_json": rmse_json,
        "top_max_error_json": max_json,
        "top_rmse_grid_png": rmse_png,
        "top_max_error_grid_png": max_png,
    }


def _percentiles(xs: List[float], ps: Tuple[int, ...] = (50, 90, 95)) -> Dict[str, float]:
    if not xs:
        return {}
    a = np.asarray(xs, dtype=np.float64)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def compute_batch_summary(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [s for s in samples if s.get("success")]
    bad = [s for s in samples if not s.get("success")]
    skipped = [s for s in ok if s.get("vtracer", {}).get("metrics_skipped")]

    def pull(subkey: str) -> List[float]:
        out: List[float] = []
        for s in ok:
            v = (s.get("vtracer") or {}).get(subkey)
            if v is not None and isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                out.append(float(v))
        return out

    rmse = pull("mean_rmse_px")
    mx = pull("max_component_error_px")
    mme = pull("mean_mean_error_px")

    sem = [float((s.get("vtracer") or {}).get("total_semantic_segments", 0)) for s in ok]
    kpt = [float((s.get("vtracer") or {}).get("total_keypoints", 0)) for s in ok]
    prim = [float((s.get("vtracer") or {}).get("total_path_primitives", 0)) for s in ok]

    by_c: Dict[str, Any] = {}
    for s in ok:
        lv = s.get("complexity_level")
        key = str(int(lv)) if lv is not None else "unknown"
        by_c.setdefault(key, []).append(s)

    per_complexity: Dict[str, Any] = {}
    for key, group in by_c.items():
        r = [float((x.get("vtracer") or {}).get("mean_rmse_px", 0.0)) for x in group if (x.get("vtracer") or {}).get("mean_rmse_px") is not None]
        per_complexity[key] = {
            "count": len(group),
            "mean_rmse_px": float(np.mean(r)) if r else None,
        }

    return {
        "total_samples": len(samples),
        "success_count": len(ok),
        "failed_count": len(bad),
        "metrics_skipped_count": len(skipped),
        "metrics_skip_rate": float(len(skipped) / max(1, len(ok))) if ok else 0.0,
        "fitting": {
            "mean_rmse_px_mean": float(np.mean(rmse)) if rmse else None,
            "mean_rmse_px_std": float(np.std(rmse)) if len(rmse) > 1 else 0.0,
            "mean_rmse_px_percentiles": _percentiles(rmse),
            "max_component_error_px_mean": float(np.mean(mx)) if mx else None,
            "max_component_error_px_percentiles": _percentiles(mx),
            "mean_mean_error_px_mean": float(np.mean(mme)) if mme else None,
        },
        "segmentation": {
            "total_semantic_segments_mean": float(np.mean(sem)) if sem else None,
            "total_semantic_segments_percentiles": _percentiles(sem),
            "total_keypoints_mean": float(np.mean(kpt)) if kpt else None,
            "total_path_primitives_mean": float(np.mean(prim)) if prim else None,
            "total_semantic_line_sum": int(sum((s.get("vtracer") or {}).get("total_semantic_line", 0) for s in ok)),
            "total_semantic_arc_sum": int(sum((s.get("vtracer") or {}).get("total_semantic_arc", 0) for s in ok)),
            "total_semantic_spline_sum": int(sum((s.get("vtracer") or {}).get("total_semantic_spline", 0) for s in ok)),
        },
        "per_complexity": per_complexity,
        "failures": [{"sample_name": s.get("sample_name"), "error": s.get("error")} for s in bad],
        "metric_groups_doc": METRIC_GROUPS,
    }


def _check_thresholds(summary: Dict[str, Any], th: BatchThresholds) -> Tuple[bool, List[str]]:
    notes: List[str] = []
    ok = True

    if th.max_mean_rmse_px is not None:
        v = summary.get("fitting", {}).get("mean_rmse_px_mean")
        if v is not None and v > th.max_mean_rmse_px:
            ok = False
            notes.append(f"mean_rmse_px_mean {v:.4f} > gate {th.max_mean_rmse_px}")

    if th.max_max_error_px is not None:
        v = summary.get("fitting", {}).get("max_component_error_px_mean")
        if v is not None and v > th.max_max_error_px:
            ok = False
            notes.append(f"max_component_error_px_mean {v:.4f} > gate {th.max_max_error_px}")

    if th.max_metrics_skip_rate is not None:
        r = float(summary.get("metrics_skip_rate", 0.0))
        if r > th.max_metrics_skip_rate:
            ok = False
            notes.append(f"metrics_skip_rate {r:.4f} > gate {th.max_metrics_skip_rate}")

    return ok, notes


def run_batch(
    validation_dataset_dir: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    start_index: int = 0,
    complexity_filter: Optional[List[int]] = None,
    log_every: int = 10,
    with_plots: bool = True,
    top_k_worst: int = 20,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary = _load_summary(validation_dataset_dir)
    subset = summary[start_index:]
    if complexity_filter:
        allow = set(complexity_filter)
        subset = [s for s in subset if int(s.get("complexity_level", -1)) in allow]
    if max_samples is not None:
        subset = subset[:max_samples]

    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, sample in enumerate(subset):
        sub_out = os.path.join(output_dir, sample.get("name", f"sample_{i}"))
        result = evaluate_sample(validation_dataset_dir, sample, sub_out)
        if with_plots and result.get("success"):
            sample_plot_path = os.path.join(sub_out, f"{result.get('sample_name')}_metrics_vis.png")
            _save_sample_visualization(result, sample_plot_path)
            result["sample_visualization_path"] = sample_plot_path
        results.append(result)
        if log_every > 0 and (i + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i + 1}/{len(subset)}] elapsed {elapsed:.1f}s", flush=True)

    batch_summary = compute_batch_summary(results)
    batch_summary["wall_clock_sec"] = round(time.perf_counter() - t0, 3)
    batch_summary["dataset_dir"] = validation_dataset_dir
    batch_summary["subset_size"] = len(subset)

    report = {
        "batch_summary": batch_summary,
        "samples": results,
    }
    report_path = os.path.join(output_dir, "vtracer_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary_only = os.path.join(output_dir, "vtracer_batch_summary.json")
    with open(summary_only, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    if with_plots:
        overall_plot_path = os.path.join(output_dir, "vtracer_overall_statistics.png")
        _save_overall_visualization(results, batch_summary, overall_plot_path)
        batch_summary["overall_visualization_path"] = overall_plot_path
        top_dir = os.path.join(output_dir, "worst_cases")
        top_paths = _save_top_error_reports(results, top_dir, top_k=max(1, int(top_k_worst)))
        batch_summary["worst_case_reports"] = top_paths
        report["batch_summary"] = batch_summary
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(summary_only, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    return results, batch_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VTracer 中心线：validation_dataset 批量系统测试与指标汇总"
    )
    parser.add_argument("--dataset", type=str, default=os.path.join(ROOT, "validation_dataset"))
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "vtracer_validation_output"))
    parser.add_argument("--max-samples", type=int, default=None, help="默认 None 表示跑完全集")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--complexity", type=int, nargs="*", default=None)
    parser.add_argument("--log-every", type=int, default=20, help="进度日志间隔，0 关闭")
    parser.add_argument("--no-plots", action="store_true", help="关闭每样本与整体统计可视化输出")
    parser.add_argument("--top-k-worst", type=int, default=20, help="导出高误差样本Top-K")
    parser.add_argument("--max-mean-rmse", type=float, default=None, help="门禁：batch mean_rmse 上限")
    parser.add_argument("--max-mean-max-error", type=float, default=None, help="门禁：batch max_error 均值上限")
    parser.add_argument("--max-skip-rate", type=float, default=None, help="门禁：metrics_skipped 在成功样本中占比上限 0~1")
    args = parser.parse_args()

    th = BatchThresholds(
        max_mean_rmse_px=args.max_mean_rmse,
        max_max_error_px=args.max_mean_max_error,
        max_metrics_skip_rate=args.max_skip_rate,
    )

    print("VTracer 系统测试 (validation_dataset)")
    print("指标组:", ", ".join(METRIC_GROUPS.keys()))
    print("=" * 60)

    _results, batch_summary = run_batch(
        args.dataset,
        args.output,
        max_samples=args.max_samples,
        start_index=args.start_index,
        complexity_filter=list(args.complexity) if args.complexity else None,
        log_every=args.log_every,
        with_plots=not args.no_plots,
        top_k_worst=args.top_k_worst,
    )

    bs = batch_summary
    print(f"子集: {bs.get('subset_size')}  wall: {bs.get('wall_clock_sec')}s")
    print(f"成功: {bs.get('success_count')}  失败: {bs.get('failed_count')}  metrics_skipped: {bs.get('metrics_skipped_count')} ({bs.get('metrics_skip_rate', 0)*100:.1f}% of success)")
    fit = bs.get("fitting", {})
    if fit.get("mean_rmse_px_mean") is not None:
        print(f"拟合 mean_rmse_px: mean={fit['mean_rmse_px_mean']:.4f}  p95={fit.get('mean_rmse_px_percentiles', {}).get('p95', 'n/a')}")
    if fit.get("max_component_error_px_mean") is not None:
        print(f"拟合 max_error: mean={fit['max_component_error_px_mean']:.4f}  p95={fit.get('max_component_error_px_percentiles', {}).get('p95', 'n/a')}")
    seg = bs.get("segmentation", {})
    if seg.get("total_semantic_segments_mean") is not None:
        print(
            f"分段 semantic_segments: mean={seg['total_semantic_segments_mean']:.2f}  "
            f"L/A/S 累计={seg.get('total_semantic_line_sum')}/{seg.get('total_semantic_arc_sum')}/{seg.get('total_semantic_spline_sum')}"
        )

    report_path = os.path.join(args.output, "vtracer_validation_report.json")
    print(f"\n完整报告: {report_path}")
    print(f"仅汇总: {os.path.join(args.output, 'vtracer_batch_summary.json')}")
    if batch_summary.get("overall_visualization_path"):
        print(f"整体可视化: {batch_summary.get('overall_visualization_path')}")
    wc = batch_summary.get("worst_case_reports", {})
    if wc:
        print(f"Worst-case报告: {wc.get('top_rmse_json')}")
        print(f"Worst-case网格图: {wc.get('top_rmse_grid_png')}")

    pass_gates, notes = _check_thresholds(bs, th)
    if notes:
        print("门禁:", "; ".join(notes))
    if not pass_gates:
        sys.exit(1)


if __name__ == "__main__":
    main()
