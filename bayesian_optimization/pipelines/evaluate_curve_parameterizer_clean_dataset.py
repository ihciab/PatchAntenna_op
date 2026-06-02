from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from Rebuild.NewParams import CurveParameterizer, NewParams


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "clean_parametric_dataset_50"
DEFAULT_OUTPUT = ROOT / "curve_parameterizer_clean_eval_output"


@dataclass
class CurveParamSampleEval:
    name: str
    success: bool
    gt_segments: int
    pred_segments: int
    breakpoints_mae_ratio: float
    breakpoints_count_error: int
    type_lcs_score: float
    effective_params_gt: int
    effective_params_pred: int
    parsimony_score: float
    param_score: float
    elapsed_sec: float
    json_path: str = ""
    visual_path: str = ""
    error: str = ""


def norm_type(value: str) -> str:
    value = value.lower().strip()
    if value == "line":
        return "line"
    if value in ("arc", "circle"):
        return "arc"
    if value in ("bspline", "spline", "curve"):
        return "spline"
    return value


def gt_boundary_ratios(gt_segments: List[Dict[str, Any]]) -> Tuple[List[float], List[str]]:
    if not gt_segments:
        return [], []
    lengths = []
    types = []
    for segment in gt_segments:
        lengths.append(int(segment.get("samples", 0) or estimate_gt_segment_samples(segment)))
        types.append(norm_type(str(segment.get("type", ""))))
    total = max(1, sum(max(1, n) for n in lengths))
    acc = 0
    boundaries = []
    for n in lengths:
        boundaries.append(acc / total)
        acc += max(1, n)
    return boundaries, types


def pred_boundary_ratios(contour_result: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    n_points = int(contour_result.get("sampled_point_count", 0) or 0)
    if n_points <= 0:
        return [], []
    boundaries = []
    types = []
    for segment in contour_result.get("segments", []):
        start_idx = int(segment.get("start_idx", 0) or 0) % max(1, n_points)
        boundaries.append(start_idx / max(1, n_points))
        types.append(norm_type(str(segment.get("type", ""))))
    return boundaries, types


def estimate_gt_segment_samples(segment: Dict[str, Any]) -> int:
    kind = norm_type(str(segment.get("type", "")))
    if kind == "line":
        p0 = np.asarray(segment.get("p0", segment.get("start", [0, 0])), dtype=np.float64)
        p1 = np.asarray(segment.get("p1", segment.get("end", [0, 0])), dtype=np.float64)
        return max(2, int(np.linalg.norm(p1 - p0)))
    if kind == "arc":
        radius = float(segment.get("radius", 20.0) or 20.0)
        sweep = float(segment.get("sweep_angle", segment.get("sweep", 2.0 * np.pi)) or 2.0 * np.pi)
        return max(8, int(abs(radius * sweep)))
    if kind == "spline":
        ctrl = [key for key in segment if isinstance(key, str) and len(key) > 1 and key[0] == "p" and key[1:].isdigit()]
        return max(24, 16 * max(2, len(ctrl)))
    return 16


def gt_effective_params(gt_segments: List[Dict[str, Any]]) -> int:
    total = 0
    for segment in gt_segments:
        kind = norm_type(str(segment.get("type", "")))
        if kind == "line":
            total += 1
        elif kind == "arc":
            total += 2
        elif kind == "spline":
            ctrl = [key for key in segment if isinstance(key, str) and len(key) > 1 and key[0] == "p" and key[1:].isdigit()]
            total += max(4, min(24, len(ctrl) if ctrl else 4))
        else:
            total += 3
    return max(1, total)


def pred_effective_params(contour_result: Dict[str, Any]) -> int:
    return max(1, sum(int(seg.get("effective_params", 0) or 0) for seg in contour_result.get("segments", [])))


def breakpoint_mae(gt_boundaries: List[float], pred_boundaries: List[float]) -> float:
    if not gt_boundaries and not pred_boundaries:
        return 0.0
    if not gt_boundaries or not pred_boundaries:
        return 1.0
    return float(np.mean([min(abs(g - p) for p in pred_boundaries) for g in gt_boundaries]))


def lcs_ratio(a: List[str], b: List[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return float(dp[-1][-1]) / float(max(len(a), len(b)))


def parsimony_score(pred_eff: int, gt_eff: int) -> float:
    if pred_eff <= gt_eff:
        return 1.0
    excess = (pred_eff - gt_eff) / float(max(1, gt_eff))
    return max(0.0, 1.0 - min(1.0, 0.32 * excess))


def select_main_contour(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    return max(results, key=lambda item: int(item.get("sampled_point_count", 0) or 0))


def resolve_sample_paths(dataset: Path, sample: Dict[str, Any]) -> Tuple[str, Path, Path]:
    name = str(sample["name"])
    image_path = dataset / "images" / f"{name}.png"
    gt_path = dataset / "gt" / f"{name}.json"
    return name, image_path, gt_path


def eval_one(sample: Dict[str, Any], dataset: Path, output_dir: Path, save_png: bool) -> CurveParamSampleEval:
    name, image_path, gt_path = resolve_sample_paths(dataset, sample)
    t0 = time.perf_counter()
    sample_out = output_dir / name
    sample_out.mkdir(parents=True, exist_ok=True)

    try:
        with gt_path.open("r", encoding="utf-8") as f:
            gt = json.load(f)
        gt_segments = gt.get("segments", [])
        gt_boundaries, gt_types = gt_boundary_ratios(gt_segments)
        gt_eff = gt_effective_params(gt_segments)

        params = NewParams(
            image_path=image_path,
            save_dir=sample_out,
            canny_threshold1=50,
            canny_threshold2=160,
            morph_kernel_size=1,
            dilate_iterations=0,
        )
        parameterizer = CurveParameterizer(
            image_path=image_path,
            edges=params.edges(),
            original_img=params.original_img(),
            save_dir=sample_out,
            output_prefix=name,
            fit_tolerance=1.5,
            resample_step=3.0,
            dp_complexity_weight=1.02,
            dp_max_segment_points=78,
            keypoint_angle_threshold_deg=32.0,
        )
        json_path = parameterizer.save_json(sample_out / f"{name}_curve_parameterization.json")
        visual_path = ""
        if save_png:
            visual_path = str(parameterizer.visualize(sample_out / f"{name}_curve_parameterization.png"))

        main = select_main_contour(parameterizer.results())
        pred_boundaries, pred_types = pred_boundary_ratios(main)
        pred_eff = pred_effective_params(main)

        bp_mae = breakpoint_mae(gt_boundaries, pred_boundaries)
        cnt_err = abs(len(gt_boundaries) - len(pred_boundaries))
        type_score = lcs_ratio(gt_types, pred_types)
        sparse_score = parsimony_score(pred_eff, gt_eff)
        bp_score = max(0.0, 1.0 - min(1.0, bp_mae / 0.08) - min(0.4, 0.06 * cnt_err))
        param_score = 0.52 * bp_score + 0.30 * type_score + 0.18 * sparse_score

        return CurveParamSampleEval(
            name=name,
            success=True,
            gt_segments=len(gt_types),
            pred_segments=len(pred_types),
            breakpoints_mae_ratio=bp_mae,
            breakpoints_count_error=cnt_err,
            type_lcs_score=type_score,
            effective_params_gt=gt_eff,
            effective_params_pred=pred_eff,
            parsimony_score=sparse_score,
            param_score=param_score,
            elapsed_sec=round(time.perf_counter() - t0, 4),
            json_path=str(json_path),
            visual_path=visual_path,
        )
    except Exception as exc:
        return CurveParamSampleEval(
            name=name,
            success=False,
            gt_segments=0,
            pred_segments=0,
            breakpoints_mae_ratio=1.0,
            breakpoints_count_error=0,
            type_lcs_score=0.0,
            effective_params_gt=0,
            effective_params_pred=0,
            parsimony_score=0.0,
            param_score=0.0,
            elapsed_sec=round(time.perf_counter() - t0, 4),
            error=str(exc),
        )


def run_eval(dataset: Path, output_dir: Path, max_samples: int | None, save_png: bool) -> Dict[str, Any]:
    with (dataset / "summary.json").open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if max_samples is not None:
        samples = samples[:max_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [eval_one(sample, dataset, output_dir, save_png=save_png) for sample in samples]
    ok = [row for row in rows if row.success]
    report = {
        "dataset_dir": str(dataset),
        "output_dir": str(output_dir),
        "total": len(rows),
        "success": len(ok),
        "failed": len(rows) - len(ok),
        "aggregate": {
            "breakpoints_mae_ratio_mean": mean_or_none([r.breakpoints_mae_ratio for r in ok]),
            "breakpoints_count_error_mean": mean_or_none([r.breakpoints_count_error for r in ok]),
            "type_lcs_score_mean": mean_or_none([r.type_lcs_score for r in ok]),
            "effective_params_gt_mean": mean_or_none([r.effective_params_gt for r in ok]),
            "effective_params_pred_mean": mean_or_none([r.effective_params_pred for r in ok]),
            "parsimony_score_mean": mean_or_none([r.parsimony_score for r in ok]),
            "param_score_mean": mean_or_none([r.param_score for r in ok]),
        },
        "samples": [asdict(row) for row in rows],
    }
    report_path = output_dir / "curve_parameterizer_clean_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report: {report_path}")
    return report


def mean_or_none(values: List[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CurveParameterizer on clean_parametric_dataset_50.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()

    report = run_eval(args.dataset, args.output, args.max_samples, save_png=not args.no_png)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
