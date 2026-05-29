"""
多进程评测 worker：可被 pickle，供 Windows spawn 下 ProcessPoolExecutor 使用。
主脚本仅在 --jobs>1 时导入本模块；通过 importlib 加载同目录下的评测脚本，避免 tests 包名依赖。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, Tuple

Payload = Tuple[Dict[str, Any], str, bool, bool, bool, float | None, int | None, str]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_TEST_SCRIPT = os.path.join(_THIS_DIR, "test_clean_parametric_segmentation_validation.py")

_eval_mod: Any = None
_MOD_NAME = "_clean_parametric_seg_eval_mod"


def _ensure_paths() -> None:
    if os.environ.get("LOKY_MAX_CPU_COUNT") is None:
        try:
            import multiprocessing as _mp

            os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, int(_mp.cpu_count() or 4)))
        except Exception:
            os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    acad = os.path.join(_ROOT, "AutoCAD_v8.5.4")
    if acad not in sys.path:
        sys.path.insert(0, acad)


def _load_eval_module() -> Any:
    global _eval_mod
    if _eval_mod is not None:
        return _eval_mod
    _ensure_paths()
    spec = importlib.util.spec_from_file_location(_MOD_NAME, _TEST_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {_TEST_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    _eval_mod = mod
    return mod


def run_one_sample(payload: Payload) -> Dict[str, Any]:
    sample, output_dir, keep_debug, save_png, tuned, dp_boundary_field_weight, boundary_candidate_merge_gap, pipeline = payload
    mod = _load_eval_module()
    ev = mod._eval_single(
        sample,
        output_dir=output_dir,
        keep_debug=keep_debug,
        save_png=save_png,
        tuned=tuned,
        dp_boundary_field_weight=dp_boundary_field_weight,
        boundary_candidate_merge_gap=boundary_candidate_merge_gap,
        pipeline=pipeline,
    )
    return asdict(ev)
