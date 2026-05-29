"""
Pytest 入口：验证 VTracer 中心线参数化系统能否在 validation_dataset 上跑通。

这个文件不是参数化算法本体，而是测试入口。它的作用是快速检查：

1. 数据集批处理函数能不能正常启动；
2. 中心线参数化流程能不能对少量样本输出结果；
3. 输出 summary 里是否包含 fitting / segmentation 两类指标；
4. 在需要时，可以通过环境变量打开全量数据集回归测试。

真正做参数化的代码在：

- vtracer_python.py
  - TraceConfig：参数化配置；
  - VTracerPython：主流程；
  - _fit_centerline_path：中心线拟合入口；
  - _dynamic_programming_segments：动态规划语义分段；
  - _fit_segment_model：在线段 / 圆弧 / 样条之间选择模型。

这个测试文件调用的是 test_vtracer_validation_dataset.run_batch，
run_batch 再调用 vtracer_python.py 完成实际参数化。
"""

from __future__ import annotations

import os
import sys

import pytest


# 项目根目录。这里取当前文件的上一级目录，保持和已有测试布局一致。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# validation_dataset 是该测试使用的数据集目录。
DATASET = os.path.join(ROOT, "validation_dataset")

# OUT 是测试输出目录前缀；冒烟测试和全量测试会分别追加后缀。
OUT = os.path.join(ROOT, "vtracer_validation_output", "pytest_run")


@pytest.mark.skipif(not os.path.isdir(DATASET), reason="validation_dataset 不存在")
def test_vtracer_batch_smoke_three_samples():
    """只跑 3 个样本的冒烟测试，确认整条参数化链路基本可用。"""

    # 把源码目录插入 sys.path，避免 pytest 从不同工作目录启动时找不到模块。
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "AutoCAD_v8.5.4"))

    # run_batch 是批处理入口；它内部会读取图片、运行 VTracer、统计指标并写输出。
    from test_vtracer_validation_dataset import run_batch

    out = OUT + "_smoke"
    samples, summary = run_batch(
        DATASET,
        out,
        max_samples=3,
        start_index=0,
        complexity_filter=None,
        log_every=0,
    )

    # 这里的断言偏“系统是否健康”，不是严格追求每个几何指标必须达到某个阈值。
    assert len(samples) == 3
    assert summary["total_samples"] == 3
    assert summary["success_count"] >= 2
    assert "fitting" in summary
    assert "segmentation" in summary


@pytest.mark.skipif(not os.path.isdir(DATASET), reason="validation_dataset 不存在")
@pytest.mark.skipif(os.environ.get("VTRACER_FULL_BATCH") != "1", reason="设置 VTRACER_FULL_BATCH=1 才跑全量")
def test_vtracer_full_validation_dataset():
    """可选全量回归测试；默认跳过，避免普通 pytest 运行太慢。"""

    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "AutoCAD_v8.5.4"))
    from test_vtracer_validation_dataset import run_batch

    out = OUT + "_full"
    _samples, summary = run_batch(DATASET, out, max_samples=None, log_every=25)

    # 全量测试主要检查批处理统计一致性：成功数 + 失败数必须等于总样本数。
    assert summary["subset_size"] >= 1
    assert summary["success_count"] + summary["failed_count"] == summary["total_samples"]
