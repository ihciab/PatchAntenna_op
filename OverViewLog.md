# OverViewLog

本文件只记录已经验证、并且值得保留为项目维护依据的成功改动。临时实验、失败尝试、已回退调试不写入本文件。

## 2026-05-27: 拆分总体架构 README 和成功改动日志

Changed files:

- `PROJECT_PIPELINE_README.md`
- `OverViewLog.md`

Purpose:

- 将原本冗长的 `TEST_FILES_OVERVIEW_*.txt` 拆分为稳定架构说明和精简成功日志。
- 让项目主线回到图片 -> FSS -> 边缘 -> Global B-spline -> 参数化 -> CST 建模。

Validation:

- 文档结构基于 `fss_parameterized_cst_pipeline.py`、`fss_simulation_pipeline.py`、`geometry_driven_parameterizer.py`、`geometry_graph_parameterizer.py`、`parameterized_json_to_cst.py` 和 `run_parameterization_only.py` 梳理。

Remaining risk:

- 旧 overview 仍作为历史材料保留，但不建议继续追加维护流水账。

## 2026-05-25: FSS + Parameterization Only Runner

Changed files:

- `run_parameterization_only.py`

Purpose:

- 提供阶段测试入口：运行 FSS repair 和参数化，但不创建 CST 工程，不运行 CST solver。

Behavior:

- 默认执行 FSS repair。
- `--skip-fss-cleanup` 可直接使用原图。
- `--honor-instance-skip` 可恢复尊重 instance skip 标记的旧行为。
- 输出 `parameterization_only_metadata.json`，明确 `cst_build_skipped=true` 和 `cst_solver_skipped=true`。

Validation:

- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\run_parameterization_only.py`
- `D:\Anaconda\envs\linefor\python.exe .\run_parameterization_only.py --help`

Remaining risk:

- 默认 FSS repair 依赖 OCR / YOLO / detector 环境。如果只想调试参数化，用 `--skip-fss-cleanup`。

## 2026-05-25: Joblib / Loky CPU Probe Guard

Changed files:

- `run_parameterization_only.py`
- `fss_parameterized_cst_pipeline.py`

Purpose:

- 避免 Windows + Anaconda 环境下 joblib/loky 物理核心探测输出非致命 traceback。

Implementation:

```python
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))
```

Validation:

- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\run_parameterization_only.py .\fss_parameterized_cst_pipeline.py`
- 在 `linefor` 环境导入 runner 后可读到 `LOKY_MAX_CPU_COUNT`。

Remaining risk:

- 该改动只处理 joblib CPU 探测噪声，不改变 FSS、参数化或 CST 行为。

## 2026-05-24: Graph-Based Local Spline Parameterizer

Changed files:

- `geometry_graph_parameterizer.py`
- `fss_parameterized_cst_pipeline.py`
- `run_parameterization_only.py`
- `parameterized_json_to_cst.py`

Purpose:

- 新增 `graph_local_primitives` 参数化模式，避免全局 B-spline 对复杂 FSS 拓扑造成过度平滑、圆角化或 primitive 爆炸。

Pipeline:

```text
repair_fig.png
  -> adaptive edge preprocessing
  -> VTracer centerline extraction
  -> topology graph extraction
  -> graph edge split
  -> local line / arc / spline fitting
  -> graph-aware JSON
```

Fallback:

```text
graph_local_primitives -> geometry_primitives -> standard
```

Validation:

- `test_graph_local_parameterizer.py` 中 right-angle corner、branch junction、close parallel lines、loop anchor 回归函数通过。
- 参数化-only runner 可生成 graph-local debug folders。

Remaining risk:

- graph-local 当前更偏拓扑保护，primitive 数量可能高于 Global B-spline 流程。

## 2026-05-24: Edge Candidate Quality Selector

Changed files:

- `geometry_driven_parameterizer.py`

Purpose:

- 在参数化前比较多个边缘候选，而不是只在 Canny 和 auto 之间做简单切换。

Successful behavior:

- 输出 `edge_candidate_scores.json`
- 输出 `edge_selection_diagnostics.json`
- 根据拓扑指标选择更合适的边缘输入

Candidate examples:

- `canny`
- `auto`
- `foreground_contour`
- `stroke_mask`
- `canny_gap_closed`
- `subject_mask_boundary`
- `color_subject_boundary`
- `canny_merged`

Remaining risk:

- 候选评分只能辅助决策。复杂截图仍需人工检查 `_candidate_*` 图和 diagnostics。

## 2026-05-24: Subject Boundary Retention Guard

Changed files:

- `geometry_driven_parameterizer.py`

Purpose:

- 防止 `subject_mask_boundary` 只保留稀疏外轮廓，删除预处理 FSS 图像中的真实边缘信息。

Successful behavior:

- 当 subject boundary 对 Canny 的保留率过低时，候选会被标记为 invalid。
- `reject_reason = subject_mask_boundary_erases_preprocessed_fss_edges`

Remaining risk:

- 保留 Canny 会增加 primitive 数量，但比删除真实结构更安全。

## 2026-05-24: Outer Square Screenshot Frame Guard

Changed files:

- `geometry_driven_parameterizer.py`

Purpose:

- 识别并移除截图产生的外层近似正方形边框，避免它进入 FSS 参数化。

Detection thresholds:

- bbox 接近图像边界
- `span_x >= 0.82`
- `span_y >= 0.82`
- `0.85 <= w/h <= 1.18`
- approx polygon vertices <= 8
- 删除比例过高时拒绝删除

Validation sample:

- `test/test23.png`
- 外框 bbox 约为 `[36, 31, 620, 619]`
- 外框删除后拓扑验证保持通过

Remaining risk:

- 如果真实 FSS 本身就是贴边大方环，需人工检查清理图。

## 2026-05-24: Color Subject Boundary Candidate

Changed files:

- `geometry_driven_parameterizer.py`

Purpose:

- 对彩色 FSS 截图，从高饱和度导体区域提取边界，避免灰色截图框或白底干扰。

Successful behavior:

- `color_subject_boundary` 可作为边缘候选参与评分。
- 对 `test/test23.png` 一类彩色截图，能减少 Canny 双边缘/多 component 问题。

Remaining risk:

- 该候选只适合彩色导体截图。黑白 mask 或低饱和度 CAD 导出会回落到其他候选。

## 2026-05-24: Relaxed FSS Line Gate

Changed files:

- `geometry_graph_parameterizer.py`

Purpose:

- 减少长直 FSS 边缘因为 1-3 px 栅格抖动而落入 spline fallback。

Successful behavior:

- 当边缘高度线性、长度/弦长比接近 1、切向稳定时，允许更宽松地拟合为 line。

Remaining risk:

- 浅曲线若非常接近直线，可能被简化成 line。需要检查局部 fit JSON 和 preview。

## 2026-05-23: Graph Local Pipeline Documentation and Debug Runner

Changed files:

- `run_parameterization_only.py`
- `geometry_graph_parameterizer.py`
- `TEST_FILES_OVERVIEW_CN.txt`
- `TEST_FILES_OVERVIEW_EN.txt`

Purpose:

- 建立参数化阶段快速测试能力。
- 将 graph-local 输出分层组织为 debug folders。

Important folders:

```text
00_edges/
01_graph/
02_edge_split/
03_local_fit/
04_topology_validation/
05_export/
```

Remaining risk:

- 参数化-only runner 不验证 CST 端口、基板、地、solver 或 S 参数导出。

