# Auto_py2cst Project Pipeline README

本文档是项目总体架构说明，替代冗长的测试/维护混合型 overview。它只描述当前主流程、关键模块、数据产物和维护入口。

## 1. 项目目标

本项目将 FSS / 天线结构图片转换为可在 CST 中重建和仿真的参数化几何。

核心链路：

```text
输入图片
  -> FSS 图像处理 / repair_fig.png
  -> 边缘提取与候选边缘选择
  -> VTracer 中心线提取
  -> ordered centerlines
  -> Global B-spline 中间表示
  -> line / arc / spline 参数化
  -> parameterization JSON
  -> CST 建模 / 可选求解
```

拓扑正确性优先级高于视觉平滑度。CST 可重建性优先级高于参数最少。

## 2. 主入口

### 完整 CST 流程

文件：

- `fss_parameterized_cst_pipeline.py`

用途：

- 读取 instance JSON 或文件内 `DEFAULT_INSTANCE_DICT`
- 执行 FSS 图像清理
- 生成端口信息
- 执行参数化
- 根据参数化 JSON 创建 CST 工程
- 根据 `BUILD_ONLY` 决定是否运行 CST solver

典型阶段：

```text
1. Load Instance JSON
2. Image Preparation
3. Port Summary
4. Parameterization
5. CST Build / Simulation
```

输出目录：

```text
pipeline_runs/<run_name>/
  input_instance.json
  image_preparation.json
  prepared_instance.json
  port_summary.json
  pipeline_metadata.json
  01_fss_clean/
  02_parameterization/
  03_cst/
```

### 只跑 FSS + 参数化

文件：

- `run_parameterization_only.py`

用途：

- 用于小规模阶段测试
- 默认执行 FSS repair + 参数化
- 不创建 CST 工程
- 不运行 CST solver

推荐命令：

```powershell
D:\Anaconda\envs\linefor\python.exe .\run_parameterization_only.py --parameterization-mode graph_local_primitives
```

直接用原图跳过 FSS：

```powershell
D:\Anaconda\envs\linefor\python.exe .\run_parameterization_only.py --parameterization-mode graph_local_primitives --skip-fss-cleanup
```

## 3. FSS 图像处理阶段

主要文件：

- `fss_simulation_pipeline.py`
- `Rebuild/FssDetector.py`
- `Rebuild/fssdetector_pipeline.py`
- `Rebuild/fssdetector_selection.py`
- `Rebuild/fssdetector_clustering.py`
- `Rebuild/fssdetector_ocr.py`

输入：

- `layers.<layer>.img_path`
- `layers.<layer>.col_mats`

输出：

```text
01_fss_clean/<layer>/<image_stem>/<result_index>/repair_fig.png
```

职责：

- OCR / YOLO 文本预检查
- 子图选择
- 文字、箭头、线条等注释清理
- 颜色聚类
- 背景/导体区域分离
- 修复后的 `repair_fig.png` 输出
- 根据 `col_mats` 将 PEC 和背景颜色规范化

跳过规则：

- 完整流程默认尊重 instance 中的 skip 标记。
- `run_parameterization_only.py` 默认忽略 instance 中的 skip 标记，用于强制测试 FSS + 参数化。
- 显式 `--skip-fss-cleanup` 会直接使用原图。

## 4. 边缘提取阶段

主要文件：

- `Rebuild/NewParams.py`
- `geometry_driven_parameterizer.py`
- `geometry_graph_parameterizer.py`

边缘候选包括：

- `canny`
- `auto`
- `foreground_contour`
- `stroke_mask`
- `canny_gap_closed`
- `subject_mask_boundary`
- `color_subject_boundary`
- `canny_merged`
- patch / solid topology 相关候选

关键诊断输出：

```text
02_parameterization/<backend>/00_edges/
  edge_selection_diagnostics.json
  edge_candidate_scores.json
  _candidate_*/repair_fig_edges.png
```

选择原则：

- 避免过度过滤真实 FSS 边缘
- 避免截图外框误判为结构
- 避免 Canny 双边缘导致拓扑膨胀
- 对预处理后的 FSS 图像优先保留真实结构信息

## 5. 参数化模式

当前代码中存在四类参数化入口。完整 CST pipeline 支持：

```text
standard
optimized_bs_seed
geometry_primitives
graph_local_primitives
graph_local_lines
```

参数化阶段测试 runner 支持：

```text
standard
geometry_primitives
graph_local_primitives
graph_local_lines
```

### standard

入口：

- `Rebuild.NewParams`
- `vtracer_python.py`

流程：

```text
repair_fig.png
  -> edge extraction
  -> VTracer centerline / contours
  -> semantic line / arc / spline
  -> curve_parameterization.json
```

特点：

- 最稳定的基础 fallback
- 输出 sampled / resampled points
- CST 端可直接用点列重建

### optimized_bs_seed

入口：

- `fss_parameterized_cst_pipeline.py`
- `core.geometry.optimized_bspline_fitter`

流程：

```text
repair_fig.png
  -> centered repair image / edges
  -> optimized Global B-spline fitting
  -> seed polylines
  -> VTracer seed mode
  -> aggregate parameterization JSON
```

特点：

- 属于历史实验路径。
- 目标是把优化后的 B-spline 结果作为 VTracer / 参数化 seed。
- 目前主要用于完整 pipeline 内对比，不作为小规模阶段测试的默认入口。

### geometry_primitives

入口：

- `geometry_driven_parameterizer.py`

主流程：

```text
repair_fig.png
  -> adaptive edge preprocessing
  -> VTracer centerline extraction
  -> ordered centerlines
  -> Global B-spline intermediate
  -> primitive decomposition
  -> compact JSON
  -> CST
```

核心思想：

- Global B-spline 不是最终表达，而是用于连续化和降噪的中间层。
- 最终仍导出 line / arc / spline primitives 和 fallback sampled points。

保护机制：

- B-spline length shrink ratio gate
- B-spline RMS error gate
- topology fallback
- standard pipeline fallback
- line-priority over arc/spline

失败链：

```text
geometry_primitives -> standard
```

### graph_local_primitives

入口：

- `geometry_graph_parameterizer.py`

主流程：

```text
repair_fig.png
  -> adaptive edge preprocessing
  -> VTracer centerline extraction
  -> skeleton graph extraction
  -> graph edge split
  -> local line / arc / spline fitting
  -> graph-aware JSON
  -> CST sampled-point fallback
```

设计目标：

- 替代全局平滑造成的拓扑失真
- 保留 junction / branch / loop / corner
- 每条 graph edge 独立拟合
- primitive 优先级：line > arc > local spline

注意：

- 这是 topology-first 实验增强模式。
- 适合分析复杂 FSS 连通关系。
- 当前主 README 仍把 `geometry_primitives` 的 Global B-spline 流程作为传统主线，同时保留 graph-local 作为拓扑保护增强。

失败链：

```text
graph_local_primitives -> geometry_primitives -> standard
```

### graph_local_lines

入口:
- `geometry_graph_parameterizer.py`
- `fss_parameterized_cst_pipeline.py`

主流程:

```text
repair_fig.png
  -> adaptive edge preprocessing
  -> VTracer centerline extraction
  -> skeleton graph extraction
  -> graph edge split
  -> forced local line fitting
  -> line-only graph-aware JSON
  -> CST compact line reconstruction
```

设计目标:
- 复用 `graph_local_primitives` 的拓扑保护能力。
- 禁止 arc / spline primitive 输出；曲线区域会按采样折线拆成多段 `line` primitive。
- 输出 `backend=graph_local_lines`，`metadata.line_only_parameterization=true`。
- component 的 `fallback_points` / `resampled_points` 保存折线采样顶点，避免把整段曲线压成首尾一根弦。
- 在折线顶点生成后，会按滑动三点组检查局部冗余点：如果三个点彼此很近，并且两段局部斜率没有正负突变，就删除中间点，直接连接首尾点。
- 对 `长线段 -> 微小偏移点 -> 长线段` 这类肉眼可见的冗余 start point，会额外检查中间点到首尾直连弦的误差；误差足够小且方向没有突变时，同样删除中间点。
- 对 degree=2 的连续图边，会按全局点顺序串成闭合链后再做一轮 `123 / 234 / 345` 滑动三点合并，因此跨 edge 边界的 start point 也会被检查。
- 三点合并会同步更新该 component 的 `fallback_points` / `resampled_points` 顺序，并重新生成后续 `line` segments；默认阈值为 `LINE_TRIPLET_MERGE_DISTANCE_PX=3.0`、`LINE_TRIPLET_MERGE_MAX_ANGLE_DEG=35.0`。

使用命令:

```powershell
D:\Anaconda\envs\linefor\python.exe .\fss_parameterized_cst_pipeline.py --parameterization-mode graph_local_lines
```

可选调参:

```powershell
D:\Anaconda\envs\linefor\python.exe .\fss_parameterized_cst_pipeline.py --parameterization-mode graph_local_lines --line-triplet-merge-distance-px 3.0 --line-triplet-merge-max-angle-deg 35.0
```

失败策略:

```text
graph_local_lines is strict line-only mode.
It does not fall back to geometry_primitives or standard because those modes may emit arc/spline primitives.
```

## 6. 参数化 JSON

常见文件：

```text
02_parameterization/curve_parameterization.json
```

重要字段：

- `backend`
- `components`
- `primitives`
- `fallback_points`
- `resampled_points`
- `metadata`

graph-local 模式还包含：

- `nodes`
- `edges`
- `constraints`
- `source_edge_id`
- `start_node`
- `end_node`

CST 兼容原则：

- compact primitives 优先
- primitive 失败时使用 `fallback_points`
- sampled points 永远是最终安全重建路径

## 7. CST 建模阶段

主要文件：

- `parameterized_json_to_cst.py`
- `Simulink/`
- `Simulink/Simulation.py`

核心类：

- `ParameterizedJsonCSTBuilder`
- `CSTParametricConfig`

流程：

```text
curve_parameterization.json
  -> load components
  -> map image coordinates to CST geometry frame
  -> draw curves
  -> extrude PEC
  -> subtract holes
  -> add substrate / ground
  -> add waveguide or patch ports
  -> save CST project
  -> optional solver
```

输出：

```text
03_cst/<project_name>.cst
```

可选行为：

- `BUILD_ONLY=True`: 只建模，不仿真
- `BUILD_ONLY=False`: 建模并运行 solver

## 8. 推荐调试顺序

### 只检查 FSS + 参数化

```powershell
D:\Anaconda\envs\linefor\python.exe .\run_parameterization_only.py --parameterization-mode graph_local_primitives
```

检查：

- `image_preparation.json`
- `01_fss_clean/.../repair_fig.png`
- `02_parameterization/.../edge_selection_diagnostics.json`
- `02_parameterization/.../graph_primitives_metrics.json`
- `02_parameterization/.../05_export/graph_primitives_preview.png`

### 直接跳过 FSS 检查边缘/参数化

```powershell
D:\Anaconda\envs\linefor\python.exe .\run_parameterization_only.py --parameterization-mode graph_local_primitives --skip-fss-cleanup
```

### 完整 CST 建模

```powershell
D:\Anaconda\envs\linefor\python.exe .\fss_parameterized_cst_pipeline.py
```

## 9. 维护原则

- 不要把实验日志继续堆进 README。
- 架构说明只记录稳定模块和当前推荐流程。
- 成功改动记录写入 `OverViewLog.md`。
- 失败、回退、临时调试记录不进入主 README。
- 修改 FSS 框架前，应先确认 standalone FSS 行为和 pipeline runner 行为差异。
- topology correctness > CST reconstructability > primitive compactness > smoothness.
