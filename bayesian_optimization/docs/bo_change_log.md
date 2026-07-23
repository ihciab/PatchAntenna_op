# BO Change Log

## 2026-07-11 - Central Bayesian Optimization Config

### Modified Files

- `beyesian_opconfig.json`
- `bayesian_optimization/pipelines/optimization_pipeline.py`
- `bayesian_optimization/optimization/multistage.py`
- `docs/bo_change_log.md`
- `bayesian_optimization/docs/bo_change_log.md`

### Summary

Added a root-level `beyesian_opconfig.json` file for the main BO run settings.
It contains the CST S11 frequency range, stage trial counts, stage loss
weights, full objective weights, optimizer hyperparameters, stopping settings,
geometry settings, and port-connection parameters.

The editor-run pipeline now merges this JSON into the legacy editor config. The
configured frequency range is applied to the CST builder config, so the current
`9.4-10.8 GHz` range is used by simulation. Stage objective weights are now
configurable through `stage_loss_weights`, and Optuna/skopt hyperparameters can
be adjusted without editing Python code.

## 2026-07-09 - Stage4 Topology Exploration

### Modified Files

- `bayesian_optimization/optimization/multistage.py`
- `bayesian_optimization/geometry/primitive_mutator.py`
- `bayesian_optimization/pipelines/optimization_pipeline.py`
- `docs/bo_change_log.md`
- `bayesian_optimization/docs/bo_change_log.md`

### Summary

Added an optional Stage4 local-escape phase after Stage1-Stage3 multi-stage
optimization. Stage4 freezes global electrical scale, port size, port position,
feedline width, and feedline position, then moves exactly one eligible conductor
contour point per trial using `stage4_delta_x` and `stage4_delta_y`.

Stage4 uses the best successful Stage3 payload as its non-cumulative reference.
The selected point is scheduled cyclically by `StageManager`; Optuna only
samples the point move in the bounded `[-stage4_delta_px, +stage4_delta_px]`
window. The default editor configuration uses `STAGE4_TRIALS = 20` and
`STAGE4_DELTA_PX = 7.0`.

Automatic CST geometry repair is disabled in Stage4. Invalid geometry,
duplicate vertices, broken segments, and validation failures skip CST
build/simulation and return the invalid-geometry penalty. Valid Stage4 trials
use the full Stage3 objective terms and write logs for the selected point,
move vector, geometry validity, ERES, EBW, and loss.

Stage4 is controlled by `stage4_trials`; setting it to `0` restores the earlier
three-stage multi-stage behavior.

## 2026-06-18 - Multi-Stage Bayesian Optimization

### Modified Files

- `bayesian_optimization/optimization/multistage.py`
- `bayesian_optimization/geometry/primitive_mutator.py`
- `bayesian_optimization/pipelines/optimization_pipeline.py`
- `bayesian_optimization/simulation/parameterized_json_to_cst.py`
- `docs/bo_change_log.md`
- `bayesian_optimization/docs/bo_change_log.md`

### Summary

Added an optional three-stage BO mode controlled by
`enable_multistage_optimization`. Stage 1 samples only `global_scale_x`,
`global_scale_y`, and `port_width_scale` for frequency locking. Stage 2 fixes
the best Stage 1 scale values and runs the existing local shape variables.
Stage 3 jointly fine tunes local variables and scale variables in a +/-5%
window around the Stage 1 best. The default remains disabled, so existing
single-stage behavior is preserved.

Multi-stage evaluations also write two explicit geometry trace files:
`mutation_stage_curve_parameterization.json` captures the sampled/scaled
parameterization before CST handoff and repair, while
`cst_input_curve_parameterization.json` captures the repaired payload that is
used as the CST input alongside `curve_parameterization.json`.

The CST builder now validates compact primitive endpoint continuity before
using primitives for `ExtrudeCurve`. When port-width scaling leaves primitive
gaps, CST handoff falls back to the closed sampled polygon to avoid projects
that contain only substrate and ground.

## 2026-05-30 - Feature-Constrained Topology-Preserving Shape Optimization

### 修改文件

- `feature_shape_optimizer.py`
- `primitive_mutator.py`
- `optimization_pipeline.py`
- `docs/bo_change_log.md`

### 修改原因

原 BO 几何层以单控制点 `dx/dy` 为变量，容易产生尖刺、锯齿、局部凸起/凹陷，并且会让 Port 附近点改变馈线宽度和 Port 接触面积。新实现将优化层从单点位移升级为特征优先、连续点组兜底的形状优化层。

### 新增功能

- 新增 Feature Detection，识别并报告 `Feedline`、`Patch`、`Slot`、`Meander` 区域。
- BO 变量优先级改为：Level 1 `feed_width/feed_length/inset_depth`，Level 2 `slot_length/slot_width/slot_position`，Level 3 `patch_edge_offset`，Level 4 `group_###_offset_distance`。
- 废弃默认单控制点 BO 变量，不再生成 `control_point_dx/control_point_dy` 或 `point_id_dx/point_id_dy`。
- 新增连续 Point Group 生成，组变量以法向距离方式整体作用于连续点段。
- 新增 Boundary Normal Offset 变形模式，并输出 `normal_offset_debug.json`。
- 新增 Port 三层约束：`PORT_CORE` 完全冻结，`PORT_NEIGHBOR` 仅允许沿馈线传播方向运动，`NORMAL_REGION` 允许特征/组法向变形。
- 新增 feedline 特征变量，保证 Feedline 中心线连续，Port 核心连接区域冻结。
- 新增 manufacturability validator 与 spike detector，检测最小边长、尖角、曲率突变、极窄尖刺；发现错误后在 CST 前按 invalid geometry 回滚。
- 新增每轮调试输出：`optimization_variable_report.json`、`group_overlay.png`、`normal_offset_overlay.png`、`port_constraint_overlay.png`、`feature_regions.png`、`geometry_before_after.png`、`deformation_vectors.png`、`port_constraint_report.json`。

### 风险分析

- Feature Detection 仍基于几何启发式，复杂天线结构可能把部分分支/槽误分到 Patch 或 Meander；已保留 Group Normal Offset 作为兜底。
- Port 方向自动识别依赖边界位置和馈线带宽估计；异常输入图或非边界馈电结构需要后续人工标注或更强语义输入。
- 制造友好检查在优化层做保守拦截，可能拒绝一部分电磁上可行但局部很尖锐的探索样本；这符合本轮“工程化轮廓优先”的目标。
- 旧 `ControlPointDeformer` 和 `constrained_shape_optimizer.py` 未删除，避免破坏已有诊断/历史文件；主优化入口已切换到新特征约束层。

## 2026-05-30 - Invalid Geometry Rollback Policy

### 修改文件

- `optimization_pipeline.py`
- `docs/bo_change_log.md`
- `logs/algorithm_updates.md`

### 修改原因

控制点可移动范围增大后，BO 更容易采样到非法几何。如果这些非法几何继续计入 invalid ratio 或 no-improvement patience，会过早触发停止条件，导致优化空间探索不足。

### 新增功能

- 将 `invalid_geometry` 样本改为 rollback-and-continue 语义：
  - 记录非法样本和 validation report。
  - 给优化器反馈 large penalty。
  - 跳过 CST build/simulation。
  - 不更新 best design。
  - 不消耗 no-improvement patience。
  - 不再触发 invalid geometry ratio early-stop。
- `run_metadata.json` 新增 `invalid_geometry_policy`，明确当前非法几何处理策略。
- 保留 `MAX_INVALID_RATIO` / `--max-invalid-ratio` 配置字段，避免破坏现有命令和配置文件兼容性。

### 风险分析

- BO 会更倾向于跑满最大轮数，非法几何比例高时总耗时可能增加。
- 非法几何仍会反馈惩罚给优化器，因此优化器可以学习避开坏区域，但不会因为坏区域多而提前退出。

## 2026-05-29 - Code Readability And Key Function Markers

### 修改文件

- `optimization_pipeline.py`
- `constrained_shape_optimizer.py`
- `deformation_engine.py`
- `primitive_mutator.py`
- `docs/bo_change_log.md`

### 修改原因

为了方便人工阅读和后续调试，对 Bayesian Optimization 相关代码进行可读性增强。此次修改只增加模块说明、函数 docstring、分区标记和关键函数标记，不改变优化变量、采样策略、目标函数、几何验证、CST Builder 或仿真流程。

### 新增功能

- 为 BO 主流程、控制点选择、控制点变形、变量提取适配层增加模块级说明。
- 为所有 class/function 补充 docstring。
- 在关键函数处增加 `【关键函数】` 标记，包括：
  - BO 主循环 `OptimizationPipeline.run`
  - 单轮评估 `OptimizationPipeline.evaluate`
  - CST handoff `prepare_cst_handoff_payload`
  - 控制点计划生成 `build_control_point_optimization_plan`
  - quota 选择 `select_by_quota`
  - 控制点范围生成 `point_ranges`
  - 控制点 offset 应用 `ControlPointDeformer.apply_offsets`
  - 几何变异入口 `mutate_geometry`
- 增加分区注释，明确 optimizer backend、run state、主循环、调试输出等代码边界。

### 风险分析

- 本次为注释和可读性修改，不应改变运行结果。
- 若后续人工继续重构，需保持当前边界：不要改 CST Builder、objective、geometry validation 和 parameterization schema。

## 2026-05-29 - Quota Selection, Class-Based Range And Diagnostics

### 修改文件

- `constrained_shape_optimizer.py`
- `deformation_engine.py`
- `optimization_pipeline.py`
- `control_point_constraints.json`
- `docs/bo_change_log.md`

### 修改原因

当前控制点选择过度依赖 Top-K score，容易集中在外边界直线区域；统一 `[-1, 1]` 位移范围也导致几何扰动过小，BO 难以观察到谐振频率、S11 和带宽变化。本次只增强变量选择、变量范围、可视化和诊断层，不修改 CST Builder、仿真流程、目标函数、几何验证或参数化 schema。

### 新增功能

- 废弃纯 Top-K 选择，新增 Quota-Based Selection：
  - `RESONANT`: 4 个
  - `FEEDLINE`: 3 个
  - `STRUCTURAL`: 3 个
  - `EXPLORATION`: 2 个
- 新增 `minimum_point_spacing` 空间分散约束，默认 20 px。
- 增强 resonance score，对槽/拐角/馈线连接附近高曲率点提高优先级。
- Feedline 点强制进入变量空间，并输出 `feedline_selection_report.json`。
- 新增按类别搜索范围：
  - `PORT`: 固定 0
  - `STRUCTURAL`: `[-2, 2]`
  - `FEEDLINE`: `[-4, 4]`
  - `RESONANT`: `[-6, 6]`
- 新增局部尺度自适应范围：`max_disp = displacement_ratio * local_feature_scale`，并加入全局结构尺度下限，避免密集采样点把变量范围压得过小。
- 每轮 evaluation 新增：
  - `selection_quota_report.json`
  - `point_distribution_report.json`
  - `feedline_selection_report.json`
  - `deformation_statistics.json`
- 新增调试可视化：
  - `selected_points_heatmap.png`
  - `selection_category_overlay.png`
  - `coverage_map.png`
  - `displacement_histogram.png`
- 优化结束后新增变量敏感度诊断：
  - `plots/parameter_sensitivity_heatmap.png`
  - `most_sensitive_variables.json`

### 风险分析

- 控制点分类仍基于几何启发式，不等价于真实电流分布；后续可结合 CST 场分布或人工标注继续校准。
- 增大位移范围会提高无效几何概率，但现有 validation/rejection 层仍会在 CST 前拦截非法结构。
- 敏感度分析使用历史样本相关性，样本数较少时只作为下一轮筛选参考，不应被视为严格因果结论。

## 2026-05-29 - Control Point Classification And Debug Enhancement

### 修改文件

- `constrained_shape_optimizer.py`
- `deformation_engine.py`
- `primitive_mutator.py`
- `optimization_pipeline.py`
- `requirements.txt`
- `docs/bo_change_log.md`

### 修改原因

增强 Bayesian Optimization 的受约束控制点位移优化层，避免仅依赖简单高曲率筛选，同时为 port / feedline / symmetry / selected points 提供可追踪的调试输出。

### 新增功能

- 新增控制点分类：
  - `PORT`
  - `FEEDLINE`
  - `RESONANT`
  - `STRUCTURAL`
- 新增 Port 点自动冻结逻辑，PORT 点不会进入 BO 变量空间。
- 新增 `port_constraints_report.json`。
- 新增 feedline group 报告：
  - `feedline_groups.json`
- 新增 symmetry group 检测与报告：
  - `symmetry_groups.json`
- 新增控制点 selection score：
  - curvature score
  - resonance score
  - symmetry bonus
  - topology risk penalty
- 新增 `point_selection_scores.json`。
- 新增 `point_classification.json`。
- 每轮 evaluation 新增：
  - `evaluation_summary.json`
  - `moved_points_overlay.png`
  - `displacement_vectors.png`
  - `point_id_overlay.png`
  - `top_selected_points.png`
  - `geometry_before_after.png`
  - `symmetry_debug.png`
- 优化结束后尝试生成：
  - `plots/optimization_animation.gif`

### 潜在风险

- 当前 PORT / FEEDLINE / RESONANT 分类基于几何启发式规则，尚未直接读取完整 EM 语义，因此需要结合具体天线结构持续校准。
- symmetry group 当前只用于检测和可视化报告，尚未把成对点压缩为共享 `symmetry_dx` 变量；后续可以在不改 CST builder 的前提下继续增强。
- feedline group 当前输出约束报告和调试信息，尚未引入 `feed_width/feed_length` 独立参数变量。
- 参数重要性和动画依赖可选库；缺失时会跳过，不影响 BO 主流程。
