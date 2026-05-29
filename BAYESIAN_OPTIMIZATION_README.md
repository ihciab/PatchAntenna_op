# Bayesian Optimization Geometry Optimization README

本文档总结当前贴片天线 Bayesian Optimization 相关新增代码。该优化层是外挂式实验模块，位于：

```text
curve_parameterization.json
  -> design variable extraction
  -> constrained geometry deformation
  -> geometry validation / repair
  -> CST build / simulation
  -> S11 extraction
  -> objective evaluation
  -> optimization history / plots
```

现有图像处理、参数化 backend、CST builder 和原 pipeline 主流程不被重构。

## 1. 主要文件

```text
optimization_pipeline.py
```

BO 总控入口。负责读取参数化 JSON、创建优化器、执行每轮变形、验证、CST build/simulation、S11 解析、目标函数计算和可视化输出。

```text
primitive_mutator.py
```

BO 变量提取和几何 mutation 入口。当前优先使用受约束控制点位移变量；如果约束配置不可用或选不出控制点，回退到 `global_scale`。

```text
constrained_shape_optimizer.py
deformation_engine.py
geometry_constraint_validator.py
control_point_constraints.json
```

受约束控制点局部形变优化层。核心思想是优化控制点位移 `dx/dy`，而不是优化控制点绝对坐标。

```text
geometry_validation.py
```

CST 前几何合法性验证与保守修复层。用于避免 open curve、非平面、自交、tiny edges 等导致 CST 建模失败的问题。

```text
s11_parser.py
optimization_objectives.py
```

S11 结果解析和目标函数计算。

## 2. 优化变量设计

### 当前优先变量：受约束控制点位移

不是直接优化：

```text
P_i = (x_i, y_i)
```

而是优化局部位移：

```text
P_i' = P_i + Δ_i
Δ_i = (dx_i, dy_i)
```

BO 变量形式：

```text
c010_p012_dx
c010_p012_dy
c003_p004_dx
c003_p004_dy
...
```

其中：

```text
c010_p012
```

表示第 10 个 component 的第 12 个控制点。

默认范围：

```text
dx ∈ [-1.0, 1.0]
dy ∈ [-1.0, 1.0]
```

默认最多选择：

```text
8 个 movable points
16 个 BO dimensions
```

总维度限制：

```text
≤ 20
```

配置文件：

```text
control_point_constraints.json
```

示例：

```json
{
  "enabled": true,
  "auto_select": true,
  "max_movable_points": 8,
  "max_dimensions": 20,
  "default_dx_range": [-1.0, 1.0],
  "default_dy_range": [-1.0, 1.0],
  "freeze_topology_endpoints": true,
  "point_constraints": {
    "c010_p012": {
      "movable": true,
      "dx_range": [-0.5, 0.5],
      "dy_range": [-1.0, 1.0],
      "priority": "high"
    }
  }
}
```

### 自动可动点选择策略

默认只选择局部关键点：

```text
high curvature points
internal control points
local resonance-sensitive candidates
```

默认冻结：

```text
loop closure points
topology anchor points
component endpoints
port boundary / graph endpoints
```

这样可以避免高维 BO 崩溃，也避免拓扑被破坏。

### 回退变量：global_scale

如果控制点优化未启用，或没有可移动点，则回退为：

```text
global_scale ∈ [0.92, 1.08]
```

该变量是 topology-preserving uniform scale。

## 3. 优化逻辑

每一轮 BO 逻辑：

```text
1. optimizer.ask()
2. 生成控制点 dx/dy 变量
3. apply_control_point_offsets()
4. graph-local open edges -> CST closed-loop handoff
5. geometry validation / repair
6. 如果 invalid：跳过 CST，objective = large penalty
7. 如果 valid：进入 CST build / simulation
8. 读取 S11
9. 计算 objective
10. optimizer.tell()
11. 保存 history / plots / best_design
```

当前优化器 backend：

```text
optuna_tpe
```

如果 Optuna 不可用，会 fallback：

```text
skopt_gp_ei
```

编辑器配置位于 `optimization_pipeline.py` 顶部：

```python
EDITOR_RUN_CONFIG = {
    "BASE_RUN_DIR": PROJECT_ROOT / "pipeline_runs" / "run_20260528_213437",
    "RUN_NAME": "bo_editor_full30_213437",
    "MAX_EVALUATIONS": 30,
    "TARGET_FREQUENCY_GHZ": 10.0,
    "TARGET_S11_DB": -10.0,
    "BUILD_ONLY": False,
    "OPTIMIZER_BACKEND": "optuna"
}
```

注意：如果 `BUILD_ONLY=True`，不会运行 solver，因此不会有 S11 和谐振频率。

## 4. 几何验证与修复

CST 前会执行 Python 内几何验证，不依赖 CST。

验证项：

```text
closed loop
segment connectivity
planarity
self intersection
minimum edge length
polygon area
duplicate points
minimum gap
curvature smoothness
```

保守修复项：

```text
remove consecutive duplicate points
merge / remove degenerate tiny edges
snap tiny open gaps
force closure for small closure gap
flatten tiny z deviation
```

禁止自动修复：

```text
self intersection
large topology break
large disconnected gap
zero-area polygon
```

如果 invalid：

```text
skip CST
objective = 100.0 penalty
record invalid reason
```

验证输出：

```text
valid_designs/eval_xxx/validation_logs/
  validation_report.json
  repaired_geometry.json
  invalid_geometry.json
  repair_history.txt
```

调试图：

```text
valid_designs/eval_xxx/geometry_debug_plots/
  original_geometry.png
  repaired_geometry.png
  invalid_features.png
```

## 5. CST Handoff 逻辑

`graph_local_primitives` 有时会把闭合 patch 轮廓拆成多条 open edge：

```text
1 -> 2
2 -> 3
...
12 -> 1
```

CST 的 `ExtrudeCurve` 要求闭合曲线。因此 BO 层会在 CST 前生成临时 handoff JSON：

```text
open graph edges
  -> merged closed loop component
  -> CST builder
```

这一步不改原始 parameterization schema，也不改 CST builder，只改变 BO 送入 CST 的临时 JSON。

输出：

```text
valid_designs/eval_xxx/
  mutation_raw_curve_parameterization.json
  curve_parameterization.json
  cst_handoff_validation.json
```

其中：

```text
mutation_raw_curve_parameterization.json
```

是控制点变形后的原始 graph-local JSON。

```text
curve_parameterization.json
```

是给 CST builder 使用的闭合 handoff JSON。

## 6. S11 解析

S11 文件由 `s11_parser.py` 解析。

支持：

```text
s11.csv
*_s11.csv
*_s11.txt
```

提取指标：

```text
resonant_frequency_ghz
minimum_s11_db
s11_at_target_db
bandwidth_ghz
point_count
```

谐振频率定义：

```text
f_resonance = frequency at minimum S11
```

## 7. Objective Function

目标函数在 `optimization_objectives.py` 中。

主目标：

```text
minimize |f_resonance - f_target|
```

同时考虑目标频点 S11：

```text
s11_at_target_db
```

次级 penalty：

```text
geometry complexity
spline curvature
tiny segments
topology instability
CST failure
invalid geometry
```

典型 penalty：

```text
invalid geometry: 100.0
CST failure: 80.0
```

如果 solver 成功，objective breakdown 会写入：

```text
optimization_history.json
```

## 8. 可视化输出

每次优化 run 输出到：

```text
optimization_runs/<run_name>/plots/
```

当前图包括：

```text
objective_history.png
```

每轮 objective 变化。

```text
variable_objective.png
```

优化变量与 objective 的关系。

```text
evaluation_status.png
```

每轮状态统计，例如 completed / invalid_geometry / cst_failed。

```text
resonance_history.png
```

每轮谐振频率和 minimum S11。该图只有 solver 运行并成功导出 S11 后才会生成。

```text
best_s11_curve.png
```

当前最佳设计的 S11 曲线。

```text
optimizer_param_importance.png
```

优化器参数重要性。Optuna 需要足够 trial 数才能计算。

每个 evaluation 的几何调试图：

```text
valid_designs/eval_xxx/geometry_debug_plots/
valid_designs/eval_xxx/deformation_debug/
```

## 9. 优化结果查看

核心结果目录：

```text
optimization_runs/<run_name>/
```

重点文件：

```text
optimization_history.json
optimizer_trials.json
best_design/best_record.json
best_design/curve_parameterization.json
logs/optimization.log
plots/
```

最佳设计：

```text
best_design/best_record.json
```

其中包含：

```json
{
  "objective": 0.123,
  "variables": {
    "c010_p012_dx": 0.2,
    "c010_p012_dy": -0.1
  },
  "s11_metrics": {
    "resonant_frequency_ghz": 10.1,
    "minimum_s11_db": -15.2,
    "s11_at_target_db": -12.8,
    "bandwidth_ghz": 0.4
  }
}
```

## 10. 推荐运行方式

使用 `paper` 环境：

```powershell
D:\Anaconda\envs\paper\python.exe optimization_pipeline.py
```

当前编辑器配置默认：

```text
MAX_EVALUATIONS = 30
BUILD_ONLY = False
OPTIMIZER_BACKEND = optuna
```

烟测时建议临时改成：

```python
"MAX_EVALUATIONS": 3,
"BUILD_ONLY": False
```

只测试几何和 CST build 时：

```python
"MAX_EVALUATIONS": 1,
"BUILD_ONLY": True
```

注意：只有 `BUILD_ONLY=False` 才能生成 S11、谐振频率和 `resonance_history.png`。

## 11. 当前设计原则

本优化层遵守：

```text
Topology-Preserving Local Shape Optimization
```

不是自由拓扑生成。

核心约束：

```text
不破坏原始拓扑
不优化所有控制点绝对坐标
不让 BO 维度过高
不让 invalid geometry 进入 CST
优先保证 CST reconstructability
```

