# Design Agent 总控 Pipeline 中文说明

## 运行入口

默认读取 `design_agent/agent_config.json`：

```powershell
python -m design_agent.scripts.run_design_agent_pipeline
```

Python 调用：

```python
from design_agent import DesignAgent

agent = DesignAgent.from_config_file("config.json")
result = agent.run_pipeline_from_config("design_agent/agent_config.json")
print(result.manifest_path)
```

临时覆盖少量参数：

```powershell
python -m design_agent.scripts.run_design_agent_pipeline --iterations 2 --run-prefix design_test
```

## 输出文件夹规范

每次运行自动创建编号目录：

```text
design_agent_runs/
  degsin_test_1/
  degsin_test_2/
  degsin_test_3/
```

默认前缀是 `degsin_test`，保持当前约定；如果需要改成 `design_test`，修改
`agent_config.json` 的 `run_folder.run_prefix`。

每个运行目录内部结构：

```text
degsin_test_N/
  agents_inputs/
  01_closed_loop/
  02_bayesian_optimization/
  manifests/
  pipeline_manifest.json
```

## 配置文件

总控 pipeline 的路径来源和超参数集中写在：

`design_agent/agent_config.json`

相对路径按项目根目录解析。

### run_folder

输入：

- `run_root`：编号运行目录的父目录。
- `run_prefix`：运行目录前缀，默认 `degsin_test`。
- `run_index`：指定编号；为 `null` 时自动取下一个编号。

输出：

- `design_agent_runs/degsin_test_N/`

### input_paths

输入：

- `seed_input_dir`：初始 `agents_inputs` 来源。
- `source_run_dir`：提供 `stackup.json`、初始设计上下文。
- `config_path`：LLM client 配置。
- `initial_geometry_json`：可选，强制指定当前 Geometry Engine JSON。

输出：

- 本次运行私有的 `degsin_test_N/agents_inputs/`

### closed_loop

输入：

- `iterations`：LLM 几何修改轮数。
- `geometry_only`：只运行几何，不跑 CST。
- `build_only`：只建 CST，不跑 solver。
- `f0_ghz` / `f1_ghz`：CST 频率范围。
- `target_frequency_ghz`、`target_s11`、`target_gain` 等目标。
- `max_geometry_repair_attempts`：Geometry Engine 报错后让 LLM 修复计划的次数。

输出：

- `01_closed_loop/iter_XXX/diagnosis.json`
- `01_closed_loop/iter_XXX/plan.json`
- `01_closed_loop/iter_XXX/operation_plan.json`
- `01_closed_loop/iter_XXX/geometry_engine_geometry.json`
- `01_closed_loop/iter_XXX/geometry_summary.json`
- `01_closed_loop/iter_XXX/simulation_summary.json`，如果实际跑了 solver
- 刷新后的 `agents_inputs/geometry_summary.json`
- 刷新后的 `agents_inputs/simulation_summary.json`
- 刷新后的 `agents_inputs/history.json`

### bo

输入：

- `prepare_bo`：是否在闭环后准备 BO 输入。
- `execute_bo`：是否真正执行 BO；当前配置为 `true`。
- `bo_max_evaluations`：BO 评估次数，执行 BO 时需要。
- `bo_build_only`：BO 阶段是否只建模。
- `bo_target_json`：BO 目标文件。
- `bo_f0_ghz` / `bo_f1_ghz`：BO CST 仿真频率范围。
- `optimizer_backend`：优化后端，例如 `optuna`。
- `enable_multistage_optimization`：是否启用多阶段优化。

输出：

- `agents_inputs/curve_parameterization.json`
- `agents_inputs/patch_port_summary.json`
- `agents_inputs/primitive_analysis.json`
- `agents_inputs/geometry_engine_bo_adapter_metadata.json`
- `agents_inputs/bo_variable_plan.json`
- `agents_inputs/bo_parameterization_summary.json`
- BO handoff manifest：
  `02_bayesian_optimization/design_agent_manifests/*.json`

当前 `agent_config.json` 已设置 `execute_bo=true`，会输出实际 BO run directory。

## 模块调用顺序

### 1. DesignAgentPipelineRunner

模块：

`design_agent.pipeline.DesignAgentPipelineRunner`

输入：

- `design_agent/agent_config.json`

输出：

- 编号运行目录
- `pipeline_manifest.json`

### 2. seed_agents_inputs

模块：

`DesignAgentPipelineRunner._seed_agents_inputs`

输入：

- `seed_input_dir/target.md`
- `seed_input_dir/geometry_summary.json`
- `seed_input_dir/simulation_summary.json`
- 可选 BO 和 history 文件

输出：

- `degsin_test_N/agents_inputs/*`

### 3. ClosedLoopDesignRunner

模块：

`design_agent.scripts.run_closed_loop_design.ClosedLoopDesignRunner`

输入：

- `agents_inputs/target.md`
- `agents_inputs/geometry_summary.json`
- `agents_inputs/simulation_summary.json`
- `agents_inputs/history.json`
- 可选 `agents_inputs/bo_parameterization_summary.json`

输出：

- LLM 诊断、计划、几何操作
- Geometry Engine JSON
- geometry summary
- simulation summary
- history

### 4. Geometry Engine 到 BO 适配

模块：

`design_agent.tools.bo_adapter.convert_geometry_engine_to_bo`

输入：

- 最新 `geometry_engine_geometry.json`
- `source_run_dir/stackup.json`

输出：

- `curve_parameterization.json`
- `patch_port_summary.json`
- `primitive_analysis.json`
- BO adapter metadata

### 5. BO Handoff / BO 执行

模块：

`design_agent.tools.bayesian_optimization_runner.BayesianOptimizationAgentRunner`

输入：

- BO parameterization
- port summary
- primitive analysis
- `target.json`
- BO 超参数

输出：

- BO manifest
- `bo_variable_plan.json`
- `bo_parameterization_summary.json`
- 可选实际 BO 优化结果

## 当前注意事项

- 当前总控 pipeline 从已有 `seed_input_dir` 启动，还没有把“初始 LLM 设计生成”自动并入第一步。
- 初次设计如果没有仿真，仍需要一个占位 `simulation_summary.json`，因为当前轻量 LLM skill 会读取它。
- BO 后 LLM 主要读取 `bo_parameterization_summary.json`。BO 最优设计转回 Geometry Engine JSON 的阶段还未实现。
