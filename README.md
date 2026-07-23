# Auto_py2cst_v0.71 项目总览

本项目用于从 FSS/天线结构图像和参数化 JSON 出发，完成几何参数化、CST 建模/仿真、S11 结果解析，以及贝叶斯优化闭环。

## 当前目录结构

根目录现在主要保留两类文件：

- `test_*.py`：测试脚本和验证入口。
- `*.json`：项目输入配置，例如 `config.json`、`pipeline_test_instance.json`、`control_point_constraints.json`。

贝叶斯优化相关实现已集中到 `bayesian_optimization/`：

```text
bayesian_optimization/
  docs/              文档、README、变更记录
  geometry/          几何参数化、primitive 分析、形变、约束和验证
  logs/              项目整理前已有的日志和算法更新记录
  optimization/      BO 目标函数、S11 解析、约束优化辅助模块
  pipelines/         参数化 pipeline、CST pipeline、BO 主流程入口
  simulation/        参数化 JSON 到 CST 的构建适配层
  tools/             VTracer、并行评测 worker、旧自动仿真入口
  requirements.txt   贝叶斯优化相关依赖说明
```

其他目录保持原项目职责：

- `core/`：图像、几何、材料、文档解析等底层模块。
- `Rebuild/`：图像修复、端口检测、参数化前处理相关模块。
- `Simulink/`、`Process/`、`Sweep/`：原有 CST/仿真流程模块。
- `models/`：模型权重。
- `pipeline_runs/`、`optimization_runs/`、`results/`：运行输出目录。
- `local_packages*`：本地依赖包。

## 主要入口

完整贝叶斯优化主流程：

```bash
python -m bayesian_optimization.pipelines.optimization_pipeline
```

FSS 参数化到 CST 的主 pipeline：

```bash
python -m bayesian_optimization.pipelines.fss_parameterized_cst_pipeline
```

Prompt + 模型抽取模块 `PDF_analy_agent` 已接到根目录入口，运行前先设置 OpenRouter API Key：

```powershell
$env:OPENROUTER_API_KEY="你的 OpenRouter API Key"
```

只运行 FSS 论文抽取，不写数据库：

```powershell
py -3.12 -m design_agent.scripts.run_fss_agent --root ".\PDF_analy_agent\FSS PDF"
```

只运行 Antenna 论文抽取，不写数据库：

```powershell
py -3.12 -m design_agent.scripts.run_antenna_agent --root ".\PDF_analy_agent\Antenna PDF"
```

如果当前机器没有 `py -3.12`，也可以使用已安装的 Python 3.13：

```powershell
D:\Python\python.exe -m design_agent.scripts.run_fss_agent --root ".\PDF_analy_agent\FSS PDF"
D:\Python\python.exe -m design_agent.scripts.run_antenna_agent --root ".\PDF_analy_agent\Antenna PDF"
```

模型可通过环境变量覆盖，默认值与 `PROMPT_MODEL_EXTRACTION_GUIDE.md` 保持一致：

```powershell
$env:OPENROUTER_IMAGE_MODEL="qwen/qwen3-vl-8b-instruct"
$env:OPENROUTER_TEXT_MODEL="deepseek/deepseek-v3.2"
```

只运行图像修复和参数化，不进入 CST：

```bash
python -m bayesian_optimization.pipelines.run_parameterization_only
```

旧版实验性 BO 入口：

```bash
python -m bayesian_optimization.pipelines.experimental_bo_optimizer
```

## 常用测试

不依赖 OpenCV/CST 的轻量测试：

```bash
pytest -q test_port_summary_constraints.py
```

迁移后的语法检查：

```bash
python -m compileall -q bayesian_optimization test_*.py
```

部分测试会依赖 `cv2`、CST COM、模型权重或本地 CST 环境。如果当前 Python 环境缺少 OpenCV，包含 `test_patch_topology_mode.py` 的 pytest 会在收集阶段报 `ModuleNotFoundError: No module named 'cv2'`。

## 导入方式

整理后不要再从根目录平铺模块导入，例如不要写：

```python
from geometry_graph_parameterizer import GraphBasedLocalSplineParameterizer
```

应改为包路径：

```python
from bayesian_optimization.geometry.geometry_graph_parameterizer import GraphBasedLocalSplineParameterizer
```

常用模块路径：

- `bayesian_optimization.pipelines.optimization_pipeline`
- `bayesian_optimization.pipelines.fss_parameterized_cst_pipeline`
- `bayesian_optimization.geometry.geometry_driven_parameterizer`
- `bayesian_optimization.geometry.geometry_graph_parameterizer`
- `bayesian_optimization.geometry.primitive_mutator`
- `bayesian_optimization.optimization.optimization_objectives`
- `bayesian_optimization.optimization.s11_parser`
- `bayesian_optimization.simulation.parameterized_json_to_cst`
- `bayesian_optimization.tools.vtracer_python`

## 迁移检查结果

已完成的整理工作：

- 贝叶斯优化相关 `.py` 文件已移动到 `bayesian_optimization/` 下的专门子目录。
- 原 `BAYESIAN_OPTIMIZATION_README.md`、`PROJECT_PIPELINE_README.md`、`docs/bo_change_log.md` 已归入 `bayesian_optimization/docs/`。
- 原 `OverViewLog.md`、`logs/algorithm_updates.md` 已归入 `bayesian_optimization/logs/`。
- 测试脚本和 `Rebuild/NewParams.py` 中引用旧模块名的导入已改为包导入。
- `PROJECT_ROOT` 已修正为仓库根目录，避免移动后找不到 `Rebuild/`、`models/`、`pipeline_runs/` 等资源。
- `experimental_bo_optimizer` 中字符串形式的 pipeline callable 已改为新的包路径。

验证命令：

```bash
python -m compileall -q bayesian_optimization test_graph_local_parameterizer.py test_patch_topology_mode.py test_port_summary_constraints.py test_patch_port_detection.py test_vtracer_validation_dataset.py test_clean_parametric_segmentation_validation.py
pytest -q test_port_summary_constraints.py
```

当前已知限制：

- OpenCV 未安装时，依赖 `cv2` 的测试无法运行。
- CST 建模/仿真入口需要本机 CST、COM 接口和项目路径配置完整后才能完整验证。

"sk-or-v1-<your-openrouter-api-key>"

## Design Agent Pipeline 主线说明

本节以 `design_agent/scripts/run_design_agent_pipeline.py` 为主线，说明 design agent 从目标读取、LLM 设计、几何生成、CST 仿真到 BO 优化反馈的完整调用关系。

### 一键入口

主入口：

```bash
python -m design_agent.scripts.run_design_agent_pipeline
```

入口脚本读取默认配置：

```text
design_agent/agent_config.json
```

也可以指定配置或临时覆盖部分运行参数：

```bash
python -m design_agent.scripts.run_design_agent_pipeline --agent-config design_agent/agent_config.json
python -m design_agent.scripts.run_design_agent_pipeline --iterations 3
python -m design_agent.scripts.run_design_agent_pipeline --build-only
python -m design_agent.scripts.run_design_agent_pipeline --geometry-only
python -m design_agent.scripts.run_design_agent_pipeline --skip-bo-prepare
python -m design_agent.scripts.run_design_agent_pipeline --execute-bo
```

常用开关含义：

- `--geometry-only`：只运行 LLM/几何生成，不进入 CST。
- `--build-only`：构建 CST 工程，但不运行 solver。
- `--skip-bo-prepare`：不准备 BO handoff 文件。
- `--execute-bo`：准备 BO 后直接执行 Bayesian Optimization。

### 顶层调用链

`run_design_agent_pipeline.py` 的工作很薄，主要负责加载配置和创建 LLM client：

```text
design_agent/scripts/run_design_agent_pipeline.py
  -> design_agent.pipeline.load_pipeline_config()
  -> design_agent.llm.client.OpenAICompatibleLLMClient.from_config_file()
  -> design_agent.pipeline.DesignAgentPipelineRunner.run()
```

`DesignAgentPipelineRunner.run()` 是顶层编排器，主要阶段是：

```text
01 bootstrap target
  target.json / agent_config.json
  -> design_agent_runs/<run_name>/agents_inputs/
  -> 初始 target.md、geometry_summary.json、simulation_summary.json

02 closed loop
  -> ClosedLoopDesignRunner.run()
  -> 每轮 LLM 诊断/规划/生成
  -> Geometry Engine JSON
  -> CST build/simulation
  -> simulation_summary.json
  -> LLM 反思，进入下一轮

03 BO handoff / BO execution
  -> convert_geometry_engine_to_bo()
  -> BayesianOptimizationRunner
  -> bayesian_optimization.pipelines.optimization_pipeline
```

关键文件：

- `design_agent/scripts/run_design_agent_pipeline.py`：命令行入口。
- `design_agent/pipeline.py`：顶层 pipeline runner、配置加载、run 目录组织、BO handoff。
- `design_agent/scripts/run_closed_loop_design.py`：闭环迭代主逻辑。
- `design_agent/skills/lightweight_design.py`：LLM prompt 链，包含 diagnose、plan、generate、repair、reflect_history、reflect_bo。
- `design_agent/skills/prompts/`：LLM prompt 模板目录。
- `design_agent/tools/bo_adapter.py`：把 Geometry Engine JSON 转成 BO 可用的 parameterization JSON 和 port summary。
- `design_agent/tools/bayesian_optimization_runner.py`：准备并可选执行 BO pipeline。

### 配置文件

顶层配置：

```text
design_agent/agent_config.json
```

主要分区：

- `run_folder`：控制输出目录，例如 `design_agent_runs/<run_prefix>_<index>`。
- `input_paths`：输入目标、初始设计、根配置路径。
- `closed_loop`：闭环次数、是否跳过 CST、CST 频率、目标 S11/gain/bandwidth。
- `bo`：是否准备 BO、是否执行 BO、BO 频率、最大评估次数、BO 是否 build-only。

LLM 和 CST Python 库路径配置：

```text
config.json
```

其中 `cstModuleBase.absolute_path` 指向 CST 官方 Python 库目录，例如：

```text
C:\Program Files (x86)\CST Studio Suite 2024\AMD64\python_cst_libraries
```

### 运行产物

默认输出在：

```text
design_agent_runs/<run_prefix>_<index>/
```

常见子目录和文件：

- `agents_inputs/`：每轮给 LLM 使用的压缩输入，包括目标、几何摘要、仿真摘要、BO 摘要。
- `01_closed_loop/iteration_xxx/`：每轮闭环输出。
- `geometry_engine_*.json`：Geometry Engine 几何结果。
- `simulation_summary.json`：CST 仿真后的 S11/gain 等摘要。
- `cst_result.json`：CST 工程路径、结果目录、是否运行 solver。
- `bo_effect_summary.json`：BO 执行后的效果总结。
- `manifests/`：顶层运行 manifest。

### Closed Loop 每轮流程

闭环 runner 在 `design_agent/scripts/run_closed_loop_design.py`：

```text
ClosedLoopDesignRunner.run()
  -> LightweightDesignSkill
     -> diagnose
     -> plan
     -> generate
  -> Geometry Builder / Geometry Engine JSON
  -> 几何校验和修复
  -> _run_cst()
  -> write_simulation_summary()
  -> reflect_iteration_effect()
```

如果 `bo.prepare_bo=true`，每轮还会执行：

```text
_run_bo_for_iteration()
  -> convert_geometry_engine_to_bo()
  -> BayesianOptimizationRunner.prepare()
  -> BayesianOptimizationRunner.run()  # 仅 execute=true 时
  -> reflect_bo_effect()
```

### Geometry Engine 到 CST

Design Agent 生成的是 Geometry Engine JSON，不是 CST 直接输入。进入 CST 前会先转换成 `ParameterizedJsonCSTBuilder` 能读的参数化 JSON：

```text
design_agent/scripts/run_geometry_engine_cst.py
  -> GeometryEngineCSTRunner.run()
  -> _build_parameterization_json()
  -> _build_port_summary()
  -> _build_cst_config()
  -> ParameterizedJsonCSTBuilder(parameterization_path, config).build()
```

这一步会生成：

- `01_adapter/parameterization_from_geometry_engine.json`
- `01_adapter/patch_port_summary.json`
- `02_cst/*.cst`
- `03_results/*_s11.txt`

### BO 接入 Design Agent

Design Agent 的 BO 接入不是直接改 Geometry Engine JSON，而是先生成 BO handoff：

```text
design_agent/tools/bo_adapter.py
  -> convert_geometry_engine_to_bo()
  -> build_bo_parameterization()
  -> build_bo_port_summary()
```

然后由：

```text
design_agent/tools/bayesian_optimization_runner.py
```

调用：

```text
bayesian_optimization/pipelines/optimization_pipeline.py
```

BO pipeline 内每个 evaluation 也会调用同一个 CST builder，因此 Design Agent 直跑 CST 和 Design Agent 内嵌 BO 最终共享同一套 CST/Python 接口。

## CST/Python 接口核心和调用点

### 最高优先级：共同 CST Builder

两个 pipeline 最终都会调用：

```text
bayesian_optimization/simulation/parameterized_json_to_cst.py
```

这个文件是 CST/Python 交互的核心适配层。

核心对象和函数：

- `CSTParametricConfig`：CST 工程、单位、频率、尺寸、材料、基板、ground、端口、solver 开关配置。
- `ParameterizedJsonCSTBuilder.build()`：完整建模和仿真主流程。
- `_open_or_create_project()`：创建/打开 CST 工程，并调用 CST 初始化。
- `_draw_substrate_and_ground()`：创建基板和地板。
- `_component_curve_command()` / `cst_extrudecurve()`：画金属轮廓并拉伸。
- `_add_waveguide_port()`：根据 `patch_port_summary.json` 添加 waveguide port。
- `_run_solver()`：调用 `self.modeler.run_solver()` 启动 CST solver。
- `_export_s11()`：用 `cst.results.ProjectFile` 读取 `1D Results\S-Parameters\S1,1` 并写出 S11 文本。
- `load_instance_config()`：把 `pipeline_test_instance*.json` 这类 instance JSON 转成 `CSTParametricConfig`。

### 最底层 CST API 封装

底层 CST 宏命令和官方接口封装在：

```text
Simulink/handle.py
```

这里直接使用 CST 官方 Python API：

```python
from cst.interface import DesignEnvironment
```

关键函数：

- `cst_create_project()`：`DesignEnvironment.new()` + `new_mws()` 创建 CST 工程。
- `cst_open_project()`：打开已有 `.cst` 工程。
- `cst_auto_init()`：自动写入频率、平面波、边界、背景、solver 类型等基础设置。
- `cst_set_frequency()`：设置 solver 频率范围。
- `cst_set_boundaries()`：设置边界条件。
- `cst_change_solver()`：切换到 HF Frequency Domain。
- `cst_set_solver2f()`：频域 solver 细节设置。
- `cst_create_brick()`：生成 Brick 宏命令。
- `cst_extrudecurve()`：曲线拉伸成金属 solid。
- `cst_waveguide_port()`：生成 waveguide port 宏命令。
- `cst_close_project()`：保存并关闭 CST 工程和 DesignEnvironment。

### CST Python 库路径

CST 官方 Python 库路径由：

```text
bayesian_optimization/simulation/cst_library_path.py
```

从根目录配置读取：

```text
config.json -> cstModuleBase.absolute_path
```

`parameterized_json_to_cst.py` import CST 前会调用：

```python
ensure_cst_library_path()
```

这样 Python 才能 import：

```python
import cst
import cst.results
from cst.interface import DesignEnvironment
```

### 两条主线的 CST 调用点

Design Agent 主线：

```text
design_agent/scripts/run_design_agent_pipeline.py
  -> design_agent/pipeline.py
  -> design_agent/scripts/run_closed_loop_design.py::_run_cst()
  -> design_agent/scripts/run_geometry_engine_cst.py::GeometryEngineCSTRunner.run()
  -> bayesian_optimization/simulation/parameterized_json_to_cst.py::ParameterizedJsonCSTBuilder.build()
  -> Simulink/handle.py
```

Bayesian Optimization 主线：

```text
bayesian_optimization/pipelines/optimization_pipeline.py::OptimizationPipeline._build_and_simulate()
  -> load_instance_config()
  -> ParameterizedJsonCSTBuilder(design_json, cst_config).build()
  -> Simulink/handle.py
```

### CST 设置来源对照

Design Agent 直跑 CST：

- `design_agent/agent_config.json`：`closed_loop.f0_ghz`、`closed_loop.f1_ghz`、`build_only`、`geometry_only`。
- `design_agent_runs/initial_design_test/stackup.json`：基板宽长、材料、厚度、ground 设置。
- `design_agent_runs/initial_design_test/design_trace.json`：当命令行未指定频率时，可作为目标频率/带宽来源。
- 运行时生成的 `patch_port_summary.json`：端口位置和方向。

BO 跑 CST：

- `beyesian_opconfig.json`：BO 频率覆盖、目标频率、目标 S11、是否 build-only。
- `pipeline_test_instance.json` / `pipeline_test_instance2.json`：CST 工程路径、单位、尺寸、默认 f0/f1、材料、基板、ground。
- `pipeline_runs/.../patch_port_summary.json` 或每轮修正后的 `port_summary_connected.json`：端口输入。

最关键的判断：

```text
如果要改“仿真参数从哪里来”，看 agent_config.json、beyesian_opconfig.json、pipeline_test_instance*.json 和 load_instance_config()。
如果要改“CST 里具体怎么建模/加端口/跑仿真”，看 parameterized_json_to_cst.py。
如果要改“CST 宏命令/API 调用细节”，看 Simulink/handle.py。
CST-Python的文件可参考D:\cst2py_box\Auto_py2cst_v0.71\bayesian_optimization\simulation
```
