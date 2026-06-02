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

