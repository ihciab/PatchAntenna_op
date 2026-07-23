# CadQuery 重构说明

## 中文

### 为什么引入 CadQuery

早期 Geometry Engine 使用项目内自定义的矩形、slot、mirror 等几何操作。这个方式适合快速验证 DSL 流程，但随着天线结构复杂度增加，自定义几何逻辑会遇到几个问题：

- 布尔运算容易出错，例如 slot cut、union、difference、孔洞提取。
- 曲线结构难以稳定表示，例如 circle、arc、ring、SRR、U-slot。
- 几何合法性需要自己维护大量边界条件。
- 下游 CST Builder 需要稳定、连续、有方向的边界数据。

CadQuery 是成熟的 Python CAD 建模库，底层基于 OpenCascade，适合承担几何内核职责。本次重构的目标是让 Geometry Engine 不再实现自己的几何内核，而是把几何构造、变换和布尔运算交给 CadQuery。

### 相比旧 DSL 的优势

- 使用 CadQuery 进行 rectangle、circle、polygon 等标准几何构造。
- 使用 CadQuery boolean union / difference 处理导体合并和 slot cutout。
- 使用 CadQuery mirror、translate、rotate、scale 执行几何变换。
- Geometry Engine 可以专注 DSL 调度、语义参数、验证和导出。
- 导出的 JSON 不依赖 CadQuery，便于未来接入其他几何内核。

### 总体工作流

```text
Geometry DSL
        |
        v
Command Registry
        |
        v
CadQuery Geometry Construction
        |
        v
Boundary Extraction
        |
        v
Geometry Validation
        |
        v
Geometry JSON
        |
        v
Downstream CST Builder
```

### 几何生命周期

1. Importer 从 `patch.json` 或 `parameterization_from_agent.json` 创建 `Patch` 语义对象。
2. `Patch.__post_init__()` 使用 CadQuery 构造初始导体模型。
3. DSL 命令通过 `GeometryEngine.execute()` 执行。
4. 修改类命令调用 CadQuery 后端，例如 cut、union、mirror、rotate。
5. Validator 从 CadQuery 模型提取最终边界并验证。
6. Exporter 将边界转换为标准 Geometry JSON。

### 当前 CadQuery DSL 操作

构造类：

```text
Rectangle(width=?, height=?, x=?, y=?)
Circle(radius=?, x=?, y=?)
Polygon(points=[(x1, y1), (x2, y2), ...])
```

旧 DSL 兼容类：

```text
ResizePatch(length=?, width=?)
MoveFeed(dx=?, dy=?)
AddSlot(shape='rectangle'|'circle', x=?, y=?, width=?, height=?)
DeleteSlot(id=?)
MirrorX()
MirrorY()
Validate()
ExportJSON(path)
```

CadQuery 变换类：

```text
Translate(dx=?, dy=?)
Rotate(angle=?, center_x=?, center_y=?)
Scale(factor=?, center_x=?, center_y=?)
```

CadQuery 布尔类：

```text
BooleanUnion(shape='rectangle'|'circle'|'polygon', ...)
BooleanDifference(shape='rectangle'|'circle'|'polygon', ...)
```

示例：

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "Rectangle(width=25, height=20, x=25, y=15)" `
  --command "BooleanDifference(shape='circle', x=25, y=15, radius=2)" `
  --command "BooleanUnion(shape='rectangle', x=25, y=27, width=6, height=4)" `
  --command "Validate()" `
  --export-geometry-json design_agent_runs\initial_design_test\geometry.json
```

### 重要约束

- Geometry construction 必须先通过 CadQuery 完成。
- Exporter 不写 CadQuery 对象。
- Geometry JSON 不包含 CST 命令、仿真设置、网格设置。
- `paper` 环境中 CadQuery 可用，建议使用：

```powershell
D:\Anaconda\envs\paper\python.exe
```

## English

### Why CadQuery Is Introduced

The earlier Geometry Engine used project-local geometry logic for rectangles, slots, mirrors, and related edits. That was useful for quickly validating the DSL flow, but it becomes fragile as antenna geometry grows more complex.

Typical problems include:

- Boolean operations such as slot cut, union, difference, and hole extraction are hard to maintain manually.
- Curved structures such as circles, arcs, rings, SRRs, and U-slots need robust CAD support.
- Geometry legality requires many edge-case checks.
- The downstream CST Builder needs stable, continuous, oriented boundaries.

CadQuery is a mature Python CAD library based on OpenCascade. In this refactor, Geometry Engine delegates geometry construction, transforms, and boolean operations to CadQuery instead of implementing a custom geometry kernel.

### Advantages Over the Previous DSL

- CadQuery creates standard rectangle, circle, and polygon geometry.
- CadQuery boolean union and difference handle conductor merges and slot cutouts.
- CadQuery mirror, translate, rotate, and scale handle transforms.
- Geometry Engine can focus on DSL dispatch, semantic parameters, validation, and export.
- The exported JSON is independent from CadQuery and can later support other geometry kernels.

### Overall Workflow

```text
Geometry DSL
        |
        v
Command Registry
        |
        v
CadQuery Geometry Construction
        |
        v
Boundary Extraction
        |
        v
Geometry Validation
        |
        v
Geometry JSON
        |
        v
Downstream CST Builder
```

### Geometry Lifecycle

1. The importer creates a semantic `Patch` object from `patch.json` or `parameterization_from_agent.json`.
2. `Patch.__post_init__()` builds the initial CadQuery conductor model.
3. DSL commands execute through `GeometryEngine.execute()`.
4. Mutating commands call the CadQuery backend for cut, union, mirror, rotate, and related operations.
5. The validator extracts the final boundary from the CadQuery model and validates it.
6. The exporter converts the validated boundary into standardized Geometry JSON.

### Current CadQuery DSL Operations

Construction:

```text
Rectangle(width=?, height=?, x=?, y=?)
Circle(radius=?, x=?, y=?)
Polygon(points=[(x1, y1), (x2, y2), ...])
```

Backward-compatible DSL:

```text
ResizePatch(length=?, width=?)
MoveFeed(dx=?, dy=?)
AddSlot(shape='rectangle'|'circle', x=?, y=?, width=?, height=?)
DeleteSlot(id=?)
MirrorX()
MirrorY()
Validate()
ExportJSON(path)
```

Transforms:

```text
Translate(dx=?, dy=?)
Rotate(angle=?, center_x=?, center_y=?)
Scale(factor=?, center_x=?, center_y=?)
```

Booleans:

```text
BooleanUnion(shape='rectangle'|'circle'|'polygon', ...)
BooleanDifference(shape='rectangle'|'circle'|'polygon', ...)
```

Example:

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "Rectangle(width=25, height=20, x=25, y=15)" `
  --command "BooleanDifference(shape='circle', x=25, y=15, radius=2)" `
  --command "BooleanUnion(shape='rectangle', x=25, y=27, width=6, height=4)" `
  --command "Validate()" `
  --export-geometry-json design_agent_runs\initial_design_test\geometry.json
```

### Important Constraints

- Geometry must be constructed with CadQuery first.
- The exporter must not serialize CadQuery objects.
- Geometry JSON must not contain CST commands, simulation settings, or mesh settings.
- CadQuery is available in the `paper` environment. Use:

```powershell
D:\Anaconda\envs\paper\python.exe
```
