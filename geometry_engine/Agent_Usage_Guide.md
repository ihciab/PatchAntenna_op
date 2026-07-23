# Geometry Engine Agent Usage Guide

本文档面向 LLM Design Agent，说明当前 Geometry Engine 已实现的能力、DSL 调用格式、输入输出文件、几何约束和推荐调用流程。

This document is intended for an LLM Design Agent. It explains the current Geometry Engine capabilities, DSL call formats, input/output files, geometry constraints, and recommended usage flow.

---

## 1. 模块定位

Geometry Engine 是 LLM Design Agent 与 CST Builder 之间的几何中间层。

```text
LLM Design Agent
        |
        v
Geometry DSL
        |
        v
Geometry Engine
        |
        v
CadQuery Geometry
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

Geometry Engine 不调用 OpenAI API，不调用 CST API，不运行仿真，不做优化。

它只负责：

- 接收 DSL 命令
- 使用 CadQuery 构造和修改几何
- 提取最终 2D conductor boundary
- 验证边界合法性
- 导出标准 Geometry JSON
- 可选生成 SVG/PNG/PDF 可视化图

---

## 2. 当前测试入口

当前推荐使用：

```text
design_agent.scripts.run_geometry_engine_dsl_test
```

这个脚本已经支持：

- 从 `design_agent_runs/initial_design_test/patch.json` 加载初始几何
- 从文件内配置读取 DSL 命令
- 执行 DSL
- 自动 Validate
- 导出 Geometry JSON
- 生成 SVG 可视化

直接运行：

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test
```

脚本顶部有可编辑配置：

```python
IN_FILE_TEST_CONFIG = {
    "run_dir": PROJECT_ROOT / "design_agent_runs" / "initial_design_test",
    "patch_json": None,
    "parameterization_json": None,
    "commands": [
        "Validate()",
        "AddSlot(shape='rectangle', x=25, y=12, width=4, height=1.5)",
        "AddSlot(shape='circle', x=31, y=12, width=2, height=2)",
        "Validate()",
    ],
    "export_geometry_json": PROJECT_ROOT / ".codex_tmp" / "slot_visual_test.json",
    "visualize": True,
    "visual_output": PROJECT_ROOT / ".codex_tmp" / "slot_visual_test.svg",
    "show_vertices": True,
    "show_labels": True,
    "max_vertex_markers": 64,
    "max_vertex_labels": 16,
}
```

Agent 如果要批量测试不同设计，只需要修改 `commands` 和输出路径。

---

## 3. 输入文件

Geometry Engine 当前支持从两类文件初始化：

### 3.1 design-agent patch.json

```text
design_agent_runs/<run_name>/patch.json
```

配置方式：

```python
"run_dir": PROJECT_ROOT / "design_agent_runs" / "initial_design_test"
```

或：

```python
"patch_json": PROJECT_ROOT / "design_agent_runs" / "initial_design_test" / "patch.json"
```

### 3.2 CST adapter parameterization_from_agent.json

```text
design_agent_runs/<run_name>_cst/01_adapter/parameterization_from_agent.json
```

配置方式：

```python
"parameterization_json": PROJECT_ROOT / "design_agent_runs" / "initial_design_test_cst" / "01_adapter" / "parameterization_from_agent.json"
```

如果 `run_dir` 不为 `None`，优先使用 `run_dir/patch.json`。

---

## 4. 输出文件

### 4.1 Geometry JSON

推荐输出：

```text
.codex_tmp/slot_visual_test.json
```

配置：

```python
"export_geometry_json": PROJECT_ROOT / ".codex_tmp" / "slot_visual_test.json"
```

输出 schema：

```text
geometry_engine_geometry_v1
```

Geometry JSON 只描述几何，不包含：

- CadQuery 对象
- CST 命令
- simulation settings
- mesh settings

### 4.2 SVG 可视化

推荐输出：

```text
.codex_tmp/slot_visual_test.svg
```

配置：

```python
"visualize": True,
"visual_output": PROJECT_ROOT / ".codex_tmp" / "slot_visual_test.svg"
```

可视化会显示：

- conductor outer boundary
- slot / hole
- feed 位置
- 可选顶点 marker
- 可选顶点编号
- 坐标轴，单位 mm

对于圆形槽，导出 JSON 中会有很多采样顶点。为了避免图上密集标注，可以调小：

```python
"max_vertex_markers": 24,
"max_vertex_labels": 8
```

或关闭：

```python
"show_vertices": False,
"show_labels": False
```

---

## 5. 当前已实现 DSL 操作

下面是 Agent 当前可以生成的 DSL 命令。

### 5.1 Validate

验证当前几何。

```text
Validate()
```

推荐在关键修改后调用。

---

### 5.2 Rectangle

使用 CadQuery 创建矩形 conductor，替换当前 conductor。

```text
Rectangle(width=?, height=?, x=?, y=?)
```

参数：

- `width`: x 方向尺寸，mm
- `height`: y 方向尺寸，mm
- `x`: 中心 x 坐标，mm
- `y`: 中心 y 坐标，mm

示例：

```text
Rectangle(width=25, height=20, x=25, y=15)
```

---

### 5.3 Circle

使用 CadQuery 创建圆形 conductor，替换当前 conductor。

```text
Circle(radius=?, x=?, y=?)
```

参数：

- `radius`: 半径，mm
- `x`: 圆心 x 坐标，mm
- `y`: 圆心 y 坐标，mm

示例：

```text
Circle(radius=8, x=25, y=15)
```

---

### 5.4 Polygon

使用 CadQuery 创建多边形 conductor，替换当前 conductor。

```text
Polygon(points=[(x1,y1),(x2,y2),(x3,y3)])
```

示例：

```text
Polygon(points=[(0,0),(20,0),(18,12),(0,10)])
```

要求：

- 至少 3 个点
- 点按边界顺序给出
- 单位 mm

---

### 5.5 AddSlot

添加 slot cutout。底层通过 CadQuery boolean difference 实现。

#### 矩形槽

```text
AddSlot(shape='rectangle', x=?, y=?, width=?, height=?)
```

参数：

- `x`: 槽中心 x
- `y`: 槽中心 y
- `width`: 槽宽度
- `height`: 槽高度

示例：

```text
AddSlot(shape='rectangle', x=25, y=12, width=4, height=1.5)
```

#### 圆形槽

```text
AddSlot(shape='circle', x=?, y=?, width=?, height=?)
```

当前约定：

- `width` 表示圆形槽直径
- `height` 保留为兼容参数，可写成与 width 相同

示例：

```text
AddSlot(shape='circle', x=31, y=12, width=2, height=2)
```

说明：

- 圆形槽导出时会采样成多个顶点，当前默认约 96 点。
- 这不是错误，而是曲线边界的离散表达。

---

### 5.6 DeleteSlot

删除由 `AddSlot` 添加的 slot。

```text
DeleteSlot(id='slot_001')
```

默认自动生成 id：

```text
slot_001
slot_002
slot_003
```

示例：

```text
DeleteSlot(id='slot_001')
```

---

### 5.7 BooleanDifference

从当前 conductor 中减去一个 CadQuery 几何。

```text
BooleanDifference(shape='rectangle', x=?, y=?, width=?, height=?)
BooleanDifference(shape='circle', x=?, y=?, radius=?)
BooleanDifference(shape='polygon', points=[...])
```

示例：

```text
BooleanDifference(shape='circle', x=25, y=15, radius=2)
```

适合表达：

- 圆孔
- 矩形槽
- 多边形切槽
- 后续复杂 slot

---

### 5.8 BooleanUnion

把一个 CadQuery 几何并入当前 conductor。

```text
BooleanUnion(shape='rectangle', x=?, y=?, width=?, height=?)
BooleanUnion(shape='circle', x=?, y=?, radius=?)
BooleanUnion(shape='polygon', points=[...])
```

示例：

```text
BooleanUnion(shape='rectangle', x=25, y=27, width=6, height=4)
```

适合表达：

- stub
- parasitic attachment
- patch extension

---

### 5.9 Translate

平移当前 conductor。

```text
Translate(dx=?, dy=?)
```

示例：

```text
Translate(dx=5, dy=0)
```

---

### 5.10 Rotate

旋转当前 conductor。

```text
Rotate(angle=?)
Rotate(angle=?, center_x=?, center_y=?)
```

参数：

- `angle`: 角度，单位 degree
- `center_x`, `center_y`: 可选旋转中心

示例：

```text
Rotate(angle=15)
Rotate(angle=30, center_x=25, center_y=15)
```

---

### 5.11 Scale

缩放当前 conductor。

```text
Scale(factor=?)
Scale(factor=?, center_x=?, center_y=?)
```

示例：

```text
Scale(factor=1.2)
Scale(factor=0.8, center_x=25, center_y=15)
```

---

### 5.12 MirrorX

沿 patch 中心水平轴镜像。

```text
MirrorX()
```

---

### 5.13 MirrorY

沿 patch 中心垂直轴镜像。

```text
MirrorY()
```

---

### 5.14 ResizePatch

兼容旧 DSL：调整矩形 patch 尺寸。

```text
ResizePatch(length=?, width=?)
```

示例：

```text
ResizePatch(length=28, width=30)
```

---

### 5.15 MoveFeed

兼容旧 DSL：移动 feed。

```text
MoveFeed(dx=?, dy=?)
```

示例：

```text
MoveFeed(dx=1.0, dy=0.0)
```

注意：当前 feed 主要作为 metadata 导出，不是 conductor boolean 的一部分。

---

## 6. 推荐 Agent 调用模板

### 6.1 添加矩形槽 + 圆形槽

```python
"commands": [
    "Validate()",
    "AddSlot(shape='rectangle', x=25, y=12, width=4, height=1.5)",
    "AddSlot(shape='circle', x=31, y=12, width=2, height=2)",
    "Validate()",
]
```

### 6.2 创建新矩形 patch，切圆孔，加 stub

```python
"commands": [
    "Rectangle(width=25, height=20, x=25, y=15)",
    "BooleanDifference(shape='circle', x=25, y=15, radius=2)",
    "BooleanUnion(shape='rectangle', x=25, y=27, width=6, height=4)",
    "Validate()",
]
```

### 6.3 多边形 patch，切圆孔，平移

```python
"commands": [
    "Polygon(points=[(0,0),(20,0),(18,12),(0,10)])",
    "BooleanDifference(shape='circle', x=8, y=5, radius=1.5)",
    "Translate(dx=5, dy=5)",
    "Validate()",
]
```

---

## 7. Geometry JSON 输出结构

输出示意：

```json
{
  "schema_version": "geometry_engine_geometry_v1",
  "generator": "geometry_engine_cadquery",
  "unit": "mm",
  "coordinate_system": {
    "plane": "XY",
    "x_axis": "right",
    "y_axis": "up",
    "orientation": "right_handed"
  },
  "geometries": [
    {
      "id": "patch_conductor",
      "type": "planar_conductor",
      "unit": "mm",
      "plane": "XY",
      "outer_boundary": {
        "closed": true,
        "orientation": "CCW",
        "vertices": []
      },
      "holes": [],
      "metadata": {
        "source_kernel": "CadQuery",
        "feed": {}
      }
    }
  ]
}
```

重要规则：

- 所有单位为 mm
- 外边界和 holes 都是有序顶点
- 所有 loop 都是 CCW
- 不重复最后一个闭合点
- `closed: true` 表示闭合
- holes 表示 slot/cutout

---

## 8. 失败处理建议

Agent 生成 DSL 后，必须执行 `Validate()`。

如果脚本输出：

```text
FAIL: ...
```

说明该 DSL 几何非法或 CadQuery boolean 失败。

Agent 应该：

1. 读取错误信息
2. 调整 slot 位置、尺寸或布尔对象
3. 重新生成 DSL
4. 再次 Validate

常见失败原因：

- slot 超出 conductor
- slot 贴边导致边界退化
- polygon 点数不足
- boolean tool 与 conductor 没有有效交集
- rotate/scale 后 feed metadata 不再适合旧矩形 edge 规则

---

## 9. 圆形槽可视化说明

圆形槽在 Geometry JSON 中会被采样成多个顶点，例如 96 个点。

这不是几何错误。

如果 SVG 看起来有密集点或编号，可以在配置中调整：

```python
"max_vertex_markers": 24,
"max_vertex_labels": 8,
```

或关闭：

```python
"show_vertices": False,
"show_labels": False,
```

---

## 10. Agent 最小调用原则

推荐 Agent 输出 DSL 时遵循：

- 每组操作最后加 `Validate()`
- slot 尺寸不要贴近 conductor 边界
- 圆形 slot 用 `shape='circle'`
- 复杂 cutout 优先用 `BooleanDifference`
- 添加 stub/扩展结构优先用 `BooleanUnion`
- 最终输出只消费 `geometry_engine_geometry_v1`

