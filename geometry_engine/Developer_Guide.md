# Geometry Engine 开发指南

## 中文

### 模块划分

当前模块按职责分离：

```text
geometry_engine/
  cadquery_backend.py      CadQuery 构造、布尔、变换、边界提取
  boundary.py              内核无关边界数据结构和边界验证
  exporter.py              Geometry JSON 导出
  validator.py             Patch 语义验证和边界验证
  engine.py                DSL 执行入口
  registry.py              DSL 命令注册
  context.py               运行时上下文
  importer.py              从旧 JSON 产物初始化 Patch
  geometry/
    patch.py               CadQuery-backed Patch 语义对象
    feed.py                Feed 语义对象
    slot.py                Slot 语义对象
  commands/
    construction.py        Rectangle / Circle / Polygon
    boolean.py             BooleanUnion / BooleanDifference
    transform.py           Translate / Rotate / Scale
    add_slot.py            AddSlot
    delete_slot.py         DeleteSlot
    mirror.py              MirrorX / MirrorY
    move_feed.py           MoveFeed
    resize_patch.py        ResizePatch
```

### 如何创建新几何

命令行：

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --command "Rectangle(width=25, height=20, x=25, y=15)" `
  --command "Validate()" `
  --export-geometry-json geometry.json
```

Python：

```python
from geometry_engine import GeometryEngine

engine = GeometryEngine()
engine.execute("Rectangle(width=25, height=20, x=25, y=15)")
engine.execute("BooleanDifference(shape='circle', x=25, y=15, radius=2)")
engine.export_json("geometry.json")
```

### 如何添加新几何操作

新增命令建议放在 `geometry_engine/commands/`。

示例：添加 `AddStub()`。

```python
from geometry_engine.cadquery_backend import CadQueryPlanarModel
from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class AddStubCommand(GeometryCommand):
    """Add a rectangular stub by CadQuery boolean union."""

    dsl_name = "AddStub"

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        """Create an AddStub command."""

        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    def execute(self, context: GeometryContext) -> None:
        """Union a CadQuery rectangle into the current conductor."""

        tool = CadQueryPlanarModel.rectangle(
            width=self.width,
            height=self.height,
            center_x=self.x,
            center_y=self.y,
            z=context.patch.z,
            thickness=context.patch.thickness,
        )
        context.patch.boolean_union(tool)
```

然后在 `registry.py` 的 `with_builtin_commands()` 中注册。

### 如何导出 JSON

推荐使用：

```python
engine.export_json("geometry.json")
```

命令行使用：

```powershell
--export-geometry-json geometry.json
```

`--export-patch-json` 仍保留为兼容别名，但现在导出的是标准 Geometry JSON，不是旧 CST adapter 风格 patch JSON。

### 编码规范

- public function 必须有类型注解。
- public function 必须有 docstring。
- 不在构造阶段直接编辑 JSON。
- 几何构造必须先使用 CadQuery。
- JSON exporter 不依赖 CadQuery。
- validator 必须验证最终 boundary，而不是只检查输入参数。
- 不要在 Geometry Engine 中加入 CST API 调用。
- 不要在 Geometry Engine 中加入 OpenAI API 调用。

### 测试建议

基础 slot：

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=25, y=12, width=4, height=1.5)" `
  --command "Validate()" `
  --export-geometry-json .codex_tmp\slot_geometry.json
```

圆孔：

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='circle', x=25, y=12, width=2, height=2)" `
  --command "Validate()" `
  --export-geometry-json .codex_tmp\circle_slot_geometry.json
```

布尔组合：

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --command "Rectangle(width=25, height=20, x=25, y=15)" `
  --command "BooleanDifference(shape='circle', x=25, y=15, radius=2)" `
  --command "BooleanUnion(shape='rectangle', x=25, y=27, width=6, height=4)" `
  --command "Validate()" `
  --export-geometry-json .codex_tmp\boolean_geometry.json
```

### 未来扩展建议

- 增加 `AddStub()`、`AddParasitic()`、`AddGroundSlot()`。
- 增加 curve primitive JSON，而不是所有曲线都采样成顶点。
- 增加多 conductor、多 layer 输出。
- 增加 slot overlap 和最小金属间隙规则。
- 在 CST Builder 中消费 `geometry_engine_geometry_v1`。

## English

### Module Layout

The current module is separated by responsibility:

```text
geometry_engine/
  cadquery_backend.py      CadQuery construction, booleans, transforms, boundary extraction
  boundary.py              Kernel-independent boundary data structures and validation
  exporter.py              Geometry JSON export
  validator.py             Patch semantic validation and boundary validation
  engine.py                DSL execution entry point
  registry.py              DSL command registry
  context.py               Runtime context
  importer.py              Initialize Patch from old JSON artifacts
  geometry/
    patch.py               CadQuery-backed Patch semantic object
    feed.py                Feed semantic object
    slot.py                Slot semantic object
  commands/
    construction.py        Rectangle / Circle / Polygon
    boolean.py             BooleanUnion / BooleanDifference
    transform.py           Translate / Rotate / Scale
    add_slot.py            AddSlot
    delete_slot.py         DeleteSlot
    mirror.py              MirrorX / MirrorY
    move_feed.py           MoveFeed
    resize_patch.py        ResizePatch
```

### Creating New Geometry

Command line:

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --command "Rectangle(width=25, height=20, x=25, y=15)" `
  --command "Validate()" `
  --export-geometry-json geometry.json
```

Python:

```python
from geometry_engine import GeometryEngine

engine = GeometryEngine()
engine.execute("Rectangle(width=25, height=20, x=25, y=15)")
engine.execute("BooleanDifference(shape='circle', x=25, y=15, radius=2)")
engine.export_json("geometry.json")
```

### Adding New Geometry Operations

New commands should usually live under `geometry_engine/commands/`.

Example: add `AddStub()`.

```python
from geometry_engine.cadquery_backend import CadQueryPlanarModel
from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class AddStubCommand(GeometryCommand):
    """Add a rectangular stub by CadQuery boolean union."""

    dsl_name = "AddStub"

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        """Create an AddStub command."""

        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    def execute(self, context: GeometryContext) -> None:
        """Union a CadQuery rectangle into the current conductor."""

        tool = CadQueryPlanarModel.rectangle(
            width=self.width,
            height=self.height,
            center_x=self.x,
            center_y=self.y,
            z=context.patch.z,
            thickness=context.patch.thickness,
        )
        context.patch.boolean_union(tool)
```

Then register it in `registry.py` inside `with_builtin_commands()`.

### Exporting JSON

Recommended Python usage:

```python
engine.export_json("geometry.json")
```

Command-line usage:

```powershell
--export-geometry-json geometry.json
```

`--export-patch-json` is kept as a compatibility alias, but it now exports standardized Geometry JSON rather than the old CST-adapter-style patch JSON.

### Coding Conventions

- Public functions must include type hints.
- Public functions must include docstrings.
- Do not edit JSON directly during construction.
- Geometry construction must go through CadQuery first.
- The JSON exporter must not depend on CadQuery.
- The validator must validate the final boundary, not only input parameters.
- Do not add CST API calls to Geometry Engine.
- Do not add OpenAI API calls to Geometry Engine.

### Suggested Tests

Basic slot:

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=25, y=12, width=4, height=1.5)" `
  --command "Validate()" `
  --export-geometry-json .codex_tmp\slot_geometry.json
```

Circular slot:

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='circle', x=25, y=12, width=2, height=2)" `
  --command "Validate()" `
  --export-geometry-json .codex_tmp\circle_slot_geometry.json
```

Boolean composition:

```powershell
D:\Anaconda\envs\paper\python.exe -m design_agent.scripts.run_geometry_engine_dsl_test `
  --command "Rectangle(width=25, height=20, x=25, y=15)" `
  --command "BooleanDifference(shape='circle', x=25, y=15, radius=2)" `
  --command "BooleanUnion(shape='rectangle', x=25, y=27, width=6, height=4)" `
  --command "Validate()" `
  --export-geometry-json .codex_tmp\boolean_geometry.json
```

### Future Extension Suggestions

- Add `AddStub()`, `AddParasitic()`, and `AddGroundSlot()`.
- Add curve primitive JSON instead of sampling every curve into vertices.
- Add multi-conductor and multi-layer output.
- Add slot overlap and minimum copper clearance rules.
- Update the CST Builder to consume `geometry_engine_geometry_v1`.
