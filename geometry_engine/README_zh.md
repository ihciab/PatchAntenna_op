# Geometry Engine 使用说明

Geometry Engine 是 LLM Design Agent 与 CST 建模程序之间的中间层。

它的职责是：

- 接收 Geometry DSL 命令
- 维护内部的参数化 `Patch` / `Feed` / `Slot` 对象
- 执行几何修改
- 验证几何合法性
- 导出 `patch.json`

它不负责调用 OpenAI API，不负责调用 CST API，也不负责优化算法。

## 系统位置

```text
LLM
  -> Geometry DSL
  -> Geometry Engine
  -> patch.json
  -> CST Builder
  -> CST Simulation
```

Geometry Engine 内部始终操作对象模型，不直接修改 JSON。只有 exporter 负责把对象转换成 `patch.json`。

## 快速测试

以现有 design agent 运行目录作为输入：

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test --run-dir design_agent_runs\initial_design_test --use-default-commands
```

这个命令会读取：

```text
design_agent_runs\initial_design_test\patch.json
```

然后自动转换成 Geometry Engine 内部的 `Patch` 对象，并依次执行默认 DSL 测试命令。

## 指定 DSL 命令测试

可以重复传入 `--command`，命令会按顺序执行。每个修改类命令执行后都会自动调用 validator，如果验证失败，Geometry Engine 会回滚到命令执行前的状态。

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "Validate()" `
  --command "AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)" `
  --command "MoveFeed(dx=1.0, dy=0.0)" `
  --command "MirrorY()" `
  --command "Validate()" `
  --export-patch-json design_agent_runs\initial_design_test\patch_with_slots.json
```

## Python 代码中使用

```python
from geometry_engine.context import GeometryContext
from geometry_engine.engine import GeometryEngine
from geometry_engine.importer import ParameterizationImporter

patch = ParameterizationImporter().from_run_dir("design_agent_runs/initial_design_test")
engine = GeometryEngine(context=GeometryContext(patch=patch))

engine.execute("Validate()")
engine.execute("AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)")
engine.execute("MoveFeed(dx=1.0, dy=0.0)")
engine.execute("Validate()")
engine.export_json("design_agent_runs/initial_design_test/patch_with_slots.json")
```

## 当前支持的 DSL 命令

### ResizePatch(length=?, width=?)

修改 patch 尺寸，单位为毫米。Patch 中心保持不变，feed 会自动保持挂在原来的边缘上。

```text
ResizePatch(length=26, width=30)
ResizePatch(width=32)
ResizePatch(length=28)
```

### MoveFeed(dx=?, dy=?)

按增量移动 feed 点，单位为毫米。移动后 feed 必须仍然位于 patch 边缘。

```text
MoveFeed(dx=1.0, dy=0.0)
MoveFeed(dx=-2.0, dy=0.0)
```

当前初始样例中 patch 范围约为：

```text
x: 11 ~ 39
y: 12 ~ 36
feed: bottom edge, y = 12
```

所以 bottom feed 通常只能沿 x 方向移动，不能随意改变 y。

### AddSlot(shape="rectangle", x=?, y=?, width=?, height=?)

添加矩形 slot。slot 的 `(x, y)` 是中心点，`width` 和 `height` 是尺寸，单位为毫米。

合法示例：

```text
AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)
AddSlot(shape='rectangle', x=19, y=24, width=2, height=6)
```

非法示例：

```text
AddSlot(shape='rectangle', x=25, y=12.2, width=4, height=1)
```

这个 slot 的下边界会低于 patch 下边界 `y=12`，因此 validator 会报错。

第一版只支持 `shape="rectangle"`。后续可以扩展圆形、环形、SRR、U-slot 等。

### DeleteSlot(id=?)

按 slot id 删除 slot。

```text
DeleteSlot(id='slot_001')
```

如果 `AddSlot` 没有显式传入 id，Geometry Engine 会自动生成：

```text
slot_001
slot_002
slot_003
```

### MirrorX()

以 patch 中心的水平轴进行镜像。会镜像 feed 和所有 slots。

```text
MirrorX()
```

对于 bottom feed，`MirrorX()` 后 feed 会变成 top feed。

### MirrorY()

以 patch 中心的垂直轴进行镜像。会镜像 feed 和所有 slots。

```text
MirrorY()
```

对于 left feed，`MirrorY()` 后 feed 会变成 right feed；bottom feed 的方向不变，只改变 x 位置。

### Validate()

验证当前几何状态。

```text
Validate()
```

返回 `ValidationResult(valid=True, errors=[])` 或错误列表。

### ExportJSON(path)

导出当前 Geometry Engine 对象为 `patch.json`。

```text
ExportJSON('design_agent_runs/initial_design_test/patch_with_slots.json')
```

也可以在测试脚本中使用：

```powershell
--export-patch-json design_agent_runs\initial_design_test\patch_with_slots.json
```

## Slot 测试建议

推荐从这几类 case 开始测试。

合法中心 slot：

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "Validate()" `
  --command "AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)" `
  --command "Validate()"
```

合法多 slot：

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)" `
  --command "AddSlot(shape='rectangle', x=19, y=24, width=2, height=6)" `
  --command "AddSlot(shape='rectangle', x=31, y=24, width=2, height=6)" `
  --command "Validate()"
```

非法贴边 slot：

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=25, y=12.2, width=4, height=1)"
```

非法超出右边界 slot：

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=38, y=24, width=4, height=2)"
```

添加后删除：

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)" `
  --command "DeleteSlot(id='slot_001')" `
  --command "Validate()"
```

## 几何规则

当前 validator 检查以下规则：

- 所有几何单位为毫米
- Patch 长度和宽度必须为正数
- Patch 必须是 PEC
- Patch 轮廓必须闭合
- Slot id 不能重复
- Slot 当前只支持 rectangle
- Slot 必须严格位于 patch 内部，不能贴边
- Feed 宽度必须为正数
- Feed 必须位于 patch 边缘
- Feed 的方向必须与所在边一致
- Feed 的端点跨度必须落在 patch 边界范围内

## 导出格式说明

Exporter 会生成 `design_agent_patch_v1` 风格的 JSON：

```text
patch.json
  schema_version
  unit
  topology
  parameters
  conductor
    components
      patch
    cutouts
      slot_001
      slot_002
  port
```

注意：第一版 `AddSlot` 会导出到 `conductor.cutouts`。这表示 slot cutout 元数据。当前 Geometry Engine 会验证 slot 合法性，但 CST Builder 是否真正执行布尔减槽，需要后续在 Builder 侧实现。

## 文件职责

### geometry_engine/__init__.py

包入口。导出常用类：

- `GeometryEngine`
- `ParameterizationImporter`

### geometry_engine/engine.py

Geometry Engine 主入口。

职责：

- 接收 DSL 字符串或命令对象
- 调用 parser 解析 DSL
- 通过 registry 创建命令对象
- 执行命令
- 修改后自动调用 validator
- 验证失败时回滚 geometry state
- 提供 `Validate()` 和 `ExportJSON(path)` 内置命令

核心类：

- `GeometryEngine`
- `ValidateCommand`
- `ExportJSONCommand`

### geometry_engine/context.py

运行时上下文。

职责：

- 保存当前 `Patch` 对象
- 保存 `GeometryValidator`
- 保存 `PatchJSONExporter`
- 生成新的 slot id
- 提供 `validate()` 和 `export_json()`

核心类：

- `GeometryContext`

### geometry_engine/registry.py

命令注册表。

职责：

- 将 DSL 命令名映射到命令类
- 注册内置命令
- 支持未来扩展命令

新增 DSL 命令时，通常只需要：

1. 新增一个 command class
2. 设置 `dsl_name`
3. 在 registry 中注册

Engine 核心不需要修改。

核心类：

- `CommandRegistry`

### geometry_engine/validator.py

几何验证器。

职责：

- 验证 patch 尺寸
- 验证 patch 材料
- 验证 patch 闭合性
- 验证 slot 是否在 patch 内部
- 验证 feed 是否在 patch 边缘

核心类：

- `GeometryValidator`
- `ValidationResult`
- `GeometryValidationError`

### geometry_engine/exporter.py

JSON 导出器。

职责：

- 将内部 `Patch` 对象转换成 `patch.json`
- 导出 patch polygon primitives
- 导出 port 信息
- 导出 slots 到 `conductor.cutouts`

核心类：

- `PatchJSONExporter`

### geometry_engine/importer.py

JSON 导入桥接层。

职责：

- 从 design-agent `patch.json` 初始化 `Patch`
- 从 CST adapter `parameterization_from_agent.json` 初始化 `Patch`
- 从 patch/feed component 的 bbox 推断 patch 尺寸和 feed 接触边

注意：Importer 只是测试和兼容旧产物的桥接层。Geometry Engine 核心仍然操作对象，而不是直接操作 JSON。

核心类：

- `ParameterizationImporter`
- `ParameterizationImportError`

### geometry_engine/geometry/patch.py

Patch 对象模型。

职责：

- 维护 patch 长度、宽度、中心点、材料、层信息
- 持有 `Feed`
- 持有多个 `Slot`
- 提供 resize、add/delete slot、mirror 等几何操作
- 生成 patch 顶点
- 判断 feed 是否在边缘

核心类：

- `Patch`

### geometry_engine/geometry/feed.py

Feed 对象模型。

职责：

- 维护 feed 点位置
- 维护 feed 宽度和方向
- 支持移动和镜像
- 生成 port edge 两端点

核心类：

- `Feed`

### geometry_engine/geometry/slot.py

Slot 对象模型。

职责：

- 维护 slot id、shape、中心点、宽度、高度
- 计算 slot 边界
- 支持镜像
- 生成 slot polygon vertices

核心类：

- `Slot`

### geometry_engine/dsl/command.py

DSL 命令抽象。

职责：

- 定义 parsed command 数据结构
- 定义所有 command class 的抽象基类

核心类：

- `ParsedCommand`
- `GeometryCommand`

### geometry_engine/dsl/parser.py

DSL parser。

职责：

- 解析函数调用形式的 DSL
- 支持单条命令
- 支持多条命令脚本
- 只接受安全 literal 参数

支持形式：

```text
AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)
Validate()
```

核心类：

- `DSLParser`
- `DSLParseError`

### geometry_engine/commands/resize_patch.py

`ResizePatch` 命令实现。

### geometry_engine/commands/move_feed.py

`MoveFeed` 命令实现。

### geometry_engine/commands/add_slot.py

`AddSlot` 命令实现。

第一版只完整支持 rectangular slot。

### geometry_engine/commands/delete_slot.py

`DeleteSlot` 命令实现。

### geometry_engine/commands/mirror.py

`MirrorX` 和 `MirrorY` 命令实现。

### design_agent.scripts.run_geometry_engine_dsl_test

命令行测试脚本。

职责：

- 加载 `--run-dir` / `--patch-json` / `--parameterization-json`
- 初始化 Geometry Engine
- 按顺序执行 DSL 命令
- 每条命令后输出 validator 结果
- 可选导出 `patch.json`

## 扩展新 DSL 命令

以未来 `AddStub()` 为例：

1. 新建文件：

```text
geometry_engine/commands/add_stub.py
```

2. 定义命令类：

```python
from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand


class AddStubCommand(GeometryCommand):
    dsl_name = "AddStub"

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    def execute(self, context: GeometryContext) -> str:
        # TODO: modify context.patch
        return "stub_001"
```

3. 在 `registry.py` 的 `with_builtin_commands()` 中注册。

Engine 的执行流程不需要修改。

## 当前限制

- Patch 当前按矩形 patch 建模
- Slot 当前只支持 rectangle
- Slot overlap 还没有检查
- Slot 最小金属间隙还没有检查
- Mirror 对非矩形 patch 的复杂轮廓检查还是 TODO
- `conductor.cutouts` 目前是 Geometry Engine 导出的 cutout 元数据，CST Builder 侧还需要实现真正布尔减槽

