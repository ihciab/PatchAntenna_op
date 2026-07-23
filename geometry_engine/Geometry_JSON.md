# Geometry JSON 规范

## 中文

### 目标

Geometry JSON 是 Geometry Engine 的标准导出格式。它只描述几何，不描述 CST 命令、仿真设置、材料库设置或网格设置。

它的目标是让下游 CST Builder 可以从一个稳定、内核无关的几何文件中自动建模。

### 顶层结构

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
  "geometries": [],
  "export_rules": {}
}
```

### geometry 对象

当前主要导出 planar conductor：

```json
{
  "id": "patch_conductor",
  "type": "planar_conductor",
  "unit": "mm",
  "plane": "XY",
  "outer_boundary": {},
  "holes": [],
  "metadata": {}
}
```

### boundary loop

边界 loop 不重复最后一个闭合点，通过 `closed: true` 表示闭合。

```json
{
  "id": "outer",
  "role": "outer",
  "closed": true,
  "orientation": "CCW",
  "vertices": [
    {"x": 37.5, "y": 5.0},
    {"x": 37.5, "y": 25.0},
    {"x": 12.5, "y": 25.0},
    {"x": 12.5, "y": 5.0}
  ]
}
```

### holes

Slot cutout 或 boolean difference 形成的孔洞放在 `holes` 中。

```json
{
  "id": "hole_001",
  "role": "hole",
  "closed": true,
  "orientation": "CCW",
  "vertices": [
    {"x": 23.0, "y": 11.25},
    {"x": 27.0, "y": 11.25},
    {"x": 27.0, "y": 12.75},
    {"x": 23.0, "y": 12.75}
  ]
}
```

### 坐标系统

- 单位：毫米
- 平面：XY
- x 轴：向右
- y 轴：向上
- z 不在边界顶点中导出
- Geometry JSON 描述 2D conductor boundary

### 导出规则

Exporter 和 validator 会保证：

- boundary closed
- boundary continuous
- vertices ordered
- orientation is counter-clockwise
- no duplicated vertices
- no self-intersection
- unit is millimeter

### 圆和曲线

Geometry JSON 当前使用顶点序列表达边界。CadQuery 中的圆、弧线和曲线会在 boundary extraction 阶段采样成有序顶点。

默认圆形边界采样点数量由 `CadQueryBoundaryExtractor.curve_samples` 控制，当前为 96。

### 未来扩展

后续可以扩展：

- 多 conductor 几何
- ground slot
- parasitic patch
- layer stack geometry
- curve primitive representation
- arc-aware CST builder
- geometry constraints

## English

### Goal

Geometry JSON is the standard export format of Geometry Engine. It describes geometry only. It does not describe CST commands, simulation settings, material-library settings, or mesh settings.

The goal is to let the downstream CST Builder reconstruct geometry from a stable, kernel-independent geometry file.

### Top-Level Structure

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
  "geometries": [],
  "export_rules": {}
}
```

### Geometry Object

The current main exported geometry is a planar conductor:

```json
{
  "id": "patch_conductor",
  "type": "planar_conductor",
  "unit": "mm",
  "plane": "XY",
  "outer_boundary": {},
  "holes": [],
  "metadata": {}
}
```

### Boundary Loop

A boundary loop does not repeat the final closing point. Closure is represented by `closed: true`.

```json
{
  "id": "outer",
  "role": "outer",
  "closed": true,
  "orientation": "CCW",
  "vertices": [
    {"x": 37.5, "y": 5.0},
    {"x": 37.5, "y": 25.0},
    {"x": 12.5, "y": 25.0},
    {"x": 12.5, "y": 5.0}
  ]
}
```

### Holes

Slot cutouts or boolean differences are exported under `holes`.

```json
{
  "id": "hole_001",
  "role": "hole",
  "closed": true,
  "orientation": "CCW",
  "vertices": [
    {"x": 23.0, "y": 11.25},
    {"x": 27.0, "y": 11.25},
    {"x": 27.0, "y": 12.75},
    {"x": 23.0, "y": 12.75}
  ]
}
```

### Coordinate System

- Unit: millimeters
- Plane: XY
- x axis: right
- y axis: up
- z is not exported in boundary vertices
- Geometry JSON describes 2D conductor boundaries

### Export Rules

The exporter and validator guarantee:

- boundary closed
- boundary continuous
- vertices ordered
- counter-clockwise orientation
- no duplicated vertices
- no self-intersection
- unit is millimeter

### Circles and Curves

Geometry JSON currently represents boundaries as vertex sequences. Circles, arcs, and curves in CadQuery are sampled into ordered vertices during boundary extraction.

The default circular boundary sample count is controlled by `CadQueryBoundaryExtractor.curve_samples`, currently 96.

### Future Extensibility

Future versions can add:

- multiple conductor geometries
- ground slots
- parasitic patches
- layer stack geometry
- curve primitive representation
- arc-aware CST builder
- geometry constraints
