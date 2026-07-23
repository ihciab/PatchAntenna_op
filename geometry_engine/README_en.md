# Geometry Engine Guide

Geometry Engine is the middle layer between the LLM Design Agent and the CST modeling program.

Its responsibilities are:

- Receive Geometry DSL commands
- Maintain internal parametric `Patch` / `Feed` / `Slot` objects
- Apply geometry edits
- Validate geometry legality
- Export `patch.json`

It does not call the OpenAI API, the CST API, or any optimizer.

## System Position

```text
LLM
  -> Geometry DSL
  -> Geometry Engine
  -> patch.json
  -> CST Builder
  -> CST Simulation
```

Geometry Engine always edits the object model internally. It does not edit JSON directly. Only the exporter converts objects into `patch.json`.

## Quick Test

Use an existing design-agent run directory as input:

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test --run-dir design_agent_runs\initial_design_test --use-default-commands
```

This reads:

```text
design_agent_runs\initial_design_test\patch.json
```

Then it converts the file into a Geometry Engine `Patch` object and runs the default DSL smoke test sequence.

## Run Specific DSL Commands

Pass `--command` repeatedly. Commands are executed in order. After every geometry-mutating command, the engine runs the validator automatically. If validation fails, the engine rolls back to the state before that command.

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

## Python Usage

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

## Supported DSL Commands

### ResizePatch(length=?, width=?)

Resize the patch in millimeters. The patch center is preserved, and the feed remains attached to its original edge.

```text
ResizePatch(length=26, width=30)
ResizePatch(width=32)
ResizePatch(length=28)
```

### MoveFeed(dx=?, dy=?)

Move the feed point by a delta in millimeters. The feed must remain on a patch edge after the move.

```text
MoveFeed(dx=1.0, dy=0.0)
MoveFeed(dx=-2.0, dy=0.0)
```

In the current initial example, the patch bounds are approximately:

```text
x: 11 ~ 39
y: 12 ~ 36
feed: bottom edge, y = 12
```

So a bottom-edge feed should normally move only along x, not freely along y.

### AddSlot(shape="rectangle", x=?, y=?, width=?, height=?)

Add a rectangular slot. `(x, y)` is the slot center. `width` and `height` are dimensions in millimeters.

Valid examples:

```text
AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)
AddSlot(shape='rectangle', x=19, y=24, width=2, height=6)
```

Invalid example:

```text
AddSlot(shape='rectangle', x=25, y=12.2, width=4, height=1)
```

This slot crosses the lower patch boundary at `y=12`, so the validator rejects it.

Version 1 only supports `shape="rectangle"`. Future versions can add circular slots, rings, SRR slots, U-slots, and other families.

### DeleteSlot(id=?)

Delete a slot by id.

```text
DeleteSlot(id='slot_001')
```

If `AddSlot` does not receive an explicit id, Geometry Engine auto-generates:

```text
slot_001
slot_002
slot_003
```

### MirrorX()

Mirror the feed and all slots across the horizontal axis through the patch center.

```text
MirrorX()
```

For a bottom feed, `MirrorX()` changes the feed to a top feed.

### MirrorY()

Mirror the feed and all slots across the vertical axis through the patch center.

```text
MirrorY()
```

For a left feed, `MirrorY()` changes the feed to a right feed. For a bottom feed, the direction stays the same and only the x position changes.

### Validate()

Validate the current geometry state.

```text
Validate()
```

It returns `ValidationResult(valid=True, errors=[])` or a list of errors.

### ExportJSON(path)

Export the current Geometry Engine object model as `patch.json`.

```text
ExportJSON('design_agent_runs/initial_design_test/patch_with_slots.json')
```

You can also use the test script option:

```powershell
--export-patch-json design_agent_runs\initial_design_test\patch_with_slots.json
```

## Slot Test Cases

Start with these recommended cases.

Valid center slot:

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "Validate()" `
  --command "AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)" `
  --command "Validate()"
```

Valid multiple slots:

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)" `
  --command "AddSlot(shape='rectangle', x=19, y=24, width=2, height=6)" `
  --command "AddSlot(shape='rectangle', x=31, y=24, width=2, height=6)" `
  --command "Validate()"
```

Invalid boundary-touching slot:

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=25, y=12.2, width=4, height=1)"
```

Invalid slot crossing the right boundary:

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=38, y=24, width=4, height=2)"
```

Add then delete:

```powershell
python -m design_agent.scripts.run_geometry_engine_dsl_test `
  --run-dir design_agent_runs\initial_design_test `
  --command "AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)" `
  --command "DeleteSlot(id='slot_001')" `
  --command "Validate()"
```

## Geometry Rules

The current validator checks:

- All geometry is measured in millimeters
- Patch length and width must be positive
- Patch must use PEC material
- Patch outline must be closed
- Slot ids must be unique
- Slot shape must be rectangle in version 1
- Slots must be strictly inside the patch and cannot touch patch edges
- Feed width must be positive
- Feed must be on a patch edge
- Feed direction must match the edge it lies on
- Feed span must remain within the patch edge

## Export Format

The exporter generates a `design_agent_patch_v1` style JSON:

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

Note: in version 1, `AddSlot` exports slots into `conductor.cutouts`. This is cutout metadata. Geometry Engine validates the slots, but the CST Builder still needs future support to perform actual boolean slot subtraction.

## File Responsibilities

### geometry_engine/__init__.py

Package entry point. Exports common classes:

- `GeometryEngine`
- `ParameterizationImporter`

### geometry_engine/engine.py

Main Geometry Engine entry point.

Responsibilities:

- Accept DSL strings or command objects
- Parse DSL through the parser
- Create command objects through the registry
- Execute commands
- Run validation automatically after mutations
- Roll back geometry state on validation failure
- Provide built-in `Validate()` and `ExportJSON(path)` commands

Core classes:

- `GeometryEngine`
- `ValidateCommand`
- `ExportJSONCommand`

### geometry_engine/context.py

Runtime context.

Responsibilities:

- Store the current `Patch` object
- Store `GeometryValidator`
- Store `PatchJSONExporter`
- Generate new slot ids
- Provide `validate()` and `export_json()`

Core class:

- `GeometryContext`

### geometry_engine/registry.py

Command registry.

Responsibilities:

- Map DSL command names to command classes
- Register built-in commands
- Support future command extension

When adding a new DSL command, you normally only need to:

1. Add a command class
2. Set `dsl_name`
3. Register it in the registry

The engine core does not need to change.

Core class:

- `CommandRegistry`

### geometry_engine/validator.py

Geometry validator.

Responsibilities:

- Validate patch dimensions
- Validate patch material
- Validate patch closure
- Validate slots are inside the patch
- Validate feed lies on a patch edge

Core classes:

- `GeometryValidator`
- `ValidationResult`
- `GeometryValidationError`

### geometry_engine/exporter.py

JSON exporter.

Responsibilities:

- Convert the internal `Patch` object to `patch.json`
- Export patch polygon primitives
- Export port information
- Export slots into `conductor.cutouts`

Core class:

- `PatchJSONExporter`

### geometry_engine/importer.py

JSON import bridge.

Responsibilities:

- Initialize `Patch` from design-agent `patch.json`
- Initialize `Patch` from CST adapter `parameterization_from_agent.json`
- Infer patch dimensions and feed contact edge from patch/feed component bboxes

Note: the importer is only a bridge for testing and compatibility with existing artifacts. The Geometry Engine core still works on objects, not JSON.

Core classes:

- `ParameterizationImporter`
- `ParameterizationImportError`

### geometry_engine/geometry/patch.py

Patch object model.

Responsibilities:

- Maintain patch length, width, center, material, and layer
- Hold a `Feed`
- Hold multiple `Slot` objects
- Provide resize, add/delete slot, and mirror operations
- Generate patch vertices
- Check whether a feed point lies on an edge

Core class:

- `Patch`

### geometry_engine/geometry/feed.py

Feed object model.

Responsibilities:

- Maintain feed point location
- Maintain feed width and direction
- Support move and mirror operations
- Generate the two port-edge endpoints

Core class:

- `Feed`

### geometry_engine/geometry/slot.py

Slot object model.

Responsibilities:

- Maintain slot id, shape, center, width, and height
- Compute slot bounds
- Support mirror operations
- Generate slot polygon vertices

Core class:

- `Slot`

### geometry_engine/dsl/command.py

DSL command abstraction.

Responsibilities:

- Define parsed command data structure
- Define the abstract base class for all command classes

Core classes:

- `ParsedCommand`
- `GeometryCommand`

### geometry_engine/dsl/parser.py

DSL parser.

Responsibilities:

- Parse function-call-style DSL
- Support one command
- Support multi-command scripts
- Accept only safe literal arguments

Supported form:

```text
AddSlot(shape='rectangle', x=25, y=24, width=4, height=1.5)
Validate()
```

Core classes:

- `DSLParser`
- `DSLParseError`

### geometry_engine/commands/resize_patch.py

Implementation of `ResizePatch`.

### geometry_engine/commands/move_feed.py

Implementation of `MoveFeed`.

### geometry_engine/commands/add_slot.py

Implementation of `AddSlot`.

Version 1 fully supports rectangular slots only.

### geometry_engine/commands/delete_slot.py

Implementation of `DeleteSlot`.

### geometry_engine/commands/mirror.py

Implementation of `MirrorX` and `MirrorY`.

### design_agent.scripts.run_geometry_engine_dsl_test

Command-line test script.

Responsibilities:

- Load `--run-dir` / `--patch-json` / `--parameterization-json`
- Initialize Geometry Engine
- Execute DSL commands in order
- Print validator result after each command
- Optionally export `patch.json`

## Adding New DSL Commands

Example: future `AddStub()`.

1. Create a new file:

```text
geometry_engine/commands/add_stub.py
```

2. Define a command class:

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

3. Register it in `registry.py` inside `with_builtin_commands()`.

The engine execution flow does not need to change.

## Current Limitations

- Patch is currently modeled as a rectangle
- Slot currently supports rectangle only
- Slot overlap is not checked yet
- Minimum copper clearance is not checked yet
- Mirror validation for non-rectangular patch outlines is still TODO
- `conductor.cutouts` is currently cutout metadata exported by Geometry Engine; the CST Builder still needs real boolean subtraction support

