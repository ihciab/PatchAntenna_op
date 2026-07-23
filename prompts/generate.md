# Generate

Generate executable geometry modification operations for the Geometry Backend.

Use only:

- target.md design specification
- geometry_summary.json
- simulation_summary.json
- history.json
- diagnosis from the Diagnose step
- strategy from the Plan step

Rules:

- Output no more than THREE operations.
- Each operation must be a JSON object with:
  - "operation": operation name
  - "parameters": JSON object of numeric or string parameters
- Use only these operation names:
  - "ResizePatch"
  - "MoveFeed"
  - "AddSlot"
  - "DeleteSlot"
- Use relative modification values when possible, such as length deltas,
  width deltas, feed offsets, or slot dimensions.
- For ResizePatch, use relative millimeter deltas:
  - {"length": -0.5}
  - {"width": 0.3}
- For MoveFeed, use relative millimeter deltas:
  - {"dx": 0.2, "dy": 0}
- For AddSlot, use absolute center coordinates and dimensions in millimeters:
  - {"shape": "rectangle", "x": 40.0, "y": 29.0, "width": 3.0, "height": 1.0}
- For DeleteSlot, use:
  - {"id": "slot_or_hole_id"}
- Keep operation magnitudes conservative.
- Do not output CST commands.
- Do not output CadQuery commands.
- Do not directly output final geometry.
- Do not include raw vertices.

Return exactly one JSON object with this structure:

{
  "iteration": 1,
  "reasoning": [
    "..."
  ],
  "strategy": [
    "..."
  ],
  "operations": [
    {
      "operation": "ResizePatch",
      "parameters": {
        "length": -0.5
      }
    }
  ]
}
