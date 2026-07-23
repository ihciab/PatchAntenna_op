# Generate

Generate executable geometry modification operations for the Geometry Backend.

Use only:

- target.md design specification
- geometry_summary.json
- simulation_summary.json
- history.json
- bo_parameterization_summary.json when present
- diagnosis from the Diagnose step
- strategy from the Plan step

Rules:

- Use history.electromagnetic_knowledge, history.bo_effect_summary, and recent
  llm_effect_summary records before selecting operations. Do not ignore prior
  simulated strategy effects.
- Output no more than FIVE operations.
- Each operation must be a JSON object with:
  - "operation": operation name
  - "parameters": JSON object of numeric or string parameters
- Use only these operation names:
  - "ResizePatch"
  - "AddSlot"
  - "DeleteSlot"
- Use relative modification values when possible, such as length deltas,
  width deltas, or slot dimensions.
- ResizePatch uses relative millimeter deltas, for example:
  - {"length": -0.8}
  - {"width": 1.2}
- ResizePatch must preserve the radiating patch center on the substrate center.
  Do not use patch displacement as a tuning mechanism; asymmetry should come
  from slot position or slot size, not from moving the patch body.
- MoveFeed is disabled. Never output an operation named "MoveFeed". Do not
  tune the antenna by moving the feed point, changing feed offset, or creating
  an inset feed with MoveFeed. Keep the feed fixed on the existing patch
  centerline.
- AddSlot uses absolute center coordinates and dimensions in millimeters, for
  example:
  - {"shape": "rectangle", "x": 40.0, "y": 29.0, "width": 5.0, "height": 1.2}
- DeleteSlot uses:
  - {"id": "slot_or_hole_id"}
- DeleteSlot is forbidden during early design. Only output DeleteSlot when
  geometry_summary.json contains at least four existing slots. If there are
  fewer than four slots, use ResizePatch or AddSlot instead.
- Let the diagnosis determine operation magnitude. Use physically meaningful
  changes, including slot loading, when matching, bandwidth, mode behavior, or
  repeated weak improvement calls for it.
- If history shows repeated ResizePatch attempts with little improvement, do
  not output only ResizePatch again. Consider AddSlot if geometry permits.
  Consider DeleteSlot only after the design already has at least four slots.
- Prefer a coherent geometry experiment over a single isolated tweak when the
  design has stagnated. Examples of coherent operation sets:
  - ResizePatch + AddSlot + AddSlot
  - DeleteSlot + AddSlot only when replacing a harmful slot pattern after at
    least four slots already exist
- If recent history shows weak improvement, prefer THREE to FIVE coordinated
  operations rather than one tiny operation, unless geometry constraints leave
  only one valid option.
- Multiple slots are allowed when they express an RF design idea, such as
  symmetric perturbation, mode splitting, current-path lengthening, or bandwidth
  broadening. Keep each slot inside the patch.
- Avoid add/delete oscillation. Do not delete a slot merely because it exists;
  delete it only if at least four slots already exist and history or current
  diagnosis indicates a specific slot is harmful or should be replaced.
- If replacing a slot, explain the replacement as a new slot pattern rather than
  a reversal. When possible, combine replacement with patch or slot adjustment
  so the iteration tests a complete RF hypothesis.
- Do not output CST commands.
- Do not output CadQuery commands.
- Do not directly output final geometry.
- Do not include raw vertices.
- When bo_parameterization_summary.json is present, use its slot IDs, slot
  centers, dimensions, variable bounds, and port context to choose operations.
  The output must still be Geometry Backend operations, not BO variables.

Return exactly one JSON object with this structure:

{
  "iteration": 1,
  "reasoning": [
    "..."
  ],
  "strategy": [
    "..."
  ],
  "design_hypothesis": "...",
  "operations": [
    {
      "operation": "ResizePatch",
      "parameters": {
        "length": -0.5
      }
    }
  ]
}

The response must be strict JSON that passes Python json.loads. Use only
double-quoted JSON strings, no comments, no trailing commas, no Markdown, no
code fences, and no non-ASCII symbols such as Greek letters or ohm signs inside
string values.
