# Repair Geometry Operations

The previous operation_plan.json failed in the Geometry Backend.

Use the same electromagnetic design intent, but produce a corrected executable
operation plan that satisfies Geometry Engine constraints.

Important backend constraints:

- MoveFeed is disabled. Do not repair a failed plan by moving the feed,
  changing feed offset, or creating an inset-feed MoveFeed operation.
- If the failed plan contains MoveFeed, replace that intent with ResizePatch
  and/or AddSlot operations that preserve the fixed centerline feed.
- Slots must be strictly inside the patch boundary.
- Slot dimensions must be positive and finite.
- DeleteSlot is only allowed when geometry_summary.json contains at least four
  existing slots. If fewer than four slots exist, repair a bad slot operation by
  moving/reducing the slot or by using ResizePatch/AddSlot, not by deleting.
- Patch dimensions must remain positive.
- The radiating patch body must remain centered on the substrate; do not repair
  a failed plan by shifting the whole patch. Slots may be asymmetric.
- Output no more than FIVE operations.
- Use only operation names supported by the Geometry Backend:
  - "ResizePatch"
  - "AddSlot"
  - "DeleteSlot"

Do not abandon antenna-design reasoning. Correct the failed operation in the
clearest way that preserves the RF intent when possible. If the failed concept
is geometrically impossible, choose another physically meaningful operation.
Do not collapse a multi-operation design into a single tiny tweak unless the
error specifically requires it.

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

The example structure above is only a schema example. The response must be
strict JSON that passes Python json.loads, with no Markdown, no comments, no
trailing commas, and no non-ASCII symbols inside string values.
