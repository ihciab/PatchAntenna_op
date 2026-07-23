# Plan

Use the diagnosis and the four summarized inputs to create an optimization
strategy for the next iteration.

Rules:

- Avoid repeating modifications that history shows were ineffective or harmful.
- Prefer small, interpretable geometry changes.
- Limit the next iteration to no more than THREE geometry modifications.
- Do not include CST commands.
- Do not include CadQuery commands.
- Do not include raw polygon or mesh edits.
- Reserve higher-risk geometry changes for later iterations unless clearly
  justified by the diagnosis.

Return exactly one JSON object with this structure:

{
  "strategy": [
    "..."
  ],
  "avoid": [
    "..."
  ],
  "modification_candidates": [
    {
      "operation": "...",
      "intent": "...",
      "expected_effect": "...",
      "risk": "low"
    }
  ]
}
