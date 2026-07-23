# Plan

Use the diagnosis and the summarized inputs to create the next antenna design
strategy.

Rules:

- First read history.electromagnetic_knowledge, history.bo_effect_summary, and
  recent attempts[*].llm_effect_summary. Treat them as the experiment memory
  from prior simulations and use them to choose or reject the next strategy.
- Avoid repeating modifications that history shows were ineffective or harmful.
- Choose modifications because they are electromagnetically meaningful, not
  because they are always the smallest possible change.
- If previous iterations show weak improvement or stagnation, propose a
  coordinated geometry experiment instead of another single small tweak.
- You may combine patch resizing and multiple slot operations when they serve
  one coherent RF hypothesis.
- MoveFeed is disabled. Do not propose feed relocation, feed offset tuning,
  inset-feed MoveFeed operations, or any modification candidate named
  "MoveFeed". Keep the feed on the existing patch centerline.
- Keep the radiating patch body centered on the substrate. Patch resizing may
  change length or width, but the patch center is not an optimization variable.
  Use slot position/size offsets for asymmetric tuning when asymmetry is needed.
- Do not repeatedly add a slot and then delete it in the next iteration unless
  the history shows the slot clearly worsened the objective. Prefer modifying
  the overall geometry direction instead of undo/redo oscillation.
- Do not propose slot deletion during early design. DeleteSlot may only be
  considered when geometry_summary.json currently lists at least four slots.
  With fewer than four slots, prefer ResizePatch or AddSlot.
- If history.recent_operation_pattern reports weak_recent_improvement or
  add_delete_oscillation, explicitly change design hypothesis. Do not continue
  the same add/delete or resize/feed-only pattern.
- Limit the next iteration to no more than FIVE geometry modifications.
- Do not include CST commands.
- Do not include CadQuery commands.
- Do not include raw polygon or mesh edits.
- Do not include "DeleteSlot" as a modification candidate unless the current
  geometry already has at least four slots.
- Do not include "MoveFeed" as a modification candidate under any condition.
- If bo_parameterization_summary.json is present, respect the BO parameterized
  slot/model surface. Prefer modifications that build on useful BO variables
  and avoid immediately undoing a slot pattern that BO just optimized unless
  the summaries show it is harmful.

Return exactly one JSON object with this structure:

{
  "design_mode": "tune | reshape | slot_load | multi_slot | recover",
  "design_hypothesis": "...",
  "strategy": [
    "..."
  ],
  "rf_rationale": [
    "..."
  ],
  "anti_repetition_rule": "...",
  "modification_candidates": [
    {
      "operation": "...",
      "intent": "...",
      "expected_effect": "...",
      "risk": "low | medium | high"
    }
  ]
}
