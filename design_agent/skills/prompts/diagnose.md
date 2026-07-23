# Diagnose

Analyze the summarized inputs:

- target.md: design specifications in Markdown
- geometry_summary.json: current summarized geometry
- simulation_summary.json: current summarized electromagnetic performance
- history.json: previous modification attempts and effects
- bo_parameterization_summary.json, when present: BO optimization surface,
  tunable slot/model variables, port context, and parameterized slot geometry

Diagnose the antenna as an RF designer. Determine:

- current design problems
- evidence from the summaries
- likely physical causes
- electromagnetic opportunities for improvement

Before diagnosing, review history.electromagnetic_knowledge,
history.bo_effect_summary, and each recent attempt's llm_effect_summary. Use
those lessons as experimental RF knowledge: which strategies improved
resonance, matching, bandwidth, or gain; which strategies were ineffective; and
which geometry ideas should not be repeated.

Use patch antenna knowledge, including electrical length, fringing fields,
feed coupling, current path perturbation, slot loading, mode splitting,
impedance matching, bandwidth, and gain tradeoffs. If a metric is missing,
state that it is missing and reason from the available summaries only.
When bo_parameterization_summary.json is present, use it as evidence for which
slot and model dimensions were optimized, which variables were sensitive, and
which geometric degrees of freedom remain useful for the next LLM edit.

Return exactly one JSON object with this structure:

{
  "current_problems": [
    "..."
  ],
  "evidence": [
    "..."
  ],
  "possible_physical_causes": [
    "..."
  ],
  "design_opportunities": [
    "..."
  ]
}
