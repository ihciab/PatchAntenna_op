# Reflect Bayesian Optimization

Summarize what the completed Bayesian optimization run learned about the
antenna design strategy.

Use only the supplied structured data:

- target.md: design objectives
- history.json: prior closed-loop knowledge
- optimization_history.json: BO evaluations and outcomes
- bo_parameterization_summary.json, when present: the BO surface and variable
  choices

Write concise electromagnetic lessons that will help the next design iteration.
Focus on what kinds of variable moves improved or hurt the objective, whether
the BO search found a useful region, and which variables should be emphasized
or avoided next.

Rules:

- Do not invent unavailable simulation metrics.
- If the BO run mostly failed, summarize the failure mode clearly.
- Prefer RF and geometry explanations over generic optimization language.
- Mention resonance, matching, bandwidth, gain, slots, patch size, and feed
  behavior when the records support it.
- Return exactly one JSON object.

Return structure:

{
  "available": true,
  "summary": "One concise paragraph describing the BO learning outcome.",
  "electromagnetic_lessons": [
    "..."
  ],
  "strategy_effect": {
    "helped": [
      "..."
    ],
    "hurt": [
      "..."
    ],
    "uncertain": [
      "..."
    ]
  },
  "next_iteration_guidance": [
    "..."
  ],
  "avoid_repeating": [
    "..."
  ]
}
