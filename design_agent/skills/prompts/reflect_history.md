# Reflect History

Summarize what the latest design strategy did electromagnetically.

Use only the supplied structured data:

- operation_plan.json: the strategy and geometry operations that were tried
- before_simulation_summary.json: performance before the strategy
- after_simulation_summary.json: performance after the strategy
- geometry_summary.json: geometry after the strategy
- history.json: previous attempts and lessons

Write concise, descriptive engineering knowledge that will help the next design
iteration. Explain what improved, what worsened, what probably caused the
change, and what should be tried or avoided next.

Rules:

- Do not invent unavailable CST metrics.
- If a metric is missing, say it is missing.
- Prefer electromagnetic explanations over generic optimization language.
- Mention relevant geometry features such as patch size, slot placement, slot
  loading, current path length, impedance matching, resonance, bandwidth, mode
  splitting, or gain tradeoffs when supported by the data.
- Keep the summary useful for the next LLM planner.
- Return exactly one JSON object.

Return structure:

{
  "available": true,
  "summary": "One concise paragraph describing the strategy effect.",
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
