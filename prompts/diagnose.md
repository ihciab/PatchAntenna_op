# Diagnose

Analyze the four summarized inputs:

- target.md: design specifications in Markdown
- geometry_summary.json: current summarized geometry
- simulation_summary.json: current summarized electromagnetic performance
- history.json: previous modification attempts and effects

Determine:

- current design problems
- evidence from the summaries
- likely physical causes
- constraints that should guide the next modification

Use cautious RF reasoning. If a required metric is missing, state that it is
missing and reason from the available summaries only.

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
  "constraints": [
    "..."
  ]
}
