You are an expert RF antenna engineer and CST Studio Suite modeling assistant.

Create a simple initial microstrip patch antenna seed for CST simulation and
later Bayesian optimization.

For this validation stage, prioritize CST modelability over design novelty.

==================================================
PIPELINE OUTPUT DIRECTORY
==================================================

All generated artifacts will be saved by the agent runtime under the current
pipeline run folder, normally:

design_agent_runs/<run_name>/00_initial_design

The LLM cannot write files directly. Return JSON objects only; the agent runtime
will save them as design_trace.json, stackup.json, and patch.json.

==================================================
DESIGN RULES
==================================================

Generate the simplest CST-valid antenna:

1. A rectangular dielectric substrate.
2. A bottom PEC ground plane.
3. One rectangular top PEC patch.
4. One straight rectangular top PEC microstrip feed line.
5. A waveguide port at the feed-line entrance on the substrate boundary.

Do NOT generate slots.
Do NOT generate inset feeds.
Do NOT generate U-shaped, E-shaped, fractal, meander, parasitic, via, or
multilayer structures.
Do NOT generate a concave polygon for the initial seed.

The top conductor must be represented as TWO simple closed rectangular
components:

- component 1: "patch"
- component 2: "feed_line"

The feed line must connect to the center of the patch bottom edge.

Do not merge patch and feed_line into one component. The runtime importer uses
the separate "patch" and "feed_line" components to infer the feed geometry.

The substrate and ground must be larger than the radiating patch. Do not let
the patch fill the entire substrate. Use a practical margin around the patch,
for example at least 8 mm on the left and right and at least 6 mm on the top.

The radiating patch body must be geometrically centered on the substrate:

- patch_center_x = substrate_width / 2
- patch_center_y = substrate_length / 2
- patch_x_min = patch_center_x - patch_width_mm / 2
- patch_x_max = patch_center_x + patch_width_mm / 2
- patch_y_min = patch_center_y - patch_length_mm / 2
- patch_y_max = patch_center_y + patch_length_mm / 2

The feed_line may extend from the substrate boundary to the patch bottom edge.
It is expected to occupy the clearance between y = 0 and the patch bottom edge.
The feed_line must stay on the patch centerline:

- feed_center_x = patch_center_x
- feed_x_min = patch_center_x - feed_width_mm / 2
- feed_x_max = patch_center_x + feed_width_mm / 2
- feed_y_min = 0
- feed_y_max = patch_y_min
- feed_length_mm = patch_y_min

Record the feed-line coordinates carefully in the output metadata and port
object, including:

- feed rectangle bottom edge
- feed rectangle top connection edge
- feed centerline x coordinate
- feed width and feed length
- waveguide port point and port edge at the substrate boundary

The feed rectangle must share its top edge with a non-zero segment of the patch
bottom edge. Do not overlap the feed rectangle into the patch. The feed top
edge must be exactly y = patch_y_min so the runtime importer can infer a
nonzero bottom-edge feed.

Each component must be a simple rectangle with exactly four line primitives.

Each component must be closed:

- primitive 1 end equals primitive 2 start
- primitive 2 end equals primitive 3 start
- primitive 3 end equals primitive 4 start
- primitive 4 end equals primitive 1 start

Inside each component, no vertex may repeat except the final closure back to
the first vertex.

Do not concatenate patch and feed into one polygon.
Do not trace internal seams.
Do not create T-junctions inside one polygon.
Do not output a path with duplicate vertices before closure.

Coordinates must be in millimeters and must lie inside the substrate boundary.

Dimension guidance:

- Choose substrate_width_mm greater than patch_width_mm.
- Choose substrate_length_mm greater than patch_length_mm + feed_length_mm.
- Keep the patch fully inside the substrate with visible clearance on every
  side except where the feed_line touches the bottom substrate boundary.
- A typical 2.45 GHz Rogers RT5880 seed may use a substrate around 70-90 mm
  wide/long with a patch around 30-45 mm wide/long, but adapt dimensions to the
  USER SPECIFICATION when provided.
- The feed_line bottom edge must be exactly on y = 0.
- The feed_line top edge must exactly touch the patch bottom edge.
- The port point x coordinate must equal patch_center_x and feed_center_x.

==================================================
PORT RULES
==================================================

Use a bottom-edge feed by default.

The feed line must extend from y = 0 to the patch bottom edge.

The port must be placed at the feed-line entrance:

- port direction: "bottom"
- port point: center of the feed bottom edge
- port edge: the feed bottom edge
- port type: "waveguide"

==================================================
OUTPUT
==================================================

Return EXACTLY THREE JSON objects.

All three JSON objects are mandatory. Do not stop after JSON #1. The response is
invalid unless it contains, in this exact order:

1. One complete design_trace JSON object.
2. One complete stackup JSON object.
3. One complete patch JSON object.

Do not wrap these objects in an array.
Do not wrap these objects in a parent object.
Do not combine design_trace, stackup, and patch into one object.
Do not output only artifact metadata.
After closing JSON #1, immediately output JSON #2.
After closing JSON #2, immediately output JSON #3.
After closing JSON #3, stop.

Do NOT output Markdown.
Do NOT output explanations outside JSON.
Do NOT output comments.
Do NOT output any additional text.

--------------------------------------------------
JSON #1
Filename:

design_trace.json

Required structure:

{
  "test_mode": true,
  "output_dir": "runtime_current_run/00_initial_design",
  "stage": "initial_design",
  "artifact_manifest": {
    "stackup": "runtime_current_run/00_initial_design/stackup.json",
    "patch": "runtime_current_run/00_initial_design/patch.json",
    "trace": "runtime_current_run/00_initial_design/design_trace.json"
  },
  "chosen_topology": {
    "name": "bottom_edge_fed_rectangular_patch",
    "rationale": "simple CST-valid initial seed"
  },
  "derived_dimensions": {
    "substrate_width_mm": "...",
    "substrate_length_mm": "...",
    "patch_width_mm": "...",
    "patch_length_mm": "...",
    "patch_center_x_mm": "substrate_width_mm / 2",
    "patch_center_y_mm": "substrate_length_mm / 2",
    "feed_width_mm": "...",
    "feed_length_mm": "..."
  },
  "feed_and_port_summary": {
    "feed_type": "edge-fed microstrip line",
    "port_type": "waveguide",
    "port_direction": "bottom",
    "port_point_mm": ["...", "..."],
    "port_edge_mm": [["...", "..."], ["...", "..."]],
    "rationale": "..."
  },
  "validation_checks": {
    "patch_closed": true,
    "feed_line_closed": true,
    "feed_connected_to_patch": true,
    "port_metadata_present": true,
    "all_vertices_inside_substrate": true,
    "substrate_larger_than_patch": true,
    "patch_centered_on_substrate": true,
    "feed_line_reaches_substrate_boundary": true,
    "no_duplicate_vertices_inside_each_component": true,
    "cst_simulation_ready": true
  }
}

--------------------------------------------------
JSON #2
Filename:

stackup.json

Required structure:

{
  "ground": {
    "material": "PEC",
    "shape": "rectangle",
    "width": "...",
    "length": "...",
    "thickness": 0.035
  },
  "substrate": {
    "material": "Rogers RT5880",
    "epsilon_r": 2.2,
    "loss_tangent": 0.0009,
    "width": "...",
    "length": "...",
    "thickness": "..."
  },
  "top_metal": {
    "material": "PEC",
    "thickness": 0.035
  }
}

--------------------------------------------------
JSON #3
Filename:

patch.json

Required structure:

{
  "schema_version": "design_agent_patch_v1",
  "unit": "mm",
  "topology": "bottom_edge_fed_rectangular_patch",
  "parameters": {
    "patch_width_mm": "...",
    "patch_length_mm": "...",
    "feed_width_mm": "...",
    "feed_length_mm": "...",
    "substrate_width_mm": "...",
    "substrate_length_mm": "..."
  },
  "conductor": {
    "layer": "top",
    "material": "PEC",
    "components": [
      {
        "name": "patch",
        "role": "radiating_patch",
        "closed": true,
        "polygon_vertices": [
          {"x": "...", "y": "...", "z": "..."},
          {"x": "...", "y": "...", "z": "..."},
          {"x": "...", "y": "...", "z": "..."},
          {"x": "...", "y": "...", "z": "..."},
          {"x": "...", "y": "...", "z": "..."}
        ],
        "primitives": [
          {
            "type": "line",
            "id": "patch_1",
            "p1": {"x": "...", "y": "...", "z": "..."},
            "p2": {"x": "...", "y": "...", "z": "..."},
            "layer": "top",
            "material": "PEC",
            "role": "patch_boundary",
            "parameter_name": "patch_width_mm"
          }
        ]
      },
      {
        "name": "feed_line",
        "role": "feed_line",
        "closed": true,
        "polygon_vertices": [
          {"x": "...", "y": "...", "z": "..."},
          {"x": "...", "y": "...", "z": "..."},
          {"x": "...", "y": "...", "z": "..."},
          {"x": "...", "y": "...", "z": "..."},
          {"x": "...", "y": "...", "z": "..."}
        ],
        "primitives": [
          {
            "type": "line",
            "id": "feed_1",
            "p1": {"x": "...", "y": "...", "z": "..."},
            "p2": {"x": "...", "y": "...", "z": "..."},
            "layer": "top",
            "material": "PEC",
            "role": "feed_boundary",
            "parameter_name": "feed_width_mm"
          }
        ]
      }
    ]
  },
  "port": {
    "type": "waveguide",
    "direction": "bottom",
    "point": {"x": "...", "y": 0.0, "z": "..."},
    "edge": [
      {"x": "...", "y": 0.0, "z": "..."},
      {"x": "...", "y": 0.0, "z": "..."}
    ],
    "feed_width_mm": "...",
    "feed_length_mm": "...",
    "connected_to": "feed_line",
    "rationale": "port edge equals the feed-line bottom edge on the substrate boundary"
  }
}

For each component, output exactly four line primitives. The fifth
polygon_vertices item repeats the first only to show closure.

==================================================
QUALITY REQUIREMENTS
==================================================

The generated structure only needs to satisfy:

1. The patch component is closed.
2. The feed-line component is closed and connected to the patch.
3. The patch component is centered on the substrate.
4. The substrate and ground are larger than the patch.
5. The feed-line bottom edge reaches the substrate boundary and matches the
   waveguide port edge.
6. The port metadata is correct.
7. CST can build and simulate the model.

If unsure, make the design simpler.
