## 2026-05-30
### Changed files
- optimization_pipeline.py
- docs/bo_change_log.md
- logs/algorithm_updates.md

### Purpose
Change invalid-geometry handling in the Bayesian Optimization loop from an early-stop trigger to a rollback-and-continue policy. Larger control-point displacement ranges are expected to produce more invalid samples during exploration; those samples should teach the optimizer through a penalty, but should not prematurely stop the run.

### Behavior
- `invalid_geometry` samples are still recorded in `optimization_history.json`.
- The optimizer still receives the invalid-geometry penalty objective.
- CST build/simulation is skipped for invalid geometry.
- Invalid geometry does not update `best_design`.
- Invalid geometry no longer consumes `no_improvement_patience`.
- Invalid geometry ratio no longer triggers early stopping.
- `MAX_INVALID_RATIO` / `--max-invalid-ratio` are retained for compatibility, but run metadata now records that invalid-ratio early stop is disabled.

### Validation samples
- `D:\Anaconda\envs\paper\python.exe -m py_compile optimization_pipeline.py`
- `python -m py_compile optimization_pipeline.py`

### Remaining risks
- Runs with many invalid samples may now use more evaluations before reaching the 30-iteration cap.
- The optimizer receives penalties for invalid regions, but very aggressive displacement ranges can still spend many trials near invalid geometry boundaries.

## 2026-05-25
### Changed files
- Rebuild/port_topology_detector.py
- Rebuild/PortSearch.py
- fss_parameterized_cst_pipeline.py
- test_patch_port_detection.py
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt
- logs/algorithm_updates.md

### Purpose
Add an optional Stage-1 patch-antenna port detector that is topology-first and plug-and-play with the existing PortSearch pipeline. Existing analyze() behavior and nearest-edge output remain unchanged.

The detector is now integrated into the main fss_parameterized_cst_pipeline.py Port Summary stage. Each full pipeline run writes patch_port_detection into port_summary.json and writes topology debug images under 03_port_detection while preserving legacy closest_edge fields for the CST builder.

### Key thresholds
- border_distance_px = 8
- min_skeleton_component_size = 3
- min_port_score = 10.0
- max_ports = 4
- min_component_area inherits the caller/analyzer setting

### Validation samples
- Simple microstrip patch with left-border feed.
- Inset-feed patch with bottom-border feed.
- Noisy screenshot-like mask with isolated conductive artifacts.
- Border-frame-only image to guard against false ports from screenshot frames.
- Main pipeline _create_port_summary integration test confirming port_summary.json contains patch_port_detection and 03_port_detection debug outputs.

### Remaining risks
- Stage 1 uses endpoint-to-centroid distance as a path-length proxy.
- No graph geodesic path search yet.
- CPW and multi-port reasoning are reserved for later stages.

## 2026-05-25
### Changed files
- Rebuild/port_topology_detector.py
- Rebuild/PortSearch.py
- fss_parameterized_cst_pipeline.py
- parameterized_json_to_cst.py
- test_patch_port_detection.py
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt
- logs/algorithm_updates.md

### Purpose
Fix the run_20260525_192027 / test48 case where legacy PortSearch selected the top image edge while the physical feed is the bottom patch port. Connect the topology port result to downstream CST waveguide-port creation.

### Key thresholds
- Low-saturation conductor fallback: saturation <= 90 and grayscale in [20, 251].
- Effective bbox-border threshold: max(border_distance_px, min(20, 15% of active bbox short side)).
- Wide solid-mask pseudo endpoint penalty: local_width/reference_width >= 0.55.

### Validation samples
- pipeline_runs/_debug_patch_port_test48_after_fix2 on test/test48.png:
  - selected patch topology port point = [213, 353]
  - direction = bottom
  - local_width = 19.0
  - fallback_from_subject_mask = true
- test_patch_port_detection.py:
  - synthetic fallback bottom-feed sample
  - CST builder preference for patch_port_detection over legacy closest_edge
  - existing microstrip/inset/noise/frame tests

### Remaining risks
- CST port span is still estimated from image-space local_width.
- CPW ground-gap semantics and multi-port disambiguation are not implemented yet.
- Legacy Simulink/Simulation.py path still has older direct port placement logic; the main fss_parameterized_cst_pipeline.py path now uses parameterized_json_to_cst.py topology-port handoff.
## 2026-05-25
### Changed files
- run_parameterization_only.py
- fss_parameterized_cst_pipeline.py
- fss_simulation_pipeline.py
- Rebuild/fssdetector_pipeline.py
- geometry_driven_parameterizer.py
- geometry_graph_parameterizer.py
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt
- logs/algorithm_updates.md

### Purpose
Restore the repair-first, solid-mask topology behavior shown by the successful `param_only_20260525_163434` run for test47-like patch antenna screenshots. The bad `param_only_20260525_213410` run used the raw image path and selected graph-local primitives from the wrong representation.

### Key thresholds
- No new geometry thresholds were introduced.
- Existing solid-mask gate remains `foreground_ratio >= max(max_stroke_mask_foreground_ratio * 2.0, 0.12)`.
- Existing `--skip-fss-cleanup` and `--honor-instance-skip` still allow explicit raw-image debugging.

### Validation samples
- `pipeline_runs/_verify_solid_mask_priority_20260525`: `image_preparation.stage=fss_repair`, `detector_passthrough=false`, `normalization_applied=true`, `parameterization_status.actual_backend=solid_mask_topology`.
- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\geometry_driven_parameterizer.py .\geometry_graph_parameterizer.py .\Rebuild\fssdetector_pipeline.py .\fss_simulation_pipeline.py .\run_parameterization_only.py .\fss_parameterized_cst_pipeline.py`
- `D:\Anaconda\envs\linefor\python.exe -m pytest -q test_patch_topology_mode.py test_graph_local_parameterizer.py`

### Remaining risks
- One existing FSS detector console message still says the no-text path skips FSS cleanup, but behavior now continues into `process_edges()`.
- Full CST build/simulation was not run in this verification turn; the main pipeline entry point now shares the same repair-first image-preparation policy.
## 2026-05-25
### Changed files
- run_parameterization_only.py
- fss_parameterized_cst_pipeline.py
- fss_simulation_pipeline.py
- Rebuild/fssdetector_pipeline.py
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt
- logs/algorithm_updates.md

### Purpose
Correction after user clarification: `skip_fss_cleanup=true` is intentional for already-processed images and should keep parameterization validation fast. Restore the no-text passthrough policy.

### Key thresholds
- No threshold changes.
- Default `honor_instance_skip=True` is restored.
- YOLO no-text precheck restores passthrough behavior.

### Validation samples
- `pipeline_runs/_verify_skip_fss_restored_20260525`: `image_preparation.stage=direct_input_image`, `skip_fss_cleanup=true`, parameterized `test48.png` without FSS detector, exported 7 graph-local primitives.
- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\fss_simulation_pipeline.py .\Rebuild\fssdetector_pipeline.py .\run_parameterization_only.py .\fss_parameterized_cst_pipeline.py .\geometry_driven_parameterizer.py .\geometry_graph_parameterizer.py`

### Remaining risks
- Direct-image validation assumes the input is already suitable for parameterization. Use `--ignore-instance-skip` when a run must force FSS repair despite instance skip flags.
## 2026-05-25
### Changed files
- geometry_driven_parameterizer.py
- geometry_graph_parameterizer.py
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt
- logs/algorithm_updates.md

### Purpose
Balance direct-input patch antenna edge selection for the user-provided test47/test48 pair while keeping FSS skipped. The goal is to preserve antenna body and details, avoid canvas-frame/double-edge dominance, and use line-rich graph-local primitives when the edge map is already clean.

### Key thresholds
- `auto_dense_filled_mask`: auto representation is `stroke_mask`, `auto_foreground_ratio >= max(max_stroke_mask_foreground_ratio * 2.0, 0.12)`, and `auto_to_canny_ratio >= 2.0`.
- Sparse auto edge path is unchanged: `auto_to_canny_ratio <= max_sparse_auto_to_canny_ratio` and close-extra evidence must pass the existing close-parallel threshold.
- No JSON schema changes.

### Validation samples
- `pipeline_runs/_iter04_test47_dense_solid_20260525`: direct input `test47.png`, FSS skipped, `actual_backend=solid_mask_topology`, full body/details preserved and canvas frame suppressed.
- `pipeline_runs/_iter05_test48_regression_20260525`: direct input `test48.png`, FSS skipped, `actual_backend=graph_local_primitives`, 12 primitives, matching the provided `run_20260525_192027` behavior.
- `D:\Anaconda\envs\linefor\python.exe -m pytest -q test_patch_topology_mode.py test_graph_local_parameterizer.py`
- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\geometry_driven_parameterizer.py .\geometry_graph_parameterizer.py .\run_parameterization_only.py .\fss_simulation_pipeline.py .\Rebuild\fssdetector_pipeline.py`

### Remaining risks
- Filled-mask topology is intentionally boundary based, not centerline graph based. It is only selected for dense stroke-mask cases so sparse edge images keep graph-local line primitives.

## 2026-05-26
### Changed files
- parameterized_json_to_cst.py
- test_patch_port_detection.py
- logs/algorithm_updates.md
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt

### Purpose
Fix the patch-antenna CST handoff where the topology port endpoint was correct in image space but the generated waveguide port face could sit just outside the reconstructed feedline. The CST builder now snaps patch-port endpoints to nearby parameterized geometry, moves the port plane slightly inward along the feed direction, and writes precise port-plane coordinates instead of integer floor/ceil values.

Also rebuilds closed CST contours from graph-local open edge fragments before extrusion. This prevents graph-local parameterizations such as test48 from producing open-curve `ExtrudeCurve` failures and leaving the CST model without the antenna metal.

### Key thresholds
- Port geometry search half-widths: `1.25 * local_width`, `2.5 * local_width`, `4.0 * local_width`, with minimums of 8, 24, and 40 px.
- Port inward overlap: `max(3 px, min(0.5 * local_width, 14 px))`.
- Open-edge contour join tolerance: median segment length times 1.5, clamped to 2..5 px.

### Validation samples
- `test_patch_port_detection.py`: 10 tests passed, including inward snapping for the test48-style bottom feed endpoint and closed-contour reconstruction from open graph edges.
- Actual `pipeline_runs/run_20260526_120947` dry check: 12 open graph edges rebuilt into 1 closed CST contour; patch port snapped from `(213, 353)` to `(204.78, 341.66)`.
- CST build-only verification: `pipeline_runs/_verify_cst_handoff_20260526/Verify_Port_Handoff.cst` built from the run_20260526_120947 parameterization.
- CST solver handoff verification: `pipeline_runs/_verify_cst_port_solver_20260526/Verify_Port_Solver.cst` completed solver startup/excitation without `ExtrudeCurve` errors in `Result/output.txt`.

### Remaining risks
- The verification solver run completed, but S11 export reported a CST result-id lookup issue (`ResultItem does not exist for run id=0`). The CST handoff now creates buildable metal and a connected port plane, but final S11 export handling may still need a separate result-tree fallback.
- The contour rebuild is intentionally conservative and only accepts open graph fragments that can be closed within the join tolerance.

## 2026-05-26
### Changed files
- Rebuild/port_geometry_builder.py
- Rebuild/port_topology_detector.py
- test_patch_port_detection.py
- logs/algorithm_updates.md
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt

### Purpose
Correction after user clarification: do not compensate for port mistakes in the CST simulation/reconstruction layer. The previous CST-side snapping/closed-contour handoff experiment was reverted. Port geometry is now refined inside the patch-port detection subsystem itself.

The new `CSTPortGeometryBuilder` converts a selected skeleton endpoint into CST-ready excitation geometry by estimating feed tangent from the skeleton path, estimating feed width from distance transform plus orthogonal cross-section scan, offsetting the port plane outward from the conductor terminal, and generating a rectangle perpendicular to the feed direction.

### Key thresholds
- Skeleton tangent trace distance: 14 px.
- Width samples: first 8 skeleton-path pixels near the endpoint.
- Outward offset: `max(3 px, 0.9 * feed_width)`.
- Port width padding: `1.35 * feed_width`.
- Port height for debug rectangle: `max(2 px, 0.25 * feed_width)`.

### Validation samples
- `pytest -q test_patch_port_detection.py` passed 10 tests.
- Actual run_20260526_173906 dry check: bottom endpoint `[213, 353]` refined to CST plane center approximately `[216, 362]`; geometry metadata keeps the original endpoint and rectangle.
- Actual run_20260526_190510 dry check: selected bottom port is refined outward while leaving CST settings untouched.

### Remaining risks
- Current CST consumption still uses the existing `point/direction/local_width` fields, so arbitrary angled rectangles are recorded in debug metadata but not yet consumed by a dedicated CST arbitrary-port builder.
- Multi-port disambiguation and CPW-specific ground-gap semantics remain future work.

## 2026-05-26
### Changed files
- Rebuild/port_topology_detector.py
- test_patch_port_detection.py
- logs/algorithm_updates.md
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt

### Purpose
Fix a regression in the port-geometry refinement layer. The previous version wrote the outward-offset geometry center back into `PatchPortCandidate.point`. Because the existing CST handoff interprets `point` as the waveguide-port contact location, this pushed the red port plane away from the feed conductor and produced a disconnected port.

The detector now keeps `PatchPortCandidate.point` as the feed-terminal contact point. The outward-offset rectangle remains available only in `debug_metadata["port_geometries"]` for visualization and future arbitrary-port support.

### Key thresholds
- No threshold changes.
- Existing geometry refinement still computes feed direction, feed width, outward center, and rectangle, but only `local_width` is used to refine the legacy CST-compatible port span.

### Validation samples
- `pytest -q test_patch_port_detection.py` passed 10 tests.
- Actual run_20260526_173906 dry check: selected port point remains `[213, 353]`, while `port_geometries[0].center` records the outward debug plane near `[215.8, 362.5]`.
- Actual run_20260526_190510 dry check: selected bottom contact point remains `[894, 899]`, avoiding the disconnected outward red block shown in the CST screenshot.

### Remaining risks
- The arbitrary port rectangle is metadata-only until a future CST builder explicitly supports non-axis-aligned excitation rectangles.

## 2026-05-27
### Changed files
- Rebuild/port_topology_detector.py
- test_patch_port_detection.py
- logs/algorithm_updates.md
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt

### Purpose
Fix the latest test47/test48 port-placement failures where skeleton endpoints could place the CST port below a conductor branch instead of on the actual feed terminal. The detector now adds terminal-face candidates on the active conductor bbox. These candidates represent a continuous conductor face near the image/bbox edge, such as the right-side feed pad terminal in test47.

### Key thresholds
- Terminal face is considered only when the active bbox side is within `max(border_distance_px, 36)` px of the image edge.
- Terminal face run length must be at least 6 px and no more than 24% of that bbox side length.
- Terminal face candidates score above ordinary skeleton endpoint candidates, but only when they are on the same connected main conductor label.
- Legacy `local_width` is now the estimated physical feed/terminal width, not the padded debug rectangle width.

### Validation samples
- `pytest -q test_patch_port_detection.py test_graph_local_parameterizer.py` passed 15 tests.
- Actual run_20260526_211823 dry check: selected port is now `(1029, 850)`, `direction=right`, `local_width=100.0`, matching the right-lower terminal shown in `port_analysis.png`.
- Actual run_20260527_170402 dry check: selected test48 port remains bottom feed terminal `(205, 354)`, `direction=bottom`.

### Remaining risks
- The terminal-face rule is intentionally local to active bbox edges. It does not yet infer CPW gaps or multi-port systems.

## 2026-05-28
### Changed files
- Rebuild/port_topology_detector.py
- Rebuild/port_em_validator.py
- logs/algorithm_updates.md
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt

### Purpose
Extend patch-port detection from image-border endpoint scoring toward EM-aware feed inference while keeping the existing pipeline and CST reconstruction untouched. The detector now traces a feed branch inward from each endpoint/terminal-face candidate, samples a width profile, estimates aspect ratio, tortuosity, curvature, and feed-to-patch width transition, then applies soft EM scoring on top of the existing conservative score.

The selected `PatchPortCandidate.point` remains the CST-compatible metal feed contact point. Outward excitation-plane information stays in `port_geometries` metadata only.

### Key thresholds
- Branch trace stops at junctions, unstable curvature, or width expansion above `2.5 * current_width`.
- Width stability target: `width_std / mean_width < 0.35`.
- Width range target: `max_width / min_width < 2.0`.
- Feed aspect ratio: `> 3.0` positive, `< 1.5` penalized.
- Tortuosity: `< 1.15` positive, `> 1.4` penalized.
- Border evidence remains available but is secondary to branch/feed EM quality.

### Validation samples
- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\Rebuild\port_topology_detector.py .\Rebuild\port_geometry_builder.py .\Rebuild\port_em_validator.py .\test_patch_port_detection.py` passed.
- `D:\Anaconda\envs\linefor\python.exe -m pytest -q test_patch_port_detection.py` passed 11 tests.
- Actual `run_20260527_223123` dry check on `test47/repair_fig.png` selected `(1029, 850)`, `direction=right`, `local_width=100.0`; contact point stays on the right-lower feed terminal and outward center is metadata-only.
- Actual `run_20260527_222906` dry check on `test48/repair_fig.png` selected `(205, 354)`, `direction=bottom`, `local_width=19.0`.

### Remaining risks
- Feed-to-patch junction detection is still a width-profile heuristic and can miss smooth tapers or anti-aliased transitions.
- CPW ground-gap semantics, multi-port disambiguation, and arbitrary angled CST rectangle consumption remain future work.

## 2026-05-28
### Changed files
- Rebuild/PortSearch.py
- Rebuild/port_topology_detector.py
- Rebuild/port_geometry_builder.py
- test_patch_port_detection.py
- logs/algorithm_updates.md
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt

### Purpose
Fix port placement on screenshots where the image canvas extends beyond the actual dielectric/substrate outline. Port detection now uses the first large subject contour from `SubjectEdgeAnalyzer` as an optional valid design region when that contour is clearly substrate-like. Candidate border reasoning, terminal-face scoring, debug overlays, and CST port geometry refinement are constrained to this first-layer contour.

When the outward excitation plane would leave the valid region, the geometry builder clips it back to the contour boundary. If the clipped point still touches the conductor mask, the exported `PatchPortCandidate.point` / `cst_contact_point` is also snapped to that contour-boundary contact point. The original raw skeleton/terminal point is preserved as `raw_endpoint`.

### Key thresholds
- Valid design region is enabled only when the first subject bbox covers at least 35% of the image and is near all canvas sides within 12% of image size.
- Candidate points may be within 2 px of the valid-region mask to absorb raster contour quantization.
- Geometry clipping is only active when `valid_region_mask` is provided; legacy no-region behavior remains unchanged.

### Validation samples
- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\Rebuild\port_topology_detector.py .\Rebuild\port_geometry_builder.py .\Rebuild\PortSearch.py .\test_patch_port_detection.py` passed.
- `D:\Anaconda\envs\linefor\python.exe -m pytest -q test_patch_port_detection.py` passed 12 tests.
- Actual `run_20260528_161404` dry check moved the bottom contact from raw `(205, 354)` to valid contour contact `(205, 353)`; `port_geometries[0].center` is also `[205.0, 353.0]` with `valid_region_limited=true`.
- Actual `run_20260528_160542` and `run_20260527_223123` right-lower feed-pad cases still select `(1029, 850)`, `direction=right`.

### Remaining risks
- The valid-region heuristic assumes the first large contour is substrate-like. Unusual non-convex substrate outlines may still need a more precise board-contour model.
- Current CST handoff still consumes the legacy axis-aligned `point/direction/local_width` fields, not the arbitrary rectangle metadata.

## 2026-05-28
### Changed files
- parameterized_json_to_cst.py
- test_patch_port_detection.py
- logs/algorithm_updates.md
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt

### Purpose
Diagnose and fix the gap between the visually correct port detector overlay and the CST model. The port detector was integrated correctly: `port_summary.json` contained `(205, 353)` and the CST history used `set topology patch waveguide port`. The remaining gap came from the handoff between image-mask detection and reconstructed CST metal: the parameterized feed terminal ended near image y=351, while the detected mask contact was y=353. The topology port handoff now snaps the detected port point along the feed direction to the nearest reconstructed PEC terminal before generating the CST waveguide port.

The topology-port CST coordinates now use precise mapped values instead of integer floor/ceil expansion. This avoids turning a terminal plane around y=-14.36 mm into a coarse y=-15..-14 mm slab that visually appears below the feed.

### Key thresholds
- Snap transverse search half-width: `max(6 px, 0.8 * local_width_px)`.
- Snap propagation-axis gap: `max(12 px, 2.0 * local_width_px)`.
- Snapping is only applied to topology patch ports; legacy closest-edge fallback is unchanged.

### Validation samples
- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\parameterized_json_to_cst.py .\test_patch_port_detection.py` passed.
- `D:\Anaconda\envs\linefor\python.exe -m pytest -q test_patch_port_detection.py` passed 13 tests.
- Actual `run_20260528_171216` dry handoff: raw topology point `(205,353)` snaps to reconstructed feed terminal `(203.81,351.00)`, producing precise CST ranges `Xrange -0.980268613..0.683964964`, `Yrange -14.3649635..-14.3649635`.

### Remaining risks
- The snap uses reconstructed curve sample points. Very sparse or malformed parameterized feed geometry can still leave the port slightly misaligned.
- Arbitrary angled port rectangles remain metadata-only; the current CST command is still axis-aligned by topology direction.

## 2026-05-28
### Changed files
- fss_parameterized_cst_pipeline.py
- test_patch_port_detection.py
- logs/algorithm_updates.md
- TEST_FILES_OVERVIEW_EN.txt
- TEST_FILES_OVERVIEW_CN.txt

### Purpose
Merge the new patch-port detector output into the main pipeline as the single port summary artifact. New pipeline runs no longer emit the old nearest-edge `port_analysis.png` or old-style `port_summary.json`. Instead, the pipeline writes `patch_port_summary.json`, whose top-level fields expose `ports`, `selected_port`, `port_geometries`, and `debug_metadata`, while retaining a `patch_port_detection` block for existing CST handoff code.

### Key thresholds
- No detector threshold changes.
- The CST handoff receives the returned `patch_port_summary.json` path as its port summary input.

### Validation samples
- `D:\Anaconda\envs\linefor\python.exe -m py_compile .\fss_parameterized_cst_pipeline.py .\parameterized_json_to_cst.py .\test_patch_port_detection.py` passed.
- `D:\Anaconda\envs\linefor\python.exe -m pytest -q test_patch_port_detection.py` passed 13 tests.
- Dry check `_verify_patch_port_summary_only` writes `patch_port_summary.json`; `port_summary.json` and `port_analysis.png` are not created; selected test47 port remains `(1029,850)`, `direction=right`.

### Remaining risks
- External scripts that hard-code `port_summary.json` must be pointed to the new `patch_port_summary.json` path or use the pipeline metadata field.
