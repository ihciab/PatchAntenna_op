# BO Change Log

## 2026-07-11 00:00:00 +08:00

### Modified Files

- `beyesian_opconfig.json`
- `bayesian_optimization/pipelines/optimization_pipeline.py`
- `bayesian_optimization/optimization/multistage.py`
- `docs/bo_change_log.md`
- `bayesian_optimization/docs/bo_change_log.md`

### Summary

Added a root-level `beyesian_opconfig.json` file that centralizes important
Bayesian optimization settings: CST S11 frequency range, stage trial counts,
stage-specific loss weights, full objective weights, optimizer hyperparameters,
stopping criteria, geometry settings, and port-connection parameters.

The editor-run pipeline now loads this JSON and uses it to override the legacy
`EDITOR_RUN_CONFIG`. The configured S11 simulation range is applied directly to
the CST builder config, so `simulation.f0 = 9.4` and `simulation.f1 = 10.8`
drive the actual CST run instead of only metadata.

Stage objective weights are now configurable. Stage1/Stage2 can override
`ERES` and `EBW`, while Stage3/Stage4 can scale the full objective. Optuna
`n_startup_trials` and `multivariate`, plus skopt `base_estimator` and
`acq_func`, are also configurable.

## 2026-07-09 00:00:00 +08:00

### Modified Files

- `bayesian_optimization/optimization/multistage.py`
- `bayesian_optimization/geometry/primitive_mutator.py`
- `bayesian_optimization/pipelines/optimization_pipeline.py`
- `docs/bo_change_log.md`
- `bayesian_optimization/docs/bo_change_log.md`

### Reason

The existing Stage1-Stage3 flow can converge to a good local solution but may
still miss alternative current paths. Stage4 was added as a bounded local
escape phase after fine tuning, keeping electrical scale and port/feedline
conditions fixed while probing one conductor contour point per trial.

### Algorithm Changes

- Added `STAGE4_TOPOLOGY_EXPLORATION` to `OptimizationStage` and extended
  `StageManager` with `stage4_trials`, `stage4_delta_px`, and cyclic selected
  point scheduling.
- Stage4 uses only `stage4_delta_x` and `stage4_delta_y` in the range
  `[-stage4_delta_px, +stage4_delta_px]`. The selected point index is injected
  by the stage manager instead of being optimized by Optuna.
- Stage4 starts from the best successful Stage3 payload and applies each trial
  as a non-cumulative single-point move. If no Stage3 reference is available,
  it falls back to the current best record or the initial payload.
- Stage4 freezes `global_scale_x`, `global_scale_y`, port size, port position,
  feedline width, feedline position, and port connection adjustment.
- Stage4 disables automatic CST geometry repair. Invalid geometry, duplicate
  points, broken segments, and failed validation return the invalid-geometry
  penalty without CST build/simulation.
- Stage4 keeps the full Stage3 objective terms and writes explicit logs for the
  selected point, move vector, geometry validity, ERES, EBW, and loss.

### Compatibility

- Stage4 is active only when multi-stage optimization is enabled and
  `stage4_trials > 0`.
- Set `STAGE4_TRIALS = 0` or `--stage4-trials 0` to keep the earlier
  three-stage behavior.
- Existing CST modeling, parameterization, S11 parsing, and objective framework
  interfaces remain unchanged.

## 2026-06-18 00:00:00 +08:00

### Modified Files

- `bayesian_optimization/optimization/multistage.py`
- `bayesian_optimization/geometry/primitive_mutator.py`
- `bayesian_optimization/pipelines/optimization_pipeline.py`
- `bayesian_optimization/simulation/parameterized_json_to_cst.py`
- `docs/bo_change_log.md`

### Reason

The antenna BO loop can waste many expensive CST evaluations when the
parameterized conductor has an electrical scale drift. A multi-stage BO mode
was added to first lock global electrical scale, then optimize local shape, and
finally jointly fine tune both.

### Algorithm Changes

- Added `OptimizationStage` and `StageManager`.
- Added optional high-level variables: `global_scale_x`, `global_scale_y`, and
  `port_width_scale`.
- Stage 1 samples only the three high-level scale variables and feeds Optuna
  `1.0 * ERES + 1.0 * EBW`.
- Stage 2 fixes the Stage 1 best scales and samples the existing local geometry
  variables with `0.2 * ERES + 1.0 * EBW`.
- Stage 3 samples high-level scales in a +/-5% window around the Stage 1 best
  values together with all local variables, then feeds the existing full
  objective total to Optuna.
- Global scaling is applied only to parameterized conductor geometry fields in
  the JSON payload and leaves canvas/substrate/ground/airbox metadata unchanged.
- Per-evaluation temporary port summaries now receive `port_width_scale` so the
  CST waveguide port width follows the high-level feed-width variable.
- In multi-stage mode, each evaluation now writes
  `mutation_stage_curve_parameterization.json` before CST handoff/repair and
  `cst_input_curve_parameterization.json` after repair/port connection so the
  exact scaled parameterization and the exact CST input can be inspected.
- CST handoff now checks compact primitive endpoint continuity before using
  primitives for extrusion. If high-level port scaling creates primitive gaps,
  the builder falls back to the closed sampled polygon so the conductor solid is
  still generated in CST.

### Compatibility

- `enable_multistage_optimization` defaults to `False`, preserving the existing
  single-stage behavior.
- Existing CST build, parameterization, S11 parsing, and objective factory
  interfaces are unchanged.
- CLI users can enable the mode with `--enable-multistage-optimization`.

### Risk Analysis

- The new high-level scale variables can move conductor geometry closer to the
  canvas/substrate boundary; existing geometry validation still rejects unsafe
  samples before CST.
- Stage 1 and Stage 3 require enough evaluation budget. Validation rejects
  configurations where `stage1_trials + stage3_trials >= max_evaluations`.
- Stage 3 uses the Stage 1 best scale as its center. If Stage 1 never produces
  a finite non-failure result, the fallback center remains `1.0`.

## 2026-06-08 11:28:06 +08:00

### Modified Files

- `bayesian_optimization/pipelines/optimization_pipeline.py`
- `bayesian_optimization/simulation/parameterized_json_to_cst.py`
- `docs/bo_change_log.md`

### Reason

The BO pipeline needs the CST S-parameter simulation range to come from the
new-format instance JSON, specifically `Antenna_package.f0` and
`Antenna_package.f1`, instead of relying on stale prepared-instance files or
hard-coded defaults.

### Algorithm Changes

- Added `INSTANCE_JSON_PATH` to the editor-run BO configuration and pointed it
  to `pipeline_test_instance.json`.
- Updated editor config conversion so `INSTANCE_JSON_PATH` is used when present.
- Added explicit CST S11 frequency-range validation before each CST build.
- Added `cst_s11_frequency_range` to BO run metadata.
- Added builder-level frequency validation before `cst_auto_init(...)`.

### Compatibility

- CLI behavior is unchanged; `--instance-json` still controls CST settings.
- Existing prepared-instance workflows still work when `INSTANCE_JSON_PATH` is
  omitted.
- `load_instance_config(...)` continues to support both `Antenna_package` and
  legacy `FSS_package`.

### Risk Analysis

- Runs with invalid frequency ranges now fail early with a clear error instead
  of producing an ambiguous CST setup.
- Direct editor runs now prefer root `pipeline_test_instance.json`; users who
  want per-run prepared instances should change `INSTANCE_JSON_PATH` or remove
  it from `EDITOR_RUN_CONFIG`.

## 2026-06-02 21:06:15 +08:00

### Modified Files

- `bayesian_optimization/optimization/objective_factory.py`
- `bayesian_optimization/optimization/optimization_objectives.py`
- `bayesian_optimization/pipelines/optimization_pipeline.py`
- `docs/bo_change_log.md`

### Reason

The previous objective mixed losses with different physical units, such as GHz,
dB and dBi, and was closer to a free-design objective than a paper
reconstruction objective. The BO target now needs to compare simulated metrics
against the paper metrics stored in the new-format instance JSON.

### Algorithm Changes

- Added an objective factory that reads `Antenna_Type` and `Paper_Performance`
  from the instance JSON.
- Added `WidebandPatchObjective` as the active profile.
- Switched continuous objective terms to normalized relative errors:
  `abs(actual - target) / max(abs(target), epsilon)`.
- Continuous loss is now:
  `4.0 * E_res + 3.0 * E_bw + 2.0 * E_s11 + 1.0 * E_gain`.
- Preserved hard-failure penalties:
  `invalid_geometry = 100`, `cst_failure = 80`.
- Disabled geometry complexity, curvature, tiny segment and topology warning
  penalties by setting their default weights to zero.
- Added `normalized_errors` to objective breakdown output.

### Compatibility

- The public `evaluate_objective(...)` signature is unchanged.
- Existing pipeline call sites continue to pass `target_frequency_ghz` and
  `target_s11_db`; when `instance_json` is available, the pipeline overrides
  those values internally from the paper targets.
- If no instance JSON is provided, the objective falls back to the existing
  target frequency and S11 settings.
- Existing breakdown fields are retained; new fields are additive.

### Risk Analysis

- Gain is included structurally, but the current pipeline does not export a
  simulated gain metric. Until a gain exporter provides `gain_dbi`,
  `peak_gain_dbi` or `simulated_gain_dbi`, the gain term is skipped instead of
  fabricating a penalty.
- `MultibandPatchObjective`, `ArrayObjective` and `FSSObjective` are selected by
  name only for now and fall back to `WidebandPatchObjective`; dedicated
  implementations should be added before optimizing those designs seriously.
- S11 error uses `s11_at_target_db`, so it remains a scalar paper-metric
  objective and intentionally does not compare full S11 curves.
