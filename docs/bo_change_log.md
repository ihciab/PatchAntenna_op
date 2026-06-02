# BO Change Log

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
