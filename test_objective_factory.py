from __future__ import annotations

from pathlib import Path

from bayesian_optimization.optimization.objective_factory import extract_objective_targets
from bayesian_optimization.optimization.optimization_objectives import ObjectiveWeights
from bayesian_optimization.optimization.s11_parser import S11Metrics


def test_zero_target_resonance_disables_resonance_loss() -> None:
    instance = {
        "Paper_Performance": {
            "Target_Resonances_GHz": [0.0],
            "Resonance_Count": 2,
            "Bandwidth_10dB": {
                "Start_GHz": 9.48,
                "End_GHz": 10.53,
                "Bandwidth_GHz": 1.05,
            },
        }
    }
    targets = extract_objective_targets(
        instance,
        fallback_target_frequency_ghz=10.0,
        fallback_target_s11_db=-10.0,
    )

    assert targets.resonance_ghz is None

    result = targets_profile_result(targets)

    assert result.normalized_errors["resonance"] is None
    assert result.weighted_terms["resonance"] == 0.0


def targets_profile_result(targets):
    from bayesian_optimization.optimization.objective_factory import WidebandPatchObjective

    metrics = S11Metrics(
        s11_path=Path("dummy_s11.csv"),
        resonant_frequency_ghz=10.0,
        minimum_s11_db=-18.0,
        s11_at_target_db=-12.0,
        bandwidth_ghz=1.0,
        bandwidth_start_ghz=9.5,
        bandwidth_end_ghz=10.5,
        point_count=101,
    )
    return WidebandPatchObjective(targets).evaluate(metrics, {}, ObjectiveWeights())
