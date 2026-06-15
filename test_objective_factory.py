from __future__ import annotations

from pathlib import Path

import pytest

from bayesian_optimization.optimization.objective_factory import extract_objective_targets
from bayesian_optimization.optimization.optimization_objectives import ObjectiveWeights
from bayesian_optimization.optimization.s11_parser import S11Metrics, parse_s11_file


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
    assert result.normalized_errors["resonance_count"] == 1.0
    assert result.weighted_terms["resonance_count"] == 8.0


def test_two_target_resonances_use_average_frequency_loss() -> None:
    instance = {
        "Paper_Performance": {
            "Target_Resonances_GHz": [9.5, 10.5],
            "Target_S11_dB": -10.0,
        }
    }
    targets = extract_objective_targets(
        instance,
        fallback_target_frequency_ghz=10.0,
        fallback_target_s11_db=-10.0,
    )

    result = targets_profile_result(
        targets,
        resonant_frequencies_ghz=(9.4, 10.8),
    )

    expected = ((abs(9.4 - 9.5) / 9.5) + (abs(10.8 - 10.5) / 10.5)) / 2.0
    assert targets.resonance_frequencies_ghz == (9.5, 10.5)
    assert result.normalized_errors["resonance"] == pytest.approx(expected)
    assert result.weighted_terms["resonance"] == pytest.approx(4.0 * expected)
    assert result.normalized_errors["resonance_count"] == 0.0
    assert result.weighted_terms["resonance_count"] == 0.0
    assert result.actuals["resonance_frequencies_ghz"] == [9.4, 10.8]
    assert result.actuals["resonance_count"] == 2


def test_missing_second_resonance_gets_modal_count_penalty() -> None:
    instance = {
        "Paper_Performance": {
            "Target_Resonances_GHz": [9.5, 10.5],
            "Target_S11_dB": -10.0,
        }
    }
    targets = extract_objective_targets(
        instance,
        fallback_target_frequency_ghz=10.0,
        fallback_target_s11_db=-10.0,
    )

    result = targets_profile_result(
        targets,
        resonant_frequencies_ghz=(9.5,),
    )

    assert targets.resonance_count == 2
    assert result.normalized_errors["resonance_count"] == 1.0
    assert result.weighted_terms["resonance_count"] == 8.0
    assert result.actuals["qualified_resonance_frequencies_ghz"] == [9.5]


def test_parse_s11_file_reports_two_local_resonances() -> None:
    s11_path = Path("test_s11_dual_resonance.csv")
    try:
        s11_path.write_text(
            "frequency,s11\n"
            "8.0,-3.0\n"
            "9.0,-14.0\n"
            "9.5,-6.0\n"
            "10.0,-8.0\n"
            "10.6,-22.0\n"
            "11.2,-7.0\n"
            "12.0,-4.0\n",
            encoding="utf-8",
        )

        metrics = parse_s11_file(s11_path, target_frequency_ghz=10.0)

        assert metrics.resonant_frequency_ghz == 10.6
        assert metrics.minimum_s11_db == -22.0
        assert metrics.resonant_frequencies_ghz == (9.0, 10.6)
        assert metrics.to_dict()["resonant_frequencies_ghz"] == (9.0, 10.6)
    finally:
        if s11_path.exists():
            s11_path.unlink()


def test_parse_s11_file_ignores_shallow_second_dip_for_modal_count() -> None:
    s11_path = Path("test_s11_single_deep_resonance.csv")
    try:
        s11_path.write_text(
            "frequency,s11\n"
            "8.0,-2.0\n"
            "9.0,-8.0\n"
            "9.5,-5.0\n"
            "10.0,-32.0\n"
            "10.5,-6.0\n"
            "11.0,-9.0\n"
            "12.0,-3.0\n",
            encoding="utf-8",
        )

        metrics = parse_s11_file(s11_path, target_frequency_ghz=10.0)

        assert metrics.resonant_frequency_ghz == 10.0
        assert metrics.minimum_s11_db == -32.0
        assert metrics.resonant_frequencies_ghz == (10.0,)
    finally:
        if s11_path.exists():
            s11_path.unlink()


def test_parse_s11_file_uses_widest_contiguous_10db_band() -> None:
    s11_path = Path("test_s11_split_bands.csv")
    try:
        s11_path.write_text(
            "frequency,s11\n"
            "8.0,-3.0\n"
            "9.0,-14.0\n"
            "9.1,-13.0\n"
            "9.2,-5.0\n"
            "10.0,-12.0\n"
            "10.3,-11.0\n"
            "10.6,-10.5\n"
            "10.9,-4.0\n",
            encoding="utf-8",
        )

        metrics = parse_s11_file(s11_path, target_frequency_ghz=10.0)

        assert metrics.bandwidth_start_ghz == 10.0
        assert metrics.bandwidth_end_ghz == 10.6
        assert metrics.bandwidth_ghz == pytest.approx(0.6)
    finally:
        if s11_path.exists():
            s11_path.unlink()


def test_bandwidth_loss_includes_width_shortfall() -> None:
    instance = {
        "Paper_Performance": {
            "Target_Resonances_GHz": [9.88, 10.12],
            "Bandwidth_10dB": {
                "Start_GHz": 9.821,
                "End_GHz": 10.162,
                "Bandwidth_GHz": 0.341,
            },
            "Target_S11_dB": -10.0,
        }
    }
    targets = extract_objective_targets(
        instance,
        fallback_target_frequency_ghz=10.0,
        fallback_target_s11_db=-10.0,
    )

    result = targets_profile_result(
        targets,
        resonant_frequencies_ghz=(9.88, 10.12),
        bandwidth_ghz=0.100,
        bandwidth_start_ghz=9.880,
        bandwidth_end_ghz=9.980,
    )

    expected_edges = abs(9.880 - 9.821) / 9.821 + abs(9.980 - 10.162) / 10.162
    expected_shortfall = (0.341 - 0.100) / 0.341
    assert result.normalized_errors["bandwidth_edges"] == pytest.approx(expected_edges)
    assert result.normalized_errors["bandwidth_width_shortfall"] == pytest.approx(expected_shortfall)
    assert result.normalized_errors["bandwidth"] == pytest.approx(expected_edges + expected_shortfall)
    assert result.weighted_terms["bandwidth"] == pytest.approx(8.0 * (expected_edges + expected_shortfall))


def targets_profile_result(
    targets,
    resonant_frequencies_ghz=(),
    bandwidth_ghz=1.0,
    bandwidth_start_ghz=9.5,
    bandwidth_end_ghz=10.5,
):
    from bayesian_optimization.optimization.objective_factory import WidebandPatchObjective

    metrics = S11Metrics(
        s11_path=Path("dummy_s11.csv"),
        resonant_frequency_ghz=10.0,
        minimum_s11_db=-18.0,
        s11_at_target_db=-12.0,
        bandwidth_ghz=bandwidth_ghz,
        bandwidth_start_ghz=bandwidth_start_ghz,
        bandwidth_end_ghz=bandwidth_end_ghz,
        point_count=101,
        resonant_frequencies_ghz=tuple(resonant_frequencies_ghz),
    )
    return WidebandPatchObjective(targets).evaluate(metrics, {}, ObjectiveWeights())
