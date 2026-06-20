from __future__ import annotations

import pytest

from bayesian_optimization.geometry.port_summary_utils import (
    ensure_port_summary_connected_to_geometry,
    normalize_port_side,
)
from bayesian_optimization.geometry.feature_shape_optimizer import detect_port_model
from bayesian_optimization.geometry.primitive_analyzer import analyze_primitives


def _left_biased_payload() -> dict:
    points = [
        [0, 0],
        [100, 0],
        [100, 100],
        [45, 100],
        [50, 100],
        [55, 100],
        [0, 60],
        [0, 55],
        [0, 50],
        [0, 45],
        [0, 40],
        [0, 0],
    ]
    return {
        "components": [
            {
                "resampled_points": points,
                "segments": [
                    {"type": "LINE", "start": [0, 40], "end": [0, 60]},
                    {"type": "LINE", "start": [45, 100], "end": [55, 100]},
                ],
            }
        ]
    }


def _bottom_port_summary() -> dict:
    return {
        "schema_version": "patch_port_summary_v1",
        "selected_port": {
            "point": [50, 100],
            "direction": "bottom",
            "local_width": 10,
        },
        "patch_port_detection": {
            "enabled": True,
            "ports": [
                {
                    "point": [50, 100],
                    "direction": "bottom",
                    "local_width": 10,
                }
            ],
        },
    }


def test_detect_port_model_prefers_port_summary_direction_over_point_statistics() -> None:
    model = detect_port_model(_left_biased_payload(), _bottom_port_summary())

    assert model.port_side == "bottom"
    assert model.axis == "vertical"
    assert model.propagation_direction == (0.0, -1.0)
    assert model.centerline == 50.0
    assert model.estimated_width == 10.0


def test_primitive_port_context_uses_port_summary_side() -> None:
    analysis = analyze_primitives(_left_biased_payload(), port_summary=_bottom_port_summary())
    context = analysis["summary"]["port_context"]

    assert context["side"] == "bottom"
    assert context["axis"] == "y"
    assert context["propagation_direction"] == [0.0, -1.0]
    assert analysis["primitives"][0]["role"] != "PORT"
    assert analysis["primitives"][1]["role"] == "PORT"


def test_normalize_port_side_accepts_bottom() -> None:
    assert normalize_port_side("bottom") == "bottom"


def test_port_summary_connection_shifts_bottom_port_inward_to_curve() -> None:
    payload = {
        "components": [
            {
                "closed": False,
                "resampled_points": [
                    [45.0, 96.0],
                    [55.0, 96.0],
                ],
            }
        ]
    }
    port_summary = _bottom_port_summary()

    connected, report = ensure_port_summary_connected_to_geometry(
        payload,
        port_summary,
        step_px=0.2,
        tolerance_px=0.05,
        max_shift_px=10.0,
    )

    assert report["status"] == "connected_by_inward_shift"
    assert report["connected_after"] is True
    assert report["shift_applied_px"] == pytest.approx(4.0)
    assert report["final_free_normal_shift_applied"] is False
    assert report["final_free_normal_deferred_to_cst_builder"] is True
    assert report["geometry_contact_point"] == pytest.approx([50.0, 96.0])
    assert connected["selected_port"]["point"] == pytest.approx([50.0, 96.0])
    assert connected["patch_port_detection"]["ports"][0]["point"] == pytest.approx([50.0, 96.0])
    assert connected["bo_port_connection_adjustment"]["final_free_normal_inward_px"] == pytest.approx(10.0)
    assert connected["bo_port_connection_adjustment"]["final_port_width_scale"] == pytest.approx(1.50)
    assert connected["selected_port"]["local_width"] == pytest.approx(15.0)
    assert connected["patch_port_detection"]["ports"][0]["local_width"] == pytest.approx(15.0)
    assert report["port_width_adjustment"]["scale"] == pytest.approx(1.50)
