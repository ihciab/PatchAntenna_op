from __future__ import annotations

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
