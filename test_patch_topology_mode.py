from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from geometry_driven_parameterizer import EDGE_REPRESENTATION_MODE, GeometryDrivenParameterizer


def _parameterizer(tmp_path: Path) -> GeometryDrivenParameterizer:
    return GeometryDrivenParameterizer(
        image_path=tmp_path / "dummy.png",
        save_dir=tmp_path,
        max_centerline_components_for_geometry=30,
    )


def test_patch_topology_mask_first_skeleton_preserves_feed(tmp_path: Path) -> None:
    image = np.full((160, 220, 3), (220, 210, 30), dtype=np.uint8)
    cv2.rectangle(image, (60, 38), (160, 96), (205, 220, 235), -1)
    cv2.rectangle(image, (104, 96), (116, 150), (205, 220, 235), -1)
    cv2.rectangle(image, (82, 86), (92, 118), (220, 210, 30), -1)
    cv2.rectangle(image, (128, 86), (138, 118), (220, 210, 30), -1)

    p = _parameterizer(tmp_path)
    result = p.build_patch_topology_vtracer_input(image, debug_dir=tmp_path / "patch_topology_debug")

    assert result["accepted"]
    skeleton = result["skeleton"]
    assert skeleton.dtype == np.uint8
    assert np.count_nonzero(skeleton[120:151, 104:117]) > 10
    assert result["metrics"]["mode"] == EDGE_REPRESENTATION_MODE
    assert result["metrics"]["skeleton_length"] > 0
    assert (tmp_path / "patch_topology_debug" / "01_pec_mask.png").exists()
    assert (tmp_path / "patch_topology_debug" / "05_pruned_skeleton.png").exists()
    assert (tmp_path / "patch_topology_debug" / "patch_topology_metrics.json").exists()


def test_patch_topology_removes_only_obvious_outer_substrate_component(tmp_path: Path) -> None:
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(mask, (0, 0), (159, 119), 255, -1)
    cv2.rectangle(mask, (50, 35), (110, 75), 255, -1)

    p = _parameterizer(tmp_path)
    cleaned, metrics = p.remove_outer_substrate_component(mask)

    assert metrics["removed_outer_frame"]
    assert np.count_nonzero(cleaned) == 0


def test_patch_topology_validation_rejects_empty_skeleton(tmp_path: Path) -> None:
    p = _parameterizer(tmp_path)
    validation = p.validate_topology_preservation(
        np.zeros((40, 40), dtype=np.uint8),
        np.zeros((40, 40), dtype=np.uint8),
    )

    assert not validation["accepted"]
    assert validation["fallback_reason"] == "empty_topology_mask_or_skeleton"


def test_patch_topology_metrics_gate_rejects_fragmented_candidate(tmp_path: Path) -> None:
    p = _parameterizer(tmp_path)
    metrics = {
        "component_count": 8,
        "endpoint_count": 10,
        "skeleton_length": 200,
        "pec_mask_pixels": 1000,
    }

    assert not p._patch_topology_metrics_sane(metrics)
