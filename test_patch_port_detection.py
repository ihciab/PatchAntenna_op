from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from Rebuild.PortSearch import SubjectEdgeAnalyzer
from Rebuild.port_topology_detector import PatchPortTopologyDetector
from parameterized_json_to_cst import CSTParametricConfig, ParameterizedJsonCSTBuilder
from fss_parameterized_cst_pipeline import FSSParameterizedCSTPipeline


def _blank_mask(size: tuple[int, int] = (120, 160)) -> np.ndarray:
    height, width = size
    return np.zeros((height, width), dtype=np.uint8)


def _simple_microstrip_patch() -> np.ndarray:
    mask = _blank_mask()
    cv2.rectangle(mask, (64, 34), (130, 86), 255, -1)
    cv2.rectangle(mask, (0, 56), (64, 66), 255, -1)
    return mask


def _inset_feed_patch() -> np.ndarray:
    mask = _blank_mask()
    cv2.rectangle(mask, (48, 24), (116, 84), 255, -1)
    cv2.rectangle(mask, (76, 84), (88, 119), 255, -1)
    cv2.rectangle(mask, (66, 84), (75, 106), 0, -1)
    cv2.rectangle(mask, (89, 84), (98, 106), 0, -1)
    return mask


def test_simple_microstrip_patch_detects_left_border_port() -> None:
    detector = PatchPortTopologyDetector(border_distance_px=8, min_component_area=100)

    result = detector.detect_ports(subject_mask=_simple_microstrip_patch())

    assert result.ports
    best = result.ports[0]
    assert best.direction == "left"
    assert best.touches_border
    assert best.connected_to_main_patch
    assert best.score >= 10.0
    assert result.endpoint_mask.dtype == np.uint8
    assert int(np.count_nonzero(result.endpoint_mask)) >= 1


def test_inset_feed_patch_detects_bottom_border_port() -> None:
    detector = PatchPortTopologyDetector(border_distance_px=8, min_component_area=100)

    result = detector.detect_ports(subject_mask=_inset_feed_patch())

    assert result.ports
    best = result.ports[0]
    assert best.direction == "bottom"
    assert best.touches_border
    assert best.connected_to_main_patch


def test_noisy_screenshot_keeps_connected_feed_as_port() -> None:
    mask = _simple_microstrip_patch()
    cv2.rectangle(mask, (4, 4), (8, 8), 255, -1)
    cv2.rectangle(mask, (148, 15), (153, 20), 255, -1)
    cv2.circle(mask, (36, 104), 3, 255, -1)

    detector = PatchPortTopologyDetector(border_distance_px=8, min_component_area=100)
    result = detector.detect_ports(subject_mask=mask)

    assert result.ports
    assert result.ports[0].direction == "left"
    assert all(port.connected_to_main_patch for port in result.ports)


def test_border_frame_image_produces_no_false_ports() -> None:
    mask = _blank_mask()
    cv2.rectangle(mask, (0, 0), (159, 119), 255, 4)

    detector = PatchPortTopologyDetector(border_distance_px=8, min_component_area=100)
    result = detector.detect_ports(subject_mask=mask)

    assert result.ports == []
    assert result.debug_metadata["main_patch"]["looks_like_frame"]


def test_subject_edge_analyzer_optional_method_writes_debug(tmp_path: Path) -> None:
    mask = _simple_microstrip_patch()
    image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    analyzer = SubjectEdgeAnalyzer(min_component_area=100)
    analyzed = analyzer.analyze(image, subject_color="white")

    debug_dir = tmp_path / "03_port_detection"
    result = analyzer.detect_patch_ports(analyzed, debug_dir=debug_dir)

    assert result.ports
    assert (debug_dir / "skeleton.png").exists()
    assert (debug_dir / "endpoints.png").exists()
    assert (debug_dir / "selected_ports.png").exists()
    assert (debug_dir / "port_debug.json").exists()


def test_main_pipeline_port_summary_includes_patch_port_stage(tmp_path: Path) -> None:
    image_path = tmp_path / "patch.png"
    cv2.imwrite(str(image_path), _simple_microstrip_patch())

    pipeline = FSSParameterizedCSTPipeline(
        instance_json=None,
        output_root=tmp_path,
        run_name="pipeline_patch_port",
        build_only=True,
        inline_instance={
            "Folder_path": str(tmp_path),
            "Instance": "Microstrip_Antenna",
            "FSS_package": {"X": 36, "Y": 36, "f0": 6, "f1": 14},
            "layers": {
                "layer0": {
                    "img_path": str(image_path),
                    "substrate": 0.6,
                    "gnd": True,
                    "col_mats": {"white": "PEC"},
                }
            },
        },
    )
    pipeline._prepare_dirs()

    summary_path = pipeline._create_port_summary(
        image_path,
        {"col_mats": {"white": "PEC"}},
    )

    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_path.name == "patch_port_summary.json"
    assert summary["schema_version"] == "patch_port_summary_v1"
    assert "closest_edge" not in summary
    assert summary["patch_port_detection"]["enabled"]
    assert summary["patch_port_detection"]["ports"]
    assert summary["patch_port_detection"]["ports"][0]["direction"] == "left"
    assert summary["selected_port"]["direction"] == "left"
    assert (pipeline.run_dir / "03_port_detection" / "selected_ports.png").exists()
    assert not (pipeline.run_dir / "port_analysis.png").exists()


def test_patch_port_fallback_prefers_low_saturation_bottom_feed(tmp_path: Path) -> None:
    image = np.full((180, 220, 3), (220, 210, 30), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (209, 169), (255, 255, 255), -1)
    cv2.rectangle(image, (12, 12), (207, 167), (220, 210, 30), -1)
    cv2.rectangle(image, (65, 55), (155, 115), (210, 225, 238), -1)
    cv2.rectangle(image, (97, 115), (123, 167), (210, 225, 238), -1)
    cv2.rectangle(image, (82, 105), (91, 138), (220, 210, 30), -1)
    cv2.rectangle(image, (129, 105), (138, 138), (220, 210, 30), -1)

    analyzer = SubjectEdgeAnalyzer(min_component_area=300, approx_epsilon_ratio=0.0025)
    analyzed = analyzer.analyze(image, subject_color="gray")
    result = analyzer.detect_patch_ports(analyzed, debug_dir=tmp_path / "fallback_debug")

    assert result.ports
    assert result.ports[0].direction == "bottom"
    assert result.ports[0].connected_to_main_patch
    assert result.debug_metadata["fallback_from_subject_mask"]


def test_port_geometry_builder_places_plane_outside_feed_terminal() -> None:
    mask = _inset_feed_patch()
    detector = PatchPortTopologyDetector(border_distance_px=8, min_component_area=50)
    result = detector.detect_ports(subject_mask=mask)

    assert result.ports
    port = result.ports[0]
    geometry = result.debug_metadata["port_geometries"][0]

    assert port.direction == "bottom"
    assert geometry["refined"] is True
    assert port.point == tuple(geometry["endpoint"])
    assert geometry["center"][1] > geometry["endpoint"][1]
    assert geometry["cst_contact_point"] == geometry["endpoint"]
    assert geometry["feed_width"] > 1.0
    assert geometry["port_width"] >= geometry["feed_width"]
    assert port.local_width == pytest.approx(geometry["feed_width"], abs=0.01)


def test_valid_region_limits_port_plane_to_first_layer_contour() -> None:
    mask = _blank_mask((120, 160))
    cv2.rectangle(mask, (55, 30), (105, 64), 255, -1)
    cv2.rectangle(mask, (78, 64), (87, 90), 255, -1)

    valid_region = _blank_mask((120, 160))
    cv2.rectangle(valid_region, (8, 8), (151, 90), 255, -1)

    detector = PatchPortTopologyDetector(border_distance_px=8, min_component_area=50)
    result = detector.detect_ports(subject_mask=mask, valid_region_mask=valid_region)

    assert result.ports
    geometry = result.debug_metadata["port_geometries"][0]
    assert result.ports[0].direction == "bottom"
    assert geometry["valid_region_limited"] is True
    assert geometry["center"][1] <= geometry["endpoint"][1]
    assert result.debug_metadata["valid_region"]["bbox"][3] == 83


def test_port_geometry_builder_estimates_tangent_from_skeleton_path() -> None:
    mask = _blank_mask((90, 90))
    cv2.line(mask, (20, 80), (45, 55), 255, 9)
    detector = PatchPortTopologyDetector(border_distance_px=12, min_component_area=30)
    result = detector.detect_ports(subject_mask=mask)

    assert result.ports
    geometry = result.debug_metadata["port_geometries"][0]
    direction = np.array(geometry["feed_direction"], dtype=float)
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=0.1)
    assert geometry["confidence"] > 0.4


def test_right_side_feed_pad_terminal_face_has_priority() -> None:
    mask = _blank_mask((180, 240))
    cv2.rectangle(mask, (30, 35), (150, 120), 255, -1)
    cv2.rectangle(mask, (85, 105), (95, 150), 255, -1)
    cv2.rectangle(mask, (95, 140), (180, 150), 255, -1)
    cv2.rectangle(mask, (180, 132), (232, 158), 255, -1)

    detector = PatchPortTopologyDetector(border_distance_px=8, min_component_area=80)
    result = detector.detect_ports(subject_mask=mask)

    assert result.ports
    best = result.ports[0]
    assert best.direction == "right"
    assert best.point[0] == 232
    assert 132 <= best.point[1] <= 158
    assert result.debug_metadata["terminal_face_candidate_count"] >= 1


def test_cst_builder_prefers_patch_topology_port_over_legacy_edge(tmp_path: Path) -> None:
    builder = ParameterizedJsonCSTBuilder.__new__(ParameterizedJsonCSTBuilder)
    builder.config = CSTParametricConfig(project_folder=tmp_path, run_solver=False)
    builder.modeler = type(
        "DummyModeler",
        (),
        {
            "__init__": lambda self: setattr(self, "history", []),
            "add_to_history": lambda self, name, command: self.history.append((name, command)),
        },
    )()

    ports = {
        "closest_edge": [[0.0, 0.0], [220.0, 0.0]],
        "closest_border_sides": ["top"],
        "patch_port_detection": {
            "ports": [
                {
                    "point": [110, 179],
                    "direction": "bottom",
                    "local_width": 18.0,
                    "connected_to_main_patch": True,
                    "score": 13.0,
                    "confidence": 1.0,
                }
            ]
        },
    }

    used = builder._add_patch_topology_port_if_available(ports, (0.0, 0.0, 220.0, 180.0))

    assert used
    assert builder.modeler.history
    name, command = builder.modeler.history[0]
    assert name == "set topology patch waveguide port"
    assert '.Orientation "ymin"' in command


def test_cst_builder_snaps_patch_port_to_reconstructed_feed_terminal(tmp_path: Path) -> None:
    builder = ParameterizedJsonCSTBuilder.__new__(ParameterizedJsonCSTBuilder)
    builder.config = CSTParametricConfig(project_folder=tmp_path, run_solver=False)
    builder.payload = {
        "components": [
            {
                "resampled_points": [
                    [100.0, 120.0],
                    [100.0, 150.0],
                    [105.0, 170.0],
                    [110.0, 170.0],
                    [115.0, 170.0],
                ]
            }
        ]
    }
    builder.modeler = type(
        "DummyModeler",
        (),
        {
            "__init__": lambda self: setattr(self, "history", []),
            "add_to_history": lambda self, name, command: self.history.append((name, command)),
        },
    )()

    ports = {
        "patch_port_detection": {
            "ports": [
                {
                    "point": [110, 179],
                    "direction": "bottom",
                    "local_width": 18.0,
                    "connected_to_main_patch": True,
                    "score": 13.0,
                    "confidence": 1.0,
                }
            ]
        },
    }

    used = builder._add_patch_topology_port_if_available(ports, (0.0, 0.0, 220.0, 180.0))

    assert used
    _, command = builder.modeler.history[0]
    assert '.Orientation "ymin"' in command
    assert '.Yrange "-13.0909091", "-13.0909091"' in command
