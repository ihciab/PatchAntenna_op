from __future__ import annotations

from typing import Any


class PortEMValidator:
    """Soft EM validator for microstrip-like patch feed candidates.

    这个 validator 不阻断 pipeline，只给出 warnings 和分数修正。
    端口检测仍保持保守输出，避免因为启发式误判导致没有端口。
    """

    def validate(self, analysis: dict[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        scores: dict[str, float] = {}

        mean_width = float(analysis.get("mean_width", 0.0) or 0.0)
        width_std = float(analysis.get("width_std", 0.0) or 0.0)
        min_width = float(analysis.get("min_width", 0.0) or 0.0)
        max_width = float(analysis.get("max_width", 0.0) or 0.0)
        aspect_ratio = float(analysis.get("feed_aspect_ratio", 0.0) or 0.0)
        tortuosity = float(analysis.get("branch_tortuosity", 99.0) or 99.0)
        branch_length = float(analysis.get("branch_length", 0.0) or 0.0)
        transition = bool(analysis.get("patch_transition_detected", False))

        width_cv = width_std / max(mean_width, 1e-6)
        width_ratio = max_width / max(min_width, 1e-6)

        scores["width_stability"] = 2.0 if width_cv < 0.35 else -2.0
        if width_cv >= 0.35:
            warnings.append("feed width profile is unstable")

        scores["width_range"] = 1.5 if width_ratio < 2.0 else -1.5
        if width_ratio >= 2.0:
            warnings.append("feed width range expands too much before patch junction")

        if aspect_ratio > 3.0:
            scores["aspect_ratio"] = 3.0
        elif aspect_ratio < 1.5:
            scores["aspect_ratio"] = -3.0
            warnings.append("candidate is blob-like; feed aspect ratio is too low")
        else:
            scores["aspect_ratio"] = 0.5

        if tortuosity < 1.15:
            scores["straightness"] = 2.0
        elif tortuosity > 1.4:
            scores["straightness"] = -2.5
            warnings.append("feed branch is too tortuous")
        else:
            scores["straightness"] = 0.5

        scores["patch_transition"] = 3.0 if transition else 0.0
        if not transition:
            warnings.append("feed-to-patch width transition was not detected")

        if branch_length < max(6.0, mean_width * 1.2):
            scores["minimum_stub_length"] = -2.0
            warnings.append("feed stub is very short")
        else:
            scores["minimum_stub_length"] = 1.0

        total = float(sum(scores.values()))
        em_valid = total >= 1.0 and aspect_ratio >= 1.0 and branch_length >= 3.0
        confidence_scale = max(0.2, min(1.0, 0.55 + total / 12.0))

        return {
            "em_valid": bool(em_valid),
            "warnings": warnings,
            "scores": {key: round(value, 3) for key, value in scores.items()},
            "total_score": round(total, 3),
            "confidence_scale": round(confidence_scale, 3),
            "recommended_port_box": {
                "recommended_port_width": round(max(mean_width, 1.0), 3),
                "recommended_air_margin": round(max(mean_width * 0.5, 2.0), 3),
                "recommended_port_height": round(max(mean_width * 1.5, 3.0), 3),
            },
        }
