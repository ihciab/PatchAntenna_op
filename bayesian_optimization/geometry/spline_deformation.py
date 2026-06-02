from __future__ import annotations

"""Smooth B-spline control cage deformation utilities."""

import math
from typing import Dict, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


def calculate_control_weights(
    control_points: Sequence[Sequence[float]],
    center_index: int,
    sigma: Optional[float] = None,
    frozen_indices: Optional[Sequence[int]] = None,
) -> List[float]:
    """Calculate Gaussian influence weights for a spline control cage.

    Input: control points, selected center control index, optional sigma, and
    optional frozen point indexes.
    Output: one influence weight per control point.
    Algorithm purpose: distribute a center movement over neighboring controls
    with exp(-distance^2 / sigma^2) so the spline remains smooth.
    """

    points = [to_point(point) for point in control_points]
    if not points:
        return []
    center_index = max(0, min(int(center_index), len(points) - 1))
    frozen = set(int(index) for index in (frozen_indices or []))
    if sigma is None:
        sigma = estimate_sigma(points)
    sigma = max(float(sigma), 1e-9)
    center = points[center_index]
    weights: List[float] = []
    for index, point in enumerate(points):
        if index in frozen:
            weights.append(0.0)
            continue
        distance = math.hypot(point[0] - center[0], point[1] - center[1])
        weights.append(math.exp(-(distance * distance) / (sigma * sigma)))
    return weights


def apply_gaussian_deformation(
    points: Sequence[Sequence[float]],
    center_index: int,
    delta: Sequence[float],
    sigma: Optional[float] = None,
    frozen_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[List[float]], List[float]]:
    """Apply Gaussian deformation to a point cage.

    Input: point cage, center index, delta vector, optional sigma, and frozen indexes.
    Output: deformed points plus applied weights.
    Algorithm purpose: move the center 100 percent while neighbors receive
    Gaussian-weighted motion, preventing spikes from single-point motion.
    """

    weights = calculate_control_weights(points, center_index, sigma=sigma, frozen_indices=frozen_indices)
    dx = float(delta[0])
    dy = float(delta[1])
    deformed: List[List[float]] = []
    for point, weight in zip(points, weights):
        p = to_point(point)
        deformed.append([p[0] + weight * dx, p[1] + weight * dy])
    return deformed, weights


def smooth_control_cage(
    control_points: Sequence[Sequence[float]],
    center_index: Optional[int] = None,
    delta: Sequence[float] = (0.0, 0.0),
    sigma: Optional[float] = None,
    endpoint_mode: str = "freeze",
) -> Dict[str, object]:
    """Smoothly deform a B-spline control cage.

    Input: control points, optional center index, delta vector, sigma, and endpoint rule.
    Output: dictionary containing deformed points, weights, frozen endpoints,
    and point role labels.
    Algorithm purpose: implement smooth control cage deformation with frozen
    endpoints and limited handle movement to preserve C1-style continuity.
    """

    points = [to_point(point) for point in control_points]
    roles = classify_spline_points(points)
    if not points:
        return {"points": [], "weights": [], "roles": [], "frozen_indices": []}
    if center_index is None:
        internal = [index for index, role in enumerate(roles) if role == "INTERNAL_POINT"]
        center_index = internal[len(internal) // 2] if internal else len(points) // 2
    frozen_indices = [0, len(points) - 1] if endpoint_mode == "freeze" and len(points) >= 2 else []

    # Spline deformation core: Gaussian weights propagate the sampled BO delta
    # over the control cage instead of allowing a single control to form a spike.
    deformed, weights = apply_gaussian_deformation(
        points,
        int(center_index),
        delta,
        sigma=sigma,
        frozen_indices=frozen_indices,
    )

    for index, role in enumerate(roles):
        if role == "HANDLE_POINT" and index not in frozen_indices:
            original = points[index]
            deformed[index][0] = original[0] + 0.55 * (deformed[index][0] - original[0])
            deformed[index][1] = original[1] + 0.55 * (deformed[index][1] - original[1])
            weights[index] *= 0.55

    return {
        "points": deformed,
        "weights": weights,
        "roles": roles,
        "frozen_indices": frozen_indices,
        "center_index": int(center_index),
        "sigma": float(sigma if sigma is not None else estimate_sigma(points)),
    }


def classify_spline_points(control_points: Sequence[Sequence[float]]) -> List[str]:
    """Classify spline control points by endpoint/handle/internal role.

    Input: spline control points.
    Output: role label per point.
    Algorithm purpose: freeze endpoints, damp handles, and deform internal
    controls during primitive-aware mutation.
    """

    count = len(control_points)
    roles: List[str] = []
    for index in range(count):
        if index == 0 or index == count - 1:
            roles.append("ENDPOINT")
        elif index == 1 or index == count - 2:
            roles.append("HANDLE_POINT")
        else:
            roles.append("INTERNAL_POINT")
    return roles


def estimate_sigma(points: Sequence[Point]) -> float:
    """Estimate Gaussian sigma from control cage size.

    Input: control cage points.
    Output: positive sigma distance.
    Algorithm purpose: produce roughly center=100%, neighbor~50%, far~10%
    influence without requiring schema changes.
    """

    if len(points) < 2:
        return 1.0
    distances = [
        math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
        for index in range(1, len(points))
    ]
    mean_spacing = sum(distances) / max(1, len(distances))
    return max(mean_spacing * 2.0, 1e-6)


def to_point(value: Sequence[float]) -> Point:
    """Convert a coordinate sequence to a 2D point.

    Input: sequence containing x and y.
    Output: 2D float tuple.
    Algorithm purpose: normalize inputs for deformation math.
    """

    return float(value[0]), float(value[1])
