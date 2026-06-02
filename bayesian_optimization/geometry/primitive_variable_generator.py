from __future__ import annotations

"""Primitive-aware BO variable generation.

The generator emits physical variables such as line offsets, feed length, slot
width, and spline bulge terms. It intentionally avoids independent point dx/dy
variables.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence, Tuple


Point = Tuple[float, float]
NON_FEED_PORT_RANGE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class PrimitiveDesignVariable:
    """A one-dimensional BO variable with physical geometry meaning."""

    name: str
    lower: float
    upper: float
    default: float
    description: str
    primitive_id: str
    variable_type: str
    target_role: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation.

        Input: self.
        Output: dictionary with optimizer bounds and primitive metadata.
        Algorithm purpose: preserve variable provenance in debug reports.
        """

        return asdict(self)


def generate_primitive_variables(
    analysis: Dict[str, Any],
    max_dimensions: int = 12,
) -> List[PrimitiveDesignVariable]:
    """Generate primitive-aware BO variables.

    Input: primitive analysis dictionary and maximum BO dimensionality.
    Output: ordered list of physical design variables.
    Algorithm purpose: replace point001_dx/point001_dy with constrained
    feature variables that preserve topology and manufacturable antenna shapes.
    """

    variables: List[PrimitiveDesignVariable] = []
    primitives = list(analysis.get("primitives", []) or [])
    scale = estimate_design_scale(analysis)

    feedlines = [primitive for primitive in primitives if primitive.get("role") == "FEEDLINE"]
    splines = [primitive for primitive in primitives if primitive.get("type") == "BSPLINE" and primitive.get("role") != "PORT"]
    lines = [primitive for primitive in primitives if primitive.get("type") == "LINE" and primitive.get("role") not in {"PORT", "FEEDLINE"}]
    curves = [primitive for primitive in primitives if primitive.get("type") in {"ARC", "CURVE"} and primitive.get("role") != "PORT"]

    for primitive in feedlines[:2]:
        variables.extend(generate_feedline_variables(primitive, scale))
        if len(variables) >= max_dimensions:
            return variables[:max_dimensions]

    primitive_by_id = {str(primitive.get("primitive_id")): primitive for primitive in primitives}
    parallel = detect_parallel_line_pairs(lines + feedlines, scale)
    for index, pair in enumerate(parallel[:2]):
        pair_fraction = parallel_spacing_fraction(pair, primitive_by_id)
        variables.append(
            PrimitiveDesignVariable(
                name=f"slot{index:03d}_width",
                lower=-pair_fraction * scale,
                upper=pair_fraction * scale,
                default=0.0,
                description="Parallel line spacing delta that changes slot/feed width while keeping both edges parallel.",
                primitive_id="|".join(pair),
                variable_type="parallel_line_spacing",
                target_role="SLOT_OR_FEED_WIDTH",
            )
        )
        if len(variables) >= max_dimensions:
            return variables[:max_dimensions]

    for primitive in splines[:4]:
        variables.extend(generate_spline_variables(primitive, scale))
        if len(variables) >= max_dimensions:
            return variables[:max_dimensions]

    for primitive in lines[:4]:
        variables.extend(generate_line_variables(primitive, scale))
        if len(variables) >= max_dimensions:
            return variables[:max_dimensions]

    for primitive in curves[:2]:
        variables.append(
            PrimitiveDesignVariable(
                name=f"{safe_name(primitive)}_smooth_offset",
                lower=-range_fraction(0.015, primitive) * scale,
                upper=range_fraction(0.015, primitive) * scale,
                default=0.0,
                description="Smooth normal offset for an arc/curve primitive.",
                primitive_id=str(primitive.get("primitive_id")),
                variable_type="curve_smooth_offset",
                target_role=str(primitive.get("role", "STRUCTURAL")),
            )
        )
        if len(variables) >= max_dimensions:
            return variables[:max_dimensions]

    if not variables and primitives:
        primitive = primitives[0]
        variables.append(
            PrimitiveDesignVariable(
                name=f"{safe_name(primitive)}_smooth_offset",
                lower=-range_fraction(0.01, primitive) * scale,
                upper=range_fraction(0.01, primitive) * scale,
                default=0.0,
                description="Fallback primitive-level smooth offset; no raw point dx/dy variables are generated.",
                primitive_id=str(primitive.get("primitive_id")),
                variable_type="primitive_smooth_offset",
                target_role=str(primitive.get("role", "STRUCTURAL")),
            )
        )
    return variables[:max_dimensions]


def generate_line_variables(primitive: Dict[str, Any], scale: float) -> List[PrimitiveDesignVariable]:
    """Generate rigid line variables.

    Input: one LINE primitive analysis record and global design scale.
    Output: line_normal_offset and line_length_delta variables.
    Algorithm purpose: keep line primitives straight by moving/extending the
    whole segment rather than drifting endpoints independently.
    """

    primitive_id = str(primitive.get("primitive_id"))
    prefix = safe_name(primitive)
    return [
        PrimitiveDesignVariable(
            name=f"{prefix}_offset",
            lower=-range_fraction(0.018, primitive) * scale,
            upper=range_fraction(0.018, primitive) * scale,
            default=0.0,
            description="Rigid line normal offset; both endpoints move together.",
            primitive_id=primitive_id,
            variable_type="line_normal_offset",
            target_role=str(primitive.get("role", "STRUCTURAL")),
        ),
        PrimitiveDesignVariable(
            name=f"{prefix}_length_delta",
            lower=-range_fraction(0.015, primitive) * scale,
            upper=range_fraction(0.015, primitive) * scale,
            default=0.0,
            description="Symmetric line length change along the line direction.",
            primitive_id=primitive_id,
            variable_type="line_length_delta",
            target_role=str(primitive.get("role", "STRUCTURAL")),
        ),
    ]


def range_fraction(base_fraction: float, primitive: Dict[str, Any]) -> float:
    """Scale BO bounds for non-feed, non-port primitive groups.

    Input: base range fraction and primitive analysis record.
    Output: adjusted range fraction.
    Algorithm purpose: slightly expand structural/resonant optimization space
    while leaving FEEDLINE and PORT constraints conservative.
    """

    if primitive.get("role") in {"FEEDLINE", "PORT"}:
        return float(base_fraction)
    return float(base_fraction) * NON_FEED_PORT_RANGE_MULTIPLIER


def parallel_spacing_fraction(pair: Sequence[str], primitive_by_id: Dict[str, Dict[str, Any]]) -> float:
    """Return spacing bounds for a parallel primitive pair.

    Input: primitive id pair and primitive lookup.
    Output: scale fraction for slot/feed spacing variable.
    Algorithm purpose: expand only non-feed, non-port parallel groups while
    keeping feedline width/contact constraints conservative.
    """

    pair_primitives = [primitive_by_id.get(str(primitive_id), {}) for primitive_id in pair]
    if any(primitive.get("role") in {"FEEDLINE", "PORT"} for primitive in pair_primitives):
        return 0.025
    return 0.025 * NON_FEED_PORT_RANGE_MULTIPLIER


def generate_feedline_variables(primitive: Dict[str, Any], scale: float) -> List[PrimitiveDesignVariable]:
    """Generate feed-region variables with port-safe semantics.

    Input: one FEEDLINE primitive analysis record and global design scale.
    Output: feed_length variable and optional feed smooth offset.
    Algorithm purpose: allow movement along feed propagation while protecting
    port width, contact area, and alignment.
    """

    primitive_id = str(primitive.get("primitive_id"))
    prefix = safe_name(primitive)
    return [
        PrimitiveDesignVariable(
            name="feed_length" if prefix.endswith("_s0") else f"{prefix}_feed_length",
            lower=-0.020 * scale,
            upper=0.020 * scale,
            default=0.0,
            description="Feedline length delta along propagation direction; port contact width is frozen.",
            primitive_id=primitive_id,
            variable_type="feed_length",
            target_role="FEEDLINE",
        )
    ]


def generate_spline_variables(primitive: Dict[str, Any], scale: float) -> List[PrimitiveDesignVariable]:
    """Generate smooth spline deformation variables.

    Input: one BSPLINE primitive analysis record and global design scale.
    Output: spline_bulge and spline_smooth_offset variables.
    Algorithm purpose: deform B-spline control cages with Gaussian influence
    instead of moving single control points.
    """

    primitive_id = str(primitive.get("primitive_id"))
    prefix = safe_name(primitive)
    return [
        PrimitiveDesignVariable(
            name=f"{prefix}_bulge",
            lower=-range_fraction(0.020, primitive) * scale,
            upper=range_fraction(0.020, primitive) * scale,
            default=0.0,
            description="Gaussian smooth control-cage bulge around the spline center.",
            primitive_id=primitive_id,
            variable_type="spline_bulge",
            target_role=str(primitive.get("role", "RESONANT_SLOT")),
        ),
        PrimitiveDesignVariable(
            name=f"{prefix}_smooth_offset",
            lower=-range_fraction(0.015, primitive) * scale,
            upper=range_fraction(0.015, primitive) * scale,
            default=0.0,
            description="Distributed spline normal offset with frozen endpoints.",
            primitive_id=primitive_id,
            variable_type="spline_smooth_offset",
            target_role=str(primitive.get("role", "RESONANT_SLOT")),
        ),
    ]


def detect_parallel_line_pairs(primitives: Sequence[Dict[str, Any]], scale: float) -> List[Tuple[str, str]]:
    """Detect nearby parallel line pairs.

    Input: primitive records and global design scale.
    Output: primitive id pairs.
    Algorithm purpose: create slot/feed width variables that keep paired edges
    parallel during mutation.
    """

    pairs: List[Tuple[str, str]] = []
    lines = [primitive for primitive in primitives if primitive.get("type") == "LINE" and len(primitive.get("points", [])) >= 2]
    for left_index, left in enumerate(lines):
        for right in lines[left_index + 1 :]:
            left_vec = unit(vector(left["points"][0], left["points"][-1]))
            right_vec = unit(vector(right["points"][0], right["points"][-1]))
            parallel = abs(dot(left_vec, right_vec)) > 0.96
            spacing = point_line_distance(tuple(right["points"][0]), tuple(left["points"][0]), tuple(left["points"][-1]))
            if parallel and 0.004 * scale <= spacing <= 0.08 * scale:
                pairs.append((str(left.get("primitive_id")), str(right.get("primitive_id"))))
    return pairs


def estimate_design_scale(analysis: Dict[str, Any]) -> float:
    """Estimate a robust design scale from the analysis bbox.

    Input: primitive analysis dictionary.
    Output: positive scalar length.
    Algorithm purpose: choose BO bounds proportional to antenna geometry size.
    """

    bbox = analysis.get("summary", {}).get("bbox") or [0.0, 0.0, 100.0, 100.0]
    width = abs(float(bbox[2]) - float(bbox[0])) if len(bbox) >= 4 else 100.0
    height = abs(float(bbox[3]) - float(bbox[1])) if len(bbox) >= 4 else 100.0
    return max(10.0, math.hypot(width, height))


def safe_name(primitive: Dict[str, Any]) -> str:
    """Convert primitive id to an optimizer-safe variable prefix.

    Input: primitive analysis record.
    Output: sanitized string.
    Algorithm purpose: keep variable names readable and stable for Optuna/skopt.
    """

    raw = str(primitive.get("primitive_id", "primitive"))
    return raw.replace("-", "_").replace("|", "_").replace(".", "_")


def vector(a: Sequence[float], b: Sequence[float]) -> Point:
    """Calculate vector b - a.

    Input: two coordinate sequences.
    Output: 2D vector.
    Algorithm purpose: support parallel line detection and mutation directions.
    """

    return float(b[0]) - float(a[0]), float(b[1]) - float(a[1])


def unit(v: Point) -> Point:
    """Normalize a vector.

    Input: 2D vector.
    Output: unit vector or zero vector.
    Algorithm purpose: avoid repeated normalization code in geometry helpers.
    """

    length = math.hypot(v[0], v[1])
    if length <= 1e-12:
        return 0.0, 0.0
    return v[0] / length, v[1] / length


def dot(a: Point, b: Point) -> float:
    """Calculate dot product.

    Input: two 2D vectors.
    Output: scalar dot product.
    Algorithm purpose: compare line direction similarity.
    """

    return a[0] * b[0] + a[1] * b[1]


def point_line_distance(point: Point, start: Point, end: Point) -> float:
    """Calculate distance from a point to an infinite line.

    Input: point and line endpoints.
    Output: perpendicular distance.
    Algorithm purpose: estimate spacing for parallel line pairs.
    """

    direction = vector(start, end)
    length = math.hypot(direction[0], direction[1])
    if length <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    return abs((point[0] - start[0]) * direction[1] - (point[1] - start[1]) * direction[0]) / length
