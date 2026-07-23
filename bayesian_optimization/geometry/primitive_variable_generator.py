from __future__ import annotations

"""Primitive-aware BO variable generation.

The generator emits one normal-offset variable for each non-port parameterized
line, plus explicit port variables for width expansion and small propagation
direction alignment. It intentionally avoids independent point dx/dy variables.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence


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
    max_dimensions: Optional[int] = None,
) -> List[PrimitiveDesignVariable]:
    """Generate primitive-aware BO variables.

    Input: primitive analysis dictionary and optional maximum BO dimensionality.
    Output: ordered list of physical design variables.
    Algorithm purpose: replace point001_dx/point001_dy with constrained
    line-normal translation variables for every parameterized line while
    preserving topology and manufacturable antenna shapes.
    """

    variables: List[PrimitiveDesignVariable] = []
    primitives = list(analysis.get("primitives", []) or [])
    scale = estimate_design_scale(analysis)

    holes = [
        primitive
        for primitive in primitives
        if primitive.get("type") == "HOLE" or primitive.get("role") == "SLOT"
    ]
    for primitive in holes:
        variables.extend(generate_hole_translation_variables(primitive, scale))
        if reached_dimension_limit(variables, max_dimensions):
            return variables[:max_dimensions]

    lines = [
        primitive
        for primitive in primitives
        if primitive.get("type") == "LINE" and primitive.get("role") != "PORT"
    ]
    for primitive in lines:
        variables.append(generate_line_normal_offset_variable(primitive, scale))
        if reached_dimension_limit(variables, max_dimensions):
            return variables[:max_dimensions]

    ports = [
        primitive
        for primitive in primitives
        if primitive.get("type") == "LINE" and primitive.get("role") == "PORT"
    ]
    for primitive in ports:
        variables.extend(generate_port_variables(primitive, scale))
        if reached_dimension_limit(variables, max_dimensions):
            return variables[:max_dimensions]

    if max_dimensions is not None:
        return variables[:max_dimensions]
    return variables


def generate_hole_translation_variables(primitive: Dict[str, Any], scale: float) -> List[PrimitiveDesignVariable]:
    """Generate translation and size variables for a slot represented as a hole."""

    primitive_id = str(primitive.get("primitive_id"))
    prefix = safe_name(primitive)
    translate_limit = max(0.25, min(3.0, 0.04 * scale))
    bbox = primitive.get("bbox") or [0.0, 0.0, 1.0, 1.0]
    width = abs(float(bbox[2]) - float(bbox[0])) if len(bbox) >= 4 else max(1.0, 0.04 * scale)
    height = abs(float(bbox[3]) - float(bbox[1])) if len(bbox) >= 4 else max(1.0, 0.02 * scale)
    width_limit = max(0.10, min(3.0, 0.50 * width))
    height_limit = max(0.10, min(3.0, 0.50 * height))
    return [
        PrimitiveDesignVariable(
            name=f"{prefix}_slot_translate_x",
            lower=-translate_limit,
            upper=translate_limit,
            default=0.0,
            description="Rigid slot translation along x; slot width/height remain unchanged.",
            primitive_id=primitive_id,
            variable_type="hole_translate_x",
            target_role="SLOT",
        ),
        PrimitiveDesignVariable(
            name=f"{prefix}_slot_translate_y",
            lower=-translate_limit,
            upper=translate_limit,
            default=0.0,
            description="Rigid slot translation along y; slot width/height remain unchanged.",
            primitive_id=primitive_id,
            variable_type="hole_translate_y",
            target_role="SLOT",
        ),
        PrimitiveDesignVariable(
            name=f"{prefix}_slot_width_delta",
            lower=-width_limit,
            upper=width_limit,
            default=0.0,
            description="Symmetric slot width change along x; slot center remains fixed.",
            primitive_id=primitive_id,
            variable_type="hole_resize_width",
            target_role="SLOT",
        ),
        PrimitiveDesignVariable(
            name=f"{prefix}_slot_height_delta",
            lower=-height_limit,
            upper=height_limit,
            default=0.0,
            description="Symmetric slot height change along y; slot center remains fixed.",
            primitive_id=primitive_id,
            variable_type="hole_resize_height",
            target_role="SLOT",
        ),
    ]


def reached_dimension_limit(
    variables: Sequence[PrimitiveDesignVariable],
    max_dimensions: Optional[int],
) -> bool:
    """Return whether the optional BO dimensionality cap has been reached.

    Input: generated variable list and optional maximum dimension count.
    Output: true only when a caller explicitly requested a positive cap.
    Algorithm purpose: keep legacy capped runs available while making the
    default policy optimize all parameterized line primitives.
    """

    return max_dimensions is not None and len(variables) >= max_dimensions


def generate_line_normal_offset_variable(primitive: Dict[str, Any], scale: float) -> PrimitiveDesignVariable:
    """Generate one normal-translation variable for a parameterized line.

    Input: one LINE primitive analysis record and global design scale.
    Output: line_normal_offset variable.
    Algorithm purpose: keep each line primitive straight by moving both
    endpoints together along the line normal.
    """

    primitive_id = str(primitive.get("primitive_id"))
    prefix = safe_name(primitive)
    return PrimitiveDesignVariable(
        name=f"{prefix}_normal_offset",
        lower=-range_fraction(0.018, primitive) * scale,
        upper=range_fraction(0.018, primitive) * scale,
        default=0.0,
        description="Rigid line normal offset; both endpoints move together along the line normal.",
        primitive_id=primitive_id,
        variable_type="line_normal_offset",
        target_role=str(primitive.get("role", "STRUCTURAL")),
    )


def generate_port_variables(primitive: Dict[str, Any], scale: float) -> List[PrimitiveDesignVariable]:
    """Generate explicit port geometry variables.

    Input: one PORT line primitive and global design scale.
    Output: width expansion and propagation-direction shift variables.
    Algorithm purpose: let BO increase port/feed contact area without allowing
    the port width to shrink below the detected baseline.
    """

    primitive_id = str(primitive.get("primitive_id"))
    prefix = safe_name(primitive)
    width = max(1.0, line_width(primitive))
    shift_limit = max(2.0, min(0.05 * width, 0.005 * scale))
    return [
        PrimitiveDesignVariable(
            name=f"{prefix}_port_width_delta",
            lower=0.0,
            upper=0.20 * width,
            default=0.0,
            description="Port width increase only; upper bound is +20% of the detected port width.",
            primitive_id=primitive_id,
            variable_type="port_width_delta",
            target_role="PORT",
        ),
        PrimitiveDesignVariable(
            name=f"{prefix}_port_propagation_shift",
            lower=-shift_limit,
            upper=shift_limit,
            default=0.0,
            description="Small port shift along the inferred propagation direction, equivalent to the port-line normal.",
            primitive_id=primitive_id,
            variable_type="port_propagation_shift",
            target_role="PORT",
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


def line_width(primitive: Dict[str, Any]) -> float:
    """Return a line primitive length from analysis points."""

    points = primitive.get("points") or []
    if len(points) < 2:
        return 1.0
    start = points[0]
    end = points[-1]
    return math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))


def safe_name(primitive: Dict[str, Any]) -> str:
    """Convert primitive id to an optimizer-safe variable prefix.

    Input: primitive analysis record.
    Output: sanitized string.
    Algorithm purpose: keep variable names readable and stable for Optuna/skopt.
    """

    raw = str(primitive.get("primitive_id", "primitive"))
    return raw.replace("-", "_").replace("|", "_").replace(".", "_")
