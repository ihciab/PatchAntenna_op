"""Geometry validation rules for CadQuery-backed Geometry Engine models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from geometry_engine.boundary import BoundaryValidator
from geometry_engine.geometry.patch import Patch
from geometry_engine.geometry.slot import Slot


@dataclass
class ValidationResult:
    """Validation result containing all discovered geometry errors."""

    valid: bool
    errors: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise a validation error if this result is invalid."""

        if not self.valid:
            raise GeometryValidationError(self.errors)


class GeometryValidationError(ValueError):
    """Raised when geometry violates one or more engine rules."""

    def __init__(self, errors: List[str]) -> None:
        """Create an exception from validation error messages."""

        super().__init__("Geometry validation failed: " + "; ".join(errors))
        self.errors = errors


class GeometryValidator:
    """Validate patch semantics and extracted CadQuery boundary geometry."""

    tolerance: float
    boundary_validator: BoundaryValidator

    def __init__(self, tolerance: float = 1e-7) -> None:
        """Initialize the validator."""

        self.tolerance = float(tolerance)
        self.boundary_validator = BoundaryValidator(tolerance=tolerance)

    def validate(self, patch: Patch) -> ValidationResult:
        """Validate all currently supported geometry rules."""

        errors: List[str] = []
        self._validate_patch(patch, errors)
        self._validate_slots(patch, errors)
        self._validate_feed(patch, errors)
        self._validate_boundary(patch, errors)
        return ValidationResult(valid=not errors, errors=errors)

    def _validate_patch(self, patch: Patch, errors: List[str]) -> None:
        """Validate the closed rectangular PEC patch."""

        if not self._positive_finite(patch.length):
            errors.append(f"Patch length must be positive and finite, got {patch.length!r}.")
        if not self._positive_finite(patch.width):
            errors.append(f"Patch width must be positive and finite, got {patch.width!r}.")
        if patch.material.upper() != "PEC":
            errors.append(f"Patch must be the only PEC conductor, got material {patch.material!r}.")

        if not self._positive_finite(patch.thickness):
            errors.append(f"CadQuery construction thickness must be positive, got {patch.thickness!r}.")

    def _validate_slots(self, patch: Patch, errors: List[str]) -> None:
        """Validate all slots are bounded inside the patch."""

        seen_ids: set[str] = set()
        for slot in patch.slots:
            if slot.id in seen_ids:
                errors.append(f"Duplicate slot id: {slot.id!r}.")
            seen_ids.add(slot.id)
            self._validate_slot(patch, slot, errors)

    def _validate_slot(self, patch: Patch, slot: Slot, errors: List[str]) -> None:
        """Validate one slot."""

        if slot.shape not in {"rectangle", "circle"}:
            errors.append(f"Only rectangle and circle slots are supported, got {slot.shape!r} for {slot.id}.")
            return
        if not self._positive_finite(slot.width):
            errors.append(f"Slot {slot.id} width must be positive and finite, got {slot.width!r}.")
        if slot.shape == "rectangle" and not self._positive_finite(slot.height):
            errors.append(f"Slot {slot.id} height must be positive and finite, got {slot.height!r}.")
        # CadQuery performs the boolean difference. The exported boundary is
        # validated after extraction, so non-rectangular validity belongs there.
        if patch.shape_kind != "rectangle" or not patch.is_axis_aligned(self.tolerance):
            return
        if (
            slot.left <= patch.left + self.tolerance
            or slot.right >= patch.right - self.tolerance
            or slot.bottom <= patch.bottom + self.tolerance
            or slot.top >= patch.top - self.tolerance
        ):
            errors.append(
                f"Slot {slot.id} must be strictly inside patch bounds. "
                f"slot=({slot.left}, {slot.bottom}, {slot.right}, {slot.top}), "
                f"patch=({patch.left}, {patch.bottom}, {patch.right}, {patch.top})."
            )
        # TODO: Add slot overlap and minimum copper clearance checks once the
        # CST builder supports real boolean cutouts.

    def _validate_feed(self, patch: Patch, errors: List[str]) -> None:
        """Validate feed location and edge span."""

        feed = patch.feed
        if patch.shape_kind != "rectangle" or not patch.is_axis_aligned(self.tolerance):
            return
        if not self._positive_finite(feed.width):
            errors.append(f"Feed width must be positive and finite, got {feed.width!r}.")
            return
        direction = patch.edge_direction_for_point(feed.x, feed.y, self.tolerance)
        if direction is None:
            errors.append(f"Feed must be located on a patch edge, got ({feed.x}, {feed.y}).")
            return
        if feed.direction != direction:
            errors.append(f"Feed direction {feed.direction!r} does not match edge {direction!r}.")

        half_width = feed.width / 2.0
        if direction in {"top", "bottom"}:
            if feed.x - half_width < patch.left - self.tolerance or feed.x + half_width > patch.right + self.tolerance:
                errors.append("Feed edge span must remain within the patch edge.")
        else:
            if feed.y - half_width < patch.bottom - self.tolerance or feed.y + half_width > patch.top + self.tolerance:
                errors.append("Feed edge span must remain within the patch edge.")

    def _validate_boundary(self, patch: Patch, errors: List[str]) -> None:
        """Validate the final extracted conductor boundary."""

        try:
            boundary = patch.boundary()
            result = self.boundary_validator.validate(boundary)
        except Exception as exc:
            errors.append(f"CadQuery boundary extraction failed: {exc}")
            return
        errors.extend(result.errors)

    @staticmethod
    def _positive_finite(value: float) -> bool:
        """Return whether a value is finite and positive."""

        return math.isfinite(float(value)) and float(value) > 0.0
