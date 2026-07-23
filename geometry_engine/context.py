"""Runtime context for Geometry Engine command execution."""

from __future__ import annotations

from pathlib import Path

from geometry_engine.exporter import GeometryJSONExporter
from geometry_engine.geometry.patch import Patch
from geometry_engine.validator import GeometryValidator, ValidationResult


class GeometryContext:
    """Holds mutable geometry state and shared services."""

    patch: Patch
    validator: GeometryValidator
    exporter: GeometryJSONExporter
    _slot_counter: int

    def __init__(
        self,
        patch: Patch | None = None,
        validator: GeometryValidator | None = None,
        exporter: GeometryJSONExporter | None = None,
    ) -> None:
        """Initialize command context with a patch model."""

        self.patch = patch or Patch()
        self.validator = validator or GeometryValidator()
        self.exporter = exporter or GeometryJSONExporter()
        self._slot_counter = self._initial_slot_counter()
        self.patch.sync_feed_direction()

    def validate(self) -> ValidationResult:
        """Validate the current patch."""

        return self.validator.validate(self.patch)

    def export_json(self, path: str | Path) -> Path:
        """Export the current patch as patch.json."""

        return self.exporter.export(self.patch, path)

    def next_slot_id(self) -> str:
        """Return a new stable slot id."""

        self._slot_counter += 1
        return f"slot_{self._slot_counter:03d}"

    def _initial_slot_counter(self) -> int:
        """Infer the next slot id counter from existing slots."""

        max_counter = 0
        for slot in self.patch.slots:
            if slot.id.startswith("slot_"):
                try:
                    max_counter = max(max_counter, int(slot.id.split("_", 1)[1]))
                except ValueError:
                    continue
        return max_counter
