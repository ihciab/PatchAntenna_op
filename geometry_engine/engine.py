"""Geometry Engine entry point.

The engine receives DSL commands, dispatches them through a registry, mutates a
Patch object through command classes, validates geometry, and exports patch.json
through the exporter layer. It does not call LLMs, CST, or optimizers.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, List

from geometry_engine.context import GeometryContext
from geometry_engine.dsl.command import GeometryCommand
from geometry_engine.dsl.parser import DSLParser
from geometry_engine.registry import CommandRegistry
from geometry_engine.validator import ValidationResult


class ValidateCommand(GeometryCommand):
    """Validate the current geometry state."""

    dsl_name = "Validate"
    mutates_geometry = False

    def execute(self, context: GeometryContext) -> ValidationResult:
        """Return a validation result."""

        return context.validate()


class ExportJSONCommand(GeometryCommand):
    """Export the current geometry state to patch.json."""

    dsl_name = "ExportJSON"
    mutates_geometry = False

    path: str

    def __init__(self, path: str) -> None:
        """Create an export command."""

        self.path = str(path)

    def execute(self, context: GeometryContext) -> Path:
        """Validate and export patch.json."""

        return context.export_json(self.path)


class GeometryEngine:
    """Execute Geometry DSL commands against a parameterized Patch object."""

    context: GeometryContext
    parser: DSLParser
    registry: CommandRegistry

    def __init__(
        self,
        context: GeometryContext | None = None,
        parser: DSLParser | None = None,
        registry: CommandRegistry | None = None,
        validate_after_mutation: bool = True,
    ) -> None:
        """Initialize the engine."""

        self.context = context or GeometryContext()
        self.parser = parser or DSLParser()
        self.registry = registry or CommandRegistry.with_builtin_commands()
        self.validate_after_mutation = bool(validate_after_mutation)

    def execute(self, command: str | GeometryCommand) -> Any:
        """Execute one DSL string or command object."""

        command_object = self._command_object(command)
        before = copy.deepcopy(self.context.patch)
        result = command_object.execute(self.context)
        if self.validate_after_mutation and command_object.mutates_geometry:
            validation = self.context.validate()
            if not validation.valid:
                self.context.patch = before
                validation.raise_if_invalid()
        return result

    def execute_script(self, source: str) -> List[Any]:
        """Execute a DSL script containing one or more commands."""

        results: List[Any] = []
        for parsed in self.parser.parse_script(source):
            command = self.registry.create(parsed)
            results.append(self.execute(command))
        return results

    def register_command(self, name: str, factory: Any) -> None:
        """Register a custom command without changing engine core code."""

        self.registry.register(name, factory)

    def validate(self) -> ValidationResult:
        """Validate the current geometry state."""

        return self.context.validate()

    def export_json(self, path: str | Path) -> Path:
        """Export the current geometry state as patch.json."""

        return self.context.export_json(path)

    def _command_object(self, command: str | GeometryCommand) -> GeometryCommand:
        """Resolve a DSL string or command instance to a command object."""

        if isinstance(command, GeometryCommand):
            return command
        parsed = self.parser.parse(command)
        return self.registry.create(parsed)

    def available_commands(self) -> Iterable[str]:
        """Return registered DSL command names."""

        return self.registry.names()
