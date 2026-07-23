"""Command registry for extensible Geometry Engine DSL dispatch."""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

from geometry_engine.dsl.command import GeometryCommand, ParsedCommand


CommandFactory = Callable[..., GeometryCommand]


class CommandRegistry:
    """Map DSL command names to command factories."""

    _factories: Dict[str, CommandFactory]

    def __init__(self) -> None:
        """Initialize an empty registry."""

        self._factories = {}

    def register(self, name: str, factory: CommandFactory) -> None:
        """Register a command factory by DSL name."""

        self._factories[name] = factory

    def register_class(self, command_class: Type[GeometryCommand]) -> Type[GeometryCommand]:
        """Register a command class using its ``dsl_name`` attribute."""

        name = getattr(command_class, "dsl_name", command_class.__name__)
        self.register(str(name), command_class)
        return command_class

    def create(self, parsed: ParsedCommand) -> GeometryCommand:
        """Instantiate a command from parsed DSL input."""

        factory = self._factories.get(parsed.name)
        if factory is None:
            available = ", ".join(sorted(self._factories))
            raise KeyError(f"Unknown Geometry DSL command {parsed.name!r}. Available commands: {available}")
        return factory(*parsed.args, **parsed.kwargs)

    @classmethod
    def with_builtin_commands(cls) -> "CommandRegistry":
        """Return a registry populated with first-version DSL commands."""

        from geometry_engine.commands.add_slot import AddSlotCommand
        from geometry_engine.commands.boolean import BooleanDifferenceCommand, BooleanUnionCommand
        from geometry_engine.commands.construction import CircleCommand, PolygonCommand, RectangleCommand
        from geometry_engine.commands.delete_slot import DeleteSlotCommand
        from geometry_engine.commands.mirror import MirrorXCommand, MirrorYCommand
        from geometry_engine.commands.move_feed import MoveFeedCommand
        from geometry_engine.commands.resize_patch import ResizePatchCommand
        from geometry_engine.commands.transform import RotateCommand, ScaleCommand, TranslateCommand
        from geometry_engine.engine import ExportJSONCommand, ValidateCommand

        registry = cls()
        for command_class in (
            RectangleCommand,
            CircleCommand,
            PolygonCommand,
            ResizePatchCommand,
            MoveFeedCommand,
            AddSlotCommand,
            DeleteSlotCommand,
            BooleanUnionCommand,
            BooleanDifferenceCommand,
            MirrorXCommand,
            MirrorYCommand,
            TranslateCommand,
            RotateCommand,
            ScaleCommand,
            ValidateCommand,
            ExportJSONCommand,
        ):
            registry.register_class(command_class)
        return registry

    def names(self) -> list[str]:
        """Return registered command names."""

        return sorted(self._factories)
