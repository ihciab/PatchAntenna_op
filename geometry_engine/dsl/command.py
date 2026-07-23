"""Command abstractions for the Geometry Engine DSL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from geometry_engine.context import GeometryContext


@dataclass(frozen=True)
class ParsedCommand:
    """A parsed DSL invocation before it becomes an executable command."""

    name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


class GeometryCommand(ABC):
    """Base class for all executable Geometry Engine commands."""

    mutates_geometry: bool = True

    @abstractmethod
    def execute(self, context: GeometryContext) -> Any:
        """Execute the command against a geometry context."""
