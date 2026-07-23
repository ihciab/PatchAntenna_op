"""Prompt loading utilities for external markdown prompt templates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


class PromptLoader:
    """Load prompt templates from the repository-level ``prompts`` directory."""

    def __init__(self, prompt_dir: Path = Path("prompts")) -> None:
        """Create a loader bound to a prompt directory."""

        self.prompt_dir = Path(prompt_dir)

    def load(self, prompt_name: str) -> str:
        """Load a prompt template by filename or stem.

        Args:
            prompt_name: Prompt filename, such as ``initial_design.md``, or a
                stem, such as ``initial_design``.

        Returns:
            Prompt template text.
        """

        path = self._resolve_path(prompt_name)
        return path.read_text(encoding="utf-8")

    def render(self, prompt_name: str, variables: Dict[str, object]) -> str:
        """Render a prompt template using simple ``str.format`` variables."""

        template = self.load(prompt_name)
        return template.format(**variables)

    def _resolve_path(self, prompt_name: str) -> Path:
        """Resolve a prompt name to an existing markdown file path."""

        path = self.prompt_dir / prompt_name
        if path.suffix == "":
            path = path.with_suffix(".md")
        if not path.exists():
            raise FileNotFoundError("Prompt template not found: {0}".format(path))
        return path
