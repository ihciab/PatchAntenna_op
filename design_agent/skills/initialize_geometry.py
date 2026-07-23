"""Initial geometry generation skill."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from design_agent.llm.client import LLMClient
from design_agent.llm.parser import LLMResponseParser
from design_agent.llm.prompt_loader import PromptLoader
from design_agent.models import GeometryCandidate
from design_agent.state import DesignState
from design_agent.skills.prompt_utils import append_state_block, require_llm_client


INITIAL_DESIGN_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "initial_design.md"
INITIAL_DESIGN_PROMPT = INITIAL_DESIGN_PROMPT_PATH.read_text(encoding="utf-8")


class InitializeGeometrySkill:
    """Generate an initial parameterized geometry from a selected topology."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_loader: Optional[PromptLoader] = None,
        parser: Optional[LLMResponseParser] = None,
    ) -> None:
        """Initialize dependencies for geometry initialization."""

        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.parser = parser or LLMResponseParser()

    def run(self, state: DesignState) -> GeometryCandidate:
        """Return an initial geometry candidate for the current design state."""

        client = require_llm_client(self.llm_client, self.__class__.__name__)
        prompt = append_state_block(INITIAL_DESIGN_PROMPT, state, "USER SPECIFICATION AND CURRENT STATE")
        response = client.generate(
            prompt,
            context={
                "system_prompt": "You are a deterministic RF antenna design JSON generator.",
                "temperature": 0.0,
                "timeout": 120,
                "log_label": "initialize_geometry",
            },
        )
        objects = self.parser.parse_json_objects(response)
        if len(objects) < 3:
            raise ValueError("Initial geometry prompt must return trace, stackup, and patch JSON objects.")

        trace, stackup, patch = objects[:3]
        return GeometryCandidate(
            topology=state.topology,
            parameters=patch.get("parameters", {}) if isinstance(patch.get("parameters"), dict) else {},
            metadata={
                "prompt_name": "initial_design",
                "prompt_path": str(INITIAL_DESIGN_PROMPT_PATH.resolve()),
                "raw_response": response,
                "design_trace": trace,
                "stackup": stackup,
                "patch": patch,
                "json_objects": objects,
            },
        )
