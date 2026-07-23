"""Design reflection skill."""

from __future__ import annotations

from typing import Optional

from design_agent.llm.client import LLMClient
from design_agent.llm.parser import LLMResponseParser
from design_agent.llm.prompt_loader import PromptLoader
from design_agent.models import OptimizationSuggestion
from design_agent.state import DesignState
from design_agent.skills.prompt_utils import append_state_block, require_llm_client


REFLECTION_PROMPT = """# Reflection Prompt

TODO: Analyze simulation and evaluation results and suggest improvements.

Return one JSON object with this structure:

{
  "parameters": {},
  "rationale": "...",
  "source": "reflection"
}
"""


class ReflectDesignSkill:
    """Analyze the latest results and propose design improvements."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_loader: Optional[PromptLoader] = None,
        parser: Optional[LLMResponseParser] = None,
    ) -> None:
        """Initialize dependencies for design reflection."""

        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.parser = parser or LLMResponseParser()

    def run(self, state: DesignState) -> OptimizationSuggestion:
        """Return a redesign or optimization suggestion."""

        client = require_llm_client(self.llm_client, self.__class__.__name__)
        prompt = append_state_block(REFLECTION_PROMPT, state)
        response = client.generate(
            prompt,
            context={
                "system_prompt": "You are a deterministic RF antenna design reviewer.",
                "temperature": 0.0,
            },
        )
        try:
            payload = self.parser.parse_json(response)
        except ValueError:
            payload = self.parser.parse_json_objects(response)[0]

        return OptimizationSuggestion(
            parameters=payload.get("parameters", {}) if isinstance(payload.get("parameters"), dict) else {},
            rationale=str(payload.get("rationale", "")),
            source=str(payload.get("source", "reflection")),
        )
