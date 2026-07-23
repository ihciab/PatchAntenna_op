"""Topology selection skill."""

from __future__ import annotations

from typing import Optional

from design_agent.llm.client import LLMClient
from design_agent.llm.parser import LLMResponseParser
from design_agent.llm.prompt_loader import PromptLoader
from design_agent.models import AntennaTopology
from design_agent.state import DesignState
from design_agent.skills.prompt_utils import append_state_block, require_llm_client


TOPOLOGY_SELECTION_PROMPT = """# Topology Selection Prompt

TODO: Select a suitable antenna topology from the target specification.

Return one JSON object with this structure:

{
  "name": "...",
  "rationale": "...",
  "parameters": {}
}
"""


class SelectTopologySkill:
    """Select an antenna topology from target design requirements."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_loader: Optional[PromptLoader] = None,
        parser: Optional[LLMResponseParser] = None,
    ) -> None:
        """Initialize dependencies for topology selection."""

        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.parser = parser or LLMResponseParser()

    def run(self, state: DesignState) -> AntennaTopology:
        """Return a selected topology for the current design state."""

        client = require_llm_client(self.llm_client, self.__class__.__name__)
        prompt = append_state_block(TOPOLOGY_SELECTION_PROMPT, state)
        response = client.generate(
            prompt,
            context={
                "system_prompt": "You are a deterministic RF antenna topology selection agent.",
                "temperature": 0.0,
            },
        )
        try:
            payload = self.parser.parse_json(response)
        except ValueError:
            payload = self.parser.parse_json_objects(response)[0]

        return AntennaTopology(
            name=str(payload.get("name", "unknown_topology")),
            rationale=str(payload.get("rationale", "")),
            parameters=payload.get("parameters", {}) if isinstance(payload.get("parameters"), dict) else {},
        )
