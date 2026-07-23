"""Shared helpers for prompt-backed design skills."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from design_agent.llm.client import LLMClient
from design_agent.state import DesignState


def require_llm_client(client: Optional[LLMClient], skill_name: str) -> LLMClient:
    """Return an LLM client or raise a clear configuration error."""

    if client is None:
        raise ValueError("{0} requires an llm_client.".format(skill_name))
    return client


def state_to_prompt_json(state: DesignState) -> str:
    """Serialize the mutable design state into prompt-friendly JSON."""

    return json.dumps(_to_jsonable(state), indent=2, ensure_ascii=False)


def append_state_block(prompt: str, state: DesignState, title: str = "CURRENT DESIGN STATE") -> str:
    """Append a stable JSON state block to a prompt template."""

    return (
        prompt
        + "\n\n==================================================\n"
        + title
        + "\n"
        + "==================================================\n"
        + state_to_prompt_json(state)
    )


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
