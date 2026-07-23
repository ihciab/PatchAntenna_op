"""Parsing helpers for LLM responses."""

from __future__ import annotations

import json
from typing import Any, Dict, List


class LLMResponseParser:
    """Parse structured content returned by an LLM."""

    def parse_json(self, response: str) -> Dict[str, Any]:
        """Parse a JSON object from an LLM response string.

        Args:
            response: Raw LLM text expected to contain a JSON object.

        Returns:
            Parsed JSON dictionary.
        """

        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from the LLM response.")
        return parsed

    def parse_design_rationale(self, response: str) -> str:
        """Parse or normalize a design rationale response."""

        return response.strip()

    def parse_json_objects(self, response: str) -> List[Dict[str, Any]]:
        """Parse one or more JSON objects from an LLM response.

        This helper is useful during testing because models may return several
        adjacent JSON objects instead of a single valid JSON array. Non-JSON text
        before, between, or after objects is ignored.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed top-level JSON objects in the order they appear.
        """

        cleaned = response.replace("```json", "").replace("```", "")
        decoder = json.JSONDecoder()
        objects: List[Dict[str, Any]] = []
        index = 0

        while index < len(cleaned):
            while index < len(cleaned) and cleaned[index] != "{":
                index += 1
            if index >= len(cleaned):
                break

            start = index
            depth = 0
            in_string = False
            escape = False
            end = None
            while index < len(cleaned):
                char = cleaned[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                else:
                    if char == '"':
                        in_string = True
                    elif char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            end = index + 1
                            break
                index += 1

            if end is None:
                break

            candidate = cleaned[start:end]
            try:
                parsed = decoder.decode(candidate)
            except json.JSONDecodeError:
                index = end
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
            index = end

        if not objects:
            raise ValueError("No JSON objects found in the LLM response.")
        return objects
