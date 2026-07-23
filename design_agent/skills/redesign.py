"""Redesign skill."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from design_agent.llm.client import LLMClient
from design_agent.llm.parser import LLMResponseParser
from design_agent.llm.prompt_loader import PromptLoader
from design_agent.models import GeometryCandidate
from design_agent.state import DesignState
from design_agent.skills.prompt_utils import append_state_block, require_llm_client
from geometry_engine.context import GeometryContext
from geometry_engine.engine import GeometryEngine
from geometry_engine.importer import ParameterizationImporter
from geometry_engine.dsl.parser import DSLParseError
from geometry_engine.dsl.parser import DSLParser
from geometry_engine.registry import CommandRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_ENGINE_AGENT_GUIDE = PROJECT_ROOT / "geometry_engine" / "Agent_Usage_Guide.md"


EXPERIMENT_1_TARGET = {
    "name": "实验1：单频贴片设计",
    "center_frequency_ghz": 2.45,
    "s11_db_less_than": -15.0,
    "gain_dbi_greater_than": 6.0,
}


OPTIMIZATION_PROMPT = """# Redesign / Geometry-Operation Prompt

You are an RF antenna redesign agent. Your job is not to draw a new antenna
from scratch. Your job is to understand the target, inspect the current initial
patch geometry produced by InitializeGeometrySkill, and propose a small,
valid sequence of Geometry Engine DSL operations.

==================================================
TARGET FOR THIS STAGE
==================================================

Experiment: single-frequency microstrip patch design

- f0 = 2.45 GHz
- S11 < -15 dB at/near f0
- Gain > 6 dBi

==================================================
GEOMETRY ENGINE ROLE
==================================================

The geometry operation kernel is the local geometry_engine package. The LLM
must only output DSL operations that geometry_engine can parse and execute.
The runtime will execute the DSL, validate the geometry, and export Geometry
JSON. Do not output CST commands. Do not output raw polygon vertices unless
they are part of an allowed DSL command.

Before proposing a design, read the GEOMETRY ENGINE AGENT USAGE GUIDE block in
this prompt. Use it to determine:

1. Which DSL commands are currently available.
2. What each DSL command changes geometrically.
3. Which commands are stable enough for the current rectangular patch redesign.
4. Which commands are risky and should be avoided unless clearly justified.

For this first redesign stage, prefer conservative operations such as
Validate(), ResizePatch(), MoveFeed(), and small AddSlot() operations. You may
use other guide-documented commands only when they are necessary and compatible
with the current geometry and validator constraints.

==================================================
DESIGN INTENT
==================================================

The initial design is a simple bottom-edge-fed rectangular patch. Preserve that
basic board and feed concept. Apply only conservative operations:

1. Tune resonance primarily by resizing patch length.
   - Longer patch length usually lowers resonant frequency.
   - Shorter patch length usually raises resonant frequency.

2. Tune impedance matching primarily by moving the feed along the patch edge or
   by adding one small rectangular slot strictly inside the patch.

3. Avoid changes that are likely to reduce gain below 6 dBi. Prefer simple,
   symmetric, low-loss changes.

4. Keep all slot centers and dimensions safely inside the patch boundary.
   Slots must not touch the patch edge or feed edge.

5. End every operation list with Validate().

==================================================
CURRENT GEOMETRY EXPECTATION
==================================================

InitializeGeometrySkill stores the current initial board in:

state.current_geometry.metadata.patch

That patch JSON contains:

- conductor.components named "patch" and "feed_line"
- parameters such as patch_width_mm, patch_length_mm, feed_width_mm,
  feed_length_mm, substrate_width_mm, substrate_length_mm
- port metadata for the bottom feed

Use those values from CURRENT DESIGN STATE when deciding operation magnitudes.
If exact simulation metrics are absent, produce a conservative first redesign:

- keep f0 near 2.45 GHz,
- keep the rectangular patch intact,
- optionally add a small central rectangular slot only if it is clearly inside
  the patch,
- use Validate().

==================================================
OUTPUT
==================================================

Return exactly one JSON object. Do not output Markdown. Do not output comments.

Required structure:

{
  "target": {
    "experiment": "实验1：单频贴片设计",
    "f0_ghz": 2.45,
    "s11_db_target": "< -15",
    "gain_dbi_target": "> 6"
  },
  "geometry_engine_guide_understanding": {
    "available_dsl_commands_used_for_reasoning": ["..."],
    "selected_dsl_commands": ["..."],
    "rejected_risky_commands": ["..."],
    "reason": "..."
  },
  "rationale": "...",
  "expected_effect": {
    "resonance": "...",
    "matching": "...",
    "gain": "..."
  },
  "geometry_engine_commands": [
    "Validate()",
    "ResizePatch(length=..., width=...)",
    "MoveFeed(dx=..., dy=0.0)",
    "Validate()"
  ],
  "parameters": {
    "changed_parameters": {},
    "constraints_checked": [
      "bottom-edge feed remains on patch edge",
      "slots, if any, are strictly inside patch",
      "final command is Validate()"
    ]
  },
  "metadata": {
    "uses_geometry_engine": true,
    "source_geometry": "state.current_geometry.metadata.patch"
  }
}
"""


class RedesignSkill:
    """Apply suggestions to produce the next geometry candidate."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_loader: Optional[PromptLoader] = None,
        parser: Optional[LLMResponseParser] = None,
        max_repair_attempts: int = 20,
    ) -> None:
        """Initialize dependencies for redesign."""

        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader()
        self.parser = parser or LLMResponseParser()
        self.max_repair_attempts = int(max_repair_attempts)

    def run(self, state: DesignState) -> GeometryCandidate:
        """Return an updated geometry candidate."""

        client = require_llm_client(self.llm_client, self.__class__.__name__)
        prompt = self._build_design_prompt(state)
        payload, response = self._generate_payload(client, prompt)
        commands, geometry_result = self._commands_or_format_error(payload)
        if geometry_result is None:
            geometry_result = self._apply_geometry_engine(state, commands)
        repair_attempts: List[Dict[str, Any]] = [
            {
                "attempt": 0,
                "raw_response": response,
                "commands": commands,
                "geometry_engine": self._compact_geometry_result(geometry_result),
            }
        ]

        repair_index = 1
        while True:
            if not self._should_repair_geometry_result(geometry_result):
                break
            if self.max_repair_attempts > 0 and repair_index > self.max_repair_attempts:
                break

            repair_prompt = self._build_repair_prompt(
                state=state,
                failed_payload=payload,
                failed_response=response,
                geometry_result=geometry_result,
                repair_index=repair_index,
            )
            payload, response = self._generate_payload(client, repair_prompt)
            commands, geometry_result = self._commands_or_format_error(payload)
            if geometry_result is None:
                geometry_result = self._apply_geometry_engine(state, commands)
            repair_attempts.append(
                {
                    "attempt": repair_index,
                    "raw_response": response,
                    "commands": commands,
                    "geometry_engine": self._compact_geometry_result(geometry_result),
                }
            )
            repair_index += 1

        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        metadata.setdefault("prompt_name", "optimization")
        metadata.setdefault("raw_response", response)
        metadata["repair_attempts"] = repair_attempts
        metadata["target"] = payload.get("target", EXPERIMENT_1_TARGET)
        metadata["geometry_engine_guide_understanding"] = payload.get(
            "geometry_engine_guide_understanding",
            {},
        )
        metadata["rationale"] = payload.get("rationale", "")
        metadata["expected_effect"] = payload.get("expected_effect", {})
        metadata["geometry_engine_commands"] = commands
        metadata["geometry_engine"] = geometry_result
        return GeometryCandidate(
            topology=state.topology,
            parameters=payload.get("parameters", {}) if isinstance(payload.get("parameters"), dict) else {},
            metadata=metadata,
        )

    def _build_design_prompt(self, state: DesignState) -> str:
        """Build the redesign prompt with the Geometry Engine guide included."""

        prompt = (
            OPTIMIZATION_PROMPT
            + "\n\n==================================================\n"
            + "GEOMETRY ENGINE AGENT USAGE GUIDE\n"
            + "==================================================\n"
            + self._load_geometry_engine_agent_guide()
        )
        return append_state_block(prompt, state, "CURRENT DESIGN STATE")

    @staticmethod
    def _load_geometry_engine_agent_guide() -> str:
        """Load the local Geometry Engine agent guide for LLM grounding."""

        if not GEOMETRY_ENGINE_AGENT_GUIDE.exists():
            return "Guide file not found: {0}".format(GEOMETRY_ENGINE_AGENT_GUIDE)
        return GEOMETRY_ENGINE_AGENT_GUIDE.read_text(encoding="utf-8", errors="replace")

    def _generate_payload(self, client: LLMClient, prompt: str) -> tuple[Dict[str, Any], str]:
        """Call the LLM and parse one JSON object payload."""

        response = client.generate(
            prompt,
            context={
                "system_prompt": (
                    "You are a deterministic RF antenna redesign agent. "
                    "Return exactly one JSON object containing Geometry Engine DSL commands."
                ),
                "temperature": 0.0,
            },
        )
        try:
            payload = self.parser.parse_json(response)
        except ValueError:
            payload = self.parser.parse_json_objects(response)[0]
        return payload, response

    def _extract_commands(self, payload: Dict[str, Any]) -> List[str]:
        """Extract and lightly normalize the Geometry Engine DSL command list."""

        raw_commands = payload.get("geometry_engine_commands")
        if not isinstance(raw_commands, list):
            raise ValueError("Redesign response must contain `geometry_engine_commands` as a list.")

        commands = [str(command).strip() for command in raw_commands if str(command).strip()]
        if not commands:
            raise ValueError("Redesign response must contain at least one Geometry Engine command.")
        if commands[-1] != "Validate()":
            commands.append("Validate()")
        return commands

    def _commands_or_format_error(self, payload: Dict[str, Any]) -> tuple[List[str], Optional[Dict[str, Any]]]:
        """Return commands or a repairable LLM-output format failure."""

        try:
            return self._extract_commands(payload), None
        except Exception as exc:
            return [], {
                "executed": False,
                "commands": [],
                "execution_log": [],
                "error": str(exc),
                "error_type": "RedesignResponseFormatError",
                "failed_payload": payload,
            }

    def _build_repair_prompt(
        self,
        state: DesignState,
        failed_payload: Dict[str, Any],
        failed_response: str,
        geometry_result: Dict[str, Any],
        repair_index: int,
    ) -> str:
        """Build a feedback prompt that asks the LLM to repair invalid DSL."""

        compact_result = self._compact_geometry_result(geometry_result)
        feedback = {
            "repair_attempt": repair_index,
            "instruction": (
                "The previous redesign response or Geometry Engine DSL failed. Produce a corrected "
                "single JSON object with a new geometry_engine_commands list. "
                "Re-read the GEOMETRY ENGINE AGENT USAGE GUIDE in this prompt, "
                "choose valid DSL operations, and end with Validate()."
            ),
            "failed_llm_response": failed_response,
            "failed_payload": failed_payload,
            "geometry_engine_failure": compact_result,
            "repair_rules": [
                "Do not repeat the same failing command sequence.",
                "The JSON object must contain top-level key `geometry_engine_commands` and its value must be a list of DSL strings.",
                "The command list must contain at least one real geometry edit, not only Validate().",
                "If a slot is outside or too close to an edge, move it inward or reduce its size.",
                "If MoveFeed fails, keep dy=0.0 for a bottom-edge feed and use a smaller dx.",
                "If ResizePatch causes feed or slot invalidity, adjust patch size more conservatively.",
                "Return exactly one JSON object and no Markdown.",
            ],
        }
        return (
            self._build_design_prompt(state)
            + "\n\n==================================================\n"
            + "GEOMETRY ENGINE ERROR FEEDBACK\n"
            + "==================================================\n"
            + json.dumps(feedback, indent=2, ensure_ascii=False)
        )

    def _apply_geometry_engine(self, state: DesignState, commands: List[str]) -> Dict[str, Any]:
        """Execute DSL operations against the current initialized patch when available."""

        patch_payload = self._current_patch_payload(state)
        if patch_payload is None:
            return {
                "executed": False,
                "reason": "state.current_geometry.metadata.patch is not available",
            }

        execution_log: List[Dict[str, Any]] = []
        try:
            if not self._has_mutating_command(commands):
                return {
                    "executed": False,
                    "commands": commands,
                    "execution_log": execution_log,
                    "error": "No geometry edit command was provided. At least one non-Validate mutating DSL command is required.",
                    "error_type": "NoGeometryEditError",
                }
            patch = ParameterizationImporter().from_dict(patch_payload)
            engine = GeometryEngine(context=GeometryContext(patch=patch))
            for index, command in enumerate(commands):
                result = engine.execute(command)
                validation = engine.validate()
                execution_log.append(
                    {
                        "index": index,
                        "command": command,
                        "result": str(result),
                        "valid": validation.valid,
                        "errors": list(validation.errors),
                    }
                )

            geometry_json = engine.context.exporter.to_dict(engine.context.patch)
            return {
                "executed": True,
                "commands": commands,
                "execution_log": execution_log,
                "geometry_json": geometry_json,
            }
        except Exception as exc:
            return {
                "executed": False,
                "commands": commands,
                "execution_log": execution_log,
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            }

    @staticmethod
    def _should_repair_geometry_result(geometry_result: Dict[str, Any]) -> bool:
        """Return whether a failed Geometry Engine result should be sent back to the LLM."""

        if geometry_result.get("executed") is True:
            return False
        message = str(geometry_result.get("error") or geometry_result.get("reason") or "")
        if "CadQuery is required" in message:
            return False
        return True

    @staticmethod
    def _has_mutating_command(commands: List[str]) -> bool:
        """Return whether the DSL list contains at least one geometry mutation."""

        parser = DSLParser()
        registry = CommandRegistry.with_builtin_commands()
        for command in commands:
            try:
                command_object = registry.create(parser.parse(command))
            except (DSLParseError, KeyError):
                return True
            if getattr(command_object, "mutates_geometry", False):
                return True
        return False

    @staticmethod
    def _compact_geometry_result(geometry_result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove large successful geometry payloads from retry metadata."""

        compact = dict(geometry_result)
        if isinstance(compact.get("geometry_json"), dict):
            compact["geometry_json"] = "<omitted>"
        return compact

    @staticmethod
    def _current_patch_payload(state: DesignState) -> Optional[Dict[str, Any]]:
        """Return the patch JSON produced by InitializeGeometrySkill, if present."""

        geometry = state.current_geometry
        if geometry is None:
            return None
        patch_payload = geometry.metadata.get("patch")
        return patch_payload if isinstance(patch_payload, dict) else None
