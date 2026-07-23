"""Lightweight design-operation generation skill."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

from design_agent.llm.client import LLMClient
from design_agent.llm.parser import LLMResponseParser
from design_agent.llm.prompt_loader import PromptLoader
from design_agent.skills.prompt_utils import require_llm_client
from design_agent.tools.geometry_summary import default_geometry_summary_path
from design_agent.tools.simulation_summary import default_simulation_summary_path
from design_agent.tools.bo_parameterization_summary import default_bo_parameterization_summary_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_INPUTS_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"
SKILL_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
MAX_OPERATIONS_PER_PLAN = 5
PARSE_FAILURE_DIR = PROJECT_ROOT / "design_agent_runs" / "llm_parse_failures"


class LightweightDesignSkill:
    """Read summarized inputs and generate geometry modification operations.

    This skill intentionally does not call CST, CadQuery, or the Geometry Engine.
    It only runs a prompt-backed diagnose -> plan -> generate reasoning chain and
    returns operation JSON for a later Geometry Backend.
    """

    REQUIRED_INPUTS = (
        "target",
        "geometry_summary",
        "simulation_summary",
        "history",
    )
    OPTIONAL_INPUTS = (
        "bo_parameterization_summary",
    )

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_loader: Optional[PromptLoader] = None,
        parser: Optional[LLMResponseParser] = None,
    ) -> None:
        """Initialize dependencies."""

        self.llm_client = llm_client
        self.prompt_loader = prompt_loader or PromptLoader(SKILL_PROMPT_DIR)
        self.parser = parser or LLMResponseParser()

    def run_from_files(
        self,
        target_path: Path | str = AGENT_INPUTS_DIR / "target.md",
        geometry_summary_path: Path | str = default_geometry_summary_path(),
        simulation_summary_path: Path | str = default_simulation_summary_path(),
        history_path: Path | str = AGENT_INPUTS_DIR / "history.json",
        bo_parameterization_summary_path: Optional[Path | str] = default_bo_parameterization_summary_path(),
    ) -> Dict[str, Any]:
        """Load target Markdown plus summarized JSON input files and return operations."""

        inputs = {
            "target": self._load_target_object(target_path),
            "geometry_summary": self._load_json_object(geometry_summary_path),
            "simulation_summary": self._load_json_object(simulation_summary_path),
            "history": self._load_history_object(history_path),
        }
        bo_summary = self._load_optional_json_object(bo_parameterization_summary_path)
        if bo_summary is not None:
            inputs["bo_parameterization_summary"] = bo_summary
        return self.run(inputs)

    def run(self, inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run diagnose -> plan -> generate using summarized JSON only."""

        self._validate_inputs(inputs)
        client = require_llm_client(self.llm_client, self.__class__.__name__)
        system_prompt = self.prompt_loader.load("system")

        diagnosis = self._call_json_step(
            client=client,
            system_prompt=system_prompt,
            prompt_name="diagnose",
            inputs=inputs,
        )
        plan = self._call_json_step(
            client=client,
            system_prompt=system_prompt,
            prompt_name="plan",
            inputs=inputs,
            step_context={"diagnosis": diagnosis},
        )
        result = self._call_json_step(
            client=client,
            system_prompt=system_prompt,
            prompt_name="generate",
            inputs=inputs,
            step_context={
                "diagnosis": diagnosis,
                "plan": plan,
            },
        )
        self._validate_result(result)
        return result

    def run_with_trace(self, inputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run the full chain and include intermediate diagnosis and plan."""

        self._validate_inputs(inputs)
        client = require_llm_client(self.llm_client, self.__class__.__name__)
        system_prompt = self.prompt_loader.load("system")

        diagnosis = self._call_json_step(
            client=client,
            system_prompt=system_prompt,
            prompt_name="diagnose",
            inputs=inputs,
        )
        plan = self._call_json_step(
            client=client,
            system_prompt=system_prompt,
            prompt_name="plan",
            inputs=inputs,
            step_context={"diagnosis": diagnosis},
        )
        result = self._call_json_step(
            client=client,
            system_prompt=system_prompt,
            prompt_name="generate",
            inputs=inputs,
            step_context={
                "diagnosis": diagnosis,
                "plan": plan,
            },
        )
        self._validate_result(result)
        return {
            "diagnosis": diagnosis,
            "plan": plan,
            "result": result,
        }

    def repair_operation_plan(
        self,
        inputs: Dict[str, Dict[str, Any]],
        diagnosis: Dict[str, Any],
        plan: Dict[str, Any],
        failed_operation_plan: Dict[str, Any],
        geometry_error: Dict[str, Any],
        repair_attempt: int,
    ) -> Dict[str, Any]:
        """Ask the LLM to repair an operation plan that failed geometry validation."""

        self._validate_inputs(inputs)
        client = require_llm_client(self.llm_client, self.__class__.__name__)
        system_prompt = self.prompt_loader.load("system")
        prompt = self._build_repair_prompt(
            inputs=inputs,
            diagnosis=diagnosis,
            plan=plan,
            failed_operation_plan=failed_operation_plan,
            geometry_error=geometry_error,
            repair_attempt=repair_attempt,
        )
        response = client.generate(
            prompt,
            context={
                "system_prompt": system_prompt,
                "temperature": 0.0,
                "log_label": "lightweight_design.repair_attempt_{0}".format(repair_attempt),
            },
        )
        result = self._parse_json_response(
            response=response,
            client=client,
            system_prompt=system_prompt,
            prompt_name="repair_attempt_{0}".format(repair_attempt),
        )
        self._validate_result(result)
        return result

    def reflect_iteration_effect(
        self,
        *,
        inputs: Dict[str, Dict[str, Any]],
        operation_plan: Dict[str, Any],
        before_simulation_summary: Dict[str, Any],
        after_simulation_summary: Dict[str, Any],
        geometry_summary: Dict[str, Any],
        effect: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the LLM to summarize the electromagnetic effect of one iteration."""

        client = require_llm_client(self.llm_client, self.__class__.__name__)
        system_prompt = self.prompt_loader.load("system")
        prompt = self._build_reflection_prompt(
            inputs=inputs,
            operation_plan=operation_plan,
            before_simulation_summary=before_simulation_summary,
            after_simulation_summary=after_simulation_summary,
            geometry_summary=geometry_summary,
            effect=effect,
        )
        response = client.generate(
            prompt,
            context={
                "system_prompt": system_prompt,
                "temperature": 0.0,
                "log_label": "lightweight_design.reflect_history",
            },
        )
        result = self._parse_json_response(
            response=response,
            client=client,
            system_prompt=system_prompt,
            prompt_name="reflect_history",
        )
        return self._normalize_reflection_result(result)

    def reflect_bo_effect(
        self,
        *,
        target: Dict[str, Any],
        history: Dict[str, Any],
        optimization_history: Dict[str, Any],
        bo_parameterization_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ask the LLM to summarize a completed BO run for future planning."""

        client = require_llm_client(self.llm_client, self.__class__.__name__)
        system_prompt = self.prompt_loader.load("system")
        prompt = self._build_bo_reflection_prompt(
            target=target,
            history=history,
            optimization_history=optimization_history,
            bo_parameterization_summary=bo_parameterization_summary,
        )
        response = client.generate(
            prompt,
            context={
                "system_prompt": system_prompt,
                "temperature": 0.0,
                "log_label": "lightweight_design.reflect_bo",
            },
        )
        result = self._parse_json_response(
            response=response,
            client=client,
            system_prompt=system_prompt,
            prompt_name="reflect_bo",
        )
        return self._normalize_reflection_result(result)

    def _call_json_step(
        self,
        client: LLMClient,
        system_prompt: str,
        prompt_name: str,
        inputs: Dict[str, Dict[str, Any]],
        step_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build one step prompt, call the LLM, and parse one JSON object."""

        prompt = self._build_step_prompt(prompt_name, inputs, step_context or {})
        response = client.generate(
            prompt,
            context={
                "system_prompt": system_prompt,
                "temperature": 0.0,
                "log_label": "lightweight_design.{0}".format(prompt_name),
            },
        )
        return self._parse_json_response(
            response=response,
            client=client,
            system_prompt=system_prompt,
            prompt_name=prompt_name,
        )

    def _parse_json_response(
        self,
        *,
        response: str,
        client: Optional[LLMClient],
        system_prompt: str,
        prompt_name: str,
    ) -> Dict[str, Any]:
        """Parse one JSON object, using an LLM repair pass for malformed JSON."""

        try:
            parsed = self.parser.parse_json(response)
            self._validate_prompt_shape(prompt_name, parsed)
            return parsed
        except ValueError as first_error:
            try:
                parsed = self.parser.parse_json_objects(response)[0]
                self._validate_prompt_shape(prompt_name, parsed)
                return parsed
            except ValueError as second_error:
                failure_path = self._write_parse_failure(prompt_name, response, second_error)
                if client is None:
                    raise ValueError(
                        "LLM response for `{0}` was not valid JSON. Raw response saved to {1}. "
                        "Initial parse error: {2}".format(prompt_name, failure_path, first_error)
                    ) from second_error

                repaired_response = client.generate(
                    self._build_json_repair_prompt(
                        prompt_name=prompt_name,
                        invalid_response=response,
                        parse_error=first_error,
                    ),
                    context={
                        "system_prompt": (
                            "You repair malformed JSON. Return exactly one strict JSON object "
                            "and no Markdown or explanatory text."
                        ),
                        "temperature": 0.0,
                        "log_label": "lightweight_design.{0}.json_repair".format(prompt_name),
                    },
                )
                repair_path = self._write_parse_failure(
                    "{0}_repair_response".format(prompt_name),
                    repaired_response,
                    None,
                )
                try:
                    parsed = self.parser.parse_json(repaired_response)
                    self._validate_prompt_shape(prompt_name, parsed)
                    return parsed
                except ValueError:
                    try:
                        parsed = self.parser.parse_json_objects(repaired_response)[0]
                        self._validate_prompt_shape(prompt_name, parsed)
                        return parsed
                    except ValueError as repair_error:
                        raise ValueError(
                            "LLM response for `{0}` was not valid JSON and automatic JSON repair failed. "
                            "Raw response saved to {1}; repair response saved to {2}. "
                            "Initial parse error: {3}; repair parse error: {4}".format(
                                prompt_name,
                                failure_path,
                                repair_path,
                                first_error,
                                repair_error,
                            )
                            ) from repair_error

    @staticmethod
    def _expected_json_keys(prompt_name: str) -> Optional[set[str]]:
        """Return required top-level keys for prompts that have a fixed schema."""

        mapping = {
            "diagnose": {"current_problems", "evidence", "possible_physical_causes", "design_opportunities"},
            "plan": {"design_mode", "design_hypothesis", "strategy", "rf_rationale", "anti_repetition_rule", "modification_candidates"},
            "generate": {"iteration", "reasoning", "strategy", "design_hypothesis", "operations"},
            "repair": {"iteration", "reasoning", "strategy", "operations"},
            "reflect_history": {"available", "summary", "electromagnetic_lessons", "strategy_effect", "next_iteration_guidance", "avoid_repeating"},
            "reflect_bo": {"available", "summary", "electromagnetic_lessons", "strategy_effect", "next_iteration_guidance", "avoid_repeating"},
        }
        for prefix, keys in mapping.items():
            if prompt_name == prefix or prompt_name.startswith(prefix + "_attempt_"):
                return keys
        return None

    def _validate_prompt_shape(self, prompt_name: str, payload: Dict[str, Any]) -> None:
        """Reject JSON that parses but does not match the expected prompt schema."""

        expected_keys = self._expected_json_keys(prompt_name)
        if expected_keys is None:
            return
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object for `{0}`.".format(prompt_name))
        missing = [key for key in expected_keys if key not in payload]
        if missing:
            raise ValueError(
                "Parsed JSON for `{0}` is missing required keys: {1}".format(
                    prompt_name,
                    ", ".join(missing),
                )
            )

    @staticmethod
    def _build_json_repair_prompt(
        *,
        prompt_name: str,
        invalid_response: str,
        parse_error: Exception,
    ) -> str:
        """Build a prompt that converts malformed model output into strict JSON."""

        return (
            "The previous response for lightweight_design.{0} was intended to be exactly one JSON object, "
            "but Python json.loads failed.\n\n"
            "Parse error:\n{1}\n\n"
            "Repair rules:\n"
            "- Output exactly one valid JSON object.\n"
            "- Use double quotes for every JSON key and string.\n"
            "- Escape any double quote that appears inside a string value.\n"
            "- Do not use comments, trailing commas, NaN, Infinity, or unescaped newlines in strings.\n"
            "- Preserve the original meaning and top-level keys.\n"
            "- Do not output Markdown, code fences, or explanations.\n\n"
            "Malformed response to repair:\n"
            "{2}"
        ).format(prompt_name, parse_error, invalid_response)

    @staticmethod
    def _write_parse_failure(
        prompt_name: str,
        response: str,
        error_value: Optional[Exception],
    ) -> Path:
        """Persist malformed raw responses for debugging provider/model changes."""

        PARSE_FAILURE_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prompt_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = PARSE_FAILURE_DIR / "{0}_{1}.txt".format(timestamp, safe_name)
        header = ""
        if error_value is not None:
            header = "PARSE_ERROR: {0}: {1}\n\n".format(error_value.__class__.__name__, error_value)
        output_path.write_text(header + str(response), encoding="utf-8")
        return output_path

    def _build_step_prompt(
        self,
        prompt_name: str,
        inputs: Dict[str, Dict[str, Any]],
        step_context: Dict[str, Any],
    ) -> str:
        """Append summarized JSON inputs to an external prompt template."""

        prompt = self.prompt_loader.load(prompt_name)
        payload = {
            "target.md": inputs["target"],
            "geometry_summary.json": inputs["geometry_summary"],
            "simulation_summary.json": inputs["simulation_summary"],
            "history.json": inputs["history"],
        }
        if "bo_parameterization_summary" in inputs:
            payload["bo_parameterization_summary.json"] = inputs["bo_parameterization_summary"]
        if step_context:
            payload["step_context"] = step_context
        return (
            prompt
            + "\n\n==================================================\n"
            + "SUMMARIZED INPUT JSON\n"
            + "==================================================\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
        )

    def _build_repair_prompt(
        self,
        inputs: Dict[str, Dict[str, Any]],
        diagnosis: Dict[str, Any],
        plan: Dict[str, Any],
        failed_operation_plan: Dict[str, Any],
        geometry_error: Dict[str, Any],
        repair_attempt: int,
    ) -> str:
        """Build the repair prompt from the external repair template."""

        prompt = self.prompt_loader.load("repair")
        payload = {
            "target.md": inputs["target"],
            "geometry_summary.json": inputs["geometry_summary"],
            "simulation_summary.json": inputs["simulation_summary"],
            "history.json": inputs["history"],
            "diagnosis": diagnosis,
            "plan": plan,
            "failed_operation_plan": failed_operation_plan,
            "geometry_backend_error": geometry_error,
            "repair_attempt": repair_attempt,
        }
        if "bo_parameterization_summary" in inputs:
            payload["bo_parameterization_summary.json"] = inputs["bo_parameterization_summary"]
        return (
            prompt
            + "\n\n==================================================\n"
            + "REPAIR CONTEXT JSON\n"
            + "==================================================\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
        )

    def _build_reflection_prompt(
        self,
        *,
        inputs: Dict[str, Dict[str, Any]],
        operation_plan: Dict[str, Any],
        before_simulation_summary: Dict[str, Any],
        after_simulation_summary: Dict[str, Any],
        geometry_summary: Dict[str, Any],
        effect: Dict[str, Any],
    ) -> str:
        """Build the post-simulation history reflection prompt."""

        prompt = self.prompt_loader.load("reflect_history")
        payload = {
            "target.md": inputs["target"],
            "history.json": inputs["history"],
            "operation_plan.json": operation_plan,
            "before_simulation_summary.json": before_simulation_summary,
            "after_simulation_summary.json": after_simulation_summary,
            "geometry_summary.json": geometry_summary,
            "computed_effect": effect,
        }
        if "bo_parameterization_summary" in inputs:
            payload["bo_parameterization_summary.json"] = inputs["bo_parameterization_summary"]
        return (
            prompt
            + "\n\n==================================================\n"
            + "REFLECTION CONTEXT JSON\n"
            + "==================================================\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
        )

    def _build_bo_reflection_prompt(
        self,
        *,
        target: Dict[str, Any],
        history: Dict[str, Any],
        optimization_history: Dict[str, Any],
        bo_parameterization_summary: Optional[Dict[str, Any]],
    ) -> str:
        """Build the post-BO history reflection prompt."""

        prompt = self.prompt_loader.load("reflect_bo")
        payload = {
            "target.md": target,
            "history.json": history,
            "optimization_history.json": _compact_bo_history(optimization_history),
        }
        if bo_parameterization_summary is not None:
            payload["bo_parameterization_summary.json"] = bo_parameterization_summary
        return (
            prompt
            + "\n\n==================================================\n"
            + "BO REFLECTION CONTEXT JSON\n"
            + "==================================================\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
        )

    @staticmethod
    def _normalize_reflection_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize the reflection object before writing it to history."""

        normalized = dict(result)
        normalized["available"] = bool(normalized.get("available", True))
        normalized["summary"] = str(normalized.get("summary", "")).strip()
        for key in ("electromagnetic_lessons", "next_iteration_guidance", "avoid_repeating"):
            value = normalized.get(key)
            if isinstance(value, str):
                normalized[key] = [value]
            elif not isinstance(value, list):
                normalized[key] = []
        strategy_effect = normalized.get("strategy_effect")
        if not isinstance(strategy_effect, dict):
            strategy_effect = {}
        for key in ("helped", "hurt", "uncertain"):
            value = strategy_effect.get(key)
            if isinstance(value, str):
                strategy_effect[key] = [value]
            elif not isinstance(value, list):
                strategy_effect[key] = []
        normalized["strategy_effect"] = strategy_effect
        return normalized

    @classmethod
    def _validate_inputs(cls, inputs: Dict[str, Dict[str, Any]]) -> None:
        """Ensure the skill receives required summarized input objects."""

        missing = [name for name in cls.REQUIRED_INPUTS if name not in inputs]
        if missing:
            raise ValueError("Missing summarized inputs: {0}".format(", ".join(missing)))
        for name in cls.REQUIRED_INPUTS:
            if not isinstance(inputs[name], dict):
                raise ValueError("Input `{0}` must be a JSON object.".format(name))
        for name in cls.OPTIONAL_INPUTS:
            if name in inputs and not isinstance(inputs[name], dict):
                raise ValueError("Input `{0}` must be a JSON object.".format(name))

    @staticmethod
    def _load_optional_json_object(path: Optional[Path | str]) -> Optional[Dict[str, Any]]:
        """Load an optional JSON object, returning None when the file is absent."""

        if path is None:
            return None
        candidate = Path(path)
        if not candidate.exists() or candidate.stat().st_size == 0:
            return None
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object in {0}".format(candidate))
        return payload

    @staticmethod
    def _validate_result(result: Dict[str, Any]) -> None:
        """Validate the generated operation payload contract."""

        operations = result.get("operations")
        if not isinstance(operations, list):
            raise ValueError("Generated result must contain `operations` as a list.")
        if len(operations) > MAX_OPERATIONS_PER_PLAN:
            raise ValueError(
                "Generated result contains more than {0} operations.".format(MAX_OPERATIONS_PER_PLAN)
            )
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise ValueError("Operation {0} must be a JSON object.".format(index))
            if not operation.get("operation"):
                raise ValueError("Operation {0} is missing `operation`.".format(index))
            parameters = operation.get("parameters")
            if parameters is None:
                operation["parameters"] = {}
            elif not isinstance(parameters, dict):
                raise ValueError("Operation {0} must contain object `parameters`.".format(index))

    @staticmethod
    def _load_json_object(path: Path | str) -> Dict[str, Any]:
        """Load one required JSON object from disk."""

        json_path = Path(path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object in {0}".format(json_path))
        return payload

    @staticmethod
    def _load_target_object(path: Path | str) -> Dict[str, Any]:
        """Load the target file, accepting Markdown as the primary format."""

        target_path = Path(path)
        if target_path.suffix.lower() == ".md":
            return {
                "format": "markdown",
                "source": str(target_path),
                "content": target_path.read_text(encoding="utf-8"),
            }
        return LightweightDesignSkill._load_json_object(target_path)

    @staticmethod
    def _load_history_object(path: Path | str) -> Dict[str, Any]:
        """Load optional history, using an empty first-iteration history if absent."""

        history_path = Path(path)
        if not history_path.exists():
            return {
                "attempts": [],
                "note": "history.json was not found; treating this as the first design iteration.",
            }
        return LightweightDesignSkill._load_json_object(history_path)


def _compact_bo_history(optimization_history: Dict[str, Any], window: int = 12) -> Dict[str, Any]:
    """Reduce a BO optimization history to a prompt-sized summary."""

    records = optimization_history.get("records", []) if isinstance(optimization_history, dict) else []
    if not isinstance(records, list):
        records = []
    compact_records = []
    for record in records[-int(window):]:
        if not isinstance(record, dict):
            continue
        compact_records.append(
            {
                "evaluation": record.get("evaluation"),
                "status": record.get("status"),
                "objective": record.get("objective"),
                "objective_breakdown": record.get("objective_breakdown"),
                "validation": record.get("validation"),
                "variables": record.get("variables"),
            }
        )
    best_record = optimization_history.get("best_record") if isinstance(optimization_history, dict) else None
    return {
        "record_count": len(records),
        "best_record": best_record,
        "recent_records": compact_records,
    }
