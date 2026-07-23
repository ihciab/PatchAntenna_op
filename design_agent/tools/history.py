"""History utilities for closed-loop design iterations."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


HISTORY_SCHEMA_VERSION = "design_agent_closed_loop_history_v1"


def empty_history() -> Dict[str, Any]:
    """Return a new closed-loop history object."""

    return {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "attempts": [],
    }


def load_history(path: Path | str) -> Dict[str, Any]:
    """Load ``history.json`` or return an empty first-iteration history."""

    history_path = Path(path)
    if not history_path.exists():
        return empty_history()
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object in history file: {0}".format(history_path))
    payload.setdefault("schema_version", HISTORY_SCHEMA_VERSION)
    if not isinstance(payload.get("attempts"), list):
        payload["attempts"] = []
    return payload


def write_history(path: Path | str, history: Dict[str, Any]) -> Path:
    """Write ``history.json`` to disk."""

    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return history_path


def next_iteration_number(history: Dict[str, Any]) -> int:
    """Return the next one-based iteration number from a history object."""

    attempts = history.get("attempts", [])
    if not isinstance(attempts, list) or not attempts:
        return 1
    values = []
    for item in attempts:
        if isinstance(item, dict):
            try:
                values.append(int(item.get("iteration")))
            except (TypeError, ValueError):
                pass
    return (max(values) + 1) if values else len(attempts) + 1


def append_history_record(history: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """Append one iteration record and refresh top-level history metadata."""

    attempts = history.setdefault("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
        history["attempts"] = attempts
    attempts.append(record)
    history["schema_version"] = HISTORY_SCHEMA_VERSION
    history["latest_iteration"] = record.get("iteration")
    return refresh_history_knowledge(history)


def refresh_history_knowledge(history: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh derived top-level history fields used by the LLM planner."""

    history["schema_version"] = HISTORY_SCHEMA_VERSION
    history["recent_operation_pattern"] = summarize_recent_operation_pattern(history)
    history["electromagnetic_knowledge"] = summarize_electromagnetic_knowledge(history)
    return history


def build_closed_loop_history_record(
    iteration: int,
    iteration_dir: Path | str,
    operation_plan: Dict[str, Any],
    geometry_result: Dict[str, Any],
    cst_result: Dict[str, Any],
    geometry_summary_path: Path | str,
    simulation_summary_path: Optional[Path | str],
    before_simulation_summary: Dict[str, Any],
    after_simulation_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build one persistent history record for a closed-loop iteration."""

    iteration_path = Path(iteration_dir)
    geometry_summary = Path(geometry_summary_path)
    simulation_summary = None if simulation_summary_path is None else Path(simulation_summary_path)
    return {
        "iteration": int(iteration),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "operation_plan": operation_plan,
        "dsl_commands": geometry_result.get("dsl_commands", []),
        "geometry_engine": {
            "geometry_json": geometry_result.get("geometry_json"),
            "execution_log": geometry_result.get("execution_log", []),
        },
        "cst": cst_result,
        "summaries": {
            "geometry_summary": str(geometry_summary.resolve()),
            "simulation_summary": None if simulation_summary is None else str(simulation_summary.resolve()),
        },
        "effect": summarize_effect(before_simulation_summary, after_simulation_summary),
        "artifact_dir": str(iteration_path.resolve()),
    }


def summarize_effect(before: Dict[str, Any], after: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize whether the latest iteration improved key simulation metrics."""

    if after is None:
        return {
            "available": False,
            "reason": "No new simulation summary was generated.",
        }

    before_current = before.get("current", {}) if isinstance(before.get("current"), dict) else {}
    after_current = after.get("current", {}) if isinstance(after.get("current"), dict) else {}
    before_gap = before.get("gap_to_target", {}) if isinstance(before.get("gap_to_target"), dict) else {}
    after_gap = after.get("gap_to_target", {}) if isinstance(after.get("gap_to_target"), dict) else {}

    s11_delta = numeric_delta(before_current.get("s11_at_target_db"), after_current.get("s11_at_target_db"))
    freq_error_abs_delta = numeric_delta_abs(
        before_gap.get("frequency_error_ghz"),
        after_gap.get("frequency_error_ghz"),
    )
    gain_delta = numeric_delta(before_current.get("gain_dbi"), after_current.get("gain_dbi"))

    return {
        "available": True,
        "s11_at_target_delta_db": s11_delta,
        "s11_improved": None if s11_delta is None else s11_delta < 0.0,
        "abs_frequency_error_delta_ghz": freq_error_abs_delta,
        "frequency_improved": None if freq_error_abs_delta is None else freq_error_abs_delta < 0.0,
        "gain_delta_dbi": gain_delta,
        "gain_improved": None if gain_delta is None else gain_delta > 0.0,
        "passed_after": after.get("passed", {}),
    }


def summarize_recent_operation_pattern(history: Dict[str, Any], window: int = 5) -> Dict[str, Any]:
    """Summarize recent operation usage for the LLM planner."""

    attempts = history.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    recent = [item for item in attempts[-int(window):] if isinstance(item, dict)]
    operation_names: List[str] = []
    operation_sequences: List[List[str]] = []
    improved_count = 0
    available_effect_count = 0

    for attempt in recent:
        plan = attempt.get("operation_plan", {})
        operations = plan.get("operations", []) if isinstance(plan, dict) else []
        sequence = []
        if isinstance(operations, list):
            for operation in operations:
                if isinstance(operation, dict):
                    name = str(operation.get("operation", ""))
                    if name:
                        sequence.append(name)
                        operation_names.append(name)
        operation_sequences.append(sequence)

        effect = attempt.get("effect", {})
        if isinstance(effect, dict) and effect.get("available"):
            available_effect_count += 1
            if effect.get("s11_improved") or effect.get("frequency_improved") or effect.get("gain_improved"):
                improved_count += 1

    add_count = operation_names.count("AddSlot")
    delete_count = operation_names.count("DeleteSlot")
    resize_count = operation_names.count("ResizePatch")
    move_count = operation_names.count("MoveFeed")
    oscillation = add_count > 0 and delete_count > 0
    weak_recent_improvement = available_effect_count >= 3 and improved_count <= 1

    recommendations: List[str] = []
    if weak_recent_improvement:
        recommendations.append("Recent iterations show weak improvement; use a coordinated multi-operation geometry experiment.")
    if oscillation:
        recommendations.append("Recent history contains both AddSlot and DeleteSlot; avoid simple add/delete oscillation.")
    if resize_count + move_count >= max(3, len(operation_names) // 2):
        recommendations.append("ResizePatch dominates recent attempts; consider a different geometry hypothesis.")

    return {
        "window": int(window),
        "operation_counts": {
            "ResizePatch": resize_count,
            "MoveFeed": move_count,
            "AddSlot": add_count,
            "DeleteSlot": delete_count,
        },
        "operation_sequences": operation_sequences,
        "weak_recent_improvement": weak_recent_improvement,
        "add_delete_oscillation": oscillation,
        "recommendations": recommendations,
    }


def summarize_electromagnetic_knowledge(history: Dict[str, Any], window: int = 8) -> Dict[str, Any]:
    """Collect recent LLM effect reflections into a compact planning memory."""

    attempts = history.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    recent = [item for item in attempts[-int(window):] if isinstance(item, dict)]
    lessons: List[str] = []
    guidance: List[str] = []
    avoid: List[str] = []
    summaries: List[Dict[str, Any]] = []

    for attempt in recent:
        reflection = attempt.get("llm_effect_summary", {})
        if not isinstance(reflection, dict) or not reflection.get("available"):
            continue
        summary = str(reflection.get("summary", "")).strip()
        if summary:
            summaries.append(
                {
                    "iteration": attempt.get("iteration"),
                    "summary": summary,
                }
            )
        _extend_unique(lessons, reflection.get("electromagnetic_lessons"))
        _extend_unique(guidance, reflection.get("next_iteration_guidance"))
        _extend_unique(avoid, reflection.get("avoid_repeating"))

    bo_reflection = history.get("bo_effect_summary")
    if isinstance(bo_reflection, dict) and bo_reflection.get("available"):
        summary = str(bo_reflection.get("summary", "")).strip()
        if summary:
            summaries.append(
                {
                    "iteration": "bo",
                    "summary": summary,
                }
            )
        _extend_unique(lessons, bo_reflection.get("electromagnetic_lessons"))
        _extend_unique(guidance, bo_reflection.get("next_iteration_guidance"))
        _extend_unique(avoid, bo_reflection.get("avoid_repeating"))

    return {
        "window": int(window),
        "recent_summaries": summaries[-int(window):],
        "lessons": lessons[-20:],
        "next_iteration_guidance": guidance[-20:],
        "avoid_repeating": avoid[-20:],
    }


def _extend_unique(target: List[str], values: Any) -> None:
    """Append unique non-empty strings from a string or list."""

    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, list):
        candidates = values
    else:
        candidates = []
    existing = set(target)
    for value in candidates:
        text = str(value).strip()
        if text and text not in existing:
            target.append(text)
            existing.add(text)


def numeric_delta(before: Any, after: Any) -> Optional[float]:
    """Return after - before when both values are finite numbers."""

    before_number = number_or_none(before)
    after_number = number_or_none(after)
    if before_number is None or after_number is None:
        return None
    return round(after_number - before_number, 6)


def numeric_delta_abs(before: Any, after: Any) -> Optional[float]:
    """Return abs(after) - abs(before) when both values are finite numbers."""

    before_number = number_or_none(before)
    after_number = number_or_none(after)
    if before_number is None or after_number is None:
        return None
    return round(abs(after_number) - abs(before_number), 6)


def number_or_none(value: Any) -> Optional[float]:
    """Return a finite float or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
