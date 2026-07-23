"""Run the lightweight Design Agent from target Markdown and summarized JSON files.

Examples:

    python -m design_agent.scripts.run_lightweight_design_agent
    python -m design_agent.scripts.run_lightweight_design_agent --input-dir design_agent_runs/agents_inputs
    python -m design_agent.scripts.run_lightweight_design_agent --target target.md --history history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from design_agent.llm.client import OpenAICompatibleLLMClient
from design_agent.skills.lightweight_design import LightweightDesignSkill


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "design_agent_runs" / "schem"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "design_agent_runs" / "agents_inputs"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description="Generate geometry modification operations from target Markdown and summarized JSON files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory used to resolve default input filenames.",
    )
    parser.add_argument("--target", default=None, help="Path to target.md.")
    parser.add_argument("--geometry-summary", default=None, help="Path to geometry_summary.json.")
    parser.add_argument("--simulation-summary", default=None, help="Path to simulation_summary.json.")
    parser.add_argument("--history", default=None, help="Path to history.json.")
    parser.add_argument(
        "--bo-parameterization-summary",
        default=None,
        help="Optional path to bo_parameterization_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where operation_plan.json and intermediate files are saved.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.json"),
        help="Config file for the OpenAI-compatible LLM client.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Save diagnosis.json and plan.json in addition to operation_plan.json.",
    )
    return parser


def main() -> int:
    """Run the lightweight design skill."""

    args = build_arg_parser().parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "target": Path(args.target).resolve() if args.target else input_dir / "target.md",
        "geometry_summary": (
            Path(args.geometry_summary).resolve()
            if args.geometry_summary
            else input_dir / "geometry_summary.json"
        ),
        "simulation_summary": (
            Path(args.simulation_summary).resolve()
            if args.simulation_summary
            else input_dir / "simulation_summary.json"
        ),
        "history": Path(args.history).resolve() if args.history else input_dir / "history.json",
        "bo_parameterization_summary": (
            Path(args.bo_parameterization_summary).resolve()
            if args.bo_parameterization_summary
            else input_dir / "bo_parameterization_summary.json"
        ),
    }

    client = OpenAICompatibleLLMClient.from_config_file(args.config)
    skill = LightweightDesignSkill(llm_client=client)
    if args.trace:
        trace = skill.run_with_trace(load_inputs(paths))
        write_json(output_dir / "diagnosis.json", trace["diagnosis"])
        write_json(output_dir / "plan.json", trace["plan"])
        result = trace["result"]
    else:
        result = skill.run_from_files(
            target_path=paths["target"],
            geometry_summary_path=paths["geometry_summary"],
            simulation_summary_path=paths["simulation_summary"],
            history_path=paths["history"],
            bo_parameterization_summary_path=paths["bo_parameterization_summary"],
        )

    result_path = output_dir / "operation_plan.json"
    write_json(result_path, result)

    print("Saved operation plan: {0}".format(result_path))
    print("Operations:")
    for operation in result.get("operations", []):
        print("  {0}".format(operation.get("operation")))
    return 0


def load_inputs(paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    """Load the summarized input objects for trace mode."""

    return {
        "target": load_target_object(paths["target"]),
        "geometry_summary": load_json_object(paths["geometry_summary"]),
        "simulation_summary": load_json_object(paths["simulation_summary"]),
        "history": load_history_object(paths["history"]),
        **optional_bo_summary(paths["bo_parameterization_summary"]),
    }


def load_json_object(path: Path) -> Dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object in {0}".format(path))
    return payload


def load_target_object(path: Path) -> Dict[str, Any]:
    """Load target.md and wrap it as structured input for the prompt."""

    if path.suffix.lower() == ".md":
        return {
            "format": "markdown",
            "source": str(path),
            "content": path.read_text(encoding="utf-8"),
        }
    return load_json_object(path)


def load_history_object(path: Path) -> Dict[str, Any]:
    """Load optional history.json, defaulting to empty history for iteration one."""

    if not path.exists():
        return {
            "attempts": [],
            "note": "history.json was not found; treating this as the first design iteration.",
        }
    return load_json_object(path)


def optional_bo_summary(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load optional BO parameterization summary for prompt context."""

    if not path.exists() or path.stat().st_size == 0:
        return {}
    return {"bo_parameterization_summary": load_json_object(path)}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON object to disk."""

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
