"""Run the antenna design agent with a selectable prompt file.

Examples:

    python -m design_agent.scripts.run_antenna_agent --list-prompts
    python -m design_agent.scripts.run_antenna_agent --prompt initial_design
    python -m design_agent.scripts.run_antenna_agent --prompt design_agent/skills/prompts/initial_design.md
    python -m design_agent.scripts.run_antenna_agent --prompt initial_design --extra-instruction "中心频率 2.45GHz"

This script is a debugging entry point for prompt/API validation. It does not
run CST or Bayesian optimization yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = PROJECT_ROOT / "design_agent" / "skills" / "prompts"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from design_agent.llm.client import OpenAICompatibleLLMClient
from design_agent.llm.parser import LLMResponseParser


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for prompt-based agent debugging."""

    parser = argparse.ArgumentParser(
        description="Run the antenna design LLM agent with a prompt selected from design_agent/skills/prompts/."
    )
    parser.add_argument(
        "--prompt",
        default="initial_design",
        help=(
            "Prompt name or path. Examples: initial_design, initial_design.md, "
            "or design_agent/skills/prompts/initial_design.md."
        ),
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt files and exit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where raw response and parsed JSON artifacts will be saved.",
    )
    parser.add_argument(
        "--spec-file",
        default=None,
        help="Optional JSON file containing target antenna specifications.",
    )
    parser.add_argument(
        "--spec-json",
        default=None,
        help="Optional inline JSON string containing target antenna specifications.",
    )
    parser.add_argument(
        "--extra-instruction",
        default=None,
        help="Optional extra user instruction appended after the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature for the debug run.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--no-parse-json",
        action="store_true",
        help="Only save raw_llm_response.txt and skip JSON object parsing.",
    )
    return parser


def list_prompts() -> None:
    """Print prompt files available in the repository prompt directory."""

    print("Available prompts:")
    for path in sorted(PROMPT_DIR.glob("*.md")):
        print("  {0}".format(path.stem))


def resolve_prompt_path(prompt: str) -> Path:
    """Resolve a prompt name or path to an existing markdown file."""

    candidate = Path(prompt)
    if candidate.exists():
        return candidate.resolve()

    if candidate.suffix == "":
        candidate = candidate.with_suffix(".md")

    prompt_path = PROMPT_DIR / candidate.name
    if prompt_path.exists():
        return prompt_path.resolve()

    raise FileNotFoundError("Prompt file not found: {0}".format(prompt))


def default_output_dir(prompt_path: Path) -> Path:
    """Return the default artifact directory for a prompt."""

    if prompt_path.stem == "initial_design":
        return PROJECT_ROOT / "design_agent_runs" / "initial_design_test"
    return PROJECT_ROOT / "design_agent_runs" / "{0}_test".format(prompt_path.stem)


def load_specification(spec_file: Optional[str], spec_json: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load optional antenna specification data from a file or inline JSON."""

    if spec_file and spec_json:
        raise ValueError("Use either --spec-file or --spec-json, not both.")
    if spec_file:
        return json.loads(Path(spec_file).read_text(encoding="utf-8"))
    if spec_json:
        return json.loads(spec_json)
    return None


def build_prompt(
    prompt_path: Path,
    specification: Optional[Dict[str, Any]],
    extra_instruction: Optional[str],
) -> str:
    """Build the final LLM prompt from a prompt template and debug inputs."""

    prompt = prompt_path.read_text(encoding="utf-8")

    if specification is not None:
        prompt += (
            "\n\n==================================================\n"
            "USER SPECIFICATION\n"
            "==================================================\n"
            + json.dumps(specification, indent=2, ensure_ascii=False)
        )

    if extra_instruction:
        prompt += (
            "\n\n==================================================\n"
            "EXTRA USER INSTRUCTION\n"
            "==================================================\n"
            + extra_instruction
        )

    return prompt


def artifact_names_for_prompt(prompt_path: Path, objects: List[Dict[str, Any]]) -> List[str]:
    """Return stable output filenames for parsed JSON objects."""

    if prompt_path.stem == "initial_design" and len(objects) >= 3:
        return ["design_trace.json", "stackup.json", "patch.json"]
    return ["response_{0:02d}.json".format(index + 1) for index in range(len(objects))]


def save_response(
    output_dir: Path,
    prompt_path: Path,
    raw_response: str,
    objects: Optional[List[Dict[str, Any]]] = None,
) -> List[Path]:
    """Save raw LLM response and optional parsed JSON objects."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    raw_path = output_dir / "raw_llm_response.txt"
    raw_path.write_text(raw_response, encoding="utf-8")
    saved_paths.append(raw_path)

    if objects:
        for filename, content in zip(artifact_names_for_prompt(prompt_path, objects), objects):
            path = output_dir / filename
            path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
            saved_paths.append(path)

    return saved_paths


def main() -> None:
    """Run one prompt against the configured LLM backend and save artifacts."""

    args = build_arg_parser().parse_args()

    if args.list_prompts:
        list_prompts()
        return

    prompt_path = resolve_prompt_path(args.prompt)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(prompt_path)
    specification = load_specification(args.spec_file, args.spec_json)
    prompt = build_prompt(prompt_path, specification, args.extra_instruction)

    client = OpenAICompatibleLLMClient.from_config_file(str(PROJECT_ROOT / "config.json"))
    response = client.generate(
        prompt,
        context={
            "system_prompt": "You are a deterministic RF antenna design agent. Return machine-readable output.",
            "temperature": args.temperature,
            "timeout": args.timeout,
        },
    )

    objects: Optional[List[Dict[str, Any]]] = None
    if not args.no_parse_json:
        objects = LLMResponseParser().parse_json_objects(response)

    saved_paths = save_response(output_dir, prompt_path, response, objects)
    print("Prompt: {0}".format(prompt_path))
    print("Output directory: {0}".format(output_dir))
    print("Saved files:")
    for path in saved_paths:
        print("  {0}".format(path))


if __name__ == "__main__":
    main()
