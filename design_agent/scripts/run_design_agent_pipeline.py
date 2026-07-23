"""Run the top-level design-agent pipeline in numbered run folders."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from design_agent.llm.client import OpenAICompatibleLLMClient
from design_agent.pipeline import (
    DEFAULT_AGENT_CONFIG_PATH,
    DesignAgentPipelineRunner,
    load_pipeline_config,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the top-level pipeline."""

    parser = argparse.ArgumentParser(description="Run the design-agent top-level pipeline.")
    parser.add_argument(
        "--agent-config",
        default=str(DEFAULT_AGENT_CONFIG_PATH),
        help="Path to design_agent/agent_config.json.",
    )
    parser.add_argument("--run-index", type=int, default=None, help="Temporary override for run_folder.run_index.")
    parser.add_argument("--run-prefix", default=None, help="Temporary override for run_folder.run_prefix.")
    parser.add_argument("--iterations", type=int, default=None, help="Temporary override for closed_loop.iterations.")
    parser.add_argument("--build-only", action="store_true", help="Temporary override: closed_loop.build_only=true.")
    parser.add_argument("--geometry-only", action="store_true", help="Temporary override: closed_loop.geometry_only=true.")
    parser.add_argument("--skip-bo-prepare", action="store_true", help="Temporary override: bo.prepare_bo=false.")
    parser.add_argument("--execute-bo", action="store_true", help="Temporary override: bo.execute_bo=true.")
    return parser


def main() -> int:
    """Run the pipeline and print the resulting manifest path."""

    args = build_arg_parser().parse_args()
    config = load_pipeline_config(args.agent_config)
    overrides = {}
    if args.run_index is not None:
        overrides["run_index"] = args.run_index
    if args.run_prefix is not None:
        overrides["run_prefix"] = args.run_prefix
    if args.iterations is not None:
        overrides["iterations"] = args.iterations
    if args.build_only:
        overrides["build_only"] = True
    if args.geometry_only:
        overrides["geometry_only"] = True
    if args.skip_bo_prepare:
        overrides["prepare_bo"] = False
    if args.execute_bo:
        overrides["execute_bo"] = True
    if overrides:
        config = replace(config, **overrides)

    client = OpenAICompatibleLLMClient.from_config_file(str(config.config_path))
    result = DesignAgentPipelineRunner(config=config, llm_client=client).run()
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
