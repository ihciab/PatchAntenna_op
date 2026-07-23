"""Create geometry_summary.json from Geometry Engine JSON.

Examples:

    python -m design_agent.scripts.summarize_geometry_json --geometry-json design_agent_runs/redesign_test/geometry_engine_redesign.json
    python -m design_agent.scripts.summarize_geometry_json --geometry-json geometry.json --target-frequency 2.45 --epsilon-r 2.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from design_agent.tools.geometry_summary import default_geometry_summary_path, write_geometry_summary


DEFAULT_GEOMETRY_JSON = PROJECT_ROOT / "design_agent_runs" / "redesign_test" / "geometry_engine_redesign.json"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Summarize Geometry Engine antenna geometry JSON.")
    parser.add_argument(
        "--geometry-json",
        default=str(DEFAULT_GEOMETRY_JSON),
        help="Path to geometry_engine_geometry_v1 JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Default: design_agent_runs/agents_inputs/geometry_summary.json.",
    )
    parser.add_argument(
        "--target-frequency",
        type=float,
        default=2.45,
        help="Target frequency in GHz for electrical-size calculation.",
    )
    parser.add_argument(
        "--epsilon-r",
        type=float,
        default=None,
        help="Optional substrate epsilon_r for guided-wavelength electrical size.",
    )
    return parser


def main() -> int:
    """CLI entry point."""

    args = build_arg_parser().parse_args()
    geometry_json = Path(args.geometry_json)
    output = Path(args.output) if args.output else default_geometry_summary_path()
    summary_path = write_geometry_summary(
        geometry_json_path=geometry_json,
        output_path=output,
        target_frequency_ghz=args.target_frequency,
        epsilon_r=args.epsilon_r,
    )
    print("Geometry JSON:", geometry_json.resolve())
    print("Geometry summary:", summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
