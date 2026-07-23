"""Create simulation_summary.json from a CST-exported S11 curve.

Examples:

    python -m design_agent.scripts.summarize_cst_simulation --s11-file path/to/result_s11.txt
    python -m design_agent.scripts.summarize_cst_simulation --search-dir design_agent_runs/geometry_engine_redesign_cst/03_results
    python -m design_agent.scripts.summarize_cst_simulation --s11-file s11.csv --gain 7.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from design_agent.tools.simulation_summary import (
    default_simulation_summary_path,
    load_gain_from_json,
    resolve_s11_path,
    write_simulation_summary,
)


DEFAULT_SEARCH_DIR = PROJECT_ROOT / "design_agent_runs" / "geometry_engine_redesign_cst" / "03_results"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(description="Summarize CST S11 simulation results for the design agent.")
    parser.add_argument(
        "--s11-file",
        default=None,
        help="Explicit S11 curve file exported by CST. Supports txt/csv and CST tuple-list exports.",
    )
    parser.add_argument(
        "--search-dir",
        default=None,
        help="Directory to search for the newest *s11*.txt/csv file when --s11-file is omitted.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output simulation_summary.json path. Default: design_agent_runs/agents_inputs/simulation_summary.json.",
    )
    parser.add_argument(
        "--target-resonance",
        type=float,
        default=2.45,
        help="Target resonance frequency in the same unit as the S11 curve, normally GHz.",
    )
    parser.add_argument(
        "--target-bandwidth",
        type=float,
        default=None,
        help="Target bandwidth in the same frequency unit. Optional.",
    )
    parser.add_argument(
        "--target-s11",
        type=float,
        default=-15.0,
        help="Maximum allowed S11 at target resonance in dB. Default: -15.",
    )
    parser.add_argument(
        "--target-gain",
        type=float,
        default=6.0,
        help="Minimum required gain in dBi. Default: 6.",
    )
    parser.add_argument(
        "--s11-threshold",
        type=float,
        default=None,
        help="S11 threshold in dB used to compute bandwidth. Default: same as --target-s11.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Optional gain value in dBi. S11 files do not contain gain.",
    )
    parser.add_argument(
        "--gain-json",
        default=None,
        help="Optional JSON file containing gain/gain_dbi/peak_gain_dbi.",
    )
    parser.add_argument(
        "--gain-key",
        default="gain",
        help="Preferred key to read from --gain-json. Nested fallback keys are also supported.",
    )
    parser.add_argument(
        "--resonance-limit",
        type=int,
        default=3,
        help="Maximum number of resonance frequencies to report.",
    )
    return parser


def main() -> int:
    """CLI entry point."""

    args = build_arg_parser().parse_args()
    search_dir = Path(args.search_dir) if args.search_dir else (None if args.s11_file else DEFAULT_SEARCH_DIR)
    s11_path = resolve_s11_path(s11_path=args.s11_file, search_dir=search_dir)
    output_path = Path(args.output) if args.output else default_simulation_summary_path()
    gain = resolve_gain(args.gain, args.gain_json, args.gain_key)

    summary_path = write_simulation_summary(
        output_path=output_path,
        s11_path=s11_path,
        target_resonance=args.target_resonance,
        target_bandwidth=args.target_bandwidth,
        target_s11=args.target_s11,
        target_gain=args.target_gain,
        s11_threshold=args.s11_threshold,
        gain=gain,
        resonance_limit=args.resonance_limit,
    )

    print("S11 file:", s11_path.resolve())
    print("Summary:", summary_path.resolve())
    return 0


def resolve_gain(gain: Optional[float], gain_json: Optional[str], gain_key: str) -> Optional[float]:
    """Resolve gain from explicit CLI value or JSON file."""

    if gain is not None and gain_json:
        raise ValueError("Use either --gain or --gain-json, not both.")
    if gain is not None:
        return float(gain)
    if gain_json:
        return load_gain_from_json(gain_json, key=gain_key)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
