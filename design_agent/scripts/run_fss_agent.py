from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_PROJECT_ROOT = next(
    path
    for path in (
        PROJECT_ROOT / "PDF_analy_agent",
        PROJECT_ROOT / "PDFanalyagent",
        PROJECT_ROOT / "test3",
    )
    if (path / "agent").exists()
)
AGENT_ROOT = AGENT_PROJECT_ROOT / "agent"


def configure_agent_paths() -> None:
    versioned_deps = AGENT_PROJECT_ROOT / f".deps_py{sys.version_info.major}{sys.version_info.minor}"
    paths = [path for path in (versioned_deps, AGENT_PROJECT_ROOT / ".deps", AGENT_ROOT) if path.exists()]
    for path in reversed(paths):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


configure_agent_paths()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PDF analysis FSS prompt/model extraction from the project root.")
    parser.add_argument(
        "--root",
        default=str(AGENT_PROJECT_ROOT / "FSS PDF"),
        help="Root directory containing FSS paper folders, or a single paper folder.",
    )
    parser.add_argument("--paper-concurrency", type=int, default=None, help="Parallel paper worker count.")
    parser.add_argument("--skip-postgres", action="store_true", default=True, help="Compatibility flag; default on.")
    parser.add_argument("--skip-nebula", action="store_true", default=True, help="Compatibility flag; default on.")
    parser.add_argument("--with-postgres", action="store_true", help="Persist extracted data to PostgreSQL.")
    parser.add_argument("--with-nebula", action="store_true", help="Persist extracted data to NebulaGraph.")
    return parser


async def main() -> None:
    args = build_arg_parser().parse_args()
    from common.config import AgentConfig
    from fss_agent.agent import FSSExtractionAgent

    config = AgentConfig()
    config.persist_to_postgres = bool(args.with_postgres)
    config.persist_to_nebula = bool(args.with_nebula)
    if args.paper_concurrency is not None:
        config.paper_concurrency = args.paper_concurrency

    agent = FSSExtractionAgent(Path(args.root), config=config)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc
