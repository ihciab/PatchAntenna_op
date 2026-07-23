"""Closed-loop design orchestration exposed through the design_agent package.

This module is the package-level entry point for the lightweight closed-loop
workflow. The current implementation reuses the package CLI runner implementation
under ``design_agent.scripts``.
"""

from __future__ import annotations

from design_agent.scripts.run_closed_loop_design import (
    AGENT_INPUTS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_RUN_DIR,
    ClosedLoopDesignRunner,
    operations_to_dsl,
    patch_from_geometry_json,
    patch_from_geometry_summary,
)


__all__ = [
    "AGENT_INPUTS_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_RUN_DIR",
    "ClosedLoopDesignRunner",
    "operations_to_dsl",
    "patch_from_geometry_json",
    "patch_from_geometry_summary",
]
