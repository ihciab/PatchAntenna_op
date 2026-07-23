"""Core data models shared by the design agent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class DesignSpecification:
    """Target antenna requirements supplied by a user or upstream optimizer.

    Attributes:
        center_frequency_hz: Desired center frequency in Hz.
        bandwidth_hz: Desired operating bandwidth in Hz.
        max_size_mm: Optional maximum antenna envelope as ``(x, y, z)`` in mm.
        substrate: Optional substrate or stack-up description.
        performance_targets: Additional target metrics such as S11, gain, or
            radiation efficiency.
        constraints: Additional fabrication, material, or simulation constraints.
    """

    center_frequency_hz: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    max_size_mm: Optional[Tuple[float, float, float]] = None
    substrate: Optional[str] = None
    performance_targets: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AntennaTopology:
    """High-level antenna topology selected before geometric initialization."""

    name: str
    rationale: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeometryCandidate:
    """Parameterized antenna geometry candidate.

    The framework intentionally stores geometry as generic structured data so
    future CST builders, CAD kernels, or JSON schemas can be plugged in without
    changing the agent interface.
    """

    topology: Optional[AntennaTopology] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationRequest:
    """Request object passed to an external electromagnetic simulator."""

    geometry: GeometryCandidate
    settings: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None


@dataclass
class SimulationResult:
    """Raw or lightly parsed result returned by CST or another simulator."""

    success: bool = False
    result_path: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Performance evaluation derived from simulation outputs."""

    score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    passed: Optional[bool] = None
    notes: str = ""


@dataclass
class OptimizationSuggestion:
    """Design update proposed by Bayesian optimization or an LLM reflection step."""

    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    source: str = ""
