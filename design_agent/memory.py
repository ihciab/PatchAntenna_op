"""Memory store for design iterations and future retrieval workflows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from design_agent.state import DesignState


class DesignMemory:
    """Append-only memory for design attempts.

    This class is intentionally lightweight. It keeps a structured history today
    and can later be backed by vector stores, databases, or document indexes for
    reflection and retrieval-augmented generation.
    """

    def __init__(self) -> None:
        """Create an empty design memory."""

        self._records: List[Dict[str, Any]] = []

    def add_record(self, state: DesignState, note: str = "") -> None:
        """Save a snapshot of the current design state.

        Args:
            state: Current workflow state to snapshot.
            note: Optional human-readable context for the saved record.
        """

        self._records.append({"state": deepcopy(state), "note": note})

    def list_records(self) -> List[Dict[str, Any]]:
        """Return all stored design records."""

        return list(self._records)

    def latest(self) -> Optional[Dict[str, Any]]:
        """Return the newest design record if one exists."""

        if not self._records:
            return None
        return self._records[-1]

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search design memory for records relevant to ``query``.

        The default implementation is a placeholder for future RAG support.
        """

        raise NotImplementedError("DesignMemory.search is reserved for future RAG support.")
