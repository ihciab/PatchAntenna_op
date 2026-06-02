from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"
DEFAULT_CST_LIBRARY_PATH = Path(r"D:\CST Studio Suite 2023\AMD64\python_cst_libraries")
ENV_CST_LIBRARY_PATH = "CST_PYTHON_LIBRARIES"


def _candidate_paths() -> List[Path]:
    candidates: List[Path] = []
    config_value: Optional[str] = None
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as file:
                config = json.load(file)
            config_value = config.get("cstModuleBase", {}).get("absolute_path")
        except (OSError, json.JSONDecodeError):
            config_value = None

    for value in (os.environ.get(ENV_CST_LIBRARY_PATH), config_value, str(DEFAULT_CST_LIBRARY_PATH)):
        if not value:
            continue
        path = Path(value)
        if path not in candidates:
            candidates.append(path)
    return candidates


def ensure_cst_library_path() -> Path:
    checked: List[str] = []
    for path in _candidate_paths():
        checked.append(str(path))
        if not path.exists():
            continue
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
        try:
            import cst  # noqa: F401
        except ModuleNotFoundError:
            continue
        return path

    raise ModuleNotFoundError(
        "Unable to import CST Python libraries. Checked paths: " + ", ".join(checked)
    )

