from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.json"
CST_CONFIG_SECTION = "cstModuleBase"
CST_CONFIG_KEY = "absolute_path"


def _load_root_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"CST config file does not exist: {CONFIG_PATH}")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"CST config file is not valid JSON: {CONFIG_PATH}",
            exc.doc,
            exc.pos,
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(f"CST config root must be a JSON object: {CONFIG_PATH}")
    return config


def configured_cst_library_path() -> Path:
    """Return the CST Python library path from the repo root config.json."""

    config = _load_root_config()
    section = config.get(CST_CONFIG_SECTION)
    if not isinstance(section, dict):
        raise KeyError(
            f"Missing `{CST_CONFIG_SECTION}` object in CST config: {CONFIG_PATH}"
        )

    value = section.get(CST_CONFIG_KEY)
    if not isinstance(value, str) or not value.strip():
        raise KeyError(
            f"Missing `{CST_CONFIG_SECTION}.{CST_CONFIG_KEY}` in CST config: {CONFIG_PATH}"
        )

    return Path(value).expanduser()


def ensure_cst_library_path() -> Path:
    cst_library_path = configured_cst_library_path()
    if not cst_library_path.exists():
        raise ModuleNotFoundError(
            "Configured CST Python library path does not exist. "
            f"Update `{CST_CONFIG_SECTION}.{CST_CONFIG_KEY}` in {CONFIG_PATH}: "
            f"{cst_library_path}"
        )

    path_text = str(cst_library_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

    try:
        import cst  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Unable to import CST Python libraries from configured path. "
            f"Check `{CST_CONFIG_SECTION}.{CST_CONFIG_KEY}` in {CONFIG_PATH}: "
            f"{cst_library_path}"
        ) from exc
    return cst_library_path
