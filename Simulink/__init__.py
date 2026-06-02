import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

__version__ = "0.7"
__author__ = "Zhijin momo Chen"

_config = None
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_CST_LIBRARY_PATH = Path(r"D:\CST Studio Suite 2023\AMD64\python_cst_libraries")
ENV_CST_LIBRARY_PATH = "CST_PYTHON_LIBRARIES"


def _candidate_cst_library_paths(config_path: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    for value in (os.environ.get(ENV_CST_LIBRARY_PATH), config_path, str(DEFAULT_CST_LIBRARY_PATH)):
        if not value:
            continue
        path = Path(value)
        if path not in paths:
            paths.append(path)
    return paths


def _import_cst_from_config(config_data: Dict) -> str:
    checked_paths: List[str] = []
    config_path = config_data.get("cstModuleBase", {}).get("absolute_path")

    for cst_path in _candidate_cst_library_paths(config_path):
        checked_paths.append(str(cst_path))
        if not cst_path.exists():
            continue
        cst_path_text = str(cst_path)
        if cst_path_text not in sys.path:
            sys.path.insert(0, cst_path_text)
        try:
            import cst  # noqa: F401
        except ModuleNotFoundError:
            continue

        config_data.setdefault("cstModuleBase", {})["absolute_path"] = cst_path_text
        if config_path != cst_path_text:
            with CONFIG_PATH.open("w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        return cst_path_text

    raise ModuleNotFoundError(
        "python_cst_libraries was not found. Checked paths: "
        + ", ".join(checked_paths)
    )

try:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        _config_data = json.load(f)
except FileNotFoundError as exc:
    raise FileNotFoundError("Error: JSON config file does not exist") from exc
except json.JSONDecodeError as exc:
    raise json.JSONDecodeError(
        "Error: JSON config format is invalid",
        exc.doc,
        exc.pos,
    ) from exc
else:
    _config = _import_cst_from_config(_config_data)

from .handle import *

__all__ = [
    "cst_auto_init",
    "cst_change_solver",
    "cst_close_project",
    "cst_create_brick",
    "cst_create_project",
    "cst_create_material",
    "cst_curves",
    "cst_define_floquetport",
    "cst_del_curves",
    "cst_del_subtract",
    "cst_del_solid",
    "cst_excitation2signal",
    "cst_excitation2use",
    "cst_export_pic",
    "cst_extrudecurve",
    "cst_heal_all_shapes",
    "cst_import_stl",
    "cst_load_material",
    "cst_open_project",
    "cst_scale",
    "cst_set_background",
    "cst_set_boundaries",
    "cst_set_frequency",
    "cst_set_material",
    "cst_set_monitor2filed",
    "cst_solid_rename",
    "cst_set_planewave",
    "cst_set_solver2f",
    "cst_spline_curves",
    "cst_straight_line",
    "cst_translate",
    "Simulation",
]
