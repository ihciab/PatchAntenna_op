import json
import sys
from pathlib import Path

__version__ = "0.7"
__author__ = "Zhijin momo Chen"

_config = None
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

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
    _config = _config_data["cstModuleBase"]["absolute_path"]
    print(_config)
    try:
        sys.path.append(_config)
        import cst  # noqa: F401
    except ModuleNotFoundError:
        print("python_cst_libraries was not found")
        cst_lib_path = input(
            "Please enter the absolute path to python_cst_libraries: "
        )
        try:
            sys.path.append(cst_lib_path)
            import cst  # noqa: F401
        except ModuleNotFoundError:
            print("python_cst_libraries was not found")
            input("Press any key to exit...")
        else:
            print("Updating config file")
            _config_data["cstModuleBase"]["absolute_path"] = cst_lib_path
            with CONFIG_PATH.open("w", encoding="utf-8") as f:
                json.dump(_config_data, f, indent=2, ensure_ascii=False)
            print("Imported cst module")

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
