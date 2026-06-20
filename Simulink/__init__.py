import sys
from pathlib import Path

__version__ = "0.7"
__author__ = "Zhijin momo Chen"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
PROJECT_ROOT = CONFIG_PATH.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesian_optimization.simulation.cst_library_path import ensure_cst_library_path


_config = str(ensure_cst_library_path())

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
