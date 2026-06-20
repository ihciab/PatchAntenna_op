from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayesian_optimization.simulation.cst_library_path import CONFIG_PATH, ensure_cst_library_path


def _load_root_config() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Root config must be a JSON object: {CONFIG_PATH}")
    return config


config_data = _load_root_config()
ensure_cst_library_path()

file_handler_config = config_data.get("file_handler", {})
if not isinstance(file_handler_config, dict):
    file_handler_config = {}
keep_project = str(file_handler_config.get("keep_project", "False")).strip().lower() == "true"


paper_dict = {
    "Analysis_of_Scanning_Error_of_Array_Antenna_by_FSS_Radome.pdf": {
        "Col_mats": {
            "white": "FR4",
            "green": "PEC",
        },
        "FSS_package": {"X": 23, "Y": 23, "Z": 0.5, "f0": 0.5, "f1": 3},
    },
    "A_Miniaturized_Dual-Band_FSS_With_Controllable_Frequency_Resonances.pdf": {
        "Col_mats": {
            "white": "FR4",
            "gray": "PEC",
        },
        "FSS_package": {"X": 6.6, "Y": 6.6, "Z": 1, "f0": 0.5, "f1": 11},
    },
    "A_FSS-_based_Single_Layer_Reflective_Polarizer_for_X_Ku-_Bands.pdf": {
        "Col_mats": {
            "red": "PEC",
            "orange": "FR4",
        },
        "FSS_package": {"X": 8.8, "Y": 8.8, "Z": 0.5, "f0": 0.5, "f1": 18},
    },
}


import Process.AutoSimulation
import Process.SweepFSS


class UIAutoFss:
    def __init__(self) -> None:
        print("########## FSS auto simulation platform ##########")
        print("Enter one or more PDF paths separated by commas.")
        image_path = input("Type 'show' to run the demo: ").strip()

        if image_path == "show":
            Process.SweepFSS.main()
            return

        paper_list = [item.strip() for item in image_path.split(",") if item.strip()]
        folder_path = Path(str(file_handler_config.get("cst_proj", PROJECT_ROOT / "cst_projects")))

        if not keep_project and folder_path.exists():
            if folder_path.is_dir():
                shutil.rmtree(folder_path)
                print(f"Deleted existing CST project folder: {folder_path}")
            else:
                print(f"Configured CST project path exists but is not a folder: {folder_path}")

        for index, paper in enumerate(paper_list):
            paper_name = Path(paper).name
            if paper_name not in paper_dict:
                available = ", ".join(sorted(paper_dict.keys()))
                raise KeyError(f"No preset FSS settings for `{paper_name}`. Available presets: {available}")

            folder_path_temp = folder_path / str(index)
            proj_name = "FSS.cst"
            preset = paper_dict[paper_name]
            fss_package = preset["FSS_package"]
            Process.AutoSimulation.main(
                paper,
                str(folder_path_temp),
                proj_name,
                float(fss_package["f0"]),
                float(fss_package["f1"]),
                col_mats1=preset["Col_mats"],
                fss=fss_package,
            )


if __name__ == "__main__":
    UIAutoFss()
