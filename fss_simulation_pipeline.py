import argparse
import copy
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
PROJECT_ROOT = Path(__file__).resolve().parent
REBUILD_DIR = PROJECT_ROOT / "Rebuild"

DEFAULT_BGR_COLORS = {
    "black": (0, 0, 0),
    "gray": (128, 128, 128),
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "cyan": (255, 255, 0),
    "blue": (255, 0, 0),
    "purple": (128, 0, 128),
}

for path in (PROJECT_ROOT, REBUILD_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


class FSSImagePreprocessor:
    def __init__(self, output_root, detector=None, result_name="repair_fig.png"):
        self.output_root = Path(output_root)
        self.result_name = result_name
        self.detector = detector
        self.yolo_model_path = PROJECT_ROOT / "models" / "bestyolo.pt"
        self.yolo_text_labels = ("text",)
        self.yolo_text_class_id = 1
        self.yolo_conf_threshold = 0.20
        self.yolo_iou_threshold = 0.50
        self.yolo_device = "cpu"
        self.last_status = {}

    def _build_default_detector(self):
        from Rebuild.FssDetector import FSSfigDetector, TextSystemOCRAdapter

        ocr = TextSystemOCRAdapter(project_dir=str(PROJECT_ROOT))
        return FSSfigDetector(
            max_k=6,
            min_color_diff=30,
            ocr_engine=ocr,
            yolo_model_path=str(PROJECT_ROOT / "models" / "bestyolo.pt"),
        )

    def process_image(self, image_path, layer_name, col_mats, result_index=0):
        image_path = Path(image_path)
        output_folder = self.output_root / layer_name / image_path.stem

        precheck = self._precheck_yolo_text(image_path)
        if precheck.get("available") and int(precheck.get("text_detection_count", 0)) == 0:
            processed_path = output_folder / "000" / self.result_name
            self._write_passthrough_image(image_path, processed_path)
            self.last_status = {
                "source_image": str(image_path),
                "output_folder": str(output_folder),
                "result_index": 0,
                "processed_path": str(processed_path),
                "skip_fss_processing": True,
                "skip_reason": "no_text_detected_by_yolo_precheck",
                "detector_result_count": 1,
                "text_detection_count": 0,
                "yolo_detection_count": 0,
                "normalization_applied": False,
                "precheck": precheck,
            }
            print(
                "[FSSImagePreprocessor] YOLO text precheck found no text; "
                "skip full FSS detector and keep passthrough image unchanged."
            )
            return str(processed_path)

        if self.detector is None:
            self.detector = self._build_default_detector()
        results = self.detector.detect(
            str(image_path),
            output_folder=str(output_folder),
            visualize=False,
        )
        if not results:
            raise RuntimeError(f"FSS detector produced no result for image: {image_path}")

        result_index = int(result_index)
        if result_index < 0 or result_index >= len(results):
            raise IndexError(
                f"detector result_index={result_index} out of range; "
                f"available results={len(results)}"
            )

        selected_result = results[result_index] if isinstance(results[result_index], dict) else {}
        skip_fss_processing = bool(selected_result.get("skip_fss_processing", False))
        skip_reason = str(selected_result.get("skip_reason", ""))
        processed_path = output_folder / f"{result_index:03d}" / self.result_name
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed image not found: {processed_path}")

        self.last_status = {
            "source_image": str(image_path),
            "output_folder": str(output_folder),
            "result_index": result_index,
            "processed_path": str(processed_path),
            "skip_fss_processing": skip_fss_processing,
            "skip_reason": skip_reason,
            "detector_result_count": len(results),
            "text_detection_count": len(selected_result.get("text_detections", []) or []),
            "yolo_detection_count": len(selected_result.get("yolo_detections", []) or []),
            "normalization_applied": False,
            "precheck": precheck,
        }
        if skip_fss_processing:
            print(
                "[FSSImagePreprocessor] FSS processing skipped by detector; "
                f"reason={skip_reason or 'unknown'}, keep passthrough image unchanged."
            )
            return str(processed_path)

        self._normalize_repair_fig_for_simulation(processed_path, col_mats)
        self.last_status["normalization_applied"] = True
        return str(processed_path)

    def _precheck_yolo_text(self, image_path):
        import re

        model_path = Path(self.yolo_model_path)
        status = {
            "available": False,
            "model_path": str(model_path),
            "text_detection_count": 0,
            "detections": [],
            "reason": "",
        }
        if not model_path.exists():
            status["reason"] = "yolo_model_missing"
            return status

        try:
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                dll_dirs = [
                    os.path.join(sys.prefix, "Library", "bin"),
                    os.path.join(sys.prefix, "DLLs"),
                    os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
                ]
                for dll_dir in dll_dirs:
                    if os.path.isdir(dll_dir):
                        try:
                            os.add_dll_directory(dll_dir)
                        except OSError:
                            pass
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            target_map = self._resolve_yolo_class_ids(model, self.yolo_text_labels)
            if not target_map:
                status["reason"] = "text_class_not_found"
                return status

            predictions = model.predict(
                source=str(image_path),
                conf=float(self.yolo_conf_threshold),
                iou=float(self.yolo_iou_threshold),
                device=self.yolo_device,
                verbose=False,
            )
            detections = []
            for prediction in predictions:
                boxes = getattr(prediction, "boxes", None)
                if boxes is None:
                    continue
                xyxy = getattr(boxes, "xyxy", None)
                cls = getattr(boxes, "cls", None)
                conf = getattr(boxes, "conf", None)
                if xyxy is None or cls is None:
                    continue
                xyxy_values = xyxy.detach().cpu().numpy()
                cls_values = cls.detach().cpu().numpy().astype(int)
                conf_values = conf.detach().cpu().numpy() if conf is not None else []
                for index, class_id in enumerate(cls_values):
                    class_id = int(class_id)
                    if class_id not in target_map:
                        continue
                    box = xyxy_values[index]
                    detections.append(
                        {
                            "xyxy": [float(value) for value in box.tolist()],
                            "class_id": class_id,
                            "class_name": target_map[class_id],
                            "score": float(conf_values[index]) if len(conf_values) > index else 0.0,
                        }
                    )

            detections.sort(key=lambda item: (item["xyxy"][1], item["xyxy"][0]))
            status.update(
                {
                    "available": True,
                    "text_detection_count": len(detections),
                    "detections": detections,
                    "reason": "ok",
                }
            )
            return status
        except Exception as exc:
            status["reason"] = f"yolo_precheck_failed: {exc}"
            return status

    def _resolve_yolo_class_ids(self, model, target_class_names):
        normalized_targets = {
            self._normalize_yolo_label(name)
            for name in target_class_names
            if str(name).strip()
        }
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            iterable = names.items()
        elif isinstance(names, list):
            iterable = enumerate(names)
        else:
            iterable = []

        resolved = {}
        for class_id, class_name in iterable:
            normalized_name = self._normalize_yolo_label(class_name)
            if normalized_name in normalized_targets:
                resolved[int(class_id)] = normalized_name
        if "text" in normalized_targets and not resolved:
            resolved[int(self.yolo_text_class_id)] = "text"
        return resolved

    @staticmethod
    def _normalize_yolo_label(label):
        import re

        return re.sub(r"[^a-z0-9_]+", "", str(label).strip().lower())

    @staticmethod
    def _write_passthrough_image(source_path, output_path):
        import cv2

        source_path = Path(source_path)
        output_path = Path(output_path)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image for passthrough: {source_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    def _normalize_repair_fig_for_simulation(self, image_path, col_mats):
        import cv2
        import numpy as np

        pec_color_name = self._find_color_name_for_material(col_mats, "PEC")
        if pec_color_name is None:
            return

        background_color_name = self._find_first_non_pec_color_name(col_mats)
        if background_color_name is None:
            background_color_name = "white"

        pec_bgr = DEFAULT_BGR_COLORS.get(pec_color_name)
        background_bgr = DEFAULT_BGR_COLORS.get(background_color_name)
        if pec_bgr is None:
            raise ValueError(f"No default BGR color is known for PEC color name: {pec_color_name}")
        if background_bgr is None:
            raise ValueError(f"No default BGR color is known for background color name: {background_color_name}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read processed image for color normalization: {image_path}")

        border = np.concatenate([
            img[0, :, :],
            img[-1, :, :],
            img[:, 0, :],
            img[:, -1, :],
        ], axis=0)
        detected_bg = np.median(border, axis=0).astype(np.float32)
        dist_from_bg = np.linalg.norm(img.astype(np.float32) - detected_bg, axis=2)
        foreground_mask = dist_from_bg > 25.0

        if np.count_nonzero(foreground_mask) < 0.001 * foreground_mask.size:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_mask = otsu == 0
            light_mask = otsu == 255
            foreground_mask = dark_mask if np.count_nonzero(dark_mask) <= np.count_nonzero(light_mask) else light_mask

        normalized = np.empty_like(img)
        normalized[:, :] = np.asarray(background_bgr, dtype=np.uint8)
        normalized[foreground_mask] = np.asarray(pec_bgr, dtype=np.uint8)
        cv2.imwrite(str(image_path), normalized)

    def _find_color_name_for_material(self, col_mats, material_name):
        for color_name, mapped_material in col_mats.items():
            if mapped_material == material_name:
                return color_name
        return None

    def _find_first_non_pec_color_name(self, col_mats):
        for color_name, mapped_material in col_mats.items():
            if mapped_material != "PEC":
                return color_name
        return None


class InstanceDictAdapter:
    def __init__(self, image_preprocessor):
        self.image_preprocessor = image_preprocessor

    def build_simulation_instance(self, raw_instance_dict):
        instance_dict = copy.deepcopy(raw_instance_dict)

        for layer_name, layer_cfg in instance_dict["layers"].items():
            raw_img_path = layer_cfg["img_path"]
            result_index = layer_cfg.get("detector_result_index", 0)
            processed_img_path = self.image_preprocessor.process_image(
                image_path=raw_img_path,
                layer_name=layer_name,
                col_mats=layer_cfg["col_mats"],
                result_index=result_index,
            )

            layer_cfg["raw_img_path"] = raw_img_path
            layer_cfg["img_path"] = processed_img_path

        return instance_dict


class FSSSimulationPipeline:
    def __init__(self, preprocess_output_root):
        preprocessor = FSSImagePreprocessor(
            output_root=preprocess_output_root,
            result_name="repair_fig.png",
        )
        self.adapter = InstanceDictAdapter(preprocessor)

    def prepare_from_dict(self, raw_instance_dict):
        return self.adapter.build_simulation_instance(raw_instance_dict)

    def prepare_from_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            raw_instance_dict = json.load(file)
        return self.prepare_from_dict(raw_instance_dict)

    def run_from_dict(self, raw_instance_dict):
        from Simulink.Simulation import Simulation

        simulation_instance = self.prepare_from_dict(raw_instance_dict)
        simulation = Simulation(simulation_instance)
        simulation.simulation_process()
        return simulation_instance

    def run_from_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            raw_instance_dict = json.load(file)
        return self.run_from_dict(raw_instance_dict)


def write_instance_dict(instance_dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(instance_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Run FSS image cleanup before CST simulation.")
    parser.add_argument(
        "--json",
        default=str(PROJECT_ROOT / "pipeline_test_instance.json"),
        help="Input JSON whose structure matches Simulation Instance_dict.",
    )
    parser.add_argument(
        "--preprocess-output",
        default=str(PROJECT_ROOT / "fss_pipeline_outputs"),
        help="Folder used to save FSS detector outputs.",
    )
    parser.add_argument(
        "--prepared-json",
        default=None,
        help="Where to write the Instance_dict after img_path replacement.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Only clean images and write the prepared JSON; do not start CST simulation.",
    )
    args = parser.parse_args()

    pipeline = FSSSimulationPipeline(preprocess_output_root=args.preprocess_output)

    if args.preprocess_only:
        prepared_instance = pipeline.prepare_from_json(args.json)
    else:
        prepared_instance = pipeline.run_from_json(args.json)

    prepared_json = args.prepared_json
    if prepared_json is None:
        prepared_json = Path(args.preprocess_output) / "prepared_instance_dict.json"
    write_instance_dict(prepared_instance, prepared_json)
    print(f"Prepared Instance_dict written to: {prepared_json}")


if __name__ == "__main__":
    main()
