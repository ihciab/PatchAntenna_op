import os

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from tools.infer import predict_system
import tools.infer.pytorchocr_utility as ocr_utility


class TextSystemOCRAdapter:
    """
    复用 `line_build.py` 中的 TextSystem 推理方式，
    对外保持 `predict(...)` 接口，便于 FSS 检测流程无缝切换 OCR 实现。
    """

    def __init__(
        self,
        project_dir=None,
        use_gpu=False,
        det_yaml_path=None,
        det_model_path=None,
        rec_yaml_path=None,
        rec_model_path=None,
        rec_char_dict_path=None,
        rec_image_shape="3,48,640",
        drop_score=0.0,
    ):
        self.project_dir = project_dir or os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.project_dir, "models")
        self.infer_dir = os.path.join(self.project_dir, "tools", "infer")
        self.use_gpu = bool(use_gpu)
        self.det_yaml_path = self._resolve_path(det_yaml_path, self.infer_dir, "myocrv5.yml")
        self.det_model_path = self._resolve_path(det_model_path, self.models_dir, "ptocr_v5_det.pth")
        self.rec_yaml_path = self._resolve_path(rec_yaml_path, self.infer_dir, "PP-OCRv5_mobile_rec.yml")
        self.rec_model_path = self._resolve_path(rec_model_path, self.models_dir, "ptocr_v5_mobile_rec.pth")
        self.rec_char_dict_path = self._resolve_path(rec_char_dict_path, self.infer_dir, "ppocrv5_dict.txt")
        self.rec_image_shape = rec_image_shape
        self.drop_score = float(drop_score)
        self._ocr_system = None

    def _resolve_path(self, path, base_dir, default_name):
        if path is None:
            return os.path.join(base_dir, default_name)
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(base_dir, path))

    def _build_args(self):
        parser = ocr_utility.init_args()
        return parser.parse_args([
            "--use_gpu", str(self.use_gpu).lower(),
            "--det_yaml_path", self.det_yaml_path,
            "--det_model_path", self.det_model_path,
            "--rec_yaml_path", self.rec_yaml_path,
            "--rec_model_path", self.rec_model_path,
            "--rec_image_shape", self.rec_image_shape,
            "--rec_char_dict_path", self.rec_char_dict_path,
            "--drop_score", str(self.drop_score),
            "--use_mp", "false",
        ])

    def _validate_model_files(self):
        required_paths = [
            self.det_yaml_path,
            self.det_model_path,
            self.rec_yaml_path,
            self.rec_model_path,
            self.rec_char_dict_path,
        ]
        missing_paths = [path for path in required_paths if not os.path.exists(path)]
        if missing_paths:
            missing_text = "\n".join(missing_paths)
            raise FileNotFoundError(f"TextSystem OCR 所需文件不存在：\n{missing_text}")

    def _get_ocr_system(self):
        if self._ocr_system is None:
            self._validate_model_files()
            self._ocr_system = predict_system.TextSystem(self._build_args())
        return self._ocr_system

    def predict(self, image_input):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"无法读取图片: {image_input}")
        else:
            img = image_input.copy() if isinstance(image_input, np.ndarray) else None
            if img is None:
                raise ValueError("predict(...) 只支持图片路径或 numpy.ndarray")

        dt_boxes, rec_res, _ = self._get_ocr_system()(img)
        if dt_boxes is None or rec_res is None:
            return [{"rec_polys": [], "rec_texts": [], "rec_scores": []}]

        return [{
            "rec_polys": [np.asarray(box, dtype=np.float32) for box in dt_boxes],
            "rec_texts": [text for text, _ in rec_res],
            "rec_scores": [float(score) for _, score in rec_res],
        }]
