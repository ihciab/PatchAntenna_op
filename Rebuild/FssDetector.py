
try:
    from .fssdetector_ocr import TextSystemOCRAdapter
    from .fssdetector_selection import FSSSelectionMixin
    from .fssdetector_clustering import FSSClusteringMixin
    from .fssdetector_legacy import FSSLegacyMixin
    from .fssdetector_pipeline import FSSPipelineMixin
except ImportError:
    from fssdetector_ocr import TextSystemOCRAdapter
    from fssdetector_selection import FSSSelectionMixin
    from fssdetector_clustering import FSSClusteringMixin
    from fssdetector_legacy import FSSLegacyMixin
    from fssdetector_pipeline import FSSPipelineMixin


class FSSfigDetector(
    FSSPipelineMixin,
    FSSClusteringMixin,
    FSSSelectionMixin,
    FSSLegacyMixin,
):
    """
    FSS 图像检测主类。

    当前实现将流程拆分到多个 mixin 中，负责颜色聚类、筛选、OCR、
    传统辅助方法以及整条检测流水线。
    """

    def __init__(
        self,
        max_k=6,
        min_color_diff=30,
        text_info_list=None,
        ocr_engine=None,
        show_debug_windows=False,
        yolo_model_path=r"D:\LineFormer-main\models\bestyolo.pt",
        yolo_subfigure_class_id=2,
        yolo_text_class_id=1,
        yolo_conf_threshold=0.20,
        yolo_iou_threshold=0.50,
        yolo_device="cpu",
        yolo_element_labels=("arrow", "line"),
        yolo_text_labels=("text",),
        yolo_element_conf_threshold=0.12,
        yolo_element_expand_pixels=5,
        yolo_element_border_width=3,
        yolo_element_min_component_area=4,
        repair_structure_dilation_pixels=10,
        yolo_element_inpaint_radius=3,
        ocr_text_score_threshold=0.30,
        yolo_ocr_match_iou_threshold=0.10,
    ):
        """
        初始化检测器。

        参数:
            max_k: 颜色量化时允许的最大聚类数。
            min_color_diff: 聚类中心之间的最小颜色差阈值。
        """

        self.max_k = max_k
        self.min_color_diff = min_color_diff
        self.text_info_list = text_info_list or []
        self.ocr_engine = ocr_engine or TextSystemOCRAdapter()
        self.show_debug_windows = show_debug_windows
        self.yolo_model_path = yolo_model_path
        self.yolo_subfigure_class_id = yolo_subfigure_class_id
        self.yolo_text_class_id = yolo_text_class_id
        self.yolo_conf_threshold = yolo_conf_threshold
        self.yolo_iou_threshold = yolo_iou_threshold
        self.yolo_device = yolo_device
        self.yolo_element_labels = tuple(str(label).strip().lower() for label in yolo_element_labels)
        self.yolo_text_labels = tuple(str(label).strip().lower() for label in yolo_text_labels)
        self.yolo_element_conf_threshold = float(yolo_element_conf_threshold)
        self.yolo_element_expand_pixels = int(yolo_element_expand_pixels)
        self.yolo_element_border_width = int(yolo_element_border_width)
        self.yolo_element_min_component_area = int(yolo_element_min_component_area)
        self.repair_structure_dilation_pixels = int(repair_structure_dilation_pixels)
        self.yolo_element_inpaint_radius = int(yolo_element_inpaint_radius)
        self.ocr_text_score_threshold = float(ocr_text_score_threshold)
        self.yolo_ocr_match_iou_threshold = float(yolo_ocr_match_iou_threshold)
        self._yolo_model = None
        self._yolo_load_failed = False
        self.supported_image_extensions = {
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
        }


# 示例用法
if __name__ == "__main__":
    detector = FSSfigDetector(max_k=6)
    image_path = "D:\\cst2py_box\\Auto_py2cst_v0.71\\test\\test28.png"
    results = detector.detect(image_path)
    print("检测完成")
