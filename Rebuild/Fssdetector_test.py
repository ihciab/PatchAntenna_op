from FssDetector import FSSfigDetector, TextSystemOCRAdapter


if __name__ == "__main__":
    image_path = "./fss_test"
    ocr = TextSystemOCRAdapter()

    detector = FSSfigDetector(
        max_k=6,
        min_color_diff=30,
        ocr_engine=ocr,
    )
    results = detector.detect(image_path, output_folder="./fss_out_results")
