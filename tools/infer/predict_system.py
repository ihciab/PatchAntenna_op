import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import copy
import numpy as np
import time
from PIL import Image
import json
import tools.infer.pytorchocr_utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from pytorchocr.utils.utility import (
    get_image_file_list,
    check_and_read,
)
from tools.infer.pytorchocr_utility import draw_ocr_box_txt



class TextSystem(object):
    def __init__(self, args, **kwargs):
        self.text_detector = predict_det.TextDetector(args, **kwargs)
        self.text_recognizer = predict_rec.TextRecognizer(args, **kwargs)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args, **kwargs)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, "mg_crop_{}.jpg".format(bno + self.crop_image_res_index)
                ),
                img_crop_list[bno],
            )
            print("{bno}, {}".format(rec_res[bno]))
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True, slice={}):
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}

        if img is None:
            print("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()

        if slice:
            slice_gen = utility.slice_generator(
                img,
                horizontal_stride=slice["horizontal_stride"],
                vertical_stride=slice["vertical_stride"],
            )
            elapsed = []
            dt_slice_boxes = []

            for slice_crop, v_start, h_start in slice_gen:
                dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                if dt_boxes.size:
                    dt_boxes[:, :, 0] += h_start
                    dt_boxes[:, :, 1] += v_start
                    dt_slice_boxes.append(dt_boxes)
                    elapsed.append(elapse)
            dt_boxes = np.concatenate(dt_slice_boxes)

            dt_boxes = utility.merge_fragmented(
                boxes=dt_boxes,
                x_threshold=slice["merge_x_thres"],
                y_threshold=slice["merge_y_thres"],
            )
            elapse = sum(elapsed)
        else:
            dt_boxes, elapse = self.text_detector(img)

        time_dict["det"] = elapse

        if dt_boxes is None:
            print("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict
        else:
            print(
                "dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse)
            )

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = utility.get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = utility.get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict["cls"] = elapse
            print("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        if len(img_crop_list) > 1000:
            print(
                "rec crops num: {}, time and memory cost may be large.".format(len(img_crop_list))
            )

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        print("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        end = time.time()
        time_dict["all"] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)

    save_results = []
    total_time = 0
    _st = time.time()

    for idx, image_file in enumerate(image_file_list):
        img, flag_gif,  flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                print("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]

        for index, img in enumerate(imgs):
            starttime = time.time()
            dt_boxes, rec_res, time_dict = text_sys(img)
            elapse = time.time() - starttime
            if len(imgs) > 1:
                print(
                    str(idx)
                    + "_"
                    + str(index)
                    + "  Predict time of %s: %.3fs" % (image_file, elapse)
                )
            else:
                print(
                    str(idx) + "  Predict time of %s: %.3fs" % (image_file, elapse)
                )

            for text, score in rec_res:
                print("{}, {:.3f}".format(text, score))

            res = [
                {
                    "transcription": rec_res[i][0],
                    "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                }
                for i in range(len(dt_boxes))
            ]
            if len(imgs) > 1:
                save_pred = (
                    os.path.basename(image_file)
                    + "_"
                    + str(index)
                    + "\t"
                    + json.dumps(res, ensure_ascii=False)
                    + "\n"
                )
            else:
                save_pred = (
                    os.path.basename(image_file)
                    + "\t"
                    + json.dumps(res, ensure_ascii=False)
                    + "\n"
                )
            save_results.append(save_pred)

            if is_visualize:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                boxes = dt_boxes
                txts = [rec_res[i][0] for i in range(len(rec_res))]
                scores = [rec_res[i][1] for i in range(len(rec_res))]

                draw_img = draw_ocr_box_txt(
                    image,
                    boxes,
                    txts,
                    scores,
                    drop_score=drop_score,
                    font_path=font_path,
                )

                if flag_gif:
                    save_file = image_file[:-3] + "png"
                elif flag_pdf:
                    save_file = image_file.replace(".pdf", "_" + str(index) + ".png")
                else:
                    save_file = image_file
                cv2.imwrite(
                    os.path.join(draw_img_save_dir, os.path.basename(save_file)),
                    draw_img[:, :, ::-1],
                )

                print(
                    "The visualized image saved in {}".format(
                        os.path.join(draw_img_save_dir, os.path.basename(save_file))
                    )
                )

    print("The predict total time is {}".format(time.time() - _st))

    with open(
        os.path.join(draw_img_save_dir, "system_results.txt"), "w", encoding="utf-8"
    ) as f:
        f.writelines(save_results)


if __name__ == '__main__':
    project_dir = os.path.abspath(os.path.join(__dir__, '../..'))
    models_dir = os.path.join(project_dir, "models")
    infer_dir = __dir__

    CUSTOM_ARGS = [
        # 固定项：脚本自身路径（sys.argv[0]）
        sys.argv[0],
        # 以下是你的命令行参数，按顺序添加
        "--use_gpu", "false",
        "--det_yaml_path", os.path.join(infer_dir, "myocrv5.yml"),
        "--det_model_path", os.path.join(models_dir, "ptocr_v5_det.pth"),
        "--det_limit_side_len", "1536",
        "--det_db_thresh", "0.32",
        "--det_db_box_thresh", "0.5",
        "--det_db_unclip_ratio", "1.1",
        "--use_dilation", "false",
        "--det_db_score_mode", "slow",
        "--rec_yaml_path", os.path.join(infer_dir, "PP-OCRv5_mobile_rec.yml"),
        "--rec_model_path", os.path.join(models_dir, "ptocr_v5_mobile_rec.pth"),
        "--rec_image_shape", "3,48,640",  # 去掉命令行的引号，直接传字符串
        "--rec_char_dict_path", os.path.join(infer_dir, "ppocrv5_dict.txt"),
        "--drop_score", "0.3",
        "--image_dir", os.path.join(project_dir, "testfig"),
        # 可选：如果需要关闭多进程，添加--use_mp false（根据你的需求）
        "--use_mp", "false"
    ]

    # 替换sys.argv为自定义参数（核心步骤）
    sys.argv = CUSTOM_ARGS
    # =================================================================================
    args = utility.parse_args()
    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = (
                [sys.executable, "-u"]
                + sys.argv
                + ["--process_id={}".format(process_id), "--use_mp={}".format(False)]
            )
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
