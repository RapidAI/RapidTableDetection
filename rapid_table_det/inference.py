import os
from pathlib import Path

import cv2
import numpy as np

from rapid_table_det.predictor import DbNet, ObjectDetector, PPLCNet, YoloSeg, YoloDet
from rapid_table_det.utils import LoadImage

root_dir = Path(__file__).resolve().parent
model_dir = os.path.join(root_dir, "models")


class TableDetector:
    def __init__(
        self,
        obj_model="yolo",
        edge_model="yolo",
        obj_model_path=os.path.join(model_dir, "obj_det_quantized.onnx"),
        edge_model_path=os.path.join(model_dir, "yolo_edge_det_s.onnx"),
        cls_model_path=os.path.join(model_dir, "cls_det.onnx"),
    ):
        self.img_loader = LoadImage()
        if obj_model == "yolo":
            self.obj_detector = YoloDet(obj_model_path)
        else:
            self.obj_detector = ObjectDetector(obj_model_path)
        if edge_model == "yolo":
            self.dbnet = YoloSeg(edge_model_path)
        else:
            self.dbnet = DbNet(edge_model_path)
        self.pplcnet = PPLCNet(cls_model_path)

    def __call__(
        self,
        img,
        det_accuracy=0.7,
        use_obj_det=True,
        use_edge_det=True,
        use_cls_det=True,
    ):
        img = self.img_loader(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_mask = img.copy()
        h, w = img.shape[:-1]
        obj_det_res, pred_label = self.init_default_output(h, w)
        result = []
        obj_det_elapse, edge_elapse, rotate_det_elapse = 0, 0, 0
        if use_obj_det:
            obj_det_res, obj_det_elapse = self.obj_detector(img, score=det_accuracy)
        for i in range(len(obj_det_res)):
            det_res = obj_det_res[i]
            score, box = det_res
            xmin, ymin, xmax, ymax = box
            edge_box = box.reshape([-1, 2])
            lb, lt, rb, rt = self.get_box_points(box)
            if use_edge_det:
                xmin_edge, ymin_edge, xmax_edge, ymax_edge = self.pad_box_points(
                    h, w, xmax, xmin, ymax, ymin, 10
                )
                crop_img = img_mask[ymin_edge:ymax_edge, xmin_edge:xmax_edge, :]
                edge_box, lt, lb, rt, rb, tmp_edge_elapse = self.dbnet(crop_img)
                edge_elapse += tmp_edge_elapse
                if edge_box is None:
                    continue
                lb, lt, rb, rt = self.adjust_edge_points_axis(
                    edge_box, lb, lt, rb, rt, xmin_edge, ymin_edge
                )
            if use_cls_det:
                xmin_cls, ymin_cls, xmax_cls, ymax_cls = self.pad_box_points(
                    h, w, xmax, xmin, ymax, ymin, 5
                )
                cls_img = img_mask[ymin_cls:ymax_cls, xmin_cls:xmax_cls, :]
                # 增加先验信息
                self.add_pre_info_for_cls(cls_img, edge_box, xmin_cls, ymin_cls)
                pred_label, tmp_rotate_det_elapse = self.pplcnet(cls_img)
                rotate_det_elapse += tmp_rotate_det_elapse
            lb1, lt1, rb1, rt1 = self.get_real_rotated_points(
                lb, lt, pred_label, rb, rt
            )
            result.append(
                {
                    "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "lb": [int(lb1[0]), int(lb1[1])],
                    "lt": [int(lt1[0]), int(lt1[1])],
                    "rt": [int(rt1[0]), int(rt1[1])],
                    "rb": [int(rb1[0]), int(rb1[1])],
                }
            )
        elapse = [obj_det_elapse, edge_elapse, rotate_det_elapse]
        return result, elapse

    def init_default_output(self, h, w):
        img_box = np.array([0, 0, w, h])
        # 初始化默认值
        obj_det_res, edge_box, pred_label = (
            [[1.0, img_box]],
            img_box.reshape([-1, 2]),
            0,
        )
        return obj_det_res, pred_label

    def add_pre_info_for_cls(self, cls_img, edge_box, xmin_cls, ymin_cls):
        cls_box = edge_box.copy()
        cls_box[:, 0] = cls_box[:, 0] - xmin_cls
        cls_box[:, 1] = cls_box[:, 1] - ymin_cls
        # 画框增加先验信息，辅助方向label识别
        cv2.polylines(
            cls_img,
            [np.array(cls_box).astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(255, 0, 255),
            thickness=5,
        )

    def adjust_edge_points_axis(self, edge_box, lb, lt, rb, rt, xmin_edge, ymin_edge):
        edge_box[:, 0] += xmin_edge
        edge_box[:, 1] += ymin_edge
        lt, lb, rt, rb = (
            lt + [xmin_edge, ymin_edge],
            lb + [xmin_edge, ymin_edge],
            rt + [xmin_edge, ymin_edge],
            rb + [xmin_edge, ymin_edge],
        )
        return lb, lt, rb, rt

    def get_box_points(self, img_box):
        x1, y1, x2, y2 = img_box
        lt = np.array([x1, y1])  # 左上角
        rt = np.array([x2, y1])  # 右上角
        rb = np.array([x2, y2])  # 右下角
        lb = np.array([x1, y2])  # 左下角
        return lb, lt, rb, rt

    def get_real_rotated_points(self, lb, lt, pred_label, rb, rt):
        if pred_label == 0:
            lt1 = lt
            rt1 = rt
            rb1 = rb
            lb1 = lb
        elif pred_label == 1:
            lt1 = rt
            rt1 = rb
            rb1 = lb
            lb1 = lt
        elif pred_label == 2:
            lt1 = rb
            rt1 = lb
            rb1 = lt
            lb1 = rt
        elif pred_label == 3:
            lt1 = lb
            rt1 = lt
            rb1 = rt
            lb1 = rb
        else:
            lt1 = lt
            rt1 = rt
            rb1 = rb
            lb1 = lb
        return lb1, lt1, rb1, rt1

    def pad_box_points(self, h, w, xmax, xmin, ymax, ymin, pad):
        ymin_edge = max(ymin - pad, 0)
        xmin_edge = max(xmin - pad, 0)
        ymax_edge = min(ymax + pad, h)
        xmax_edge = min(xmax + pad, w)
        return xmin_edge, ymin_edge, xmax_edge, ymax_edge
