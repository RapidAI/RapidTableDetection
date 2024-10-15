import cv2
import numpy as np
from rapid_table_det_paddle.predictor import DbNet, ObjectDetector, PPLCNet
from rapid_table_det_paddle.utils import LoadImage


class TableDetector:
    def __init__(
        self, edge_model_path=None, obj_model_path=None, cls_model_path=None, **kwargs
    ):
        self.use_obj_det = kwargs.get("use_obj_det", True)
        self.use_edge_det = kwargs.get("use_edge_det", True)
        self.use_rotate_det = kwargs.get("use_rotate_det", True)
        self.img_loader = LoadImage()
        if self.use_obj_det:
            self.obj_detector = (
                ObjectDetector(obj_model_path) if obj_model_path else ObjectDetector()
            )
        if self.use_edge_det:
            self.dbnet = DbNet(edge_model_path) if edge_model_path else DbNet()
        if self.use_rotate_det:
            self.pplcnet = PPLCNet(cls_model_path) if cls_model_path else PPLCNet()

    def __call__(self, img, det_accuracy=0.4):
        img = self.img_loader(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_mask = img.copy()
        h, w = img.shape[:-1]
        img_box = np.array([1, 1, w - 1, h - 1])
        lb, lt, rb, rt = self.get_box_points(img_box)
        # 初始化默认值
        obj_det_res, edge_box, pred_label = (
            [[1.0, img_box]],
            img_box.reshape([-1, 2]),
            0,
        )
        result = []
        obj_det_elapse, edge_elapse, rotate_det_elapse = 0, 0, 0
        if self.use_obj_det:
            obj_det_res, obj_det_elapse = self.obj_detector(img, score=det_accuracy)
        for i in range(len(obj_det_res)):
            det_res = obj_det_res[i]
            score, box = det_res
            xmin, ymin, xmax, ymax = box
            edge_box = box.reshape([-1, 2])
            lb, lt, rb, rt = self.get_box_points(box)
            if self.use_edge_det:
                xmin_edge, ymin_edge, xmax_edge, ymax_edge = self.pad_box_points(
                    h, w, xmax, xmin, ymax, ymin, 10
                )
                crop_img = img_mask[ymin_edge:ymax_edge, xmin_edge:xmax_edge, :]
                edge_box, lt, lb, rt, rb, tmp_edge_elapse = self.dbnet(crop_img)
                edge_elapse += tmp_edge_elapse
                if edge_box is None:
                    continue
                edge_box[:, 0] += xmin_edge
                edge_box[:, 1] += ymin_edge
                lt, lb, rt, rb = (
                    lt + [xmin_edge, ymin_edge],
                    lb + [xmin_edge, ymin_edge],
                    rt + [xmin_edge, ymin_edge],
                    rb + [xmin_edge, ymin_edge],
                )
            if self.use_rotate_det:
                xmin_cls, ymin_cls, xmax_cls, ymax_cls = self.pad_box_points(
                    h, w, xmax, xmin, ymax, ymin, 10
                )
                cls_box = edge_box.copy()
                cls_img = img_mask[ymin_cls:ymax_cls, xmin_cls:xmax_cls, :]
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
