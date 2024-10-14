import cv2
import numpy as np
from paddle_predictor import DbNet, ObjectDetector, PPLCNet
from utils import LoadImage


class TableDetector:
    def __init__(self, dbnet_model_path, obj_model_path,pplcnet_model_path, **kwargs):
        self.use_obj_det = kwargs.get("use_obj_det")
        self.use_edge_det = kwargs.get("use_edge_det")
        self.use_rotate_det = kwargs.get("use_rotate_det")
        self.img_loader = LoadImage()
        if self.use_obj_det:
            self.obj_detector = ObjectDetector(obj_model_path)
        if self.use_edge_det:
            self.dbnet = DbNet(dbnet_model_path)
        if self.use_rotate_det:
            self.pplcnet = PPLCNet(pplcnet_model_path)
    def __call__(self, img, det_accuracy=0.4):
        img = self.img_loader(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_mask = img.copy()
        h, w = img.shape[:-1]
        img_box = np.array([1, 1, w - 1, h - 1])
        x1, y1, x2, y2 = img_box
        lt = np.array([x1, y1])  # 左上角
        lb = np.array([x1, y2])  # 左下角
        rt = np.array([x2, y1])  # 右上角
        rb = np.array([x2, y2])  # 右下角
        obj_det_res = [[1.0, img_box]]
        edge_box = img_box
        pred_label = 0
        result = []
        if self.use_obj_det:
             obj_det_res = self.obj_detector(img, score=det_accuracy)
        for i in range(len(obj_det_res)):
             det_res = obj_det_res[i]
             score, box = det_res
             xmin, ymin, xmax, ymax = box
             if self.use_edge_det:
                 xmin_edge, ymin_edge, xmax_edge, ymax_edge = self.pad_box_points(h, w, xmax, xmin, ymax, ymin, 10)
                 crop_img = img_mask[ymin_edge:ymax_edge, xmin_edge:xmax_edge, :]
                 edge_box, lt, lb, rt, rb = self.dbnet(crop_img)
                 edge_box[:, 0] += xmin_edge
                 edge_box[:, 1] += ymin_edge
                 lt, lb, rt, rb = lt + [xmin_edge, ymin_edge], lb + [xmin_edge, ymin_edge], rt + [xmin_edge, ymin_edge], rb + [xmin_edge, ymin_edge]
             if self.use_rotate_det:
                 xmin_cls, ymin_cls, xmax_cls, ymax_cls = self.pad_box_points(h, w, xmax, xmin, ymax, ymin, 10)
                 cls_box = edge_box.copy()
                 cls_img = img_mask[ymin_cls:ymax_cls, xmin_cls:xmax_cls, :]
                 cls_box[:, 0] = cls_box[:, 0] - xmin_cls
                 cls_box[:, 1] = cls_box[:, 1] - ymin_cls
                 # 画框增加先验信息，辅助方向label识别
                 cv2.polylines(cls_img, [np.array(cls_box).astype(np.int32).reshape((-1, 1, 2))], True,
                               color=(255, 0, 255), thickness=5)
                 pred_label = self.pplcnet(cls_img)
             lb1, lt1, rb1, rt1 = self.get_real_rotated_points(lb, lt, pred_label, rb, rt)
             result.append({
                 "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
                 "lb": [int(lb1[0]), int(lb1[1])],
                 "lt": [int(lt1[0]), int(lt1[1])],
                 "rt": [int(rt1[0]), int(rt1[1])],
                 "rb": [int(rb1[0]), int(rb1[1])],
             })
        return result

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