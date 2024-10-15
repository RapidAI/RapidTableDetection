import math
import traceback
from io import BytesIO
from pathlib import Path
from typing import List, Union
import itertools
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import (
    SessionOptions,
    GraphOptimizationLevel,
)

root_dir = Path(__file__).resolve().parent
InputType = Union[str, np.ndarray, bytes, Path]


class LoadImage:
    def __init__(
        self,
    ):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        img = self.load_img(img)
        img = self.convert_img(img)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = np.array(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = np.array(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    def convert_img(self, img: np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 4:
                return self.cvt_four_to_three(img)

            if channel == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


img_loader = LoadImage()


class LoadImageError(Exception):
    pass


class OrtInferSession:
    def __init__(self, model_path: Union[str, Path], num_threads: int = -1):
        self.verify_exist(model_path)

        self.num_threads = num_threads
        self._init_sess_opt()

        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        EP_list = [(cpu_ep, cpu_provider_options)]
        try:
            self.session = InferenceSession(
                str(model_path), sess_options=self.sess_opt, providers=EP_list
            )
        except TypeError:
            # 这里兼容ort 1.5.2
            self.session = InferenceSession(str(model_path), sess_options=self.sess_opt)

    def _init_sess_opt(self):
        self.sess_opt = SessionOptions()
        self.sess_opt.log_severity_level = 4
        self.sess_opt.enable_cpu_mem_arena = False

        if self.num_threads != -1:
            self.sess_opt.intra_op_num_threads = self.num_threads

        self.sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    @staticmethod
    def verify_exist(model_path: Union[Path, str]):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")


class ONNXRuntimeError(Exception):
    pass


def generate_scale(im, resize_shape, keep_ratio):
    """
    Args:
        im (np.ndarray): image (np.ndarray)
    Returns:
        im_scale_x: the resize ratio of X
        im_scale_y: the resize ratio of Y
    """
    target_size = (resize_shape[0], resize_shape[1])
    # target_size = (800, 1333)
    origin_shape = im.shape[:2]

    if keep_ratio:
        im_size_min = np.min(origin_shape)
        im_size_max = np.max(origin_shape)
        target_size_min = np.min(target_size)
        target_size_max = np.max(target_size)
        im_scale = float(target_size_min) / float(im_size_min)
        if np.round(im_scale * im_size_max) > target_size_max:
            im_scale = float(target_size_max) / float(im_size_max)
        im_scale_x = im_scale
        im_scale_y = im_scale
    else:
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
    return im_scale_y, im_scale_x


def resize(im, im_info, resize_shape, keep_ratio, interp=2):
    im_scale_y, im_scale_x = generate_scale(im, resize_shape, keep_ratio)
    im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    im_info["im_shape"] = np.array(im.shape[:2]).astype("float32")
    im_info["scale_factor"] = np.array([im_scale_y, im_scale_x]).astype("float32")

    return im, im_info


def pad(im, im_info, resize_shape):
    im_h, im_w = im.shape[:2]
    fill_value = [114.0, 114.0, 114.0]
    h, w = resize_shape[0], resize_shape[1]
    if h == im_h and w == im_w:
        im = im.astype(np.float32)
        return im, im_info

    canvas = np.ones((h, w, 3), dtype=np.float32)
    canvas *= np.array(fill_value, dtype=np.float32)
    canvas[0:im_h, 0:im_w, :] = im.astype(np.float32)
    im = canvas
    return im, im_info


def ResizePad(img, target_size):
    h, w = img.shape[:2]
    m = max(h, w)
    ratio = target_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    top = (target_size - new_h) // 2
    bottom = (target_size - new_h) - top
    left = (target_size - new_w) // 2
    right = (target_size - new_w) - left
    img1 = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    return img1, new_w, new_h, left, top


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def minboundquad(hull):
    len_hull = len(hull)
    xy = np.array(hull).reshape([-1, 2])
    idx = np.arange(0, len_hull)
    idx_roll = np.roll(idx, -1, axis=0)
    edges = np.array([idx, idx_roll]).reshape([2, -1])
    edges = np.transpose(edges, [1, 0])
    edgeangles1 = []
    for i in range(len_hull):
        y = xy[edges[i, 1], 1] - xy[edges[i, 0], 1]
        x = xy[edges[i, 1], 0] - xy[edges[i, 0], 0]
        angle = math.atan2(y, x)
        if angle < 0:
            angle = angle + 2 * math.pi
        edgeangles1.append([angle, i])
    edgeangles1_idx = sorted(list(edgeangles1), key=lambda x: x[0])
    edges1 = []
    edgeangle1 = []
    for item in edgeangles1_idx:
        idx = item[1]
        edges1.append(edges[idx, :])
        edgeangle1.append(item[0])
    edgeangles = np.array(edgeangle1)
    edges = np.array(edges1)
    eps = 2.2204e-16
    angletol = eps * 100

    k = np.diff(edgeangles) < angletol
    idx = np.where(k == 1)
    edges = np.delete(edges, idx, 0)
    edgeangles = np.delete(edgeangles, idx, 0)
    nedges = edges.shape[0]
    edgelist = np.array(nchoosek(0, nedges - 1, 1, 4))
    k = edgeangles[edgelist[:, 3]] - edgeangles[edgelist[:, 0]] <= math.pi
    k_idx = np.where(k == 1)
    edgelist = np.delete(edgelist, k_idx, 0)

    nquads = edgelist.shape[0]
    quadareas = math.inf
    qxi = np.zeros([5])
    qyi = np.zeros([5])
    cnt = np.zeros([4, 1, 2])
    edgelist = list(edgelist)
    edges = list(edges)
    xy = list(xy)

    for i in range(nquads):
        edgeind = list(edgelist[i])
        edgeind.append(edgelist[i][0])
        edgesi = []
        edgeang = []
        for idx in edgeind:
            edgesi.append(edges[idx])
            edgeang.append(edgeangles[idx])
        is_continue = False
        for idx in range(len(edgeang) - 1):
            diff = edgeang[idx + 1] - edgeang[idx]
            if diff > math.pi:
                is_continue = True
        if is_continue:
            continue
        for j in range(4):
            jplus1 = j + 1
            shared = np.intersect1d(edgesi[j], edgesi[jplus1])
            if shared.size != 0:
                qxi[j] = xy[shared[0]][0]
                qyi[j] = xy[shared[0]][1]
            else:
                A = xy[edgesi[j][0]]
                B = xy[edgesi[j][1]]
                C = xy[edgesi[jplus1][0]]
                D = xy[edgesi[jplus1][1]]
                concat = np.hstack(((A - B).reshape([2, -1]), (D - C).reshape([2, -1])))
                div = (A - C).reshape([2, -1])
                inv_result = get_inv(concat)
                a = inv_result[0, 0]
                b = inv_result[0, 1]
                c = inv_result[1, 0]
                d = inv_result[1, 1]
                e = div[0, 0]
                f = div[1, 0]
                ts1 = [a * e + b * f, c * e + d * f]
                Q = A + (B - A) * ts1[0]
                qxi[j] = Q[0]
                qyi[j] = Q[1]

        contour = np.array([qxi[:4], qyi[:4]]).astype(np.int32)
        contour = np.transpose(contour, [1, 0])
        contour = contour[:, np.newaxis, :]
        A_i = cv2.contourArea(contour)
        # break

        if A_i < quadareas:
            quadareas = A_i
            cnt = contour
    return cnt


def nchoosek(startnum, endnum, step=1, n=1):
    c = []
    for i in itertools.combinations(range(startnum, endnum + 1, step), n):
        c.append(list(i))
    return c


def get_inv(concat):
    a = concat[0][0]
    b = concat[0][1]
    c = concat[1][0]
    d = concat[1][1]
    det_concat = a * d - b * c
    inv_result = np.array(
        [[d / det_concat, -b / det_concat], [-c / det_concat, a / det_concat]]
    )
    return inv_result


def get_max_adjacent_bbox(mask):
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    max_size = 0
    cnt_save = None
    # 找到最大边缘邻接矩形
    for cont in contours:
        points, sside = get_mini_boxes(cont)
        if sside > max_size:
            max_size = sside
            cnt_save = cont
    if cnt_save is not None:
        epsilon = 0.01 * cv2.arcLength(cnt_save, True)
        box = cv2.approxPolyDP(cnt_save, epsilon, True)
        hull = cv2.convexHull(box)
        points, sside = get_mini_boxes(cnt_save)
        len_hull = len(hull)

        if len_hull == 4:
            target_box = np.array(hull)
        elif len_hull > 4:
            target_box = minboundquad(hull)
        else:
            target_box = np.array(points)

        return np.array(target_box).reshape([-1, 2])


def visuallize(img, box, lt, rt, rb, lb):
    # img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    xmin, ymin, xmax, ymax = box
    draw_box = np.array([lt, rt, rb, lb]).reshape([-1, 2])
    cv2.circle(img, (int(lt[0]), int(lt[1])), 50, (255, 0, 0), 10)
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 10)
    cv2.polylines(
        img,
        [np.array(draw_box).astype(np.int32).reshape((-1, 1, 2))],
        True,
        color=(255, 0, 255),
        thickness=6,
    )
    return img


def extract_table_img(img, lt, rt, rb, lb):
    """
    根据四个角点进行透视变换，并提取出角点区域的图片。

    参数:
    img (numpy.ndarray): 输入图像
    lt (numpy.ndarray): 左上角坐标
    rt (numpy.ndarray): 右上角坐标
    lb (numpy.ndarray): 左下角坐标
    rb (numpy.ndarray): 右下角坐标

    返回:
    numpy.ndarray: 提取出的角点区域图片
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 源点坐标
    src_points = np.float32([lt, rt, lb, rb])

    # 目标点坐标
    width_a = np.sqrt(((rb[0] - lb[0]) ** 2) + ((rb[1] - lb[1]) ** 2))
    width_b = np.sqrt(((rt[0] - lt[0]) ** 2) + ((rt[1] - lt[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((rt[0] - rb[0]) ** 2) + ((rt[1] - rb[1]) ** 2))
    height_b = np.sqrt(((lt[0] - lb[0]) ** 2) + ((lt[1] - lb[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst_points = np.float32(
        [
            [0, 0],
            [max_width - 1, 0],
            [0, max_height - 1],
            [max_width - 1, max_height - 1],
        ]
    )

    # 获取透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # 应用透视变换
    warped = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped
