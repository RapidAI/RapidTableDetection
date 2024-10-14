# 代码示例
# python predict.py [src_image_dir] [results]

import os
import sys
import glob
import json
import cv2
import paddle
import math
import itertools
import numpy as np
from PIL import Image
from utils import *
MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}






# def generate_scale(im, resize_shape, keep_ratio):
#     """
#     Args:
#         im (np.ndarray): image (np.ndarray)
#     Returns:
#         im_scale_x: the resize ratio of X
#         im_scale_y: the resize ratio of Y
#     """
#     target_size = (resize_shape[0], resize_shape[1])
#     # target_size = (800, 1333)
#     origin_shape = im.shape[:2]
#
#     if keep_ratio:
#         im_size_min = np.min(origin_shape)
#         im_size_max = np.max(origin_shape)
#         target_size_min = np.min(target_size)
#         target_size_max = np.max(target_size)
#         im_scale = float(target_size_min) / float(im_size_min)
#         if np.round(im_scale * im_size_max) > target_size_max:
#             im_scale = float(target_size_max) / float(im_size_max)
#         im_scale_x = im_scale
#         im_scale_y = im_scale
#     else:
#         resize_h, resize_w = target_size
#         im_scale_y = resize_h / float(origin_shape[0])
#         im_scale_x = resize_w / float(origin_shape[1])
#     return im_scale_y, im_scale_x


# def normalize_img(im):
#     is_scale = True
#     im = im.astype(np.float32, copy=False)
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     norm_type = 'mean_std'
#     if is_scale:
#         scale = 1.0 / 255.0
#         im *= scale
#     if norm_type == 'mean_std':
#         mean = np.array(mean)[np.newaxis, np.newaxis, :]
#         std = np.array(std)[np.newaxis, np.newaxis, :]
#         im -= mean
#         im /= std
#     return im


# def normalize_img1(im):
#     is_scale = True
#     im = im.astype(np.float32, copy=False)
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     norm_type = 'none'
#     if is_scale:
#         scale = 1.0 / 255.0
#     if norm_type == 'mean_std':
#         mean = np.array(mean)[np.newaxis, np.newaxis, :]
#         std = np.array(std)[np.newaxis, np.newaxis, :]
#         im -= mean
#         im /= std
#     return im


# def resize(im, im_info, resize_shape, keep_ratio, interp=2):
#     im_scale_y, im_scale_x = generate_scale(im, resize_shape, keep_ratio)
#     im = cv2.resize(
#         im,
#         None,
#         None,
#         fx=im_scale_x,
#         fy=im_scale_y,
#         interpolation=interp)
#     im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
#     im_info['scale_factor'] = np.array(
#         [im_scale_y, im_scale_x]).astype('float32')
#
#     return im, im_info
#
#
# def pad(im, im_info, resize_shape):
#     im_h, im_w = im.shape[:2]
#     fill_value = [114.0, 114.0, 114.0]
#     h, w = resize_shape[0], resize_shape[1]
#     if h == im_h and w == im_w:
#         im = im.astype(np.float32)
#         return im, im_info
#
#     canvas = np.ones((h, w, 3), dtype=np.float32)
#     canvas *= np.array(fill_value, dtype=np.float32)
#     canvas[0:im_h, 0:im_w, :] = im.astype(np.float32)
#     im = canvas
#     return im, im_info


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
    inv_result = np.array([[d / det_concat, -b / det_concat],
                           [-c / det_concat, a / det_concat]])
    return inv_result


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


def process_yoloe(im, im_info, resize_shape):
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    # print(im)

    im, im_info = resize(im, im_info, resize_shape, False)
    h_n, w_n = im.shape[:-1]
    im, im_info = pad(im, im_info, resize_shape)

    # im = normalize_img(im)
    im = im / 255.0
    im = im.transpose((2, 0, 1)).copy()

    im = paddle.to_tensor(im, dtype='float32')
    im = im.unsqueeze(0)
    factor = paddle.to_tensor(im_info['scale_factor']).reshape((1, 2)).astype('float32')
    im_shape = paddle.to_tensor(im_info['im_shape'].reshape((1, 2)), dtype='float32')
    return im, im_shape, factor


def ResizePad1(img, target_size):
    h, w = img.shape[:2]
    m = max(h, w)
    ratio = target_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    top = (target_size - new_h) // 2
    bottom = (target_size - new_h) - top
    left = (target_size - new_w) // 2
    right = (target_size - new_w) - left
    img1 = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img1, new_w, new_h, left, top


def process_db(im, im_info, resize_shape):
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)

    # im, im_info = resize(im, im_info, resize_shape,False)

    resize_h, resize_w = im.shape[:-1]
    h_n, w_n = im.shape[:-1]
    im, new_w, new_h, left, top = ResizePad1(im, 800)

    # im = transforms.ToTensor()(im)
    im = im / 255.0
    # im = normalize_img(im)
    im = im.transpose((2, 0, 1)).copy()

    im = paddle.to_tensor(im, dtype='float32')

    # im = transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])(im)
    im = im.unsqueeze(0)
    return im, new_h, new_w, left, top


def crop_image(img, target_size, center):
    width, height = img.shape[1:]
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[:, int(h_start):int(h_end), int(w_start):int(w_end)]
    return img


def ResizePad(img, target_size):
    img = np.array(img)
    h, w = img.shape[:2]
    m = max(h, w)
    ratio = target_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    top = (target_size - new_h) // 2
    bottom = (target_size - new_h) - top
    left = (target_size - new_w) // 2
    right = (target_size - new_w) - left
    img1 = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    #     img1 = cv2.copyMakeBorder(img, top, bottom, left, right,cv2.BORDER_REPLICATE)
    return img1


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        # print(self.std,self.scale,self.mean)
        img = (
                      img.astype('float32') * self.scale - self.mean) / self.std
        return img


def process_image1(img, mode, rotate):
    resize_width = 624
    img = ResizePad(img, resize_width)
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255.0
    #     img = normalize_img1(img)
    # #     img = crop_image(img, 600,True)
    #     norm= NormalizeImage()
    #     img = norm(img)
    return img


def pad_stride(im):
    coarsest_stride = 32
    if coarsest_stride <= 0:
        return im
    im_c, im_h, im_w = im.shape
    pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
    pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
    padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
    padding_im[:, :im_h, :im_w] = im
    return padding_im


def box_score_fast(bitmap, _box):
    '''
    box_score_fast: use bbox mean score as the mean score
    '''
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


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

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])


def nms(box_List):
    np_boxes = np.array(box_List)
    x1 = np_boxes[:, 2]
    y1 = np_boxes[:, 3]
    x2 = np_boxes[:, 4]
    y2 = np_boxes[:, 5]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = np_boxes[:, 1]
    keep = []
    index = scores.argsort()[::-1]
    thresh = 0.01
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]

        index = index[idx + 1]  # because index start from 1
        # break
    np_boxes = np_boxes[keep, :]
    return np_boxes


def expand_poly(data, sec_dis):
    """多边形等距缩放
    Args:
        data: 多边形按照逆时针顺序排列的的点集
        sec_dis: 缩放距离

    Returns:
        缩放后的多边形点集
    """
    num = len(data)
    scal_data = []
    for i in range(num):
        x1 = data[(i) % num][0] - data[(i - 1) % num][0]
        y1 = data[(i) % num][1] - data[(i - 1) % num][1]
        x2 = data[(i + 1) % num][0] - data[(i) % num][0]
        y2 = data[(i + 1) % num][1] - data[(i) % num][1]

        d_A = (x1 ** 2 + y1 ** 2) ** 0.5
        d_B = (x2 ** 2 + y2 ** 2) ** 0.5

        Vec_Cross = (x1 * y2) - (x2 * y1)
        if (d_A * d_B == 0):
            continue
        sin_theta = Vec_Cross / (d_A * d_B)
        if (sin_theta == 0):
            continue
        dv = sec_dis / sin_theta

        v1_x = (dv / d_A) * x1
        v1_y = (dv / d_A) * y1

        v2_x = (dv / d_B) * x2
        v2_y = (dv / d_B) * y2

        PQ_x = v1_x - v2_x
        PQ_y = v1_y - v2_y

        Q_x = data[(i) % num][0] + PQ_x
        Q_y = data[(i) % num][1] + PQ_y
        scal_data.append([Q_x, Q_y])
    return scal_data


def step_function(x, y):
    return paddle.reciprocal(1 + paddle.exp(-50 * (x - y)))


def process(src_image_dir, save_dir):
    model = paddle.jit.load('models/obj_det/model')
    model.eval()


    easy_model = paddle.jit.load('models/db_net/model')
    easy_model.eval()


    dir_model = paddle.jit.load('models/pplcnet/model')
    dir_model.eval()

    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    image_paths.extend(glob.glob(os.path.join(src_image_dir, "*.jpeg")))
    image_paths.extend(glob.glob(os.path.join(src_image_dir, "*.png")))
    result = {}
    for image_path in image_paths:

        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }

        filename = os.path.split(image_path)[1]
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im = np.asarray(im)
        temp = im.copy()
        h1, w1 = im.shape[:-1]
        im_mask = im.copy()
        im, im_shape, factor = process_yoloe(im, im_info, [928, 928])
        print(image_path)
        pre = model(im, factor)
        if filename not in result:
            result[filename] = []
        for item in pre[0].numpy():

            cls, value, xmin, ymin, xmax, ymax = list(item)
            cls, xmin, ymin, xmax, ymax = [int(x) for x in [cls, xmin, ymin, xmax, ymax]]
            # xmin, ymin, xmax, ymax = xmin -10, ymin -10, xmax +10, ymax +10
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > w1:
                xmax = w1
            if ymax > h1:
                ymax = h1

            if value > 0.5:

                im_info = {}
                ymin_a = max(ymin - 10, 0)
                xmin_a = max(xmin - 10, 0)
                ymax_a = min(ymax + 10, h1)
                xmax_a = min(xmax + 10, w1)

                crop_img = im_mask[ymin_a:ymax_a, xmin_a:xmax_a, :]
                # crop_img = im_mask
                ymin_c = max(ymin - 5, 0)
                xmin_c = max(xmin - 5, 0)
                ymax_c = min(ymax + 5, h1)
                xmax_c = min(xmax + 5, w1)
                pred_label = 0

                cls_img = im_mask[ymin_c:ymax_c, xmin_c:xmax_c, :]

                destHeight, destWidth = crop_img.shape[:-1]
                crop_img, resize_h, resize_w, left, top = process_db(crop_img, im_info, [800, 800])

                with paddle.no_grad():
                    predicts = easy_model(crop_img)
                predict_maps = predicts.cpu()
                pred = predict_maps[0, 0].numpy()
                segmentation = pred > 0.7
                # print(segmentation.shape)
                # dilation_kernel = np.array([[1, 1], [1, 1]])
                mask = np.array(segmentation).astype(np.uint8)

                contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                lt = np.array([int(xmin), int(ymin)])
                rt = np.array([int(xmax), int(ymin)])
                rb = np.array([int(xmax), int(ymax)])
                lb = np.array([int(xmin), int(ymax)])

                # lt = np.array([int(xmin_a), int(ymin_a)])
                # rt = np.array([int(xmax_a), int(ymin_a)])
                # rb = np.array([int(xmax_a), int(ymax_a)])
                # lb = np.array([int(xmin_a), int(ymax_a)])

                max_size = 0
                cnt_save = None
                for cont in contours:
                    points, sside = get_mini_boxes(cont)
                    if sside > max_size:
                        max_size = sside
                        cnt_save = cont
                # cnt_save = None
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

                    box = np.array(target_box).reshape([-1, 2])

                    #     print(box.shape)
                    box[:, 0] = np.clip(
                        (np.round(box[:, 0] - left) / resize_w * destWidth), 0, destWidth) + xmin_a
                    box[:, 1] = np.clip(
                        (np.round(box[:, 1] - top) / resize_h * destHeight), 0, destHeight) + ymin_a
                    x = box[:, 0]
                    l_idx = x.argsort()
                    l_box = np.array([box[l_idx[0]], box[l_idx[1]]])
                    r_box = np.array([box[l_idx[2]], box[l_idx[3]]])
                    l_idx_1 = np.array(l_box[:, 1]).argsort()
                    lt = l_box[l_idx_1[0]]
                    lt[lt < 0] = 0
                    lb = l_box[l_idx_1[1]]
                    r_idx_1 = np.array(r_box[:, 1]).argsort()
                    rt = r_box[r_idx_1[0]]
                    rt[rt < 0] = 0
                    rb = r_box[r_idx_1[1]]
                    cls_box = box.copy()
                    cls_box[:, 0] = cls_box[:, 0] - xmin_c
                    cls_box[:, 1] = cls_box[:, 1] - ymin_c
                    cv2.polylines(cls_img, [np.array(cls_box).astype(np.int32).reshape((-1, 1, 2))], True,
                                  color=(255, 0, 255), thickness=5)

                    cls_img = process_image1(cls_img, 'test', True)
                    cls_img = paddle.to_tensor(cls_img)
                    cls_img = cls_img.unsqueeze(0)
                    with paddle.no_grad():
                        # print(cls_img.shape)
                        label = dir_model(cls_img)
                    label = label.unsqueeze(0).numpy()
                    mini_batch_result = np.argsort(label)
                    mini_batch_result = mini_batch_result[0][-1]  # 把这些列标拿出来
                    mini_batch_result = mini_batch_result.flatten()  # 拉平了，只吐出一个 array
                    mini_batch_result = mini_batch_result[::-1]  # 逆序

                    pred_label = mini_batch_result[0]

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
                draw_box = np.array([lt1, rt1, rb1, lb1]).reshape([-1, 2])
                cv2.circle(temp, (int(lt1[0]), int(lt1[1])), 50, (255, 0, 0), 10)
                cv2.rectangle(temp, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 10)
                cv2.polylines(temp, [np.array(draw_box).astype(np.int32).reshape((-1, 1, 2))], True,
                              color=(255, 0, 255), thickness=6)

                result[filename].append({
                    "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
                    "lb": [int(lb1[0]), int(lb1[1])],
                    "lt": [int(lt1[0]), int(lt1[1])],
                    "rt": [int(rt1[0]), int(rt1[1])],
                    "rb": [int(rb1[0]), int(rb1[1])],
                })
        print(f"{image_path} process done!")

        save_p = os.path.join(save_dir, filename)
        h, w = temp.shape[:-1]
        target = 512
        w_p = h / w * target
        # temp = cv2.resize(temp, (int(target), int(w_p)))
        cv2.imwrite(save_p, temp)

    with open(os.path.join(save_dir, "result.txt"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(result))


if __name__ == "__main__":

    src_image_dir = "images"
    save_dir = "outputs"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)