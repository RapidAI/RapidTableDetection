import math
import itertools
import cv2
import numpy as np


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
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_iou(box, other_boxes):
    """
    计算给定边界框与一组其他边界框之间的交并比（IoU）。

    参数：
    - box: 单个边界框，格式为 [x1, y1, width, height]。
    - other_boxes: 其他边界框的数组，每个边界框的格式也为 [x1, y1, width, height]。

    返回值：
    - iou: 一个数组，包含给定边界框与每个其他边界框的IoU值。
    """

    # 计算交集的左上角坐标
    x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
    y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
    # 计算交集的右下角坐标
    x2 = np.minimum(box[2], np.array(other_boxes)[:, 2])
    y2 = np.minimum(box[3], np.array(other_boxes)[:, 3])
    # 计算交集区域的面积
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    # 计算给定边界框的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    # 计算其他边界框的面积
    other_boxes_area = np.array(other_boxes[:, 2] - other_boxes[:, 0]) * np.array(
        other_boxes[:, 3] - other_boxes[:, 1]
    )
    # 计算IoU值
    iou = intersection_area / (box_area + other_boxes_area - intersection_area)
    return iou


def custom_NMSBoxes(boxes, scores, iou_threshold=0.4):
    # 如果没有边界框，则直接返回空列表
    if len(boxes) == 0:
        return []
    # 将得分和边界框转换为NumPy数组
    scores = np.array(scores)
    boxes = np.array(boxes)
    # 根据置信度阈值过滤边界框
    # filtered_boxes = boxes[mask]
    # filtered_scores = scores[mask]
    # 如果过滤后没有边界框，则返回空列表
    if len(boxes) == 0:
        return []
    # 根据置信度得分对边界框进行排序
    sorted_indices = np.argsort(scores)[::-1]
    # 初始化一个空列表来存储选择的边界框索引
    indices = []
    # 当还有未处理的边界框时，循环继续
    while len(sorted_indices) > 0:
        # 选择得分最高的边界框索引
        current_index = sorted_indices[0]
        indices.append(current_index)
        # 如果只剩一个边界框，则结束循环
        if len(sorted_indices) == 1:
            break
        # 获取当前边界框和其他边界框
        current_box = boxes[current_index]
        other_boxes = boxes[sorted_indices[1:]]
        # 计算当前边界框与其他边界框的IoU
        iou = calculate_iou(current_box, other_boxes)
        # 找到IoU低于阈值的边界框，即与当前边界框不重叠的边界框
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        # 更新sorted_indices以仅包含不重叠的边界框
        sorted_indices = sorted_indices[non_overlapping_indices + 1]
    # 返回选择的边界框索引
    return indices
