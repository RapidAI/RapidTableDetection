import cv2

from rapid_table_det.utils.load_image import LoadImage
import numpy as np

img_loader = LoadImage()


def visuallize(img, box, lt, rt, rb, lb):
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
