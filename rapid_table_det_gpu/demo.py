import os

import cv2

from inference import TableDetector
from utils import visuallize, extract_table_img

if __name__ == '__main__':
    img_path = f"../images/lineless1.png"
    out_dir = "outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    table_det = TableDetector()
    result, elapse = table_det(img_path)
    obj_det_elapse, edge_elapse, rotate_det_elapse = elapse
    print(f"obj_det_elapse:{obj_det_elapse}, edge_elapse={edge_elapse}, rotate_det_elapse={rotate_det_elapse}")
    for i, res in enumerate(result):
        box = res['box']
        lt, rt, rb, lb = res['lt'], res['rt'], res['rb'], res['lb']
        img_with_box = visuallize(img_path, box, lt, rt, rb, lb)
        cv2.imwrite(f"{out_dir}/visualize-{i}.jpg", img_with_box)
        extract_img = extract_table_img(img_path,  lt, rt, rb, lb)
        cv2.imwrite(f"{out_dir}/extract-{i}.jpg", extract_img)
