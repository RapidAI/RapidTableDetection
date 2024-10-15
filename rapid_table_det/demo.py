import os

import cv2

from rapid_table_det.inference import TableDetector
from rapid_table_det.utils import visuallize, extract_table_img

if __name__ == '__main__':
    img_path = f"../images/wtw_img02678.jpg"
    out_dir = "outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    table_det = TableDetector(
        # obj_model_path="models/obj_det.onnx",
        obj_model_path="models/obj_det.onnx",
        edge_model_path="models/modified_obj_det.onnx",
        cls_model_path="models/cls_det.onnx",
        use_obj_det=True,
        use_edge_det=True,
        use_rotate_det=True)
    result, elapse = table_det(img_path)
    obj_det_elapse, edge_elapse, rotate_det_elapse = elapse
    print(f"obj_det_elapse:{obj_det_elapse}, edge_elapse={edge_elapse}, rotate_det_elapse={rotate_det_elapse}")
    # 一张图片中可能有多个表格
    for i, res in enumerate(result):
        box = res['box']
        lt, rt, rb, lb = res['lt'], res['rt'], res['rb'], res['lb']
        # 带识别框和左上角方向位置
        img_with_box = visuallize(img_path, box, lt, rt, rb, lb)
        cv2.imwrite(f"{out_dir}/visualize-{i}.jpg", img_with_box)
        # 透视变换提取表格图片
        extract_img = extract_table_img(img_path,  lt, rt, rb, lb)
        cv2.imwrite(f"{out_dir}/extract-{i}.jpg", extract_img)
