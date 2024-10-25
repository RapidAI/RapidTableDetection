from rapid_table_det.inference import TableDetector

img_path = f"tests/test_files/chip2.jpg"
table_det = TableDetector(
    obj_model_path="rapid_table_det/models/obj_det.onnx",
    edge_model_path="rapid_table_det/models/edge_det.onnx",
)
result, elapse = table_det(img_path)
obj_det_elapse, edge_elapse, rotate_det_elapse = elapse
print(
    f"obj_det_elapse:{obj_det_elapse}, edge_elapse={edge_elapse}, rotate_det_elapse={rotate_det_elapse}"
)
# 输出可视化
import os
import cv2
from rapid_table_det.utils import img_loader, visuallize, extract_table_img

img = img_loader(img_path)
file_name_with_ext = os.path.basename(img_path)
file_name, file_ext = os.path.splitext(file_name_with_ext)
out_dir = "rapid_table_det/outputs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
extract_img = img.copy()
for i, res in enumerate(result):
    box = res["box"]
    lt, rt, rb, lb = res["lt"], res["rt"], res["rb"], res["lb"]
    # 带识别框和左上角方向位置
    img = visuallize(img, box, lt, rt, rb, lb)
    # 透视变换提取表格图片
    wrapped_img = extract_table_img(extract_img.copy(), lt, rt, rb, lb)
    cv2.imwrite(f"{out_dir}/{file_name}-extract-{i}.jpg", wrapped_img)
cv2.imwrite(f"{out_dir}/{file_name}-visualize.jpg", img)
