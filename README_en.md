<div align="center">
  <div align="center">
    <h1><b>📊RapidTableDetection</b></h1>
  </div>
  <a href=""><img src="https://img.shields.io/badge/Python->=3.8,<3.12-aff.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Mac%2C%20Win-pink.svg"></a>
<a href="https://semver.org/"><img alt="SemVer2.0" src="https://img.shields.io/badge/SemVer-2.0-brightgreen"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/license-Apache 2.0-blue"></a>
</div>

### Recent Updates

- **2024.10.15**
    - Completed the initial version of the code, including three modules: object detection, semantic segmentation, and corner direction recognition.
- **2024.11.2**
    - Added new YOLOv11 object detection models and edge detection models.
    - Increased automatic downloading and reduced package size.
    - Added ONNX-GPU inference support and provided benchmark test results.
    - Added online example usage.

### Introduction

💡✨ RapidTableDetection is a powerful and efficient table detection system that supports various types of tables, including those in papers, journals, magazines, invoices, receipts, and sign-in sheets.

🚀 It supports versions derived from PaddlePaddle and YOLO, with the default model combination requiring only 1.2 seconds for single-image CPU inference, and 0.4 seconds for the smallest ONNX-GPU (V100) combination, or 0.2 seconds for the PaddlePaddle-GPU version.

🛠️ It supports free combination and independent training optimization of three modules, providing ONNX conversion scripts and fine-tuning training solutions.

🌟 The whl package is easy to integrate and use, providing strong support for downstream OCR, table recognition, and data collection.

Refer to the implementation solution of the [2nd place in the Baidu Table Detection Competition](https://aistudio.baidu.com/projectdetail/5398861?searchKeyword=%E8%A1%A8%E6%A0%BC%E6%A3%80%E6%B5%8B%E5%A4%A7%E8%B5%9B&searchTab=ALL), and retrain with a large amount of real-world scenario data.
![img.png](readme_resource/structure.png) \
The training dataset is acknowledged. The author works on open-source projects during spare time, please support by giving a star.


### Usage Recommendations

- Document scenarios: No perspective rotation, use only object detection.
- Photography scenarios with small angle rotation (-90~90): Default top-left corner, do not use corner direction recognition.
- Use the online experience to find the suitable model combination for your scenario.

### Online Experience
[modelscope](https://www.modelscope.cn/studios/jockerK/RapidTableDetDemo) [huggingface](https://huggingface.co/spaces/Joker1212/RapidTableDetection)

### Effect Demonstration

![res_show.jpg](readme_resource/res_show.jpg)![res_show2.jpg](readme_resource/res_show2.jpg)

### Installation

Models will be automatically downloaded, or you can download them from the repository [modelscope model warehouse](https://www.modelscope.cn/models/jockerK/TableExtractor).

``` python {linenos=table}
pip install rapid-table-det
```

#### Parameter Explanation

Default values:
- `use_cuda: False`: Enable GPU acceleration for inference.
- `obj_model_type="yolo_obj_det"`: Object detection model type.
- `edge_model_type="yolo_edge_det"`: Edge detection model type.
- `cls_model_type="paddle_cls_det"`: Corner direction classification model type.


Since ONNX has limited GPU acceleration, it is still recommended to directly use YOLOX or install PaddlePaddle for faster model execution (I can provide the entire process if needed).
The PaddlePaddle S model, due to quantization, actually slows down and reduces accuracy, but significantly reduces model size.


| `model_type`         | Task Type | Training Source                                 | Size     | Single Table Inference Time (V100-16G, cuda12, cudnn9, ubuntu) |
|:---------------------|:---------|:-------------------------------------|:-------|:-------------------------------------|
| **yolo_obj_det**     | Table Object Detection | `yolo11-l`                           | `100m` | `cpu:570ms, gpu:400ms`               |
| `paddle_obj_det`     | Table Object Detection | `paddle yoloe-plus-x`                | `380m` | `cpu:1000ms, gpu:300ms`              |
| `paddle_obj_det_s`   | Table Object Detection | `paddle yoloe-plus-x + quantization` | `95m`  | `cpu:1200ms, gpu:1000ms`             |
| **yolo_edge_det**    | Semantic Segmentation | `yolo11-l-segment`                   | `108m` | `cpu:570ms, gpu:200ms`               |
| `yolo_edge_det_s`    | Semantic Segmentation | `yolo11-s-segment`                   | `11m`  | `cpu:260ms, gpu:200ms`               |
| `paddle_edge_det`    | Semantic Segmentation | `paddle-dbnet`                       | `99m`  | `cpu:1200ms, gpu:120ms`              |
| `paddle_edge_det_s`  | Semantic Segmentation | `paddle-dbnet + quantization`        | `25m`  | `cpu:860ms, gpu:760ms`               |
| **paddle_cls_det**   | Direction Classification | `paddle pplcnet`                     | `6.5m` | `cpu:70ms, gpu:60ms`                 |

Execution parameters:
- `det_accuracy=0.7`
- `use_obj_det=True`
- `use_edge_det=True`
- `use_cls_det=True`

### Quick Start

``` python {linenos=table}
from rapid_table_det.inference import TableDetector

img_path = f"tests/test_files/chip.jpg"
table_det = TableDetector()

result, elapse = table_det(img_path)
obj_det_elapse, edge_elapse, rotate_det_elapse = elapse
print(
    f"obj_det_elapse:{obj_det_elapse}, edge_elapse={edge_elapse}, rotate_det_elapse={rotate_det_elapse}"
)
# Output visualization
# import os
# import cv2
# from rapid_table_det.utils.visuallize import img_loader, visuallize, extract_table_img
# 
# img = img_loader(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# file_name_with_ext = os.path.basename(img_path)
# file_name, file_ext = os.path.splitext(file_name_with_ext)
# out_dir = "rapid_table_det/outputs"
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
# extract_img = img.copy()
# for i, res in enumerate(result):
#     box = res["box"]
#     lt, rt, rb, lb = res["lt"], res["rt"], res["rb"], res["lb"]
#     # With detection box and top-left corner position
#     img = visuallize(img, box, lt, rt, rb, lb)
#     # Perspective transformation to extract table image
#     wrapped_img = extract_table_img(extract_img.copy(), lt, rt, rb, lb)
#     cv2.imwrite(f"{out_dir}/{file_name}-extract-{i}.jpg", wrapped_img)
# cv2.imwrite(f"{out_dir}/{file_name}-visualize.jpg", img)

```
### Using PaddlePaddle Version
You must download the models and specify their locations!
``` python {linenos=table}
#(default installation is GPU version, you can override with CPU version paddlepaddle)
pip install rapid-table-det-paddle 
```
```python
from rapid_table_det_paddle.inference import TableDetector

img_path = f"tests/test_files/chip.jpg"

table_det = TableDetector(
    obj_model_path="models/obj_det_paddle",
    edge_model_path="models/edge_det_paddle",
    cls_model_path="models/cls_det_paddle",
    use_obj_det=True,
    use_edge_det=True,
    use_cls_det=True,
)
result, elapse = table_det(img_path)
obj_det_elapse, edge_elapse, rotate_det_elapse = elapse
print(
    f"obj_det_elapse:{obj_det_elapse}, edge_elapse={edge_elapse}, rotate_det_elapse={rotate_det_elapse}"
)
# more than one table in one image
# img = img_loader(img_path)
# file_name_with_ext = os.path.basename(img_path)
# file_name, file_ext = os.path.splitext(file_name_with_ext)
# out_dir = "rapid_table_det_paddle/outputs"
# if not os.path.exists(out_dir):
#     os.makedirs(out_dir)
# extract_img = img.copy()
# for i, res in enumerate(result):
#     box = res["box"]
#     lt, rt, rb, lb = res["lt"], res["rt"], res["rb"], res["lb"]
#     # With detection box and top-left corner position
#     img = visuallize(img, box, lt, rt, rb, lb)
#     # Perspective transformation to extract table image
#     wrapped_img = extract_table_img(extract_img.copy(), lt, rt, rb, lb)
#     cv2.imwrite(f"{out_dir}/{file_name}-extract-{i}.jpg", wrapped_img)
# cv2.imwrite(f"{out_dir}/{file_name}-visualize.jpg", img)

```

## FAQ (Frequently Asked Questions)

1. **Q: How to fine-tune the model for specific scenarios?**
    - A: Refer to this project, which provides detailed visualization steps and datasets. You can get the PaddlePaddle inference model from [Baidu Table Detection Competition](https://aistudio.baidu.com/projectdetail/5398861?searchKeyword=%E8%A1%A8%E6%A0%BC%E6%A3%80%E6%B5%8B%E5%A4%A7%E8%B5%9B&searchTab=ALL). For YOLOv11, use the official script, which is simple enough, and convert the data to COCO format for training as per the official guidelines.
2. **Q: How to export ONNX?**
    - A: For PaddlePaddle models, use the `onnx_transform.ipynb` file in the `tools` directory of this project. For YOLOv11, follow the official method, which can be done in one line.
3. **Q: Can distorted images be corrected?**
    - A: This project only handles rotation and perspective scenarios for table extraction. For distorted images, you need to correct the distortion first.

### Acknowledgments

- [2nd Place Solution in Baidu Table Detection Competition](https://aistudio.baidu.com/projectdetail/5398861?searchKeyword=%E8%A1%A8%E6%A0%BC%E6%A3%80%E6%B5%8B%E5%A4%A7%E8%B5%9B&searchTab=ALL)
- [WTW Natural Scene Table Dataset](https://tianchi.aliyun.com/dataset/108587)
- [FinTabNet PDF Document Table Dataset](https://developer.ibm.com/exchanges/data/all/fintabnet/)
- [TableBank Table Dataset](https://doc-analysis.github.io/tablebank-page/)
- [TableGeneration Table Auto-Generation Tool](https://github.com/WenmuZhou/TableGeneration)

### Contribution Guidelines

Pull requests are welcome. For major changes, please open an issue to discuss what you would like to change.

If you have other good suggestions and integration scenarios, the author will actively respond and support them.

### Open Source License

This project is licensed under the [Apache 2.0](https://github.com/RapidAI/TableStructureRec/blob/c41bbd23898cb27a957ed962b0ffee3c74dfeff1/LICENSE) open source license.

