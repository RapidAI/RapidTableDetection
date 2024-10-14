from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
root_dir = Path(__file__).resolve().parent.parent
root_dir_str = str(root_dir)
if __name__ == '__main__':
    model_dir = f"rapid_table_det/models"
    # 加载原始模型
    pre_fix = "db_net"
    original_model_path = f"{pre_fix}.onnx"
    quantized_model_path = f"{pre_fix}_quantized.onnx"
    original_model_path = f"{root_dir_str}/{model_dir}/{original_model_path}"
    quantized_model_path = f"{root_dir_str}/{model_dir}/{quantized_model_path}"
    # ori_model = onnx.load(original_model_path)
    quant_pre_process(original_model_path, quantized_model_path, auto_merge=True)
    # 进行动态量化
    quantize_dynamic(
        model_input=quantized_model_path,
        model_output=quantized_model_path,
        weight_type=QuantType.QInt8
    )

    # 验证量化后的模型
    quantized_model = onnx.load(quantized_model_path)
    onnx.checker.check_model(quantized_model)
    print("Quantized model is valid.")
