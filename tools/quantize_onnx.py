import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
def quantize_model(root_dir_str, model_dir, pre_fix):

    original_model_path = f"{pre_fix}.onnx"
    quantized_model_path = f"{pre_fix}_quantized.onnx"
    # quantized_model_path = original_model_path
    original_model_path = f"{root_dir_str}/{model_dir}/{original_model_path}"
    quantized_model_path = f"{root_dir_str}/{model_dir}/{quantized_model_path}"
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
if __name__ == '__main__':
    root_dir_str = ".."
    model_dir = f"rapid_table_det/models"
    # quantize_model(root_dir_str, model_dir, "obj_det")
    quantize_model(root_dir_str, model_dir, "edge_det")
