# paddle2onnx --model_dir models/obj_det --model_filename model.pdmodel --params_filename model.pdiparams --save_file model.onnx --opset_version 16 --enable_onnx_checker True
# paddle2onnx --model_dir models/db_net --model_filename model.pdmodel --params_filename model.pdiparams --save_file model.onnx --opset_version 16 --enable_onnx_checker True
# paddle2onnx --model_dir models/pplcnet --model_filename model.pdmodel --params_filename model.pdiparams --save_file model.onnx --opset_version 16 --enable_onnx_checker True
# onnxslim model.onnx slim.onnx
