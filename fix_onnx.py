import onnx
import os
if __name__ == '__main__':
    deleted_nodes = ['p2o.Squeeze.3','p2o.Squeeze.5']
    # 指定目录路径
    model_dir = 'rapid_table_det/models'

    # 加载现有 ONNX 模型
    model_path = os.path.join(model_dir, 'obj_det.onnx')
    model = onnx.load(model_path)

    # 找到 'p2o.Squeeze.3' 节点并删除
    nodes_to_delete = []
    for node in model.graph.node:
        if node.name in deleted_nodes:
            nodes_to_delete.append(node)

    # 删除节点
    for node in nodes_to_delete:
        model.graph.node.remove(node)

    # 找到 'p2o.Gather.1' 节点
    for node in model.graph.node:
        if node.name == 'p2o.Gather.0':
            gather_output_name = node.output[0]
            break
        if node.name == 'p2o.Gather.2':
            gather_output_name1 = node.output[0]
            break
    # 更新 'p2o.Gather.8' 节点的输入
    for node in model.graph.node:
        if node.name == 'p2o.Gather.8':
            node.input[0] = gather_output_name
            break
        if node.name == 'p2o.Gather.10':
            node.input[0] = gather_output_name1
            break
    # 保存修改后的模型
    modified_model_path = os.path.join(model_dir, 'modified_obj_det.onnx')
    onnx.save(model, modified_model_path)

    # 使用 onnx-check 验证模型
    import onnx.checker
    try:
        onnx.checker.check_model(modified_model_path)
        print("Model is valid.")
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        print(f"Invalid model: {e}")
