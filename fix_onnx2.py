import os
import onnx
from onnx import helper

# 指定目录路径
model_dir = 'rapid_table_det/models'

# 加载现有 ONNX 模型
model_path = os.path.join(model_dir, 'obj_det.onnx')
model = onnx.load(model_path)

# 删除指定的节点
deleted_nodes = ['p2o.Squeeze.3', 'p2o.Squeeze.5']
nodes_to_delete = [node for node in model.graph.node if node.name in deleted_nodes]
for node in nodes_to_delete:
    model.graph.node.remove(node)


# 创建新的 squeeze 节点
def create_squeeze_node(name, input_name):
    return helper.make_node(
        'Squeeze',
        inputs=[input_name],
        outputs=[input_name + '_squeezed'],
        name=name + '_Squeeze',
        axes=[1]  # 假设需要squeeze的轴是第0轴
    )


# 找到 gather8 和 gather10 的输出
gather8_output = None
gather10_output = None
# 找到 'p2o.Gather.1' 节点
for node in model.graph.node:
    if node.name == 'p2o.Gather.0':
        gather_output_name = node.output[0]
        # break
    if node.name == 'p2o.Gather.2':
        gather_output_name1 = node.output[0]
        # break

for node in model.graph.node:
    if node.name == 'p2o.Gather.8':
        node.input[0] = gather_output_name
        gather8_output = node.output[0]
    elif node.name == 'p2o.Gather.10':
        node.input[0] = gather_output_name1
        gather10_output = node.output[0]

# 创建新的 squeeze 节点并插入到模型中
if gather8_output:
    new_squeeze_node_8 = create_squeeze_node('p2o.Gather.8', gather8_output)
    model.graph.node.append(new_squeeze_node_8)

if gather10_output:
    new_squeeze_node_10 = create_squeeze_node('p2o.Gather.10', gather10_output)
    model.graph.node.append(new_squeeze_node_10)

# 更新依赖于 gather8 和 gather10 的节点输入
for node in model.graph.node:
    if node.name == 'p2o.Cast.0':
        node.input[0] = new_squeeze_node_8.output[0]

    if node.name == 'p2o.Gather.12':
        node.input[1] = new_squeeze_node_10.output[0]

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