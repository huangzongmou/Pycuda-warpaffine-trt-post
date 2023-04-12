import onnx
from onnx import numpy_helper, helper, shape_inference
import onnxruntime as ort
import numpy as np

model_path = './model/yolov8n.onnx'
model = onnx.load(model_path)

last_node = model.graph.node[-1]
output_name = "output"
perm = [0, 2, 1]

transpose_node = helper.make_node('Transpose', inputs=["Transpose_output"], outputs=[output_name], perm=perm)
last_node.output[0] = "Transpose_output"
model.graph.node.extend([transpose_node])

last_node1 = model.graph.node[-1]
output_tensor1 = last_node.output[0]

inferred_model = shape_inference.infer_shapes(model)
output_shape = inferred_model.graph.output[0].type.tensor_type.shape.dim
model.graph.output[0].name = output_name
model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = output_shape[2].dim_value
model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = output_shape[1].dim_value
# # Save the modified ONNX model
onnx.save(model, 'yolov8n_transpose.onnx')
