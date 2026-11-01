import numpy as np
import onnxruntime as ort
import onnx

model = onnx.load("gtcrn.onnx")
sess = ort.InferenceSession("gtcrn.onnx")

# 寻找前两个 Conv 节点
conv_names = []
for node in model.graph.node:
    if node.op_type == "Conv" and "en_convs" in node.name:
        conv_names.append(node.output[0])

inferred_model = onnx.shape_inference.infer_shapes(model)
for out_info in inferred_model.graph.value_info:
    if out_info.name in conv_names[:2]:
        model.graph.output.append(out_info)

onnx.save(model, "temp_gtcrn.onnx")
sess = ort.InferenceSession("temp_gtcrn.onnx")

input_data = np.zeros((1, 257, 10, 2), dtype=np.float32)
input_data[:, :, :, 0] = 1.0

outputs = sess.run(conv_names[:2], {"input_spec": input_data})

print(f"Python Conv1: {outputs[0].flatten()[0]:.6f}")
print(f"Python Conv2: {outputs[1].flatten()[0]:.6f}")