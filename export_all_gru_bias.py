import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("gtcrn.onnx")

# 万能查找函数：翻遍 initializer 和 Constant 节点
def find_const(name):
    for init in m.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init).flatten().astype(np.float32)
    for node in m.graph.node:
        if node.op_type == 'Constant' and name in node.output:
            for attr in node.attribute:
                if attr.name == 'value':
                    return numpy_helper.to_array(attr.t).flatten().astype(np.float32)
    return None

targets = [
    ('val_568', 'val_568_f'), ('val_650', 'val_650_f'), 
    ('val_703', 'val_703_f'), ('val_750', 'val_750_f'),
    ('val_868', 'val_868_f'), ('val_950', 'val_950_f'), 
    ('val_1003', 'val_1003_f'), ('val_1050', 'val_1050_f')
]

with open("c_test/dpgrnn_bias.h", "w") as f:
    f.write("#ifndef DPGRNN_BIAS_H\n#define DPGRNN_BIAS_H\n\n")
    for orig_name, c_name in targets:
        b = find_const(orig_name)
        if b is None:
            print(f"WARN: {orig_name} not found, generating zeros.")
            f.write(f"static const float {c_name}[48] = {{0.0f}};\n\n")
        else:
            f.write(f"static const float {c_name}[{len(b)}] = {{\n    ")
            for i, v in enumerate(b):
                f.write(f"{v:.8f}f, ")
                if (i+1) % 8 == 0: f.write("\n    ")
            f.write("\n};\n\n")
            print(f"OK: {c_name} (len {len(b)})")
    f.write("#endif\n")
print("written c_test/dpgrnn_bias.h")
