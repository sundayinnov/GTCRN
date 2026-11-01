import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("/home/gtcrn/gtcrn.onnx")

# dpgrnn1.inter_rnn.rnn1/rnn2, dpgrnn2.inter_rnn.rnn1/rnn2
targets = [
    ('node_gru_5__1',  'val_703'),    # dpgrnn1.inter_rnn.rnn1.bias
    ('node_gru_6__1',  'val_750'),    # dpgrnn1.inter_rnn.rnn2.bias
    ('node_gru_9__1',  'val_1003'),   # dpgrnn2.inter_rnn.rnn1.bias
    ('node_gru_10__1', 'val_1050'),   # dpgrnn2.inter_rnn.rnn2.bias
]

# 找每个 GRU 的 bias 输入名
bias_map = {}
for node in m.graph.node:
    if node.op_type == 'GRU':
        for name in node.input:
            for init in m.graph.initializer:
                if init.name == name:
                    w = numpy_helper.to_array(init)
                    if w.shape == (1, 48) or w.shape == (48,):
                        bias_map[node.name] = name
                        print(f'{node.name}: bias = {name}, shape={w.shape}, max_abs={np.abs(w).max():.6f}')

with open("/home/gtcrn/c_test/dpgrnn_bias.h", "w") as f:
    f.write("#ifndef DPGRNN_BIAS_H\n#define DPGRNN_BIAS_H\n#include <stdint.h>\n\n")
    for node_name, c_name in targets:
        if node_name not in bias_map:
            print(f'WARNING: {node_name} bias not found!')
            continue
        bias_name = bias_map[node_name]
        for init in m.graph.initializer:
            if init.name == bias_name:
                b = numpy_helper.to_array(init).flatten().astype(np.float32)
                maxv = float(np.abs(b).max())
                scale = maxv / 127.0 if maxv > 0 else 1.0
                q = np.round(b / scale).astype(np.int8)
                f.write(f"#define {c_name}_scale {scale:.8f}f\n")
                f.write(f"static const int8_t {c_name}[48] __attribute__((aligned(4))) = {{\n    ")
                for i, v in enumerate(q):
                    f.write(f"{int(v)}, ")
                    if (i+1) % 16 == 0: f.write("\n    ")
                f.write("\n};\n\n")
                print(f'{c_name}: scale={scale:.6f}, max_abs={maxv:.4f}')
    f.write("#endif\n")

print("written /home/gtcrn/c_test/dpgrnn_bias.h")
