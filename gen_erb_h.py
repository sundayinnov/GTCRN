import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("/home/gtcrn/gtcrn.onnx")
w = None
for node in m.graph.node:
    if node.op_type == "MatMul" and "linear" in node.name:
        for init in m.graph.initializer:
            if init.name == node.input[1]:
                w = numpy_helper.to_array(init).astype(np.float32)
                break
        break

assert w is not None
assert w.shape == (192, 64), w.shape

with open("/home/gtcrn/c_test/erb_w.h", "w") as f:
    f.write("#ifndef ERB_W_H\n#define ERB_W_H\n")
    f.write("static const float erb_w[192][64] = {\n")
    for i in range(192):
        row = ",".join(f"{v:.8f}f" for v in w[i])
        f.write("{" + row + "},\n")
    f.write("};\n#endif\n")
print("written erb_w.h, shape =", w.shape)
