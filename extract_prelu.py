import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("gtcrn.onnx")
slopes = []
for node in m.graph.node:
    if node.op_type == "PRelu":
        slope_name = node.input[1]
        for init in m.graph.initializer:
            if init.name == slope_name:
                s = numpy_helper.to_array(init).flatten()[0]
                slopes.append((node.name, float(s)))
                break

print("PReLU 总数:", len(slopes))
for i, (name, s) in enumerate(slopes):
    print(f"  [{i}] {name}: {s:.8f}")

with open("prelu_slopes.h", "w") as f:
    f.write("#ifndef PRELU_SLOPES_H\n#define PRELU_SLOPES_H\n")
    f.write(f"static const float prelu_slopes[{len(slopes)}] = {{\n")
    for i, (_, s) in enumerate(slopes):
        f.write(f"    {s:.8f}f,  // [{i}] {slopes[i][0]}\n")
    f.write("};\n#endif\n")
print("written prelu_slopes.h")
