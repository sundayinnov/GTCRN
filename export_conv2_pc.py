import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("/home/gtcrn/gtcrn.onnx")
w = None
for init in m.graph.initializer:
    if init.name == "encoder.en_convs.1.conv.weight":
        w = numpy_helper.to_array(init).astype(np.float32)
        break

assert w is not None and w.shape == (16, 8, 1, 5), w.shape

OC = 16
scales = []
int8_w = np.zeros((OC, 8, 5), dtype=np.int8)
for oc in range(OC):
    maxv = float(np.abs(w[oc]).max())
    s = maxv / 127.0 if maxv > 0 else 1.0
    scales.append(s)
    int8_w[oc] = np.round(w[oc, :, 0, :] / s).astype(np.int8)

with open("/home/gtcrn/c_test/conv2_pc.h", "w") as f:
    f.write("#ifndef CONV2_PC_H\n#define CONV2_PC_H\n#include <stdint.h>\n\n")
    f.write("// per-channel int8 weights, shape (16,8,1,5) -> flatten to (16,40)\n")
    f.write("static const float conv2_scale[16] = {\n    ")
    f.write(", ".join(f"{s:.8f}f" for s in scales))
    f.write("\n};\n\n")
    f.write("static const int8_t conv2_w[16][40] __attribute__((aligned(4))) = {\n")
    for oc in range(OC):
        flat = int8_w[oc].flatten()
        f.write("    {" + ",".join(str(int(v)) for v in flat) + "},\n")
    f.write("};\n\n#endif\n")

print("Conv2 per-channel scales:")
for oc in range(OC):
    print(f"  oc={oc:2d}: scale={scales[oc]:.6f}, max_abs={float(np.abs(w[oc]).max()):.4f}")
print("written /home/gtcrn/c_test/conv2_pc.h")
