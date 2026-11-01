import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("/home/gtcrn/gtcrn.onnx")
w = None
for init in m.graph.initializer:
    if init.name == "encoder.en_convs.2.point_conv1.weight":
        w = numpy_helper.to_array(init).astype(np.float32)
        break

assert w is not None and w.shape == (16, 24, 1, 1), w.shape

OC = 16
scales, int8_w = [], np.zeros((OC, 24), dtype=np.int8)
for oc in range(OC):
    maxv = float(np.abs(w[oc]).max())
    s = maxv / 127.0 if maxv > 0 else 1.0
    scales.append(s)
    int8_w[oc] = np.round(w[oc, :, 0, 0] / s).astype(np.int8)

with open("/home/gtcrn/c_test/pc1_pc.h", "w") as f:
    f.write("#ifndef PC1_PC_H\n#define PC1_PC_H\n#include <stdint.h>\n\n")
    f.write("static const float pc1_scale[16] = {\n    ")
    f.write(", ".join(f"{s:.8f}f" for s in scales))
    f.write("\n};\n\n")
    f.write("static const int8_t pc1_w[16][24] __attribute__((aligned(4))) = {\n")
    for oc in range(OC):
        f.write("    {" + ",".join(str(int(v)) for v in int8_w[oc]) + "},\n")
    f.write("};\n\n#endif\n")

print("point_conv1 per-channel scales:")
for oc in range(OC):
    print(f"  oc={oc:2d}: scale={scales[oc]:.6f}, max_abs={float(np.abs(w[oc]).max()):.4f}")
