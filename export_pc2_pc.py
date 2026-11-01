import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("/home/gtcrn/gtcrn.onnx")
w = None
for init in m.graph.initializer:
    if init.name == "encoder.en_convs.2.point_conv2.weight":
        w = numpy_helper.to_array(init).astype(np.float32)
        break

assert w is not None and w.shape == (8, 16, 1, 1), w.shape

OC = 8
scales, int8_w = [], np.zeros((OC, 16), dtype=np.int8)
for oc in range(OC):
    maxv = float(np.abs(w[oc]).max())
    s = maxv / 127.0 if maxv > 0 else 1.0
    scales.append(s)
    int8_w[oc] = np.round(w[oc, :, 0, 0] / s).astype(np.int8)

with open("/home/gtcrn/c_test/pc2_pc.h", "w") as f:
    f.write("#ifndef PC2_PC_H\n#define PC2_PC_H\n#include <stdint.h>\n\n")
    f.write("static const float pc2_scale[8] = {\n    ")
    f.write(", ".join(f"{s:.8f}f" for s in scales))
    f.write("\n};\n\n")
    f.write("static const int8_t pc2_w[8][16] __attribute__((aligned(4))) = {\n")
    for oc in range(OC):
        f.write("    {" + ",".join(str(int(v)) for v in int8_w[oc]) + "},\n")
    f.write("};\n\n#endif\n")

print("point_conv2 per-channel scales:")
for oc in range(OC):
    print(f"  oc={oc}: scale={scales[oc]:.6f}, max_abs={float(np.abs(w[oc]).max()):.4f}")
