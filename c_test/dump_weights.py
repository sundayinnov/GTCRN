import onnx, onnx.numpy_helper as nh, numpy as np
m = onnx.load("gtcrn.onnx")
init = {i.name: nh.to_array(i) for i in m.graph.initializer}

# 列出 decoder 相关权重
for name, arr in init.items():
    if "de_convs" in name and ("weight" in name or "bias" in name):
        print(f"{name:55s} shape={arr.shape} sum={arr.sum():.4f} "
              f"min={arr.min():.4f} max={arr.max():.4f}")
