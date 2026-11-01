import onnx
from onnx import numpy_helper
import numpy as np

print("正在读取 gtcrn.onnx...")
onnx_model = onnx.load("gtcrn.onnx")

with open("gtcrn_weights.h", "w") as f:
    f.write("#ifndef GTCRN_WEIGHTS_H\n#define GTCRN_WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n\n")
    
    # 遍历所有初始化的权重（常量）
    for initializer in onnx_model.graph.initializer:
        name = initializer.name.replace('.', '_').replace('/', '_').replace(':', '_')
        
        # 将权重转为 numpy 数组
        weight_array = numpy_helper.to_array(initializer)
        
        # 按 float32 导出
        flat_array = weight_array.flatten()
        
        print(f"导出权重: {name}, 形状: {weight_array.shape}, 元素个数: {len(flat_array)}")
        
        # 写入 C 数组
        f.write(f"// Shape: {weight_array.shape}\n")
        f.write(f"static const float {name}[{len(flat_array)}] = {{\n    ")
        
        for i, val in enumerate(flat_array):
            f.write(f"{val:.8f}f, ")
            if (i + 1) % 8 == 0:
                f.write("\n    ")
        f.write("\n};\n\n")
        
    f.write("#endif // GTCRN_WEIGHTS_H\n")

print("\n权重提取完毕！已生成 gtcrn_weights.h")
