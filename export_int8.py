import onnx
from onnx import numpy_helper
import numpy as np

print("正在读取 gtcrn.onnx...")
onnx_model = onnx.load("gtcrn.onnx")

with open("gtcrn_int8_weights.h", "w") as f:
    f.write("#ifndef GTCRN_INT8_WEIGHTS_H\n#define GTCRN_INT8_WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n\n")
    
    for initializer in onnx_model.graph.initializer:
        name = initializer.name.replace('.', '_').replace('/', '_').replace(':', '_')
        weight_array = numpy_helper.to_array(initializer)
        flat_array = weight_array.flatten()
        
        # 判断是权重还是偏置 (bias)
        is_bias = 'bias' in name.lower() or 'ln_weight' in name.lower() or 'ln_bias' in name.lower()
        
        if not is_bias:
            # 1. 对称量化权重为 INT8
            max_val = np.max(np.abs(flat_array))
            scale = max_val / 127.0 if max_val > 0 else 1.0
            
            int8_array = np.round(flat_array / scale).astype(np.int8)
            
            # 导出 scale 和 INT8 数组
            f.write(f"#define {name}_scale {scale:.8f}f\n")
            f.write(f"// Shape: {weight_array.shape}, Elements: {len(int8_array)}\n")
            # 加上 4 字节对齐，专为 dprod 准备
            f.write(f"static const int8_t {name}[{len(int8_array)}] __attribute__((aligned(4))) = {{\n    ")
            
            for i, val in enumerate(int8_array):
                f.write(f"{val}, ")
                if (i + 1) % 16 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")
            
        else:
            # 偏置 (bias) 和 LayerNorm 参数建议保留为 float32
            f.write(f"// Shape: {weight_array.shape}\n")
            f.write(f"static const float {name}[{len(flat_array)}] __attribute__((aligned(4))) = {{\n    ")
            for i, val in enumerate(flat_array):
                f.write(f"{val:.8f}f, ")
                if (i + 1) % 8 == 0:
                    f.write("\n    ")
            f.write("\n};\n\n")

    f.write("#endif // GTCRN_INT8_WEIGHTS_H\n")

print("\nINT8 权重导出完毕！生成了 gtcrn_int8_weights.h")
