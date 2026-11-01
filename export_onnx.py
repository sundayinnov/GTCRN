import torch
import os
from gtcrn import GTCRN

# 1. 加载 DNS3 预训练模型
print("正在加载模型...")
model = GTCRN().eval()
ckpt_path = os.path.join('checkpoints', 'model_trained_on_dns3.tar')
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['model'])

# 2. 构造一个假的输入张量 (Dummy Input)
# 形状说明: (Batch_Size=1, Freq_Bins=257, Time_Frames=10, Real_Imag=2)
# 这里的 10 就是固定的时间帧数
dummy_input = torch.randn(1, 257, 10, 2)

# 3. 导出为 ONNX 格式
print("正在导出 ONNX 模型...")
onnx_file_path = "gtcrn.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file_path, 
    export_params=True,
    opset_version=11,           
    do_constant_folding=True,   
    input_names=['input_spec'], 
    output_names=['output_mask']
    # 删除了 dynamic_axes，固定为静态形状 [1, 257, 10, 2]
)

print(f"导出成功！文件已保存为: {onnx_file_path}")
print(f"文件大小: {os.path.getsize(onnx_file_path) / 1024:.2f} KB")