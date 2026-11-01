import onnx, onnxruntime as ort, numpy as np, wave

# ---- 生成 debug ONNX ----
m = onnx.load("gtcrn.onnx")
dbg = []
for node in m.graph.node:
    if node.op_type in ("Conv", "ConvTranspose", "GRU"):
        name = node.output[0]
        new = name + "_dbg"
        m.graph.node.append(onnx.helper.make_node("Identity", [name], [new]))
        dbg.append((node.op_type, node.name, new))
for _, _, new in dbg:
    m.graph.output.append(
        onnx.helper.make_tensor_value_info(new, onnx.TensorProto.FLOAT, None)
    )
onnx.save(m, "gtcrn_dbg.onnx")
print(f"debug 节点: {len(dbg)}")

# ---- 读 STFT ----
def read_wav(p):
    with wave.open(p, 'rb') as f:
        return np.frombuffer(f.readframes(f.getnframes()),
                             dtype=np.int16).astype(np.float32) / 32768.0

def stft(x, n_fft=512, hop=256):
    win = np.sqrt(np.hanning(n_fft))
    p = n_fft // 2
    x = np.pad(x, (p, p), mode='reflect')
    nf = (len(x) - n_fft) // hop + 1
    fr = np.lib.stride_tricks.as_strided(
        x, (nf, n_fft), (hop * x.strides[0], x.strides[0])
    ) * win
    return np.fft.rfft(fr, axis=1).T

x = read_wav("noisy_1007.wav")
sp = stft(x)
c0 = sp[:, :10]
cin = np.stack([c0.real, c0.imag], -1)[np.newaxis].astype(np.float32)

# ---- 推理 ----
sess = ort.InferenceSession("gtcrn_dbg.onnx", providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
outs = sess.run(None, {in_name: cin})
onames = [o.name for o in sess.get_outputs()]
res = dict(zip(onames, outs))

# ---- 按 graph 顺序打印 ----
i = 0
for node in onnx.load("gtcrn.onnx").graph.node:
    if node.op_type in ("Conv", "ConvTranspose", "GRU"):
        key = node.output[0] + "_dbg"
        if key in res:
            arr = res[key]
            print(f"[PY] #{i:02d} {node.op_type:15s} {node.name:35s} "
                  f"shape={arr.shape} min={arr.min():.4f} "
                  f"max={arr.max():.4f} mean={arr.mean():.4f}")
            i += 1
