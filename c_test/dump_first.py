import onnx, onnxruntime as ort, numpy as np, wave

m = onnx.load("gtcrn.onnx")
want = ["node_ConvTranspose_1619", "node_ConvTranspose_1620"]
for node in m.graph.node:
    if node.name in want:
        m.graph.node.append(onnx.helper.make_node("Identity", [node.output[0]], [node.name + "_out"]))
        m.graph.output.append(onnx.helper.make_tensor_value_info(node.name + "_out", onnx.TensorProto.FLOAT, None))
onnx.save(m, "gtcrn_first.onnx")

def read_wav(p):
    with wave.open(p, 'rb') as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
def stft(x, n=512, h=256):
    w = np.sqrt(np.hanning(n)); p = n//2
    x = np.pad(x, (p,p), mode='reflect')
    nf = (len(x)-n)//h+1
    fr = np.lib.stride_tricks.as_strided(x, (nf,n), (h*x.strides[0], x.strides[0]))*w
    return np.fft.rfft(fr, axis=1).T

x = read_wav("noisy_1007.wav")
sp = stft(x)[:, :10]
cin = np.stack([sp.real, sp.imag], -1)[np.newaxis].astype(np.float32)
sess = ort.InferenceSession("gtcrn_first.onnx", providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
res = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {in_name: cin})))

for key in ["node_ConvTranspose_1619_out", "node_ConvTranspose_1620_out"]:
    arr = res[key].flatten()
    print(f"\n{key} (前 10 个):")
    for i in range(10):
        print(f"  [{i}] {arr[i]:+.6f}")
