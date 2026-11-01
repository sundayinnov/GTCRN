import numpy as np
import onnxruntime as ort
import wave
import math

# ================= 1. 读取音频 =================
def read_wav(filename):
    with wave.open(filename, 'rb') as f:
        params = f.getparams()
        print(f"Input: {params.nchannels} ch, {params.framerate} Hz, {params.sampwidth*8} bit")
        n_frames = f.getnframes()
        data = f.readframes(n_frames)
        # 转换为 float32 [-1, 1]
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio

# ================= 2. STFT (对齐原版) =================
def stft(x, n_fft=512, hop=256):
    # 原版使用 sqrt(hann)
    window = np.sqrt(np.hanning(n_fft)) 
    # 补零使得最后一帧完整
    pad = n_fft // 2
    x = np.pad(x, (pad, pad), mode='reflect')
    
    num_frames = (len(x) - n_fft) // hop + 1
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(num_frames, n_fft), strides=(hop * x.strides[0], x.strides[0])
    )
    frames = frames * window
    # rfft 返回 (257,)，转置为 (freq, time)
    spec = np.fft.rfft(frames, axis=1).T
    return spec

# ================= 3. ISTFT =================
def istft(spec, n_fft=512, hop=256):
    window = np.sqrt(np.hanning(n_fft))
    frames = np.fft.irfft(spec.T, n=n_fft)
    frames = frames * window
    
    output_len = hop * (frames.shape[0] - 1) + n_fft
    audio = np.zeros(output_len)
    window_sum = np.zeros(output_len)
    
    for i in range(frames.shape[0]):
        start = i * hop
        audio[start:start+n_fft] += frames[i]
        window_sum[start:start+n_fft] += window ** 2
        
    # 去除首尾补零
    pad = n_fft // 2
    audio = audio[pad:-pad]
    window_sum = window_sum[pad:-pad]
    
    # 避免除以零
    window_sum[window_sum < 1e-8] = 1.0
    audio = audio / window_sum
    return audio

# ================= 4. 主推理流程 =================
def main():
    # 读取
    x = read_wav("noisy.wav")
    
    # STFT
    spec = stft(x, 512, 256) # shape (257, T)
    T = spec.shape[1]
    print(f"STFT shape: {spec.shape}")
    
    # 扩展为复数 (F, T, 2)
    spec_c = np.stack([spec.real, spec.imag], axis=-1) # (257, T, 2)
    
    # 加载模型
    sess = ort.InferenceSession("gtcrn.onnx", providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    # 分块推理 (每次 10 帧)
    chunk_size = 10
    out_spec_c = np.zeros_like(spec_c)
    
    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)
        chunk = spec_c[:, i:end, :]
        
        # 如果不足 10 帧，补零
        if end - i < chunk_size:
            pad_len = chunk_size - (end - i)
            chunk = np.pad(chunk, ((0, 0), (0, pad_len), (0, 0)))
            
        # 调整形状为 (B, F, T, 2) -> (1, 257, 10, 2)
        chunk_input = chunk[np.newaxis, ...].astype(np.float32)
        
        # 推理
        chunk_out = sess.run(None, {input_name: chunk_input})[0] # (1, 257, 10, 2)
        
        # 截断有效帧并保存
        valid_len = end - i
        out_spec_c[:, i:end, :] = chunk_out[0, :, :valid_len, :]

    print("推理完成，正在合成音频...")

    # ISTFT
    out_spec = out_spec_c[..., 0] + 1j * out_spec_c[..., 1] # (257, T)
    out_audio = istft(out_spec, 512, 256)

    # 写出
    out_audio = np.clip(out_audio, -1.0, 1.0)
    out_int16 = (out_audio * 32767).astype(np.int16)
    with wave.open("enhanced_python.wav", 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(out_int16.tobytes())
        
    print("Done! Saved to enhanced_python.wav")

if __name__ == "__main__":
    main()
