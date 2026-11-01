import numpy as np
import onnxruntime as ort
import wave

def read_wav(filename, max_seconds=10):
    with wave.open(filename, 'rb') as f:
        params = f.getparams()
        print(f"Input: {params.nchannels} ch, {params.framerate} Hz, {params.sampwidth*8} bit")
        n_frames = f.getnframes()
        max_samples = params.framerate * max_seconds
        if n_frames > max_samples:
            n_frames = max_samples
        data = f.readframes(n_frames)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, params.framerate

def stft(x, n_fft=512, hop=256):
    window = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    pad = n_fft // 2
    x = np.pad(x, (pad, pad), mode='reflect')
    num_frames = (len(x) - n_fft) // hop + 1
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(num_frames, n_fft), strides=(hop * x.strides[0], x.strides[0])
    )
    frames = frames * window
    spec = np.fft.rfft(frames, axis=1).T
    return spec, window

def istft(spec, window, n_fft=512, hop=256):
    frames = np.fft.irfft(spec.T, n=n_fft, axis=1)
    frames = frames * window
    output_len = hop * (frames.shape[0] - 1) + n_fft
    audio = np.zeros(output_len, dtype=np.float32)
    window_sum = np.zeros(output_len, dtype=np.float32)
    for i in range(frames.shape[0]):
        start = i * hop
        audio[start:start+n_fft] += frames[i]
        window_sum[start:start+n_fft] += window ** 2
    pad = n_fft // 2
    audio = audio[pad:-pad]
    window_sum = window_sum[pad:-pad]
    window_sum[window_sum < 1e-4] = 1.0
    audio = audio / window_sum
    return audio

def main():
    x, sr = read_wav("noisy_1007.wav", max_seconds=10)
    print(f"Audio length: {len(x)} samples")
    
    spec, window = stft(x, 512, 256)
    spec_c = np.stack([spec.real, spec.imag], axis=-1)
    
    sess = ort.InferenceSession("gtcrn.onnx", providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    chunk_size = 10
    out_spec_c = np.zeros_like(spec_c)
    T = spec.shape[1]
    
    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)
        chunk = spec_c[:, i:end, :]
        if end - i < chunk_size:
            pad_len = chunk_size - (end - i)
            chunk = np.pad(chunk, ((0, 0), (0, pad_len), (0, 0)))
            
        chunk_input = chunk[np.newaxis, ...].astype(np.float32)
        chunk_out = sess.run(None, {input_name: chunk_input})[0]
        
        if i == 0:
            m_abs = np.sqrt(chunk_out[0, :, :, 0]**2 + chunk_out[0, :, :, 1]**2)
            print(f"Python Mask Chunk0: min={m_abs.min():.3f} max={m_abs.max():.3f} mean={m_abs.mean():.3f}")

        out_spec_c[:, i:end, :] = chunk_out[0, :, :end - i, :]

    out_spec = out_spec_c[..., 0] + 1j * out_spec_c[..., 1]
    out_audio = istft(out_spec, window, 512, 256)
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
