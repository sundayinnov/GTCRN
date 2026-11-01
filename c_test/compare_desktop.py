import numpy as np
import onnxruntime as ort
import wave

def read_wav(filename):
    with wave.open(filename, 'rb') as f:
        p = f.getparams()
        print(f"Input: {p.nchannels} ch, {p.framerate} Hz, {p.sampwidth*8} bit")
        data = f.readframes(f.getnframes())
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

def stft(x, n_fft=512, hop=256):
    window = np.sqrt(np.hanning(n_fft))
    pad = n_fft // 2
    x = np.pad(x, (pad, pad), mode='reflect')
    n_frames = (len(x) - n_fft) // hop + 1
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n_fft),
        strides=(hop * x.strides[0], x.strides[0])
    ) * window
    return np.fft.rfft(frames, axis=1).T

def istft(spec, n_fft=512, hop=256):
    window = np.sqrt(np.hanning(n_fft))
    frames = np.fft.irfft(spec.T, n=n_fft) * window
    out_len = hop * (frames.shape[0] - 1) + n_fft
    audio = np.zeros(out_len)
    wsum  = np.zeros(out_len)
    for i in range(frames.shape[0]):
        s = i * hop
        audio[s:s+n_fft] += frames[i]
        wsum [s:s+n_fft] += window ** 2
    pad = n_fft // 2
    audio, wsum = audio[pad:-pad], wsum[pad:-pad]
    wsum[wsum < 1e-8] = 1.0
    return audio / wsum

def main():
    x = read_wav("test_desktop.wav")
    spec = stft(x, 512, 256)                  # (F=257, T)
    T = spec.shape[1]
    print(f"STFT shape: {spec.shape}")

    spec_c = np.stack([spec.real, spec.imag], axis=-1)  # (F, T, 2)

    sess = ort.InferenceSession("gtcrn.onnx", providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    print(f"ONNX input: {in_name} shape={sess.get_inputs()[0].shape}")
    print(f"ONNX output shape: {sess.get_outputs()[0].shape}")

    chunk = 10
    out = np.zeros_like(spec_c)
    dbg = 0

    for i in range(0, T, chunk):
        end = min(i + chunk, T)
        c = spec_c[:, i:end, :]
        if end - i < chunk:
            c = np.pad(c, ((0,0), (0, chunk-(end-i)), (0,0)))
        cin = c[np.newaxis, ...].astype(np.float32)          # (1, F, T, 2)
        cout = sess.run(None, {in_name: cin})[0]             # (1, F, T, 2)

        if dbg < 5:
            s, e = cin[0], cout[0]
            sm = np.sqrt(s[...,0]**2 + s[...,1]**2) + 1e-8
            em = np.sqrt(e[...,0]**2 + e[...,1]**2)
            M = em / sm
            print(f"[PY] chunk{dbg} mask |M|: "
                  f"min={M.min():.4f} max={M.max():.4f} mean={M.mean():.4f}")
            dbg += 1

        out[:, i:end, :] = cout[0, :, :end-i, :]

    out_spec = out[...,0] + 1j * out[...,1]
    y = np.clip(istft(out_spec, 512, 256), -1.0, 1.0)
    with wave.open("desktop_denoised_py.wav", 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes((y * 32767).astype(np.int16).tobytes())
    print("Done -> desktop_denoised_py.wav")

if __name__ == "__main__":
    main()
