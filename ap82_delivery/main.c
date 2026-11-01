#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "gtcrn.h"

#define PI           3.14159265358979323846f
#define FRAME_SIZE   512
#define HOP_SIZE     256
#define FFT_SIZE     512
#define NFFT_BINS    257
#define CHUNK_FRAMES 10
#define MAX_SECONDS  10

static float g_win[FRAME_SIZE];

static void die(const char *msg) { fprintf(stderr, "错误: %s\n", msg); exit(1); }

static unsigned int rd_u32(const unsigned char *p) {
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8)
         | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}
static unsigned short rd_u16(const unsigned char *p) {
    return (unsigned short)((unsigned short)p[0] | ((unsigned short)p[1] << 8));
}

static float *read_wav(const char *path, int *out_len, int *out_sr) {
    FILE *f = fopen(path, "rb");
    if (!f) die("无法打开输入 WAV 文件");
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 44) die("文件太小，不是合法 WAV");

    unsigned char *buf = (unsigned char *)malloc((size_t)sz);
    if (!buf) die("内存不足");
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) die("WAV 读取失败");
    fclose(f);

    if (memcmp(buf, "RIFF", 4) || memcmp(buf + 8, "WAVE", 4)) die("不是合法的 WAV 文件");

    int fmt = 1, ch = 1, sr = 16000, bits = 16;
    long data_off = -1, data_len = 0;
    long pos = 12;
    while (pos + 8 <= sz) {
        unsigned int csize = rd_u32(buf + pos + 4);
        long body = pos + 8;
        if (!memcmp(buf + pos, "fmt ", 4)) {
            fmt  = rd_u16(buf + body);
            ch   = rd_u16(buf + body + 2);
            sr   = (int)rd_u32(buf + body + 4);
            bits = rd_u16(buf + body + 14);
        } else if (!memcmp(buf + pos, "data", 4)) {
            data_off = body;
            data_len = (long)csize;
            if (data_off + data_len > sz) data_len = sz - data_off;
        }
        pos = body + (long)csize + (csize & 1);
    }
    if (data_off < 0) die("WAV 中找不到 data 块");
    if (ch < 1) die("声道数非法");

    int bps = bits / 8;
    if (bps <= 0) die("不支持的位深");
    long nframes = data_len / ((long)bps * ch);

    float *mono = (float *)malloc(sizeof(float) * (size_t)nframes);
    if (!mono) die("内存不足");
    const unsigned char *d = buf + data_off;

    for (long i = 0; i < nframes; i++) {
        float acc = 0.0f;
        for (int c = 0; c < ch; c++) {
            const unsigned char *p = d + (i * ch + c) * bps;
            float v = 0.0f;
            if (fmt == 3 && bits == 32) {
                memcpy(&v, p, 4);
            } else if (bits == 8) {
                v = ((int)p[0] - 128) / 128.0f;
            } else if (bits == 16) {
                short s; memcpy(&s, p, 2); v = s / 32768.0f;
            } else if (bits == 24) {
                int s = (int)p[0] | ((int)p[1] << 8) | ((int)p[2] << 16);
                if (s & 0x800000) s |= ~0xFFFFFF;
                v = s / 8388608.0f;
            } else if (bits == 32) {
                int s; memcpy(&s, p, 4); v = (float)(s / 2147483648.0);
            }
            acc += v;
        }
        mono[i] = acc / (float)ch;
    }
    free(buf);
    *out_len = (int)nframes;
    *out_sr = sr;
    return mono;
}

static void write_wav(const char *path, const float *data, int len, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) die("无法写出输出 WAV");
    int datasize = len * 2;
    int chunksize = 36 + datasize;
    int fmtsize = 16, srate = sr, brate = sr * 2;
    short fmt = 1, ch = 1, ba = 2, bits = 16;
    fwrite("RIFF", 1, 4, f);   fwrite(&chunksize, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);   fwrite(&fmtsize, 4, 1, f);
    fwrite(&fmt, 2, 1, f);     fwrite(&ch, 2, 1, f);
    fwrite(&srate, 4, 1, f);   fwrite(&brate, 4, 1, f);
    fwrite(&ba, 2, 1, f);      fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);   fwrite(&datasize, 4, 1, f);
    for (int i = 0; i < len; i++) {
        float x = data[i];
        if (x > 1.0f) x = 1.0f;
        if (x < -1.0f) x = -1.0f;
        short s = (short)lrintf(x * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
}

static void fft(float *re, float *im, int n, int inverse) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * (double)PI / len * (inverse ? 1.0 : -1.0);
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; j++) {
                double a = ang * j;
                float wr = (float)cos(a);
                float wi = (float)sin(a);
                float ur = re[i + j], ui = im[i + j];
                float vr = re[i + j + len / 2] * wr - im[i + j + len / 2] * wi;
                float vi = re[i + j + len / 2] * wi + im[i + j + len / 2] * wr;
                re[i + j] = ur + vr;                 im[i + j] = ui + vi;
                re[i + j + len / 2] = ur - vr;       im[i + j + len / 2] = ui - vi;
            }
        }
    }
    if (inverse) for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

int main(int argc, char **argv) {
    const char *inpath  = (argc > 1) ? argv[1] : "noisy.wav";
    const char *outpath = (argc > 2) ? argv[2] : "denoised_10s.wav";

    printf("=== GTCRN 降噪（仅处理前 %d 秒）===\n", MAX_SECONDS);

    int sr = 16000, nlen = 0;
    float *x = read_wav(inpath, &nlen, &sr);
    printf("输入文件 : %s\n", inpath);
    printf("采样率   : %d Hz\n", sr);
    printf("音频长度 : %.2f 秒 (%d 样本)\n", (double)nlen / sr, nlen);

    int total = (nlen < sr * MAX_SECONDS) ? nlen : sr * MAX_SECONDS;
    printf("处理范围 : 前 %.2f 秒 (%d 样本)\n\n", (double)total / sr, total);

    for (int i = 0; i < FRAME_SIZE; i++)
        g_win[i] = sqrtf(0.5f * (1.0f - cosf(2.0f * PI * i / FRAME_SIZE)));

    gtcrn_init();

    int pad = FRAME_SIZE / 2;
    int padded = total + 2 * pad;
    float *xpad = (float *)malloc((size_t)padded * sizeof(float));
    float *out  = (float *)calloc((size_t)padded, sizeof(float));
    float *wnrm = (float *)calloc((size_t)padded, sizeof(float));
    if (!xpad || !out || !wnrm) die("内存不足");
    for (int i = 0; i < pad; i++) xpad[i] = x[pad - i];
    memcpy(xpad + pad, x, sizeof(float) * (size_t)total);
    for (int i = 0; i < pad; i++) xpad[pad + total + i] = x[total - 2 - i];

    int nframes = (padded - FRAME_SIZE) / HOP_SIZE + 1;
    int nchunks = (nframes + CHUNK_FRAMES - 1) / CHUNK_FRAMES;
    printf("总帧数 %d，分 %d 个 chunk（每 chunk %d 帧）\n", nframes, nchunks, CHUNK_FRAMES);

    float fft_re[FFT_SIZE], fft_im[FFT_SIZE];

    for (int c = 0; c < nchunks; c++) {
        float in_re[NFFT_BINS * CHUNK_FRAMES] = {0};
        float in_im[NFFT_BINS * CHUNK_FRAMES] = {0};
        float o_re [NFFT_BINS * CHUNK_FRAMES] = {0};
        float o_im [NFFT_BINS * CHUNK_FRAMES] = {0};

        for (int t = 0; t < CHUNK_FRAMES; t++) {
            int fi = c * CHUNK_FRAMES + t;
            if (fi >= nframes) break;
            int start = fi * HOP_SIZE;
            memset(fft_re, 0, sizeof(fft_re));
            memset(fft_im, 0, sizeof(fft_im));
            for (int i = 0; i < FRAME_SIZE; i++)
                fft_re[i] = xpad[start + i] * g_win[i];
            fft(fft_re, fft_im, FFT_SIZE, 0);
            for (int f = 0; f < NFFT_BINS; f++) {
                // 👈 修改：配套 gtcrn.c，改为 t * 257 + f
                in_re[t * 257 + f] = fft_re[f];
                in_im[t * 257 + f] = fft_im[f];
            }
        }

        gtcrn_process_chunk(in_re, in_im, o_re, o_im);

        for (int t = 0; t < CHUNK_FRAMES; t++) {
            int fi = c * CHUNK_FRAMES + t;
            if (fi >= nframes) break;
            int start = fi * HOP_SIZE;
            for (int f = 0; f < NFFT_BINS; f++) {
                // 👈 修改：配套 gtcrn.c，改为 t * 257 + f
                fft_re[f] = o_re[t * 257 + f];
                fft_im[f] = o_im[t * 257 + f];
            }
            for (int f = NFFT_BINS; f < FFT_SIZE; f++) {
                fft_re[f] =  o_re[t * 257 + (FFT_SIZE - f)];
                fft_im[f] = -o_im[t * 257 + (FFT_SIZE - f)];
            }
            fft(fft_re, fft_im, FFT_SIZE, 1);
            for (int i = 0; i < FRAME_SIZE; i++) {
                out [start + i] += fft_re[i] * g_win[i];
                wnrm[start + i] += g_win[i] * g_win[i];
            }
        }
        if ((c + 1) % 10 == 0 || c + 1 == nchunks)
            printf("  已处理 %d/%d chunk\n", c + 1, nchunks);
    }

    float *den = (float *)calloc((size_t)total, sizeof(float));
    if (!den) die("内存不足");
    for (int i = 0; i < total; i++) {
        int p = i + pad;
        den[i] = (wnrm[p] > 1e-3f) ? out[p] / (wnrm[p] + 1e-8f) : 0.0f;
    }

    write_wav(outpath, den, total, sr);
    printf("\n降噪完成 -> %s (前 %.2f 秒)\n", outpath, (double)total / sr);

    free(x); free(xpad); free(out); free(wnrm); free(den);
    return 0;
}