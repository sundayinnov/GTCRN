#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gtcrn.h"
#include "gtcrn_weights_compat.h"
#include "prelu_slopes.h"

#ifdef DEBUG_LAYER
static void _stat(const char *tag, const float *p, int n) {
    float mn=1e9f, mx=-1e9f, s=0;
    for (int i=0;i<n;i++){float v=p[i];if(v<mn)mn=v;if(v>mx)mx=v;s+=v;}
    fprintf(stderr, "[C] %-15s min=%.4f max=%.4f mean=%.4f\n", tag, mn, mx, s/n);
}
static int g_dbg = 0;   /* 只打印第一个 chunk */
#endif

static float h_dp1_1[33][8] = {0}, h_dp1_2[33][8] = {0};
static float h_dp2_1[33][8] = {0}, h_dp2_2[33][8] = {0};

static inline float ps(float x, float s) { return x > 0 ? x : x * s; }
static inline float sig(float x) { return 1.0f / (1.0f + expf(-x)); }

static inline void gru_step(const float *x, int in_dim, float *h, int H,
    const float *wih, const float *whh, const float *b) {
    float z[16], r[16], n[16];
    for (int i = 0; i < H; i++) {
        float sz = b[i] + b[3*H+i];
        float sr = b[H+i] + b[3*H+H+i];
        float sn = b[2*H+i];
        for (int j = 0; j < in_dim; j++) sz += x[j] * wih[i*in_dim+j];
        for (int j = 0; j < in_dim; j++) sr += x[j] * wih[(H+i)*in_dim+j];
        for (int j = 0; j < in_dim; j++) sn += x[j] * wih[(2*H+i)*in_dim+j];
        for (int j = 0; j < H; j++) sz += h[j] * whh[i*H+j];
        for (int j = 0; j < H; j++) sr += h[j] * whh[(H+i)*H+j];
        z[i] = sig(sz); r[i] = sig(sr);
        for (int j = 0; j < H; j++) sn += r[i] * h[j] * whh[(2*H+i)*H+j];
        sn += r[i] * b[3*H+2*H+i];
        n[i] = tanhf(sn);
    }
    for (int i = 0; i < H; i++) h[i] = (1-z[i])*n[i] + z[i]*h[i];
}

static void tra_fwd(float in[8][10][33], float attn[8][10],
    const float *wih, const float *whh, const float *b,
    const float *fcw, const float *fcb) {
    float erb[10][8];
    for (int t = 0; t < 10; t++) for (int c = 0; c < 8; c++) {
        float s = 0;
        for (int f = 0; f < 33; f++) { float v = in[c][t][f]; s += v*v; }
        erb[t][c] = s / 33.0f;
    }
    float h[16] = {0}, g[10][16];
    for (int t = 0; t < 10; t++) {
        gru_step(erb[t], 8, h, 16, wih, whh, b);
        for (int i = 0; i < 16; i++) g[t][i] = h[i];
    }
    for (int t = 0; t < 10; t++) for (int c = 0; c < 8; c++) {
        float s = fcb[c];
        // 核心修复：Linear 权重是 [in][out]，即 16x8，索引应该是 i*8 + c
        for (int i = 0; i < 16; i++) s += g[t][i] * fcw[i*8 + c]; 
        attn[c][t] = sig(s);
    }
}

static void gtblock(float in[16][10][33], float out[16][10][33],
    int dil, int is_decoder, int block_id,
    float p1, float p2,
    const float *w1, const float *b1,
    const float *wd, const float *bd,
    const float *w2, const float *b2,
    const float *wih, const float *whh, const float *b,
    const float *fcw, const float *fcb) {
    float main_[8][10][33], skip_[8][10][33];
    for (int t = 0; t < 10; t++) for (int f = 0; f < 33; f++) for (int c = 0; c < 8; c++) {
        main_[c][t][f] = in[c][t][f];          /* 前 8 = main */
        skip_[c][t][f] = in[c+8][t][f];        /* 后 8 = skip */
    }
    float c1[16][10][33], c2[16][10][33], c3[8][10][33];

    /* ===== ① point_conv1 =====
     * 强制加入 SFE 的 24 通道展开 (k = -1, 0, 1)
     */
    for (int t = 0; t < 10; t++) for (int of = 0; of < 33; of++) for (int oc = 0; oc < 16; oc++) {
        float s = b1[oc];
        for (int k = 0; k < 3; k++) {              
            int ff = of + k - 1;
            if (ff < 0 || ff >= 33) continue;
            for (int ic = 0; ic < 8; ic++) {
                int ic_global = ic * 3 + k;
                int widx = is_decoder ? (ic_global * 16 + oc)   /* ConvTranspose2d */
                                      : (oc * 24 + ic_global);  /* Conv2d */
                s += main_[ic][t][ff] * w1[widx];
            }
        }
        c1[oc][t][of] = ps(s, p1);
    }

    /* ===== ② depth_conv ===== 
     * 保持原样，无需反转频率，因果时间正确
     */
    for (int t = 0; t < 10; t++) for (int of = 0; of < 33; of++) for (int oc = 0; oc < 16; oc++) {
        float s = bd[oc];
        for (int ti = 0; ti < 3; ti++) {
            int tt = is_decoder ? (t - dil*ti) : (t - dil*(2-ti));
            if (tt < 0 || tt >= 10) continue;
            for (int fi = -1; fi <= 1; fi++) {
                int ff = of + fi;
                if (ff >= 0 && ff < 33) {
                    s += c1[oc][tt][ff] * wd[oc*9 + ti*3 + (fi+1)];
                }
            }
        }
        c2[oc][t][of] = ps(s, p2);
    }

    /* ===== ③ point_conv2 ===== */
    for (int t = 0; t < 10; t++) for (int of = 0; of < 33; of++) for (int oc = 0; oc < 8; oc++) {
        float s = b2[oc];
        for (int ic = 0; ic < 16; ic++) {
            int widx = is_decoder ? (ic * 8 + oc)   /* ConvTranspose2d */
                                  : (oc * 16 + ic); /* Conv2d */
            s += c2[ic][t][of] * w2[widx];
        }
        c3[oc][t][of] = s;
    }

    float attn[8][10];
    tra_fwd(c3, attn, wih, whh, b, fcw, fcb);
    for (int t = 0; t < 10; t++) for (int f = 0; f < 33; f++) for (int c = 0; c < 8; c++) {
        out[c*2+0][t][f] = c3[c][t][f] * attn[c][t];   /* 偶数位 = main 结果 */
        out[c*2+1][t][f] = skip_[c][t][f];             /* 奇数位 = skip 原样 */
    }
}

static void dpgrnn(float in[16][10][33], float out[16][10][33],
    float h1[33][8], float h2[33][8],
    const float *ih1, const float *hh1, const float *bb1,
    const float *ih2, const float *hh2, const float *bb2,
    const float *fcI, const float *bfcI,
    const float *lnIw, const float *lnIb,
    const float *ih3, const float *hh3, const float *bb3,
    const float *ih4, const float *hh4, const float *bb4,
    const float *fcT, const float *bfcT,
    const float *lnTw, const float *lnTb) {
    float intra[16][10][33];
    for (int t = 0; t < 10; t++) {
        float y[33][16];
        for (int half = 0; half < 2; half++) {
            const float *wih = half==0 ? ih1 : ih2;
            const float *whh = half==0 ? hh1 : hh2;
            const float *bias = half==0 ? bb1 : bb2;
            int off = half * 8;
            float hf[4] = {0}, yf[33][4];
            for (int f = 0; f < 33; f++) {
                float x[8]; for (int j = 0; j < 8; j++) x[j] = in[off+j][t][f];
                gru_step(x, 8, hf, 4, wih, whh, bias);
                for (int i = 0; i < 4; i++) yf[f][i] = hf[i];
            }
            float hb[4] = {0}, yb[33][4];
            for (int f = 32; f >= 0; f--) {
                float x[8]; for (int j = 0; j < 8; j++) x[j] = in[off+j][t][f];
                const float *wih_bwd = half==0 ? ih1+96 : ih2+96;
                const float *whh_bwd = half==0 ? hh1+48 : hh2+48;
                const float *bias_bwd = half==0 ? bb1+24 : bb2+24;
                gru_step(x, 8, hb, 4, wih_bwd, whh_bwd, bias_bwd);
                for (int i = 0; i < 4; i++) yb[f][i] = hb[i];
            }
            for (int f = 0; f < 33; f++)
                for (int i = 0; i < 4; i++) { y[f][off+i] = yf[f][i]; y[f][off+4+i] = yb[f][i]; }
        }
        float tmp[33][16], mean = 0, var = 0;
        for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++) {
            float s = bfcI[c];
            // 核心修复：fcI 也是 [in][out] 布局，索引换成 i*16 + c
            for (int i = 0; i < 16; i++) s += y[f][i] * fcI[i*16 + c];
            tmp[f][c] = s; mean += s;
        }
        mean /= 33*16;
        for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++) { float d = tmp[f][c]-mean; var += d*d; }
        var /= 33*16;
        float inv = 1.0f / sqrtf(var + 1e-8f);
        for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++)
            intra[c][t][f] = in[c][t][f] + (tmp[f][c]-mean)*inv*lnIw[f*16+c] + lnIb[f*16+c];
    }
    float y_all[10][33][16];
    for (int f = 0; f < 33; f++) {
        float y[10][16];
        const float *wih = ih3; const float *whh = hh3; const float *bias = bb3;
        float h[8];
        for (int i = 0; i < 8; i++) h[i] = h1[f][i];
        for (int t = 0; t < 10; t++) {
            float x[8]; for (int j = 0; j < 8; j++) x[j] = intra[j][t][f];
            gru_step(x, 8, h, 8, wih, whh, bias);
            for (int i = 0; i < 8; i++) y[t][i] = h[i];
        }
        for (int i = 0; i < 8; i++) h1[f][i] = h[i];
        wih = ih4; whh = hh4; bias = bb4;
        for (int i = 0; i < 8; i++) h[i] = h2[f][i];
        for (int t = 0; t < 10; t++) {
            float x[8]; for (int j = 0; j < 8; j++) x[j] = intra[8+j][t][f];
            gru_step(x, 8, h, 8, wih, whh, bias);
            for (int i = 0; i < 8; i++) y[t][8+i] = h[i];
        }
        for (int i = 0; i < 8; i++) h2[f][i] = h[i];
        for (int t = 0; t < 10; t++) for (int c = 0; c < 16; c++) {
            float s = bfcT[c];
            // 核心修复：fcT 也是 [in][out] 布局
            for (int i = 0; i < 16; i++) s += y[t][i] * fcT[i*16 + c];
            y_all[t][f][c] = s;
        }
    }
    for (int t = 0; t < 10; t++) {
        float mean = 0, var = 0;
        for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++) mean += y_all[t][f][c];
        mean /= 33*16;
        for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++) { float d = y_all[t][f][c]-mean; var += d*d; }
        var /= 33*16;
        float inv = 1.0f / sqrtf(var + 1e-8f);
        for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++)
            out[c][t][f] = intra[c][t][f] + (y_all[t][f][c]-mean)*inv*lnTw[f*16+c] + lnTb[f*16+c];
    }
}

void gtcrn_init(void) {
    memset(h_dp1_1, 0, sizeof(h_dp1_1));
    memset(h_dp1_2, 0, sizeof(h_dp1_2));
    memset(h_dp2_1, 0, sizeof(h_dp2_1));
    memset(h_dp2_2, 0, sizeof(h_dp2_2));
#ifdef DEBUG_LAYER
    g_dbg = 1;
#endif
}

void gtcrn_process_chunk(const float *in_re, const float *in_im, float *out_re, float *out_im) {
#ifdef RESET_STATE_PER_CHUNK
    memset(h_dp1_1, 0, sizeof(h_dp1_1));
    memset(h_dp1_2, 0, sizeof(h_dp1_2));
    memset(h_dp2_1, 0, sizeof(h_dp2_1));
    memset(h_dp2_2, 0, sizeof(h_dp2_2));
#endif
#ifdef DEBUG_LAYER
    static int _chunk_cnt = 0;
    g_dbg = (_chunk_cnt++ == 0);   /* 只打印第一个 chunk */
#endif

    float raw[3][10][257];
    for (int t = 0; t < 10; t++) for (int f = 0; f < 257; f++) {
        int i = t * 257 + f;
        float re = in_re[i], im = in_im[i];
        raw[0][t][f] = sqrtf(re*re + im*im + 1e-12f);
        raw[1][t][f] = re;
        raw[2][t][f] = im;
    }
    float feat[3][10][129];
    for (int t = 0; t < 10; t++) for (int c = 0; c < 3; c++) {
        for (int f = 0; f < 65; f++) feat[c][t][f] = raw[c][t][f];
        for (int k = 0; k < 64; k++) {
            float s = 0;
            for (int m = 0; m < 192; m++) s += raw[c][t][65+m] * val_1511[k*192 + m];
            feat[c][t][65+k] = s;
        }
    }
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("feat", (float*)feat, 3*10*129);
#endif

    float conv1[16][10][65];
#ifdef DEBUG_LAYER
    float _c0mn=1e9f,_c0mx=-1e9f,_c0s=0;
#endif
    for (int t = 0; t < 10; t++) for (int of = 0; of < 65; of++) for (int oc = 0; oc < 16; oc++) {
        float s = encoder_conv0_b[oc];
        for (int j = 0; j < 5; j++) {
            int inf = of*2+j-2; if (inf < 0 || inf >= 129) continue;
            for (int ch = 0; ch < 9; ch++) {
                int ic = ch/3, lf = ch%3, af = inf+lf-1;
                float x = (af >= 0 && af < 129) ? feat[ic][t][af] : 0.0f;
                s += x * encoder_conv0_w[oc*45+ch*5+j];
            }
        }
#ifdef DEBUG_LAYER
        if(s<_c0mn)_c0mn=s; if(s>_c0mx)_c0mx=s; _c0s+=s;
#endif
        conv1[oc][t][of] = ps(s, prelu_slopes[0]);
    }
#ifdef DEBUG_LAYER
    if (g_dbg) fprintf(stderr, "[C] %-15s min=%.4f max=%.4f mean=%.4f\n",
        "en_conv0_pre", _c0mn, _c0mx, _c0s/10400);
#endif

    float conv2[16][10][33];
#ifdef DEBUG_LAYER
    float _c1mn=1e9f,_c1mx=-1e9f,_c1s=0;
#endif
    for (int t = 0; t < 10; t++) for (int of = 0; of < 33; of++) for (int g = 0; g < 2; g++) for (int og = 0; og < 8; og++) {
        int oc = g*8+og;
        float s = encoder_conv1_b[oc];
        for (int j = 0; j < 5; j++) {
            int fi = of*2+j-2; if (fi < 0 || fi >= 65) continue;
            for (int ig = 0; ig < 8; ig++) s += conv1[g*8+ig][t][fi] * encoder_conv1_w[oc*40 + ig*5 + j];
        }
#ifdef DEBUG_LAYER
        if(s<_c1mn)_c1mn=s; if(s>_c1mx)_c1mx=s; _c1s+=s;
#endif
        conv2[oc][t][of] = ps(s, prelu_slopes[1]);
    }
#ifdef DEBUG_LAYER
    if (g_dbg) fprintf(stderr, "[C] %-15s min=%.4f max=%.4f mean=%.4f\n",
        "en_conv1_pre", _c1mn, _c1mx, _c1s/5280);
#endif

    float eo2[16][10][33], eo3[16][10][33], eo4[16][10][33];
    gtblock(conv2, eo2, 1, 0, 2, prelu_slopes[2], prelu_slopes[3],
        eb2_pc1_w, eb2_pc1_b, eb2_dc_w, eb2_dc_b, eb2_pc2_w, eb2_pc2_b,
        eb2_attn_wih, eb2_attn_whh, eb2_attn_b, eb2_attn_fc_w, eb2_attn_fc_b);
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("eo2", (float*)eo2, 16*10*33);
#endif
    gtblock(eo2, eo3, 2, 0, 3, prelu_slopes[4], prelu_slopes[5],
        eb3_pc1_w, eb3_pc1_b, eb3_dc_w, eb3_dc_b, eb3_pc2_w, eb3_pc2_b,
        eb3_attn_wih, eb3_attn_whh, eb3_attn_b, eb3_attn_fc_w, eb3_attn_fc_b);
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("eo3", (float*)eo3, 16*10*33);
#endif
    gtblock(eo3, eo4, 5, 0, 4, prelu_slopes[6], prelu_slopes[7],
        eb4_pc1_w, eb4_pc1_b, eb4_dc_w, eb4_dc_b, eb4_pc2_w, eb4_pc2_b,
        eb4_attn_wih, eb4_attn_whh, eb4_attn_b, eb4_attn_fc_w, eb4_attn_fc_b);
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("eo4", (float*)eo4, 16*10*33);
#endif

    float dp1[16][10][33], dp2[16][10][33];
    dpgrnn(eo4, dp1, h_dp1_1, h_dp1_2,
        val_569, val_570, val_571, val_651, val_652, val_653,
        val_666, val_679, val_685, val_686,
        val_704, val_705, val_721, val_751, val_752, val_768,
        val_781, val_794, val_800, val_801);
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("dp1", (float*)dp1, 16*10*33);
#endif
    dpgrnn(dp1, dp2, h_dp2_1, h_dp2_2,
        val_869, val_870, val_871, val_951, val_952, val_953,
        val_966, val_979, val_985, val_986,
        val_1004, val_1005, val_1021, val_1051, val_1052, val_1068,
        val_1081, val_1094, val_1100, val_1101);
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("dp2", (float*)dp2, 16*10*33);
#endif

    float di[16][10][33], d0[16][10][33], d1[16][10][33], d2[16][10][33];
    for (int t = 0; t < 10; t++) for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++) di[c][t][f] = dp2[c][t][f] + eo4[c][t][f];
    gtblock(di, d0, 5, 1, 0, prelu_slopes[8], prelu_slopes[9],
        db0_pc1_w, db0_pc1_b, db0_dc_w, db0_dc_b, db0_pc2_w, db0_pc2_b,
        db0_attn_wih, db0_attn_whh, db0_attn_b, db0_attn_fc_w, db0_attn_fc_b);
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("d0", (float*)d0, 16*10*33);
#endif
    for (int t = 0; t < 10; t++) for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++) di[c][t][f] = d0[c][t][f] + eo3[c][t][f];
    gtblock(di, d1, 2, 1, 1, prelu_slopes[10], prelu_slopes[11],
        db1_pc1_w, db1_pc1_b, db1_dc_w, db1_dc_b, db1_pc2_w, db1_pc2_b,
        db1_attn_wih, db1_attn_whh, db1_attn_b, db1_attn_fc_w, db1_attn_fc_b);
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("d1", (float*)d1, 16*10*33);
#endif
    for (int t = 0; t < 10; t++) for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++) di[c][t][f] = d1[c][t][f] + eo2[c][t][f];
    gtblock(di, d2, 1, 1, 2, prelu_slopes[12], prelu_slopes[13],
        db2_pc1_w, db2_pc1_b, db2_dc_w, db2_dc_b, db2_pc2_w, db2_pc2_b,
        db2_attn_wih, db2_attn_whh, db2_attn_b, db2_attn_fc_w, db2_attn_fc_b);
#ifdef DEBUG_LAYER
    if (g_dbg) _stat("d2", (float*)d2, 16*10*33);
#endif

    float ct3_in[16][10][33];
    for (int t = 0; t < 10; t++) for (int f = 0; f < 33; f++) for (int c = 0; c < 16; c++) ct3_in[c][t][f] = d2[c][t][f] + conv2[c][t][f];
    float ct3[16][10][65];
#ifdef DEBUG_LAYER
    float _c3mn=1e9f,_c3mx=-1e9f,_c3s=0;
#endif
    for (int t = 0; t < 10; t++) for (int of = 0; of < 65; of++) for (int oc = 0; oc < 16; oc++) {
        int g = oc/8, og = oc%8;
        float s = decoder_conv3_b[oc];
        for (int j = 0; j < 5; j++) {
            int tmp = of+2-j;
            if (tmp >= 0 && tmp%2 == 0) {
                int inf = tmp/2;
                if (inf >= 0 && inf < 33) for (int ig = 0; ig < 8; ig++)
                    s += ct3_in[g*8+ig][t][inf] * decoder_conv3_w[(g*8+ig)*40 + og*5 + j];
            }
        }
#ifdef DEBUG_LAYER
        if(s<_c3mn)_c3mn=s; if(s>_c3mx)_c3mx=s; _c3s+=s;
#endif
        ct3[oc][t][of] = ps(s, prelu_slopes[14]);
    }
#ifdef DEBUG_LAYER
    if (g_dbg) fprintf(stderr, "[C] %-15s min=%.4f max=%.4f mean=%.4f\n",
        "dec3_pre", _c3mn, _c3mx, _c3s/10400);
#endif

    float ct4_in[16][10][65];
    for (int t = 0; t < 10; t++) for (int f = 0; f < 65; f++) for (int c = 0; c < 16; c++) ct4_in[c][t][f] = ct3[c][t][f] + conv1[c][t][f];

    float mre[10][129], mim[10][129];
#ifdef DEBUG_LAYER
    float _d4rmn=1e9f,_d4rmx=-1e9f,_d4rs=0;
    float _d4imn=1e9f,_d4imx=-1e9f,_d4is=0;
#endif
    for (int t = 0; t < 10; t++) for (int of = 0; of < 129; of++) {
        float sr = decoder_conv4_b[0], si = decoder_conv4_b[1];
        for (int j = 0; j < 5; j++) {
            int tmp = of+2-j;
            if (tmp >= 0 && tmp%2 == 0) {
                int inf = tmp/2;
                if (inf >= 0 && inf < 65) for (int ic = 0; ic < 16; ic++) {
                    float x = ct4_in[ic][t][inf];
                    sr += x * decoder_conv4_w[ic*10 + 0*5 + j];
                    si += x * decoder_conv4_w[ic*10 + 1*5 + j];
                }
            }
        }
#ifdef DEBUG_LAYER
        if(sr<_d4rmn)_d4rmn=sr; if(sr>_d4rmx)_d4rmx=sr; _d4rs+=sr;
        if(si<_d4imn)_d4imn=si; if(si>_d4imx)_d4imx=si; _d4is+=si;
#endif
        mre[t][of] = tanhf(sr); mim[t][of] = tanhf(si);
    }
#ifdef DEBUG_LAYER
    if (g_dbg) {
        fprintf(stderr, "[C] %-15s min=%.4f max=%.4f mean=%.4f\n",
            "dec4_re_pre", _d4rmn, _d4rmx, _d4rs/1290);
        fprintf(stderr, "[C] %-15s min=%.4f max=%.4f mean=%.4f\n",
            "dec4_im_pre", _d4imn, _d4imx, _d4is/1290);
    }
#endif

    float mre_full[10][257], mim_full[10][257];
    for (int t = 0; t < 10; t++) {
        for (int f = 0; f < 65; f++) { mre_full[t][f] = mre[t][f]; mim_full[t][f] = mim[t][f]; }
        for (int k = 0; k < 192; k++) {
            float sr = 0, si = 0;
            for (int m = 0; m < 64; m++) {
                sr += mre[t][65 + m] * val_38[k*64 + m];
                si += mim[t][65 + m] * val_38[k*64 + m];
            }
            mre_full[t][65+k] = sr;
            mim_full[t][65+k] = si;
        }
    }

#ifdef DEBUG_MASK
    {
        static int ncall = 0;
        if (ncall++ < 3) {
            float mn=1e9f, mx=-1e9f, s=0;
            for (int t=0;t<10;t++) for (int f=0;f<257;f++) {
                float v = sqrtf(mre_full[t][f]*mre_full[t][f] + mim_full[t][f]*mim_full[t][f]);
                if (v<mn)mn=v; if(v>mx)mx=v; s+=v;
            }
            fprintf(stderr, "[C] chunk%d mask |M|: min=%.3f max=%.3f mean=%.3f\n",
                    ncall-1, mn, mx, s/(10*257));
        }
    }
#endif

    for (int t = 0; t < 10; t++) for (int f = 0; f < 257; f++) {
        int i = t * 257 + f;
        float mr = mre_full[t][f], mi = mim_full[t][f];
        float re = in_re[i], im = in_im[i];
        out_re[i] = re*mr - im*mi;
        out_im[i] = re*mi + im*mr;
    }
}