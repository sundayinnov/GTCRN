#ifndef GTCRN_H
#define GTCRN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

void gtcrn_init(void);
void gtcrn_process_chunk(const float *in_re, const float *in_im, float *out_re, float *out_im);

#ifdef __cplusplus
}
#endif

#endif
