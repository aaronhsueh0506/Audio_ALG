#ifndef VAD_H
#define VAD_H

#include <stddef.h>

typedef struct {
    int F;            // number of frequency bins

    int   mode;       // 0:any, 1:ratio, 2:count
    float mask_thr;   // ratio threshold
    int   min_bins;   // count threshold

    int   enable_median;
    int   median_k;

    int   enable_smooth;
    int   hangover;
} VadConfig;
typedef struct {

    VadConfig cfg;

    /* ---------- median state ---------- */
    int med_buf[9];
    int med_pos;

    /* ---------- hangover state ---------- */
    int state;
    int hang_cnt;

    /* ---------- output state ---------- */
    int vad_raw;
    int vad_out;

    /* Non-NULL only on the vad_create() heap path: the block backing this
     * struct, freed by vad_destroy(). NULL on the vad_init() caller-pool
     * path, where the caller owns the memory and vad_destroy() must not free
     * it. Same precedent as GSC's owned_heap. */
    void* owned_heap;

} MaskVAD;


/* Heap constructor. */
MaskVAD* vad_create(const VadConfig* cfg);

/* Caller-pool constructor. vad_get_mem_size() returns the byte requirement;
 * vad_init() places the struct at mem[0]. `mem` must be 16-byte aligned and
 * `mem_size` at least that requirement, else NULL. The caller owns the memory
 * and vad_destroy() will not free it. */
size_t   vad_get_mem_size(void);
MaskVAD* vad_init(void* mem, size_t mem_size, const VadConfig* cfg);

/* per-frame step */
/* Whether vad_init()/vad_create() will accept this config. Exported so a
 * caller that sizes before it constructs -- vad_get_mem_size() takes no
 * config -- can ask the same question the constructor will. */
int vad_config_is_valid(const VadConfig* cfg);

void vad_step(MaskVAD* v,
              const int* mask);

int vad_get_raw(const MaskVAD* v);
int vad_get_out(const MaskVAD* v);

void vad_destroy(MaskVAD* v);

#endif
