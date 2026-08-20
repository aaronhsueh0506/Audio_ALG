#ifndef VAD_API_H
#define VAD_API_H

#include <stddef.h>

#include "fft_wrapper.h"
#include "masker.h"
#include "vad.h"

typedef enum {
    VAD_BACKEND_MASKER = 0
} VADBackendType;

typedef struct {
    VADBackendType backend;

    MaskerConfig masker_cfg;
    VadConfig vad_cfg;
} VADApiConfig;

typedef struct VADApi VADApi;

/* Heap constructor. */
VADApi* vad_api_create(const VADApiConfig* cfg);

/* Caller-pool constructor. One pool backs the VADApi plus its masker and VAD
 * sub-objects, so an integrator sizes and owns exactly one block. `mem` must
 * be 16-byte aligned and `mem_size` at least vad_api_get_mem_size(cfg), else
 * NULL. vad_api_destroy() will not free caller-owned memory. */
size_t  vad_api_get_mem_size(const VADApiConfig* cfg);
VADApi* vad_api_init(const VADApiConfig* cfg, void* mem, size_t mem_size);

void vad_api_process(VADApi* v,
                     const Complex* frame_in);

int vad_api_get(const VADApi* v);
int vad_api_get_raw(const VADApi* v);
const int* vad_api_get_mask(const VADApi* v);

MaskEstimator* vad_api_get_masker(VADApi* v);
MaskVAD* vad_api_get_mask_vad(VADApi* v);

void vad_api_destroy(VADApi* v);

#endif