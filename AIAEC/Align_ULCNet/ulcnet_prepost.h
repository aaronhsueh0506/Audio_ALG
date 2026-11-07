/* Align-ULCNet pre/post class, frequency domain only (framing: ulcnet_process.h): the caller's spectra to the
 * accelerator's tensors and back. Per hop: pre_process_freq, frame_open, the NPU run, frame_finish (or frame_bypass),
 * post_process_freq. The recurrent state is outside the pool: it belongs to the application and is optionally
 * registered in the config. */

#ifndef ULCNET_PREPOST_H
#define ULCNET_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_process.h"   /* ulcnet_model_io.h: UlcnetGrid */

#ifdef __cplusplus
extern "C" {
#endif

#define ULCNET_PREPOST_DESCRIPTOR_VERSION 1u
/* Folded into build_flags_hash; bump on any change of pool size, carve order or descriptor content. */
#define ULCNET_PREPOST_CARVE_VERSION 8u

typedef struct UlcnetPrepostConfig {
    UlcnetGrid   grid;         /* ulcnet_grid_make(); copied at init; D is a sanity bound only */
    uint32_t model_layout_version; /* 0 = ULCNET_MODEL_IO_LAYOUT_VERSION; else must equal it and the graph metadata */
    /* Optional registration of the application's recurrent state: borrowed buffers (they must outlive the instance),
     * element counts not bytes; a float host registers all four (zeroed at init and reset, checked whole at every
     * frame_finish), a board none. Every pair set or all NULL/0, else refused; not in the pool size or hash. */
    float       *key_history;
    size_t       key_history_elements;
    float       *value_history;
    size_t       value_history_elements;
    float       *logit_history;
    size_t       logit_history_elements;
    float       *h_gru;
    size_t       gru_hidden_elements;
} UlcnetPrepostConfig;

typedef struct UlcnetPrepostMemReq {
    uint32_t descriptor_version;  /* ULCNET_PREPOST_DESCRIPTOR_VERSION      */
    uint32_t layout_version;      /* ULCNET_MODEL_IO_LAYOUT_VERSION         */
    uint32_t reserved0;           /* 0 */
    uint32_t build_flags_hash;    /* FNV-1a-32: sample rate, n_fft, layout, carve, D */
    uint32_t alignment;
    uint32_t reserved1;           /* 0 */
    uint64_t bytes;
} UlcnetPrepostMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(UlcnetPrepostMemReq) == 32,
               "UlcnetPrepostMemReq must stay 32 bytes");
#endif
#endif

typedef struct UlcnetPrepost UlcnetPrepost;

/* Fill a config for `grid` (copied): nothing registered, the built layout. 0, or -1 on NULL or an invalid grid. */
int ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg, const UlcnetGrid *grid);

/* Exact pool size (instance header, model-I/O block, frame staging, 16-byte aligned; a function of n_bins, not of D).
 * -1 with *req untouched on NULL, an invalid grid, another layout version or an inconsistent state registration. */
int ulcnet_prepost_get_mem_size(const UlcnetPrepostConfig *cfg,
                                UlcnetPrepostMemReq *req);

/* Construct inside caller memory; starts reset (the registered buffers are zeroed). NULL on a bad config or a
 * misaligned or undersized pool; a refused call touches neither the pool nor the registered buffers. */
UlcnetPrepost *ulcnet_prepost_init(void *pool, size_t bytes,
                                   const UlcnetPrepostConfig *cfg);

/* As _init, plus the stale-pool gate: a pool recorded for another D, layout or build is refused (NULL = _init). */
UlcnetPrepost *ulcnet_prepost_init_ex(void *pool, size_t bytes,
                                      const UlcnetPrepostConfig *cfg,
                                      const UlcnetPrepostMemReq *expected);

UlcnetPrepost *ulcnet_prepost_create(const UlcnetPrepostConfig *cfg);   /* host path: allocates the pool */

/* Frees only what _create allocated (never the registered buffers); NULL-safe, a no-op on an _init instance. */
void ulcnet_prepost_destroy(UlcnetPrepost *p);

/* Zero the registered recurrent state and the output staging, drop any open frame, advance the state generation.
 * Every pointer, the registration, the config and the pool stay; NULL is ignored. */
void ulcnet_prepost_reset(UlcnetPrepost *p);

/* Counter of state restarts: advances on reset and when frame_finish() refused a frame, registered or not; 0 for a
 * fresh instance or NULL. A fixed-point accelerator loads quantized zeros at start and on every change. */
uint32_t ulcnet_prepost_state_generation(const UlcnetPrepost *p);

int ulcnet_prepost_hop_size(const UlcnetPrepost *p);   /* or -1 */
int ulcnet_prepost_num_bins(const UlcnetPrepost *p);   /* or -1 */
/* Built from the config's grid and layout; the five interior fields are 0. */
const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p);

/* One analysed frame per stream (n_bins bins; unnormalised rfft of a sqrt-Hann n_fft block on the hop grid, center
 * 0). The error spectrum is AecResContext.error_spec, the analysis seam the model is trained on, never
 * AecOutputBase.base_spec of aec_get_output_base() (on a guard hop it does not pair with the formed hop); the far
 * spectrum is the raw far of the same hop. Returns 1 (one invocation); -1 on NULL args or while a frame is open. */
int ulcnet_prepost_pre_process_freq(UlcnetPrepost *p,
                                    const float *err_re, const float *err_im,
                                    const float *far_re, const float *far_im);

/* Publish the current frame's accelerator boundary. `output` is NaN-prefilled so an unwritten frame is caught by
 * frame_finish. Each state output has the address of its state input: read every state input before writing any state
 * output. The state pointers are the registered buffers (NULL and 0 elements when none), the others point into the
 * pool; all are stable for the life of the instance. Returns 0, or -1 if no frame is open. */
int ulcnet_prepost_frame_open(UlcnetPrepost *p,
                                UlcnetModelIoInputs *inputs,
                                UlcnetModelIoOutputs *outputs);

/* Call after a successful accelerator run (frame_bypass() after a failed one); needs an open frame from frame_open(),
 * else refused. Validates a finite mask and, when state is registered, finite state (every registered buffer scanned
 * whole), applies the mask and inverse compression, stages the enhanced spectrum. On failure (-1) the registered
 * state is zeroed, the generation advances and the frame stays open: then frame_bypass() or a fresh frame_open(). */
int ulcnet_prepost_frame_finish(UlcnetPrepost *p);

/* Identity for the current frame (after a failed run or a reprime): the error spectrum passes through without the AEC
 * output guard, so on a guard hop it can exceed the capture. State is left as is. Returns 0, or -1. */
int ulcnet_prepost_frame_bypass(UlcnetPrepost *p);

/* The enhanced spectrum (n_bins bins) of the frame just finished or bypassed. Returns 0, or -1. */
int ulcnet_prepost_post_process_freq(UlcnetPrepost *p, float *re, float *im);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_PREPOST_H */
