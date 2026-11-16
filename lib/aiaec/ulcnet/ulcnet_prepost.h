/* Align-ULCNet pre/post class, frequency domain only: per-frame functions between the caller's spectra and the
 * accelerator's tensors. Per run (one frame, or a packet of N): pre_frame for every slot, views, the NPU run,
 * validate_run, then post_frame for every slot (zero_state after a failed run or a non-finite frame). The call
 * lifecycle, the staging of what must survive to post (the error spectrum, the events) and the identity frames belong
 * to the stream. The recurrent state is outside the pool: it belongs to the application and is optionally registered
 * in the config.
 *
 * Framing (the application's, with audiolib on the run-time UlcnetGrid; Python reference aiaec_streaming). Trained
 * against: sqrt-Hann (ac_window_build default); analysis AcStft from ac_stft_config_default(&sc, grid.n_fft),
 * hop = grid.hop, spec_scale 1, center 0 (one frame per push; the Python reference is centered: 0, 2, then 1);
 * synthesis AcIstft, same config, ola AC_OLA_ACCUMULATE, ispec_scale 1, every push emits a full hop (hop #0
 * included), latency grid.hop. FftHandle and window table: caller-owned, shareable.
 *
 * Packet boundary (config frames_per_call N >= 2, model layout ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION): the same
 * sequence over N slots, one NPU run for the N frames. A frame marked as an alignment event is reset in the graph and
 * bypassed there; the host does not restart the state for it and fills the frame with the error spectrum. */

#ifndef ULCNET_PREPOST_H
#define ULCNET_PREPOST_H

#include <stddef.h>
#include <stdint.h>

#include "ulcnet_model_io.h"   /* UlcnetGrid */

#ifdef __cplusplus
extern "C" {
#endif

#define ULCNET_PREPOST_DESCRIPTOR_VERSION 1u
/* Folded into build_flags_hash; bump on any change of pool size, carve order or descriptor content. */
#define ULCNET_PREPOST_CARVE_VERSION 10u

typedef struct UlcnetPrepostConfig {
    UlcnetGrid   grid;         /* ulcnet_grid_make(); copied at init; D is a sanity bound only */
    uint32_t model_layout_version; /* 0 = ULCNET_MODEL_IO_LAYOUT_VERSION (PACKET_LAYOUT_VERSION when frames_per_call >= 2);
                                    * else must equal it and the graph metadata */
    int          frames_per_call;  /* 0 or 1: one frame per graph call; 2..ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL: a packet */
    /* Optional registration of the application's recurrent state: borrowed buffers (they must outlive the instance),
     * element counts not bytes; a float host registers all four (zeroed at init and reset, checked whole at every
     * validate_run), a board none. Every pair set or all NULL/0, else refused; not in the pool size or hash. */
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
    uint32_t layout_version;      /* descriptor layout: 18, or 19 for a packet */
    uint32_t reserved0;           /* 0 */
    uint32_t build_flags_hash;    /* FNV-1a-32: sample rate, n_fft, layout, carve, D */
    uint32_t alignment;
    uint32_t reserved1;           /* 0 */
    uint64_t bytes;
} UlcnetPrepostMemReq;

typedef struct UlcnetPrepost UlcnetPrepost;

/* Fill a config for `grid` (copied): nothing registered, the built layout. 0, or -1 on NULL or an invalid grid. */
int ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg, const UlcnetGrid *grid);

/* Exact pool size (instance header, model-I/O block, 16-byte aligned; a function of n_bins, not of D).
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

#define ULCNET_PREPOST_OK            0
#define ULCNET_PREPOST_ERR_ARG      (-1)   /* NULL, slot out of range, or an event at one frame: nothing changed */
#define ULCNET_PREPOST_ERR_NONFINITE (-2)  /* post_frame: a non-finite enhanced bin; the caller restarts once */

/* Full reset: zero the registered recurrent state and advance the state generation. Every pointer, the registration,
 * the config and the pool stay; NULL is ignored. */
void ulcnet_prepost_reset(UlcnetPrepost *p);

/* The restart verb of a run: the same effect as reset (registered state zeroed, generation + 1). */
void ulcnet_prepost_zero_state(UlcnetPrepost *p);

/* Counter of state restarts (registered state or not): advances on reset, zero_state and a refused validate_run();
 * 0 for a fresh instance or NULL. A fixed-point accelerator reloads quantized zeros when it changes. */
uint32_t ulcnet_prepost_state_generation(const UlcnetPrepost *p);

/* Built from the config's grid and layout; the five interior fields are 0. */
const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p);

/* Features of frame `slot` from one analysed frame per plane (n_bins bins; unnormalised rfft of a sqrt-Hann n_fft
 * block, center 0): the error spectrum is AecResidualView.error_spec (never base_spec), the far spectrum the raw far of
 * the same hop. Read here, not kept. `event` != 0 marks an alignment event (packet only). OK or ERR_ARG. */
int ulcnet_prepost_pre_frame(UlcnetPrepost *p, int slot,
                             const float *err_re, const float *err_im,
                             const float *far_re, const float *far_im, int event);

/* Once per run, after the last pre_frame: NaN-fill `output` and publish the boundary. Each state output has the address
 * of its state input (read every input before writing any output); the state pointers are the registered buffers,
 * the others point into the pool, all stable for the life of the instance. 0, or -1 on NULL. */
int ulcnet_prepost_views(UlcnetPrepost *p, UlcnetModelIoInputs *inputs, UlcnetModelIoOutputs *outputs);

/* After a successful run the whole mask and every registered state buffer must be finite. 0, or -1 when refused: the
 * registered state is zeroed and the generation advanced once. */
int ulcnet_prepost_validate_run(UlcnetPrepost *p);

/* Frame `slot` of the validated run: compressed error times the slot's mask, inverse compression, into out_re / out_im
 * (n_bins each). OK; ERR_ARG (nothing written); ERR_NONFINITE (state and generation untouched, the caller restarts). */
int ulcnet_prepost_post_frame(const UlcnetPrepost *p, int slot, float *out_re, float *out_im);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_PREPOST_H */
