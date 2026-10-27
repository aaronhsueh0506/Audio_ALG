/* Rate bridge: runs the DFN2 stage on its 48 kHz grid inside a pipeline whose
 * own grid is 8 or 16 kHz. AEC, RES, BF and comfort noise stay on the native grid.
 * Per native hop:
 *   E, P (native spectra) -> iFFT + sqrt-Hann WOLA -> e[n], p[n] (native)
 *   -> resample up (x3 at 16 kHz, x6 at 8 kHz) -> 48 kHz FIFOs
 *   every 512 new 48 kHz samples: sqrt-Hann 1024/512 analysis of both
 *     -> dfn_res_stage_process(E48, P48) -> 48 kHz iFFT + WOLA -> resample down
 *   -> native FIFO -> emit exactly one native hop.
 *
 * Clocks: the stage runs a fixed per-hop schedule (16 kHz/hop 128: 0,1,1,1;
 * 16 kHz/hop 256 and 8 kHz/hop 128: 1,2), so a hop carries at most two inferences.
 * The native output FIFO is prefilled with `prefill` zeros (init derives it from the
 * resamplers' tick counts).
 * Delay added to the hosting pipeline's own output: prefill + up group delay +
 * (L + 1)*512*sr/48000 + down group delay native samples, L = DFN_RES_STAGE_LOOKAHEAD_FRAMES;
 * for the shipped 1/1 pair 16 kHz/hop 128 -> 672 (42.0 ms), 16 kHz/hop 256 -> 629 (39.3 ms),
 * 8 kHz/hop 128 -> 330 (41.3 ms), as returned by dfn_rate_bridge_added_delay_samples(), which
 * floors the sum. The stage term is a whole number of native samples only when L + 1 is a
 * multiple of 3; every other term is whole, so the exact delay is the floored one plus 0, 1/3
 * or 2/3 of a sample, from dfn_rate_bridge_added_delay_exact() (e.g. pair 0/1 at 16 kHz/hop
 * 128: 501 1/3). The resamplers are not an identity (pass-band
 * response applied twice).
 * Fail-open: with no model the stage emits P48 delayed L frames.
 * Memory: static pool with the 32-byte descriptor; the stage and resampler pools
 * are carved inside it. No heap after init, no stdio. */

#ifndef DFN_RATE_BRIDGE_H
#define DFN_RATE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "dfn_res_stage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DFN_RATE_BRIDGE_DESCRIPTOR_VERSION 1u
/* Carve layout version; bumped on any change of pool size, carve order or descriptor. */
#define DFN_RATE_BRIDGE_LAYOUT_VERSION     3u
/* Denominator of dfn_rate_bridge_added_delay_exact(): sixths of a native sample. */
#define DFN_RATE_BRIDGE_DELAY_DENOMINATOR  6

/* Grid contract: hop = fft_size / 2 (50 % overlap). Supported native grids are
 * 8 kHz/256, 16 kHz/256 and 16 kHz/512 only. */
typedef struct DfnRateBridgeConfig {
    int sample_rate;            /* 8000 or 16000 (48000 uses the stage directly) */
    int fft_size;               /* 0 = rate default (256); 16 kHz also 512     */
    DfnResStageConfig stage;    /* 48 kHz stage config: matrices, model, limit */
} DfnRateBridgeConfig;

typedef struct {
    uint32_t descriptor_version;  /* DFN_RATE_BRIDGE_DESCRIPTOR_VERSION      */
    uint32_t layout_version;      /* DFN_RATE_BRIDGE_LAYOUT_VERSION          */
    uint32_t backend_id;          /* the stage's backend id                  */
    uint32_t build_flags_hash;    /* FNV-1a-32: the stage's hash, FIFO caps, resampler pool sizes */
    uint32_t alignment;           /* 16                                      */
    uint32_t reserved;            /* 0                                       */
    uint64_t bytes;
} DfnRateBridgeMemReq;

_Static_assert(sizeof(DfnRateBridgeMemReq) == 32,
               "DfnRateBridgeMemReq must be exactly 32 bytes");

typedef struct DfnRateBridge DfnRateBridge;

/* Native grid `sample_rate` with the rate-default fft and the stage defaults
 * (48 kHz, no limit, no matrices, identity model). */
DfnRateBridgeConfig dfn_rate_bridge_default_config(int sample_rate);

/* Returns -1 on NULL, a rate other than 8000/16000, an fft_size the rate does not
 * offer, or any stage rejection (dfn_res_stage.h). */
int dfn_rate_bridge_get_mem_requirements(const DfnRateBridgeConfig *cfg,
                                         DfnRateBridgeMemReq *out);
DfnRateBridge *dfn_rate_bridge_init(void *mem, size_t bytes,
                                    const DfnRateBridgeConfig *cfg);
/* init plus the stale-pool descriptor gate against `expected`. */
DfnRateBridge *dfn_rate_bridge_init_ex(void *mem, size_t bytes,
                                       const DfnRateBridgeConfig *cfg,
                                       const DfnRateBridgeMemReq *expected);
DfnRateBridge *dfn_rate_bridge_create(const DfnRateBridgeConfig *cfg);
void dfn_rate_bridge_destroy(DfnRateBridge *b);   /* frees only create()'s heap */
/* Resets the resamplers, FIFOs, OLAs, the stage and the prefill. */
void dfn_rate_bridge_reset(DfnRateBridge *b);

/* One native hop: the two native spectra in (n_freqs each), one native hop out.
 * Returns 0, or -1 on NULL arguments or a stage contract error (reset required).
 * Phased use, for a caller that runs the model itself (a hop carries 0, 1 or 2
 * stage frames):
 *     dfn_rate_bridge_begin(b, E, P);                  // WOLA + resample up
 *     while (dfn_rate_bridge_nn_begin(b, &job) > 0) {  // 48 kHz frame open
 *         rc = run_model(&job);                        // 0 = every output written
 *         dfn_rate_bridge_nn_end(b, rc);               // commit, or the identity
 *     }
 *     dfn_rate_bridge_finish(b, out_hop);              // emit one native hop
 * nn_begin returns 1 with the open frame's views in *job, 0 when the hop has no
 * further frame, -1 on a contract error. finish takes any frame left open as the
 * identity (fail-open). dfn_rate_bridge_process() runs this loop with the
 * configured model callback. */
int dfn_rate_bridge_begin(DfnRateBridge *b,
                          const Complex *estimate_spec,
                          const Complex *apply_spec);
int dfn_rate_bridge_nn_begin(DfnRateBridge *b, DfnResStageJob *job);
int dfn_rate_bridge_nn_end(DfnRateBridge *b, int model_rc);
int dfn_rate_bridge_finish(DfnRateBridge *b, float *out_hop);
int dfn_rate_bridge_process(DfnRateBridge *b,
                            const Complex *estimate_spec,
                            const Complex *apply_spec,
                            float *out_hop);

int dfn_rate_bridge_hop_size(const DfnRateBridge *b);      /* -1 on NULL */
int dfn_rate_bridge_n_freqs(const DfnRateBridge *b);       /* -1 on NULL */
/* Native samples the bridge adds to the hosting pipeline's own latency. */
int dfn_rate_bridge_added_delay_samples(const DfnRateBridge *b);   /* floor of the exact value */
/* The same delay without the floor: *numerator / *denominator native samples, the denominator
 * always DFN_RATE_BRIDGE_DELAY_DENOMINATOR (not reduced). With the grid's rate `sr` (8000 or
 * 16000):
 *     numerator = (prefill + up latency) * 6
 *               + ((L + 1) * 512 + down latency in 48 kHz frames) * sr / 8000
 * whose division is exact. Returns 0, or -1 on a NULL argument. */
int dfn_rate_bridge_added_delay_exact(const DfnRateBridge *b, int *numerator,
                                      int *denominator);
DfnResStage *dfn_rate_bridge_stage(const DfnRateBridge *b);
/* Frame counters for diagnostics: 48 kHz frames run in total and in the last hop. */
void dfn_rate_bridge_get_counters(const DfnRateBridge *b,
                                  long long *frames_48k, int *last_hop_frames);

#ifdef __cplusplus
}
#endif

#endif /* DFN_RATE_BRIDGE_H */
