/* Rate bridge: runs the DFN2 stage on its 48 kHz grid inside a pipeline whose
 * own grid is 8 or 16 kHz. A thin adapter: the resampling, 48 kHz framing, FIFOs and delay are
 * stream/ainr/bridge's AinrBridge (which embeds no model); this layer puts the configured
 * dfn_res_stage between its "model-rate spectra out" and "enhanced spectrum back". AEC, RES, BF and comfort noise stay on the native grid.
 * Per native hop:
 *   E, P (native spectra) -> iFFT + sqrt-Hann WOLA -> e[n], p[n] (native)
 *   -> resample up (x3 at 16 kHz, x6 at 8 kHz) -> 48 kHz FIFOs
 *   every 512 new 48 kHz samples: sqrt-Hann 1024/512 analysis of both
 *     -> dfn_res_stage_process(E48, P48) -> 48 kHz iFFT + WOLA -> resample down
 *   -> native FIFO -> emit exactly one native hop.
 *
 * Clocks, the added delay (L = dfn_res_stage_lookahead_frames() of the stage's grid; 672 / 629 / 330 native
 * samples for the shipped 1/1 pair) and the post-chain reset timing are those of stream/ainr/bridge/ainr_bridge.h;
 * dfn_rate_bridge_added_delay_samples() floors the sum and dfn_rate_bridge_added_delay_exact() gives the remainder.
 * Fail-open: with no model the stage emits P48 delayed L frames.
 * Memory: static pool with the 32-byte descriptor; the AinrBridge and stage pools are carved
 * inside it. No heap after init, no stdio. */

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
#define DFN_RATE_BRIDGE_LAYOUT_VERSION     7u
/* Denominator of dfn_rate_bridge_added_delay_exact(): sixths of a native sample. */
#define DFN_RATE_BRIDGE_DELAY_DENOMINATOR  6

/* Grid contract: hop = fft_size / 2 (50 % overlap). Supported native grids are
 * 8 kHz/256, 16 kHz/256 and 16 kHz/512 only, and the model (stage.grid) must be the 48 kHz /
 * 1024 grid with one frame per call: the resampling ratios, the per-hop inference schedule, the
 * prefill calibration and the FIFO capacities are written for it, so a bridge over another model
 * grid is refused (-1). Its 48 kHz AcStft / AcIstft run at audiolib's default scales: the stage
 * applies dfn2_stft_scale(n_fft) / dfn2_stft_unscale(n_fft). */
typedef struct DfnRateBridgeConfig {
    int sample_rate;            /* 8000 or 16000 (48000 uses the stage directly) */
    int fft_size;               /* 0 = rate default (256); 16 kHz also 512     */
    DfnResStageConfig stage;    /* the stage config: model grid (48 kHz / 1024), matrices, model, limit */
} DfnRateBridgeConfig;

typedef struct {
    uint32_t descriptor_version;  /* DFN_RATE_BRIDGE_DESCRIPTOR_VERSION      */
    uint32_t layout_version;      /* DFN_RATE_BRIDGE_LAYOUT_VERSION          */
    uint32_t backend_id;          /* the stage's backend id                  */
    uint32_t build_flags_hash;    /* FNV-1a-32: this layout, the stage's hash and the AinrBridge's */
    uint32_t alignment;           /* 16                                      */
    uint32_t reserved;            /* 0                                       */
    uint64_t bytes;
} DfnRateBridgeMemReq;

typedef struct DfnRateBridge DfnRateBridge;

/* Native grid `sample_rate` with the rate-default fft and the stage defaults for the model grid
 * `model_grid` (copied; no limit, no matrices, identity model). */
DfnRateBridgeConfig dfn_rate_bridge_default_config(int sample_rate, const DFN2Grid *model_grid);

/* Returns -1 on NULL, a rate other than 8000/16000, an fft_size the rate does not offer, a model
 * grid (stage.grid) that is not 48 kHz / 1024 with one frame per call, or any stage rejection
 * (dfn_res_stage.h). */
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
/* Synthesis-only reset (ainr_bridge_reset_synthesis): the stage, resamplers, FIFOs and counters keep running. Call it
 * between hops, before begin() / process() of the hop on which aec_post_reset_serial() moved, and reseed the
 * comfort noise (AEC_CNG_SEED) next to it; the AEC is not reset. 0, or -1 on NULL or while a hop is in progress. */
int dfn_rate_bridge_reset_synthesis(DfnRateBridge *b);

/* One native hop: the two native spectra in (n_freqs each), one native hop out.
 * Returns 0, or -1 on NULL arguments or a stage contract error (reset required).
 * Phased use, for a caller that runs the model itself (a hop carries 0, 1 or 2
 * stage frames; the first mask_lookahead stage frames after a start need no model frame and
 * are closed inside nn_begin):
 *     dfn_rate_bridge_begin(b, E, P);                  // WOLA + resample up
 *     while (dfn_rate_bridge_nn_begin(b, &job) > 0) {  // 48 kHz frame open
 *         rc = run_model(&job);                        // run status (stream_run.h): 0 = every output written
 *         dfn_rate_bridge_nn_end(b, rc);               // commit, or the identity
 *     }
 *     dfn_rate_bridge_finish(b, out_hop);              // emit one native hop
 * nn_begin returns 1 with the open frame's views in *job, 0 when the hop has no
 * further frame, -1 on a contract error. finish takes any frame left open (STREAM_RUN_FAILED) or never
 * fetched (STREAM_RUN_NOT_STARTED) as the identity (fail-open). dfn_rate_bridge_process() runs this loop with the
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
/* The delay without the floor, *numerator / *denominator native samples (ainr_bridge_added_delay_exact()'s
 * formula; denominator always DFN_RATE_BRIDGE_DELAY_DENOMINATOR). 0, or -1 on NULL. */
int dfn_rate_bridge_added_delay_exact(const DfnRateBridge *b, int *numerator,
                                      int *denominator);
DfnResStage *dfn_rate_bridge_stage(const DfnRateBridge *b);
/* The stage's state generation (dfn_res_stage_state_generation(): resets, refused
 * commits and dirty run statuses advance it); 0 for NULL. */
unsigned int dfn_rate_bridge_state_generation(const DfnRateBridge *b);
/* Frame counters for diagnostics: 48 kHz frames run in total and in the last hop. */
void dfn_rate_bridge_get_counters(const DfnRateBridge *b,
                                  long long *frames_48k, int *last_hop_frames);
/* Diagnostics: the native-grid hops the host synthesis produced from the spectra of the last
 * dfn_rate_bridge_begin() / process() (hop_size floats each, before the resamplers). Valid until
 * the next begin() or reset; NULL for an output pointer you do not need. Returns 0, or -1 on NULL. */
int dfn_rate_bridge_get_host_hops(const DfnRateBridge *b, const float **estimate_hop,
                                  const float **apply_hop);

#ifdef __cplusplus
}
#endif

#endif /* DFN_RATE_BRIDGE_H */
