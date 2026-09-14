/* ============================================================================
 * Rate bridge: run the DFN2 stage on its 48 kHz grid inside a pipeline whose
 * own grid is 8 or 16 kHz.
 *
 * Everything conventional (AEC, RES, BF/GSC, comfort noise) stays on the
 * product's native grid. Only this bridge crosses to 48 kHz and back, per
 * native hop:
 *
 *     E, P (native spectra)  --iFFT + sqrt-Hann WOLA-->  e[n], p[n]  (native)
 *     e, p  --audio_resampler up (x3 at 16 kHz, x6 at 8 kHz)-->  48 kHz FIFOs
 *     every 512 new 48 kHz samples: sqrt-Hann 1024/512 analysis of both
 *         --> dfn_res_stage_process(E48, P48) --> Y48 (source frame t-2)
 *         --> 48 kHz iFFT + WOLA --> audio_resampler down --> native FIFO
 *     emit exactly one native hop from that FIFO.
 *
 * The 48 kHz analysis is the same periodic sqrt-Hann, center=False, 50%
 * overlap transform the 48 kHz pipelines hand the stage directly, so the
 * stage sees the same kind of frame either way; only the resampler round
 * trip separates the two paths.
 *
 * CLOCKS. A native hop of h samples yields h*up 48 kHz samples, so the
 * stage runs a fixed schedule per hop: 16 kHz/hop 128 -> 0,1,1,1 (period
 * 4); 16 kHz/hop 256 and 8 kHz/hop 128 -> 1,2 (period 2). The peak of two inferences in
 * one hop is a real per-hop compute fact for the board budget. The native
 * output FIFO is prefilled with `prefill` zeros, the largest deficit that
 * schedule can leave between samples emitted and samples produced. Init
 * derives it from the resamplers' own tick counts rather than assuming it:
 * 16 kHz/hop 128 -> 128 (the schedule empties the FIFO every fourth hop),
 * 16 kHz/hop 256 -> 85, 8 kHz/hop 128 -> 42.
 *
 * DELAY. Relative to the hosting pipeline's own output the bridge adds
 *     prefill + up group delay + 1536*sr/48000 + down group delay
 * native samples (the 1536 are the stage's analysis/WOLA hop plus its two
 * lookahead hops at 48 kHz). Grids are named <rate>/hop <h>: 16 kHz/hop 128
 * -> 672 (42.0 ms), 16 kHz/hop 256 -> 629 (39.3 ms), 8 kHz/hop 128 -> 330
 * (41.3 ms). dfn_rate_bridge_added_delay_samples()
 * returns the value init computed. The delay is exact; the samples are not
 * an identity of P: the two FIR resamplers apply their pass-band response
 * twice, so the tests hold the output to a tolerance, not a memcmp.
 *
 * FAIL-OPEN. With no model the stage emits P48 delayed two frames, so the
 * bridge output is the hosting pipeline's own post-RES signal through the
 * resampler round trip, delayed by the constant above.
 *
 * MEMORY. Static pool with the house 32-byte descriptor. The stage pool and
 * the three resampler pools are carved inside this one; the stage's hash is
 * folded into build_flags_hash. No heap after init, no stdio,
 * -ffp-contract=off.
 * ========================================================================== */

#ifndef DFN_RATE_BRIDGE_H
#define DFN_RATE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "dfn_res_stage.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DFN_RATE_BRIDGE_DESCRIPTOR_VERSION 1u
#define DFN_RATE_BRIDGE_LAYOUT_VERSION     1u

/* GRID CONTRACT. The hop is always fft_size / 2: the hosting pipelines'
 * 50% overlap is part of what this bridge synthesises and analyses, so it is
 * fixed here rather than configured. The native grids are exactly 8 kHz/256,
 * 16 kHz/256 and 16 kHz/512, the bridge's own capability gate (its
 * resamplers, FIFO capacities, prefill calibration window and delay table
 * are proven for these three), not a copy of the hosts' list. A further
 * 50%-overlap grid extends that gate, the table and the tests; a different
 * overlap is a different bridge, not a config value. */
typedef struct DfnRateBridgeConfig {
    int sample_rate;            /* 8000 or 16000 (48000 uses the stage directly) */
    int fft_size;               /* 0 = rate default (256); 16 kHz also 512     */
    DfnResStageConfig stage;    /* 48 kHz stage config: matrices, model, limit */
} DfnRateBridgeConfig;

typedef struct {
    uint32_t descriptor_version;  /* DFN_RATE_BRIDGE_DESCRIPTOR_VERSION      */
    uint32_t layout_version;      /* DFN_RATE_BRIDGE_LAYOUT_VERSION          */
    uint32_t backend_id;          /* the stage's backend id                  */
    uint32_t build_flags_hash;    /* FNV-1a-32, folds the stage's hash       */
    uint32_t alignment;           /* 16                                      */
    uint32_t reserved;            /* 0                                       */
    uint64_t bytes;
} DfnRateBridgeMemReq;

_Static_assert(sizeof(DfnRateBridgeMemReq) == 32,
               "DfnRateBridgeMemReq must be exactly 32 bytes");

typedef struct DfnRateBridge DfnRateBridge;

/* Native grid `sample_rate` with the rate-default fft; the stage defaults
 * (48 kHz, no limit, no matrices, identity model). */
DfnRateBridgeConfig dfn_rate_bridge_default_config(int sample_rate);

/* Reject-first: -1 on NULL, a rate other than 8000/16000, an fft_size the
 * rate does not offer, or any stage rejection (dfn_res_stage.h). */
int dfn_rate_bridge_get_mem_requirements(const DfnRateBridgeConfig *cfg,
                                         DfnRateBridgeMemReq *out);
DfnRateBridge *dfn_rate_bridge_init(void *mem, size_t bytes,
                                    const DfnRateBridgeConfig *cfg);
/* init plus the 8-point stale-pool gate against `expected`. */
DfnRateBridge *dfn_rate_bridge_init_ex(void *mem, size_t bytes,
                                       const DfnRateBridgeConfig *cfg,
                                       const DfnRateBridgeMemReq *expected);
DfnRateBridge *dfn_rate_bridge_create(const DfnRateBridgeConfig *cfg);
void dfn_rate_bridge_destroy(DfnRateBridge *b);   /* frees only create()'s heap */
/* Resamplers, FIFOs, OLAs, the stage and the prefill. Whole-pipeline resets
 * only, as for the stage. */
void dfn_rate_bridge_reset(DfnRateBridge *b);

/* One native hop: the two native spectra in (n_freqs each), one native hop
 * of output. Returns 0, or -1 on NULL arguments or a stage contract error
 * (the bridge is then in an undefined state and must be reset). */
int dfn_rate_bridge_process(DfnRateBridge *b,
                            const Complex *estimate_spec,
                            const Complex *apply_spec,
                            float *out_hop);

int dfn_rate_bridge_hop_size(const DfnRateBridge *b);      /* -1 on NULL */
int dfn_rate_bridge_n_freqs(const DfnRateBridge *b);       /* -1 on NULL */
/* Native samples the bridge adds to the hosting pipeline's own latency. */
int dfn_rate_bridge_added_delay_samples(const DfnRateBridge *b);
/* Between hops only; -1 on NULL or a refused value. */
int dfn_rate_bridge_set_atten_lim(DfnRateBridge *b, float atten_lim_db);
DfnResStage *dfn_rate_bridge_stage(const DfnRateBridge *b);
/* Frame counters, for tests and diagnostics: 48 kHz frames run in total
 * and during the last hop (the schedule). */
void dfn_rate_bridge_get_counters(const DfnRateBridge *b,
                                  long long *frames_48k, int *last_hop_frames);

#ifdef __cplusplus
}
#endif

#endif /* DFN_RATE_BRIDGE_H */
