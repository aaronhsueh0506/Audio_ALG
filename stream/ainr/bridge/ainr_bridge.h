/* Sample-rate bridge between a host whose own grid is 8 or 16 kHz and a model that runs on the
 * 48 kHz / 1024 / hop 512 grid (DFN2, DFN3). The bridge holds no model: the caller runs it between
 * "take the model-rate spectra" and "put the enhanced spectrum back". AEC, RES, BF and comfort noise
 * stay on the host grid. Per host hop:
 *   E, P (host spectra) -> iFFT + sqrt-Hann WOLA -> e[n], p[n] (host)
 *   -> resample up (x3 at 16 kHz, x6 at 8 kHz) -> 48 kHz FIFOs
 *   every 512 new 48 kHz samples: sqrt-Hann 1024/512 analysis of both -> one model frame
 *     the caller's model -> one 48 kHz spectrum -> iFFT + WOLA -> resample down
 *   -> host FIFO -> exactly one host hop out.
 *
 * Clocks: a fixed per-hop schedule (16 kHz/hop 128: 0,1,1,1 frames; 16 kHz/hop 256 and 8 kHz/hop 128:
 * 1,2), so a hop carries at most two model frames. The host output FIFO is prefilled with `prefill`
 * zeros (init derives it from the resamplers' tick counts).
 * Delay added to the host's own output: prefill + up group delay + (L + 1)*512*sr/48000 + down group
 * delay host samples, L = cfg.lookahead_frames, the frames the caller's model trails its newest input
 * by (the bridge cannot see them: the model hands back the spectrum of frame t - L). For L = 2: 16 kHz/
 * hop 128 -> 672 (42.0 ms), 16 kHz/hop 256 -> 629 (39.3 ms), 8 kHz/hop 128 -> 330 (41.3 ms), as
 * returned by ainr_bridge_added_delay_samples(), which floors the sum. The model term is a whole number
 * of host samples only when L + 1 is a multiple of 3; every other term is whole, so the exact delay
 * is the floored one plus 0, 1/3 or 2/3 of a sample (ainr_bridge_added_delay_exact()). The resamplers
 * are not an identity (pass-band response applied twice).
 * Post-chain reset of the host (aec_post_reset_serial() moved): the two host-grid synthesis objects
 * (estimate and apply lane) run at the top of ainr_bridge_begin() with no FIFO or lookahead between
 * the spectra and them, so ainr_bridge_reset_synthesis() on that hop, before its begin(), lands on
 * the hop the serial protocol demands. Everything downstream is a continuous time-domain stream and
 * is left alone.
 * Scales: the 48 kHz AcStft / AcIstft run at audiolib's default scales (1); a model that expects
 * normalised spectra applies its own scale to the views and the inverse to what it puts back.
 * Model grid: 48000 / 1024 / 512 / 513 bins are constants here (AINR_BRIDGE_MODEL_*); the FIFO
 * capacities, the schedule and the prefill calibration are written for them. Taking the model grid from
 * the caller is a later step.
 * Layer: stream/ (it frames with audiolib's AcStft / AcIstft, which lib/ model cores never do). It
 * embeds no model, so the library DFN2 core, dfn2_stream_*_freq or a DFN3 stream plugs in alike.
 * Memory: static pool with the 32-byte descriptor; no heap after init, no stdio. */

#ifndef AINR_BRIDGE_H
#define AINR_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#include "fft_wrapper.h"   /* Complex */

#ifdef __cplusplus
extern "C" {
#endif

#define AINR_BRIDGE_DESCRIPTOR_VERSION 1u
/* Carve layout version; bumped on any change of pool size, carve order or descriptor content. */
#define AINR_BRIDGE_LAYOUT_VERSION     1u
#define AINR_BRIDGE_BACKEND_NE10       2u   /* the only FFT backend; id 1 is never returned */
/* Denominator of ainr_bridge_added_delay_exact(): sixths of a host sample. */
#define AINR_BRIDGE_DELAY_DENOMINATOR  6

#define AINR_BRIDGE_MODEL_RATE 48000
#define AINR_BRIDGE_MODEL_FFT  1024
#define AINR_BRIDGE_MODEL_HOP  512
#define AINR_BRIDGE_MODEL_BINS 513
/* Largest accepted lookahead_frames. */
#define AINR_BRIDGE_MAX_LOOKAHEAD 16

/* Host grids: 8 kHz/256, 16 kHz/256 and 16 kHz/512 (hop = fft_size / 2). */
typedef struct AinrBridgeConfig {
    int sample_rate;         /* 8000 or 16000 (48000 hosts call the model directly) */
    int fft_size;            /* 0 = rate default (256); 16 kHz also 512             */
    int lookahead_frames;    /* model frames of delay L, 0..AINR_BRIDGE_MAX_LOOKAHEAD */
} AinrBridgeConfig;

typedef struct {
    uint32_t descriptor_version;  /* AINR_BRIDGE_DESCRIPTOR_VERSION          */
    uint32_t layout_version;      /* AINR_BRIDGE_LAYOUT_VERSION              */
    uint32_t backend_id;          /* AINR_BRIDGE_BACKEND_NE10                */
    uint32_t build_flags_hash;    /* FNV-1a-32: grid, FIFO caps, resampler pool sizes */
    uint32_t alignment;           /* 16                                      */
    uint32_t reserved;            /* 0                                       */
    uint64_t bytes;
} AinrBridgeMemReq;

typedef struct AinrBridge AinrBridge;

/* One open model frame. The views are valid until ainr_bridge_model_end(). */
typedef struct AinrBridgeFrame {
    const Complex *estimate_spec;   /* AINR_BRIDGE_MODEL_BINS bins from the host's estimate lane */
    const Complex *apply_spec;      /* the same from the apply lane                              */
    Complex *out_spec;              /* AINR_BRIDGE_MODEL_BINS bins to write; zero-filled at open */
} AinrBridgeFrame;

/* Host grid `sample_rate`, the rate-default fft size, `lookahead_frames` of model delay. */
AinrBridgeConfig ainr_bridge_default_config(int sample_rate, int lookahead_frames);

/* Returns -1 on NULL, a rate other than 8000/16000, an fft_size the rate does not offer, or a
 * lookahead_frames outside 0..AINR_BRIDGE_MAX_LOOKAHEAD. */
int ainr_bridge_get_mem_requirements(const AinrBridgeConfig *cfg, AinrBridgeMemReq *out);
/* Constructs inside caller memory (16-byte aligned, >= req.bytes); init_ex also checks the stale-pool
 * descriptor against `expected`. A config, size or descriptor rejection writes nothing into the pool. Starts reset. */
AinrBridge *ainr_bridge_init(void *mem, size_t bytes, const AinrBridgeConfig *cfg);
AinrBridge *ainr_bridge_init_ex(void *mem, size_t bytes, const AinrBridgeConfig *cfg,
                                const AinrBridgeMemReq *expected);
AinrBridge *ainr_bridge_create(const AinrBridgeConfig *cfg);
void ainr_bridge_destroy(AinrBridge *b);   /* frees only create()'s heap */
/* Resets the resamplers, FIFOs, overlap-adds and the prefill. The model is the caller's to reset. */
void ainr_bridge_reset(AinrBridge *b);
/* Synthesis-only reset: clears the overlap-add tail and frame counter of the two host-grid synthesis
 * objects and nothing else (resamplers, FIFOs, analysis histories, 48 kHz synthesis, output FIFO with
 * its prefill and the counters keep running, so the added delay is unchanged). Call it between hops,
 * before ainr_bridge_begin() of the hop on which the serial moved. Returns 0, or -1 on NULL or while
 * a hop is in progress (nothing is cleared). */
int ainr_bridge_reset_synthesis(AinrBridge *b);

/* One host hop, phased: the caller runs the model for every frame the hop carries (0, 1 or 2):
 *     ainr_bridge_begin(b, E, P);                          // WOLA + resample up
 *     while (ainr_bridge_model_begin(b, &f) > 0) {         // a 48 kHz frame opens
 *         run_model(f.estimate_spec, f.apply_spec, f.out_spec);
 *         ainr_bridge_model_end(b);                        // f.out_spec goes back
 *     }
 *     ainr_bridge_finish(b, out_hop);                      // one host hop out
 * begin takes the host spectra (n_freqs bins each) and returns 0, or -1 on NULL arguments or a hop
 * already begun. model_begin returns 1 with the open frame in *frame, 0 when the hop has no further
 * frame, -1 on a contract error (no hop begun, or a frame still open). model_end returns 0, or -1 when
 * no frame is open. A frame must be answered with a full out_spec before the next model_begin: the
 * bridge does not skip or repeat frames (the model's own failure handling decides what it writes;
 * out_spec is zero until written). finish returns 0 and writes hop_size floats, or -1 on NULL, no hop
 * begun, a frame still open or a due frame not yet fetched; nothing changes then. After a -1 whose
 * cause was the caller's inability to answer an open frame, ainr_bridge_reset() is the way out. */
int ainr_bridge_begin(AinrBridge *b, const Complex *estimate_spec, const Complex *apply_spec);
int ainr_bridge_model_begin(AinrBridge *b, AinrBridgeFrame *frame);
int ainr_bridge_model_end(AinrBridge *b);
int ainr_bridge_finish(AinrBridge *b, float *out_hop);

int ainr_bridge_hop_size(const AinrBridge *b);      /* -1 on NULL */
int ainr_bridge_n_freqs(const AinrBridge *b);       /* -1 on NULL */
/* Host samples the bridge adds to the host's own latency. */
int ainr_bridge_added_delay_samples(const AinrBridge *b);   /* floor of the exact value */
/* The same delay without the floor: *numerator / *denominator host samples, the denominator always
 * AINR_BRIDGE_DELAY_DENOMINATOR (not reduced). With the host rate `sr` (8000 or 16000):
 *     numerator = (prefill + up latency) * 6
 *               + ((L + 1) * 512 + down latency in 48 kHz frames) * sr / 8000
 * whose division is exact. Returns 0, or -1 on a NULL argument. */
int ainr_bridge_added_delay_exact(const AinrBridge *b, int *numerator, int *denominator);
/* Diagnostics: 48 kHz frames run in total and in the last hop. */
void ainr_bridge_get_counters(const AinrBridge *b, long long *frames_48k, int *last_hop_frames);
/* Diagnostics: the host-grid hops the synthesis produced from the spectra of the last ainr_bridge_begin()
 * (hop_size floats each, before the resamplers). Valid until the next begin() or reset; NULL for an
 * output pointer you do not need. Returns 0, or -1 on NULL. */
int ainr_bridge_get_host_hops(const AinrBridge *b, const float **estimate_hop, const float **apply_hop);

#ifdef __cplusplus
}
#endif

#endif /* AINR_BRIDGE_H */
