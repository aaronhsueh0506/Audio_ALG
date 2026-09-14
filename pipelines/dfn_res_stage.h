/* ============================================================================
 * Post-RES DeepFilterNet2 stage, shared by the mono and 4-channel
 * "AEC + RES + DFN2" pipelines.
 *
 * One hop in, one spectrum out. The stage sits between a conventional
 * pipeline's post-RES spectrum and its synthesis:
 *
 *     A      = E  * 2^-5                        (E: the pre-RES spectrum the
 *                                                RES gain was estimated from)
 *     B      = P  * 2^-5                        (P: the post-RES spectrum the
 *                                                pipeline would synthesise,
 *                                                E * G_res + comfort noise)
 *     heads  = DFN2(estimated on A)             (dfn2_prepost_pre_process_freq_dual)
 *     Y      = compose(heads, applied on B)     (ERB mask, deep filter, alpha
 *                                                blend, attenuation limit)
 *     out    = Y * 2^5                          (source frame t - 2)
 *
 * The network ESTIMATES from the pre-RES spectrum, because DFN2 was trained
 * on unprocessed noisy speech and must not see the spectral holes RES cuts,
 * and APPLIES to the post-RES spectrum. With the host's comfort-noise fill
 * off, E and P share phase bin for bin (G_res is real), so the complex
 * deep-filter taps fitted on A apply to B without a phase mismatch. With the
 * fill on, the additive comfort noise perturbs P's phase exactly in the
 * RES-cut bins where it is largest, and the deep filter's estimate/apply
 * consistency no longer holds there (the ERB mask, being real, is far less
 * sensitive). Both hosting pipelines therefore keep the fill OFF by default
 * and expose it only as a product A/B switch; when it is on it travels
 * through the model like every other component of P, and atten_lim_db
 * bounds how far any bin can be attenuated. The dual-input topology with
 * the full deep filter is itself the product hypothesis: this class proves
 * it structurally (the identity gate below), and its quality against a
 * mask-only compose is settled by the Python evaluators once a checkpoint of
 * the current model contract exists.
 *
 * 2^-5 / 2^5: the pipelines' spectra are unnormalised STFTs (audio scale);
 * DFN2's FREQ boundary expects torch.stft(normalized=True), i.e. 1/32 of
 * that. Both factors are exact powers of two.
 *
 * TIMING. The emitted spectrum belongs to source frame t - 2
 * (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD, a cascade). process() returns 0
 * and writes an all-zero spectrum on the first two hops, so the caller
 * synthesises unconditionally and the delay is a pure delay.
 *
 * FAIL-OPEN. With no model (model.infer == NULL), a nonzero infer() result,
 * or an unwritten/non-finite output, the frame is taken with
 * dfn2_prepost_frame_skip(): an exact identity through the cascade (unit ERB
 * mask through a partition-of-unity erb_inv, zero taps, alpha 0), the
 * recurrent state not stepped, the clocks advanced. The stage output is then
 * P delayed by two hops: the conventional pipeline with its NR disabled,
 * byte for byte, comfort noise included. That is the backbone gate of both
 * hosting pipelines.
 *
 * RESET. Whole-pipeline resets only. The stage consumes no far-end, so an AEC
 * delay change leaves both of its inputs continuous and must not reset it (a
 * reset costs two zero hops and restarts the recurrent state).
 *
 * GRID. 48 kHz / 1024 / 512 only: DFN2's own grid, which the pipelines'
 * 48 kHz grid already is (same periodic sqrt-Hann, same 50% overlap), so E
 * is a valid DFN2 analysis frame without a second transform. Other rates are
 * refused at get_mem_requirements.
 *
 * MEMORY. Static pool with the house 32-byte descriptor. The DFN2 class pool
 * is carved inside this one and its own build hash is folded into
 * build_flags_hash, so a DFN2 carve change reaches the hosting pipeline's
 * descriptor. No heap after init, no stdio, -ffp-contract=off.
 * ========================================================================== */

#ifndef DFN_RES_STAGE_H
#define DFN_RES_STAGE_H

#include <stddef.h>
#include <stdint.h>

#include "dfn2_prepost.h"   /* DFN2Model, DFN2ModelIoDescriptor, the grid */
#include "fft_wrapper.h"    /* Complex */

#ifdef __cplusplus
extern "C" {
#endif

#define DFN_RES_STAGE_DESCRIPTOR_VERSION 1u
#define DFN_RES_STAGE_LAYOUT_VERSION     1u
#define DFN_RES_STAGE_BACKEND_KISS       1u
#define DFN_RES_STAGE_BACKEND_NE10       2u

/* Pipeline audio-scale spectra <-> DFN2 normalized=True spectra. Hex-float
 * literals: no division, no decimal round trip, exact powers of two. */
#define DFN_RES_STAGE_IN_SCALE  0x1p-5f
#define DFN_RES_STAGE_OUT_SCALE 0x1p5f

/* Frames the emitted spectrum trails the newest input by. */
#define DFN_RES_STAGE_LOOKAHEAD_FRAMES (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD)

typedef struct DfnResStageConfig {
    int          sample_rate;   /* 48000 only                                */
    int          fft_size;      /* 0 or 1024                                 */
    float        atten_lim_db;  /* DFN2 attenuation limit; 0 = unlimited     */
    const float *erb_fwd;       /* borrowed, bin-major [DFN2_N_BINS][DFN2_N_ERB] */
    const float *erb_inv;       /* borrowed, band-major, partition of unity  */
    DFN2Model    model;         /* by value; all-zero = identity             */
} DfnResStageConfig;

typedef struct {
    uint32_t descriptor_version;  /* DFN_RES_STAGE_DESCRIPTOR_VERSION         */
    uint32_t layout_version;      /* DFN_RES_STAGE_LAYOUT_VERSION             */
    uint32_t backend_id;          /* DFN_RES_STAGE_BACKEND_KISS / _NE10       */
    uint32_t build_flags_hash;    /* FNV-1a-32, folds the DFN2 class hash     */
    uint32_t alignment;           /* 16                                       */
    uint32_t reserved;            /* 0                                        */
    uint64_t bytes;
} DfnResStageMemReq;

_Static_assert(sizeof(DfnResStageMemReq) == 32,
               "DfnResStageMemReq must be exactly 32 bytes");

typedef struct DfnResStage DfnResStage;

/* Defaults: 48 kHz, fft 1024, no attenuation limit, no matrices, identity
 * model. */
DfnResStageConfig dfn_res_stage_default_config(int sample_rate);

/* Reject-first: -1 on NULL, a rate other than 48000, an fft_size other than
 * 0/1024, a missing ERB matrix, an erb_inv through which a unit band mask
 * does not expand to exactly 1.0f in every bin (walked in the DFN2
 * expansion order, so the fail-open identity holds bit for bit), a non-finite
 * attenuation limit, a model that infers without a descriptor or with a
 * descriptor dfn2_model_io_descriptor_validate() refuses, or a build outside
 * pipelines/Makefile (backend id 0). */
int dfn_res_stage_get_mem_requirements(const DfnResStageConfig *cfg,
                                       DfnResStageMemReq *out);

/* Construct inside caller memory (16-byte aligned, >= req.bytes). Both apply
 * every get_mem_requirements() rejection; init_ex additionally applies the
 * 8-point stale-pool gate against `expected`. Starts reset. */
DfnResStage *dfn_res_stage_init(void *mem, size_t bytes,
                                const DfnResStageConfig *cfg);
DfnResStage *dfn_res_stage_init_ex(void *mem, size_t bytes,
                                   const DfnResStageConfig *cfg,
                                   const DfnResStageMemReq *expected);
DfnResStage *dfn_res_stage_create(const DfnResStageConfig *cfg);
void dfn_res_stage_destroy(DfnResStage *s);   /* frees only create()'s heap */

/* Reset the DFN2 class and the hop counters; calls model.reset when
 * present. See RESET above: whole-pipeline resets only. */
void dfn_res_stage_reset(DfnResStage *s);

/* Between hops only; -1 on NULL or a non-finite value. */
int dfn_res_stage_set_atten_lim(DfnResStage *s, float atten_lim_db);

/* Source frame of the last emitted spectrum (the DFN2 class's own clock);
 * -1 before the first emission. */
int dfn_res_stage_output_frame_index(const DfnResStage *s, long long *frame);

/* Hop counters, for tests and diagnostics: frames pushed, head evaluations
 * committed, frames taken as the identity. */
void dfn_res_stage_get_counters(const DfnResStage *s, long long *frames_in,
                                long long *commits, long long *skips);

/* One hop. `estimate_spec` is the pre-RES spectrum the model estimates from
 * (E); `apply_spec` is the post-RES spectrum the heads are applied to (P).
 * Keeping the two pointers distinct is the contract of this class: the
 * estimate/apply split is visible at the seam and testable with distinct
 * inputs. Both are borrowed for the duration of the call (they may alias the
 * hosting pipeline's per-hop buffers). `out_spec` is ALWAYS fully written:
 * zeros on the two warm-up hops. Returns 1 when out_spec carries an emitted
 * frame, 0 during warm-up, -1 on a contract error (NULL or non-finite
 * arguments, or the DFN2 class refusing the frame), in which case out_spec
 * is untouched. */
int dfn_res_stage_process(DfnResStage *s,
                          const Complex *estimate_spec,
                          const Complex *apply_spec,
                          Complex *out_spec);

#ifdef __cplusplus
}
#endif

#endif /* DFN_RES_STAGE_H */
