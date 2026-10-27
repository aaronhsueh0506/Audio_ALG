/**
 * tests/test_mcab_sample_beam.c -- the SRP-PHAT/BF/fallback-VAD sample helper over an mcab flow
 * (mcab_sample_beam.h), on all three production signal grids: the split front / beam / fuse
 * sequence and its guards, the helper's own pool contract, fuse leaving the residual gain to the
 * caller, the helper followed by the examples' RES / NR / min / comfort noise / synthesis (the
 * RES an AecRes the caller builds with the libaec functions) against the independent oracle
 * (conv_chain_mcab.h), that caller-owned RES across reset and preset changes, and mcab_raw run
 * as a black box against that oracle. The flow-only checks are in test_mcab.c.
 * Arguments: the directory of the example binaries and a scratch directory.
 */

#define MCAB_TEST_RIG_WITH_BEAM 1
#include "mcab_test_rig.h"
#include "mcab_sample_flow.h"
#include "bf.h"
#include "srp.h"
#include "conv_chain_mcab.h"
#include "mcab_sample_post.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
            return 0;                                                  \
        }                                                              \
    } while (0)

static int spectrum_finite(const Complex* spectrum, int n) {
    for (int k = 0; k < n; ++k)
        if (!isfinite(spectrum[k].r) || !isfinite(spectrum[k].i)) return 0;
    return 1;
}

static int run_grid(int sample_rate, int fft_size) {
    McabTestUnitConfig cfg =
        mcab_test_unit_default_config(sample_rate);
    McabSampleBeam* p;
    McabSampleBeamInfo info;
    McabSampleBeamResult beam;
    McabFrame frame;
    float* microphones;
    float* far;
    float phase = 0.0f;
    int hop;
    int n_freqs;
    int doa_analysis_total = 0;

    cfg.core.fft_size = fft_size;
    cfg.beam.bf_fixed_mode = 1;
    cfg.beam.bf_fixed_doa_rad = 0.4f;
    cfg.beam.bf_mu = 0.02f;
    p = mcab_test_beam_create(&cfg);
    CHECK(p != NULL, "create 4ch linear AEC + BF unit");
    hop = mcab_hop_size(mcab_test_beam_flow(p));
    n_freqs = mcab_n_freqs(mcab_test_beam_flow(p));
    CHECK(hop == fft_size / 2,
          "rate-specific hop");
    CHECK(mcab_fft_size(mcab_test_beam_flow(p)) == fft_size,
          "zero-padding-free frame/FFT contract");
    CHECK(n_freqs == fft_size / 2 + 1,
          "spatial bin count follows selected FFT");
    {
        /* What the helper hands SRP and BF is on the flow's grid. */
        McabSampleSpatialConfig sc;
        CHECK(mcab_sample_spatial_config(&cfg.core, &cfg.beam, &sc) == 0 &&
                  sc.srp.sr == (float)sample_rate && sc.srp.NFFT == (float)fft_size &&
                  sc.srp.F == n_freqs && sc.srp.M == MCAB_CHANNELS,
              "DOA grid follows selected main AEC grid");
        CHECK(sc.n_freqs == n_freqs && sc.n_channels == MCAB_CHANNELS &&
                  sc.bf.adapt_interval == mcab_sample_beam_bf_effective_adapt_interval(p),
              "BF grid follows selected main AEC grid");
    }
    CHECK(mcab_matched_filter_count(mcab_test_beam_flow(p)) == 1,
          "one shared matcher");
    CHECK(mcab_linear_aec_count(mcab_test_beam_flow(p)) ==
              MCAB_CHANNELS,
          "four linear AEC lanes");

    microphones = (float*)malloc(
        (size_t)hop * MCAB_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far, "allocate unit buffers");

    for (int frame_no = 0; frame_no < 80; ++frame_no) {
        for (int i = 0; i < hop; ++i) {
            float echo = 0.08f * sinf(phase);
            float near = frame_no >= 30
                ? 0.025f * sinf(phase * 1.73f + 0.2f) : 0.0f;
            far[i] = echo;
            for (int ch = 0; ch < MCAB_CHANNELS; ++ch) {
                microphones[i * MCAB_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch) +
                    near * (1.0f + 0.02f * ch);
            }
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(mcab_sample_beam_process_with_activity(
                  p, microphones, far,
                  frame_no >= 30, frame_no >= 30, NULL, &beam, &info) ==
                  MCAB_OK,
              "explicit-VAD frame");
        CHECK(info.frame_index == (uint64_t)frame_no,
              "monotonic spatial frame index");
        CHECK(info.doa_analysis_frames >= 0 &&
              info.doa_analysis_frames <= 1,
              "bounded DOA analyses per main hop");
        doa_analysis_total += info.doa_analysis_frames;
        CHECK(isfinite(info.doa_used_rad), "fixed BF DOA must be finite");
        CHECK(beam.spectrum && beam.weights &&
              spectrum_finite(beam.spectrum, n_freqs) &&
              spectrum_finite(beam.weights, MCAB_CHANNELS * n_freqs),
              "beam spectrum and weights are finite");
        CHECK(mcab_sample_beam_fuse(p, &frame) == MCAB_OK &&
              frame.n_freqs == n_freqs && frame.e == beam.spectrum &&
              frame.r2 && frame.cn && frame.far_hop,
              "fuse publishes the beam spectrum and the post-stage inputs");
    }
    CHECK(doa_analysis_total == 80,
          "same-grid DOA consumes every main analysis frame");

    mcab_sample_beam_reset(p);
    CHECK(mcab_sample_beam_process(
              p, microphones, far, &beam, &info) ==
              MCAB_OK,
          "fallback-VAD processing after reset");
    CHECK(info.frame_index == 0, "reset restarts the unit frame index");
    CHECK(info.doa_analysis_frames == 1,
          "same-grid DOA resumes immediately after reset");

    free(far);
    free(microphones);
    mcab_test_beam_destroy(p);
    return 1;
}

/* Contract test: in fixed-notebook mode (bf_fixed_mode && bf_fixed_align_notebook),
 * bf_create() forces the BF RLS update cadence to 1 hop whatever bf_adapt_interval is.
 * The lambda retime in mcab_test_beam_create() must use that forced cadence, or
 * lambda is calibrated for a slower update period than the one that runs. */
static int test_bf_fixed_notebook_lambda_matches_forced_cadence(void) {
    McabTestUnitConfig cfg =
        mcab_test_unit_default_config(16000);
    McabSampleBeam* p;
    int hop_size;
    int actual_interval;
    float actual_lambda;
    float expected_lambda;
    float buggy_lambda;

    cfg.beam.bf_fixed_mode = 1;
    cfg.beam.bf_fixed_align_notebook = 1;
    cfg.beam.bf_adapt_interval = 4; /* caller asks for a slower cadence... */
    cfg.beam.bf_lambda = 0.98f;

    p = mcab_test_beam_create(&cfg);
    CHECK(p != NULL, "fixed-notebook cadence test: create pipeline");

    hop_size = mcab_hop_size(mcab_test_beam_flow(p));
    actual_interval = mcab_sample_beam_bf_effective_adapt_interval(p);
    actual_lambda = mcab_sample_beam_bf_lambda(p);

    /* ...but fixed_align_notebook forces the real BF cadence to every hop
     * (1), regardless of the requested 4. */
    CHECK(actual_interval == 1,
          "fixed-notebook mode forces the actual BF cadence to 1");

    /* Lambda must be retimed assuming that SAME forced interval (1), i.e.
     * exactly anr_retime_alpha(bf_lambda, sr, hop_size*1) -- not
     * hop_size*4, which scaling with the raw pre-forced bf_adapt_interval
     * would give. */
    expected_lambda = anr_retime_alpha(
        cfg.beam.bf_lambda, cfg.core.sample_rate, hop_size * 1);
    CHECK(fabsf(actual_lambda - expected_lambda) < 1e-9f,
          "lambda retime uses the forced (not the requested) adapt_interval");

    /* Sanity check that this test can fail: the lambda scaled against the
     * pre-forced interval of 4 must visibly differ from the correct one. */
    buggy_lambda = anr_retime_alpha(
        cfg.beam.bf_lambda, cfg.core.sample_rate, hop_size * 4);
    CHECK(fabsf(actual_lambda - buggy_lambda) > 1e-4f,
          "forced-cadence lambda differs from one retimed with the requested interval");

    mcab_test_beam_destroy(p);
    return 1;
}

/* Companion check: outside fixed-notebook mode, a configured
 * bf_adapt_interval > 1 is NOT forced down, so the effective interval used
 * for both the actual BF cadence and the lambda retime must equal the
 * requested value itself (guards the other branch of
 * bf_effective_adapt_interval() from regressing). */
static int test_bf_auto_mode_respects_configured_interval(void) {
    McabTestUnitConfig cfg =
        mcab_test_unit_default_config(16000);
    McabSampleBeam* p;
    int hop_size;
    float expected_lambda;

    cfg.beam.bf_adapt_interval = 4;
    cfg.beam.bf_lambda = 0.98f;

    p = mcab_test_beam_create(&cfg);
    CHECK(p != NULL, "auto-mode cadence test: create pipeline");

    hop_size = mcab_hop_size(mcab_test_beam_flow(p));
    CHECK(mcab_sample_beam_bf_effective_adapt_interval(p) == 4,
          "auto mode keeps the requested adapt_interval unforced");

    expected_lambda = anr_retime_alpha(
        cfg.beam.bf_lambda, cfg.core.sample_rate, hop_size * 4);
    CHECK(fabsf(mcab_sample_beam_bf_lambda(p) - expected_lambda) < 1e-9f,
          "lambda retime uses the requested adapt_interval in auto mode");

    mcab_test_beam_destroy(p);
    return 1;
}

/* mcab_sample_beam_process() must reject a bad call (NULL far_reference/beam,
 * non-finite input) before auto_vad() can fold it into the noise_power EMA; a NaN
 * noise_power would make every later comparison false and disable VAD for good. Two
 * identically-fed pipelines run side by side and only "attacked" also receives the calls
 * that must be rejected; any leak into VAD state makes it diverge from "clean". */
static int test_rejected_calls_preserve_vad_state(void) {
    const int sample_rate = 16000;
    const int fft_size = 256;
    McabTestUnitConfig cfg =
        mcab_test_unit_default_config(sample_rate);
    McabSampleBeam *clean, *attacked;
    int hop, n_freqs;
    float *mic, *far, *nan_mic, *nan_far, *inf_far, *quiet_mic;
    float phase = 0.0f;
    McabSampleBeamResult beam_clean, beam_attacked, throwaway_beam;
    McabSampleBeamInfo info_clean, info_attacked, throwaway;

    cfg.core.fft_size = fft_size;
    clean = mcab_test_beam_create(&cfg);
    attacked = mcab_test_beam_create(&cfg);
    CHECK(clean && attacked, "create clean/attacked pipeline pair");
    hop = mcab_hop_size(mcab_test_beam_flow(clean));
    n_freqs = mcab_n_freqs(mcab_test_beam_flow(clean));

    mic = (float*)malloc(
        (size_t)hop * MCAB_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    nan_mic = (float*)malloc(
        (size_t)hop * MCAB_CHANNELS * sizeof(float));
    nan_far = (float*)malloc((size_t)hop * sizeof(float));
    inf_far = (float*)malloc((size_t)hop * sizeof(float));
    quiet_mic = (float*)malloc(
        (size_t)hop * MCAB_CHANNELS * sizeof(float));
    CHECK(mic && far && nan_mic
              && nan_far && inf_far && quiet_mic,
          "allocate VAD-contamination test buffers");

    /* Quiet warm-up so noise_power settles away from its init value
     * identically on both instances before the attack. */
    for (int frame = 0; frame < 5; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float v = 0.002f * sinf(phase);
            far[i] = v;
            for (int ch = 0; ch < MCAB_CHANNELS; ++ch)
                mic[i * MCAB_CHANNELS + ch] = v;
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(mcab_sample_beam_process(
                  clean, mic, far, &beam_clean, &info_clean) ==
                  MCAB_OK,
              "clean warm-up frame");
        CHECK(mcab_sample_beam_process(
                  attacked, mic, far, &beam_attacked, &info_attacked) ==
                  MCAB_OK,
              "attacked warm-up frame");
    }

    /* One loud frame on BOTH instances arms vad_hangover to the same
     * nonzero value on each -- a quiet warm-up alone never exercises the
     * `if (speech) p->vad_hangover -= 1;` branch a leaking rejected call
     * would still execute (hangover stays 0 the whole time otherwise, so
     * `speech` there is always false and that line never runs). */
    for (int i = 0; i < hop; ++i) {
        float v = 0.05f * sinf(phase);
        far[i] = 0.0f;
        for (int ch = 0; ch < MCAB_CHANNELS; ++ch)
            mic[i * MCAB_CHANNELS + ch] = v;
        phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
    }
    CHECK(mcab_sample_beam_process(
              clean, mic, far, &beam_clean, &info_clean) ==
              MCAB_OK,
          "clean hangover-arming frame");
    CHECK(mcab_sample_beam_process(
              attacked, mic, far, &beam_attacked, &info_attacked) ==
              MCAB_OK,
          "attacked hangover-arming frame");
    CHECK(info_clean.vad_raw == 1 && info_attacked.vad_raw == 1,
          "loud tone actually arms the hangover on both instances");

    memcpy(nan_mic, mic,
           (size_t)hop * MCAB_CHANNELS * sizeof(float));
    nan_mic[0] = NAN;
    memcpy(nan_far, far, (size_t)hop * sizeof(float));
    nan_far[0] = NAN;
    memcpy(inf_far, far, (size_t)hop * sizeof(float));
    inf_far[0] = (float)INFINITY;
    CHECK(mcab_sample_beam_process(
              attacked, mic, NULL, &throwaway_beam, &throwaway) ==
              MCAB_INVALID_ARGUMENT,
          "NULL far_reference is rejected");
    CHECK(mcab_sample_beam_process(
              attacked, mic, far, NULL, &throwaway) ==
              MCAB_INVALID_ARGUMENT,
          "NULL beam is rejected");
    CHECK(mcab_sample_beam_process(
              attacked, nan_mic, far, &throwaway_beam, &throwaway) !=
              MCAB_OK,
          "NaN mic input is rejected, not silently processed");
    /* auto_vad() reads only the microphones, so on finite mic data a non-finite far would
     * pass its own guard and only be caught by the finite check inside
     * mcab_process_front(), after auto_vad() has already moved
     * noise_power/vad_hangover. NaN and Inf are both exercised. The mic content is quiet
     * on purpose: a loud frame would re-arm vad_hangover to the value it already has and
     * hide a missing far_reference check, while quiet content takes the EMA's silence
     * branch and changes noise_power if auto_vad() wrongly runs. */
    for (int i = 0; i < hop; ++i) {
        float v = 0.003f * sinf(phase + 1.7f);
        for (int ch = 0; ch < MCAB_CHANNELS; ++ch)
            quiet_mic[i * MCAB_CHANNELS + ch] = v;
    }
    CHECK(mcab_sample_beam_process(
              attacked, quiet_mic, nan_far, &throwaway_beam, &throwaway) ==
              MCAB_INVALID_ARGUMENT,
          "NaN far_reference is rejected before touching VAD state");
    CHECK(mcab_sample_beam_process(
              attacked, quiet_mic, inf_far, &throwaway_beam, &throwaway) ==
              MCAB_INVALID_ARGUMENT,
          "Inf far_reference is rejected before touching VAD state");

    /* Resume identical valid input on both, QUIET so the hangover counter walks down to 0
     * instead of being re-armed every hop (a loud frame would mask a 1-frame hangover
     * discrepancy). A leak in any rejected call above desyncs "attacked" from "clean". */
    for (int frame = 0; frame < 40; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float v = 0.002f * sinf(phase);
            far[i] = v;
            for (int ch = 0; ch < MCAB_CHANNELS; ++ch)
                mic[i * MCAB_CHANNELS + ch] = v;
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(mcab_sample_beam_process(
                  clean, mic, far, &beam_clean, &info_clean) ==
                  MCAB_OK,
              "clean post-attack frame");
        CHECK(mcab_sample_beam_process(
                  attacked, mic, far, &beam_attacked, &info_attacked) ==
                  MCAB_OK,
              "attacked post-attack frame");
        CHECK(info_clean.vad_raw == info_attacked.vad_raw &&
              info_clean.vad_out == info_attacked.vad_out,
              "rejected calls must not desync VAD decisions from an "
              "identically-fed clean instance");
        CHECK(memcmp(beam_clean.spectrum, beam_attacked.spectrum,
                     (size_t)n_freqs * sizeof(Complex)) == 0,
              "rejected calls must not desync the beam spectrum from an "
              "identically-fed clean instance");
    }

    /* Positive check, not just clean==attacked: guards against both
     * instances being identically broken (e.g. if noise_power corruption
     * were somehow shared or the loud tone were mistuned). */
    for (int i = 0; i < hop; ++i) {
        float v = 0.05f * sinf(phase);
        far[i] = 0.0f;
        for (int ch = 0; ch < MCAB_CHANNELS; ++ch)
            mic[i * MCAB_CHANNELS + ch] = v;
        phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
    }
    CHECK(mcab_sample_beam_process(
              clean, mic, far, &beam_clean, &info_clean) ==
              MCAB_OK,
          "clean final loud frame");
    CHECK(info_clean.vad_raw == 1,
          "loud tone is actually detected as speech");

    free(mic); free(far); free(nan_mic);
    free(nan_far); free(inf_far); free(quiet_mic);
    mcab_test_beam_destroy(clean);
    mcab_test_beam_destroy(attacked);
    return 1;
}

/* Static-memory pool coverage for the descriptor-tier composition layer: poison/bounds,
 * descriptor-staleness rejection and a heap-vs-pool byte-equal run. Follows
 * run_static_parity() in test_mcab.c (0xa5 poison with a trailing-byte bounds
 * check, the 8-point descriptor mutation block, destroy-then-reinit pool reuse) plus a
 * heap vs. pool run over many hops comparing the beam spectrum, the weights, the fused
 * frame and every scalar field of McabSampleBeamInfo. The composite pool can be
 * several MB at 48 kHz/1024, so there is no full-pool scan around every rejection. */
static int run_static_parity(int sample_rate, int fft_size) {
    enum { EXTRA = 32, FRAMES = 40 };
    McabTestUnitConfig cfg =
        mcab_test_unit_default_config(sample_rate);
    McabSampleFlow objs;
    McabSampleBeam* heap;
    McabSampleBeam* stat;
    unsigned char* pool = NULL;
    unsigned char* beam_pool;
    float* microphones;
    float* far;
    float phase_step;
    size_t flow_bytes, beam_bytes, pool_bytes;
    size_t k;
    int hop;
    int n_freqs;
    int frame;

    cfg.core.fft_size = fft_size;
    cfg.beam.bf_fixed_mode = 1;
    cfg.beam.bf_fixed_doa_rad = 0.35f;
    cfg.beam.bf_mu = 0.02f;

    /* One caller pool: the flow's objects (mcab_sample_flow) and then the helper. mcab's own
     * descriptor contract is test_mcab.c's run_static_parity. */
    CHECK(mcab_sample_flow_get_mem_size(&cfg.core, &flow_bytes) == 0 &&
              mcab_sample_beam_get_mem_size(&cfg.core, &cfg.beam, &beam_bytes) == 0,
          "static memory requirement query succeeds");
    flow_bytes = ALIGN16(flow_bytes);
    pool_bytes = flow_bytes + beam_bytes + (size_t)EXTRA;

    CHECK(posix_memalign((void**)&pool, 16u, pool_bytes) == 0 && pool,
          "aligned caller pool allocates");
    memset(pool, 0xa5, pool_bytes);
    CHECK(mcab_sample_flow_init(&objs, pool, flow_bytes, &cfg.core) == 0,
          "the flow builds in the caller pool");
    beam_pool = pool + flow_bytes;

    CHECK(mcab_sample_beam_init(beam_pool + 1, beam_bytes, &cfg.core, &cfg.beam, objs.flow) == NULL,
          "static init rejects a misaligned pool");
    CHECK(mcab_sample_beam_init(beam_pool, beam_bytes - 1u, &cfg.core, &cfg.beam, objs.flow) == NULL,
          "static init rejects an undersized pool");
    {
        McabTestUnitConfig other = cfg;
        other.core.fft_size = fft_size == 512 ? 256 : (sample_rate == 16000 ? 512 : 1024);
        CHECK(sample_rate == 48000 ||
                  mcab_sample_beam_init(beam_pool, beam_bytes * 4u, &other.core, &other.beam,
                                        objs.flow) == NULL,
              "static init rejects a flow of another grid");
        other = cfg;
        other.core.enable_post = 0;
        CHECK(mcab_sample_beam_init(beam_pool, beam_bytes, &other.core, &other.beam, objs.flow) == NULL,
              "static init rejects a flow whose enable_post disagrees with the config");
        other = cfg;
        other.beam.bf_lambda = 2.0f;
        CHECK(mcab_sample_beam_init(beam_pool, beam_bytes, &other.core, &other.beam, objs.flow) == NULL,
              "static init rejects an invalid beamformer parameter set");
        CHECK(mcab_sample_beam_init(beam_pool, beam_bytes, &cfg.core, &cfg.beam, NULL) == NULL,
              "static init rejects a missing flow");
    }
    for (k = 0; k < beam_bytes + (size_t)EXTRA; ++k)
        if (beam_pool[k] != 0xa5) break;
    CHECK(k == beam_bytes + (size_t)EXTRA, "every refused init left the helper pool unwritten");

    stat = mcab_sample_beam_init(beam_pool, beam_bytes, &cfg.core, &cfg.beam, objs.flow);
    heap = mcab_test_beam_create(&cfg);
    CHECK(stat != NULL && heap != NULL,
          "poisoned caller pool and heap construction both succeed");

    for (k = 0; k < (size_t)EXTRA; ++k) {
        if (beam_pool[beam_bytes + k] != 0xa5) break;
    }
    CHECK(k == (size_t)EXTRA, "static init stays inside the queried pool");

    hop = mcab_hop_size(objs.flow);
    n_freqs = mcab_n_freqs(objs.flow);
    CHECK(hop == mcab_hop_size(mcab_test_beam_flow(heap)),
          "heap and pool instances agree on hop size");
    CHECK(hop == fft_size / 2, "rate-specific hop");

    microphones = (float*)malloc(
        (size_t)hop * MCAB_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far,
          "static parity buffers allocate");

    phase_step = (float)(2.0 * M_PI * 700.0 / sample_rate);
    for (frame = 0; frame < FRAMES; ++frame) {
        McabSampleBeamInfo heap_info;
        McabSampleBeamInfo stat_info;
        McabSampleBeamResult heap_beam;
        McabSampleBeamResult stat_beam;
        McabFrame heap_frame;
        McabFrame stat_frame;
        int vad = frame >= 15;
        int i;
        int ch;

        memset(&heap_info, 0, sizeof(heap_info));
        memset(&stat_info, 0, sizeof(stat_info));

        for (i = 0; i < hop; ++i) {
            int64_t absolute = (int64_t)frame * hop + i;
            float phase = phase_step * (float)absolute;
            float echo = 0.08f * sinf(phase);
            float near = vad ? 0.02f * sinf(phase * 1.6f + 0.3f) : 0.0f;
            far[i] = echo;
            for (ch = 0; ch < MCAB_CHANNELS; ++ch) {
                microphones[i * MCAB_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch) + near * (1.0f + 0.02f * ch);
            }
        }

        CHECK(mcab_sample_beam_process_with_activity(
                  heap, microphones, far, vad, vad, NULL,
                  &heap_beam, &heap_info) == MCAB_OK,
              "heap instance processes a synthetic hop");
        CHECK(mcab_sample_beam_process_with_activity(
                  stat, microphones, far, vad, vad, NULL,
                  &stat_beam, &stat_info) == MCAB_OK,
              "pool instance processes a synthetic hop");
        CHECK(memcmp(heap_beam.spectrum, stat_beam.spectrum,
                     (size_t)n_freqs * sizeof(Complex)) == 0,
              "heap/pool beam spectrum is byte-identical");
        CHECK(memcmp(heap_beam.weights, stat_beam.weights,
                     (size_t)MCAB_CHANNELS * (size_t)n_freqs *
                         sizeof(Complex)) == 0,
              "heap/pool beam weights are byte-identical");
        CHECK(mcab_sample_beam_fuse(heap, &heap_frame) == MCAB_OK &&
              mcab_sample_beam_fuse(stat, &stat_frame) == MCAB_OK &&
              memcmp(heap_frame.r2, stat_frame.r2, (size_t)n_freqs * sizeof(float)) == 0 &&
              memcmp(heap_frame.cn, stat_frame.cn, (size_t)n_freqs * sizeof(float)) == 0 &&
              memcmp(heap_frame.y, stat_frame.y, (size_t)n_freqs * sizeof(Complex)) == 0,
              "heap/pool fused frames are byte-identical");
        CHECK(heap_info.frame_index == stat_info.frame_index,
              "heap/pool frame_index matches");
        CHECK(heap_info.delay.delay_samples ==
                      stat_info.delay.delay_samples &&
                  heap_info.delay.confidence == stat_info.delay.confidence &&
                  heap_info.delay.solid == stat_info.delay.solid &&
                  heap_info.delay.changed == stat_info.delay.changed &&
                  heap_info.delay.estimator_calls ==
                      stat_info.delay.estimator_calls &&
                  heap_info.delay.estimator_updates ==
                      stat_info.delay.estimator_updates,
              "heap/pool delay state matches");
        CHECK((isnan(heap_info.doa_raw_rad) &&
                   isnan(stat_info.doa_raw_rad)) ||
                  heap_info.doa_raw_rad == stat_info.doa_raw_rad,
              "heap/pool doa_raw_rad matches");
        CHECK((isnan(heap_info.doa_smooth_rad) &&
                   isnan(stat_info.doa_smooth_rad)) ||
                  heap_info.doa_smooth_rad == stat_info.doa_smooth_rad,
              "heap/pool doa_smooth_rad matches");
        CHECK((isnan(heap_info.doa_used_rad) &&
                   isnan(stat_info.doa_used_rad)) ||
                  heap_info.doa_used_rad == stat_info.doa_used_rad,
              "heap/pool doa_used_rad matches");
        CHECK(heap_info.vad_raw == stat_info.vad_raw &&
                  heap_info.vad_out == stat_info.vad_out &&
                  heap_info.bf_adaptive == stat_info.bf_adaptive &&
                  heap_info.doa_analysis_frames ==
                      stat_info.doa_analysis_frames,
              "heap/pool remaining frame info fields match");
    }

    mcab_sample_beam_destroy(stat);
    CHECK(isnan(mcab_sample_beam_bf_lambda(stat)) &&
              mcab_sample_beam_bf_effective_adapt_interval(stat) == -1,
          "destroy marks a caller-pool instance inactive");
    mcab_sample_beam_destroy(stat);
    CHECK(mcab_sample_beam_bf_effective_adapt_interval(stat) == -1,
          "caller-pool destroy is idempotent");
    stat = mcab_sample_beam_init(beam_pool, beam_bytes, &cfg.core, &cfg.beam, objs.flow);
    CHECK(stat != NULL, "caller pool is reusable after destroy");

    mcab_sample_beam_destroy(stat);
    mcab_sample_flow_destroy(&objs);
    mcab_test_beam_destroy(heap);
    free(far);
    free(microphones);
    free(pool);
    return 1;
}

/* ---------------------------------------------------------------------------
 * The chain the examples run behind the unit (RES, NR, min, comfort noise, synthesis), against
 * the reference chain written over the front end (conv_chain_mcab.h).
 * ------------------------------------------------------------------------ */

static void* aligned_block(size_t bytes) {
    void* mem = NULL;
    return posix_memalign(&mem, 16, bytes) == 0 ? mem : NULL;
}

/* The stages the examples carve behind the unit (mcab_static.c): the RES (an
 * AecRes built here from the preset's AecConfig on the unit's grid), NR, synthesis and the
 * gain/spectrum buffers. A missing RES or NR reads as the unity vector. */
typedef struct ExPost {
    AecRes* res;              /* NULL: no suppressor */
    AnrDenoiser* nr;          /* NULL: no denoiser */
    AcIstft* istft;
    float *unity, *total, *extra;
    Complex* spec;
    uint32_t rng;
    int n_freqs, enable_cng;
    void* mem;
} ExPost;

/* >= 0: ex_post_init() fills the block with this byte before the stages are placed in it. */
static int ex_post_poison = -1;

/* What an application's reset does to the stages it owns: RES, denoiser, synthesis, comfort-noise
 * seed (the unit's own reset does not touch them). */
static void ex_post_reset(ExPost* q, int reset_res) {
    if (q->res && reset_res) aec_res_reset(q->res);
    if (q->nr) anr_reset(q->nr);
    ac_istft_reset(q->istft);
    q->rng = MCAB_SAMPLE_CNG_RNG_SEED;
}

static void ex_post_free(ExPost* q) {
    if (q->nr) anr_destroy(q->nr);
    free(q->mem);
    memset(q, 0, sizeof(*q));
}

static int ex_post_init(ExPost* q, int sample_rate, int fft_size, int enable_res, AecPreset preset,
                        int enable_nr, int enable_cng) {
    AecConfig rc;
    AnrConfig nc = anr_config_for_mode_grid(sample_rate, fft_size, ANR_MODE_BALANCED);
    AcStftConfig sc;
    size_t res_b = 0, nr_b, syn_b, vec, cvec;
    unsigned char* cur;
    int k;
    memset(q, 0, sizeof(*q));
    q->n_freqs = fft_size / 2 + 1;
    q->enable_cng = enable_cng;
    ac_stft_config_default(&sc, fft_size);
    if (enable_res) {
        aec_config_from_preset(&rc, preset, sample_rate);
        rc.fft_size = fft_size;
        res_b = ALIGN16(aec_res_get_mem_size(&rc));
    }
    nr_b = enable_nr ? ALIGN16(anr_get_mem_size(&nc)) : 0;
    syn_b = ALIGN16(ac_istft_get_mem_size(&sc));
    vec = ALIGN16((size_t)q->n_freqs * sizeof(float));
    cvec = ALIGN16((size_t)q->n_freqs * sizeof(Complex));
    q->mem = aligned_block(res_b + nr_b + syn_b + 3 * vec + cvec);
    if (!q->mem) return -1;
    if (ex_post_poison >= 0) memset(q->mem, ex_post_poison, res_b + nr_b + syn_b + 3 * vec + cvec);
    cur = (unsigned char*)q->mem;
    if (enable_res) { q->res = aec_res_init(cur, aec_res_get_mem_size(&rc), &rc); cur += res_b; }
    if (enable_nr) { q->nr = anr_init(cur, anr_get_mem_size(&nc), &nc); cur += nr_b; }
    q->istft = ac_istft_init(cur, ac_istft_get_mem_size(&sc), &sc); cur += syn_b;
    q->unity = (float*)cur; cur += vec;
    q->total = (float*)cur; cur += vec;
    q->extra = (float*)cur; cur += vec;
    q->spec = (Complex*)cur;
    for (k = 0; k < q->n_freqs; ++k) q->unity[k] = 1.0f;
    q->rng = MCAB_SAMPLE_CNG_RNG_SEED;
    return (enable_res && !q->res) || (enable_nr && !q->nr) || !q->istft ? -1 : 0;
}

/* The per-hop chain of the examples. 0, or -1 when a stage refuses. */
static int ex_post_hop(ExPost* q, const McabFrame* f, float* out) {
    const float* res = q->unity;
    const float* nr = q->unity;
    if (q->res) {
        AecResInput in;
        mcab_sample_res_input(f, &in);
        if (!(res = aec_res_process(q->res, &in))) return -1;
    }
    if (q->nr && !(nr = mcab_sample_anr_gain(q->nr, f, q->extra, 1))) return -1;
    sk_min_f32(q->total, nr, res, q->n_freqs);
    sk_capply_gain_f32(q->spec, f->e, q->total, q->n_freqs);
    if (q->enable_cng && q->res && f->cn)
        aec_cng_fill(&q->rng, q->spec, f->cn, res, nr, MCAB_SAMPLE_CNG_NR_GAIN_FLOOR, q->n_freqs);
    (void)ac_istft_push(q->istft, q->spec, out);
    return 0;
}

/* A broadband far end through a nonlinear echo path (so a residual stays after the linear
 * filters converge and the RES floors bind), with double-talk bursts, on four channels. A
 * function of the hop index only, so twins see the same input. */
static void res_scene_hop(int h, int hop, float* mics, float* far) {
    int i, ch;
    for (i = 0; i < hop; ++i) {
        float t = (float)(h * hop + i);
        uint32_t x = (uint32_t)(h * hop + i) * 2654435761u;
        float r, echo;
        float near = ((h / 25) % 3 == 2) ? 0.2f * sinf(0.023f * t) * sinf(0.0011f * t) : 0.0f;
        x ^= x >> 15;
        x *= 2246822519u;
        x ^= x >> 13;
        r = 0.3f * sinf(0.05f * t) + 0.2f * ((float)(x >> 8) / 16777216.0f - 0.5f);
        echo = r + 1.5f * r * r * r;
        far[i] = r;
        for (ch = 0; ch < MCAB_CHANNELS; ++ch)
            mics[(size_t)i * MCAB_CHANNELS + (size_t)ch] =
                0.4f * echo + near + 0.03f * sinf(0.017f * t + (float)ch);
    }
}

/* A scene whose echo the linear filters cannot model: the far end is broadband noise under a
 * slow envelope, the echo in the microphones is a different noise under the same envelope
 * (uncorrelated with the reference, so the filters stay near zero), plus double-talk bursts. The
 * residual-echo estimate stays high and the RES keeps cutting for the whole run instead of only
 * during the first convergence. A function of the hop index only. */
static float hard_noise(uint32_t seed) {
    seed *= 2654435761u;
    seed ^= seed >> 15;
    seed *= 2246822519u;
    seed ^= seed >> 13;
    return (float)(seed >> 8) / 16777216.0f - 0.5f;
}

static void res_scene_hard_hop(int h, int hop, float* mics, float* far) {
    float env = 0.35f + 0.65f * (0.5f + 0.5f * sinf(0.09f * (float)h));
    int i, ch;
    for (i = 0; i < hop; ++i) {
        int idx = h * hop + i;
        float t = (float)idx;
        float near = ((h / 40) % 3 == 2) ? 0.15f * sinf(0.023f * t) * sinf(0.0011f * t) : 0.0f;
        far[i] = 0.5f * env * hard_noise((uint32_t)idx);
        for (ch = 0; ch < MCAB_CHANNELS; ++ch)
            mics[(size_t)i * MCAB_CHANNELS + (size_t)ch] =
                0.5f * env * hard_noise((uint32_t)(idx * 4 + ch) + 0x51ed270bu) + near +
                0.002f * sinf(0.017f * t + (float)ch);
    }
}

/* fuse() hands the later stages the beamformed frame and leaves the residual gain to them:
 * g_res_ctx stays NULL on every hop of a busy scene, on every grid, while the inputs the
 * caller's RES reads (E, Y, R^2, comfort noise, aligned far end) are all present and finite. */
static int test_fuse_leaves_the_residual_gain_to_the_caller(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int grid;
    for (grid = 0; grid < 3; ++grid) {
        McabTestUnitConfig cfg = mcab_test_unit_default_config(grids[grid][0]);
        McabSampleBeam* p;
        float *mics, *far;
        int hop, n, h, ok = 1;
        char label[160];
        cfg.core.fft_size = grids[grid][1];
        p = mcab_test_beam_create(&cfg);
        CHECK(p != NULL, "fuse frame: the unit builds");
        hop = mcab_hop_size(mcab_test_beam_flow(p));
        n = mcab_n_freqs(mcab_test_beam_flow(p));
        mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        for (h = 0; h < 120 && ok; ++h) {
            McabSampleBeamResult beam;
            McabFrame f;
            int k;
            res_scene_hop(h, hop, mics, far);
            if (mcab_sample_beam_process(p, mics, far, &beam, NULL) != MCAB_OK ||
                mcab_sample_beam_fuse(p, &f) != MCAB_OK ||
                f.g_res_ctx != NULL || !f.e || !f.y || !f.r2 || !f.cn || !f.far_hop ||
                f.n_freqs != n || f.hop != hop) {
                ok = 0;
                break;
            }
            for (k = 0; k < n; ++k)
                if (!isfinite(f.e[k].r) || !isfinite(f.e[k].i) || !isfinite(f.y[k].r) ||
                    !isfinite(f.r2[k]) || !isfinite(f.cn[k])) ok = 0;
        }
        snprintf(label, sizeof label,
                 "fuse frame: %d Hz/%d g_res_ctx is NULL on every hop and the RES inputs are present",
                 grids[grid][0], grids[grid][1]);
        CHECK(ok, label);
        free(mics); free(far);
        mcab_test_beam_destroy(p);
    }
    return 1;
}

/* With the BF bypassed the beamformed error is lane 0's error spectrum and the effective weights
 * are [1,0,0,0]; the reference chain fed the same two must reproduce unit + fuse + the examples'
 * chain bit for bit, for every combination of the RES, NR and comfort-noise switches and, with
 * the RES on, at every AEC preset. */
static int test_unit_chain_equals_reference_chain(void) {
    int mode, grid, preset;
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    for (grid = 0; grid < 3; ++grid) {
        for (mode = 0; mode < 8; ++mode) {
         for (preset = 0; preset < 4; ++preset) {
            McabTestUnitConfig cfg = mcab_test_unit_default_config(grids[grid][0]);
            ConvChainOptions opt = conv_default_options();
            McabSampleBeam* unit;
            ExPost post;
            ConvChain* ref;
            McabSampleBeamResult beam;
            McabFrame frame;
            Complex* weights;
            float *mics, *far, *out_w, *out_r;
            int hop, n, h, i, k, identical = 1, nonzero = 0, ok = 1;
            char label[160];

            if (!((mode >> 1) & 1) && preset != (int)AEC_PRESET_BALANCED) continue;   /* no RES: no preset */
            cfg.core.fft_size = grids[grid][1];
            cfg.beam.bf_enable = 0;
            opt.aec_preset = (AecPreset)preset;
            opt.enable_nr = mode & 1;
            opt.enable_res = (mode >> 1) & 1;
            opt.enable_cng = (mode >> 2) & 1;
            unit = mcab_test_beam_create(&cfg);
            ref = conv_create(&cfg.core, &opt);
            {   /* The oracle writes the denoiser window out on its own; it must agree with the
                 * examples' denoiser, since the short scene below cannot tell the
                 * noise-tracking window apart. */
                int ghop = grids[grid][1] / 2;
                AnrConfig oc = conv_nr_config(grids[grid][0], grids[grid][1], ghop, ANR_MODE_BALANCED);
                AnrConfig ec = anr_config_for_mode_grid(grids[grid][0], grids[grid][1], ANR_MODE_BALANCED);
                CHECK(memcmp(&oc, &ec, sizeof oc) == 0,
                      "oracle and example agree on the denoiser configuration");
            }
            snprintf(label, sizeof(label), "reference chain: %d Hz/%d nr=%d res=%d(preset %d) cng=%d build",
                     grids[grid][0], grids[grid][1], mode & 1, (mode >> 1) & 1, preset, (mode >> 2) & 1);
            CHECK(unit && ref &&
                  ex_post_init(&post, grids[grid][0], grids[grid][1], (mode >> 1) & 1,
                               (AecPreset)preset, mode & 1, (mode >> 2) & 1) == 0,
                  label);
            hop = mcab_hop_size(mcab_test_beam_flow(unit));
            n = mcab_n_freqs(mcab_test_beam_flow(unit));
            mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
            far = (float*)calloc((size_t)hop, sizeof(float));
            out_w = (float*)calloc((size_t)hop, sizeof(float));
            out_r = (float*)calloc((size_t)hop, sizeof(float));
            weights = (Complex*)calloc((size_t)MCAB_CHANNELS * (size_t)n, sizeof(Complex));
            for (k = 0; k < n; ++k) weights[k].r = 1.0f;
            /* Past the denoiser's initialisation window, so its gain shapes the output. */
            for (h = 0; h < 340 && ok; ++h) {
                res_scene_hop(h, hop, mics, far);   /* nonlinear echo + double talk: the RES cuts */
                if (mcab_sample_beam_process_with_activity(unit, mics, far, 0, 0, NULL, &beam, NULL) !=
                        MCAB_OK ||
                    mcab_sample_beam_fuse(unit, &frame) != MCAB_OK ||
                    ex_post_hop(&post, &frame, out_w) != 0 ||
                    mcab_process_front(ref->core, mics, far, &ref->front) != MCAB_OK ||
                    conv_post(ref, weights, ref->front.linear_spectra[0], out_r) != MCAB_OK) {
                    ok = 0;
                    break;
                }
                if (memcmp(out_w, out_r, (size_t)hop * sizeof(float)) != 0) identical = 0;
                for (i = 0; i < hop; ++i) if (out_r[i] != 0.0f) nonzero = 1;
            }
            snprintf(label, sizeof(label),
                     "reference chain: %d Hz/%d nr=%d res=%d(preset %d) cng=%d unit + chain output is bit-identical",
                     grids[grid][0], grids[grid][1], mode & 1, (mode >> 1) & 1, preset, (mode >> 2) & 1);
            CHECK(ok && identical && nonzero, label);
            free(mics); free(far); free(out_w); free(out_r); free(weights);
            mcab_test_beam_destroy(unit);
            ex_post_free(&post);
            conv_destroy(ref);
         }
        }
    }
    return 1;
}

/* The unit with its caller-owned RES (ExPost with the RES only): one hop of the RES scene
 * through process + fuse + the AecRes the caller built; the gain, or NULL on a refusal. */
typedef struct ResRig {
    McabSampleBeam* unit;
    ExPost post;
} ResRig;

static int res_rig_init(ResRig* r, int sample_rate, int fft_size, AecPreset preset) {
    McabTestUnitConfig cfg = mcab_test_unit_default_config(sample_rate);
    cfg.core.fft_size = fft_size;
    r->unit = mcab_test_beam_create(&cfg);
    return r->unit && ex_post_init(&r->post, sample_rate, fft_size, 1, preset, 0, 0) == 0 ? 0 : -1;
}

static void res_rig_free(ResRig* r) {
    mcab_test_beam_destroy(r->unit);
    ex_post_free(&r->post);
}

/* What an application's reset does: the unit and the stages it chains. */
static void res_rig_reset(ResRig* r) {
    mcab_sample_beam_reset(r->unit);
    aec_res_reset(r->post.res);
}

static const float* res_rig_hop(ResRig* r, int h, float* mics, float* far) {
    McabSampleBeamResult beam;
    McabFrame f;
    AecResInput in;
    res_scene_hop(h, mcab_hop_size(mcab_test_beam_flow(r->unit)), mics, far);
    if (mcab_sample_beam_process(r->unit, mics, far, &beam, NULL) != MCAB_OK ||
        mcab_sample_beam_fuse(r->unit, &f) != MCAB_OK)
        return NULL;
    mcab_sample_res_input(&f, &in);
    return aec_res_process(r->post.res, &in);
}

/* The caller-owned RES across reset: a used and reset rig equals a fresh one on the gain, and a
 * preset set with aec_res_set_preset() survives the reset and then equals a rig built at that
 * preset (while the two presets give different gains on this scene). */
static int test_caller_res_reset_and_preset(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int g;
    for (g = 0; g < 3; ++g) {
        ResRig used, fresh, retarget, built;
        float *mics, *far;
        int h, n, hop, same = 1, same_preset = 1, differs = 0, ok;
        char label[160];
        ok = res_rig_init(&used, grids[g][0], grids[g][1], AEC_PRESET_BALANCED) == 0 &&
             res_rig_init(&fresh, grids[g][0], grids[g][1], AEC_PRESET_BALANCED) == 0 &&
             res_rig_init(&retarget, grids[g][0], grids[g][1], AEC_PRESET_BALANCED) == 0 &&
             res_rig_init(&built, grids[g][0], grids[g][1], AEC_PRESET_AGGRESSIVE) == 0;
        CHECK(ok, "caller res: rigs build");
        hop = mcab_hop_size(mcab_test_beam_flow(used.unit));
        n = mcab_n_freqs(mcab_test_beam_flow(used.unit));
        mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        for (h = 0; h < 150; ++h) {
            (void)res_rig_hop(&used, h + 1000, mics, far);
            (void)res_rig_hop(&retarget, h + 1000, mics, far);
        }
        res_rig_reset(&used);
        CHECK(aec_res_set_preset(retarget.post.res, AEC_PRESET_AGGRESSIVE, 0.0f) == 0,
              "caller res: a preset is accepted");
        res_rig_reset(&retarget);
        for (h = 0; h < 250; ++h) {
            static float keep_u[513], keep_r[513];
            const float *gu = res_rig_hop(&used, h, mics, far), *gf, *gr, *gb;
            if (!gu) { same = 0; break; }
            memcpy(keep_u, gu, (size_t)n * sizeof(float));
            gf = res_rig_hop(&fresh, h, mics, far);
            gr = res_rig_hop(&retarget, h, mics, far);
            if (!gr) { same_preset = 0; break; }
            memcpy(keep_r, gr, (size_t)n * sizeof(float));
            gb = res_rig_hop(&built, h, mics, far);
            if (!gf || memcmp(keep_u, gf, (size_t)n * sizeof(float)) != 0) same = 0;
            if (!gb || memcmp(keep_r, gb, (size_t)n * sizeof(float)) != 0) same_preset = 0;
            if (memcmp(keep_u, keep_r, (size_t)n * sizeof(float)) != 0) differs = 1;
        }
        snprintf(label, sizeof label,
                 "caller res: %d Hz/%d a used rig after reset gives the gain of a fresh rig bit for bit",
                 grids[g][0], grids[g][1]);
        CHECK(same, label);
        snprintf(label, sizeof label,
                 "caller res: %d Hz/%d a preset set before reset survives it and equals a rig built "
                 "at that preset", grids[g][0], grids[g][1]);
        CHECK(same_preset, label);
        snprintf(label, sizeof label, "caller res: %d Hz/%d balanced and aggressive differ on this scene",
                 grids[g][0], grids[g][1]);
        CHECK(differs, label);
        free(mics); free(far);
        res_rig_free(&used); res_rig_free(&fresh); res_rig_free(&retarget); res_rig_free(&built);
    }
    return 1;
}

/* A ramped preset on the caller-owned RES is genuinely in flight, lands, and changes the gain
 * against an untouched twin; bad presets and ramps are refused. */
static int test_caller_res_ramped_preset(void) {
    ResRig p, q;
    float *mics, *far, keep[513];
    float live0 = 0.0f, target0 = 0.0f, lv = 0.0f, tg = 0.0f;
    int hop, n, h, same_before = 1, differs_after = 0, ramp_seen = 0, landed = 0;
    CHECK(res_rig_init(&p, 16000, 256, AEC_PRESET_BALANCED) == 0 &&
              res_rig_init(&q, 16000, 256, AEC_PRESET_BALANCED) == 0,
          "ramped preset: rigs build");
    hop = mcab_hop_size(mcab_test_beam_flow(p.unit));
    n = mcab_n_freqs(mcab_test_beam_flow(p.unit));
    mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    for (h = 0; h < 240; ++h) {
        const float* gp = res_rig_hop(&p, h, mics, far);
        const float* gq;
        if (!gp) { same_before = 0; break; }
        memcpy(keep, gp, (size_t)n * sizeof(float));
        gq = res_rig_hop(&q, h, mics, far);
        if (!gq || memcmp(keep, gq, (size_t)n * sizeof(float)) != 0) same_before = 0;
    }
    CHECK(aec_res_split_floor(p.post.res, &live0, &target0) == 0 &&
              aec_res_set_preset(p.post.res, AEC_PRESET_MILD, 100.0f) == 0 &&
              aec_res_set_preset(p.post.res, (AecPreset)77, 0.0f) == -1 &&
              aec_res_set_preset(p.post.res, AEC_PRESET_MILD, 60001.0f) == -1,
          "ramped preset: a ramped preset is accepted, bad ones are refused");
    for (h = 240; h < 640; ++h) {
        const float* gp = res_rig_hop(&p, h, mics, far);
        const float* gq;
        if (!gp) break;
        memcpy(keep, gp, (size_t)n * sizeof(float));
        gq = res_rig_hop(&q, h, mics, far);
        if (!gq || memcmp(keep, gq, (size_t)n * sizeof(float)) != 0) differs_after = 1;
        aec_res_split_floor(p.post.res, &lv, &tg);
        if (h == 241 && lv != tg) ramp_seen = 1;
        if (h == 639 && lv == tg && tg != target0) landed = 1;
    }
    CHECK(same_before, "ramped preset: untouched twins agree before the change");
    CHECK(differs_after, "ramped preset: the new preset changes the gain");
    CHECK(ramp_seen && landed, "ramped preset: the floor ramps, then lands on the new target");
    free(mics); free(far);
    res_rig_free(&p); res_rig_free(&q);
    return 1;
}

/* ---------------------------------------------------------------------------
 * The whole chain of an example (unit, RES, NR, comfort noise, synthesis) as one rig, with the
 * unit built either on the heap (create) or in a caller pool (init_ex).
 * ------------------------------------------------------------------------ */

typedef struct ChainRig {
    McabSampleBeam* unit;
    void* unit_pool;          /* non-NULL: flow + helper placed in this poisoned pool */
    McabSampleFlow objs;
    ExPost post;
} ChainRig;

static int chain_rig_init(ChainRig* r, int sample_rate, int fft_size, int in_pool) {
    McabTestUnitConfig cfg = mcab_test_unit_default_config(sample_rate);
    size_t flow_bytes, beam_bytes;
    memset(r, 0, sizeof(*r));
    cfg.core.fft_size = fft_size;
    if (in_pool) {
        if (mcab_sample_flow_get_mem_size(&cfg.core, &flow_bytes) != 0 ||
            mcab_sample_beam_get_mem_size(&cfg.core, &cfg.beam, &beam_bytes) != 0)
            return -1;
        flow_bytes = ALIGN16(flow_bytes);
        if (posix_memalign(&r->unit_pool, 16u, flow_bytes + beam_bytes) != 0) return -1;
        memset(r->unit_pool, 0xa5, flow_bytes + beam_bytes);
        if (mcab_sample_flow_init(&r->objs, r->unit_pool, flow_bytes, &cfg.core) != 0) return -1;
        r->unit = mcab_sample_beam_init((unsigned char*)r->unit_pool + flow_bytes, beam_bytes,
                                        &cfg.core, &cfg.beam, r->objs.flow);
    } else {
        r->unit = mcab_test_beam_create(&cfg);
    }
    ex_post_poison = in_pool ? 0xa5 : -1;
    {
        int rc = ex_post_init(&r->post, sample_rate, fft_size, 1, AEC_PRESET_BALANCED, 1, 1);
        ex_post_poison = -1;
        if (rc != 0) return -1;
    }
    return r->unit ? 0 : -1;
}

static void chain_rig_free(ChainRig* r) {
    if (r->unit_pool) {
        mcab_sample_beam_destroy(r->unit);
        mcab_sample_flow_destroy(&r->objs);
    } else {
        mcab_test_beam_destroy(r->unit);
    }
    free(r->unit_pool);
    ex_post_free(&r->post);
}

static void chain_rig_reset(ChainRig* r, int reset_res) {
    mcab_sample_beam_reset(r->unit);
    ex_post_reset(&r->post, reset_res);
}

/* 0 on success; `out` is the hop's output. */
static int chain_rig_hop(ChainRig* r, int h, float* mics, float* far, float* out) {
    McabSampleBeamResult beam;
    McabFrame f;
    res_scene_hop(h, r->unit_pool ? mcab_hop_size(r->objs.flow)
                                  : mcab_hop_size(mcab_test_beam_flow(r->unit)), mics, far);
    if (mcab_sample_beam_process(r->unit, mics, far, &beam, NULL) != MCAB_OK ||
        mcab_sample_beam_fuse(r->unit, &f) != MCAB_OK)
        return -1;
    return ex_post_hop(&r->post, &f, out);
}

/* After mcab_sample_beam_reset() plus the application's own resets (aec_res_reset, denoiser,
 * synthesis, comfort-noise seed) the whole chain equals a fresh instance bit for bit; the same
 * with the AecRes left unreset must differ, so the test would catch an application (or an
 * example) that forgets it. */
static int test_chain_reset_equals_fresh(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int g;
    for (g = 0; g < 3; ++g) {
        ChainRig used, fresh, forgot;
        float *mics, *far, *oa, *ob, *oc;
        int hop, h, same = 1, forgot_differs = 0, nonzero = 0, ok;
        char label[160];
        ok = chain_rig_init(&used, grids[g][0], grids[g][1], 0) == 0 &&
             chain_rig_init(&fresh, grids[g][0], grids[g][1], 0) == 0 &&
             chain_rig_init(&forgot, grids[g][0], grids[g][1], 0) == 0;
        CHECK(ok, "chain reset: rigs build");
        hop = mcab_hop_size(mcab_test_beam_flow(used.unit));
        mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        oa = (float*)calloc((size_t)hop, sizeof(float));
        ob = (float*)calloc((size_t)hop, sizeof(float));
        oc = (float*)calloc((size_t)hop, sizeof(float));
        for (h = 0; h < 200; ++h) {
            (void)chain_rig_hop(&used, h + 1000, mics, far, oa);
            (void)chain_rig_hop(&forgot, h + 1000, mics, far, oc);
        }
        chain_rig_reset(&used, 1);
        chain_rig_reset(&forgot, 0);
        for (h = 0; h < 300; ++h) {
            int i;
            if (chain_rig_hop(&used, h, mics, far, oa) != 0 ||
                chain_rig_hop(&fresh, h, mics, far, ob) != 0 ||
                chain_rig_hop(&forgot, h, mics, far, oc) != 0) { same = 0; break; }
            if (memcmp(oa, ob, (size_t)hop * sizeof(float)) != 0) same = 0;
            if (memcmp(oc, ob, (size_t)hop * sizeof(float)) != 0) forgot_differs = 1;
            for (i = 0; i < hop; ++i) if (ob[i] != 0.0f) nonzero = 1;
        }
        snprintf(label, sizeof label,
                 "chain reset: %d Hz/%d reset-then-rerun equals a fresh chain bit for bit",
                 grids[g][0], grids[g][1]);
        CHECK(same && nonzero, label);
        snprintf(label, sizeof label,
                 "chain reset: %d Hz/%d without aec_res_reset the chain does NOT equal a fresh one",
                 grids[g][0], grids[g][1]);
        CHECK(forgot_differs, label);
        free(mics); free(far); free(oa); free(ob); free(oc);
        chain_rig_free(&used); chain_rig_free(&fresh); chain_rig_free(&forgot);
    }
    return 1;
}

/* The heap create() unit and the caller-pool init_ex() unit (0xa5-poisoned pool, poisoned stage
 * block) behind the same example chain give the same output bit for bit. */
static int test_chain_heap_equals_pool(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int g;
    for (g = 0; g < 3; ++g) {
        ChainRig heap, pool;
        float *mics, *far, *oa, *ob;
        int hop, h, same = 1, nonzero = 0, i;
        char label[160];
        CHECK(chain_rig_init(&heap, grids[g][0], grids[g][1], 0) == 0 &&
                  chain_rig_init(&pool, grids[g][0], grids[g][1], 1) == 0,
              "chain heap/pool: rigs build");
        hop = mcab_hop_size(mcab_test_beam_flow(heap.unit));
        mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        oa = (float*)calloc((size_t)hop, sizeof(float));
        ob = (float*)calloc((size_t)hop, sizeof(float));
        for (h = 0; h < 300; ++h) {
            if (chain_rig_hop(&heap, h, mics, far, oa) != 0 ||
                chain_rig_hop(&pool, h, mics, far, ob) != 0) { same = 0; break; }
            if (memcmp(oa, ob, (size_t)hop * sizeof(float)) != 0) same = 0;
            for (i = 0; i < hop; ++i) if (oa[i] != 0.0f) nonzero = 1;
        }
        snprintf(label, sizeof label,
                 "chain heap/pool: %d Hz/%d create() and a poisoned init_ex() pool agree bit for bit",
                 grids[g][0], grids[g][1]);
        CHECK(same && nonzero, label);
        free(mics); free(far); free(oa); free(ob);
        chain_rig_free(&heap); chain_rig_free(&pool);
    }
    return 1;
}

/* aec_res_set_preset(res, preset, ramp_ms) on the application-owned AecRes in an echo scene where
 * the RES acts (an echo the linear filters cannot model + double talk, many bins cut, denoiser off so the RES is not
 * masked): the chain output follows an independent oracle (conv_chain_mcab.h: its own front end
 * and AecRes, the same call at the same hop) bit for bit; the run without the call differs from
 * both, so a skipped call is caught. */
static int test_caller_res_preset_ramp_vs_oracle(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    enum { SET_AT = 200, HOPS = 520 };
    int g;
    for (g = 0; g < 3; ++g) {
        McabTestUnitConfig cfg = mcab_test_unit_default_config(grids[g][0]);
        ConvChainOptions opt = conv_default_options();
        McabSampleBeam *u_set, *u_skip, *u_ref;
        ExPost p_set, p_skip;
        ConvChain* ref;
        Complex* weights;
        float *mics, *far, *o_set, *o_skip, *o_ref;
        long cut_before = 0, cut_after = 0;   /* (hop, bin) pairs the RES cut below 0.9 */
        int hop, n, h, k, i, ok = 1, set_matches = 1, skip_differs = 0, nonzero = 0;
        char label[200];
        cfg.core.fft_size = grids[g][1];
        cfg.beam.bf_enable = 0;
        opt.enable_nr = 0;
        opt.enable_cng = 1;
        u_set = mcab_test_beam_create(&cfg);
        u_skip = mcab_test_beam_create(&cfg);
        u_ref = mcab_test_beam_create(&cfg);
        ref = conv_create(&cfg.core, &opt);
        CHECK(u_set && u_skip && u_ref && ref &&
                  ex_post_init(&p_set, grids[g][0], grids[g][1], 1, AEC_PRESET_BALANCED, 0, 1) == 0 &&
                  ex_post_init(&p_skip, grids[g][0], grids[g][1], 1, AEC_PRESET_BALANCED, 0, 1) == 0,
              "preset ramp: units, oracle and stages build");
        hop = mcab_hop_size(mcab_test_beam_flow(u_set));
        n = mcab_n_freqs(mcab_test_beam_flow(u_set));
        mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        o_set = (float*)calloc((size_t)hop, sizeof(float));
        o_skip = (float*)calloc((size_t)hop, sizeof(float));
        o_ref = (float*)calloc((size_t)hop, sizeof(float));
        weights = (Complex*)calloc((size_t)MCAB_CHANNELS * (size_t)n, sizeof(Complex));
        for (k = 0; k < n; ++k) weights[k].r = 1.0f;
        for (h = 0; h < HOPS && ok; ++h) {
            McabSampleBeamResult beam;
            McabFrame fs, fk;
            int cut = 0;
            if (h == SET_AT) {
                if (aec_res_set_preset(p_set.res, AEC_PRESET_AGGRESSIVE, 100.0f) != 0 ||
                    aec_res_set_preset(ref->res, AEC_PRESET_AGGRESSIVE, 100.0f) != 0) { ok = 0; break; }
            }
            res_scene_hard_hop(h, hop, mics, far);
            if (mcab_sample_beam_process(u_set, mics, far, &beam, NULL) != MCAB_OK ||
                mcab_sample_beam_fuse(u_set, &fs) != MCAB_OK ||
                ex_post_hop(&p_set, &fs, o_set) != 0 ||
                mcab_sample_beam_process(u_skip, mics, far, &beam, NULL) != MCAB_OK ||
                mcab_sample_beam_fuse(u_skip, &fk) != MCAB_OK ||
                ex_post_hop(&p_skip, &fk, o_skip) != 0 ||
                mcab_sample_beam_process(u_ref, mics, far, &beam, NULL) != MCAB_OK ||
                mcab_process_front(ref->core, mics, far, &ref->front) != MCAB_OK ||
                conv_post(ref, weights, ref->front.linear_spectra[0], o_ref) != MCAB_OK) {
                ok = 0;
                break;
            }
            /* denoiser off: total = min(1, G_res) = G_res */
            for (k = 0; k < n; ++k) if (p_set.total[k] < 0.9f) ++cut;
            if (h >= 100 && h < SET_AT) cut_before += cut;   /* after the initial-state phase */
            if (h >= SET_AT) cut_after += cut;
            if (memcmp(o_set, o_ref, (size_t)hop * sizeof(float)) != 0) set_matches = 0;
            if (h >= SET_AT && memcmp(o_skip, o_ref, (size_t)hop * sizeof(float)) != 0) skip_differs = 1;
            if (h < SET_AT && memcmp(o_skip, o_set, (size_t)hop * sizeof(float)) != 0) set_matches = 0;
            for (i = 0; i < hop; ++i) if (o_ref[i] != 0.0f) nonzero = 1;
        }
        snprintf(label, sizeof label,
                 "preset ramp: %d Hz/%d the RES acts in the scene beyond its initial state (gain "
                 "below 0.9 on many bins, before and after the change)", grids[g][0], grids[g][1]);
        CHECK(ok && cut_before > 2L * n && cut_after > 10L * n, label);
        snprintf(label, sizeof label,
                 "preset ramp: %d Hz/%d output with the ramped preset equals the oracle bit for "
                 "bit on every hop", grids[g][0], grids[g][1]);
        CHECK(set_matches && nonzero, label);
        snprintf(label, sizeof label,
                 "preset ramp: %d Hz/%d the run with the preset call skipped differs from the "
                 "oracle", grids[g][0], grids[g][1]);
        CHECK(skip_differs, label);
        free(mics); free(far); free(o_set); free(o_skip); free(o_ref); free(weights);
        mcab_test_beam_destroy(u_set); mcab_test_beam_destroy(u_skip);
        mcab_test_beam_destroy(u_ref);
        ex_post_free(&p_set); ex_post_free(&p_skip);
        conv_destroy(ref);
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * mcab_raw as a black box: its output file equals the reference chain (own front
 * end, AecRes, denoiser, comfort noise, synthesis) fed the beam of the unit on the same input,
 * bit for bit.
 * ------------------------------------------------------------------------ */

static int raw_case(const char* bin, const char* tmp, int sample_rate, int fft_size, int fixed_doa) {
    char mic_path[512], ref_path[512], out_path[512], cmd[2048];
    McabTestUnitConfig cfg = mcab_test_unit_default_config(sample_rate);
    ConvChainOptions opt = conv_default_options();
    McabSampleBeam* unit;
    ConvChain* ref;
    FILE *fm, *fr, *fo;
    float *mics, *far, *want, *got;
    int hop, h, hops = 300, ok = 1, nonzero = 0;
    snprintf(mic_path, sizeof mic_path, "%s/raw_mic.f32", tmp);
    snprintf(ref_path, sizeof ref_path, "%s/raw_ref.f32", tmp);
    snprintf(out_path, sizeof out_path, "%s/raw_out.f32", tmp);
    cfg.core.fft_size = fft_size;
    if (fixed_doa) {
        cfg.beam.bf_fixed_mode = 1;
        cfg.beam.bf_fixed_doa_rad = 45.0f * (float)(3.14159265358979323846 / 180.0);   /* as the CLI */
    }
    unit = mcab_test_beam_create(&cfg);
    ref = conv_create(&cfg.core, &opt);
    if (!unit || !ref) return -1;
    hop = mcab_hop_size(mcab_test_beam_flow(unit));
    mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    want = (float*)calloc((size_t)hop, sizeof(float));
    got = (float*)calloc((size_t)hop, sizeof(float));
    fm = fopen(mic_path, "wb");
    fr = fopen(ref_path, "wb");
    if (!fm || !fr) return -1;
    for (h = 0; h < hops; ++h) {
        res_scene_hard_hop(h, hop, mics, far);
        fwrite(mics, sizeof(float), (size_t)hop * MCAB_CHANNELS, fm);
        fwrite(far, sizeof(float), (size_t)hop, fr);
    }
    fclose(fm);
    fclose(fr);
    snprintf(cmd, sizeof cmd,
             "'%s' --mic-raw '%s' --ref-raw '%s' --output-raw '%s' --sample-rate %d "
             "--fft-size %d%s > '%s/raw_stdout.txt' 2>&1",
             bin, mic_path, ref_path, out_path, sample_rate, fft_size,
             fixed_doa ? " --fixed-doa-deg 45" : "", tmp);
    if (system(cmd) != 0) return -1;
    fo = fopen(out_path, "rb");
    if (!fo) return -1;
    for (h = 0; h < hops && ok; ++h) {
        McabSampleBeamResult beam;
        int i;
        res_scene_hard_hop(h, hop, mics, far);
        if (fread(got, sizeof(float), (size_t)hop, fo) != (size_t)hop ||
            mcab_sample_beam_process(unit, mics, far, &beam, NULL) != MCAB_OK ||
            conv_hop(ref, mics, far, beam.weights, beam.spectrum, want) != MCAB_OK) {
            ok = 0;
            break;
        }
        if (memcmp(want, got, (size_t)hop * sizeof(float)) != 0) ok = 0;
        for (i = 0; i < hop; ++i) if (want[i] != 0.0f) nonzero = 1;
    }
    if (ok && fread(got, sizeof(float), 1, fo) != 0) ok = 0;   /* nothing beyond the hops */
    fclose(fo);
    free(mics); free(far); free(want); free(got);
    mcab_test_beam_destroy(unit);
    conv_destroy(ref);
    return ok && nonzero ? 1 : 0;
}

static int test_raw_runner_black_box(const char* bin_dir, const char* tmp) {
    static const int cases[4][3] = {{16000, 256, 0}, {16000, 512, 0}, {48000, 1024, 0}, {16000, 256, 1}};
    char bin[512], label[160];
    int c;
    snprintf(bin, sizeof bin, "%s/mcab_raw", bin_dir);
    for (c = 0; c < 4; ++c) {
        snprintf(label, sizeof label,
                 "black box: mcab_raw %d Hz/%d%s equals the reference chain bit for bit",
                 cases[c][0], cases[c][1], cases[c][2] ? " fixed DOA" : "");
        CHECK(raw_case(bin, tmp, cases[c][0], cases[c][1], cases[c][2]) == 1, label);
    }
    return 1;
}

/* ============================================================================
 * Unit contract: reset, core view, the front-only profile
 * ========================================================================== */

static uint32_t g_rng = 0x1234567u;
static float frand(void) {
    uint32_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng = x;
    return ((float)(x >> 8) * (1.0f / 16777216.0f)) - 0.5f;
}

/* One hop of the shared synthetic scene every delay-lock test drives: fresh
 * far noise appended to far_hist, and a true_delay-delayed 0.6x echo fanned
 * out to the four mics with a per-channel taper. */
static void fill_echo_hop(int frame, int hop, int true_delay,
                          float* far_hist, float* far, float* microphones) {
    for (int i = 0; i < hop; ++i) {
        int64_t t = (int64_t)frame * hop + i;
        float noise = 0.25f * frand();
        float echo;
        far_hist[t] = noise;
        far[i] = noise;
        echo = t >= true_delay ? 0.6f * far_hist[t - true_delay] : 0.0f;
        for (int ch = 0; ch < MCAB_CHANNELS; ++ch) {
            microphones[i * MCAB_CHANNELS + ch] =
                echo * (1.0f - 0.02f * ch);
        }
    }
}

/* The echo scene plus a directional near source: one noise stream reaches microphone ch
 * ch*2 samples later than microphone 0, so SRP has a bearing to find. */
static void fill_split_hop(int frame, int hop, float* far_hist,
                           float* near_hist, float* far, float* microphones) {
    enum { TRUE_DELAY = 400 };
    fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
    for (int i = 0; i < hop; ++i) {
        int64_t t = (int64_t)frame * hop + i;
        near_hist[t] = 0.2f * frand();
        for (int ch = 0; ch < MCAB_CHANNELS; ++ch) {
            int64_t u = t - 2 * ch;
            microphones[i * MCAB_CHANNELS + ch] +=
                u >= 0 ? near_hist[u] : 0.0f;
        }
    }
}

typedef struct SplitScene {
    int hop, frames;
    float *far_hist, *near_hist, *mics, *far;
} SplitScene;

static int scene_open(SplitScene* s, int hop, int frames) {
    s->hop = hop;
    s->frames = frames;
    s->far_hist = (float*)calloc((size_t)frames * hop, sizeof(float));
    s->near_hist = (float*)calloc((size_t)frames * hop, sizeof(float));
    s->mics = (float*)malloc((size_t)hop * MCAB_CHANNELS * sizeof(float));
    s->far = (float*)malloc((size_t)hop * sizeof(float));
    return s->far_hist && s->near_hist && s->mics && s->far;
}

static void scene_close(SplitScene* s) {
    free(s->far_hist); free(s->near_hist); free(s->mics); free(s->far);
}

static int bits_equal_f(float a, float b) {
    return memcmp(&a, &b, sizeof(float)) == 0;
}

/* The state words compare bitwise: NaN is a legal value in them. */
static int same_info(const McabSampleBeamInfo* a, const McabSampleBeamInfo* b) {
    return a->frame_index == b->frame_index &&
           a->delay.delay_samples == b->delay.delay_samples &&
           a->delay.solid == b->delay.solid && a->delay.changed == b->delay.changed &&
           a->delay.estimator_calls == b->delay.estimator_calls &&
           bits_equal_f(a->delay.confidence, b->delay.confidence) &&
           bits_equal_f(a->doa_raw_rad, b->doa_raw_rad) &&
           bits_equal_f(a->doa_smooth_rad, b->doa_smooth_rad) &&
           bits_equal_f(a->doa_used_rad, b->doa_used_rad) &&
           a->vad_raw == b->vad_raw && a->vad_out == b->vad_out &&
           a->bf_adaptive == b->bf_adaptive &&
           a->doa_analysis_frames == b->doa_analysis_frames;
}

static McabTestUnitConfig split_config(void) {
    McabTestUnitConfig cfg = mcab_test_unit_default_config(16000);
    cfg.core.fft_size = 512;
    cfg.core.enable_post = 0;
    cfg.beam.bf_mu = 0.02f;
    cfg.beam.doa_update_interval = 1;
    return cfg;
}

/* Front and beam as two calls. Returns 1 when both succeeded. */
static int split_hop(McabSampleBeam* p, const float* mics, const float* far,
                     int vad_raw, int vad_out, const int* mask,
                     McabSampleBeamResult* beam, McabSampleBeamInfo* info) {
    McabFront front;
    if (mcab_sample_beam_process_front(p, mics, far, &front) != MCAB_OK) return 0;
    return mcab_sample_beam_process_beam(p, &front, vad_raw, vad_out, mask, beam, info) ==
           MCAB_OK;
}

/* The front view's aligned reference is the delayed far, whole-hop exact, on every
 * hop; the delay state it carries agrees with the unit's frame info. */
static int test_core_aligned_ref(void) {
    enum { FRAMES = 240, TRUE_DELAY = 400 };
    McabTestUnitConfig cfg = mcab_test_unit_default_config(16000);
    McabSampleBeam* p;
    McabSampleBeamResult beam;
    McabSampleBeamInfo info;
    McabFront pre;
    float* microphones;
    float* far;
    float* far_hist;
    int hop;
    int locked = 0;
    int aligned_checked = 0;

    cfg.core.fft_size = 512;
    p = mcab_test_beam_create(&cfg);
    CHECK(p != NULL, "create 512/256 unit");
    hop = mcab_hop_size(mcab_test_beam_flow(p));
    CHECK(hop == 256, "unit hop is 256");

    microphones = (float*)malloc(
        (size_t)hop * MCAB_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && far_hist, "allocate core-test buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        /* Zero the whole out-struct first: if process_front ever stopped
         * assigning aligned_ref, the non-NULL check below would fail
         * instead of reading leftover stack garbage. */
        memset(&pre, 0, sizeof(pre));
        CHECK(mcab_sample_beam_process_front(
                  p, microphones, far, &pre) == MCAB_OK,
              "unit front processes");
        CHECK(pre.aligned_ref != NULL,
              "pre.aligned_ref is assigned (non-NULL)");

        /* Byte-exact aligned-far check against our own far history: the
         * ring path is a pure copy, so equality is exact, on every frame
         * and for whatever delay the estimator currently applies. The
         * shift is a WHOLE-hop decision -- until the samples already seen
         * cover the applied delay the seam serves the RAW far hop -- so the
         * expectation is chosen per hop, never per sample. */
        CHECK(pre.delay.delay_samples >= 0, "applied delay is non-negative");
        {
            int64_t hop_start = (int64_t)frame * hop;
            int servable = hop_start >= (int64_t)pre.delay.delay_samples;
            for (int i = 0; i < hop; ++i) {
                int64_t t = hop_start + i;
                float expected = servable
                    ? far_hist[t - pre.delay.delay_samples]
                    : far_hist[t];
                if (pre.aligned_ref[i] != expected) {
                    CHECK(0, "aligned_ref content matches the delayed far");
                }
                aligned_checked += 1;
            }
        }

        CHECK(mcab_sample_beam_process_beam(
                  p, &pre, 0, 0, NULL, &beam, &info) == MCAB_OK &&
              info.delay.delay_samples == pre.delay.delay_samples &&
              info.delay.solid == pre.delay.solid &&
              info.delay.changed == pre.delay.changed,
              "frame info carries the front's delay state");
        if (pre.delay.solid && pre.delay.delay_samples > 0) locked = 1;
    }

    CHECK(locked,
          "delayed far eventually locks with positive delay_samples");
    CHECK(aligned_checked == FRAMES * hop,
          "aligned_ref was checked on every sample");

    free(far_hist);
    free(far);
    free(microphones);
    mcab_test_beam_destroy(p);
    return 1;
}

/* process_with_activity() against front + beam(v, v, NULL): a null mask and a single flag
 * reproduce it. */
static int test_split_null_mask_equals_begin(void) {
    McabTestUnitConfig cfg = split_config();
    McabSampleBeam *a, *b;
    SplitScene s;
    int hop, nf, ok = 1, nonzero = 0;
    const int frames = 90;

    a = mcab_test_beam_create(&cfg);
    b = mcab_test_beam_create(&cfg);
    CHECK(a && b, "split twin: instances create");
    hop = mcab_hop_size(mcab_test_beam_flow(a));
    nf = mcab_n_freqs(mcab_test_beam_flow(a));
    CHECK(scene_open(&s, hop, frames), "split twin: scene allocates");
    for (int f = 0; f < frames && ok; ++f) {
        McabSampleBeamResult ba, bb;
        McabSampleBeamInfo ia, ib;
        int vad = (f / 7) % 2;
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        ok = mcab_sample_beam_process_with_activity(a, s.mics, s.far, vad, vad, NULL, &ba, &ia) ==
                 MCAB_OK &&
             split_hop(b, s.mics, s.far, vad, vad, NULL, &bb, &ib) &&
             memcmp(ba.spectrum, bb.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
             memcmp(ba.weights, bb.weights,
                    (size_t)MCAB_CHANNELS * (size_t)nf * sizeof(Complex)) == 0 &&
             same_info(&ia, &ib);
        for (int k = 0; k < nf; ++k) if (ba.spectrum[k].r != 0.0f) nonzero = 1;
    }
    CHECK(ok, "split twin: front + beam(v, v, NULL) matches process_with_activity byte for byte, state included");
    CHECK(nonzero, "split twin: the compared output is not silence");
    scene_close(&s);
    mcab_test_beam_destroy(a);
    mcab_test_beam_destroy(b);
    return 1;
}

/* A mask and a separate vad_raw / vad_out change what SRP and BF see. */
static int test_split_mask_and_vad_reach_srp_bf(void) {
    McabTestUnitConfig cfg = split_config();
    enum { NP = 5 };   /* NULL, all-zero, all-one, even bins, NULL with vad_raw 0 */
    McabSampleBeam* p[NP];
    int* mask[NP] = {0};
    SplitScene s;
    Complex* out[NP];
    int hop, nf, differs_doa[NP] = {0}, differs_out[NP] = {0};
    int raw_finite[NP] = {0}, adaptive_seen[NP] = {0};
    const int frames = 80;

    for (int i = 0; i < NP; ++i) {
        p[i] = mcab_test_beam_create(&cfg);
        CHECK(p[i] != NULL, "split mask: instances create");
    }
    hop = mcab_hop_size(mcab_test_beam_flow(p[0]));
    nf = mcab_n_freqs(mcab_test_beam_flow(p[0]));
    CHECK(scene_open(&s, hop, frames), "split mask: scene allocates");
    for (int i = 1; i < 4; ++i) {
        mask[i] = (int*)malloc((size_t)nf * sizeof(int));
        for (int k = 0; k < nf; ++k)
            mask[i][k] = i == 1 ? 0 : (i == 2 ? 1 : (k % 2 == 0));
    }
    for (int i = 0; i < NP; ++i) out[i] = (Complex*)malloc((size_t)nf * sizeof(Complex));
    for (int f = 0; f < frames; ++f) {
        McabSampleBeamInfo st[NP];
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        for (int i = 0; i < NP; ++i) {
            McabSampleBeamResult beam;
            /* Speech (vad_out 1) lets the DOA smoother settle; the quiet
             * stretches (vad_out 0) are where BF adapts. */
            int vad_raw = i == 4 ? 0 : 1;
            int vad_out = (f % 20) < 10;
            CHECK(split_hop(p[i], s.mics, s.far, vad_raw, vad_out, mask[i], &beam, &st[i]),
                  "split mask: hop runs");
            memcpy(out[i], beam.spectrum, (size_t)nf * sizeof(Complex));
            if (isfinite(st[i].doa_raw_rad)) raw_finite[i] = 1;
            if (st[i].bf_adaptive) adaptive_seen[i] = 1;
        }
        for (int i = 1; i < NP; ++i) {
            if (!bits_equal_f(st[0].doa_raw_rad, st[i].doa_raw_rad) ||
                !bits_equal_f(st[0].doa_smooth_rad, st[i].doa_smooth_rad) ||
                !bits_equal_f(st[0].doa_used_rad, st[i].doa_used_rad) ||
                st[0].bf_adaptive != st[i].bf_adaptive) differs_doa[i] = 1;
            if (memcmp(out[0], out[i], (size_t)nf * sizeof(Complex)) != 0) differs_out[i] = 1;
        }
    }
    CHECK(raw_finite[0], "split mask: with vad_raw 1 SRP produces a finite raw DOA");
    CHECK(adaptive_seen[0], "split mask: BF adapts on some quiet hop once a DOA exists");
    CHECK(differs_doa[1], "split mask: an all-zero mask changes the SRP result (bins are read)");
    CHECK(differs_doa[3], "split mask: a half-bin mask changes the SRP result");
    CHECK(differs_out[2], "split mask: an all-one mask freezes BF adaptation, the output changes");
    CHECK(differs_out[3], "split mask: a half-bin mask changes the beamformed output");
    CHECK(!raw_finite[4], "split vad: vad_raw 0 never runs SRP");
    CHECK(differs_doa[4], "split vad: vad_raw reaches SRP independently of vad_out");
    for (int i = 0; i < NP; ++i) {
        mcab_test_beam_destroy(p[i]);
        free(out[i]);
        free(mask[i]);
    }
    scene_close(&s);
    return 1;
}

/* The frame info against SRP and BF objects built here from the same derivation and fed the
 * same lane spectra: every field, the beamformed spectrum and the weights. */
static int split_state_case(int fixed_bf) {
    McabTestUnitConfig cfg = split_config();
    McabSampleBeam* p;
    cfg.beam.bf_fixed_mode = fixed_bf;
    cfg.beam.bf_fixed_doa_rad = 0.3f;
    p = mcab_test_beam_create(&cfg);
    SRP_Config sc;
    BF_Config gc;
    float gx[MCAB_CHANNELS], gy[MCAB_CHANNELS];
    ArrayGeometry geom = { MCAB_CHANNELS, gx, gy };
    SRP* srp;
    BF* bf;
    Complex *spec, *weights;
    SplitScene s;
    int *mask;
    int hop, nf, interval, finite_smooth = 0, finite_used = 0;
    int fixed_used_seen = 0;
    const int frames = 70;

    CHECK(p != NULL, "split state: instance creates");
    hop = mcab_hop_size(mcab_test_beam_flow(p));
    nf = mcab_n_freqs(mcab_test_beam_flow(p));
    for (int m = 0; m < MCAB_CHANNELS; ++m) {
        float phi = 2.0f * M_PI * m / MCAB_CHANNELS;
        gx[m] = cfg.beam.uca_radius_m * cosf(phi);
        gy[m] = cfg.beam.uca_radius_m * sinf(phi);
    }
    memset(&sc, 0, sizeof(sc));
    sc.M = MCAB_CHANNELS;
    sc.F = nf;
    sc.num_angles = cfg.beam.num_angles;
    sc.sr = (float)cfg.core.sample_rate;
    sc.NFFT = (float)mcab_fft_size(mcab_test_beam_flow(p));
    sc.c = cfg.beam.speed_of_sound_m_s;
    sc.low_freq = cfg.beam.doa_low_freq_hz;
    sc.high_freq = cfg.beam.doa_high_freq_hz;
    sc.enable_smoothing = cfg.beam.doa_enable_smoothing;
    sc.switch_consec = cfg.beam.doa_switch_consecutive;
    sc.angle_tol = cfg.beam.doa_angle_tolerance_rad;
    sc.update_interval = cfg.beam.doa_update_interval;
    srp = srp_create_from_geometry(&sc, &geom);
    CHECK(srp != NULL, "split state: lower-level SRP creates");
    interval = bf_effective_adapt_interval(cfg.beam.bf_fixed_mode,
                                            cfg.beam.bf_fixed_align_notebook,
                                            cfg.beam.bf_adapt_interval);
    memset(&gc, 0, sizeof(gc));
    gc.enable = cfg.beam.bf_enable;
    gc.lambda = anr_retime_alpha(cfg.beam.bf_lambda, cfg.core.sample_rate, hop * interval);
    gc.mu = cfg.beam.bf_mu;
    gc.enable_fix_mode = cfg.beam.bf_fixed_mode;
    gc.fixed_doa_rad = cfg.beam.bf_fixed_doa_rad;
    gc.fixed_align_notebook = cfg.beam.bf_fixed_align_notebook;
    gc.adapt_interval = interval;
    bf = bf_create(MCAB_CHANNELS, nf, srp->num_angles, srp->a_array, &gc);
    CHECK(bf != NULL, "split state: lower-level BF creates");
    spec = (Complex*)calloc((size_t)nf, sizeof(Complex));
    weights = (Complex*)calloc((size_t)nf * MCAB_CHANNELS, sizeof(Complex));
    mask = (int*)malloc((size_t)nf * sizeof(int));
    for (int k = 0; k < nf; ++k) mask[k] = (k % 3) != 0;
    CHECK(scene_open(&s, hop, frames), "split state: scene allocates");
    for (int f = 0; f < frames; ++f) {
        McabFront front;
        McabSampleBeamResult beam;
        McabSampleBeamInfo st;
        int vad_raw = (f / 5) % 3 != 0, vad_out = (f / 9) % 2;
        const int* m = (f % 4) == 3 ? NULL : mask;
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        CHECK(mcab_sample_beam_process_front(p, s.mics, s.far, &front) == MCAB_OK,
              "split state: front runs");
        CHECK(front.n_freqs == nf && front.linear_spectra[0] && front.linear_spectra[3],
              "split state: front publishes the four lane spectra");
        doa_step(srp, front.linear_spectra, m, vad_raw, vad_out);
        bf_process_with_weights(bf, front.linear_spectra, doa_get_smooth(srp),
                                 vad_out ? 0 : 1, m, spec, weights);
        CHECK(mcab_sample_beam_process_beam(p, &front, vad_raw, vad_out, m, &beam, &st) == MCAB_OK,
              "split state: beam runs");
        CHECK(bits_equal_f(st.doa_raw_rad, doa_get_raw(srp)) &&
              bits_equal_f(st.doa_smooth_rad, doa_get_smooth(srp)) &&
              bits_equal_f(st.doa_used_rad, bf_get_doa_used(bf)) &&
              st.bf_adaptive == bf_get_adaptive(bf),
              "split state: getters equal the lower-level SRP/BF getters");
        CHECK(!isinf(st.doa_raw_rad) && !isinf(st.doa_smooth_rad) && !isinf(st.doa_used_rad),
              "split state: no infinite angle");
        if (isfinite(st.doa_smooth_rad)) finite_smooth = 1;
        if (isfinite(st.doa_used_rad)) finite_used = 1;
        if (fixed_bf && st.doa_used_rad == 0.3f && !bits_equal_f(st.doa_used_rad, st.doa_smooth_rad))
            fixed_used_seen = 1;
        CHECK(memcmp(beam.spectrum, spec, (size_t)nf * sizeof(Complex)) == 0,
              "split state: the beam spectrum is the lower-level BF spectrum");
        CHECK(memcmp(beam.weights, weights,
                     (size_t)nf * MCAB_CHANNELS * sizeof(Complex)) == 0,
              "split state: the beam weights are the lower-level BF weights");
    }
    CHECK(finite_smooth && finite_used, "split state: the smoothed and used DOA become finite");
    CHECK(!fixed_bf || fixed_used_seen, "split state: a fixed beamformer reports its fixed angle as the used DOA");
    bf_destroy(bf);
    srp_destroy(srp);
    free(spec); free(weights); free(mask);
    scene_close(&s);
    mcab_test_beam_destroy(p);
    return 1;
}

static int test_split_beam_state_matches_lower_level(void) {
    return split_state_case(0) && split_state_case(1);
}

/* Call order: beam needs a pending front, fuse needs this hop's beam, a new front supersedes
 * whatever is pending, reset drops it; refused calls change nothing. */
static int test_split_sequence_guards(void) {
    McabTestUnitConfig cfg = split_config();
    McabTestUnitConfig cfg_post = split_config();
    McabSampleBeam *p, *q, *post;
    McabFront front, other, bad;
    McabSampleBeamResult beam;
    McabSampleBeamInfo info;
    McabFrame frame;
    float *mics, *far;
    int hop, nf;

    cfg_post.core.enable_post = 1;
    p = mcab_test_beam_create(&cfg);
    q = mcab_test_beam_create(&cfg);
    post = mcab_test_beam_create(&cfg_post);
    CHECK(p && q && post, "split guards: instances create");
    hop = mcab_hop_size(mcab_test_beam_flow(p));
    nf = mcab_n_freqs(mcab_test_beam_flow(p));
    mics = (float*)calloc((size_t)hop * MCAB_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    memset(&bad, 0, sizeof(bad));
    CHECK(mcab_sample_beam_process_beam(p, &bad, 0, 0, NULL, &beam, &info) == MCAB_INVALID_ARGUMENT,
          "split guards: a front with the wrong bin count or NULL spectra is refused");
    CHECK(mcab_sample_beam_process_front(p, mics, far, NULL) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_process_front(NULL, mics, far, &front) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_process_front(p, NULL, far, &front) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_process_front(p, mics, NULL, &front) == MCAB_INVALID_ARGUMENT,
          "split guards: NULL arguments of process_front are refused");
    CHECK(mcab_sample_beam_process_with_activity(p, mics, far, 2, 0, NULL, &beam, &info) ==
              MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_process_with_activity(p, mics, far, 0, 0, NULL, NULL, &info) ==
              MCAB_INVALID_ARGUMENT,
          "split guards: a non-boolean flag or a NULL beam is refused before any state moves");

    /* A real front of another instance: valid in shape, but p has none pending. */
    CHECK(mcab_sample_beam_process_front(q, mics, far, &other) == MCAB_OK,
          "split guards: the other instance's front runs");
    CHECK(mcab_sample_beam_process_beam(p, &other, 0, 0, NULL, &beam, &info) == MCAB_SEQUENCE_ERROR,
          "split guards: beam without a pending front is a sequence error");
    CHECK(mcab_sample_beam_fuse(post, &frame) == MCAB_SEQUENCE_ERROR,
          "split guards: fuse without a beam is a sequence error");

    CHECK(mcab_sample_beam_process_front(post, mics, far, &front) == MCAB_OK &&
          mcab_sample_beam_fuse(post, &frame) == MCAB_SEQUENCE_ERROR,
          "split guards: fuse between front and beam is a sequence error");
    bad = front;
    bad.n_freqs = nf + 1;
    CHECK(mcab_sample_beam_process_beam(post, &bad, 0, 0, NULL, &beam, &info) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_process_beam(post, &front, 2, 0, NULL, &beam, &info) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_process_beam(post, &front, 0, 2, NULL, &beam, &info) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_process_beam(post, NULL, 0, 0, NULL, &beam, &info) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_process_beam(post, &front, 0, 0, NULL, NULL, &info) == MCAB_INVALID_ARGUMENT,
          "split guards: a refused beam call changes nothing");
    CHECK(mcab_sample_beam_process_beam(post, &front, 0, 0, NULL, &beam, NULL) == MCAB_OK &&
          mcab_sample_beam_process_beam(post, &front, 0, 0, NULL, &beam, &info) == MCAB_SEQUENCE_ERROR,
          "split guards: the beam half runs once per hop; info is optional");
    CHECK(mcab_sample_beam_fuse(post, NULL) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_fuse(NULL, &frame) == MCAB_INVALID_ARGUMENT &&
          mcab_sample_beam_fuse(post, &frame) == MCAB_OK &&
          frame.e == beam.spectrum,
          "split guards: fuse follows this hop's beam");
    CHECK(mcab_sample_beam_process_front(post, mics, far, &front) == MCAB_OK &&
          mcab_sample_beam_fuse(post, &frame) == MCAB_SEQUENCE_ERROR,
          "split guards: a new front supersedes the beam of the last hop");
    CHECK(mcab_sample_beam_process_front(post, mics, far, &front) == MCAB_OK &&
          mcab_sample_beam_process_beam(post, &front, 0, 0, NULL, &beam, &info) == MCAB_OK,
          "split guards: a second front without a beam in between is legal");
    mcab_sample_beam_reset(post);
    CHECK(mcab_sample_beam_process_beam(post, &front, 0, 0, NULL, &beam, &info) == MCAB_SEQUENCE_ERROR &&
          mcab_sample_beam_fuse(post, &frame) == MCAB_SEQUENCE_ERROR,
          "split guards: a reset drops the pending front and beam");

    /* The front-only profile (a neural post-filter reads the beam spectrum): it builds, runs,
     * and has no fuse; a refused fuse leaves the unit as it was. */
    CHECK(mcab_sample_beam_process_front(p, mics, far, &front) == MCAB_OK &&
          mcab_sample_beam_process_beam(p, &front, 0, 0, NULL, &beam, &info) == MCAB_OK &&
          info.frame_index == 0,
          "split guards: the enable_post = 0 unit runs front and beam");
    CHECK(mcab_sample_beam_fuse(p, &frame) == MCAB_INVALID_ARGUMENT,
          "split guards: fuse is refused without the post context");
    CHECK(mcab_sample_beam_process_front(p, mics, far, &front) == MCAB_OK &&
          mcab_sample_beam_process_beam(p, &front, 0, 0, NULL, &beam, &info) == MCAB_OK &&
          info.frame_index == 1,
          "split guards: the refused fuse did not reset the unit");

    free(mics); free(far);
    mcab_test_beam_destroy(p);
    mcab_test_beam_destroy(q);
    mcab_test_beam_destroy(post);
    return 1;
}

static int same_bytes(const void* a, const void* b, size_t bytes) {
    if (!a || !b) return a == b;
    return memcmp(a, b, bytes) == 0;
}

static int same_frame(const McabFrame* a, const McabFrame* b) {
    size_t nf = (size_t)a->n_freqs;
    return a->n_freqs == b->n_freqs && a->hop == b->hop &&
           same_bytes(a->e, b->e, nf * sizeof(Complex)) &&
           same_bytes(a->y, b->y, nf * sizeof(Complex)) &&
           same_bytes(a->x, b->x, nf * sizeof(Complex)) &&
           same_bytes(a->r2, b->r2, nf * sizeof(float)) &&
           same_bytes(a->cn, b->cn, nf * sizeof(float)) &&
           same_bytes(a->g_res_ctx, b->g_res_ctx, nf * sizeof(float)) &&
           same_bytes(a->far_hop, b->far_hop, (size_t)a->hop * sizeof(float)) &&
           bits_equal_f(a->far_power, b->far_power) && bits_equal_f(a->max_dt, b->max_dt) &&
           bits_equal_f(a->max_sat, b->max_sat) && a->all_converged == b->all_converged &&
           a->usable_linear == b->usable_linear && a->clock_drift == b->clock_drift &&
           a->delay_changed == b->delay_changed && a->delay_locked == b->delay_locked;
}

/* A hop refused for a non-finite sample changes no adaptive state and leaves nothing pending:
 * the caller's fuse() (or beam) after it is a sequence error that changes nothing, and the
 * stream then continues exactly as if the refused hop had never been offered. */
static int test_refused_hop_then_fuse_keeps_state(void) {
    enum { FRAMES = 60, NAN_HOP = 20, STALE_HOP = 35 };
    McabTestUnitConfig cfg = split_config();
    McabSampleBeam *a, *b;
    SplitScene s;
    float* nan_mics;
    int hop, nf, ok = 1, nonzero = 0;

    cfg.core.enable_post = 1;
    a = mcab_test_beam_create(&cfg);
    b = mcab_test_beam_create(&cfg);
    CHECK(a && b, "refused hop: instances create");
    hop = mcab_hop_size(mcab_test_beam_flow(a));
    nf = mcab_n_freqs(mcab_test_beam_flow(a));
    CHECK(scene_open(&s, hop, FRAMES), "refused hop: scene allocates");
    nan_mics = (float*)malloc((size_t)hop * MCAB_CHANNELS * sizeof(float));
    CHECK(nan_mics != NULL, "refused hop: buffer allocates");
    g_rng = 0x1234567u;
    for (int f = 0; f < FRAMES && ok; ++f) {
        McabSampleBeamResult ba, bb;
        McabSampleBeamInfo ia, ib;
        McabFrame fa, fb;
        int vad = (f / 7) % 2;
        if (f == NAN_HOP) {
            /* Only `a` is offered the bad hop; the scene does not advance for it. */
            memcpy(nan_mics, s.mics, (size_t)hop * MCAB_CHANNELS * sizeof(float));
            nan_mics[0] = NAN;
            CHECK(mcab_sample_beam_process_with_activity(a, nan_mics, s.far, vad, vad, NULL, &ba,
                                                           &ia) == MCAB_INVALID_ARGUMENT,
                  "refused hop: a non-finite microphone sample is refused");
            CHECK(mcab_sample_beam_fuse(a, &fa) == MCAB_SEQUENCE_ERROR,
                  "refused hop: fuse after the refused hop is a sequence error");
        }
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        if (f == STALE_HOP) {
            /* Both twins: a front, then a refused front; the first front's view is stale. */
            McabFront front, unused;
            McabSampleBeam* twins[2] = { a, b };
            memcpy(nan_mics, s.mics, (size_t)hop * MCAB_CHANNELS * sizeof(float));
            nan_mics[0] = NAN;
            for (int t = 0; t < 2; ++t) {
                CHECK(mcab_sample_beam_process_front(twins[t], s.mics, s.far, &front) ==
                          MCAB_OK &&
                      mcab_sample_beam_process_front(twins[t], nan_mics, s.far, &unused) ==
                          MCAB_INVALID_ARGUMENT,
                      "refused hop: a refused front after an accepted one");
                CHECK(mcab_sample_beam_process_beam(twins[t], &front, vad, vad, NULL, &ba, &ia) ==
                          MCAB_SEQUENCE_ERROR &&
                      mcab_sample_beam_fuse(twins[t], &fa) == MCAB_SEQUENCE_ERROR,
                      "refused hop: the refused front leaves no front pending for beam or fuse");
            }
            continue;
        }
        ok = mcab_sample_beam_process_with_activity(a, s.mics, s.far, vad, vad, NULL, &ba, &ia) ==
                 MCAB_OK &&
             mcab_sample_beam_process_with_activity(b, s.mics, s.far, vad, vad, NULL, &bb, &ib) ==
                 MCAB_OK &&
             mcab_sample_beam_fuse(a, &fa) == MCAB_OK &&
             mcab_sample_beam_fuse(b, &fb) == MCAB_OK &&
             memcmp(ba.spectrum, bb.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
             memcmp(ba.weights, bb.weights,
                    (size_t)MCAB_CHANNELS * (size_t)nf * sizeof(Complex)) == 0 &&
             same_info(&ia, &ib) && same_frame(&fa, &fb);
        if (f > NAN_HOP)
            for (int k = 0; k < nf; ++k) if (ba.spectrum[k].r != 0.0f) nonzero = 1;
    }
    CHECK(ok, "refused hop: the stream after the refused hop and fuse equals the uninterrupted "
              "twin byte for byte (beam, weights, frame info, fused frame)");
    CHECK(nonzero, "refused hop: the compared output is not silence");
    free(nan_mics);
    scene_close(&s);
    mcab_test_beam_destroy(a);
    mcab_test_beam_destroy(b);
    return 1;
}

/* reset() returns the unit to the state of a fresh instance: the same input afterwards gives
 * the same beam bytes and frame info as the first pass did. */
static int test_reset_equals_fresh(void) {
    enum { FRAMES = 40 };
    McabTestUnitConfig cfg = split_config();
    McabSampleBeam *warm, *fresh;
    SplitScene s;
    Complex* first[FRAMES] = {0};
    int hop, nf, ok = 1;

    cfg.core.enable_post = 1;
    warm = mcab_test_beam_create(&cfg);
    fresh = mcab_test_beam_create(&cfg);
    CHECK(warm && fresh, "reset: instances create");
    hop = mcab_hop_size(mcab_test_beam_flow(warm));
    nf = mcab_n_freqs(mcab_test_beam_flow(warm));
    CHECK(scene_open(&s, hop, FRAMES), "reset: scene allocates");
    for (int pass = 0; pass < 2 && ok; ++pass) {
        g_rng = 0x1234567u;
        memset(s.far_hist, 0, (size_t)FRAMES * hop * sizeof(float));
        memset(s.near_hist, 0, (size_t)FRAMES * hop * sizeof(float));
        if (pass == 1) mcab_sample_beam_reset(warm);
        for (int f = 0; f < FRAMES && ok; ++f) {
            McabSampleBeamResult beam;
            McabSampleBeamInfo info;
            int vad = (f / 7) % 2;
            fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
            ok = mcab_sample_beam_process_with_activity(warm, s.mics, s.far, vad, vad, NULL, &beam, &info) ==
                 MCAB_OK;
            if (!ok) break;
            if (pass == 0) {
                first[f] = (Complex*)malloc((size_t)nf * sizeof(Complex));
                ok = first[f] != NULL;
                if (ok) memcpy(first[f], beam.spectrum, (size_t)nf * sizeof(Complex));
            } else {
                McabSampleBeamResult fb;
                McabSampleBeamInfo ref_info;
                ok = mcab_sample_beam_process_with_activity(fresh, s.mics, s.far, vad, vad, NULL, &fb, &ref_info) ==
                         MCAB_OK &&
                     memcmp(first[f], beam.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
                     memcmp(fb.spectrum, beam.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
                     same_info(&info, &ref_info);
            }
        }
    }
    for (int f = 0; f < FRAMES; ++f) free(first[f]);
    CHECK(ok, "reset: a reset unit reproduces its first pass and a fresh instance byte for byte");
    scene_close(&s);
    mcab_test_beam_destroy(warm);
    mcab_test_beam_destroy(fresh);
    return 1;
}

/* The flow under the helper reads back the 16 kHz default grid (independent constants and the
 * helper's own spatial config), a NULL handle is safe, and a destroyed flow in caller memory
 * reports -1. */
static int test_core_view_and_timing(void) {
    McabTestUnitConfig cfg = mcab_test_unit_default_config(16000);
    McabSampleSpatialConfig sc;
    McabMemReq req;
    McabLastTiming t, z;
    void* pool = NULL;
    McabSampleBeam* p = NULL;
    Mcab* flow;
    if (mcab_test_unit_get_mem_requirements(&cfg, &req) == 0 &&
        posix_memalign(&pool, 16, (size_t)req.bytes) == 0)
        p = mcab_test_beam_init(pool, (size_t)req.bytes, &cfg);
    flow = mcab_test_beam_flow(p);
    CHECK(p != NULL && flow != NULL, "core view: unit builds in a caller pool");
    CHECK(mcab_linear_aec_count(flow) == MCAB_CHANNELS && mcab_hop_size(flow) == 128 &&
              mcab_fft_size(flow) == 256 && mcab_n_freqs(flow) == 129 &&
              mcab_sample_spatial_config(&cfg.core, &cfg.beam, &sc) == 0 &&
              sc.n_freqs == mcab_n_freqs(flow) && sc.srp.NFFT == (float)mcab_fft_size(flow),
          "core view: the unit's flow is on the 16 kHz default grid the helper also uses");
    CHECK(mcab_test_beam_flow(NULL) == NULL, "core view: NULL handle");
    memset(&t, 0xff, sizeof(t));
    memset(&z, 0, sizeof(z));
    mcab_get_last_timing(mcab_test_beam_flow(NULL), &t);
    CHECK(memcmp(&t, &z, sizeof(t)) == 0, "core view: a NULL handle zeroes the timing record");
    mcab_test_beam_destroy(p);
    CHECK(mcab_hop_size(flow) == -1 && mcab_n_freqs(flow) == -1 && mcab_enable_post(flow) == -1,
          "core view: the destroyed flow reports -1");
    free(pool);
    return 1;
}

/* bf_adapt_interval is bounded (hop * interval stays inside int on every grid), and a cadence
 * long enough to underflow the retimed RLS retention is refused by the size query as by init. */
static int test_adapt_interval_bound(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int g;
    for (g = 0; g < 3; ++g) {
        McabTestUnitConfig cfg = mcab_test_unit_default_config(grids[g][0]);
        McabSampleSpatialConfig sc;
        size_t bytes;
        McabSampleBeam* b;
        cfg.core.fft_size = grids[g][1];
        cfg.beam.bf_lambda = 1.0f;    /* retimes to 1 at any cadence */
        cfg.beam.bf_adapt_interval = MCAB_SAMPLE_BEAM_MAX_ADAPT_INTERVAL;
        CHECK(mcab_sample_spatial_config(&cfg.core, &cfg.beam, &sc) == 0 &&
                  sc.bf.adapt_interval == MCAB_SAMPLE_BEAM_MAX_ADAPT_INTERVAL &&
                  sc.bf.lambda == 1.0f &&
                  mcab_sample_beam_get_mem_size(&cfg.core, &cfg.beam, &bytes) == 0,
              "adapt interval: the bound is accepted");
        b = mcab_test_beam_create(&cfg);
        CHECK(b != NULL, "adapt interval: a helper at the bound builds");
        mcab_test_beam_destroy(b);
        cfg.beam.bf_adapt_interval = MCAB_SAMPLE_BEAM_MAX_ADAPT_INTERVAL + 1;
        CHECK(mcab_sample_spatial_config(&cfg.core, &cfg.beam, &sc) != 0 &&
                  mcab_sample_beam_get_mem_size(&cfg.core, &cfg.beam, &bytes) != 0 &&
                  mcab_test_beam_create(&cfg) == NULL,
              "adapt interval: bound + 1 is refused by the query and by init");
        cfg.beam.bf_adapt_interval = 0x7fffffff;
        CHECK(mcab_sample_spatial_config(&cfg.core, &cfg.beam, &sc) != 0,
              "adapt interval: INT_MAX is refused");
    }
    {
        /* 0.5 per 10 ms over 2000 hops of 8 ms: 0.5^1600 underflows to 0 in float. */
        McabTestUnitConfig cfg = mcab_test_unit_default_config(16000);
        size_t bytes;
        cfg.beam.bf_lambda = 0.5f;
        cfg.beam.bf_adapt_interval = 2000;
        CHECK(mcab_sample_beam_get_mem_size(&cfg.core, &cfg.beam, &bytes) != 0 &&
                  mcab_test_beam_create(&cfg) == NULL,
              "adapt interval: an underflowing retention is refused by the query and by init");
    }
    return 1;
}

/* The DOA band is checked as SRP gets it: high 0 resolves to min(7000, Nyquist), and a band
 * whose low end is not below that is refused by the query and by init. */
static int test_doa_band_bounds(void) {
    static const struct { int sr; float low, high; int accept; } rows[] = {
        {16000, 6999.0f, 0.0f, 1},    {16000, 7000.0f, 0.0f, 0},    {16000, 7001.0f, 0.0f, 0},
        {48000, 6999.0f, 0.0f, 1},    {48000, 7000.0f, 0.0f, 0},    {48000, 7001.0f, 0.0f, 0},
        {16000, 7500.0f, 7999.0f, 1}, {16000, 7999.0f, 7999.0f, 0}, {16000, 8000.0f, 8000.0f, 0},
        {16000, 300.0f, 300.0f, 0},   {16000, 300.0f, 299.0f, 0},   {16000, 300.0f, 0.0f, 1},
        {16000, 0.0f, 8000.0f, 1},    {16000, 7999.0f, 8000.0f, 1},
        {48000, 23999.0f, 24000.0f, 1},
    };
    int r;
    for (r = 0; r < (int)(sizeof(rows) / sizeof(rows[0])); ++r) {
        McabTestUnitConfig cfg = mcab_test_unit_default_config(rows[r].sr);
        McabSampleSpatialConfig sc;
        McabSampleBeam* b;
        size_t bytes;
        int query, init;
        cfg.beam.doa_low_freq_hz = rows[r].low;
        cfg.beam.doa_high_freq_hz = rows[r].high;
        query = mcab_sample_spatial_config(&cfg.core, &cfg.beam, &sc) == 0 &&
                mcab_sample_beam_get_mem_size(&cfg.core, &cfg.beam, &bytes) == 0;
        b = mcab_test_beam_create(&cfg);
        init = b != NULL;
        mcab_test_beam_destroy(b);
        if (query != rows[r].accept || init != rows[r].accept)
            fprintf(stderr, "  DOA band row %d (sr %d low %g high %g): query %d init %d\n", r,
                    rows[r].sr, (double)rows[r].low, (double)rows[r].high, query, init);
        CHECK(query == rows[r].accept && init == rows[r].accept,
              "DOA band: low against the resolved high, by the query and by init");
    }
    {
        McabTestUnitConfig d16 = mcab_test_unit_default_config(16000);
        McabTestUnitConfig d48 = mcab_test_unit_default_config(48000);
        McabSampleSpatialConfig sc;
        CHECK(mcab_sample_spatial_config(&d16.core, &d16.beam, &sc) == 0 &&
                  mcab_sample_spatial_config(&d48.core, &d48.beam, &sc) == 0,
              "DOA band: the default band is valid at both rates");
    }
    return 1;
}

static int run_all_tests(const char* bin_dir, const char* tmp) {
    McabTestUnitConfig invalid =
        mcab_test_unit_default_config(16000);
    invalid.core.fft_size = 1024;
    CHECK(mcab_test_beam_create(&invalid) == NULL,
          "16 kHz rejects cross-rate FFT 1024");
    invalid = mcab_test_unit_default_config(48000);
    invalid.core.fft_size = 512;
    CHECK(mcab_test_beam_create(&invalid) == NULL,
          "48 kHz rejects cross-rate FFT 512");
    invalid = mcab_test_unit_default_config(16000);
    invalid.core.enable_post = 0;
    {
        McabSampleBeam* front_only = mcab_test_beam_create(&invalid);
        CHECK(front_only != NULL,
              "the unit accepts the front-only profile (no fuse)");
        mcab_test_beam_destroy(front_only);
    }
    invalid.core.enable_post = 2;
    CHECK(mcab_test_beam_create(&invalid) == NULL,
          "a non-boolean enable_post is rejected");
    invalid = mcab_test_unit_default_config(16000);
    invalid.beam.bf_lambda = 1.0f + 1e-6f;
    CHECK(mcab_test_beam_create(&invalid) == NULL,
          "BF forgetting factor above 1.0 is rejected");
    invalid.beam.bf_lambda = 1.0f;
    {
        McabSampleBeam* boundary = mcab_test_beam_create(&invalid);
        CHECK(boundary != NULL,
              "BF forgetting factor of exactly 1.0 is still accepted");
        mcab_test_beam_destroy(boundary);
    }
    CHECK(run_grid(16000, 256),
          "16 kHz 256/128 spatial unit");
    CHECK(run_grid(16000, 512),
          "16 kHz 512/256 spatial unit");
    CHECK(run_grid(48000, 1024),
          "48 kHz 1024/512 spatial unit");
    CHECK(test_bf_fixed_notebook_lambda_matches_forced_cadence(),
          "BF fixed-notebook lambda/forced-cadence match test");
    CHECK(test_bf_auto_mode_respects_configured_interval(),
          "BF auto-mode configured-interval test");
    CHECK(test_rejected_calls_preserve_vad_state(),
          "rejected calls (NULL args, NaN input) must not contaminate VAD state");
    CHECK(run_static_parity(16000, 256),
          "16 kHz 256/128 static-memory pool parity");
    CHECK(run_static_parity(16000, 512),
          "16 kHz 512/256 static-memory pool parity");
    CHECK(run_static_parity(48000, 1024),
          "48 kHz 1024/512 static-memory pool parity");
    CHECK(test_fuse_leaves_the_residual_gain_to_the_caller(),
          "fuse leaves the residual gain to the caller");
    CHECK(test_unit_chain_equals_reference_chain(),
          "unit + fuse + the examples' chain equals the reference chain over the front end");
    CHECK(test_caller_res_reset_and_preset(), "the caller-owned RES: reset restarts it; a preset survives it");
    CHECK(test_caller_res_ramped_preset(), "the caller-owned RES: runtime strength");
    CHECK(test_caller_res_preset_ramp_vs_oracle(),
          "the caller-owned RES: a ramped preset follows the oracle; skipping it is caught");
    CHECK(test_chain_reset_equals_fresh(), "reset-then-rerun equals a fresh chain");
    CHECK(test_chain_heap_equals_pool(), "heap and pool units give the same chain output");
    CHECK(test_raw_runner_black_box(bin_dir, tmp), "mcab_raw against the oracle");
    CHECK(test_core_aligned_ref(), "front view aligned reference");
    CHECK(test_split_null_mask_equals_begin(),
          "front + beam equals process_with_activity");
    CHECK(test_split_mask_and_vad_reach_srp_bf(),
          "mask and activity flags reach SRP and BF");
    CHECK(test_split_beam_state_matches_lower_level(),
          "beam result and frame info match the lower-level SRP/BF");
    CHECK(test_split_sequence_guards(), "front / beam / fuse call order");
    CHECK(test_refused_hop_then_fuse_keeps_state(),
          "a refused hop followed by fuse changes nothing");
    CHECK(test_reset_equals_fresh(), "reset equals a fresh instance");
    CHECK(test_core_view_and_timing(), "core view and timing record");
    CHECK(test_adapt_interval_bound(), "BF adapt interval bound");
    CHECK(test_doa_band_bounds(), "DOA band bounds");
    printf("All mcab_sample_beam tests passed (spatial=%s)\n",
           mcab_sample_beam_spatial_backend());
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <example-bin-dir> <scratch-dir>\n", argv[0]);
        return 2;
    }
    return run_all_tests(argv[1], argv[2]) ? 0 : 1;
}
