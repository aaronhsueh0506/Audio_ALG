/**
 * tests/test_array_sample_beam.c -- the SRP-PHAT/BF/fallback-VAD sample helper over an array
 * instance (aec.h, n_mics = 4; array_sample_beam.h), on all three production signal grids: the
 * split front / beam / fuse sequence and its guards, the helper's own pool contract, fuse
 * publishing the array's residual gain, the helper followed by the examples' gain / NR / comfort
 * noise / synthesis (aec_apply_gain / aec_apply_cng) against the independent oracle
 * (conv_chain_4ch.h) including a ramped level change, the array's gain stage across reset and a level
 * change, and 4ch_raw run as a black box against that oracle. The array-only checks (level refusals and
 * ramp, views, timing) are in the AEC library's test_aec_array*.c, the chain-level checks without the helper in test_4ch_chain.c.
 * Arguments: the directory of the example binaries and a scratch directory.
 */

#define ARRAY_TEST_RIG_WITH_BEAM 1
#include "array_test_rig.h"
#include "bf.h"
#include "srp.h"
#include "conv_chain_4ch.h"
#include "array_sample_post.h"
#include "array_sample_unit.h"

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
    ArrayTestUnitConfig cfg =
        array_test_unit_default_config(sample_rate);
    ArraySampleBeam* p;
    ArraySampleBeamInfo info;
    ArraySampleBeamResult beam;
    AecResidualView frame;
    float* microphones;
    float* far;
    float phase = 0.0f;
    int hop;
    int n_freqs;

    cfg.aec_cfg.fft_size = fft_size;
    cfg.beam.bf_fixed_mode = 1;
    cfg.beam.bf_fixed_doa_rad = 0.4f;
    cfg.beam.bf_mu = 0.02f;
    p = array_test_beam_create(&cfg);
    CHECK(p != NULL, "create 4ch linear AEC + BF unit");
    hop = aec_hop_size(array_test_beam_aec(p));
    n_freqs = array_test_n_freqs(array_test_beam_aec(p));
    CHECK(hop == fft_size / 2,
          "rate-specific hop");
    CHECK(array_test_fft_size(array_test_beam_aec(p)) == fft_size,
          "zero-padding-free frame/FFT contract");
    CHECK(n_freqs == fft_size / 2 + 1,
          "spatial bin count follows selected FFT");
    {
        /* What the helper hands SRP and BF is on the array's grid. */
        ArraySampleSpatialConfig sc;
        CHECK(array_sample_spatial_config(&cfg.aec_cfg, &cfg.beam, &sc) == 0 &&
                  sc.srp.sr == (float)sample_rate && sc.srp.NFFT == (float)fft_size &&
                  sc.srp.F == n_freqs && sc.srp.M == AEC_ARRAY_MICS,
              "DOA grid follows selected main AEC grid");
        CHECK(sc.n_freqs == n_freqs && sc.n_channels == AEC_ARRAY_MICS &&
                  sc.bf.adapt_interval == array_sample_beam_bf_effective_adapt_interval(p),
              "BF grid follows selected main AEC grid");
    }
    CHECK(aec_array_matched_filter_count(array_test_beam_aec(p)) == 1,
          "one shared matcher");
    CHECK(aec_array_linear_aec_count(array_test_beam_aec(p)) ==
              AEC_ARRAY_MICS,
          "four linear AEC lanes");

    microphones = (float*)malloc(
        (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far, "allocate unit buffers");

    for (int frame_no = 0; frame_no < 80; ++frame_no) {
        for (int i = 0; i < hop; ++i) {
            float echo = 0.08f * sinf(phase);
            float near = frame_no >= 30
                ? 0.025f * sinf(phase * 1.73f + 0.2f) : 0.0f;
            far[i] = echo;
            for (int ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
                microphones[i * AEC_ARRAY_MICS + ch] =
                    echo * (1.0f - 0.03f * ch) +
                    near * (1.0f + 0.02f * ch);
            }
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(array_sample_beam_process_with_activity(
                  p, microphones, far,
                  frame_no >= 30, frame_no >= 30, NULL, &beam, &info) ==
                  AEC_OK,
              "explicit-VAD frame");
        CHECK(info.frame_index == (uint64_t)frame_no,
              "monotonic spatial frame index");
        CHECK(isfinite(info.doa_used_rad), "fixed BF DOA must be finite");
        CHECK(beam.spectrum && beam.weights &&
              spectrum_finite(beam.spectrum, n_freqs) &&
              spectrum_finite(beam.weights, AEC_ARRAY_MICS * n_freqs),
              "beam spectrum and weights are finite");
        CHECK(array_sample_beam_fuse(p, &frame) == AEC_OK &&
              frame.n_freqs == n_freqs && frame.error_spec == beam.spectrum &&
              frame.r2 && frame.comfort_noise && frame.far_hop && frame.res_gain,
              "fuse publishes the beam spectrum and the post-stage inputs");
    }

    array_sample_beam_reset(p);
    CHECK(array_sample_beam_process(
              p, microphones, far, &beam, &info) ==
              AEC_OK,
          "fallback-VAD processing after reset");
    CHECK(info.frame_index == 0, "reset restarts the unit frame index");

    free(far);
    free(microphones);
    array_test_beam_destroy(p);
    return 1;
}

/* Contract test: in fixed-notebook mode (bf_fixed_mode && bf_fixed_align_notebook),
 * bf_create() forces the BF RLS update cadence to 1 hop whatever bf_adapt_interval is.
 * The lambda retime in array_test_beam_create() must use that forced cadence, or
 * lambda is calibrated for a slower update period than the one that runs. */
static int test_bf_fixed_notebook_lambda_matches_forced_cadence(void) {
    ArrayTestUnitConfig cfg =
        array_test_unit_default_config(16000);
    ArraySampleBeam* p;
    int hop_size;
    int actual_interval;
    float actual_lambda;
    float expected_lambda;
    float buggy_lambda;

    cfg.beam.bf_fixed_mode = 1;
    cfg.beam.bf_fixed_align_notebook = 1;
    cfg.beam.bf_adapt_interval = 4; /* caller asks for a slower cadence... */
    cfg.beam.bf_lambda = 0.98f;

    p = array_test_beam_create(&cfg);
    CHECK(p != NULL, "fixed-notebook cadence test: create pipeline");

    hop_size = aec_hop_size(array_test_beam_aec(p));
    actual_interval = array_sample_beam_bf_effective_adapt_interval(p);
    actual_lambda = array_sample_beam_bf_lambda(p);

    /* ...but fixed_align_notebook forces the real BF cadence to every hop
     * (1), regardless of the requested 4. */
    CHECK(actual_interval == 1,
          "fixed-notebook mode forces the actual BF cadence to 1");

    /* Lambda must be retimed assuming that SAME forced interval (1), i.e.
     * exactly anr_retime_alpha(bf_lambda, sr, hop_size*1) -- not
     * hop_size*4, which scaling with the raw pre-forced bf_adapt_interval
     * would give. */
    expected_lambda = anr_retime_alpha(
        cfg.beam.bf_lambda, cfg.aec_cfg.sample_rate, hop_size * 1);
    CHECK(fabsf(actual_lambda - expected_lambda) < 1e-9f,
          "lambda retime uses the forced (not the requested) adapt_interval");

    /* Sanity check that this test can fail: the lambda scaled against the
     * pre-forced interval of 4 must visibly differ from the correct one. */
    buggy_lambda = anr_retime_alpha(
        cfg.beam.bf_lambda, cfg.aec_cfg.sample_rate, hop_size * 4);
    CHECK(fabsf(actual_lambda - buggy_lambda) > 1e-4f,
          "forced-cadence lambda differs from one retimed with the requested interval");

    array_test_beam_destroy(p);
    return 1;
}

/* Companion check: outside fixed-notebook mode, a configured
 * bf_adapt_interval > 1 is NOT forced down, so the effective interval used
 * for both the actual BF cadence and the lambda retime must equal the
 * requested value itself (guards the other branch of
 * bf_effective_adapt_interval() from regressing). */
static int test_bf_auto_mode_respects_configured_interval(void) {
    ArrayTestUnitConfig cfg =
        array_test_unit_default_config(16000);
    ArraySampleBeam* p;
    int hop_size;
    float expected_lambda;

    cfg.beam.bf_adapt_interval = 4;
    cfg.beam.bf_lambda = 0.98f;

    p = array_test_beam_create(&cfg);
    CHECK(p != NULL, "auto-mode cadence test: create pipeline");

    hop_size = aec_hop_size(array_test_beam_aec(p));
    CHECK(array_sample_beam_bf_effective_adapt_interval(p) == 4,
          "auto mode keeps the requested adapt_interval unforced");

    expected_lambda = anr_retime_alpha(
        cfg.beam.bf_lambda, cfg.aec_cfg.sample_rate, hop_size * 4);
    CHECK(fabsf(array_sample_beam_bf_lambda(p) - expected_lambda) < 1e-9f,
          "lambda retime uses the requested adapt_interval in auto mode");

    array_test_beam_destroy(p);
    return 1;
}

/* array_sample_beam_process() must reject a bad call (NULL far_reference/beam,
 * non-finite input) before auto_vad() can fold it into the noise_power EMA; a NaN
 * noise_power would make every later comparison false and disable VAD for good. Two
 * identically-fed pipelines run side by side and only "attacked" also receives the calls
 * that must be rejected; any leak into VAD state makes it diverge from "clean". */
static int test_rejected_calls_preserve_vad_state(void) {
    const int sample_rate = 16000;
    const int fft_size = 256;
    ArrayTestUnitConfig cfg =
        array_test_unit_default_config(sample_rate);
    ArraySampleBeam *clean, *attacked;
    int hop, n_freqs;
    float *mic, *far, *nan_mic, *nan_far, *inf_far, *quiet_mic;
    float phase = 0.0f;
    ArraySampleBeamResult beam_clean, beam_attacked, throwaway_beam;
    ArraySampleBeamInfo info_clean, info_attacked, throwaway;

    cfg.aec_cfg.fft_size = fft_size;
    clean = array_test_beam_create(&cfg);
    attacked = array_test_beam_create(&cfg);
    CHECK(clean && attacked, "create clean/attacked pipeline pair");
    hop = aec_hop_size(array_test_beam_aec(clean));
    n_freqs = array_test_n_freqs(array_test_beam_aec(clean));

    mic = (float*)malloc(
        (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    nan_mic = (float*)malloc(
        (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    nan_far = (float*)malloc((size_t)hop * sizeof(float));
    inf_far = (float*)malloc((size_t)hop * sizeof(float));
    quiet_mic = (float*)malloc(
        (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    CHECK(mic && far && nan_mic
              && nan_far && inf_far && quiet_mic,
          "allocate VAD-contamination test buffers");

    /* Quiet warm-up so noise_power settles away from its init value
     * identically on both instances before the attack. */
    for (int frame = 0; frame < 5; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float v = 0.002f * sinf(phase);
            far[i] = v;
            for (int ch = 0; ch < AEC_ARRAY_MICS; ++ch)
                mic[i * AEC_ARRAY_MICS + ch] = v;
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(array_sample_beam_process(
                  clean, mic, far, &beam_clean, &info_clean) ==
                  AEC_OK,
              "clean warm-up frame");
        CHECK(array_sample_beam_process(
                  attacked, mic, far, &beam_attacked, &info_attacked) ==
                  AEC_OK,
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
        for (int ch = 0; ch < AEC_ARRAY_MICS; ++ch)
            mic[i * AEC_ARRAY_MICS + ch] = v;
        phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
    }
    CHECK(array_sample_beam_process(
              clean, mic, far, &beam_clean, &info_clean) ==
              AEC_OK,
          "clean hangover-arming frame");
    CHECK(array_sample_beam_process(
              attacked, mic, far, &beam_attacked, &info_attacked) ==
              AEC_OK,
          "attacked hangover-arming frame");
    CHECK(info_clean.vad_raw == 1 && info_attacked.vad_raw == 1,
          "loud tone actually arms the hangover on both instances");

    memcpy(nan_mic, mic,
           (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    nan_mic[0] = NAN;
    memcpy(nan_far, far, (size_t)hop * sizeof(float));
    nan_far[0] = NAN;
    memcpy(inf_far, far, (size_t)hop * sizeof(float));
    inf_far[0] = (float)INFINITY;
    CHECK(array_sample_beam_process(
              attacked, mic, NULL, &throwaway_beam, &throwaway) ==
              AEC_ERR_ARG,
          "NULL far_reference is rejected");
    CHECK(array_sample_beam_process(
              attacked, mic, far, NULL, &throwaway) ==
              AEC_ERR_ARG,
          "NULL beam is rejected");
    CHECK(array_sample_beam_process(
              attacked, nan_mic, far, &throwaway_beam, &throwaway) !=
              AEC_OK,
          "NaN mic input is rejected, not silently processed");
    /* auto_vad() reads only the microphones, so on finite mic data a non-finite far would
     * pass its own guard and only be caught by the finite check inside
     * array_test_front(), after auto_vad() has already moved
     * noise_power/vad_hangover. NaN and Inf are both exercised. The mic content is quiet
     * on purpose: a loud frame would re-arm vad_hangover to the value it already has and
     * hide a missing far_reference check, while quiet content takes the EMA's silence
     * branch and changes noise_power if auto_vad() wrongly runs. */
    for (int i = 0; i < hop; ++i) {
        float v = 0.003f * sinf(phase + 1.7f);
        for (int ch = 0; ch < AEC_ARRAY_MICS; ++ch)
            quiet_mic[i * AEC_ARRAY_MICS + ch] = v;
    }
    CHECK(array_sample_beam_process(
              attacked, quiet_mic, nan_far, &throwaway_beam, &throwaway) ==
              AEC_ERR_ARG,
          "NaN far_reference is rejected before touching VAD state");
    CHECK(array_sample_beam_process(
              attacked, quiet_mic, inf_far, &throwaway_beam, &throwaway) ==
              AEC_ERR_ARG,
          "Inf far_reference is rejected before touching VAD state");

    /* Resume identical valid input on both, QUIET so the hangover counter walks down to 0
     * instead of being re-armed every hop (a loud frame would mask a 1-frame hangover
     * discrepancy). A leak in any rejected call above desyncs "attacked" from "clean". */
    for (int frame = 0; frame < 40; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float v = 0.002f * sinf(phase);
            far[i] = v;
            for (int ch = 0; ch < AEC_ARRAY_MICS; ++ch)
                mic[i * AEC_ARRAY_MICS + ch] = v;
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(array_sample_beam_process(
                  clean, mic, far, &beam_clean, &info_clean) ==
                  AEC_OK,
              "clean post-attack frame");
        CHECK(array_sample_beam_process(
                  attacked, mic, far, &beam_attacked, &info_attacked) ==
                  AEC_OK,
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
        for (int ch = 0; ch < AEC_ARRAY_MICS; ++ch)
            mic[i * AEC_ARRAY_MICS + ch] = v;
        phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
    }
    CHECK(array_sample_beam_process(
              clean, mic, far, &beam_clean, &info_clean) ==
              AEC_OK,
          "clean final loud frame");
    CHECK(info_clean.vad_raw == 1,
          "loud tone is actually detected as speech");

    free(mic); free(far); free(nan_mic);
    free(nan_far); free(inf_far); free(quiet_mic);
    array_test_beam_destroy(clean);
    array_test_beam_destroy(attacked);
    return 1;
}

/* Static-memory pool coverage for the descriptor-tier composition layer: poison/bounds,
 * descriptor-staleness rejection and a heap-vs-pool byte-equal run. Follows
 * the AEC library's test_aec_array.c (0xa5 poison with a trailing-byte bounds
 * check, the 8-point descriptor mutation block, destroy-then-reinit pool reuse) plus a
 * heap vs. pool run over many hops comparing the beam spectrum, the weights, the fused
 * frame and every scalar field of ArraySampleBeamInfo. The composite pool can be
 * several MB at 48 kHz/1024, so there is no full-pool scan around every rejection. */
static int run_static_parity(int sample_rate, int fft_size) {
    enum { EXTRA = 32, FRAMES = 40 };
    ArrayTestUnitConfig cfg =
        array_test_unit_default_config(sample_rate);
    Aec* aec;
    AecMemReq aec_req;
    ArraySampleBeam* heap;
    ArraySampleBeam* stat;
    unsigned char* pool = NULL;
    unsigned char* beam_pool;
    float* microphones;
    float* far;
    float phase_step;
    size_t aec_bytes, beam_bytes, pool_bytes;
    size_t k;
    int hop;
    int n_freqs;
    int frame;

    cfg.aec_cfg.fft_size = fft_size;
    cfg.beam.bf_fixed_mode = 1;
    cfg.beam.bf_fixed_doa_rad = 0.35f;
    cfg.beam.bf_mu = 0.02f;

    /* One caller pool: the array and then the helper. The array's descriptor contract is the AEC
     * library's test_aec_array.c (test_descriptor). */
    CHECK(aec_get_mem_requirements(&cfg.aec_cfg, &aec_req) == 0 &&
              array_sample_beam_get_mem_size(&cfg.aec_cfg, &cfg.beam, &beam_bytes) == 0,
          "static memory requirement query succeeds");
    aec_bytes = ALIGN16((size_t)aec_req.bytes);
    pool_bytes = aec_bytes + beam_bytes + (size_t)EXTRA;

    CHECK(posix_memalign((void**)&pool, 16u, pool_bytes) == 0 && pool,
          "aligned caller pool allocates");
    memset(pool, 0xa5, pool_bytes);
    aec = aec_init_ex(pool, aec_bytes, &cfg.aec_cfg, &aec_req);
    CHECK(aec != NULL, "the array builds in the caller pool");
    beam_pool = pool + aec_bytes;

    CHECK(array_sample_beam_init(beam_pool + 1, beam_bytes, &cfg.aec_cfg, &cfg.beam, aec) == NULL,
          "static init rejects a misaligned pool");
    CHECK(array_sample_beam_init(beam_pool, beam_bytes - 1u, &cfg.aec_cfg, &cfg.beam, aec) == NULL,
          "static init rejects an undersized pool");
    {
        ArrayTestUnitConfig other = cfg;
        other.aec_cfg.fft_size = fft_size == 512 ? 256 : (sample_rate == 16000 ? 512 : 1024);
        CHECK(sample_rate == 48000 ||
                  array_sample_beam_init(beam_pool, beam_bytes * 4u, &other.aec_cfg, &other.beam,
                                        aec) == NULL,
              "static init rejects an array of another grid");
        other = cfg;
        other.beam.bf_lambda = 2.0f;
        CHECK(array_sample_beam_init(beam_pool, beam_bytes, &other.aec_cfg, &other.beam, aec) == NULL,
              "static init rejects an invalid beamformer parameter set");
        CHECK(array_sample_beam_init(beam_pool, beam_bytes, &cfg.aec_cfg, &cfg.beam, NULL) == NULL,
              "static init rejects a missing array");
    }
    for (k = 0; k < beam_bytes + (size_t)EXTRA; ++k)
        if (beam_pool[k] != 0xa5) break;
    CHECK(k == beam_bytes + (size_t)EXTRA, "every refused init left the helper pool unwritten");

    stat = array_sample_beam_init(beam_pool, beam_bytes, &cfg.aec_cfg, &cfg.beam, aec);
    heap = array_test_beam_create(&cfg);
    CHECK(stat != NULL && heap != NULL,
          "poisoned caller pool and heap construction both succeed");

    for (k = 0; k < (size_t)EXTRA; ++k) {
        if (beam_pool[beam_bytes + k] != 0xa5) break;
    }
    CHECK(k == (size_t)EXTRA, "static init stays inside the queried pool");

    hop = aec_hop_size(aec);
    n_freqs = array_test_n_freqs(aec);
    CHECK(hop == aec_hop_size(array_test_beam_aec(heap)),
          "heap and pool instances agree on hop size");
    CHECK(hop == fft_size / 2, "rate-specific hop");

    microphones = (float*)malloc(
        (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far,
          "static parity buffers allocate");

    phase_step = (float)(2.0 * M_PI * 700.0 / sample_rate);
    for (frame = 0; frame < FRAMES; ++frame) {
        ArraySampleBeamInfo heap_info;
        ArraySampleBeamInfo stat_info;
        ArraySampleBeamResult heap_beam;
        ArraySampleBeamResult stat_beam;
        AecResidualView heap_frame;
        AecResidualView stat_frame;
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
            for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
                microphones[i * AEC_ARRAY_MICS + ch] =
                    echo * (1.0f - 0.03f * ch) + near * (1.0f + 0.02f * ch);
            }
        }

        CHECK(array_sample_beam_process_with_activity(
                  heap, microphones, far, vad, vad, NULL,
                  &heap_beam, &heap_info) == AEC_OK,
              "heap instance processes a synthetic hop");
        CHECK(array_sample_beam_process_with_activity(
                  stat, microphones, far, vad, vad, NULL,
                  &stat_beam, &stat_info) == AEC_OK,
              "pool instance processes a synthetic hop");
        CHECK(memcmp(heap_beam.spectrum, stat_beam.spectrum,
                     (size_t)n_freqs * sizeof(Complex)) == 0,
              "heap/pool beam spectrum is byte-identical");
        CHECK(memcmp(heap_beam.weights, stat_beam.weights,
                     (size_t)AEC_ARRAY_MICS * (size_t)n_freqs *
                         sizeof(Complex)) == 0,
              "heap/pool beam weights are byte-identical");
        CHECK(array_sample_beam_fuse(heap, &heap_frame) == AEC_OK &&
              array_sample_beam_fuse(stat, &stat_frame) == AEC_OK &&
              memcmp(heap_frame.r2, stat_frame.r2, (size_t)n_freqs * sizeof(float)) == 0 &&
              memcmp(heap_frame.comfort_noise, stat_frame.comfort_noise, (size_t)n_freqs * sizeof(float)) == 0 &&
              memcmp(heap_frame.near_spec, stat_frame.near_spec, (size_t)n_freqs * sizeof(Complex)) == 0 &&
              memcmp(heap_frame.res_gain, stat_frame.res_gain, (size_t)n_freqs * sizeof(float)) == 0,
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
                  heap_info.bf_adaptive == stat_info.bf_adaptive,
              "heap/pool remaining frame info fields match");
    }

    array_sample_beam_destroy(stat);
    CHECK(isnan(array_sample_beam_bf_lambda(stat)) &&
              array_sample_beam_bf_effective_adapt_interval(stat) == -1,
          "destroy marks a caller-pool instance inactive");
    array_sample_beam_destroy(stat);
    CHECK(array_sample_beam_bf_effective_adapt_interval(stat) == -1,
          "caller-pool destroy is idempotent");
    stat = array_sample_beam_init(beam_pool, beam_bytes, &cfg.aec_cfg, &cfg.beam, aec);
    CHECK(stat != NULL, "caller pool is reusable after destroy");

    array_sample_beam_destroy(stat);
    aec_destroy(aec);
    array_test_beam_destroy(heap);
    free(far);
    free(microphones);
    free(pool);
    return 1;
}

/* ---------------------------------------------------------------------------
 * The chain the examples run behind the unit (RES, NR, min, comfort noise, synthesis), against
 * the reference chain written over the array instance (conv_chain_4ch.h).
 * ------------------------------------------------------------------------ */

static void* aligned_block(size_t bytes) {
    void* mem = NULL;
    return posix_memalign(&mem, 16, bytes) == 0 ? mem : NULL;
}

/* The stages the examples carve behind the unit (4ch_static.c): NR, synthesis and the
 * gain/spectrum buffers; the gain, the comfort noise and its generator are the array's
 * (aec_apply_gain / aec_apply_cng). A missing NR reads as the unity vector; without a gain stage
 * (enable_res = 0, an AEC_RESIDUAL_ESTIMATE array) the denoiser gain alone is applied. */
typedef struct ExPost {
    AnrDenoiser* nr;          /* NULL: no denoiser */
    AcIstft* istft;
    float *unity, *extra;
    Complex* spec;
    uint32_t reset_handled;   /* AecResidualView.post_reset_serial the synthesis last followed */
    int n_freqs, enable_res, enable_cng;
    void* mem;
} ExPost;

/* >= 0: ex_post_init() fills the block with this byte before the stages are placed in it. */
static int ex_post_poison = -1;

/* What an application's reset does to the stages it owns: denoiser and synthesis (the unit's own
 * reset covers the array, its gain stage and its comfort-noise generator). */
static void ex_post_reset(ExPost* q) {
    if (q->nr) anr_reset(q->nr);
    ac_istft_reset(q->istft);
}

static void ex_post_free(ExPost* q) {
    if (q->nr) anr_destroy(q->nr);
    free(q->mem);
    memset(q, 0, sizeof(*q));
}

static int ex_post_init(ExPost* q, int sample_rate, int fft_size, int enable_res,
                        int enable_nr, int enable_cng) {
    AnrConfig nc = anr_config_for_level_grid(sample_rate, fft_size, ANR_LEVEL_BALANCED);
    AcStftConfig sc;
    size_t nr_b, syn_b, vec, cvec;
    unsigned char* cur;
    int k;
    memset(q, 0, sizeof(*q));
    q->n_freqs = fft_size / 2 + 1;
    q->enable_res = enable_res;
    q->enable_cng = enable_cng;
    ac_stft_config_default(&sc, fft_size);
    nr_b = enable_nr ? ALIGN16(anr_get_mem_size(&nc)) : 0;
    syn_b = ALIGN16(ac_istft_get_mem_size(&sc));
    vec = ALIGN16((size_t)q->n_freqs * sizeof(float));
    cvec = ALIGN16((size_t)q->n_freqs * sizeof(Complex));
    q->mem = aligned_block(nr_b + syn_b + 2 * vec + cvec);
    if (!q->mem) return -1;
    if (ex_post_poison >= 0) memset(q->mem, ex_post_poison, nr_b + syn_b + 2 * vec + cvec);
    cur = (unsigned char*)q->mem;
    if (enable_nr) { q->nr = anr_init(cur, anr_get_mem_size(&nc), &nc); cur += nr_b; }
    q->istft = ac_istft_init(cur, ac_istft_get_mem_size(&sc), &sc); cur += syn_b;
    q->unity = (float*)cur; cur += vec;
    q->extra = (float*)cur; cur += vec;
    q->spec = (Complex*)cur;
    for (k = 0; k < q->n_freqs; ++k) q->unity[k] = 1.0f;
    return (enable_nr && !q->nr) || !q->istft ? -1 : 0;
}

/* The per-hop chain of the examples on the array behind `aec`. 0, or -1 when a stage refuses. */
static int ex_post_hop(ExPost* q, Aec* aec, const AecResidualView* f, float* out) {
    const float* nr = NULL;
    if (q->nr && !(nr = array_sample_anr_gain(q->nr, f, q->extra, 1))) return -1;
    if (array_sample_post_reset_due(&q->reset_handled, f)) ac_istft_reset(q->istft);
    if (q->enable_res) {
        if (aec_apply_gain(aec, nr, q->spec) != AEC_OK) return -1;
        if (q->enable_cng && aec_apply_cng(aec, nr, q->spec) != AEC_OK) return -1;
    } else {
        aec_apply_gain_spec(q->spec, f->base_spec, q->unity, nr ? nr : q->unity, q->n_freqs);
    }
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
        for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
            mics[(size_t)i * AEC_ARRAY_MICS + (size_t)ch] =
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
        for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
            mics[(size_t)i * AEC_ARRAY_MICS + (size_t)ch] =
                0.5f * env * hard_noise((uint32_t)(idx * 4 + ch) + 0x51ed270bu) + near +
                0.002f * sinf(0.017f * t + (float)ch);
    }
}

/* fuse() publishes the array's residual gain with the beamformed frame: res_gain is the same
 * buffer a second aec_get_residual() view returns, finite and inside [0, 1] on every hop of a busy
 * scene, on every grid, while the inputs of the gain (E, near, R^2, comfort noise, aligned far
 * end) are all present and finite. The lanes' hops and echo estimates are not part of a fused
 * view. */
static int test_fuse_publishes_the_gain(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int grid;
    for (grid = 0; grid < 3; ++grid) {
        ArrayTestUnitConfig cfg = array_test_unit_default_config(grids[grid][0]);
        ArraySampleBeam* p;
        float *mics, *far;
        int hop, n, h, ok = 1;
        char label[160];
        cfg.aec_cfg.fft_size = grids[grid][1];
        p = array_test_beam_create(&cfg);
        CHECK(p != NULL, "fuse frame: the unit builds");
        hop = aec_hop_size(array_test_beam_aec(p));
        n = array_test_n_freqs(array_test_beam_aec(p));
        mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        for (h = 0; h < 120 && ok; ++h) {
            ArraySampleBeamResult beam;
            AecResidualView f, again;
            int k;
            res_scene_hop(h, hop, mics, far);
            memset(&again, 0, sizeof(again));
            again.struct_size = sizeof(again);
            if (array_sample_beam_process(p, mics, far, &beam, NULL) != AEC_OK ||
                array_sample_beam_fuse(p, &f) != AEC_OK ||
                aec_get_residual(array_test_beam_aec(p), &again) != AEC_OK ||
                f.res_gain == NULL || f.res_gain != again.res_gain ||
                f.formed_hop != NULL || f.echo_spec != NULL ||
                !f.error_spec || !f.near_spec || !f.r2 || !f.comfort_noise || !f.far_hop ||
                f.n_freqs != n || f.hop_size != hop) {
                ok = 0;
                break;
            }
            for (k = 0; k < n; ++k)
                if (!isfinite(f.error_spec[k].r) || !isfinite(f.error_spec[k].i) ||
                    !isfinite(f.near_spec[k].r) || !isfinite(f.r2[k]) ||
                    !isfinite(f.comfort_noise[k]) ||
                    !(f.res_gain[k] >= 0.0f && f.res_gain[k] <= 1.0f)) ok = 0;
        }
        snprintf(label, sizeof label,
                 "fuse frame: %d Hz/%d res_gain is the array's gain in [0, 1] on every hop and the "
                 "RES inputs are present", grids[grid][0], grids[grid][1]);
        CHECK(ok, label);
        free(mics); free(far);
        array_test_beam_destroy(p);
    }
    return 1;
}

/* With the BF bypassed the beamformed error is lane 0's error spectrum and the effective weights
 * are [1,0,0,0]; the reference chain fed the same two must reproduce unit + fuse + the examples'
 * chain bit for bit, for every combination of the RES, NR and comfort-noise switches and, with
 * the RES on, at every AEC level. */
static int test_unit_chain_equals_reference_chain(void) {
    int mode, grid, level;
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    for (grid = 0; grid < 3; ++grid) {
        for (mode = 0; mode < 8; ++mode) {
         for (level = 0; level < 4; ++level) {
            ArrayTestUnitConfig cfg = array_test_unit_default_config(grids[grid][0]);
            ConvChainOptions opt = conv_default_options();
            ArraySampleBeam* unit;
            ExPost post;
            ConvChain* ref;
            ArraySampleBeamResult beam;
            AecResidualView frame;
            Complex* weights;
            float *mics, *far, *out_w, *out_r;
            int hop, n, h, i, k, identical = 1, nonzero = 0, ok = 1;
            char label[160];

            if (!((mode >> 1) & 1) && level != (int)AEC_LEVEL_BALANCED) continue;   /* no RES: no level */
            cfg.aec_cfg = array_test_level_config(grids[grid][0], grids[grid][1], (AecLevel)level);
            cfg.aec_cfg.residual = ((mode >> 1) & 1) ? AEC_RESIDUAL_FULL : AEC_RESIDUAL_ESTIMATE;
            cfg.beam.bf_enable = 0;
            opt.enable_nr = mode & 1;
            opt.enable_res = (mode >> 1) & 1;
            opt.enable_cng = (mode >> 2) & 1;
            unit = array_test_beam_create(&cfg);
            ref = conv_create(&cfg.aec_cfg, &opt);
            {   /* The oracle writes the denoiser window out on its own; it must agree with the
                 * examples' denoiser, since the short scene below cannot tell the
                 * noise-tracking window apart. */
                int ghop = grids[grid][1] / 2;
                AnrConfig oc = conv_nr_config(grids[grid][0], grids[grid][1], ghop, ANR_LEVEL_BALANCED);
                AnrConfig ec = anr_config_for_level_grid(grids[grid][0], grids[grid][1], ANR_LEVEL_BALANCED);
                CHECK(memcmp(&oc, &ec, sizeof oc) == 0,
                      "oracle and example agree on the denoiser configuration");
            }
            snprintf(label, sizeof(label), "reference chain: %d Hz/%d nr=%d res=%d(level %d) cng=%d build",
                     grids[grid][0], grids[grid][1], mode & 1, (mode >> 1) & 1, level, (mode >> 2) & 1);
            CHECK(unit && ref &&
                  ex_post_init(&post, grids[grid][0], grids[grid][1], (mode >> 1) & 1,
                               mode & 1, (mode >> 2) & 1) == 0,
                  label);
            hop = aec_hop_size(array_test_beam_aec(unit));
            n = array_test_n_freqs(array_test_beam_aec(unit));
            mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
            far = (float*)calloc((size_t)hop, sizeof(float));
            out_w = (float*)calloc((size_t)hop, sizeof(float));
            out_r = (float*)calloc((size_t)hop, sizeof(float));
            weights = (Complex*)calloc((size_t)AEC_ARRAY_MICS * (size_t)n, sizeof(Complex));
            for (k = 0; k < n; ++k) weights[k].r = 1.0f;
            /* Past the denoiser's initialisation window, so its gain shapes the output. */
            for (h = 0; h < 340 && ok; ++h) {
                res_scene_hop(h, hop, mics, far);   /* nonlinear echo + double talk: the RES cuts */
                if (array_sample_beam_process_with_activity(unit, mics, far, 0, 0, NULL, &beam, NULL) !=
                        AEC_OK ||
                    array_sample_beam_fuse(unit, &frame) != AEC_OK ||
                    ex_post_hop(&post, array_test_beam_aec(unit), &frame, out_w) != 0 ||
                    array_test_front(ref->aec, mics, far, &ref->front) != AEC_OK ||
                    conv_post(ref, weights, ref->front.lane_error_spec[0], out_r) != AEC_OK) {
                    ok = 0;
                    break;
                }
                if (memcmp(out_w, out_r, (size_t)hop * sizeof(float)) != 0) identical = 0;
                for (i = 0; i < hop; ++i) if (out_r[i] != 0.0f) nonzero = 1;
            }
            snprintf(label, sizeof(label),
                     "reference chain: %d Hz/%d nr=%d res=%d(level %d) cng=%d unit + chain output is bit-identical",
                     grids[grid][0], grids[grid][1], mode & 1, (mode >> 1) & 1, level, (mode >> 2) & 1);
            CHECK(ok && identical && nonzero, label);
            free(mics); free(far); free(out_w); free(out_r); free(weights);
            array_test_beam_destroy(unit);
            ex_post_free(&post);
            conv_destroy(ref);
         }
        }
    }
    return 1;
}

/* array_sample_stages_chain (the stages the 4ch examples carve behind the unit) against the reference chain over the same
 * front end, across hard resets of the array in mid-run: the synthesis follows AecResidualView.post_reset_serial (its
 * overlap-add tail is cleared on the hop the serial moves), as the reference chain does. At least one reset event must
 * have been seen. */
static int test_stages_chain_follows_the_post_reset(void) {
    static const int grids[2][2] = {{16000, 256}, {48000, 1024}};
    static const int reset_hops[] = {120, 121, 200};
    int grid;
    for (grid = 0; grid < 2; ++grid) {
        ArrayTestUnitConfig cfg = array_test_unit_default_config(grids[grid][0]);
        ConvChainOptions opt = conv_default_options();
        ArraySampleBeam* unit;
        ArraySampleStages ps;
        ConvChain* ref;
        AecResidualView frame;
        Complex* weights;
        float *mics, *far, *out_w, *out_r;
        void* mem = NULL;
        uint32_t serial = 0u;
        int hop, n, h, k, r, identical = 1, nonzero = 0, ok = 1, events = 0, built;
        char label[160];

        cfg.aec_cfg = array_test_level_config(grids[grid][0], grids[grid][1], AEC_LEVEL_BALANCED);
        cfg.aec_cfg.residual = AEC_RESIDUAL_FULL;
        cfg.beam.bf_enable = 0;
        opt.enable_nr = 1;
        opt.enable_res = 1;
        opt.enable_cng = 1;
        unit = array_test_beam_create(&cfg);
        ref = conv_create(&cfg.aec_cfg, &opt);
        built = unit && ref && array_sample_stages_plan(&ps, grids[grid][0], grids[grid][1]) == 0 &&
                posix_memalign(&mem, 16, ps.total) == 0 && array_sample_stages_init(&ps, mem) == 0;
        snprintf(label, sizeof(label), "stages chain: %d Hz/%d build", grids[grid][0], grids[grid][1]);
        CHECK(built, label);
        if (!built) { free(mem); array_test_beam_destroy(unit); conv_destroy(ref); continue; }
        hop = aec_hop_size(array_test_beam_aec(unit));
        n = array_test_n_freqs(array_test_beam_aec(unit));
        mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        out_w = (float*)calloc((size_t)hop, sizeof(float));
        out_r = (float*)calloc((size_t)hop, sizeof(float));
        weights = (Complex*)calloc((size_t)AEC_ARRAY_MICS * (size_t)n, sizeof(Complex));
        for (k = 0; k < n; ++k) weights[k].r = 1.0f;
        for (h = 0; h < 340 && ok; ++h) {
            for (r = 0; r < (int)(sizeof reset_hops / sizeof reset_hops[0]); ++r) {
                if (h == reset_hops[r]) {   /* the application resets the array only: stages follow the serial */
                    aec_reset(array_test_beam_aec(unit));
                    aec_reset(ref->aec);
                }
            }
            res_scene_hop(h, hop, mics, far);
            if (array_sample_beam_process_with_activity(unit, mics, far, 0, 0, NULL, &(ArraySampleBeamResult){0}, NULL) !=
                    AEC_OK ||
                array_sample_beam_fuse(unit, &frame) != AEC_OK ||
                array_sample_stages_chain(&ps, array_test_beam_aec(unit), &frame, out_w) != 0 ||
                array_test_front(ref->aec, mics, far, &ref->front) != AEC_OK ||
                conv_post(ref, weights, ref->front.lane_error_spec[0], out_r) != AEC_OK) {
                ok = 0;
                break;
            }
            if (frame.post_reset_serial != serial) { serial = frame.post_reset_serial; ++events; }
            if (memcmp(out_w, out_r, (size_t)hop * sizeof(float)) != 0) identical = 0;
            for (k = 0; k < hop; ++k) if (out_r[k] != 0.0f) nonzero = 1;
        }
        snprintf(label, sizeof(label), "stages chain: %d Hz/%d bit-identical to the reference chain across %d reset events",
                 grids[grid][0], grids[grid][1], events);
        CHECK(ok && identical && nonzero && events >= 1, label);
        free(mics); free(far); free(out_w); free(out_r); free(weights);
        anr_destroy(ps.nr);
        free(mem);
        array_test_beam_destroy(unit);
        conv_destroy(ref);
    }
    return 1;
}

/* The unit alone behind the RES scene: one hop through process + fuse; the array's gain, or NULL
 * on a refusal. */
typedef struct ResRig {
    ArraySampleBeam* unit;
} ResRig;

static int res_rig_init(ResRig* r, int sample_rate, int fft_size, AecLevel level) {
    ArrayTestUnitConfig cfg = array_test_unit_default_config(sample_rate);
    cfg.aec_cfg = array_test_level_config(sample_rate, fft_size, level);
    r->unit = array_test_beam_create(&cfg);
    return r->unit ? 0 : -1;
}

static void res_rig_free(ResRig* r) {
    array_test_beam_destroy(r->unit);
}

/* What an application's reset does: the unit, which resets the array and its gain stage. */
static void res_rig_reset(ResRig* r) {
    array_sample_beam_reset(r->unit);
}

static Aec* res_rig_aec(ResRig* r) { return array_test_beam_aec(r->unit); }

static const float* res_rig_hop(ResRig* r, int h, float* mics, float* far) {
    ArraySampleBeamResult beam;
    AecResidualView f;
    res_scene_hop(h, aec_hop_size(array_test_beam_aec(r->unit)), mics, far);
    if (array_sample_beam_process(r->unit, mics, far, &beam, NULL) != AEC_OK ||
        array_sample_beam_fuse(r->unit, &f) != AEC_OK)
        return NULL;
    return f.res_gain;
}

/* The array's gain stage across reset: a used and reset rig equals a fresh one on the gain, and a
 * level set with aec_set_level() survives the reset and then equals a rig built at that level
 * (while the two levels give different gains on this scene). */
static int test_array_gain_reset_and_level(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int g;
    for (g = 0; g < 3; ++g) {
        ResRig used, fresh, retarget, built;
        float *mics, *far;
        int h, n, hop, same = 1, same_level = 1, differs = 0, ok;
        char label[160];
        ok = res_rig_init(&used, grids[g][0], grids[g][1], AEC_LEVEL_BALANCED) == 0 &&
             res_rig_init(&fresh, grids[g][0], grids[g][1], AEC_LEVEL_BALANCED) == 0 &&
             res_rig_init(&retarget, grids[g][0], grids[g][1], AEC_LEVEL_BALANCED) == 0 &&
             res_rig_init(&built, grids[g][0], grids[g][1], AEC_LEVEL_AGGRESSIVE) == 0;
        CHECK(ok, "array gain: rigs build");
        hop = aec_hop_size(array_test_beam_aec(used.unit));
        n = array_test_n_freqs(array_test_beam_aec(used.unit));
        mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        for (h = 0; h < 150; ++h) {
            (void)res_rig_hop(&used, h + 1000, mics, far);
            (void)res_rig_hop(&retarget, h + 1000, mics, far);
        }
        res_rig_reset(&used);
        CHECK(aec_set_level(res_rig_aec(&retarget), AEC_LEVEL_AGGRESSIVE) == 0,
              "array gain: a level is accepted");
        res_rig_reset(&retarget);
        for (h = 0; h < 250; ++h) {
            static float keep_u[513], keep_r[513];
            const float *gu = res_rig_hop(&used, h, mics, far), *gf, *gr, *gb;
            if (!gu) { same = 0; break; }
            memcpy(keep_u, gu, (size_t)n * sizeof(float));
            gf = res_rig_hop(&fresh, h, mics, far);
            gr = res_rig_hop(&retarget, h, mics, far);
            if (!gr) { same_level = 0; break; }
            memcpy(keep_r, gr, (size_t)n * sizeof(float));
            gb = res_rig_hop(&built, h, mics, far);
            if (!gf || memcmp(keep_u, gf, (size_t)n * sizeof(float)) != 0) same = 0;
            if (!gb || memcmp(keep_r, gb, (size_t)n * sizeof(float)) != 0) same_level = 0;
            if (memcmp(keep_u, keep_r, (size_t)n * sizeof(float)) != 0) differs = 1;
        }
        snprintf(label, sizeof label,
                 "array gain: %d Hz/%d a used rig after reset gives the gain of a fresh rig bit for bit",
                 grids[g][0], grids[g][1]);
        CHECK(same, label);
        snprintf(label, sizeof label,
                 "array gain: %d Hz/%d a level set before reset survives it and equals a rig built "
                 "at that level", grids[g][0], grids[g][1]);
        CHECK(same_level, label);
        snprintf(label, sizeof label, "array gain: %d Hz/%d balanced and aggressive differ on this scene",
                 grids[g][0], grids[g][1]);
        CHECK(differs, label);
        free(mics); free(far);
        res_rig_free(&used); res_rig_free(&fresh); res_rig_free(&retarget); res_rig_free(&built);
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * The whole chain of an example (unit, gain, NR, comfort noise, synthesis) as one rig, with the
 * unit built either on the heap (create) or in a caller pool (init_ex).
 * ------------------------------------------------------------------------ */

typedef struct ChainRig {
    ArraySampleBeam* unit;
    void* unit_pool;          /* non-NULL: array + helper placed in this poisoned pool */
    Aec* aec;
    ExPost post;
} ChainRig;

static int chain_rig_init(ChainRig* r, int sample_rate, int fft_size, int in_pool) {
    ArrayTestUnitConfig cfg = array_test_unit_default_config(sample_rate);
    AecMemReq aec_req;
    size_t aec_bytes, beam_bytes;
    memset(r, 0, sizeof(*r));
    cfg.aec_cfg.fft_size = fft_size;
    if (in_pool) {
        if (aec_get_mem_requirements(&cfg.aec_cfg, &aec_req) != 0 ||
            array_sample_beam_get_mem_size(&cfg.aec_cfg, &cfg.beam, &beam_bytes) != 0)
            return -1;
        aec_bytes = ALIGN16((size_t)aec_req.bytes);
        if (posix_memalign(&r->unit_pool, 16u, aec_bytes + beam_bytes) != 0) return -1;
        memset(r->unit_pool, 0xa5, aec_bytes + beam_bytes);
        r->aec = aec_init_ex(r->unit_pool, aec_bytes, &cfg.aec_cfg, &aec_req);
        if (!r->aec) return -1;
        r->unit = array_sample_beam_init((unsigned char*)r->unit_pool + aec_bytes, beam_bytes,
                                        &cfg.aec_cfg, &cfg.beam, r->aec);
    } else {
        r->unit = array_test_beam_create(&cfg);
    }
    ex_post_poison = in_pool ? 0xa5 : -1;
    {
        int rc = ex_post_init(&r->post, sample_rate, fft_size, 1, 1, 1);
        ex_post_poison = -1;
        if (rc != 0) return -1;
    }
    return r->unit ? 0 : -1;
}

static void chain_rig_free(ChainRig* r) {
    if (r->unit_pool) {
        array_sample_beam_destroy(r->unit);
        aec_destroy(r->aec);
    } else {
        array_test_beam_destroy(r->unit);
    }
    free(r->unit_pool);
    ex_post_free(&r->post);
}

/* The unit's reset covers the array (gain stage, comfort-noise generator); reset_stages also
 * clears the denoiser and the synthesis, as an application's reset does. */
static void chain_rig_reset(ChainRig* r, int reset_stages) {
    array_sample_beam_reset(r->unit);
    if (reset_stages) ex_post_reset(&r->post);
}

/* 0 on success; `out` is the hop's output. */
static int chain_rig_hop(ChainRig* r, int h, float* mics, float* far, float* out) {
    ArraySampleBeamResult beam;
    AecResidualView f;
    Aec* aec = r->unit_pool ? r->aec : array_test_beam_aec(r->unit);
    res_scene_hop(h, aec_hop_size(aec), mics, far);
    if (array_sample_beam_process(r->unit, mics, far, &beam, NULL) != AEC_OK ||
        array_sample_beam_fuse(r->unit, &f) != AEC_OK)
        return -1;
    return ex_post_hop(&r->post, aec, &f, out);
}

/* After array_sample_beam_reset() plus the application's own resets (denoiser, synthesis) the
 * whole chain equals a fresh instance bit for bit; the same with the denoiser and synthesis left
 * unreset must differ, so the test would catch an application (or an example) that forgets it. */
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
        hop = aec_hop_size(array_test_beam_aec(used.unit));
        mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
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
                 "chain reset: %d Hz/%d without the denoiser and synthesis reset the chain does NOT equal a fresh one",
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
        hop = aec_hop_size(array_test_beam_aec(heap.unit));
        mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
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

/* aec_set_level(array, level) in an echo scene where the gain acts (an echo the linear
 * filters cannot model + double talk, many bins cut, denoiser off so the gain is not masked): the
 * chain output follows an independent oracle (conv_chain_4ch.h: its own array, the same call at
 * the same hop, the chain on explicit arrays) bit for bit; the run without the call differs from
 * both, so a skipped call is caught. */
static int test_array_level_ramp_vs_oracle(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    enum { SET_AT = 200, HOPS = 520 };
    int g;
    for (g = 0; g < 3; ++g) {
        ArrayTestUnitConfig cfg = array_test_unit_default_config(grids[g][0]);
        ConvChainOptions opt = conv_default_options();
        ArraySampleBeam *u_set, *u_skip, *u_ref;
        ExPost p_set, p_skip;
        ConvChain* ref;
        Complex* weights;
        float *mics, *far, *o_set, *o_skip, *o_ref;
        long cut_before = 0, cut_after = 0;   /* (hop, bin) pairs the gain cut below 0.9 */
        int hop, n, h, k, i, ok = 1, set_matches = 1, skip_differs = 0, nonzero = 0;
        char label[200];
        cfg.aec_cfg.fft_size = grids[g][1];
        cfg.beam.bf_enable = 0;
        opt.enable_nr = 0;
        opt.enable_cng = 1;
        u_set = array_test_beam_create(&cfg);
        u_skip = array_test_beam_create(&cfg);
        u_ref = array_test_beam_create(&cfg);
        ref = conv_create(&cfg.aec_cfg, &opt);
        CHECK(u_set && u_skip && u_ref && ref &&
                  ex_post_init(&p_set, grids[g][0], grids[g][1], 1, 0, 1) == 0 &&
                  ex_post_init(&p_skip, grids[g][0], grids[g][1], 1, 0, 1) == 0,
              "level ramp: units, oracle and stages build");
        hop = aec_hop_size(array_test_beam_aec(u_set));
        n = array_test_n_freqs(array_test_beam_aec(u_set));
        mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        o_set = (float*)calloc((size_t)hop, sizeof(float));
        o_skip = (float*)calloc((size_t)hop, sizeof(float));
        o_ref = (float*)calloc((size_t)hop, sizeof(float));
        weights = (Complex*)calloc((size_t)AEC_ARRAY_MICS * (size_t)n, sizeof(Complex));
        for (k = 0; k < n; ++k) weights[k].r = 1.0f;
        for (h = 0; h < HOPS && ok; ++h) {
            ArraySampleBeamResult beam;
            AecResidualView fs, fk;
            const float* gain;
            int cut = 0;
            if (h == SET_AT) {
                if (aec_set_level(array_test_beam_aec(u_set), AEC_LEVEL_AGGRESSIVE) != 0 ||
                    aec_set_level(ref->aec, AEC_LEVEL_AGGRESSIVE) != 0) { ok = 0; break; }
            }
            res_scene_hard_hop(h, hop, mics, far);
            if (array_sample_beam_process(u_set, mics, far, &beam, NULL) != AEC_OK ||
                array_sample_beam_fuse(u_set, &fs) != AEC_OK ||
                ex_post_hop(&p_set, array_test_beam_aec(u_set), &fs, o_set) != 0 ||
                array_sample_beam_process(u_skip, mics, far, &beam, NULL) != AEC_OK ||
                array_sample_beam_fuse(u_skip, &fk) != AEC_OK ||
                ex_post_hop(&p_skip, array_test_beam_aec(u_skip), &fk, o_skip) != 0 ||
                array_sample_beam_process(u_ref, mics, far, &beam, NULL) != AEC_OK ||
                array_test_front(ref->aec, mics, far, &ref->front) != AEC_OK ||
                conv_post(ref, weights, ref->front.lane_error_spec[0], o_ref) != AEC_OK) {
                ok = 0;
                break;
            }
            /* denoiser off: total = min(1, G_res) = G_res */
            gain = fs.res_gain;
            for (k = 0; k < n; ++k) if (gain[k] < 0.9f) ++cut;
            if (h >= 100 && h < SET_AT) cut_before += cut;   /* after the initial-state phase */
            if (h >= SET_AT) cut_after += cut;
            if (memcmp(o_set, o_ref, (size_t)hop * sizeof(float)) != 0) set_matches = 0;
            if (h >= SET_AT && memcmp(o_skip, o_ref, (size_t)hop * sizeof(float)) != 0) skip_differs = 1;
            if (h < SET_AT && memcmp(o_skip, o_set, (size_t)hop * sizeof(float)) != 0) set_matches = 0;
            for (i = 0; i < hop; ++i) if (o_ref[i] != 0.0f) nonzero = 1;
        }
        snprintf(label, sizeof label,
                 "level ramp: %d Hz/%d the gain acts in the scene beyond its initial state (gain "
                 "below 0.9 on many bins, before and after the change)", grids[g][0], grids[g][1]);
        CHECK(ok && cut_before > 2L * n && cut_after > 10L * n, label);
        snprintf(label, sizeof label,
                 "level ramp: %d Hz/%d output with the ramped level equals the oracle bit for "
                 "bit on every hop", grids[g][0], grids[g][1]);
        CHECK(set_matches && nonzero, label);
        snprintf(label, sizeof label,
                 "level ramp: %d Hz/%d the run with the level call skipped differs from the "
                 "oracle", grids[g][0], grids[g][1]);
        CHECK(skip_differs, label);
        free(mics); free(far); free(o_set); free(o_skip); free(o_ref); free(weights);
        array_test_beam_destroy(u_set); array_test_beam_destroy(u_skip);
        array_test_beam_destroy(u_ref);
        ex_post_free(&p_set); ex_post_free(&p_skip);
        conv_destroy(ref);
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * 4ch_raw as a black box: its output file equals the reference chain (own array
 * instance, denoiser, comfort noise, synthesis) fed the beam of the unit on the same input,
 * bit for bit.
 * ------------------------------------------------------------------------ */

static int raw_case(const char* bin, const char* tmp, int sample_rate, int fft_size, int fixed_doa) {
    char mic_path[512], ref_path[512], out_path[512], cmd[2048];
    ArrayTestUnitConfig cfg = array_test_unit_default_config(sample_rate);
    ConvChainOptions opt = conv_default_options();
    ArraySampleBeam* unit;
    ConvChain* ref;
    FILE *fm, *fr, *fo;
    float *mics, *far, *want, *got;
    int hop, h, hops = 300, ok = 1, nonzero = 0;
    snprintf(mic_path, sizeof mic_path, "%s/raw_mic.f32", tmp);
    snprintf(ref_path, sizeof ref_path, "%s/raw_ref.f32", tmp);
    snprintf(out_path, sizeof out_path, "%s/raw_out.f32", tmp);
    cfg.aec_cfg.fft_size = fft_size;
    if (fixed_doa) {
        cfg.beam.bf_fixed_mode = 1;
        cfg.beam.bf_fixed_doa_rad = 45.0f * (float)(3.14159265358979323846 / 180.0);   /* as the CLI */
    }
    unit = array_test_beam_create(&cfg);
    ref = conv_create(&cfg.aec_cfg, &opt);
    if (!unit || !ref) return -1;
    hop = aec_hop_size(array_test_beam_aec(unit));
    mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    want = (float*)calloc((size_t)hop, sizeof(float));
    got = (float*)calloc((size_t)hop, sizeof(float));
    fm = fopen(mic_path, "wb");
    fr = fopen(ref_path, "wb");
    if (!fm || !fr) return -1;
    for (h = 0; h < hops; ++h) {
        res_scene_hard_hop(h, hop, mics, far);
        fwrite(mics, sizeof(float), (size_t)hop * AEC_ARRAY_MICS, fm);
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
        ArraySampleBeamResult beam;
        int i;
        res_scene_hard_hop(h, hop, mics, far);
        if (fread(got, sizeof(float), (size_t)hop, fo) != (size_t)hop ||
            array_sample_beam_process(unit, mics, far, &beam, NULL) != AEC_OK ||
            conv_hop(ref, mics, far, beam.weights, beam.spectrum, want) != AEC_OK) {
            ok = 0;
            break;
        }
        if (memcmp(want, got, (size_t)hop * sizeof(float)) != 0) ok = 0;
        for (i = 0; i < hop; ++i) if (want[i] != 0.0f) nonzero = 1;
    }
    if (ok && fread(got, sizeof(float), 1, fo) != 0) ok = 0;   /* nothing beyond the hops */
    fclose(fo);
    free(mics); free(far); free(want); free(got);
    array_test_beam_destroy(unit);
    conv_destroy(ref);
    return ok && nonzero ? 1 : 0;
}

static int test_raw_runner_black_box(const char* bin_dir, const char* tmp) {
    static const int cases[4][3] = {{16000, 256, 0}, {16000, 512, 0}, {48000, 1024, 0}, {16000, 256, 1}};
    char bin[512], label[160];
    int c;
    snprintf(bin, sizeof bin, "%s/4ch_raw", bin_dir);
    for (c = 0; c < 4; ++c) {
        snprintf(label, sizeof label,
                 "black box: 4ch_raw %d Hz/%d%s equals the reference chain bit for bit",
                 cases[c][0], cases[c][1], cases[c][2] ? " fixed DOA" : "");
        CHECK(raw_case(bin, tmp, cases[c][0], cases[c][1], cases[c][2]) == 1, label);
    }
    return 1;
}

/* ============================================================================
 * Unit contract: reset, array view, the front-only profile
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
        for (int ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
            microphones[i * AEC_ARRAY_MICS + ch] =
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
        for (int ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
            int64_t u = t - 2 * ch;
            microphones[i * AEC_ARRAY_MICS + ch] +=
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
    s->mics = (float*)malloc((size_t)hop * AEC_ARRAY_MICS * sizeof(float));
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
static int same_info(const ArraySampleBeamInfo* a, const ArraySampleBeamInfo* b) {
    return a->frame_index == b->frame_index &&
           a->delay.delay_samples == b->delay.delay_samples &&
           a->delay.solid == b->delay.solid && a->delay.changed == b->delay.changed &&
           a->delay.estimator_calls == b->delay.estimator_calls &&
           bits_equal_f(a->delay.confidence, b->delay.confidence) &&
           bits_equal_f(a->doa_raw_rad, b->doa_raw_rad) &&
           bits_equal_f(a->doa_smooth_rad, b->doa_smooth_rad) &&
           bits_equal_f(a->doa_used_rad, b->doa_used_rad) &&
           a->vad_raw == b->vad_raw && a->vad_out == b->vad_out &&
           a->bf_adaptive == b->bf_adaptive;
}

static ArrayTestUnitConfig split_config(void) {
    ArrayTestUnitConfig cfg = array_test_unit_default_config(16000);
    cfg.aec_cfg.fft_size = 512;
    cfg.aec_cfg.residual = AEC_RESIDUAL_STATE;
    cfg.beam.bf_mu = 0.02f;
    cfg.beam.doa_update_interval = 1;
    return cfg;
}

/* Front and beam as two calls. Returns 1 when both succeeded. */
static int split_hop(ArraySampleBeam* p, const float* mics, const float* far,
                     int vad_raw, int vad_out, const int* mask,
                     ArraySampleBeamResult* beam, ArraySampleBeamInfo* info) {
    AecArrayView front;
    if (array_sample_beam_process_front(p, mics, far, &front) != AEC_OK) return 0;
    return array_sample_beam_process_beam(p, &front, vad_raw, vad_out, mask, beam, info) ==
           AEC_OK;
}

/* The front view's aligned reference is the delayed far, whole-hop exact, on every
 * hop; the delay state it carries agrees with the unit's frame info. */
static int test_front_aligned_ref(void) {
    enum { FRAMES = 240, TRUE_DELAY = 400 };
    ArrayTestUnitConfig cfg = array_test_unit_default_config(16000);
    ArraySampleBeam* p;
    ArraySampleBeamResult beam;
    ArraySampleBeamInfo info;
    AecArrayView pre;
    float* microphones;
    float* far;
    float* far_hist;
    int hop;
    int locked = 0;
    int aligned_checked = 0;

    cfg.aec_cfg.fft_size = 512;
    p = array_test_beam_create(&cfg);
    CHECK(p != NULL, "create 512/256 unit");
    hop = aec_hop_size(array_test_beam_aec(p));
    CHECK(hop == 256, "unit hop is 256");

    microphones = (float*)malloc(
        (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && far_hist, "allocate front-aligned-ref buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        /* Zero the whole out-struct first: if process_front ever stopped
         * assigning aligned_ref, the non-NULL check below would fail
         * instead of reading leftover stack garbage. */
        memset(&pre, 0, sizeof(pre));
        CHECK(array_sample_beam_process_front(
                  p, microphones, far, &pre) == AEC_OK,
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

        CHECK(array_sample_beam_process_beam(
                  p, &pre, 0, 0, NULL, &beam, &info) == AEC_OK &&
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
    array_test_beam_destroy(p);
    return 1;
}

/* process_with_activity() against front + beam(v, v, NULL): a null mask and a single flag
 * reproduce it. */
static int test_split_null_mask_equals_begin(void) {
    ArrayTestUnitConfig cfg = split_config();
    ArraySampleBeam *a, *b;
    SplitScene s;
    int hop, nf, ok = 1, nonzero = 0;
    const int frames = 90;

    a = array_test_beam_create(&cfg);
    b = array_test_beam_create(&cfg);
    CHECK(a && b, "split twin: instances create");
    hop = aec_hop_size(array_test_beam_aec(a));
    nf = array_test_n_freqs(array_test_beam_aec(a));
    CHECK(scene_open(&s, hop, frames), "split twin: scene allocates");
    for (int f = 0; f < frames && ok; ++f) {
        ArraySampleBeamResult ba, bb;
        ArraySampleBeamInfo ia, ib;
        int vad = (f / 7) % 2;
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        ok = array_sample_beam_process_with_activity(a, s.mics, s.far, vad, vad, NULL, &ba, &ia) ==
                 AEC_OK &&
             split_hop(b, s.mics, s.far, vad, vad, NULL, &bb, &ib) &&
             memcmp(ba.spectrum, bb.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
             memcmp(ba.weights, bb.weights,
                    (size_t)AEC_ARRAY_MICS * (size_t)nf * sizeof(Complex)) == 0 &&
             same_info(&ia, &ib);
        for (int k = 0; k < nf; ++k) if (ba.spectrum[k].r != 0.0f) nonzero = 1;
    }
    CHECK(ok, "split twin: front + beam(v, v, NULL) matches process_with_activity byte for byte, state included");
    CHECK(nonzero, "split twin: the compared output is not silence");
    scene_close(&s);
    array_test_beam_destroy(a);
    array_test_beam_destroy(b);
    return 1;
}

/* A mask and a separate vad_raw / vad_out change what SRP and BF see. */
static int test_split_mask_and_vad_reach_srp_bf(void) {
    ArrayTestUnitConfig cfg = split_config();
    enum { NP = 5 };   /* NULL, all-zero, all-one, even bins, NULL with vad_raw 0 */
    ArraySampleBeam* p[NP];
    int* mask[NP] = {0};
    SplitScene s;
    Complex* out[NP];
    int hop, nf, differs_doa[NP] = {0}, differs_out[NP] = {0};
    int raw_finite[NP] = {0}, adaptive_seen[NP] = {0};
    const int frames = 80;

    for (int i = 0; i < NP; ++i) {
        p[i] = array_test_beam_create(&cfg);
        CHECK(p[i] != NULL, "split mask: instances create");
    }
    hop = aec_hop_size(array_test_beam_aec(p[0]));
    nf = array_test_n_freqs(array_test_beam_aec(p[0]));
    CHECK(scene_open(&s, hop, frames), "split mask: scene allocates");
    for (int i = 1; i < 4; ++i) {
        mask[i] = (int*)malloc((size_t)nf * sizeof(int));
        for (int k = 0; k < nf; ++k)
            mask[i][k] = i == 1 ? 0 : (i == 2 ? 1 : (k % 2 == 0));
    }
    for (int i = 0; i < NP; ++i) out[i] = (Complex*)malloc((size_t)nf * sizeof(Complex));
    for (int f = 0; f < frames; ++f) {
        ArraySampleBeamInfo st[NP];
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        for (int i = 0; i < NP; ++i) {
            ArraySampleBeamResult beam;
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
        array_test_beam_destroy(p[i]);
        free(out[i]);
        free(mask[i]);
    }
    scene_close(&s);
    return 1;
}

/* The frame info against SRP and BF objects built here from the same derivation and fed the
 * same lane spectra: every field, the beamformed spectrum and the weights. */
static int split_state_case(int fixed_bf) {
    ArrayTestUnitConfig cfg = split_config();
    ArraySampleBeam* p;
    cfg.beam.bf_fixed_mode = fixed_bf;
    cfg.beam.bf_fixed_doa_rad = 0.3f;
    p = array_test_beam_create(&cfg);
    SRP_Config sc;
    BF_Config gc;
    float gx[AEC_ARRAY_MICS], gy[AEC_ARRAY_MICS];
    ArrayGeometry geom = { AEC_ARRAY_MICS, gx, gy };
    SRP* srp;
    BF* bf;
    Complex *spec, *weights;
    SplitScene s;
    int *mask;
    int hop, nf, interval, finite_smooth = 0, finite_used = 0;
    int fixed_used_seen = 0;
    const int frames = 70;

    CHECK(p != NULL, "split state: instance creates");
    hop = aec_hop_size(array_test_beam_aec(p));
    nf = array_test_n_freqs(array_test_beam_aec(p));
    for (int m = 0; m < AEC_ARRAY_MICS; ++m) {
        float phi = 2.0f * M_PI * m / AEC_ARRAY_MICS;
        gx[m] = cfg.beam.uca_radius_m * cosf(phi);
        gy[m] = cfg.beam.uca_radius_m * sinf(phi);
    }
    memset(&sc, 0, sizeof(sc));
    sc.M = AEC_ARRAY_MICS;
    sc.F = nf;
    sc.num_angles = cfg.beam.num_angles;
    sc.sr = (float)cfg.aec_cfg.sample_rate;
    sc.NFFT = (float)array_test_fft_size(array_test_beam_aec(p));
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
    gc.lambda = anr_retime_alpha(cfg.beam.bf_lambda, cfg.aec_cfg.sample_rate, hop * interval);
    gc.mu = cfg.beam.bf_mu;
    gc.enable_fix_mode = cfg.beam.bf_fixed_mode;
    gc.fixed_doa_rad = cfg.beam.bf_fixed_doa_rad;
    gc.fixed_align_notebook = cfg.beam.bf_fixed_align_notebook;
    gc.adapt_interval = interval;
    bf = bf_create(AEC_ARRAY_MICS, nf, srp->num_angles, srp->a_array, &gc);
    CHECK(bf != NULL, "split state: lower-level BF creates");
    spec = (Complex*)calloc((size_t)nf, sizeof(Complex));
    weights = (Complex*)calloc((size_t)nf * AEC_ARRAY_MICS, sizeof(Complex));
    mask = (int*)malloc((size_t)nf * sizeof(int));
    for (int k = 0; k < nf; ++k) mask[k] = (k % 3) != 0;
    CHECK(scene_open(&s, hop, frames), "split state: scene allocates");
    for (int f = 0; f < frames; ++f) {
        AecArrayView front;
        ArraySampleBeamResult beam;
        ArraySampleBeamInfo st;
        int vad_raw = (f / 5) % 3 != 0, vad_out = (f / 9) % 2;
        const int* m = (f % 4) == 3 ? NULL : mask;
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        CHECK(array_sample_beam_process_front(p, s.mics, s.far, &front) == AEC_OK,
              "split state: front runs");
        CHECK(front.n_freqs == nf && front.lane_error_spec[0] && front.lane_error_spec[3],
              "split state: front publishes the four lane spectra");
        doa_step(srp, front.lane_error_spec, m, vad_raw, vad_out);
        bf_process_with_weights(bf, front.lane_error_spec, doa_get_smooth(srp),
                                 vad_out ? 0 : 1, m, spec, weights);
        CHECK(array_sample_beam_process_beam(p, &front, vad_raw, vad_out, m, &beam, &st) == AEC_OK,
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
                     (size_t)nf * AEC_ARRAY_MICS * sizeof(Complex)) == 0,
              "split state: the beam weights are the lower-level BF weights");
    }
    CHECK(finite_smooth && finite_used, "split state: the smoothed and used DOA become finite");
    CHECK(!fixed_bf || fixed_used_seen, "split state: a fixed beamformer reports its fixed angle as the used DOA");
    bf_destroy(bf);
    srp_destroy(srp);
    free(spec); free(weights); free(mask);
    scene_close(&s);
    array_test_beam_destroy(p);
    return 1;
}

static int test_split_beam_state_matches_lower_level(void) {
    return split_state_case(0) && split_state_case(1);
}

/* Call order: beam needs a pending front, fuse needs this hop's beam, a new front supersedes
 * whatever is pending, reset drops it; refused calls change nothing. */
static int test_split_sequence_guards(void) {
    ArrayTestUnitConfig cfg = split_config();
    ArrayTestUnitConfig cfg_post = split_config();
    ArraySampleBeam *p, *q, *post;
    AecArrayView front, other, bad;
    ArraySampleBeamResult beam;
    ArraySampleBeamInfo info;
    AecResidualView frame;
    float *mics, *far;
    int hop, nf;

    cfg_post.aec_cfg.residual = AEC_RESIDUAL_FULL;
    p = array_test_beam_create(&cfg);
    q = array_test_beam_create(&cfg);
    post = array_test_beam_create(&cfg_post);
    CHECK(p && q && post, "split guards: instances create");
    hop = aec_hop_size(array_test_beam_aec(p));
    nf = array_test_n_freqs(array_test_beam_aec(p));
    mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    memset(&bad, 0, sizeof(bad));
    CHECK(array_sample_beam_process_beam(p, &bad, 0, 0, NULL, &beam, &info) == AEC_ERR_ARG,
          "split guards: a front with the wrong bin count or NULL spectra is refused");
    CHECK(array_sample_beam_process_front(p, mics, far, NULL) == AEC_ERR_ARG &&
          array_sample_beam_process_front(NULL, mics, far, &front) == AEC_ERR_ARG &&
          array_sample_beam_process_front(p, NULL, far, &front) == AEC_ERR_ARG &&
          array_sample_beam_process_front(p, mics, NULL, &front) == AEC_ERR_ARG,
          "split guards: NULL arguments of process_front are refused");
    CHECK(array_sample_beam_process_with_activity(p, mics, far, 2, 0, NULL, &beam, &info) ==
              AEC_ERR_ARG &&
          array_sample_beam_process_with_activity(p, mics, far, 0, 0, NULL, NULL, &info) ==
              AEC_ERR_ARG,
          "split guards: a non-boolean flag or a NULL beam is refused before any state moves");

    /* A real front of another instance: valid in shape, but p has none pending. */
    CHECK(array_sample_beam_process_front(q, mics, far, &other) == AEC_OK,
          "split guards: the other instance's front runs");
    CHECK(array_sample_beam_process_beam(p, &other, 0, 0, NULL, &beam, &info) == AEC_ERR_SEQUENCE,
          "split guards: beam without a pending front is a sequence error");
    CHECK(array_sample_beam_fuse(post, &frame) == AEC_ERR_SEQUENCE,
          "split guards: fuse without a beam is a sequence error");

    CHECK(array_sample_beam_process_front(post, mics, far, &front) == AEC_OK &&
          array_sample_beam_fuse(post, &frame) == AEC_ERR_SEQUENCE,
          "split guards: fuse between front and beam is a sequence error");
    bad = front;
    bad.n_freqs = nf + 1;
    CHECK(array_sample_beam_process_beam(post, &bad, 0, 0, NULL, &beam, &info) == AEC_ERR_ARG &&
          array_sample_beam_process_beam(post, &front, 2, 0, NULL, &beam, &info) == AEC_ERR_ARG &&
          array_sample_beam_process_beam(post, &front, 0, 2, NULL, &beam, &info) == AEC_ERR_ARG &&
          array_sample_beam_process_beam(post, NULL, 0, 0, NULL, &beam, &info) == AEC_ERR_ARG &&
          array_sample_beam_process_beam(post, &front, 0, 0, NULL, NULL, &info) == AEC_ERR_ARG,
          "split guards: a refused beam call changes nothing");
    CHECK(array_sample_beam_process_beam(post, &front, 0, 0, NULL, &beam, NULL) == AEC_OK &&
          array_sample_beam_process_beam(post, &front, 0, 0, NULL, &beam, &info) == AEC_ERR_SEQUENCE,
          "split guards: the beam half runs once per hop; info is optional");
    CHECK(array_sample_beam_fuse(post, NULL) == AEC_ERR_ARG &&
          array_sample_beam_fuse(NULL, &frame) == AEC_ERR_ARG &&
          array_sample_beam_fuse(post, &frame) == AEC_OK &&
          frame.error_spec == beam.spectrum,
          "split guards: fuse follows this hop's beam");
    CHECK(array_sample_beam_process_front(post, mics, far, &front) == AEC_OK &&
          array_sample_beam_fuse(post, &frame) == AEC_ERR_SEQUENCE,
          "split guards: a new front supersedes the beam of the last hop");
    CHECK(array_sample_beam_process_front(post, mics, far, &front) == AEC_OK &&
          array_sample_beam_process_beam(post, &front, 0, 0, NULL, &beam, &info) == AEC_OK,
          "split guards: a second front without a beam in between is legal");
    array_sample_beam_reset(post);
    CHECK(array_sample_beam_process_beam(post, &front, 0, 0, NULL, &beam, &info) == AEC_ERR_SEQUENCE &&
          array_sample_beam_fuse(post, &frame) == AEC_ERR_SEQUENCE,
          "split guards: a reset drops the pending front and beam");

    /* The front-only profile (a neural post-filter reads the beam spectrum): it builds, runs,
     * and has no fuse (AEC_ERR_MODE); a refused fuse leaves the unit as it was. */
    CHECK(array_sample_beam_process_front(p, mics, far, &front) == AEC_OK &&
          array_sample_beam_process_beam(p, &front, 0, 0, NULL, &beam, &info) == AEC_OK &&
          info.frame_index == 0,
          "split guards: the lanes-only unit runs front and beam");
    CHECK(array_sample_beam_fuse(p, &frame) == AEC_ERR_MODE,
          "split guards: fuse is refused on a lanes-only array (AEC_RESIDUAL_STATE)");
    CHECK(array_sample_beam_process_front(p, mics, far, &front) == AEC_OK &&
          array_sample_beam_process_beam(p, &front, 0, 0, NULL, &beam, &info) == AEC_OK &&
          info.frame_index == 1,
          "split guards: the refused fuse did not reset the unit");

    free(mics); free(far);
    array_test_beam_destroy(p);
    array_test_beam_destroy(q);
    array_test_beam_destroy(post);
    return 1;
}

static int same_bytes(const void* a, const void* b, size_t bytes) {
    if (!a || !b) return a == b;
    return memcmp(a, b, bytes) == 0;
}

static int same_frame(const AecResidualView* a, const AecResidualView* b) {
    size_t nf = (size_t)a->n_freqs;
    return a->n_freqs == b->n_freqs && a->hop_size == b->hop_size &&
           same_bytes(a->error_spec, b->error_spec, nf * sizeof(Complex)) &&
           same_bytes(a->base_spec, b->base_spec, nf * sizeof(Complex)) &&
           same_bytes(a->near_spec, b->near_spec, nf * sizeof(Complex)) &&
           same_bytes(a->far_spec, b->far_spec, nf * sizeof(Complex)) &&
           same_bytes(a->r2, b->r2, nf * sizeof(float)) &&
           same_bytes(a->comfort_noise, b->comfort_noise, nf * sizeof(float)) &&
           same_bytes(a->res_gain, b->res_gain, nf * sizeof(float)) &&
           same_bytes(a->far_hop, b->far_hop, (size_t)a->hop_size * sizeof(float)) &&
           a->guard_lanes == b->guard_lanes && a->post_reset_serial == b->post_reset_serial &&
           bits_equal_f(a->max_dt, b->max_dt) &&
           bits_equal_f(a->max_saturation, b->max_saturation) && a->all_converged == b->all_converged &&
           a->usable_linear == b->usable_linear && a->clock_drift == b->clock_drift;
}

/* A hop refused for a non-finite sample changes no adaptive state and leaves nothing pending:
 * the caller's fuse() (or beam) after it is a sequence error that changes nothing, and the
 * stream then continues exactly as if the refused hop had never been offered. */
static int test_refused_hop_then_fuse_keeps_state(void) {
    enum { FRAMES = 60, NAN_HOP = 20, STALE_HOP = 35 };
    ArrayTestUnitConfig cfg = split_config();
    ArraySampleBeam *a, *b;
    SplitScene s;
    float* nan_mics;
    int hop, nf, ok = 1, nonzero = 0;

    cfg.aec_cfg.residual = AEC_RESIDUAL_FULL;
    a = array_test_beam_create(&cfg);
    b = array_test_beam_create(&cfg);
    CHECK(a && b, "refused hop: instances create");
    hop = aec_hop_size(array_test_beam_aec(a));
    nf = array_test_n_freqs(array_test_beam_aec(a));
    CHECK(scene_open(&s, hop, FRAMES), "refused hop: scene allocates");
    nan_mics = (float*)malloc((size_t)hop * AEC_ARRAY_MICS * sizeof(float));
    CHECK(nan_mics != NULL, "refused hop: buffer allocates");
    g_rng = 0x1234567u;
    for (int f = 0; f < FRAMES && ok; ++f) {
        ArraySampleBeamResult ba, bb;
        ArraySampleBeamInfo ia, ib;
        AecResidualView fa, fb;
        int vad = (f / 7) % 2;
        if (f == NAN_HOP) {
            /* Only `a` is offered the bad hop; the scene does not advance for it. */
            memcpy(nan_mics, s.mics, (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
            nan_mics[0] = NAN;
            CHECK(array_sample_beam_process_with_activity(a, nan_mics, s.far, vad, vad, NULL, &ba,
                                                           &ia) == AEC_ERR_ARG,
                  "refused hop: a non-finite microphone sample is refused");
            CHECK(array_sample_beam_fuse(a, &fa) == AEC_ERR_SEQUENCE,
                  "refused hop: fuse after the refused hop is a sequence error");
        }
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        if (f == STALE_HOP) {
            /* Both twins: a front, then a refused front; the first front's view is stale. */
            AecArrayView front, unused;
            ArraySampleBeam* twins[2] = { a, b };
            memcpy(nan_mics, s.mics, (size_t)hop * AEC_ARRAY_MICS * sizeof(float));
            nan_mics[0] = NAN;
            for (int t = 0; t < 2; ++t) {
                CHECK(array_sample_beam_process_front(twins[t], s.mics, s.far, &front) ==
                          AEC_OK &&
                      array_sample_beam_process_front(twins[t], nan_mics, s.far, &unused) ==
                          AEC_ERR_ARG,
                      "refused hop: a refused front after an accepted one");
                CHECK(array_sample_beam_process_beam(twins[t], &front, vad, vad, NULL, &ba, &ia) ==
                          AEC_ERR_SEQUENCE &&
                      array_sample_beam_fuse(twins[t], &fa) == AEC_ERR_SEQUENCE,
                      "refused hop: the refused front leaves no front pending for beam or fuse");
            }
            continue;
        }
        ok = array_sample_beam_process_with_activity(a, s.mics, s.far, vad, vad, NULL, &ba, &ia) ==
                 AEC_OK &&
             array_sample_beam_process_with_activity(b, s.mics, s.far, vad, vad, NULL, &bb, &ib) ==
                 AEC_OK &&
             array_sample_beam_fuse(a, &fa) == AEC_OK &&
             array_sample_beam_fuse(b, &fb) == AEC_OK &&
             memcmp(ba.spectrum, bb.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
             memcmp(ba.weights, bb.weights,
                    (size_t)AEC_ARRAY_MICS * (size_t)nf * sizeof(Complex)) == 0 &&
             same_info(&ia, &ib) && same_frame(&fa, &fb);
        if (f > NAN_HOP)
            for (int k = 0; k < nf; ++k) if (ba.spectrum[k].r != 0.0f) nonzero = 1;
    }
    CHECK(ok, "refused hop: the stream after the refused hop and fuse equals the uninterrupted "
              "twin byte for byte (beam, weights, frame info, fused frame)");
    CHECK(nonzero, "refused hop: the compared output is not silence");
    free(nan_mics);
    scene_close(&s);
    array_test_beam_destroy(a);
    array_test_beam_destroy(b);
    return 1;
}

/* The application resets the array between the beam half and the fuse half of one hop: the library refuses the fuse (its pending
 * front is gone) and the helper, for which a sequence refusal changed nothing, keeps its own state, so the frame counter goes on.
 * The next hop runs and fuses again. */
static int test_array_reset_between_beam_and_fuse(void) {
    enum { FRAMES = 12, RESET_HOP = 6 };
    ArrayTestUnitConfig cfg = split_config();
    ArraySampleBeam* p;
    SplitScene s;
    int hop, ok = 1;

    cfg.aec_cfg.residual = AEC_RESIDUAL_FULL;
    p = array_test_beam_create(&cfg);
    CHECK(p != NULL, "array reset: instance creates");
    hop = aec_hop_size(array_test_beam_aec(p));
    CHECK(scene_open(&s, hop, FRAMES), "array reset: scene allocates");
    g_rng = 0x1234567u;
    for (int f = 0; f < FRAMES && ok; ++f) {
        ArraySampleBeamResult beam;
        ArraySampleBeamInfo info;
        AecResidualView frame;
        int vad = (f / 7) % 2;
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        ok = array_sample_beam_process_with_activity(p, s.mics, s.far, vad, vad, NULL, &beam, &info) == AEC_OK &&
             info.frame_index == (uint64_t)f;
        if (!ok) break;
        if (f == RESET_HOP) {
            aec_reset(array_test_beam_aec(p));
            ok = array_sample_beam_fuse(p, &frame) == AEC_ERR_SEQUENCE &&
                 array_sample_beam_fuse(p, &frame) == AEC_ERR_SEQUENCE;
        } else {
            ok = array_sample_beam_fuse(p, &frame) == AEC_OK;
        }
    }
    CHECK(ok, "array reset: the fuse after an array reset is a sequence error, the frame counter continues (the helper "
              "was not reset) and every other hop fuses");
    scene_close(&s);
    array_test_beam_destroy(p);
    return 1;
}

/* reset() returns the unit to the state of a fresh instance: the same input afterwards gives
 * the same beam bytes and frame info as the first pass did. */
static int test_reset_equals_fresh(void) {
    enum { FRAMES = 40 };
    ArrayTestUnitConfig cfg = split_config();
    ArraySampleBeam *warm, *fresh;
    SplitScene s;
    Complex* first[FRAMES] = {0};
    int hop, nf, ok = 1;

    cfg.aec_cfg.residual = AEC_RESIDUAL_FULL;
    warm = array_test_beam_create(&cfg);
    fresh = array_test_beam_create(&cfg);
    CHECK(warm && fresh, "reset: instances create");
    hop = aec_hop_size(array_test_beam_aec(warm));
    nf = array_test_n_freqs(array_test_beam_aec(warm));
    CHECK(scene_open(&s, hop, FRAMES), "reset: scene allocates");
    for (int pass = 0; pass < 2 && ok; ++pass) {
        g_rng = 0x1234567u;
        memset(s.far_hist, 0, (size_t)FRAMES * hop * sizeof(float));
        memset(s.near_hist, 0, (size_t)FRAMES * hop * sizeof(float));
        if (pass == 1) array_sample_beam_reset(warm);
        for (int f = 0; f < FRAMES && ok; ++f) {
            ArraySampleBeamResult beam;
            ArraySampleBeamInfo info;
            int vad = (f / 7) % 2;
            fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
            ok = array_sample_beam_process_with_activity(warm, s.mics, s.far, vad, vad, NULL, &beam, &info) ==
                 AEC_OK;
            if (!ok) break;
            if (pass == 0) {
                first[f] = (Complex*)malloc((size_t)nf * sizeof(Complex));
                ok = first[f] != NULL;
                if (ok) memcpy(first[f], beam.spectrum, (size_t)nf * sizeof(Complex));
            } else {
                ArraySampleBeamResult fb;
                ArraySampleBeamInfo ref_info;
                ok = array_sample_beam_process_with_activity(fresh, s.mics, s.far, vad, vad, NULL, &fb, &ref_info) ==
                         AEC_OK &&
                     memcmp(first[f], beam.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
                     memcmp(fb.spectrum, beam.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
                     same_info(&info, &ref_info);
            }
        }
    }
    for (int f = 0; f < FRAMES; ++f) free(first[f]);
    CHECK(ok, "reset: a reset unit reproduces its first pass and a fresh instance byte for byte");
    scene_close(&s);
    array_test_beam_destroy(warm);
    array_test_beam_destroy(fresh);
    return 1;
}

/* bf_adapt_interval is bounded (hop * interval stays inside int on every grid), and a cadence
 * long enough to underflow the retimed RLS retention is refused by the size query as by init. */
static int test_adapt_interval_bound(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    int g;
    for (g = 0; g < 3; ++g) {
        ArrayTestUnitConfig cfg = array_test_unit_default_config(grids[g][0]);
        ArraySampleSpatialConfig sc;
        size_t bytes;
        ArraySampleBeam* b;
        cfg.aec_cfg.fft_size = grids[g][1];
        cfg.beam.bf_lambda = 1.0f;    /* retimes to 1 at any cadence */
        cfg.beam.bf_adapt_interval = ARRAY_SAMPLE_BEAM_MAX_ADAPT_INTERVAL;
        CHECK(array_sample_spatial_config(&cfg.aec_cfg, &cfg.beam, &sc) == 0 &&
                  sc.bf.adapt_interval == ARRAY_SAMPLE_BEAM_MAX_ADAPT_INTERVAL &&
                  sc.bf.lambda == 1.0f &&
                  array_sample_beam_get_mem_size(&cfg.aec_cfg, &cfg.beam, &bytes) == 0,
              "adapt interval: the bound is accepted");
        b = array_test_beam_create(&cfg);
        CHECK(b != NULL, "adapt interval: a helper at the bound builds");
        array_test_beam_destroy(b);
        cfg.beam.bf_adapt_interval = ARRAY_SAMPLE_BEAM_MAX_ADAPT_INTERVAL + 1;
        CHECK(array_sample_spatial_config(&cfg.aec_cfg, &cfg.beam, &sc) != 0 &&
                  array_sample_beam_get_mem_size(&cfg.aec_cfg, &cfg.beam, &bytes) != 0 &&
                  array_test_beam_create(&cfg) == NULL,
              "adapt interval: bound + 1 is refused by the query and by init");
        cfg.beam.bf_adapt_interval = 0x7fffffff;
        CHECK(array_sample_spatial_config(&cfg.aec_cfg, &cfg.beam, &sc) != 0,
              "adapt interval: INT_MAX is refused");
    }
    {
        /* 0.5 per 10 ms over 2000 hops of 8 ms: 0.5^1600 underflows to 0 in float. */
        ArrayTestUnitConfig cfg = array_test_unit_default_config(16000);
        size_t bytes;
        cfg.beam.bf_lambda = 0.5f;
        cfg.beam.bf_adapt_interval = 2000;
        CHECK(array_sample_beam_get_mem_size(&cfg.aec_cfg, &cfg.beam, &bytes) != 0 &&
                  array_test_beam_create(&cfg) == NULL,
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
        ArrayTestUnitConfig cfg = array_test_unit_default_config(rows[r].sr);
        ArraySampleSpatialConfig sc;
        ArraySampleBeam* b;
        size_t bytes;
        int query, init;
        cfg.beam.doa_low_freq_hz = rows[r].low;
        cfg.beam.doa_high_freq_hz = rows[r].high;
        query = array_sample_spatial_config(&cfg.aec_cfg, &cfg.beam, &sc) == 0 &&
                array_sample_beam_get_mem_size(&cfg.aec_cfg, &cfg.beam, &bytes) == 0;
        b = array_test_beam_create(&cfg);
        init = b != NULL;
        array_test_beam_destroy(b);
        if (query != rows[r].accept || init != rows[r].accept)
            fprintf(stderr, "  DOA band row %d (sr %d low %g high %g): query %d init %d\n", r,
                    rows[r].sr, (double)rows[r].low, (double)rows[r].high, query, init);
        CHECK(query == rows[r].accept && init == rows[r].accept,
              "DOA band: low against the resolved high, by the query and by init");
    }
    {
        ArrayTestUnitConfig d16 = array_test_unit_default_config(16000);
        ArrayTestUnitConfig d48 = array_test_unit_default_config(48000);
        ArraySampleSpatialConfig sc;
        CHECK(array_sample_spatial_config(&d16.aec_cfg, &d16.beam, &sc) == 0 &&
                  array_sample_spatial_config(&d48.aec_cfg, &d48.beam, &sc) == 0,
              "DOA band: the default band is valid at both rates");
    }
    return 1;
}

static int run_all_tests(const char* bin_dir, const char* tmp) {
    ArrayTestUnitConfig invalid =
        array_test_unit_default_config(16000);
    invalid.aec_cfg.fft_size = 1024;
    CHECK(array_test_beam_create(&invalid) == NULL,
          "16 kHz rejects cross-rate FFT 1024");
    invalid = array_test_unit_default_config(48000);
    invalid.aec_cfg.fft_size = 512;
    CHECK(array_test_beam_create(&invalid) == NULL,
          "48 kHz rejects cross-rate FFT 512");
    invalid = array_test_unit_default_config(16000);
    invalid.aec_cfg.residual = AEC_RESIDUAL_STATE;
    {
        ArraySampleBeam* front_only = array_test_beam_create(&invalid);
        CHECK(front_only != NULL,
              "the unit accepts the front-only profile (no fuse)");
        array_test_beam_destroy(front_only);
    }
    invalid.aec_cfg.residual = (AecResidualMode)99;
    CHECK(array_test_beam_create(&invalid) == NULL,
          "an out-of-range residual mode is rejected");
    invalid.aec_cfg.residual = AEC_RESIDUAL_OFF;
    CHECK(array_test_beam_create(&invalid) == NULL,
          "an array with no residual stage (OFF) is rejected");
    invalid = array_test_unit_default_config(16000);
    invalid.beam.bf_lambda = 1.0f + 1e-6f;
    CHECK(array_test_beam_create(&invalid) == NULL,
          "BF forgetting factor above 1.0 is rejected");
    invalid.beam.bf_lambda = 1.0f;
    {
        ArraySampleBeam* boundary = array_test_beam_create(&invalid);
        CHECK(boundary != NULL,
              "BF forgetting factor of exactly 1.0 is still accepted");
        array_test_beam_destroy(boundary);
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
    CHECK(test_fuse_publishes_the_gain(),
          "fuse publishes the array's residual gain");
    CHECK(test_unit_chain_equals_reference_chain(),
          "unit + fuse + the examples' chain equals the reference chain over the front end");
    CHECK(test_stages_chain_follows_the_post_reset(),
          "the examples' stages follow a post-chain reset like the reference chain");
    CHECK(test_array_gain_reset_and_level(), "the array's gain stage: reset restarts it; a level survives it");
    CHECK(test_array_level_ramp_vs_oracle(),
          "the array's gain stage: a ramped level follows the oracle; skipping it is caught");
    CHECK(test_chain_reset_equals_fresh(), "reset-then-rerun equals a fresh chain");
    CHECK(test_chain_heap_equals_pool(), "heap and pool units give the same chain output");
    CHECK(test_raw_runner_black_box(bin_dir, tmp), "4ch_raw against the oracle");
    CHECK(test_front_aligned_ref(), "front view aligned reference");
    CHECK(test_split_null_mask_equals_begin(),
          "front + beam equals process_with_activity");
    CHECK(test_split_mask_and_vad_reach_srp_bf(),
          "mask and activity flags reach SRP and BF");
    CHECK(test_split_beam_state_matches_lower_level(),
          "beam result and frame info match the lower-level SRP/BF");
    CHECK(test_split_sequence_guards(), "front / beam / fuse call order");
    CHECK(test_refused_hop_then_fuse_keeps_state(),
          "a refused hop followed by fuse changes nothing");
    CHECK(test_array_reset_between_beam_and_fuse(), "an array reset between beam and fuse");
    CHECK(test_reset_equals_fresh(), "reset equals a fresh instance");
    CHECK(test_adapt_interval_bound(), "BF adapt interval bound");
    CHECK(test_doa_band_bounds(), "DOA band bounds");
    printf("All array_sample_beam tests passed (spatial=%s)\n",
           array_sample_beam_spatial_backend());
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <example-bin-dir> <scratch-dir>\n", argv[0]);
        return 2;
    }
    return run_all_tests(argv[1], argv[2]) ? 0 : 1;
}
