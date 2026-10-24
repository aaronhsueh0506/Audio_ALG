/**
 * tests/test_audio_pipeline_4ch.c -- four-channel linear AEC + beamformer unit tests: the
 * public-API acceptance style of the 4aec_nr_res tests plus the SRP-PHAT/BF stage, on all three
 * production signal grids; the split front / beam / fuse sequence; fuse() leaving the residual
 * gain to the caller; the unit followed by the examples' RES / NR / min / comfort noise /
 * synthesis (the RES an AecRes the caller builds with the libaec functions) against the
 * independent oracle (conv_chain_4ch.h); that caller-owned RES across reset and preset changes;
 * and audio_pipeline_4ch_raw run as a black box against that oracle.
 * Arguments: the directory of the example binaries and a scratch directory.
 */

#include "audio_pipeline_4ch.h"
#include "bf.h"
#include "srp.h"
#include "conv_chain_4ch.h"
#include "se_anr_stage.h"

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
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(sample_rate);
    AudioPipeline4Ch* p;
    AudioPipeline4ChFrameInfo info;
    AudioPipeline4ChBeam beam;
    SeSpecFrame frame;
    float* microphones;
    float* far;
    float phase = 0.0f;
    int hop;
    int n_freqs;
    int doa_analysis_total = 0;

    cfg.core.fft_size = fft_size;
    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.4f;
    cfg.bf_mu = 0.02f;
    p = audio_pipeline_4ch_create(&cfg);
    CHECK(p != NULL, "create 4ch linear AEC + BF unit");
    hop = audio_pipeline_4ch_hop_size(p);
    n_freqs = audio_pipeline_4ch_n_freqs(p);
    CHECK(hop == fft_size / 2,
          "rate-specific hop");
    CHECK(audio_pipeline_4ch_frame_size(p) == fft_size &&
          audio_pipeline_4ch_fft_size(p) == fft_size,
          "zero-padding-free frame/FFT contract");
    CHECK(n_freqs == fft_size / 2 + 1,
          "spatial bin count follows selected FFT");
    CHECK(
        audio_pipeline_4ch_doa_sample_rate(p) ==
            sample_rate &&
        audio_pipeline_4ch_doa_frame_size(p) == fft_size &&
        audio_pipeline_4ch_doa_hop_size(p) == fft_size / 2 &&
        audio_pipeline_4ch_doa_fft_size(p) == fft_size,
        "DOA grid follows selected main AEC grid");
    CHECK(
        audio_pipeline_4ch_bf_sample_rate(p) == sample_rate &&
        audio_pipeline_4ch_bf_frame_size(p) == fft_size &&
        audio_pipeline_4ch_bf_hop_size(p) == fft_size / 2 &&
        audio_pipeline_4ch_bf_fft_size(p) == fft_size,
        "BF grid follows selected main AEC grid");
    CHECK(audio_pipeline_4ch_matched_filter_count(p) == 1,
          "one shared matcher");
    CHECK(audio_pipeline_4ch_linear_aec_count(p) ==
              FOUR_AEC_NR_RES_CHANNELS,
          "four linear AEC lanes");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far, "allocate unit buffers");

    for (int frame_no = 0; frame_no < 80; ++frame_no) {
        for (int i = 0; i < hop; ++i) {
            float echo = 0.08f * sinf(phase);
            float near = frame_no >= 30
                ? 0.025f * sinf(phase * 1.73f + 0.2f) : 0.0f;
            far[i] = echo;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch) +
                    near * (1.0f + 0.02f * ch);
            }
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(audio_pipeline_4ch_process_with_activity(
                  p, microphones, far,
                  frame_no >= 30, frame_no >= 30, NULL, &beam, &info) ==
                  FOUR_AEC_NR_RES_OK,
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
              spectrum_finite(beam.weights, FOUR_AEC_NR_RES_CHANNELS * n_freqs),
              "beam spectrum and weights are finite");
        CHECK(audio_pipeline_4ch_fuse(p, &frame) == FOUR_AEC_NR_RES_OK &&
              frame.n_freqs == n_freqs && frame.e == beam.spectrum &&
              frame.r2 && frame.cn && frame.far_hop,
              "fuse publishes the beam spectrum and the post-stage inputs");
    }
    CHECK(doa_analysis_total == 80,
          "same-grid DOA consumes every main analysis frame");

    audio_pipeline_4ch_reset(p);
    CHECK(audio_pipeline_4ch_process(
              p, microphones, far, &beam, &info) ==
              FOUR_AEC_NR_RES_OK,
          "fallback-VAD processing after reset");
    CHECK(info.frame_index == 0, "reset restarts the unit frame index");
    CHECK(info.doa_analysis_frames == 1,
          "same-grid DOA resumes immediately after reset");

    free(far);
    free(microphones);
    audio_pipeline_4ch_destroy(p);
    return 1;
}

/* Contract test: in fixed-notebook mode (bf_fixed_mode && bf_fixed_align_notebook),
 * bf_create() forces the BF RLS update cadence to 1 hop whatever bf_adapt_interval is.
 * The lambda retime in audio_pipeline_4ch_create() must use that forced cadence, or
 * lambda is calibrated for a slower update period than the one that runs. */
static int test_bf_fixed_notebook_lambda_matches_forced_cadence(void) {
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(16000);
    AudioPipeline4Ch* p;
    int hop_size;
    int actual_interval;
    float actual_lambda;
    float expected_lambda;
    float buggy_lambda;

    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_align_notebook = 1;
    cfg.bf_adapt_interval = 4; /* caller asks for a slower cadence... */
    cfg.bf_lambda = 0.98f;

    p = audio_pipeline_4ch_create(&cfg);
    CHECK(p != NULL, "fixed-notebook cadence test: create pipeline");

    hop_size = audio_pipeline_4ch_bf_hop_size(p);
    actual_interval = audio_pipeline_4ch_bf_effective_adapt_interval(p);
    actual_lambda = audio_pipeline_4ch_bf_lambda(p);

    /* ...but fixed_align_notebook forces the real BF cadence to every hop
     * (1), regardless of the requested 4. */
    CHECK(actual_interval == 1,
          "fixed-notebook mode forces the actual BF cadence to 1");

    /* Lambda must be retimed assuming that SAME forced interval (1), i.e.
     * exactly anr_retime_alpha(bf_lambda, sr, hop_size*1) -- not
     * hop_size*4, which scaling with the raw pre-forced bf_adapt_interval
     * would give. */
    expected_lambda = anr_retime_alpha(
        cfg.bf_lambda, cfg.core.sample_rate, hop_size * 1);
    CHECK(fabsf(actual_lambda - expected_lambda) < 1e-9f,
          "lambda retime uses the forced (not the requested) adapt_interval");

    /* Sanity check that this test can fail: the lambda scaled against the
     * pre-forced interval of 4 must visibly differ from the correct one. */
    buggy_lambda = anr_retime_alpha(
        cfg.bf_lambda, cfg.core.sample_rate, hop_size * 4);
    CHECK(fabsf(actual_lambda - buggy_lambda) > 1e-4f,
          "forced-cadence lambda differs from the old mis-scaled lambda");

    audio_pipeline_4ch_destroy(p);
    return 1;
}

/* Companion check: outside fixed-notebook mode, a configured
 * bf_adapt_interval > 1 is NOT forced down, so the effective interval used
 * for both the actual BF cadence and the lambda retime must equal the
 * requested value itself (guards the other branch of
 * bf_effective_adapt_interval() from regressing). */
static int test_bf_auto_mode_respects_configured_interval(void) {
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(16000);
    AudioPipeline4Ch* p;
    int hop_size;
    float expected_lambda;

    cfg.bf_adapt_interval = 4;
    cfg.bf_lambda = 0.98f;

    p = audio_pipeline_4ch_create(&cfg);
    CHECK(p != NULL, "auto-mode cadence test: create pipeline");

    hop_size = audio_pipeline_4ch_bf_hop_size(p);
    CHECK(audio_pipeline_4ch_bf_effective_adapt_interval(p) == 4,
          "auto mode keeps the requested adapt_interval unforced");

    expected_lambda = anr_retime_alpha(
        cfg.bf_lambda, cfg.core.sample_rate, hop_size * 4);
    CHECK(fabsf(audio_pipeline_4ch_bf_lambda(p) - expected_lambda) < 1e-9f,
          "lambda retime uses the requested adapt_interval in auto mode");

    audio_pipeline_4ch_destroy(p);
    return 1;
}

/* audio_pipeline_4ch_process() must reject a bad call (NULL far_reference/beam,
 * non-finite input) before auto_vad() can fold it into the noise_power EMA; a NaN
 * noise_power would make every later comparison false and disable VAD for good. Two
 * identically-fed pipelines run side by side and only "attacked" also receives the calls
 * that must be rejected; any leak into VAD state makes it diverge from "clean". */
static int test_rejected_calls_preserve_vad_state(void) {
    const int sample_rate = 16000;
    const int fft_size = 256;
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(sample_rate);
    AudioPipeline4Ch *clean, *attacked;
    int hop, n_freqs;
    float *mic, *far, *nan_mic, *nan_far, *inf_far, *quiet_mic;
    float phase = 0.0f;
    AudioPipeline4ChBeam beam_clean, beam_attacked, throwaway_beam;
    AudioPipeline4ChFrameInfo info_clean, info_attacked, throwaway;

    cfg.core.fft_size = fft_size;
    clean = audio_pipeline_4ch_create(&cfg);
    attacked = audio_pipeline_4ch_create(&cfg);
    CHECK(clean && attacked, "create clean/attacked pipeline pair");
    hop = audio_pipeline_4ch_hop_size(clean);
    n_freqs = audio_pipeline_4ch_n_freqs(clean);

    mic = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    nan_mic = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    nan_far = (float*)malloc((size_t)hop * sizeof(float));
    inf_far = (float*)malloc((size_t)hop * sizeof(float));
    quiet_mic = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    CHECK(mic && far && nan_mic
              && nan_far && inf_far && quiet_mic,
          "allocate VAD-contamination test buffers");

    /* Quiet warm-up so noise_power settles away from its init value
     * identically on both instances before the attack. */
    for (int frame = 0; frame < 5; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float v = 0.002f * sinf(phase);
            far[i] = v;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(audio_pipeline_4ch_process(
                  clean, mic, far, &beam_clean, &info_clean) ==
                  FOUR_AEC_NR_RES_OK,
              "clean warm-up frame");
        CHECK(audio_pipeline_4ch_process(
                  attacked, mic, far, &beam_attacked, &info_attacked) ==
                  FOUR_AEC_NR_RES_OK,
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
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
        phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
    }
    CHECK(audio_pipeline_4ch_process(
              clean, mic, far, &beam_clean, &info_clean) ==
              FOUR_AEC_NR_RES_OK,
          "clean hangover-arming frame");
    CHECK(audio_pipeline_4ch_process(
              attacked, mic, far, &beam_attacked, &info_attacked) ==
              FOUR_AEC_NR_RES_OK,
          "attacked hangover-arming frame");
    CHECK(info_clean.vad_raw == 1 && info_attacked.vad_raw == 1,
          "loud tone actually arms the hangover on both instances");

    memcpy(nan_mic, mic,
           (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    nan_mic[0] = NAN;
    memcpy(nan_far, far, (size_t)hop * sizeof(float));
    nan_far[0] = NAN;
    memcpy(inf_far, far, (size_t)hop * sizeof(float));
    inf_far[0] = (float)INFINITY;
    CHECK(audio_pipeline_4ch_process(
              attacked, mic, NULL, &throwaway_beam, &throwaway) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "NULL far_reference is rejected");
    CHECK(audio_pipeline_4ch_process(
              attacked, mic, far, NULL, &throwaway) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "NULL beam is rejected");
    CHECK(audio_pipeline_4ch_process(
              attacked, nan_mic, far, &throwaway_beam, &throwaway) !=
              FOUR_AEC_NR_RES_OK,
          "NaN mic input is rejected, not silently processed");
    /* auto_vad() reads only the microphones, so on finite mic data a non-finite far would
     * pass its own guard and only be caught by the finite check inside
     * four_aec_front_process(), after auto_vad() has already moved
     * noise_power/vad_hangover. NaN and Inf are both exercised. The mic content is quiet
     * on purpose: a loud frame would re-arm vad_hangover to the value it already has and
     * hide a missing far_reference check, while quiet content takes the EMA's silence
     * branch and changes noise_power if auto_vad() wrongly runs. */
    for (int i = 0; i < hop; ++i) {
        float v = 0.003f * sinf(phase + 1.7f);
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            quiet_mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
    }
    CHECK(audio_pipeline_4ch_process(
              attacked, quiet_mic, nan_far, &throwaway_beam, &throwaway) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "NaN far_reference is rejected before touching VAD state");
    CHECK(audio_pipeline_4ch_process(
              attacked, quiet_mic, inf_far, &throwaway_beam, &throwaway) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "Inf far_reference is rejected before touching VAD state");

    /* Resume identical valid input on both, QUIET so the hangover counter walks down to 0
     * instead of being re-armed every hop (a loud frame would mask a 1-frame hangover
     * discrepancy). A leak in any rejected call above desyncs "attacked" from "clean". */
    for (int frame = 0; frame < 40; ++frame) {
        for (int i = 0; i < hop; ++i) {
            float v = 0.002f * sinf(phase);
            far[i] = v;
            for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
                mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
            phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
        }
        CHECK(audio_pipeline_4ch_process(
                  clean, mic, far, &beam_clean, &info_clean) ==
                  FOUR_AEC_NR_RES_OK,
              "clean post-attack frame");
        CHECK(audio_pipeline_4ch_process(
                  attacked, mic, far, &beam_attacked, &info_attacked) ==
                  FOUR_AEC_NR_RES_OK,
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
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            mic[i * FOUR_AEC_NR_RES_CHANNELS + ch] = v;
        phase += (float)(2.0 * M_PI * 700.0 / sample_rate);
    }
    CHECK(audio_pipeline_4ch_process(
              clean, mic, far, &beam_clean, &info_clean) ==
              FOUR_AEC_NR_RES_OK,
          "clean final loud frame");
    CHECK(info_clean.vad_raw == 1,
          "loud tone is actually detected as speech");

    free(mic); free(far); free(nan_mic);
    free(nan_far); free(inf_far); free(quiet_mic);
    audio_pipeline_4ch_destroy(clean);
    audio_pipeline_4ch_destroy(attacked);
    return 1;
}

/* Static-memory pool coverage for the descriptor-tier composition layer: poison/bounds,
 * descriptor-staleness rejection and a heap-vs-pool byte-equal run. Follows
 * run_static_parity() in test_4aec_nr_res.c (0xa5 poison with a trailing-byte bounds
 * check, the 8-point descriptor mutation block, destroy-then-reinit pool reuse) plus a
 * heap vs. pool run over many hops comparing the beam spectrum, the weights, the fused
 * frame and every scalar field of AudioPipeline4ChFrameInfo. The composite pool can be
 * several MB at 48 kHz/1024, so there is no full-pool scan around every rejection. */
static int run_static_parity(int sample_rate, int fft_size) {
    enum { EXTRA = 32, FRAMES = 40 };
    AudioPipeline4ChConfig cfg =
        audio_pipeline_4ch_default_config(sample_rate);
    AudioPipeline4ChMemReq req;
    AudioPipeline4ChMemReq stale;
    AudioPipeline4Ch* heap;
    AudioPipeline4Ch* stat;
    unsigned char* pool = NULL;
    float* microphones;
    float* far;
    float phase_step;
    size_t pool_bytes;
    size_t k;
    int hop;
    int n_freqs;
    int frame;

    cfg.core.fft_size = fft_size;
    cfg.bf_fixed_mode = 1;
    cfg.bf_fixed_doa_rad = 0.35f;
    cfg.bf_mu = 0.02f;

    CHECK(audio_pipeline_4ch_get_mem_requirements(&cfg, &req) == 0,
          "static memory requirement query succeeds");
    CHECK(req.layout_version == AUDIO_PIPELINE_4CH_LAYOUT_VERSION &&
              AUDIO_PIPELINE_4CH_LAYOUT_VERSION == 21u,
          "the queried descriptor publishes the current carve layout");
    CHECK(req.bytes <= (uint64_t)SIZE_MAX,
          "static memory requirement fits size_t");
    pool_bytes = (size_t)req.bytes + (size_t)EXTRA;

    CHECK(posix_memalign(
              (void**)&pool, (size_t)req.alignment, pool_bytes) == 0 && pool,
          "aligned caller pool allocates");
    memset(pool, 0xa5, pool_bytes);

    CHECK(audio_pipeline_4ch_init(pool + 1, (size_t)req.bytes, &cfg) == NULL,
          "static init rejects a misaligned pool");
    CHECK(audio_pipeline_4ch_init(pool, (size_t)req.bytes - 1u, &cfg) == NULL,
          "static init rejects an undersized pool");

    stale = req;
    stale.descriptor_version += 1u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a stale descriptor ABI");
    stale = req;
    stale.layout_version += 1u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a stale carve layout");
    stale = req;
    stale.backend_id = 99u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a backend mismatch");
    stale = req;
    stale.build_flags_hash ^= 0xffffffffu;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a build-layout mismatch");
    stale = req;
    stale.alignment *= 2u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects an alignment mismatch");
    stale = req;
    stale.reserved = 1u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a corrupt reserved field");
    stale = req;
    stale.bytes -= 1u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects stale descriptor bytes");
    stale = req;
    stale.layout_version -= 1u;
    stale.bytes += 4096u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a stale layout even when its cached bytes "
          "are larger than current (byte count fitting must never "
          "substitute for layout/hash agreement)");
    stale = req;
    stale.layout_version = 17u;
    stale.bytes += 4096u;
    CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) ==
              NULL,
          "static init_ex rejects a descriptor of an older layout (17) "
          "even when its cached bytes are larger than current");

    {   /* The previous unit (layout 20) carried its own AecRes: one aligned AecRes plus the
         * 16-byte-aligned handle pointer more than this one. Its descriptor, in every
         * combination with the current fields, is refused; the current layout 21 is accepted
         * just below. */
        AecConfig rc;
        uint64_t old_bytes;
        aec_config_from_preset(&rc, AEC_PRESET_BALANCED, sample_rate);
        rc.fft_size = fft_size;
        old_bytes = req.bytes + (uint64_t)ALIGN16(aec_res_get_mem_size(&rc)) + 16u;
        stale = req;
        stale.layout_version = 20u;
        stale.bytes = old_bytes;
        CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) == NULL &&
              audio_pipeline_4ch_init_ex(pool, (size_t)old_bytes, &cfg, &stale) == NULL,
              "static init_ex rejects the layout-20 unit descriptor (old size, with its RES) "
              "on a pool of the new size and of the old size");
        stale = req;
        stale.layout_version = 20u;
        CHECK(audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &stale) == NULL,
              "static init_ex rejects layout 20 even when every other field is current");
    }

    stat = audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    heap = audio_pipeline_4ch_create(&cfg);
    CHECK(stat != NULL && heap != NULL,
          "poisoned caller pool and heap construction both succeed");

    for (k = 0; k < (size_t)EXTRA; ++k) {
        if (pool[(size_t)req.bytes + k] != 0xa5) break;
    }
    CHECK(k == (size_t)EXTRA, "static init stays inside the queried pool");

    hop = audio_pipeline_4ch_hop_size(stat);
    n_freqs = audio_pipeline_4ch_n_freqs(stat);
    CHECK(hop == audio_pipeline_4ch_hop_size(heap),
          "heap and pool instances agree on hop size");
    CHECK(hop == fft_size / 2, "rate-specific hop");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    CHECK(microphones && far,
          "static parity buffers allocate");

    phase_step = (float)(2.0 * M_PI * 700.0 / sample_rate);
    for (frame = 0; frame < FRAMES; ++frame) {
        AudioPipeline4ChFrameInfo heap_info;
        AudioPipeline4ChFrameInfo stat_info;
        AudioPipeline4ChBeam heap_beam;
        AudioPipeline4ChBeam stat_beam;
        SeSpecFrame heap_frame;
        SeSpecFrame stat_frame;
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
            for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
                microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
                    echo * (1.0f - 0.03f * ch) + near * (1.0f + 0.02f * ch);
            }
        }

        CHECK(audio_pipeline_4ch_process_with_activity(
                  heap, microphones, far, vad, vad, NULL,
                  &heap_beam, &heap_info) == FOUR_AEC_NR_RES_OK,
              "heap instance processes a synthetic hop");
        CHECK(audio_pipeline_4ch_process_with_activity(
                  stat, microphones, far, vad, vad, NULL,
                  &stat_beam, &stat_info) == FOUR_AEC_NR_RES_OK,
              "pool instance processes a synthetic hop");
        CHECK(memcmp(heap_beam.spectrum, stat_beam.spectrum,
                     (size_t)n_freqs * sizeof(Complex)) == 0,
              "heap/pool beam spectrum is byte-identical");
        CHECK(memcmp(heap_beam.weights, stat_beam.weights,
                     (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n_freqs *
                         sizeof(Complex)) == 0,
              "heap/pool beam weights are byte-identical");
        CHECK(audio_pipeline_4ch_fuse(heap, &heap_frame) == FOUR_AEC_NR_RES_OK &&
              audio_pipeline_4ch_fuse(stat, &stat_frame) == FOUR_AEC_NR_RES_OK &&
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

    audio_pipeline_4ch_destroy(stat);
    CHECK(audio_pipeline_4ch_hop_size(stat) == -1,
          "destroy marks a caller-pool instance inactive");
    audio_pipeline_4ch_destroy(stat);
    CHECK(audio_pipeline_4ch_hop_size(stat) == -1,
          "caller-pool destroy is idempotent");
    stat = audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &req);
    CHECK(stat != NULL, "caller pool is reusable after destroy");

    audio_pipeline_4ch_destroy(stat);
    audio_pipeline_4ch_destroy(heap);
    free(far);
    free(microphones);
    free(pool);
    return 1;
}

/* ---------------------------------------------------------------------------
 * The chain the examples run behind the unit (RES, NR, min, comfort noise, synthesis), against
 * the reference chain written over the front end (conv_chain_4ch.h).
 * ------------------------------------------------------------------------ */

static void* aligned_block(size_t bytes) {
    void* mem = NULL;
    return posix_memalign(&mem, 16, bytes) == 0 ? mem : NULL;
}

/* The stages the examples carve behind the unit (audio_pipeline_4ch_static.c): the RES (an
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
    q->rng = SE_CNG_RNG_SEED;
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
    q->rng = SE_CNG_RNG_SEED;
    return (enable_res && !q->res) || (enable_nr && !q->nr) || !q->istft ? -1 : 0;
}

/* The per-hop chain of the examples. 0, or -1 when a stage refuses. */
static int ex_post_hop(ExPost* q, const SeSpecFrame* f, float* out) {
    const float* res = q->unity;
    const float* nr = q->unity;
    if (q->res && !(res = se_res_stage_gain(q->res, f))) return -1;
    if (q->nr && !(nr = se_anr_stage_gain(q->nr, f, q->extra, 1))) return -1;
    sk_min_f32(q->total, nr, res, q->n_freqs);
    sk_capply_gain_f32(q->spec, f->e, q->total, q->n_freqs);
    if (q->enable_cng && q->res && f->cn)
        aec_cng_fill(&q->rng, q->spec, f->cn, res, nr, SE_CNG_NR_GAIN_FLOOR, q->n_freqs);
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
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            mics[(size_t)i * FOUR_AEC_NR_RES_CHANNELS + (size_t)ch] =
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
        for (ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch)
            mics[(size_t)i * FOUR_AEC_NR_RES_CHANNELS + (size_t)ch] =
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
        AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(grids[grid][0]);
        AudioPipeline4Ch* p;
        float *mics, *far;
        int hop, n, h, ok = 1;
        char label[160];
        cfg.core.fft_size = grids[grid][1];
        p = audio_pipeline_4ch_create(&cfg);
        CHECK(p != NULL, "fuse frame: the unit builds");
        hop = audio_pipeline_4ch_hop_size(p);
        n = audio_pipeline_4ch_n_freqs(p);
        mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        for (h = 0; h < 120 && ok; ++h) {
            AudioPipeline4ChBeam beam;
            SeSpecFrame f;
            int k;
            res_scene_hop(h, hop, mics, far);
            if (audio_pipeline_4ch_process(p, mics, far, &beam, NULL) != FOUR_AEC_NR_RES_OK ||
                audio_pipeline_4ch_fuse(p, &f) != FOUR_AEC_NR_RES_OK ||
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
        audio_pipeline_4ch_destroy(p);
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
            AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(grids[grid][0]);
            ConvChainOptions opt = conv_default_options();
            AudioPipeline4Ch* unit;
            ExPost post;
            ConvChain* ref;
            AudioPipeline4ChBeam beam;
            SeSpecFrame frame;
            Complex* weights;
            float *mics, *far, *out_w, *out_r;
            int hop, n, h, i, k, identical = 1, nonzero = 0, ok = 1;
            char label[160];

            if (!((mode >> 1) & 1) && preset != (int)AEC_PRESET_BALANCED) continue;   /* no RES: no preset */
            cfg.core.fft_size = grids[grid][1];
            cfg.bf_enable = 0;
            opt.aec_preset = (AecPreset)preset;
            opt.enable_nr = mode & 1;
            opt.enable_res = (mode >> 1) & 1;
            opt.enable_cng = (mode >> 2) & 1;
            unit = audio_pipeline_4ch_create(&cfg);
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
            hop = audio_pipeline_4ch_hop_size(unit);
            n = audio_pipeline_4ch_n_freqs(unit);
            mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
            far = (float*)calloc((size_t)hop, sizeof(float));
            out_w = (float*)calloc((size_t)hop, sizeof(float));
            out_r = (float*)calloc((size_t)hop, sizeof(float));
            weights = (Complex*)calloc((size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n, sizeof(Complex));
            for (k = 0; k < n; ++k) weights[k].r = 1.0f;
            /* Past the denoiser's initialisation window, so its gain shapes the output. */
            for (h = 0; h < 340 && ok; ++h) {
                res_scene_hop(h, hop, mics, far);   /* nonlinear echo + double talk: the RES cuts */
                if (audio_pipeline_4ch_process_with_activity(unit, mics, far, 0, 0, NULL, &beam, NULL) !=
                        FOUR_AEC_NR_RES_OK ||
                    audio_pipeline_4ch_fuse(unit, &frame) != FOUR_AEC_NR_RES_OK ||
                    ex_post_hop(&post, &frame, out_w) != 0 ||
                    four_aec_front_process(ref->core, mics, far, &ref->front) != FOUR_AEC_NR_RES_OK ||
                    conv_post(ref, weights, ref->front.linear_spectra[0], out_r) != FOUR_AEC_NR_RES_OK) {
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
            audio_pipeline_4ch_destroy(unit);
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
    AudioPipeline4Ch* unit;
    ExPost post;
} ResRig;

static int res_rig_init(ResRig* r, int sample_rate, int fft_size, AecPreset preset) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(sample_rate);
    cfg.core.fft_size = fft_size;
    r->unit = audio_pipeline_4ch_create(&cfg);
    return r->unit && ex_post_init(&r->post, sample_rate, fft_size, 1, preset, 0, 0) == 0 ? 0 : -1;
}

static void res_rig_free(ResRig* r) {
    audio_pipeline_4ch_destroy(r->unit);
    ex_post_free(&r->post);
}

/* What an application's reset does: the unit and the stages it chains. */
static void res_rig_reset(ResRig* r) {
    audio_pipeline_4ch_reset(r->unit);
    aec_res_reset(r->post.res);
}

static const float* res_rig_hop(ResRig* r, int h, float* mics, float* far) {
    AudioPipeline4ChBeam beam;
    SeSpecFrame f;
    res_scene_hop(h, audio_pipeline_4ch_hop_size(r->unit), mics, far);
    if (audio_pipeline_4ch_process(r->unit, mics, far, &beam, NULL) != FOUR_AEC_NR_RES_OK ||
        audio_pipeline_4ch_fuse(r->unit, &f) != FOUR_AEC_NR_RES_OK)
        return NULL;
    return se_res_stage_gain(r->post.res, &f);
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
        hop = audio_pipeline_4ch_hop_size(used.unit);
        n = audio_pipeline_4ch_n_freqs(used.unit);
        mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
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
    hop = audio_pipeline_4ch_hop_size(p.unit);
    n = audio_pipeline_4ch_n_freqs(p.unit);
    mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
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
    AudioPipeline4Ch* unit;
    void* unit_pool;          /* non-NULL: init_ex() in this poisoned pool */
    ExPost post;
} ChainRig;

static int chain_rig_init(ChainRig* r, int sample_rate, int fft_size, int in_pool) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(sample_rate);
    AudioPipeline4ChMemReq req;
    memset(r, 0, sizeof(*r));
    cfg.core.fft_size = fft_size;
    if (in_pool) {
        if (audio_pipeline_4ch_get_mem_requirements(&cfg, &req) != 0 ||
            posix_memalign(&r->unit_pool, req.alignment, (size_t)req.bytes) != 0)
            return -1;
        memset(r->unit_pool, 0xa5, (size_t)req.bytes);
        r->unit = audio_pipeline_4ch_init_ex(r->unit_pool, (size_t)req.bytes, &cfg, &req);
    } else {
        r->unit = audio_pipeline_4ch_create(&cfg);
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
    audio_pipeline_4ch_destroy(r->unit);
    free(r->unit_pool);
    ex_post_free(&r->post);
}

static void chain_rig_reset(ChainRig* r, int reset_res) {
    audio_pipeline_4ch_reset(r->unit);
    ex_post_reset(&r->post, reset_res);
}

/* 0 on success; `out` is the hop's output. */
static int chain_rig_hop(ChainRig* r, int h, float* mics, float* far, float* out) {
    AudioPipeline4ChBeam beam;
    SeSpecFrame f;
    res_scene_hop(h, audio_pipeline_4ch_hop_size(r->unit), mics, far);
    if (audio_pipeline_4ch_process(r->unit, mics, far, &beam, NULL) != FOUR_AEC_NR_RES_OK ||
        audio_pipeline_4ch_fuse(r->unit, &f) != FOUR_AEC_NR_RES_OK)
        return -1;
    return ex_post_hop(&r->post, &f, out);
}

/* After audio_pipeline_4ch_reset() plus the application's own resets (aec_res_reset, denoiser,
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
        hop = audio_pipeline_4ch_hop_size(used.unit);
        mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
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
        hop = audio_pipeline_4ch_hop_size(heap.unit);
        mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
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
 * masked): the chain output follows an independent oracle (conv_chain_4ch.h: its own front end
 * and AecRes, the same call at the same hop) bit for bit; the run without the call differs from
 * both, so a skipped call is caught. */
static int test_caller_res_preset_ramp_vs_oracle(void) {
    static const int grids[3][2] = {{16000, 256}, {16000, 512}, {48000, 1024}};
    enum { SET_AT = 200, HOPS = 520 };
    int g;
    for (g = 0; g < 3; ++g) {
        AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(grids[g][0]);
        ConvChainOptions opt = conv_default_options();
        AudioPipeline4Ch *u_set, *u_skip, *u_ref;
        ExPost p_set, p_skip;
        ConvChain* ref;
        Complex* weights;
        float *mics, *far, *o_set, *o_skip, *o_ref;
        long cut_before = 0, cut_after = 0;   /* (hop, bin) pairs the RES cut below 0.9 */
        int hop, n, h, k, i, ok = 1, set_matches = 1, skip_differs = 0, nonzero = 0;
        char label[200];
        cfg.core.fft_size = grids[g][1];
        cfg.bf_enable = 0;
        opt.enable_nr = 0;
        opt.enable_cng = 1;
        u_set = audio_pipeline_4ch_create(&cfg);
        u_skip = audio_pipeline_4ch_create(&cfg);
        u_ref = audio_pipeline_4ch_create(&cfg);
        ref = conv_create(&cfg.core, &opt);
        CHECK(u_set && u_skip && u_ref && ref &&
                  ex_post_init(&p_set, grids[g][0], grids[g][1], 1, AEC_PRESET_BALANCED, 0, 1) == 0 &&
                  ex_post_init(&p_skip, grids[g][0], grids[g][1], 1, AEC_PRESET_BALANCED, 0, 1) == 0,
              "preset ramp: units, oracle and stages build");
        hop = audio_pipeline_4ch_hop_size(u_set);
        n = audio_pipeline_4ch_n_freqs(u_set);
        mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
        far = (float*)calloc((size_t)hop, sizeof(float));
        o_set = (float*)calloc((size_t)hop, sizeof(float));
        o_skip = (float*)calloc((size_t)hop, sizeof(float));
        o_ref = (float*)calloc((size_t)hop, sizeof(float));
        weights = (Complex*)calloc((size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)n, sizeof(Complex));
        for (k = 0; k < n; ++k) weights[k].r = 1.0f;
        for (h = 0; h < HOPS && ok; ++h) {
            AudioPipeline4ChBeam beam;
            SeSpecFrame fs, fk;
            int cut = 0;
            if (h == SET_AT) {
                if (aec_res_set_preset(p_set.res, AEC_PRESET_AGGRESSIVE, 100.0f) != 0 ||
                    aec_res_set_preset(ref->res, AEC_PRESET_AGGRESSIVE, 100.0f) != 0) { ok = 0; break; }
            }
            res_scene_hard_hop(h, hop, mics, far);
            if (audio_pipeline_4ch_process(u_set, mics, far, &beam, NULL) != FOUR_AEC_NR_RES_OK ||
                audio_pipeline_4ch_fuse(u_set, &fs) != FOUR_AEC_NR_RES_OK ||
                ex_post_hop(&p_set, &fs, o_set) != 0 ||
                audio_pipeline_4ch_process(u_skip, mics, far, &beam, NULL) != FOUR_AEC_NR_RES_OK ||
                audio_pipeline_4ch_fuse(u_skip, &fk) != FOUR_AEC_NR_RES_OK ||
                ex_post_hop(&p_skip, &fk, o_skip) != 0 ||
                audio_pipeline_4ch_process(u_ref, mics, far, &beam, NULL) != FOUR_AEC_NR_RES_OK ||
                four_aec_front_process(ref->core, mics, far, &ref->front) != FOUR_AEC_NR_RES_OK ||
                conv_post(ref, weights, ref->front.linear_spectra[0], o_ref) != FOUR_AEC_NR_RES_OK) {
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
        audio_pipeline_4ch_destroy(u_set); audio_pipeline_4ch_destroy(u_skip);
        audio_pipeline_4ch_destroy(u_ref);
        ex_post_free(&p_set); ex_post_free(&p_skip);
        conv_destroy(ref);
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * audio_pipeline_4ch_raw as a black box: its output file equals the reference chain (own front
 * end, AecRes, denoiser, comfort noise, synthesis) fed the beam of the unit on the same input,
 * bit for bit.
 * ------------------------------------------------------------------------ */

static int raw_case(const char* bin, const char* tmp, int sample_rate, int fft_size, int fixed_doa) {
    char mic_path[512], ref_path[512], out_path[512], cmd[2048];
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(sample_rate);
    ConvChainOptions opt = conv_default_options();
    AudioPipeline4Ch* unit;
    ConvChain* ref;
    FILE *fm, *fr, *fo;
    float *mics, *far, *want, *got;
    int hop, h, hops = 300, ok = 1, nonzero = 0;
    snprintf(mic_path, sizeof mic_path, "%s/raw_mic.f32", tmp);
    snprintf(ref_path, sizeof ref_path, "%s/raw_ref.f32", tmp);
    snprintf(out_path, sizeof out_path, "%s/raw_out.f32", tmp);
    cfg.core.fft_size = fft_size;
    if (fixed_doa) {
        cfg.bf_fixed_mode = 1;
        cfg.bf_fixed_doa_rad = 45.0f * (float)(3.14159265358979323846 / 180.0);   /* as the CLI */
    }
    unit = audio_pipeline_4ch_create(&cfg);
    ref = conv_create(&cfg.core, &opt);
    if (!unit || !ref) return -1;
    hop = audio_pipeline_4ch_hop_size(unit);
    mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    want = (float*)calloc((size_t)hop, sizeof(float));
    got = (float*)calloc((size_t)hop, sizeof(float));
    fm = fopen(mic_path, "wb");
    fr = fopen(ref_path, "wb");
    if (!fm || !fr) return -1;
    for (h = 0; h < hops; ++h) {
        res_scene_hard_hop(h, hop, mics, far);
        fwrite(mics, sizeof(float), (size_t)hop * FOUR_AEC_NR_RES_CHANNELS, fm);
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
        AudioPipeline4ChBeam beam;
        int i;
        res_scene_hard_hop(h, hop, mics, far);
        if (fread(got, sizeof(float), (size_t)hop, fo) != (size_t)hop ||
            audio_pipeline_4ch_process(unit, mics, far, &beam, NULL) != FOUR_AEC_NR_RES_OK ||
            conv_hop(ref, mics, far, beam.weights, beam.spectrum, want) != FOUR_AEC_NR_RES_OK) {
            ok = 0;
            break;
        }
        if (memcmp(want, got, (size_t)hop * sizeof(float)) != 0) ok = 0;
        for (i = 0; i < hop; ++i) if (want[i] != 0.0f) nonzero = 1;
    }
    if (ok && fread(got, sizeof(float), 1, fo) != 0) ok = 0;   /* nothing beyond the hops */
    fclose(fo);
    free(mics); free(far); free(want); free(got);
    audio_pipeline_4ch_destroy(unit);
    conv_destroy(ref);
    return ok && nonzero ? 1 : 0;
}

static int test_raw_runner_black_box(const char* bin_dir, const char* tmp) {
    static const int cases[4][3] = {{16000, 256, 0}, {16000, 512, 0}, {48000, 1024, 0}, {16000, 256, 1}};
    char bin[512], label[160];
    int c;
    snprintf(bin, sizeof bin, "%s/audio_pipeline_4ch_raw", bin_dir);
    for (c = 0; c < 4; ++c) {
        snprintf(label, sizeof label,
                 "black box: audio_pipeline_4ch_raw %d Hz/%d%s equals the reference chain bit for bit",
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
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] =
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
        for (int ch = 0; ch < FOUR_AEC_NR_RES_CHANNELS; ++ch) {
            int64_t u = t - 2 * ch;
            microphones[i * FOUR_AEC_NR_RES_CHANNELS + ch] +=
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
    s->mics = (float*)malloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
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
static int same_info(const AudioPipeline4ChFrameInfo* a, const AudioPipeline4ChFrameInfo* b) {
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

static AudioPipeline4ChConfig split_config(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
    cfg.core.fft_size = 512;
    cfg.core.enable_post = 0;
    cfg.bf_mu = 0.02f;
    cfg.doa_update_interval = 1;
    return cfg;
}

/* Front and beam as two calls. Returns 1 when both succeeded. */
static int split_hop(AudioPipeline4Ch* p, const float* mics, const float* far,
                     int vad_raw, int vad_out, const int* mask,
                     AudioPipeline4ChBeam* beam, AudioPipeline4ChFrameInfo* info) {
    FourAecFront front;
    if (audio_pipeline_4ch_process_front(p, mics, far, &front) != FOUR_AEC_NR_RES_OK) return 0;
    return audio_pipeline_4ch_process_beam(p, &front, vad_raw, vad_out, mask, beam, info) ==
           FOUR_AEC_NR_RES_OK;
}

/* The front view's aligned reference is the delayed far, whole-hop exact, on every
 * hop; the delay state it carries agrees with the unit's frame info. */
static int test_core_aligned_ref(void) {
    enum { FRAMES = 240, TRUE_DELAY = 400 };
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
    AudioPipeline4Ch* p;
    AudioPipeline4ChBeam beam;
    AudioPipeline4ChFrameInfo info;
    FourAecFront pre;
    float* microphones;
    float* far;
    float* far_hist;
    int hop;
    int locked = 0;
    int aligned_checked = 0;

    cfg.core.fft_size = 512;
    p = audio_pipeline_4ch_create(&cfg);
    CHECK(p != NULL, "create 512/256 unit");
    hop = audio_pipeline_4ch_hop_size(p);
    CHECK(hop == 256, "unit hop is 256");

    microphones = (float*)malloc(
        (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    far = (float*)malloc((size_t)hop * sizeof(float));
    far_hist = (float*)calloc((size_t)FRAMES * hop, sizeof(float));
    CHECK(microphones && far && far_hist, "allocate core-test buffers");

    for (int frame = 0; frame < FRAMES; ++frame) {
        fill_echo_hop(frame, hop, TRUE_DELAY, far_hist, far, microphones);
        /* Zero the whole out-struct first: if process_front ever stopped
         * assigning aligned_ref, the non-NULL check below would fail
         * instead of reading leftover stack garbage. */
        memset(&pre, 0, sizeof(pre));
        CHECK(audio_pipeline_4ch_process_front(
                  p, microphones, far, &pre) == FOUR_AEC_NR_RES_OK,
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

        CHECK(audio_pipeline_4ch_process_beam(
                  p, &pre, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_OK &&
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
    audio_pipeline_4ch_destroy(p);
    return 1;
}

/* process_with_activity() against front + beam(v, v, NULL): a null mask and a single flag
 * reproduce it. */
static int test_split_null_mask_equals_begin(void) {
    AudioPipeline4ChConfig cfg = split_config();
    AudioPipeline4Ch *a, *b;
    SplitScene s;
    int hop, nf, ok = 1, nonzero = 0;
    const int frames = 90;

    a = audio_pipeline_4ch_create(&cfg);
    b = audio_pipeline_4ch_create(&cfg);
    CHECK(a && b, "split twin: instances create");
    hop = audio_pipeline_4ch_hop_size(a);
    nf = audio_pipeline_4ch_n_freqs(a);
    CHECK(scene_open(&s, hop, frames), "split twin: scene allocates");
    for (int f = 0; f < frames && ok; ++f) {
        AudioPipeline4ChBeam ba, bb;
        AudioPipeline4ChFrameInfo ia, ib;
        int vad = (f / 7) % 2;
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        ok = audio_pipeline_4ch_process_with_activity(a, s.mics, s.far, vad, vad, NULL, &ba, &ia) ==
                 FOUR_AEC_NR_RES_OK &&
             split_hop(b, s.mics, s.far, vad, vad, NULL, &bb, &ib) &&
             memcmp(ba.spectrum, bb.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
             memcmp(ba.weights, bb.weights,
                    (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)nf * sizeof(Complex)) == 0 &&
             same_info(&ia, &ib);
        for (int k = 0; k < nf; ++k) if (ba.spectrum[k].r != 0.0f) nonzero = 1;
    }
    CHECK(ok, "split twin: front + beam(v, v, NULL) matches process_with_activity byte for byte, state included");
    CHECK(nonzero, "split twin: the compared output is not silence");
    scene_close(&s);
    audio_pipeline_4ch_destroy(a);
    audio_pipeline_4ch_destroy(b);
    return 1;
}

/* A mask and a separate vad_raw / vad_out change what SRP and BF see. */
static int test_split_mask_and_vad_reach_srp_bf(void) {
    AudioPipeline4ChConfig cfg = split_config();
    enum { NP = 5 };   /* NULL, all-zero, all-one, even bins, NULL with vad_raw 0 */
    AudioPipeline4Ch* p[NP];
    int* mask[NP] = {0};
    SplitScene s;
    Complex* out[NP];
    int hop, nf, differs_doa[NP] = {0}, differs_out[NP] = {0};
    int raw_finite[NP] = {0}, adaptive_seen[NP] = {0};
    const int frames = 80;

    for (int i = 0; i < NP; ++i) {
        p[i] = audio_pipeline_4ch_create(&cfg);
        CHECK(p[i] != NULL, "split mask: instances create");
    }
    hop = audio_pipeline_4ch_hop_size(p[0]);
    nf = audio_pipeline_4ch_n_freqs(p[0]);
    CHECK(scene_open(&s, hop, frames), "split mask: scene allocates");
    for (int i = 1; i < 4; ++i) {
        mask[i] = (int*)malloc((size_t)nf * sizeof(int));
        for (int k = 0; k < nf; ++k)
            mask[i][k] = i == 1 ? 0 : (i == 2 ? 1 : (k % 2 == 0));
    }
    for (int i = 0; i < NP; ++i) out[i] = (Complex*)malloc((size_t)nf * sizeof(Complex));
    for (int f = 0; f < frames; ++f) {
        AudioPipeline4ChFrameInfo st[NP];
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        for (int i = 0; i < NP; ++i) {
            AudioPipeline4ChBeam beam;
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
        audio_pipeline_4ch_destroy(p[i]);
        free(out[i]);
        free(mask[i]);
    }
    scene_close(&s);
    return 1;
}

/* The frame info against SRP and BF objects built here from the same derivation and fed the
 * same lane spectra: every field, the beamformed spectrum and the weights. */
static int split_state_case(int fixed_bf) {
    AudioPipeline4ChConfig cfg = split_config();
    AudioPipeline4Ch* p;
    cfg.bf_fixed_mode = fixed_bf;
    cfg.bf_fixed_doa_rad = 0.3f;
    p = audio_pipeline_4ch_create(&cfg);
    SRP_Config sc;
    BF_Config gc;
    float gx[FOUR_AEC_NR_RES_CHANNELS], gy[FOUR_AEC_NR_RES_CHANNELS];
    ArrayGeometry geom = { FOUR_AEC_NR_RES_CHANNELS, gx, gy };
    SRP* srp;
    BF* bf;
    Complex *spec, *weights;
    SplitScene s;
    int *mask;
    int hop, nf, interval, finite_smooth = 0, finite_used = 0;
    int fixed_used_seen = 0;
    const int frames = 70;

    CHECK(p != NULL, "split state: instance creates");
    hop = audio_pipeline_4ch_hop_size(p);
    nf = audio_pipeline_4ch_n_freqs(p);
    for (int m = 0; m < FOUR_AEC_NR_RES_CHANNELS; ++m) {
        float phi = 2.0f * M_PI * m / FOUR_AEC_NR_RES_CHANNELS;
        gx[m] = cfg.uca_radius_m * cosf(phi);
        gy[m] = cfg.uca_radius_m * sinf(phi);
    }
    memset(&sc, 0, sizeof(sc));
    sc.M = FOUR_AEC_NR_RES_CHANNELS;
    sc.F = nf;
    sc.num_angles = cfg.num_angles;
    sc.sr = (float)cfg.core.sample_rate;
    sc.NFFT = (float)audio_pipeline_4ch_fft_size(p);
    sc.c = cfg.speed_of_sound_m_s;
    sc.low_freq = cfg.doa_low_freq_hz;
    sc.high_freq = cfg.doa_high_freq_hz;
    sc.enable_smoothing = cfg.doa_enable_smoothing;
    sc.switch_consec = cfg.doa_switch_consecutive;
    sc.angle_tol = cfg.doa_angle_tolerance_rad;
    sc.update_interval = cfg.doa_update_interval;
    srp = srp_create_from_geometry(&sc, &geom);
    CHECK(srp != NULL, "split state: lower-level SRP creates");
    interval = bf_effective_adapt_interval(cfg.bf_fixed_mode,
                                            cfg.bf_fixed_align_notebook,
                                            cfg.bf_adapt_interval);
    memset(&gc, 0, sizeof(gc));
    gc.enable = cfg.bf_enable;
    gc.lambda = anr_retime_alpha(cfg.bf_lambda, cfg.core.sample_rate, hop * interval);
    gc.mu = cfg.bf_mu;
    gc.enable_fix_mode = cfg.bf_fixed_mode;
    gc.fixed_doa_rad = cfg.bf_fixed_doa_rad;
    gc.fixed_align_notebook = cfg.bf_fixed_align_notebook;
    gc.adapt_interval = interval;
    bf = bf_create(FOUR_AEC_NR_RES_CHANNELS, nf, srp->num_angles, srp->a_array, &gc);
    CHECK(bf != NULL, "split state: lower-level BF creates");
    spec = (Complex*)calloc((size_t)nf, sizeof(Complex));
    weights = (Complex*)calloc((size_t)nf * FOUR_AEC_NR_RES_CHANNELS, sizeof(Complex));
    mask = (int*)malloc((size_t)nf * sizeof(int));
    for (int k = 0; k < nf; ++k) mask[k] = (k % 3) != 0;
    CHECK(scene_open(&s, hop, frames), "split state: scene allocates");
    for (int f = 0; f < frames; ++f) {
        FourAecFront front;
        AudioPipeline4ChBeam beam;
        AudioPipeline4ChFrameInfo st;
        int vad_raw = (f / 5) % 3 != 0, vad_out = (f / 9) % 2;
        const int* m = (f % 4) == 3 ? NULL : mask;
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        CHECK(audio_pipeline_4ch_process_front(p, s.mics, s.far, &front) == FOUR_AEC_NR_RES_OK,
              "split state: front runs");
        CHECK(front.n_freqs == nf && front.linear_spectra[0] && front.linear_spectra[3],
              "split state: front publishes the four lane spectra");
        doa_step(srp, front.linear_spectra, m, vad_raw, vad_out);
        bf_process_with_weights(bf, front.linear_spectra, doa_get_smooth(srp),
                                 vad_out ? 0 : 1, m, spec, weights);
        CHECK(audio_pipeline_4ch_process_beam(p, &front, vad_raw, vad_out, m, &beam, &st) == FOUR_AEC_NR_RES_OK,
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
                     (size_t)nf * FOUR_AEC_NR_RES_CHANNELS * sizeof(Complex)) == 0,
              "split state: the beam weights are the lower-level BF weights");
    }
    CHECK(finite_smooth && finite_used, "split state: the smoothed and used DOA become finite");
    CHECK(!fixed_bf || fixed_used_seen, "split state: a fixed beamformer reports its fixed angle as the used DOA");
    bf_destroy(bf);
    srp_destroy(srp);
    free(spec); free(weights); free(mask);
    scene_close(&s);
    audio_pipeline_4ch_destroy(p);
    return 1;
}

static int test_split_beam_state_matches_lower_level(void) {
    return split_state_case(0) && split_state_case(1);
}

/* Call order: beam needs a pending front, fuse needs this hop's beam, a new front supersedes
 * whatever is pending, reset drops it; refused calls change nothing. */
static int test_split_sequence_guards(void) {
    AudioPipeline4ChConfig cfg = split_config();
    AudioPipeline4ChConfig cfg_post = split_config();
    AudioPipeline4Ch *p, *q, *post;
    FourAecFront front, other, bad;
    AudioPipeline4ChBeam beam;
    AudioPipeline4ChFrameInfo info;
    SeSpecFrame frame;
    float *mics, *far;
    int hop, nf;

    cfg_post.core.enable_post = 1;
    p = audio_pipeline_4ch_create(&cfg);
    q = audio_pipeline_4ch_create(&cfg);
    post = audio_pipeline_4ch_create(&cfg_post);
    CHECK(p && q && post, "split guards: instances create");
    hop = audio_pipeline_4ch_hop_size(p);
    nf = audio_pipeline_4ch_n_freqs(p);
    mics = (float*)calloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    memset(&bad, 0, sizeof(bad));
    CHECK(audio_pipeline_4ch_process_beam(p, &bad, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "split guards: a front with the wrong bin count or NULL spectra is refused");
    CHECK(audio_pipeline_4ch_process_front(p, mics, far, NULL) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_process_front(NULL, mics, far, &front) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_process_front(p, NULL, far, &front) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_process_front(p, mics, NULL, &front) == FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "split guards: NULL arguments of process_front are refused");
    CHECK(audio_pipeline_4ch_process_with_activity(p, mics, far, 2, 0, NULL, &beam, &info) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_process_with_activity(p, mics, far, 0, 0, NULL, NULL, &info) ==
              FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "split guards: a non-boolean flag or a NULL beam is refused before any state moves");

    /* A real front of another instance: valid in shape, but p has none pending. */
    CHECK(audio_pipeline_4ch_process_front(q, mics, far, &other) == FOUR_AEC_NR_RES_OK,
          "split guards: the other instance's front runs");
    CHECK(audio_pipeline_4ch_process_beam(p, &other, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "split guards: beam without a pending front is a sequence error");
    CHECK(audio_pipeline_4ch_fuse(post, &frame) == FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "split guards: fuse without a beam is a sequence error");

    CHECK(audio_pipeline_4ch_process_front(post, mics, far, &front) == FOUR_AEC_NR_RES_OK &&
          audio_pipeline_4ch_fuse(post, &frame) == FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "split guards: fuse between front and beam is a sequence error");
    bad = front;
    bad.n_freqs = nf + 1;
    CHECK(audio_pipeline_4ch_process_beam(post, &bad, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_process_beam(post, &front, 2, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_process_beam(post, &front, 0, 2, NULL, &beam, &info) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_process_beam(post, NULL, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_process_beam(post, &front, 0, 0, NULL, NULL, &info) == FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "split guards: a refused beam call changes nothing");
    CHECK(audio_pipeline_4ch_process_beam(post, &front, 0, 0, NULL, &beam, NULL) == FOUR_AEC_NR_RES_OK &&
          audio_pipeline_4ch_process_beam(post, &front, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "split guards: the beam half runs once per hop; info is optional");
    CHECK(audio_pipeline_4ch_fuse(post, NULL) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_fuse(NULL, &frame) == FOUR_AEC_NR_RES_INVALID_ARGUMENT &&
          audio_pipeline_4ch_fuse(post, &frame) == FOUR_AEC_NR_RES_OK &&
          frame.e == beam.spectrum,
          "split guards: fuse follows this hop's beam");
    CHECK(audio_pipeline_4ch_process_front(post, mics, far, &front) == FOUR_AEC_NR_RES_OK &&
          audio_pipeline_4ch_fuse(post, &frame) == FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "split guards: a new front supersedes the beam of the last hop");
    CHECK(audio_pipeline_4ch_process_front(post, mics, far, &front) == FOUR_AEC_NR_RES_OK &&
          audio_pipeline_4ch_process_beam(post, &front, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_OK,
          "split guards: a second front without a beam in between is legal");
    audio_pipeline_4ch_reset(post);
    CHECK(audio_pipeline_4ch_process_beam(post, &front, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_SEQUENCE_ERROR &&
          audio_pipeline_4ch_fuse(post, &frame) == FOUR_AEC_NR_RES_SEQUENCE_ERROR,
          "split guards: a reset drops the pending front and beam");

    /* The front-only profile (a neural post-filter reads the beam spectrum): it builds, runs,
     * and has no fuse; a refused fuse leaves the unit as it was. */
    CHECK(audio_pipeline_4ch_process_front(p, mics, far, &front) == FOUR_AEC_NR_RES_OK &&
          audio_pipeline_4ch_process_beam(p, &front, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_OK &&
          info.frame_index == 0,
          "split guards: the enable_post = 0 unit runs front and beam");
    CHECK(audio_pipeline_4ch_fuse(p, &frame) == FOUR_AEC_NR_RES_INVALID_ARGUMENT,
          "split guards: fuse is refused without the post context");
    CHECK(audio_pipeline_4ch_process_front(p, mics, far, &front) == FOUR_AEC_NR_RES_OK &&
          audio_pipeline_4ch_process_beam(p, &front, 0, 0, NULL, &beam, &info) == FOUR_AEC_NR_RES_OK &&
          info.frame_index == 1,
          "split guards: the refused fuse did not reset the unit");

    free(mics); free(far);
    audio_pipeline_4ch_destroy(p);
    audio_pipeline_4ch_destroy(q);
    audio_pipeline_4ch_destroy(post);
    return 1;
}

static int same_bytes(const void* a, const void* b, size_t bytes) {
    if (!a || !b) return a == b;
    return memcmp(a, b, bytes) == 0;
}

static int same_frame(const SeSpecFrame* a, const SeSpecFrame* b) {
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
    AudioPipeline4ChConfig cfg = split_config();
    AudioPipeline4Ch *a, *b;
    SplitScene s;
    float* nan_mics;
    int hop, nf, ok = 1, nonzero = 0;

    cfg.core.enable_post = 1;
    a = audio_pipeline_4ch_create(&cfg);
    b = audio_pipeline_4ch_create(&cfg);
    CHECK(a && b, "refused hop: instances create");
    hop = audio_pipeline_4ch_hop_size(a);
    nf = audio_pipeline_4ch_n_freqs(a);
    CHECK(scene_open(&s, hop, FRAMES), "refused hop: scene allocates");
    nan_mics = (float*)malloc((size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
    CHECK(nan_mics != NULL, "refused hop: buffer allocates");
    g_rng = 0x1234567u;
    for (int f = 0; f < FRAMES && ok; ++f) {
        AudioPipeline4ChBeam ba, bb;
        AudioPipeline4ChFrameInfo ia, ib;
        SeSpecFrame fa, fb;
        int vad = (f / 7) % 2;
        if (f == NAN_HOP) {
            /* Only `a` is offered the bad hop; the scene does not advance for it. */
            memcpy(nan_mics, s.mics, (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
            nan_mics[0] = NAN;
            CHECK(audio_pipeline_4ch_process_with_activity(a, nan_mics, s.far, vad, vad, NULL, &ba,
                                                           &ia) == FOUR_AEC_NR_RES_INVALID_ARGUMENT,
                  "refused hop: a non-finite microphone sample is refused");
            CHECK(audio_pipeline_4ch_fuse(a, &fa) == FOUR_AEC_NR_RES_SEQUENCE_ERROR,
                  "refused hop: fuse after the refused hop is a sequence error");
        }
        fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
        if (f == STALE_HOP) {
            /* Both twins: a front, then a refused front; the first front's view is stale. */
            FourAecFront front, unused;
            AudioPipeline4Ch* twins[2] = { a, b };
            memcpy(nan_mics, s.mics, (size_t)hop * FOUR_AEC_NR_RES_CHANNELS * sizeof(float));
            nan_mics[0] = NAN;
            for (int t = 0; t < 2; ++t) {
                CHECK(audio_pipeline_4ch_process_front(twins[t], s.mics, s.far, &front) ==
                          FOUR_AEC_NR_RES_OK &&
                      audio_pipeline_4ch_process_front(twins[t], nan_mics, s.far, &unused) ==
                          FOUR_AEC_NR_RES_INVALID_ARGUMENT,
                      "refused hop: a refused front after an accepted one");
                CHECK(audio_pipeline_4ch_process_beam(twins[t], &front, vad, vad, NULL, &ba, &ia) ==
                          FOUR_AEC_NR_RES_SEQUENCE_ERROR &&
                      audio_pipeline_4ch_fuse(twins[t], &fa) == FOUR_AEC_NR_RES_SEQUENCE_ERROR,
                      "refused hop: the refused front leaves no front pending for beam or fuse");
            }
            continue;
        }
        ok = audio_pipeline_4ch_process_with_activity(a, s.mics, s.far, vad, vad, NULL, &ba, &ia) ==
                 FOUR_AEC_NR_RES_OK &&
             audio_pipeline_4ch_process_with_activity(b, s.mics, s.far, vad, vad, NULL, &bb, &ib) ==
                 FOUR_AEC_NR_RES_OK &&
             audio_pipeline_4ch_fuse(a, &fa) == FOUR_AEC_NR_RES_OK &&
             audio_pipeline_4ch_fuse(b, &fb) == FOUR_AEC_NR_RES_OK &&
             memcmp(ba.spectrum, bb.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
             memcmp(ba.weights, bb.weights,
                    (size_t)FOUR_AEC_NR_RES_CHANNELS * (size_t)nf * sizeof(Complex)) == 0 &&
             same_info(&ia, &ib) && same_frame(&fa, &fb);
        if (f > NAN_HOP)
            for (int k = 0; k < nf; ++k) if (ba.spectrum[k].r != 0.0f) nonzero = 1;
    }
    CHECK(ok, "refused hop: the stream after the refused hop and fuse equals the uninterrupted "
              "twin byte for byte (beam, weights, frame info, fused frame)");
    CHECK(nonzero, "refused hop: the compared output is not silence");
    free(nan_mics);
    scene_close(&s);
    audio_pipeline_4ch_destroy(a);
    audio_pipeline_4ch_destroy(b);
    return 1;
}

/* reset() returns the unit to the state of a fresh instance: the same input afterwards gives
 * the same beam bytes and frame info as the first pass did. */
static int test_reset_equals_fresh(void) {
    enum { FRAMES = 40 };
    AudioPipeline4ChConfig cfg = split_config();
    AudioPipeline4Ch *warm, *fresh;
    SplitScene s;
    Complex* first[FRAMES] = {0};
    int hop, nf, ok = 1;

    cfg.core.enable_post = 1;
    warm = audio_pipeline_4ch_create(&cfg);
    fresh = audio_pipeline_4ch_create(&cfg);
    CHECK(warm && fresh, "reset: instances create");
    hop = audio_pipeline_4ch_hop_size(warm);
    nf = audio_pipeline_4ch_n_freqs(warm);
    CHECK(scene_open(&s, hop, FRAMES), "reset: scene allocates");
    for (int pass = 0; pass < 2 && ok; ++pass) {
        g_rng = 0x1234567u;
        memset(s.far_hist, 0, (size_t)FRAMES * hop * sizeof(float));
        memset(s.near_hist, 0, (size_t)FRAMES * hop * sizeof(float));
        if (pass == 1) audio_pipeline_4ch_reset(warm);
        for (int f = 0; f < FRAMES && ok; ++f) {
            AudioPipeline4ChBeam beam;
            AudioPipeline4ChFrameInfo info;
            int vad = (f / 7) % 2;
            fill_split_hop(f, hop, s.far_hist, s.near_hist, s.far, s.mics);
            ok = audio_pipeline_4ch_process_with_activity(warm, s.mics, s.far, vad, vad, NULL, &beam, &info) ==
                 FOUR_AEC_NR_RES_OK;
            if (!ok) break;
            if (pass == 0) {
                first[f] = (Complex*)malloc((size_t)nf * sizeof(Complex));
                ok = first[f] != NULL;
                if (ok) memcpy(first[f], beam.spectrum, (size_t)nf * sizeof(Complex));
            } else {
                AudioPipeline4ChBeam fb;
                AudioPipeline4ChFrameInfo ref_info;
                ok = audio_pipeline_4ch_process_with_activity(fresh, s.mics, s.far, vad, vad, NULL, &fb, &ref_info) ==
                         FOUR_AEC_NR_RES_OK &&
                     memcmp(first[f], beam.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
                     memcmp(fb.spectrum, beam.spectrum, (size_t)nf * sizeof(Complex)) == 0 &&
                     same_info(&info, &ref_info);
            }
        }
    }
    for (int f = 0; f < FRAMES; ++f) free(first[f]);
    CHECK(ok, "reset: a reset unit reproduces its first pass and a fresh instance byte for byte");
    scene_close(&s);
    audio_pipeline_4ch_destroy(warm);
    audio_pipeline_4ch_destroy(fresh);
    return 1;
}

/* The composed core is readable, the timing record is the core's, and a NULL handle is safe. */
static int test_core_view_and_timing(void) {
    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
    AudioPipeline4Ch* p = audio_pipeline_4ch_create(&cfg);
    FourAecNrResLastTiming t, z;
    CHECK(p != NULL, "core view: unit creates");
    CHECK(audio_pipeline_4ch_get_core(p) != NULL &&
              four_aec_nr_res_linear_aec_count(audio_pipeline_4ch_get_core(p)) == FOUR_AEC_NR_RES_CHANNELS &&
              four_aec_nr_res_hop_size(audio_pipeline_4ch_get_core(p)) == audio_pipeline_4ch_hop_size(p),
          "core view: the unit exposes its core");
    CHECK(audio_pipeline_4ch_get_core(NULL) == NULL, "core view: NULL handle");
    memset(&t, 0xff, sizeof(t));
    memset(&z, 0, sizeof(z));
    audio_pipeline_4ch_last_core_timing(NULL, &t);
    CHECK(memcmp(&t, &z, sizeof(t)) == 0, "core view: a NULL handle zeroes the timing record");
    audio_pipeline_4ch_destroy(p);
    CHECK(audio_pipeline_4ch_get_core(p) == NULL, "core view: destroyed handle");
    return 1;
}

static int run_all_tests(const char* bin_dir, const char* tmp) {
    AudioPipeline4ChConfig invalid =
        audio_pipeline_4ch_default_config(16000);
    invalid.core.fft_size = 1024;
    CHECK(audio_pipeline_4ch_create(&invalid) == NULL,
          "16 kHz rejects cross-rate FFT 1024");
    invalid = audio_pipeline_4ch_default_config(48000);
    invalid.core.fft_size = 512;
    CHECK(audio_pipeline_4ch_create(&invalid) == NULL,
          "48 kHz rejects cross-rate FFT 512");
    invalid = audio_pipeline_4ch_default_config(16000);
    invalid.core.enable_post = 0;
    {
        AudioPipeline4Ch* front_only = audio_pipeline_4ch_create(&invalid);
        CHECK(front_only != NULL,
              "the unit accepts the front-only profile (no fuse)");
        audio_pipeline_4ch_destroy(front_only);
    }
    invalid.core.enable_post = 2;
    CHECK(audio_pipeline_4ch_create(&invalid) == NULL,
          "a non-boolean enable_post is rejected");
    invalid = audio_pipeline_4ch_default_config(16000);
    invalid.bf_lambda = 1.0f + 1e-6f;
    CHECK(audio_pipeline_4ch_create(&invalid) == NULL,
          "BF forgetting factor above 1.0 is rejected");
    invalid.bf_lambda = 1.0f;
    {
        AudioPipeline4Ch* boundary = audio_pipeline_4ch_create(&invalid);
        CHECK(boundary != NULL,
              "BF forgetting factor of exactly 1.0 is still accepted");
        audio_pipeline_4ch_destroy(boundary);
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
    CHECK(test_raw_runner_black_box(bin_dir, tmp), "audio_pipeline_4ch_raw against the oracle");
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
    printf("All audio_pipeline_4ch tests passed (spatial=%s)\n",
           audio_pipeline_4ch_spatial_backend());
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <example-bin-dir> <scratch-dir>\n", argv[0]);
        return 2;
    }
    return run_all_tests(argv[1], argv[2]) ? 0 : 1;
}
