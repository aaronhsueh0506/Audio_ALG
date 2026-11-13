/**
 * Chain-level tests of the four-microphone array instance (aec.h, AecConfig.n_mics == 4) behind the
 * conventional NR / synthesis chain (conv_chain_4ch.h): the groups that need the NR library, the
 * gain and comfort-noise stages or the synthesis. The array itself (lanes, shared delay, admission,
 * fusion, descriptor, reset and placement of its spectra and gain) is tested by the AEC library's
 * test_aec_array.c and test_aec_array_flow.c. The helper-level chain tests are in
 * test_array_sample_beam.c.
 *
 * The tests use equal weights as an external-beamformer stand in. The library itself never
 * chooses those weights in production.
 */

#include "aec.h"
#include "conv_chain_4ch.h"
#include "array_test_rig.h"
#include "audio_window.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* sqrt(2)*sin(2*pi*i/32): the phase table the comfort-noise fill (aec_apply_cng) indexes. */
static const float kCngTable[32] = {
    0.0f, 0.27589938f, 0.541196108f, 0.785694957f, 1.0f, 1.17587554f,
    1.30656302f, 1.3870399f, 1.41421354f, 1.3870399f, 1.30656302f,
    1.17587554f, 1.0f, 0.785694957f, 0.541196108f, 0.27589938f,
    1.73191212e-16f, -0.27589938f, -0.541196108f, -0.785694957f, -1.0f,
    -1.17587554f, -1.30656302f, -1.3870399f, -1.41421354f, -1.3870399f,
    -1.30656302f, -1.17587554f, -1.0f, -0.785694957f, -0.541196108f,
    -0.27589938f};

static int failures = 0;

#define CHECK(condition, label)                                            \
    do {                                                                   \
        if (condition) {                                                   \
            printf("PASS: %s\n", label);                                   \
        } else {                                                           \
            printf("FAIL: %s\n", label);                                   \
            failures += 1;                                                 \
        }                                                                  \
    } while (0)

static void fill_inputs(float* microphones, float* ref, int hop,
                        int sample_rate, int frame_index) {
    int i;
    int ch;
    for (i = 0; i < hop; ++i) {
        int64_t absolute = (int64_t)frame_index * hop + i;
        float phase = 2.0f * 3.14159265358979323846f *
                      440.0f * (float)absolute / (float)sample_rate;
        ref[i] = 0.08f * sinf(phase);
        for (ch = 0; ch < AEC_ARRAY_MICS; ++ch) {
            float near = 0.01f * sinf(
                phase * (1.0f + 0.05f * (float)ch));
            microphones[
                i * AEC_ARRAY_MICS + ch] =
                ref[i] * (0.4f + 0.05f * (float)ch) + near;
        }
    }
}


/* One hop of the echo stimulus the bypass tests share: a far-end tone, its
 * half-level echo on every microphone, a weak near-end tone and a
 * deterministic broadband term so the denoiser has something to shape once
 * it leaves init -- a pure tone pair can leave the gain at its ceiling and a
 * divergence row would never fire. */
static void synth_echo_hop(int h, int hop, float* mics, float* far,
                           uint32_t* rng) {
    int i, ch;
    for (i = 0; i < hop; ++i) {
        float t = (float)(h * hop + i);
        float r = 0.6f * sinf(0.06f * t);
        float noise;
        *rng = *rng * 1664525u + 1013904223u;
        noise = 0.02f * ((float)((*rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f);
        far[i] = r;
        for (ch = 0; ch < 4; ++ch)
            mics[(size_t)i * 4u + (size_t)ch] =
                0.5f * r + 0.02f * sinf(0.017f * t) + noise;
    }
}

/* The NR-disabled post path must apply exactly the RES gain with no scaling of its own.
 * MMSE-LSA passes through (gain == 1.0 in every bin) for its first num_init_frames hops,
 * and min(1.0, res_gain) is res_gain, so over that window an NR-enabled and an NR-disabled
 * instance fed the same stimulus must emit the same samples bit for bit. Also asserted: a
 * non-zero output (the equality would hold trivially on silence) and a later divergence (a
 * denoiser that never shapes anything would make the equality prove nothing). */
static void test_nr_bypass_output_identity(void) {
    AecConfig cfg = array_test_default_config(16000);
    ConvChainOptions on_opt = conv_default_options();
    ConvChainOptions off_opt = on_opt;
    ConvChain* on;
    ConvChain* off;
    float *mics, *far, *out_on, *out_off;
    Complex* w;
    uint32_t rng = 12345u;
    int hop, n, init_hops, total_hops, h, i, ch, k;
    int prefix_equal = 1, prefix_nonzero = 0, diverged_after_init = 0;

    off_opt.enable_nr = 0;
    on = conv_create(&cfg, &on_opt);
    off = conv_create(&cfg, &off_opt);
    CHECK(on != NULL && off != NULL,
          "NR-enabled and NR-disabled instances both create");
    if (!on || !off) {
        conv_destroy(on);
        conv_destroy(off);
        return;
    }
    hop = on->hop;
    n = on->n_freqs;
    CHECK(off->hop == hop && off->n_freqs == n,
          "the two instances resolve the same grid");

    /* The canonical noise-init length: the pipeline's NR overlay retimes L,
     * not num_init_frames, so the denoiser's pass-through window is the
     * canonical one for this grid. */
    init_hops = anr_retime_frames(20, 16000, hop);
    total_hops = init_hops + 80;

    mics = (float*)calloc((size_t)hop * 4u, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    out_on = (float*)calloc((size_t)hop, sizeof(float));
    out_off = (float*)calloc((size_t)hop, sizeof(float));
    w = (Complex*)calloc(
        (size_t)AEC_ARRAY_MICS * (size_t)n, sizeof(Complex));
    if (!mics || !far || !out_on || !out_off || !w) {
        free(mics); free(far); free(out_on); free(out_off); free(w);
        conv_destroy(on);
        conv_destroy(off);
        CHECK(0, "NR bypass identity stimulus allocates");
        return;
    }
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
        for (k = 0; k < n; ++k) {
            w[(size_t)ch * (size_t)n + (size_t)k].r = 0.25f;
            w[(size_t)ch * (size_t)n + (size_t)k].i = 0.0f;
        }

    for (h = 0; h < total_hops; ++h) {
        synth_echo_hop(h, hop, mics, far, &rng);
        if (conv_hop(on, mics, far, w, NULL, out_on) != AEC_OK ||
            conv_hop(off, mics, far, w, NULL, out_off) != AEC_OK)
            break;

        if (h < init_hops) {
            if (memcmp(out_on, out_off, (size_t)hop * sizeof(float)) != 0)
                prefix_equal = 0;
            for (i = 0; i < hop; ++i)
                if (out_off[i] != 0.0f) prefix_nonzero = 1;
        } else if (memcmp(out_on, out_off,
                          (size_t)hop * sizeof(float)) != 0) {
            diverged_after_init = 1;
        }
    }
    CHECK(h == total_hops, "both instances process the whole stimulus");
    CHECK(prefix_equal,
          "NR-disabled output is bit-identical to NR-enabled while the "
          "denoiser passes through -- the RES-only branch applies res_gain "
          "with nothing of its own");
    CHECK(prefix_nonzero,
          "that identity window carries a non-zero signal");
    CHECK(diverged_after_init,
          "the two paths diverge once the denoiser starts shaping, so the "
          "identity above is a constraint and not a silent no-op");

    free(mics); free(far); free(out_on); free(out_off); free(w);
    conv_destroy(on);
    conv_destroy(off);
}

/* With RES, NR and CNG all disabled the post path has no gain source, so the core must be
 * a transparent synthesis of the beamformed linear error: with identity weights on lane 0
 * the fused error is lane 0's analysis spectrum, and a sqrt-Hann WOLA round trip
 * reconstructs the previous hop of formed_hop[0] to float precision. The default core on
 * the same stimulus must not satisfy it, or the identity proves nothing about the switch.
 * Structural rows: no suppressor is built. */
static void test_res_bypass_passthrough(void) {
    AecConfig cfg = array_test_default_config(16000);
    ConvChainOptions def_opt = conv_default_options();
    ConvChainOptions off_opt = def_opt;
    ConvChain* off;
    ConvChain* def;
    float *mics, *far, *out_off, *out_def, *prev_lin;
    Complex* w;
    uint32_t rng = 4242u;
    double max_abs = 0.0, max_err_off = 0.0, max_err_def = 0.0;
    int hop, n, h, i, k, total_hops = 200;

    def_opt.enable_cng = 0;
    off_opt.enable_cng = 0;
    off_opt.enable_res = 0;
    off_opt.enable_nr = 0;
    off = conv_create(&cfg, &off_opt);
    def = conv_create(&cfg, &def_opt);
    CHECK(off != NULL && def != NULL, "RES-disabled and default chains create");
    if (!off || !def) {
        conv_destroy(off);
        conv_destroy(def);
        return;
    }
    hop = off->hop;
    n = off->n_freqs;
    mics = (float*)calloc((size_t)hop * 4u, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    out_off = (float*)calloc((size_t)hop, sizeof(float));
    out_def = (float*)calloc((size_t)hop, sizeof(float));
    prev_lin = (float*)calloc((size_t)hop, sizeof(float));
    w = (Complex*)calloc(
        (size_t)AEC_ARRAY_MICS * (size_t)n, sizeof(Complex));
    if (!mics || !far || !out_off || !out_def || !prev_lin || !w) {
        free(mics); free(far); free(out_off); free(out_def); free(prev_lin);
        free(w);
        conv_destroy(off);
        conv_destroy(def);
        CHECK(0, "RES bypass stimulus allocates");
        return;
    }
    for (k = 0; k < n; ++k) w[k].r = 1.0f;
    for (h = 0; h < total_hops; ++h) {
        synth_echo_hop(h, hop, mics, far, &rng);
        if (conv_hop(off, mics, far, w, NULL, out_off) != AEC_OK ||
            conv_hop(def, mics, far, w, NULL, out_def) != AEC_OK)
            break;
        /* out(h) reconstructs lane 0's hop h-1; the two lanes see the same
         * input, so either instance's copy of it is the reference. */
        if (h >= 4)
            for (i = 0; i < hop; ++i) {
                double ref = prev_lin[i];
                double e_off = fabs((double)out_off[i] - ref);
                double e_def = fabs((double)out_def[i] - ref);
                if (fabs(ref) > max_abs) max_abs = fabs(ref);
                if (e_off > max_err_off) max_err_off = e_off;
                if (e_def > max_err_def) max_err_def = e_def;
            }
        for (i = 0; i < hop; ++i)
            prev_lin[i] = off->front.lane_formed_hop[0][i];
    }
    CHECK(h == total_hops, "both cores process the whole stimulus");
    CHECK(off->frame.res_gain == NULL && def->frame.res_gain != NULL,
          "the RES-disabled chain builds no gain stage (ESTIMATE), the default chain does (FULL)");
    printf("  RES bypass: max|lin|=%.4f  max err off=%.2e  def=%.2e\n",
           max_abs, max_err_off, max_err_def);
    CHECK(max_abs > 1e-3, "the linear reference carries signal");
    CHECK(max_err_off <= 1e-4 * max_abs,
          "with RES, NR and CNG off the core is a transparent synthesis of "
          "the beamformed linear error");
    CHECK(max_err_def > 1e-2 * max_abs,
          "the default core shapes the same stimulus, so the identity above "
          "is a constraint on the bypass and not on the stimulus");
    free(mics); free(far); free(out_off); free(out_def); free(prev_lin); free(w);
    conv_destroy(off);
    conv_destroy(def);
}

/* Comfort noise fills the AEC-suppressed bins from lib/aec's recipe: an LCG step per bin
 * whose top five bits index kCngTable, the imaginary part a quarter turn (+8 of 32) ahead
 * of the real one. The injected noise owes the caller bounded samples, the same expected
 * per-bin power as the unit-variance Gaussian pair and an advancing sequence; no output
 * bytes are pinned. The pipeline half differences a CNG-on run against a byte-identical
 * CNG-off one: comfort noise is added after every gain, so the difference is the noise. */
static void test_comfort_noise_contract(void) {
    AecConfig cfg = array_test_default_config(16000);
    ConvChainOptions on = conv_default_options();
    ConvChainOptions off = conv_default_options();
    ConvChain* p_on;
    ConvChain* p_off;
    float* mics;
    float* far;
    float* out_on;
    float* out_off;
    float* prev_noise;
    float* noise_stream;
    Complex* w;
    double sum = 0.0;
    double sq = 0.0;
    double worst_pair = 0.0;
    double noise_sq = 0.0;
    double corr_sum = 0.0;
    float bound = 0.0f;
    const int hops = 240;
    int hop;
    int n;
    int h, i, ch, k;
    int injected = 0;
    int prev_active = 0;
    int corr_n = 0;
    int finite = 1;

    for (i = 0; i < 32; ++i) {
        float v = kCngTable[i];
        float q = kCngTable[(i + 8) & 31];
        double power = (double)v * v + (double)q * q;
        if (fabsf(v) > bound) bound = fabsf(v);
        sum += v;
        sq += (double)v * v;
        if (fabs(power - 2.0) > worst_pair) worst_pair = fabs(power - 2.0);
    }
    CHECK(bound <= 1.41421366f,
          "cng: every table entry is bounded by sqrt(2) (no unbounded tail)");
    CHECK(fabs(sum / 32.0) < 1e-6,
          "cng: the table is zero-mean over a uniform index");
    CHECK(fabs(sq / 32.0 - 1.0) < 1e-6,
          "cng: the table has unit mean square -- same expected per-bin power "
          "as the unit-variance Gaussian pair it replaced");
    CHECK(worst_pair < 1e-6,
          "cng: real and imaginary entries are a quarter turn apart, so each "
          "bin's injected power is exactly 2*amplitude^2");

    off.enable_cng = 0;
    p_on = conv_create(&cfg, &on);
    p_off = conv_create(&cfg, &off);
    CHECK(p_on != NULL && p_off != NULL, "cng: both chains create");
    if (!p_on || !p_off) {
        conv_destroy(p_on);
        conv_destroy(p_off);
        return;
    }
    hop = p_on->hop;
    n = p_on->n_freqs;
    mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    out_on = (float*)calloc((size_t)hop, sizeof(float));
    out_off = (float*)calloc((size_t)hop, sizeof(float));
    prev_noise = (float*)calloc((size_t)hop, sizeof(float));
    noise_stream = (float*)calloc((size_t)hops * (size_t)hop, sizeof(float));
    w = (Complex*)calloc(
        (size_t)AEC_ARRAY_MICS * (size_t)n, sizeof(Complex));
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
        for (k = 0; k < n; ++k) w[(size_t)ch * (size_t)n + (size_t)k].r = 0.25f;

    for (h = 0; h < hops; ++h) {
        fill_inputs(mics, far, hop, 16000, h);
        if (conv_hop(p_on, mics, far, w, NULL, out_on) != AEC_OK ||
            conv_hop(p_off, mics, far, w, NULL, out_off) != AEC_OK) {
            finite = 0;
            break;
        }
        {
            double hop_sq = 0.0;
            double cross = 0.0;
            double prev_sq = 0.0;
            for (i = 0; i < hop; ++i) {
                float noise = out_on[i] - out_off[i];
                noise_stream[(size_t)h * (size_t)hop + (size_t)i] = noise;
                if (!isfinite(out_on[i]) || !isfinite(noise)) finite = 0;
                noise_sq += (double)noise * noise;
                hop_sq += (double)noise * noise;
                cross += (double)noise * prev_noise[i];
                prev_sq += (double)prev_noise[i] * prev_noise[i];
                if (noise != 0.0f) injected = 1;
            }
            /* A generator whose state never advanced would hand every bin of
             * every hop the same table entry, so each hop's injected spectrum
             * would be the amplitude profile times one fixed complex constant
             * -- successive hops would then be nearly collinear. Advancing per
             * bin decorrelates them. */
            if (hop_sq > 0.0 && prev_active && prev_sq > 0.0) {
                corr_sum += fabs(cross / sqrt(hop_sq * prev_sq));
                corr_n += 1;
            }
            prev_active = (hop_sq > 0.0);
            if (prev_active)
                for (i = 0; i < hop; ++i) prev_noise[i] = out_on[i] - out_off[i];
        }
    }
    CHECK(finite, "cng: every CNG-on sample and every injected sample is finite");
    CHECK(injected && noise_sq > 0.0,
          "cng: comfort noise is actually injected (CNG-on and CNG-off cores "
          "carry different energy)");
    CHECK(corr_n > 20 && corr_sum / corr_n < 0.5,
          "cng: successive hops' injected noise is decorrelated -- the "
          "generator advances instead of re-injecting one fixed spectrum");

    /* The seed is per-INSTANCE: a second, independently created pair fed the
     * same input must inject the byte-identical stream. A process-wide
     * generator state would fail here. */
    {
        ConvChain* q_on = conv_create(&cfg, &on);
        ConvChain* q_off = conv_create(&cfg, &off);
        int same = 1;
        if (q_on && q_off) {
            for (h = 0; h < hops && same; ++h) {
                fill_inputs(mics, far, hop, 16000, h);
                if (conv_hop(q_on, mics, far, w, NULL, out_on) != AEC_OK ||
                    conv_hop(q_off, mics, far, w, NULL, out_off) != AEC_OK) {
                    same = 0;
                    break;
                }
                for (i = 0; i < hop; ++i)
                    if (out_on[i] - out_off[i] !=
                        noise_stream[(size_t)h * (size_t)hop + (size_t)i])
                        same = 0;
            }
        }
        CHECK(q_on && q_off && same,
              "cng: a second, independently created core injects the identical "
              "noise stream (the seed is per-instance)");
        conv_destroy(q_on);
        conv_destroy(q_off);
    }

    printf("cng: %d hops, injected RMS %.3e, hop-to-hop |corr| %.4f over %d "
           "pairs (bound %.8f)\n",
           hops, sqrt(noise_sq / ((double)hops * (double)hop)),
           corr_n ? corr_sum / corr_n : -1.0, corr_n, bound);
    free(mics); free(far); free(out_on); free(out_off);
    free(prev_noise); free(noise_stream); free(w);
    conv_destroy(p_on);
    conv_destroy(p_off);
}

/* ---------------------------------------------------------------------------
 * Pool placement invariance: everything this module owns is carved from one caller block,
 * so where it sits and what surrounds it must not reach the audio (a consumer reading past
 * its carve, or a decision on a pointer value). One build runs four times with the pool at
 * four different 16-aligned bases and a different filler on both sides; byte-equal output
 * is the main assertion, and the filler check afterwards is the write-side half.
 * ------------------------------------------------------------------------ */

#define POOL_PLACEMENT_GUARD  4096u
#define POOL_PLACEMENT_FRAMES 64

static void test_pool_placement_invariance(int sample_rate, int fft_size) {
    /* 16-aligned (the published alignment) but deliberately not all congruent
     * modulo 32 or 64: a kernel quietly depending on a wider alignment than
     * the descriptor promises separates these. */
    static const size_t bases[] = { 0u, 16u, 32u, 48u };
    static const int fillers[] = { 0xa5, 0x5a, 0x00, 0xff };
    const int placements = (int)(sizeof(bases) / sizeof(bases[0]));
    AecConfig cfg;
    size_t app_bytes = 0;   /* the array pool, the whole application block */
    AecMemReq req;
    float* reference = NULL;
    float* current = NULL;
    float* microphones = NULL;
    float* ref = NULL;
    float* out = NULL;
    Complex* weights = NULL;
    size_t stream_bytes = 0;
    int hop = 0;
    int n_freqs = 0;
    int placement;
    int built = 0;
    int streamed = 0;
    int identical = 1;
    int guards_intact = 1;

    cfg = array_test_default_config(sample_rate);
    cfg.fft_size = fft_size;
    if (aec_get_mem_requirements(&cfg, &req) != 0) {
        CHECK(0, "pool placement memory requirement query succeeds");
        return;
    }
    app_bytes = ALIGN16((size_t)req.bytes);

    for (placement = 0; placement < placements; ++placement) {
        size_t lead = POOL_PLACEMENT_GUARD + bases[placement];
        size_t block_bytes = lead + app_bytes + POOL_PLACEMENT_GUARD;
        unsigned char* block = NULL;
        unsigned char* pool;
        Aec* p;
        ConvChain* chain;
        AecArrayView pre;
        float* destination;
        size_t i;
        int frame;

        if (posix_memalign(
                (void**)&block, 16u, block_bytes) != 0)
            break;
        memset(block, fillers[placement], block_bytes);
        pool = block + lead;

        p = aec_init_ex(pool, app_bytes, &cfg, &req);
        if (!p) { free(block); break; }
        chain = conv_attach(p, &cfg, NULL);
        if (!chain) { aec_destroy(p); free(block); break; }
        built += 1;

        if (built == 1) {
            hop = aec_hop_size(p);
            n_freqs = array_test_n_freqs(p);
            stream_bytes = (size_t)POOL_PLACEMENT_FRAMES * (size_t)hop *
                           sizeof(float);
            reference = (float*)malloc(stream_bytes);
            current = (float*)malloc(stream_bytes);
            microphones = (float*)calloc(
                (size_t)hop * AEC_ARRAY_MICS, sizeof(float));
            ref = (float*)calloc((size_t)hop, sizeof(float));
            out = (float*)calloc((size_t)hop, sizeof(float));
            weights = (Complex*)calloc(
                (size_t)AEC_ARRAY_MICS * n_freqs, sizeof(Complex));
            CHECK(reference && current && microphones && ref && out && weights,
                  "pool placement buffers allocate");
            if (!reference || !current || !microphones || !ref || !out ||
                !weights) {
                conv_destroy(chain);
                aec_destroy(p);
                free(block);
                break;
            }
            for (i = 0; i < (size_t)AEC_ARRAY_MICS * n_freqs; ++i)
                weights[i].r = 0.25f;
        }

        destination = (built == 1) ? reference : current;
        for (frame = 0; frame < POOL_PLACEMENT_FRAMES; ++frame) {
            fill_inputs(microphones, ref, hop, sample_rate, frame);
            if (array_test_front(
                    p, microphones, ref, &pre) != AEC_OK ||
                conv_post(chain, weights, NULL, out) != AEC_OK)
                break;
            memcpy(destination + (size_t)frame * hop, out,
                   (size_t)hop * sizeof(float));
        }
        if (frame == POOL_PLACEMENT_FRAMES) streamed += 1;

        for (i = 0; i < lead; ++i)
            if (block[i] != (unsigned char)fillers[placement])
                guards_intact = 0;
        for (i = lead + app_bytes; i < block_bytes; ++i)
            if (block[i] != (unsigned char)fillers[placement])
                guards_intact = 0;

        if (built > 1 && memcmp(reference, current, stream_bytes) != 0)
            identical = 0;

        conv_destroy(chain);
        aec_destroy(p);
        free(block);
    }

    CHECK(built == placements && streamed == placements,
          "every shifted pool base constructs and streams");
    CHECK(streamed == placements && identical,
          "output is byte-identical across four pool bases surrounded by "
          "four different fillers");
    CHECK(streamed == placements && guards_intact,
          "processing writes nothing outside the queried pool");

    free(weights);
    free(out);
    free(ref);
    free(microphones);
    free(current);
    free(reference);
}

/* After aec_reset() and the reset of the stages it chains (conv_reset()), the
 * core and the chain equal a freshly created pair with the same config. The subject
 * is warmed on a different stretch of the scene than the one compared, so what it remembers
 * is wrong for what follows. Run with CNG both ON and OFF: conv_reset() re-seeds the
 * comfort-noise generator, so the CNG path is compared for real. */
static void test_reset_equals_fresh_instance(int sample_rate, int fft_size,
                                             int enable_cng) {
    enum { WARM = 400, WARM_OFFSET = 100000, COMPARE = 200 };
    AecConfig cfg;
    ConvChainOptions opt = conv_default_options();
    ConvChain* fresh;
    ConvChain* warmed;
    float* mics;
    float* far;
    float* out_a;
    float* out_b;
    Complex* w;
    long differing = 0;
    int first_bad = -1;
    int hop, n_freqs, h, ch, k, ok = 1;

    cfg = array_test_default_config(sample_rate);
    cfg.fft_size = fft_size;
    opt.enable_cng = enable_cng;
    fresh = conv_create(&cfg, &opt);
    warmed = conv_create(&cfg, &opt);
    CHECK(fresh != NULL && warmed != NULL, "reset parity: both chains create");
    if (!fresh || !warmed) {
        conv_destroy(fresh);
        conv_destroy(warmed);
        return;
    }

    hop = fresh->hop;
    n_freqs = fresh->n_freqs;
    mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    out_a = (float*)calloc((size_t)hop, sizeof(float));
    out_b = (float*)calloc((size_t)hop, sizeof(float));
    w = (Complex*)calloc((size_t)AEC_ARRAY_MICS * (size_t)n_freqs,
                         sizeof(Complex));
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
        for (k = 0; k < n_freqs; ++k)
            w[(size_t)ch * (size_t)n_freqs + (size_t)k].r = 0.25f;

    for (h = 0; h < WARM && ok; ++h) {
        fill_inputs(mics, far, hop, sample_rate, WARM_OFFSET + h);
        if (conv_hop(warmed, mics, far, w, NULL, out_b) != AEC_OK)
            ok = 0;
    }
    CHECK(ok, "reset parity: the warm phase runs to completion");
    conv_reset(warmed);

    for (h = 0; h < COMPARE && ok; ++h) {
        fill_inputs(mics, far, hop, sample_rate, h);
        if (conv_hop(fresh, mics, far, w, NULL, out_a) != AEC_OK ||
            conv_hop(warmed, mics, far, w, NULL, out_b) != AEC_OK) {
            ok = 0;
            break;
        }
        if (memcmp(out_a, out_b, (size_t)hop * sizeof(float)) != 0) {
            differing++;
            if (first_bad < 0) first_bad = h;
        }
    }
    CHECK(ok, "reset parity: the compare phase runs to completion");
    if (differing != 0)
        printf("  reset parity @ %d Hz/%d cng=%d: %ld of %d hops differ, "
               "first at %d\n",
               sample_rate, fft_size, enable_cng, differing, COMPARE, first_bad);
    CHECK(differing == 0,
          "aec_reset leaves an array chain byte-identical to a fresh one");

    free(mics); free(far); free(out_a); free(out_b); free(w);
    conv_destroy(fresh);
    conv_destroy(warmed);
}

/* The comfort noise is scaled by the denoiser gain of its bin, bounded below by
 * CNG_NR_GAIN_FLOOR (the denoiser never sees the fill). Witness on a broadband
 * echo-plus-noise stimulus where the denoiser settles far below the floor: the injected
 * noise is the exact difference between a CNG-on and a CNG-off core (added after every gain,
 * same draw sequence). With the denoiser off that difference is the unscaled reference
 * d_ref; with it on, d_nr must equal CNG_NR_GAIN_FLOOR * d_ref sample for sample. The
 * residual d_nr - floor * d_ref must carry almost no energy, and the energy ratio must sit
 * at floor^2 (no scaling at all would leave it at 1). */
static void test_comfort_noise_follows_nr_gain(void) {
    const double floor_amp = 0.31622777, floor_sq = floor_amp * floor_amp;
    AecConfig cfg = array_test_default_config(16000);
    ConvChainOptions opt[4];
    ConvChain* p[4] = { NULL, NULL, NULL, NULL };
    float* out[4] = { NULL, NULL, NULL, NULL };
    float *mics, *far;
    Complex* w;
    uint32_t rng = 0xC0FFEEu;
    int hop, n, settle_hops, hops, h, k, ch, i, created = 1, finite = 1;
    double e_nr = 0.0, e_ref = 0.0, e_res = 0.0;

    /* [0] CNG off / NR on, [1] CNG on / NR on, [2] CNG off / NR off,
     * [3] CNG on / NR off. */
    for (i = 0; i < 4; ++i) {
        opt[i] = conv_default_options();
        opt[i].enable_cng = (i & 1);
        opt[i].enable_nr = (i < 2);
        p[i] = conv_create(&cfg, &opt[i]);
        if (!p[i]) created = 0;
    }
    CHECK(created, "cng/nr: all four chains create");
    if (!created) { for (i = 0; i < 4; ++i) conv_destroy(p[i]); return; }
    hop = p[0]->hop;
    n = p[0]->n_freqs;
    settle_hops = anr_retime_frames(20, 16000, hop) + 40;
    hops = settle_hops + 200;
    mics = (float*)calloc((size_t)hop * AEC_ARRAY_MICS, sizeof(float));
    far = (float*)calloc((size_t)hop, sizeof(float));
    for (i = 0; i < 4; ++i) out[i] = (float*)calloc((size_t)hop, sizeof(float));
    w = (Complex*)calloc(
        (size_t)AEC_ARRAY_MICS * (size_t)n, sizeof(Complex));
    for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
        for (k = 0; k < n; ++k) w[(size_t)ch * (size_t)n + (size_t)k].r = 0.25f;

    for (h = 0; h < hops; ++h) {
        for (k = 0; k < hop; ++k) {
            float r, noise;
            rng = rng * 1664525u + 1013904223u;
            r = 0.25f * ((float)((rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f);
            rng = rng * 1664525u + 1013904223u;
            noise = 0.3f * ((float)((rng >> 8) & 0xFFFFu) / 32768.0f - 1.0f);
            far[k] = r;
            for (ch = 0; ch < AEC_ARRAY_MICS; ++ch)
                mics[(size_t)k * AEC_ARRAY_MICS + (size_t)ch] = 0.5f * r + noise;
        }
        for (i = 0; i < 4; ++i) {
            if (conv_hop(p[i], mics, far, w, NULL, out[i]) != AEC_OK)
                finite = 0;
        }
        if (h < settle_hops) continue;
        for (k = 0; k < hop; ++k) {
            double d_nr = out[1][k] - out[0][k];
            double d_ref = out[3][k] - out[2][k];
            double res = d_nr - floor_amp * d_ref;
            if (!isfinite(d_nr) || !isfinite(d_ref)) finite = 0;
            e_nr += d_nr * d_nr;
            e_ref += d_ref * d_ref;
            e_res += res * res;
        }
    }
    CHECK(finite && e_ref > 0.0,
          "cng/nr: the NR-off reference core injects comfort noise");
    CHECK(e_nr > 0.0 && e_nr > 0.8 * floor_sq * e_ref && e_nr < 1.5 * floor_sq * e_ref,
          "cng/nr: with the denoiser on, the injected noise carries floor^2 of "
          "the NR-off energy -- the fill is scaled by G_nr down to the floor");
    CHECK(e_res < 0.02 * e_ref,
          "cng/nr: the injected noise is floor * reference sample for sample "
          "-- the scaling is bounded by exactly CNG_NR_GAIN_FLOOR");
    for (i = 0; i < 4; ++i) { conv_destroy(p[i]); free(out[i]); }
    free(mics); free(far); free(w);
}

int main(void) {
    test_nr_bypass_output_identity();
    test_res_bypass_passthrough();
    test_comfort_noise_contract();
    test_comfort_noise_follows_nr_gain();
    test_pool_placement_invariance(16000, 256);
    test_pool_placement_invariance(16000, 512);
    test_pool_placement_invariance(48000, 1024);
    test_reset_equals_fresh_instance(16000, 256, 1);
    test_reset_equals_fresh_instance(16000, 256, 0);
    test_reset_equals_fresh_instance(16000, 512, 1);
    test_reset_equals_fresh_instance(48000, 1024, 1);

    if (failures) {
        printf("%d test(s) failed\n", failures);
        return 1;
    }
    printf("All 4ch chain tests passed\n");
    return 0;
}
