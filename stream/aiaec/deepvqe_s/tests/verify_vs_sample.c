/* Verifies deepvqe_stream.c (deepvqe_stream_pre_process/post_process) against an independent
 * re-implementation of the same per-hop lifecycle (driver A below): int16->float mic+far (no
 * HPF) -> ac_stft_push_planar x2 -> deepvqe_prepost_pre_process_freq -> deepvqe_prepost_frame_open
 * -> NPU -> deepvqe_prepost_frame_finish/frame_bypass -> deepvqe_prepost_post_process_freq ->
 * ac_istft_push_planar -> int16 saturate, built directly against
 * lib/aiaec/deepvqe_s/deepvqe_prepost.h and audiolib, never
 * against deepvqe_stream.c itself -- the same lib-level core deepvqe_stream.c wraps, called by
 * hand instead of through the wrapper. Both drivers run the SAME deterministic fake accelerator
 * (pure function of a global hop counter + a force_fail flag) and must produce byte-identical
 * PCM and an identical (pre_rc, ai_rc, post_rc, written, generation) trace, hop for hop, at
 * frames_per_call == 1 -- an independent-implementation cross-check of the wrapper, not a
 * comparison against any specific caller's source file (the board sample calls
 * aiaec_preprocess()/aiaec_postprocess(), one layer up, not deepvqe_stream_*
 * directly; stream/aiaec/tests/test_aiaec_block.c covers that layer, including
 * frames_per_call > 1).
 *
 * --inject-shift: delays the comparison's view of driver B's PCM by one hop, to prove this
 * harness's comparison can actually detect a divergence. Injects nothing into deepvqe_stream.c.
 *
 * --ola-matters: a separate run (run_ola_matters_proof(), its own pair of driver instances, not
 * involved in the A-vs-B comparison above) builds driver A with AC_OLA_ENV_NORM and driver C
 * with AC_OLA_ACCUMULATE, same board_init()/board_pre()/board_post() helpers, and diffs the two
 * to show the OLA mode itself is load-bearing -- not just hop alignment. This never touches
 * deepvqe_stream.c; DeepVqeStreamConfig has no public hook to override the OLA mode, so proving
 * this on the actual B driver isn't possible without editing deepvqe_stream.c, which this test
 * does not do. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "deepvqe_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"

/* DEEPVQE_TIME_ORDER itself comes from deepvqe_process.h via deepvqe_stream.h ->
 * deepvqe_prepost.h -> deepvqe_process.h; used directly below instead of a local, driftable
 * copy. */

/* ---- shared deterministic fake accelerator: pure function of (global hop, force_fail) ---- */
static int shared_fake(long long ghop, int force_fail, DeepVqePrepostOutputs *out) {
    size_t i;
    for (i = 0; i < out->taps_elements; ++i)
        out->taps[i] = 0.05f * (float)(((ghop + (long long)i) % 37) - 18);
    if (force_fail) return -1;   /* valid finite taps AND a failure status: only the status can fail the hop closed */
    /* No state registered in this test (state[id] all NULL), so state_out[*] are all NULL too --
     * nothing else to write. */
    return 0;
}

/* ---- driver: independent re-implementation of deepvqe_stream.c's own per-hop lifecycle, OLA
 * mode parameterized so the same code builds run_matrix()'s reference driver (A, ACCUMULATE,
 * matching deepvqe_stream.c) and the OLA-matters proof's two driver instances (ENV_NORM vs
 * ACCUMULATE). ---- */
typedef struct BoardDriver {
    DeepVqeGrid grid;
    FftHandle *fft;
    AcStft *stft_mic; void *stft_mic_pool;
    AcStft *stft_far; void *stft_far_pool;
    AcIstft *istft; void *istft_pool;
    DeepVqePrepost *prepost; void *prepost_pool;
    float *mic_f, *far_f, *hop_out, *mic_re, *mic_im, *far_re, *far_im, *enh_re, *enh_im;
} BoardDriver;

static int board_init(BoardDriver *d, const DeepVqeGrid *grid, AcOlaMode ola) {
    AcStftConfig scfg;
    DeepVqePrepostConfig pcfg;
    DeepVqePrepostMemReq preq;
    size_t n_bins = (size_t)grid->n_bins, hop = (size_t)grid->hop;
    size_t smb, sfb, ib;

    memset(d, 0, sizeof(*d));
    d->grid = *grid;
    d->fft = fft_create(grid->n_fft);
    if (!d->fft) return -1;

    ac_stft_config_default(&scfg, grid->n_fft);
    scfg.ola = ola;
    scfg.fft = d->fft;

    smb = ac_stft_get_mem_size(&scfg);
    if (smb == 0 || posix_memalign(&d->stft_mic_pool, 16, smb) != 0) return -1;
    d->stft_mic = ac_stft_init(d->stft_mic_pool, smb, &scfg);
    if (!d->stft_mic) return -1;

    sfb = ac_stft_get_mem_size(&scfg);
    if (sfb == 0 || posix_memalign(&d->stft_far_pool, 16, sfb) != 0) return -1;
    d->stft_far = ac_stft_init(d->stft_far_pool, sfb, &scfg);
    if (!d->stft_far) return -1;

    ib = ac_istft_get_mem_size(&scfg);
    if (ib == 0 || posix_memalign(&d->istft_pool, 16, ib) != 0) return -1;
    d->istft = ac_istft_init(d->istft_pool, ib, &scfg);
    if (!d->istft) return -1;

    if (deepvqe_prepost_config_defaults(&pcfg, grid) != 0) return -1;
    if (deepvqe_prepost_get_mem_size(&pcfg, &preq) != 0) return -1;
    if (posix_memalign(&d->prepost_pool, (size_t)preq.alignment, (size_t)preq.bytes) != 0) return -1;
    d->prepost = deepvqe_prepost_init(d->prepost_pool, (size_t)preq.bytes, &pcfg);
    if (!d->prepost) return -1;

    d->mic_f = malloc(hop * sizeof(float));
    d->far_f = malloc(hop * sizeof(float));
    d->hop_out = malloc(hop * sizeof(float));
    d->mic_re = malloc(n_bins * sizeof(float));
    d->mic_im = malloc(n_bins * sizeof(float));
    d->far_re = malloc(n_bins * sizeof(float));
    d->far_im = malloc(n_bins * sizeof(float));
    d->enh_re = malloc(n_bins * sizeof(float));
    d->enh_im = malloc(n_bins * sizeof(float));
    if (!d->mic_f || !d->far_f || !d->hop_out || !d->mic_re || !d->mic_im ||
        !d->far_re || !d->far_im || !d->enh_re || !d->enh_im) return -1;
    return 0;
}

static int board_pre(BoardDriver *d, const int16_t *mic_in, const int16_t *far_in,
                     DeepVqePrepostInputs *in, DeepVqePrepostOutputs *out) {
    int hop = d->grid.hop, k;
    for (k = 0; k < hop; ++k) {
        d->mic_f[k] = (float)mic_in[k] / 32768.0f;
        d->far_f[k] = (float)far_in[k] / 32768.0f;
    }
    ac_stft_push_planar(d->stft_mic, d->mic_f, d->mic_re, d->mic_im);
    ac_stft_push_planar(d->stft_far, d->far_f, d->far_re, d->far_im);
    if (deepvqe_prepost_pre_process_freq(d->prepost, d->mic_re, d->mic_im, d->far_re, d->far_im) != 1)
        return -1;
    if (deepvqe_prepost_frame_open(d->prepost, in, out) != 0) return -1;
    return 1;
}

/* Returns 1 (full hop synthesized), 0 (ENV_NORM's first-push-emits-nothing case, or any mode
 * where the push legitimately returns < hop), -1 (post_process_freq contract error). *written_out
 * carries ac_istft_push_planar's own return value for the trace. */
static int board_post(BoardDriver *d, int ai_rc, int16_t *pcm_out, int *written_out) {
    int rc, prc, hop = d->grid.hop, k, written = 0;
    rc = (ai_rc == 0) ? deepvqe_prepost_frame_finish(d->prepost) : -1;
    if (rc != 0) deepvqe_prepost_frame_bypass(d->prepost);

    prc = deepvqe_prepost_post_process_freq(d->prepost, d->enh_re, d->enh_im);
    if (prc != 0) {
        memset(pcm_out, 0, (size_t)hop * sizeof(int16_t));
        if (written_out) *written_out = -1;
        return -1;
    }
    written = ac_istft_push_planar(d->istft, d->enh_re, d->enh_im, d->hop_out);
    if (written_out) *written_out = written;
    if (written != hop) {
        memset(pcm_out, 0, (size_t)hop * sizeof(int16_t));
        return 0;
    }
    for (k = 0; k < hop; ++k) {
        float v = d->hop_out[k] * 32768.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        pcm_out[k] = (int16_t)v;
    }
    return 1;
}

static void board_reset(BoardDriver *d) {
    ac_stft_reset(d->stft_mic);
    ac_stft_reset(d->stft_far);
    ac_istft_reset(d->istft);
    deepvqe_prepost_reset(d->prepost);
}

static uint32_t board_generation(const BoardDriver *d) {
    return deepvqe_prepost_state_generation(d->prepost);
}

/* ---- main comparison run: hand-rolled reference driver A (ACCUMULATE) vs. deepvqe_stream B ---- */

typedef struct Stats {
    long long hops, mismatches, feat_mismatches;
    long long passthrough_violations, generation_advanced_on_bypass;
} Stats;

static int cmp_floats_exact(const float *a, const float *b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

static int run_matrix(int inject_shift) {
    DeepVqeGridSpec spec;
    DeepVqeGrid grid;
    BoardDriver A;
    DeepVqeStreamConfig cfg;
    DeepVqeStreamMemReq req;
    DeepVqeStream *B;
    void *pool_b;
    int hop;
    long long ghop = 0;
    Stats st;
    int16_t *mic_in, *far_in, *pcm_out_a, *pcm_out_b, *prev_pcm_out_b;
    long long fault_hop = DEEPVQE_TIME_ORDER + 1;   /* well past the ring's own startup window */
    long long phase0_hops = DEEPVQE_TIME_ORDER + 3;
    long long phase1_hops = 3;
    long long phase, phase_hops, hip;

    memset(&st, 0, sizeof(st));
    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000; spec.n_fft = 512; spec.delay_depth = 32; spec.frames_per_call = 1;
    if (deepvqe_grid_make(&grid, &spec) != DEEPVQE_GRID_OK) {
        fprintf(stderr, "deepvqe_grid_make failed\n");
        return -1;
    }
    hop = grid.hop;

    /* ACCUMULATE, matching deepvqe_stream.c -- the reference B is compared against (run_ola_matters_proof()
     * contrasts it with ENV_NORM). */
    if (board_init(&A, &grid, AC_OLA_ACCUMULATE) != 0) { fprintf(stderr, "board_init failed\n"); return -1; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    if (deepvqe_stream_get_mem_size(&cfg, &req) != 0) { fprintf(stderr, "get_mem_size failed\n"); return -1; }
    if (posix_memalign(&pool_b, (size_t)req.alignment, (size_t)req.bytes) != 0) return -1;
    B = deepvqe_stream_init_ex(&cfg, pool_b, (size_t)req.bytes, &req);
    if (!B) { fprintf(stderr, "deepvqe_stream_init_ex failed\n"); return -1; }

    mic_in = calloc((size_t)hop, sizeof(int16_t));
    far_in = calloc((size_t)hop, sizeof(int16_t));
    pcm_out_a = calloc((size_t)hop, sizeof(int16_t));
    pcm_out_b = calloc((size_t)hop, sizeof(int16_t));
    prev_pcm_out_b = calloc((size_t)hop, sizeof(int16_t));

    for (phase = 0; phase < 2; ++phase) {
        if (phase == 0) phase_hops = phase0_hops;
        else { board_reset(&A); deepvqe_stream_reset(B); phase_hops = phase1_hops; }

        for (hip = 0; hip < phase_hops; ++hip, ++ghop) {
            DeepVqeHopIO io_b;
            DeepVqePrepostInputs in_a;
            DeepVqePrepostOutputs out_a;
            int pre_rc_a, pre_rc_b, post_rc_a, post_rc_b, ai_rc;
            int written_a = -2, written_b = -2;
            uint32_t gen_before_a, gen_before_b, gen_after_a, gen_after_b;
            int force_fail = (phase == 0 && hip == fault_hop) ? 1 : 0;
            int k;

            for (k = 0; k < hop; ++k) {
                mic_in[k] = (int16_t)(((long long)k * 5 + ghop * 3) % 97 - 48);
                far_in[k] = (int16_t)(((long long)k * 11 + ghop * 7) % 89 - 44);
            }

            gen_before_a = board_generation(&A);
            gen_before_b = deepvqe_stream_state_generation(B);

            pre_rc_a = board_pre(&A, mic_in, far_in, &in_a, &out_a);
            pre_rc_b = deepvqe_stream_pre_process(B, mic_in, far_in, &io_b);
            if (pre_rc_a != pre_rc_b) {
                fprintf(stderr, "hop %lld: pre_rc mismatch A=%d B=%d\n", ghop, pre_rc_a, pre_rc_b);
                st.mismatches++;
            }
            if (pre_rc_a == 1 && pre_rc_b == 1) {
                if (in_a.spectrum_ri_elements != io_b.inputs.spectrum_ri_elements ||
                    !cmp_floats_exact(in_a.mic, io_b.inputs.mic, in_a.spectrum_ri_elements) ||
                    !cmp_floats_exact(in_a.far, io_b.inputs.far, in_a.spectrum_ri_elements)) {
                    fprintf(stderr, "hop %lld: feature (mic/far spectrum) mismatch A vs B\n", ghop);
                    st.feat_mismatches++;
                }
            }

            ai_rc = shared_fake(ghop, force_fail, &out_a);
            {
                int ai_rc_b = shared_fake(ghop, force_fail, &io_b.outputs);
                if (ai_rc_b != ai_rc) { fprintf(stderr, "hop %lld: fake rc mismatch\n", ghop); st.mismatches++; }
            }

            post_rc_a = board_post(&A, ai_rc, pcm_out_a, &written_a);
            post_rc_b = deepvqe_stream_post_process(B, ai_rc, pcm_out_b);
            /* B doesn't expose `written` directly; infer it from post_rc_b per deepvqe_stream.h's
             * documented 3-way split (1 => written==hop; 0 => written==0, the frames_per_call > 1
             * pre-roll case, never reached here since frames_per_call == 1; -1 => post_process_freq
             * contract error, written undefined/not applicable). */
            written_b = (post_rc_b == 1) ? hop : (post_rc_b == 0 ? 0 : -1);

            if (post_rc_a != post_rc_b) {
                fprintf(stderr, "hop %lld: post_rc mismatch A=%d B=%d\n", ghop, post_rc_a, post_rc_b);
                st.mismatches++;
            }
            if (written_a != written_b) {
                fprintf(stderr, "hop %lld: written mismatch A=%d B=%d\n", ghop, written_a, written_b);
                st.mismatches++;
            }
            /* Explicit assertion of the documented return-code pattern, not just "PCM happens to
             * be zero": AC_OLA_ACCUMULATE has no first-push special case, so every non-faulted
             * hop from hop 0 onward must be post_rc == 1 on both sides. */
            if (!force_fail) {
                if (post_rc_a != 1 || post_rc_b != 1) {
                    fprintf(stderr, "hop %lld: expected post_rc==1 on BOTH sides, got A=%d B=%d\n",
                           ghop, post_rc_a, post_rc_b);
                    st.mismatches++;
                }
            }

            {
                const int16_t *b_for_compare = inject_shift ? prev_pcm_out_b : pcm_out_b;
                if (memcmp(pcm_out_a, b_for_compare, (size_t)hop * sizeof(int16_t)) != 0) {
                    fprintf(stderr, "hop %lld: PCM mismatch (inject_shift=%d)\n", ghop, inject_shift);
                    st.mismatches++;
                }
            }
            memcpy(prev_pcm_out_b, pcm_out_b, (size_t)hop * sizeof(int16_t));

            gen_after_a = board_generation(&A);
            gen_after_b = deepvqe_stream_state_generation(B);
            if (gen_after_a != gen_after_b) {
                fprintf(stderr, "hop %lld: generation mismatch A=%u B=%u\n", ghop, gen_after_a, gen_after_b);
                st.mismatches++;
            }

            if (force_fail) {
                /* Bypass-is-silence-not-passthrough, and bypass must not advance the generation
                 * (only a non-finite frame_finish rejection would -- not exercised by this fake,
                 * which fails via ai_rc != 0, i.e. a pure bypass). */
                int identical_to_mic = (memcmp(pcm_out_b, mic_in, (size_t)hop * sizeof(int16_t)) == 0);
                if (identical_to_mic) {
                    fprintf(stderr, "hop %lld: FAULT-INJECTED hop's PCM equals mic_in verbatim -- "
                                    "bypass looks like a passthrough, not fail-closed silence\n", ghop);
                    st.passthrough_violations++;
                }
                if (gen_after_b != gen_before_b || gen_after_a != gen_before_a) {
                    fprintf(stderr, "hop %lld: generation advanced on a plain bypass hop "
                                    "(before A=%u B=%u, after A=%u B=%u) -- bypass must not touch "
                                    "the generation\n", ghop, gen_before_a, gen_before_b, gen_after_a, gen_after_b);
                    st.generation_advanced_on_bypass++;
                }
            }
            st.hops++;
        }
    }

    printf("hops=%lld mismatches=%lld feat_mismatches=%lld passthrough_violations=%lld "
          "generation_advanced_on_bypass=%lld %s\n",
          st.hops, st.mismatches, st.feat_mismatches, st.passthrough_violations,
          st.generation_advanced_on_bypass,
          (st.mismatches == 0 && st.feat_mismatches == 0 && st.passthrough_violations == 0 &&
           st.generation_advanced_on_bypass == 0) ? "PASS" : "FAIL");

    free(mic_in); free(far_in); free(pcm_out_a); free(pcm_out_b); free(prev_pcm_out_b);
    return (st.mismatches == 0 && st.feat_mismatches == 0 && st.passthrough_violations == 0 &&
            st.generation_advanced_on_bypass == 0) ? 0 : 1;
}

/* ---- OLA-matters proof: driver A (AC_OLA_ENV_NORM) vs. driver C (AC_OLA_ACCUMULATE) on
 * IDENTICAL inputs and an identical fake, to show the OLA mode itself changes the output (not
 * just hop alignment). Both are independent re-implementations built by this test file;
 * deepvqe_stream.c is never touched or involved in this comparison. ---- */
static int run_ola_matters_proof(void) {
    DeepVqeGridSpec spec;
    DeepVqeGrid grid;
    BoardDriver A, C;
    int hop, k;
    long long ghop;
    int16_t *mic_in, *far_in, *pcm_a, *pcm_c;
    long long mismatches = 0;
    const long long hops = DEEPVQE_TIME_ORDER + 4;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000; spec.n_fft = 512; spec.delay_depth = 32; spec.frames_per_call = 1;
    if (deepvqe_grid_make(&grid, &spec) != DEEPVQE_GRID_OK) return -1;
    hop = grid.hop;

    if (board_init(&A, &grid, AC_OLA_ENV_NORM) != 0) return -1;
    if (board_init(&C, &grid, AC_OLA_ACCUMULATE) != 0) return -1;

    mic_in = calloc((size_t)hop, sizeof(int16_t));
    far_in = calloc((size_t)hop, sizeof(int16_t));
    pcm_a = calloc((size_t)hop, sizeof(int16_t));
    pcm_c = calloc((size_t)hop, sizeof(int16_t));

    for (ghop = 0; ghop < hops; ++ghop) {
        DeepVqePrepostInputs in_a, in_c;
        DeepVqePrepostOutputs out_a, out_c;
        int pre_a, pre_c, post_a, post_c, written_a = -2, written_c = -2, ai_rc;

        for (k = 0; k < hop; ++k) {
            mic_in[k] = (int16_t)(((long long)k * 5 + ghop * 3) % 97 - 48);
            far_in[k] = (int16_t)(((long long)k * 11 + ghop * 7) % 89 - 44);
        }
        pre_a = board_pre(&A, mic_in, far_in, &in_a, &out_a);
        pre_c = board_pre(&C, mic_in, far_in, &in_c, &out_c);
        ai_rc = shared_fake(ghop, 0, &out_a);
        (void)shared_fake(ghop, 0, &out_c);
        post_a = board_post(&A, ai_rc, pcm_a, &written_a);
        post_c = board_post(&C, ai_rc, pcm_c, &written_c);
        (void)pre_c;

        if (pre_a == 1 && pre_c == 1) {
            if (post_a != post_c || written_a != written_c ||
                memcmp(pcm_a, pcm_c, (size_t)hop * sizeof(int16_t)) != 0) {
                printf("ola-matters: hop %lld diverges as expected (ENV_NORM post=%d written=%d "
                      "vs ACCUMULATE post=%d written=%d, PCM %s)\n",
                      ghop, post_a, written_a, post_c, written_c,
                      memcmp(pcm_a, pcm_c, (size_t)hop * sizeof(int16_t)) != 0 ? "differs" : "same");
                mismatches++;
            }
        }
    }

    free(mic_in); free(far_in); free(pcm_a); free(pcm_c);
    if (mismatches == 0) {
        fprintf(stderr, "ola-matters FAILED: ENV_NORM and ACCUMULATE produced identical traces -- "
                        "the OLA mode has no observable effect in this harness, which means it "
                        "cannot prove the mode is load-bearing\n");
        return 1;
    }
    printf("ola-matters: %lld/%lld hops diverged between ENV_NORM and ACCUMULATE -- OLA mode is "
          "confirmed load-bearing\n", mismatches, hops);
    return 0;
}

/* ---- fail-closed oracle: the stream against an STFT -> ISTFT chain that never sees the library's bypass ----
 * An identity accelerator (CCM taps = 1 + 0j at delay 0, centre frequency) makes the stream's output the mic one hop
 * late. The oracle is that chain built by hand from ac_stft/ac_istft: mic spectrum in, the same spectrum out, except a
 * ZERO spectrum at the failed hop. The stream must equal it sample for sample (+-1 LSB, int16 conversion), so a failed
 * hop that kept the mic spectrum (a passthrough), or one that ignored a reported infer_status, differs from the oracle
 * by thousands of LSB. Three failures: a reported infer_status != 0 with valid finite taps, non-finite taps with
 * infer_status 0 (refused), and both. The oracle is independent of deepvqe_prepost_frame_bypass(). */
static void identity_taps(DeepVqePrepostOutputs *out) {
    const size_t tapn = 9, bins = out->taps_elements / (tapn * 2u);
    size_t b;
    memset(out->taps, 0, out->taps_elements * sizeof(float));
    for (b = 0; b < bins; ++b) out->taps[b * tapn * 2u + 1u * 2u] = 1.0f;
}

static int run_fail_closed_oracle(int kind) {   /* 0 status only, 1 non-finite taps, 2 both */
    DeepVqeGridSpec spec;
    DeepVqeGrid grid;
    BoardDriver O;
    DeepVqeStreamConfig cfg;
    DeepVqeStreamMemReq req;
    DeepVqeStream *B;
    void *pool_b = NULL;
    int16_t *mic_in, *prev_mic, *far_in, *pcm_b, *pcm_o;
    const long long hops = DEEPVQE_TIME_ORDER + 12, fault_hop = DEEPVQE_TIME_ORDER + 4;
    long long ghop, bad = 0, discriminating = 0;
    int hop, k;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000; spec.n_fft = 512; spec.delay_depth = 32; spec.frames_per_call = 1;
    if (deepvqe_grid_make(&grid, &spec) != DEEPVQE_GRID_OK) return 1;
    hop = grid.hop;
    if (board_init(&O, &grid, AC_OLA_ACCUMULATE) != 0) return 1;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    if (deepvqe_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&pool_b, (size_t)req.alignment, (size_t)req.bytes) != 0) return 1;
    B = deepvqe_stream_init_ex(&cfg, pool_b, (size_t)req.bytes, &req);
    if (!B) return 1;
    mic_in = calloc((size_t)hop, sizeof(int16_t)); prev_mic = calloc((size_t)hop, sizeof(int16_t));
    far_in = calloc((size_t)hop, sizeof(int16_t));
    pcm_b = calloc((size_t)hop, sizeof(int16_t)); pcm_o = calloc((size_t)hop, sizeof(int16_t));

    for (ghop = 0; ghop < hops; ++ghop) {
        DeepVqeHopIO io;
        const int fail = ghop == fault_hop;
        long long maxd = 0;

        memcpy(prev_mic, mic_in, (size_t)hop * sizeof(int16_t));
        for (k = 0; k < hop; ++k) mic_in[k] = (int16_t)(((((long long)ghop * hop + k) * 977) % 20001 - 10000) / 2);
        /* the oracle: mic spectrum through, a zero spectrum at the failed hop */
        for (k = 0; k < hop; ++k) O.mic_f[k] = (float)mic_in[k] / 32768.0f;
        ac_stft_push_planar(O.stft_mic, O.mic_f, O.mic_re, O.mic_im);
        if (fail) { memset(O.mic_re, 0, (size_t)grid.n_bins * sizeof(float)); memset(O.mic_im, 0, (size_t)grid.n_bins * sizeof(float)); }
        if (ac_istft_push_planar(O.istft, O.mic_re, O.mic_im, O.hop_out) != hop) { bad++; continue; }
        for (k = 0; k < hop; ++k) {
            float v = O.hop_out[k] * 32768.0f;
            pcm_o[k] = (int16_t)(v > 32767.0f ? 32767.0f : v < -32768.0f ? -32768.0f : v);
        }

        if (deepvqe_stream_pre_process(B, mic_in, far_in, &io) != 1) { bad++; continue; }
        identity_taps(&io.outputs);
        if (fail && kind != 0) io.outputs.taps[io.outputs.taps_elements / 3] = NAN;
        if (deepvqe_stream_post_process(B, fail && kind != 1 ? -1 : 0, pcm_b) < 0) { bad++; continue; }
        for (k = 0; k < hop; ++k) {
            long long d = (long long)pcm_b[k] - pcm_o[k];
            if (d < 0) d = -d;
            if (d > maxd) maxd = d;
        }
        if (maxd > 1) {
            fprintf(stderr, "fail-closed oracle (kind %d) hop %lld: stream differs from the zero-spectrum oracle by %lld LSB%s\n", kind, ghop, maxd,
                    fail ? " (the failed hop)" : "");
            bad++;
        }
        if (fail) {   /* the oracle's failed hop is far from the hop an identity run would return (the previous mic hop) */
            for (k = 0; k < hop; ++k) { long long d = (long long)pcm_o[k] - prev_mic[k]; if (d < 0) d = -d; if (d > 1000) ++discriminating; }
        }
    }
    if (discriminating < hop / 4) { fprintf(stderr, "fail-closed oracle (kind %d): the failed hop is not distinguishable from a normal one (%lld samples)\n", kind, discriminating); bad++; }
    free(mic_in); free(prev_mic); free(far_in); free(pcm_b); free(pcm_o); free(pool_b);
    printf("fail-closed oracle (kind %d: %s): %lld deviations %s\n", kind,
           kind == 0 ? "infer_status only" : kind == 1 ? "non-finite taps" : "both", bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    int inject_shift = (argc > 1 && strcmp(argv[1], "--inject-shift") == 0);
    int ola_matters = (argc > 1 && strcmp(argv[1], "--ola-matters") == 0);

    if (ola_matters) return run_ola_matters_proof();

    if (inject_shift) {
        int rc = run_matrix(1);
        if (rc == 0) {
            fprintf(stderr, "INJECT-SHIFT CHECK FAILED: harness stayed green under a deliberately "
                            "injected one-hop comparison shift -- the comparison has no teeth\n");
            return 1;
        }
        printf("inject-shift check: comparator correctly went RED under the injected shift\n");
        return 0;
    }

    return (run_matrix(0) ? 1 : 0) | run_fail_closed_oracle(0) | run_fail_closed_oracle(1) | run_fail_closed_oracle(2);
}
