/* Verifies ulcnet_stream_pre_process/post_process against an independent re-implementation of the
 * board sample's own per-hop loop (alg_ai3_alignulcnet.c's simple_thread(): Stage 3 alignment-edge
 * check -> stream_reset() + arm reprime -> Stage 4 int16->float + 2 STFT pushes + pre_process_freq
 * -> Stage 5 reprime-bypass or frame_open/accelerator/frame_finish-or-bypass -> post_process_freq
 * -> ac_istft_push_planar -> int16 saturate), built directly against
 * lib/aiaec/ulcnet/ulcnet_prepost.h and audiolib, never against ulcnet_stream.c. Both drivers run
 * the SAME deterministic fake accelerator and must produce byte-identical PCM and an identical
 * (pre_rc, ai_rc, post_rc, written, generation) trace, hop for hop.
 *
 * Board sample read-only reference: ~/Desktop/exp/audio_sample/alg_ai3_alignulcnet.c, stream_reset()
 * (~line 1304: ulcnet_prepost_reset + ac_stft_reset x2 + ac_istft_reset), the per-hop loop
 * (~1555-1634) and its grid constants (ALIGNULCNET_SR=16000, ALIGNULCNET_FFT=512, delay_depth=32).
 * This model's own internal AEC (the sample's Stage 1-2) is NOT part of what this harness verifies
 * -- the stream's contract starts at "caller already has a linear-error hop and a far hop," so
 * both drivers here take err_in/far_in directly, exactly like ulcnet_stream_pre_process does.
 * Never edited; this file re-implements Stage 3-5 against the public
 * lib/aiaec/ulcnet/ulcnet_prepost.h API instead of calling into it.
 *
 * --inject-shift: delays the comparison's view of driver B's PCM by one hop, to prove this
 * harness's comparison can actually detect a divergence. Injects nothing into ulcnet_stream.c. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ulcnet_stream.h"
#include "fft_wrapper.h"
#include "audio_stft.h"

#define ULCNET_SR   16000
#define ULCNET_FFT  512
#define ULCNET_D    32

/* ---- shared deterministic fake accelerator: pure function of (global hop, force_fail) ---- */
static int shared_fake(long long ghop, int force_fail, UlcnetModelIoOutputs *out) {
    size_t i;
    if (force_fail) return -1;   /* writes nothing: mirrors the real accelerator leaving the
                                  * NaN-prefilled output untouched on failure */
    for (i = 0; i < out->spectrum_ri_elements; ++i)
        out->output[i] = 0.03f * (float)(((ghop + (long long)i) % 59) - 29);
    /* No state registered in this test, so the four *_out pointers are all NULL -- nothing else
     * to write. */
    return 0;
}

/* ---- driver: independent re-implementation of the board sample's loop, including its own
 * alignment-edge/reprime bookkeeping (had_hop/prev_solid/reprime_remaining), so this test can
 * exercise the SAME contract ulcnet_stream.c implements, hop-identical, from a clean reference. */
typedef struct BoardDriver {
    UlcnetGrid grid;
    FftHandle *fft;
    AcStft *stft_err; void *stft_err_pool;
    AcStft *stft_far; void *stft_far_pool;
    AcIstft *istft; void *istft_pool;
    UlcnetPrepost *prepost; void *prepost_pool;
    float *err_f, *far_f, *hop_out, *err_re, *err_im, *far_re, *far_im, *enh_re, *enh_im;
    int had_hop;
    int prev_solid;
    unsigned reprime_remaining;
} BoardDriver;

static int board_init(BoardDriver *d, const UlcnetGrid *grid) {
    AcStftConfig scfg;
    UlcnetPrepostConfig pcfg;
    UlcnetPrepostMemReq preq;
    size_t n_bins = (size_t)grid->n_bins, hop = (size_t)grid->hop;
    size_t seb, sfb, ib;

    memset(d, 0, sizeof(*d));
    d->grid = *grid;
    d->fft = fft_create(grid->n_fft);
    if (!d->fft) return -1;

    ac_stft_config_default(&scfg, grid->n_fft);
    scfg.ola = AC_OLA_ACCUMULATE;   /* must track ulcnet_stream.c's OLA mode */
    scfg.fft = d->fft;

    seb = ac_stft_get_mem_size(&scfg);
    if (seb == 0 || posix_memalign(&d->stft_err_pool, 16, seb) != 0) return -1;
    d->stft_err = ac_stft_init(d->stft_err_pool, seb, &scfg);
    if (!d->stft_err) return -1;

    sfb = ac_stft_get_mem_size(&scfg);
    if (sfb == 0 || posix_memalign(&d->stft_far_pool, 16, sfb) != 0) return -1;
    d->stft_far = ac_stft_init(d->stft_far_pool, sfb, &scfg);
    if (!d->stft_far) return -1;

    ib = ac_istft_get_mem_size(&scfg);
    if (ib == 0 || posix_memalign(&d->istft_pool, 16, ib) != 0) return -1;
    d->istft = ac_istft_init(d->istft_pool, ib, &scfg);
    if (!d->istft) return -1;

    if (ulcnet_prepost_config_defaults(&pcfg, grid) != 0) return -1;
    if (ulcnet_prepost_get_mem_size(&pcfg, &preq) != 0) return -1;
    if (posix_memalign(&d->prepost_pool, (size_t)preq.alignment, (size_t)preq.bytes) != 0) return -1;
    d->prepost = ulcnet_prepost_init(d->prepost_pool, (size_t)preq.bytes, &pcfg);
    if (!d->prepost) return -1;

    d->err_f = malloc(hop * sizeof(float));
    d->far_f = malloc(hop * sizeof(float));
    d->hop_out = malloc(hop * sizeof(float));
    d->err_re = malloc(n_bins * sizeof(float));
    d->err_im = malloc(n_bins * sizeof(float));
    d->far_re = malloc(n_bins * sizeof(float));
    d->far_im = malloc(n_bins * sizeof(float));
    d->enh_re = malloc(n_bins * sizeof(float));
    d->enh_im = malloc(n_bins * sizeof(float));
    if (!d->err_f || !d->far_f || !d->hop_out || !d->err_re || !d->err_im ||
        !d->far_re || !d->far_im || !d->enh_re || !d->enh_im) return -1;

    d->had_hop = 0;
    d->prev_solid = 0;
    d->reprime_remaining = 0;
    return 0;
}

/* The 4-function reset stream_reset() performs (alg_ai3_alignulcnet.c:1304), in its exact order. */
static void board_reset_only(BoardDriver *d) {
    ulcnet_prepost_reset(d->prepost);
    ac_stft_reset(d->stft_err);
    ac_stft_reset(d->stft_far);
    ac_istft_reset(d->istft);
}

/* Explicit, caller-driven reset (NOT from an alignment edge): the board sample has no equivalent
 * call site, so this is built directly from ulcnet_stream.h's documented contract for
 * ulcnet_stream_reset() -- also clears had_hop/prev_solid/reprime_remaining, so the next hop is
 * exactly like the first hop after init. */
static void board_explicit_reset(BoardDriver *d) {
    board_reset_only(d);
    d->reprime_remaining = 0;
    d->had_hop = 0;
    d->prev_solid = 0;
}

static int board_always_bypass_pre(BoardDriver *d, const int16_t *err_in, const int16_t *far_in) {
    int hop = d->grid.hop, k;
    for (k = 0; k < hop; ++k) {
        d->err_f[k] = (float)err_in[k] / 32768.0f;
        d->far_f[k] = (float)far_in[k] / 32768.0f;
    }
    ac_stft_push_planar(d->stft_err, d->err_f, d->err_re, d->err_im);
    ac_stft_push_planar(d->stft_far, d->far_f, d->far_re, d->far_im);
    if (ulcnet_prepost_pre_process_freq(d->prepost, d->err_re, d->err_im, d->far_re, d->far_im) != 1)
        return -1;
    (void)ulcnet_prepost_frame_bypass(d->prepost);
    return 0;
}

/* Stage 3 (alignment edge) + Stage 4 + the reprime-or-frame_open half of Stage 5, taking
 * err_in/far_in already as full-scale float, exactly like the real board sample's own Stage 4
 * (alg_ai3_alignulcnet.c:1582: AecResContext.formed_hop and its float ref_f straight into
 * ac_stft_push_planar, no int16 step at all). Returns 1 (a real accelerator job is due, in/out
 * filled), 0 (reprime: already bypassed, in/out untouched/meaningless), -1 (contract error). */
static int board_pre_f32(BoardDriver *d, const float *err_in, const float *far_in,
                         UlcnetStreamDelay delay, UlcnetModelIoInputs *in, UlcnetModelIoOutputs *out) {
    int edge = delay.changed || (d->had_hop && !d->prev_solid && delay.solid);
    if (edge) {
        board_reset_only(d);
        d->reprime_remaining = ULCNET_STREAM_REPRIME_FRAMES;
    }
    d->had_hop = 1;
    d->prev_solid = delay.solid;

    ac_stft_push_planar(d->stft_err, err_in, d->err_re, d->err_im);
    ac_stft_push_planar(d->stft_far, far_in, d->far_re, d->far_im);
    if (ulcnet_prepost_pre_process_freq(d->prepost, d->err_re, d->err_im, d->far_re, d->far_im) != 1)
        return -1;

    if (d->reprime_remaining > 0) {
        d->reprime_remaining--;
        (void)ulcnet_prepost_frame_bypass(d->prepost);
        return 0;
    }
    if (ulcnet_prepost_frame_open(d->prepost, in, out) != 0) return -1;
    return 1;
}

/* int16 convenience wrapper around board_pre_f32, mirroring ulcnet_stream.c's own
 * int16-wrapper-around-the-f32-entry-point shape: converts into the per-hop float scratch (never
 * read by anyone else), then runs exactly board_pre_f32's path -- the int16 -> /32768.0f
 * conversion this harness's every OTHER row relies on, kept in exactly one place instead of
 * duplicated alongside the edge/reset/STFT/reprime logic a second time. */
static int board_pre(BoardDriver *d, const int16_t *err_in, const int16_t *far_in,
                     UlcnetStreamDelay delay, UlcnetModelIoInputs *in, UlcnetModelIoOutputs *out) {
    int hop = d->grid.hop, k;
    for (k = 0; k < hop; ++k) {
        d->err_f[k] = (float)err_in[k] / 32768.0f;
        d->far_f[k] = (float)far_in[k] / 32768.0f;
    }
    return board_pre_f32(d, d->err_f, d->far_f, delay, in, out);
}

/* Stage 5's close + post_process_freq + synthesis. `pre_rc` selects whether to dispatch
 * frame_finish/frame_bypass (1: a real frame was opened) or skip it entirely (0: already bypassed
 * inside board_pre -- ai_rc is ignored). */
static int board_post(BoardDriver *d, int pre_rc, int ai_rc, int16_t *pcm_out, int *written_out) {
    int hop = d->grid.hop, k, prc, written = 0;
    if (pre_rc == 1) {
        int rc = (ai_rc == 0) ? ulcnet_prepost_frame_finish(d->prepost) : -1;
        if (rc != 0) ulcnet_prepost_frame_bypass(d->prepost);
    }
    prc = ulcnet_prepost_post_process_freq(d->prepost, d->enh_re, d->enh_im);
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

static uint32_t board_generation(const BoardDriver *d) {
    return ulcnet_prepost_state_generation(d->prepost);
}

static int cmp_floats_exact(const float *a, const float *b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

static UlcnetGrid make_grid(void) {
    UlcnetGridSpec spec;
    UlcnetGrid grid;
    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = ULCNET_SR;
    spec.n_fft = ULCNET_FFT;
    spec.delay_depth = ULCNET_D;
    if (ulcnet_grid_make(&grid, &spec) != ULCNET_GRID_OK) {
        fprintf(stderr, "ulcnet_grid_make failed\n");
        exit(1);
    }
    return grid;
}

typedef struct Stats {
    long long hops, mismatches, feat_mismatches;
} Stats;

/* ---- main matrix: steady state + reprime row + solid-transition row + fail-open row, one
 * continuous run so generation/history carries correctly between rows, exactly as a real caller
 * would see it. ---- */
static int run_matrix(int inject_shift) {
    UlcnetGrid grid = make_grid();
    BoardDriver A, C;   /* C: always-bypass reference, same input sequence from hop 0, for the
                         * fail-open row -- the OLA's overlap memory is exactly one hop deep
                         * (50% overlap), so comparing A against C only becomes meaningful once
                         * at least two CONSECUTIVE hops are bypassed on A (see the fail-open row
                         * below for why one forced-fail hop alone is not a clean comparison). */
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    UlcnetStream *B;
    void *pool_b;
    int hop = grid.hop;
    long long ghop = 0;
    Stats st;
    int16_t *err_in, *far_in, *pcm_a, *pcm_b, *pcm_c, *prev_pcm_b;
    int fail_violations = 0;

    memset(&st, 0, sizeof(st));
    if (board_init(&A, &grid) != 0) { fprintf(stderr, "board_init A failed\n"); return -1; }
    if (board_init(&C, &grid) != 0) { fprintf(stderr, "board_init C failed\n"); return -1; }

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    if (ulcnet_stream_get_mem_size(&cfg, &req) != 0) { fprintf(stderr, "get_mem_size failed\n"); return -1; }
    if (posix_memalign(&pool_b, (size_t)req.alignment, (size_t)req.bytes) != 0) return -1;
    B = ulcnet_stream_init_ex(&cfg, pool_b, (size_t)req.bytes, &req);
    if (!B) { fprintf(stderr, "ulcnet_stream_init_ex failed\n"); return -1; }

    err_in = calloc((size_t)hop, sizeof(int16_t));
    far_in = calloc((size_t)hop, sizeof(int16_t));
    pcm_a = calloc((size_t)hop, sizeof(int16_t));
    pcm_b = calloc((size_t)hop, sizeof(int16_t));
    pcm_c = calloc((size_t)hop, sizeof(int16_t));
    prev_pcm_b = calloc((size_t)hop, sizeof(int16_t));

    /* Hop plan (ghop -> event):
     *   0..4    steady (changed=0, solid=1)                       -- baseline
     *   5       reprime row: changed=1                            -- Row 2
     *   6..8    steady (post-reprime)
     *   9..12   solid=0 (changed=0)                                -- quiet period
     *   13      solid flips to 1 (changed=0)                       -- Row 3 (edge via solid, not changed)
     *   14..16  steady (post-reprime)
     *   17,18   fail-open row: ai_rc forced -1 for TWO consecutive hops -- Row 4
     *   19      steady again (sanity: generation did not move on 17/18)
     */
    const long long total_hops = 20;
    for (; ghop < total_hops; ++ghop) {
        UlcnetHopIO io_b;
        UlcnetModelIoInputs in_a;
        UlcnetModelIoOutputs out_a;
        UlcnetStreamDelay delay;
        int pre_rc_a, pre_rc_b, post_rc_a, post_rc_b, ai_rc;
        int written_a = -2, written_b = -2;
        uint32_t gen_before_a, gen_before_b, gen_after_a, gen_after_b;
        int k, force_fail;
        int expect_reprime_hop = (ghop == 5 || ghop == 13);
        int in_fail_window = (ghop == 17 || ghop == 18);

        memset(&delay, 0, sizeof(delay));
        if (ghop == 5) { delay.changed = 1; delay.solid = 1; }
        else if (ghop >= 9 && ghop <= 12) { delay.changed = 0; delay.solid = 0; }
        else if (ghop == 13) { delay.changed = 0; delay.solid = 1; }   /* the 0->1 transition */
        else { delay.changed = 0; delay.solid = 1; }

        force_fail = in_fail_window;

        for (k = 0; k < hop; ++k) {
            err_in[k] = (int16_t)(((long long)k * 7 + ghop * 5) % 101 - 50);
            far_in[k] = (int16_t)(((long long)k * 13 + ghop * 3) % 97 - 48);
        }

        gen_before_a = board_generation(&A);
        gen_before_b = ulcnet_stream_state_generation(B);
        if (gen_before_a != gen_before_b) {
            fprintf(stderr, "hop %lld: generation mismatch BEFORE this hop's own work, A=%u B=%u\n",
                   ghop, gen_before_a, gen_before_b);
            st.mismatches++;
        }

        pre_rc_a = board_pre(&A, err_in, far_in, delay, &in_a, &out_a);
        pre_rc_b = ulcnet_stream_pre_process(B, err_in, far_in, delay, &io_b);
        if (pre_rc_a != pre_rc_b) {
            fprintf(stderr, "hop %lld: pre_rc mismatch A=%d B=%d\n", ghop, pre_rc_a, pre_rc_b);
            st.mismatches++;
        }
        if (expect_reprime_hop && pre_rc_a != 0) {
            fprintf(stderr, "hop %lld: expected pre_rc==0 (reprime) on A, got %d\n", ghop, pre_rc_a);
            st.mismatches++;
        }
        if (!expect_reprime_hop && pre_rc_a == 0) {
            fprintf(stderr, "hop %lld: unexpected pre_rc==0 (reprime) on A\n", ghop);
            st.mismatches++;
        }

        /* Feature-array memcmp gated on BOTH sides returning 1 -- on a reprime hop io_b's inputs
         * are zeroed (ulcnet_stream.h's own documented contract) and A never called frame_open
         * either, so comparing them would be meaningless, not just noisy. */
        if (pre_rc_a == 1 && pre_rc_b == 1) {
            if (in_a.spectrum_bins_elements != io_b.inputs.spectrum_bins_elements ||
                !cmp_floats_exact(in_a.error_mag, io_b.inputs.error_mag, in_a.spectrum_bins_elements) ||
                !cmp_floats_exact(in_a.far_mag, io_b.inputs.far_mag, in_a.spectrum_bins_elements) ||
                !cmp_floats_exact(in_a.error_cos, io_b.inputs.error_cos, in_a.spectrum_bins_elements) ||
                !cmp_floats_exact(in_a.error_sin, io_b.inputs.error_sin, in_a.spectrum_bins_elements)) {
                fprintf(stderr, "hop %lld: feature mismatch A vs B\n", ghop);
                st.feat_mismatches++;
            }
        }

        ai_rc = (pre_rc_a == 1) ? shared_fake(ghop, force_fail, &out_a) : 0;
        if (pre_rc_b == 1) {
            int ai_rc_b = shared_fake(ghop, force_fail, &io_b.outputs);
            if (pre_rc_a == 1 && ai_rc_b != ai_rc) {
                fprintf(stderr, "hop %lld: fake rc mismatch A=%d B=%d\n", ghop, ai_rc, ai_rc_b);
                st.mismatches++;
            }
        }

        post_rc_a = board_post(&A, pre_rc_a, ai_rc, pcm_a, &written_a);
        post_rc_b = ulcnet_stream_post_process(B, ai_rc, pcm_b);
        written_b = (post_rc_b == 1) ? hop : (post_rc_b == 0 ? 0 : -1);

        if (post_rc_a != post_rc_b) {
            fprintf(stderr, "hop %lld: post_rc mismatch A=%d B=%d\n", ghop, post_rc_a, post_rc_b);
            st.mismatches++;
        }
        if (written_a != written_b) {
            fprintf(stderr, "hop %lld: written mismatch A=%d B=%d\n", ghop, written_a, written_b);
            st.mismatches++;
        }
        /* AC_OLA_ACCUMULATE has no first-push-after-reset suppression: a reprime hop's post_process
         * returns 1 on both sides, like every other hop, so post_rc is not compared against
         * expect_reprime_hop here. */

        {
            const int16_t *b_for_compare = inject_shift ? prev_pcm_b : pcm_b;
            if (memcmp(pcm_a, b_for_compare, (size_t)hop * sizeof(int16_t)) != 0) {
                fprintf(stderr, "hop %lld: PCM mismatch (inject_shift=%d)\n", ghop, inject_shift);
                st.mismatches++;
            }
        }
        memcpy(prev_pcm_b, pcm_b, (size_t)hop * sizeof(int16_t));

        gen_after_a = board_generation(&A);
        gen_after_b = ulcnet_stream_state_generation(B);
        if (gen_after_a != gen_after_b) {
            fprintf(stderr, "hop %lld: generation mismatch A=%u B=%u\n", ghop, gen_after_a, gen_after_b);
            st.mismatches++;
        }
        if (expect_reprime_hop) {
            if (gen_after_a != gen_before_a + 1) {
                fprintf(stderr, "hop %lld: expected generation to advance by exactly 1 on the reset "
                                "hop, A: before=%u after=%u\n", ghop, gen_before_a, gen_after_a);
                st.mismatches++;
            }
        } else if (ghop == 6 || ghop == 14) {
            /* The hop right after a reprime hop must NOT advance the generation again. */
            if (gen_after_a != gen_before_a) {
                fprintf(stderr, "hop %lld: generation advanced again right after a reprime hop "
                                "(before=%u after=%u)\n", ghop, gen_before_a, gen_after_a);
                st.mismatches++;
            }
        }

        /* Always advance C (the always-bypass reference) with the identical input, so its OLA
         * history is the pure-passthrough-from-hop-0 reference at every ghop. */
        if (board_always_bypass_pre(&C, err_in, far_in) != 0) {
            fprintf(stderr, "hop %lld: board_always_bypass_pre (C) failed\n", ghop);
            st.mismatches++;
        } else {
            int wr_c = 0;
            (void)board_post(&C, 0, 0, pcm_c, &wr_c);
        }

        /* Fail-open row (Row 4): the OLA's overlap memory is exactly one hop deep (50% overlap,
         * sqrt-Hann), so hop 17 alone mixes a real/model-shaped hop-16 tail with hop-17's bypass
         * contribution -- not a clean "is bypass correct" signal. By hop 18 (the SECOND
         * consecutive bypassed hop), both the current and the carried-over tail come from
         * bypassed hops, so A's (and B's) PCM at hop 18 must exactly equal C's (always-bypass
         * from hop 0) PCM at hop 18 -- a confound-free proof that frame_bypass() really does pass
         * the error spectrum through unchanged, not some other transformation, and specifically
         * not silence. */
        if (ghop == 18) {
            int a_matches_c = (memcmp(pcm_a, pcm_c, (size_t)hop * sizeof(int16_t)) == 0);
            int b_matches_c = (memcmp(pcm_b, pcm_c, (size_t)hop * sizeof(int16_t)) == 0);
            int is_silence = 1;
            for (k = 0; k < hop; ++k) if (pcm_c[k] != 0) { is_silence = 0; break; }
            if (!a_matches_c) {
                fprintf(stderr, "hop %lld: FAIL-OPEN check: A's PCM does not match the always-bypass "
                                "reference C -- bypass is not passing the error spectrum through\n", ghop);
                fail_violations++;
            }
            if (!b_matches_c) {
                fprintf(stderr, "hop %lld: FAIL-OPEN check: B's (ulcnet_stream) PCM does not match "
                                "the always-bypass reference C\n", ghop);
                fail_violations++;
            }
            if (is_silence) {
                fprintf(stderr, "hop %lld: FAIL-OPEN check: the always-bypass reference C is itself "
                                "silence -- the input signal or test setup makes this check vacuous\n", ghop);
                fail_violations++;
            }
            if (a_matches_c && b_matches_c && !is_silence) {
                printf("fail-open row: hop %lld -- A and B both match the always-bypass reference "
                      "C exactly, and it is NOT silence: bypass correctly passes the error "
                      "spectrum through\n", ghop);
            }
        }
        if (in_fail_window && (gen_after_a != gen_before_a)) {
            fprintf(stderr, "hop %lld: generation advanced on a plain (ai_rc != 0) bypass hop "
                            "(before=%u after=%u) -- a pure bypass must not touch the generation\n",
                           ghop, gen_before_a, gen_after_a);
            st.mismatches++;
        }

        st.hops++;
    }

    printf("hops=%lld mismatches=%lld feat_mismatches=%lld fail_open_violations=%d %s\n",
          st.hops, st.mismatches, st.feat_mismatches, fail_violations,
          (st.mismatches == 0 && st.feat_mismatches == 0 && fail_violations == 0) ? "PASS" : "FAIL");

    free(err_in); free(far_in); free(pcm_a); free(pcm_b); free(pcm_c); free(prev_pcm_b);
    return (st.mismatches == 0 && st.feat_mismatches == 0 && fail_violations == 0) ? 0 : 1;
}

/* ---- Row 5: explicit ulcnet_stream_reset() must not leave had_hop/prev_solid stale, or the very
 * next hop with solid=1 would spuriously re-trigger a second reset+reprime. This is an API-level
 * check (the board sample has no call site for an "explicit reset separate from an alignment
 * edge"); board_explicit_reset() is this test's own reference, built from ulcnet_stream.h's own
 * documented ulcnet_stream_reset() contract, against the independent board re-implementation. ---- */
static int run_explicit_reset_row(void) {
    UlcnetGrid grid = make_grid();
    BoardDriver A;
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    UlcnetStream *B;
    void *pool_b;
    int hop = grid.hop;
    int16_t *err_in, *far_in, *pcm_a, *pcm_b;
    int k, violations = 0;
    UlcnetStreamDelay delay;
    UlcnetModelIoInputs in_a;
    UlcnetModelIoOutputs out_a;
    UlcnetHopIO io_b;
    int pre_rc_a, pre_rc_b;

    if (board_init(&A, &grid) != 0) return -1;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    if (ulcnet_stream_get_mem_size(&cfg, &req) != 0) return -1;
    if (posix_memalign(&pool_b, (size_t)req.alignment, (size_t)req.bytes) != 0) return -1;
    B = ulcnet_stream_init_ex(&cfg, pool_b, (size_t)req.bytes, &req);
    if (!B) return -1;

    err_in = calloc((size_t)hop, sizeof(int16_t));
    far_in = calloc((size_t)hop, sizeof(int16_t));
    pcm_a = calloc((size_t)hop, sizeof(int16_t));
    pcm_b = calloc((size_t)hop, sizeof(int16_t));
    for (k = 0; k < hop; ++k) { err_in[k] = (int16_t)(k * 3 - 20); far_in[k] = (int16_t)(k * 5 - 30); }

    /* A few steady hops with solid=0 recorded as the last state. */
    memset(&delay, 0, sizeof(delay));
    delay.changed = 0; delay.solid = 0;
    for (k = 0; k < 3; ++k) {
        pre_rc_a = board_pre(&A, err_in, far_in, delay, &in_a, &out_a);
        pre_rc_b = ulcnet_stream_pre_process(B, err_in, far_in, delay, &io_b);
        (void)board_post(&A, pre_rc_a, 0, pcm_a, NULL);
        (void)ulcnet_stream_post_process(B, 0, pcm_b);
    }

    /* Explicit reset, NOT via an alignment edge, on both sides. */
    board_explicit_reset(&A);
    ulcnet_stream_reset(B);

    /* Next hop: solid flips to 1, changed stays 0. If had_hop/prev_solid were left stale, this
     * would wrongly look like a 0->1 transition and re-trigger a second reset+reprime
     * (pre_rc == 0). Correct behavior: this is "the first hop after a reset," so no edge fires --
     * pre_rc == 1. */
    delay.changed = 0; delay.solid = 1;
    pre_rc_a = board_pre(&A, err_in, far_in, delay, &in_a, &out_a);
    pre_rc_b = ulcnet_stream_pre_process(B, err_in, far_in, delay, &io_b);

    if (pre_rc_a != 1) {
        fprintf(stderr, "explicit-reset row: A spuriously re-triggered a reprime after an explicit "
                        "reset (pre_rc=%d, expected 1)\n", pre_rc_a);
        violations++;
    }
    if (pre_rc_b != 1) {
        fprintf(stderr, "explicit-reset row: B (ulcnet_stream) spuriously re-triggered a reprime "
                        "after an explicit ulcnet_stream_reset() (pre_rc=%d, expected 1) -- the "
                        "had_hop/prev_solid fix did not hold\n", pre_rc_b);
        violations++;
    }
    if (pre_rc_a != pre_rc_b) {
        fprintf(stderr, "explicit-reset row: A/B disagree (A=%d B=%d)\n", pre_rc_a, pre_rc_b);
        violations++;
    }

    (void)board_post(&A, pre_rc_a, 0, pcm_a, NULL);
    (void)ulcnet_stream_post_process(B, 0, pcm_b);

    printf("explicit-reset row: %s (pre_rc after explicit reset + solid=1 next hop: A=%d B=%d, "
          "expected 1/1 -- no spurious re-trigger)\n",
          violations == 0 ? "PASS" : "FAIL", pre_rc_a, pre_rc_b);

    free(err_in); free(far_in); free(pcm_a); free(pcm_b);
    return violations == 0 ? 0 : 1;
}

/* ---- Row 6 (float entry point): drives ulcnet_stream_pre_process_f32() with REAL float input
 * (deliberately not exactly representable in int16, via a fractional-sine signal) against
 * board_pre_f32() -- a board reference that, like the real board sample, never quantizes to
 * int16 either. Then, separately, quantizes that SAME float signal to int16 first (as
 * ulcnet_stream_pre_process(), the existing int16 entry, requires) and runs it through the
 * already-verified int16 path, which produces a DIFFERENT PCM result -- the quantization gap the
 * float entry point exists to avoid. Without this row, every other row in this file drives both
 * compared sides from int16 and cannot see this gap on either side. ---- */
static int run_float_entry_row(void) {
    UlcnetGrid grid = make_grid();
    BoardDriver D;
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    UlcnetStream *E;
    void *pool_e;
    int hop = grid.hop;
    float *err_f, *far_f;
    int16_t *err_q, *far_q, *pcm_d, *pcm_e, *pcm_q;
    int h, k, violations = 0;
    int any_quantized_differs = 0;
    const int total_hops = 6;
    UlcnetStream *Q;
    void *pool_q;

    if (board_init(&D, &grid) != 0) return -1;
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    if (ulcnet_stream_get_mem_size(&cfg, &req) != 0) return -1;
    if (posix_memalign(&pool_e, (size_t)req.alignment, (size_t)req.bytes) != 0) return -1;
    E = ulcnet_stream_init_ex(&cfg, pool_e, (size_t)req.bytes, &req);
    if (!E) return -1;
    if (posix_memalign(&pool_q, (size_t)req.alignment, (size_t)req.bytes) != 0) return -1;
    Q = ulcnet_stream_init_ex(&cfg, pool_q, (size_t)req.bytes, &req);
    if (!Q) return -1;

    err_f = malloc((size_t)hop * sizeof(float));
    far_f = malloc((size_t)hop * sizeof(float));
    err_q = malloc((size_t)hop * sizeof(int16_t));
    far_q = malloc((size_t)hop * sizeof(int16_t));
    pcm_d = malloc((size_t)hop * sizeof(int16_t));
    pcm_e = malloc((size_t)hop * sizeof(int16_t));
    pcm_q = malloc((size_t)hop * sizeof(int16_t));

    for (h = 0; h < total_hops; ++h) {
        UlcnetStreamDelay delay;
        UlcnetModelIoInputs in_d;
        UlcnetModelIoOutputs out_d;
        UlcnetHopIO io_e;
        int pre_rc_d, pre_rc_e, post_rc_d, post_rc_e, ai_rc;

        memset(&delay, 0, sizeof(delay));
        delay.changed = (h == 2) ? 1 : 0;
        delay.solid = 1;

        /* A fractional-sine signal: scaled so int16 quantization has sub-LSB rounding error on
         * nearly every sample (not an integer multiple of 1/32768), unlike this file's other rows'
         * int16-sourced-then-converted inputs, which are exactly representable by construction. */
        for (k = 0; k < hop; ++k) {
            err_f[k] = 0.1f * sinf(0.013f * (float)(k + h * hop)) + 0.0001234f;
            far_f[k] = 0.08f * sinf(0.021f * (float)(k + h * hop) + 1.0f) - 0.0000777f;
            err_q[k] = (int16_t)(err_f[k] * 32768.0f);
            far_q[k] = (int16_t)(far_f[k] * 32768.0f);
        }

        pre_rc_d = board_pre_f32(&D, err_f, far_f, delay, &in_d, &out_d);
        pre_rc_e = ulcnet_stream_pre_process_f32(E, err_f, far_f, delay, &io_e);
        if (pre_rc_d != pre_rc_e) {
            fprintf(stderr, "float-entry row: hop %d: pre_rc mismatch D=%d E=%d\n", h, pre_rc_d, pre_rc_e);
            violations++;
        }
        if (pre_rc_d == 1 && pre_rc_e == 1) {
            if (in_d.spectrum_bins_elements != io_e.inputs.spectrum_bins_elements ||
                !cmp_floats_exact(in_d.error_mag, io_e.inputs.error_mag, in_d.spectrum_bins_elements) ||
                !cmp_floats_exact(in_d.far_mag, io_e.inputs.far_mag, in_d.spectrum_bins_elements) ||
                !cmp_floats_exact(in_d.error_cos, io_e.inputs.error_cos, in_d.spectrum_bins_elements) ||
                !cmp_floats_exact(in_d.error_sin, io_e.inputs.error_sin, in_d.spectrum_bins_elements)) {
                fprintf(stderr, "float-entry row: hop %d: feature mismatch D vs E\n", h);
                violations++;
            }
        }

        ai_rc = (pre_rc_d == 1) ? shared_fake(h, 0, &out_d) : 0;
        if (pre_rc_e == 1) (void)shared_fake(h, 0, &io_e.outputs);

        post_rc_d = board_post(&D, pre_rc_d, ai_rc, pcm_d, NULL);
        post_rc_e = ulcnet_stream_post_process(E, ai_rc, pcm_e);
        if (post_rc_d != post_rc_e || memcmp(pcm_d, pcm_e, (size_t)hop * sizeof(int16_t)) != 0) {
            fprintf(stderr, "float-entry row: hop %d: PCM/rc mismatch D vs E (rc %d/%d)\n",
                   h, post_rc_d, post_rc_e);
            violations++;
        }

        /* Teeth: the SAME underlying signal, quantized to int16 first (as ulcnet_stream_pre_process()
         * requires), run through the already-verified int16 path on a throwaway stream Q, must NOT
         * produce the same PCM as the float path above once both are past the reprime/startup
         * window -- otherwise this row would not actually be exercising the quantization gap. */
        {
            UlcnetHopIO io_q;
            int pre_rc_q, post_rc_q, ai_rc_q;
            pre_rc_q = ulcnet_stream_pre_process(Q, err_q, far_q, delay, &io_q);
            ai_rc_q = (pre_rc_q == 1) ? shared_fake(h, 0, &io_q.outputs) : 0;
            post_rc_q = ulcnet_stream_post_process(Q, ai_rc_q, pcm_q);
            if (h >= 3 && post_rc_q == 1 && post_rc_e == 1 &&
                memcmp(pcm_q, pcm_e, (size_t)hop * sizeof(int16_t)) != 0) {
                any_quantized_differs = 1;
            }
        }
    }

    if (!any_quantized_differs) {
        fprintf(stderr, "float-entry row: FAIL -- quantizing to int16 first made no difference to "
                        "the output; this row is not exercising the gap it claims to\n");
        violations++;
    } else {
        printf("float-entry row: PASS (f32 entry matches an unquantized board reference "
              "bit-exact; quantizing the same signal to int16 first, as the int16 entry point "
              "requires, measurably changes the output -- confirming the gap the f32 entry "
              "point avoids)\n");
    }

    free(err_f); free(far_f); free(err_q); free(far_q); free(pcm_d); free(pcm_e); free(pcm_q);
    free(pool_e); free(pool_q);
    return violations == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    int inject_shift = (argc > 1 && strcmp(argv[1], "--inject-shift") == 0);

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

    {
        int rc1 = run_matrix(0);
        int rc2 = run_explicit_reset_row();
        int rc3 = run_float_entry_row();
        return (rc1 == 0 && rc2 == 0 && rc3 == 0) ? 0 : 1;
    }
}
