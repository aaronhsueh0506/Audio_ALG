/* Smoke build only: does deepvqe_stream link and run several hops without crashing. Not a
 * correctness test -- it does not check PCM against a reference; that's the separate
 * verification harness. Also asserts the one documented return value: with AC_OLA_ACCUMULATE
 * (no first-push suppression) post_process returns 1 (a real, full-hop synthesis result) at every
 * hop from hop 0 onward, N == 1 here. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepvqe_stream.h"

/* Writes zeros/success to taps and every registered state_out -- shape correctness only, no
 * numerical realism. This smoke test registers no state, so only taps is touched. */
static int fake_accelerator(const DeepVqePrepostInputs *in, DeepVqePrepostOutputs *out) {
    size_t i;
    (void)in;
    for (i = 0; i < out->taps_elements; ++i) out->taps[i] = 0.0f;
    for (i = 0; i < DEEPVQE_STATE_COUNT; ++i) {
        size_t k;
        if (!out->state_out[i]) continue;
        for (k = 0; k < out->state_elements[i]; ++k) out->state_out[i][k] = 0.0f;
    }
    return 0;
}

/* Pairing probe at frames_per_call == 8: pre_process's first call to a fresh/reset stream
 * returns 0 (batch still gathering, no job this hop) -- `s->open` is NEVER set in that case, so
 * a check gated on `open` alone cannot catch a second pre_process before the matching
 * post_process here; only `pending` (set on every successful pre_process, job-ready or not) can.
 * Returns 0 on success, 1 on failure (same convention as main()'s own body). */
static int pairing_probe_n8(void) {
    DeepVqeGridSpec spec;
    DeepVqeGrid grid;
    DeepVqeStreamConfig cfg;
    DeepVqeStreamMemReq req;
    DeepVqeStream *s;
    void *pool = NULL;
    int16_t *mic_in, *far_in, *pcm_out;
    int hop, k, pre_rc, misuse_rc;
    DeepVqeHopIO io;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000;
    spec.n_fft = 256;
    spec.delay_depth = 63;
    spec.frames_per_call = 8;
    if (deepvqe_grid_make(&grid, &spec) != DEEPVQE_GRID_OK) {
        fprintf(stderr, "pairing probe (N=8): deepvqe_grid_make failed\n");
        return 1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    if (deepvqe_stream_get_mem_size(&cfg, &req) != 0) {
        fprintf(stderr, "pairing probe (N=8): deepvqe_stream_get_mem_size failed\n");
        return 1;
    }
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "pairing probe (N=8): pool allocation failed\n");
        return 1;
    }
    s = deepvqe_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) {
        fprintf(stderr, "pairing probe (N=8): deepvqe_stream_init_ex failed\n");
        free(pool);
        return 1;
    }

    hop = deepvqe_stream_hop_size(s);
    mic_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    far_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    pcm_out = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    if (!mic_in || !far_in || !pcm_out) {
        fprintf(stderr, "pairing probe (N=8): pcm buffer allocation failed\n");
        free(pool);
        return 1;
    }
    for (k = 0; k < hop; ++k) { mic_in[k] = (int16_t)k; far_in[k] = (int16_t)(k * 2); }

    /* Fresh/reset stream, first hop of an N=8 batch: must return 0 (gathering), not 1 -- this is
     * the case `open` never sets, so the misuse probe below is only meaningful because of it. */
    pre_rc = deepvqe_stream_pre_process(s, mic_in, far_in, &io);
    if (pre_rc != 0) {
        fprintf(stderr, "pairing probe (N=8): setup pre_process rc=%d, expected 0 (batch "
                        "gathering, hop 1 of 8)\n", pre_rc);
        free(mic_in); free(far_in); free(pcm_out); free(pool);
        return 1;
    }

    /* A second pre_process with no intervening post_process must be rejected even though no
     * model frame is open (rc == 0 above never sets `open`). */
    misuse_rc = deepvqe_stream_pre_process(s, mic_in, far_in, &io);
    if (misuse_rc != -1) {
        fprintf(stderr, "pairing probe (N=8): pre_process called twice in a row during the "
                        "gather phase returned %d, expected -1\n", misuse_rc);
        free(mic_in); free(far_in); free(pcm_out); free(pool);
        return 1;
    }

    /* Resynchronize with the documented contract (post_process is still due for the first call)
     * before tearing down, confirming the stream is left usable, not permanently wedged. */
    if (deepvqe_stream_post_process(s, 0, pcm_out) != 0) {
        fprintf(stderr, "pairing probe (N=8): cleanup post_process did not return the expected "
                        "pre-roll 0\n");
        free(mic_in); free(far_in); free(pcm_out); free(pool);
        return 1;
    }

    printf("smoke_deepvqe_stream: pairing probe (N=8, repeated pre during gather) -- PASS\n");
    free(mic_in); free(far_in); free(pcm_out); free(pool);
    return 0;
}

int main(void) {
    DeepVqeGridSpec spec;
    DeepVqeGrid grid;
    DeepVqeStreamConfig cfg;
    DeepVqeStreamMemReq req;
    DeepVqeStream *s;
    void *pool;
    int16_t *mic_in, *far_in, *pcm_out;
    int hop, hops, i, k;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000;
    spec.n_fft = 512;
    spec.delay_depth = 32;   /* MAX_DELAY_FRAMES in the board sample */
    spec.frames_per_call = 1;

    if (deepvqe_grid_make(&grid, &spec) != DEEPVQE_GRID_OK) {
        fprintf(stderr, "deepvqe_grid_make failed\n");
        return 1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;   /* cfg.state[]/state_elements[] all zero: no state registered */

    if (deepvqe_stream_get_mem_size(&cfg, &req) != 0) {
        fprintf(stderr, "deepvqe_stream_get_mem_size failed\n");
        return 1;
    }

    pool = NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "pool allocation failed (%llu bytes)\n", (unsigned long long)req.bytes);
        return 1;
    }

    s = deepvqe_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) {
        fprintf(stderr, "deepvqe_stream_init_ex failed\n");
        return 1;
    }

    hop = deepvqe_stream_hop_size(s);
    if (hop <= 0) {
        fprintf(stderr, "deepvqe_stream_hop_size returned %d\n", hop);
        return 1;
    }
    mic_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    far_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    pcm_out = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    if (!mic_in || !far_in || !pcm_out) {
        fprintf(stderr, "pcm buffer allocation failed\n");
        return 1;
    }

    hops = DEEPVQE_TIME_ORDER + 8;   /* past the CCM ring's own startup window */
    for (i = 0; i < hops; ++i) {
        DeepVqeHopIO io;
        int pre_rc, post_rc, ai_rc;

        for (k = 0; k < hop; ++k) {
            mic_in[k] = (int16_t)((k * 7 + i * 3) % 101 - 50);
            far_in[k] = (int16_t)((k * 11 + i * 5) % 97 - 48);
        }

        pre_rc = deepvqe_stream_pre_process(s, mic_in, far_in, &io);
        if (pre_rc != 1) {
            fprintf(stderr, "pre_process rc=%d at hop %d (expected 1)\n", pre_rc, i);
            return 1;
        }

        ai_rc = fake_accelerator(&io.inputs, &io.outputs);

        post_rc = deepvqe_stream_post_process(s, ai_rc, pcm_out);
        if (post_rc != 1) {
            fprintf(stderr, "post_process rc=%d at hop %d (expected exactly 1: AC_OLA_ACCUMULATE "
                            "has no first-push startup case, N == 1 here)\n", post_rc, i);
            return 1;
        }
    }

    deepvqe_stream_reset(s);

    /* Pairing contract: a pre_process/post_process call out of turn is a caller bug and must
     * return -1, never silently succeed. Drive one legitimate pair, then probe both misuse directions. */
    {
        DeepVqeHopIO io;
        int pre_rc, post_rc, misuse_rc;

        for (k = 0; k < hop; ++k) { mic_in[k] = (int16_t)k; far_in[k] = (int16_t)(k * 2); }
        pre_rc = deepvqe_stream_pre_process(s, mic_in, far_in, &io);
        if (pre_rc != 1) { fprintf(stderr, "pairing probe: setup pre_process rc=%d\n", pre_rc); return 1; }
        (void)fake_accelerator(&io.inputs, &io.outputs);
        post_rc = deepvqe_stream_post_process(s, 0, pcm_out);
        if (post_rc != 1) { fprintf(stderr, "pairing probe: setup post_process rc=%d\n", post_rc); return 1; }

        /* A second post_process with no intervening pre_process must be rejected, not re-deliver
         * the same frame again. */
        misuse_rc = deepvqe_stream_post_process(s, 0, pcm_out);
        if (misuse_rc != -1) {
            fprintf(stderr, "pairing probe: post_process called twice in a row returned %d, "
                            "expected -1 (it must not silently re-deliver)\n", misuse_rc);
            return 1;
        }

        /* Back to a clean, paired state before probing the other direction. */
        pre_rc = deepvqe_stream_pre_process(s, mic_in, far_in, &io);
        if (pre_rc != 1) { fprintf(stderr, "pairing probe: re-sync pre_process rc=%d\n", pre_rc); return 1; }
        (void)fake_accelerator(&io.inputs, &io.outputs);

        /* A second pre_process with no intervening post_process must be rejected too. */
        misuse_rc = deepvqe_stream_pre_process(s, mic_in, far_in, &io);
        if (misuse_rc != -1) {
            fprintf(stderr, "pairing probe: pre_process called twice in a row returned %d, "
                            "expected -1\n", misuse_rc);
            return 1;
        }
        post_rc = deepvqe_stream_post_process(s, 0, pcm_out);
        if (post_rc != 1) { fprintf(stderr, "pairing probe: cleanup post_process rc=%d\n", post_rc); return 1; }

        printf("smoke_deepvqe_stream: pairing probe (double post_process, double pre_process) -- PASS\n");
    }

    printf("smoke_deepvqe_stream: %d hops, hop=%d, generation after reset=%u -- PASS\n",
          hops, hop, deepvqe_stream_state_generation(s));

    free(mic_in);
    free(far_in);
    free(pcm_out);
    free(pool);

    if (pairing_probe_n8() != 0) return 1;

    return 0;
}
