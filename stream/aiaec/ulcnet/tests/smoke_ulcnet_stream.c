/* Smoke build only: does ulcnet_stream link and run several hops without crashing. Not a
 * correctness test -- it does not check PCM against a reference; that's the separate
 * verification harness. Asserts the documented return-value splits: post_process returns exactly
 * 1 at every hop (AC_OLA_ACCUMULATE has no first-push startup quirk); pre_process
 * returns 0 at the reprime hop triggered by delay.changed, 1 elsewhere. Also independently
 * confirms (against the bare UlcnetPrepost class, not through the
 * stream) what ulcnet_prepost_frame_bypass() returns when called right after
 * ulcnet_prepost_pre_process_freq() with no intervening frame_open() -- the exact sequence a
 * reprime hop uses -- since that return value is not otherwise observable through the stream's
 * own public API. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet_stream.h"

/* Writes a zero mask and zero to every registered state_out -- shape correctness only, no
 * numerical realism. This smoke test registers no state, so only output is touched. */
static int fake_accelerator(const UlcnetModelIoInputs *in, UlcnetModelIoOutputs *out) {
    size_t i;
    (void)in;
    for (i = 0; i < out->spectrum_ri_elements; ++i) out->output[i] = 0.0f;
    if (out->key_history_out)
        for (i = 0; i < out->key_history_elements; ++i) out->key_history_out[i] = 0.0f;
    if (out->value_history_out)
        for (i = 0; i < out->value_history_elements; ++i) out->value_history_out[i] = 0.0f;
    if (out->logit_history_out)
        for (i = 0; i < out->logit_history_elements; ++i) out->logit_history_out[i] = 0.0f;
    if (out->h_gru_out)
        for (i = 0; i < out->gru_hidden_elements; ++i) out->h_gru_out[i] = 0.0f;
    return 0;
}

/* Independent confirmation against the bare UlcnetPrepost class: pre_process_freq() then
 * frame_bypass() with no frame_open() in between -- exactly the sequence a reprime hop uses
 * inside ulcnet_stream_pre_process(). Prints and returns frame_bypass()'s actual return value. */
static int confirm_bare_frame_bypass_after_pre_process_freq(const UlcnetGrid *grid) {
    UlcnetPrepostConfig pcfg;
    UlcnetPrepostMemReq preq;
    UlcnetPrepost *p;
    void *pool;
    float re[4096], im[4096];   /* far larger than any n_bins this test uses */
    int rc_pre, rc_bypass;

    if (ulcnet_prepost_config_defaults(&pcfg, grid) != 0) {
        fprintf(stderr, "ulcnet_prepost_config_defaults failed\n");
        return -2;
    }
    if (ulcnet_prepost_get_mem_size(&pcfg, &preq) != 0) {
        fprintf(stderr, "ulcnet_prepost_get_mem_size failed\n");
        return -2;
    }
    pool = NULL;
    if (posix_memalign(&pool, (size_t)preq.alignment, (size_t)preq.bytes) != 0 || !pool) {
        fprintf(stderr, "bare prepost pool allocation failed\n");
        return -2;
    }
    p = ulcnet_prepost_init_ex(pool, (size_t)preq.bytes, &pcfg, &preq);
    if (!p) {
        fprintf(stderr, "ulcnet_prepost_init_ex failed\n");
        free(pool);
        return -2;
    }

    memset(re, 0, sizeof(re));
    memset(im, 0, sizeof(im));
    rc_pre = ulcnet_prepost_pre_process_freq(p, re, im, re, im);
    if (rc_pre != 1) {
        fprintf(stderr, "bare pre_process_freq rc=%d (expected 1)\n", rc_pre);
        free(pool);
        return -2;
    }
    rc_bypass = ulcnet_prepost_frame_bypass(p);
    printf("confirm: ulcnet_prepost_frame_bypass() right after pre_process_freq() (no prior "
          "frame_open()) returns %d\n", rc_bypass);

    free(pool);
    return rc_bypass;
}

int main(void) {
    UlcnetGridSpec spec;
    UlcnetGrid grid;
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    UlcnetStream *s;
    void *pool;
    int16_t *err_in, *far_in, *pcm_out;
    int hop, hops, i, k;
    int bare_bypass_rc;
    const int reprime_hop = 5;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000;
    spec.n_fft = 512;
    spec.delay_depth = 32;   /* matches the board sample's --delay-depth default */

    if (ulcnet_grid_make(&grid, &spec) != ULCNET_GRID_OK) {
        fprintf(stderr, "ulcnet_grid_make failed\n");
        return 1;
    }

    bare_bypass_rc = confirm_bare_frame_bypass_after_pre_process_freq(&grid);
    if (bare_bypass_rc != 0) {
        fprintf(stderr, "bare frame_bypass() check did not return 0 (got %d) -- the reprime path "
                        "inside ulcnet_stream_pre_process() relies on this returning 0\n",
                bare_bypass_rc);
        return 1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;   /* all four state pointers/counts zero: no state registered */

    if (ulcnet_stream_get_mem_size(&cfg, &req) != 0) {
        fprintf(stderr, "ulcnet_stream_get_mem_size failed\n");
        return 1;
    }
    printf("pool: hop=%d n_bins=%d bytes=%llu alignment=%u layout_version=%u\n",
          grid.hop, grid.n_bins, (unsigned long long)req.bytes, req.alignment,
          req.layout_version);

    pool = NULL;
    if (posix_memalign(&pool, (size_t)req.alignment, (size_t)req.bytes) != 0 || !pool) {
        fprintf(stderr, "pool allocation failed (%llu bytes)\n", (unsigned long long)req.bytes);
        return 1;
    }

    /* Prove the stale-pool gate can actually refuse (project rule: a gate must be shown capable
     * of failing, not just capable of passing) before trusting the real init_ex() call below. A
     * deliberately corrupted build_flags_hash must make init_ex() return NULL. */
    {
        UlcnetStreamMemReq bad_req = req;
        UlcnetStream *refused;
        bad_req.build_flags_hash ^= 1u;
        refused = ulcnet_stream_init_ex(&cfg, pool, (size_t)req.bytes, &bad_req);
        if (refused) {
            fprintf(stderr, "stale-pool gate FAILED to refuse a corrupted build_flags_hash\n");
            return 1;
        }
        printf("stale-pool gate: correctly refused a corrupted build_flags_hash\n");
    }

    s = ulcnet_stream_init_ex(&cfg, pool, (size_t)req.bytes, &req);
    if (!s) {
        fprintf(stderr, "ulcnet_stream_init_ex failed\n");
        return 1;
    }

    hop = ulcnet_stream_hop_size(s);
    if (hop <= 0) {
        fprintf(stderr, "ulcnet_stream_hop_size returned %d\n", hop);
        return 1;
    }
    err_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    far_in = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    pcm_out = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    if (!err_in || !far_in || !pcm_out) {
        fprintf(stderr, "pcm buffer allocation failed\n");
        return 1;
    }

    hops = reprime_hop + 6;
    for (i = 0; i < hops; ++i) {
        UlcnetHopIO io;
        UlcnetStreamDelay delay;
        int pre_rc, post_rc, ai_rc = 0;

        for (k = 0; k < hop; ++k) {
            err_in[k] = (int16_t)((k * 7 + i * 3) % 101 - 50);
            far_in[k] = (int16_t)((k * 11 + i * 5) % 97 - 48);
        }

        delay.changed = (i == reprime_hop) ? 1 : 0;
        delay.solid = 1;

        pre_rc = ulcnet_stream_pre_process(s, err_in, far_in, delay, &io);
        if (pre_rc < 0) {
            fprintf(stderr, "pre_process rc=%d at hop %d (expected 0 or 1)\n", pre_rc, i);
            return 1;
        }
        if (i == reprime_hop && pre_rc != 0) {
            fprintf(stderr, "pre_process rc=%d at reprime hop %d (expected exactly 0)\n",
                   pre_rc, i);
            return 1;
        }
        if (i != reprime_hop && pre_rc != 1) {
            fprintf(stderr, "pre_process rc=%d at hop %d (expected exactly 1)\n", pre_rc, i);
            return 1;
        }

        if (pre_rc == 1) ai_rc = fake_accelerator(&io.inputs, &io.outputs);

        post_rc = ulcnet_stream_post_process(s, ai_rc, pcm_out);
        if (post_rc != 1) {
            fprintf(stderr, "post_process rc=%d at hop %d (expected exactly 1: AC_OLA_ACCUMULATE "
                            "has no first-push-after-reset suppression)\n", post_rc, i);
            return 1;
        }
    }

    ulcnet_stream_reset(s);
    printf("smoke_ulcnet_stream: %d hops, hop=%d, reprime at hop %d, generation after reset=%u "
          "-- PASS\n", hops, hop, reprime_hop, ulcnet_stream_state_generation(s));

    free(err_in);
    free(far_in);
    free(pcm_out);
    free(pool);
    return 0;
}
