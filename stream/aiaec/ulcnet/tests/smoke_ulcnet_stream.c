/* Smoke build only: does ulcnet_stream link and run several hops without crashing. Not a
 * correctness test -- it does not check PCM against a reference; that's the separate
 * verification harness. Asserts the documented return-value splits: post_process returns exactly
 * 1 at every hop (AC_OLA_ACCUMULATE has no first-push startup quirk); pre_process
 * returns 0 at the reprime hop triggered by delay.changed, 1 elsewhere. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet_stream.h"
#include "utility/mem_align.h"

/* Writes a zero mask and zero to every registered state_out -- shape correctness only, no
 * numerical realism. This smoke test registers no state, so only output is touched. */
static int fake_accelerator(UlcnetModelIoOutputs *out) {
    size_t i;
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

int main(void) {
    UlcnetGridSpec spec;
    UlcnetGrid grid;
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    UlcnetStream *s;
    void *pool;
    float *err_in, *far_in;
    int16_t *pcm_out;
    int hop, hops, i, k;
    const int reprime_hop = 5;

    /* The descriptor is a fixed-width 32-byte POD and the carve alignment is audiolib's 16 bytes. */
    if (sizeof(UlcnetStreamMemReq) != 32u) {
        fprintf(stderr, "FAIL: UlcnetStreamMemReq is %zu bytes, pinned to 32\n", sizeof(UlcnetStreamMemReq));
        return 1;
    }
    if ((size_t)ULCNET_STREAM_ALIGNMENT != 16u || ck_align16_size(1u) != (size_t)ULCNET_STREAM_ALIGNMENT) {
        fprintf(stderr, "FAIL: ULCNET_STREAM_ALIGNMENT is %zu (ck_align16_size(1) = %zu), pinned to the shared 16\n",
                (size_t)ULCNET_STREAM_ALIGNMENT, ck_align16_size(1u));
        return 1;
    }

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000;
    spec.n_fft = 512;
    spec.delay_depth = 32;   /* matches the board sample's --delay-depth default */

    if (ulcnet_grid_make(&grid, &spec) != ULCNET_GRID_OK) {
        fprintf(stderr, "ulcnet_grid_make failed\n");
        return 1;
    }

    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;

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
    err_in = (float *)calloc((size_t)hop, sizeof(float));
    far_in = (float *)calloc((size_t)hop, sizeof(float));
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
            err_in[k] = (float)((k * 7 + i * 3) % 101 - 50) / 32768.0f;
            far_in[k] = (float)((k * 11 + i * 5) % 97 - 48) / 32768.0f;
        }

        delay.changed = (i == reprime_hop) ? 1 : 0;
        delay.solid = 1;

        pre_rc = ulcnet_stream_pre_process_f32(s, err_in, far_in, delay, &io);
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

        if (pre_rc == 1) ai_rc = fake_accelerator(&io.outputs);

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
