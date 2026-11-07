/* Smoke build only: does deepvqe_stream link and run several hops without crashing. Not a
 * correctness test -- it does not check PCM against a reference; that's the separate
 * verification harness. Also asserts the one documented return-value split: post_process returns
 * exactly 0 at hop 0 (AC_OLA_ENV_NORM's first-push startup quirk) and 1 at every hop after. */

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
        if (post_rc != 0 && post_rc != 1) {
            fprintf(stderr, "post_process rc=%d at hop %d (expected 0 or 1)\n", post_rc, i);
            return 1;
        }
        if (i == 0 && post_rc != 0) {
            fprintf(stderr, "post_process rc=%d at hop 0 (expected exactly 0: AC_OLA_ENV_NORM's "
                            "first-push startup case)\n", post_rc);
            return 1;
        }
        if (i > 0 && post_rc != 1) {
            fprintf(stderr, "post_process rc=%d at hop %d (expected exactly 1 past hop 0)\n",
                   post_rc, i);
            return 1;
        }
    }

    deepvqe_stream_reset(s);
    printf("smoke_deepvqe_stream: %d hops, hop=%d, generation after reset=%u -- PASS\n",
          hops, hop, deepvqe_stream_state_generation(s));

    free(mic_in);
    free(far_in);
    free(pcm_out);
    free(pool);
    return 0;
}
