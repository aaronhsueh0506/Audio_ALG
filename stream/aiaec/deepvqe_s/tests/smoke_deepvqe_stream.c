/* Smoke build only: does deepvqe_stream link and run several hops without crashing. Not a
 * correctness test -- it does not check PCM against a reference; that's the separate
 * verification harness. Also asserts the documented returns: post_process returns 1 at every hop
 * from hop 0 onward (N == 1 here), the one-hop entries refuse a stream with N > 1, and at N == 1
 * they equal the packet entries. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepvqe_stream.h"
#include "utility/mem_align.h"

/* Writes zeros/success to taps and every registered state_out -- shape correctness only, no
 * numerical realism. This smoke test registers no state, so only taps is touched. */
static int fake_accelerator(DeepVqePrepostOutputs *out) {
    size_t i;
    for (i = 0; i < out->taps_elements; ++i) out->taps[i] = 0.0f;
    for (i = 0; i < DEEPVQE_STATE_COUNT; ++i) {
        size_t k;
        if (!out->state_out[i]) continue;
        for (k = 0; k < out->state_elements[i]; ++k) out->state_out[i][k] = 0.0f;
    }
    return 0;
}

static void make_stream(int fft, int depth, int n, void **pool_out, DeepVqeStream **s_out, size_t *bytes_out) {
    DeepVqeGridSpec spec;
    DeepVqeGrid grid;
    DeepVqeStreamConfig cfg;
    DeepVqeStreamMemReq req;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = 16000;
    spec.n_fft = fft;
    spec.delay_depth = depth;
    spec.frames_per_call = n;
    memset(&cfg, 0, sizeof(cfg));
    if (deepvqe_grid_make(&grid, &spec) != DEEPVQE_GRID_OK) { fprintf(stderr, "deepvqe_grid_make failed\n"); exit(2); }
    cfg.grid = grid;
    if (deepvqe_stream_get_mem_size(&cfg, &req) != 0) { fprintf(stderr, "get_mem_size failed\n"); exit(2); }
    *pool_out = NULL;
    if (posix_memalign(pool_out, (size_t)req.alignment, (size_t)req.bytes) != 0 || !*pool_out) exit(2);
    *s_out = deepvqe_stream_init_ex(&cfg, *pool_out, (size_t)req.bytes, &req);
    if (!*s_out) { fprintf(stderr, "deepvqe_stream_init_ex failed\n"); exit(2); }
    *bytes_out = (size_t)req.bytes;
}

/* frames_per_call == 8: the one-hop entries refuse and touch nothing; the packet entries pair (a second pre or post out
 * of turn is -1) and one packet writes N * hop samples. Returns 0 on success, 1 on failure. */
static int packet_probe_n8(void) {
    const int n = 8;
    DeepVqeStream *s;
    void *pool;
    size_t bytes, k;
    unsigned char *snap;
    int16_t *mic_in, *far_in, *pcm_out;
    int hop;
    DeepVqeHopIO io, before;
    float *spec;
    int fail = 0;

    make_stream(256, 63, n, &pool, &s, &bytes);
    hop = deepvqe_stream_hop_size(s);
    mic_in = (int16_t *)calloc((size_t)n * (size_t)hop, sizeof(int16_t));
    far_in = (int16_t *)calloc((size_t)n * (size_t)hop, sizeof(int16_t));
    pcm_out = (int16_t *)calloc((size_t)n * (size_t)hop, sizeof(int16_t));
    snap = (unsigned char *)malloc(bytes);
    spec = (float *)calloc((size_t)n * (size_t)(hop + 1) * 2u, sizeof(float));
    if (!mic_in || !far_in || !pcm_out || !snap || !spec) exit(2);
    for (k = 0; k < (size_t)n * (size_t)hop; ++k) { mic_in[k] = (int16_t)(k % 211); far_in[k] = (int16_t)(k * 2 % 199); pcm_out[k] = 777; }

    memset(&io, 0x5A, sizeof io);
    before = io;
    memcpy(snap, pool, bytes);
    if (deepvqe_stream_pre_process(s, mic_in, far_in, &io) != -1) { fprintf(stderr, "N=8: one-hop pre_process accepted\n"); fail = 1; }
    if (deepvqe_stream_pre_process_freq(s, spec, spec, spec, spec, &io) != -1) { fprintf(stderr, "N=8: one-hop pre_process_freq accepted\n"); fail = 1; }
    if (memcmp(snap, pool, bytes) != 0 || memcmp(&io, &before, sizeof io) != 0) { fprintf(stderr, "N=8: a refused one-hop pre_process changed the stream\n"); fail = 1; }

    if (deepvqe_stream_pre_process_packet(s, mic_in, far_in, &io) != 1) { fprintf(stderr, "N=8: packet pre_process did not return 1\n"); fail = 1; }
    if (deepvqe_stream_pre_process_packet(s, mic_in, far_in, &io) != -1) { fprintf(stderr, "N=8: packet pre_process twice in a row did not return -1\n"); fail = 1; }
    memcpy(snap, pool, bytes);
    if (deepvqe_stream_post_process(s, 0, pcm_out) != -1) { fprintf(stderr, "N=8: one-hop post_process accepted\n"); fail = 1; }
    for (k = 0; k < (size_t)n * (size_t)hop; ++k) if (pcm_out[k] != 777) { fprintf(stderr, "N=8: a refused one-hop post_process wrote pcm_out\n"); fail = 1; break; }
    if (deepvqe_stream_post_process_freq(s, 0, spec, spec + (size_t)n * (size_t)(hop + 1)) != -1) { fprintf(stderr, "N=8: one-hop post_process_freq accepted\n"); fail = 1; }
    if (memcmp(snap, pool, bytes) != 0) { fprintf(stderr, "N=8: a refused one-hop post_process changed the stream\n"); fail = 1; }

    (void)fake_accelerator(&io.outputs);
    if (deepvqe_stream_post_process_packet(s, 0, pcm_out) != 1) { fprintf(stderr, "N=8: packet post_process did not return 1\n"); fail = 1; }
    if (deepvqe_stream_post_process_packet(s, 0, pcm_out) != -1) { fprintf(stderr, "N=8: packet post_process twice in a row did not return -1\n"); fail = 1; }
    if (deepvqe_stream_pre_process_packet(s, mic_in, far_in, &io) != 1) { fprintf(stderr, "N=8: the stream is wedged after a packet\n"); fail = 1; }

    if (!fail) printf("smoke_deepvqe_stream: packet probe (N=8, one-hop entries refuse, packet pairing) -- PASS\n");
    free(mic_in); free(far_in); free(pcm_out); free(snap); free(spec); free(pool);
    return fail;
}

/* N == 1: the one-hop entries return what the packet entries return, PCM and spectrum, over a run with an input-dependent
 * accelerator stand-in (a failed hop included). Returns 0 on success, 1 on failure. */
static int one_hop_equals_packet_n1(void) {
    DeepVqeStream *a, *b;
    void *pa, *pb;
    size_t bytes;
    int hop, nb, i, k, fail = 0;
    int16_t *mic, *far, *oa, *ob;
    float *spec, *ra, *rb, *ia, *ib;

    make_stream(256, 63, 1, &pa, &a, &bytes);
    make_stream(256, 63, 1, &pb, &b, &bytes);
    hop = deepvqe_stream_hop_size(a);
    nb = hop + 1;
    mic = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    far = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    oa = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    ob = (int16_t *)calloc((size_t)hop, sizeof(int16_t));
    spec = (float *)calloc((size_t)nb * 4u, sizeof(float));
    ra = (float *)calloc((size_t)nb * 4u, sizeof(float));
    rb = ra + nb; ia = rb + nb; ib = ia + nb;
    if (!mic || !far || !oa || !ob || !spec || !ra) exit(2);
    for (i = 0; i < 40; ++i) {
        DeepVqeHopIO ioa, iob;
        int status = i == 17 ? -1 : 0;

        for (k = 0; k < hop; ++k) { mic[k] = (int16_t)((k * 13 + i * 7) % 2001 - 1000); far[k] = (int16_t)((k * 5 + i * 11) % 1601 - 800); }
        if (deepvqe_stream_pre_process(a, mic, far, &ioa) != 1 || deepvqe_stream_pre_process_packet(b, mic, far, &iob) != 1 ||
            ioa.state_generation != iob.state_generation ||
            memcmp(ioa.inputs.mic, iob.inputs.mic, ioa.inputs.spectrum_ri_elements * sizeof(float)) != 0 ||
            memcmp(ioa.inputs.far, iob.inputs.far, ioa.inputs.spectrum_ri_elements * sizeof(float)) != 0) { fprintf(stderr, "N=1 PCM hop %d: pre differs\n", i); fail = 1; break; }
        for (k = 0; k < (int)ioa.outputs.taps_elements; ++k)
            ioa.outputs.taps[k] = iob.outputs.taps[k] = 0.05f * (float)((k * 31 + i * 17) % 23 - 11) / 11.0f + (k % 18 == 8 ? 0.5f : 0.0f);
        if (deepvqe_stream_post_process(a, status, oa) != 1 || deepvqe_stream_post_process_packet(b, status, ob) != 1 ||
            memcmp(oa, ob, (size_t)hop * sizeof(int16_t)) != 0 ||
            deepvqe_stream_state_generation(a) != deepvqe_stream_state_generation(b)) { fprintf(stderr, "N=1 PCM hop %d: post differs\n", i); fail = 1; break; }
    }
    deepvqe_stream_reset(a);
    deepvqe_stream_reset(b);
    for (i = 0; i < 40 && !fail; ++i) {
        DeepVqeHopIO ioa, iob;
        float *mr = spec, *mi = spec + nb, *fr = spec + 2 * nb, *fi = spec + 3 * nb;
        int status = i == 17 ? -1 : 0;

        for (k = 0; k < nb; ++k) {
            mr[k] = (float)((k * 13 + i * 7) % 2001 - 1000) / 5000.0f; mi[k] = (float)((k * 3 + i * 19) % 1001 - 500) / 5000.0f;
            fr[k] = (float)((k * 5 + i * 11) % 1601 - 800) / 5000.0f; fi[k] = (float)((k * 17 + i) % 701 - 350) / 5000.0f;
        }
        if (deepvqe_stream_pre_process_freq(a, mr, mi, fr, fi, &ioa) != 1 ||
            deepvqe_stream_pre_process_packet_freq(b, mr, mi, fr, fi, &iob) != 1) { fprintf(stderr, "N=1 freq hop %d: pre differs\n", i); fail = 1; break; }
        for (k = 0; k < (int)ioa.outputs.taps_elements; ++k)
            ioa.outputs.taps[k] = iob.outputs.taps[k] = 0.05f * (float)((k * 29 + i * 13) % 23 - 11) / 11.0f + (k % 18 == 8 ? 0.5f : 0.0f);
        if (deepvqe_stream_post_process_freq(a, status, ra, ia) != 1 || deepvqe_stream_post_process_packet_freq(b, status, rb, ib) != 1 ||
            memcmp(ra, rb, (size_t)nb * sizeof(float)) != 0 || memcmp(ia, ib, (size_t)nb * sizeof(float)) != 0 ||
            deepvqe_stream_state_generation(a) != deepvqe_stream_state_generation(b)) { fprintf(stderr, "N=1 freq hop %d: post differs\n", i); fail = 1; break; }
    }
    if (!fail) printf("smoke_deepvqe_stream: N=1 one-hop entries equal the packet entries (PCM and spectrum) -- PASS\n");
    free(mic); free(far); free(oa); free(ob); free(spec); free(ra); free(pa); free(pb);
    return fail;
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

    /* The descriptor is a fixed-width 32-byte POD and the carve alignment is audiolib's 16 bytes. */
    if (sizeof(DeepVqeStreamMemReq) != 32u) {
        fprintf(stderr, "FAIL: DeepVqeStreamMemReq is %zu bytes, pinned to 32\n", sizeof(DeepVqeStreamMemReq));
        return 1;
    }
    if ((size_t)DEEPVQE_STREAM_ALIGNMENT != 16u || ck_align16_size(1u) != (size_t)DEEPVQE_STREAM_ALIGNMENT) {
        fprintf(stderr, "FAIL: DEEPVQE_STREAM_ALIGNMENT is %zu (ck_align16_size(1) = %zu), pinned to the shared 16\n",
                (size_t)DEEPVQE_STREAM_ALIGNMENT, ck_align16_size(1u));
        return 1;
    }

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

        ai_rc = fake_accelerator(&io.outputs);

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
        (void)fake_accelerator(&io.outputs);
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
        (void)fake_accelerator(&io.outputs);

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

    if (packet_probe_n8() != 0) return 1;
    if (one_hop_equals_packet_n1() != 0) return 1;

    return 0;
}
