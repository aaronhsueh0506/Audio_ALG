/* Event mapping, configuration gates, pool gates and input gates of aiaec_front, without needing the
 * AEC to converge. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aiaec_front.h"
#include "aiaec_front_test.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

static AiaecFrontConfig cfg_of(int sr, int fft, int hop_in) {
    AiaecFrontConfig fc;
    AiaecConfig g;
    g.sr = sr; g.fft = fft; g.hop_in = hop_in;
    aiaec_front_config_defaults(&fc, &g);
    return fc;
}

static int init_err(const AiaecFrontConfig *fc) {
    int err = 12345;
    void *pool = NULL;
    size_t bytes = aiaec_front_get_mem_size_cfg(fc);
    AiaecFront *f;
    if (bytes == 0) {
        /* the size query refuses; init must name the same reason */
        char scratch[64] __attribute__((aligned(16)));
        f = aiaec_front_init_cfg(scratch, sizeof(scratch), fc, &err);
        CHECK(f == NULL, "init accepted a config the size query refused");
        return err;
    }
    if (posix_memalign(&pool, 16, bytes) != 0) abort();
    f = aiaec_front_init_cfg(pool, bytes, fc, &err);
    if (f) { err = AIAEC_OK; }
    free(pool);
    return err;
}

static void test_event_mapping(void) {
    AiaecAlignEvent e;
    e = aiaec_front_event_from_delay_state(AEC_LINEAR_DELAY_UNLOCKED);
    CHECK(e.changed == 0 && e.solid == 0, "UNLOCKED -> %d/%d", e.changed, e.solid);
    e = aiaec_front_event_from_delay_state(AEC_LINEAR_DELAY_LOCKED);
    CHECK(e.changed == 0 && e.solid == 1, "LOCKED -> changed=%d solid=%d", e.changed, e.solid);
    e = aiaec_front_event_from_delay_state(AEC_LINEAR_DELAY_CHANGED);
    CHECK(e.changed == 1 && e.solid == 1, "CHANGED -> changed=%d solid=%d", e.changed, e.solid);
    e = aiaec_front_event_from_delay_state(-1);
    CHECK(e.changed == 0 && e.solid == 0, "-1 -> %d/%d", e.changed, e.solid);
    e = aiaec_front_event_from_delay_state(3);
    CHECK(e.changed == 0 && e.solid == 0, "3 -> %d/%d", e.changed, e.solid);
}

static void test_config_gates(void) {
    AiaecFrontConfig fc;

    fc = cfg_of(16000, 256, 128);
    CHECK(fc.delay_mode == AIAEC_FRONT_DELAY_MATCHED && fc.delay_num_filters == 5 && fc.fixed_delay_samples == -1,
          "defaults");
    CHECK(aiaec_front_get_mem_size_cfg(&fc) > 0, "16k/256/128 refused");
    CHECK(init_err(&fc) == AIAEC_OK, "16k/256/128 init");
    fc = cfg_of(16000, 256, 1024);
    CHECK(init_err(&fc) == AIAEC_OK, "16k/256/1024 init");
    fc = cfg_of(16000, 512, 256);
    CHECK(init_err(&fc) == AIAEC_OK, "16k/512/256 init");
    fc = cfg_of(48000, 1024, 512);
    CHECK(init_err(&fc) == AIAEC_OK, "48k/1024/512 init");

    fc = cfg_of(16000, 256, 129);   CHECK(init_err(&fc) == AIAEC_ERR_IN_HOP, "hop_in 129");
    fc = cfg_of(16000, 256, 0);     CHECK(init_err(&fc) == AIAEC_ERR_IN_HOP, "hop_in 0");
    fc = cfg_of(16000, 256, -128);  CHECK(init_err(&fc) == AIAEC_ERR_IN_HOP, "hop_in -128");
    fc = cfg_of(16000, 256, 128 * 1024); CHECK(init_err(&fc) == AIAEC_ERR_IN_HOP, "hop_in too large");
    fc = cfg_of(44100, 256, 128);   CHECK(init_err(&fc) == AIAEC_ERR_GRID, "44.1k");
    fc = cfg_of(16000, 1024, 512);  CHECK(init_err(&fc) == AIAEC_ERR_GRID, "16k/1024");
    fc = cfg_of(8000, 256, 128);    CHECK(init_err(&fc) == AIAEC_ERR_GRID, "8k legacy grid");
    CHECK(aiaec_front_get_mem_size_cfg(NULL) == 0, "NULL config size");
    CHECK(aiaec_front_init_cfg(NULL, 0, NULL, NULL) == NULL, "NULL config init");

    fc = cfg_of(16000, 256, 128); fc.delay_mode = 3;
    CHECK(init_err(&fc) == AIAEC_FRONT_ERR_DELAY, "mode 3");
    fc = cfg_of(16000, 256, 128); fc.delay_num_filters = 0;
    CHECK(init_err(&fc) == AIAEC_FRONT_ERR_DELAY, "0 filters");
    fc = cfg_of(16000, 256, 128); fc.delay_num_filters = 6;
    CHECK(init_err(&fc) == AIAEC_FRONT_ERR_DELAY, "6 filters");
    fc = cfg_of(16000, 256, 128); fc.fixed_delay_samples = 100;
    CHECK(init_err(&fc) == AIAEC_FRONT_ERR_DELAY, "matched with a fixed delay");
    fc = cfg_of(16000, 256, 128); fc.delay_mode = AIAEC_FRONT_DELAY_FIXED;
    CHECK(init_err(&fc) == AIAEC_FRONT_ERR_DELAY, "fixed without a delay");
    fc.fixed_delay_samples = 160;
    CHECK(init_err(&fc) == AIAEC_OK, "fixed 160");
    fc = cfg_of(16000, 256, 128); fc.delay_mode = AIAEC_FRONT_DELAY_EXTERNAL; fc.fixed_delay_samples = 5;
    CHECK(init_err(&fc) == AIAEC_FRONT_ERR_DELAY, "external with a delay");
    fc.fixed_delay_samples = -1;
    CHECK(init_err(&fc) == AIAEC_OK, "external");
}

/* The product entry takes the grid only and uses the default policy: same pool size and same refusals as the
 * config-based entry with aiaec_front_config_defaults(). */
static void test_product_grid(void) {
    static const int grids[][3] = { {16000, 256, 128}, {16000, 256, 1024}, {16000, 512, 256}, {48000, 1024, 512},
                                    {16000, 256, 129}, {16000, 256, 0}, {44100, 256, 128}, {16000, 1024, 512}, {8000, 256, 128} };
    size_t k;
    for (k = 0; k < sizeof(grids) / sizeof(grids[0]); ++k) {
        AiaecConfig g;
        AiaecFrontConfig fc = cfg_of(grids[k][0], grids[k][1], grids[k][2]);
        size_t bytes;
        void *pool = NULL;
        int err = 12345;
        AiaecFront *f;
        g.sr = grids[k][0]; g.fft = grids[k][1]; g.hop_in = grids[k][2];
        bytes = aiaec_front_get_mem_size(&g);
        CHECK(bytes == aiaec_front_get_mem_size_cfg(&fc), "grid %d/%d/%d: size %zu vs defaults %zu", g.sr, g.fft, g.hop_in,
              bytes, aiaec_front_get_mem_size_cfg(&fc));
        if (bytes == 0) {
            char scratch[64] __attribute__((aligned(16)));
            CHECK(aiaec_front_init(scratch, sizeof(scratch), &g, &err) == NULL && err == init_err(&fc), "grid %d/%d/%d: refusal %d vs %d",
                  g.sr, g.fft, g.hop_in, err, init_err(&fc));
            continue;
        }
        if (posix_memalign(&pool, 16, bytes) != 0) abort();
        f = aiaec_front_init(pool, bytes, &g, &err);
        CHECK(f != NULL && err == AIAEC_OK, "grid %d/%d/%d: init err=%d", g.sr, g.fft, g.hop_in, err);
        free(pool);
    }
    CHECK(aiaec_front_get_mem_size(NULL) == 0, "NULL grid size");
    {
        int err = 0;
        CHECK(aiaec_front_init(NULL, 0, NULL, &err) == NULL && err == AIAEC_ERR_NULL, "NULL grid init err=%d", err);
    }
}

static void test_pool_and_input_gates(void) {
    AiaecFrontConfig fc = cfg_of(16000, 256, 256);
    const size_t bytes = aiaec_front_get_mem_size_cfg(&fc);
    const size_t guard = 64;
    unsigned char *raw = NULL;
    unsigned char *pool;
    AiaecFront *f;
    AiaecFrontOut out;
    int16_t mic[256], far[256];
    float mf[256], ff[256];
    int err = 0, i;

    if (posix_memalign((void **)&raw, 16, bytes + 2 * guard) != 0) abort();
    pool = raw + guard;
    memset(raw, 0xA5, bytes + 2 * guard);

    CHECK(aiaec_front_init_cfg(pool, bytes - 16, &fc, &err) == NULL && err == AIAEC_ERR_POOL, "undersized pool err=%d", err);
    CHECK(aiaec_front_init_cfg(pool + 1, bytes, &fc, &err) == NULL && err == AIAEC_ERR_POOL, "misaligned pool err=%d", err);
    CHECK(aiaec_front_init_cfg(NULL, bytes, &fc, &err) == NULL && err == AIAEC_ERR_NULL, "NULL pool err=%d", err);

    f = aiaec_front_init_cfg(pool, bytes, &fc, &err);
    CHECK(f != NULL && err == AIAEC_OK, "exact pool err=%d", err);
    if (!f) { free(raw); return; }

    for (i = 0; i < 256; ++i) { mic[i] = (int16_t)(i * 7); far[i] = (int16_t)(-i * 5); mf[i] = 0.01f; ff[i] = 0.02f; }
    CHECK(aiaec_front_process_i16(f, mic, far, 128, &out) == AIAEC_ERR_INPUT_SIZE, "short block");
    CHECK(out.formed_hop == NULL && out.events == NULL && out.event_count == 0, "out not cleared on a refusal");
    CHECK(aiaec_front_process_i16(f, mic, far, 512, &out) == AIAEC_ERR_INPUT_SIZE, "long block");
    CHECK(aiaec_front_process_i16(NULL, mic, far, 256, &out) == AIAEC_ERR_NULL, "NULL front");
    CHECK(aiaec_front_process_i16(f, NULL, far, 256, &out) == AIAEC_ERR_NULL, "NULL mic");
    CHECK(aiaec_front_process_f32(f, mf, NULL, 256, &out) == AIAEC_ERR_NULL, "NULL far");
    CHECK(aiaec_front_process_f32(f, mf, ff, 256, NULL) == AIAEC_ERR_NULL, "NULL out");

    CHECK(aiaec_front_process_i16(f, mic, far, 256, &out) == AIAEC_OK, "i16 block");
    CHECK(out.event_count == 2 && out.samples == 256 && out.formed_hop && out.raw_far && out.events, "out shape");
    CHECK(out.raw_far[3] == (float)far[3] / 32768.0f, "raw_far scale");
    CHECK(aiaec_front_process_f32(f, mf, ff, 256, &out) == AIAEC_OK, "f32 block");
    CHECK(out.raw_far == ff, "f32 raw_far not the caller's far");
    aiaec_front_reset(f);
    aiaec_front_reset(NULL);

    {
        AiaecInput in;
        aiaec_front_fill_align_input(&out, &in);
        CHECK(in.kind == AIAEC_INPUT_ALIGN_ULCNET && in.samples == 256 &&
              in.u.align_ulcnet.residual_in == out.formed_hop && in.u.align_ulcnet.far_in == out.raw_far &&
              in.u.align_ulcnet.events == out.events && in.u.align_ulcnet.event_count == 2, "fill_align_input");
    }

    for (i = 0; i < (int)guard; ++i)
        CHECK(raw[i] == 0xA5 && raw[guard + bytes + (size_t)i] == 0xA5, "pool guard byte %d overwritten", i);
    free(raw);
}

int main(void) {
    test_event_mapping();
    test_config_gates();
    test_product_grid();
    test_pool_and_input_gates();
    if (g_fail) { fprintf(stderr, "test_front_events: %d failure(s)\n", g_fail); return 1; }
    printf("test_front_events: PASS\n");
    return 0;
}
