/* The board samples' AIAEC graph rows (ai3_model_io_deepvqes.h, ai3_model_io_alignulcnet.h) against the AiaecWork of
 * real pre calls (PCM, _spec, _freq) for the product this binary is built for (-DAIAEC_PRODUCT_MODEL_ID: 1 DeepVQE-S,
 * 10 Align-ULCNet): every SUBMIT block passes AI3_MODEL_IO_CHECK_WORK against the row of its configuration, and with
 * the test catalog (-DAIAEC_TEST_CATALOG) the library accepts a (sr, fft, N) exactly when the row exists; with the
 * production catalog every accepted configuration has a row. Align-ULCNet also runs an event hop: N = 1 is BYPASS with
 * an empty work, N = 8 at hop 2 puts frame_reset 00100000 in the last signal. The rows against a fresh export are
 * pinned by AIAEC/tests/test_aiaec_bind_info_pin.py. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aiaec.h"
#include "aiaec_product.h"
#include "ai3_model_io.h"
#include "deepvqe_prepost.h"
#include "ulcnet_model_io.h"
#include "ai3_model_io_alignulcnet.h"
#include "ai3_model_io_deepvqes.h"
#if AIAEC_PRODUCT_MODEL_ID == 10
#define FIND(sr, fft, n, mc) ai3_model_io_alignulcnet_find(sr, fft, n, mc)
#define ROW_COUNT AI3_MODEL_IO_ALIGNULCNET_ROW_COUNT
#define IS_ALIGN 1
#else
#define FIND(sr, fft, n, mc) ai3_model_io_deepvqes_find(sr, fft, n, mc)
#define ROW_COUNT AI3_MODEL_IO_DEEPVQES_ROW_COUNT
#define IS_ALIGN 0
#endif

_Static_assert(AI3_MODEL_IO_DEEPVQES_LAYOUT == DEEPVQE_PREPOST_LAYOUT_VERSION, "DeepVQE-S layout version");
_Static_assert(AI3_MODEL_IO_ALIGNULCNET_LAYOUT_ONE_FRAME == ULCNET_MODEL_IO_LAYOUT_VERSION, "Align-ULCNet layout 18");
_Static_assert(AI3_MODEL_IO_ALIGNULCNET_LAYOUT_PACKET == ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION, "Align-ULCNet layout 19");

static int g_fail, g_patterns;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static void compare_work(const char *tag, const Ai3ModelIoConfig *mc, const AiaecWork *work)
{
    int i, rc;
    const StreamWiring *w = &mc->wiring;
    CHECK(work->action == AIAEC_SUBMIT, "%s action %d", tag, (int)work->action);
    CHECK(work->input_count == w->signal_in_count, "%s input_count %d vs %d", tag, work->input_count, w->signal_in_count);
    CHECK(work->output_count == w->head_out_count, "%s output_count %d vs %d", tag, work->output_count, w->head_out_count);
    for (i = 0; i < work->input_count && i < w->signal_in_count; ++i)
        CHECK(work->inputs[i].b == w->inputs[i].b && work->inputs[i].c == w->inputs[i].c &&
              work->inputs[i].h == w->inputs[i].h && work->inputs[i].w == w->inputs[i].w &&
              work->inputs[i].elements == w->inputs[i].elements, "%s signal %d shape", tag, i);
    for (i = 0; i < work->output_count && i < w->head_out_count; ++i)
        CHECK(work->outputs[i].b == w->outputs[i].b && work->outputs[i].c == w->outputs[i].c &&
              work->outputs[i].h == w->outputs[i].h && work->outputs[i].w == w->outputs[i].w &&
              work->outputs[i].elements == w->outputs[i].elements, "%s head %d shape", tag, i);
    AI3_MODEL_IO_CHECK_WORK(rc, mc, work);
    CHECK(rc == AI3_MODEL_IO_OK, "%s AI3_MODEL_IO_CHECK_WORK rc %d", tag, rc);
    CHECK(ai3_model_io_fits(mc, AIAEC_MAX_INPUT_TENSORS, AIAEC_MAX_OUTPUT_TENSORS) == AI3_MODEL_IO_OK, "%s fits", tag);
}

/* One pre call of `family` on a fresh init; event_hop < 0: no event. Returns the pre status. */
static int run_pre(const AiaecConfig *cfg, int family, int event_hop, AiaecWork *work, AiaecInfo *info)
{
    StreamMemReq req;
    AiaecCtx *pcm = NULL; AiaecSpecCtx *spec = NULL; AiaecFreqCtx *freq = NULL;
    AiaecInput in; AiaecFreqInput fin;
    AiaecAlignEvent *events;
    int16_t *s16; float *f32;
    void *pool = NULL;
    int rc, n = cfg->hop_in / (cfg->fft / 2), bins = cfg->fft / 2 + 1;

    rc = family == 0 ? aiaec_get_mem_size(cfg, &req) : family == 1 ? aiaec_get_mem_size_spec(cfg, &req)
                                                                   : aiaec_get_mem_size_freq(cfg, &req);
    if (rc != AIAEC_OK) return rc;
    if (posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) exit(2);
    rc = family == 0 ? aiaec_init(cfg, pool, (size_t)req.bytes, &pcm, info)
       : family == 1 ? aiaec_init_spec(cfg, pool, (size_t)req.bytes, &spec, info)
                     : aiaec_init_freq(cfg, pool, (size_t)req.bytes, &freq, info);
    if (rc != AIAEC_OK) { free(pool); return rc; }
    events = (AiaecAlignEvent *)calloc((size_t)n, sizeof *events);
    s16 = (int16_t *)calloc((size_t)cfg->hop_in, sizeof *s16);
    f32 = (float *)calloc((size_t)(cfg->hop_in > n * bins ? cfg->hop_in : n * bins), sizeof *f32);
    if (event_hop >= 0 && event_hop < n) events[event_hop].changed = 1;
    memset(work, 0, sizeof *work); memset(&in, 0, sizeof in); memset(&fin, 0, sizeof fin);
#if IS_ALIGN
    in.kind = AIAEC_INPUT_ALIGN_ULCNET; in.samples = cfg->hop_in;
    in.u.align_ulcnet.residual_in = f32; in.u.align_ulcnet.far_in = f32;
    in.u.align_ulcnet.events = events; in.u.align_ulcnet.event_count = n;
    fin.kind = AIAEC_INPUT_ALIGN_ULCNET; fin.frames = n;
    fin.u.align_ulcnet.err_re = fin.u.align_ulcnet.err_im = fin.u.align_ulcnet.far_re =
        fin.u.align_ulcnet.far_im = f32;
    fin.u.align_ulcnet.events = events; fin.u.align_ulcnet.event_count = n;
#else
    in.kind = AIAEC_INPUT_DEEPVQE_S; in.samples = cfg->hop_in;
    in.u.deepvqe_s.mic_in = s16; in.u.deepvqe_s.far_in = s16;
    fin.kind = AIAEC_INPUT_DEEPVQE_S; fin.frames = n;
    fin.u.deepvqe_s.mic_re = fin.u.deepvqe_s.mic_im = fin.u.deepvqe_s.far_re = fin.u.deepvqe_s.far_im = f32;
#endif
    rc = family == 0 ? aiaec_preprocess(pcm, &in, work)
       : family == 1 ? aiaec_preprocess_spec(spec, &in, work) : aiaec_preprocess_freq(freq, &fin, work);
    if (rc == AIAEC_OK && IS_ALIGN && work->input_count == 5 && event_hop >= 0) {
        char pattern[65];   /* frame_reset is the last signal: 1.0 at the event hop only */
        int k;
        for (k = 0; k < n; ++k) pattern[k] = work->inputs[4].data[k] == 1.0f ? '1' : work->inputs[4].data[k] == 0.0f ? '0' : '?';
        pattern[n] = '\0';
        CHECK(n != 8 || event_hop != 2 || strcmp(pattern, "00100000") == 0, "frame_reset %s, want 00100000", pattern);
        CHECK(pattern[event_hop] == '1' && strchr(pattern, '1') == strrchr(pattern, '1'), "frame_reset %s: one event at %d",
              pattern, event_hop);
        ++g_patterns;
    }
    free(pool); free(events); free(s16); free(f32);
    return rc;
}

int main(void)
{
    static const int ffts[] = { 128, 256, 512, 1024 };
    static const int srs[] = { 16000, 48000 };
    int si, fi, n, accepted = 0, rows_hit = 0, compared = 0, family;

    for (si = 0; si < 2; ++si) for (fi = 0; fi < 4; ++fi) for (n = 1; n <= 8; ++n) {
        AiaecConfig cfg;
        Ai3ModelIoConfig mc;
        StreamMemReq req;
        int lib_ok, row_ok;

        cfg.sr = srs[si]; cfg.fft = ffts[fi]; cfg.hop_in = n * (ffts[fi] / 2);
        lib_ok = aiaec_get_mem_size(&cfg, &req) == AIAEC_OK;
        row_ok = FIND(cfg.sr, cfg.fft, ai3_model_io_frames(cfg.fft, cfg.hop_in), &mc) == AI3_MODEL_IO_OK;
        accepted += lib_ok; rows_hit += row_ok;
#ifdef AIAEC_TEST_CATALOG
        CHECK(lib_ok == row_ok, "sr %d fft %d N %d: library %s, model config %s", cfg.sr, cfg.fft, n,
              lib_ok ? "accepts" : "refuses", row_ok ? "has a row" : "has no row");
#else
        CHECK(!lib_ok || row_ok, "sr %d fft %d N %d: production catalog accepts, model config has no row", cfg.sr,
              cfg.fft, n);
#endif
        if (!lib_ok || !row_ok) continue;
        for (family = 0; family < 3; ++family) {
            AiaecWork work; AiaecInfo info;
            char tag[96];
            int rc, e, ev, n_events = IS_ALIGN ? 2 : 1;   /* no event, then (Align-ULCNet) an event at hop 2 or the last */
            for (e = 0; e < n_events; ++e) {
                ev = e == 0 ? -1 : (n > 2 ? 2 : n - 1);
                snprintf(tag, sizeof tag, "sr %d fft %d N %d family %d event_hop %d", cfg.sr, cfg.fft, n, family, ev);
                rc = run_pre(&cfg, family, ev, &work, &info);
                CHECK(rc == AIAEC_OK, "%s pre rc %d", tag, rc);
                if (rc != AIAEC_OK) continue;
                if (IS_ALIGN && n == 1 && ev == 0) {
                    /* an N = 1 event hop is BYPASS: no tensors, so no boundary check (and none would pass) */
                    CHECK(work.action == AIAEC_BYPASS && work.input_count == 0 && work.output_count == 0,
                          "%s expected BYPASS with an empty work", tag);
                    CHECK(ai3_model_io_check_counts(&mc, work.input_count, work.output_count) == AI3_MODEL_IO_ERR_COUNT,
                          "%s the boundary check must reject an empty BYPASS work", tag);
                    ++compared;
                    continue;
                }
                compare_work(tag, &mc, &work);
                ++compared;
            }
        }
    }
#ifdef AIAEC_TEST_CATALOG
    CHECK(rows_hit == ROW_COUNT && accepted == ROW_COUNT, "%d rows, %d hit, %d configs accepted", ROW_COUNT, rows_hit, accepted);
#endif
    if (g_fail) return 1;
    printf("test_aiaec_model_io[%s, %s catalog]: PASS (library-accepted configs %d, rows hit %d of %d, pre calls compared %d, frame_reset patterns %d)\n",
           IS_ALIGN ? "align_ulcnet" : "deepvqe_s",
#ifdef AIAEC_TEST_CATALOG
           "test",
#else
           "production",
#endif
           accepted, rows_hit, ROW_COUNT, compared, g_patterns);
    return 0;
}
