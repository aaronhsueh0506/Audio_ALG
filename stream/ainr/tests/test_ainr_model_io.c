/* The board samples' DFN2 graph rows (stream/ainr/ai3_model_io_dfn2.h) against the AinrWork a real preprocess
 * call returns, for this build's variant (full, or lite under -DDFN2_MASK_ONLY) and product id
 * (AINR_PRODUCT_MODEL_ID, incl. the quantization alternates): every PCM, _spec and _freq block must pass
 * AI3_MODEL_IO_CHECK_WORK against the row of its configuration, and the library accepts a (sr, fft, N) exactly
 * when the row exists (a product id with no catalog rows refuses every configuration). The other variant's row
 * of the same grid must reject this variant's work. The second block of a family carries the same shapes as the
 * first, and every refusal of ainr_get_mem_size() comes back from the three init families with *ctx NULL and
 * *info untouched. The rows against a fresh export are pinned by AINR/tests/test_dfn2_port_order.py. Not built
 * under NO_STDIO (ainr.c is file-mode). */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ainr.h"
#include "ainr_test.h"
#include "ainr_product.h"
#include "dfn2_model_io.h"
#include "ai3_model_io.h"
#include "ai3_model_io_dfn2.h"

_Static_assert(AI3_MODEL_IO_DFN2_LAYOUT_FULL == DFN2_MODEL_IO_LAYOUT_VERSION_FULL, "full layout version");
_Static_assert(AI3_MODEL_IO_DFN2_LAYOUT_LITE == DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY, "lite layout version");

/* The product id has catalog rows in this build (ainr.c: the default and _ALT2 of the variant; _ALT1 and the
 * other variant's ids are refused by every entry). */
#ifdef DFN2_MASK_ONLY
#define ID_HAS_ROWS (AINR_PRODUCT_MODEL_ID == AINR_MODEL_ID_DFN2_LITE || AINR_PRODUCT_MODEL_ID == AINR_MODEL_ID_DFN2_LITE_ALT2)
#else
#define ID_HAS_ROWS (AINR_PRODUCT_MODEL_ID == AINR_MODEL_ID_DFN2_FULL || AINR_PRODUCT_MODEL_ID == AINR_MODEL_ID_DFN2_FULL_ALT2)
#endif

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

static int write_erb(const char *path, int n_bins, int n_erb)
{
    size_t elems = (size_t)n_bins * (size_t)n_erb * 2u, i;
    int32_t hdr[2];
    float z = 0.0f;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    hdr[0] = n_bins; hdr[1] = n_erb;
    if (fwrite(hdr, sizeof(int32_t), 2, fp) != 2) { fclose(fp); return -1; }
    for (i = 0; i < elems; ++i)
        if (fwrite(&z, sizeof z, 1, fp) != 1) { fclose(fp); return -1; }
    return fclose(fp);
}

static void compare_work(const char *tag, const Ai3ModelIoConfig *mc, const AinrWork *work)
{
    const StreamWiring *w = &mc->wiring;
    int i, rc;

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
    CHECK(ai3_model_io_fits(mc, AINR_MAX_INPUT_TENSORS, AINR_MAX_OUTPUT_TENSORS) == AI3_MODEL_IO_OK, "%s fits", tag);
}

static void fill_heads(AinrWork *w)
{
    int i;
    size_t k;
    for (i = 0; i < w->output_count; ++i)
        for (k = 0; k < w->outputs[i].elements; ++k) w->outputs[i].data[k] = i == 0 ? 0.5f : 0.0f;
}

/* Every refusal of the sizing call comes back from the three init families, with *ctx NULL and *info untouched. */
static void check_refusals(const char *erb)
{
    static const struct { int sr, fft, hop_in, want; const char *what; } cases[] = {
        { 44100, 256, 128, AINR_ERR_GRID, "unregistered grid" },
        { 48000, 1024, 100, AINR_ERR_IN_HOP, "hop_in not a multiple of model_hop" },
        { 48000, 1024, 1024, AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED, "lookahead grid with N = 2" },
#ifndef DFN2_MASK_ONLY
        { 16000, 256, 512, AINR_ERR_FRAMES_PER_CALL, "N = 4 has no graph" },
#endif
    };
    _Alignas(16) static unsigned char pool[1 << 21];
    size_t i;
    AinrInfo info, sentinel;
    AinrConfig cfg;
    StreamMemReq req;
    memset(&sentinel, 0x5a, sizeof(sentinel));
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        AinrCtx *c0 = (AinrCtx *)pool;
        AinrSpecCtx *c1 = (AinrSpecCtx *)pool;
        AinrFreqCtx *c2 = (AinrFreqCtx *)pool;
        cfg.sr = cases[i].sr; cfg.fft = cases[i].fft; cfg.hop_in = cases[i].hop_in;
        CHECK(ainr_get_mem_size(&cfg, &req) == cases[i].want, "refusal (%s): get_mem_size", cases[i].what);
        info = sentinel;
        CHECK(ainr_init_from_erb_file(&cfg, erb, pool, sizeof(pool), &c0, &info) == cases[i].want, "refusal (%s): init",
              cases[i].what);
        CHECK(c0 == NULL && memcmp(&info, &sentinel, sizeof(info)) == 0, "refusal (%s): ctx/info after init", cases[i].what);
        info = sentinel;
        CHECK(ainr_init_spec_from_erb_file(&cfg, erb, pool, sizeof(pool), &c1, &info) == cases[i].want,
              "refusal (%s): init_spec", cases[i].what);
        CHECK(c1 == NULL && memcmp(&info, &sentinel, sizeof(info)) == 0, "refusal (%s): ctx/info after init_spec",
              cases[i].what);
        info = sentinel;
        CHECK(ainr_init_freq_from_erb_file(&cfg, erb, pool, sizeof(pool), &c2, &info) == cases[i].want,
              "refusal (%s): init_freq", cases[i].what);
        CHECK(c2 == NULL && memcmp(&info, &sentinel, sizeof(info)) == 0, "refusal (%s): ctx/info after init_freq",
              cases[i].what);
    }
}

/* One pre call of `family` on a fresh init; compares the work with the row, then closes the block and compares
 * the next one. */
static void run_family(const AinrConfig *cfg, int family, int n, const Ai3ModelIoConfig *mc, const char *tag)
{
    AinrInfo info;
    AinrWork work;
    StreamMemReq req;
    AinrCtx *pcm = NULL;
    AinrSpecCtx *spec = NULL;
    AinrFreqCtx *freq = NULL;
    void *pool = NULL;
    int16_t *pcm_in;
    float *zeros, *out_re, *out_im;
    char erb[96];
    int rc, bins = cfg->fft / 2 + 1;

    rc = family == 0 ? ainr_get_mem_size(cfg, &req) : family == 1 ? ainr_get_mem_size_spec(cfg, &req)
                                                                  : ainr_get_mem_size_freq(cfg, &req);
    CHECK(rc == AINR_OK, "%s mem size %d", tag, rc);
    if (rc != AINR_OK) return;
    snprintf(erb, sizeof erb, "/tmp/ainr_model_io_erb_%d.bin", (int)getpid());
    if (write_erb(erb, bins, (int)mc->wiring.outputs[0].w) != 0) exit(2);
    if (posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) exit(2);
    rc = family == 0 ? ainr_init_from_erb_file(cfg, erb, pool, (size_t)req.bytes, &pcm, &info)
       : family == 1 ? ainr_init_spec_from_erb_file(cfg, erb, pool, (size_t)req.bytes, &spec, &info)
                     : ainr_init_freq_from_erb_file(cfg, erb, pool, (size_t)req.bytes, &freq, &info);
    unlink(erb);
    CHECK(rc == AINR_OK, "%s init %d", tag, rc);
    pcm_in = (int16_t *)calloc((size_t)cfg->hop_in, sizeof(int16_t));
    zeros = (float *)calloc((size_t)n * (size_t)bins, sizeof(float));
    out_re = (float *)calloc((size_t)n * (size_t)bins, sizeof(float));
    out_im = (float *)calloc((size_t)n * (size_t)bins, sizeof(float));
    if (!pcm_in || !zeros || !out_re || !out_im) exit(2);
    memset(&work, 0, sizeof work);
    if (rc == AINR_OK) {
        rc = family == 0 ? ainr_preprocess(pcm, pcm_in, &work)
           : family == 1 ? ainr_preprocess_spec(spec, pcm_in, &work)
                         : ainr_preprocess_freq(freq, zeros, zeros, zeros, zeros, &work);
        CHECK(rc == AINR_OK, "%s preprocess %d", tag, rc);
        if (rc == AINR_OK) {
            Ai3ModelIoConfig other;
            int other_model = AI3_MODEL_IO_DFN2_PRODUCT == AI3_MODEL_IO_DFN2_LITE ? AI3_MODEL_IO_DFN2_FULL
                                                                                  : AI3_MODEL_IO_DFN2_LITE;
            compare_work(tag, mc, &work);
            /* the other variant's row of the same grid must refuse this variant's work */
            if (ai3_model_io_dfn2_find(other_model, cfg->sr, cfg->fft, n, &other) == AI3_MODEL_IO_OK) {
                int rc2;
                AI3_MODEL_IO_CHECK_WORK(rc2, &other, &work);
                CHECK(rc2 != AI3_MODEL_IO_OK, "%s the other variant's row accepted this work", tag);
            }
            fill_heads(&work);
            rc = family == 0 ? ainr_postprocess(pcm, 0, pcm_in) : family == 1 ? ainr_postprocess_spec(spec, 0, out_re, out_im)
                                                                              : ainr_postprocess_freq(freq, 0, out_re, out_im);
            CHECK(rc == AINR_OK, "%s postprocess %d", tag, rc);
            memset(&work, 0, sizeof work);
            rc = family == 0 ? ainr_preprocess(pcm, pcm_in, &work)
               : family == 1 ? ainr_preprocess_spec(spec, pcm_in, &work)
                             : ainr_preprocess_freq(freq, zeros, zeros, zeros, zeros, &work);
            CHECK(rc == AINR_OK, "%s second preprocess %d", tag, rc);
            if (rc == AINR_OK) compare_work(tag, mc, &work);
        }
    }
    free(pool); free(pcm_in); free(zeros); free(out_re); free(out_im);
}

int main(void)
{
    static const int srs[] = { 8000, 16000, 32000, 48000 };
    static const int ffts[] = { 128, 256, 512, 1024, 2048 };
    int si, fi, n, i, accepted = 0, own_rows = 0, compared = 0;

    for (i = 0; i < AI3_MODEL_IO_DFN2_ROW_COUNT; ++i) {
        Ai3ModelIoConfig c;
        CHECK(ai3_model_io_dfn2_row(i, &c) == AI3_MODEL_IO_OK, "row %d expands", i);
        own_rows += c.model == AI3_MODEL_IO_DFN2_PRODUCT;
    }
    CHECK(ai3_model_io_dfn2_row(AI3_MODEL_IO_DFN2_ROW_COUNT, NULL) == AI3_MODEL_IO_ERR_NULL, "row NULL cfg");

    if (ID_HAS_ROWS) {
        char erb[96];
        snprintf(erb, sizeof erb, "/tmp/ainr_model_io_refusal_%d.bin", (int)getpid());
        if (write_erb(erb, 513, 32) != 0) exit(2);
        check_refusals(erb);
        unlink(erb);
    }

    for (si = 0; si < 4; ++si) for (fi = 0; fi < 5; ++fi) for (n = 1; n <= 8; ++n) {
        AinrConfig cfg;
        Ai3ModelIoConfig mc;
        StreamMemReq req;
        int lib_ok, row_ok, family;

        cfg.sr = srs[si]; cfg.fft = ffts[fi]; cfg.hop_in = n * (ffts[fi] / 2);
        CHECK(ai3_model_io_frames(cfg.fft, cfg.hop_in) == n, "frames arithmetic");
        lib_ok = ainr_get_mem_size(&cfg, &req) == AINR_OK;
        row_ok = ai3_model_io_dfn2_find(AI3_MODEL_IO_DFN2_PRODUCT, cfg.sr, cfg.fft, n, &mc) == AI3_MODEL_IO_OK;
        accepted += lib_ok;
        CHECK(lib_ok == (ID_HAS_ROWS && row_ok), "sr %d fft %d N %d: library %s, model config %s", cfg.sr, cfg.fft, n,
              lib_ok ? "accepts" : "refuses", row_ok ? "has a row" : "has no row");
        CHECK(row_ok || (mc.n == 0 && mc.wiring.input_count == 0), "a refused find leaves cfg zeroed");
        if (!lib_ok || !row_ok) continue;
        CHECK(mc.model == AI3_MODEL_IO_DFN2_PRODUCT && mc.n == n && mc.sample_rate == cfg.sr && mc.fft == cfg.fft,
              "key echo");
        CHECK(mc.layout_version == (AI3_MODEL_IO_DFN2_PRODUCT == AI3_MODEL_IO_DFN2_LITE ? DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY
                                                                                        : DFN2_MODEL_IO_LAYOUT_VERSION_FULL),
              "layout version");
        {   /* the row's state count against the library's: a build constant, so any valid grid of the row's rate and size asks it */
            DFN2GridSpec gs;
            DFN2Grid grid;
            DFN2ModelIoDescriptor desc;
            memset(&gs, 0, sizeof gs);
            gs.sample_rate = cfg.sr; gs.n_fft = cfg.fft; gs.n_erb = (int)mc.wiring.outputs[0].w; gs.df_bins = 96;
#ifndef DFN2_MASK_ONLY
            gs.df_order = 5;
#endif
            gs.frames_per_call = n;
            CHECK(dfn2_grid_make(&grid, &gs) == DFN2_GRID_OK && dfn2_model_io_descriptor_default(&grid, &desc) == 0,
                  "sr %d fft %d N %d: no descriptor to compare the state count with", cfg.sr, cfg.fft, n);
            CHECK(ai3_model_io_state_count(&mc) == desc.state_tensor_count, "sr %d fft %d N %d: row has %d states, the library %d",
                  cfg.sr, cfg.fft, n, ai3_model_io_state_count(&mc), desc.state_tensor_count);
        }
        for (family = 0; family < 3; ++family) {
            char tag[96];
            snprintf(tag, sizeof tag, "sr %d fft %d N %d family %d", cfg.sr, cfg.fft, n, family);
            run_family(&cfg, family, n, &mc, tag);
            ++compared;
        }
    }
    /* every row of this build's model is the configuration of one accepted library config, and none is spare */
    CHECK(accepted == (ID_HAS_ROWS ? own_rows : 0), "library accepted %d configs, model config has %d rows (id has rows: %d)",
          accepted, own_rows, ID_HAS_ROWS);
    /* the variant switch: a model with no row at a grid is refused, an unknown model id is refused */
    {
        Ai3ModelIoConfig c;
        CHECK(ai3_model_io_dfn2_find(AI3_MODEL_IO_DFN2_LITE, 16000, 256, 1, &c) == AI3_MODEL_IO_ERR_NO_ROW, "lite has no 16 kHz row");
        CHECK(ai3_model_io_dfn2_find(AI3_MODEL_IO_DEEPVQE_S, 16000, 256, 1, &c) == AI3_MODEL_IO_ERR_NO_ROW, "foreign model");
        CHECK(ai3_model_io_dfn2_find(AI3_MODEL_IO_DFN2_FULL, 16000, 256, 1, NULL) == AI3_MODEL_IO_ERR_NULL, "NULL cfg");
    }
    if (g_fail) return 1;
    printf("test_ainr_model_io[%s id %d]: PASS (library-accepted configs %d, rows %d, real pre calls compared %d)\n",
           AI3_MODEL_IO_DFN2_PRODUCT == AI3_MODEL_IO_DFN2_LITE ? "lite" : "full", (int)AINR_PRODUCT_MODEL_ID, accepted,
           own_rows, compared);
    return 0;
}
